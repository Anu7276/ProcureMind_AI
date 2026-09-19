"""GET /standard/{key} — direct lookup + graph neighbours."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ai.knowledge import knowledge_loader as kl
from backend.schemas.api_schemas import CertificationInfo, RelatedStandard, StandardDetail

router = APIRouter(tags=["Standards"])


@router.get("/standard/{key:path}", response_model=StandardDetail)
async def get_standard(key: str):
    """
    Direct lookup for a single IS code.
    Returns full metadata + graph neighbours (for frontend drill-down).
    The {key:path} converter allows keys with slashes like 'IS/IEC 60898 (Part 1)'.
    """
    # In-memory lookup first (fastest path)
    record = kl.get_standard(key)

    if record is None:
        # Try without edition year suffix
        bare_key = key.split(":")[0].strip()
        record = kl.get_standard(bare_key)

    if record is None:
        raise HTTPException(status_code=404, detail=f"Standard not found: {key}")

    dq = record.get("data_quality", {})

    # Graph neighbours
    related: list[RelatedStandard] = []
    try:
        from backend.services import neo4j_service
        avail, _ = await neo4j_service.is_available()
        if avail:
            neighbours = await neo4j_service.get_related_standards(record["key"])
            related = [
                RelatedStandard(
                    key=n.get("key", ""),
                    display_code=n.get("display_code", ""),
                    title=n.get("title", ""),
                    relationship_type=n.get("relationship_type", "RELATED"),
                )
                for n in neighbours
                if n.get("key")
            ]
    except Exception:
        pass  # graph neighbours are supplementary; don't fail the request

    # Compliance info (in-memory)
    cert_info = None
    qco = kl.get_qco_for_key(record["key"])
    if qco:
        cert_info = CertificationInfo(
            mandatory=qco.get("enforcement_status") == "MANDATORY_ENFORCED",
            scheme_name=qco.get("certification_required"),
            penalty=qco.get("penalty"),
            gazette_reference=qco.get("gazette_reference"),
            enforcement_status=qco.get("enforcement_status"),
        )

    superseded_by = record.get("superseded_by") or []
    if isinstance(superseded_by, str):
        superseded_by = [superseded_by]

    return StandardDetail(
        key=record["key"],
        display_code=record.get("display_code", key),
        edition_year=record.get("edition_year"),
        title=record.get("title", ""),
        scope=record.get("scope"),
        status=record.get("status", "UNKNOWN"),
        category=record.get("category"),
        subcategory=record.get("subcategory"),
        verification_level=dq.get("verification_level", "single_source_unconfirmed"),
        flags=dq.get("flags", []),
        has_full_text=(record.get("text") or {}).get("has_full_text", False),
        superseded_by=superseded_by,
        keywords=record.get("keywords") or [],
        compliance=cert_info,
        related_standards=related,
    )
