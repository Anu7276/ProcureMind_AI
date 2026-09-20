"""
POST /recommend

Accepts either:
  A) A structured requirement (from /ingest) + optional audit_id
  B) A raw query string (runs full pipeline internally)

Returns Node05 output: ranked IS code recommendations with reasoning.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from ai.pipeline.graph import run_pipeline, run_pipeline_from_requirement
from backend.schemas.api_schemas import (
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    CertificationInfo,
    RelatedStandard,
    PipelineMeta,
)
from backend.config.settings import settings

router = APIRouter(tags=["Pipeline"])


def _build_recommendation_item(r: dict) -> RecommendationItem:
    cert = r.get("certification")
    cert_info = (
        CertificationInfo(
            mandatory=cert.get("mandatory"),
            scheme_name=cert.get("scheme_name"),
            scheme_code=cert.get("scheme_code"),
            lead_time_weeks=cert.get("lead_time_weeks"),
            penalty=cert.get("penalty"),
            gazette_reference=cert.get("gazette_reference"),
            enforcement_status=cert.get("enforcement_status"),
            evidence_source=cert.get("evidence_source", "qco_orders.json"),
        )
        if cert
        else None
    )

    related = [
        RelatedStandard(
            key=rs.get("key", ""),
            display_code=rs.get("display_code", ""),
            title=rs.get("title", ""),
            relationship_type=rs.get("relationship_type", "RELATED"),
        )
        for rs in r.get("related_standards", [])
    ]

    return RecommendationItem(
        is_code=r.get("is_code", ""),
        key=r.get("key", ""),
        title=r.get("title", ""),
        confidence=r.get("confidence", 0.5),
        verification_level=r.get("verification_level", "single_source_unconfirmed"),
        status=r.get("status", "UNKNOWN"),
        superseded_by=r.get("superseded_by", []),
        flags=r.get("flags", []),
        certification=cert_info,
        related_standards=related,
        reasoning=r.get("reasoning", ""),
        evidence_sources=r.get("evidence_sources", []),
        data_quality_note=r.get("data_quality_note"),
        relevance_score=r.get("relevance_score"),
        version_info=r.get("version_info"),
        replaced_by=r.get("replaced_by", []),
        match_strength=r.get("match_strength"),
        low_match=r.get("low_match"),
        spec_line=r.get("spec_line"),
        clarification_prompt=r.get("clarification_prompt"),
        matched_terms=r.get("matched_terms", []),
        evidence_clause=r.get("evidence_clause"),
        scope=r.get("scope"),
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest):
    """
    Run the full recommendation pipeline.
    Accepts either a raw query or a structured requirement from /ingest.
    """
    audit_id = body.audit_id or str(uuid.uuid4())

    # Path A: structured requirement provided (from prior /ingest call)
    if body.structured_requirement is not None:
        req = body.structured_requirement
        final_state = await run_pipeline_from_requirement(
            structured_requirement=req.model_dump(),
            normalized_text=req.normalized_text or "",
            audit_id=audit_id,
            language=req.language or getattr(body, "language", "en") or "en",
            input_type=req.input_type or getattr(body, "input_type", "text") or "text",
            user_edited=getattr(body, "user_edited", False),
        )

    # Path B: raw text query — run full pipeline
    elif body.raw_query and body.raw_query.strip():
        final_state = await run_pipeline(
            raw_input=body.raw_query.strip(),
            input_type="text",
            audit_id=audit_id,
        )

    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either raw_query or structured_requirement.",
        )

    raw_recs = final_state.get("recommendations", [])
    recommendations = [_build_recommendation_item(r) for r in raw_recs]

    raw_closest = final_state.get("closest_matches", [])
    closest_matches = [_build_recommendation_item(r) for r in raw_closest]

    abstained = bool(final_state.get("abstained", False))
    abstain_reason = final_state.get("abstain_reason")

    req_data = final_state.get("structured_requirement", {})
    query_summary = (
        req_data.get("product")
        or body.raw_query
        or final_state.get("normalized_text", "")[:100]
        or "Procurement requirement"
    )

    pipeline_meta = PipelineMeta(
        llm_mode=final_state.get("pipeline_meta", {}).get("llm_mode") or getattr(settings, "LLM_PROVIDER", "mock"),
        llm_model=getattr(settings, "llm_model_name", ""),
        embedder=getattr(settings, "EMBEDDING_MODEL_NAME", ""),
        retrieval_sources_used=final_state.get("retrieval_sources_used", []),
        reranker_mode="cross_encoder" if final_state.get("cross_encoder_used") else "bm25_lexical",
        abstained=abstained,
        abstain_reason=abstain_reason,
        audit_saved=final_state.get("audit_saved"),
        warnings=final_state.get("pipeline_warnings", []),
    )

    return RecommendResponse(
        audit_id=audit_id,
        query_summary=str(query_summary),
        recommendations=recommendations,
        warnings=final_state.get("pipeline_warnings", []),
        pipeline_stages_completed=final_state.get("stages_completed", []),
        abstained=abstained,
        abstain_reason=abstain_reason,
        closest_matches=closest_matches,
        pipeline_meta=pipeline_meta,
    )
