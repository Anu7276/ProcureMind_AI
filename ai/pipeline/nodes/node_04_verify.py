"""
Node 04 — Verify

For each candidate from Node03:
  1. Resolve status (ACTIVE/WITHDRAWN/SUPERSEDED) from in-memory standards dict
     → WITHDRAWN/SUPERSEDED are surfaced prominently, NOT silently dropped
     → Pilot case: IS 8828 WITHDRAWN → IS/IEC 60898 (Part 1) & (Part 2)
  2. Look up QCO/certification from PostgreSQL (degraded: skip with warning)
  3. Surface verification_level and flags — needs_review / unverified are VISIBLE
  4. Validate every code against is_code_whitelist_clean.json
     → Drop any not on whitelist (hard safety net against hallucination)

Degraded-response behavior:
  ┌──────────────────┬────────────────────────────────────────────────────────┐
  │ Store Down       │ Behavior                                               │
  ├──────────────────┼────────────────────────────────────────────────────────┤
  │ PostgreSQL only  │ Status/superseded_by from in-memory dict (always avail)│
  │                  │ QCO/certification skipped, warning added per candidate  │
  └──────────────────┴────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def _resolve_status(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract status-related fields from a standards_clean.json record.
    Returns: {status, superseded_by, verification_level, flags, has_full_text}
    """
    status = record.get("status", "UNKNOWN")
    superseded_by = record.get("superseded_by") or []
    if isinstance(superseded_by, str):
        superseded_by = [superseded_by]

    dq = record.get("data_quality", {})
    verification_level = dq.get("verification_level", "single_source_unconfirmed")
    flags = dq.get("flags", [])
    has_full_text = (record.get("text") or {}).get("has_full_text", False)

    return {
        "status": status,
        "superseded_by": superseded_by,
        "verification_level": verification_level,
        "flags": flags,
        "has_full_text": has_full_text,
    }


async def _get_compliance(
    standard_key: str, postgres_ok: bool
) -> Optional[Dict[str, Any]]:
    """
    Build compliance block.
    Primary: QCO in-memory index (always available).
    Supplement: PostgreSQL for scheme details and product rules.
    """
    # In-memory QCO lookup (always works, even with Postgres down)
    qco = kl.get_qco_for_key(standard_key)

    if qco:
        scheme_code = _extract_scheme_code(qco.get("certification_required", ""))
        scheme = kl.get_cert_scheme(scheme_code) if scheme_code else None
        return {
            "mandatory": qco.get("enforcement_status") == "MANDATORY_ENFORCED",
            "scheme_name": scheme.get("name") if scheme else qco.get("certification_required"),
            "scheme_code": scheme_code,
            "lead_time_weeks": scheme.get("lead_time_weeks") if scheme else None,
            "penalty": qco.get("penalty"),
            "gazette_reference": qco.get("gazette_reference"),
            "enforcement_status": qco.get("enforcement_status"),
            "evidence_source": f"qco_orders.json ({qco.get('qco_id', 'unknown')})",
        }

    # Supplement from PostgreSQL if available
    if postgres_ok:
        try:
            from backend.services import postgres_service
            qco_row = await postgres_service.get_qco_for_standard(standard_key)
            if qco_row:
                return {
                    "mandatory": qco_row.get("enforcement_status") == "MANDATORY_ENFORCED",
                    "scheme_name": qco_row.get("certification_required"),
                    "scheme_code": None,
                    "penalty": qco_row.get("penalty"),
                    "gazette_reference": qco_row.get("gazette_reference"),
                    "enforcement_status": qco_row.get("enforcement_status"),
                    "evidence_source": f"qco_orders.postgres ({qco_row.get('qco_id', '')})",
                }
        except Exception as exc:
            logger.debug("Postgres compliance lookup failed for %s: %s", standard_key, exc)

    # Check product_rules in-memory (via standards compliance field)
    record = kl.get_standard(standard_key)
    if record:
        compliance = record.get("compliance", {}) or {}
        mandatory = compliance.get("mandatory_certification")
        schemes = compliance.get("schemes", [])
        if mandatory is not None or schemes:
            return {
                "mandatory": mandatory,
                "scheme_name": schemes[0] if schemes else None,
                "scheme_code": None,
                "penalty": None,
                "gazette_reference": None,
                "enforcement_status": None,
                "evidence_source": "standards_clean.json (compliance field)",
            }

    return None


def _extract_scheme_code(cert_text: str) -> Optional[str]:
    """Extract scheme code from strings like 'Scheme-I / ISI Mark'."""
    if not cert_text:
        return None
    if "Scheme-I" in cert_text or "ISI" in cert_text:
        return "Scheme-I"
    if "Scheme-IV" in cert_text:
        return "Scheme-IV"
    if "CRS" in cert_text:
        return "CRS"
    if "FMCS" in cert_text:
        return "FMCS"
    return None


def _build_evidence_sources(
    candidate: Dict, record: Optional[Dict], compliance: Optional[Dict]
) -> List[str]:
    """Assemble evidence source citations."""
    sources = [f"IS {candidate.get('display_code', '')}, BIS"]
    if compliance and compliance.get("gazette_reference"):
        sources.append(compliance["gazette_reference"])
    if compliance and compliance.get("evidence_source"):
        sources.append(compliance["evidence_source"])
    return list(dict.fromkeys(sources))  # deduplicate


async def node_04_verify(state: PipelineState) -> dict:
    """Verify status, compliance, quality flags for each candidate."""
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))
    candidates = state.get("candidates", [])

    if not candidates:
        stages.append("verify")
        return {
            "verified_candidates": [],
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    # Check Postgres availability once (for compliance supplement)
    postgres_ok = False
    try:
        from backend.services import postgres_service
        postgres_ok = await postgres_service.is_available()
        if not postgres_ok:
            warnings.append(
                "PostgreSQL unavailable — compliance/QCO data will use in-memory index only"
            )
    except Exception:
        warnings.append(
            "PostgreSQL unavailable — compliance/QCO data will use in-memory index only"
        )

    verified: List[Dict[str, Any]] = []

    for cand in candidates:
        key = cand.get("key", "")

        # ── Whitelist validation (hard safety net) ────────────────────────
        display_code = cand.get("display_code", key)
        if not kl.validate_whitelist(key) and not kl.validate_whitelist(display_code):
            logger.warning(
                "Node04: dropping %s — not on whitelist (hallucination guard)", key
            )
            warnings.append(
                f"Candidate {key} dropped — not found in is_code_whitelist_clean.json"
            )
            continue

        # ── Status resolution (in-memory, always available) ───────────────
        record = kl.get_standard(key)
        if record:
            status_info = _resolve_status(record)
        else:
            # Key in Qdrant but not loaded in memory (edge case)
            status_info = {
                "status": cand.get("status", "UNKNOWN"),
                "superseded_by": [],
                "verification_level": cand.get("verification_level", "single_source_unconfirmed"),
                "flags": [],
                "has_full_text": False,
            }

        # ── Prominent WITHDRAWN / SUPERSEDED surfacing ────────────────────
        # IS 8828 pilot case: WITHDRAWN → IS/IEC 60898 (Part 1) & (Part 2)
        if status_info["status"] in ("WITHDRAWN", "SUPERSEDED"):
            successors = status_info.get("superseded_by", [])
            successor_str = ", ".join(successors) if successors else "unknown"
            logger.info(
                "Node04: %s is %s → succeeded by %s",
                key, status_info["status"], successor_str,
            )
            # Do NOT drop — surface with prominent flag so Node05 explains it

        # ── Compliance lookup ─────────────────────────────────────────────
        compliance = await _get_compliance(key, postgres_ok)

        # ── Evidence sources ──────────────────────────────────────────────
        evidence = _build_evidence_sources(cand, record, compliance)

        # ── Assemble verified candidate ────────────────────────────────────
        verified.append({
            **cand,
            "status": status_info["status"],
            "superseded_by": status_info["superseded_by"],
            "verification_level": status_info["verification_level"],
            "flags": status_info["flags"],
            "has_full_text": status_info["has_full_text"],
            "certification": compliance,
            "evidence_sources": evidence,
            "whitelist_valid": True,
        })

    logger.info(
        "Node04: %d/%d candidates passed whitelist validation",
        len(verified), len(candidates),
    )
    stages.append("verify")
    return {
        "verified_candidates": verified,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
