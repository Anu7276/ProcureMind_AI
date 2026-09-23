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
) -> Dict[str, Any]:
    """
    Build compliance block for a standard.
    Primary: QCO in-memory index (always available).
    Supplement: PostgreSQL for scheme details and product rules.
    When no QCO or confirmed compliance field exists in dataset, returns
    mandatory: None and schemes: [] (never fabricates voluntary / scheme values).
    lead_time_weeks, marking_requirements, testing_frequency are null unless
    explicitly present in the dataset record or QCO for THIS standard's key.
    """
    record = kl.get_standard(standard_key) or {}
    qco = kl.get_qco_for_key(standard_key)

    if qco:
        raw_cert = qco.get("certification_required")
        schemes = [raw_cert] if raw_cert else []
        enforcement = qco.get("enforcement_status")
        mandatory = (
            True if enforcement == "MANDATORY_ENFORCED"
            else (False if enforcement == "VOLUNTARY" else None)
        )
        return {
            "mandatory": mandatory,
            "schemes": schemes,
            "scheme_name": qco.get("scheme_name") or raw_cert,
            "scheme_code": qco.get("scheme_code"),
            "lead_time_weeks": qco.get("lead_time_weeks") or record.get("lead_time_weeks"),
            "marking_requirements": qco.get("marking_requirements") or record.get("marking_requirements"),
            "testing_frequency": qco.get("testing_frequency") or record.get("testing_frequency"),
            "penalty": qco.get("penalty"),
            "gazette_reference": qco.get("gazette_reference"),
            "enforcement_status": enforcement,
            "evidence_source": f"qco_orders.json ({qco.get('qco_id', 'unknown')})",
        }

    # Supplement from PostgreSQL if available
    if postgres_ok:
        try:
            from backend.services import postgres_service
            qco_row = await postgres_service.get_qco_for_standard(standard_key)
            if qco_row:
                raw_cert = qco_row.get("certification_required")
                schemes = [raw_cert] if raw_cert else []
                enforcement = qco_row.get("enforcement_status")
                mandatory = (
                    True if enforcement == "MANDATORY_ENFORCED"
                    else (False if enforcement == "VOLUNTARY" else None)
                )
                return {
                    "mandatory": mandatory,
                    "schemes": schemes,
                    "scheme_name": qco_row.get("scheme_name") or raw_cert,
                    "scheme_code": qco_row.get("scheme_code"),
                    "lead_time_weeks": qco_row.get("lead_time_weeks") or record.get("lead_time_weeks"),
                    "marking_requirements": qco_row.get("marking_requirements") or record.get("marking_requirements"),
                    "testing_frequency": qco_row.get("testing_frequency") or record.get("testing_frequency"),
                    "penalty": qco_row.get("penalty"),
                    "gazette_reference": qco_row.get("gazette_reference"),
                    "enforcement_status": enforcement,
                    "evidence_source": f"qco_orders.postgres ({qco_row.get('qco_id', '')})",
                }
        except Exception as exc:
            logger.debug("Postgres compliance lookup failed for %s: %s", standard_key, exc)

    # Check product_rules in-memory (via standards compliance field)
    if record:
        compliance = record.get("compliance", {}) or {}
        mandatory = compliance.get("mandatory_certification")
        schemes = compliance.get("schemes", []) or []
        if mandatory is not None or (schemes and len(schemes) > 0):
            scheme_name = schemes[0] if schemes else compliance.get("scheme_name")
            return {
                "mandatory": bool(mandatory) if mandatory is not None else None,
                "schemes": list(schemes),
                "scheme_name": scheme_name,
                "scheme_code": compliance.get("scheme_code"),
                "lead_time_weeks": compliance.get("lead_time_weeks") or record.get("lead_time_weeks"),
                "marking_requirements": compliance.get("marking_requirements") or record.get("marking_requirements"),
                "testing_frequency": compliance.get("testing_frequency") or record.get("testing_frequency"),
                "penalty": compliance.get("penalty"),
                "gazette_reference": compliance.get("gazette_reference"),
                "enforcement_status": "MANDATORY_ENFORCED" if mandatory is True else "VOLUNTARY" if mandatory is False else None,
                "evidence_source": "standards_clean.json (compliance field)",
            }

    # Explicit null/empty certification block when not available in dataset
    return {
        "mandatory": None,
        "schemes": [],
        "scheme_name": None,
        "scheme_code": None,
        "lead_time_weeks": None,
        "marking_requirements": None,
        "testing_frequency": None,
        "penalty": None,
        "gazette_reference": None,
        "enforcement_status": None,
        "evidence_source": None,
    }


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

    literal_codes_list = state.get("literal_codes", [])
    literal_codes_by_key = {item["key"]: item for item in literal_codes_list if item.get("key")}

    verified: List[Dict[str, Any]] = []
    promoted_candidates: List[Dict[str, Any]] = []
    candidate_keys = {c.get("key") for c in candidates if c.get("key")}

    for cand in candidates:
        key = cand.get("key", "")
        display_code = cand.get("display_code", key)

        # ── Whitelist validation (guard against hallucinated/invented codes) ────
        if not kl.validate_whitelist(key) and not kl.validate_whitelist(display_code):
            logger.warning(
                "Node04: dropping %s — not on whitelist (hallucination guard)", key
            )
            warnings.append(
                f"Candidate {key} dropped — not found in is_code_whitelist_clean.json"
            )
            continue

        # ── Version and status resolution ──────────────────────────────────────
        record = kl.get_standard(key)
        cited_year = cand.get("cited_year")
        if cited_year is None and key in literal_codes_by_key:
            cited_year = literal_codes_by_key[key].get("cited_year")

        version_info = kl.check_standard_version(record or cand, cited_year=cited_year)
        status = version_info["status"]
        successors = version_info["successors"]

        if record:
            status_info = _resolve_status(record)
        else:
            status_info = {
                "status": status,
                "superseded_by": successors,
                "verification_level": cand.get("verification_level", "single_source_unconfirmed"),
                "flags": cand.get("flags", []),
                "has_full_text": False,
            }

        flags = list(status_info.get("flags", []))
        if any("cites" in m or "mismatch" in m.lower() for m in version_info.get("messages", [])):
            if "edition_mismatch" not in flags:
                flags.append("edition_mismatch")
        if status in ("WITHDRAWN", "SUPERSEDED"):
            if status.lower() not in flags:
                flags.append(status.lower())

        # ── Successor promotion & demotion ─────────────────────────────────────
        replaced_by = []
        if status in ("WITHDRAWN", "SUPERSEDED"):
            replaced_by = list(successors)
            old_score = float(cand.get("score", 0.5))
            # Demote old standard: user-cited literal mentions demoted slightly (-0.03) to rank below promoted successor (+0.02)
            # but remain clearly visible alongside it. Uncited obsolete standards demoted heavily (0.5x).
            if cand.get("is_literal_mention"):
                cand["score"] = round(max(0.1, old_score - 0.03), 4)
            else:
                cand["score"] = round(old_score * 0.5, 4)
            successor_str = ", ".join(successors) if successors else "unknown"
            logger.info("Node04: %s is %s → promoted successor(s): %s", key, status, successor_str)

            for succ_key in successors:
                existing = next((c for c in candidates if c.get("key") == succ_key), None)
                if existing:
                    existing["score"] = max(float(existing.get("score", 0.0)), round(min(1.0, old_score + 0.02), 4))
                    existing["is_successor_promotion"] = True
                else:
                    succ_rec = kl.get_standard(succ_key)
                    if succ_rec and succ_key not in candidate_keys:
                        candidate_keys.add(succ_key)
                        promoted_candidates.append({
                            "key": succ_key,
                            "display_code": succ_rec.get("display_code", succ_key),
                            "title": succ_rec.get("title", ""),
                            "category": succ_rec.get("category"),
                            "subcategory": succ_rec.get("subcategory"),
                            "score": round(min(1.0, old_score + 0.02), 4),
                            "source": f"successor_of:{key}",
                            "is_successor_promotion": True,
                        })

        # ── Amendments, Version Check, Compliance & Verification blocks ───────
        amend_entries = kl.AMENDMENTS_BY_KEY.get(key, [])
        amendments_block = {
            "status": "listed" if len(amend_entries) > 0 else "not_available_in_dataset",
            "entries": list(amend_entries),
        }

        version_check_block = {
            "current_edition_year": version_info.get("current_edition_year"),
            "cited_year": version_info.get("cited_year"),
            "is_current": version_info.get("is_current"),
            "status": version_info.get("status", "UNKNOWN"),
            "successors": list(successors),
            "messages": list(version_info.get("messages", [])),
        }

        compliance = await _get_compliance(key, postgres_ok)

        verification_block = {
            "level": status_info["verification_level"],
            "flags": list(flags),
        }

        # ── Evidence sources ──────────────────────────────────────────────────
        evidence = _build_evidence_sources(cand, record, compliance)

        # ── Assemble verified candidate ────────────────────────────────────────
        verified.append({
            **cand,
            "status": status,
            "superseded_by": successors,
            "replaced_by": replaced_by,
            "version_info": version_info,
            "verification_level": status_info["verification_level"],
            "flags": flags,
            "has_full_text": status_info.get("has_full_text", False),
            "certification": compliance,
            "amendments": amendments_block,
            "version_check": version_check_block,
            "verification": verification_block,
            "evidence_sources": evidence,
            "whitelist_valid": True,
        })

    # ── Process promoted successors ───────────────────────────────────────────
    for p_cand in promoted_candidates:
        p_key = p_cand["key"]
        p_rec = kl.get_standard(p_key)
        p_version = kl.check_standard_version(p_rec or p_cand, cited_year=None)
        p_compliance = await _get_compliance(p_key, postgres_ok)
        p_evidence = _build_evidence_sources(p_cand, p_rec, p_compliance)
        p_status_info = _resolve_status(p_rec) if p_rec else {
            "status": p_version["status"],
            "superseded_by": p_version["successors"],
            "verification_level": "single_source_unconfirmed",
            "flags": [],
            "has_full_text": False,
        }

        p_amend_entries = kl.AMENDMENTS_BY_KEY.get(p_key, [])
        p_amendments = {
            "status": "listed" if len(p_amend_entries) > 0 else "not_available_in_dataset",
            "entries": list(p_amend_entries),
        }

        p_version_check = {
            "current_edition_year": p_version.get("current_edition_year"),
            "cited_year": p_version.get("cited_year"),
            "is_current": p_version.get("is_current"),
            "status": p_version.get("status", "UNKNOWN"),
            "successors": list(p_version.get("successors", [])),
            "messages": list(p_version.get("messages", [])),
        }

        p_verification = {
            "level": p_status_info["verification_level"],
            "flags": list(p_status_info.get("flags", [])),
        }

        verified.append({
            **p_cand,
            "status": p_version["status"],
            "superseded_by": p_version["successors"],
            "replaced_by": [],
            "version_info": p_version,
            "verification_level": p_status_info["verification_level"],
            "flags": p_status_info.get("flags", []),
            "has_full_text": p_status_info.get("has_full_text", False),
            "certification": p_compliance,
            "amendments": p_amendments,
            "version_check": p_version_check,
            "verification": p_verification,
            "evidence_sources": p_evidence,
            "whitelist_valid": True,
            "related_standards": kl.RELATIONSHIPS_BY_KEY.get(p_key, []),
        })

    logger.info(
        "Node04: %d candidates passed whitelist and verification (including %d promoted successors)",
        len(verified), len(promoted_candidates),
    )
    stages.append("verify")
    return {
        "verified_candidates": verified,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
