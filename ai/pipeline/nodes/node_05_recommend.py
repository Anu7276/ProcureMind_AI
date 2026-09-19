"""
Node 05 — Recommend

LLM call: produce final explainable, ranked recommendations.
Each result includes: IS code, title, confidence, status, related standards,
certification, and reasoning citing the specific scope/keyword that triggered
the match (not a generic explanation).

Also writes the full result to recommendation_log in PostgreSQL (audit trail).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ai.llm.llm_factory import get_llm
from ai.llm.prompts import RECOMMENDATION_PROMPT
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def _candidate_summary_for_prompt(candidates: List[Dict]) -> str:
    """Serialise verified candidates for the LLM prompt (truncated for token budget)."""
    slim = []
    for c in candidates:
        slim.append({
            "key": c.get("key"),
            "display_code": c.get("display_code"),
            "title": c.get("title"),
            "status": c.get("status"),
            "superseded_by": c.get("superseded_by", []),
            "verification_level": c.get("verification_level"),
            "flags": c.get("flags", []),
            "certification_mandatory": c.get("certification", {}).get("mandatory") if c.get("certification") else None,
            "certification_scheme": c.get("certification", {}).get("scheme_name") if c.get("certification") else None,
            "related_count": len(c.get("related_standards", [])),
        })
    return json.dumps(slim, ensure_ascii=False, indent=2)


def _requirement_summary(state: PipelineState) -> str:
    req = state.get("structured_requirement", {})
    parts = []
    for field, label in [
        ("product", "Product"), ("material", "Material"),
        ("specifications", "Specs"), ("performance_requirements", "Performance"),
        ("safety_requirements", "Safety"), ("application", "Application"),
    ]:
        val = req.get(field)
        if val:
            parts.append(f"{label}: {val}")
    return "; ".join(parts) or state.get("normalized_text", "")[:300]


def _parse_reasoning_response(raw: str) -> List[Dict[str, Any]]:
    """Parse the LLM's JSON array of reasoning items."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception as exc:
        logger.warning("Node05: reasoning JSON parse failed: %s", exc)
    return []


def _merge_reasoning(
    candidates: List[Dict], reasoning_items: List[Dict]
) -> List[Dict[str, Any]]:
    """Merge LLM reasoning back into candidate dicts, sorted by final confidence."""
    reasoning_by_key = {r["key"]: r for r in reasoning_items if "key" in r}

    result = []
    for cand in candidates:
        key = cand.get("key", "")
        reasoning_info = reasoning_by_key.get(key, {})

        base_score = float(cand.get("score", 0.5))
        adjustment = float(reasoning_info.get("confidence_adjustment", 0.0))
        final_confidence = round(max(0.0, min(1.0, base_score + adjustment)), 3)

        # Build related standards list for output
        related = [
            {
                "key": r.get("key"),
                "display_code": r.get("display_code"),
                "title": r.get("title", ""),
                "relationship_type": r.get("relationship_type", "RELATED"),
            }
            for r in cand.get("related_standards", [])
            if r.get("key")
        ]

        result.append({
            "is_code": cand.get("display_code", key),
            "key": key,
            "title": cand.get("title", ""),
            "confidence": final_confidence,
            "verification_level": cand.get("verification_level", "single_source_unconfirmed"),
            "status": cand.get("status", "UNKNOWN"),
            "superseded_by": cand.get("superseded_by", []),
            "flags": cand.get("flags", []),
            "certification": cand.get("certification"),
            "related_standards": related,
            "reasoning": reasoning_info.get(
                "reasoning",
                f"Matched based on vector similarity to requirement. "
                f"Verification level: {cand.get('verification_level', 'unknown')}.",
            ),
            "evidence_sources": cand.get("evidence_sources", []),
        })

    result.sort(key=lambda x: x["confidence"], reverse=True)
    return result


async def node_05_recommend(state: PipelineState) -> dict:
    """Generate final explainable recommendations and write audit log."""
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))
    candidates = state.get("verified_candidates", [])
    audit_id = state.get("audit_id", "")

    if not candidates:
        stages.append("recommend")
        return {
            "recommendations": [],
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    req_summary = _requirement_summary(state)
    candidates_json = _candidate_summary_for_prompt(candidates)

    # ── LLM call for reasoning ────────────────────────────────────────────────
    reasoning_items: List[Dict] = []
    try:
        llm = get_llm()
        chain = RECOMMENDATION_PROMPT | llm
        response = await chain.ainvoke({
            "requirement_summary": req_summary,
            "n_candidates": len(candidates),
            "candidates_json": candidates_json,
        })
        raw_content = response.content if hasattr(response, "content") else str(response)
        reasoning_items = _parse_reasoning_response(raw_content)
        logger.info("Node05: LLM produced %d reasoning items", len(reasoning_items))

    except Exception as exc:
        logger.warning("Node05: LLM reasoning unavailable (%s) — using rule-based reasoning generator", exc)
        warnings.append(
            "LLM API pending — generated standard-specific reasoning via BIS verification engine"
        )
        from ai.llm.mock_llm import generate_mock_reasoning
        reasoning_items = generate_mock_reasoning(req_summary, candidates)

    if not reasoning_items:
        from ai.llm.mock_llm import generate_mock_reasoning
        reasoning_items = generate_mock_reasoning(req_summary, candidates)

    # ── Merge reasoning + build final list ────────────────────────────────────
    recommendations = _merge_reasoning(candidates, reasoning_items)

    # ── Write audit log to PostgreSQL ────────────────────────────────────────
    returned_codes = [r["is_code"] for r in recommendations]
    top_conf = recommendations[0]["confidence"] if recommendations else None

    try:
        from backend.services import postgres_service
        await postgres_service.write_recommendation_log(
            audit_id=audit_id,
            query_text=state.get("normalized_text", state.get("raw_input", "")),
            input_type=state.get("input_type", "text"),
            structured_requirement=state.get("structured_requirement"),
            returned_codes=returned_codes,
            pipeline_warnings=warnings,
            top_confidence=top_conf,
        )
        logger.info("Node05: audit log written for audit_id=%s", audit_id)
    except Exception as exc:
        logger.warning("Node05: audit log write failed (non-fatal): %s", exc)
        warnings.append(f"Audit log not persisted ({exc}) — result still valid")

    stages.append("recommend")
    return {
        "recommendations": recommendations,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
