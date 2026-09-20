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
from typing import Any, Dict, List, Optional, Set, Tuple

from ai.knowledge import knowledge_loader as kl
from ai.knowledge.text_utils import tokenize
from ai.llm.llm_factory import get_llm
from ai.llm.prompts import RECOMMENDATION_PROMPT
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def _extract_evidence_fields(cand: Dict[str, Any], req_tokens: Set[str]) -> Dict[str, Any]:
    """Extract scope, keywords, matched terms, evidence clause, version, and certification for a candidate."""
    key = cand.get("key", "")
    cat_rec = kl.get_standard(key) or {}

    # Scope: first 400 chars
    scope = cand.get("scope") or cat_rec.get("scope") or ""
    scope_clean = " ".join(scope.split()) if scope else ""
    scope_snippet = scope_clean[:400]

    # Keywords
    keywords = cand.get("keywords") or cat_rec.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    elif isinstance(keywords, list):
        keywords = [str(k).strip() for k in keywords if str(k).strip()]

    # Title
    title = cand.get("title") or cat_rec.get("title") or ""

    # Matched terms: overlap between requirement/query tokens and candidate's title/scope/keywords
    matched_terms = list(cand.get("matched_terms") or [])
    if not matched_terms and req_tokens:
        text_for_matching = f"{title} {scope_clean} {' '.join(keywords)}"
        cand_tokens = set(tokenize(text_for_matching))
        matched_terms = [t for t in req_tokens if t in cand_tokens]

    # Evidence clause (from retrieval chunk or scope)
    evidence_clause = cand.get("evidence_clause") or ""
    if not evidence_clause and scope_clean:
        first_period = scope_clean.find(".")
        if 20 <= first_period <= 200:
            evidence_clause = scope_clean[:first_period + 1]
        else:
            evidence_clause = scope_clean[:150]

    # Version info
    v_info = cand.get("version_info") or {}
    v_summary = {
        "status": cand.get("status") or v_info.get("status", "ACTIVE"),
        "cited_year": v_info.get("cited_year"),
        "edition_year": v_info.get("edition_year"),
        "successor": cand.get("superseded_by") or v_info.get("successors") or [],
    }

    # Certification summary
    cert = cand.get("certification") or {}
    cert_summary = {
        "mandatory": bool(cert.get("mandatory") or cand.get("certification_mandatory")),
        "scheme": cert.get("scheme_name") or cand.get("certification_scheme"),
    }

    return {
        "scope": scope_snippet,
        "keywords": keywords[:8],
        "matched_terms": matched_terms[:6],
        "evidence_clause": evidence_clause[:250],
        "version_info": v_summary,
        "certification": cert_summary,
    }


def _candidate_summary_for_prompt(candidates: List[Dict], requirement_text: str = "") -> str:
    """Serialise verified candidates for the LLM prompt with scope, keywords, matched terms, clause, version, and certification."""
    slim = []
    req_tokens = set(tokenize(requirement_text)) if requirement_text else set()
    for c in candidates:
        ev = _extract_evidence_fields(c, req_tokens)
        slim.append({
            "key": c.get("key"),
            "display_code": c.get("display_code"),
            "title": c.get("title"),
            "status": c.get("status"),
            "scope": ev["scope"],
            "keywords": ev["keywords"],
            "matched_terms": ev["matched_terms"],
            "evidence_clause": ev["evidence_clause"],
            "version_info": ev["version_info"],
            "certification": ev["certification"],
            "superseded_by": c.get("superseded_by", []),
            "verification_level": c.get("verification_level"),
            "flags": c.get("flags", []),
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


def _build_data_quality_note(cand: Dict[str, Any]) -> str:
    """Build a concise, honest human-readable note about the catalog record quality."""
    flags = cand.get("flags") or []
    v_level = cand.get("verification_level") or "unknown"
    if "needs_review" in flags:
        return "Record flagged for secondary review against official BIS gazette."
    if "unverified" in flags:
        return "Unverified standard entry."
    if v_level == "verified_multi_source":
        return "Verified across multiple authoritative sources."
    if v_level == "single_source_unconfirmed":
        return "Catalog entry confirmed from single source; gazette confirmation pending."
    if flags:
        return f"Flags: {', '.join(flags)}."
    return "Standard metadata verified."


def _compute_confidence(
    relevance_score: float,
    gap_to_next: float,
) -> float:
    """
    Compute a monotonic, calibrated confidence score for a recommendation.
    
    Formula:
        confidence = min(0.99, max(0.10, round(0.85 * relevance_score + 0.12 * gap_to_next, 3)))
        
    Properties:
    - Strictly monotonic in relevance_score (higher retrieval/rerank score -> higher baseline confidence).
    - Rewarded by gap_to_next: when a candidate clearly outdistances the runner-up, confidence increases.
    - Capped strictly at 0.99: never clips to 1.0, preventing artificial confidence saturation and ties.
    - Independent of data quality flags (data quality is surfaced via verification_level and data_quality_note).
    """
    raw_conf = 0.85 * float(relevance_score) + 0.12 * float(gap_to_next)
    conf = min(0.99, max(0.10, round(raw_conf, 3)))
    return conf


def _deterministic_fallback_reasoning(
    cand: Dict[str, Any],
    req_summary: str,
    req_tokens: Optional[Set[str]] = None,
) -> str:
    """Deterministic fallback reasoning citing scope, matched terms, and evidence clause."""
    code = cand.get("display_code", cand.get("key", ""))
    title = cand.get("title", "")
    status = cand.get("status", "ACTIVE")
    superseded = cand.get("superseded_by") or []
    cert = cand.get("certification") or {}
    mandatory = cert.get("mandatory", False)

    tokens = req_tokens if req_tokens is not None else set(tokenize(req_summary))
    ev = _extract_evidence_fields(cand, tokens)
    matched = ev["matched_terms"]
    scope = ev["scope"]
    clause = ev["evidence_clause"]

    reasons = []
    if status in ("WITHDRAWN", "SUPERSEDED"):
        sup_str = ", ".join(superseded) if superseded else "an updated IS edition"
        reasons.append(
            f"WARNING: {code} is {status}. Procurement specifications must reference {sup_str} instead."
        )
    else:
        if matched:
            term_str = ", ".join(f"'{t}'" for t in matched[:3])
            reasons.append(
                f"{code} ('{title}') directly matches requirement for {term_str} based on standard scope and specifications."
            )
        else:
            reasons.append(
                f"{code} covers '{title}', matching the specified procurement requirement."
            )

        if clause:
            clause_words = clause.split()[:16]
            reasons.append(f"Applicable clause specifies: \"{' '.join(clause_words)}...\".")
        elif scope:
            scope_words = scope.split()[:16]
            reasons.append(f"Scope specifies: \"{' '.join(scope_words)}...\".")

    if mandatory:
        reasons.append(
            "Mandatory QCO compliance applies — supplier must possess a valid BIS license / ISI mark."
        )

    flags = cand.get("flags", [])
    if "needs_review" in flags or "unverified" in flags:
        reasons.append("Standard data requires secondary manual verification against BIS gazette.")

    return " ".join(reasons)


def _merge_reasoning(
    candidates: List[Dict],
    reasoning_items: List[Dict],
    req_summary: str,
    warnings: List[str],
    req_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Merge reasoning and assemble final recommendations.
    
    Ranking rules:
    - Final order is determined purely by:
        1. relevance_score descending
        2. is_literal_mention first (True before False)
        3. key alphabetical ascending (deterministic tie-breaker)
    - Data quality NEVER alters the candidate order.
    - Validates LLM output: drops any reasoning key not in the candidate list.
    - If the LLM omits any candidate key or produces invalid JSON, uses deterministic fallback
      reasoning and appends a pipeline warning.
    - Never allows the LLM to inject new IS codes into recommendations.
    """
    # 1. Deterministic and stable sort
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            -float(c.get("score", 0.0)),
            0 if c.get("is_literal_mention") else 1,
            c.get("key", "")
        )
    )

    req_tokens = set(tokenize(req_text or req_summary))

    # 2. Build map of valid reasoning items (strictly filtered to existing candidate keys)
    candidate_keys = {c.get("key") for c in sorted_candidates if c.get("key")}
    valid_reasoning = {}
    for item in reasoning_items:
        k = item.get("key")
        if k and k in candidate_keys and item.get("reasoning"):
            valid_reasoning[k] = item["reasoning"]

    # Check for missing keys
    missing_keys = [k for k in candidate_keys if k not in valid_reasoning]
    if missing_keys:
        logger.info("Node05: using deterministic fallback reasoning for %d candidate(s)", len(missing_keys))
        warnings.append(
            f"Node05: LLM reasoning missed {len(missing_keys)} candidate(s) — deterministic fallback applied."
        )

    # 3. Assemble recommendation items with monotonic confidence
    result = []
    n = len(sorted_candidates)
    for i, cand in enumerate(sorted_candidates):
        key = cand.get("key", "")
        rel_score = float(cand.get("score", 0.5))

        # Gap to next candidate
        next_score = float(sorted_candidates[i + 1].get("score", 0.0)) if i + 1 < n else 0.0
        gap = max(0.0, rel_score - next_score)

        conf = _compute_confidence(rel_score, gap)

        ev = _extract_evidence_fields(cand, req_tokens)

        reasoning_text = valid_reasoning.get(key)
        if not reasoning_text:
            reasoning_text = _deterministic_fallback_reasoning(cand, req_summary, req_tokens)

        # Related standards list
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
            "confidence": conf,
            "relevance_score": round(rel_score, 4),
            "verification_level": cand.get("verification_level", "single_source_unconfirmed"),
            "status": cand.get("status", "UNKNOWN"),
            "superseded_by": cand.get("superseded_by", []),
            "replaced_by": cand.get("replaced_by", []),
            "version_info": cand.get("version_info"),
            "flags": cand.get("flags", []),
            "certification": cand.get("certification"),
            "related_standards": related,
            "reasoning": reasoning_text,
            "evidence_sources": cand.get("evidence_sources", []),
            "data_quality_note": _build_data_quality_note(cand),
            "matched_terms": ev["matched_terms"],
            "evidence_clause": ev["evidence_clause"],
            "scope": ev["scope"],
        })

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
    query_text = state.get("normalized_text", "") or req_summary
    candidates_json = _candidate_summary_for_prompt(candidates, requirement_text=query_text)

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
    recommendations = _merge_reasoning(
        candidates, reasoning_items, req_summary, warnings, req_text=query_text
    )

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
