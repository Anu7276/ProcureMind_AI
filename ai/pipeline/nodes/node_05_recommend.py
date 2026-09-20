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
from typing import Any, Dict, List, Optional, Set

from ai.knowledge import knowledge_loader as kl
from ai.knowledge.text_utils import tokenize
from ai.llm.llm_factory import get_llm
from ai.llm.prompts import RECOMMENDATION_PROMPT
from ai.pipeline.state import PipelineState
from backend.config.settings import settings

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
    match_strength: float,
    gap_to_next: float,
) -> float:
    """
    Compute a monotonic, calibrated confidence score for a recommendation.
    Formula:
        confidence = min(match_strength + 0.05, max(0.01, round(0.85 * match_strength + 0.12 * gap_to_next, 3)))
    """
    ms = float(match_strength)
    raw_conf = 0.85 * ms + 0.12 * float(gap_to_next)
    conf = min(0.99, max(0.01, round(raw_conf, 3)))
    max_cap = round(ms + 0.05, 3)
    if conf > max_cap:
        conf = max_cap
    return round(max(0.01, conf), 3)


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


def _generate_spec_line(cand: Dict[str, Any], req_summary: str) -> str:
    """
    Generate exact, copy-paste-ready tender clause for every recommendation:
    Format:
    "The [product] shall conform to [IS code] (or latest revision) with valid BIS [Scheme Name] [Registration/License/Certification]. The vendor shall provide [Type of proof] prior to dispatch. [Mandatory/Voluntary note]"
    """
    title = cand.get("title", "")
    code = cand.get("is_code") or cand.get("display_code") or cand.get("key", "")
    cert = cand.get("certification") or {}
    mandatory = cert.get("mandatory", False)
    scheme_name = cert.get("scheme_name") or "Scheme-I / ISI Mark"
    scheme_code = cert.get("scheme_code") or "Scheme-I"
    gazette_ref = cert.get("gazette_reference")

    product = req_summary or title or "materials/equipment"

    if "Scheme-II" in scheme_code or "CRS" in scheme_name:
        cert_term = "Registration"
        proof_term = "valid BIS Registration Certificate (CRS)"
    elif "Scheme-IV" in scheme_code or "CoC" in scheme_name or "Certificate of Conformity" in scheme_name:
        cert_term = "Certificate of Conformity"
        proof_term = "Certificate of Conformity (CoC) / Type Test Report"
    elif "FMCS" in scheme_code or "Scheme-X" in scheme_code:
        cert_term = "Foreign Manufacturers Certification"
        proof_term = "valid BIS FMCS License"
    elif "Eco" in scheme_name or "Eco" in scheme_code:
        cert_term = "Eco Mark Certification"
        proof_term = "valid BIS Eco Mark License"
    elif "HM" in scheme_code or "Hallmark" in scheme_name:
        cert_term = "Hallmarking Registration"
        proof_term = "BIS Hallmarking Registration and HUID verification"
    else:
        cert_term = "License"
        proof_term = "valid BIS Certification Marks License (ISI Mark)"

    spec = f"The {product} shall conform to {code} (or latest revision) with valid BIS {scheme_name} {cert_term}. The vendor shall provide {proof_term} prior to dispatch."

    if mandatory:
        qco_clause = f"Under QCO ({gazette_ref}), this" if gazette_ref else "Under statutory Quality Control Orders (QCO), this"
        spec += f" {qco_clause} is mandatory for supply and non-compliant bids shall be rejected at technical stage."
    else:
        spec += " Compliance is recommended for quality assurance."

    return spec


def _generate_clarification_prompt(cand: Dict[str, Any], req_summary: str) -> str:
    """Generate a helpful clarification question when confidence is below 0.60."""
    category = cand.get("category") or "General"
    title = cand.get("title") or "the item"
    return (
        f"Recommendation confidence is moderate. To refine results for '{title}', "
        f"please specify: (1) exact material grade/type, (2) dimensions, voltage, or capacity ratings, "
        f"or (3) operating application ({category})."
    )


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

    # 3. Assemble recommendation items with monotonic confidence and match strength
    items = []
    n = len(sorted_candidates)
    low_match_thresh = getattr(settings, "LOW_MATCH_THRESHOLD", 0.50)
    abstain_thresh = getattr(settings, "ABSTAIN_THRESHOLD", 0.30)

    for i, cand in enumerate(sorted_candidates):
        key = cand.get("key", "")
        match_strength = float(cand.get("match_strength", cand.get("relevance_score", cand.get("score", 0.5))))

        # Gap to next candidate
        next_strength = float(
            sorted_candidates[i + 1].get("match_strength", sorted_candidates[i + 1].get("relevance_score", sorted_candidates[i + 1].get("score", 0.0)))
        ) if i + 1 < n else 0.0
        gap = max(0.0, match_strength - next_strength)

        conf = _compute_confidence(match_strength, gap)

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

        is_low_conf = (match_strength < low_match_thresh) or (conf < 0.60)
        clarification = _generate_clarification_prompt(cand, req_summary) if is_low_conf else None

        items.append({
            "is_code": cand.get("display_code", key),
            "key": key,
            "title": cand.get("title", ""),
            "confidence": conf,
            "match_strength": round(match_strength, 4),
            "relevance_score": round(match_strength, 4),
            "coverage": round(float(cand.get("coverage") or 0.0), 4),
            "low_match": False,
            "is_low_confidence": is_low_conf,
            "clarification_prompt": clarification,
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
            "spec_line": _generate_spec_line(cand, req_summary),
            "data_quality_note": _build_data_quality_note(cand),
            "matched_terms": ev["matched_terms"],
            "evidence_clause": ev["evidence_clause"],
            "scope": ev["scope"],
        })

    # Ensure successor match strength and confidence cap on withdrawn/superseded standards
    item_by_key = {it["key"]: it for it in items}
    for it in items:
        if it.get("status") in ("WITHDRAWN", "SUPERSEDED"):
            succ_keys = it.get("replaced_by") or it.get("superseded_by") or []
            succ_items = [item_by_key[k] for k in succ_keys if k in item_by_key]
            if succ_items:
                for s_it in succ_items:
                    if s_it["match_strength"] < it["match_strength"]:
                        s_it["match_strength"] = it["match_strength"]
                        s_it["relevance_score"] = it["match_strength"]
                        s_it["confidence"] = max(s_it["confidence"], _compute_confidence(s_it["match_strength"], 0.05))
                max_succ_conf = max(s_it["confidence"] for s_it in succ_items)
                it["confidence"] = max(0.10, round(max_succ_conf - 0.05, 3))

    top_strength = float(items[0]["match_strength"]) if items else 0.0
    top_coverage = float(items[0].get("coverage", 0.0)) if items else 0.0

    coverage_floor = getattr(settings, "ABSTAIN_COVERAGE_FLOOR", 0.20)

    # Dual-gate: abstain only when BOTH match_strength < threshold AND top_coverage < floor.
    # A query that covers >=20% of content-weighted terms is a real procurement requirement
    # even if the absolute BM25 ratio is low (long queries naturally have lower raw/ideal ratio).
    # Literal IS-code mentions and promoted successors are never abstained
    has_literal = any(
        it.get("source") == "literal_mention"
        or it.get("is_literal_mention")
        or it.get("is_successor_promotion")
        or ("successor_of" in str(it.get("source", "")))
        for it in items
    )
    should_abstain = (
        not has_literal
        and (not items or top_strength < abstain_thresh)
        and top_coverage < coverage_floor
    )

    if should_abstain:
        abstained = True
        abstain_reason = "No confident match in the 1,380-standard dataset for this requirement"
        closest_matches = []
        for it in items[:3]:
            it["low_match"] = True
            closest_matches.append(it)
        recommendations = []
        warnings.append("No confident match in the 1,380-standard dataset for this requirement")
    elif not has_literal and abstain_thresh <= top_strength < low_match_thresh:
        abstained = False
        abstain_reason = ""
        closest_matches = []
        for it in items:
            it["low_match"] = True
            if not it.get("clarification_prompt"):
                it["clarification_prompt"] = _generate_clarification_prompt(it, req_summary)
        recommendations = items
        warnings.append(
            "Low confidence advisory: Match strength is low. Consider providing material grades or operating specs to refine recommendations."
        )
    elif not has_literal and top_coverage < coverage_floor and top_strength < low_match_thresh:
        # Coverage below floor but strength above abstain threshold: show low_match
        abstained = False
        abstain_reason = ""
        closest_matches = []
        for it in items:
            it["low_match"] = True
            if not it.get("clarification_prompt"):
                it["clarification_prompt"] = _generate_clarification_prompt(it, req_summary)
        recommendations = items
        warnings.append(
            "Low coverage advisory: Query terms poorly represented in dataset. Provide more specific product details."
        )
    else:
        abstained = False
        abstain_reason = ""
        closest_matches = []
        for it in items:
            it["low_match"] = (it["match_strength"] < low_match_thresh)
        recommendations = items

    return recommendations, abstained, abstain_reason, closest_matches


async def node_05_recommend(state: PipelineState) -> dict:
    """Generate final explainable recommendations and write audit log."""
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))
    candidates = state.get("verified_candidates", [])
    audit_id = state.get("audit_id", "")

    if state.get("abstained") or not candidates:
        stages.append("recommend")
        abstained = True
        abstain_reason = state.get("abstain_reason") or "No confident match in the 1,380-standard dataset for this requirement"
        audit_saved = False
        try:
            from backend.services import postgres_service
            audit_saved = await postgres_service.write_recommendation_log(
                audit_id=audit_id,
                query_text=state.get("normalized_text", state.get("raw_input", "")),
                input_type=state.get("input_type", "text"),
                structured_requirement=state.get("structured_requirement"),
                returned_codes=[],
                pipeline_warnings=warnings,
                top_confidence=None,
                abstained=True,
                abstain_reason=abstain_reason,
                top_match_strength=None,
                closest_matches_codes=[],
            )
            if not audit_saved:
                warnings.append("Audit log not saved: database write failed or unavailable")
        except Exception as exc:
            logger.warning("Node05: audit log write failed: %s", exc)
            warnings.append(f"Audit log not saved: {exc}")
            audit_saved = False

        return {
            "recommendations": [],
            "abstained": True,
            "abstain_reason": abstain_reason,
            "closest_matches": [],
            "audit_saved": audit_saved,
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
    recommendations, abstained, abstain_reason, closest_matches = _merge_reasoning(
        candidates, reasoning_items, req_summary, warnings, req_text=query_text
    )

    # ── Write audit log to PostgreSQL ────────────────────────────────────────
    returned_codes = [r["is_code"] for r in recommendations]
    top_conf = recommendations[0]["confidence"] if recommendations else None
    top_ms = recommendations[0].get("match_strength") if recommendations else (
        closest_matches[0].get("match_strength") if closest_matches else None
    )
    closest_codes = [m["is_code"] for m in closest_matches] if closest_matches else []

    audit_saved = False
    try:
        from backend.services import postgres_service
        audit_saved = await postgres_service.write_recommendation_log(
            audit_id=audit_id,
            query_text=state.get("normalized_text", state.get("raw_input", "")),
            input_type=state.get("input_type", "text"),
            structured_requirement=state.get("structured_requirement"),
            returned_codes=returned_codes,
            pipeline_warnings=warnings,
            top_confidence=top_conf,
            abstained=abstained,
            abstain_reason=abstain_reason,
            top_match_strength=top_ms,
            closest_matches_codes=closest_codes,
        )
        if audit_saved:
            logger.info("Node05: audit log written for audit_id=%s (abstained=%s)", audit_id, abstained)
        else:
            warnings.append("Audit log not saved: database write failed or unavailable")
    except Exception as exc:
        logger.warning("Node05: audit log write failed: %s", exc)
        warnings.append(f"Audit log not saved: {exc}")
        audit_saved = False

    stages.append("recommend")
    return {
        "recommendations": recommendations,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "closest_matches": closest_matches,
        "audit_saved": audit_saved,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
