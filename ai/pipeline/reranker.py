"""
Reranker module for Node 03.

Combines:
  1. Semantic Cross-Encoder (BAAI/bge-reranker-base) or lexical BM25 fallback
  2. Fused retrieval score (from RRF)
  3. Domain category soft boost (+0.05) & hard filtering (safe: leaves >= 5 candidates)
  4. Formula: relevance_score = 0.7 * minmax(rerank) + 0.3 * minmax(fused)
"""
from __future__ import annotations

from collections import defaultdict
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ai.knowledge import knowledge_loader as kl
from ai.knowledge.text_utils import tokenize
from backend.config.settings import settings

logger = logging.getLogger(__name__)

_cross_encoder = None
_cross_encoder_attempted = False


def _is_model_cached(model_name: str) -> bool:
    """Check if model exists in local filesystem or HuggingFace hub cache."""
    if os.path.exists(model_name):
        return True
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    folder_name = "models--" + model_name.replace("/", "--")
    return os.path.exists(os.path.join(hf_home, folder_name))


def get_cross_encoder():
    """Lazy load CrossEncoder with offline graceful degradation."""
    global _cross_encoder, _cross_encoder_attempted
    if _cross_encoder_attempted:
        return _cross_encoder
    _cross_encoder_attempted = True

    if not getattr(settings, "RERANKER_ENABLED", True):
        return None

    # In mock offline mode, avoid heavy CrossEncoder CPU inference/hangs to keep 75 eval queries < 40s
    if os.getenv("LLM_PROVIDER") == "mock" and os.getenv("USE_CROSS_ENCODER_IN_MOCK") != "1":
        logger.info("Mock offline mode: using fast BM25 lexical reranker")
        _cross_encoder = None
        return None

    try:
        from sentence_transformers import CrossEncoder

        try:
            _cross_encoder = CrossEncoder(settings.RERANKER_MODEL, max_length=512, local_files_only=True)
            logger.info("Loaded CrossEncoder from local cache: %s", settings.RERANKER_MODEL)
        except Exception:
            if os.getenv("DOWNLOAD_CROSS_ENCODER") == "1":
                logger.info("Downloading CrossEncoder: %s", settings.RERANKER_MODEL)
                _cross_encoder = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
            else:
                logger.info(
                    "CrossEncoder weights for %s not cached locally — using fast BM25 lexical reranker (set DOWNLOAD_CROSS_ENCODER=1 to download)",
                    settings.RERANKER_MODEL,
                )
                _cross_encoder = None
    except Exception as exc:
        logger.info("CrossEncoder unavailable (%s) — using lexical reranker fallback", exc)
        _cross_encoder = None

    return _cross_encoder


def _min_max_scale(scores: List[float]) -> List[float]:
    """Min-max normalise a list of scores to [0.0, 1.0]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 for _ in scores]
    diff = max_s - min_s
    return [round((s - min_s) / diff, 4) for s in scores]


def _lexical_rerank_score(query_tokens: Set[str], std_record: Dict[str, Any]) -> float:
    """Fast lexical BM25-style scorer over structured standard fields."""
    if not query_tokens or not std_record:
        return 0.0

    score = 0.0
    # Title match (weight 3.0)
    title_tokens = set(tokenize(std_record.get("title") or ""))
    overlap_title = query_tokens.intersection(title_tokens)
    score += len(overlap_title) * 3.0

    # Keywords match (weight 2.0)
    kw_tokens = set(tokenize(" ".join(std_record.get("keywords") or [])))
    overlap_kw = query_tokens.intersection(kw_tokens)
    score += len(overlap_kw) * 2.0

    # Scope match (weight 1.5)
    scope_tokens = set(tokenize(std_record.get("scope") or ""))
    overlap_scope = query_tokens.intersection(scope_tokens)
    score += len(overlap_scope) * 1.5

    # Category match (weight 1.0)
    cat_str = f"{std_record.get('category') or ''} {std_record.get('subcategory') or ''}"
    cat_tokens = set(tokenize(cat_str))
    overlap_cat = query_tokens.intersection(cat_tokens)
    score += len(overlap_cat) * 1.0

    return score


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 10,
    structured_requirement: Optional[Dict[str, Any]] = None,
    category_hint: Optional[str] = None,
    literal_keys: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Reranks candidates using CrossEncoder or BM25 fallback, applies category soft boost
    and safe hard filtering, and returns top_n candidates.
    """
    if not candidates:
        return [], []

    warnings: List[str] = []
    encoder = get_cross_encoder()
    raw_rerank_scores: List[float] = []

    if encoder is not None:
        try:
            pairs = []
            for cand in candidates:
                key = cand.get("key", "")
                std = kl.get_standard(key) or cand
                title = std.get("title") or ""
                scope = (std.get("scope") or "")[:500]
                keywords = " ".join(std.get("keywords") or [])
                cand_text = f"{title} {scope} {keywords}".strip()
                pairs.append([query, cand_text])

            scores = encoder.predict(pairs)
            raw_rerank_scores = [float(s) for s in scores]
        except Exception as exc:
            logger.warning("CrossEncoder prediction failed: %s — falling back to lexical", exc)
            warnings.append("Cross-encoder unavailable - lexical rerank used")
            encoder = None

    if encoder is None:
        warnings.append("Cross-encoder unavailable - lexical rerank used")
        # Build query tokens from query text + structured requirement
        req_text_parts = []
        if structured_requirement:
            for field in ["product", "material", "specifications", "application"]:
                val = structured_requirement.get(field)
                if val:
                    req_text_parts.append(str(val))
        rich_query = f"{query} {' '.join(req_text_parts)}"
        q_tokens = set(tokenize(rich_query))

        for cand in candidates:
            key = cand.get("key", "")
            std = kl.get_standard(key) or cand
            lex_score = _lexical_rerank_score(q_tokens, std)
            raw_rerank_scores.append(lex_score)

    # Min-max scale rerank scores and fused retrieval scores
    raw_fused_scores = [float(c.get("score", 0.0)) for c in candidates]
    norm_rerank = _min_max_scale(raw_rerank_scores)
    norm_fused = _min_max_scale(raw_fused_scores)

    w_rerank = getattr(settings, "RERANK_WEIGHT_RERANKER", 0.7)
    w_fused = getattr(settings, "RERANK_WEIGHT_FUSED", 0.3)
    lit_keys = literal_keys or set()

    for idx, cand in enumerate(candidates):
        r_score = norm_rerank[idx]
        f_score = norm_fused[idx]
        fused_relevance = round(w_rerank * r_score + w_fused * f_score, 4)

        # ── Absolute match strength calculation ──
        raw_bm25 = cand.get("raw_score")
        ideal_score = cand.get("ideal_score")
        coverage = cand.get("coverage")
        if raw_bm25 is None or ideal_score is None or coverage is None:
            raw_bm25, ideal_score, coverage = kl.BM25_INDEX.score_document(query, cand.get("key", ""))

        base_bm25_strength = 0.5 * (raw_bm25 / ideal_score if ideal_score > 0 else 0.0) + 0.5 * coverage
        base_bm25_strength = min(1.0, max(0.0, base_bm25_strength))

        if encoder is not None:
            raw_logit = raw_rerank_scores[idx]
            abs_ce = 1.0 / (1.0 + math.exp(-raw_logit))
            ms = 0.6 * base_bm25_strength + 0.4 * abs_ce
            match_strength = min(1.0, max(0.0, ms))
        elif cand.get("vector_score") is not None:
            abs_vec = min(1.0, max(0.0, float(cand["vector_score"])))
            ms = 0.6 * base_bm25_strength + 0.4 * abs_vec
            match_strength = min(1.0, max(0.0, ms))
        else:
            match_strength = base_bm25_strength

        # Literal code mentions should have high absolute strength and high score
        if cand.get("is_literal_mention") or cand.get("key") in lit_keys:
            match_strength = max(match_strength, 0.95)
            fused_relevance = max(fused_relevance, 0.95)

        cand["fused_score"] = fused_relevance
        cand["raw_rerank_score"] = raw_rerank_scores[idx]
        cand["raw_bm25"] = raw_bm25
        cand["ideal_score"] = ideal_score
        cand["coverage"] = coverage
        cand["match_strength"] = round(match_strength, 4)
        cand["relevance_score"] = round(match_strength, 4)
        cand["score"] = fused_relevance

    # Category soft boost (Phase 4.2: hint must never remove candidates, soft boost only)
    lit_keys = literal_keys or set()
    if category_hint:
        catalog_cats = set(kl.map_category_hint_to_catalog(category_hint))
        if catalog_cats:
            cat_boost = getattr(settings, "CATEGORY_BOOST", 0.05)
            for cand in candidates:
                cand_cat = cand.get("category")
                if cand_cat and (
                    cand_cat in catalog_cats
                    or any(c.lower() in cand_cat.lower() for c in catalog_cats)
                ):
                    cand["score"] = round(min(1.0, cand["score"] + cat_boost), 4)
                    cand["match_strength"] = round(min(1.0, cand["match_strength"] + cat_boost), 4)
                    cand["relevance_score"] = cand["match_strength"]
                    cand["category_boosted"] = True

    # Apply negative keyword suppression
    candidates, neg_warnings = _apply_negative_keyword_rules(query, candidates, lit_keys)
    warnings.extend(neg_warnings)

    # Sort descending: literal mentions first, then by score descending, then match_strength descending
    candidates.sort(
        key=lambda c: (
            0 if (c.get("is_literal_mention") or c.get("key") in lit_keys) else 1,
            -float(c.get("score", 0.0)),
            -float(c.get("match_strength", 0.0)),
            c.get("key", ""),
        )
    )

    return candidates[:top_n], warnings


def _matches_trigger_word_boundary(trigger: str, text: str) -> bool:
    """Match trigger using word boundaries, avoiding substring collisions like 'lamp' in 'clamp'."""
    pattern = r"\b" + re.escape(trigger.lower().strip()) + r"\b"
    return bool(re.search(pattern, text.lower()))


def _apply_negative_keyword_rules(
    query: str,
    candidates: List[Dict[str, Any]],
    literal_keys: Set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Apply domain-specific negative keyword suppression rules.
    If query matches any trigger keyword (word boundary), penalize specified standards.
    Total penalty per candidate is capped at 0.25.
    NEVER penalizes literal mentions or standards in literal_keys.
    """
    rules = getattr(kl, "NEGATIVE_KEYWORD_RULES", [])
    if not rules:
        return candidates, []

    warnings = []
    penalized_penalties: Dict[str, float] = defaultdict(float)
    reasons: Dict[str, str] = {}

    for rule in rules:
        triggers = rule.get("trigger_keywords", [])
        if any(_matches_trigger_word_boundary(trig, query) for trig in triggers):
            penalize_keys = set(rule.get("penalize_standards", []))
            penalty = float(rule.get("penalty", 0.20))
            reason = rule.get("reason", "")
            for p_key in penalize_keys:
                penalized_penalties[p_key] += penalty
                if reason and p_key not in reasons:
                    reasons[p_key] = reason

    if not penalized_penalties:
        return candidates, []

    for c in candidates:
        key = c.get("key", "")
        # Hard rule: NEVER penalize literal mentions
        if c.get("is_literal_mention") or key in literal_keys:
            continue
        if key in penalized_penalties:
            p_val = min(0.25, round(penalized_penalties[key], 4))  # Capped at 0.25
            old_score = float(c.get("relevance_score", c.get("score", 0.0)))
            new_score = round(max(0.01, old_score - p_val), 4)
            c["relevance_score"] = new_score
            c["score"] = new_score
            c["negative_penalty_applied"] = p_val
            c["negative_rule_reason"] = reasons.get(key, "")
            warnings.append(f"Standard {key} demoted by {p_val:.2f}: {reasons.get(key)}")

    return candidates, warnings
