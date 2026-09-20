"""
Node 03 — Retrieve

True hybrid retrieval pipeline:
  1. Parallel retrieval (asyncio.gather) across:
     - (a) Vector search on standards_vectors (Qdrant)
     - (b) Lexical search (PostgreSQL FTS if available, else in-memory BM25)
     - (c) Clause-level search on standards_fulltext_chunks (Qdrant, aggregated to parent_key)
  2. Reciprocal Rank Fusion (RRF, k=60):
     RRF_score(d) = sum_{m in sources} 1 / (60 + rank_m(d))
     Normalized to [0, 1] for downstream compatibility.
     Per-source ranks saved to candidate["retrieval_trace"].
  3. Literal code boost (+0.15, injected at 0.9).
  4. Thesaurus expansions (+0.05, injected at 0.55).
  5. Knowledge graph expansion (Neo4j if available, else in-memory relationships).
     Used ONLY for related_standards metadata, NEVER for ranking.
     Ordered: requires_testing > requires_safety > allied > performance > references > others.
     Capped at 5 per candidate and labeled.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ai.knowledge import knowledge_loader as kl
from ai.llm.llm_factory import get_embedder
from ai.pipeline.state import PipelineState
from backend.config.settings import settings

logger = logging.getLogger(__name__)


def _build_query_text(state: PipelineState) -> str:
    """Compose a rich query string from the structured requirement."""
    req: Dict = state.get("structured_requirement", {})
    parts = []
    for field in [
        "product", "material", "specifications",
        "performance_requirements", "safety_requirements", "application"
    ]:
        val = req.get(field)
        if val:
            parts.append(str(val))
    norm = state.get("normalized_text", "")
    if norm:
        parts.append(norm[:500])
    return " ".join(parts) or state.get("raw_input", "")


def _order_and_cap_relationships(
    relations: List[Dict[str, Any]], cap: int = 5
) -> List[Dict[str, Any]]:
    """
    Orders related standards by relationship priority:
      requires_testing > requires_safety > allied > performance > references > others
    Caps at 5 per candidate, dedupes by key, and ensures human-readable labels.
    Used ONLY for related_standards metadata, NEVER for ranking.
    """
    PRIORITY = {
        "requires_testing": 1,
        "requires_safety": 2,
        "allied": 3,
        "performance": 4,
        "references": 5,
    }

    seen_keys = set()
    deduped = []
    for r in relations:
        k = r.get("key")
        if not k or k in seen_keys:
            continue
        seen_keys.add(k)
        raw_type = (r.get("relationship_type") or "references").strip()
        type_lower = raw_type.lower()
        label = r.get("label") or type_lower.replace("_", " ").title()
        deduped.append({
            "key": k,
            "display_code": r.get("display_code") or k,
            "title": r.get("title") or "",
            "relationship_type": type_lower,
            "label": label,
        })

    def _sort_key(item: Dict[str, Any]):
        rtype = item["relationship_type"]
        prio = PRIORITY.get(rtype, 99)
        return (prio, rtype, item["key"])

    deduped.sort(key=_sort_key)
    return deduped[:cap]


def _boost_literal_mentions(
    candidates: List[Dict[str, Any]],
    literal_codes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Boost score (+0.15) for candidates literally cited by the user in text.
    Inject missing ones at 0.9 with source='literal_mention'.
    Attaches cited_year and literal_code info.
    """
    if not literal_codes:
        return candidates

    code_map = {item["key"]: item for item in literal_codes if item.get("key")}
    candidate_keys = {c["key"] for c in candidates if c.get("key")}

    boosted = []
    for c in candidates:
        key = c.get("key")
        if key in code_map:
            item = code_map[key]
            new_score = round(min(1.0, float(c.get("score", 0.0)) + 0.15), 4)
            boosted.append({
                **c,
                "score": new_score,
                "source": c.get("source") or "literal_mention",
                "is_literal_mention": True,
                "cited_year": item.get("cited_year"),
                "literal_code_info": item,
            })
        else:
            boosted.append(c)

    # Inject missing ones at 0.9
    for key, item in code_map.items():
        if key not in candidate_keys:
            record = kl.get_standard(key)
            if record:
                dq = record.get("data_quality") or {}
                boosted.append({
                    "key": key,
                    "display_code": record.get("display_code", key),
                    "title": record.get("title", ""),
                    "category": record.get("category"),
                    "subcategory": record.get("subcategory"),
                    "verification_level": dq.get("verification_level", "single_source_unconfirmed"),
                    "status": record.get("status", "ACTIVE"),
                    "superseded_by": record.get("superseded_by") or [],
                    "flags": dq.get("flags", []),
                    "score": 0.9,
                    "source": "literal_mention",
                    "is_literal_mention": True,
                    "cited_year": item.get("cited_year"),
                    "literal_code_info": item,
                    "retrieval_trace": {"literal_mention": 1},
                    "related_standards": _order_and_cap_relationships(
                        kl.get_related_standards_in_memory(key)
                    ),
                })

    boosted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return boosted


def _apply_thesaurus_hints(
    candidates: List[Dict[str, Any]],
    thesaurus_keys: List[str],
) -> List[Dict[str, Any]]:
    """
    Separate small step for thesaurus expansions:
    - Boost +0.05 if already retrieved
    - Inject at 0.55 with source='thesaurus_hint' if not retrieved
    - Thesaurus-injected items must never outrank a candidate with score >= 0.70
    """
    if not thesaurus_keys:
        return candidates

    candidate_keys = {c["key"] for c in candidates if c.get("key")}
    updated = []
    for c in candidates:
        if c.get("key") in thesaurus_keys and not c.get("is_literal_mention"):
            new_score = round(min(1.0, float(c.get("score", 0.0)) + 0.05), 4)
            updated.append({**c, "score": new_score})
        else:
            updated.append(c)

    # Inject missing ones at 0.55
    for key in thesaurus_keys:
        if key not in candidate_keys:
            record = kl.get_standard(key)
            if record:
                dq = record.get("data_quality") or {}
                updated.append({
                    "key": key,
                    "display_code": record.get("display_code", key),
                    "title": record.get("title", ""),
                    "category": record.get("category"),
                    "subcategory": record.get("subcategory"),
                    "verification_level": dq.get("verification_level", "single_source_unconfirmed"),
                    "status": record.get("status", "ACTIVE"),
                    "superseded_by": record.get("superseded_by") or [],
                    "flags": dq.get("flags", []),
                    "score": 0.55,
                    "source": "thesaurus_hint",
                    "retrieval_trace": {"thesaurus_hint": 1},
                    "related_standards": _order_and_cap_relationships(
                        kl.get_related_standards_in_memory(key)
                    ),
                })

    updated.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return updated


# ── Parallel Retrieval Sources ───────────────────────────────────────────────

async def _fetch_vector(query_vec: Optional[List[float]], top_k: int) -> List[Dict[str, Any]]:
    """Source (a): semantic search on standards_vectors via Qdrant."""
    if not query_vec:
        raise RuntimeError("Vector embedding not available")
    from backend.services import qdrant_service
    avail, _ = await qdrant_service.is_available()
    if not avail:
        raise RuntimeError("Qdrant unavailable")
    return await asyncio.wait_for(
        qdrant_service.search_standards(query_vec, top_k=top_k),
        timeout=settings.QDRANT_TIMEOUT_S,
    )


async def _fetch_lexical(query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """Source (b): lexical search via PostgreSQL FTS or in-memory BM25."""
    try:
        from backend.services import postgres_service
        if await postgres_service.is_available():
            hits = await postgres_service.keyword_search_standards(query_text, limit=top_k)
            if hits:
                return hits
    except Exception as exc:
        logger.debug("PostgreSQL FTS unavailable: %s", exc)

    # In-memory BM25 fallback
    return kl.search_standards_in_memory(query_text, top_k=top_k)


async def _fetch_clause(query_vec: Optional[List[float]], top_k: int) -> List[Dict[str, Any]]:
    """
    Source (c): clause-level vector search on standards_fulltext_chunks.
    Aggregated to parent_key, retaining best chunk text as evidence_clause.
    """
    if not query_vec:
        raise RuntimeError("Vector embedding not available")
    from backend.services import qdrant_service
    avail, _ = await qdrant_service.is_available()
    if not avail:
        raise RuntimeError("Qdrant unavailable")

    raw_chunks = await asyncio.wait_for(
        qdrant_service.search_fulltext_chunks(query_vec, top_k=top_k * 2),
        timeout=settings.QDRANT_TIMEOUT_S,
    )
    if not raw_chunks:
        return []

    # Aggregate by parent_key, keeping chunk with highest score
    by_parent: Dict[str, Dict[str, Any]] = {}
    for chunk in raw_chunks:
        parent_key = chunk.get("parent_key")
        if not parent_key:
            continue
        score = float(chunk.get("score", 0.0))
        chunk_text = chunk.get("chunk_text") or ""
        if parent_key not in by_parent or score > by_parent[parent_key]["score"]:
            by_parent[parent_key] = {
                "key": parent_key,
                "score": score,
                "evidence_clause": chunk_text,
                "display_code": chunk.get("display_code") or parent_key,
                "source": "qdrant_fulltext",
            }

    aggregated = sorted(by_parent.values(), key=lambda x: x["score"], reverse=True)
    return aggregated[:top_k]


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    sources_results: Dict[str, List[Dict[str, Any]]],
    k: int = 60,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Reciprocal Rank Fusion over multiple retrieval sources.
      RRF_score(d) = sum_{m in sources} 1 / (k + rank_m(d))
    where rank_m(d) is 1-indexed.
    Returns (fused_candidates, active_sources).
    """
    active_sources = [m for m, hits in sources_results.items() if hits]
    if not active_sources:
        return [], []

    rrf_scores: Dict[str, float] = defaultdict(float)
    traces: Dict[str, Dict[str, int]] = defaultdict(dict)
    evidence_clauses: Dict[str, str] = {}
    item_payloads: Dict[str, Dict[str, Any]] = {}

    for source_name in active_sources:
        hits = sources_results[source_name]
        for idx, hit in enumerate(hits):
            rank = idx + 1
            key = hit.get("key")
            if not key:
                continue
            rrf_scores[key] += 1.0 / (k + rank)
            traces[key][source_name] = rank
            if hit.get("evidence_clause") and key not in evidence_clauses:
                evidence_clauses[key] = hit["evidence_clause"]
            if key not in item_payloads:
                item_payloads[key] = hit

    # Theoretical maximum score if an item was #1 across all active sources
    max_possible = len(active_sources) / (k + 1)

    # Sort descending by RRF score, tie-break by key
    sorted_keys = sorted(
        rrf_scores.keys(),
        key=lambda k_val: (rrf_scores[k_val], k_val),
        reverse=True,
    )

    fused_candidates: List[Dict[str, Any]] = []
    for key in sorted_keys:
        raw_rrf = rrf_scores[key]
        normalized_score = round(raw_rrf / max_possible, 4) if max_possible > 0 else 0.0
        trace = traces[key]
        source_label = "+".join(sorted(trace.keys()))

        record = kl.get_standard(key) or item_payloads.get(key, {})
        dq = record.get("data_quality") or {}

        cand = {
            "key": key,
            "display_code": record.get("display_code") or key,
            "title": record.get("title", ""),
            "category": record.get("category"),
            "subcategory": record.get("subcategory"),
            "verification_level": dq.get("verification_level", record.get("verification_level", "single_source_unconfirmed")),
            "status": record.get("status", "ACTIVE"),
            "superseded_by": record.get("superseded_by") or [],
            "flags": dq.get("flags", record.get("flags", [])),
            "score": normalized_score,
            "rrf_score": round(raw_rrf, 6),
            "retrieval_trace": trace,
            "source": source_label,
            "related_standards": _order_and_cap_relationships(
                record.get("related_standards") or kl.get_related_standards_in_memory(key)
            ),
        }
        if key in evidence_clauses:
            cand["evidence_clause"] = evidence_clauses[key]

        fused_candidates.append(cand)

    return fused_candidates, active_sources


# ── Node 03 Main Entrypoint ──────────────────────────────────────────────────

async def node_03_retrieve(state: PipelineState) -> dict:
    """True hybrid vector + lexical + clause retrieval with RRF and graceful degradation."""
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))
    top_k = settings.RETRIEVE_TOP_K

    query_text = _build_query_text(state)
    literal_codes = state.get("literal_codes", [])
    thesaurus_keys = state.get("thesaurus_expansions", [])

    if not kl._loaded:
        kl.load_all()

    from ai.knowledge.text_utils import tokenize
    if not tokenize(query_text):
        warnings.append("Query has no content terms")
        stages.append("retrieve")
        return {
            "candidates": [],
            "retrieval_sources_used": [],
            "pipeline_warnings": warnings,
            "stages_completed": stages,
        }

    # Attempt query vector generation if Qdrant is potentially available
    query_vec: Optional[List[float]] = None
    try:
        from backend.services import qdrant_service
        avail, _ = await qdrant_service.is_available()
        if avail:
            embedder = get_embedder()
            query_vec = embedder.encode_one(query_text)
    except Exception as exc:
        logger.debug("Node03: Query embedding generation skipped/failed: %s", exc)

    retrieve_top_n = getattr(settings, "RETRIEVE_CANDIDATES", 20)
    final_top_k = getattr(settings, "FINAL_TOP_K", 10)
    req = state.get("structured_requirement", {})
    category_hint = req.get("category_hint")

    # ── Step 1: Run all 3 retrieval branches in parallel ─────────────────────
    vector_task = _fetch_vector(query_vec, retrieve_top_n)
    lexical_task = _fetch_lexical(query_text, retrieve_top_n)
    clause_task = _fetch_clause(query_vec, retrieve_top_n)

    results = await asyncio.gather(vector_task, lexical_task, clause_task, return_exceptions=True)
    res_vec, res_lex, res_clause = results

    sources_results: Dict[str, List[Dict[str, Any]]] = {}

    # Check vector search outcome
    if isinstance(res_vec, Exception) or not res_vec:
        if isinstance(res_vec, Exception):
            logger.debug("Vector search error: %s", res_vec)
        warnings.append("Vector store (Qdrant) unavailable — vector retrieval skipped")
    else:
        sources_results["vector"] = res_vec

    # Check lexical search outcome
    if isinstance(res_lex, Exception) or not res_lex:
        if isinstance(res_lex, Exception):
            logger.debug("Lexical search error: %s", res_lex)
        warnings.append("Lexical search unavailable — lexical retrieval skipped")
    else:
        sources_results["lexical"] = res_lex

    # Check clause-level search outcome
    if isinstance(res_clause, Exception) or not res_clause:
        if isinstance(res_clause, Exception):
            logger.debug("Clause search error: %s", res_clause)
        warnings.append("Clause-level vector store unavailable — clause retrieval skipped")
    else:
        sources_results["clause"] = res_clause

    # ── Step 2: Reciprocal Rank Fusion ───────────────────────────────────────
    candidates, retrieval_sources_used = _reciprocal_rank_fusion(sources_results, k=60)
    logger.info("Node03: RRF fused %d candidates from sources: %s", len(candidates), retrieval_sources_used)

    # ── Step 3: Semantic/Lexical Reranker + Category Filter/Boost ─────────────
    from ai.pipeline.reranker import rerank
    literal_keys = {lc["key"] for lc in literal_codes if lc.get("key")}
    candidates, rerank_warnings = rerank(
        query=query_text,
        candidates=candidates,
        top_n=retrieve_top_n,
        structured_requirement=req,
        category_hint=category_hint,
        literal_keys=literal_keys,
    )
    for rw in rerank_warnings:
        if rw not in warnings:
            warnings.append(rw)

    # ── Step 4: Boost literal IS code mentions ───────────────────────────────
    if literal_codes:
        candidates = _boost_literal_mentions(candidates, literal_codes)

    # ── Step 5: Apply thesaurus expansions ──────────────────────────────────
    if thesaurus_keys:
        candidates = _apply_thesaurus_hints(candidates, thesaurus_keys)

    # Keep final_top_k after boosts
    candidates = candidates[:final_top_k]

    # ── Step 5: Knowledge Graph Expansion (Neo4j / in-memory) ─────────────────
    if candidates:
        try:
            from backend.services import neo4j_service
            avail, _ = await neo4j_service.is_available()
            if avail:
                enriched = []
                for cand in candidates:
                    key = cand.get("key")
                    neighbours: List[Dict] = []
                    if key:
                        try:
                            raw_neighbours = await neo4j_service.expand_standard(
                                key, max_hops=settings.GRAPH_MAX_HOPS
                            )
                            neighbours = _order_and_cap_relationships(raw_neighbours)
                        except Exception as exc:
                            logger.debug("Graph expand failed for %s: %s", key, exc)
                    enriched.append({**cand, "related_standards": neighbours})
                candidates = enriched
                logger.info("Node03: Neo4j graph expansion complete")
            else:
                raise RuntimeError("Neo4j ping failed")
        except Exception as exc:
            logger.debug("Node03: Neo4j unavailable (%s) — using in-memory relationship graph", exc)
            warnings.append(
                "Neo4j offline — populated related standards from in-memory BIS relationships graph"
            )
            candidates = [
                {
                    **c,
                    "related_standards": _order_and_cap_relationships(
                        c.get("related_standards") or kl.get_related_standards_in_memory(c.get("key", ""))
                    ),
                }
                for c in candidates
            ]

    stages.append("retrieve")
    return {
        "candidates": candidates,
        "retrieval_sources_used": retrieval_sources_used,
        "pipeline_warnings": warnings,
        "stages_completed": stages,
    }
