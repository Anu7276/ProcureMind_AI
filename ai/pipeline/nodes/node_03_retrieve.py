"""
Node 03 — Retrieve

Hybrid search strategy:
  1. Qdrant vector similarity on structured requirement
  2. Exact/keyword boost for literal IS codes found in text (Node01)
  3. Neo4j graph traversal to expand top candidates with related standards

Degraded-response behavior (all failures are caught and logged — never raise):
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │ Store Down      │ Behavior                                             │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │ Neo4j only      │ Return Qdrant-only results; graph expansion skipped  │
  │ Qdrant only     │ Keyword fallback via PostgreSQL FTS; no graph expand │
  │ Both down       │ Empty candidates list; warnings surfaced in output   │
  └─────────────────┴──────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

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
    # Also include normalized text snippet for richer signal
    norm = state.get("normalized_text", "")
    if norm:
        parts.append(norm[:500])
    return " ".join(parts) or state.get("raw_input", "")


async def _qdrant_search(query_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    from backend.services import qdrant_service
    return await qdrant_service.search_standards(query_vec, top_k=top_k)


async def _postgres_keyword_fallback(query_text: str, top_k: int) -> List[Dict[str, Any]]:
    from backend.services import postgres_service
    rows = await postgres_service.keyword_search_standards(query_text, limit=top_k)
    return [
        {
            **row,
            "score": 0.5,  # neutral score for keyword results
            "source": "keyword_fallback",
        }
        for row in rows
    ]


async def _neo4j_expand(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add graph-related standards to each candidate."""
    from backend.services import neo4j_service

    enriched = []
    for cand in candidates:
        key = cand.get("key")
        related: List[Dict] = []
        if key:
            try:
                neighbours = await neo4j_service.expand_standard(
                    key, max_hops=settings.GRAPH_MAX_HOPS
                )
                related = [
                    {
                        "key": n.get("key"),
                        "display_code": n.get("display_code"),
                        "title": n.get("title"),
                        "relationship_type": n.get("relationship_type", "RELATED"),
                    }
                    for n in neighbours
                    if n.get("key")
                ]
            except Exception as exc:
                logger.debug("Graph expand failed for %s: %s", key, exc)
        enriched.append({**cand, "related_standards": related})
    return enriched


def _boost_literal_mentions(
    candidates: List[Dict[str, Any]],
    literal_keys: List[str],
) -> List[Dict[str, Any]]:
    """
    Boost score for candidates whose key was literally mentioned in the text.
    Also inject any mentioned standards not already in candidates.
    """
    candidate_keys = {c["key"] for c in candidates if c.get("key")}

    boosted = []
    for c in candidates:
        boost = 0.15 if c.get("key") in literal_keys else 0.0
        boosted.append({**c, "score": min(1.0, c.get("score", 0) + boost)})

    # Inject any mentioned keys not already retrieved
    for key in literal_keys:
        if key not in candidate_keys:
            record = kl.get_standard(key)
            if record:
                boosted.append({
                    "key": key,
                    "display_code": record.get("display_code", key),
                    "title": record.get("title", ""),
                    "category": record.get("category"),
                    "verification_level": record["data_quality"].get("verification_level"),
                    "score": 0.9,  # high confidence — user literally mentioned it
                    "source": "literal_mention",
                    "related_standards": [],
                })

    # Sort by score descending
    boosted.sort(key=lambda x: x.get("score", 0), reverse=True)
    return boosted


async def node_03_retrieve(state: PipelineState) -> dict:
    """Hybrid vector + keyword + graph retrieval with degraded fallbacks."""
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))
    top_k = settings.RETRIEVE_TOP_K

    query_text = _build_query_text(state)
    literal_keys = state.get("thesaurus_expansions", [])
    req = state.get("structured_requirement", {})
    category_hint = req.get("category_hint")

    candidates: List[Dict[str, Any]] = []
    qdrant_available = False
    neo4j_available = False

    # ── Step 1: Vector search (Qdrant) ───────────────────────────────────────
    try:
        from backend.services import qdrant_service
        avail, _ = await qdrant_service.is_available()
        if avail:
            embedder = get_embedder()
            query_vec = embedder.encode_one(query_text)
            candidates = await _qdrant_search(query_vec, top_k)
            qdrant_available = True
            logger.info("Node03: Qdrant returned %d candidates", len(candidates))
        else:
            raise RuntimeError("Qdrant ping failed")

    except Exception as exc:
        logger.warning("Node03: Qdrant unavailable (%s) — falling back to keyword search", exc)
        warnings.append(
            "Qdrant unavailable — using keyword-only fallback from PostgreSQL"
        )

    # ── Step 2: Keyword fallback if Qdrant is down ───────────────────────────
    if not qdrant_available:
        try:
            candidates = await _postgres_keyword_fallback(query_text, top_k)
            if candidates:
                logger.info(
                    "Node03: keyword fallback returned %d candidates", len(candidates)
                )
        except Exception as exc:
            logger.warning("Node03: PostgreSQL keyword fallback also unavailable (%s)", exc)

        if not candidates:
            logger.info("Node03: using in-memory verified BIS standards catalog")
            warnings.append(
                "External stores offline — using in-memory verified BIS standards catalog"
            )
            candidates = kl.search_standards_in_memory(query_text, top_k)

    # ── Step 3: Boost literal IS code mentions ───────────────────────────────
    if literal_keys:
        candidates = _boost_literal_mentions(candidates, literal_keys)

    # Keep top_k after boost
    candidates = candidates[:top_k]

    # ── Step 4: Graph enrichment (Neo4j) ─────────────────────────────────────
    if candidates:
        try:
            from backend.services import neo4j_service
            avail, _ = await neo4j_service.is_available()
            if avail:
                candidates = await _neo4j_expand(candidates)
                neo4j_available = True
                logger.info("Node03: graph expansion complete")
            else:
                raise RuntimeError("Neo4j ping failed")

        except Exception as exc:
            logger.warning("Node03: Neo4j unavailable (%s) — using in-memory relationship graph", exc)
            warnings.append(
                "Neo4j offline — populated related standards from in-memory BIS relationships graph"
            )
            # Use in-memory relationships index
            candidates = [
                {
                    **c,
                    "related_standards": c.get("related_standards") or kl.get_related_standards_in_memory(c.get("key", "")),
                }
                for c in candidates
            ]

    stages.append("retrieve")
    return {
        "candidates": candidates,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
