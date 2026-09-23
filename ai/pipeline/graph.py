"""
LangGraph state graph — wires the 5 pipeline nodes (+ Node0 for file inputs).

Graph topology:
    [start]
       ↓
  document_understanding (Node 0 — always runs; no-op for text inputs)
       ↓
     ingest (Node 01)
       ↓
    extract (Node 02)
       ↓
    retrieve (Node 03)
       ↓
     verify (Node 04)
       ↓
   recommend (Node 05)
       ↓
    [END]

Three compiled graphs are exposed:
  • _compiled_graph          – full 6-node pipeline (Nodes 00→05)
  • _extract_only_graph      – Nodes 00→02 only (document → ingest → extract)
  • _from_requirement_graph  – Nodes 03→05 only (retrieve → verify → recommend)

run_pipeline(), run_extract_only(), and run_pipeline_from_requirement()
are the public entry points.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from ai.pipeline.nodes.node_00_document import node_00_document
from ai.pipeline.nodes.node_01_ingest import node_01_ingest
from ai.pipeline.nodes.node_02_extract import node_02_extract
from ai.pipeline.nodes.node_03_retrieve import node_03_retrieve
from ai.pipeline.nodes.node_04_verify import node_04_verify
from ai.pipeline.nodes.node_05_recommend import node_05_recommend
from ai.pipeline.segmenter import segment_tender, LineItem
from ai.pipeline.state import PipelineState


# ─────────────────────────────────────────────────────────────────────────────
# Graph builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_graph() -> StateGraph:
    """Full 6-node pipeline: Node 00 → 01 → 02 → 03 → 04 → 05."""
    g = StateGraph(PipelineState)

    g.add_node("document_understanding", node_00_document)
    g.add_node("ingest", node_01_ingest)
    g.add_node("extract", node_02_extract)
    g.add_node("retrieve", node_03_retrieve)
    g.add_node("verify", node_04_verify)
    g.add_node("recommend", node_05_recommend)

    g.set_entry_point("document_understanding")
    g.add_edge("document_understanding", "ingest")
    g.add_edge("ingest", "extract")
    g.add_edge("extract", "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_edge("verify", "recommend")
    g.add_edge("recommend", END)

    return g


def _build_extract_only_graph() -> StateGraph:
    """Extraction-only graph: Node 00 → 01 → 02 then END.

    Used by the /ingest endpoint to extract structured requirements from a
    document without running retrieval.  The human review step (ReviewPage)
    then lets the user confirm/edit before calling /recommend.
    """
    g = StateGraph(PipelineState)

    g.add_node("document_understanding", node_00_document)
    g.add_node("ingest", node_01_ingest)
    g.add_node("extract", node_02_extract)

    g.set_entry_point("document_understanding")
    g.add_edge("document_understanding", "ingest")
    g.add_edge("ingest", "extract")
    g.add_edge("extract", END)

    return g


async def ingest_codes_only(state: PipelineState) -> dict:
    """
    Lightweight node for from_requirement graph:
    Derives literal IS codes (with cited_year) and thesaurus_expansions.
    When user_edited is True, scans the structured_requirement as the source of truth.
    When user_edited is False, scans normalized_text.
    """
    from ai.pipeline.nodes.node_01_ingest import extract_literal_codes
    from ai.knowledge import knowledge_loader as kl

    user_edited = state.get("user_edited", False)
    if user_edited:
        req = state.get("structured_requirement", {}) or {}
        text_to_scan = " ".join(str(v) for v in req.values() if v)
    else:
        text_to_scan = state.get("normalized_text", "") or state.get("raw_input", "") or ""

    literal_codes = extract_literal_codes(text_to_scan) if text_to_scan else []
    thesaurus_exp = kl.expand_query_with_thesaurus(text_to_scan) if text_to_scan else []
    thesaurus_hint = ", ".join(thesaurus_exp) if thesaurus_exp else None

    stages = list(state.get("stages_completed", []))
    stages.append("ingest_codes_only")

    return {
        "literal_codes": literal_codes,
        "thesaurus_expansions": thesaurus_exp,
        "thesaurus_hint": thesaurus_hint,
        "stages_completed": stages,
    }


def _build_from_requirement_graph() -> StateGraph:
    """Retrieval + verify + recommend graph: ingest_codes_only → retrieve → verify → recommend.

    Used when a structured requirement is already known (e.g. after human
    review on ReviewPage) and we only need to run the retrieval/ranking half.
    """
    g = StateGraph(PipelineState)

    g.add_node("ingest_codes_only", ingest_codes_only)
    g.add_node("retrieve", node_03_retrieve)
    g.add_node("verify", node_04_verify)
    g.add_node("recommend", node_05_recommend)

    g.set_entry_point("ingest_codes_only")
    g.add_edge("ingest_codes_only", "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_edge("verify", "recommend")
    g.add_edge("recommend", END)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# Compile once at import time — reused across all requests
# ─────────────────────────────────────────────────────────────────────────────

_compiled_graph = _build_full_graph().compile()
_extract_only_graph = _build_extract_only_graph().compile()
_from_requirement_graph = _build_from_requirement_graph().compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    raw_input: str,
    input_type: str = "text",
    raw_bytes: bytes = b"",
    audit_id: str | None = None,
) -> Dict[str, Any]:
    """
    Entry point for the full 6-node pipeline.

    Args:
        raw_input:  Query string (for text) or filename hint (for files).
        input_type: "text" | "pdf" | "docx" | "image"
        raw_bytes:  File content (empty for text input).
        audit_id:   Pre-assigned UUID string; one is generated if not provided.

    Returns:
        Final PipelineState dict with all node outputs.
    """
    if audit_id is None:
        audit_id = str(uuid.uuid4())

    # If text input contains multiple distinct line items, process each item independently
    if input_type == "text" and raw_input and not raw_bytes:
        items = segment_tender(raw_input)
        if len(items) > 1:
            def _item_uuid(parent_id: str, idx: int) -> str:
                try:
                    ns = uuid.UUID(parent_id)
                except Exception:
                    ns = uuid.NAMESPACE_DNS
                return str(uuid.uuid5(ns, f"item_{idx}"))

            async def _run_item(it: LineItem) -> tuple[LineItem, Dict[str, Any]]:
                st: PipelineState = {
                    "raw_input": it.cleaned_text,
                    "raw_bytes": None,
                    "input_type": "text",
                    "audit_id": _item_uuid(audit_id, it.item_index),
                    "suppress_audit": True,
                    "pipeline_warnings": [],
                    "stages_completed": [],
                }
                res = await _compiled_graph.ainvoke(st)
                return it, res

            item_results_pairs = await asyncio.gather(*[_run_item(it) for it in items])

            per_item_results = []
            all_recs_map: Dict[str, Dict[str, Any]] = {}
            all_closest_map: Dict[str, Dict[str, Any]] = {}
            all_warnings: List[str] = []
            all_stages: set[str] = set()
            all_abstained = True

            for it, st in item_results_pairs:
                it_recs = st.get("recommendations", [])
                it_abstained = st.get("abstained", False)
                it_abstain_reason = st.get("abstain_reason")
                it_warnings = st.get("pipeline_warnings", [])
                all_warnings.extend(it_warnings)
                for s in st.get("stages_completed", []):
                    all_stages.add(s)

                if not it_abstained:
                    all_abstained = False

                literal_codes = st.get("literal_codes", [])
                cited_codes_found = [
                    c.get("display_code", c.get("raw", "")) if isinstance(c, dict) else str(c)
                    for c in literal_codes
                ] or [c for c in it.cited_codes]

                per_item_results.append({
                    "item_index": it.item_index,
                    "item_text": it.cleaned_text,
                    "estimated_quantity": it.estimated_quantity,
                    "recommendations": it_recs,
                    "cited_codes_found": cited_codes_found,
                    "warnings": it_warnings,
                    "abstained": it_abstained,
                    "abstain_reason": it_abstain_reason,
                })

                for r in it_recs:
                    k = r.get("key") or r.get("is_code")
                    if not k:
                        continue
                    if k not in all_recs_map:
                        all_recs_map[k] = r
                    else:
                        if r.get("confidence", 0.0) > all_recs_map[k].get("confidence", 0.0):
                            all_recs_map[k] = r

                for r in st.get("closest_matches", []):
                    k = r.get("key") or r.get("is_code")
                    if k and k not in all_closest_map:
                        all_closest_map[k] = r

            merged_recs = sorted(
                all_recs_map.values(),
                key=lambda x: (x.get("confidence", 0.0), x.get("match_strength", 0.0) or 0.0),
                reverse=True,
            )
            merged_closest = sorted(
                all_closest_map.values(),
                key=lambda x: (x.get("confidence", 0.0), x.get("match_strength", 0.0) or 0.0),
                reverse=True,
            )

            merged_returned_codes = [r["is_code"] for r in merged_recs if r.get("is_code")]
            merged_structured_req = {
                "items": [
                    st.get("structured_requirement")
                    for _, st in item_results_pairs
                    if st.get("structured_requirement")
                ]
            }
            top_conf = merged_recs[0]["confidence"] if merged_recs else None
            top_ms = max(
                [r.get("confidence", 0.0) for r in merged_recs]
                + [r.get("match_strength", 0.0) or 0.0 for r in merged_recs],
                default=None,
            ) if merged_recs else (
                merged_closest[0].get("match_strength") if merged_closest else None
            )
            merged_closest_codes = [m["is_code"] for m in merged_closest if m.get("is_code")]

            audit_saved = False
            try:
                from backend.services import postgres_service
                audit_saved = await postgres_service.write_recommendation_log(
                    audit_id=audit_id,
                    query_text=raw_input,
                    input_type=input_type,
                    structured_requirement=merged_structured_req,
                    returned_codes=merged_returned_codes,
                    pipeline_warnings=list(dict.fromkeys(all_warnings)),
                    top_confidence=top_conf,
                    abstained=all_abstained,
                    abstain_reason="All tender items abstained" if all_abstained else None,
                    top_match_strength=top_ms,
                    closest_matches_codes=merged_closest_codes,
                )
                if not audit_saved:
                    all_warnings.append("Audit log not saved: database write failed or unavailable")
            except Exception as exc:
                all_warnings.append(f"Audit log not saved: {exc}")
                audit_saved = False

            stages_list = ["document_understanding", "segment", "extract", "retrieve", "verify", "recommend"]

            return {
                "raw_input": raw_input,
                "raw_bytes": None,
                "input_type": input_type,
                "audit_id": audit_id,
                "recommendations": merged_recs,
                "per_item_results": per_item_results,
                "line_items": [
                    {
                        "item_index": it.item_index,
                        "raw_text": it.raw_text,
                        "cleaned_text": it.cleaned_text,
                        "cited_codes": it.cited_codes,
                        "estimated_quantity": it.estimated_quantity,
                    }
                    for it in items
                ],
                "abstained": all_abstained,
                "abstain_reason": "All tender items abstained" if all_abstained else None,
                "closest_matches": merged_closest,
                "pipeline_warnings": list(dict.fromkeys(all_warnings)),
                "stages_completed": stages_list,
                "audit_saved": audit_saved,
            }

    initial_state: PipelineState = {
        "raw_input": raw_input,
        "raw_bytes": raw_bytes if raw_bytes else None,
        "input_type": input_type,
        "audit_id": audit_id,
        "pipeline_warnings": [],
        "stages_completed": [],
    }

    final_state = await _compiled_graph.ainvoke(initial_state)
    return final_state


async def run_extract_only(
    raw_input: str,
    input_type: str = "text",
    raw_bytes: bytes = b"",
    audit_id: str | None = None,
) -> Dict[str, Any]:
    """
    Run Nodes 00-02 only: document understanding → ingest → extract.

    Used by /ingest to produce a structured requirement for human review.
    Returns the state after Node 02 (structured_requirement populated).
    """
    if audit_id is None:
        audit_id = str(uuid.uuid4())

    initial_state: PipelineState = {
        "raw_input": raw_input,
        "raw_bytes": raw_bytes if raw_bytes else None,
        "input_type": input_type,
        "audit_id": audit_id,
        "pipeline_warnings": [],
        "stages_completed": [],
    }

    final_state = await _extract_only_graph.ainvoke(initial_state)
    return final_state


async def run_pipeline_from_requirement(
    structured_requirement: Dict[str, Any],
    normalized_text: str,
    audit_id: str | None = None,
    language: str = "en",
    input_type: str = "text",
    user_edited: bool = False,
) -> Dict[str, Any]:
    """
    Run ingest_codes_only → Nodes 03–05 using a pre-computed structured requirement.

    Used when /recommend receives confirmed output from a prior /ingest call
    (i.e. after the user has reviewed and confirmed on ReviewPage).
    """
    if audit_id is None:
        audit_id = str(uuid.uuid4())

    initial_state: PipelineState = {
        "raw_input": normalized_text,
        "input_type": input_type or "text",
        "audit_id": audit_id,
        "extracted_text": normalized_text,
        "normalized_text": normalized_text,
        "language": language or "en",
        "thesaurus_expansions": [],
        "structured_requirement": structured_requirement,
        "user_edited": user_edited,
        "pipeline_warnings": [],
        "stages_completed": [],
    }

    final_state = await _from_requirement_graph.ainvoke(initial_state)
    return final_state
