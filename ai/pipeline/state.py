"""
LangGraph pipeline state definition.
Every node receives the full state dict and returns a partial update dict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    raw_input: str              # original text query or filename for file uploads
    raw_bytes: Optional[bytes]  # file bytes for PDF/DOCX/image inputs
    input_type: str             # "text" | "pdf" | "docx" | "image"
    audit_id: str               # UUID string, set at pipeline entry

    # ── Node 00 output — Document Understanding ───────────────────────────────
    # Only populated when input_type != "text"
    extracted_text: Optional[str]       # machine-readable text from file
    doc_structure: Optional[Dict]       # sections, tables identified by Docling

    # ── Node 01 output — Ingest ───────────────────────────────────────────────
    normalized_text: str                # cleaned text (without thesaurus pollution)
    language: str                       # detected language code e.g. "en", "hi"
    literal_codes: List[Dict[str, Any]] # each {raw, key, part, cited_year}
    thesaurus_expansions: List[str]     # IS keys suggested by thesaurus
    thesaurus_hint: Optional[str]       # optional thesaurus hint passed to Node 02 prompt

    # ── Node 02 output — Extract ──────────────────────────────────────────────
    structured_requirement: Dict[str, Any]
    # Keys: product, material, specifications, performance_requirements,
    #       safety_requirements, application, category_hint

    # ── Node 03 output — Retrieve ─────────────────────────────────────────────
    candidates: List[Dict[str, Any]]
    # Each: key, display_code, title, score, category, verification_level,
    #       related_standards (from graph), source, retrieval_trace, evidence_clause
    retrieval_sources_used: List[str]  # e.g. ["vector", "lexical", "clause"]

    # ── Node 04 output — Verify ───────────────────────────────────────────────
    verified_candidates: List[Dict[str, Any]]
    # Each candidate augmented with: status, superseded_by, flags,
    #   verification_level, certification (mandatory/voluntary + scheme + penalty)
    #   whitelist_valid (bool)

    # ── Node 05 output — Recommend ────────────────────────────────────────────
    recommendations: List[Dict[str, Any]]
    # Final ranked list matching RecommendationItem schema

    # ── Cross-cutting ──────────────────────────────────────────────────────────
    pipeline_warnings: List[str]
    # Accumulated non-fatal warnings (degraded store, low-quality data, etc.)
    # Key messages:
    #   "Neo4j unavailable — graph enrichment skipped, results are vector-only"
    #   "Qdrant unavailable — using keyword-only fallback from PostgreSQL"
    #   "All vector stores unavailable — no candidates retrieved"
    #   "PostgreSQL unavailable — compliance/QCO data not available for this result"

    stages_completed: List[str]
    # e.g. ["document_understanding", "ingest", "extract", "retrieve", "verify", "recommend"]
