"""
Pydantic request / response schemas for the FastAPI layer.
These are separate from the ORM models to keep API contracts stable.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared sub-models ─────────────────────────────────────────────────────────

class CertificationInfo(BaseModel):
    mandatory: Optional[bool] = None
    scheme_name: Optional[str] = None
    scheme_code: Optional[str] = None
    lead_time_weeks: Optional[int] = None
    penalty: Optional[str] = None
    gazette_reference: Optional[str] = None
    enforcement_status: Optional[str] = None
    evidence_source: str = "qco_orders.json"


class RelatedStandard(BaseModel):
    key: str
    display_code: str
    title: str
    relationship_type: str      # e.g. REFERENCES, ALLIED, REQUIRES_TESTING


class RecommendationItem(BaseModel):
    is_code: str                              # e.g. "IS 1786:2008"
    key: str                                  # canonical key e.g. "IS 1786"
    title: str
    confidence: float = Field(ge=0.0, le=1.0)
    verification_level: str                   # verified_multi_source | needs_review | ...
    status: str                               # ACTIVE | WITHDRAWN | SUPERSEDED
    superseded_by: List[str] = []
    flags: List[str] = []
    certification: Optional[CertificationInfo] = None
    related_standards: List[RelatedStandard] = []
    reasoning: str                            # LLM-generated, clause-level
    evidence_sources: List[str] = []          # e.g. ["IS 1786, BIS", "QCO-ELEC-2024-01"]
    data_quality_note: Optional[str] = None   # data quality explanation
    relevance_score: Optional[float] = None   # match_strength [0.0, 1.0]
    version_info: Optional[Dict[str, Any]] = None
    replaced_by: List[str] = []
    match_strength: Optional[float] = None
    low_match: Optional[bool] = None
    spec_line: Optional[str] = None
    clarification_prompt: Optional[str] = None
    matched_terms: List[str] = []
    evidence_clause: Optional[str] = None
    scope: Optional[str] = None


# ── /ingest ───────────────────────────────────────────────────────────────────

class StructuredRequirement(BaseModel):
    """Output of Node01+02 — shown to user for review before /recommend."""
    product: Optional[str] = None
    material: Optional[str] = None
    specifications: Optional[str] = None
    performance_requirements: Optional[str] = None
    safety_requirements: Optional[str] = None
    application: Optional[str] = None
    category_hint: Optional[str] = None
    normalized_text: str = ""
    language: str = "en"
    input_type: str = "text"
    thesaurus_expansions: List[str] = []
    extraction_method: Optional[str] = None
    pages: Optional[int] = None


class IngestResponse(BaseModel):
    audit_id: str
    structured_requirement: StructuredRequirement
    warnings: List[str] = []


# ── /recommend ────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """
    Either pass `structured_requirement` (from a prior /ingest call)
    or `raw_query` to run the full pipeline from scratch.
    """
    raw_query: Optional[str] = None
    structured_requirement: Optional[StructuredRequirement] = None
    audit_id: Optional[str] = None          # carry forward from /ingest if available


class PipelineMeta(BaseModel):
    llm_mode: Optional[str] = None
    llm_model: Optional[str] = None
    embedder: Optional[str] = None
    retrieval_sources_used: List[str] = []
    reranker_mode: Optional[str] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None
    audit_saved: Optional[bool] = None
    warnings: List[str] = []


class RecommendResponse(BaseModel):
    audit_id: str
    query_summary: str
    recommendations: List[RecommendationItem]
    warnings: List[str] = []               # degraded-store warnings
    pipeline_stages_completed: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    abstained: Optional[bool] = False
    abstain_reason: Optional[str] = None
    closest_matches: Optional[List[RecommendationItem]] = []
    pipeline_meta: Optional[PipelineMeta] = None


# ── GET /standard/{key} ───────────────────────────────────────────────────────

class StandardDetail(BaseModel):
    key: str
    display_code: str
    edition_year: Optional[int] = None
    title: str
    scope: Optional[str] = None
    status: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    verification_level: str
    flags: List[str] = []
    has_full_text: bool
    superseded_by: List[str] = []
    keywords: List[str] = []
    compliance: Optional[CertificationInfo] = None
    related_standards: List[RelatedStandard] = []


# ── GET /health ───────────────────────────────────────────────────────────────

class StoreHealth(BaseModel):
    available: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                       # "ok" | "degraded" | "critical"
    stores: Dict[str, StoreHealth]
    llm_provider: str
    embedding_model: str
    standards_loaded: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
