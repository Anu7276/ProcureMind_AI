"""
GET /graph/info — returns metadata about available pipeline graphs.
POST /graph/extract — run extract-only graph (Nodes 00–02) from text.
"""
from __future__ import annotations

import uuid
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.pipeline.graph import run_extract_only, run_pipeline_from_requirement
from backend.schemas.api_schemas import StructuredRequirement

router = APIRouter(prefix="/graph", tags=["Graph"])


class ExtractRequest(BaseModel):
    text: str
    input_type: str = "text"
    audit_id: Optional[str] = None


class ExtractResponse(BaseModel):
    audit_id: str
    structured_requirement: Optional[Dict[str, Any]] = None
    normalized_text: Optional[str] = None
    language: Optional[str] = None
    thesaurus_expansions: list = []
    warnings: list = []
    stages_completed: list = []


@router.get("/info")
async def graph_info():
    """Return metadata about available pipeline graph variants."""
    return {
        "graphs": {
            "full": {
                "nodes": ["document_understanding", "ingest", "extract", "retrieve", "verify", "recommend"],
                "description": "Full 6-node pipeline. Used by POST /recommend with raw_query.",
                "entry_point": "document_understanding",
            },
            "extract_only": {
                "nodes": ["document_understanding", "ingest", "extract"],
                "description": "Extraction-only graph. Used by POST /ingest — returns structured requirement for human review.",
                "entry_point": "document_understanding",
            },
            "from_requirement": {
                "nodes": ["retrieve", "verify", "recommend"],
                "description": "Retrieval+ranking graph. Used by POST /recommend with structured_requirement after human review.",
                "entry_point": "retrieve",
            },
        }
    }


@router.post("/extract", response_model=ExtractResponse)
async def extract_only(body: ExtractRequest):
    """
    Run extraction-only graph (Nodes 00–02) on raw text.
    Returns structured requirement for human review before /recommend.
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text cannot be empty")

    audit_id = body.audit_id or str(uuid.uuid4())

    final_state = await run_extract_only(
        raw_input=text,
        input_type=body.input_type,
        audit_id=audit_id,
    )

    return ExtractResponse(
        audit_id=audit_id,
        structured_requirement=final_state.get("structured_requirement"),
        normalized_text=final_state.get("normalized_text"),
        language=final_state.get("language", "en"),
        thesaurus_expansions=final_state.get("thesaurus_expansions", []),
        warnings=final_state.get("pipeline_warnings", []),
        stages_completed=final_state.get("stages_completed", []),
    )
