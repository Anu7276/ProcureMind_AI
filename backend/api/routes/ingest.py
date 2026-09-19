"""
POST /ingest

Accepts: multipart file upload OR raw text body.
Runs: Node00 (document understanding) + Node01 (ingest) + Node02 (extract).
Returns: structured requirement for human review before /recommend is called.

This is the "human review checkpoint" from the diagram — the frontend shows
what the pipeline extracted before running retrieval.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai.pipeline.graph import run_pipeline
from backend.schemas.api_schemas import IngestResponse, StructuredRequirement

router = APIRouter(tags=["Pipeline"])

ALLOWED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "image/png": "image",
    "image/jpeg": "image",
    "image/tiff": "image",
    "text/plain": "text",
}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
):
    """
    Submit a tender document (PDF/DOCX/image) or raw text.
    Returns the structured requirement extracted by the pipeline for user review.
    """
    audit_id = str(uuid.uuid4())

    if file is not None:
        # File upload path
        content_type = file.content_type or ""
        input_type = ALLOWED_TYPES.get(content_type)
        if input_type is None:
            # Try extension-based detection
            filename = file.filename or ""
            ext = filename.rsplit(".", 1)[-1].lower()
            ext_map = {"pdf": "pdf", "docx": "docx", "doc": "docx",
                       "png": "image", "jpg": "image", "jpeg": "image",
                       "tif": "image", "tiff": "image", "txt": "text"}
            input_type = ext_map.get(ext)
            if input_type is None:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type: {content_type or ext}. "
                           f"Supported: PDF, DOCX, PNG, JPEG, TXT.",
                )
        raw_bytes = await file.read()
        raw_input = file.filename or "uploaded_file"

    elif text is not None and text.strip():
        # Raw text path
        input_type = "text"
        raw_bytes = b""
        raw_input = text.strip()

    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either a file upload or a text body.",
        )

    # Run pipeline Nodes 00–02 only
    # We run the full graph but stop caring about nodes 03-05 output
    final_state = await run_pipeline(
        raw_input=raw_input,
        input_type=input_type,
        raw_bytes=raw_bytes,
        audit_id=audit_id,
    )

    req_data = final_state.get("structured_requirement", {})
    req = StructuredRequirement(
        product=req_data.get("product"),
        material=req_data.get("material"),
        specifications=req_data.get("specifications"),
        performance_requirements=req_data.get("performance_requirements"),
        safety_requirements=req_data.get("safety_requirements"),
        application=req_data.get("application"),
        category_hint=req_data.get("category_hint"),
        normalized_text=final_state.get("normalized_text", ""),
        language=final_state.get("language", "en"),
        input_type=input_type,
        thesaurus_expansions=final_state.get("thesaurus_expansions", []),
    )

    return IngestResponse(
        audit_id=audit_id,
        structured_requirement=req,
        warnings=final_state.get("pipeline_warnings", []),
    )
