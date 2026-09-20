"""
POST /ingest

Accepts: multipart file upload OR raw text body.
Runs: Document Understanding (Node00) + Ingest (Node01) + Extract (Node02).
Returns: structured requirement for human review before /recommend is called.

Enforces strict magic byte validation, 10 MB file limits, and 20k char text limits.
Never returns HTTP 200 with unreadable junk or raw PDF byte streams.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai.pipeline.document_extractors import extract_text_file
from ai.pipeline.graph import run_extract_only
from backend.schemas.api_schemas import IngestResponse, StructuredRequirement

router = APIRouter(tags=["Pipeline"])

MAX_FILE_SIZE = 10 * 1024 * 1024       # 10 MB
MAX_TEXT_CHARS = 20_000                # 20,000 chars
CHUNK_SIZE = 64 * 1024                 # 64 KB streaming chunks


def detect_file_type_by_magic_bytes(file_bytes: bytes, filename: str = "") -> str:
    """Detect format strictly from magic bytes, rejecting masqueraded binary files."""
    if not file_bytes:
        raise HTTPException(
            status_code=422,
            detail={"code": "EMPTY_DOCUMENT", "message": "Uploaded file is empty."},
        )

    # 1. PDF: %PDF-
    if file_bytes.startswith(b"%PDF-"):
        return "pdf"

    # 2. DOCX: PK\x03\x04 (zip container)
    if file_bytes.startswith(b"PK\x03\x04"):
        return "docx"

    # 3. Images: PNG, JPEG, TIFF
    if (
        file_bytes.startswith(b"\x89PNG")
        or file_bytes.startswith(b"\xff\xd8\xff")
        or file_bytes.startswith(b"II*\x00")
        or file_bytes.startswith(b"MM\x00*")
    ):
        return "image"

    # 4. Text file test: strictly no NUL bytes and decodable as utf-8 / cp1252
    if b"\x00" not in file_bytes:
        try:
            extract_text_file(file_bytes)
            return "txt"
        except Exception:
            pass

    raise HTTPException(
        status_code=415,
        detail={
            "code": "UNSUPPORTED_TYPE",
            "message": f"Unsupported or unrecognized file format for '{filename or 'upload'}'. Supported: PDF, DOCX, PNG, JPEG, TXT.",
        },
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
):
    """
    Submit a tender document (PDF/DOCX/image/TXT) or raw text.
    Validates by magic bytes, checks text quality, and returns structured requirements.
    """
    audit_id = str(uuid.uuid4())

    if file is not None:
        chunks = []
        total_read = 0
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "FILE_TOO_LARGE",
                        "message": f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)} MB.",
                    },
                )
            chunks.append(chunk)

        raw_bytes = b"".join(chunks)
        filename = file.filename or "uploaded_file"
        input_type = detect_file_type_by_magic_bytes(raw_bytes, filename)
        raw_input = filename

    elif text is not None and text.strip():
        raw_input = text.strip()
        if len(raw_input) > MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "TEXT_TOO_LONG",
                    "message": f"Text input exceeds maximum limit of {MAX_TEXT_CHARS:,} characters.",
                },
            )
        input_type = "text"
        raw_bytes = b""

    else:
        raise HTTPException(
            status_code=422,
            detail={"code": "UNREADABLE_DOCUMENT", "message": "Provide either a file upload or text content."},
        )

    # Run extraction-only pipeline (Nodes 00–02) — produces structured requirement for human review
    final_state = await run_extract_only(
        raw_input=raw_input,
        input_type=input_type,
        raw_bytes=raw_bytes,
        audit_id=audit_id,
    )

    doc_structure = final_state.get("doc_structure", {})
    extracted_text = final_state.get("extracted_text", "")

    # If extraction failed or document was rejected by quality gate, never return HTTP 200 with junk
    if not extracted_text or doc_structure.get("error"):
        error_code = doc_structure.get("error", "UNREADABLE_DOCUMENT")
        status_code = doc_structure.get("status_code", 422)
        message = doc_structure.get("message") or doc_structure.get("detail") or "Failed to extract readable document text."
        raise HTTPException(
            status_code=status_code,
            detail={"code": error_code, "message": message},
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
        normalized_text=final_state.get("normalized_text", extracted_text),
        language=final_state.get("language", "en"),
        input_type=input_type,
        thesaurus_expansions=final_state.get("thesaurus_expansions", []),
        extraction_method=doc_structure.get("method"),
        pages=doc_structure.get("pages"),
    )

    return IngestResponse(
        audit_id=audit_id,
        structured_requirement=req,
        warnings=final_state.get("pipeline_warnings", []),
    )
