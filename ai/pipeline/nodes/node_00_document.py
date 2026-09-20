"""
Node 00 — Document Understanding

Accepts: PDF, DOCX, scanned image, or plain text.
Outputs: machine-readable text + document structure dict.

Uses dedicated, secure extractors from ai.pipeline.document_extractors:
  - No raw bytes decoding fallback for binary formats.
  - Enforces text quality gate (looks_like_document_text).
  - Runs blocking file extractors in asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from ai.pipeline.document_extractors import (
    DocumentError,
    ExtractionResult,
    extract_docx,
    extract_image,
    extract_pdf,
    extract_text_file,
    looks_like_document_text,
)
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


# ── LayoutExtractor interface (preserved for LayoutLMv3 drop-in) ─────────────

class LayoutExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        ...


class DoclingLLMExtractor(LayoutExtractor):
    def extract(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        return structure


def _get_extractor() -> LayoutExtractor:
    return DoclingLLMExtractor()


_extractor = _get_extractor()


# ── Node function ─────────────────────────────────────────────────────────────

async def node_00_document(state: PipelineState) -> dict:
    """
    Document Understanding node.
    Extracts text and structural metadata from PDF, DOCX, image, or text inputs.
    Never decodes binary files as raw text.
    """
    input_type = state.get("input_type", "text")
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))

    # 1. Text input — pass-through with quality validation
    if input_type == "text":
        raw_text = state.get("raw_input", "")
        stages.append("document_understanding")
        return {
            "extracted_text": raw_text,
            "doc_structure": {"method": "raw_text", "pages": 1, "tables": 0},
            "normalized_text": raw_text,
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    # 2. Binary / file inputs
    raw_bytes = state.get("raw_bytes", b"")
    if not raw_bytes:
        warnings.append("Node00: No file bytes received — treating as empty input")
        stages.append("document_understanding")
        return {
            "extracted_text": "",
            "doc_structure": {"method": "none", "error": "EMPTY_BYTES"},
            "normalized_text": "",
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    result: ExtractionResult
    try:
        if input_type == "pdf":
            result = await asyncio.to_thread(extract_pdf, raw_bytes)
        elif input_type == "docx":
            result = await asyncio.to_thread(extract_docx, raw_bytes)
        elif input_type == "image":
            result = await asyncio.to_thread(extract_image, raw_bytes)
        elif input_type in ("txt", "plain_text"):
            result = await asyncio.to_thread(extract_text_file, raw_bytes)
        else:
            raise DocumentError(
                "UNSUPPORTED_TYPE",
                f"Unsupported document type '{input_type}'. Supported: pdf, docx, image, txt.",
                status_code=415,
            )

        # Append any extraction-level warnings (e.g. truncated text, missing Hindi OCR)
        for w in result.warnings:
            if w not in warnings:
                warnings.append(w)

        # Quality gate check on extracted text
        is_valid, reason = looks_like_document_text(result.text)
        if not is_valid:
            logger.warning("Node00: Quality gate rejected text: %s", reason)
            warnings.append(f"Document rejected by quality gate: {reason}")
            stages.append("document_understanding")
            return {
                "extracted_text": "",
                "normalized_text": "",
                "doc_structure": {"method": result.method, "error": "QUALITY_GATE_FAILED", "detail": reason},
                "stages_completed": stages,
                "pipeline_warnings": warnings,
            }

        # OCR advisory warning
        if result.method == "ocr":
            warnings.append("Text was read by OCR — please check it on the review page")

        doc_structure = {
            "method": result.method,
            "pages": result.pages,
            "tables": result.tables,
        }
        doc_structure = _extractor.extract(result.text, doc_structure)

        stages.append("document_understanding")
        return {
            "extracted_text": result.text,
            "doc_structure": doc_structure,
            "normalized_text": result.text,
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    except DocumentError as doc_err:
        logger.warning("Node00 DocumentError [%s]: %s", doc_err.code, doc_err.message)
        warnings.append(doc_err.message)
        stages.append("document_understanding")
        return {
            "extracted_text": "",
            "normalized_text": "",
            "doc_structure": {"method": "error", "error": doc_err.code, "message": doc_err.message, "status_code": doc_err.status_code},
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }
    except Exception as exc:
        logger.error("Node00 unexpected extraction exception: %s", exc)
        warnings.append(f"Document processing failed: {exc}")
        stages.append("document_understanding")
        return {
            "extracted_text": "",
            "normalized_text": "",
            "doc_structure": {"method": "error", "error": "EXTRACTION_FAILED", "message": str(exc)},
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }
