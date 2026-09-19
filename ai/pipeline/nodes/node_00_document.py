"""
Node 00 — Document Understanding

Accepts: PDF, DOCX, scanned image, or plain text.
Outputs: machine-readable text + document structure dict.

For text inputs this node is a no-op (pass-through).

LayoutLMv3 status — DOCUMENTED FALLBACK TAKEN for v1:
  LayoutLMv3 requires ~1.3GB model download and a GPU-friendly runtime.
  v1 uses Docling's native structure extraction + LLM-based field parsing
  in Node02 instead. The LayoutExtractor interface is defined below so
  LayoutLMv3 can be dropped in without restructuring the pipeline.

  To activate LayoutLMv3 in a future version:
    1. Install: pip install transformers torch layoutlmv3
    2. Implement LayoutLMv3Extractor(LayoutExtractor) below
    3. Set USE_LAYOUTLMV3=true in .env
    4. Replace _get_extractor() to return the new class
"""
from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


# ── LayoutExtractor interface (preserved for LayoutLMv3 drop-in) ─────────────

class LayoutExtractor(ABC):
    """
    Abstract interface for layout-aware field extraction.
    Implement this to add LayoutLMv3 or any other layout model.
    """

    @abstractmethod
    def extract(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given extracted text + structural hints from Docling,
        return a dict of identified fields (sections, tables, line items).
        """
        ...


class DoclingLLMExtractor(LayoutExtractor):
    """
    v1 fallback: uses Docling's own structure output.
    No LayoutLMv3. Field extraction is delegated to the LLM in Node02.
    """

    def extract(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        # Structure is passed through as-is; Node02 does the LLM extraction.
        return structure


def _get_extractor() -> LayoutExtractor:
    """
    Factory — swap LayoutLMv3Extractor in here when ready.
    Currently returns DoclingLLMExtractor (v1 fallback).
    """
    return DoclingLLMExtractor()


_extractor = _get_extractor()


# ── PDF / DOCX parsing via Docling ────────────────────────────────────────────

def _parse_with_docling(file_bytes: bytes, file_type: str) -> tuple[str, dict]:
    """
    Returns (text, structure_dict).
    Falls back gracefully if Docling is not installed.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        import tempfile, os

        suffix = ".pdf" if file_type == "pdf" else ".docx"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            text = result.document.export_to_text()
            structure = {
                "pages": getattr(result.document, "num_pages", None),
                "tables": len(getattr(result.document, "tables", [])),
            }
            return text, structure
        finally:
            os.unlink(tmp_path)

    except ImportError:
        logger.warning("Docling not installed — falling back to raw bytes decode")
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = str(file_bytes)
        return text, {}
    except Exception as exc:
        logger.error("Docling parsing failed: %s", exc)
        return "", {}


# ── OCR for scanned images ────────────────────────────────────────────────────

def _ocr_image(file_bytes: bytes) -> str:
    """
    Run Tesseract OCR on an image.
    Returns extracted text, or empty string if pytesseract is not available.
    Note: Tesseract binary must be installed separately.
    """
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()

    except ImportError:
        logger.warning(
            "pytesseract / Pillow not installed — OCR unavailable. "
            "Install: pip install pytesseract Pillow && install Tesseract binary"
        )
        return ""
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        return ""


# ── Node function ─────────────────────────────────────────────────────────────

async def node_00_document(state: PipelineState) -> dict:
    """
    Document Understanding node.
    No-op for text inputs; parses files for PDF/DOCX/image inputs.
    """
    input_type = state.get("input_type", "text")
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))

    # Text input — nothing to do
    if input_type == "text":
        stages.append("document_understanding")
        return {
            "extracted_text": state.get("raw_input", ""),
            "doc_structure": {},
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    raw_bytes = state.get("raw_bytes", b"")
    if not raw_bytes:
        warnings.append("Node00: No file bytes received — treating as empty input")
        stages.append("document_understanding")
        return {
            "extracted_text": "",
            "doc_structure": {},
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    text = ""
    structure: Dict[str, Any] = {}

    if input_type in ("pdf", "docx"):
        logger.info("Node00: Parsing %s with Docling", input_type.upper())
        text, structure = _parse_with_docling(raw_bytes, input_type)
        if not text:
            warnings.append(
                f"Node00: Docling returned empty text for {input_type.upper()} — "
                "check the file is not corrupted"
            )

    elif input_type == "image":
        logger.info("Node00: Running Tesseract OCR on image")
        text = _ocr_image(raw_bytes)
        structure = {"ocr_backend": "tesseract"}
        if not text:
            warnings.append(
                "Node00: OCR returned empty text — "
                "ensure Tesseract binary is installed and image is readable"
            )

    else:
        warnings.append(f"Node00: Unknown input_type '{input_type}' — treating as plain text")
        text = raw_bytes.decode("utf-8", errors="replace")

    # Apply layout extractor (v1 = pass-through; LayoutLMv3 would transform here)
    structure = _extractor.extract(text, structure)

    stages.append("document_understanding")
    return {
        "extracted_text": text,
        "doc_structure": structure,
        "normalized_text": text,  # pre-populate so Node01 can build on it
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
