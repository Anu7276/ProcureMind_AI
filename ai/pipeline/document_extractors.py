"""
Document Extractors — Secure and Robust Document Intake for ProcureMind AI.

Supports:
  - PDF: PyMuPDF (with table extraction and 200 dpi scanned-PDF OCR fallback) or pypdf
  - DOCX: python-docx (paragraphs + tables rendered with ' | ')
  - Image: Pillow + pytesseract OCR (max 25 MP)
  - Text: utf-8-sig / utf-16 / cp1252 (strictly rejects binary NUL bytes)
  - Docling: optional lazy singleton when settings.USE_DOCLING is enabled

Safety gates:
  - Encryption detection -> DocumentError("PDF_ENCRYPTED")
  - Page count limits -> DocumentError("TOO_MANY_PAGES")
  - Text quality gate -> looks_like_document_text()
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import io
import logging
import re
import string
from typing import List, Optional, Tuple

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentError(Exception):
    """Domain exception for document parsing and validation failures."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class ExtractionResult:
    text: str
    method: str
    pages: int
    tables: int
    warnings: List[str] = field(default_factory=list)
    ok: bool = True


# ── Quality Gate ─────────────────────────────────────────────────────────────

def looks_like_document_text(text: str) -> Tuple[bool, str]:
    """
    Validates that the extracted text is human-readable document text
    rather than binary garbage, PDF bytecode, or degenerate repetitive characters.
    """
    if not text or not text.strip():
        return False, "Extracted text is empty"

    # 1. Reject raw PDF bytecode / stream markers
    raw_pdf_markers = ["%PDF-", "endobj", "/BaseFont", "/FlateDecode", "/Helvetica", "/Encoding"]
    for marker in raw_pdf_markers:
        if marker in text:
            return False, f"Contains raw PDF stream marker '{marker}'"

    # 2. Check character validity ratio (letters, digits, whitespace, common punctuation)
    # Hindi and multilingual unicode letters return True for isalnum()
    allowed_punct = set(string.punctuation + "–—₹°±µ½¼¾•·“”‘’…|/\\[]{}()<>+=-_*&^%$#@!~`'\";:,.")
    valid_count = sum(1 for c in text if c.isalnum() or c.isspace() or c in allowed_punct)
    valid_ratio = valid_count / len(text)
    if valid_ratio < 0.60:
        return False, f"Fewer than 60% valid characters ({valid_ratio:.1%})"

    # 3. Check for at least 3 words of 3+ letters
    strip_chars = string.punctuation + "–—₹°±µ½¼¾•·“”‘’…|/\\[]{}()<>+=-_*&^%$#@!~`'\";:.,।॥"
    words = [w.strip(strip_chars) for w in text.split()]
    words_3plus = [w for w in words if len(w) >= 3 and any(c.isalpha() for c in w)]
    if len(words_3plus) < 3:
        return False, f"Found only {len(words_3plus)} words of 3+ characters (need at least 3)"

    # 4. Check that no single character makes up more than 40% of text (for texts > 20 chars)
    if len(text) > 20:
        counts = Counter(text)
        most_common_char, count = counts.most_common(1)[0]
        if (count / len(text)) > 0.40:
            return False, f"Single character '{most_common_char}' makes up {count / len(text):.1%} of text"

    return True, ""


# ── Docling Optional Singleton ───────────────────────────────────────────────

_docling_converter = None


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


# ── PDF Extraction ───────────────────────────────────────────────────────────

def extract_pdf(file_bytes: bytes) -> ExtractionResult:
    """
    Extracts text and tables from a PDF using PyMuPDF (or pypdf fallback).
    Detects scanned pages and triggers OCR if average characters per page < 40.
    """
    warnings: List[str] = []

    # Optional Docling path if explicitly enabled
    if getattr(settings, "USE_DOCLING", False):
        try:
            import tempfile, os
            converter = _get_docling_converter()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                res = converter.convert(tmp_path)
                doc_text = res.document.export_to_text()
                pages = getattr(res.document, "num_pages", 1)
                tables = len(getattr(res.document, "tables", []))
                return ExtractionResult(
                    text=doc_text,
                    method="docling",
                    pages=pages,
                    tables=tables,
                    warnings=warnings,
                    ok=True,
                )
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as exc:
            logger.warning("Docling extraction failed (%s); falling back to PyMuPDF/pypdf", exc)
            warnings.append(f"Docling conversion failed ({exc}); used native PDF extractor fallback")

    # Primary path: PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        used_pymupdf = True
    except ImportError:
        fitz = None
        used_pymupdf = False

    if used_pymupdf and fitz is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise DocumentError("UNREADABLE_DOCUMENT", f"Unable to open PDF: {exc}", status_code=422)

        if doc.is_encrypted:
            raise DocumentError(
                "PDF_ENCRYPTED",
                "This PDF is password-protected. Remove the password and upload again.",
                status_code=422,
            )

        num_pages = len(doc)
        if num_pages > 50:
            raise DocumentError(
                "TOO_MANY_PAGES",
                f"This PDF has {num_pages} pages, which exceeds the 50-page limit.",
                status_code=422,
            )

        page_texts: List[str] = []
        total_tables = 0

        for page_idx, page in enumerate(doc):
            p_text = page.get_text("text", sort=True) or ""

            # Extract tables if supported by this PyMuPDF build
            if hasattr(page, "find_tables"):
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        total_tables += len(tabs.tables)
                        table_lines = []
                        for tab in tabs:
                            for row in tab.extract():
                                row_cells = [str(c).strip() if c is not None else "" for c in row]
                                if any(row_cells):
                                    table_lines.append(" | ".join(row_cells))
                        if table_lines:
                            p_text = p_text + "\n" + "\n".join(table_lines)
                except Exception as exc:
                    logger.debug("Table extraction error on page %d: %s", page_idx + 1, exc)

            page_texts.append(p_text.strip())

        full_text = "\n\n".join(page_texts).strip()

        # Scanned-PDF detection: average non-whitespace characters per page < 40
        non_ws_count = len(re.sub(r"\s+", "", full_text))
        avg_non_ws = non_ws_count / max(1, num_pages)

        if avg_non_ws < 40:
            logger.info("PDF appears scanned (avg %0.1f chars/page). Initiating OCR...", avg_non_ws)
            return _ocr_pdf_pages(doc, num_pages)

        # Truncate at 200,000 characters if necessary
        if len(full_text) > 200_000:
            full_text = full_text[:200_000]
            warnings.append("Text truncated to 200,000 characters limit.")

        return ExtractionResult(
            text=full_text,
            method="pymupdf",
            pages=num_pages,
            tables=total_tables,
            warnings=warnings,
            ok=True,
        )

    # Fallback path: pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            raise DocumentError(
                "PDF_ENCRYPTED",
                "This PDF is password-protected. Remove the password and upload again.",
                status_code=422,
            )

        num_pages = len(reader.pages)
        if num_pages > 50:
            raise DocumentError(
                "TOO_MANY_PAGES",
                f"This PDF has {num_pages} pages, which exceeds the 50-page limit.",
                status_code=422,
            )

        page_texts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            page_texts.append(t.strip())

        full_text = "\n\n".join(page_texts).strip()
        if len(full_text) > 200_000:
            full_text = full_text[:200_000]
            warnings.append("Text truncated to 200,000 characters limit.")

        return ExtractionResult(
            text=full_text,
            method="pypdf",
            pages=num_pages,
            tables=0,
            warnings=warnings,
            ok=True,
        )
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("UNREADABLE_DOCUMENT", f"Failed to extract text from PDF: {exc}", status_code=422)


def _ocr_pdf_pages(doc: Any, num_pages: int) -> ExtractionResult:
    """Runs 200 dpi OCR via PyMuPDF and pytesseract on scanned PDF pages."""
    try:
        import pytesseract
        from PIL import Image

        try:
            available_langs = pytesseract.get_languages()
        except Exception:
            available_langs = ["eng"]

        ocr_lang = "eng+hin" if "hin" in available_langs else "eng"
        warnings: List[str] = []
        if "hin" not in available_langs:
            warnings.append("Hindi OCR data not installed; Hindi text may be missed")

        ocr_pages_text: List[str] = []
        max_ocr_pages = min(num_pages, 30)

        for page_idx in range(max_ocr_pages):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                txt = pytesseract.image_to_string(img, lang=ocr_lang, timeout=30)
            except Exception as e:
                logger.warning("OCR timeout/error on page %d: %s", page_idx + 1, e)
                txt = ""
            ocr_pages_text.append(txt.strip())

        full_text = "\n\n".join(ocr_pages_text).strip()
        if len(full_text) > 200_000:
            full_text = full_text[:200_000]
            warnings.append("Text truncated to 200,000 characters limit.")

        return ExtractionResult(
            text=full_text,
            method="ocr",
            pages=num_pages,
            tables=0,
            warnings=warnings,
            ok=True,
        )

    except Exception as exc:
        err_str = str(exc).lower()
        if "tesseractnotfounderror" in type(exc).__name__.lower() or "tesseract is not installed" in err_str:
            raise DocumentError(
                "OCR_UNAVAILABLE",
                "OCR engine (Tesseract) is not installed. Windows: install UB-Mannheim build; Ubuntu: apt install tesseract-ocr",
                status_code=503,
            )
        raise DocumentError("OCR_UNAVAILABLE", f"OCR extraction failed: {exc}", status_code=503)


# ── DOCX Extraction ──────────────────────────────────────────────────────────

def extract_docx(file_bytes: bytes) -> ExtractionResult:
    """
    Extracts text and table rows from a .docx file using python-docx.
    Each table row is formatted with cells separated by ' | '.
    """
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentError("DOCX_UNREADABLE", f"Unable to open DOCX file: {exc}", status_code=422)

    elements: List[str] = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            elements.append(p.text.strip())

    table_count = len(doc.tables)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text]
            if cells:
                elements.append(" | ".join(cells))

    full_text = "\n".join(elements).strip()
    return ExtractionResult(
        text=full_text,
        method="docx",
        pages=1,
        tables=table_count,
        warnings=[],
        ok=True,
    )


# ── Image Extraction ─────────────────────────────────────────────────────────

def extract_image(file_bytes: bytes) -> ExtractionResult:
    """
    Extracts text from a standalone image using Pillow and pytesseract.
    Rejects images exceeding 25 megapixels.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        if w * h > 25_000_000:
            raise DocumentError(
                "IMAGE_TOO_LARGE",
                f"Image size ({w}x{h} = {w*h/1_000_000:.1f} MP) exceeds the 25 MP limit.",
                status_code=422,
            )
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("UNREADABLE_DOCUMENT", f"Invalid image file: {exc}", status_code=422)

    try:
        import pytesseract
        try:
            available_langs = pytesseract.get_languages()
        except Exception:
            available_langs = ["eng"]

        ocr_lang = "eng+hin" if "hin" in available_langs else "eng"
        warnings: List[str] = []
        if "hin" not in available_langs:
            warnings.append("Hindi OCR data not installed; Hindi text may be missed")

        text = pytesseract.image_to_string(img, lang=ocr_lang, timeout=30).strip()
        return ExtractionResult(
            text=text,
            method="ocr",
            pages=1,
            tables=0,
            warnings=warnings,
            ok=True,
        )
    except Exception as exc:
        err_str = str(exc).lower()
        if "tesseractnotfounderror" in type(exc).__name__.lower() or "tesseract is not installed" in err_str:
            raise DocumentError(
                "OCR_UNAVAILABLE",
                "OCR engine (Tesseract) is not installed. Windows: install UB-Mannheim build; Ubuntu: apt install tesseract-ocr",
                status_code=503,
            )
        raise DocumentError("UNREADABLE_DOCUMENT", f"OCR failed on image: {exc}", status_code=422)


# ── Plain Text Extraction ────────────────────────────────────────────────────

def extract_text_file(file_bytes: bytes) -> ExtractionResult:
    """
    Extracts text from a plain text file.
    The ONLY place where character decoding is allowed.
    Rejects binary files containing NUL bytes.
    Tries utf-8-sig, utf-16 (if BOM present), and cp1252.
    """
    if b"\x00" in file_bytes:
        raise DocumentError(
            "UNREADABLE_DOCUMENT",
            "File contains binary NUL bytes; not a valid text file.",
            status_code=422,
        )

    # 1. UTF-16 with BOM
    if file_bytes.startswith(b"\xff\xfe") or file_bytes.startswith(b"\xfe\xff"):
        try:
            text = file_bytes.decode("utf-16")
            return ExtractionResult(text=text, method="plain_text", pages=1, tables=0, warnings=[], ok=True)
        except Exception:
            pass

    # 2. UTF-8 (with optional BOM)
    try:
        text = file_bytes.decode("utf-8-sig")
        return ExtractionResult(text=text, method="plain_text", pages=1, tables=0, warnings=[], ok=True)
    except UnicodeDecodeError:
        pass

    # 3. Windows CP-1252 fallback
    try:
        text = file_bytes.decode("cp1252")
        return ExtractionResult(text=text, method="plain_text", pages=1, tables=0, warnings=[], ok=True)
    except UnicodeDecodeError:
        pass

    raise DocumentError(
        "UNREADABLE_DOCUMENT",
        "Unable to decode text file with utf-8 or cp1252.",
        status_code=422,
    )
