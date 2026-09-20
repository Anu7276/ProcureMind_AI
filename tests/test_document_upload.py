"""
Tests for Phase 1 — Document Intake: Stop feeding raw PDF bytes to the pipeline.
"""
from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pypdf
import fitz

from ai.pipeline.document_extractors import (
    DocumentError,
    extract_pdf,
    extract_text_file,
    looks_like_document_text,
)
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def generate_tender_pdf() -> bytes:
    """Generates a clean PDF tender using reportlab with text layer."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(50, 750, "Tender for electrical works and civil infrastructure construction.")
    c.drawString(50, 720, "Scope of work includes supply of the following items:")
    c.drawString(50, 690, "(1) TMT reinforcement bars, Fe 500D grade for RCC structures.")
    c.drawString(50, 660, "(2) internal electrical wiring installation work in conduits.")
    c.drawString(50, 630, "(3) switches for fixed domestic-type electrical points.")
    c.save()
    return buffer.getvalue()


def generate_encrypted_pdf() -> bytes:
    """Generates an encrypted, password-protected PDF."""
    pdf_bytes = generate_tender_pdf()
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    writer = pypdf.PdfWriter()
    for p in reader.pages:
        writer.add_page(p)
    writer.encrypt("password123")
    enc_buf = io.BytesIO()
    writer.write(enc_buf)
    return enc_buf.getvalue()


def generate_scanned_image_pdf() -> bytes:
    """Generates a PDF containing only an image without any digital text layer."""
    from PIL import Image
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    img = Image.new("RGB", (300, 150), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    return doc.tobytes()


# ── Quality Gate Unit Tests ──────────────────────────────────────────────────

def test_looks_like_document_text_rejects_raw_pdf_bytes():
    raw_pdf_str = "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R /BaseFont /Helvetica >>\nendobj\nstream\n/FlateDecode"
    ok, reason = looks_like_document_text(raw_pdf_str)
    assert not ok
    assert "stream marker" in reason.lower() or "pdf" in reason.lower()


def test_looks_like_document_text_accepts_tender_text():
    tender = "Supply of 1000 meters of 4-core XLPE insulated armored copper electric cable for 1.1 kV."
    ok, reason = looks_like_document_text(tender)
    assert ok
    assert reason == ""


def test_looks_like_document_text_accepts_hindi_text():
    hindi_text = "पीने के पानी के लिए अनप्लास्टिककृत पीवीसी पाइप और फिटिंग की आपूर्ति।"
    ok, reason = looks_like_document_text(hindi_text)
    assert ok
    assert reason == ""


def test_looks_like_document_text_rejects_garbage():
    garbage = "???????????????!!!!!!!!!!!!!!!!!!!!!###########$$$$$$$"
    ok, reason = looks_like_document_text(garbage)
    assert not ok


# ── Ingest Route Tests ───────────────────────────────────────────────────────

def test_ingest_tender_pdf_success_and_recommendation(client):
    """
    Uploading a clean PDF tender must extract genuine text, contain no bytecode tags,
    and return expected procurement recommendation.
    """
    pdf_bytes = generate_tender_pdf()
    files = {"file": ("tender_document.pdf", pdf_bytes, "application/pdf")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    sr = data.get("structured_requirement", {})
    norm_text = sr.get("normalized_text", "")
    assert "TMT reinforcement bars" in norm_text
    assert "%PDF" not in norm_text
    assert "endobj" not in norm_text

    # Now verify that running recommendation on this extracted requirement yields IS 1786
    rec_resp = client.post("/recommend", json={"raw_query": norm_text})
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    keys = [r["key"] for r in rec_data.get("recommendations", [])]
    assert any("IS 1786" in k for k in keys)


def test_ingest_random_bytes_returns_error(client):
    """Uploading random junk bytes named x.pdf must be rejected (415 or 422), never 200."""
    fake_pdf = b"\x00\x01\x02\x03\xff\xfe\xab\xcd\xefRandomJunkBytesNotAPdfAtAll"
    files = {"file": ("x.pdf", fake_pdf, "application/pdf")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code in (415, 422)
    assert "code" in resp.json().get("detail", {})


def test_ingest_encrypted_pdf_returns_422_pdf_encrypted(client):
    """Uploading password protected PDF must return 422 with code PDF_ENCRYPTED."""
    enc_pdf = generate_encrypted_pdf()
    files = {"file": ("secure.pdf", enc_pdf, "application/pdf")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 422
    detail = resp.json().get("detail", {})
    assert detail.get("code") == "PDF_ENCRYPTED"


def test_ingest_file_exceeding_10mb_returns_413(client):
    """Uploading file > 10 MB must return HTTP 413 FILE_TOO_LARGE."""
    oversized = b"%PDF-" + b"0" * (10 * 1024 * 1024 + 1024)
    files = {"file": ("oversized.pdf", oversized, "application/pdf")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 413
    detail = resp.json().get("detail", {})
    assert detail.get("code") == "FILE_TOO_LARGE"


def test_ingest_scanned_pdf_handles_ocr(client):
    """A scanned image PDF must trigger OCR or return 503 OCR_UNAVAILABLE if Tesseract is missing."""
    scanned_pdf = generate_scanned_image_pdf()
    files = {"file": ("scanned.pdf", scanned_pdf, "application/pdf")}
    resp = client.post("/ingest", files=files)
    # Either succeeds if Tesseract is installed or returns 503 OCR_UNAVAILABLE or 422
    assert resp.status_code in (200, 422, 503)
    if resp.status_code == 503:
        assert resp.json().get("detail", {}).get("code") == "OCR_UNAVAILABLE"
