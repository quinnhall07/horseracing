"""
tests/test_parser/test_ocr.py
─────────────────────────────
Phase 9 / ADR-048 — OCR fallback for scanned PDFs.

The dominant input to this system is rasterized (image-only) Brisnet UP
cards — pdfplumber + pypdf both produce zero extractable text on those.
`_extract_with_ocr` rasterizes each page via Poppler and OCRs it with
Tesseract; this test module verifies:

  1. OCR path is actually invoked when text extraction returns empty.
  2. Missing OCR deps (pytesseract / pdf2image) fail gracefully — the
     function returns "" and the caller raises a clean ValueError.
  3. Poppler / Tesseract runtime errors are caught per page so a single
     bad page doesn't kill the whole document.
  4. Output preserves the form-feed page delimiter the downstream cleaner
     and parser already rely on.

Tests use lightweight mocks rather than real OCR — they assert wiring,
not OCR accuracy (accuracy is verified by the end-to-end PDF ingest tests
that run against the EXAMPLE_RACE_CARDS fixtures when the binaries are
available).
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

import pytest

from app.services.pdf_parser import extractor
from app.services.pdf_parser.extractor import (
    _extract_with_ocr,
    extract_text_from_pdf,
    ingest_pdf,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _minimal_pdf_bytes() -> bytes:
    """A tiny single-page PDF with no text. pdfplumber + pypdf return ""
    on this, so the extractor's fallback chain falls through to OCR."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    # Don't draw any text — only a rectangle. The resulting PDF has no
    # text layer and serves as a stand-in for a scanned image PDF.
    c.rect(50, 50, 100, 100, stroke=1, fill=0)
    c.save()
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# _extract_with_ocr
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractWithOcr:
    def test_returns_empty_when_pdf2image_missing(self, monkeypatch):
        """Simulate pdf2image being uninstalled."""
        # Hide both modules so the import inside _extract_with_ocr fails.
        for mod in ("pdf2image", "pytesseract"):
            monkeypatch.setitem(sys.modules, mod, None)
        out = _extract_with_ocr(_minimal_pdf_bytes())
        assert out == ""

    def test_returns_empty_when_pdf2image_raises(self):
        """If Poppler fails to rasterize, OCR returns "" gracefully."""
        with patch.object(extractor, "_extract_with_ocr") as orig:
            # Re-route to the real function so we exercise it, but patch
            # the imported pdf2image so it errors out at call time.
            orig.side_effect = None
        with patch("pdf2image.convert_from_bytes",
                   side_effect=RuntimeError("poppler exploded")):
            out = _extract_with_ocr(_minimal_pdf_bytes())
            assert out == ""

    def test_pypdf_image_path_produces_form_feed_delimited_output(self):
        """Path 1: pypdf direct image extraction. Pages with at least one
        embedded image OCR to per-page text joined by \\x0c."""
        # Fake page that yields one image; fake reader yields 3 such pages.
        fake_image_file = MagicMock(data=b"\x89PNG\x00fakebytes")
        fake_page = MagicMock()
        fake_page.images = [fake_image_file]
        fake_reader = MagicMock(pages=[fake_page, fake_page, fake_page])

        with patch("pypdf.PdfReader", return_value=fake_reader), \
             patch("PIL.Image.open", return_value=MagicMock(name="PILImage")), \
             patch("pytesseract.image_to_string",
                   side_effect=["PAGE ONE", "PAGE TWO", "PAGE THREE"]) as ocr:
            out = _extract_with_ocr(_minimal_pdf_bytes())
        assert out == "PAGE ONE\x0cPAGE TWO\x0cPAGE THREE"
        assert ocr.call_count == 3

    def test_single_page_failure_does_not_kill_document(self):
        """If Tesseract crashes on one page, the other pages still produce
        text. The bad page's slot is empty rather than aborting the whole
        document — partial extraction is better than none."""
        fake_image_file = MagicMock(data=b"\x89PNG\x00fakebytes")
        fake_page = MagicMock()
        fake_page.images = [fake_image_file]
        fake_reader = MagicMock(pages=[fake_page] * 3)

        with patch("pypdf.PdfReader", return_value=fake_reader), \
             patch("PIL.Image.open", return_value=MagicMock(name="PILImage")), \
             patch("pytesseract.image_to_string",
                   side_effect=["ONE",
                                RuntimeError("tesseract crashed"),
                                "THREE"]):
            out = _extract_with_ocr(_minimal_pdf_bytes())
        assert out == "ONE\x0c\x0cTHREE"

    def test_falls_through_to_raster_when_pypdf_yields_no_images(self):
        """Path 2: when no page has embedded images (vector PDFs), the
        function falls through to pdf2image rasterization at 200 DPI."""
        # pypdf returns pages with no embedded images.
        fake_page = MagicMock()
        fake_page.images = []
        fake_reader = MagicMock(pages=[fake_page, fake_page])

        with patch("pypdf.PdfReader", return_value=fake_reader), \
             patch("pdf2image.convert_from_bytes",
                   return_value=[MagicMock(name="raster1"),
                                 MagicMock(name="raster2")]) as p2i, \
             patch("pytesseract.image_to_string",
                   side_effect=["RASTER ONE", "RASTER TWO"]):
            out = _extract_with_ocr(_minimal_pdf_bytes())
        assert "RASTER ONE" in out and "RASTER TWO" in out
        p2i.assert_called_once()
        # ADR-048: 200 DPI for the raster fallback (sized for Brisnet UP
        # small-print legibility without producing decompression-bomb images).
        assert p2i.call_args.kwargs.get("dpi") == 200


# ──────────────────────────────────────────────────────────────────────────────
# extract_text_from_pdf — integration with the fallback chain
# ──────────────────────────────────────────────────────────────────────────────


class TestFallbackChain:
    def test_ocr_fires_when_pdfplumber_and_pypdf_empty(self):
        """A no-text-layer PDF (scan stand-in) flows through pdfplumber → pypdf
        → OCR. We mock OCR rather than running Tesseract for speed."""
        with patch.object(extractor, "_extract_with_ocr",
                          return_value="OCR-EXTRACTED TEXT\x0cPAGE TWO") as m:
            text, n = extract_text_from_pdf(_minimal_pdf_bytes())
        assert "OCR-EXTRACTED" in text
        assert m.called

    def test_ocr_not_called_when_pdfplumber_succeeds(self):
        """A text-layer PDF must NOT trigger OCR — keeps fast path fast."""
        # Build a PDF with actual text in it.
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        c.setFont("Courier", 12)
        c.drawString(50, 700, "Real text layer here — pdfplumber wins.")
        c.save()
        pdf = buf.getvalue()
        with patch.object(extractor, "_extract_with_ocr") as m:
            text, _ = extract_text_from_pdf(pdf)
        assert "Real text layer" in text
        m.assert_not_called()

    def test_raises_when_no_method_yields_text(self):
        """OCR returns "", pdfplumber + pypdf already empty → clean ValueError."""
        with patch.object(extractor, "_extract_with_ocr", return_value=""):
            with pytest.raises(ValueError, match="no extractable text"):
                extract_text_from_pdf(_minimal_pdf_bytes())


# ──────────────────────────────────────────────────────────────────────────────
# ingest_pdf — error surfacing
# ──────────────────────────────────────────────────────────────────────────────


class TestIngestPdfWithOcrFailure:
    def test_returns_failure_when_ocr_returns_empty(self):
        """If OCR can't read the PDF, ingest_pdf should surface a clear error
        (not crash), preserving the success=False contract."""
        with patch.object(extractor, "_extract_with_ocr", return_value=""):
            r = ingest_pdf(_minimal_pdf_bytes(), source_filename="scan.pdf")
        assert r.success is False
        assert r.card is None
        assert any("no extractable text" in e.lower() for e in r.errors)
        assert r.processing_ms is not None
