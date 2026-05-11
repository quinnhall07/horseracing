"""
tests/test_parser/test_extractor.py
─────────────────────────────────────
Tests for `app.services.pdf_parser.extractor`.

Covers:
  • Format detection from text signatures
  • Parser dispatch routing
  • Size guard in `ingest_pdf`
  • Text extraction via pdfplumber (reportlab-generated PDF fixture)
  • End-to-end ingest pipeline shape (not full parse correctness — pdfplumber's
    text extraction does not preserve multi-space columns the way our parser
    regex expects, so deep parser assertions belong in test_brisnet_parser.py)
  • Graceful failure on corrupt input
"""

from __future__ import annotations

import io

import pytest

from app.schemas.race import IngestionResult, RaceCard
from app.services.pdf_parser.brisnet_parser import BrisnetParser
from app.services.pdf_parser.extractor import (
    _detect_format,
    _get_parser,
    extract_text_from_pdf,
    ingest_pdf,
)


# ──────────────────────────────────────────────────────────────────────────────
# PDF fixture helper
# ──────────────────────────────────────────────────────────────────────────────


def _make_pdf(text_lines: list[str]) -> bytes:
    """
    Build a single-page PDF whose body is the given lines, drawn in Courier
    (monospaced) so pdfplumber's geometry-based extraction has a fighting
    chance to preserve column alignment. Returns the raw bytes.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.setFont("Courier", 9)
    y = 750
    for line in text_lines:
        c.drawString(36, y, line)
        y -= 11
        if y < 36:  # new page if we run out of vertical space
            c.showPage()
            c.setFont("Courier", 9)
            y = 750
    c.save()
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# _detect_format
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectFormat:
    @pytest.mark.parametrize(
        "snippet,expected",
        [
            ("Brisnet Ultimate Past Performances\nRACE 1", "brisnet_up"),
            ("BRISNET UP - Today's Card", "brisnet_up"),
            ("Daily Racing Form\nRACE 1", "drf"),
            ("DRF Past Performances", "drf"),
            ("Equibase Past Performances", "equibase"),
            ("Equi-Base Premier", "equibase"),
        ],
    )
    def test_known_signatures(self, snippet, expected):
        assert _detect_format(snippet) == expected

    def test_empty_returns_unknown(self):
        assert _detect_format("") == "unknown"

    def test_no_signature_returns_unknown(self):
        assert _detect_format("Just some random text with no markers") == "unknown"

    def test_only_searches_first_2000_chars(self):
        # Signature buried past the 2000-char window is ignored.
        long_prefix = "x" * 2100
        text = long_prefix + " brisnet "
        assert _detect_format(text) == "unknown"

    def test_signature_is_case_insensitive(self):
        # Detection lowercases the sample before checking — capitalization
        # of the source text must not matter.
        assert _detect_format("BRISNET ULTIMATE PAST PERFORMANCES") == "brisnet_up"


# ──────────────────────────────────────────────────────────────────────────────
# _get_parser dispatch
# ──────────────────────────────────────────────────────────────────────────────


class TestGetParser:
    @pytest.mark.parametrize("fmt", ["brisnet_up", "drf", "equibase", "unknown"])
    def test_returns_brisnet_parser_for_any_format(self, fmt):
        # Phase 1 dispatch falls back to BrisnetParser for every format —
        # the DRF / Equibase parsers are stubs.
        assert isinstance(_get_parser(fmt), BrisnetParser)


# ──────────────────────────────────────────────────────────────────────────────
# Size guard
# ──────────────────────────────────────────────────────────────────────────────


class TestSizeGuard:
    def test_oversized_payload_rejected(self, monkeypatch):
        # Set a tiny max to avoid allocating real megabytes.
        from app.services.pdf_parser import extractor
        monkeypatch.setattr(extractor.settings, "MAX_UPLOAD_SIZE_BYTES", 1024)
        big_bytes = b"\x00" * 2048
        result = ingest_pdf(big_bytes, source_filename="too_big.pdf")
        assert result.success is False
        assert result.card is None
        assert any("exceeds maximum" in err for err in result.errors)

    def test_at_limit_passes_size_guard(self, monkeypatch):
        # Exactly-at-limit byte count must NOT be rejected by the guard.
        # The downstream extraction will fail on the garbage bytes, but the
        # failure mode should be a parse/extract error, not a size error.
        from app.services.pdf_parser import extractor
        monkeypatch.setattr(extractor.settings, "MAX_UPLOAD_SIZE_BYTES", 1024)
        at_limit = b"\x00" * 1024
        result = ingest_pdf(at_limit, source_filename="at_limit.pdf")
        # Whatever the result, the rejection reason must not be size.
        if not result.success:
            assert not any("exceeds maximum" in err for err in result.errors)


# ──────────────────────────────────────────────────────────────────────────────
# extract_text_from_pdf
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractTextFromPdf:
    def test_extracts_text_from_simple_pdf(self):
        pdf_bytes = _make_pdf(["Hello PDF", "Second line"])
        text, pages = extract_text_from_pdf(pdf_bytes)
        assert pages == 1
        assert "Hello PDF" in text
        assert "Second line" in text

    def test_page_count_correct_for_multipage(self):
        # 80 lines at 11pt with margin → forces ≥ 2 pages
        many_lines = [f"Line {i}" for i in range(120)]
        pdf_bytes = _make_pdf(many_lines)
        text, pages = extract_text_from_pdf(pdf_bytes)
        assert pages >= 2
        # Each page is form-feed separated
        assert "\x0c" in text

    def test_raises_on_unparseable_input(self):
        # Random bytes that aren't a PDF should raise.
        with pytest.raises(Exception):
            extract_text_from_pdf(b"not a pdf at all, definitely not")


# ──────────────────────────────────────────────────────────────────────────────
# ingest_pdf end-to-end
# ──────────────────────────────────────────────────────────────────────────────


class TestIngestPdfEndToEnd:
    """
    These tests exercise the full bytes → IngestionResult pipeline.

    pdfplumber's default text extraction collapses multi-space column
    separators, which means the BrisnetParser's strict regexes will not
    recover every field from a reportlab-rendered fixture. So we only
    assert on the pipeline's SHAPE here — content correctness is the
    job of test_brisnet_parser.py against pre-extracted text.
    """

    def test_returns_ingestion_result(self):
        pdf_bytes = _make_pdf([
            "Brisnet Ultimate Past Performances",
            "RACE 1",
            "BEL  05/03/2025",
            "6 Furlongs   (Dirt)   Fast",
        ])
        result = ingest_pdf(pdf_bytes, source_filename="test.pdf")
        assert isinstance(result, IngestionResult)

    def test_processing_time_recorded(self):
        pdf_bytes = _make_pdf(["Brisnet UP", "RACE 1", "BEL 05/03/2025"])
        result = ingest_pdf(pdf_bytes)
        assert result.processing_ms is not None
        assert result.processing_ms >= 0

    def test_source_filename_preserved_on_success(self):
        pdf_bytes = _make_pdf(["Brisnet UP", "RACE 1", "BEL 05/03/2025"])
        result = ingest_pdf(pdf_bytes, source_filename="card_2025_05_03.pdf")
        # The filename should propagate into the card if parsing succeeded
        # (or be absent if it failed — either way no exception).
        if result.success:
            assert isinstance(result.card, RaceCard)
            assert result.card.source_filename == "card_2025_05_03.pdf"

    def test_total_pages_set_on_card(self):
        pdf_bytes = _make_pdf(["Brisnet UP", "RACE 1", "BEL 05/03/2025"])
        result = ingest_pdf(pdf_bytes)
        if result.success and result.card is not None:
            assert result.card.total_pages >= 1

    def test_no_race_header_yields_failure(self):
        # PDF containing no "RACE N" lines → parser returns zero races →
        # ingest_pdf sets success=False with explanatory error.
        pdf_bytes = _make_pdf(["Some other document", "Not a race card"])
        result = ingest_pdf(pdf_bytes)
        assert result.success is False
        assert any("zero races" in err.lower() or "race card" in err.lower()
                   for err in result.errors)

    def test_corrupt_bytes_caught_gracefully(self):
        # Garbage bytes → extraction raises → caught → IngestionResult with errors.
        result = ingest_pdf(b"\xff\xd8\xff\xe0not-a-pdf", source_filename="bad.pdf")
        assert result.success is False
        assert len(result.errors) >= 1
