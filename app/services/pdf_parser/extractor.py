"""
app/services/pdf_parser/extractor.py
──────────────────────────────────────
Orchestrates PDF-to-structured-data conversion.

Responsibilities:
  1. Accept a raw bytes payload (PDF file upload).
  2. Detect the PDF format (Brisnet, Equibase, DRF, unknown).
  3. Route to the appropriate format-specific parser.
  4. Return an IngestionResult (RaceCard + parse metadata).

Two-pass extraction strategy:
  Pass A — pdfplumber "layout" mode (preserves column whitespace; best for PP tables)
  Pass B — pdfplumber "text" mode (fallback if layout is garbled)
  Pass C — pypdf character-level extraction (last resort for scanned/image PDFs)

If Pass C still produces no usable text, the function raises a ValueError
with a human-readable error that the API layer converts to a 422 response.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import structlog

from app.core.config import settings
from app.schemas.race import IngestionResult, RaceCard
from app.services.pdf_parser.brisnet_parser import BrisnetParser

logger = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Format detection signatures
# ──────────────────────────────────────────────────────────────────────────────

# Brisnet UP PDFs contain this string in their first two pages
_BRISNET_SIGNATURES = ["brisnet", "ultimate past performances", "up -"]
_DRF_SIGNATURES = ["daily racing form", "drf"]
_EQUIBASE_SIGNATURES = ["equibase", "equi-base"]


def _detect_format(text_sample: str) -> str:
    """
    Identify PDF format from the first 2000 characters of extracted text.

    Returns one of: "brisnet_up" | "drf" | "equibase" | "unknown"
    """
    sample = text_sample[:2000].lower()
    if any(sig in sample for sig in _BRISNET_SIGNATURES):
        return "brisnet_up"
    if any(sig in sample for sig in _DRF_SIGNATURES):
        return "drf"
    if any(sig in sample for sig in _EQUIBASE_SIGNATURES):
        return "equibase"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction function
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(
    pdf_bytes: bytes,
    strategy: str = "layout",
) -> tuple[str, int]:
    """
    Extract all text from a PDF bytes payload.

    Returns:
        (full_text, page_count)

    Raises:
        ImportError if neither pdfplumber nor pypdf is installed.
        ValueError  if the PDF produces no extractable text (likely image scan).
    """
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError("pdfplumber not installed — run: pip install pdfplumber") from e

    pages_text: list[str] = []
    page_count = 0

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            # Primary extraction strategy
            try:
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            except Exception as exc:
                logger.warning("pdfplumber page extraction failed", error=str(exc))
                text = ""

            if not text.strip() and strategy == "layout":
                # Retry with simple text mode
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

            pages_text.append(text)

    full_text = "\x0c".join(pages_text)  # form-feed as page delimiter

    if not full_text.strip():
        # Last resort: pypdf
        full_text = _extract_with_pypdf(pdf_bytes)
        if not full_text.strip():
            raise ValueError(
                "PDF produced no extractable text. "
                "This may be a scanned/image-based PDF requiring OCR preprocessing."
            )

    return full_text, page_count


def _extract_with_pypdf(pdf_bytes: bytes) -> str:
    """pypdf fallback extraction for PDFs that pdfplumber cannot handle."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\x0c".join(pages)


# ──────────────────────────────────────────────────────────────────────────────
# Format-specific parser dispatch
# ──────────────────────────────────────────────────────────────────────────────

def _get_parser(fmt: str):
    """
    Return the appropriate parser instance for the detected format.

    Currently only Brisnet UP is implemented (Phase 1).  DRF and Equibase
    parsers are stubbed — they will emit a warning and attempt Brisnet parsing
    as a best-effort fallback.
    """
    if fmt == "brisnet_up":
        return BrisnetParser()
    # Stubs — will be replaced by format-specific parsers in later phases
    if fmt in ("drf", "equibase"):
        logger.warning(
            "Format-specific parser not yet implemented; using Brisnet parser as fallback",
            format=fmt,
        )
        return BrisnetParser()
    # Unknown format — try Brisnet anyway; better than nothing
    return BrisnetParser()


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_bytes: bytes, source_filename: str = "") -> IngestionResult:
    """
    Full ingestion pipeline: bytes → IngestionResult.

    Steps:
      1. Validate size
      2. Extract raw text (pdfplumber → pypdf fallback)
      3. Detect format
      4. Route to parser
      5. Wrap in IngestionResult with timing metadata

    This function is synchronous — it is called from the FastAPI route via
    run_in_executor so it does not block the async event loop.

    `processing_ms` is always populated in the response, regardless of
    success — slow failures need to be observable too.
    """
    start_ms = time.perf_counter()

    def _elapsed() -> float:
        return round((time.perf_counter() - start_ms) * 1000, 1)

    # ── Size guard ─────────────────────────────────────────────────────────────
    if len(pdf_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        size_mb = len(pdf_bytes) / (1024 * 1024)
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        return IngestionResult(
            success=False,
            errors=[f"File size {size_mb:.1f} MB exceeds maximum {max_mb:.0f} MB"],
            processing_ms=_elapsed(),
        )

    # ── Text extraction ────────────────────────────────────────────────────────
    try:
        raw_text, page_count = extract_text_from_pdf(
            pdf_bytes,
            strategy=settings.PDF_EXTRACTION_STRATEGY,
        )
    except ValueError as exc:
        return IngestionResult(
            success=False, errors=[str(exc)], processing_ms=_elapsed()
        )
    except Exception as exc:
        logger.error("Unexpected extraction error", error=str(exc))
        return IngestionResult(
            success=False,
            errors=[f"PDF extraction failed: {type(exc).__name__}: {exc}"],
            processing_ms=_elapsed(),
        )

    # ── Format detection ───────────────────────────────────────────────────────
    fmt = _detect_format(raw_text)
    logger.info("PDF format detected", format=fmt, pages=page_count, filename=source_filename)

    # ── Parsing ────────────────────────────────────────────────────────────────
    parser = _get_parser(fmt)
    try:
        card: RaceCard = parser.parse(raw_text, source_filename=source_filename)
        card.total_pages = page_count
    except Exception as exc:
        logger.error("Parser raised unexpected exception", error=str(exc), format=fmt)
        return IngestionResult(
            success=False,
            errors=[f"Parse error ({fmt}): {type(exc).__name__}: {exc}"],
            processing_ms=_elapsed(),
        )

    errors: list[str] = []
    success = card.n_races > 0
    if not success:
        errors.append("Parser produced zero races. Check that the PDF is a race card.")

    logger.info(
        "Ingestion complete",
        races=card.n_races,
        qualified=card.n_qualified_races,
        confidence_avg=_avg_confidence(card),
        elapsed_ms=_elapsed(),
    )

    return IngestionResult(
        success=success,
        card=card if success else None,
        errors=errors,
        processing_ms=_elapsed(),
    )


def _avg_confidence(card: RaceCard) -> float:
    if not card.races:
        return 0.0
    return round(sum(r.parse_confidence for r in card.races) / len(card.races), 3)
