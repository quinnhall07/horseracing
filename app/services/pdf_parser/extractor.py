"""
app/services/pdf_parser/extractor.py
──────────────────────────────────────
Orchestrates PDF-to-structured-data conversion.

Responsibilities:
  1. Accept a raw bytes payload (PDF file upload).
  2. Detect the PDF format (Brisnet, Equibase, DRF, unknown).
  3. Route to the appropriate format-specific parser.
  4. Return an IngestionResult (RaceCard + parse metadata).

Four-pass extraction strategy:
  Pass A — pdfplumber "layout" mode (preserves column whitespace; best for PP tables)
  Pass B — pdfplumber "text" mode (fallback if layout is garbled)
  Pass C — pypdf character-level extraction (catches text-with-broken-pdfplumber-layout)
  Pass D — Tesseract OCR via Poppler raster (ADR-048; scanned/image PDFs)

Scanned/image PDFs (the dominant input for this system) have no embedded
text layer — passes A-C all return empty strings and Pass D produces the
parseable text. OCR is CPU-bound (~10-30 s per page at 300 DPI) but runs
inside `run_in_executor` from the FastAPI route, so it never blocks the
async loop.

If even OCR produces no text, the function raises a ValueError with a
human-readable error that the API layer converts to a 422 response —
this should now only happen if Tesseract / Poppler aren't installed.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import structlog

from app.core.config import settings
from app.schemas.race import IngestionResult, RaceCard
from app.services.pdf_parser.brisnet_parser import BrisnetParser
from app.services.pdf_parser.equibase_parser import EquibaseParser
from app.services.pdf_parser.llm_parser import LLMParser

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
        # pypdf fallback (text PDFs that pdfplumber struggles with)
        full_text = _extract_with_pypdf(pdf_bytes)

    if not full_text.strip():
        # OCR fallback for scanned/image PDFs — ADR-048. The dominant input
        # to this system (Brisnet UP cards re-rasterized by the operator's
        # workflow) lands here.
        logger.info("extractor.ocr_fallback", page_count=page_count)
        full_text = _extract_with_ocr(pdf_bytes)

    if not full_text.strip():
        raise ValueError(
            "PDF produced no extractable text, even with OCR. "
            "If this PDF is scanned, confirm `tesseract` and `poppler` are "
            "installed (see README). Otherwise the PDF may be corrupt."
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


# Fallback raster DPI for PDFs where direct image extraction yields nothing
# (vector-content scans, multi-image-per-page layouts). 200 dpi at US Letter
# is ~1700×2200 px — Tesseract reads 6-pt type cleanly at this scale and
# the page-count × per-page cost stays manageable.
_OCR_RASTER_DPI: int = 200


def _extract_with_ocr(pdf_bytes: bytes) -> str:
    """Tesseract OCR fallback for scanned / image-based PDFs (ADR-048).

    Strategy (cheapest first):

    1. **Direct image extraction via pypdf.** iOS / mobile / scanner-app
       PDFs typically embed each page as a single full-page JPG. pypdf can
       hand us that JPG directly — no rasterization needed. ~4 s/page vs
       ~30 s for the rasterization path, and the source resolution is
       preserved so Tesseract sees the exact pixels the scanner captured.

    2. **Rasterization via pdf2image / Poppler.** Only invoked when pypdf
       finds no embedded images (vector-content scans, etc.). Rendered at
       `_OCR_RASTER_DPI` (200 dpi at the page's native size), capped at
       the PIL decompression-bomb limit so iOS PDFs with non-standard
       30-inch page dimensions don't produce 100M-pixel rasters.

    Per-page text is joined with the same `\\x0c` form-feed delimiter the
    primary path uses, so the downstream cleaner / format detector /
    Brisnet parser see an identical interface regardless of which path
    produced the text.

    Returns the empty string if every dep is missing — the caller already
    raises a ValueError in that case with a clearer message.
    """
    try:
        import pytesseract  # type: ignore[import-untyped]
    except ImportError as e:
        logger.warning("extractor.ocr_deps_missing", error=str(e))
        return ""

    # ── Path 1: pypdf direct image extraction. ─────────────────────────────
    page_texts = _ocr_via_embedded_images(pdf_bytes, pytesseract)
    if any(t.strip() for t in page_texts):
        return "\x0c".join(page_texts)

    # ── Path 2: pdf2image rasterization fallback. ───────────────────────────
    logger.info("extractor.ocr_falling_back_to_raster")
    return _ocr_via_rasterization(pdf_bytes, pytesseract)


def _ocr_via_embedded_images(pdf_bytes: bytes, pytesseract) -> list[str]:
    """Extract embedded images per page and OCR each one. Pages with no
    embedded image yield "", which the caller flattens into form-feed
    delimiters.
    """
    try:
        from pypdf import PdfReader
        from PIL import Image
    except ImportError:
        return []

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("extractor.pypdf_open_failed", error=str(exc))
        return []

    out: list[str] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)
        except Exception:  # noqa: BLE001
            images = []
        if not images:
            out.append("")
            continue
        # If a page has multiple images, concatenate their OCR outputs.
        page_parts: list[str] = []
        for img_idx, img_file in enumerate(images, start=1):
            try:
                pil_img = Image.open(io.BytesIO(img_file.data))
                text = pytesseract.image_to_string(pil_img) or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "extractor.ocr_image_failed",
                    page=page_idx,
                    image=img_idx,
                    error=str(exc),
                )
                text = ""
            page_parts.append(text)
        page_text = "\n".join(page_parts)
        out.append(page_text)
        logger.info(
            "extractor.ocr_page_complete",
            page=page_idx,
            total=len(reader.pages),
            chars=len(page_text),
            source="embedded",
        )
    return out


def _ocr_via_rasterization(pdf_bytes: bytes, pytesseract) -> str:
    """Fallback: rasterize each page with Poppler then OCR. Slower than
    embedded-image extraction but catches vector PDFs."""
    try:
        from pdf2image import convert_from_bytes  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("extractor.pdf2image_missing")
        return ""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=_OCR_RASTER_DPI, fmt="png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("extractor.ocr_pdf2image_failed", error=str(exc))
        return ""

    pages: list[str] = []
    for i, img in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(img) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "extractor.ocr_tesseract_failed", page=i, error=str(exc)
            )
            text = ""
        pages.append(text)
        logger.info(
            "extractor.ocr_page_complete",
            page=i,
            total=len(images),
            chars=len(text),
            source="raster",
        )
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
    if fmt == "equibase":
        return EquibaseParser()
    if fmt == "drf":
        logger.warning(
            "DRF parser not yet implemented; using Brisnet parser as fallback",
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

    # ── LLM fallback (Phase 9 / ADR-049) ───────────────────────────────────────
    # Regex parsing is strict and frequently fails on OCR-noisy text. When
    # the regex parser produces 0 races OR 0 qualified races, fall through
    # to the LLM parser. The LLM call is gated on `ANTHROPIC_API_KEY` —
    # absence is logged but not fatal, so dev / test environments without
    # credentials still return the (empty) regex result rather than crashing.
    if card.n_races == 0 or card.n_qualified_races == 0:
        logger.info(
            "extractor.llm_fallback_triggered",
            regex_races=card.n_races,
            regex_qualified=card.n_qualified_races,
            format=fmt,
        )
        llm_result = LLMParser().parse(raw_text, source_format=fmt)
        # Always surface LLM-level warnings (missing API key, malformed
        # JSON, etc.) so the user / UI can see why the fallback didn't help.
        errors.extend(llm_result.warnings)
        if llm_result.races:
            # Replace the empty regex card with the LLM-built one, but
            # preserve the original filename / page count / format tag and
            # mark the source_format so downstream consumers know this
            # came from the LLM path.
            inferred_date = next(
                (r.header.race_date for r in llm_result.races if r.header.race_date),
                None,
            )
            inferred_track = next(
                (r.header.track_code for r in llm_result.races if r.header.track_code),
                None,
            )
            card = RaceCard(
                source_filename=source_filename,
                source_format=f"{fmt}+llm",
                total_pages=page_count,
                card_date=inferred_date,
                track_code=inferred_track,
                races=llm_result.races,
            )
            errors.append(
                f"Parsed via LLM fallback ({llm_result.model}); "
                f"{len(llm_result.races)} race(s), "
                f"{llm_result.cached_tokens} cached input tokens."
            )

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
