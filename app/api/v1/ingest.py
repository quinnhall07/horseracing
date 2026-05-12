"""
app/api/v1/ingest.py
────────────────────
POST /api/v1/ingest/upload — accept a Brisnet / Equibase / DRF PDF, parse
it into a structured RaceCard, persist to the live DB, and return the
IngestionResult to the caller.

Because `ingest_pdf` is CPU-bound (PDF text extraction + regex parsing),
it's dispatched onto the default thread pool via `run_in_executor`. This
keeps the FastAPI event loop free for concurrent uploads.

The endpoint enforces an upper size guard at the HTTP boundary using
`settings.MAX_UPLOAD_SIZE_BYTES` BEFORE materialising the bytes in memory.
The same limit is re-checked inside `ingest_pdf` for defence in depth.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.persistence import persist_ingestion_result
from app.db.session import get_session
from app.schemas.race import IngestionResult
from app.services.pdf_parser.extractor import ingest_pdf

router = APIRouter()
log = get_logger(__name__)


@router.post("/upload", response_model=IngestionResult, status_code=status.HTTP_200_OK)
async def upload_card(
    file: UploadFile = File(..., description="Race card PDF (Brisnet / Equibase / DRF)"),
    session: AsyncSession = Depends(get_session),
) -> IngestionResult:
    """Parse a race-card PDF and persist the structured result."""
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected PDF upload, got content-type={content_type!r}",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload",
        )
    if len(pdf_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        size_mb = len(pdf_bytes) / (1024 * 1024)
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size_mb:.1f} MB exceeds maximum {max_mb:.0f} MB",
        )

    loop = asyncio.get_running_loop()
    result: IngestionResult = await loop.run_in_executor(
        None, ingest_pdf, pdf_bytes, file.filename or "",
    )

    if result.success and result.card is not None:
        try:
            card_id = await persist_ingestion_result(session, result)
            log.info(
                "ingest.persisted",
                card_id=card_id,
                filename=file.filename,
                races=result.card.n_races,
            )
        except Exception as exc:
            log.error("ingest.persist_failed", error=str(exc), filename=file.filename)
            result.errors.append(f"persistence failed: {type(exc).__name__}: {exc}")
    else:
        log.warning(
            "ingest.failed", filename=file.filename, errors=result.errors,
        )

    return result
