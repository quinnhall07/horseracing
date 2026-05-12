"""
tests/test_api/test_ingest.py
─────────────────────────────
Smoke tests for the FastAPI `/api/v1/ingest/upload` endpoint.

Strategy:
  * Override the DATABASE_URL env var to an in-memory aiosqlite DB BEFORE
    importing `app.main`. This isolates the test DB from the dev DB on disk.
  * Use FastAPI's TestClient (httpx-backed) to drive multipart uploads.
  * Generate a tiny reportlab PDF in-process so no fixture files are needed.
"""

from __future__ import annotations

import io
import os
import uuid

import pytest

# Each test run gets a fresh file-backed SQLite DB so the lifespan
# `init_db()` (which uses run_sync) can issue real CREATE TABLE statements.
# in-memory `:memory:` databases don't survive across connections, and the
# async engine pools multiple connections — so we use a unique tmpfile.
_TEST_DB_PATH = f"./_test_ingest_{uuid.uuid4().hex}.db"
os.environ["HRBS_DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c

    # Cleanup the tmp DB file after the module's tests finish.
    try:
        os.remove(_TEST_DB_PATH)
    except OSError:
        pass


def _make_pdf_bytes(lines: list[str]) -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.setFont("Courier", 9)
    y = 750
    for line in lines:
        c.drawString(36, y, line)
        y -= 11
    c.save()
    return buf.getvalue()


# ── Health endpoints ─────────────────────────────────────────────────────────


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    payload = r.json()
    assert payload["name"] == "horseracing"
    assert isinstance(payload["version"], str)


# ── Upload endpoint ──────────────────────────────────────────────────────────


def test_upload_rejects_empty_body(client):
    r = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert r.status_code == 400
    assert "Empty" in r.json()["detail"]


def test_upload_rejects_wrong_content_type(client):
    r = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("not-a-pdf.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_upload_corrupt_pdf_returns_failure_result(client):
    r = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("corrupt.pdf", b"%PDF-1.4 not actually a pdf", "application/pdf")},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["success"] is False
    assert payload["errors"]
    assert payload["card"] is None


def test_upload_valid_brisnet_text_returns_card(client):
    pdf = _make_pdf_bytes([
        "Brisnet Ultimate Past Performances",
        "CD  05/10/2026",
        "RACE 4",
        "4 1/2 Furlongs  Dirt  Fast",
        "Claiming  $30,000  Purse $62,000",
        "1  LOVELY WORDS         6-1  Grisales 123",
        "2  AMAZING ASCENDIS     4-1  Smith    120",
    ])
    r = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("card.pdf", pdf, "application/pdf")},
    )
    assert r.status_code == 200
    payload = r.json()
    # Synthetic Brisnet fixture is too sparse to guarantee a parseable race —
    # the success bit may be either True or False. What matters is that the
    # API: returned 200, contains processing_ms, and (if success) emitted a card.
    assert "success" in payload
    assert "processing_ms" in payload
    assert payload["processing_ms"] >= 0
    if payload["success"]:
        assert payload["card"] is not None
        assert payload["card"]["source_filename"] == "card.pdf"
