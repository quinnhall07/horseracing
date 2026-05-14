"""
app/main.py
───────────
FastAPI application factory.

Lifespan responsibilities:
  * Configure structlog before any request handler runs.
  * Create live-ingestion DB tables on startup (idempotent).
  * Load the Stream A inference artifacts (sub-models + meta + calibrator)
    into `app.state.artifacts`. Missing/broken models log a warning and
    leave `app.state.artifacts = None` so the `/cards` and `/portfolio`
    endpoints can respond with 503.
  * Dispose the SQLAlchemy engine on shutdown to release the SQLite handle.

Mounted routers:
  * /api/v1/ingest    — PDF upload + parse + persist
  * /api/v1/cards     — Hydrated RaceCard with calibrated probs
  * /api/v1/portfolio — CVaR-aware +EV Portfolio
  * GET /healthz      — liveness probe
  * GET /version      — project metadata
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import cards as cards_router
from app.api.v1 import ingest as ingest_router
from app.api.v1 import portfolio as portfolio_router
from app.core.logging import configure_logging, get_logger
from app.db.session import dispose_engine, init_db
from app.services.inference.pipeline import InferenceArtifacts


def _models_dir() -> Path:
    """Resolve the trained-model directory.

    Override via `HRBS_MODELS_DIR` for tests / alternative installs.
    Default: ./models/baseline_full relative to CWD.
    """
    raw = os.environ.get("HRBS_MODELS_DIR", "models/baseline_full")
    return Path(raw)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log = get_logger("app.main")
    log.info("startup", database_url_kind=_database_url_kind())
    await init_db()

    # Inference artifacts — graceful degradation when missing.
    artifacts: Optional[InferenceArtifacts] = None
    models_dir = _models_dir()
    try:
        if models_dir.exists():
            artifacts = InferenceArtifacts.load(models_dir)
            log.info(
                "startup.artifacts_loaded",
                models_dir=str(models_dir),
                available=artifacts.available_sub_models,
            )
        else:
            log.warning(
                "startup.models_dir_missing",
                models_dir=str(models_dir),
                hint="GET /api/v1/cards/{id} and /portfolio/{id} will 503.",
            )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "startup.artifacts_load_failed",
            models_dir=str(models_dir),
            error=str(exc),
        )
        artifacts = None
    app.state.artifacts = artifacts
    app.state.inference_cache = {}

    try:
        yield
    finally:
        log.info("shutdown")
        await dispose_engine()


def _database_url_kind() -> str:
    """Return the scheme part of the DATABASE_URL (no credentials)."""
    from app.core.config import settings
    url = settings.DATABASE_URL
    return url.split("://", 1)[0] if "://" in url else url


def _project_version() -> str:
    try:
        return metadata.version("horseracing")
    except metadata.PackageNotFoundError:
        return "0.0.0+dev"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Horse Racing Betting System",
        version=_project_version(),
        description="Pari-mutuel wagering analytics — PDF ingestion + ML pipeline.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest_router.router, prefix="/api/v1/ingest", tags=["ingest"])
    app.include_router(cards_router.router, prefix="/api/v1/cards", tags=["cards"])
    app.include_router(portfolio_router.router, prefix="/api/v1/portfolio", tags=["portfolio"])

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/version", tags=["meta"])
    async def version() -> dict:
        return {"name": "horseracing", "version": _project_version()}

    return app


app = create_app()
