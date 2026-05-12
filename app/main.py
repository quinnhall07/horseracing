"""
app/main.py
───────────
FastAPI application factory.

Lifespan responsibilities:
  * Configure structlog before any request handler runs.
  * Create live-ingestion DB tables on startup (idempotent).
  * Dispose the SQLAlchemy engine on shutdown to release the SQLite handle.

Mounted routers:
  * /api/v1/ingest — PDF upload + parse
  * GET /healthz   — liveness probe
  * GET /version   — project metadata

Phase 5 will add /api/v1/analyze and /api/v1/portfolio. They live alongside
the ingest router so each phase plugs in independently.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib import metadata

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import ingest as ingest_router
from app.core.logging import configure_logging, get_logger
from app.db.session import dispose_engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log = get_logger("app.main")
    log.info("startup", database_url_kind=_database_url_kind())
    await init_db()
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

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/version", tags=["meta"])
    async def version() -> dict:
        return {"name": "horseracing", "version": _project_version()}

    return app


app = create_app()
