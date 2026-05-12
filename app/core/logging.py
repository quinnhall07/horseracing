"""
app/core/logging.py
───────────────────
Centralised structlog configuration.

All modules call `from app.core.logging import get_logger` and rely on
`configure_logging()` being invoked once at process start (FastAPI lifespan
hook in `app/main.py`, or explicitly in scripts).

Renderer is selected by `settings.LOG_JSON`:
  - True  → JSON output (one event per line) for prod log aggregation.
  - False → Pretty key=value for dev terminals (uses ConsoleRenderer with colours
            when the stream is a TTY).

If `configure_logging()` is never called, structlog uses its lazy defaults —
that's fine for unit tests and for ad-hoc script invocations.
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent structlog setup. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = settings.LOG_LEVEL.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(message)s",
    )

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_JSON:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None):
    """Return a bound structlog logger. Configures logging on first call."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
