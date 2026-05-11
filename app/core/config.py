"""
app/core/config.py
──────────────────
Application configuration via Pydantic BaseSettings.

Reads from environment variables (with `.env` fallback if python-dotenv is
installed). All fields have sensible defaults so the package is importable
without any environment setup — required for unit tests that exercise pure
logic without spinning up the full app.
"""

from __future__ import annotations

from typing import Literal

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # fallback for environments without pydantic-settings
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore
    _HAS_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    """Global application settings. Override via env vars (HRBS_*) or .env."""

    # ── PDF ingestion ─────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    PDF_EXTRACTION_STRATEGY: Literal["layout", "text"] = "layout"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./horseracing.db"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_JSON: bool = False  # human-readable in dev, JSON in prod

    # ── Bankroll / Bet sizing (Phase 5 inputs; kept here so all knobs are central)
    BANKROLL_USD: float = 10_000.0
    KELLY_FRACTION: float = 0.25
    MAX_BET_FRACTION: float = 0.03  # hard cap: no single bet > 3% bankroll
    CVAR_ALPHA: float = 0.05         # 95th percentile shortfall
    MAX_DAILY_DRAWDOWN_PCT: float = 0.10

    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="HRBS_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )


settings = Settings()
