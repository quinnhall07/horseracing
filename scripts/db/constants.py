"""Shared constants for the Phase 0 data pipeline.

Path resolution is relative to the repository root (the parent of `scripts/`).
This keeps every script runnable from any working directory.
"""

from __future__ import annotations

from pathlib import Path

# Repository root = parent of scripts/db/
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Data tree
DATA_DIR: Path     = REPO_ROOT / "data"
STAGING_DIR: Path  = DATA_DIR / "staging"
CLEANED_DIR: Path  = DATA_DIR / "cleaned"
EXPORTS_DIR: Path  = DATA_DIR / "exports"
DB_DIR: Path       = DATA_DIR / "db"
DB_PATH: Path      = DB_DIR / "master.db"

# Phase 0 module assets
SCHEMA_PATH: Path  = Path(__file__).resolve().parent / "schema.sql"

# Schema version — bump when DATA_PIPELINE.md §2 SQL changes.
SCHEMA_VERSION: str = "1.0"

# Quality thresholds (DATA_PIPELINE.md §6 dataset / §8 per-row).
DATASET_MIN_SCORE: float = 0.70
ROW_MIN_SCORE: float     = 0.60

# Source-priority order (highest → lowest). Higher beats lower on first insert
# only; existing rows are never overwritten (see DATA_PIPELINE.md §3).
SOURCE_PRIORITY: dict[str, int] = {
    "equibase": 4,
    "brisnet":  3,
    "drf":      2,
    "kaggle":   1,
}
