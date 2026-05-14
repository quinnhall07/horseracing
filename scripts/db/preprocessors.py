"""Per-dataset preprocessing hooks for `map_and_clean.py`.

Some Kaggle datasets ship as multi-CSV bundles (e.g., a races table joined
to a runs/horses table by race_id) rather than a single denormalized CSV.
The default `_pick_primary_csv` in map_and_clean only knows how to read one
CSV; for multi-table datasets we register a preprocessor here that returns
the merged DataFrame.

Field-map entries reference these by name:
    "preprocess": "gdaley_hkracing_merge"

Adding a new preprocessor:
    1. Write the function. Signature: `(staging_dir: Path) -> pd.DataFrame`.
    2. Register it in PREPROCESSORS dict at the bottom.
    3. Add `"preprocess": "<name>"` to the field_map entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


# ─── gdaley/hkracing (HK, race_id INT, IDs only — no names) ───────────────

def gdaley_hkracing_merge(staging_dir: Path) -> pd.DataFrame:
    """Merge runs.csv (per-horse) with races.csv (per-race) on race_id."""
    races = pd.read_csv(staging_dir / "races.csv", low_memory=False)
    runs  = pd.read_csv(staging_dir / "runs.csv",  low_memory=False)
    return runs.merge(races, on="race_id", how="left", suffixes=("", "_race"))


# ─── lantanacamara/hong-kong-horse-racing (HK, real names) ────────────────

def lantanacamara_hk_merge(staging_dir: Path) -> pd.DataFrame:
    """Merge race-result-horse.csv with race-result-race.csv on race_id."""
    races  = pd.read_csv(staging_dir / "race-result-race.csv",  low_memory=False)
    horses = pd.read_csv(staging_dir / "race-result-horse.csv", low_memory=False)
    return horses.merge(races, on="race_id", how="left", suffixes=("", "_race"))


# ─── gdaley/horseracing-in-hk (HK, expanded — INCLUDES SECTIONAL PACE DATA) ─

def gdaley_horseracing_in_hk_merge(staging_dir: Path) -> pd.DataFrame:
    """Merge runs.csv (per-horse, with per-call sectional times) with races.csv.

    Same row-shape as gdaley/hkracing but the runs CSV carries `time1`…`time6`
    (cumulative call-point times in seconds) and `behind_sec1`…`behind_sec6`
    (lengths behind leader at each call). These are exactly what
    PaceScenarioModel needs — see ADR-047.
    """
    races = pd.read_csv(staging_dir / "races.csv", low_memory=False)
    runs  = pd.read_csv(staging_dir / "runs.csv",  low_memory=False)
    return runs.merge(races, on="race_id", how="left", suffixes=("", "_race"))


# ─── registry ─────────────────────────────────────────────────────────────

PREPROCESSORS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "gdaley_hkracing_merge":             gdaley_hkracing_merge,
    "lantanacamara_hk_merge":            lantanacamara_hk_merge,
    "gdaley_horseracing_in_hk_merge":    gdaley_horseracing_in_hk_merge,
}
