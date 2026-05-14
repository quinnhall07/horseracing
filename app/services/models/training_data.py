"""
app/services/models/training_data.py
────────────────────────────────────
Convert the master-DB training parquet into ML-ready feature matrices.

INPUTS
    data/exports/training_<YYYYMMDD>.parquet — one row per historical
    horse-race result with the columns documented in DATA_PIPELINE.md §12.

OUTPUTS
    A DataFrame where each row carries:
      * an integer `win` label (1 if the horse finished first, else 0)
      * per-horse features derived from PRIOR results only (no leakage)
      * today-race scalar features (knowable pre-race: distance, surface, …)
      * field-relative columns computed within the today-race group

CRITICAL: every per-horse feature is computed with `groupby('horse_key').shift(1)`
before the rolling aggregation, so the value on row N depends ONLY on rows
1..N-1 for that horse. This is what CLAUDE.md §2 means by "time-based
validation" — there must be no leakage at the FEATURE level either, not
just the train/val split.

Per CLAUDE.md §8 the EWM uses alpha=0.4 — same constant as the live-inference
path in `app/services/feature_engineering/speed_features.py`. Locked in.

The module is intentionally framework-agnostic: it returns a pandas DataFrame
that any downstream model (LightGBM, sklearn, PyTorch) can consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger

log = get_logger(__name__)

EWM_ALPHA: float = 0.4
"""Locked per CLAUDE.md §8."""

# How many recent prior races the rolling aggregates summarise.
ROLLING_WINDOW: int = 6

# Layoff parameters (mirror feature_engineering/layoff.py).
LAYOFF_RECOVERY_THRESHOLD_DAYS: float = 30.0
LAYOFF_DECAY_LAMBDA: float = math.log(2.0) / 60.0
FIRST_TIME_STARTER_FITNESS: float = 0.6


@dataclass(frozen=True)
class TrainValSplit:
    """Outputs of `time_based_split`."""

    train: pd.DataFrame
    val: pd.DataFrame
    split_date: pd.Timestamp


# ─── Parquet loader ──────────────────────────────────────────────────────────


def load_training_parquet(path: Path) -> pd.DataFrame:
    """Load the training parquet with the canonical dtype coercions applied."""
    if not path.exists():
        raise FileNotFoundError(f"Training parquet not found: {path}")
    df = pd.read_parquet(path)

    # The parquet stores `race_date` as datetime64[us]; coerce to a normalised
    # day-level timestamp for stable groupby keys and day-delta arithmetic.
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.normalize()

    # Some columns ship as `object` because they're 100% null in the parquet
    # (see PROGRESS.md "Known Export Caveats"). Coerce the numerics so downstream
    # rolling/arithmetic doesn't crash on the first non-null value.
    for col in ("fraction_q1_sec", "fraction_q2_sec", "beaten_lengths_q1",
                "beaten_lengths_q2", "claiming_price", "foaling_year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─── Feature engineering ─────────────────────────────────────────────────────


def _race_id(df: pd.DataFrame) -> pd.Series:
    """Stable in-frame race identifier built from (date, track, race#)."""
    return (
        df["race_date"].dt.strftime("%Y%m%d")
        + "|" + df["track_code"].astype(str)
        + "|" + df["race_number"].astype(str)
    )


def _horse_key(df: pd.DataFrame) -> pd.Series:
    """Per-horse grouping key. Prefers the master-DB `horse_dedup_key` (a
    SHA-256 of (name, foaling_year, country) — globally unique by construction
    per DATA_PIPELINE.md). Falls back to the legacy `horse_name|jurisdiction`
    compromise when the column is missing — see ADR-027 for the history."""
    if "horse_dedup_key" in df.columns and df["horse_dedup_key"].notna().any():
        # 100% non-null in v2 exports; if any row leaks through with NaN,
        # back-fill from the legacy key so groupby still has something stable.
        legacy = df["horse_name"].astype(str) + "|" + df["jurisdiction"].astype(str)
        return df["horse_dedup_key"].astype("string").fillna(legacy)
    return df["horse_name"].astype(str) + "|" + df["jurisdiction"].astype(str)


def _safe_zscore(s: pd.Series) -> pd.Series:
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)
    if not std or pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def _layoff_fitness(days: pd.Series) -> pd.Series:
    """Vectorised twin of layoff.layoff_fitness — handles NaN as FTS."""
    out = np.where(
        days.isna(),
        FIRST_TIME_STARTER_FITNESS,
        np.where(
            days <= LAYOFF_RECOVERY_THRESHOLD_DAYS,
            1.0,
            np.exp(-LAYOFF_DECAY_LAMBDA * (days.fillna(0) - LAYOFF_RECOVERY_THRESHOLD_DAYS)),
        ),
    )
    return pd.Series(out, index=days.index, dtype=float)


def prepare_training_features(
    df: pd.DataFrame,
    *,
    drop_first_starts: bool = False,
) -> pd.DataFrame:
    """Build the per-row feature matrix.

    The returned frame is sorted by `race_date` (ascending) so a downstream
    time-based train/val split can slice on the last K rows. The `win` label
    column is included; downstream callers do `X = df.drop(columns=["win"])`.

    Set `drop_first_starts=True` to exclude rows where the horse has no prior
    history. Default False so the model can learn FTS behaviour.
    """
    if df.empty:
        return df.copy()

    # ── Sort + identifiers ──────────────────────────────────────────────────
    df = df.copy()
    df["horse_key"] = _horse_key(df)
    df["race_id"] = _race_id(df)
    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)

    # ── Derive field_size where the source dataset omitted it ───────────────
    derived = df.groupby("race_id")["race_id"].transform("size")
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce").fillna(derived).astype("int64")

    # ── Labels (derived directly from finish_position — no leakage; this IS
    # the target). Computed BEFORE the rolling aggregates so they can read
    # the past `win` column.
    df["win"] = (df["finish_position"] == 1).astype("int8")
    # finish_pct ∈ (0, 1], lower = better finish. Useful as a regression
    # auxiliary; downstream rolling uses the prior shift.
    df["finish_pct"] = df["finish_position"] / df["field_size"]

    grp = df.groupby("horse_key", sort=False, group_keys=False)

    # ── Per-horse prior features (groupby + shift(1) before any aggregation) ──
    speed_prior = grp["speed_figure"].shift(1)
    df["last_speed_prior"] = speed_prior
    df["ewm_speed_prior"] = grp["speed_figure"].transform(
        lambda s: s.shift(1).ewm(alpha=EWM_ALPHA, adjust=True).mean()
    )
    df["best_speed_prior"] = grp["speed_figure"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).max()
    )
    df["mean_speed_prior"] = grp["speed_figure"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    df["speed_delta_prior"] = grp["speed_figure"].transform(
        lambda s: s.shift(1) - s.shift(2)
    )

    df["n_prior_starts"] = grp.cumcount()

    df["mean_finish_pos_prior"] = grp["finish_position"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["mean_finish_pct_prior"] = grp["finish_pct"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["win_rate_prior"] = grp["win"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    df["mean_purse_prior"] = grp["purse_usd"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    # ── Days since previous run + layoff fitness ────────────────────────────
    prev_date = grp["race_date"].shift(1)
    df["days_since_prev"] = (df["race_date"] - prev_date).dt.days
    df["layoff_fitness"] = _layoff_fitness(df["days_since_prev"])

    # ── Field-relative columns (within today's race) ────────────────────────
    race_grp = df.groupby("race_id", sort=False)
    df["ewm_speed_zscore"] = race_grp["ewm_speed_prior"].transform(_safe_zscore)
    df["ewm_speed_rank"] = race_grp["ewm_speed_prior"].rank(
        ascending=False, method="min", na_option="keep"
    )
    df["ewm_speed_pct"] = race_grp["ewm_speed_prior"].rank(pct=True, na_option="keep")

    df["weight_lbs_delta"] = (
        df["weight_lbs"] - race_grp["weight_lbs"].transform("mean")
    )

    # ── Categorical encodings ───────────────────────────────────────────────
    for col in ("surface", "condition", "race_type", "jurisdiction"):
        df[col] = df[col].astype("category")

    # ── Optional pruning ────────────────────────────────────────────────────
    if drop_first_starts:
        df = df[df["n_prior_starts"] > 0].reset_index(drop=True)

    log.info(
        "training_data.features_prepared",
        rows=len(df),
        horses=df["horse_key"].nunique(),
        races=df["race_id"].nunique(),
        date_min=df["race_date"].min().date().isoformat() if len(df) else None,
        date_max=df["race_date"].max().date().isoformat() if len(df) else None,
        win_rate=float(df["win"].mean()) if len(df) else 0.0,
    )
    return df


# ─── Train/val split ─────────────────────────────────────────────────────────


def time_based_split(
    df: pd.DataFrame,
    val_fraction: float = 0.10,
    date_col: str = "race_date",
) -> TrainValSplit:
    """Slice the trailing `val_fraction` of rows (by date) into the validation set.

    Per CLAUDE.md §2 — never random-split; future information leaks.
    Returns the cut-off date used so callers can log it in metric artifacts.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if df.empty:
        empty = df.iloc[0:0]
        return TrainValSplit(train=empty, val=empty, split_date=pd.NaT)

    # Compute the quantile boundary on the date axis (not on row position) —
    # rows are unevenly distributed across dates, and we want to validate on
    # an actual time window, not a random 10% of recent rows.
    cutoff = df[date_col].quantile(1.0 - val_fraction)
    train = df[df[date_col] <= cutoff].reset_index(drop=True)
    val = df[df[date_col] > cutoff].reset_index(drop=True)
    return TrainValSplit(train=train, val=val, split_date=cutoff)


# ─── Feature column registry ─────────────────────────────────────────────────

SPEED_FORM_FEATURE_COLUMNS: list[str] = [
    # Per-horse priors
    "ewm_speed_prior",
    "last_speed_prior",
    "best_speed_prior",
    "mean_speed_prior",
    "speed_delta_prior",
    "n_prior_starts",
    "mean_finish_pos_prior",
    "mean_finish_pct_prior",
    "win_rate_prior",
    "mean_purse_prior",
    "days_since_prev",
    "layoff_fitness",
    # Today-race numeric scalars
    "distance_furlongs",
    "field_size",
    "weight_lbs",
    "purse_usd",
    # Field-relative
    "ewm_speed_zscore",
    "ewm_speed_rank",
    "ewm_speed_pct",
    "weight_lbs_delta",
    # Categoricals (LightGBM handles these natively as category dtype)
    "surface",
    "condition",
    "race_type",
    "jurisdiction",
]
"""Canonical feature set for the Layer-1a Speed/Form model. Locked — any
addition is an ADR (downstream artifacts encode the column order)."""


def three_way_time_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
    date_col: str = "race_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by date, then slice into [train | calib | test] by row quantile.

    Per ADR-003 — never random. Calibration slice sits between train and test,
    so the calibrator sees model behaviour on a held-out window that's still
    in-distribution relative to test.
    """
    if not (0 < train_frac < 1) or not (0 < cal_frac < 1) or train_frac + cal_frac >= 1:
        raise ValueError(
            f"Bad split fractions train={train_frac}, cal={cal_frac}; both must be"
            f" in (0, 1) and sum to < 1."
        )
    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    n = len(sorted_df)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    train = sorted_df.iloc[:n_train].reset_index(drop=True)
    calib = sorted_df.iloc[n_train : n_train + n_cal].reset_index(drop=True)
    test = sorted_df.iloc[n_train + n_cal :].reset_index(drop=True)
    return train, calib, test


__all__ = [
    "EWM_ALPHA",
    "ROLLING_WINDOW",
    "TrainValSplit",
    "SPEED_FORM_FEATURE_COLUMNS",
    "load_training_parquet",
    "prepare_training_features",
    "time_based_split",
    "three_way_time_split",
]
