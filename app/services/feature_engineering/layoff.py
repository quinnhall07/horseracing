"""
app/services/feature_engineering/layoff.py
──────────────────────────────────────────
Layoff / freshening fitness curve.

From CLAUDE.md §8:
    fitness(days) = exp(-lambda * max(0, days - recovery_threshold))

`recovery_threshold` (≈30 days) models the observed empirical fact that
a short freshening is neutral-to-positive — horses do not lose form during
a brief letup. Beyond the threshold, fitness decays exponentially.

`lambda` is fit empirically per surface/distance category from the master
training corpus during Phase 3. Until that fit exists, we use a single
default decay constant that puts a horse at ~50% fitness after 90 days off
(a layoff conventional wisdom calls "long").

Public surface:
    layoff_fitness(days_since_last, ...) -> float in [0, 1]
    apply_layoff_features(df, ...)       -> df + layoff_days + layoff_fitness cols
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

# Default tunable parameters. ADR: any layer that wants a fitted value
# overrides via the function args — this module never reads from a global config.
DEFAULT_RECOVERY_THRESHOLD_DAYS: float = 30.0
DEFAULT_LAMBDA: float = math.log(2.0) / 60.0
"""λ such that fitness drops to 0.5 at days=recovery_threshold+60 (≈90 days)."""

# Horses with no PP at all → first-time starter. We treat them as a separate
# regime; downstream models can learn to handle them. We expose a sentinel
# instead of None for the days column so the dtype stays float and the
# fitness column stays computable.
FIRST_TIME_STARTER_FITNESS: float = 0.6
"""Neutral fitness for first-time starters (no PP history)."""


def layoff_fitness(
    days_since_last: float | int | None,
    recovery_threshold: float = DEFAULT_RECOVERY_THRESHOLD_DAYS,
    decay_lambda: float = DEFAULT_LAMBDA,
) -> float:
    """Return a fitness score in [0, 1] given the number of days since last race.

    NaN / None days → FIRST_TIME_STARTER_FITNESS.
    Negative days   → coerced to 0 (data error, but fail soft).
    days <= recovery_threshold → 1.0 (no decay yet).
    """
    if days_since_last is None:
        return FIRST_TIME_STARTER_FITNESS
    try:
        d = float(days_since_last)
    except (TypeError, ValueError):
        return FIRST_TIME_STARTER_FITNESS
    if math.isnan(d):
        return FIRST_TIME_STARTER_FITNESS
    if d < 0:
        d = 0.0
    if d <= recovery_threshold:
        return 1.0
    return math.exp(-decay_lambda * (d - recovery_threshold))


def layoff_fitness_series(
    days: Iterable[float | int | None],
    recovery_threshold: float = DEFAULT_RECOVERY_THRESHOLD_DAYS,
    decay_lambda: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """Vectorised version returning a numpy array."""
    arr = np.array(
        [layoff_fitness(d, recovery_threshold, decay_lambda) for d in days],
        dtype=float,
    )
    return arr


def apply_layoff_features(
    df: pd.DataFrame,
    days_col: str = "days_since_last",
    out_fitness_col: str = "layoff_fitness",
    recovery_threshold: float = DEFAULT_RECOVERY_THRESHOLD_DAYS,
    decay_lambda: float = DEFAULT_LAMBDA,
) -> pd.DataFrame:
    """Attach `layoff_fitness` to a per-horse DataFrame.

    The frame is modified in place AND returned (pandas convention).
    """
    df[out_fitness_col] = layoff_fitness_series(
        df[days_col].tolist(),
        recovery_threshold=recovery_threshold,
        decay_lambda=decay_lambda,
    )
    return df


__all__ = [
    "DEFAULT_RECOVERY_THRESHOLD_DAYS",
    "DEFAULT_LAMBDA",
    "FIRST_TIME_STARTER_FITNESS",
    "layoff_fitness",
    "layoff_fitness_series",
    "apply_layoff_features",
]
