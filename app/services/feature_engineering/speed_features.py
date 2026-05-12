"""
app/services/feature_engineering/speed_features.py
──────────────────────────────────────────────────
Per-horse and field-relative speed-figure features.

Per-horse signals (from a horse's PP lines, most-recent-first):
  * ewm_speed_figure  — exponentially weighted mean (alpha=0.4) of speed figures.
                         Heavier weight on recent races.
  * best_speed_figure — max of the last N PP lines (default N=6).
  * last_speed_figure — most recent PP speed figure.
  * speed_figure_delta — last - second-last; captures form trajectory.
  * n_speed_figures   — count of non-null figs available (data-quality signal).

Field-relative signals (computed across all horses entered in the same race):
  * ewm_speed_zscore  — z-score of `ewm_speed_figure` within the field.
  * ewm_speed_rank    — 1-based descending rank within the field (1 = fastest).
  * ewm_speed_pct     — percentile within the field (0–1, higher = faster).

Per CLAUDE.md §2: features must be FIELD-RELATIVE, not absolute. A Beyer of 95
means nothing without the field mean. The rank/zscore/percentile columns are
the canonical inputs the downstream models consume.

Per CLAUDE.md §8: EWM uses pandas `.ewm(alpha=0.4)`, with the input series
ordered most-recent-LAST (so the last value carries the most weight). The
parsed schema enforces most-recent-FIRST on `HorseEntry.pp_lines`, so this
module reverses the iterable before feeding pandas.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from app.schemas.race import HorseEntry, ParsedRace, PastPerformanceLine

EWM_ALPHA: float = 0.4
"""Per CLAUDE.md §8. Locked in — do not change without an ADR."""

BEST_FIG_WINDOW: int = 6
"""Lookback window for the `best_speed_figure` aggregate."""


def _speed_figures(pp_lines: Sequence[PastPerformanceLine]) -> list[float]:
    """Extract non-null speed figures in MOST-RECENT-FIRST order."""
    return [float(pp.speed_figure) for pp in pp_lines if pp.speed_figure is not None]


def ewm_speed(figures_most_recent_first: Iterable[float], alpha: float = EWM_ALPHA) -> Optional[float]:
    """EWM with `alpha` over a most-recent-first sequence.

    Returns None for an empty input. pandas convention has the most-recent
    sample weighted highest when it is the LAST element, so we reverse the
    list before applying `.ewm()`.
    """
    figs = list(figures_most_recent_first)
    if not figs:
        return None
    figs_oldest_first = list(reversed(figs))
    s = pd.Series(figs_oldest_first, dtype=float)
    return float(s.ewm(alpha=alpha, adjust=True).mean().iloc[-1])


def horse_speed_summary(entry: HorseEntry, alpha: float = EWM_ALPHA) -> dict[str, Optional[float]]:
    """Per-horse speed-figure summary. Pure function — does not mutate entry."""
    figs = _speed_figures(entry.pp_lines)  # most-recent-first
    n = len(figs)

    last_fig = figs[0] if n >= 1 else None
    delta = (figs[0] - figs[1]) if n >= 2 else None

    best_window = figs[:BEST_FIG_WINDOW]
    best = max(best_window) if best_window else None

    return {
        "ewm_speed_figure":  ewm_speed(figs, alpha=alpha),
        "best_speed_figure": best,
        "last_speed_figure": last_fig,
        "speed_figure_delta": delta,
        "n_speed_figures":   float(n),
    }


def _zscore(series: pd.Series) -> pd.Series:
    """Within-series z-score. NaN-safe; constant series → zeros."""
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    return (series - mean) / std


def _rank_descending(series: pd.Series) -> pd.Series:
    """1-based descending rank — NaN preserved as NaN."""
    return series.rank(ascending=False, method="min", na_option="keep")


def _percentile(series: pd.Series) -> pd.Series:
    """0-1 percentile within the series — NaN preserved as NaN. Higher = faster."""
    return series.rank(pct=True, ascending=True, na_option="keep")


def build_speed_feature_frame(race: ParsedRace, alpha: float = EWM_ALPHA) -> pd.DataFrame:
    """Per-horse speed features for one race, with field-relative columns appended.

    Index is `post_position`. Caller assembles per-race frames and concatenates.
    """
    rows = []
    for entry in race.entries:
        summary = horse_speed_summary(entry, alpha=alpha)
        summary["post_position"] = entry.post_position
        summary["horse_name"] = entry.horse_name
        rows.append(summary)

    df = pd.DataFrame(rows).set_index("post_position")
    # Field-relative columns. These are the canonical model inputs.
    df["ewm_speed_zscore"] = _zscore(df["ewm_speed_figure"])
    df["ewm_speed_rank"] = _rank_descending(df["ewm_speed_figure"])
    df["ewm_speed_pct"] = _percentile(df["ewm_speed_figure"])
    return df


__all__ = [
    "EWM_ALPHA",
    "BEST_FIG_WINDOW",
    "ewm_speed",
    "horse_speed_summary",
    "build_speed_feature_frame",
]
