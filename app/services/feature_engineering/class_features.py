"""
app/services/feature_engineering/class_features.py
──────────────────────────────────────────────────
Class trajectory features.

Class in racing is the level of competition (claiming price for claiming
races, purse for allowance/stakes, race type for non-claiming). Horses
moving UP in class face tougher competition; horses dropping DOWN face
weaker rivals. The delta between recent races and today is one of the
most reliable handicapping signals.

Per CLAUDE.md §8: class trajectory ∈ R, normalised by recent class level.

Implementation:
  * `claiming_price_delta`  — today's claiming_price − recent average (None for
                              non-claiming races on either side).
  * `purse_delta`           — today's purse − recent average purse.
  * `class_trajectory`      — claiming delta when both today and recent are
                              claiming races; otherwise purse delta scaled so
                              the two metrics live on a comparable axis.
  * `dropping_in_class`     — boolean: today's class < recent average.
  * `race_type_change`      — categorical: today_type vs modal recent_type.

Per CLAUDE.md §2: features must be FIELD-RELATIVE where possible. The class
delta itself is per-horse; the field-relative z-score is appended downstream.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.schemas.race import HorseEntry, ParsedRace, PastPerformanceLine, RaceType

CLAIMING_RACE_TYPES = {RaceType.MAIDEN_CLAIMING, RaceType.CLAIMING}
RECENT_PP_WINDOW: int = 4
"""How many recent PPs to average over for class-level baselines."""


def _avg(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def _modal_race_type(pps: Sequence[PastPerformanceLine]) -> Optional[RaceType]:
    types = [p.race_type for p in pps if p.race_type != RaceType.UNKNOWN]
    if not types:
        return None
    return Counter(types).most_common(1)[0][0]


def horse_class_summary(
    entry: HorseEntry,
    today_race_type: RaceType,
    today_claiming_price: Optional[float],
    today_purse: Optional[float],
    window: int = RECENT_PP_WINDOW,
) -> dict[str, Optional[float] | bool]:
    """Per-horse class trajectory summary."""
    pp_window = entry.pp_lines[:window]

    claiming_prices = [p.claiming_price for p in pp_window if p.claiming_price is not None]
    purses = [p.purse_usd for p in pp_window if p.purse_usd is not None]

    avg_recent_claiming = _avg(claiming_prices)
    avg_recent_purse = _avg(purses)

    # ── Claiming price delta ──────────────────────────────────────────────────
    today_is_claiming = today_race_type in CLAIMING_RACE_TYPES
    recent_was_claiming = bool(claiming_prices)
    if today_is_claiming and recent_was_claiming and today_claiming_price is not None:
        claiming_delta = today_claiming_price - avg_recent_claiming  # type: ignore[operator]
    else:
        claiming_delta = None

    # ── Purse delta ───────────────────────────────────────────────────────────
    if today_purse is not None and avg_recent_purse is not None:
        purse_delta = today_purse - avg_recent_purse
    else:
        purse_delta = None

    # ── Composite class_trajectory ────────────────────────────────────────────
    # If both sides are claiming, use the claiming delta directly (in dollars).
    # Otherwise fall back to the purse delta. Both live in dollars so the units
    # match; field-relative normalisation happens downstream.
    if claiming_delta is not None:
        trajectory: Optional[float] = float(claiming_delta)
    elif purse_delta is not None:
        trajectory = float(purse_delta)
    else:
        trajectory = None

    # Dropping in class = today is cheaper / lower-purse than recent average.
    dropping: Optional[bool]
    if trajectory is None:
        dropping = None
    else:
        dropping = trajectory < 0

    # ── Race-type change ──────────────────────────────────────────────────────
    modal = _modal_race_type(pp_window)
    if modal is None or today_race_type == RaceType.UNKNOWN:
        race_type_change: Optional[str] = None
    elif modal == today_race_type:
        race_type_change = "same"
    else:
        race_type_change = f"{modal.value}->{today_race_type.value}"

    return {
        "avg_recent_claiming":   avg_recent_claiming,
        "avg_recent_purse":      avg_recent_purse,
        "claiming_price_delta":  float(claiming_delta) if claiming_delta is not None else None,
        "purse_delta":           float(purse_delta) if purse_delta is not None else None,
        "class_trajectory":      trajectory,
        "dropping_in_class":     dropping,
        "race_type_change":      race_type_change,
    }


def build_class_feature_frame(race: ParsedRace) -> pd.DataFrame:
    """Per-horse class trajectory features + field-relative z-score."""
    today_race_type = race.header.race_type
    today_claiming = race.header.claiming_price
    today_purse = race.header.purse_usd

    rows = []
    for entry in race.entries:
        summary = horse_class_summary(
            entry,
            today_race_type=today_race_type,
            today_claiming_price=today_claiming,
            today_purse=today_purse,
        )
        summary["post_position"] = entry.post_position
        summary["horse_name"] = entry.horse_name
        rows.append(summary)

    df = pd.DataFrame(rows).set_index("post_position")

    # Field-relative class_trajectory z-score (NaN where trajectory is None).
    traj = pd.to_numeric(df["class_trajectory"], errors="coerce")
    if traj.notna().sum() >= 2:
        m, s = traj.mean(skipna=True), traj.std(skipna=True, ddof=0)
        df["class_trajectory_zscore"] = (traj - m) / s if s and s > 0 else 0.0
    else:
        df["class_trajectory_zscore"] = 0.0

    return df


__all__ = [
    "CLAIMING_RACE_TYPES",
    "RECENT_PP_WINDOW",
    "horse_class_summary",
    "build_class_feature_frame",
]
