"""
app/services/feature_engineering/pace_features.py
─────────────────────────────────────────────────
Pace shape construction + per-horse pace-style proxies + field-level pace
pressure index.

Pace shape — per CLAUDE.md and the master reference:
  * `early_speed`  — lower beaten_lengths_q1 ⇒ higher early-speed score.
                     We use (-1 * beaten_lengths_q1) so positive = faster early.
  * `late_kick`    — improvement from second call to finish:
                     beaten_lengths_q2 - beaten_lengths_finish_proxy
                     (finish proxy = lengths_behind; positive = late closer).
  * `fraction_q1_ratio` — fraction_q1 / fraction_q2 (early share of mid-race time).
  * `fraction_q2_ratio` — (fraction_q2 - fraction_q1) / fraction_finish (middle share).

Pace style proxy (per-horse, summarised across PP lines):
  * `pace_style_score` — average position_at_first_call_proxy.
                          Approximated from beaten_lengths_q1 (no explicit position
                          column on PastPerformanceLine; lengths_behind at first
                          call is the best proxy). Low value ⇒ runs near the front.

Field-level pace pressure (race-level scalar replicated per-row):
  * `pace_pressure_index` — count of horses with `pace_style_score <= 1.5` lengths
                             behind at the first call on their recent form. More
                             front-runners ⇒ more pressure ⇒ contested early pace.

This module is intentionally lightweight: the Phase 1b Pace Scenario Model
(LightGBM) consumes these summary stats and learns the actual pace shape; we
only need the building blocks here. Phase 2 = features, Phase 3 = models.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.schemas.race import HorseEntry, ParsedRace, PastPerformanceLine

# A horse is "front-running" today (per-PP heuristic) if it was within this many
# lengths of the lead at the first call. 1.5 lengths is the conventional cut.
FRONT_RUNNER_LENGTHS_THRESHOLD: float = 1.5
RECENT_PP_WINDOW: int = 5
"""How many recent PPs to average over for the per-horse pace-style proxy."""


def _safe_div(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def fraction_ratios(pp: PastPerformanceLine) -> dict[str, Optional[float]]:
    """Compute fractional time ratios for a single PP line."""
    q1, q2, fin = pp.fraction_q1, pp.fraction_q2, pp.fraction_finish

    q1_ratio = _safe_div(q1, q2)
    mid = (q2 - q1) if (q1 is not None and q2 is not None) else None
    q2_ratio = _safe_div(mid, fin)
    return {"fraction_q1_ratio": q1_ratio, "fraction_q2_ratio": q2_ratio}


def pace_shape_metrics(pp: PastPerformanceLine) -> dict[str, Optional[float]]:
    """Per-PP pace metrics. Returns None for missing inputs (don't fabricate)."""
    bl_q1, bl_q2, bl_fin = (
        pp.beaten_lengths_q1, pp.beaten_lengths_q2, pp.lengths_behind,
    )

    early_speed = -bl_q1 if bl_q1 is not None else None  # negative-lengths-behind
    late_kick = (
        (bl_q2 - bl_fin)
        if (bl_q2 is not None and bl_fin is not None)
        else None
    )

    return {
        "early_speed": early_speed,
        "late_kick": late_kick,
        **fraction_ratios(pp),
    }


def horse_pace_summary(
    entry: HorseEntry, window: int = RECENT_PP_WINDOW
) -> dict[str, Optional[float]]:
    """Summarise pace tendencies over the horse's last `window` PP lines.

    `pace_style_score` is the mean of `beaten_lengths_q1` across the window —
    smaller values mean the horse habitually sits near the lead at the first
    call. None when no first-call data is available.

    `early_speed_avg` is the mean early_speed (i.e. -lengths_behind_q1).
    `late_kick_avg` is the mean late_kick.
    """
    pp_window = entry.pp_lines[:window]
    bl_q1 = [p.beaten_lengths_q1 for p in pp_window if p.beaten_lengths_q1 is not None]
    early = [
        -p.beaten_lengths_q1 for p in pp_window if p.beaten_lengths_q1 is not None
    ]
    late = []
    for p in pp_window:
        if p.beaten_lengths_q2 is not None and p.lengths_behind is not None:
            late.append(p.beaten_lengths_q2 - p.lengths_behind)

    return {
        "pace_style_score": float(np.mean(bl_q1)) if bl_q1 else None,
        "early_speed_avg":  float(np.mean(early)) if early else None,
        "late_kick_avg":    float(np.mean(late)) if late else None,
        "n_pace_pps":       float(len(pp_window)),
    }


def build_pace_feature_frame(race: ParsedRace) -> pd.DataFrame:
    """Per-horse pace summary + race-level pace pressure index.

    `pace_pressure_index` is constant across the frame (one value per race)
    but stored on every row so the FeatureEngine output is a tidy long-form
    DataFrame.
    """
    rows = []
    for entry in race.entries:
        summary = horse_pace_summary(entry)
        summary["post_position"] = entry.post_position
        summary["horse_name"] = entry.horse_name
        rows.append(summary)

    df = pd.DataFrame(rows).set_index("post_position")

    # Field-relative early-speed z-score (intra-race standardisation).
    early = df["early_speed_avg"]
    if early.notna().sum() >= 2:
        m, s = early.mean(skipna=True), early.std(skipna=True, ddof=0)
        df["early_speed_zscore"] = (early - m) / s if s and s > 0 else 0.0
    else:
        df["early_speed_zscore"] = 0.0

    # Count of probable front-runners. Use pace_style_score (mean lengths-behind
    # at first call) <= threshold as the proxy.
    front_count = int(
        (df["pace_style_score"] <= FRONT_RUNNER_LENGTHS_THRESHOLD).fillna(False).sum()
    )
    df["pace_pressure_index"] = float(front_count)

    return df


__all__ = [
    "FRONT_RUNNER_LENGTHS_THRESHOLD",
    "RECENT_PP_WINDOW",
    "fraction_ratios",
    "pace_shape_metrics",
    "horse_pace_summary",
    "build_pace_feature_frame",
]
