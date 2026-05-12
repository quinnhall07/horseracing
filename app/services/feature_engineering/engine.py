"""
app/services/feature_engineering/engine.py
──────────────────────────────────────────
Orchestrator: RaceCard → ML-ready DataFrame.

The downstream Phase 3 sub-models (Speed/Form LightGBM, Pace Scenario,
Sequence Transformer, etc.) consume a long-form DataFrame with one row per
(race_number, post_position). This module:

  1. Iterates over every ParsedRace in the card.
  2. Builds per-horse summary stats:
     * speed_features.build_speed_feature_frame
     * pace_features.build_pace_feature_frame
     * class_features.build_class_feature_frame
     * connections.build_connection_feature_frame
  3. Computes layoff fitness from days_since_last (derived from PP gap).
  4. Joins everything horizontally on (race_number, post_position) and
     appends a small set of universally-present identifier columns.

The output schema is stable. New features should be ADDITIVE — never change
the meaning of an existing column without an ADR + an explicit migration
plan in PROGRESS.md.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from app.core.logging import get_logger
from app.schemas.race import HorseEntry, ParsedRace, RaceCard
from app.services.feature_engineering.class_features import build_class_feature_frame
from app.services.feature_engineering.connections import (
    build_connection_feature_frame,
)
from app.services.feature_engineering.layoff import (
    DEFAULT_LAMBDA,
    DEFAULT_RECOVERY_THRESHOLD_DAYS,
    apply_layoff_features,
)
from app.services.feature_engineering.pace_features import build_pace_feature_frame
from app.services.feature_engineering.speed_features import build_speed_feature_frame

log = get_logger(__name__)


def _days_since_last(entry: HorseEntry, today: date) -> Optional[int]:
    """Compute the gap (in days) between the horse's most recent PP and today.

    Returns None for first-time starters (no PP history) so downstream layoff
    code can treat them as their own regime.
    """
    if not entry.pp_lines:
        return None
    return (today - entry.pp_lines[0].race_date).days


class FeatureEngine:
    """Stateless transformer.

    Parameters
    ----------
    layoff_recovery_threshold_days
        Pass-through to layoff.apply_layoff_features. Overridable when the
        Phase 3 calibration produces a tuned value.
    layoff_decay_lambda
        Same — pass-through to layoff.apply_layoff_features.
    """

    def __init__(
        self,
        layoff_recovery_threshold_days: float = DEFAULT_RECOVERY_THRESHOLD_DAYS,
        layoff_decay_lambda: float = DEFAULT_LAMBDA,
    ):
        self.layoff_recovery_threshold_days = layoff_recovery_threshold_days
        self.layoff_decay_lambda = layoff_decay_lambda

    # ── Per-race ────────────────────────────────────────────────────────────

    def transform_race(self, race: ParsedRace) -> pd.DataFrame:
        """Build the feature matrix for a single race (indexed by post_position)."""
        speed = build_speed_feature_frame(race)
        pace = build_pace_feature_frame(race)
        cls = build_class_feature_frame(race)
        conn = build_connection_feature_frame(race)

        # Per-horse identifiers + simple parse-time fields (entry-level scalars).
        rows = []
        today = race.header.race_date
        for entry in race.entries:
            rows.append({
                "post_position":      entry.post_position,
                "horse_name":         entry.horse_name,
                "morning_line_odds":  entry.morning_line_odds,
                "ml_implied_prob":    entry.ml_implied_prob,
                "weight_lbs":         entry.weight_lbs,
                "days_since_last":    _days_since_last(entry, today),
                "n_pp":               entry.n_pp,
            })
        base = pd.DataFrame(rows).set_index("post_position")

        apply_layoff_features(
            base,
            days_col="days_since_last",
            recovery_threshold=self.layoff_recovery_threshold_days,
            decay_lambda=self.layoff_decay_lambda,
        )

        # Drop duplicate horse_name columns introduced by per-module frames;
        # keep the one in `base` as canonical.
        for sub in (speed, pace, cls, conn):
            if "horse_name" in sub.columns:
                sub.drop(columns=["horse_name"], inplace=True)

        df = base.join(speed).join(pace).join(cls).join(conn)

        # Race-level identifiers replicated across rows for tidy long-form output.
        h = race.header
        df.insert(0, "race_number", h.race_number)
        df.insert(1, "race_date", h.race_date)
        df.insert(2, "track_code", h.track_code)
        df.insert(3, "distance_furlongs", h.distance_furlongs)
        df.insert(4, "surface", h.surface.value)
        df.insert(5, "condition", h.condition.value)
        df.insert(6, "race_type", h.race_type.value)
        df.insert(7, "field_size", race.field_size)
        df.insert(8, "is_sprint", h.is_sprint)

        # Field-relative weight (lightweight; saves a round-trip in modelling).
        if df["weight_lbs"].notna().any():
            mean_w = df["weight_lbs"].mean(skipna=True)
            df["weight_lbs_delta"] = df["weight_lbs"] - mean_w
        else:
            df["weight_lbs_delta"] = pd.NA

        df.reset_index(inplace=True)  # post_position becomes a column
        return df

    # ── Per-card ────────────────────────────────────────────────────────────

    def transform(self, card: RaceCard) -> pd.DataFrame:
        """Build the full feature matrix for a card.

        Concatenates per-race frames; preserves the (race_number, post_position)
        primary-key contract by including both as columns. Returns an empty
        DataFrame for cards with zero races (graceful no-op).
        """
        frames: list[pd.DataFrame] = []
        for race in card.races:
            try:
                frames.append(self.transform_race(race))
            except Exception as exc:
                log.error(
                    "feature_engine.race_failed",
                    race_number=race.header.race_number,
                    error=str(exc),
                )
                continue

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


__all__ = ["FeatureEngine"]
