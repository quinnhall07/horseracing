"""
app/services/models/market.py
─────────────────────────────
Layer 1e — Market / Smart Money Model.

Inputs: market odds-derived signals from PP history (final odds movement,
favourite frequency). Output: a market-derived prior P(win) for today's
horse — used as a feature by the meta-learner so the meta can learn how
much to trust the public on this kind of race.

STATUS — V1: empirical odds-to-probability calibration with shrinkage.

The simplest market model is just the implied probability of the final
odds, normalised for the track take. We approximate the take as 18%
(industry-typical for win pools) — i.e., raw implied probs across the
field sum to ~1.18. The model:
  1. Converts `odds_final` (decimal) to implied prob = 1 / odds.
  2. Within each historical race, computes the inverse-take normalisation.
  3. At training time, learns the historical (binned-implied-prob → actual
     win rate) calibration curve via isotonic regression — this is Phase 4
     machinery applied here at training to produce a market-prior column.

The "smart money" component (late-money odds drift) requires per-PP odds
trajectories that aren't in the current parquet, so it's deferred. The
V1 output column is `market_implied_prob_calibrated`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class MarketModelConfig:
    odds_col: str = "odds_final"
    race_id_col: str = "race_id"
    label_col: str = "win"
    assumed_track_take: float = 0.18
    """Used only for the unnormalised implied prob — the isotonic step
    overwrites it during fit."""


class MarketModel:
    """Empirical odds-to-probability calibrator."""

    ARTIFACT_VERSION: str = "1"

    def __init__(self, config: Optional[MarketModelConfig] = None):
        self.config = config or MarketModelConfig()
        self.iso: Optional[IsotonicRegression] = None
        self.is_fitted = False

    @staticmethod
    def raw_implied_prob(odds: pd.Series) -> pd.Series:
        """Implied prob from decimal odds. NaN-propagating, no smoothing."""
        return 1.0 / odds.where(odds > 0)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "MarketModel":
        cfg = self.config
        df = train_df[[cfg.odds_col, cfg.race_id_col, cfg.label_col]].copy()
        df["raw"] = self.raw_implied_prob(df[cfg.odds_col])
        df = df.dropna(subset=["raw"])

        # Normalise within race so each field sums to 1 (removes take per-race).
        df["norm"] = df["raw"] / df.groupby(cfg.race_id_col)["raw"].transform("sum")

        # Isotonic calibration: predicted norm-implied-prob → actual win rate.
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(df["norm"].to_numpy(), df[cfg.label_col].to_numpy())
        self.is_fitted = True

        log.info(
            "market.fitted",
            rows_used=len(df),
            mean_raw=float(df["raw"].mean()),
            mean_norm=float(df["norm"].mean()),
            mean_label=float(df[cfg.label_col].mean()),
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.iso is None:
            raise RuntimeError("MarketModel not fitted.")
        cfg = self.config
        raw = self.raw_implied_prob(df[cfg.odds_col])
        # Field normalisation (within-race)
        sums = raw.groupby(df[cfg.race_id_col]).transform("sum")
        norm = raw / sums
        out = self.iso.predict(norm.fillna(0.0).to_numpy())
        # If odds were missing for the row, NaN the prediction so downstream
        # can decide whether to fall back to the global mean.
        out = np.where(raw.isna().to_numpy(), np.nan, out)
        return out

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> dict:
        if not self.is_fitted or self.iso is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": self.ARTIFACT_VERSION,
            "config": asdict(self.config),
            "iso_x": self.iso.X_thresholds_.tolist(),
            "iso_y": self.iso.y_thresholds_.tolist(),
        }
        with open(path / "model.json", "w") as fh:
            json.dump(payload, fh, indent=2)
        log.info("market.saved", path=str(path))
        return payload

    @classmethod
    def load(cls, path: Path) -> "MarketModel":
        path = Path(path)
        with open(path / "model.json") as fh:
            payload = json.load(fh)
        cfg = MarketModelConfig(**payload["config"])
        obj = cls(config=cfg)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        xs = np.asarray(payload["iso_x"], dtype=float)
        ys = np.asarray(payload["iso_y"], dtype=float)
        iso.X_thresholds_ = xs
        iso.y_thresholds_ = ys
        iso.X_min_ = float(xs.min())
        iso.X_max_ = float(xs.max())
        iso.increasing_ = True
        # Rebuild the linear interpolation that sklearn's IsotonicRegression
        # constructs at fit time. `out_of_bounds="clip"` means we leave the
        # boundary handling to numpy.clip in _transform; bounds_error=False
        # mirrors what sklearn._build_f does for the same setting.
        iso.f_ = interp1d(
            xs, ys, kind="linear", bounds_error=False,
            fill_value=(float(ys[0]), float(ys[-1])),
        )
        obj.iso = iso
        obj.is_fitted = True
        return obj


__all__ = ["MarketModelConfig", "MarketModel"]
