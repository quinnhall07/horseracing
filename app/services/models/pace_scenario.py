"""
app/services/models/pace_scenario.py
────────────────────────────────────
Layer 1b — Pace Scenario Model (LightGBM).

The pace model takes the field's mix of running styles and predicts a
per-horse `pace_advantage` score: how favourable is the race shape for
this individual horse's preferred style?

STATUS — SCAFFOLDING ONLY (Phase 3 deferred work).

The current master DB does not populate `fraction_q1_sec`, `fraction_q2_sec`,
`beaten_lengths_q1`, or `beaten_lengths_q2` for any row (see PROGRESS.md
"Known Export Caveats"). Without those fields we cannot:
  * reconstruct pace shape from historical PPs
  * train a model targeting late-kick / front-running outcomes
  * compute the `pace_pressure_index` over historical races

The class structure here is a placeholder so other layers can `from
app.services.models.pace_scenario import PaceScenarioModel` without an
ImportError, and so the meta-learner can wire its slot in advance. The
`fit()` method raises NotImplementedError — calling code MUST check
`is_trainable_with(df)` first or rely on `predict_proba()` returning a
neutral 0.5 constant.

Unblock path: source datasets with fraction times (Equibase PP feeds,
DRF charts). Once available, the implementation here will mirror
`SpeedFormModel` (LightGBM binary, in-race softmax) with the target being
"finished within 1 length of the winner" — a robust pace-shape outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger

log = get_logger(__name__)


REQUIRED_FRACTION_COLUMNS: tuple[str, ...] = (
    "fraction_q1_sec",
    "fraction_q2_sec",
    "beaten_lengths_q1",
    "beaten_lengths_q2",
)


@dataclass
class PaceScenarioConfig:
    min_non_null_fraction_pct: float = 0.50
    """Refuse to train if fewer than half the rows have fraction data."""


class PaceScenarioModel:
    """Placeholder — see module docstring for unblock criteria."""

    ARTIFACT_VERSION: str = "0"

    def __init__(self, config: Optional[PaceScenarioConfig] = None):
        self.config = config or PaceScenarioConfig()
        self.is_fitted = False

    @classmethod
    def is_trainable_with(cls, df: pd.DataFrame) -> bool:
        for col in REQUIRED_FRACTION_COLUMNS:
            if col not in df.columns:
                return False
            non_null_pct = float(df[col].notna().mean()) if len(df) else 0.0
            if non_null_pct < 0.10:
                return False
        return True

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "PaceScenarioModel":
        if not self.is_trainable_with(train_df):
            raise NotImplementedError(
                "PaceScenarioModel cannot be trained — the master DB lacks "
                "the fractional time columns this model targets. See "
                "PROGRESS.md 'Known Export Caveats'."
            )
        # Future implementation hook — once data is available, this branch
        # gets the LightGBM training loop modelled after SpeedFormModel.
        raise NotImplementedError("Pace data unavailable; cannot fit.")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Neutral fallback — uniform 0.5 across rows.

        The meta-learner is allowed to consume this; orthogonalisation
        (CLAUDE.md §2) means a constant feature simply contributes zero
        information once standardised."""
        return np.full(len(df), 0.5, dtype=float)

    def save(self, path: Path) -> dict:
        log.warning("pace_scenario.save_called_on_stub", path=str(path))
        return {"artifact_version": self.ARTIFACT_VERSION, "stub": True}

    @classmethod
    def load(cls, path: Path) -> "PaceScenarioModel":
        return cls()


__all__ = ["PaceScenarioConfig", "PaceScenarioModel"]
