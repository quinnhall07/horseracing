"""
app/services/models/meta_learner.py
───────────────────────────────────
Layer 2 — Stacking Meta-Learner.

Inputs: the raw probabilities from each of the Layer-1 sub-models, plus a
small set of meta-features (field size, distance, race type). Output: a
single P(win) per horse that the calibration layer (Phase 4) then transforms
into a calibrated probability.

Per CLAUDE.md §2: sub-model inputs MUST be orthogonalised before being fed
to the meta-learner. Speed figures already incorporate pace; the residual
of pace_scenario.predict_proba after regressing it against speed_form is
what carries the unique pace information. This module applies the
residualisation step explicitly via `_orthogonalise`.

The current implementation uses LightGBM as the meta-model — same library
as Speed/Form for consistency, but with a much smaller capacity (few leaves,
heavy regularisation) since the inputs are already strong learners.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class MetaLearnerConfig:
    """Hyperparameters for the stacking head."""
    num_leaves: int = 15
    learning_rate: float = 0.05
    num_boost_round: int = 200
    early_stopping_rounds: int = 30
    min_data_in_leaf: int = 500
    lambda_l2: float = 5.0
    seed: int = 19

    sub_model_columns: tuple[str, ...] = (
        "speed_form_proba",
        "pace_scenario_proba",
        "sequence_proba",
        "connections_proba",
        "market_proba",
    )
    """Column names the meta-learner expects from upstream sub-models."""

    meta_feature_columns: tuple[str, ...] = (
        "field_size",
        "distance_furlongs",
    )

    label_col: str = "win"
    race_id_col: str = "race_id"

    def to_lgb_params(self) -> dict:
        return {
            "objective": "binary",
            "metric": ["binary_logloss"],
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l2": self.lambda_l2,
            "verbose": -1,
            "seed": self.seed,
        }


def _orthogonalise(
    X: pd.DataFrame, anchor_col: str
) -> pd.DataFrame:
    """Replace each sub-model column (except `anchor_col`) with the residual
    after regressing it linearly on the anchor. Speed/Form is the anchor; the
    other layers' residuals are what they add beyond what speed already knows.
    """
    Y = X.copy()
    anchor = X[anchor_col].fillna(X[anchor_col].mean()).to_numpy().reshape(-1, 1)
    for col in X.columns:
        if col == anchor_col:
            continue
        y = X[col].fillna(X[col].mean()).to_numpy()
        if np.allclose(y.std(), 0.0):
            Y[col] = 0.0
            continue
        reg = LinearRegression().fit(anchor, y)
        Y[col] = y - reg.predict(anchor)
    return Y


class MetaLearner:
    """LightGBM stacking head over the 5 sub-model outputs + meta features."""

    ARTIFACT_VERSION: str = "1"

    def __init__(self, config: Optional[MetaLearnerConfig] = None):
        self.config = config or MetaLearnerConfig()
        self.booster: Optional[lgb.Booster] = None
        self.is_fitted = False

    # ── Fit ────────────────────────────────────────────────────────────────

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        sub = df[list(cfg.sub_model_columns)].copy()
        sub = _orthogonalise(sub, anchor_col=cfg.sub_model_columns[0])
        meta = df[list(cfg.meta_feature_columns)].copy()
        return pd.concat([sub, meta], axis=1)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "MetaLearner":
        cfg = self.config

        X_train = self._features(train_df)
        y_train = train_df[cfg.label_col].astype(int).to_numpy()
        train_dset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        valid_sets = [train_dset]
        valid_names = ["train"]
        if val_df is not None and len(val_df):
            X_val = self._features(val_df)
            y_val = val_df[cfg.label_col].astype(int).to_numpy()
            val_dset = lgb.Dataset(X_val, label=y_val, reference=train_dset, free_raw_data=False)
            valid_sets.append(val_dset)
            valid_names.append("val")

        callbacks = [lgb.log_evaluation(period=0)]
        if val_df is not None and cfg.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            cfg.to_lgb_params(),
            train_dset,
            num_boost_round=cfg.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.is_fitted = True
        log.info("meta_learner.fitted",
                 train_size=len(train_df),
                 val_size=len(val_df) if val_df is not None else 0,
                 n_rounds=self.booster.current_iteration())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.booster is None:
            raise RuntimeError("MetaLearner not fitted.")
        return self.booster.predict(self._features(df))

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> dict:
        if not self.is_fitted or self.booster is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "booster.txt"))
        meta = {
            "artifact_version": self.ARTIFACT_VERSION,
            "config": asdict(self.config),
        }
        with open(path / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        return meta

    @classmethod
    def load(cls, path: Path) -> "MetaLearner":
        path = Path(path)
        with open(path / "metadata.json") as fh:
            meta = json.load(fh)
        cfg_kwargs = meta["config"]
        cfg_kwargs["sub_model_columns"] = tuple(cfg_kwargs["sub_model_columns"])
        cfg_kwargs["meta_feature_columns"] = tuple(cfg_kwargs["meta_feature_columns"])
        cfg = MetaLearnerConfig(**cfg_kwargs)
        obj = cls(config=cfg)
        obj.booster = lgb.Booster(model_file=str(path / "booster.txt"))
        obj.is_fitted = True
        return obj


__all__ = ["MetaLearnerConfig", "MetaLearner"]
