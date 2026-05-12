"""
app/services/models/speed_form.py
──────────────────────────────────
Layer 1a — Speed / Form Model (LightGBM).

This is the workhorse baseline. Each horse-race becomes one row with the
features registered in `training_data.SPEED_FORM_FEATURE_COLUMNS`; the
label is the binary win indicator. LightGBM produces a per-row raw score;
softmax across the in-race field then converts the scores into a
probability distribution over the starters.

Per CLAUDE.md §2:
  * The raw LightGBM output is NOT a calibrated probability. Phase 4
    calibration (Platt / isotonic) sits BETWEEN this model and the EV
    engine. We expose the raw probability and the in-race softmax
    separately so the calibration layer can choose its input.
  * Features must be field-relative. The training_data module already
    appends z-score / rank / percentile columns; this class just consumes
    them.

API contract:
    model = SpeedFormModel().fit(train_df, val_df=val_df)
    model.predict_proba(df)           → raw P(win) per row (uncalibrated)
    model.predict_softmax(df)         → race-relative probabilities (sum=1)
    model.save(path) / load(path)     → joblib artifact (model + metadata)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from app.core.logging import get_logger
from app.services.models.training_data import SPEED_FORM_FEATURE_COLUMNS

log = get_logger(__name__)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "surface", "condition", "race_type", "jurisdiction",
)
"""Columns LightGBM should treat as native categoricals."""


@dataclass
class SpeedFormMetrics:
    """Evaluation metrics emitted after `fit`. Stored alongside the artifact."""
    train_log_loss: float
    val_log_loss: float
    train_auc: float
    val_auc: float
    val_race_top1_accuracy: float
    """Fraction of validation races where the predicted top-1 horse won."""
    n_train: int
    n_val: int
    n_features: int


@dataclass
class SpeedFormConfig:
    """LightGBM hyperparameters + training meta. Lightly tuned defaults for
    a baseline — Phase 4 calibration will surface whether deeper search is
    warranted before we touch this."""
    num_leaves: int = 63
    learning_rate: float = 0.05
    num_boost_round: int = 800
    early_stopping_rounds: int = 50
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    seed: int = 17

    feature_columns: tuple[str, ...] = field(
        default_factory=lambda: tuple(SPEED_FORM_FEATURE_COLUMNS)
    )
    categorical_columns: tuple[str, ...] = CATEGORICAL_FEATURES
    label_column: str = "win"
    race_id_column: str = "race_id"

    def to_lgb_params(self) -> dict:
        return {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "verbose": -1,
            "seed": self.seed,
        }


def _race_softmax(scores: pd.Series, race_ids: pd.Series) -> pd.Series:
    """Apply a numerically-stable softmax to `scores` within each race group."""
    df = pd.DataFrame({"score": scores.values, "race_id": race_ids.values},
                      index=scores.index)
    df["score"] = df["score"].astype(float)
    # Stable softmax: subtract per-group max before exp.
    df["centered"] = df["score"] - df.groupby("race_id")["score"].transform("max")
    df["exp"] = np.exp(df["centered"])
    df["sum"] = df.groupby("race_id")["exp"].transform("sum")
    out = df["exp"] / df["sum"]
    out.index = scores.index
    return out


def _race_top1_accuracy(
    proba: pd.Series, labels: pd.Series, race_ids: pd.Series
) -> float:
    """Fraction of races whose argmax(proba) row matches the winner row.

    Skips races with no winner row (rare but possible if `finish_position`
    was NULL upstream — the training pipeline already filters those, but be
    defensive)."""
    df = pd.DataFrame({"p": proba.values, "y": labels.values, "r": race_ids.values})
    correct = 0
    total = 0
    for race_id, sub in df.groupby("r", sort=False):
        if sub["y"].sum() == 0:
            continue
        total += 1
        if sub.loc[sub["p"].idxmax(), "y"] == 1:
            correct += 1
    return correct / total if total else 0.0


class SpeedFormModel:
    """LightGBM binary classifier targeting the win label."""

    ARTIFACT_VERSION: str = "1"

    def __init__(self, config: Optional[SpeedFormConfig] = None):
        self.config = config or SpeedFormConfig()
        self.booster: Optional[lgb.Booster] = None
        self.metrics: Optional[SpeedFormMetrics] = None

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "SpeedFormModel":
        cfg = self.config

        X_train, y_train = self._extract_xy(train_df)
        train_dset = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=list(cfg.categorical_columns),
            free_raw_data=False,
        )

        valid_sets = [train_dset]
        valid_names = ["train"]
        if val_df is not None and len(val_df):
            X_val, y_val = self._extract_xy(val_df)
            val_dset = lgb.Dataset(
                X_val, label=y_val,
                categorical_feature=list(cfg.categorical_columns),
                reference=train_dset,
                free_raw_data=False,
            )
            valid_sets.append(val_dset)
            valid_names.append("val")
        else:
            X_val = y_val = None

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

        # ── Metrics ─────────────────────────────────────────────────────────
        train_pred = self.booster.predict(X_train)
        train_ll = log_loss(y_train, np.clip(train_pred, 1e-7, 1 - 1e-7))
        try:
            train_auc = roc_auc_score(y_train, train_pred)
        except ValueError:
            train_auc = float("nan")

        if val_df is not None and len(val_df) and X_val is not None:
            val_pred = self.booster.predict(X_val)
            val_ll = log_loss(y_val, np.clip(val_pred, 1e-7, 1 - 1e-7))
            try:
                val_auc = roc_auc_score(y_val, val_pred)
            except ValueError:
                val_auc = float("nan")
            race_acc = _race_top1_accuracy(
                pd.Series(val_pred, index=val_df.index),
                val_df[cfg.label_column],
                val_df[cfg.race_id_column],
            )
            n_val = len(val_df)
        else:
            val_ll = float("nan")
            val_auc = float("nan")
            race_acc = float("nan")
            n_val = 0

        self.metrics = SpeedFormMetrics(
            train_log_loss=float(train_ll),
            val_log_loss=float(val_ll),
            train_auc=float(train_auc),
            val_auc=float(val_auc),
            val_race_top1_accuracy=float(race_acc),
            n_train=len(train_df),
            n_val=n_val,
            n_features=len(cfg.feature_columns),
        )
        log.info("speed_form.fit_complete", **asdict(self.metrics))
        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def _extract_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        cfg = self.config
        X = df[list(cfg.feature_columns)].copy()
        for col in cfg.categorical_columns:
            if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
        y = df[cfg.label_column].astype(int).to_numpy()
        return X, y

    def _extract_x(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        X = df[list(cfg.feature_columns)].copy()
        for col in cfg.categorical_columns:
            if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
        return X

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Raw uncalibrated P(win) per row. Output range [0, 1] but not
        calibrated — Phase 4 wraps this in Platt/isotonic."""
        if self.booster is None:
            raise RuntimeError("Model not fitted.")
        return self.booster.predict(self._extract_x(df))

    def predict_softmax(self, df: pd.DataFrame) -> pd.Series:
        """In-race softmax normalisation — guaranteed to sum to 1 per race.

        The softmax is applied to the LightGBM RAW SCORE (pre-sigmoid),
        because that's where additivity makes sense; sigmoiding first then
        renormalising would double-squash the dynamic range and discard
        information.
        """
        if self.booster is None:
            raise RuntimeError("Model not fitted.")
        raw_scores = self.booster.predict(
            self._extract_x(df), raw_score=True
        )
        race_ids = df[self.config.race_id_column]
        return _race_softmax(
            pd.Series(raw_scores, index=df.index, dtype=float),
            race_ids,
        )

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> dict:
        """Persist the booster + config + metrics as a directory of files.

        Layout:
            path/
              booster.txt          ← LightGBM native text format
              metadata.json        ← config, metrics, artifact version
        """
        if self.booster is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.booster.save_model(str(path / "booster.txt"))
        meta = {
            "artifact_version": self.ARTIFACT_VERSION,
            "config": asdict(self.config),
            "metrics": asdict(self.metrics) if self.metrics else None,
        }
        with open(path / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        log.info("speed_form.saved", path=str(path))
        return meta

    @classmethod
    def load(cls, path: Path) -> "SpeedFormModel":
        path = Path(path)
        with open(path / "metadata.json") as fh:
            meta = json.load(fh)
        cfg_kwargs = meta["config"]
        cfg_kwargs["feature_columns"] = tuple(cfg_kwargs["feature_columns"])
        cfg_kwargs["categorical_columns"] = tuple(cfg_kwargs["categorical_columns"])
        cfg = SpeedFormConfig(**cfg_kwargs)
        obj = cls(config=cfg)
        obj.booster = lgb.Booster(model_file=str(path / "booster.txt"))
        if meta.get("metrics"):
            obj.metrics = SpeedFormMetrics(**meta["metrics"])
        return obj


__all__ = [
    "SpeedFormConfig",
    "SpeedFormMetrics",
    "SpeedFormModel",
    "CATEGORICAL_FEATURES",
]
