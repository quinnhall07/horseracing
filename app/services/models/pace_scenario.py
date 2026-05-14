"""
app/services/models/pace_scenario.py
────────────────────────────────────
Layer 1b — Pace Scenario Model (LightGBM).

The model consumes the per-horse sectional times exposed in the master DB:

    fraction_q1_sec   — cumulative seconds at first call
    fraction_q2_sec   — cumulative seconds at second call
    beaten_lengths_q1 — lengths behind leader at first call
    beaten_lengths_q2 — lengths behind leader at second call

…plus race-relative derivations (pace pressure, position rank, fraction
deltas) and trains a LightGBM binary classifier on the win label, like
Speed/Form but on the orthogonal pace signal. Rows without pace data fall
back to 0.5 — the gate (`is_trainable_with`) enforces a minimum absolute
row count of 5 000 with non-null pace cols before declaring the model
trainable.

Status — TRAINABLE on the v2 master DB (ADR-047 supersedes the Pace
portion of ADR-026 once the parquet carries fractional data; the gate
keeps the model in stub mode for parquets that don't yet).
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

log = get_logger(__name__)


REQUIRED_FRACTION_COLUMNS: tuple[str, ...] = (
    "fraction_q1_sec",
    "fraction_q2_sec",
    "beaten_lengths_q1",
    "beaten_lengths_q2",
)


PACE_FEATURE_COLUMNS: tuple[str, ...] = (
    # Raw per-horse pace features
    "fraction_q1_sec",
    "fraction_q2_sec",
    "beaten_lengths_q1",
    "beaten_lengths_q2",
    # Derived per-row features
    "fraction_q1_per_furlong",      # pace at first call, normalised
    "fraction_q2_delta",            # seconds from call 1 to call 2
    "beaten_lengths_delta",         # gaining/losing ground between calls
    # Field-relative within today's race
    "pace_pressure",                # std of fraction_q1_sec within race
    "beaten_lengths_q1_rank",       # 1 = on the lead at first call
    "fraction_q1_zscore",           # how fast vs. field at first call
    # Race context the meta-learner can also pivot on
    "distance_furlongs",
    "field_size",
)


CATEGORICAL_FEATURES: tuple[str, ...] = (
    "surface", "condition", "race_type", "jurisdiction",
)


@dataclass
class PaceScenarioConfig:
    min_non_null_fraction_pct: float = 0.001
    """Refuse to train if no rows have pace data. Set deliberately low —
    the absolute-count gate is the real guard."""

    min_rows_with_data: int = 5_000
    """Absolute minimum rows with non-null pace columns. Pace data is
    jurisdiction-sparse (HK only in the v2 export) so a 10% relative
    threshold rejects perfectly usable training corpora."""

    # LightGBM hyperparameters — copied from SpeedForm with smaller leaf
    # counts since the labelled subset is roughly an order of magnitude
    # smaller.
    num_leaves: int = 31
    learning_rate: float = 0.05
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    seed: int = 19

    feature_columns: tuple[str, ...] = field(
        default_factory=lambda: tuple(PACE_FEATURE_COLUMNS)
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


@dataclass
class PaceMetrics:
    train_log_loss: float
    val_log_loss: float
    train_auc: float
    val_auc: float
    n_train_with_pace: int
    n_val_with_pace: int


def _add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the pace feature columns on the rows that have raw fraction data.

    For rows where the raw fraction cols are NaN the derived cols stay NaN —
    LightGBM handles NaN natively and predict_proba forces the output to 0.5
    for such rows (matching the neutral-fallback contract of the stub).
    """
    out = df.copy()
    if "fraction_q1_sec" not in out.columns:
        return out

    f1 = pd.to_numeric(out["fraction_q1_sec"], errors="coerce")
    f2 = pd.to_numeric(out["fraction_q2_sec"], errors="coerce")
    b1 = pd.to_numeric(out["beaten_lengths_q1"], errors="coerce")
    b2 = pd.to_numeric(out["beaten_lengths_q2"], errors="coerce")
    dist = pd.to_numeric(out.get("distance_furlongs"), errors="coerce")

    # Pace at first call, normalised by distance. Avoid div-by-zero with NaN
    # propagation; LightGBM will still see the NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_furlong = f1 / (dist * 0.25).replace(0, np.nan)
    out["fraction_q1_per_furlong"] = per_furlong
    out["fraction_q2_delta"] = f2 - f1
    out["beaten_lengths_delta"] = b2 - b1

    if "race_id" in out.columns:
        race_grp = out.groupby("race_id", sort=False)
        # std of first-call times within the race — high = duel/contested early
        out["pace_pressure"] = race_grp["fraction_q1_sec"].transform("std")
        # rank of beaten_lengths_q1 within race (smaller = closer to lead).
        # rank skips NaN by default (na_option='keep' is the default for rank()).
        out["beaten_lengths_q1_rank"] = race_grp["beaten_lengths_q1"].rank(
            method="min", na_option="keep"
        )
        # z-score of fraction_q1_sec within race
        race_mean = race_grp["fraction_q1_sec"].transform("mean")
        race_std = race_grp["fraction_q1_sec"].transform("std")
        out["fraction_q1_zscore"] = (f1 - race_mean) / race_std.replace(0, np.nan)
    else:
        out["pace_pressure"] = np.nan
        out["beaten_lengths_q1_rank"] = np.nan
        out["fraction_q1_zscore"] = np.nan

    return out


def _race_softmax(scores: pd.Series, race_ids: pd.Series) -> pd.Series:
    df = pd.DataFrame({"score": scores.values, "race_id": race_ids.values},
                      index=scores.index)
    df["score"] = df["score"].astype(float)
    df["centered"] = df["score"] - df.groupby("race_id")["score"].transform("max")
    df["exp"] = np.exp(df["centered"])
    df["sum"] = df.groupby("race_id")["exp"].transform("sum")
    out = df["exp"] / df["sum"]
    out.index = scores.index
    return out


class PaceScenarioModel:
    """LightGBM binary classifier on the per-horse pace signal."""

    ARTIFACT_VERSION: str = "1"

    def __init__(self, config: Optional[PaceScenarioConfig] = None):
        self.config = config or PaceScenarioConfig()
        self.booster: Optional[lgb.Booster] = None
        self.metrics: Optional[PaceMetrics] = None
        self.is_fitted: bool = False

    # ── Trainability gate ────────────────────────────────────────────────

    @classmethod
    def is_trainable_with(
        cls,
        df: pd.DataFrame,
        config: Optional[PaceScenarioConfig] = None,
    ) -> bool:
        cfg = config or PaceScenarioConfig()
        for col in REQUIRED_FRACTION_COLUMNS:
            if col not in df.columns:
                return False
            non_null = int(df[col].notna().sum()) if len(df) else 0
            non_null_pct = float(non_null / max(1, len(df)))
            if non_null < cfg.min_rows_with_data:
                return False
            if non_null_pct < cfg.min_non_null_fraction_pct:
                return False
        return True

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "PaceScenarioModel":
        if not self.is_trainable_with(train_df, self.config):
            raise RuntimeError(
                "PaceScenarioModel cannot be trained on this parquet — the "
                "fractional-time columns are missing or below the row-count "
                "threshold. See ADR-047."
            )

        cfg = self.config

        # Restrict to rows with non-null pace data. LightGBM handles NaN in
        # the FEATURES, but we don't want to train on rows whose pace signal
        # is entirely absent — they add no information to the pace head.
        train_mask = train_df[list(REQUIRED_FRACTION_COLUMNS)].notna().all(axis=1)
        train_slice = _add_pace_features(train_df[train_mask].copy())

        val_mask = pd.Series(False, index=val_df.index) if val_df is not None else None
        val_slice = None
        if val_df is not None and len(val_df):
            val_mask = val_df[list(REQUIRED_FRACTION_COLUMNS)].notna().all(axis=1)
            if val_mask.any():
                val_slice = _add_pace_features(val_df[val_mask].copy())

        X_train, y_train = self._extract_xy(train_slice)
        train_dset = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=[c for c in cfg.categorical_columns if c in X_train.columns],
            free_raw_data=False,
        )

        valid_sets = [train_dset]
        valid_names = ["train"]
        if val_slice is not None and len(val_slice):
            X_val, y_val = self._extract_xy(val_slice)
            val_dset = lgb.Dataset(
                X_val, label=y_val,
                categorical_feature=[c for c in cfg.categorical_columns if c in X_val.columns],
                free_raw_data=False,
                reference=train_dset,
            )
            valid_sets.append(val_dset)
            valid_names.append("val")

        log.info(
            "pace.fit.start",
            n_train=int(len(train_slice)),
            n_val=int(len(val_slice)) if val_slice is not None else 0,
        )
        self.booster = lgb.train(
            cfg.to_lgb_params(),
            train_dset,
            num_boost_round=cfg.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        train_pred = self.booster.predict(X_train, num_iteration=self.booster.best_iteration)
        train_ll = float(log_loss(y_train, np.clip(train_pred, 1e-7, 1 - 1e-7), labels=[0, 1]))
        train_auc = float(roc_auc_score(y_train, train_pred)) if len(set(y_train)) > 1 else float("nan")

        val_ll = val_auc = float("nan")
        if val_slice is not None and len(val_slice):
            X_val, y_val = self._extract_xy(val_slice)
            val_pred = self.booster.predict(X_val, num_iteration=self.booster.best_iteration)
            val_ll = float(log_loss(y_val, np.clip(val_pred, 1e-7, 1 - 1e-7), labels=[0, 1]))
            val_auc = float(roc_auc_score(y_val, val_pred)) if len(set(y_val)) > 1 else float("nan")

        self.metrics = PaceMetrics(
            train_log_loss=train_ll,
            val_log_loss=val_ll,
            train_auc=train_auc,
            val_auc=val_auc,
            n_train_with_pace=int(train_mask.sum()),
            n_val_with_pace=int(val_mask.sum()) if val_mask is not None else 0,
        )
        self.is_fitted = True
        log.info("pace.fit.complete", **asdict(self.metrics))
        return self

    def _extract_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        cfg = self.config
        # Filter to columns that actually exist (forward-compat with sparser parquets).
        feature_cols = [c for c in cfg.feature_columns if c in df.columns]
        cat_cols = [c for c in cfg.categorical_columns if c in df.columns]
        X = df[feature_cols + cat_cols].copy()
        for c in cat_cols:
            X[c] = X[c].astype("category")
        y = df[cfg.label_column].astype(int).to_numpy()
        return X, y

    # ── Predict ──────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row P(win). Rows with NO pace data return 0.5 (neutral)."""
        out = np.full(len(df), 0.5, dtype=float)
        if not self.is_fitted or self.booster is None:
            return out
        # If pace cols are absent entirely, return all-0.5.
        for c in REQUIRED_FRACTION_COLUMNS:
            if c not in df.columns:
                return out
        # Only predict on rows where the raw pace columns are present. Rows
        # with ANY null pace col stay at the neutral 0.5 default — LightGBM
        # rejects object-dtype columns (pandas keeps the col as object when
        # all values are None), and the rows are uninformative anyway.
        has_pace_mask = df[list(REQUIRED_FRACTION_COLUMNS)].notna().all(axis=1).to_numpy()
        if not has_pace_mask.any():
            return out
        feats = _add_pace_features(df.loc[has_pace_mask].copy())
        cfg = self.config
        feature_cols = [c for c in cfg.feature_columns if c in feats.columns]
        cat_cols = [c for c in cfg.categorical_columns if c in feats.columns]
        X = feats[feature_cols + cat_cols].copy()
        for c in cat_cols:
            X[c] = X[c].astype("category")
        proba = self.booster.predict(X, num_iteration=self.booster.best_iteration)
        out[has_pace_mask] = np.asarray(proba, dtype=float)
        return out

    def predict_softmax(
        self, df: pd.DataFrame, race_id_column: Optional[str] = None
    ) -> np.ndarray:
        col = race_id_column or self.config.race_id_column
        proba = self.predict_proba(df)
        if col not in df.columns:
            return proba
        race_ids = df[col]
        out = _race_softmax(pd.Series(proba, index=df.index), race_ids)
        return out.to_numpy()

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: Path) -> dict:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not self.is_fitted or self.booster is None:
            (path / "metadata.json").write_text(json.dumps(
                {"artifact_version": self.ARTIFACT_VERSION, "stub": True}, indent=2
            ))
            log.warning("pace.save_called_on_stub", path=str(path))
            return {"artifact_version": self.ARTIFACT_VERSION, "stub": True}

        self.booster.save_model(str(path / "booster.txt"))
        meta = {
            "artifact_version": self.ARTIFACT_VERSION,
            "stub": False,
            "config": asdict(self.config),
            "metrics": asdict(self.metrics) if self.metrics else None,
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta

    @classmethod
    def load(cls, path: Path) -> "PaceScenarioModel":
        path = Path(path)
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            return cls()
        meta = json.loads(meta_path.read_text())
        if meta.get("stub", True):
            return cls()
        cfg = PaceScenarioConfig(**meta["config"])
        m = cls(cfg)
        m.booster = lgb.Booster(model_file=str(path / "booster.txt"))
        if meta.get("metrics"):
            m.metrics = PaceMetrics(**meta["metrics"])
        m.is_fitted = True
        return m


__all__ = [
    "PaceScenarioConfig",
    "PaceScenarioModel",
    "REQUIRED_FRACTION_COLUMNS",
    "PACE_FEATURE_COLUMNS",
]
