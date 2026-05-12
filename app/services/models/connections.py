"""
app/services/models/connections.py
──────────────────────────────────
Layer 1d — Jockey × Trainer Bayesian Hierarchical Model.

In production this is a partial-pooling Bayesian hierarchical model that
estimates per-jockey, per-trainer, and per-pair win rates with shrinkage
toward the population mean. The output for a given horse-race is
P(win | jockey, trainer, jurisdiction) — a complementary signal to the
LightGBM Speed/Form output.

STATUS — V1: simple empirical-Bayes shrinkage estimator.

We compute, on the training set:
  * win_rate(jockey, trainer) over a shared horse-race population
  * shrunk toward win_rate(jurisdiction) by a beta-binomial prior with
    pseudo-count `prior_strength`

At inference time, for each row, we look up the (jockey, trainer)
combination's shrunken win rate. Out-of-vocabulary pairs fall back to the
solo jockey rate, then solo trainer rate, then jurisdiction baseline.

This is intentionally simple — the full PyMC hierarchical model is a Phase 4
follow-up. The empirical estimator captures most of the signal a horse-race
model gets from connections.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ConnectionsConfig:
    prior_strength: float = 30.0
    """Beta-binomial pseudo-count for shrinkage. 30 ≈ "see 30 races before
    trusting an empirical rate". Lightly tuned."""
    min_jockey_starts: int = 10
    """Below this, the solo-jockey rate falls back to jurisdiction baseline."""
    min_trainer_starts: int = 10
    min_pair_starts: int = 5

    jockey_col: str = "jockey_name"
    trainer_col: str = "trainer_name"
    jurisdiction_col: str = "jurisdiction"
    label_col: str = "win"


def _shrunk_rate(wins: int, starts: int, prior_mean: float, alpha: float) -> float:
    """Beta-binomial posterior mean with pseudo-count `alpha`."""
    if starts <= 0:
        return prior_mean
    return (wins + alpha * prior_mean) / (starts + alpha)


class ConnectionsModel:
    """Empirical-Bayes connections estimator."""

    ARTIFACT_VERSION: str = "1"

    def __init__(self, config: Optional[ConnectionsConfig] = None):
        self.config = config or ConnectionsConfig()
        self.jurisdiction_rate: dict[str, float] = {}
        self.jockey_rate: dict[tuple[str, str], float] = {}
        self.trainer_rate: dict[tuple[str, str], float] = {}
        self.pair_rate: dict[tuple[str, str, str], float] = {}
        self.global_rate: float = 0.0
        self.is_fitted = False

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "ConnectionsModel":
        cfg = self.config
        df = train_df

        self.global_rate = float(df[cfg.label_col].mean())

        # Per-jurisdiction baseline.
        for jur, grp in df.groupby(cfg.jurisdiction_col, observed=True):
            self.jurisdiction_rate[str(jur)] = float(grp[cfg.label_col].mean())

        # Solo-jockey shrunken rate.
        for (jur, jockey), grp in df.groupby([cfg.jurisdiction_col, cfg.jockey_col], observed=True):
            if pd.isna(jockey) or not str(jockey).strip():
                continue
            n = len(grp)
            if n < cfg.min_jockey_starts:
                continue
            self.jockey_rate[(str(jur), str(jockey))] = _shrunk_rate(
                int(grp[cfg.label_col].sum()), n,
                self.jurisdiction_rate.get(str(jur), self.global_rate),
                cfg.prior_strength,
            )

        # Solo-trainer shrunken rate.
        for (jur, trainer), grp in df.groupby([cfg.jurisdiction_col, cfg.trainer_col], observed=True):
            if pd.isna(trainer) or not str(trainer).strip():
                continue
            n = len(grp)
            if n < cfg.min_trainer_starts:
                continue
            self.trainer_rate[(str(jur), str(trainer))] = _shrunk_rate(
                int(grp[cfg.label_col].sum()), n,
                self.jurisdiction_rate.get(str(jur), self.global_rate),
                cfg.prior_strength,
            )

        # Jockey × trainer pair shrunken rate.
        for (jur, jockey, trainer), grp in df.groupby(
            [cfg.jurisdiction_col, cfg.jockey_col, cfg.trainer_col], observed=True,
        ):
            if pd.isna(jockey) or pd.isna(trainer):
                continue
            n = len(grp)
            if n < cfg.min_pair_starts:
                continue
            self.pair_rate[(str(jur), str(jockey), str(trainer))] = _shrunk_rate(
                int(grp[cfg.label_col].sum()), n,
                self.jurisdiction_rate.get(str(jur), self.global_rate),
                cfg.prior_strength,
            )

        self.is_fitted = True
        log.info(
            "connections.fitted",
            global_rate=self.global_rate,
            jurisdictions=len(self.jurisdiction_rate),
            jockeys=len(self.jockey_rate),
            trainers=len(self.trainer_rate),
            pairs=len(self.pair_rate),
        )
        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ConnectionsModel not fitted.")
        cfg = self.config

        out = np.empty(len(df), dtype=float)
        jurs = df[cfg.jurisdiction_col].astype(str).to_numpy()
        jocks = df[cfg.jockey_col].astype(str).to_numpy()
        trns = df[cfg.trainer_col].astype(str).to_numpy()

        for i in range(len(df)):
            jur, jck, trn = jurs[i], jocks[i], trns[i]
            # Try pair, then solo jockey, then solo trainer, then jurisdiction baseline.
            val = (
                self.pair_rate.get((jur, jck, trn))
                or self.jockey_rate.get((jur, jck))
                or self.trainer_rate.get((jur, trn))
                or self.jurisdiction_rate.get(jur)
                or self.global_rate
            )
            out[i] = val
        return out

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Use string-joined keys so JSON-serializable.
        payload = {
            "artifact_version": self.ARTIFACT_VERSION,
            "config": asdict(self.config),
            "global_rate": self.global_rate,
            "jurisdiction_rate": self.jurisdiction_rate,
            "jockey_rate": {"||".join(k): v for k, v in self.jockey_rate.items()},
            "trainer_rate": {"||".join(k): v for k, v in self.trainer_rate.items()},
            "pair_rate": {"||".join(k): v for k, v in self.pair_rate.items()},
        }
        with open(path / "model.json", "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        log.info("connections.saved", path=str(path),
                 n_jockeys=len(self.jockey_rate),
                 n_trainers=len(self.trainer_rate),
                 n_pairs=len(self.pair_rate))
        return payload

    @classmethod
    def load(cls, path: Path) -> "ConnectionsModel":
        path = Path(path)
        with open(path / "model.json") as fh:
            payload = json.load(fh)
        cfg = ConnectionsConfig(**payload["config"])
        obj = cls(config=cfg)
        obj.global_rate = float(payload["global_rate"])
        obj.jurisdiction_rate = {k: float(v) for k, v in payload["jurisdiction_rate"].items()}
        obj.jockey_rate = {tuple(k.split("||")): float(v) for k, v in payload["jockey_rate"].items()}
        obj.trainer_rate = {tuple(k.split("||")): float(v) for k, v in payload["trainer_rate"].items()}
        obj.pair_rate = {tuple(k.split("||")): float(v) for k, v in payload["pair_rate"].items()}
        obj.is_fitted = True
        return obj


__all__ = ["ConnectionsConfig", "ConnectionsModel"]
