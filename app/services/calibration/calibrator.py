"""
app/services/calibration/calibrator.py
──────────────────────────────────────
Phase 4 — Probability Calibration.

Raw scores from LightGBM (Speed/Form, Meta-learner) are NOT calibrated
probabilities — see CLAUDE.md §2 and ADR-008. This module sits between
those models and the EV engine, transforming raw outputs into probabilities
whose mean matches the empirical win frequency at every confidence level.

Two estimators are implemented:
    * Platt scaling — fits σ(a·x + b) via logistic regression on the raw
      scores. Parametric, low-variance, fast. Fails when the calibration
      distortion is non-sigmoidal (e.g. a staircase).
    * Isotonic regression — non-parametric piecewise-constant fit. Handles
      arbitrary monotone distortions but needs more data to be stable.

The "auto" method fits both, computes Expected Calibration Error (ECE) on
the fit set, and keeps the one with lower ECE — per ADR-008.

Per-race softmax (with optional temperature) renormalises the calibrated
per-row probabilities so each race sums to 1.

API:
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    p   = cal.predict_proba(raw)                       # per-row calibrated
    p   = cal.predict_softmax(raw, race_ids)           # per-race sum=1
    cal.save(path) / Calibrator.load(path)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.core.logging import get_logger

log = get_logger(__name__)


# ── Metric helpers ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    observed_rate: float
    count: int


def _validate_lengths(probs: np.ndarray, labels: np.ndarray) -> None:
    if len(probs) != len(labels):
        raise ValueError(
            f"probs and labels must have the same length; got {len(probs)} and {len(labels)}"
        )


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error: weighted mean |mean_pred − observed_rate|
    across equal-width bins in [0, 1]. Lower is better; 0 = perfect."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    _validate_lengths(probs, labels)
    if len(probs) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(probs)
    total_err = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        mean_pred = probs[mask].mean()
        observed = labels[mask].mean()
        weight = mask.sum() / n
        total_err += weight * abs(mean_pred - observed)
    return float(total_err)


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> list[ReliabilityBin]:
    """Return one `ReliabilityBin` per equal-width bin of [0, 1].
    Empty bins are included with `count=0` and NaN summary stats so the
    output length is always exactly `n_bins`."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    _validate_lengths(probs, labels)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)

    out: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(probs[mask].mean())
            observed = float(labels[mask].mean())
        else:
            mean_pred = float("nan")
            observed = float("nan")
        out.append(
            ReliabilityBin(
                bin_lower=float(edges[b]),
                bin_upper=float(edges[b + 1]),
                mean_predicted=mean_pred,
                observed_rate=observed,
                count=count,
            )
        )
    return out


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and label.
    0 = perfect, 0.25 = always-0.5 against 50/50 labels."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    _validate_lengths(probs, labels)
    return float(np.mean((probs - labels) ** 2))


# ── Calibrator ─────────────────────────────────────────────────────────────


@dataclass
class CalibratorConfig:
    """Calibrator hyperparameters."""
    method: str = "auto"
    """One of 'platt', 'isotonic', 'auto'. 'auto' selects via ECE."""
    n_bins_for_selection: int = 15
    """Bin count fed into the ECE that drives auto-selection."""
    softmax_temperature: float = 1.0
    """Temperature T applied to the logit form of calibrated probabilities
    before per-race softmax. T<1 sharpens, T>1 flattens."""
    seed: int = 23


@dataclass
class CalibratorMetrics:
    """ECE / Brier / log-loss recorded for each candidate method at fit time."""
    ece: float
    brier: float
    n_samples: int

    def asdict(self) -> dict:
        return {"ece": self.ece, "brier": self.brier, "n_samples": self.n_samples}


_VALID_METHODS = ("platt", "isotonic", "auto")


class Calibrator:
    """Wraps a Platt or isotonic estimator. `fit` accepts uncalibrated scores
    in [0, 1] (e.g. `SpeedFormModel.predict_proba` output) together with the
    binary win labels.

    The artifact persisted by `save` contains the sklearn estimator joblib
    pickle + a JSON sidecar with config / metrics / chosen method.
    """

    ARTIFACT_VERSION: str = "1"

    def __init__(self, config: Optional[CalibratorConfig] = None):
        self.config = config or CalibratorConfig()
        self._platt: Optional[LogisticRegression] = None
        self._isotonic: Optional[IsotonicRegression] = None
        self.chosen_method: Optional[str] = None
        self.metrics: Optional[dict[str, dict]] = None

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "Calibrator":
        if self.config.method not in _VALID_METHODS:
            raise ValueError(
                f"Calibrator.method must be one of {_VALID_METHODS}; "
                f"got {self.config.method!r}"
            )
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=int).ravel()
        _validate_lengths(scores, labels)

        method = self.config.method
        if method in ("platt", "auto"):
            self._platt = _fit_platt(scores, labels, seed=self.config.seed)
        if method in ("isotonic", "auto"):
            self._isotonic = _fit_isotonic(scores, labels)

        # Compute metrics for whatever was fit.
        metrics: dict[str, dict] = {}
        if self._platt is not None:
            p = _predict_platt(self._platt, scores)
            metrics["platt"] = CalibratorMetrics(
                ece=expected_calibration_error(
                    p, labels, n_bins=self.config.n_bins_for_selection
                ),
                brier=brier_score(p, labels),
                n_samples=len(scores),
            ).asdict()
        if self._isotonic is not None:
            p = _predict_isotonic(self._isotonic, scores)
            metrics["isotonic"] = CalibratorMetrics(
                ece=expected_calibration_error(
                    p, labels, n_bins=self.config.n_bins_for_selection
                ),
                brier=brier_score(p, labels),
                n_samples=len(scores),
            ).asdict()

        if method == "auto":
            # Pick whichever achieves the lower ECE.
            self.chosen_method = min(metrics, key=lambda k: metrics[k]["ece"])
        else:
            self.chosen_method = method

        self.metrics = metrics
        log.info(
            "calibrator.fit_complete",
            method=method,
            chosen=self.chosen_method,
            metrics=metrics,
            n_samples=len(scores),
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────

    def _ensure_fitted(self) -> None:
        if self.chosen_method is None:
            raise RuntimeError("Calibrator not fitted.")

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Calibrated per-row probabilities in [0, 1]. Output is NOT
        per-race normalised — use `predict_softmax` for that."""
        self._ensure_fitted()
        scores = np.asarray(scores, dtype=float).ravel()
        if self.chosen_method == "platt":
            return _predict_platt(self._platt, scores)  # type: ignore[arg-type]
        return _predict_isotonic(self._isotonic, scores)  # type: ignore[arg-type]

    def predict_softmax(
        self, scores: np.ndarray, race_ids: np.ndarray
    ) -> np.ndarray:
        """Per-race softmax over the LOGIT of the calibrated probability
        (divided by T). Output is non-negative and sums to 1 within each
        race group. T < 1 sharpens the distribution; T > 1 flattens it."""
        p = self.predict_proba(scores)
        # Clip away exact 0 / 1 so the logit is finite.
        p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
        logits = np.log(p_clipped) - np.log(1 - p_clipped)
        logits = logits / float(self.config.softmax_temperature)

        race_ids = np.asarray(race_ids).ravel()
        if len(race_ids) != len(scores):
            raise ValueError(
                f"race_ids length {len(race_ids)} != scores length {len(scores)}"
            )

        out = np.empty_like(logits)
        # Group-wise stable softmax.
        for rid in np.unique(race_ids):
            mask = race_ids == rid
            ls = logits[mask]
            ls = ls - ls.max()
            e = np.exp(ls)
            out[mask] = e / e.sum()
        return out

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> dict:
        self._ensure_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._platt is not None:
            joblib.dump(self._platt, path / "platt.joblib")
        if self._isotonic is not None:
            joblib.dump(self._isotonic, path / "isotonic.joblib")

        meta = {
            "artifact_version": self.ARTIFACT_VERSION,
            "config": asdict(self.config),
            "chosen_method": self.chosen_method,
            "metrics": self.metrics or {},
        }
        with open(path / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        log.info("calibrator.saved", path=str(path), chosen=self.chosen_method)
        return meta

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        path = Path(path)
        with open(path / "metadata.json") as fh:
            meta = json.load(fh)
        cfg = CalibratorConfig(**meta["config"])
        obj = cls(config=cfg)
        platt_path = path / "platt.joblib"
        iso_path = path / "isotonic.joblib"
        if platt_path.exists():
            obj._platt = joblib.load(platt_path)
        if iso_path.exists():
            obj._isotonic = joblib.load(iso_path)
        obj.chosen_method = meta["chosen_method"]
        obj.metrics = meta.get("metrics") or None
        return obj


# ── Private fit / predict helpers ─────────────────────────────────────────


def _fit_platt(
    scores: np.ndarray, labels: np.ndarray, seed: int = 23
) -> LogisticRegression:
    """Platt scaling: logistic regression on the raw score, one feature."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", random_state=seed)
    lr.fit(scores.reshape(-1, 1), labels)
    return lr


def _predict_platt(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(scores.reshape(-1, 1))[:, 1]


def _fit_isotonic(
    scores: np.ndarray, labels: np.ndarray
) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels.astype(float))
    return iso


def _predict_isotonic(model: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict(scores)


__all__ = [
    "Calibrator",
    "CalibratorConfig",
    "CalibratorMetrics",
    "ReliabilityBin",
    "brier_score",
    "expected_calibration_error",
    "reliability_bins",
]
