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

The "auto" method fits both on an inner-train slice (default 80% of the
fit data, seeded random shuffle), evaluates ECE on the held-out inner-val
slice (default 20%), and keeps the one with lower inner-val ECE —
isotonic must beat Platt by `auto_min_delta_ece` (default 0.001) to be
chosen, giving a one-sided protective bias toward the simpler model.
Both calibrators are then re-fit on the FULL fit data so no data is
wasted. See ADR-037 — this corrects the fit-slice criterion in ADR-030
that overfit to isotonic memorisation.

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
    """One of 'platt', 'isotonic', 'auto'. 'auto' selects via held-out
    inner-val ECE (NOT fit-slice ECE — see ADR-037)."""
    n_bins_for_selection: int = 15
    """Bin count fed into the ECE that drives auto-selection."""
    softmax_temperature: float = 1.0
    """Temperature T applied to the logit form of calibrated probabilities
    before per-race softmax. T<1 sharpens, T>1 flattens."""
    seed: int = 23
    auto_val_fraction: float = 0.2
    """Fraction of the fit slice held out as an inner-val set for auto-
    selection. Random seeded split. Used only when `method='auto'` AND
    the fit slice has at least `auto_min_inner_val_size` would-be val
    rows; otherwise we fall back to fit-slice ECE with a warning."""
    auto_min_inner_val_size: int = 100
    """If the inner-val split would have fewer rows than this, fall back
    to fit-slice ECE selection (with a warning). Inner-val ECE on a tiny
    slice is too noisy to rely on."""
    auto_min_delta_ece: float = 0.001
    """Minimum inner-val ECE advantage isotonic must show over Platt to
    be chosen by `method='auto'`. Default 0.001 (= 0.1% ECE) introduces
    a small protective bias toward the simpler model when the two are
    statistically tied. Set to 0 to choose strictly by lower ECE."""


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
        self.inner_val_metrics: Optional[dict[str, dict]] = None
        self.auto_selection_mode: Optional[str] = None  # "held_out" | "fit_slice_fallback" | None

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
        if method == "auto":
            chosen, inner_val_metrics, sel_mode = self._auto_select(scores, labels)
            self.chosen_method = chosen
            self.inner_val_metrics = inner_val_metrics
            self.auto_selection_mode = sel_mode
        else:
            self.chosen_method = method
            self.inner_val_metrics = None
            self.auto_selection_mode = None

        # Always (re-)fit on the FULL fit slice so no data is wasted, AND
        # so the metrics dict can report fit-slice ECE/Brier for both
        # candidate methods (useful for diagnostics / comparison with the
        # inner-val choice).
        if method in ("platt", "auto"):
            self._platt = _fit_platt(scores, labels, seed=self.config.seed)
        if method in ("isotonic", "auto"):
            self._isotonic = _fit_isotonic(scores, labels)

        # Compute fit-slice metrics for whatever was fit.
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

        self.metrics = metrics
        log.info(
            "calibrator.fit_complete",
            method=method,
            chosen=self.chosen_method,
            selection_mode=self.auto_selection_mode,
            metrics=metrics,
            inner_val_metrics=self.inner_val_metrics,
            n_samples=len(scores),
        )
        return self

    def _auto_select(
        self, scores: np.ndarray, labels: np.ndarray,
    ) -> tuple[str, dict, str]:
        """Pick platt vs isotonic by held-out inner-val ECE.

        Returns (chosen_method, inner_val_metrics_dict, selection_mode).
        selection_mode is "held_out" when an inner split was used,
        "fit_slice_fallback" when the fit slice was too small.
        """
        n = len(scores)
        n_val = int(n * self.config.auto_val_fraction)

        # Fall back to fit-slice ECE for very small calib slices — inner-val
        # ECE on < min_inner_val_size rows is too noisy to trust.
        if n_val < self.config.auto_min_inner_val_size:
            log.warning(
                "calibrator.auto_using_fit_slice_fallback",
                n=n, n_val=n_val,
                min_inner_val_size=self.config.auto_min_inner_val_size,
            )
            platt = _fit_platt(scores, labels, seed=self.config.seed)
            iso = _fit_isotonic(scores, labels)
            platt_ece = expected_calibration_error(
                _predict_platt(platt, scores), labels,
                n_bins=self.config.n_bins_for_selection,
            )
            iso_ece = expected_calibration_error(
                _predict_isotonic(iso, scores), labels,
                n_bins=self.config.n_bins_for_selection,
            )
            chosen = (
                "isotonic" if iso_ece + self.config.auto_min_delta_ece < platt_ece
                else "platt"
            )
            return (
                chosen,
                {
                    "platt": {"ece": float(platt_ece), "n_val": int(n)},
                    "isotonic": {"ece": float(iso_ece), "n_val": int(n)},
                },
                "fit_slice_fallback",
            )

        # Random seeded split inside the fit slice. Time-isolation is
        # enforced at the OUTER split (the calib slice itself is a
        # contiguous time window); inside the slice a random split is
        # appropriate for choosing between two 1-D calibration maps.
        rng = np.random.default_rng(self.config.seed)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        s_tr, l_tr = scores[train_idx], labels[train_idx]
        s_val, l_val = scores[val_idx], labels[val_idx]

        platt_inner = _fit_platt(s_tr, l_tr, seed=self.config.seed)
        iso_inner = _fit_isotonic(s_tr, l_tr)

        platt_ece = expected_calibration_error(
            _predict_platt(platt_inner, s_val), l_val,
            n_bins=self.config.n_bins_for_selection,
        )
        iso_ece = expected_calibration_error(
            _predict_isotonic(iso_inner, s_val), l_val,
            n_bins=self.config.n_bins_for_selection,
        )
        # Isotonic wins only if it beats Platt by min_delta_ece on inner-val.
        # Default min_delta=0.001 protects against noise-driven swaps.
        chosen = (
            "isotonic" if iso_ece + self.config.auto_min_delta_ece < platt_ece
            else "platt"
        )
        return (
            chosen,
            {
                "platt": {"ece": float(platt_ece), "n_val": int(n_val)},
                "isotonic": {"ece": float(iso_ece), "n_val": int(n_val)},
            },
            "held_out",
        )

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
            "inner_val_metrics": self.inner_val_metrics or {},
            "auto_selection_mode": self.auto_selection_mode,
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
        obj.inner_val_metrics = meta.get("inner_val_metrics") or None
        obj.auto_selection_mode = meta.get("auto_selection_mode")
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
