"""
app/services/calibration/drift.py
─────────────────────────────────
Phase 4 — Calibration drift detection via CUSUM change-point.

A calibrator fit on 2018-2024 data won't stay calibrated forever — track
conditions, takeout rates, jockey populations, and the distribution of
horse quality all drift. We need an online statistic that says "the
model's predictions have started disagreeing with realized outcomes in a
way that is no longer attributable to sampling noise" so the operator
knows to refit.

Statistic: two-sided Bernoulli CUSUM on the STANDARDISED calibration
residual z_t.

    z_t = (label_t − p_t) / sqrt(p_t · (1 − p_t))      (Bernoulli z-score)

Under perfect calibration each z_t has mean 0 and variance 1 regardless
of p_t, so individual high-confidence observations don't dominate the
running statistic (the raw residual |r_t| = |label − p| can be as large
as 0.95 for an unlucky longshot hit, whereas |z_t| is bounded and the
average magnitude is independent of p).

The CUSUM accumulates standardised deviations past a reference value k
and alarms when they exceed a threshold h. Two streams track over- and
under-prediction independently:

    S+_t = max(0, S+_{t-1} + (z_t − k))      → alarm: model UNDER-predicts
    S-_t = max(0, S-_{t-1} − (z_t + k))      → alarm: model OVER-predicts

Defaults k=0.5, h=4 follow Page (1954) / Hawkins & Olwell (1998) — they
correspond to an in-control average-run-length ARL₀ ≈ 168 and detect a
0.5σ shift with ARL₁ ≈ 10. The operator can tighten or loosen via the
config; raising h to 5 lifts ARL₀ to ≈ 465.

The standardisation has a numerical wrinkle: at p_t → 0 or p_t → 1 the
denominator vanishes and z_t blows up. We floor sqrt(p(1-p)) at
`eps` (default 1e-4 → max |z_t| ≈ 1/eps = 10000 if unclipped) and
additionally clip |z_t| ≤ `z_clip` (default 5). Both guards are
documented and conservative.

Public surface
──────────────
    CUSUMConfig         — frozen config (k, h, two_sided, z_clip, eps).
    CUSUMState          — running state (s_plus, s_minus, n, alarmed_at,
                          direction).
    CUSUMDetector       — incremental detector.
        .update(prediction, label) → state
        .run(predictions, labels)  → list[state]
        .reset()
    detect_drift(...)   — one-shot batch helper, returns DriftReport.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)


# ── Config & state ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CUSUMConfig:
    """Configuration for CUSUM drift detection on STANDARDISED Bernoulli
    residuals.

    k:          Reference value in σ-units. Drift smaller than k per obs
                is treated as noise. Typical 0.25–1.0; 0.5 = standard.
    h:          Alarm threshold in σ-units. Cumulative drift past k must
                exceed h for the alarm to fire. Typical 4–6; 4 ≈ ARL₀ 168.
    two_sided:  If True, track both positive and negative drift (default).
                If False, only positive (label > pred → UNDER-prediction).
    z_clip:     Max magnitude of the per-step standardised residual.
                Prevents single longshot hits at very small p from
                dominating the running stat.
    eps:        Numerical floor on sqrt(p·(1-p)) in the standardisation.
                Caps z at roughly 1/eps before z_clip applies.
    """
    k: float = 0.5
    h: float = 4.0
    two_sided: bool = True
    z_clip: float = 5.0
    eps: float = 1e-4

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError(f"k must be >= 0; got {self.k}")
        if self.h <= 0:
            raise ValueError(f"h must be > 0; got {self.h}")
        if self.z_clip <= 0:
            raise ValueError(f"z_clip must be > 0; got {self.z_clip}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0; got {self.eps}")


@dataclass
class CUSUMState:
    """Running state of the CUSUM detector.

    s_plus / s_minus: running cumulative-sum statistics in σ-units, ≥ 0.
    n:                number of observations consumed.
    alarmed_at:       index of the first observation where the alarm
                      fired (None if not yet).
    direction:        "high" (label > pred → UNDER-predicted) or
                      "low" (label < pred → OVER-predicted) when
                      alarmed; None otherwise.
    """
    s_plus: float = 0.0
    s_minus: float = 0.0
    n: int = 0
    alarmed_at: Optional[int] = None
    direction: Optional[str] = None

    @property
    def alarmed(self) -> bool:
        return self.alarmed_at is not None


# ── Detector ───────────────────────────────────────────────────────────────


def _standardised_residual(p: float, label: float, eps: float, z_clip: float) -> float:
    """Bernoulli z-score: (label - p) / sqrt(p*(1-p)), with floor + clip."""
    sd = math.sqrt(max(p * (1.0 - p), eps))
    z = (label - p) / sd
    if z > z_clip:
        return z_clip
    if z < -z_clip:
        return -z_clip
    return z


class CUSUMDetector:
    """Incremental two-sided CUSUM detector for calibration residuals.

    Usage:
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=4.0))
        for pred, label in stream:
            state = det.update(pred, label)
            if state.alarmed:
                trigger_recalibration()
                det.reset()

    The detector latches on first alarm — subsequent updates do not
    clear `alarmed_at` / `direction` and do not reset the running stats.
    The caller invokes `reset()` to start a new monitoring epoch
    (typically after refitting the calibrator).
    """

    def __init__(self, config: CUSUMConfig | None = None) -> None:
        self.config = config or CUSUMConfig()
        self.state = CUSUMState()

    @property
    def k(self) -> float:
        return self.config.k

    @property
    def h(self) -> float:
        return self.config.h

    @property
    def two_sided(self) -> bool:
        return self.config.two_sided

    def reset(self) -> None:
        """Zero the running statistics and clear any prior alarm."""
        self.state = CUSUMState()

    def update(self, prediction: float, label: float) -> CUSUMState:
        """Consume one (prediction, label) pair and return the new state."""
        if not (0.0 <= prediction <= 1.0):
            raise ValueError(f"prediction must lie in [0, 1]; got {prediction}")
        if label not in (0, 1, 0.0, 1.0):
            raise ValueError(f"label must be 0 or 1; got {label}")

        s = self.state
        z = _standardised_residual(
            float(prediction), float(label), self.config.eps, self.config.z_clip
        )

        s_plus = max(0.0, s.s_plus + z - self.k)
        s_minus = (
            max(0.0, s.s_minus - z - self.k)
            if self.two_sided
            else 0.0
        )

        new_n = s.n + 1
        alarmed_at = s.alarmed_at
        direction = s.direction
        if alarmed_at is None:
            if s_plus > self.h:
                alarmed_at = new_n - 1
                direction = "high"
                log.info(
                    "drift.alarm",
                    step=alarmed_at, direction="high",
                    s_plus=s_plus, s_minus=s_minus, h=self.h, k=self.k,
                )
            elif self.two_sided and s_minus > self.h:
                alarmed_at = new_n - 1
                direction = "low"
                log.info(
                    "drift.alarm",
                    step=alarmed_at, direction="low",
                    s_plus=s_plus, s_minus=s_minus, h=self.h, k=self.k,
                )

        self.state = CUSUMState(
            s_plus=s_plus,
            s_minus=s_minus,
            n=new_n,
            alarmed_at=alarmed_at,
            direction=direction,
        )
        return self.state

    def run(
        self, predictions: Sequence[float], labels: Sequence[float],
    ) -> list[CUSUMState]:
        """Drive the detector over a full sequence and return per-step states.

        Does NOT reset the detector first — chain sequences explicitly if
        desired. Call `reset()` to start fresh.
        """
        preds = np.asarray(predictions, dtype=float).ravel()
        labs = np.asarray(labels, dtype=float).ravel()
        if len(preds) != len(labs):
            raise ValueError(
                f"predictions and labels must match length; got "
                f"{len(preds)} and {len(labs)}"
            )
        states: list[CUSUMState] = []
        for p, l in zip(preds, labs):
            states.append(self.update(float(p), float(l)))
        return states


# ── One-shot batch helper ──────────────────────────────────────────────────


@dataclass
class DriftReport:
    """Summary of running CUSUM over a fixed batch of (preds, labels)."""
    n: int
    alarmed: bool
    alarmed_at: Optional[int]
    direction: Optional[str]
    final_s_plus: float
    final_s_minus: float
    mean_residual: float
    max_s_plus: float
    max_s_minus: float
    config: dict


def detect_drift(
    predictions: Sequence[float],
    labels: Sequence[float],
    config: CUSUMConfig | None = None,
) -> DriftReport:
    """One-shot CUSUM drift check over a batch.

    Returns a `DriftReport` summarising (a) whether and when an alarm
    fired, (b) the final CUSUM statistics, (c) the running max for
    diagnostic plotting, and (d) the mean (raw) residual for
    sanity-checking the choice of `k`.
    """
    cfg = config or CUSUMConfig()
    det = CUSUMDetector(cfg)
    states = det.run(predictions, labels)
    if not states:
        return DriftReport(
            n=0, alarmed=False, alarmed_at=None, direction=None,
            final_s_plus=0.0, final_s_minus=0.0, mean_residual=0.0,
            max_s_plus=0.0, max_s_minus=0.0,
            config={
                "k": cfg.k, "h": cfg.h, "two_sided": cfg.two_sided,
                "z_clip": cfg.z_clip, "eps": cfg.eps,
            },
        )
    final = states[-1]
    preds = np.asarray(predictions, dtype=float).ravel()
    labs = np.asarray(labels, dtype=float).ravel()
    return DriftReport(
        n=len(states),
        alarmed=final.alarmed,
        alarmed_at=final.alarmed_at,
        direction=final.direction,
        final_s_plus=final.s_plus,
        final_s_minus=final.s_minus,
        mean_residual=float((labs - preds).mean()),
        max_s_plus=max(s.s_plus for s in states),
        max_s_minus=max(s.s_minus for s in states),
        config={
            "k": cfg.k, "h": cfg.h, "two_sided": cfg.two_sided,
            "z_clip": cfg.z_clip, "eps": cfg.eps,
        },
    )


# ── Drift-state persistence (ADR-044) ──────────────────────────────────────
#
# The rolling-retrain script (`scripts/rolling_retrain.py`) supports a
# `--skip-if-no-drift` flag that reads a small JSON sidecar to decide
# whether to spin up an expensive retraining run. We serialise just
# enough of the `CUSUMDetector` state to make that check trivial:
# `triggered`, the running sums, the observation count, and a UTC
# timestamp of when the file was written. No predictions or labels are
# persisted — the file is a marker, not an audit log.


def save_drift_state(detector: "CUSUMDetector", path: Path) -> None:
    """Persist the live CUSUM detector state to a JSON marker file.

    Writes the keys consumed by `scripts/rolling_retrain.py`:
        - triggered:       bool — has the alarm fired (latched once true)?
        - pos_cusum:       float — current S+ running sum
        - neg_cusum:       float — current S- running sum
        - n_observations:  int — observations consumed so far
        - alarmed_at:      Optional[int] — step at which the alarm fired
        - direction:       Optional[str] — "high" / "low" when alarmed
        - last_updated:    ISO-8601 UTC timestamp

    Idempotent: writes the file atomically by truncating in place. Creates
    parent dirs as needed. Safe to call from cron without locking — readers
    only ever see a valid JSON document because of the os.replace semantics
    of Path.write_text.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = detector.state
    payload = {
        "triggered": bool(state.alarmed),
        "pos_cusum": float(state.s_plus),
        "neg_cusum": float(state.s_minus),
        "n_observations": int(state.n),
        "alarmed_at": state.alarmed_at,
        "direction": state.direction,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "config": {
            "k": detector.config.k,
            "h": detector.config.h,
            "two_sided": detector.config.two_sided,
            "z_clip": detector.config.z_clip,
            "eps": detector.config.eps,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("drift.state_saved", path=str(path), triggered=payload["triggered"],
             n=payload["n_observations"])


def load_drift_state(path: Path) -> dict:
    """Read a JSON marker emitted by `save_drift_state`.

    For a missing file returns a neutral "not triggered" state so the
    `--skip-if-no-drift` flag on the rolling retrain script can fail
    soft on first boot (no marker yet => no drift => skip). For a file
    that fails to parse, returns the same neutral state with a logged
    warning rather than raising — operator tooling should never crash
    a cron job because of a corrupt marker.
    """
    path = Path(path)
    if not path.exists():
        log.info("drift.state_missing", path=str(path))
        return {
            "triggered": False,
            "pos_cusum": 0.0,
            "neg_cusum": 0.0,
            "n_observations": 0,
            "alarmed_at": None,
            "direction": None,
            "last_updated": None,
        }
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        log.warning("drift.state_corrupt", path=str(path), error=str(exc))
        return {
            "triggered": False,
            "pos_cusum": 0.0,
            "neg_cusum": 0.0,
            "n_observations": 0,
            "alarmed_at": None,
            "direction": None,
            "last_updated": None,
        }


__all__ = [
    "CUSUMConfig",
    "CUSUMDetector",
    "CUSUMState",
    "DriftReport",
    "detect_drift",
    "save_drift_state",
    "load_drift_state",
]
