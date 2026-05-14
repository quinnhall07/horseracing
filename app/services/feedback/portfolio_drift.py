"""
app/services/feedback/portfolio_drift.py
────────────────────────────────────────
Phase 5b / Layer 7 — portfolio-level realised-vs-expected PnL CUSUM.

Companion to `app/services/calibration/drift.py`. Where that detector watches
the standardised Bernoulli residual of (label − pred) on a single-event
calibration stream, this one watches the standardised portfolio residual of
(realised_pnl − expected_pnl) over a stream of settled bets.

Statistic (per settled bet i):
    σ_i  = stake_i · sqrt(p_i · (1 − p_i)) · decimal_odds_at_settlement_i
                                              # Bernoulli-derived stdev of
                                              # pnl_i for a win-or-lose bet
    z_i  = (pnl_i − E[pnl_i]) / σ_i

Under perfect calibration AND correct model probabilities, z_i has mean 0
and variance 1 — identical guarantees to the calibration-drift z-score, so
we reuse the same CUSUM defaults (`k=0.5`, `h=4`).

Two-sided rationale:
    s_plus_t  = max(0, s_plus_{t-1}  + (z_t − k))   → realised > expected
                                                      (model UNDER-confident
                                                       OR upside variance burst)
    s_minus_t = max(0, s_minus_{t-1} − (z_t + k))   → realised < expected
                                                      (model OVER-confident,
                                                       paper-trading bleeding)

Both directions are operationally meaningful: persistent over-performance
suggests we are systematically leaving money on the table by under-betting,
and persistent under-performance is the well-known calibration-drift failure
mode that we MUST detect before we hemorrhage bankroll.

Numerical guards (mirroring calibration drift):
    eps    — floor on σ_i to avoid blow-up when p_i ≈ 0 or 1, OR stake_i = 0.
    z_clip — magnitude clip on z_i so a single longshot does not dominate.

The detector latches on first alarm; the caller calls `reset()` to begin a
new monitoring epoch (after retraining, recalibrating, or shifting the
betting policy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)


# ── z-score helper ─────────────────────────────────────────────────────────


def portfolio_pnl_zscore(
    expected_returns: np.ndarray,
    realised_pnl: np.ndarray,
    stakes: np.ndarray,
    *,
    model_probs: np.ndarray | None = None,
    decimal_odds: np.ndarray | None = None,
    eps: float = 1e-6,
    z_clip: float = 5.0,
) -> np.ndarray:
    """Standardised per-bet residual: (pnl − E[pnl]) / σ_estimate.

    Two parameterisations are supported for the σ_estimate:

    1.  If `model_probs` AND `decimal_odds` are supplied (preferred — the
        analytic Bernoulli stdev): σ_i = stake_i · √(p_i (1−p_i)) · odds_i.
        This is the closed-form stdev of (pnl_i | bet placed) under the
        single-event Bernoulli (won/lost) with E[pnl_i] = stake_i ·
        (p_i · odds_i − 1) = stake_i · edge_i.

    2.  If those are not supplied, fall back to `σ_i = |stakes_i|`, which
        is a conservative O(stake)-scaled stdev (correct order of magnitude
        for E[|pnl_i|] under any reasonable p_i). This matches the
        docstring spec ("σ_estimate using stakes as a variance proxy")
        while still keeping the closed-form Bernoulli path available for
        tests + production.

    All arrays must be 1-D and the same length. Returns a 1-D array of
    z-scores, clipped to ±z_clip.
    """
    er = np.asarray(expected_returns, dtype=float).ravel()
    pnl = np.asarray(realised_pnl, dtype=float).ravel()
    st = np.asarray(stakes, dtype=float).ravel()
    if not (len(er) == len(pnl) == len(st)):
        raise ValueError(
            f"expected_returns/realised_pnl/stakes lengths must match; "
            f"got {len(er)}/{len(pnl)}/{len(st)}"
        )

    if model_probs is not None and decimal_odds is not None:
        p = np.asarray(model_probs, dtype=float).ravel()
        o = np.asarray(decimal_odds, dtype=float).ravel()
        if not (len(p) == len(o) == len(er)):
            raise ValueError(
                "model_probs / decimal_odds must match length of stakes"
            )
        var_factor = np.clip(p * (1.0 - p), eps, None)
        sigma = np.abs(st) * np.sqrt(var_factor) * o
    else:
        sigma = np.abs(st)

    sigma = np.maximum(sigma, eps)
    z = (pnl - er) / sigma
    z = np.clip(z, -z_clip, z_clip)
    return z


# ── State + detector ───────────────────────────────────────────────────────


@dataclass
class PortfolioDriftState:
    """Running CUSUM state for the portfolio drift detector."""
    pos_cusum: float = 0.0
    neg_cusum: float = 0.0
    n_observations: int = 0
    triggered: bool = False
    triggered_at: Optional[int] = None
    direction: Optional[str] = None  # "high" → over-performance, "low" → under.


@dataclass
class PortfolioDriftDetector:
    """Two-sided CUSUM detector on standardised portfolio PnL residuals.

    Defaults (k=0.5, h=4) match the calibration-drift detector by design
    (ADR-036, ADR-043): a single set of σ-unit thresholds for every drift
    statistic in the system keeps the operator's mental model simple and
    the ARL₀ comparable across modules.

    Usage (incremental):
        det = PortfolioDriftDetector()
        for z in zscores:
            if det.update(z):
                trigger_review_or_retrain()
                det.reset()

    Usage (batch via run(...)):
        det.run(zscores) → list[bool] per-step triggered flags.
    """

    k: float = 0.5
    h: float = 4.0
    two_sided: bool = True
    pos_cusum: float = 0.0
    neg_cusum: float = 0.0
    triggered: bool = False
    triggered_at: Optional[int] = None
    direction: Optional[str] = None
    n_observations: int = 0

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError(f"k must be >= 0; got {self.k}")
        if self.h <= 0:
            raise ValueError(f"h must be > 0; got {self.h}")

    # ─── core API ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Zero the running statistics and clear any prior alarm."""
        self.pos_cusum = 0.0
        self.neg_cusum = 0.0
        self.triggered = False
        self.triggered_at = None
        self.direction = None
        self.n_observations = 0

    def update(self, z: float) -> bool:
        """Append a z-score, update both-direction CUSUMs.

        Returns True iff THIS update was the one that crossed the threshold
        (so the caller can act once per epoch). Once triggered, subsequent
        updates return False (latched) until `reset()` is called.
        """
        if not math.isfinite(z):
            raise ValueError(f"z must be finite; got {z}")

        already_triggered = self.triggered
        self.pos_cusum = max(0.0, self.pos_cusum + (z - self.k))
        if self.two_sided:
            self.neg_cusum = max(0.0, self.neg_cusum - (z + self.k))
        else:
            self.neg_cusum = 0.0
        self.n_observations += 1

        if already_triggered:
            return False

        if self.pos_cusum > self.h:
            self.triggered = True
            self.triggered_at = self.n_observations - 1
            self.direction = "high"
            log.info(
                "portfolio_drift.alarm",
                step=self.triggered_at, direction="high",
                pos_cusum=self.pos_cusum, neg_cusum=self.neg_cusum,
                k=self.k, h=self.h,
            )
            return True

        if self.two_sided and self.neg_cusum > self.h:
            self.triggered = True
            self.triggered_at = self.n_observations - 1
            self.direction = "low"
            log.info(
                "portfolio_drift.alarm",
                step=self.triggered_at, direction="low",
                pos_cusum=self.pos_cusum, neg_cusum=self.neg_cusum,
                k=self.k, h=self.h,
            )
            return True

        return False

    def run(self, z_scores: np.ndarray | list[float]) -> list[bool]:
        """Drive the detector over a sequence; return per-step trigger flags.

        Does NOT reset first — caller chains sequences explicitly.
        """
        arr = np.asarray(z_scores, dtype=float).ravel()
        flags: list[bool] = []
        for z in arr:
            flags.append(self.update(float(z)))
        return flags

    @property
    def state(self) -> PortfolioDriftState:
        """Snapshot of the running state as a dataclass (immutable view)."""
        return PortfolioDriftState(
            pos_cusum=self.pos_cusum,
            neg_cusum=self.neg_cusum,
            n_observations=self.n_observations,
            triggered=self.triggered,
            triggered_at=self.triggered_at,
            direction=self.direction,
        )


__all__ = [
    "PortfolioDriftDetector",
    "PortfolioDriftState",
    "portfolio_pnl_zscore",
]
