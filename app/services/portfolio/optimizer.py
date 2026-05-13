"""
app/services/portfolio/optimizer.py
───────────────────────────────────
Phase 5b — CVaR-constrained portfolio optimiser (Layer 6).

Allocates capital across a list of `BetCandidate` objects on a single race
(per ADR-039 + ADR-041, cross-race optimisation is deferred). The objective
is expected return; the binding risk constraint is CVaR_α — Conditional
Value at Risk at tail probability α (default 0.05, i.e. 95th-percentile
shortfall) — capped at `max_drawdown_pct` of bankroll.

Math: linear-programming formulation of CVaR per Rockafellar & Uryasev
(2000) "Optimization of Conditional Value-at-Risk." For a vector of stake
fractions `f`, an auxiliary VaR proxy `v`, and shortfall slack variables
`u_s ≥ 0` over Monte Carlo scenarios s = 1..S, the LP is:

    minimise   −Σᵢ fᵢ · EVᵢ                          (negate to maximise EV)
    subject to
               −Σᵢ payoffᵢₛ · fᵢ − v − uₛ ≤ 0    for each s   (S rows)
               v + (1/(α·S)) · Σₛ uₛ ≤ max_drawdown_pct        (1 row)
               Σᵢ fᵢ ≤ 1                                       (1 row, budget)
               0 ≤ fᵢ ≤ min(¼-Kelly_cap_i, max_bet_fraction)
               v free, uₛ ≥ 0

The decision vector x has length n + 1 + S, packed as
    x = [f₀, f₁, …, f_{n-1}, v, u₀, u₁, …, u_{S-1}].

Scenario generation: vectorised Plackett-Luce sampling via the Gumbel
trick. For each scenario we draw a single finishing order from PL with
strengths = race_win_probs; the per-candidate payoff multiplier for a
$1 stake is +(decimal_odds − 1) on a hit and −1 on a miss.

Public API:
    optimize_portfolio(candidates, race_win_probs, bankroll, *,
                       cvar_alpha=0.05, max_drawdown_pct=0.20,
                       n_scenarios=1000, seed=42,
                       max_bet_fraction=0.03, card_id="unknown") -> Portfolio
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linprog

from app.core.logging import get_logger
from app.schemas.bets import BetCandidate, BetRecommendation, Portfolio
from app.schemas.race import BetType
from app.services.ordering.plackett_luce import sample_orderings_vectorised
from app.services.portfolio.sizing import (
    DEFAULT_MAX_BET_FRACTION,
    apply_bet_cap,
    kelly_fraction,
)

log = get_logger(__name__)


_F_TOL: float = 1e-9
"""Stake-fraction tolerance: numerical noise below this is dropped from
the recommendation list. linprog rarely returns exact zeros for inactive
variables, so a small threshold avoids cluttering the portfolio with
fractions that round-trip to a $0 stake."""


def _build_payoff_matrix(
    candidates: list[BetCandidate],
    orderings: np.ndarray,
) -> np.ndarray:
    """Construct an (n_candidates, n_scenarios) payoff-per-$1 matrix.

    The inner loop over candidates is a python loop, but the inner
    comparisons against orderings are vectorised across scenarios — for
    most practical card sizes (n_candidates ≤ ~50, n_scenarios ≤ 5000)
    this is fast enough that the LP solve dominates.
    """
    n_candidates = len(candidates)
    n_scenarios = orderings.shape[0]
    payoff = np.empty((n_candidates, n_scenarios), dtype=float)
    for i, c in enumerate(candidates):
        bt = c.bet_type
        sel = c.selection
        gain = c.decimal_odds - 1.0
        if bt == BetType.WIN:
            hits = orderings[:, 0] == sel[0]
        elif bt == BetType.EXACTA:
            hits = (orderings[:, 0] == sel[0]) & (orderings[:, 1] == sel[1])
        elif bt == BetType.TRIFECTA:
            hits = (
                (orderings[:, 0] == sel[0])
                & (orderings[:, 1] == sel[1])
                & (orderings[:, 2] == sel[2])
            )
        elif bt == BetType.SUPERFECTA:
            hits = (
                (orderings[:, 0] == sel[0])
                & (orderings[:, 1] == sel[1])
                & (orderings[:, 2] == sel[2])
                & (orderings[:, 3] == sel[3])
            )
        else:
            raise ValueError(f"unsupported bet_type for optimizer: {bt}")
        payoff[i, :] = np.where(hits, gain, -1.0)
    return payoff


def _empty_portfolio(card_id: str, bankroll: float) -> Portfolio:
    return Portfolio(
        card_id=card_id,
        bankroll=bankroll,
        recommendations=[],
        expected_return=0.0,
        var_95=0.0,
        cvar_95=0.0,
        total_stake_fraction=0.0,
    )


def _validate_inputs(
    candidates: list[BetCandidate],
    race_win_probs: dict[str, np.ndarray],
) -> None:
    """Raise on inputs the LP cannot handle. Caller is expected to have
    a coherent (race_id → win prob vector) mapping for every race that
    appears in `candidates`."""
    for c in candidates:
        if c.race_id not in race_win_probs:
            raise ValueError(
                f"candidate refers to race_id={c.race_id!r} but "
                f"race_win_probs has no entry for it"
            )
        n_horses = len(race_win_probs[c.race_id])
        for idx in c.selection:
            if not (0 <= idx < n_horses):
                raise ValueError(
                    f"selection index {idx} out of range for race "
                    f"{c.race_id!r} (n_horses={n_horses})"
                )


def _per_candidate_upper_bound(
    candidate: BetCandidate,
    max_bet_fraction: float,
) -> float:
    """Per-candidate stake-fraction upper bound: min(¼-Kelly cap on edge,
    global max_bet_fraction)."""
    kelly = kelly_fraction(candidate.edge, candidate.decimal_odds)
    return min(apply_bet_cap(kelly, cap=max_bet_fraction), max_bet_fraction)


def optimize_portfolio(
    candidates: list[BetCandidate],
    race_win_probs: dict[str, np.ndarray],
    bankroll: float,
    *,
    cvar_alpha: float = 0.05,
    max_drawdown_pct: float = 0.20,
    n_scenarios: int = 1000,
    seed: int = 42,
    max_bet_fraction: float = DEFAULT_MAX_BET_FRACTION,
    card_id: str = "unknown",
) -> Portfolio:
    """CVaR-constrained allocation across a set of bet candidates.

    Args:
        candidates: list of `BetCandidate`. May span multiple races IF the
            caller has chosen to group cross-race; current Phase 5b usage
            (per-race) gives the cleanest interpretation of correlation.
        race_win_probs: race_id → calibrated win-probability vector.
            Used to draw PL scenarios for the Monte Carlo CVaR estimate.
            Strengths can be the raw probability vector (PL is invariant
            to scale).
        bankroll: dollar bankroll. All Portfolio.* metrics are in USD;
            stake_fraction is the unit-less fraction of `bankroll`.
        cvar_alpha: tail probability. 0.05 ⇒ 95th-percentile shortfall.
        max_drawdown_pct: upper bound on CVaR_α as a fraction of bankroll.
        n_scenarios: number of PL Monte Carlo scenarios per race.
        seed: PRNG seed for the Gumbel-trick sampler. Same seed + same
            candidates + same probs ⇒ identical Portfolio across calls.
        max_bet_fraction: hard cap on any single stake fraction
            (ADR-002). Default 0.03.
        card_id: forwarded to the returned Portfolio.

    Returns:
        Portfolio. Empty `recommendations` if (a) `candidates` is empty,
        (b) the LP is infeasible, or (c) the LP solver fails. The
        latter two are logged as warnings.

    Raises:
        ValueError: race_win_probs missing a referenced race_id, or a
            candidate's selection index is out of range for its race.
    """
    if not candidates:
        return _empty_portfolio(card_id, bankroll)
    _validate_inputs(candidates, race_win_probs)

    n = len(candidates)
    S = int(n_scenarios)
    alpha = float(cvar_alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"cvar_alpha must be in (0, 1); got {alpha}")
    if S <= 0:
        raise ValueError(f"n_scenarios must be positive; got {n_scenarios}")

    # ── Monte Carlo PL scenarios ───────────────────────────────────────
    # Build the (n_candidates, n_scenarios) payoff-per-$1 matrix. When
    # candidates span multiple races, each race needs its own ordering
    # draw because the per-race PL distributions are independent. We
    # build one (S, n_horses_r) ordering matrix per unique race using a
    # PER-RACE seeded RNG so that the same (race_id, seed) → same draws,
    # then dispatch per-candidate into _build_payoff_matrix.
    unique_races = sorted({c.race_id for c in candidates})
    race_orderings: dict[str, np.ndarray] = {}
    for r_idx, race_id in enumerate(unique_races):
        # Per-race seeding makes results invariant to the order of unique
        # races in `candidates` and stable when callers add/remove races.
        race_rng = np.random.default_rng(np.array([seed, hash(race_id) & 0xFFFFFFFF]))
        race_orderings[race_id] = sample_orderings_vectorised(
            strengths=np.asarray(race_win_probs[race_id], dtype=float),
            n_samples=S,
            rng=race_rng,
        )

    # Build the full payoff matrix candidate-by-candidate.
    payoff_matrix = np.empty((n, S), dtype=float)
    for i, c in enumerate(candidates):
        orderings = race_orderings[c.race_id]
        # Reuse the helper but on this candidate's own race orderings.
        payoff_matrix[i : i + 1, :] = _build_payoff_matrix([c], orderings)

    # ── EV per dollar (used in LP objective) ───────────────────────────
    # Use the candidate's own .expected_value, which was computed by the
    # EV engine. linprog minimises c·x; we want to MAXIMISE EV·f, so we
    # negate. Auxiliary variables v and u_s have zero EV coefficient.
    ev_per_dollar = np.array([c.expected_value for c in candidates], dtype=float)
    c_obj = np.concatenate([
        -ev_per_dollar,                        # f_i terms (negated to maximise)
        np.zeros(1, dtype=float),              # v term
        np.zeros(S, dtype=float),              # u_s terms
    ])

    # ── Inequality constraints (A_ub @ x ≤ b_ub) ───────────────────────
    # Row 0..S-1: scenario constraints  −payoffᵀ · f − v − u_s ≤ 0.
    A_scen = np.zeros((S, n + 1 + S), dtype=float)
    A_scen[:, :n] = -payoff_matrix.T          # (S, n)
    A_scen[:, n] = -1.0                       # −v column
    A_scen[np.arange(S), n + 1 + np.arange(S)] = -1.0  # −u_s diagonal
    b_scen = np.zeros(S, dtype=float)

    # Row S: CVaR constraint  v + (1/(α·S)) · Σ_s u_s ≤ max_drawdown_pct.
    A_cvar = np.zeros((1, n + 1 + S), dtype=float)
    A_cvar[0, n] = 1.0
    A_cvar[0, n + 1 :] = 1.0 / (alpha * S)
    b_cvar = np.array([float(max_drawdown_pct)], dtype=float)

    # Row S+1: budget  Σ_i f_i ≤ 1.
    A_budget = np.zeros((1, n + 1 + S), dtype=float)
    A_budget[0, :n] = 1.0
    b_budget = np.array([1.0], dtype=float)

    A_ub = np.vstack([A_scen, A_cvar, A_budget])
    b_ub = np.concatenate([b_scen, b_cvar, b_budget])

    # ── Bounds: per-candidate ¼-Kelly cap; v free; u_s ≥ 0 ────────────
    f_bounds: list[tuple[Optional[float], Optional[float]]] = [
        (0.0, _per_candidate_upper_bound(c, max_bet_fraction))
        for c in candidates
    ]
    v_bound: tuple[Optional[float], Optional[float]] = (None, None)
    u_bounds: list[tuple[Optional[float], Optional[float]]] = [(0.0, None)] * S
    bounds = f_bounds + [v_bound] + u_bounds

    result = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        log.warning(
            "portfolio.optimize.lp_failed",
            card_id=card_id,
            message=result.message,
            status=result.status,
            n_candidates=n,
            n_scenarios=S,
        )
        return _empty_portfolio(card_id, bankroll)

    x = result.x
    f = x[:n]
    v = float(x[n])
    u = x[n + 1 :]

    # ── Assemble Portfolio ─────────────────────────────────────────────
    # Clip floating-point noise at the lower bound (linprog may return
    # tiny negative numbers for variables at 0).
    f = np.clip(f, 0.0, None)

    recommendations: list[BetRecommendation] = []
    for c, f_i in zip(candidates, f):
        if f_i <= _F_TOL:
            continue
        recommendations.append(
            BetRecommendation(
                candidate=c,
                stake=float(f_i * bankroll),
                stake_fraction=float(f_i),
            )
        )

    expected_return = float(bankroll * np.dot(f, ev_per_dollar))
    var_95_usd = float(bankroll * max(0.0, v))
    cvar_lhs = v + (1.0 / (alpha * S)) * float(np.sum(u))
    cvar_95_usd = float(bankroll * cvar_lhs)
    # Clamp the total stake fraction to [0, 1] to absorb floating-point
    # tolerance against the Portfolio schema's upper bound.
    total_stake_fraction = float(min(1.0, max(0.0, float(np.sum(f)))))

    return Portfolio(
        card_id=card_id,
        bankroll=bankroll,
        recommendations=recommendations,
        expected_return=expected_return,
        var_95=var_95_usd,
        cvar_95=cvar_95_usd,
        total_stake_fraction=total_stake_fraction,
    )


__all__ = ["optimize_portfolio"]
