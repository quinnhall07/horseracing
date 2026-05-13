"""
app/services/ev_engine/calculator.py
────────────────────────────────────
Phase 5a — Expected Value calculator (orchestrator).

For each race, given a calibrated vector of win probabilities and a parallel
vector of decimal odds, produce a list of BetCandidate objects covering
WIN, EXACTA, TRIFECTA, and SUPERFECTA. Each candidate carries:
    - the model probability (PL-derived for exotics)
    - the (possibly market-impact-adjusted) decimal odds
    - edge = model_prob − market_prob
    - expected value per $1 = model_prob × odds − 1
    - 1/4 Kelly stake fraction (capped at 3% per ADR-002)

Place/Show and Pick 3/4/6 are deferred (ADR-039).

The calculator is source-agnostic for odds: caller supplies the array.
For backtesting, callers pass historical `odds_final`. For live, callers
pass morning-line or live tote. Same module, different data.

Market impact (Master Reference §10): when callers supply `pool_sizes`, the
calculator uses `post_bet_decimal_odds` to compute the odds AT WHICH the
proposed stake will actually settle. Default pool_size=None ⇒ no impact,
suitable for analyses without live tote data.

Public API:
    compute_ev_candidates(
        race_id, win_probs, decimal_odds, bet_types,
        min_edge=DEFAULT_MIN_EDGE, bankroll=1.0,
        pool_sizes=None, takeout_rates=None, exotic_odds=None,
    ) -> list[BetCandidate]
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from app.core.logging import get_logger
from app.schemas.bets import BetCandidate
from app.schemas.race import BetType
from app.services.ev_engine.market_impact import (
    DEFAULT_TAKEOUT,
    post_bet_decimal_odds,
)
from app.services.ordering.plackett_luce import (
    exacta_prob,
    trifecta_prob,
    superfecta_prob,
)
from app.services.portfolio.sizing import apply_bet_cap, kelly_fraction

log = get_logger(__name__)


DEFAULT_MIN_EDGE: float = 0.05
"""Per Master Reference §151: 5-10% minimum edge threshold. 0.05 is the
permissive default; tighten for production by passing min_edge=0.08."""


_SUM_TOL: float = 1e-5


def expected_value_per_dollar(model_prob: float, decimal_odds: float) -> float:
    """EV per $1 staked: p × O − 1.

    Equivalent to p × (O − 1) − (1 − p) × 1; we use the algebraic
    simplification because it's one multiply.
    """
    return float(model_prob * decimal_odds - 1.0)


def _validate_inputs(win_probs: np.ndarray, decimal_odds: np.ndarray) -> None:
    if len(win_probs) != len(decimal_odds):
        raise ValueError(
            f"win_probs and decimal_odds must have the same length; "
            f"got {len(win_probs)} and {len(decimal_odds)}"
        )
    s = float(np.sum(win_probs))
    if abs(s - 1.0) > _SUM_TOL:
        raise ValueError(f"win_probs must sum to 1; got {s}")
    if (win_probs < -1e-12).any():
        raise ValueError("win_probs must be non-negative")
    if (decimal_odds < 1.0).any():
        raise ValueError("decimal_odds must all be >= 1")


def _candidate_for_win(
    race_id: str,
    horse_idx: int,
    model_prob: float,
    pre_odds: float,
    bankroll: float,
    pool_size: Optional[float],
    takeout_rate: float,
    min_edge: float,
) -> Optional[BetCandidate]:
    """Build a single WIN BetCandidate. Returns None if edge < min_edge."""
    # Apply market impact if pool_size is available. We need the proposed
    # stake to compute the post-bet odds, but the stake itself depends on
    # the post-bet odds via Kelly. We use the PRE-impact Kelly as a first
    # estimate, then recompute odds at that stake. Because post_odds ≤
    # pre_odds always, post_kelly ≤ pre_kelly, so using pre_kelly as the
    # impact-estimate stake OVERESTIMATES impact — i.e., the reported EV
    # is a conservative lower bound on the true settled EV. When stake
    # vastly exceeds pool, the post-impact edge recheck correctly returns
    # None rather than producing a silently-wrong candidate.
    pre_market_prob = 1.0 / pre_odds
    pre_edge = model_prob - pre_market_prob
    if pre_edge < min_edge:
        return None

    if pool_size is not None:
        pre_kelly = kelly_fraction(pre_edge, pre_odds)
        pre_stake_frac_capped = apply_bet_cap(pre_kelly)
        pre_stake = pre_stake_frac_capped * bankroll
        decimal_odds = post_bet_decimal_odds(
            pre_odds=pre_odds,
            bet_amount=pre_stake,
            pool_size=pool_size,
            takeout_rate=takeout_rate,
        )
        market_impact_applied = True
    else:
        decimal_odds = pre_odds
        market_impact_applied = False

    market_prob = 1.0 / decimal_odds
    edge = model_prob - market_prob
    if edge < min_edge:
        return None

    ev = expected_value_per_dollar(model_prob, decimal_odds)
    kelly = kelly_fraction(edge, decimal_odds)
    kelly_capped = apply_bet_cap(kelly)

    return BetCandidate(
        race_id=race_id,
        bet_type=BetType.WIN,
        selection=(horse_idx,),
        model_prob=float(model_prob),
        decimal_odds=float(decimal_odds),
        market_prob=float(market_prob),
        edge=float(edge),
        expected_value=float(ev),
        kelly_fraction=float(kelly_capped),
        market_impact_applied=market_impact_applied,
        pool_size=pool_size,
    )


_PL_PROB_FN = {
    BetType.EXACTA: lambda p, sel: exacta_prob(p, sel[0], sel[1]),
    BetType.TRIFECTA: lambda p, sel: trifecta_prob(p, sel[0], sel[1], sel[2]),
    BetType.SUPERFECTA: lambda p, sel: superfecta_prob(p, sel[0], sel[1], sel[2], sel[3]),
}


def _candidate_for_exotic(
    race_id: str,
    bet_type: BetType,
    selection: tuple[int, ...],
    win_probs: np.ndarray,
    pre_odds: float,
    bankroll: float,
    pool_size: Optional[float],
    takeout_rate: float,
    min_edge: float,
) -> Optional[BetCandidate]:
    """Build a single exotic BetCandidate. Returns None if edge < min_edge."""
    pl_fn = _PL_PROB_FN[bet_type]
    model_prob = float(pl_fn(win_probs, selection))

    pre_market_prob = 1.0 / pre_odds
    pre_edge = model_prob - pre_market_prob
    if pre_edge < min_edge:
        return None

    if pool_size is not None:
        pre_kelly = kelly_fraction(pre_edge, pre_odds)
        pre_stake_frac = apply_bet_cap(pre_kelly)
        pre_stake = pre_stake_frac * bankroll
        decimal_odds = post_bet_decimal_odds(
            pre_odds=pre_odds,
            bet_amount=pre_stake,
            pool_size=pool_size,
            takeout_rate=takeout_rate,
        )
        market_impact_applied = True
    else:
        decimal_odds = pre_odds
        market_impact_applied = False

    market_prob = 1.0 / decimal_odds
    edge = model_prob - market_prob
    if edge < min_edge:
        return None

    ev = expected_value_per_dollar(model_prob, decimal_odds)
    kelly = kelly_fraction(edge, decimal_odds)
    kelly_capped = apply_bet_cap(kelly)

    return BetCandidate(
        race_id=race_id,
        bet_type=bet_type,
        selection=selection,
        model_prob=model_prob,
        decimal_odds=float(decimal_odds),
        market_prob=float(market_prob),
        edge=float(edge),
        expected_value=float(ev),
        kelly_fraction=float(kelly_capped),
        market_impact_applied=market_impact_applied,
        pool_size=pool_size,
    )


def compute_ev_candidates(
    race_id: str,
    win_probs: np.ndarray,
    decimal_odds: np.ndarray,
    bet_types: Iterable[BetType],
    min_edge: float = DEFAULT_MIN_EDGE,
    bankroll: float = 1.0,
    pool_sizes: Optional[dict[BetType, Optional[float]]] = None,
    takeout_rates: Optional[dict[BetType, float]] = None,
    exotic_odds: Optional[dict[BetType, dict[tuple[int, ...], float]]] = None,
) -> list[BetCandidate]:
    """Generate all +EV BetCandidate objects for a single race.

    Args:
        race_id, win_probs, decimal_odds: as before. decimal_odds is used
            only for Win-pool bets; exotics use `exotic_odds`.
        bet_types: iterable of BetType enums to evaluate.
        min_edge:  minimum edge to include. Default 0.05.
        bankroll:  used for dollar-stake estimation in market impact. Default 1.0.
        pool_sizes: dict mapping BetType → pool size in $. None disables impact.
        takeout_rates: dict overriding DEFAULT_TAKEOUT per bet type.
        exotic_odds:   dict[BetType, dict[selection-tuple, decimal_odds]]
                       Required for EXACTA / TRIFECTA / SUPERFECTA.

    Returns:
        List of BetCandidate sorted by descending expected_value.
    """
    win_probs = np.asarray(win_probs, dtype=float).ravel()
    decimal_odds = np.asarray(decimal_odds, dtype=float).ravel()
    _validate_inputs(win_probs, decimal_odds)

    pool_sizes = pool_sizes or {}
    takeout_rates = takeout_rates or {}
    exotic_odds = exotic_odds or {}

    candidates: list[BetCandidate] = []

    for bet_type in bet_types:
        pool_size = pool_sizes.get(bet_type)
        takeout = takeout_rates.get(bet_type, DEFAULT_TAKEOUT[bet_type.value])

        if bet_type == BetType.WIN:
            for i in range(len(win_probs)):
                c = _candidate_for_win(
                    race_id=race_id,
                    horse_idx=i,
                    model_prob=float(win_probs[i]),
                    pre_odds=float(decimal_odds[i]),
                    bankroll=bankroll,
                    pool_size=pool_size,
                    takeout_rate=takeout,
                    min_edge=min_edge,
                )
                if c is not None:
                    candidates.append(c)
        elif bet_type in (BetType.EXACTA, BetType.TRIFECTA, BetType.SUPERFECTA):
            if bet_type not in exotic_odds:
                raise ValueError(
                    f"exotic_odds dict required for {bet_type}; "
                    f"caller must supply per-permutation gross decimal odds."
                )
            for selection, pre_odds in exotic_odds[bet_type].items():
                c = _candidate_for_exotic(
                    race_id=race_id,
                    bet_type=bet_type,
                    selection=tuple(selection),
                    win_probs=win_probs,
                    pre_odds=float(pre_odds),
                    bankroll=bankroll,
                    pool_size=pool_size,
                    takeout_rate=takeout,
                    min_edge=min_edge,
                )
                if c is not None:
                    candidates.append(c)
        else:
            raise ValueError(
                f"bet_type {bet_type} not supported in Phase 5a "
                f"(ADR-039). Use WIN, EXACTA, TRIFECTA, or SUPERFECTA."
            )

    candidates.sort(key=lambda c: c.expected_value, reverse=True)
    return candidates


__all__ = [
    "DEFAULT_MIN_EDGE",
    "compute_ev_candidates",
    "expected_value_per_dollar",
]
