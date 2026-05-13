"""
app/services/ev_engine/market_impact.py
───────────────────────────────────────
Phase 5a — Pari-mutuel market impact model.

In a pari-mutuel pool of size P with takeout rate τ, the public's gross
decimal odds on a winning outcome that received B of the pool is

    pre_odds = (1 − τ) × P / B

When you add stake x to the bet (which must therefore be a winning bet for
this calculation to matter), the pool grows to P + x and the winning share
denominator grows to B + x. The post-bet decimal odds your stake will
actually pay out at is

    post_odds(x) = (1 − τ) × (P + x) / (B + x)

This module exposes:
    * `post_bet_decimal_odds(pre_odds, bet_amount, pool_size, takeout_rate)`
      — primary API; returns post-bet odds for inserting bet_amount into
        a pool of pool_size. pool_size=None disables impact (returns pre_odds).
    * `inferred_winning_bets(pre_odds, pool_size, takeout_rate)`
      — helper that recovers B from the public odds. Useful for diagnostics
        and for the calculator when only pre_odds and pool_size are known.
    * `DEFAULT_TAKEOUT` — typical track takeout rates per bet type from
      Master Reference §190 (Section 5: Bet Type Strategy).

Per Master Reference §10 ("The Two Genuine Edges"), market impact modelling
is one of the two real competitive moats over public/commercial systems.
Phase 5a wires it in but defaults `pool_size=None` (zero impact) so that
the EV engine produces valid output even without live tote data. When live
pool sizes are available, callers populate pool_size and the engine
automatically accounts for self-impact.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_TAKEOUT: dict[str, float] = {
    "win": 0.17,
    "place": 0.17,
    "show": 0.17,
    "exacta": 0.21,
    "trifecta": 0.25,
    "superfecta": 0.25,
    "pick3": 0.25,
    "pick4": 0.25,
    "pick6": 0.25,
}
"""Typical US pari-mutuel takeout rates by bet type. Per Master Reference §190.
Track-specific overrides should be passed explicitly when known."""


def _validate(
    pre_odds: float,
    bet_amount: float,
    takeout_rate: float,
    pool_size: Optional[float] = None,
) -> None:
    """Single source of truth for all market-impact input validation.

    `bet_amount=0` and `pool_size=None` are both valid; the caller decides
    what to do with those (e.g., the zero-impact early return). `pool_size`
    is checked only when provided.
    """
    if pre_odds < 1.0:
        raise ValueError(f"pre_odds must be >= 1; got {pre_odds}")
    if bet_amount < 0.0:
        raise ValueError(f"bet_amount must be >= 0; got {bet_amount}")
    if not 0.0 <= takeout_rate < 1.0:
        raise ValueError(f"takeout_rate must be in [0, 1); got {takeout_rate}")
    if pool_size is not None and pool_size <= 0.0:
        raise ValueError(f"pool_size must be > 0; got {pool_size}")


def inferred_winning_bets(
    pre_odds: float, pool_size: float, takeout_rate: float
) -> float:
    """Recover the size of the winning-outcome bets from the public odds.

    pre_odds = (1 − τ) × pool_size / B   →   B = (1 − τ) × pool_size / pre_odds
    """
    _validate(pre_odds, 0.0, takeout_rate, pool_size=pool_size)
    return (1.0 - takeout_rate) * pool_size / pre_odds


def post_bet_decimal_odds(
    pre_odds: float,
    bet_amount: float,
    pool_size: Optional[float],
    takeout_rate: float,
) -> float:
    """Decimal odds after adding `bet_amount` to a pari-mutuel pool.

    Args:
        pre_odds:     public gross decimal odds before your bet. >= 1.
        bet_amount:   the stake you propose to add. >= 0.
        pool_size:    total pre-bet pool in $. If None, no impact applied
                      (callable for systems without live tote data).
        takeout_rate: track takeout fraction in [0, 1). e.g. 0.17 for Win.

    Returns:
        post_odds = (1 − τ)(P + x) / (B + x), where B is inferred from
        pre_odds and pool_size. Returns pre_odds when pool_size is None
        or bet_amount is 0.
    """
    _validate(pre_odds, bet_amount, takeout_rate, pool_size=pool_size)
    if pool_size is None or bet_amount == 0.0:
        return float(pre_odds)

    B = inferred_winning_bets(pre_odds, pool_size, takeout_rate)
    new_pool = pool_size + bet_amount
    new_B = B + bet_amount
    return float((1.0 - takeout_rate) * new_pool / new_B)


__all__ = [
    "DEFAULT_TAKEOUT",
    "inferred_winning_bets",
    "post_bet_decimal_odds",
]
