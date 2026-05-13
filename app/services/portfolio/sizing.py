"""
app/services/portfolio/sizing.py
────────────────────────────────
Phase 5 — Fractional Kelly bet sizing.

Per ADR-002 (CLAUDE.md §2): all bet sizing uses 1/4 Kelly, capped at 3% of
bankroll per single bet. The Kelly fraction is a STAKE FRACTION of the
bankroll, not a multiple of expected value.

    full_kelly  = max(0, (edge × decimal_odds − (1 − edge)) / decimal_odds)
    bet_fraction = full_kelly × fraction        # fraction=0.25 by default
    capped      = min(bet_fraction, max_bet_fraction)   # 0.03 by default

These are pure functions; no state, no I/O. They live in `portfolio/`
because the CVaR optimiser (Phase 5b) consumes them as upper-bound inputs.
"""

from __future__ import annotations

DEFAULT_KELLY_FRACTION: float = 0.25
"""Per ADR-002: 1/4 Kelly is the universal fraction. Override only with
strong rationale; see ADR-002 'Rejected Alternatives' before changing."""

DEFAULT_MAX_BET_FRACTION: float = 0.03
"""Per ADR-002: hard cap of 3% of bankroll on any single bet, regardless of
what the Kelly formula returns. Guards against Kelly blowup on high-edge
exotic combinations where the formula can output very large fractions."""


def kelly_fraction(
    edge: float,
    decimal_odds: float,
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> float:
    """Fractional Kelly bet size as a fraction of bankroll.

    Args:
        edge:          model_prob − market_prob. Can be negative; if so or 0,
                       returns 0.
        decimal_odds:  gross decimal odds (e.g., 3-1 == 4.0). Must be >= 1.
        fraction:      Kelly multiplier in [0, 1]. Default 0.25 (ADR-002).

    Returns:
        Stake fraction of bankroll in [0, fraction]. Truncated at 0 for
        negative-EV positions.
    """
    if decimal_odds < 1.0:
        raise ValueError(f"decimal_odds must be >= 1; got {decimal_odds}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}")

    full = (edge * decimal_odds - (1.0 - edge)) / decimal_odds
    return max(0.0, full) * fraction


def apply_bet_cap(
    stake_fraction: float,
    cap: float = DEFAULT_MAX_BET_FRACTION,
) -> float:
    """Clamp a stake fraction to the per-bet hard cap.

    Args:
        stake_fraction: a non-negative fraction of bankroll.
        cap:            upper bound in [0, 1]. Default 0.03 (ADR-002).

    Returns:
        min(stake_fraction, cap).
    """
    if stake_fraction < 0.0:
        raise ValueError(f"stake_fraction must be >= 0; got {stake_fraction}")
    if not 0.0 <= cap <= 1.0:
        raise ValueError(f"cap must be in [0, 1]; got {cap}")
    return min(stake_fraction, cap)


__all__ = [
    "DEFAULT_KELLY_FRACTION",
    "DEFAULT_MAX_BET_FRACTION",
    "kelly_fraction",
    "apply_bet_cap",
]
