"""Unit tests for app/services/ev_engine/market_impact.py."""
from __future__ import annotations

import pytest

from app.services.ev_engine.market_impact import (
    inferred_winning_bets,
    post_bet_decimal_odds,
    DEFAULT_TAKEOUT,
)


def test_default_takeout_per_bet_type_present():
    assert DEFAULT_TAKEOUT["win"] == pytest.approx(0.17)
    assert DEFAULT_TAKEOUT["exacta"] == pytest.approx(0.21)
    assert DEFAULT_TAKEOUT["trifecta"] == pytest.approx(0.25)
    assert DEFAULT_TAKEOUT["superfecta"] == pytest.approx(0.25)


def test_no_pool_size_returns_pre_odds_unchanged():
    """pool_size=None means 'infinite pool' → no market impact."""
    assert post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=100.0, pool_size=None, takeout_rate=0.17
    ) == pytest.approx(5.0)


def test_zero_bet_returns_pre_odds():
    """Adding $0 to the pool cannot change the odds."""
    assert post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=0.0, pool_size=10_000.0, takeout_rate=0.17
    ) == pytest.approx(5.0)


def test_post_odds_monotonically_decreases_with_bet_size():
    """Larger stake → smaller decimal odds (more competition for the pool)."""
    pool = 10_000.0
    odds = [
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=x, pool_size=pool, takeout_rate=0.17
        )
        for x in [0.0, 10.0, 100.0, 1_000.0, 10_000.0]
    ]
    for i in range(1, len(odds)):
        assert odds[i] < odds[i - 1], (
            f"post-odds must decrease with stake; got {odds}"
        )


def test_inferred_winning_bets_matches_pari_mutuel_definition():
    """Given pre_odds = (1-τ) × pool / B_winning, recover B_winning."""
    pool = 10_000.0
    pre_odds = 5.0
    tau = 0.17
    B = inferred_winning_bets(pre_odds=pre_odds, pool_size=pool, takeout_rate=tau)
    # 5 = (1-0.17) × 10000 / B  →  B = 0.83 × 10000 / 5 = 1660.0
    assert B == pytest.approx(1660.0)


def test_post_odds_asymptotes_to_one_minus_takeout():
    """As bet → ∞, all winners are you; decimal odds → (1 − τ)."""
    pool = 10_000.0
    tau = 0.17
    odds = post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=1e12, pool_size=pool, takeout_rate=tau
    )
    assert odds == pytest.approx(1.0 - tau, abs=1e-3)


def test_post_odds_explicit_closed_form_value():
    """pre_odds=5.0, pool=10000, takeout=0.17, bet=1000:
        B_winning = 0.83*10000/5 = 1660
        post = 0.83*(10000+1000)/(1660+1000) = 0.83*11000/2660 = 3.4323...
    """
    odds = post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=1000.0, pool_size=10_000.0, takeout_rate=0.17
    )
    expected = 0.83 * 11_000.0 / 2_660.0
    assert odds == pytest.approx(expected, abs=1e-9)


def test_rejects_pre_odds_below_one():
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=0.5, bet_amount=100.0, pool_size=10_000.0, takeout_rate=0.17
        )


def test_rejects_negative_bet():
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=-1.0, pool_size=10_000.0, takeout_rate=0.17
        )


def test_rejects_takeout_outside_unit():
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=100.0, pool_size=10_000.0, takeout_rate=1.1
        )
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=100.0, pool_size=10_000.0, takeout_rate=-0.01
        )


def test_post_odds_at_winning_pool_saturation():
    """bet_amount == B (winning-pool size): the user doubles the winning
    side. Post odds = (1 − τ)(P + B)/(2B). This is the 50% saturation
    structural case — sits between zero-impact and full-asymptote."""
    pool = 10_000.0
    tau = 0.17
    pre_odds = 5.0
    B = (1.0 - tau) * pool / pre_odds  # = 1660.0
    odds = post_bet_decimal_odds(
        pre_odds=pre_odds, bet_amount=B, pool_size=pool, takeout_rate=tau
    )
    expected = (1.0 - tau) * (pool + B) / (2.0 * B)
    assert odds == pytest.approx(expected, abs=1e-9)


def test_rejects_non_positive_pool_size():
    """pool_size <= 0 must raise regardless of which entry point is used."""
    with pytest.raises(ValueError, match="pool_size"):
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=100.0, pool_size=0.0, takeout_rate=0.17
        )
    with pytest.raises(ValueError, match="pool_size"):
        inferred_winning_bets(pre_odds=5.0, pool_size=-1.0, takeout_rate=0.17)
