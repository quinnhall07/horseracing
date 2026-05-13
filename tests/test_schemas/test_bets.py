"""Schema validation tests for app/schemas/bets.py."""
from __future__ import annotations

import pytest

from app.schemas.bets import BetCandidate, BetRecommendation, Portfolio
from app.schemas.race import BetType


def _candidate(**overrides) -> BetCandidate:
    base = dict(
        race_id="CD-2026-05-10-R4",
        bet_type=BetType.WIN,
        selection=(0,),
        model_prob=0.30,
        decimal_odds=4.0,
        market_prob=0.25,
        edge=0.05,
        expected_value=0.20,
        kelly_fraction=0.0125,
        market_impact_applied=False,
        pool_size=None,
    )
    base.update(overrides)
    return BetCandidate(**base)


def test_win_candidate_round_trip():
    c = _candidate()
    assert c.bet_type == BetType.WIN
    assert c.selection == (0,)
    assert c.edge == pytest.approx(0.05)


def test_exacta_requires_two_distinct_indices():
    c = _candidate(bet_type=BetType.EXACTA, selection=(0, 1))
    assert c.selection == (0, 1)
    with pytest.raises(ValueError, match="selection length"):
        _candidate(bet_type=BetType.EXACTA, selection=(0,))
    with pytest.raises(ValueError, match="distinct"):
        _candidate(bet_type=BetType.EXACTA, selection=(0, 0))


def test_trifecta_length_three():
    _candidate(bet_type=BetType.TRIFECTA, selection=(0, 1, 2))
    with pytest.raises(ValueError, match="selection length"):
        _candidate(bet_type=BetType.TRIFECTA, selection=(0, 1))


def test_superfecta_length_four():
    _candidate(bet_type=BetType.SUPERFECTA, selection=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="selection length"):
        _candidate(bet_type=BetType.SUPERFECTA, selection=(0, 1, 2))


def test_probabilities_in_unit_interval():
    with pytest.raises(ValueError):
        _candidate(model_prob=1.5)
    with pytest.raises(ValueError):
        _candidate(market_prob=-0.1)


def test_decimal_odds_at_least_one():
    with pytest.raises(ValueError):
        _candidate(decimal_odds=0.5)


def test_kelly_fraction_non_negative_and_capped_below_one():
    with pytest.raises(ValueError):
        _candidate(kelly_fraction=-0.01)
    with pytest.raises(ValueError):
        _candidate(kelly_fraction=1.5)


def test_pick_n_bet_types_rejected_in_5a():
    """Phase 5a does not support cross-race bets. ADR-039 defers Pick3/4/6."""
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.PICK3, selection=(0, 1, 2))
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.PICK4, selection=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.PICK6, selection=(0, 1, 2, 3, 4, 5))


def test_place_and_show_rejected_in_5a():
    """ADR-039 defers Place/Show until live pool composition is available."""
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.PLACE, selection=(0,))
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.SHOW, selection=(0,))


def test_recommendation_round_trip():
    rec = BetRecommendation(
        candidate=_candidate(),
        stake=125.00,
        stake_fraction=0.0125,
    )
    assert rec.stake == pytest.approx(125.0)
    assert rec.stake_fraction == pytest.approx(0.0125)


def test_portfolio_empty_recommendations_allowed():
    p = Portfolio(
        card_id="CD-2026-05-10",
        bankroll=10000.0,
        recommendations=[],
        expected_return=0.0,
        var_95=0.0,
        cvar_95=0.0,
        total_stake_fraction=0.0,
    )
    assert p.recommendations == []
