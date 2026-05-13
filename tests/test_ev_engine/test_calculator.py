"""Unit tests for app/services/ev_engine/calculator.py."""
from __future__ import annotations

import numpy as np
import pytest

from app.schemas.race import BetType
from app.services.ev_engine.calculator import (
    DEFAULT_MIN_EDGE,
    compute_ev_candidates,
    expected_value_per_dollar,
)


def test_expected_value_per_dollar_positive_edge():
    """EV per $1 = model_prob × decimal_odds − 1."""
    assert expected_value_per_dollar(0.30, 4.0) == pytest.approx(0.20)


def test_expected_value_per_dollar_negative_edge():
    assert expected_value_per_dollar(0.20, 4.0) == pytest.approx(-0.20)


def test_expected_value_per_dollar_zero_edge():
    assert expected_value_per_dollar(0.25, 4.0) == pytest.approx(0.0)


def test_default_min_edge_matches_master_reference():
    """Master Reference §151: 5–10% minimum edge threshold."""
    assert DEFAULT_MIN_EDGE == pytest.approx(0.05)


def test_compute_win_candidates_filters_below_threshold():
    """Only candidates with edge >= min_edge are returned."""
    # Field of 4 horses; horse 0 is strongly +EV, others -EV.
    win_probs = np.array([0.50, 0.20, 0.20, 0.10])
    # Public odds (1/p, fair, no take): 2.0, 5.0, 5.0, 10.0
    # Suppose market is "wrong": public has ML 4.0 on horse 0 (implied 0.25),
    # 4.0 on horse 1, 5.0 on horse 2, 10.0 on horse 3.
    decimal_odds = np.array([4.0, 4.0, 5.0, 10.0])
    # Edges: horse 0: 0.50 - 0.25 = 0.25 (PASS); horse 1: 0.20 - 0.25 = -0.05 (FAIL)
    # horse 2: 0.20 - 0.20 = 0.0 (FAIL); horse 3: 0.10 - 0.10 = 0.0 (FAIL)
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    win_cands = [c for c in candidates if c.bet_type == BetType.WIN]
    assert len(win_cands) == 1
    assert win_cands[0].selection == (0,)
    assert win_cands[0].edge == pytest.approx(0.25)


def test_compute_win_candidates_sorted_by_descending_ev():
    """Output must be sorted by descending expected_value."""
    win_probs = np.array([0.50, 0.30, 0.20])
    decimal_odds = np.array([4.0, 5.0, 8.0])
    # Edges: h0: 0.50-0.25=0.25; h1: 0.30-0.20=0.10; h2: 0.20-0.125=0.075
    # EVs: h0: 0.50*4-1=1.0; h1: 0.30*5-1=0.5; h2: 0.20*8-1=0.6
    # Sorted by descending EV: h0 (1.0), h2 (0.6), h1 (0.5)
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    evs = [c.expected_value for c in candidates]
    assert evs == sorted(evs, reverse=True)


def test_compute_win_candidates_kelly_sized_and_capped():
    """Each candidate's kelly_fraction must respect 1/4 Kelly + 3% cap."""
    # Massive edge to force Kelly above cap
    win_probs = np.array([0.95, 0.05])
    decimal_odds = np.array([10.0, 2.0])
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    for c in candidates:
        assert c.kelly_fraction <= 0.03, "must respect 3% cap"
        assert c.kelly_fraction >= 0.0


def test_compute_win_candidates_returns_empty_when_no_edge():
    """No bets at fair odds."""
    win_probs = np.array([0.40, 0.30, 0.20, 0.10])
    decimal_odds = np.array([2.5, 1 / 0.30, 5.0, 10.0])  # all fair
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    assert candidates == []


def test_compute_win_candidates_validates_lengths_match():
    """win_probs and decimal_odds must have the same length."""
    with pytest.raises(ValueError, match="length"):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.5, 0.5]),
            decimal_odds=np.array([2.0, 2.0, 2.0]),
            bet_types=[BetType.WIN],
            min_edge=0.05,
        )


def test_compute_win_candidates_validates_probs_sum_to_one():
    with pytest.raises(ValueError, match="sum"):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.3, 0.3, 0.3]),  # sums to 0.9
            decimal_odds=np.array([4.0, 4.0, 4.0]),
            bet_types=[BetType.WIN],
            min_edge=0.05,
        )


def test_market_impact_lowers_post_odds_in_candidate():
    """When pool_size is passed, post-bet odds (and EV) should decrease."""
    win_probs = np.array([0.40, 0.30, 0.30])
    decimal_odds = np.array([5.0, 4.0, 4.0])

    no_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
        bankroll=10_000.0,
        pool_sizes={BetType.WIN: None},
    )
    with_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
        bankroll=10_000.0,
        pool_sizes={BetType.WIN: 5_000.0},
    )
    # horse 0 has positive edge in both. With impact, EV should be lower.
    assert no_impact[0].selection == (0,)
    assert with_impact[0].selection == (0,)
    assert with_impact[0].expected_value < no_impact[0].expected_value
    assert with_impact[0].market_impact_applied is True
    assert no_impact[0].market_impact_applied is False
