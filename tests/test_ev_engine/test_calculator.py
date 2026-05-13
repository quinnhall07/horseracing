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


# ── Exotic bets ────────────────────────────────────────────────────────────


def test_compute_exacta_candidates_from_exotic_odds_dict():
    """Caller supplies per-permutation gross odds; calculator filters by edge."""
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.50, 0.30, 0.20])
    # Public exacta odds (gross decimal); we make some +EV and one -EV to
    # exercise the filter from both directions.
    exotic_odds = {
        BetType.EXACTA: {
            (0, 1): 8.0,   # PL prob: 0.50 * 0.30 / 0.50 = 0.30; market 0.125; edge 0.175 → PASS
            (0, 2): 12.0,  # PL prob: 0.50 * 0.20 / 0.50 = 0.20; market 0.083; edge 0.117 → PASS
            (1, 0): 8.0,   # PL prob: 0.30 * 0.50 / 0.70 ≈ 0.214; market 0.125; edge 0.089 → PASS
            (1, 2): 30.0,  # PL prob: 0.30 * 0.20 / 0.70 ≈ 0.0857; market 0.033; edge 0.052 → PASS
            (2, 1): 3.0,   # PL prob: 0.20 * 0.30 / 0.80 = 0.075; market 0.333; edge -0.258 → FAIL
        }
    }
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.5, 5.0]),
        bet_types=[BetType.EXACTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
    )
    selections = {c.selection for c in candidates}
    # Only the four positive-edge permutations pass; (2, 1) is filtered out.
    assert selections == {(0, 1), (0, 2), (1, 0), (1, 2)}
    assert (2, 1) not in selections


def test_compute_trifecta_candidates():
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.50, 0.30, 0.15, 0.05])
    # Pick one strongly +EV trifecta and one -EV; calculator must filter.
    exotic_odds = {
        BetType.TRIFECTA: {
            (0, 1, 2): 30.0,   # PL: 0.50 * (0.30/0.50) * (0.15/0.20) = 0.225;
                               #     market: 0.0333; edge: 0.19 → PASS
            (0, 1, 3): 200.0,  # PL: 0.50 * (0.30/0.50) * (0.05/0.20) = 0.075;
                               #     market: 0.005; edge: 0.07 → PASS
            (3, 2, 1): 100.0,  # PL: 0.05 * (0.15/0.95) * (0.30/0.80) ≈ 0.00296;
                               #     market: 0.01; edge: ≈-0.007 → FAIL
        }
    }
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.3, 6.7, 20.0]),
        bet_types=[BetType.TRIFECTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
    )
    selections = {c.selection for c in candidates}
    assert selections == {(0, 1, 2), (0, 1, 3)}


def test_compute_superfecta_candidate():
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.40, 0.30, 0.20, 0.10])
    exotic_odds = {
        BetType.SUPERFECTA: {
            (0, 1, 2, 3): 50.0,  # PL: 0.40 * (0.30/0.60) * (0.20/0.30) * (0.10/0.10)
                                 #     = 0.40 * 0.5 * 0.667 * 1.0 = 0.1333
                                 # market: 0.02; edge: 0.1133 → PASS
        }
    }
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.5, 3.3, 5.0, 10.0]),
        bet_types=[BetType.SUPERFECTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
    )
    assert len(candidates) == 1
    assert candidates[0].selection == (0, 1, 2, 3)
    assert candidates[0].edge == pytest.approx(0.1133, abs=1e-3)


def test_exotic_bet_without_odds_dict_raises():
    from app.services.ev_engine.calculator import compute_ev_candidates

    with pytest.raises(ValueError, match="exotic_odds"):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.5, 0.3, 0.2]),
            decimal_odds=np.array([2.0, 3.3, 5.0]),
            bet_types=[BetType.EXACTA],
            min_edge=0.05,
            exotic_odds=None,
        )


def test_exotic_market_impact_reduces_ev():
    """Same as Win market-impact test, but for an exotic pool."""
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.50, 0.30, 0.20])
    exotic_odds = {BetType.EXACTA: {(0, 1): 8.0}}

    no_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.3, 5.0]),
        bet_types=[BetType.EXACTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
        bankroll=10_000.0,
        pool_sizes={BetType.EXACTA: None},
    )
    with_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.3, 5.0]),
        bet_types=[BetType.EXACTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
        bankroll=10_000.0,
        pool_sizes={BetType.EXACTA: 5_000.0},
    )
    assert with_impact[0].expected_value < no_impact[0].expected_value


def test_exotic_selection_validates_distinct_indices():
    """Repeated indices in an exotic selection must raise.

    The error is raised by `plackett_luce._validate_indices` when the PL
    probability is computed — before the BetCandidate is constructed. The
    BetCandidate schema's own distinctness validator is a second line of
    defence at the schema layer. Either layer firing is acceptable; what
    matters for the calculator's contract is that bad caller data does not
    silently produce a malformed candidate.
    """
    from app.services.ev_engine.calculator import compute_ev_candidates

    with pytest.raises(ValueError):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.5, 0.3, 0.2]),
            decimal_odds=np.array([2.0, 3.3, 5.0]),
            bet_types=[BetType.EXACTA],
            min_edge=0.05,
            exotic_odds={BetType.EXACTA: {(0, 0): 10.0}},  # repeated index
        )
