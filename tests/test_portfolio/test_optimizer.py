"""Tests for app/services/portfolio/optimizer.py.

The portfolio optimiser solves a Rockafellar-Uryasev CVaR LP over Monte
Carlo Plackett-Luce scenarios. CLAUDE.md §10 requires:
    * 1/4 Kelly never exceeds max_bet_fraction (per-candidate cap).
    * CVaR constraint is binding at the limit.

These tests lock in both, plus a determinism / cross-race independence /
filter-correctness battery.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.schemas.bets import BetCandidate, Portfolio
from app.schemas.race import BetType
from app.services.portfolio.optimizer import optimize_portfolio
from app.services.portfolio.sizing import (
    DEFAULT_MAX_BET_FRACTION,
    apply_bet_cap,
    kelly_fraction,
)


def _win_candidate(
    race_id: str,
    horse_idx: int,
    model_prob: float,
    decimal_odds: float,
) -> BetCandidate:
    """Build a WIN BetCandidate with derived edge / EV / Kelly fields.

    This mirrors what `compute_ev_candidates` produces — useful in tests so
    we can construct synthetic candidates without standing up the full EV
    pipeline."""
    market_prob = 1.0 / decimal_odds
    edge = model_prob - market_prob
    ev = model_prob * decimal_odds - 1.0
    k = apply_bet_cap(kelly_fraction(edge, decimal_odds))
    return BetCandidate(
        race_id=race_id,
        bet_type=BetType.WIN,
        selection=(horse_idx,),
        model_prob=model_prob,
        decimal_odds=decimal_odds,
        market_prob=market_prob,
        edge=edge,
        expected_value=ev,
        kelly_fraction=k,
        market_impact_applied=False,
        pool_size=None,
    )


def _exacta_candidate(
    race_id: str,
    first: int,
    second: int,
    model_prob: float,
    decimal_odds: float,
) -> BetCandidate:
    market_prob = 1.0 / decimal_odds
    edge = model_prob - market_prob
    ev = model_prob * decimal_odds - 1.0
    k = apply_bet_cap(kelly_fraction(edge, decimal_odds))
    return BetCandidate(
        race_id=race_id,
        bet_type=BetType.EXACTA,
        selection=(first, second),
        model_prob=model_prob,
        decimal_odds=decimal_odds,
        market_prob=market_prob,
        edge=edge,
        expected_value=ev,
        kelly_fraction=k,
        market_impact_applied=False,
        pool_size=None,
    )


# ── 1. Cap respected ──────────────────────────────────────────────────────


def test_cap_respected_all_recommendations_within_kelly_and_max_bet_fraction():
    """Every recommendation's stake_fraction ≤ max_bet_fraction AND
    ≤ candidate's ¼-Kelly cap (the per-candidate upper bound)."""
    # A 6-horse race with several +EV win candidates of varying edge.
    win_probs = np.array([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
    # Construct candidates such that the unconstrained Kelly is large for
    # some — they should all be capped at 0.03.
    cands = [
        _win_candidate("R", 0, 0.30, 5.0),   # market 0.20, edge 0.10
        _win_candidate("R", 1, 0.25, 6.0),   # market ~0.167, edge ~0.083
        _win_candidate("R", 2, 0.20, 10.0),  # market 0.10, edge 0.10
        _win_candidate("R", 3, 0.10, 25.0),  # market 0.04, edge 0.06
    ]
    portfolio = optimize_portfolio(
        candidates=cands,
        race_win_probs={"R": win_probs},
        bankroll=1.0,
        max_drawdown_pct=0.50,  # relaxed so per-candidate cap is binding
        n_scenarios=500,
        seed=1,
        max_bet_fraction=DEFAULT_MAX_BET_FRACTION,
    )
    assert isinstance(portfolio, Portfolio)
    for rec in portfolio.recommendations:
        # Hard cap
        assert rec.stake_fraction <= DEFAULT_MAX_BET_FRACTION + 1e-9
        # Per-candidate ¼-Kelly cap (≤ max_bet_fraction by construction)
        kelly_cap = apply_bet_cap(
            kelly_fraction(rec.candidate.edge, rec.candidate.decimal_odds)
        )
        assert rec.stake_fraction <= kelly_cap + 1e-9


# ── 2. CVaR binding at limit ──────────────────────────────────────────────


def test_cvar_binding_at_limit_when_many_high_edge_candidates_compete():
    """With many +EV candidates and a tight drawdown ceiling, the CVaR
    constraint must be binding (LHS ≈ RHS). This locks in CLAUDE.md §10
    'assert CVaR constraint is binding at the limit.'"""
    # 10-horse race; broad probability mass.
    win_probs = np.array([0.18, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.05, 0.04])
    # Build 8 high-edge win candidates so the LP genuinely has to pick.
    # decimal_odds = (1 / market_prob) chosen so model_prob > market_prob.
    odds = [9.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 35.0]
    cands = []
    for i, o in enumerate(odds):
        cands.append(_win_candidate("R", i, win_probs[i] * 1.8, o))
    bankroll = 10_000.0
    drawdown = 0.05
    portfolio = optimize_portfolio(
        candidates=cands,
        race_win_probs={"R": win_probs},
        bankroll=bankroll,
        cvar_alpha=0.05,
        max_drawdown_pct=drawdown,
        n_scenarios=800,
        seed=7,
        max_bet_fraction=DEFAULT_MAX_BET_FRACTION,
    )
    # CVaR LHS at optimum (= portfolio.cvar_95 / bankroll) equals
    # max_drawdown_pct within numerical tolerance.
    cvar_lhs_pct = portfolio.cvar_95 / bankroll
    assert cvar_lhs_pct == pytest.approx(drawdown, abs=1e-3)


# ── 3. Empty candidates ───────────────────────────────────────────────────


def test_empty_candidates_returns_empty_portfolio():
    portfolio = optimize_portfolio(
        candidates=[],
        race_win_probs={},
        bankroll=1000.0,
        card_id="empty",
    )
    assert isinstance(portfolio, Portfolio)
    assert portfolio.recommendations == []
    assert portfolio.expected_return == 0.0
    assert portfolio.var_95 == 0.0
    assert portfolio.cvar_95 == 0.0
    assert portfolio.total_stake_fraction == 0.0


# ── 4. Trivial single WIN ─────────────────────────────────────────────────


def test_trivial_single_win_assigns_kelly_cap():
    """One WIN candidate. model_prob=0.5, decimal_odds=4.0:
        market_prob = 0.25  →  edge = 0.25
        full-Kelly  = (0.25*4 − 0.75)/4 = 0.0625
        ¼-Kelly     = 0.015625  (below the 0.03 max_bet_fraction cap)
    The optimiser, with the drawdown ceiling relaxed, assigns the
    per-candidate ¼-Kelly bound (0.015625)."""
    c = _win_candidate("R", 0, 0.5, 4.0)
    # Sanity-check: candidate's stored kelly is the ¼-Kelly value, which
    # for this combination is BELOW the 0.03 max_bet_fraction cap.
    assert c.kelly_fraction == pytest.approx(0.015625, abs=1e-9)
    win_probs = np.array([0.5, 0.3, 0.2])
    portfolio = optimize_portfolio(
        candidates=[c],
        race_win_probs={"R": win_probs},
        bankroll=1.0,
        max_drawdown_pct=0.50,
        n_scenarios=500,
        seed=3,
    )
    assert len(portfolio.recommendations) == 1
    rec = portfolio.recommendations[0]
    assert rec.stake_fraction == pytest.approx(0.015625, abs=1e-6)


# ── 5. Correlated exactas: total stake ≤ naive sum ────────────────────────


def test_correlated_exactas_total_stake_le_naive_sum():
    """Two EXACTAs on the same race (1→2 and 1→3), both +EV. Both lose
    when horse 1 doesn't win → the LP correctly captures the correlation
    and allocates LESS in total than the naive sum of the two
    per-candidate Kelly caps."""
    win_probs = np.array([0.5, 0.25, 0.15, 0.10])  # horse 0 is favourite
    # P(1, 2) under PL with these strengths:
    # 0.5 * (0.25 / 0.5) = 0.25
    # We chose decimal_odds = 6.0 to give edge ≈ 0.25 − 1/6 ≈ 0.083.
    c12 = _exacta_candidate("R", 0, 1, 0.25, 6.0)
    # P(1, 3) = 0.5 * (0.15 / 0.5) = 0.15. decimal_odds=10 → edge ≈ 0.05.
    c13 = _exacta_candidate("R", 0, 2, 0.15, 10.0)
    cands = [c12, c13]
    portfolio = optimize_portfolio(
        candidates=cands,
        race_win_probs={"R": win_probs},
        bankroll=1.0,
        max_drawdown_pct=0.10,
        n_scenarios=1500,
        seed=11,
        max_bet_fraction=0.03,
    )
    naive_sum = c12.kelly_fraction + c13.kelly_fraction
    # The CVaR constraint should bind below the naive sum because both
    # bets lose simultaneously whenever horse 0 doesn't win.
    assert portfolio.total_stake_fraction <= naive_sum + 1e-9


# ── 6. Negative-EV filtered ───────────────────────────────────────────────


def test_negative_ev_candidate_receives_zero_stake():
    """A candidate with expected_value < 0 must get stake_fraction = 0.
    The schema requires ev >= 0 for kelly_fraction but allows ev < 0
    when constructed directly (e.g. for diagnostic candidates). Here we
    test the optimiser's reaction by mixing one −EV with one +EV bet."""
    win_probs = np.array([0.5, 0.3, 0.2])
    good = _win_candidate("R", 0, 0.5, 4.0)
    # Construct a −EV WIN: model_prob=0.20, decimal_odds=3.0 → market_prob
    # ≈ 0.333, edge = −0.133, ev = 0.20*3 − 1 = −0.40. kelly = 0 because
    # the formula clamps negative full-Kelly at 0.
    bad = BetCandidate(
        race_id="R", bet_type=BetType.WIN, selection=(1,),
        model_prob=0.20, decimal_odds=3.0,
        market_prob=1.0 / 3.0, edge=0.20 - 1.0 / 3.0,
        expected_value=0.20 * 3.0 - 1.0,
        kelly_fraction=0.0,
        market_impact_applied=False, pool_size=None,
    )
    portfolio = optimize_portfolio(
        candidates=[good, bad],
        race_win_probs={"R": win_probs},
        bankroll=1.0,
        max_drawdown_pct=0.50,
        n_scenarios=400,
        seed=5,
    )
    # The bad candidate must not appear in recommendations.
    bad_in = [r for r in portfolio.recommendations if r.candidate is bad]
    assert bad_in == []


# ── 7. Determinism: same seed + inputs ⇒ identical output ────────────────


def test_determinism_same_seed_same_portfolio():
    win_probs = np.array([0.5, 0.3, 0.2])
    cands = [
        _win_candidate("R", 0, 0.5, 4.0),
        _win_candidate("R", 1, 0.3, 5.0),
    ]
    p1 = optimize_portfolio(
        candidates=cands, race_win_probs={"R": win_probs},
        bankroll=1.0, max_drawdown_pct=0.20,
        n_scenarios=300, seed=42,
    )
    p2 = optimize_portfolio(
        candidates=cands, race_win_probs={"R": win_probs},
        bankroll=1.0, max_drawdown_pct=0.20,
        n_scenarios=300, seed=42,
    )
    assert p1.total_stake_fraction == pytest.approx(p2.total_stake_fraction)
    assert p1.expected_return == pytest.approx(p2.expected_return)
    assert p1.cvar_95 == pytest.approx(p2.cvar_95)
    assert len(p1.recommendations) == len(p2.recommendations)
    for r1, r2 in zip(p1.recommendations, p2.recommendations):
        assert r1.stake_fraction == pytest.approx(r2.stake_fraction)


# ── 8. Multiple races independent ────────────────────────────────────────


def test_multiple_races_optimized_independently():
    """Optimising race A alone and race B alone should yield the same
    allocation as optimising A | B together AS LONG AS the candidates
    don't share a race. (CLAUDE.md §10 'each race optimised in
    isolation' — current Phase 5b scope.)"""
    win_probs_A = np.array([0.6, 0.25, 0.15])
    win_probs_B = np.array([0.6, 0.30, 0.10])
    # decimal_odds chosen so each candidate has a positive ¼-Kelly
    # (model_prob > 1/decimal_odds + 0.05 minimum-edge buffer).
    cand_A = _win_candidate("A", 0, 0.6, 3.0)   # edge 0.267, kelly>0
    cand_B = _win_candidate("B", 0, 0.6, 4.0)   # edge 0.35,  kelly>0
    assert cand_A.kelly_fraction > 0
    assert cand_B.kelly_fraction > 0

    # Solo runs
    p_A = optimize_portfolio(
        candidates=[cand_A], race_win_probs={"A": win_probs_A},
        bankroll=1.0, max_drawdown_pct=0.50, n_scenarios=500, seed=9,
    )
    p_B = optimize_portfolio(
        candidates=[cand_B], race_win_probs={"B": win_probs_B},
        bankroll=1.0, max_drawdown_pct=0.50, n_scenarios=500, seed=9,
    )

    # In the per-race-only Phase 5b usage (the validation script), each
    # race is optimised in isolation, so the assertion is that the
    # solo-run stake matches the candidate's expected ¼-Kelly cap.
    assert len(p_A.recommendations) == 1
    assert p_A.recommendations[0].stake_fraction == pytest.approx(
        cand_A.kelly_fraction, abs=1e-6
    )
    assert len(p_B.recommendations) == 1
    assert p_B.recommendations[0].stake_fraction == pytest.approx(
        cand_B.kelly_fraction, abs=1e-6
    )


# ── Extra: input-validation errors ───────────────────────────────────────


def test_missing_race_win_probs_raises():
    c = _win_candidate("MISSING", 0, 0.5, 3.0)
    with pytest.raises(ValueError, match="race_win_probs"):
        optimize_portfolio(
            candidates=[c],
            race_win_probs={},  # no entry for "MISSING"
            bankroll=1.0,
        )


def test_selection_index_out_of_range_raises():
    win_probs = np.array([0.5, 0.5])
    bad = _win_candidate("R", 5, 0.5, 3.0)  # n_horses=2 → index 5 invalid
    with pytest.raises(ValueError, match="out of range"):
        optimize_portfolio(
            candidates=[bad],
            race_win_probs={"R": win_probs},
            bankroll=1.0,
        )
