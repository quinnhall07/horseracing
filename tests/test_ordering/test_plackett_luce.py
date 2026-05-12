"""Tests for app/services/ordering/plackett_luce.py.

The Plackett-Luce module has two surfaces:
    * Analytical exotic enumeration — given win probabilities for a race,
      compute P(exacta), P(trifecta), P(superfecta), or all top-k orderings.
    * MLE strength fitting — given a corpus of finishing orderings, fit
      per-horse strength parameters via maximum likelihood (scipy).

CLAUDE.md §1 and ADR-001 prohibit Harville; PL is the minimum standard.
These tests lock in PL's defining properties so a future refactor cannot
silently slip back to Harville.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.services.ordering.plackett_luce import (
    enumerate_exotic_probs,
    exacta_prob,
    fit_plackett_luce_mle,
    sample_ordering,
    superfecta_prob,
    trifecta_prob,
)


# ── Analytical: 2-horse field ─────────────────────────────────────────────


def test_exacta_two_horse_sums_to_one():
    p = np.array([0.7, 0.3])
    p12 = exacta_prob(p, 0, 1)
    p21 = exacta_prob(p, 1, 0)
    assert p12 + p21 == pytest.approx(1.0)


def test_exacta_two_horse_marginal_matches_win_prob():
    p = np.array([0.7, 0.3])
    # P(horse 0 wins exacta with anything) should equal p[0].
    assert exacta_prob(p, 0, 1) == pytest.approx(p[0])
    assert exacta_prob(p, 1, 0) == pytest.approx(p[1])


# ── Analytical: 3-horse field ─────────────────────────────────────────────


def test_trifecta_three_horse_uniform_field():
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    probs = []
    for i, j, k in itertools.permutations(range(3), 3):
        probs.append(trifecta_prob(p, i, j, k))
    assert sum(probs) == pytest.approx(1.0)
    # All 6 orderings should have equal probability = 1/6.
    assert all(p_ == pytest.approx(1.0 / 6.0, abs=1e-9) for p_ in probs)


def test_trifecta_three_horse_skewed():
    p = np.array([0.6, 0.3, 0.1])
    # P(0,1,2) = 0.6 * (0.3 / 0.4) * (0.1 / 0.1) = 0.45.
    assert trifecta_prob(p, 0, 1, 2) == pytest.approx(0.45)
    # P(2,1,0) = 0.1 * (0.3 / 0.9) * (0.6 / 0.6) = 0.1/3 ≈ 0.0333.
    assert trifecta_prob(p, 2, 1, 0) == pytest.approx(1.0 / 30.0)
    # All 6 perms sum to 1.
    s = sum(
        trifecta_prob(p, i, j, k)
        for i, j, k in itertools.permutations(range(3), 3)
    )
    assert s == pytest.approx(1.0)


def test_exacta_marginal_recovers_win_prob():
    """Σ_{j≠i} P(exacta i, j) = p[i] for every horse i."""
    p = np.array([0.5, 0.3, 0.15, 0.05])
    for i in range(4):
        marginal = sum(exacta_prob(p, i, j) for j in range(4) if j != i)
        assert marginal == pytest.approx(p[i])


# ── Analytical: 4-horse superfecta ────────────────────────────────────────


def test_superfecta_uniform_field():
    p = np.full(4, 0.25)
    perms = list(itertools.permutations(range(4), 4))
    probs = [superfecta_prob(p, *perm) for perm in perms]
    # 24 equally likely orderings.
    assert all(pp == pytest.approx(1.0 / 24.0) for pp in probs)
    assert sum(probs) == pytest.approx(1.0)


def test_superfecta_extends_trifecta_consistently():
    """Σ_{l} P(superfecta i, j, k, l) = P(trifecta i, j, k)."""
    p = np.array([0.4, 0.3, 0.2, 0.1])
    tri = trifecta_prob(p, 0, 1, 2)
    summed = sum(superfecta_prob(p, 0, 1, 2, l) for l in range(4) if l not in (0, 1, 2))
    assert summed == pytest.approx(tri)


# ── enumerate_exotic_probs ────────────────────────────────────────────────


def test_enumerate_exactas_returns_all_pairs():
    p = np.array([0.5, 0.3, 0.2])
    out = enumerate_exotic_probs(p, k=2)
    # Expect 3 * 2 = 6 ordered pairs.
    assert len(out) == 6
    assert sum(out.values()) == pytest.approx(1.0)
    # Keys are tuples of indices.
    for key in out:
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert len(set(key)) == 2


def test_enumerate_trifectas_match_per_call_results():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    out = enumerate_exotic_probs(p, k=3)
    # 4 * 3 * 2 = 24 ordered triples.
    assert len(out) == 24
    assert sum(out.values()) == pytest.approx(1.0)
    for (i, j, k), v in out.items():
        assert v == pytest.approx(trifecta_prob(p, i, j, k))


def test_enumerate_k_equals_field_size_returns_full_permutations():
    p = np.array([0.5, 0.3, 0.2])
    out = enumerate_exotic_probs(p, k=3)
    assert len(out) == 6  # 3!


# ── Edge cases ────────────────────────────────────────────────────────────


def test_exacta_handles_zero_probability_horse():
    """Horse with p=0 should yield 0 in any position."""
    p = np.array([0.5, 0.5, 0.0])
    assert exacta_prob(p, 2, 0) == pytest.approx(0.0)
    assert exacta_prob(p, 0, 2) == pytest.approx(0.0)


def test_exacta_handles_certain_horse():
    """When one horse has p=1, P(exacta i, j) = p[j] iff i is the certain one."""
    p = np.array([1.0, 0.0, 0.0])
    assert exacta_prob(p, 0, 1) == pytest.approx(0.0)  # p[1] = 0
    assert exacta_prob(p, 1, 0) == pytest.approx(0.0)  # p[1] = 0


def test_invalid_probs_raises():
    # Probs must be non-negative and sum to ~1.
    with pytest.raises(ValueError):
        exacta_prob(np.array([0.5, 0.7]), 0, 1)
    with pytest.raises(ValueError):
        exacta_prob(np.array([-0.1, 1.1]), 0, 1)


def test_invalid_indices_raises():
    p = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        exacta_prob(p, 0, 0)  # i == j
    with pytest.raises(IndexError):
        exacta_prob(p, 0, 5)


# ── MLE fit ───────────────────────────────────────────────────────────────


def test_fit_mle_recovers_known_strengths():
    """Fit PL on orderings drawn from a known strength vector → recover ratios."""
    true_strengths = np.array([0.6, 0.3, 0.1])
    rng = np.random.default_rng(7)
    orderings = [sample_ordering(true_strengths, rng) for _ in range(5000)]
    fit = fit_plackett_luce_mle(orderings, n_items=3)
    # Strengths are normalized to sum=1.
    assert fit.strengths.sum() == pytest.approx(1.0, abs=1e-6)
    # Ratios should be in the right order and roughly proportional.
    assert fit.strengths[0] > fit.strengths[1] > fit.strengths[2]
    # Allow 15% relative error given finite sample.
    assert abs(fit.strengths[0] - true_strengths[0]) < 0.05
    assert abs(fit.strengths[1] - true_strengths[1]) < 0.05
    assert abs(fit.strengths[2] - true_strengths[2]) < 0.05


def test_fit_mle_returns_n_items_strengths():
    rng = np.random.default_rng(8)
    orderings = [sample_ordering(np.array([0.5, 0.3, 0.2]), rng) for _ in range(200)]
    fit = fit_plackett_luce_mle(orderings, n_items=3)
    assert fit.strengths.shape == (3,)
    assert (fit.strengths > 0).all()


def test_fit_mle_partial_orderings_supported():
    """The MLE handles orderings with only top-k positions (k < n_items)."""
    rng = np.random.default_rng(9)
    true_strengths = np.array([0.4, 0.3, 0.2, 0.1])
    full = [sample_ordering(true_strengths, rng) for _ in range(800)]
    top3 = [order[:3] for order in full]  # only top-3 known
    fit = fit_plackett_luce_mle(top3, n_items=4)
    # The dropped 4th-position horse should still have non-zero strength.
    assert (fit.strengths > 0).all()
    assert fit.strengths.sum() == pytest.approx(1.0, abs=1e-6)


def test_fit_mle_rejects_empty_orderings():
    with pytest.raises(ValueError):
        fit_plackett_luce_mle([], n_items=3)


def test_fit_mle_rejects_inconsistent_n_items():
    with pytest.raises(ValueError):
        # Ordering references item 5 but n_items=3.
        fit_plackett_luce_mle([[0, 1, 5]], n_items=3)


# ── Sample helper ─────────────────────────────────────────────────────────


def test_sample_ordering_returns_permutation():
    rng = np.random.default_rng(0)
    p = np.array([0.4, 0.3, 0.2, 0.1])
    for _ in range(50):
        order = sample_ordering(p, rng)
        assert len(order) == 4
        assert sorted(order) == [0, 1, 2, 3]


def test_sample_ordering_respects_strength_ordering_in_expectation():
    rng = np.random.default_rng(0)
    p = np.array([0.7, 0.2, 0.1])
    first_position = [sample_ordering(p, rng)[0] for _ in range(2000)]
    counts = np.bincount(first_position, minlength=3)
    rates = counts / counts.sum()
    assert rates[0] > rates[1] > rates[2]
    assert abs(rates[0] - 0.7) < 0.05
