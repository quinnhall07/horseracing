"""Tests for app/services/ordering/stern.py.

The Stern Gamma module has three surfaces:
    * Configuration & validation (shape > 0, strengths sum to 1, indices distinct).
    * Sampling + exotic probability queries (analytic at shape=1, Monte Carlo otherwise).
    * Shape MLE — grid search over a corpus of (ordering, strengths) pairs.

CLAUDE.md §1 / ADR-001 prohibit Harville. At shape=1 the Stern model is
mathematically Plackett-Luce, and the analytical path is shared. These tests
lock in:
    (a) Shape=1 produces identical results to plackett_luce.* — no MC drift.
    (b) Shape>1 sharpens, shape<1 flattens, both in the right direction.
    (c) Sampler returns valid permutations matching expected marginals.
    (d) infer_strengths recovers target win probs even when shape ≠ 1.
    (e) Shape grid-search picks the correct shape on synthetic data.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.services.ordering import plackett_luce as pl
from app.services.ordering.stern import (
    SternConfig,
    SternFit,
    SternModel,
    fit_stern_shape,
)


# ── Config validation ────────────────────────────────────────────────────


def test_config_rejects_non_positive_shape():
    with pytest.raises(ValueError):
        SternConfig(shape=0.0)
    with pytest.raises(ValueError):
        SternConfig(shape=-1.0)


def test_config_rejects_non_positive_n_samples():
    with pytest.raises(ValueError):
        SternConfig(n_samples=0)
    with pytest.raises(ValueError):
        SternConfig(n_samples=-100)


def test_default_config_is_shape_one_pl_equivalent():
    m = SternModel()
    assert m.shape == 1.0
    assert m.n_samples > 0


# ── Strength validation ──────────────────────────────────────────────────


def test_strengths_must_sum_to_one():
    m = SternModel(SternConfig(shape=1.0))
    with pytest.raises(ValueError):
        m.exacta_prob(np.array([0.4, 0.4]), 0, 1)


def test_strengths_reject_negative():
    m = SternModel(SternConfig(shape=1.0))
    with pytest.raises(ValueError):
        m.exacta_prob(np.array([-0.1, 1.1]), 0, 1)


def test_indices_reject_repeats():
    m = SternModel(SternConfig(shape=1.0))
    with pytest.raises(ValueError):
        m.exacta_prob(np.array([0.5, 0.5]), 0, 0)


def test_indices_reject_out_of_range():
    m = SternModel(SternConfig(shape=1.0))
    with pytest.raises(IndexError):
        m.exacta_prob(np.array([0.5, 0.5]), 0, 5)


# ── Shape = 1.0 → PL equivalence (analytical, exact) ─────────────────────


def test_shape_one_exacta_matches_plackett_luce():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    m = SternModel(SternConfig(shape=1.0))
    for i, j in itertools.permutations(range(4), 2):
        assert m.exacta_prob(p, i, j) == pytest.approx(pl.exacta_prob(p, i, j))


def test_shape_one_trifecta_matches_plackett_luce():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    m = SternModel(SternConfig(shape=1.0))
    for i, j, k in itertools.permutations(range(4), 3):
        assert m.trifecta_prob(p, i, j, k) == pytest.approx(pl.trifecta_prob(p, i, j, k))


def test_shape_one_superfecta_matches_plackett_luce():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = SternModel(SternConfig(shape=1.0))
    for perm in itertools.permutations(range(4), 4):
        assert m.superfecta_prob(p, *perm) == pytest.approx(pl.superfecta_prob(p, *perm))


def test_shape_one_enumerate_matches_plackett_luce():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = SternModel(SternConfig(shape=1.0))
    a = m.enumerate_exotic_probs(p, k=3)
    b = pl.enumerate_exotic_probs(p, k=3)
    assert set(a) == set(b)
    for key, val in a.items():
        assert val == pytest.approx(b[key])


# ── Shape != 1.0 → Monte Carlo behaviour ─────────────────────────────────


def test_shape_above_one_sharpens_win_probability_of_top_horse():
    """At shape > 1, P(top horse wins) > strengths[top]. At shape < 1, less."""
    p = np.array([0.5, 0.3, 0.15, 0.05])
    sharp = SternModel(SternConfig(shape=4.0, n_samples=40_000, seed=1))
    flat = SternModel(SternConfig(shape=0.25, n_samples=40_000, seed=1))
    sharp_marginals = sharp.implied_win_probs(p)
    flat_marginals = flat.implied_win_probs(p)
    assert sharp_marginals[0] > p[0] + 0.02
    assert flat_marginals[0] < p[0] - 0.02


def test_implied_win_probs_sum_to_one():
    m = SternModel(SternConfig(shape=2.0, n_samples=10_000, seed=2))
    p = np.array([0.5, 0.3, 0.15, 0.05])
    implied = m.implied_win_probs(p)
    assert implied.sum() == pytest.approx(1.0, abs=1e-9)


def test_implied_win_probs_at_shape_one_equals_input():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    m = SternModel(SternConfig(shape=1.0))
    out = m.implied_win_probs(p)
    np.testing.assert_array_equal(out, p)


# ── Sampling ─────────────────────────────────────────────────────────────


def test_sample_ordering_returns_permutation():
    m = SternModel(SternConfig(shape=2.0, seed=0))
    p = np.array([0.4, 0.3, 0.2, 0.1])
    for _ in range(20):
        order = m.sample_ordering(p)
        assert sorted(order) == [0, 1, 2, 3]


def test_sample_orderings_batch_shape_and_marginals():
    m = SternModel(SternConfig(shape=1.0, seed=0))
    p = np.array([0.6, 0.3, 0.1])
    samples = m.sample_orderings(p, n=10_000)
    assert samples.shape == (10_000, 3)
    # At shape=1 Luce property: winners ratio should match p.
    winners = samples[:, 0]
    rates = np.bincount(winners, minlength=3) / 10_000
    # 1.96 / √10_000 = 0.02 SE on a Bernoulli at p=0.5 → use 0.025 tolerance.
    np.testing.assert_allclose(rates, p, atol=0.025)


def test_sample_orderings_deterministic_with_seed():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    a = SternModel(SternConfig(shape=2.0, n_samples=500, seed=42)).sample_orderings(p, 500)
    b = SternModel(SternConfig(shape=2.0, n_samples=500, seed=42)).sample_orderings(p, 500)
    np.testing.assert_array_equal(a, b)


# ── Exotic prob MC behaviour ─────────────────────────────────────────────


def test_exacta_prob_close_to_pl_at_shape_near_one():
    """Continuity: at shape just above 1, exacta probs should be close to PL."""
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = SternModel(SternConfig(shape=1.05, n_samples=40_000, seed=7))
    for i, j in itertools.permutations(range(4), 2):
        ster = m.exacta_prob(p, i, j)
        analytic = pl.exacta_prob(p, i, j)
        # 40k samples + small shape delta → within 0.02 abs of analytic PL.
        assert abs(ster - analytic) < 0.02


def test_enumerate_exotic_probs_returns_full_permutation_set():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = SternModel(SternConfig(shape=2.0, n_samples=5_000, seed=3))
    out = m.enumerate_exotic_probs(p, k=2)
    # All 4*3 = 12 ordered pairs present.
    assert len(out) == 12
    assert all(isinstance(key, tuple) and len(key) == 2 for key in out)


def test_enumerate_exotic_probs_sum_close_to_one_under_mc():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = SternModel(SternConfig(shape=2.0, n_samples=20_000, seed=4))
    out = m.enumerate_exotic_probs(p, k=3)
    total = sum(out.values())
    # Total ≤ 1 by construction; with 20k samples on 24 buckets the
    # missing mass from un-observed perms is small (<1%).
    assert 0.99 < total <= 1.0 + 1e-9


# ── infer_strengths ──────────────────────────────────────────────────────


def test_infer_strengths_at_shape_one_returns_target():
    m = SternModel(SternConfig(shape=1.0))
    target = np.array([0.5, 0.3, 0.15, 0.05])
    out = m.infer_strengths(target)
    np.testing.assert_array_equal(out, target)


def test_infer_strengths_recovers_target_marginals_at_shape_two():
    """At shape=2, fixed-point iteration finds rates producing target win probs."""
    m = SternModel(SternConfig(shape=2.0, n_samples=20_000, seed=11))
    target = np.array([0.5, 0.3, 0.15, 0.05])
    rates = m.infer_strengths(target, max_iter=80, tol=1e-3)
    implied = m.implied_win_probs(rates)
    np.testing.assert_allclose(implied, target, atol=0.015)


def test_infer_strengths_normalised_to_one():
    m = SternModel(SternConfig(shape=1.5, n_samples=5_000, seed=12))
    target = np.array([0.6, 0.25, 0.1, 0.05])
    rates = m.infer_strengths(target, max_iter=30)
    assert rates.sum() == pytest.approx(1.0, abs=1e-9)


# ── Shape MLE (fit_stern_shape) ──────────────────────────────────────────


def _synthesise_corpus(true_shape: float, n_races: int, seed: int) -> list[tuple[list[int], np.ndarray]]:
    """Build a corpus of (ordering, strengths) from a known-shape Stern model."""
    rng = np.random.default_rng(seed)
    model = SternModel(SternConfig(shape=true_shape, seed=seed))
    corpus: list[tuple[list[int], np.ndarray]] = []
    for _ in range(n_races):
        # Random 4-horse strengths.
        raw = rng.uniform(0.05, 1.0, size=4)
        strengths = raw / raw.sum()
        order = model.sample_ordering(strengths, rng=rng)
        corpus.append((order, strengths))
    return corpus


def test_fit_stern_shape_recovers_known_shape_above_one():
    """Synthesise orderings from shape=2.5, recover via grid search."""
    corpus = _synthesise_corpus(true_shape=2.5, n_races=300, seed=21)
    grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    fit = fit_stern_shape(corpus, shape_grid=grid, n_samples=8_000, top_k=2, seed=22)
    # We expect the maximum to land within one grid step of true shape.
    assert abs(fit.shape - 2.5) <= 1.0
    assert fit.n_orderings == 300


def test_fit_stern_shape_recovers_known_shape_below_one():
    """Synthesise from shape=0.5, recover via grid search."""
    corpus = _synthesise_corpus(true_shape=0.5, n_races=300, seed=31)
    grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    fit = fit_stern_shape(corpus, shape_grid=grid, n_samples=8_000, top_k=2, seed=32)
    # Distinguish low-shape (high-variance) regime from high-shape regime.
    assert fit.shape <= 1.0


def test_fit_stern_shape_grid_default_runs():
    """Default grid path runs end-to-end with a small corpus."""
    corpus = _synthesise_corpus(true_shape=1.0, n_races=20, seed=41)
    fit = fit_stern_shape(corpus, n_samples=2_000, top_k=2, seed=42)
    assert isinstance(fit, SternFit)
    assert len(fit.shape_grid) == len(fit.ll_per_shape)


def test_fit_stern_shape_rejects_empty_corpus():
    with pytest.raises(ValueError):
        fit_stern_shape([], n_samples=1_000)


def test_fit_stern_shape_rejects_top_k_below_two():
    """Top-1 is invariant under shape — fitting is identifiable only at k≥2."""
    corpus = _synthesise_corpus(true_shape=1.0, n_races=5, seed=51)
    with pytest.raises(ValueError):
        fit_stern_shape(corpus, top_k=1)


def test_fit_stern_shape_rejects_non_positive_shape_grid_entry():
    corpus = _synthesise_corpus(true_shape=1.0, n_races=5, seed=52)
    with pytest.raises(ValueError):
        fit_stern_shape(corpus, shape_grid=[0.0, 1.0])
    with pytest.raises(ValueError):
        fit_stern_shape(corpus, shape_grid=[-1.0, 1.0])
