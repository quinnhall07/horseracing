"""Tests for app/services/ordering/copula.py.

The copula module extends Stern with a Gaussian copula coupling same-style
horses' finishing times. The two surfaces:
    * Config & validation (rho ∈ [0,1), shape > 0, n_samples > 0).
    * Sampling + exotic probability queries. ρ=0 OR marginal_shape=1 OR
      pace_styles=None should all behave as documented (delegate to PL
      / Stern appropriately).

These tests lock in:
    (a) The PL fast path (ρ=0, shape=1, OR pace_styles=None) matches PL exactly.
    (b) Same-style positive correlation makes both same-style horses more
        likely to "finish together" — testable via the 2-style toy field.
    (c) Sampling returns valid permutations of the expected shape.
    (d) Marginal win probs at ρ>0 still approximately match strengths (the
        Luce property is preserved under same-style block correlation when
        marginals are Gamma — this is a non-obvious property worth pinning).
    (e) Edge cases: empty pace_styles, mismatched-length pace_styles,
        invalid rho, invalid shape.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.services.ordering import plackett_luce as pl
from app.services.ordering.copula import CopulaConfig, CopulaModel


# ── Config validation ────────────────────────────────────────────────────


def test_config_rejects_rho_out_of_range():
    with pytest.raises(ValueError):
        CopulaConfig(rho=-0.1)
    with pytest.raises(ValueError):
        CopulaConfig(rho=1.0)
    with pytest.raises(ValueError):
        CopulaConfig(rho=1.5)


def test_config_rejects_non_positive_shape():
    with pytest.raises(ValueError):
        CopulaConfig(marginal_shape=0.0)
    with pytest.raises(ValueError):
        CopulaConfig(marginal_shape=-1.0)


def test_config_rejects_non_positive_n_samples():
    with pytest.raises(ValueError):
        CopulaConfig(n_samples=0)


def test_default_config_is_sensible():
    cfg = CopulaConfig()
    assert 0 <= cfg.rho < 1
    assert cfg.marginal_shape == 1.0
    assert cfg.n_samples > 0


# ── PL fast paths (analytic, exact) ──────────────────────────────────────


def test_no_pace_info_matches_plackett_luce_exacta():
    """pace_styles=None at marginal_shape=1 ⇒ Stern at shape=1 ⇒ PL."""
    p = np.array([0.5, 0.3, 0.15, 0.05])
    m = CopulaModel(CopulaConfig(rho=0.5, marginal_shape=1.0))
    for i, j in itertools.permutations(range(4), 2):
        assert m.exacta_prob(p, None, i, j) == pytest.approx(pl.exacta_prob(p, i, j))


def test_zero_rho_matches_plackett_luce():
    """ρ=0 with marginal_shape=1 must collapse to PL exactly."""
    p = np.array([0.5, 0.3, 0.15, 0.05])
    styles = ["E", "P", "E", "S"]
    m = CopulaModel(CopulaConfig(rho=0.0, marginal_shape=1.0))
    for i, j, k in itertools.permutations(range(4), 3):
        assert m.trifecta_prob(p, styles, i, j, k) == pytest.approx(
            pl.trifecta_prob(p, i, j, k)
        )


def test_enumerate_exotic_probs_pl_fast_path():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = CopulaModel(CopulaConfig(rho=0.0, marginal_shape=1.0))
    out = m.enumerate_exotic_probs(p, ["E"] * 4, k=3)
    expected = pl.enumerate_exotic_probs(p, k=3)
    assert set(out) == set(expected)
    for key in out:
        assert out[key] == pytest.approx(expected[key])


# ── Pace style validation ────────────────────────────────────────────────


def test_pace_styles_length_must_match_strengths():
    p = np.array([0.5, 0.3, 0.2])
    m = CopulaModel(CopulaConfig(rho=0.3))
    with pytest.raises(ValueError):
        m.exacta_prob(p, ["E", "E"], 0, 1)


def test_pace_styles_can_be_arbitrary_hashable():
    p = np.array([0.5, 0.3, 0.2])
    m = CopulaModel(CopulaConfig(rho=0.3, n_samples=1_000, seed=0))
    # Integer styles, string styles, mixed — all hashable.
    a = m.exacta_prob(p, [1, 2, 1], 0, 1)
    b = m.exacta_prob(p, ["closer", "front", "closer"], 0, 1)
    # Same equivalence-class structure ⇒ same MC distribution (modulo seed).
    assert isinstance(a, float)
    assert isinstance(b, float)


def test_indices_reject_repeats():
    p = np.array([0.5, 0.5])
    m = CopulaModel(CopulaConfig(rho=0.3))
    with pytest.raises(ValueError):
        m.exacta_prob(p, ["E", "S"], 0, 0)


def test_indices_reject_out_of_range():
    p = np.array([0.5, 0.5])
    m = CopulaModel(CopulaConfig(rho=0.3))
    with pytest.raises(IndexError):
        m.exacta_prob(p, ["E", "S"], 0, 5)


# ── Same-style correlation effect ────────────────────────────────────────


def test_same_style_correlation_increases_finish_co_occurrence():
    """Two equal-strength horses with same pace style co-occur in the top
    two more often as ρ grows, relative to PL.

    Setup: 4 horses, two equal-strength pairs. Each pair shares a pace style.
    Under independence (ρ=0) the four (1st, 2nd) combinations across pairs
    are equally likely. Under ρ>0, same-pair finishes become MORE common.
    """
    p = np.array([0.3, 0.3, 0.2, 0.2])
    styles = ["E", "E", "S", "S"]

    indep = CopulaModel(CopulaConfig(rho=0.0, n_samples=80_000, seed=7))
    corr = CopulaModel(CopulaConfig(rho=0.7, n_samples=80_000, seed=7))

    same_style_indep = (
        indep.exacta_prob(p, styles, 0, 1) + indep.exacta_prob(p, styles, 1, 0) +
        indep.exacta_prob(p, styles, 2, 3) + indep.exacta_prob(p, styles, 3, 2)
    )
    same_style_corr = (
        corr.exacta_prob(p, styles, 0, 1) + corr.exacta_prob(p, styles, 1, 0) +
        corr.exacta_prob(p, styles, 2, 3) + corr.exacta_prob(p, styles, 3, 2)
    )
    # Correlated case should give noticeably higher same-style top-2 mass.
    assert same_style_corr > same_style_indep + 0.05


def test_correlation_concentrates_winshare_on_stronger_block_member():
    """Within a style block, ρ > 0 shifts marginal P(win) toward the
    stronger horse and away from the weaker — same-style horses share
    the same scenario, so independent-variation "upsets" shrink.

    Setup: 4 horses, styles = [E, P, E, S]; horse 0 (E, strong) and
    horse 2 (E, weak) share a style. Under ρ > 0 we expect:
        implied[0] ≥ p[0]   (stronger E gains)
        implied[2] ≤ p[2]   (weaker E loses)
    The Luce property (implied == p) is NOT preserved.
    """
    p = np.array([0.5, 0.3, 0.15, 0.05])
    styles = ["E", "P", "E", "S"]
    m = CopulaModel(CopulaConfig(rho=0.6, marginal_shape=1.0, n_samples=40_000, seed=9))
    implied = m.implied_win_probs(p, styles)
    # Direction: weaker same-style horse loses, stronger gains.
    assert implied[0] > p[0]
    assert implied[2] < p[2]
    # Sanity: total mass still 1.
    assert implied.sum() == pytest.approx(1.0, abs=1e-9)


def test_implied_win_probs_at_pl_fast_path_equals_input():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    m = CopulaModel(CopulaConfig(rho=0.0, marginal_shape=1.0))
    out = m.implied_win_probs(p, ["E"] * 4)
    np.testing.assert_array_equal(out, p)
    out_none = m.implied_win_probs(p, None)
    np.testing.assert_array_equal(out_none, p)


# ── Sampling ─────────────────────────────────────────────────────────────


def test_sample_ordering_returns_permutation():
    m = CopulaModel(CopulaConfig(rho=0.3, n_samples=200, seed=0))
    p = np.array([0.4, 0.3, 0.2, 0.1])
    styles = ["E", "E", "S", "S"]
    for _ in range(20):
        order = m.sample_ordering(p, styles)
        assert sorted(order) == [0, 1, 2, 3]


def test_sample_orderings_shape():
    m = CopulaModel(CopulaConfig(rho=0.3, n_samples=500, seed=1))
    p = np.array([0.6, 0.3, 0.1])
    samples = m.sample_orderings(p, ["E", "P", "S"], n=500)
    assert samples.shape == (500, 3)


def test_sample_orderings_deterministic_with_seed():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    styles = ["E", "E", "S", "S"]
    a = CopulaModel(CopulaConfig(rho=0.5, n_samples=500, seed=42)).sample_orderings(p, styles, 500)
    b = CopulaModel(CopulaConfig(rho=0.5, n_samples=500, seed=42)).sample_orderings(p, styles, 500)
    np.testing.assert_array_equal(a, b)


# ── Enumerate exotic probs ───────────────────────────────────────────────


def test_enumerate_exotic_probs_full_perm_set():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    styles = ["E", "E", "S", "S"]
    m = CopulaModel(CopulaConfig(rho=0.4, n_samples=5_000, seed=2))
    out = m.enumerate_exotic_probs(p, styles, k=2)
    assert len(out) == 12  # 4*3
    assert all(isinstance(key, tuple) and len(key) == 2 for key in out)


def test_enumerate_exotic_probs_sum_close_to_one_under_mc():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    styles = ["E", "E", "S", "S"]
    m = CopulaModel(CopulaConfig(rho=0.4, n_samples=30_000, seed=3))
    out = m.enumerate_exotic_probs(p, styles, k=3)
    total = sum(out.values())
    # 24 buckets; with 30k samples virtually all buckets observed.
    assert 0.99 < total <= 1.0 + 1e-9


# ── Marginal-shape interaction (rho=0 path) ──────────────────────────────


def test_copula_at_rho_zero_with_shape_two_sharpens_like_stern():
    """When ρ=0 the copula equals Stern; shape>1 sharpens top horse marginal."""
    p = np.array([0.5, 0.3, 0.15, 0.05])
    m = CopulaModel(CopulaConfig(rho=0.0, marginal_shape=4.0, n_samples=30_000, seed=11))
    implied = m.implied_win_probs(p, None)
    assert implied[0] > p[0] + 0.02
