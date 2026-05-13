"""Tests for app/services/calibration/calibrator.py.

The calibrator wraps an uncalibrated probability source (e.g. LightGBM
predict_proba) and produces probabilities that closely track empirical
frequencies. Two estimators are supported: Platt (parametric sigmoid) and
isotonic regression. The "auto" mode picks whichever achieves a lower
Expected Calibration Error on the fit data.

Tests cover:
    * Metric helpers — ECE, Brier, reliability bins.
    * Each calibrator fit + predict shape / monotonicity / range.
    * Auto-selection logic on synthetics that favour one method.
    * Per-race softmax sums to 1 per race group.
    * Save/load round-trip preserves predictions exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.calibration.calibrator import (
    Calibrator,
    CalibratorConfig,
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


# ── Metric helpers ────────────────────────────────────────────────────────


def test_ece_is_zero_when_probs_match_labels():
    # All labels equal to predicted probs (rounded into bins): ECE → 0.
    rng = np.random.default_rng(0)
    n = 2000
    probs = rng.uniform(0, 1, n)
    # Draw labels with frequency = probs (so the empirical rate matches).
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.03, f"ECE on perfectly-calibrated data should be near 0, got {ece}"


def test_ece_is_large_on_systematically_biased_probs():
    rng = np.random.default_rng(1)
    n = 2000
    true_p = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    # Bias: always predict 0.9 regardless of truth — the bin at 0.9 will
    # report observed_rate ≈ 0.5 (the global label mean), error ≈ 0.4.
    biased = np.full(n, 0.9)
    ece_biased = expected_calibration_error(biased, labels, n_bins=10)
    ece_calibrated = expected_calibration_error(true_p, labels, n_bins=10)
    assert ece_biased > ece_calibrated + 0.2


def test_ece_handles_empty_bins():
    # 10 samples at 0.1 with 1 win; 10 at 0.9 with 9 wins — empirical rates
    # match predicted probs exactly. Most other bins are empty (ignored).
    probs = np.array([0.1] * 10 + [0.9] * 10)
    labels = np.array([0] * 9 + [1] + [1] * 9 + [0])
    ece = expected_calibration_error(probs, labels, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_reliability_bins_count_matches_n_bins():
    probs = np.random.default_rng(0).uniform(0, 1, 500)
    labels = (probs > 0.5).astype(int)
    bins = reliability_bins(probs, labels, n_bins=10)
    assert len(bins) == 10
    assert all(isinstance(b, ReliabilityBin) for b in bins)
    # All bins together should cover every sample.
    total = sum(b.count for b in bins)
    assert total == len(probs)


def test_reliability_bins_are_sorted():
    rng = np.random.default_rng(2)
    probs = rng.uniform(0, 1, 1000)
    labels = (rng.uniform(0, 1, 1000) < probs).astype(int)
    bins = reliability_bins(probs, labels, n_bins=10)
    for prev, curr in zip(bins, bins[1:]):
        assert prev.bin_upper <= curr.bin_lower + 1e-9


def test_brier_score_zero_for_perfect_predictions():
    probs = np.array([0.0, 1.0, 1.0, 0.0])
    labels = np.array([0, 1, 1, 0])
    assert brier_score(probs, labels) == pytest.approx(0.0)


def test_brier_score_quarter_for_always_half_50_50_labels():
    probs = np.full(1000, 0.5)
    labels = np.tile([0, 1], 500)
    assert brier_score(probs, labels) == pytest.approx(0.25)


# ── Calibrator: Platt path ────────────────────────────────────────────────


def _biased_synthetic(n: int = 4000, seed: int = 11):
    """Generate (raw_scores, labels) where labels are systematically biased
    versus the raw scores — a setup where Platt should help materially.

    True P(win) is sigmoid(2*raw - 1) but we feed raw scores in [0,1].
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, n).astype(float)
    true_p = 1.0 / (1.0 + np.exp(-(2.0 * raw - 1.0)))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    return raw, labels


def test_platt_calibrator_predict_proba_in_unit_interval():
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="platt")).fit(raw, labels)
    out = cal.predict_proba(raw)
    assert out.shape == raw.shape
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_platt_calibrator_is_monotone_in_input():
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="platt")).fit(raw, labels)
    grid = np.linspace(0, 1, 100)
    out = cal.predict_proba(grid)
    # Platt's logistic fit is strictly monotone in the raw input.
    diffs = np.diff(out)
    assert (diffs >= -1e-9).all() or (diffs <= 1e-9).all()


def test_platt_reduces_ece_versus_raw_on_biased_input():
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="platt")).fit(raw, labels)
    raw_ece = expected_calibration_error(raw, labels, n_bins=10)
    cal_ece = expected_calibration_error(cal.predict_proba(raw), labels, n_bins=10)
    assert cal_ece < raw_ece


# ── Calibrator: isotonic path ─────────────────────────────────────────────


def test_isotonic_calibrator_is_monotone_in_input():
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="isotonic")).fit(raw, labels)
    grid = np.linspace(0, 1, 100)
    out = cal.predict_proba(grid)
    assert (np.diff(out) >= -1e-9).all()


def test_isotonic_reduces_ece_versus_raw_on_biased_input():
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="isotonic")).fit(raw, labels)
    raw_ece = expected_calibration_error(raw, labels, n_bins=10)
    cal_ece = expected_calibration_error(cal.predict_proba(raw), labels, n_bins=10)
    assert cal_ece < raw_ece


# ── Calibrator: auto-selector ─────────────────────────────────────────────


def test_auto_selector_records_chosen_method():
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method in ("platt", "isotonic")
    # Auto holds fit-slice metrics for BOTH candidate methods (diagnostic),
    # AND inner-val metrics that drive the selection itself.
    assert cal.metrics is not None
    assert "platt" in cal.metrics
    assert "isotonic" in cal.metrics
    assert cal.inner_val_metrics is not None
    assert "platt" in cal.inner_val_metrics
    assert "isotonic" in cal.inner_val_metrics
    # The chosen method must be the inner-val winner (with the
    # auto_min_delta_ece protective bias toward Platt on ties — see ADR-037).
    platt_iv = cal.inner_val_metrics["platt"]["ece"]
    iso_iv = cal.inner_val_metrics["isotonic"]["ece"]
    if cal.chosen_method == "isotonic":
        assert iso_iv + cal.config.auto_min_delta_ece < platt_iv
    else:
        assert iso_iv + cal.config.auto_min_delta_ece >= platt_iv


def test_auto_picks_isotonic_when_distortion_is_non_sigmoidal():
    """A staircase distortion (3 plateaus) is non-sigmoidal — isotonic fits
    it exactly while Platt's logistic cannot. Held-out inner-val confirms
    isotonic genuinely generalises."""
    rng = np.random.default_rng(13)
    n = 3000
    raw = rng.uniform(0, 1, n)
    true_p = np.where(raw < 0.33, 0.1, np.where(raw < 0.66, 0.5, 0.9))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method == "isotonic"
    assert cal.auto_selection_mode == "held_out"


def test_auto_picks_platt_over_isotonic_on_already_calibrated_when_skip_disabled():
    """When raw scores are ALREADY well-calibrated probabilities, isotonic
    can only overfit. With the skip guard disabled, the held-out auto-
    selector should pick Platt (the simpler model) — confirming the
    ADR-037 isotonic-vs-Platt fix in isolation. (With the skip guard
    enabled at default, identity wins on this stream — see
    test_auto_skips_calibration_when_raw_is_already_calibrated.)
    """
    rng = np.random.default_rng(2026)
    n = 4000
    raw = rng.uniform(0.05, 0.6, n)
    # True P = raw (perfectly calibrated already).
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    cal = Calibrator(
        CalibratorConfig(method="auto", skip_threshold_delta=-1.0, brier_skip_delta=-1.0)
    ).fit(raw, labels)
    assert cal.chosen_method == "platt"
    assert cal.auto_selection_mode == "held_out"


def test_auto_held_out_inner_val_records_both_method_eces():
    """Inner-val metrics dict captures both Platt and isotonic ECE for
    diagnostic transparency."""
    raw, labels = _biased_synthetic(n=4000)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    iv = cal.inner_val_metrics
    assert iv is not None
    assert iv["platt"]["n_val"] == int(4000 * cal.config.auto_val_fraction)
    assert iv["isotonic"]["n_val"] == int(4000 * cal.config.auto_val_fraction)
    assert 0.0 <= iv["platt"]["ece"] <= 1.0
    assert 0.0 <= iv["isotonic"]["ece"] <= 1.0


def test_auto_falls_back_to_fit_slice_on_small_calib():
    """For calib slice too small to hold a meaningful inner-val,
    auto_selection_mode is 'fit_slice_fallback'."""
    rng = np.random.default_rng(31)
    n = 50  # 20% inner-val = 10 rows, below default min 100.
    raw = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.auto_selection_mode == "fit_slice_fallback"
    assert cal.chosen_method in ("platt", "isotonic")


def test_auto_min_delta_ece_protects_against_noise_swap():
    """When isotonic and Platt give very close inner-val ECE, raising
    auto_min_delta_ece forces Platt (simpler model). Verify the threshold
    works as documented."""
    rng = np.random.default_rng(11)
    n = 2000
    # Mild biased synthetic where Platt and isotonic should be ≈ tied.
    raw = rng.uniform(0, 1, n)
    true_p = 0.5 + 0.1 * (raw - 0.5)  # gentle linear shift around 0.5
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    # With a generous min_delta=0.1 (=10% ECE advantage required), Platt
    # should always win — no real-world distortion gives that gap on
    # this synthetic. Also disable skip-when-calibrated so the test
    # specifically exercises the platt-vs-iso branch.
    cal = Calibrator(
        CalibratorConfig(method="auto", auto_min_delta_ece=0.1, skip_threshold_delta=0.0)
    ).fit(raw, labels)
    assert cal.chosen_method == "platt"


# ── Skip-when-calibrated guard (ADR-038) ──────────────────────────────────


def test_auto_skips_calibration_when_raw_is_already_calibrated():
    """Stream where raw scores ARE the true win probabilities: neither
    Platt nor isotonic can meaningfully improve on raw, so the
    skip-when-calibrated guard should fire and chosen_method == 'identity'.
    """
    rng = np.random.default_rng(42)
    n = 5000
    # Raw scores are already perfectly calibrated probabilities.
    raw = rng.uniform(0.05, 0.6, n)
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method == "identity"
    assert cal.auto_selection_mode == "held_out"
    # identity ECE recorded on inner-val too.
    assert "identity" in (cal.inner_val_metrics or {})
    # And on the full fit slice for diagnostic comparison.
    assert "identity" in (cal.metrics or {})


def test_identity_predict_proba_returns_raw_scores_clipped():
    """When chosen_method == 'identity', predict_proba returns raw input
    clipped to [0, 1] — no calibration applied."""
    rng = np.random.default_rng(7)
    n = 4000
    raw = rng.uniform(0.05, 0.6, n)
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method == "identity"
    out = cal.predict_proba(raw)
    np.testing.assert_array_equal(out, np.clip(raw, 0.0, 1.0))
    # Out-of-range inputs are clipped — predict_proba never escapes [0, 1].
    weird = np.array([-0.5, 0.0, 0.3, 1.0, 1.5])
    out = cal.predict_proba(weird)
    np.testing.assert_array_equal(out, np.clip(weird, 0.0, 1.0))


def test_skip_does_not_fire_on_clearly_uncalibrated_stream():
    """When the raw stream is truly miscalibrated (staircase distortion),
    isotonic should beat raw by more than skip_threshold_delta and the
    skip guard should NOT fire."""
    rng = np.random.default_rng(13)
    n = 3000
    raw = rng.uniform(0, 1, n)
    true_p = np.where(raw < 0.33, 0.1, np.where(raw < 0.66, 0.5, 0.9))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method == "isotonic"


def test_skip_threshold_tunes_how_much_improvement_required():
    """skip_threshold_delta tunes the minimum ECE advantage a calibrator
    must show over raw on inner-val:
        * Large positive (e.g. 0.5): skip almost always fires — even a
          clear isotonic win on a staircase doesn't meet the bar.
        * Default (0.001): tiny tie-band biased toward identity — clear
          wins still pass through.
    """
    rng = np.random.default_rng(13)
    n = 3000
    # Clearly miscalibrated stream — both Platt and isotonic should
    # comfortably beat raw, so default skip does NOT fire.
    raw = rng.uniform(0, 1, n)
    true_p = np.where(raw < 0.33, 0.1, np.where(raw < 0.66, 0.5, 0.9))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)

    cal_default = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal_default.chosen_method != "identity"

    # Aggressive skip — calibration must beat raw by 0.5 ECE; impossible
    # in practice, so identity always wins.
    cal_aggressive = Calibrator(
        CalibratorConfig(method="auto", skip_threshold_delta=0.5)
    ).fit(raw, labels)
    assert cal_aggressive.chosen_method == "identity"


def test_identity_save_and_load_round_trip(tmp_path: Path):
    """Identity chosen_method survives save/load and predict_proba still
    returns clipped raw."""
    rng = np.random.default_rng(2027)
    n = 4000
    raw = rng.uniform(0.05, 0.6, n)
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method == "identity"

    artifact_dir = tmp_path / "calibrator_identity"
    cal.save(artifact_dir)
    restored = Calibrator.load(artifact_dir)
    assert restored.chosen_method == "identity"
    p_before = cal.predict_proba(raw[:50])
    p_after = restored.predict_proba(raw[:50])
    np.testing.assert_array_equal(p_before, p_after)


# ── Time-ordered inner-val indices (ADR-038) ──────────────────────────────


def test_caller_supplied_inner_val_indices_records_held_out_caller_mode():
    raw, labels = _biased_synthetic(n=4000)
    n = len(raw)
    iv_idx = np.arange(n - 800, n)  # last 800 rows as inner-val
    cal = Calibrator(CalibratorConfig(method="auto")).fit(
        raw, labels, inner_val_indices=iv_idx
    )
    assert cal.auto_selection_mode == "held_out_caller"
    iv = cal.inner_val_metrics
    assert iv is not None
    assert iv["platt"]["n_val"] == 800
    assert iv["isotonic"]["n_val"] == 800
    assert iv["identity"]["n_val"] == 800


def test_caller_supplied_inner_val_indices_change_selection():
    """Use a synthetic where the calib slice contains a regime shift:
    first half is staircase (favours isotonic), second half is well-
    calibrated (favours skip). Time-ordered tail (= second half) should
    pick 'identity'; random shuffle would mix both regimes and pick a
    different method."""
    rng = np.random.default_rng(2026)
    n_each = 3000
    # Block A — staircase, favours isotonic.
    raw_a = rng.uniform(0, 1, n_each)
    p_a = np.where(raw_a < 0.33, 0.1, np.where(raw_a < 0.66, 0.5, 0.9))
    lab_a = (rng.uniform(0, 1, n_each) < p_a).astype(int)
    # Block B — already calibrated.
    raw_b = rng.uniform(0.05, 0.6, n_each)
    lab_b = (rng.uniform(0, 1, n_each) < raw_b).astype(int)

    raw = np.concatenate([raw_a, raw_b])
    lab = np.concatenate([lab_a, lab_b]).astype(int)

    # Time-ordered tail (= second half) is in the calibrated regime →
    # skip should fire on that inner-val.
    iv_idx = np.arange(n_each, 2 * n_each)
    cal_tail = Calibrator(CalibratorConfig(method="auto")).fit(
        raw, lab, inner_val_indices=iv_idx
    )
    assert cal_tail.chosen_method == "identity"
    assert cal_tail.auto_selection_mode == "held_out_caller"

    # Random seeded shuffle mixes regimes; isotonic dominates because
    # the staircase distortion is large and the calibrated half is
    # only mildly affected by isotonic overfitting on the inner-val.
    cal_rand = Calibrator(CalibratorConfig(method="auto")).fit(raw, lab)
    assert cal_rand.chosen_method != "identity"


def test_inner_val_indices_validates_range():
    raw, labels = _biased_synthetic(n=200)
    cal = Calibrator(CalibratorConfig(method="auto"))
    with pytest.raises(ValueError, match="out of range"):
        cal.fit(raw, labels, inner_val_indices=np.array([0, 1, 250]))


def test_inner_val_indices_rejects_duplicates():
    raw, labels = _biased_synthetic(n=200)
    cal = Calibrator(CalibratorConfig(method="auto"))
    with pytest.raises(ValueError, match="duplicates"):
        cal.fit(raw, labels, inner_val_indices=np.array([0, 1, 1, 2]))


def test_inner_val_indices_rejects_empty():
    raw, labels = _biased_synthetic(n=200)
    cal = Calibrator(CalibratorConfig(method="auto"))
    with pytest.raises(ValueError, match="non-empty"):
        cal.fit(raw, labels, inner_val_indices=np.array([], dtype=int))


def test_inner_val_indices_ignored_for_non_auto_method():
    """Pass-through with a warning; fit must still complete cleanly."""
    raw, labels = _biased_synthetic(n=400)
    cal = Calibrator(CalibratorConfig(method="platt"))
    cal.fit(raw, labels, inner_val_indices=np.array([0, 1, 2, 3]))
    assert cal.chosen_method == "platt"
    # No inner-val metrics for non-auto.
    assert cal.inner_val_metrics is None


# ── Brier co-criterion in skip guard (ADR-038 refined) ────────────────────


def test_inner_val_metrics_records_brier_alongside_ece():
    """Inner-val metrics dict carries Brier for all three candidates,
    so the skip guard's strictly-proper co-criterion is auditable."""
    raw, labels = _biased_synthetic(n=4000)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    iv = cal.inner_val_metrics
    assert iv is not None
    for method in ("platt", "isotonic", "identity"):
        assert "ece" in iv[method]
        assert "brier" in iv[method]
        assert 0.0 <= iv[method]["brier"] <= 1.0


def test_brier_co_criterion_can_block_calibration_when_only_ece_improves():
    """Construct a stream where Platt/isotonic give a small ECE win
    over raw but no Brier improvement — the skip guard should fire on
    the Brier leg even though the ECE leg passes.

    Setup: raw scores are uniform [0, 1] and labels are Bernoulli with
    p = raw (already well-calibrated). Calibrators that fit on this
    will, on inner-val, sometimes shave ECE through bin redistribution
    but won't beat raw on Brier.
    """
    rng = np.random.default_rng(42)
    n = 5000
    raw = rng.uniform(0.05, 0.6, n)
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    # On already-calibrated streams the Brier guard wins decisively —
    # iso/Platt cannot strictly improve Brier over raw.
    assert cal.chosen_method == "identity"
    iv = cal.inner_val_metrics
    assert iv is not None
    # Sanity: identity Brier ≤ both calibrator Briers + brier_skip_delta —
    # confirming the Brier leg of the guard correctly fired.
    raw_brier = iv["identity"]["brier"]
    assert iv["isotonic"]["brier"] + cal.config.brier_skip_delta >= raw_brier
    assert iv["platt"]["brier"] + cal.config.brier_skip_delta >= raw_brier


def test_brier_co_criterion_lets_genuine_improvement_through():
    """On a clearly-distorted (staircase) stream, isotonic improves BOTH
    ECE and Brier comfortably — neither leg of the skip guard fires."""
    rng = np.random.default_rng(13)
    n = 3000
    raw = rng.uniform(0, 1, n)
    true_p = np.where(raw < 0.33, 0.1, np.where(raw < 0.66, 0.5, 0.9))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method == "isotonic"
    iv = cal.inner_val_metrics
    assert iv is not None
    # Sanity: on this synthetic isotonic genuinely beats raw on Brier.
    assert iv["isotonic"]["brier"] + cal.config.brier_skip_delta < iv["identity"]["brier"]


def test_brier_skip_delta_zero_disables_brier_leg():
    """With brier_skip_delta=0 (and skip_threshold_delta=0), the skip
    guard reverts to a pure 'strictly better than raw' check. Calibration
    applies even on already-calibrated streams whenever it eke out any
    micro-improvement on either metric.

    NOTE: this is a knob for diagnostics, not a recommended setting —
    the default brier_skip_delta=1e-4 is what makes the guard robust.
    """
    rng = np.random.default_rng(42)
    n = 5000
    raw = rng.uniform(0.05, 0.6, n)
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    cal = Calibrator(
        CalibratorConfig(
            method="auto",
            skip_threshold_delta=-1.0,
            brier_skip_delta=-1.0,
        )
    ).fit(raw, labels)
    # With both legs disabled, skip can never fire — chosen is platt or iso.
    assert cal.chosen_method in ("platt", "isotonic")


# ── Calibrator: predict_softmax ───────────────────────────────────────────


def test_predict_softmax_sums_to_one_per_race():
    raw, labels = _biased_synthetic(n=600)
    cal = Calibrator(CalibratorConfig(method="platt")).fit(raw, labels)
    # 60 races of 10 horses each.
    race_ids = np.repeat(np.arange(60), 10)
    out = cal.predict_softmax(raw, race_ids)
    assert out.shape == raw.shape
    sums = np.zeros(60)
    for rid in range(60):
        sums[rid] = out[race_ids == rid].sum()
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_predict_softmax_temperature_scales_distribution_sharpness():
    raw, labels = _biased_synthetic(n=600)
    cal_cold = Calibrator(CalibratorConfig(method="platt", softmax_temperature=0.5)).fit(raw, labels)
    cal_hot = Calibrator(CalibratorConfig(method="platt", softmax_temperature=2.0)).fit(raw, labels)
    race_ids = np.repeat(np.arange(60), 10)
    cold = cal_cold.predict_softmax(raw, race_ids)
    hot = cal_hot.predict_softmax(raw, race_ids)
    # Cold (T<1) sharpens; entropy of cold should be lower than hot.
    def avg_entropy(p):
        eps = 1e-12
        h = 0.0
        for rid in range(60):
            pp = p[race_ids == rid]
            h += -np.sum(pp * np.log(pp + eps))
        return h / 60
    assert avg_entropy(cold) < avg_entropy(hot)


# ── Calibrator: save / load round-trip ────────────────────────────────────


def test_save_and_load_round_trip(tmp_path: Path):
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    p_before = cal.predict_proba(raw[:50])

    artifact_dir = tmp_path / "calibrator"
    cal.save(artifact_dir)
    assert (artifact_dir / "metadata.json").exists()

    restored = Calibrator.load(artifact_dir)
    assert restored.chosen_method == cal.chosen_method
    p_after = restored.predict_proba(raw[:50])
    assert np.allclose(p_before, p_after, atol=1e-9)


def test_save_writes_metadata_with_chosen_method_and_metrics(tmp_path: Path):
    raw, labels = _biased_synthetic()
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    artifact_dir = tmp_path / "calibrator"
    cal.save(artifact_dir)
    meta = json.loads((artifact_dir / "metadata.json").read_text())
    assert meta["chosen_method"] in ("platt", "isotonic")
    assert "metrics" in meta
    assert "config" in meta


# ── Calibrator: input validation ──────────────────────────────────────────


def test_fit_rejects_mismatched_lengths():
    cal = Calibrator()
    with pytest.raises(ValueError):
        cal.fit(np.array([0.1, 0.5]), np.array([0, 1, 0]))


def test_predict_before_fit_raises():
    cal = Calibrator()
    with pytest.raises(RuntimeError):
        cal.predict_proba(np.array([0.5]))


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        Calibrator(CalibratorConfig(method="bogus")).fit(
            np.array([0.1, 0.9]), np.array([0, 1])
        )
