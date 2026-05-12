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
    # Auto must hold metrics for BOTH candidate methods.
    assert cal.metrics is not None
    assert "platt" in cal.metrics
    assert "isotonic" in cal.metrics
    # The chosen method's ECE must be the minimum.
    chosen_ece = cal.metrics[cal.chosen_method]["ece"]
    other_ece = cal.metrics["isotonic" if cal.chosen_method == "platt" else "platt"]["ece"]
    assert chosen_ece <= other_ece + 1e-9


def test_auto_picks_isotonic_when_distortion_is_non_sigmoidal():
    """A staircase distortion (3 plateaus) is non-sigmoidal — isotonic fits
    it exactly while Platt's logistic cannot."""
    rng = np.random.default_rng(13)
    n = 3000
    raw = rng.uniform(0, 1, n)
    # Staircase distortion: 0 → 0.1, mid → 0.5, high → 0.9.
    true_p = np.where(raw < 0.33, 0.1, np.where(raw < 0.66, 0.5, 0.9))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto")).fit(raw, labels)
    assert cal.chosen_method == "isotonic"


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
