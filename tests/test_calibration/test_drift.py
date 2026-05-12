"""Tests for app/services/calibration/drift.py.

The CUSUM drift detector watches a stream of (prediction, label) pairs
and fires when sustained calibration error in *standardised* (Bernoulli
z-score) units exceeds the configured threshold. Tests lock in:

    (a) Config validation (k ≥ 0, h > 0, z_clip > 0, eps > 0).
    (b) Standardisation produces z=0 for label-matches-rate, with the
        expected sign + magnitude for mismatches.
    (c) Alternating-balanced calibrated streams do NOT alarm under
        default k=0.5, h=4.
    (d) Sustained under- or over-prediction DOES alarm, in the correct
        direction, within a reasonable detection delay.
    (e) Alarm latches across subsequent observations; reset() clears.
    (f) One-shot detect_drift wrapper agrees with incremental run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.calibration.drift import (
    CUSUMConfig,
    CUSUMDetector,
    CUSUMState,
    DriftReport,
    _standardised_residual,
    detect_drift,
)


# ── Config validation ────────────────────────────────────────────────────


def test_config_rejects_negative_k():
    with pytest.raises(ValueError):
        CUSUMConfig(k=-0.01)


def test_config_rejects_non_positive_h():
    with pytest.raises(ValueError):
        CUSUMConfig(h=0.0)
    with pytest.raises(ValueError):
        CUSUMConfig(h=-1.0)


def test_config_rejects_non_positive_z_clip():
    with pytest.raises(ValueError):
        CUSUMConfig(z_clip=0.0)


def test_config_rejects_non_positive_eps():
    with pytest.raises(ValueError):
        CUSUMConfig(eps=0.0)


def test_default_config_is_sensible():
    cfg = CUSUMConfig()
    assert cfg.k >= 0
    assert cfg.h > 0
    assert cfg.two_sided is True


# ── Standardised residual helper ─────────────────────────────────────────


def test_standardised_residual_zero_when_label_matches_pred_at_half():
    """At p=0.5 with label=1: z = 0.5 / sqrt(0.25) = 1.0 (one σ)."""
    z = _standardised_residual(0.5, 1.0, eps=1e-4, z_clip=5.0)
    assert z == pytest.approx(1.0)
    z = _standardised_residual(0.5, 0.0, eps=1e-4, z_clip=5.0)
    assert z == pytest.approx(-1.0)


def test_standardised_residual_clip_applies_at_extremes():
    # p = 0.001, label = 1 → naive z ≈ 0.999 / sqrt(0.001*0.999) ≈ 31.6.
    # Clipped to z_clip = 5.0.
    z = _standardised_residual(0.001, 1.0, eps=1e-6, z_clip=5.0)
    assert z == 5.0
    z = _standardised_residual(0.999, 0.0, eps=1e-6, z_clip=5.0)
    assert z == -5.0


def test_standardised_residual_eps_floor_kicks_in_at_extremes():
    """At p=0 or p=1, sqrt(p(1-p))=0 → would div-by-zero without eps."""
    # No exception — eps floor produces a finite value, then clipping caps it.
    z = _standardised_residual(0.0, 1.0, eps=1e-4, z_clip=5.0)
    assert math.isfinite(z)


# ── State + update mechanics ─────────────────────────────────────────────


def test_initial_state_zero():
    det = CUSUMDetector()
    assert det.state.s_plus == 0.0
    assert det.state.s_minus == 0.0
    assert det.state.n == 0
    assert det.state.alarmed_at is None
    assert det.state.alarmed is False


def test_update_rejects_prediction_out_of_unit_interval():
    det = CUSUMDetector()
    with pytest.raises(ValueError):
        det.update(prediction=1.5, label=1)
    with pytest.raises(ValueError):
        det.update(prediction=-0.1, label=1)


def test_update_rejects_non_binary_label():
    det = CUSUMDetector()
    with pytest.raises(ValueError):
        det.update(prediction=0.5, label=0.3)
    with pytest.raises(ValueError):
        det.update(prediction=0.5, label=2)


def test_state_increments_n_on_each_update():
    det = CUSUMDetector()
    for _ in range(5):
        det.update(prediction=0.3, label=0)
    assert det.state.n == 5


def test_s_plus_grows_under_systematic_under_prediction():
    """p=0.5, label=1 every step. z=+1 each step. With k=0.5, s_plus
    grows by 0.5 per step. After 4 steps s_plus=2.0."""
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=10.0))
    for _ in range(4):
        det.update(prediction=0.5, label=1)
    assert det.state.s_plus == pytest.approx(2.0)


def test_s_minus_grows_under_systematic_over_prediction():
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=10.0, two_sided=True))
    for _ in range(4):
        det.update(prediction=0.5, label=0)
    assert det.state.s_minus == pytest.approx(2.0)


def test_one_sided_does_not_track_negative_drift():
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=10.0, two_sided=False))
    for _ in range(10):
        det.update(prediction=0.5, label=0)
    assert det.state.s_minus == 0.0
    assert det.state.alarmed is False


# ── Calibrated streams should NOT alarm under default config ─────────────


def test_alternating_calibrated_stream_does_not_alarm():
    """Alternating label=0/1 with p=0.5 — perfectly calibrated. z ∈ {+1, -1}
    alternating. Both s_plus and s_minus oscillate around 0; never exceed h=4.
    """
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=4.0))
    for i in range(200):
        det.update(prediction=0.5, label=i % 2)
    assert det.state.alarmed is False


def test_well_calibrated_random_stream_quiet_at_h6():
    """Random calibrated stream of 500 obs with h=6 → ARL₀ ≈ 1290 so
    we very rarely alarm. With a fixed seed we verify a quiet run."""
    rng = np.random.default_rng(2026)
    preds = rng.uniform(0.2, 0.8, size=500)
    labels = (rng.uniform(size=500) < preds).astype(int)
    report = detect_drift(preds, labels, CUSUMConfig(k=0.5, h=6.0))
    assert report.alarmed is False


# ── Drifted streams SHOULD alarm ─────────────────────────────────────────


def _calibrated_prefix(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic p=0.5 stream with alternating labels — exactly calibrated,
    z alternates ±1, CUSUM stays bounded near 0. Used as a quiet warm-up to
    isolate the drift signal in the subsequent biased segment."""
    preds = np.full(n, 0.5)
    labels = np.tile([0, 1], n // 2 + 1)[:n].astype(int)
    return preds, labels


def test_under_prediction_alarms_high():
    """Predict 0.3 but realised rate 0.7 → expected z ≈ +0.87/step. With
    k=0.5, drift = 0.37/step; alarms within ~12 steps at h=4.

    Setup: 50 deterministic-calibrated obs (quiet warm-up), then 100 biased.
    """
    calib_p, calib_l = _calibrated_prefix(50)
    rng = np.random.default_rng(101)
    drift_p = np.full(100, 0.3)
    drift_l = (rng.uniform(size=100) < 0.7).astype(int)
    preds = np.concatenate([calib_p, drift_p])
    labels = np.concatenate([calib_l, drift_l])

    report = detect_drift(preds, labels, CUSUMConfig(k=0.5, h=4.0))
    assert report.alarmed is True
    assert report.direction == "high"
    assert report.alarmed_at is not None and report.alarmed_at >= 50


def test_over_prediction_alarms_low():
    calib_p, calib_l = _calibrated_prefix(50)
    rng = np.random.default_rng(202)
    drift_p = np.full(100, 0.7)
    drift_l = (rng.uniform(size=100) < 0.3).astype(int)
    preds = np.concatenate([calib_p, drift_p])
    labels = np.concatenate([calib_l, drift_l])

    report = detect_drift(preds, labels, CUSUMConfig(k=0.5, h=4.0))
    assert report.alarmed is True
    assert report.direction == "low"
    assert report.alarmed_at is not None and report.alarmed_at >= 50


def test_alarm_latches_after_first_crossing():
    """Once alarmed, the detector stays alarmed even if drift reverses."""
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=2.0))
    # Force alarm with 5 strong-direction obs.
    for _ in range(5):
        det.update(prediction=0.5, label=1)
    assert det.state.alarmed is True
    first_alarm_at = det.state.alarmed_at
    # Feed perfectly-balanced obs.
    for i in range(20):
        det.update(prediction=0.5, label=i % 2)
    assert det.state.alarmed is True
    assert det.state.alarmed_at == first_alarm_at


# ── Reset ────────────────────────────────────────────────────────────────


def test_reset_clears_state():
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=2.0))
    for _ in range(5):
        det.update(prediction=0.5, label=1)
    assert det.state.alarmed is True
    det.reset()
    assert det.state.s_plus == 0.0
    assert det.state.s_minus == 0.0
    assert det.state.n == 0
    assert det.state.alarmed_at is None


# ── Helpers + DriftReport ────────────────────────────────────────────────


def test_run_returns_state_per_step():
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=10.0))
    states = det.run(predictions=[0.5, 0.5, 0.5], labels=[1, 1, 1])
    assert len(states) == 3
    assert states[0].s_plus < states[1].s_plus < states[2].s_plus


def test_run_rejects_mismatched_lengths():
    det = CUSUMDetector()
    with pytest.raises(ValueError):
        det.run(predictions=[0.5, 0.5], labels=[1, 0, 1])


def test_detect_drift_report_fields():
    rng = np.random.default_rng(11)
    preds = rng.uniform(0.3, 0.6, size=50)
    labels = (rng.uniform(size=50) < preds).astype(int)
    report = detect_drift(preds, labels, CUSUMConfig(k=0.5, h=4.0))
    assert isinstance(report, DriftReport)
    assert report.n == 50
    assert report.config["k"] == 0.5
    assert report.config["h"] == 4.0
    assert report.max_s_plus >= report.final_s_plus
    assert report.max_s_minus >= report.final_s_minus


def test_detect_drift_handles_empty_input():
    report = detect_drift([], [], CUSUMConfig())
    assert report.n == 0
    assert report.alarmed is False
    assert report.alarmed_at is None


def test_detect_drift_one_shot_matches_incremental():
    """detect_drift over a batch agrees with manual CUSUMDetector.run."""
    rng = np.random.default_rng(303)
    preds = rng.uniform(0.2, 0.8, size=100)
    labels = (rng.uniform(size=100) < preds).astype(int)
    cfg = CUSUMConfig(k=0.5, h=4.0)
    report = detect_drift(preds, labels, cfg)
    det = CUSUMDetector(cfg)
    states = det.run(preds, labels)
    final = states[-1]
    assert report.final_s_plus == pytest.approx(final.s_plus)
    assert report.final_s_minus == pytest.approx(final.s_minus)
    assert report.alarmed == final.alarmed
    assert report.alarmed_at == final.alarmed_at
