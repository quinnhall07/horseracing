"""
tests/test_models/test_sequence.py
──────────────────────────────────
Step 5 — SequenceModel (Layer 1c, Transformer-over-history) unit tests.

These tests exercise the trainable code path. They are skipped when torch
isn't importable, mirroring the optional-dep contract documented in the
module docstring of sequence.py.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.models.sequence import (
    SequenceModel,
    SequenceModelConfig,
    _build_sequences,
)

torch = pytest.importorskip("torch")  # skip whole module if torch missing


def _synth(n_horses: int = 60, n_races_per_horse: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_horses):
        base_speed = 70 + rng.uniform(-5, 15)
        for r in range(n_races_per_horse):
            rows.append(
                {
                    "horse_dedup_key": f"H{h:04d}",
                    "horse_name": f"H{h:04d}",
                    "jurisdiction": "US",
                    "race_date": date(2024, 1, 1) + timedelta(days=30 * r),
                    "speed_figure": float(base_speed + rng.normal(0, 3)),
                    "distance_furlongs": 6.0,
                    "surface": "dirt",
                    "finish_position": int(max(1, rng.poisson(2) + 1)),
                    "field_size": 6,
                    "weight_lbs": float(120.0 + rng.uniform(-3, 3)),
                    "odds_final": float(rng.uniform(2.5, 12.0)),
                    "purse_usd": 50000.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg() -> SequenceModelConfig:
    # Small + fast so the test suite stays quick; large enough to surface real bugs.
    return SequenceModelConfig(
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        max_sequence_length=6,
        batch_size=64,
        n_epochs=3,
        learning_rate=1e-3,
        device="cpu",  # CPU keeps the test deterministic; MPS is opt-in.
    )


@pytest.fixture(scope="module")
def fitted(cfg) -> SequenceModel:
    df = _synth(n_horses=60, n_races_per_horse=10, seed=42)
    split = int(len(df) * 0.8)
    return SequenceModel(cfg).fit(df.iloc[:split], val_df=df.iloc[split:])


# ── Trainability gate ─────────────────────────────────────────────────────────


def test_is_trainable_with_dedup_key():
    df = _synth(n_horses=2, n_races_per_horse=3)
    assert SequenceModel.is_trainable_with(df)


def test_is_trainable_with_legacy_fallback():
    df = _synth(n_horses=2, n_races_per_horse=3).drop(columns=["horse_dedup_key"])
    assert SequenceModel.is_trainable_with(df)  # horse_name + jurisdiction left


def test_is_trainable_with_empty_df():
    assert not SequenceModel.is_trainable_with(pd.DataFrame())


# ── _build_sequences contract ────────────────────────────────────────────────


def test_build_sequences_shape_and_padding(cfg):
    df = _synth(n_horses=3, n_races_per_horse=5)
    Xs, mask, Xt, y = _build_sequences(df, cfg)
    assert Xs.shape == (15, cfg.max_sequence_length, cfg.n_seq_features)
    assert mask.shape == (15, cfg.max_sequence_length)
    assert Xt.shape == (15, cfg.n_today_features)
    assert y.shape == (15,)
    # Output is in the input frame's order (not sorted): the first row of df
    # belongs to horse H0000 race 1 and should have all-padding (no priors).
    # That row in the input order is exactly index 0 since _synth writes
    # horses in horse-major order.
    assert mask[0].all(), "row 0 has no priors → all-pad mask"
    # The 5th row in the input order is H0000 race 5; it has 4 priors → 4
    # non-pad slots.
    assert (~mask[4]).sum() == 4


def test_build_sequences_padding_left(cfg):
    """Priors are packed at the END of the sequence — model treats the
    latest prior as the rightmost token."""
    df = _synth(n_horses=1, n_races_per_horse=3)
    Xs, mask, _, _ = _build_sequences(df, cfg)
    T = cfg.max_sequence_length
    # Row 1 has 1 prior; padding occupies the leftmost (T-1) slots.
    assert (mask[1, : T - 1]).all()
    assert not mask[1, T - 1]


# ── Predict + train loop ─────────────────────────────────────────────────────


def test_predict_proba_unfitted_returns_05():
    df = _synth(n_horses=5, n_races_per_horse=3)
    m = SequenceModel(SequenceModelConfig(device="cpu"))
    out = m.predict_proba(df)
    assert out.shape == (15,)
    assert np.all(out == 0.5)


def test_fit_decreases_val_brier(fitted):
    """fit() runs end-to-end without NaN and the val_brier trend points down."""
    metrics = fitted._metrics  # noqa: SLF001
    assert metrics["epochs"], "metrics should record at least one epoch"
    briers = [
        e["val_brier"] for e in metrics["epochs"] if not np.isnan(e["val_brier"])
    ]
    assert briers, "val_brier should be populated when val_df is provided"
    assert briers[-1] < briers[0] + 1e-3, (
        f"val_brier did not improve: {briers[0]} → {briers[-1]}"
    )


def test_predict_proba_output_shape_and_range(fitted, cfg):
    df = _synth(n_horses=10, n_races_per_horse=4, seed=7)
    out = fitted.predict_proba(df)
    assert out.shape == (len(df),)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_first_time_starter_returns_05(fitted):
    """Rows with no prior history bypass the encoder and return 0.5 exactly."""
    df = _synth(n_horses=4, n_races_per_horse=3)
    out = fitted.predict_proba(df)
    # _synth lays out horses in horse-major order, so rows 0, 3, 6, 9 are
    # the no-prior rows for horses 0..3.
    for i in (0, 3, 6, 9):
        assert out[i] == pytest.approx(0.5)


# ── Save / Load round-trip ───────────────────────────────────────────────────


def test_save_load_round_trip(fitted, tmp_path: Path):
    df = _synth(n_horses=5, n_races_per_horse=4, seed=99)
    proba_before = fitted.predict_proba(df)
    out = tmp_path / "sequence"
    fitted.save(out)
    assert (out / "metadata.json").exists()
    assert (out / "encoder.pt").exists()
    assert (out / "scalers.npz").exists()

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["stub"] is False
    assert meta["artifact_version"] == SequenceModel.ARTIFACT_VERSION

    reloaded = SequenceModel.load(out)
    assert reloaded.is_fitted
    proba_after = reloaded.predict_proba(df)
    # Identical predictions (CPU is fully deterministic for these tiny networks).
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)


def test_load_missing_artifact_returns_stub(tmp_path: Path):
    """Loading from an empty dir yields a clean stub — predict_proba returns 0.5."""
    m = SequenceModel.load(tmp_path)
    assert m.is_fitted is False
    df = _synth(n_horses=2, n_races_per_horse=3)
    out = m.predict_proba(df)
    assert np.all(out == 0.5)
