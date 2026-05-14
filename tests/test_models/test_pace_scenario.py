"""
tests/test_models/test_pace_scenario.py
───────────────────────────────────────
PaceScenarioModel unit tests. The model trains a LightGBM binary classifier
on per-horse pace data (fraction_q1_sec, fraction_q2_sec, beaten_lengths_q1,
beaten_lengths_q2) when the parquet provides it. See ADR-047.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.models.pace_scenario import (
    PaceScenarioConfig,
    PaceScenarioModel,
)


def _synth(n_races: int = 800, seed: int = 0, with_pace: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for race_id in range(n_races):
        n_horses = int(rng.integers(6, 12))
        f1_base = 21.0 + rng.normal(0, 0.5)
        ability = rng.normal(0, 1, size=n_horses)
        win_idx = int(ability.argmax())
        for i in range(n_horses):
            beaten_q1 = max(0.0, float(2.0 - ability[i] + rng.normal(0, 0.3)))
            beaten_q2 = beaten_q1 + rng.normal(0, 0.5)
            row = {
                "race_id": f"r{race_id}",
                "win": int(i == win_idx),
                "distance_furlongs": 6.0,
                "field_size": n_horses,
                "surface": "turf",
                "condition": "firm",
                "race_type": "allowance",
                "jurisdiction": "HK",
            }
            if with_pace:
                row.update({
                    "fraction_q1_sec": f1_base + beaten_q1 * 0.2,
                    "fraction_q2_sec": f1_base + 22.0 + beaten_q2 * 0.2,
                    "beaten_lengths_q1": beaten_q1,
                    "beaten_lengths_q2": beaten_q2,
                })
            else:
                row.update({
                    "fraction_q1_sec": None,
                    "fraction_q2_sec": None,
                    "beaten_lengths_q1": None,
                    "beaten_lengths_q2": None,
                })
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg() -> PaceScenarioConfig:
    return PaceScenarioConfig(min_rows_with_data=500, num_boost_round=80)


@pytest.fixture(scope="module")
def fitted(cfg) -> PaceScenarioModel:
    df = _synth(n_races=600, seed=42)
    split = int(len(df) * 0.8)
    return PaceScenarioModel(cfg).fit(df.iloc[:split], val_df=df.iloc[split:])


# ── Trainability gate ─────────────────────────────────────────────────────────


def test_is_trainable_with_pace_data(cfg):
    df = _synth(n_races=600, seed=1)
    assert PaceScenarioModel.is_trainable_with(df, cfg)


def test_is_not_trainable_without_pace_data(cfg):
    df = _synth(n_races=600, seed=1, with_pace=False)
    assert not PaceScenarioModel.is_trainable_with(df, cfg)


def test_is_not_trainable_below_row_threshold():
    df = _synth(n_races=10, seed=1)  # too few rows
    cfg_strict = PaceScenarioConfig(min_rows_with_data=5_000)
    assert not PaceScenarioModel.is_trainable_with(df, cfg_strict)


def test_is_not_trainable_missing_columns(cfg):
    df = _synth(n_races=600, seed=1).drop(columns=["fraction_q1_sec"])
    assert not PaceScenarioModel.is_trainable_with(df, cfg)


# ── Predict + fit ────────────────────────────────────────────────────────────


def test_predict_proba_unfitted_returns_05():
    df = _synth(n_races=10)
    m = PaceScenarioModel()
    out = m.predict_proba(df)
    assert out.shape == (len(df),)
    assert np.all(out == 0.5)


def test_fit_discriminates_winners_from_losers(fitted):
    """Model assigns higher P(win) to actual winners than to losers."""
    df = _synth(n_races=200, seed=99)
    out = fitted.predict_proba(df)
    win_mean = float(out[df["win"] == 1].mean())
    lose_mean = float(out[df["win"] == 0].mean())
    assert win_mean > lose_mean + 0.1, f"winners {win_mean} not above losers {lose_mean}"


def test_predict_proba_no_pace_data_returns_neutral(fitted):
    """A row with all pace cols NaN falls back to 0.5 even on a fitted model."""
    df = _synth(n_races=10, seed=1, with_pace=False)
    out = fitted.predict_proba(df)
    assert out.shape == (len(df),)
    assert np.allclose(out, 0.5)


def test_predict_proba_missing_columns_returns_neutral(fitted):
    """A df lacking the fraction columns altogether returns 0.5."""
    df = _synth(n_races=10, seed=1).drop(columns=["fraction_q1_sec"])
    out = fitted.predict_proba(df)
    assert np.allclose(out, 0.5)


def test_fit_metrics_populated(fitted):
    assert fitted.metrics is not None
    assert fitted.metrics.n_train_with_pace > 0
    assert 0.0 < fitted.metrics.train_auc <= 1.0


# ── Save / Load round-trip ────────────────────────────────────────────────────


def test_save_load_round_trip(fitted, tmp_path: Path):
    df = _synth(n_races=50, seed=33)
    proba_before = fitted.predict_proba(df)
    out = tmp_path / "pace"
    fitted.save(out)
    assert (out / "booster.txt").exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["stub"] is False
    reloaded = PaceScenarioModel.load(out)
    assert reloaded.is_fitted
    proba_after = reloaded.predict_proba(df)
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)


def test_save_stub_writes_stub_marker(tmp_path: Path):
    m = PaceScenarioModel()  # unfitted
    out = tmp_path / "pace_stub"
    m.save(out)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["stub"] is True
    assert not (out / "booster.txt").exists()


def test_load_missing_returns_stub(tmp_path: Path):
    m = PaceScenarioModel.load(tmp_path / "nonexistent")
    assert m.is_fitted is False
    df = _synth(n_races=5)
    assert np.allclose(m.predict_proba(df), 0.5)
