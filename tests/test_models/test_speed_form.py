"""Tests for app/services/models/speed_form.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.models.speed_form import (
    SpeedFormConfig,
    SpeedFormModel,
    _race_softmax,
)
from app.services.models.training_data import (
    prepare_training_features,
    time_based_split,
)
from tests.test_models._synth import make_synthetic_results


@pytest.fixture(scope="module")
def trained_model():
    raw = make_synthetic_results(n_horses=60, n_races_per_horse=10, seed=11)
    df = prepare_training_features(raw)
    split = time_based_split(df, val_fraction=0.20)
    cfg = SpeedFormConfig(num_boost_round=120, early_stopping_rounds=20,
                          min_data_in_leaf=10)
    model = SpeedFormModel(config=cfg).fit(split.train, val_df=split.val)
    return model, split


def test_fit_records_metrics(trained_model):
    model, _ = trained_model
    assert model.metrics is not None
    assert model.metrics.n_train > 0
    assert model.metrics.n_val > 0
    assert model.metrics.n_features == len(model.config.feature_columns)


def test_val_race_top1_beats_random_pick(trained_model):
    """The synthetic generator wires speed_figure to actual finish; the model
    should identify the within-race winner far more often than 1/field_size.

    NOTE: per-row AUC is a poor metric here because the relevant
    discrimination is WITHIN race, not across the whole dataset. Top-1
    accuracy within race is the canonical signal."""
    model, _ = trained_model
    # Synthetic fields average ~5 horses → random pick = 1/5 = 0.20.
    # Real signal should land well above that.
    assert model.metrics.val_race_top1_accuracy > 0.40, (
        f"top1_acc={model.metrics.val_race_top1_accuracy}"
    )


def test_predict_softmax_sums_to_one_per_race(trained_model):
    model, split = trained_model
    p = model.predict_softmax(split.val)
    # Group by race_id and confirm sums are ~1.
    by_race = pd.DataFrame({"p": p.values, "race_id": split.val["race_id"].values})
    sums = by_race.groupby("race_id")["p"].sum()
    assert (np.abs(sums - 1.0) < 1e-6).all()


def test_predict_proba_returns_array_of_length_n(trained_model):
    model, split = trained_model
    out = model.predict_proba(split.val)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(split.val)
    assert (out >= 0).all() and (out <= 1).all()


def test_predict_without_fit_raises():
    model = SpeedFormModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame())


def test_save_and_load_round_trip(tmp_path, trained_model):
    model, split = trained_model
    save_dir = tmp_path / "sf"
    model.save(save_dir)
    loaded = SpeedFormModel.load(save_dir)
    # Same predictions to numerical precision.
    p1 = model.predict_proba(split.val)
    p2 = loaded.predict_proba(split.val)
    np.testing.assert_allclose(p1, p2, atol=1e-9)
    assert loaded.metrics is not None
    assert loaded.metrics.val_auc == model.metrics.val_auc


def test_save_unfitted_raises(tmp_path):
    with pytest.raises(RuntimeError):
        SpeedFormModel().save(tmp_path / "x")


# ── helper: _race_softmax ────────────────────────────────────────────────────


def test_race_softmax_sums_to_one_per_group():
    s = pd.Series([1.0, 2.0, 3.0, 0.0, 1.0])
    r = pd.Series(["A", "A", "A", "B", "B"])
    out = _race_softmax(s, r)
    grouped = pd.DataFrame({"out": out, "r": r}).groupby("r")["out"].sum()
    assert np.allclose(grouped.values, 1.0)


def test_race_softmax_orders_match_score_order():
    s = pd.Series([3.0, 1.0, 2.0])
    r = pd.Series(["A", "A", "A"])
    out = _race_softmax(s, r)
    # Highest input score → highest probability.
    assert out.iloc[0] > out.iloc[2] > out.iloc[1]


def test_race_softmax_numerical_stability_with_large_inputs():
    s = pd.Series([1000.0, 1001.0])
    r = pd.Series(["A", "A"])
    out = _race_softmax(s, r)
    assert np.isfinite(out).all()
    assert np.isclose(out.sum(), 1.0)
