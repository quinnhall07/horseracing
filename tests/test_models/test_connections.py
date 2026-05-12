"""Tests for app/services/models/connections.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.models.connections import ConnectionsConfig, ConnectionsModel
from app.services.models.training_data import prepare_training_features
from tests.test_models._synth import make_synthetic_results


@pytest.fixture(scope="module")
def fitted_model():
    raw = make_synthetic_results(n_horses=40, n_races_per_horse=8)
    df = prepare_training_features(raw)
    model = ConnectionsModel(
        config=ConnectionsConfig(min_jockey_starts=2, min_trainer_starts=2,
                                  min_pair_starts=2, prior_strength=5.0)
    ).fit(df)
    return model, df


def test_predict_returns_one_value_per_row(fitted_model):
    model, df = fitted_model
    out = model.predict_proba(df)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(df)
    assert (out >= 0).all() and (out <= 1).all()


def test_unknown_pair_falls_back_to_jurisdiction_baseline(fitted_model):
    model, df = fitted_model
    novel = df.head(1).copy()
    novel["jockey_name"] = "unseen_jockey"
    novel["trainer_name"] = "unseen_trainer"
    out = model.predict_proba(novel)
    jur = novel.iloc[0]["jurisdiction"]
    expected = model.jurisdiction_rate.get(str(jur), model.global_rate)
    assert abs(out[0] - expected) < 1e-9


def test_shrinkage_keeps_rates_in_unit_interval(fitted_model):
    model, _ = fitted_model
    # Shrunken rates can touch the boundary [0, 1] when both the sample and
    # the prior are uniformly winning/losing, but must never escape it.
    for v in model.jockey_rate.values():
        assert 0.0 <= v <= 1.0
    for v in model.pair_rate.values():
        assert 0.0 <= v <= 1.0


def test_shrinkage_average_close_to_jurisdiction_baseline(fitted_model):
    model, _ = fitted_model
    # Across many jockeys, the mean shrunken rate should land near the
    # underlying global rate — extreme outliers get pulled toward baseline.
    if len(model.jockey_rate) >= 5:
        mean_jockey_rate = sum(model.jockey_rate.values()) / len(model.jockey_rate)
        assert abs(mean_jockey_rate - model.global_rate) < 0.15


def test_global_rate_matches_label_mean(fitted_model):
    model, df = fitted_model
    assert abs(model.global_rate - df["win"].mean()) < 1e-9


def test_save_load_round_trip(tmp_path, fitted_model):
    model, df = fitted_model
    target = tmp_path / "conn"
    model.save(target)
    loaded = ConnectionsModel.load(target)
    p1 = model.predict_proba(df.head(50))
    p2 = loaded.predict_proba(df.head(50))
    np.testing.assert_allclose(p1, p2)
    assert loaded.is_fitted


def test_predict_without_fit_raises():
    model = ConnectionsModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame({"jurisdiction": [], "jockey_name": [], "trainer_name": []}))
