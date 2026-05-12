"""Tests for app/services/models/market.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.models.market import MarketModel
from app.services.models.training_data import prepare_training_features
from tests.test_models._synth import make_synthetic_results


@pytest.fixture(scope="module")
def fitted_model():
    raw = make_synthetic_results(n_horses=40, n_races_per_horse=8)
    df = prepare_training_features(raw)
    return MarketModel().fit(df), df


def test_raw_implied_prob_is_one_over_odds():
    s = pd.Series([2.0, 4.0, 10.0])
    out = MarketModel.raw_implied_prob(s)
    assert list(out) == [0.5, 0.25, 0.1]


def test_raw_implied_prob_handles_nonpositive():
    s = pd.Series([0.0, -1.0, 1.5])
    out = MarketModel.raw_implied_prob(s)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert abs(out.iloc[2] - 1.0 / 1.5) < 1e-12


def test_predict_returns_array_of_length_n(fitted_model):
    model, df = fitted_model
    out = model.predict_proba(df)
    assert len(out) == len(df)
    # Output should be in [0, 1] (or NaN for missing odds).
    finite = out[np.isfinite(out)]
    assert (finite >= 0).all() and (finite <= 1).all()


def test_predict_nan_for_missing_odds(fitted_model):
    model, df = fitted_model
    bad = df.head(3).copy()
    bad["odds_final"] = np.nan
    out = model.predict_proba(bad)
    assert np.isnan(out).all()


def test_predict_without_fit_raises():
    model = MarketModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame({"odds_final": [2.0], "race_id": ["x"], "win": [0]}))
