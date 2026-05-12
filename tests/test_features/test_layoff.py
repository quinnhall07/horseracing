"""Tests for app/services/feature_engineering/layoff.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.services.feature_engineering.layoff import (
    DEFAULT_LAMBDA,
    DEFAULT_RECOVERY_THRESHOLD_DAYS,
    FIRST_TIME_STARTER_FITNESS,
    apply_layoff_features,
    layoff_fitness,
    layoff_fitness_series,
)


def test_within_recovery_threshold_is_full_fitness():
    assert layoff_fitness(0) == 1.0
    assert layoff_fitness(15) == 1.0
    assert layoff_fitness(DEFAULT_RECOVERY_THRESHOLD_DAYS) == 1.0


def test_first_time_starter_returns_sentinel():
    assert layoff_fitness(None) == FIRST_TIME_STARTER_FITNESS
    assert layoff_fitness(float("nan")) == FIRST_TIME_STARTER_FITNESS


def test_invalid_input_falls_through_to_sentinel():
    assert layoff_fitness("not a number") == FIRST_TIME_STARTER_FITNESS


def test_negative_days_clamp_to_zero():
    # Negative is a data error; we coerce to zero and return full fitness.
    assert layoff_fitness(-5) == 1.0


def test_monotone_decay_past_threshold():
    f60 = layoff_fitness(60)
    f120 = layoff_fitness(120)
    f365 = layoff_fitness(365)
    assert 1.0 > f60 > f120 > f365 > 0.0


def test_half_life_default_is_60_days_past_threshold():
    # With DEFAULT_LAMBDA = ln(2)/60, fitness at recovery+60 should be 0.5.
    days = DEFAULT_RECOVERY_THRESHOLD_DAYS + 60
    assert math.isclose(layoff_fitness(days), 0.5, abs_tol=1e-9)


def test_custom_lambda_overrides_default():
    # Lambda 0 → never decays, even at very long layoffs.
    assert layoff_fitness(10_000, decay_lambda=0.0) == 1.0


def test_custom_recovery_threshold():
    # Threshold of 100 days means 90 days off is still full fitness.
    assert layoff_fitness(90, recovery_threshold=100) == 1.0
    assert layoff_fitness(101, recovery_threshold=100) < 1.0


def test_series_version_returns_numpy_array_of_floats():
    out = layoff_fitness_series([0, 30, 90, None])
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert out.shape == (4,)
    assert out[0] == 1.0
    assert out[1] == 1.0
    assert 0 < out[2] < 1
    assert out[3] == FIRST_TIME_STARTER_FITNESS


def test_apply_layoff_features_writes_column_in_place():
    df = pd.DataFrame({"days_since_last": [0, 30, 90, None]})
    out = apply_layoff_features(df)
    assert "layoff_fitness" in out.columns
    assert out is df  # mutates in place
    assert out.loc[0, "layoff_fitness"] == 1.0
    assert out.loc[1, "layoff_fitness"] == 1.0
    assert 0 < out.loc[2, "layoff_fitness"] < 1
    assert out.loc[3, "layoff_fitness"] == FIRST_TIME_STARTER_FITNESS


def test_apply_respects_custom_columns():
    df = pd.DataFrame({"layoff_days_alt": [60]})
    apply_layoff_features(df, days_col="layoff_days_alt", out_fitness_col="fit")
    assert "fit" in df.columns
    assert 0 < df.loc[0, "fit"] < 1


def test_default_lambda_value_documented():
    # The half-life default contract is part of the public API; lock it in.
    assert math.isclose(DEFAULT_LAMBDA, math.log(2) / 60.0)
    assert DEFAULT_RECOVERY_THRESHOLD_DAYS == 30.0
