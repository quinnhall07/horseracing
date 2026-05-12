"""Tests for app/services/models/training_data.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.models.training_data import (
    EWM_ALPHA,
    SPEED_FORM_FEATURE_COLUMNS,
    load_training_parquet,
    prepare_training_features,
    time_based_split,
)
from tests.test_models._synth import make_synthetic_results


def test_prepare_features_returns_one_row_per_input_row():
    raw = make_synthetic_results(n_horses=10, n_races_per_horse=5)
    out = prepare_training_features(raw)
    assert len(out) == len(raw)


def test_prepare_features_no_leakage_first_row_per_horse_has_null_priors():
    raw = make_synthetic_results(n_horses=8, n_races_per_horse=4)
    out = prepare_training_features(raw)
    # For the FIRST start of each horse (by date), every "_prior" column must
    # be NaN — the per-horse rolling features use shift(1) so the first row
    # cannot leak its own finish into its own features.
    first_starts = out.sort_values(["horse_key", "race_date"]).groupby("horse_key").head(1)
    for col in ("ewm_speed_prior", "last_speed_prior", "win_rate_prior",
                "mean_finish_pos_prior", "days_since_prev"):
        assert first_starts[col].isna().all(), f"{col} leaked on first start"


def test_field_size_derived_when_source_is_null():
    raw = make_synthetic_results(n_horses=6, n_races_per_horse=3)
    raw["field_size"] = None
    out = prepare_training_features(raw)
    # field_size should now reflect the in-frame group counts (>=1 always).
    assert (out["field_size"] >= 1).all()
    assert out["field_size"].isna().sum() == 0


def test_field_size_respects_existing_value_when_provided():
    raw = make_synthetic_results(n_horses=4, n_races_per_horse=3)
    raw["field_size"] = 99  # noisy explicit value
    out = prepare_training_features(raw)
    assert (out["field_size"] == 99).all()


def test_win_label_matches_finish_position_one():
    raw = make_synthetic_results(n_horses=10, n_races_per_horse=5)
    out = prepare_training_features(raw)
    assert ((out["finish_position"] == 1) == out["win"].astype(bool)).all()


def test_field_relative_zscore_has_zero_mean_per_race():
    raw = make_synthetic_results(n_horses=20, n_races_per_horse=8)
    out = prepare_training_features(raw)
    # Drop NaN rows (first starts have no ewm_speed_prior)
    zs = out.dropna(subset=["ewm_speed_zscore"]).groupby("race_id")["ewm_speed_zscore"].mean()
    # Allow tiny float drift; mean should be very near zero per race.
    assert zs.abs().max() < 1e-6 or np.isclose(zs.abs().max(), 0.0, atol=1e-6)


def test_ewm_prior_uses_shifted_history():
    """EWM prior on row i should ONLY use rows 0..i-1 for the same horse."""
    raw = make_synthetic_results(n_horses=1, n_races_per_horse=3, seed=42)
    # Force three distinct races / speed figures.
    raw = raw.sort_values("race_date").reset_index(drop=True)
    raw["speed_figure"] = [50.0, 100.0, 200.0]
    raw["race_date"] = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
    out = prepare_training_features(raw).sort_values("race_date").reset_index(drop=True)

    # Row 0: no prior → NaN
    assert pd.isna(out.loc[0, "ewm_speed_prior"])
    # Row 1: only the 50 prior — EWM of a single value is that value
    assert out.loc[1, "ewm_speed_prior"] == 50.0
    # Row 2: EWM of [50, 100] in oldest-first order with alpha=0.4 via pandas
    expected = pd.Series([50.0, 100.0]).ewm(alpha=EWM_ALPHA, adjust=True).mean().iloc[-1]
    assert abs(out.loc[2, "ewm_speed_prior"] - expected) < 1e-9


def test_n_prior_starts_increments_per_horse():
    raw = make_synthetic_results(n_horses=1, n_races_per_horse=5)
    out = prepare_training_features(raw).sort_values("race_date").reset_index(drop=True)
    assert list(out["n_prior_starts"]) == [0, 1, 2, 3, 4]


def test_categorical_columns_are_category_dtype():
    raw = make_synthetic_results(n_horses=8, n_races_per_horse=3)
    out = prepare_training_features(raw)
    for col in ("surface", "condition", "race_type", "jurisdiction"):
        assert isinstance(out[col].dtype, pd.CategoricalDtype), f"{col} not categorical"


# ── time_based_split ─────────────────────────────────────────────────────────


def test_time_split_separates_by_date_only():
    raw = make_synthetic_results(n_horses=15, n_races_per_horse=10)
    out = prepare_training_features(raw)
    split = time_based_split(out, val_fraction=0.20)
    assert len(split.train) > 0 and len(split.val) > 0
    assert split.train["race_date"].max() <= split.val["race_date"].min()


def test_time_split_rejects_invalid_fraction():
    raw = make_synthetic_results(n_horses=4, n_races_per_horse=3)
    out = prepare_training_features(raw)
    with pytest.raises(ValueError):
        time_based_split(out, val_fraction=0.0)
    with pytest.raises(ValueError):
        time_based_split(out, val_fraction=1.5)


def test_time_split_on_empty_input_returns_empty():
    empty = pd.DataFrame({"race_date": pd.to_datetime([])})
    split = time_based_split(empty, val_fraction=0.10)
    assert split.train.empty and split.val.empty


def test_feature_columns_registry_is_subset_of_output_columns():
    raw = make_synthetic_results(n_horses=8, n_races_per_horse=4)
    out = prepare_training_features(raw)
    missing = set(SPEED_FORM_FEATURE_COLUMNS) - set(out.columns)
    assert not missing, f"feature registry references missing columns: {missing}"


# ── parquet loader ───────────────────────────────────────────────────────────


def test_load_training_parquet_round_trip(tmp_path):
    raw = make_synthetic_results(n_horses=5, n_races_per_horse=3)
    target = tmp_path / "training.parquet"
    raw.to_parquet(target, index=False)
    loaded = load_training_parquet(target)
    assert len(loaded) == len(raw)
    # race_date should be normalised to midnight.
    assert (loaded["race_date"].dt.hour == 0).all()


def test_load_training_parquet_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_parquet(tmp_path / "does-not-exist.parquet")
