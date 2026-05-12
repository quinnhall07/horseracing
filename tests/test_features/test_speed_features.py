"""Tests for app/services/feature_engineering/speed_features.py."""

from __future__ import annotations

import math

import pandas as pd

from app.services.feature_engineering.speed_features import (
    EWM_ALPHA,
    build_speed_feature_frame,
    ewm_speed,
    horse_speed_summary,
)
from tests.test_features._fixtures import make_horse, make_pp, make_race


# ── ewm_speed ────────────────────────────────────────────────────────────────


def test_ewm_speed_empty_returns_none():
    assert ewm_speed([]) is None


def test_ewm_speed_single_value_returns_that_value():
    assert ewm_speed([95.0]) == 95.0


def test_ewm_speed_weights_recent_value_more_heavily():
    # Input order is MOST-RECENT-FIRST.
    # Recent=100, prior=80 → EWM closer to 100 than the simple mean of 90.
    val = ewm_speed([100.0, 80.0])
    assert val is not None
    simple_mean = 90.0
    assert val > simple_mean


def test_ewm_speed_matches_pandas_reference_implementation():
    figs_recent_first = [100.0, 80.0, 60.0]
    figs_oldest_first = list(reversed(figs_recent_first))
    expected = pd.Series(figs_oldest_first).ewm(alpha=EWM_ALPHA, adjust=True).mean().iloc[-1]
    assert math.isclose(ewm_speed(figs_recent_first), expected, rel_tol=1e-12)


# ── horse_speed_summary ──────────────────────────────────────────────────────


def test_horse_summary_no_pp_returns_all_none_with_zero_count():
    horse = make_horse(pp_lines=[])
    summary = horse_speed_summary(horse)
    assert summary["ewm_speed_figure"] is None
    assert summary["best_speed_figure"] is None
    assert summary["last_speed_figure"] is None
    assert summary["speed_figure_delta"] is None
    assert summary["n_speed_figures"] == 0.0


def test_horse_summary_single_pp_has_no_delta_but_has_last_and_best():
    horse = make_horse(pp_lines=[make_pp(speed_figure=92.0)])
    summary = horse_speed_summary(horse)
    assert summary["last_speed_figure"] == 92.0
    assert summary["best_speed_figure"] == 92.0
    assert summary["ewm_speed_figure"] == 92.0
    assert summary["speed_figure_delta"] is None
    assert summary["n_speed_figures"] == 1.0


def test_horse_summary_delta_is_last_minus_second_last():
    pps = [
        make_pp(days_ago=10, speed_figure=95.0),
        make_pp(days_ago=40, speed_figure=85.0),
        make_pp(days_ago=70, speed_figure=80.0),
    ]
    horse = make_horse(pp_lines=pps)
    summary = horse_speed_summary(horse)
    assert summary["speed_figure_delta"] == 10.0  # 95 - 85
    assert summary["last_speed_figure"] == 95.0
    assert summary["best_speed_figure"] == 95.0


def test_horse_summary_skips_null_speed_figures():
    pps = [
        make_pp(days_ago=10, speed_figure=None),
        make_pp(days_ago=40, speed_figure=88.0),
    ]
    horse = make_horse(pp_lines=pps)
    summary = horse_speed_summary(horse)
    assert summary["n_speed_figures"] == 1.0
    assert summary["last_speed_figure"] == 88.0


def test_horse_summary_best_window_limits_to_window_size():
    figs_recent_first = [60, 70, 80, 90, 100, 110, 120]  # 120 is OUTSIDE the 6-window
    pps = [make_pp(days_ago=10 * (i + 1), speed_figure=float(f))
           for i, f in enumerate(figs_recent_first)]
    horse = make_horse(pp_lines=pps)
    summary = horse_speed_summary(horse)
    # 6-element window covers [60..110]; the trailing 120 is excluded.
    assert summary["best_speed_figure"] == 110.0


# ── build_speed_feature_frame ────────────────────────────────────────────────


def test_field_relative_features_have_zero_mean_zscore():
    horses = [
        make_horse(post=1, pp_lines=[make_pp(speed_figure=70.0)]),
        make_horse(post=2, pp_lines=[make_pp(speed_figure=85.0)]),
        make_horse(post=3, pp_lines=[make_pp(speed_figure=100.0)]),
    ]
    race = make_race(horses=horses)
    df = build_speed_feature_frame(race)
    assert math.isclose(df["ewm_speed_zscore"].mean(), 0.0, abs_tol=1e-12)


def test_rank_is_monotonic_with_speed():
    horses = [
        make_horse(post=1, pp_lines=[make_pp(speed_figure=70.0)]),
        make_horse(post=2, pp_lines=[make_pp(speed_figure=100.0)]),
        make_horse(post=3, pp_lines=[make_pp(speed_figure=85.0)]),
    ]
    race = make_race(horses=horses)
    df = build_speed_feature_frame(race)
    # Higher speed → lower (= better) rank.
    assert df.loc[2, "ewm_speed_rank"] == 1
    assert df.loc[3, "ewm_speed_rank"] == 2
    assert df.loc[1, "ewm_speed_rank"] == 3


def test_percentile_higher_means_faster():
    horses = [
        make_horse(post=1, pp_lines=[make_pp(speed_figure=50.0)]),
        make_horse(post=2, pp_lines=[make_pp(speed_figure=110.0)]),
    ]
    race = make_race(horses=horses)
    df = build_speed_feature_frame(race)
    assert df.loc[2, "ewm_speed_pct"] > df.loc[1, "ewm_speed_pct"]


def test_constant_field_zscore_is_zero():
    """If every horse has the same speed figure, z-scores should all be 0 (no NaN)."""
    horses = [make_horse(post=i + 1, pp_lines=[make_pp(speed_figure=90.0)]) for i in range(4)]
    race = make_race(horses=horses)
    df = build_speed_feature_frame(race)
    assert (df["ewm_speed_zscore"] == 0.0).all()
