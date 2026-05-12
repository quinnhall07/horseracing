"""Tests for app/services/feature_engineering/pace_features.py."""

from __future__ import annotations

import math

import pandas as pd

from app.services.feature_engineering.pace_features import (
    FRONT_RUNNER_LENGTHS_THRESHOLD,
    build_pace_feature_frame,
    fraction_ratios,
    horse_pace_summary,
    pace_shape_metrics,
)
from tests.test_features._fixtures import make_horse, make_pp, make_race


# ── pace_shape_metrics ───────────────────────────────────────────────────────


def test_pace_shape_handles_missing_inputs_gracefully():
    pp = make_pp(beaten_lengths_q1=None, beaten_lengths_q2=None)
    m = pace_shape_metrics(pp)
    assert m["early_speed"] is None
    assert m["late_kick"] is None


def test_pace_shape_early_speed_is_negative_of_beaten_lengths_q1():
    pp = make_pp(beaten_lengths_q1=2.5)
    m = pace_shape_metrics(pp)
    assert m["early_speed"] == -2.5


def test_pace_shape_late_kick_is_q2_minus_finish():
    pp = make_pp(beaten_lengths_q1=1.0, beaten_lengths_q2=3.0)
    # Default fixture finishes 3rd, so lengths_behind=2.0; late_kick = 3 - 2 = 1.0
    m = pace_shape_metrics(pp)
    assert m["late_kick"] == 1.0


def test_pace_shape_late_kick_can_be_negative_for_pace_collapsers():
    # Was 1L behind at the second call but finished 6L behind → big collapse.
    pp = make_pp(beaten_lengths_q2=1.0, finish=5)
    pp_copy = pp.model_copy(update={"lengths_behind": 6.0})
    m = pace_shape_metrics(pp_copy)
    assert m["late_kick"] == -5.0


# ── fraction_ratios ──────────────────────────────────────────────────────────


def test_fraction_ratios_basic_math():
    pp = make_pp(fraction_q1=22.0, fraction_q2=46.0, fraction_finish=71.5)
    r = fraction_ratios(pp)
    assert math.isclose(r["fraction_q1_ratio"], 22.0 / 46.0)
    # Mid = q2 - q1 = 24.0; ratio vs finish = 24/71.5
    assert math.isclose(r["fraction_q2_ratio"], 24.0 / 71.5)


def test_fraction_ratios_none_when_missing():
    pp = make_pp(fraction_q1=None, fraction_q2=None)
    r = fraction_ratios(pp)
    assert r["fraction_q1_ratio"] is None
    assert r["fraction_q2_ratio"] is None


# ── horse_pace_summary ───────────────────────────────────────────────────────


def test_horse_pace_summary_no_pp_returns_all_none():
    horse = make_horse(pp_lines=[])
    summary = horse_pace_summary(horse)
    assert summary["pace_style_score"] is None
    assert summary["early_speed_avg"] is None
    assert summary["late_kick_avg"] is None
    assert summary["n_pace_pps"] == 0.0


def test_horse_pace_summary_averages_over_recent_window():
    pps = [
        make_pp(days_ago=10, beaten_lengths_q1=0.5),
        make_pp(days_ago=40, beaten_lengths_q1=1.0),
        make_pp(days_ago=70, beaten_lengths_q1=1.5),
    ]
    horse = make_horse(pp_lines=pps)
    summary = horse_pace_summary(horse)
    assert math.isclose(summary["pace_style_score"], 1.0)
    assert math.isclose(summary["early_speed_avg"], -1.0)


def test_horse_pace_summary_skips_null_inputs():
    pps = [
        make_pp(days_ago=10, beaten_lengths_q1=None),
        make_pp(days_ago=40, beaten_lengths_q1=2.0),
    ]
    horse = make_horse(pp_lines=pps)
    summary = horse_pace_summary(horse)
    assert summary["pace_style_score"] == 2.0
    assert summary["n_pace_pps"] == 2.0  # window still counts all PPs


# ── build_pace_feature_frame ─────────────────────────────────────────────────


def test_pace_pressure_index_counts_front_runners():
    # Three horses, two of them habitual front-runners (low pace_style_score).
    front_pps = [make_pp(beaten_lengths_q1=0.5)]
    mid_pps = [make_pp(beaten_lengths_q1=3.0)]
    horses = [
        make_horse(post=1, pp_lines=front_pps),
        make_horse(post=2, pp_lines=front_pps),
        make_horse(post=3, pp_lines=mid_pps),
    ]
    race = make_race(horses=horses)
    df = build_pace_feature_frame(race)
    assert df["pace_pressure_index"].iloc[0] == 2.0


def test_pace_pressure_index_is_constant_across_rows():
    horses = [make_horse(post=i + 1) for i in range(5)]
    race = make_race(horses=horses)
    df = build_pace_feature_frame(race)
    assert df["pace_pressure_index"].nunique() == 1


def test_early_speed_zscore_zero_mean_within_race():
    horses = [
        make_horse(post=1, pp_lines=[make_pp(beaten_lengths_q1=0.0)]),
        make_horse(post=2, pp_lines=[make_pp(beaten_lengths_q1=2.0)]),
        make_horse(post=3, pp_lines=[make_pp(beaten_lengths_q1=4.0)]),
    ]
    race = make_race(horses=horses)
    df = build_pace_feature_frame(race)
    assert math.isclose(df["early_speed_zscore"].mean(), 0.0, abs_tol=1e-12)


def test_threshold_constant_is_documented():
    assert FRONT_RUNNER_LENGTHS_THRESHOLD == 1.5
