"""Tests for app/services/feature_engineering/class_features.py."""

from __future__ import annotations

import math

from app.schemas.race import RaceType
from app.services.feature_engineering.class_features import (
    build_class_feature_frame,
    horse_class_summary,
)
from tests.test_features._fixtures import make_horse, make_pp, make_race


# ── horse_class_summary ──────────────────────────────────────────────────────


def test_class_summary_empty_pp_returns_all_none():
    horse = make_horse(pp_lines=[])
    summary = horse_class_summary(
        horse,
        today_race_type=RaceType.CLAIMING,
        today_claiming_price=20_000,
        today_purse=30_000,
    )
    assert summary["avg_recent_claiming"] is None
    assert summary["avg_recent_purse"] is None
    assert summary["claiming_price_delta"] is None
    assert summary["purse_delta"] is None
    assert summary["class_trajectory"] is None
    assert summary["dropping_in_class"] is None


def test_claiming_to_claiming_uses_claiming_delta():
    pps = [
        make_pp(claiming_price=40_000, race_type=RaceType.CLAIMING),
        make_pp(claiming_price=50_000, race_type=RaceType.CLAIMING),
    ]
    horse = make_horse(pp_lines=pps)
    summary = horse_class_summary(
        horse,
        today_race_type=RaceType.CLAIMING,
        today_claiming_price=30_000,
        today_purse=20_000,
    )
    assert summary["avg_recent_claiming"] == 45_000
    assert summary["claiming_price_delta"] == 30_000 - 45_000
    assert summary["class_trajectory"] == -15_000
    assert summary["dropping_in_class"] is True


def test_climbing_claiming_class_is_not_dropping():
    pps = [make_pp(claiming_price=20_000, race_type=RaceType.CLAIMING)]
    horse = make_horse(pp_lines=pps)
    summary = horse_class_summary(
        horse,
        today_race_type=RaceType.CLAIMING,
        today_claiming_price=40_000,
        today_purse=50_000,
    )
    assert summary["class_trajectory"] == 20_000
    assert summary["dropping_in_class"] is False


def test_non_claiming_today_falls_back_to_purse_delta():
    pps = [
        make_pp(purse=20_000, race_type=RaceType.ALLOWANCE),
        make_pp(purse=30_000, race_type=RaceType.ALLOWANCE),
    ]
    horse = make_horse(pp_lines=pps)
    summary = horse_class_summary(
        horse,
        today_race_type=RaceType.ALLOWANCE,
        today_claiming_price=None,
        today_purse=60_000,
    )
    assert summary["claiming_price_delta"] is None
    assert summary["purse_delta"] == 60_000 - 25_000
    assert summary["class_trajectory"] == 35_000
    assert summary["dropping_in_class"] is False


def test_race_type_change_same_label():
    pps = [make_pp(race_type=RaceType.ALLOWANCE) for _ in range(3)]
    horse = make_horse(pp_lines=pps)
    summary = horse_class_summary(
        horse,
        today_race_type=RaceType.ALLOWANCE,
        today_claiming_price=None,
        today_purse=40_000,
    )
    assert summary["race_type_change"] == "same"


def test_race_type_change_records_transition():
    pps = [make_pp(race_type=RaceType.CLAIMING) for _ in range(3)]
    horse = make_horse(pp_lines=pps)
    summary = horse_class_summary(
        horse,
        today_race_type=RaceType.ALLOWANCE,
        today_claiming_price=None,
        today_purse=40_000,
    )
    assert summary["race_type_change"] == "claiming->allowance"


def test_purse_delta_window_is_average_of_recent_pps():
    pps = [
        make_pp(days_ago=10, purse=20_000),
        make_pp(days_ago=40, purse=40_000),
    ]
    horse = make_horse(pp_lines=pps)
    summary = horse_class_summary(
        horse,
        today_race_type=RaceType.ALLOWANCE,
        today_claiming_price=None,
        today_purse=50_000,
    )
    assert summary["avg_recent_purse"] == 30_000
    assert summary["purse_delta"] == 20_000


# ── build_class_feature_frame ────────────────────────────────────────────────


def test_class_frame_zscore_zero_mean_within_race():
    pps_low = [make_pp(claiming_price=15_000, race_type=RaceType.CLAIMING)]
    pps_mid = [make_pp(claiming_price=25_000, race_type=RaceType.CLAIMING)]
    pps_high = [make_pp(claiming_price=35_000, race_type=RaceType.CLAIMING)]

    horses = [
        make_horse(post=1, pp_lines=pps_low),
        make_horse(post=2, pp_lines=pps_mid),
        make_horse(post=3, pp_lines=pps_high),
    ]
    race = make_race(horses=horses, race_type=RaceType.CLAIMING)
    # Force today to be a claiming race with a fixed price
    race.header.race_type = RaceType.CLAIMING
    race.header.claiming_price = 20_000

    df = build_class_feature_frame(race)
    assert math.isclose(df["class_trajectory_zscore"].mean(), 0.0, abs_tol=1e-12)
    # Horse 1 has the smallest "drop" from its lower-class recent races; horse 3
    # has the biggest drop. So horse 3 should have the most-negative trajectory.
    assert df.loc[3, "class_trajectory"] < df.loc[1, "class_trajectory"]
