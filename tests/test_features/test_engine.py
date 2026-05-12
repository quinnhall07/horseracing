"""Integration tests for the FeatureEngine orchestrator."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.schemas.race import (
    HorseEntry,
    ParsedRace,
    PastPerformanceLine,
    RaceCard,
    RaceHeader,
    RaceType,
    Surface,
    TrackCondition,
)
from app.services.feature_engineering import FeatureEngine
from tests.test_features._fixtures import make_card, make_horse, make_pp, make_race


def test_empty_card_returns_empty_dataframe():
    card = RaceCard(source_filename="x.pdf", source_format="brisnet_up")
    df = FeatureEngine().transform(card)
    assert df.empty


def test_transform_single_race_has_one_row_per_horse():
    horses = [make_horse(post=i + 1) for i in range(5)]
    race = make_race(horses=horses)
    card = make_card(races=[race])
    df = FeatureEngine().transform(card)
    assert len(df) == 5
    # Primary key columns present.
    assert "race_number" in df.columns
    assert "post_position" in df.columns
    assert set(df["post_position"]) == {1, 2, 3, 4, 5}


def test_transform_attaches_layoff_fitness():
    horse = make_horse(pp_lines=[make_pp(days_ago=10)])
    race = make_race(horses=[horse])
    df = FeatureEngine().transform(make_card(races=[race]))
    assert "layoff_fitness" in df.columns
    assert df.loc[0, "layoff_fitness"] == 1.0  # within recovery threshold


def test_transform_includes_all_feature_module_columns():
    df = FeatureEngine().transform(make_card())
    # Pick representative columns from each module to confirm joins worked.
    must_have = {
        "ewm_speed_figure", "ewm_speed_rank", "ewm_speed_zscore",
        "pace_style_score", "pace_pressure_index", "early_speed_zscore",
        "class_trajectory", "class_trajectory_zscore",
        "today_jt_same_pair", "jockey_repeat_streak",
        "layoff_fitness", "days_since_last",
        "weight_lbs_delta",
    }
    missing = must_have - set(df.columns)
    assert not missing, f"missing columns: {missing}"


def test_transform_multi_race_card():
    r1 = make_race(race_number=1, horses=[make_horse(post=p + 1) for p in range(4)])
    r2 = make_race(race_number=2, horses=[make_horse(post=p + 1) for p in range(6)])
    df = FeatureEngine().transform(make_card(races=[r1, r2]))
    assert df["race_number"].nunique() == 2
    assert len(df) == 4 + 6


def test_transform_propagates_race_header_metadata():
    race = make_race(distance=6.0, surface=Surface.DIRT)
    df = FeatureEngine().transform(make_card(races=[race]))
    assert (df["distance_furlongs"] == 6.0).all()
    assert (df["surface"] == "dirt").all()
    assert df["is_sprint"].all()


def test_weight_lbs_delta_zero_mean_within_race():
    horses = [
        make_horse(post=1, weight=115.0),
        make_horse(post=2, weight=120.0),
        make_horse(post=3, weight=125.0),
    ]
    race = make_race(horses=horses)
    df = FeatureEngine().transform(make_card(races=[race]))
    assert abs(df["weight_lbs_delta"].mean()) < 1e-9


def test_days_since_last_handles_first_time_starters():
    # First-time starter: no PP lines
    fts = HorseEntry(
        horse_name="DEBUT",
        post_position=1,
        morning_line_odds=10.0,
        weight_lbs=120.0,
        pp_lines=[],
    )
    veteran = make_horse(post=2, pp_lines=[make_pp(days_ago=20)])
    race = make_race(horses=[fts, veteran])
    df = FeatureEngine().transform(make_card(races=[race]))

    fts_row = df[df["post_position"] == 1].iloc[0]
    vet_row = df[df["post_position"] == 2].iloc[0]
    assert pd.isna(fts_row["days_since_last"])
    assert vet_row["days_since_last"] == 20
    # FTS layoff_fitness uses the sentinel, not 1.0.
    assert 0 < fts_row["layoff_fitness"] < 1
    assert vet_row["layoff_fitness"] == 1.0


def test_one_bad_race_does_not_kill_the_card():
    """If a race blows up inside transform_race, the engine logs and skips it."""
    good_race = make_race(race_number=1)

    # Build a race that will break feature math — empty entries list is enough to
    # break some of the per-module frames (set ops on empty DataFrame).
    bad_header = RaceHeader(
        race_number=2,
        race_date=date.today(),
        track_code="CD",
        distance_furlongs=6.0,
        distance_raw="6 Furlongs",
        surface=Surface.DIRT,
        condition=TrackCondition.FAST,
        race_type=RaceType.ALLOWANCE,
    )
    bad_race = ParsedRace(header=bad_header, entries=[], parse_confidence=0.0)

    df = FeatureEngine().transform(make_card(races=[good_race, bad_race]))
    # Good race still produces output.
    assert set(df["race_number"]) == {1}
