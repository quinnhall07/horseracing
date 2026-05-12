"""Quality gate scoring tests — DATA_PIPELINE.md §8 rubric."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts.db.constants import ROW_MIN_SCORE
from scripts.db.quality_gate import race_level_issues, score_row


def _full_row() -> dict:
    """A row that scores 1.0 — every field present and in range."""
    return {
        "race_race_date":          date(2026, 5, 10),
        "race_track_code":         "CD",
        "race_race_number":        4,
        "race_distance_furlongs":  4.5,
        "race_surface":            "dirt",
        "race_jurisdiction":       "US",
        "horse_name_normalized":   "lovely words",
        "horse_name_display":      "Lovely Words",
        "finish_position":         3,
        "post_position":           1,
        "weight_lbs":              123.0,
        "odds_final":              4.0,
        "speed_figure":            71.0,
        "fraction_finish_sec":     71.6,
        "jockey_name_normalized":  "danilo grisales rave",
    }


# ─── hard failures (return 0.0 immediately) ───────────────────────────────

@pytest.mark.parametrize("missing_field", [
    "race_race_date", "race_track_code", "horse_name_normalized",
    "finish_position", "race_race_number", "race_distance_furlongs",
    "race_surface", "race_jurisdiction",
])
def test_hard_failure_zero_score(missing_field):
    row = _full_row()
    row[missing_field] = None
    score, issues = score_row(row)
    assert score == 0.0
    assert any(missing_field.replace("race_", "").replace("horse_", "") in i.lower()
               or "track_code" in i.lower() or "horse_name" in i.lower()
               for i in issues), issues


# ─── soft penalties ───────────────────────────────────────────────────────

def test_full_row_scores_one():
    score, issues = score_row(_full_row())
    assert score == 1.0
    assert issues == []


def test_missing_odds_subtracts_010():
    row = _full_row()
    row["odds_final"] = None
    score, issues = score_row(row)
    assert score == pytest.approx(0.90)
    assert "missing odds" in issues


def test_missing_speed_figure_subtracts_010():
    row = _full_row()
    row["speed_figure"] = None
    score, _ = score_row(row)
    assert score == pytest.approx(0.90)


def test_missing_jockey_subtracts_005():
    row = _full_row()
    row["jockey_name_normalized"] = None
    score, _ = score_row(row)
    assert score == pytest.approx(0.95)


def test_missing_weight_subtracts_005():
    row = _full_row()
    row["weight_lbs"] = None
    score, _ = score_row(row)
    assert score == pytest.approx(0.95)


def test_missing_final_time_subtracts_010():
    row = _full_row()
    row["fraction_finish_sec"] = None
    score, _ = score_row(row)
    assert score == pytest.approx(0.90)


# ─── range checks ─────────────────────────────────────────────────────────

def test_invalid_finish_position_subtracts_015():
    row = _full_row()
    row["finish_position"] = 99
    score, issues = score_row(row)
    assert score == pytest.approx(0.85)
    assert any("finish_position" in i for i in issues)


def test_implausible_distance_subtracts_020():
    row = _full_row()
    row["race_distance_furlongs"] = 25.0  # too long for any real flat race
    score, issues = score_row(row)
    assert score == pytest.approx(0.80)
    assert any("distance" in i for i in issues)


def test_invalid_odds_subtracts_015():
    row = _full_row()
    row["odds_final"] = 0.5  # decimal odds < 1.0 means negative implied prob
    score, _ = score_row(row)
    assert score == pytest.approx(0.85)


def test_implausible_weight_subtracts_010():
    row = _full_row()
    row["weight_lbs"] = 200.0  # absurd
    score, _ = score_row(row)
    assert score == pytest.approx(0.90)


def test_score_clamps_to_zero():
    """Stack enough penalties — final score must not go negative."""
    row = _full_row()
    row["odds_final"]              = None
    row["speed_figure"]            = None
    row["jockey_name_normalized"]  = None
    row["weight_lbs"]              = None
    row["fraction_finish_sec"]    = None
    row["finish_position"]         = 99       # also bad
    row["race_distance_furlongs"]  = 1.0      # also bad
    row["odds_final"]              = 0.5      # also bad
    row["weight_lbs"]              = 200.0    # also bad
    score, _ = score_row(row)
    assert score >= 0.0
    assert score < ROW_MIN_SCORE


# ─── cross-row checks ─────────────────────────────────────────────────────

def test_race_level_no_problems_when_clean():
    df = pd.DataFrame([
        {"race_race_date": date(2026, 5, 10), "race_track_code": "CD",
         "race_race_number": 4, "race_distance_furlongs": 4.5, "race_surface": "dirt",
         "horse_name_normalized": "horse a", "finish_position": 1},
        {"race_race_date": date(2026, 5, 10), "race_track_code": "CD",
         "race_race_number": 4, "race_distance_furlongs": 4.5, "race_surface": "dirt",
         "horse_name_normalized": "horse b", "finish_position": 2},
    ])
    assert race_level_issues(df) == {}


def test_race_level_detects_multiple_winners():
    df = pd.DataFrame([
        {"race_race_date": date(2026, 5, 10), "race_track_code": "CD",
         "race_race_number": 4, "race_distance_furlongs": 4.5, "race_surface": "dirt",
         "horse_name_normalized": "horse a", "finish_position": 1},
        {"race_race_date": date(2026, 5, 10), "race_track_code": "CD",
         "race_race_number": 4, "race_distance_furlongs": 4.5, "race_surface": "dirt",
         "horse_name_normalized": "horse b", "finish_position": 1},
    ])
    issues = race_level_issues(df)
    assert len(issues) == 1
    assert any("multiple winners" in s for s in next(iter(issues.values())))


def test_race_level_treats_different_distances_as_different_races():
    """Same (date, track, race_num) at different distances → 2 separate races,
    NOT a 'mixed distances' violation. Argentine venues that run turf + dirt
    cards in parallel use the same nro for distinct physical races, and the
    dedup_key (which includes distance + surface) correctly stores them as
    separate `races` rows. Quality_gate must not falsely flag this.
    """
    df = pd.DataFrame([
        {"race_race_date": date(2026, 5, 10), "race_track_code": "AR",
         "race_race_number": 5, "race_distance_furlongs": 5.0, "race_surface": "turf",
         "horse_name_normalized": "horse a", "finish_position": 1},
        {"race_race_date": date(2026, 5, 10), "race_track_code": "AR",
         "race_race_number": 5, "race_distance_furlongs": 7.0, "race_surface": "dirt",
         "horse_name_normalized": "horse b", "finish_position": 1},
    ])
    issues = race_level_issues(df)
    # No violation — these are two separate races sharing nro=5
    assert issues == {}


def test_race_level_detects_duplicate_horse():
    df = pd.DataFrame([
        {"race_race_date": date(2026, 5, 10), "race_track_code": "CD",
         "race_race_number": 4, "race_distance_furlongs": 4.5, "race_surface": "dirt",
         "horse_name_normalized": "horse a", "finish_position": 1},
        {"race_race_date": date(2026, 5, 10), "race_track_code": "CD",
         "race_race_number": 4, "race_distance_furlongs": 4.5, "race_surface": "dirt",
         "horse_name_normalized": "horse a", "finish_position": 5},
    ])
    issues = race_level_issues(df)
    assert any("duplicate horses" in s for s in next(iter(issues.values())))
