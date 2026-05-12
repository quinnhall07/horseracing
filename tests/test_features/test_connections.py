"""Tests for app/services/feature_engineering/connections.py."""

from __future__ import annotations

from app.services.feature_engineering.connections import (
    build_connection_feature_frame,
    horse_connections_summary,
)
from tests.test_features._fixtures import make_horse, make_pp, make_race


# ── horse_connections_summary ────────────────────────────────────────────────


def test_empty_pp_returns_safe_defaults():
    horse = make_horse(pp_lines=[])
    summary = horse_connections_summary(horse)
    assert summary["today_jt_same_pair"] is False
    assert summary["jockey_repeat_streak"] == 0.0
    assert summary["n_jockey_pps"] == 0.0
    assert summary["today_jockey_win_rate_in_pps"] is None
    assert summary["trainer_continuity"] is None


def test_today_jt_pair_same_jockey():
    pps = [make_pp(jockey="J Doe", finish=2), make_pp(jockey="A Other")]
    horse = make_horse(jockey="J Doe", pp_lines=pps)
    summary = horse_connections_summary(horse)
    assert summary["today_jt_same_pair"] is True
    assert summary["jockey_repeat_streak"] == 1.0


def test_jockey_streak_counts_only_consecutive_leaders():
    pps = [
        make_pp(jockey="J Doe", finish=2),
        make_pp(jockey="J Doe", finish=3),
        make_pp(jockey="Other"),  # breaks the streak
        make_pp(jockey="J Doe"),
    ]
    horse = make_horse(jockey="J Doe", pp_lines=pps)
    summary = horse_connections_summary(horse)
    assert summary["jockey_repeat_streak"] == 2.0
    # n_jockey_pps is total recent PPs the jockey rode, not just consecutive
    assert summary["n_jockey_pps"] == 3.0


def test_win_rate_with_today_jockey():
    pps = [
        make_pp(jockey="J Doe", finish=1),
        make_pp(jockey="J Doe", finish=4),
        make_pp(jockey="J Doe", finish=1),
        make_pp(jockey="Other"),
    ]
    horse = make_horse(jockey="J Doe", pp_lines=pps)
    summary = horse_connections_summary(horse)
    assert summary["n_jockey_pps"] == 3.0
    assert summary["today_jockey_win_rate_in_pps"] == 2 / 3


def test_case_and_whitespace_insensitive():
    pps = [make_pp(jockey=" j doe ")]
    horse = make_horse(jockey="J Doe", pp_lines=pps)
    summary = horse_connections_summary(horse)
    assert summary["today_jt_same_pair"] is True


def test_trainer_continuity_proxy_requires_trainer_and_matching_jockey():
    pps = [make_pp(jockey="J Doe")]
    horse_match = make_horse(jockey="J Doe", trainer="T Smith", pp_lines=pps)
    horse_mismatch = make_horse(jockey="A Other", trainer="T Smith", pp_lines=pps)
    horse_no_trainer = make_horse(jockey="J Doe", trainer=None, pp_lines=pps)

    assert horse_connections_summary(horse_match)["trainer_continuity"] == 1.0
    assert horse_connections_summary(horse_mismatch)["trainer_continuity"] == 0.0
    assert horse_connections_summary(horse_no_trainer)["trainer_continuity"] is None


def test_no_today_jockey_no_streak():
    pps = [make_pp(jockey="A Other")]
    horse = make_horse(jockey=None, pp_lines=pps)
    summary = horse_connections_summary(horse)
    assert summary["jockey_repeat_streak"] == 0.0
    assert summary["today_jt_same_pair"] is False


# ── build_connection_feature_frame ───────────────────────────────────────────


def test_frame_is_indexed_by_post_position():
    horses = [
        make_horse(post=1, jockey="J Doe", pp_lines=[make_pp(jockey="J Doe", finish=1)]),
        make_horse(post=2, jockey="Other"),
    ]
    race = make_race(horses=horses)
    df = build_connection_feature_frame(race)
    assert list(df.index) == [1, 2]
    assert df.loc[1, "today_jt_same_pair"]
    assert not df.loc[2, "today_jt_same_pair"]
