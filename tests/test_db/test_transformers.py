"""Unit conversion tests — every transformer must return None on garbage."""

from __future__ import annotations

import pytest

from scripts.db.transformers import (
    TRANSFORMERS,
    gbp_to_usd,
    hk_going_to_condition,
    hkd_to_usd,
    metres_to_furlongs,
    normalize_condition,
    normalize_surface,
    parse_odds_to_decimal,
    stones_to_lbs,
    time_string_to_seconds,
    uk_distance_to_furlongs,
    uk_going_to_condition,
    uk_going_to_surface,
    uk_sp_to_decimal,
)


# ─── distance ────────────────────────────────────────────────────────────

def test_metres_to_furlongs():
    assert metres_to_furlongs(1200) == pytest.approx(5.965, rel=1e-3)
    assert metres_to_furlongs("1600") == pytest.approx(7.954, rel=1e-3)
    assert metres_to_furlongs(None) is None
    assert metres_to_furlongs("nonsense") is None


def test_uk_distance_to_furlongs_basic():
    assert uk_distance_to_furlongs("5f") == 5.0
    assert uk_distance_to_furlongs("1m") == 8.0
    assert uk_distance_to_furlongs("1m2f") == 10.0
    assert uk_distance_to_furlongs("2m") == 16.0


def test_uk_distance_to_furlongs_with_yards():
    """6 furlongs + 110 yards = 6.5 furlongs."""
    assert uk_distance_to_furlongs("6f 110y") == pytest.approx(6.5, abs=1e-3)


def test_uk_distance_passthrough_numeric():
    assert uk_distance_to_furlongs(5.0) == 5.0
    assert uk_distance_to_furlongs("7.5") == 7.5


def test_uk_distance_returns_none_on_garbage():
    assert uk_distance_to_furlongs(None) is None
    assert uk_distance_to_furlongs("") is None


# ─── weight ──────────────────────────────────────────────────────────────

def test_stones_to_lbs():
    assert stones_to_lbs("9-0") == 126.0    # 9 * 14
    assert stones_to_lbs("9-2") == 128.0
    assert stones_to_lbs(123) == 123.0       # numeric passthrough
    assert stones_to_lbs(None) is None
    assert stones_to_lbs("nope") is None


# ─── odds ────────────────────────────────────────────────────────────────

def test_parse_odds_fractional_to_decimal():
    """5/2 → 3.5 (numerator/denominator + 1)."""
    assert parse_odds_to_decimal("5/2") == 3.5
    assert parse_odds_to_decimal("7/4") == 2.75
    assert parse_odds_to_decimal("100/30") == pytest.approx(4.333, abs=1e-3)


def test_parse_odds_evens():
    assert parse_odds_to_decimal("EVS") == 2.0
    assert parse_odds_to_decimal("EVENS") == 2.0


def test_parse_odds_decimal_passthrough():
    assert parse_odds_to_decimal(4.0) == 4.0
    assert parse_odds_to_decimal("3.75") == 3.75


def test_parse_odds_us_hyphenated():
    """3-1 → 4.0; same as fractional-style."""
    assert parse_odds_to_decimal("3-1") == 4.0


def test_parse_odds_invalid_returns_none():
    assert parse_odds_to_decimal(None) is None
    assert parse_odds_to_decimal("") is None
    assert parse_odds_to_decimal("scratched") is None
    assert parse_odds_to_decimal("0/0") is None  # zero denominator
    assert parse_odds_to_decimal(-1.0) is None   # non-positive


def test_uk_sp_strips_favourite_marker():
    assert uk_sp_to_decimal("7/4 F") == 2.75
    assert uk_sp_to_decimal("5/2 J-F") == 3.5


# ─── surface / condition ─────────────────────────────────────────────────

def test_normalize_surface():
    assert normalize_surface("D") == "dirt"
    assert normalize_surface("Turf") == "turf"
    assert normalize_surface("AW") == "synthetic"
    assert normalize_surface("Tapeta") == "synthetic"
    assert normalize_surface(None) is None


def test_normalize_condition():
    assert normalize_condition("FT") == "fast"
    assert normalize_condition("good") == "good"
    assert normalize_condition("SY") == "sloppy"
    assert normalize_condition(None) is None


def test_uk_going_to_surface_default_turf():
    assert uk_going_to_surface("Good", "ASC") == "turf"


def test_uk_going_to_surface_aw_track():
    assert uk_going_to_surface("Good", "LIN") == "synthetic"
    assert uk_going_to_surface("Standard", None) == "synthetic"


def test_uk_going_to_condition_buckets():
    assert uk_going_to_condition("Good to Firm") == "good"
    assert uk_going_to_condition("Heavy") == "heavy"
    assert uk_going_to_condition("Standard") == "fast"
    assert uk_going_to_condition(None) is None


def test_hk_going_to_condition_buckets():
    assert hk_going_to_condition("Good") == "good"
    assert hk_going_to_condition("Yielding") == "yielding"
    assert hk_going_to_condition("Heavy") == "heavy"


# ─── time ────────────────────────────────────────────────────────────────

def test_time_string_to_seconds_with_minutes():
    assert time_string_to_seconds("1:10.40") == 70.40
    assert time_string_to_seconds("1:11.6") == 71.6


def test_time_string_to_seconds_seconds_only():
    assert time_string_to_seconds("70.40") == 70.40
    assert time_string_to_seconds(":22.4") == 22.4


def test_time_string_returns_none_on_garbage():
    assert time_string_to_seconds(None) is None
    assert time_string_to_seconds("") is None
    assert time_string_to_seconds(0) is None  # non-positive


# ─── money ───────────────────────────────────────────────────────────────

def test_gbp_to_usd_uses_fixed_rate():
    assert gbp_to_usd(100) == 127.0
    assert gbp_to_usd(None) is None
    assert gbp_to_usd("nope") is None


def test_hkd_to_usd_uses_fixed_rate():
    assert hkd_to_usd(1000) == 128.0
    assert hkd_to_usd(None) is None


# ─── registry ────────────────────────────────────────────────────────────

def test_transformers_registry_has_no_orphans():
    """Every callable in TRANSFORMERS dict must actually exist as a fn."""
    for name, fn in TRANSFORMERS.items():
        assert callable(fn), f"{name} not callable"


def test_field_maps_only_reference_registered_transformers():
    from scripts.db.field_maps import FIELD_MAPS
    for slug, fm in FIELD_MAPS.items():
        for canonical, transformer_name in fm.get("transformers", {}).items():
            assert transformer_name in TRANSFORMERS, (
                f"field_maps[{slug}] references unknown transformer "
                f"{transformer_name!r} for {canonical}"
            )
