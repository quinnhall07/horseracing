"""
tests/test_parser/test_cleaner.py
─────────────────────────────────
Tests for `app.services.pdf_parser.cleaner`.

Every public function in cleaner.py is exercised against documented input
shapes drawn from the docstrings and the Brisnet UP format spec in CLAUDE.md.
"""

from __future__ import annotations

import pytest

from app.services.pdf_parser.cleaner import (
    clean_name,
    extract_claiming_price,
    extract_first_number,
    normalize_text,
    parse_condition,
    parse_distance_to_furlongs,
    parse_odds_to_decimal,
    parse_race_type,
    parse_surface,
    parse_time_to_seconds,
)


# ──────────────────────────────────────────────────────────────────────────────
# normalize_text
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizeText:
    def test_empty_input_returns_empty_string(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore[arg-type]

    def test_caps_pathological_whitespace_runs(self):
        # 3+ space runs collapse to exactly 2 spaces. This preserves the
        # columnar gap downstream regex parsers rely on as a column separator.
        assert normalize_text("hello    world") == "hello  world"
        assert normalize_text("hello          world") == "hello  world"

    def test_preserves_single_and_double_spaces(self):
        # 1 and 2 space gaps survive unchanged.
        assert normalize_text("a b") == "a b"
        assert normalize_text("a  b") == "a  b"

    def test_preserves_newlines(self):
        # Newlines must survive — they are layout-significant in PDF text.
        assert "\n" in normalize_text("line1\nline2")

    def test_strips_control_characters(self):
        assert normalize_text("foo\x00bar\x07") == "foobar"

    def test_keeps_tab_and_newline(self):
        out = normalize_text("a\tb\nc")
        assert "\t" in out and "\n" in out

    def test_maps_smart_quotes_to_ascii(self):
        assert normalize_text("‘Big Brown’") == "'Big Brown'"
        assert normalize_text("“Quoted”") == '"Quoted"'

    def test_maps_em_and_en_dash_to_hyphen(self):
        assert normalize_text("Smarty–Jones") == "Smarty-Jones"
        assert normalize_text("Big—Brown") == "Big-Brown"

    def test_maps_unicode_fractions_to_ascii(self):
        # Distance strings often contain ½ / ¼ / ¾
        assert normalize_text("5½f") == "51/2f"
        assert normalize_text("¼Mile") == "1/4Mile"

    def test_strips_combining_accents(self):
        # "Bébé" → "Bebe"
        assert normalize_text("Bébé") == "Bebe"

    def test_trims_leading_trailing_whitespace(self):
        assert normalize_text("   foo   ") == "foo"


# ──────────────────────────────────────────────────────────────────────────────
# clean_name
# ──────────────────────────────────────────────────────────────────────────────


class TestCleanName:
    def test_title_cases_all_caps_horse_name(self):
        assert clean_name("BIG BROWN") == "Big Brown"

    def test_keeps_mixed_case_name(self):
        # Mixed case names are not aggressively re-cased; only ALL CAPS triggers.
        assert clean_name("Big Brown") == "Big Brown"

    def test_strips_trailing_apprentice_marker(self):
        # Brisnet marks apprentice jockeys with trailing "*" or "**"
        assert clean_name("Smith, J*") == "Smith, J"
        assert clean_name("Smith, J**") == "Smith, J"

    def test_normalizes_smart_quotes_before_titling(self):
        # Apostrophe in O'Brien must survive normalization
        assert clean_name("O’BRIEN, A") == "O'Brien, A"


# ──────────────────────────────────────────────────────────────────────────────
# parse_odds_to_decimal
# ──────────────────────────────────────────────────────────────────────────────


class TestParseOddsToDecimal:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("3-1", 4.0),
            ("5/2", 3.5),
            ("9-5", 2.8),
            ("even", 2.0),
            ("Evens", 2.0),
            ("EVEN", 2.0),
            ("4.50", 4.50),
            ("1.00", 1.00),
            ("100-1", 101.0),
        ],
    )
    def test_known_fractional_and_decimal_conversions(self, raw, expected):
        assert parse_odds_to_decimal(raw) == pytest.approx(expected)

    def test_empty_returns_none(self):
        assert parse_odds_to_decimal("") is None

    def test_garbage_returns_none(self):
        assert parse_odds_to_decimal("xyz") is None

    def test_division_by_zero_returns_none(self):
        assert parse_odds_to_decimal("5/0") is None

    def test_decimal_below_one_rejected(self):
        # Decimal odds must be ≥ 1.0 — anything below is non-sensical.
        assert parse_odds_to_decimal("0.50") is None

    def test_decimal_above_thousand_rejected(self):
        # Cap at 1001 = 1000-1 longshot.
        assert parse_odds_to_decimal("2000") is None


# ──────────────────────────────────────────────────────────────────────────────
# parse_distance_to_furlongs
# ──────────────────────────────────────────────────────────────────────────────


class TestParseDistanceToFurlongs:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("6 Furlongs", 6.0),
            ("6f", 6.0),
            ("5f", 5.0),
            ("1 Mile", 8.0),
            ("1 1/16 Miles", 8.5),       # 1.0625 × 8 = 8.5
            ("1 1/8 Miles", 9.0),         # 1.125 × 8 = 9.0
            ("1 3/16 Miles", 9.5),        # 1.1875 × 8 = 9.5
            ("1 1/4 Miles", 10.0),
            ("1 1/2 Miles", 12.0),
            ("5½f", 5.5),            # unicode ½
            ("About 5f", 5.0),
        ],
    )
    def test_standard_distance_strings(self, raw, expected):
        assert parse_distance_to_furlongs(raw) == pytest.approx(expected)

    def test_mile_and_yards_two_segment(self):
        # "1 Mile 70 Yards" → 8 furlongs + 70/220 furlongs ≈ 8.318
        result = parse_distance_to_furlongs("1 Mile 70 Yards")
        assert result == pytest.approx(8.0 + 70 / 220.0, abs=1e-3)

    def test_empty_returns_none(self):
        assert parse_distance_to_furlongs("") is None

    def test_no_unit_returns_none(self):
        # Unit is required by the regex.
        assert parse_distance_to_furlongs("6") is None

    def test_unrecognized_format_returns_none(self):
        assert parse_distance_to_furlongs("six furlongs") is None  # word numeral


# ──────────────────────────────────────────────────────────────────────────────
# parse_time_to_seconds
# ──────────────────────────────────────────────────────────────────────────────


class TestParseTimeToSeconds:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (":22.2", 22.20),
            ("22.20", 22.20),
            ("1:10.40", 70.40),
            ("2:02.00", 122.00),
            (":45.05", 45.05),
            ("1:35.55", 95.55),
        ],
    )
    def test_known_time_strings(self, raw, expected):
        assert parse_time_to_seconds(raw) == pytest.approx(expected, abs=1e-3)

    def test_single_digit_hundredths_pads_right(self):
        # ":22.2" must be 22.20s, not 22.02s.  This is a common bug source.
        assert parse_time_to_seconds(":22.2") == pytest.approx(22.20, abs=1e-3)

    def test_empty_returns_none(self):
        assert parse_time_to_seconds("") is None

    def test_garbage_returns_none(self):
        assert parse_time_to_seconds("abc") is None


# ──────────────────────────────────────────────────────────────────────────────
# parse_surface
# ──────────────────────────────────────────────────────────────────────────────


class TestParseSurface:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("d", "dirt"),
            ("D", "dirt"),
            ("dirt", "dirt"),
            ("Dirt", "dirt"),
            ("t", "turf"),
            ("Turf", "turf"),
            ("grass", "turf"),
            ("s", "synthetic"),
            ("synthetic", "synthetic"),
            ("a", "synthetic"),  # All-weather → synthetic
        ],
    )
    def test_known_surface_codes(self, raw, expected):
        assert parse_surface(raw) == expected

    def test_unknown_returns_unknown(self):
        assert parse_surface("xyz") == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# parse_condition
# ──────────────────────────────────────────────────────────────────────────────


class TestParseCondition:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ft", "fast"),
            ("fast", "fast"),
            ("Fast", "fast"),
            ("gd", "good"),
            ("sl", "sloppy"),
            ("my", "muddy"),
            ("fm", "firm"),
            ("yl", "yielding"),
            ("sf", "soft"),
            ("hy", "heavy"),
            ("fr", "frozen"),
        ],
    )
    def test_known_condition_codes(self, raw, expected):
        assert parse_condition(raw) == expected

    def test_unknown_returns_unknown(self):
        assert parse_condition("xyz") == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# parse_race_type
# ──────────────────────────────────────────────────────────────────────────────


class TestParseRaceType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Maiden Claiming $20,000", "maiden_claiming"),
            ("Maiden Special Weight", "maiden_special_weight"),
            ("MSW for 3yo", "maiden_special_weight"),
            ("Allowance Optional Claiming", "allowance_optional_claiming"),
            ("AOC", "allowance_optional_claiming"),
            ("Allowance", "allowance"),
            ("ALW", "allowance"),
            ("Handicap", "handicap"),
            ("Grade 1 Stakes", "graded_stakes"),
            ("Graded Stakes", "graded_stakes"),
            ("Stakes", "stakes"),
            ("Claiming $20,000", "claiming"),
        ],
    )
    def test_known_race_types(self, raw, expected):
        assert parse_race_type(raw) == expected

    def test_unknown_returns_unknown(self):
        assert parse_race_type("Some random text") == "unknown"

    def test_maiden_claiming_takes_priority_over_claiming(self):
        # If both patterns could match, maiden_claiming must win (it's checked first).
        assert parse_race_type("Maiden Claiming for 2yo") == "maiden_claiming"

    def test_graded_takes_priority_over_stakes(self):
        # "Graded Stakes" must resolve to graded_stakes, not stakes.
        assert parse_race_type("Grade 2 Stakes") == "graded_stakes"


# ──────────────────────────────────────────────────────────────────────────────
# extract_first_number
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractFirstNumber:
    def test_pulls_integer(self):
        assert extract_first_number("12 lbs") == 12.0

    def test_pulls_float(self):
        assert extract_first_number("20.5 lengths") == 20.5

    def test_pulls_negative(self):
        assert extract_first_number("-3.5") == -3.5

    def test_returns_first_only(self):
        assert extract_first_number("3 of 5 horses") == 3.0

    def test_no_number_returns_none(self):
        assert extract_first_number("no numbers here") is None

    def test_empty_returns_none(self):
        assert extract_first_number("") is None


# ──────────────────────────────────────────────────────────────────────────────
# extract_claiming_price
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractClaimingPrice:
    def test_dollar_sign_and_commas_stripped(self):
        assert extract_claiming_price("$20,000") == 20000.0

    def test_bare_integer(self):
        assert extract_claiming_price("15000") == 15000.0

    def test_with_label(self):
        # "Clm 15000" → 15000
        assert extract_claiming_price("Clm 15000") == 15000.0

    def test_decimal_preserved(self):
        assert extract_claiming_price("$1,234.56") == pytest.approx(1234.56)

    def test_empty_returns_none(self):
        assert extract_claiming_price("") is None
