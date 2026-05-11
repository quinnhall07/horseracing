"""
tests/test_parser/test_brisnet_parser.py
─────────────────────────────────────────
Tests for `app.services.pdf_parser.brisnet_parser.BrisnetParser`.

Strategy: feed synthetic, line-formatted text that mirrors the column
layout pdfplumber produces from a Brisnet UP PDF, and assert on the
resulting RaceCard / ParsedRace / HorseEntry / PastPerformanceLine shape.

Real-PDF integration validation is deferred until Phase 1b — these
tests pin the parser's behavior against a controlled input fixture so
regex regressions are caught immediately.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.schemas.race import (
    RaceCard,
    RaceType,
    Surface,
    TrackCondition,
    PaceStyle,
)
from app.services.pdf_parser.brisnet_parser import BrisnetParser


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

# A single fully-populated synthetic race. Two horses, two PP lines each.
# Multi-space gaps (≥2) are preserved by cleaner.normalize_text and serve as
# column separators for the parser's _RE_HORSE_LINE / _RE_PP_LINE patterns.
SYNTHETIC_RACE_TEXT = """\
BRISNET ULTIMATE PAST PERFORMANCES

RACE 1
BEL  05/03/2025
6 Furlongs   (Dirt)   Fast
Maiden Claiming $20,000     Purse: $35,000

 1  BIG BROWN                 SMITH, J         120  3-1     L
    Trainer: Pletcher, T
04/15/25  BEL  6f  d  ft  Clm  7  3  SMITH, J  120  4.0  92  :22.40  :45.20  1:10.40
03/22/25  AQU  6f  d  ft  Clm  5  2  SMITH, J  120  3.5  88  :22.60  :45.80  1:11.00

 2  CIGAR                     JONES, T         118  5-2     L
    Trainer: Mott, B
04/20/25  BEL  6f  d  ft  Alw  3  1  JONES, T  118  2.5  95  :22.20  :45.00  1:10.20
03/29/25  AQU  6f  d  ft  Alw  2  4  JONES, T  118  4.0  85  :22.80  :45.90  1:11.20
"""


@pytest.fixture
def parser() -> BrisnetParser:
    return BrisnetParser()


# ──────────────────────────────────────────────────────────────────────────────
# Card-level structure
# ──────────────────────────────────────────────────────────────────────────────


class TestCardLevelStructure:
    def test_parse_returns_racecard(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT, source_filename="test.pdf")
        assert isinstance(card, RaceCard)

    def test_source_metadata_preserved(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT, source_filename="card_2025_05_03.pdf")
        assert card.source_filename == "card_2025_05_03.pdf"
        assert card.source_format == "brisnet_up"

    def test_card_contains_one_race(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.n_races == 1

    def test_card_date_propagated_from_race_header(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.card_date == date(2025, 5, 3)

    def test_track_code_propagated(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.track_code == "BEL"


# ──────────────────────────────────────────────────────────────────────────────
# Race header parsing
# ──────────────────────────────────────────────────────────────────────────────


class TestRaceHeader:
    def test_race_number_extracted(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.races[0].header.race_number == 1

    def test_distance_in_furlongs(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.races[0].header.distance_furlongs == pytest.approx(6.0)

    def test_distance_raw_preserved(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert "Furlongs" in card.races[0].header.distance_raw

    def test_surface_detected_from_paren(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.races[0].header.surface == Surface.DIRT

    def test_condition_detected(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.races[0].header.condition == TrackCondition.FAST

    def test_race_type_detected(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        # "Maiden Claiming $20,000" must beat the generic "Claiming" pattern.
        assert card.races[0].header.race_type == RaceType.MAIDEN_CLAIMING

    def test_purse_extracted(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.races[0].header.purse_usd == 35000.0

    def test_claiming_price_extracted(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        # "Maiden Claiming $20,000" → claiming price 20,000
        # Note: _RE_CLAIMING matches "Clm" or "Claiming"; here the bareword "Claiming"
        # in the race-type line carries the price.
        assert card.races[0].header.claiming_price == 20000.0


# ──────────────────────────────────────────────────────────────────────────────
# Horse entry parsing
# ──────────────────────────────────────────────────────────────────────────────


class TestHorseEntries:
    def test_correct_field_size(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.races[0].field_size == 2

    def test_post_positions_assigned(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        posts = [e.post_position for e in card.races[0].entries]
        assert posts == [1, 2]

    def test_horse_names_title_cased(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        names = [e.horse_name for e in card.races[0].entries]
        assert names == ["Big Brown", "Cigar"]

    def test_morning_line_odds_converted_to_decimal(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        entries = card.races[0].entries
        assert entries[0].morning_line_odds == pytest.approx(4.0)   # 3-1 → 4.0
        assert entries[1].morning_line_odds == pytest.approx(3.5)   # 5-2 → 3.5

    def test_ml_implied_prob_computed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        entries = card.races[0].entries
        # ml_implied_prob = 1 / morning_line_odds, computed by HorseEntry validator
        assert entries[0].ml_implied_prob == pytest.approx(0.25, abs=1e-3)
        assert entries[1].ml_implied_prob == pytest.approx(1 / 3.5, abs=1e-3)

    def test_jockey_parsed_and_cleaned(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        jockeys = [e.jockey for e in card.races[0].entries]
        assert jockeys == ["Smith, J", "Jones, T"]

    def test_weight_parsed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        weights = [e.weight_lbs for e in card.races[0].entries]
        assert weights == [120.0, 118.0]

    def test_trainer_parsed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        trainers = [e.trainer for e in card.races[0].entries]
        assert trainers == ["Pletcher, T", "Mott, B"]

    def test_default_pace_style_is_unknown(self, parser):
        # Pace style is assigned downstream by the Pace Model (Layer 1b).
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        for entry in card.races[0].entries:
            assert entry.pace_style == PaceStyle.UNKNOWN


# ──────────────────────────────────────────────────────────────────────────────
# Past performance line parsing
# ──────────────────────────────────────────────────────────────────────────────


class TestPastPerformanceLines:
    def test_two_pp_lines_per_horse(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        for entry in card.races[0].entries:
            assert entry.n_pp == 2

    def test_pp_lines_most_recent_first(self, parser):
        """Schema invariant: pp_lines must be sorted most-recent-first."""
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        horse_1 = card.races[0].entries[0]
        assert horse_1.pp_lines[0].race_date == date(2025, 4, 15)
        assert horse_1.pp_lines[1].race_date == date(2025, 3, 22)

    def test_last_pp_property_returns_most_recent(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        horse_1 = card.races[0].entries[0]
        assert horse_1.last_pp is not None
        assert horse_1.last_pp.race_date == date(2025, 4, 15)

    def test_pp_distance_parsed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        pp = card.races[0].entries[0].pp_lines[0]
        assert pp.distance_furlongs == pytest.approx(6.0)

    def test_pp_surface_and_condition(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        pp = card.races[0].entries[0].pp_lines[0]
        assert pp.surface == Surface.DIRT
        assert pp.condition == TrackCondition.FAST

    def test_pp_speed_figure_parsed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        # Big Brown's PP lines: speed figures 92, 88
        figs = [pp.speed_figure for pp in card.races[0].entries[0].pp_lines]
        assert figs == [92.0, 88.0]

    def test_pp_speed_figure_source_tagged_brisnet(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        pp = card.races[0].entries[0].pp_lines[0]
        assert pp.speed_figure_source == "brisnet"

    def test_pp_fractions_parsed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        pp = card.races[0].entries[0].pp_lines[0]  # 04/15/25: :22.40, :45.20, 1:10.40
        assert pp.fraction_q1 == pytest.approx(22.40, abs=1e-2)
        assert pp.fraction_q2 == pytest.approx(45.20, abs=1e-2)
        assert pp.fraction_finish == pytest.approx(70.40, abs=1e-2)

    def test_pp_finish_position_parsed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        # Big Brown's PPs: finish 3, finish 2
        finishes = [pp.finish_position for pp in card.races[0].entries[0].pp_lines]
        assert finishes == [3, 2]

    def test_days_since_prev_computed(self, parser):
        """The parser fills days_since_prev as (this_date - previous_date).days."""
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        horse_1 = card.races[0].entries[0]
        # 04/15/25 - 03/22/25 = 24 days
        assert horse_1.pp_lines[0].days_since_prev == 24
        # Last PP has no prior — days_since_prev stays None
        assert horse_1.pp_lines[1].days_since_prev is None

    def test_pp_track_code_parsed(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        codes = [pp.track_code for pp in card.races[0].entries[0].pp_lines]
        assert codes == ["BEL", "AQU"]


# ──────────────────────────────────────────────────────────────────────────────
# Parse confidence
# ──────────────────────────────────────────────────────────────────────────────


class TestParseConfidence:
    def test_confidence_in_unit_interval(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        conf = card.races[0].parse_confidence
        assert 0.0 <= conf <= 1.0

    def test_complete_race_has_high_confidence(self, parser):
        # Synthetic race has all header fields + every horse has PP lines,
        # so confidence should be very close to 1.0.
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.races[0].parse_confidence >= 0.95

    def test_qualified_card_passes_quality_gate(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        # has_enough_data requires confidence ≥ 0.5 AND ≥4 horses with ≥1 PP.
        # Our synthetic only has 2 horses, so this should be False even though
        # confidence is high.
        assert card.races[0].has_enough_data is False

    def test_qualified_count_zero_for_small_field(self, parser):
        card = parser.parse(SYNTHETIC_RACE_TEXT)
        assert card.n_qualified_races == 0


# ──────────────────────────────────────────────────────────────────────────────
# Multi-race input
# ──────────────────────────────────────────────────────────────────────────────


MULTI_RACE_TEXT = SYNTHETIC_RACE_TEXT + """

RACE 2
BEL  05/03/2025
1 1/16 Miles   (Turf)   Firm
Allowance     Purse: $80,000

 1  ROUTE HORSE               SMITH, J         122  2-1     L
    Trainer: Brown, C
04/22/25  BEL  1m  t  fm  Alw  4  1  SMITH, J  122  3.0  98  :24.40  1:13.20  1:36.80
"""


class TestMultiRace:
    def test_two_races_parsed(self, parser):
        card = parser.parse(MULTI_RACE_TEXT)
        assert card.n_races == 2

    def test_race_numbers_in_order(self, parser):
        card = parser.parse(MULTI_RACE_TEXT)
        assert [r.header.race_number for r in card.races] == [1, 2]

    def test_route_distance_with_fraction_parsed(self, parser):
        """1 1/16 Miles → 8.5 furlongs. Validates the 1/16 fraction patch."""
        card = parser.parse(MULTI_RACE_TEXT)
        assert card.races[1].header.distance_furlongs == pytest.approx(8.5)

    def test_route_surface_turf(self, parser):
        card = parser.parse(MULTI_RACE_TEXT)
        assert card.races[1].header.surface == Surface.TURF

    def test_route_condition_firm(self, parser):
        card = parser.parse(MULTI_RACE_TEXT)
        assert card.races[1].header.condition == TrackCondition.FIRM


# ──────────────────────────────────────────────────────────────────────────────
# Degenerate inputs
# ──────────────────────────────────────────────────────────────────────────────


class TestDegenerateInputs:
    def test_empty_input_returns_empty_card(self, parser):
        card = parser.parse("")
        assert card.n_races == 0
        assert card.source_format == "brisnet_up"

    def test_input_without_race_header_produces_no_races(self, parser):
        card = parser.parse("just some random text without race markers")
        assert card.n_races == 0

    def test_race_header_only_still_returns_race(self, parser):
        # A race block with only a header line and no horses returns an empty entries list.
        text = "RACE 1\nBEL  05/03/2025\n6 Furlongs (Dirt) Fast\n"
        card = parser.parse(text)
        assert card.n_races == 1
        assert card.races[0].field_size == 0
