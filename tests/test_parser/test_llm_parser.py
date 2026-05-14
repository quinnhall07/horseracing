"""
tests/test_parser/test_llm_parser.py
─────────────────────────────────────
Phase 9 / ADR-049 — LLM-based parser fallback for OCR-noisy text.

These tests mock the Anthropic SDK rather than making real API calls
(no credentials in CI, no API spend). They verify:

  1. The call is dispatched to the configured model with the correct
     cached system prompt structure.
  2. A well-formed JSON response is validated against ParsedRace and
     returned as `result.races`.
  3. Malformed JSON degrades gracefully — empty `races`, warning logged.
  4. Missing ANTHROPIC_API_KEY returns empty result without raising.
  5. Missing anthropic SDK returns empty result without raising.
  6. The `_extract_json_object` helper handles direct JSON, fenced
     blocks, and brace-balanced extraction.

When the real `anthropic` SDK isn't installed, a stub module is
registered in `sys.modules` at conftest-collection time so the in-
function import inside `LLMParser.parse()` succeeds and the tests can
patch `anthropic.Anthropic` as usual. The stub is patched away in the
single test that exercises the "SDK not installed" branch.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Inject a stub `anthropic` module if the real SDK isn't installed.
# `LLMParser.parse()` imports anthropic lazily inside the function body,
# so the module itself is importable without the SDK. Tests need the
# import to succeed so they can patch `anthropic.Anthropic` — the stub
# satisfies that without pulling in the real package.
if "anthropic" not in sys.modules:
    _stub = types.ModuleType("anthropic")
    _stub.Anthropic = MagicMock  # type: ignore[attr-defined]
    sys.modules["anthropic"] = _stub

import anthropic  # noqa: E402

from app.schemas.race import ParsedRace, Surface, TrackCondition, RaceType  # noqa: E402
from app.services.pdf_parser import llm_parser  # noqa: E402
from app.services.pdf_parser.llm_parser import (  # noqa: E402
    DEFAULT_MODEL,
    LLMParser,
    ParsedResult,
    _build_horse_entry,
    _build_parsed_race,
    _build_pp_line,
    _coerce_date,
    _coerce_enum,
    _coerce_float,
    _coerce_int,
    _extract_json_object,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: a canonical "good" LLM response payload
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def good_payload() -> dict:
    """A minimal but fully-valid race + horse + PP line."""
    return {
        "races": [
            {
                "race_number": 4,
                "race_date": "2026-05-10",
                "track_code": "CD",
                "track_name": "Churchill Downs",
                "distance_furlongs": 4.5,
                "distance_raw": "4 1/2 Furlongs",
                "surface": "dirt",
                "condition": "fast",
                "race_type": "claiming",
                "claiming_price": 30000.0,
                "purse_usd": 62000.0,
                "entries": [
                    {
                        "horse_name": "Lovely Words",
                        "post_position": 1,
                        "morning_line_odds": 7.0,
                        "weight_lbs": 123.0,
                        "jockey": "Danilo Grisales Rave",
                        "trainer": "John A. Ortiz",
                        "medication_flags": ["L"],
                        "pp_lines": [
                            {
                                "race_date": "2026-04-12",
                                "track_code": "KEE",
                                "race_number": 3,
                                "distance_furlongs": 6.0,
                                "surface": "dirt",
                                "condition": "fast",
                                "race_type": "claiming",
                                "post_position": 4,
                                "finish_position": 3,
                                "lengths_behind": 3.25,
                                "field_size": 7,
                                "jockey": "Elliott J",
                                "weight_lbs": 124.0,
                                "odds_final": 4.0,
                                "speed_figure": 71,
                                "speed_figure_source": "brisnet",
                                "fraction_q1": 22.4,
                                "fraction_q2": 46.1,
                                "fraction_finish": 71.6,
                                "comment": "Bided",
                            }
                        ],
                    },
                    {
                        "horse_name": "Big Brown",
                        "post_position": 2,
                        "morning_line_odds": 4.0,
                        "weight_lbs": 120.0,
                        "jockey": "Smith J",
                        "trainer": "Pletcher T",
                        "medication_flags": ["L"],
                        "pp_lines": [],
                    },
                ],
            }
        ]
    }


def _make_mock_response(payload: dict | str) -> MagicMock:
    """Build a MagicMock response that mimics anthropic.types.Message."""
    body = json.dumps(payload) if isinstance(payload, dict) else payload
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = body

    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(cache_read_input_tokens=1500)
    return response


# ──────────────────────────────────────────────────────────────────────────────
# LLMParser.parse — happy path
# ──────────────────────────────────────────────────────────────────────────────


class TestLLMParserHappyPath:
    def test_calls_anthropic_with_correct_model_and_caching(self, good_payload):
        """The call must (a) target the configured model and (b) cache the
        system prompt + field-primer prefix."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(good_payload)

        with patch("anthropic.Anthropic", return_value=mock_client) as ctor:
            parser = LLMParser(model="claude-haiku-4-5")
            result = parser.parse("some ocr text", source_format="brisnet_up")

        assert result.races, "expected at least one race"
        # API key is passed (test-key from conftest)
        assert ctor.call_args.kwargs.get("api_key") == "test-key"
        # Model + max_tokens were passed
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5"
        assert call_kwargs["max_tokens"] >= 4096
        # System prompt has 2 blocks with cache_control on the trailing one
        sys_blocks = call_kwargs["system"]
        assert isinstance(sys_blocks, list)
        assert len(sys_blocks) == 2
        assert sys_blocks[0]["type"] == "text"
        assert sys_blocks[1]["type"] == "text"
        assert sys_blocks[1].get("cache_control") == {"type": "ephemeral"}
        # The user message carries the OCR text
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert "some ocr text" in messages[0]["content"]

    def test_validates_payload_into_parsed_race(self, good_payload):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(good_payload)
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = LLMParser().parse("ocr")
        assert len(result.races) == 1
        race = result.races[0]
        assert isinstance(race, ParsedRace)
        assert race.header.race_number == 4
        assert race.header.track_code == "CD"
        assert race.header.distance_furlongs == 4.5
        assert race.header.surface == Surface.DIRT
        assert race.header.condition == TrackCondition.FAST
        assert race.header.race_type == RaceType.CLAIMING
        assert race.field_size == 2
        # Entries
        e0 = race.entries[0]
        assert e0.horse_name == "Lovely Words"
        assert e0.post_position == 1
        assert e0.morning_line_odds == 7.0
        assert e0.n_pp == 1
        assert e0.pp_lines[0].speed_figure == 71
        assert e0.pp_lines[0].track_code == "KEE"
        # Confidence: 1 of 2 horses with PP → header(1.0)*0.4 + pp(0.5)*0.6 = 0.7
        assert race.parse_confidence == pytest.approx(0.7, abs=0.01)

    def test_cached_tokens_surfaced(self, good_payload):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(good_payload)
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = LLMParser().parse("ocr")
        assert result.cached_tokens == 1500

    def test_returns_model_id(self, good_payload):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(good_payload)
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = LLMParser(model="claude-haiku-4-5").parse("ocr")
        assert result.model == "claude-haiku-4-5"


# ──────────────────────────────────────────────────────────────────────────────
# LLMParser.parse — failure modes (all degrade gracefully)
# ──────────────────────────────────────────────────────────────────────────────


class TestLLMParserFailureModes:
    def test_missing_api_key_returns_empty_with_warning(self, monkeypatch):
        """When ANTHROPIC_API_KEY is unset, parse() must not crash and
        must not call the SDK at all."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("anthropic.Anthropic") as ctor:
            result = LLMParser().parse("some ocr text")
        ctor.assert_not_called()
        assert result.races == []
        assert any("ANTHROPIC_API_KEY" in w for w in result.warnings)

    def test_empty_ocr_text_returns_empty_with_warning(self):
        result = LLMParser().parse("")
        assert result.races == []
        assert any("empty" in w.lower() for w in result.warnings)

    def test_malformed_json_returns_empty_with_warning(self):
        """A response that isn't JSON at all — empty list, warning."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(
            "I cannot parse this card sorry."
        )
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = LLMParser().parse("ocr")
        assert result.races == []
        assert any("not parseable JSON" in w for w in result.warnings)

    def test_api_exception_returns_empty_with_warning(self):
        """If the SDK raises (rate-limit, auth, network), the parser
        swallows and surfaces a clean warning."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("rate limited")
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = LLMParser().parse("ocr")
        assert result.races == []
        assert any("API call failed" in w for w in result.warnings)

    def test_missing_races_key_returns_empty_with_warning(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(
            {"not_races": "wrong shape"}
        )
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = LLMParser().parse("ocr")
        assert result.races == []
        assert any("no 'races' array" in w for w in result.warnings)

    def test_individual_race_validation_failure_skips_only_that_race(self):
        """Bad races are skipped, good races are kept."""
        payload = {
            "races": [
                {
                    "race_number": 999,  # invalid: must be 1-20
                    "track_code": "CD",
                    "distance_furlongs": 6.0,
                    "race_date": "2026-05-10",
                },
                {
                    "race_number": 5,
                    "track_code": "CD",
                    "distance_furlongs": 6.0,
                    "distance_raw": "6 Furlongs",
                    "race_date": "2026-05-10",
                    "surface": "dirt",
                    "entries": [],
                },
            ]
        }
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(payload)
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = LLMParser().parse("ocr")
        # Bad race is silently dropped (race_number 999 fails _coerce_int range
        # check before Pydantic ever sees it, so no warning is generated; the
        # good race makes it through).
        assert len(result.races) == 1
        assert result.races[0].header.race_number == 5

    def test_sdk_not_installed_returns_empty(self, monkeypatch):
        """Simulate `anthropic` not being installed at all."""
        # Hide anthropic from sys.modules so the in-function import fails.
        monkeypatch.setitem(sys.modules, "anthropic", None)
        result = LLMParser().parse("some ocr")
        assert result.races == []
        assert any("anthropic SDK" in w for w in result.warnings)


# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction utility
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractJsonObject:
    def test_direct_json(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_json_with_whitespace(self):
        assert _extract_json_object('  \n{"a": 1}\n  ') == {"a": 1}

    def test_fenced_json_block(self):
        raw = 'Here is the result:\n```json\n{"a": 1, "b": 2}\n```\n'
        assert _extract_json_object(raw) == {"a": 1, "b": 2}

    def test_fenced_block_without_language_tag(self):
        raw = '```\n{"a": 1}\n```'
        assert _extract_json_object(raw) == {"a": 1}

    def test_brace_balanced_extraction(self):
        raw = 'Some prose. Then a JSON: {"a": {"b": 1}, "c": [1,2]} trailing.'
        assert _extract_json_object(raw) == {"a": {"b": 1}, "c": [1, 2]}

    def test_handles_escaped_quote_in_string(self):
        raw = r'{"name": "O\"Brien"}'
        assert _extract_json_object(raw) == {"name": 'O"Brien'}

    def test_returns_none_on_garbage(self):
        assert _extract_json_object("totally not json {{{") is None

    def test_returns_none_when_no_brace(self):
        assert _extract_json_object("plain text only") is None


# ──────────────────────────────────────────────────────────────────────────────
# Coercion helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestCoercion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (1.5, 1.5),
            (2, 2.0),
            ("$30,000", 30000.0),
            ("--", None),
            ("n/a", None),
            ("unknown", None),
            (None, None),
            ("garbage", None),
        ],
    )
    def test_coerce_float(self, raw, expected):
        assert _coerce_float(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (5, 5),
            (5.9, 6),
            ("3", 3),
            (None, None),
        ],
    )
    def test_coerce_int(self, raw, expected):
        assert _coerce_int(raw) == expected

    def test_coerce_date_iso(self):
        from datetime import date

        assert _coerce_date("2026-05-10") == date(2026, 5, 10)

    def test_coerce_date_us(self):
        from datetime import date

        assert _coerce_date("05/10/2026") == date(2026, 5, 10)

    def test_coerce_date_short_year(self):
        from datetime import date

        assert _coerce_date("05/10/26") == date(2026, 5, 10)

    def test_coerce_date_invalid_returns_none(self):
        assert _coerce_date("not a date") is None

    def test_coerce_enum_valid(self):
        from app.services.pdf_parser.llm_parser import _SURFACE_VALUES

        assert _coerce_enum("dirt", _SURFACE_VALUES, "unknown") == "dirt"
        assert _coerce_enum("Dirt", _SURFACE_VALUES, "unknown") == "dirt"

    def test_coerce_enum_invalid_falls_back(self):
        from app.services.pdf_parser.llm_parser import _SURFACE_VALUES

        assert _coerce_enum("rocks", _SURFACE_VALUES, "unknown") == "unknown"
        assert _coerce_enum(None, _SURFACE_VALUES, "unknown") == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Row-level builders
# ──────────────────────────────────────────────────────────────────────────────


class TestRowBuilders:
    def test_pp_line_drops_when_distance_out_of_range(self):
        line = _build_pp_line({
            "race_date": "2026-04-12",
            "track_code": "CD",
            "distance_furlongs": 50.0,  # garbage
            "post_position": 3,
        })
        assert line is None

    def test_pp_line_drops_when_track_code_too_long(self):
        line = _build_pp_line({
            "race_date": "2026-04-12",
            "track_code": "TOOLONG",
            "distance_furlongs": 6.0,
            "post_position": 3,
        })
        assert line is None

    def test_pp_line_clamps_invalid_weight_to_none(self):
        line = _build_pp_line({
            "race_date": "2026-04-12",
            "track_code": "CD",
            "distance_furlongs": 6.0,
            "post_position": 3,
            "weight_lbs": 999,  # out of range
            "finish_position": 2,
        })
        assert line is not None
        assert line.weight_lbs is None

    def test_pp_line_normalizes_winner_lengths_behind(self):
        """finish_position=1 with positive lengths_behind would crash the
        Pydantic validator; the builder must zero it out first."""
        line = _build_pp_line({
            "race_date": "2026-04-12",
            "track_code": "CD",
            "distance_furlongs": 6.0,
            "post_position": 3,
            "finish_position": 1,
            "lengths_behind": 2.5,
        })
        assert line is not None
        assert line.lengths_behind == 0.0

    def test_horse_entry_drops_when_name_missing(self):
        he = _build_horse_entry({"post_position": 1})
        assert he is None

    def test_horse_entry_drops_when_post_invalid(self):
        he = _build_horse_entry({"horse_name": "X", "post_position": 99})
        assert he is None

    def test_horse_entry_accepts_minimal(self):
        he = _build_horse_entry({"horse_name": "Test Horse", "post_position": 3})
        assert he is not None
        assert he.horse_name == "Test Horse"
        assert he.post_position == 3
        assert he.n_pp == 0

    def test_parsed_race_drops_when_track_code_missing(self):
        race = _build_parsed_race({
            "race_number": 4,
            "distance_furlongs": 6.0,
            "race_date": "2026-05-10",
        })
        assert race is None

    def test_parsed_race_drops_when_distance_invalid(self):
        race = _build_parsed_race({
            "race_number": 4,
            "track_code": "CD",
            "distance_furlongs": 100.0,
            "race_date": "2026-05-10",
        })
        assert race is None


# ──────────────────────────────────────────────────────────────────────────────
# Integration: extractor.ingest_pdf falls through to LLM when regex empty
# ──────────────────────────────────────────────────────────────────────────────


class TestIngestPdfLLMFallback:
    """Verify the wiring in `extractor.ingest_pdf` actually invokes the
    LLM parser when the regex parser produces nothing usable."""

    def _minimal_pdf_with_text(self, text: str) -> bytes:
        from io import BytesIO
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        c.setFont("Courier", 9)
        y = 750
        for line in text.split("\n"):
            c.drawString(36, y, line)
            y -= 11
        c.save()
        return buf.getvalue()

    def test_llm_fallback_invoked_when_regex_yields_zero_races(self, good_payload):
        """A PDF whose text doesn't match RaceN regex → 0 races from regex →
        LLM parser is called and its races are returned in the IngestionResult."""
        from app.services.pdf_parser import extractor

        # PDF text that contains the Brisnet signature but no parseable
        # "RACE N" headers — regex parser returns 0 races, LLM fallback fires.
        pdf_bytes = self._minimal_pdf_with_text(
            "Brisnet Ultimate Past Performances\nGarbage line 1\nGarbage line 2"
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(good_payload)
        with patch("anthropic.Anthropic", return_value=mock_client) as ctor:
            r = extractor.ingest_pdf(pdf_bytes, source_filename="t.pdf")

        # LLM client was instantiated (fallback triggered)
        ctor.assert_called_once()
        # Result includes the LLM-extracted race
        assert r.success is True
        assert r.card is not None
        assert r.card.n_races == 1
        assert r.card.source_format.endswith("+llm")
        # Provenance string in errors so UI can show "parsed via LLM"
        assert any("Parsed via LLM fallback" in e for e in r.errors)

    def test_llm_fallback_skipped_when_api_key_missing(self, monkeypatch):
        """No API key → LLM call returns empty + warning; ingest_pdf
        still returns success=False (no races) but doesn't crash."""
        from app.services.pdf_parser import extractor

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pdf_bytes = self._minimal_pdf_with_text(
            "Brisnet Ultimate Past Performances\nGarbage line\n"
        )
        r = extractor.ingest_pdf(pdf_bytes, source_filename="t.pdf")
        assert r.success is False
        assert any("ANTHROPIC_API_KEY" in e for e in r.errors)
