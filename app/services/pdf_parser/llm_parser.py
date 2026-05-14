"""
app/services/pdf_parser/llm_parser.py
──────────────────────────────────────
LLM-based fallback parser for OCR-noisy Brisnet UP race-card text (Phase 9).

The regex-based `BrisnetParser` (`brisnet_parser.py`) is strict: it expects
multi-space column separators, well-formed monospaced columns, and clean
date / weight / odds tokens. OCR'd scans of Brisnet UP cards routinely
break those assumptions — Tesseract collapses columns, misreads "L" as
"|" or "1", drops accent marks, splits horse names mid-token, etc. The
result is that the regex parser produces 0 horses on the user's actual
input PDFs even though the OCR text is humanly readable.

This module wraps the Anthropic Claude API to extract structured
`ParsedRace` records from OCR text. It is invoked by `extractor.py` only
when the regex parser produces 0 races or 0 qualified races — a fallback,
not a replacement.

Configuration
─────────────
- Requires the `anthropic` Python SDK: `pip install -e .[llm-parse]`.
- Requires the `ANTHROPIC_API_KEY` environment variable. When unset,
  `LLMParser.parse()` returns an empty result with a warning rather than
  raising, so the pipeline degrades gracefully on systems without
  credentials.
- Default model is `claude-haiku-4-5` — cheap, fast, sufficient for
  structured-text extraction from a single PDF's worth of OCR.

Prompt caching
──────────────
The system prompt + the field-spec primer are constant across uploads.
We mark both with `cache_control={"type": "ephemeral"}` so that
repeat uploads (the common case in the user's workflow) hit the cache
and pay ~0.1× input price instead of full price. See ADR-049.

Output contract
───────────────
`LLMParser.parse(ocr_text, source_format)` returns a `ParsedResult`
dataclass holding:
  - `races: list[ParsedRace]` (Pydantic-validated)
  - `warnings: list[str]` (LLM-level issues, prompt for the
    `IngestionResult.errors` field)
  - `model: str` (the model ID used)
  - `cached_tokens: int` (cache-read input tokens, for telemetry)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import structlog

from app.schemas.race import (
    HorseEntry,
    PastPerformanceLine,
    ParsedRace,
    RaceHeader,
    RaceType,
    Surface,
    TrackCondition,
)

logger = structlog.get_logger(__name__)


# Default Claude model. Haiku is the right call here: structured extraction
# from a single PDF of OCR text doesn't benefit from Opus/Sonnet reasoning,
# and cost matters because every upload triggers this path. See ADR-049.
DEFAULT_MODEL: str = "claude-haiku-4-5"

# Max tokens for the JSON response. A full 8-horse Brisnet UP card with
# 10 PP lines each ≈ 8000–14000 output tokens. 16000 leaves headroom.
DEFAULT_MAX_TOKENS: int = 16000


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ParsedResult:
    """Output of `LLMParser.parse()`.

    `races` is a list of fully-validated `ParsedRace` objects (each may
    be empty of entries if the LLM couldn't extract horses for a given
    race). `warnings` is a string list surfacing LLM-level issues
    (missing API key, malformed JSON, validation failures) that the
    caller propagates to `IngestionResult.errors`.
    """

    races: list[ParsedRace] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    cached_tokens: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# System prompt + field-spec primer (cacheable; never changes per call)
# ──────────────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a race-card data extraction agent. You receive OCR text from
scanned Brisnet UP (Ultimate Past Performances) PDFs of US horse racing
cards. Your job is to extract every race and every horse from the text
and emit a single strict JSON object matching the schema below.

Rules:
1. Output ONLY a JSON object — no prose, no markdown, no code fences.
2. If you cannot determine a field with confidence, emit null. Never
   guess; OCR noise is the norm here.
3. All dates must be ISO 8601 (YYYY-MM-DD).
4. Convert fractional odds ("6-1") to decimal floats (numerator/denom + 1
   for fractional, so "6-1" → 7.0; "5-2" → 3.5; "even" or "evens" → 2.0).
5. Convert distances to furlongs as floats ("6 Furlongs" → 6.0,
   "1 1/16 Miles" → 8.5, "5 1/2 Furlongs" → 5.5).
6. Surfaces are exactly one of: "dirt", "turf", "synthetic", "unknown".
7. Track conditions are exactly one of: "fast", "good", "sloppy",
   "muddy", "heavy", "frozen", "firm", "yielding", "soft", "unknown".
8. Race types are exactly one of: "maiden_claiming",
   "maiden_special_weight", "claiming", "allowance",
   "allowance_optional_claiming", "stakes", "graded_stakes", "handicap",
   "unknown".
9. Weights are pounds, in [95, 145]. If outside, emit null.
10. PP-line speed figures are floats in [-20, 150]. Out of range → null.
11. Times in seconds (e.g., "1:10.40" → 70.40, ":22.40" → 22.40).
12. Preserve horse names exactly as they appear (title case, not all-caps,
    unless the OCR clearly preserved all-caps). Strip leading/trailing
    whitespace and obvious OCR junk (lone backslashes, vertical bars
    between letters of one word, etc).
13. Output the JSON in the exact structure shown below.

Required JSON shape:

{
  "races": [
    {
      "race_number": <int 1-20>,
      "race_date": "YYYY-MM-DD",
      "track_code": "<2-5 letter uppercase track abbrev, e.g. CD>",
      "track_name": "<optional full track name>" or null,
      "distance_furlongs": <float 2.0-20.0>,
      "distance_raw": "<original distance string>",
      "surface": "dirt"|"turf"|"synthetic"|"unknown",
      "condition": "fast"|"good"|"sloppy"|"muddy"|"heavy"|"frozen"|"firm"|"yielding"|"soft"|"unknown",
      "race_type": "claiming"|"allowance"|"maiden_claiming"|"maiden_special_weight"|"stakes"|"graded_stakes"|"handicap"|"allowance_optional_claiming"|"unknown",
      "claiming_price": <float USD> or null,
      "purse_usd": <float USD> or null,
      "race_name": "<stakes name>" or null,
      "grade": 1|2|3 or null,
      "age_sex_restrictions": "<text>" or null,
      "post_time": "<text>" or null,
      "entries": [
        {
          "horse_name": "<name>",
          "post_position": <int 1-24>,
          "morning_line_odds": <float decimal odds, e.g. 7.0 for "6-1"> or null,
          "weight_lbs": <float 95-145> or null,
          "jockey": "<jockey name>" or null,
          "trainer": "<trainer name>" or null,
          "owner": "<owner>" or null,
          "medication_flags": ["L"|"B"|...],
          "equipment_changes": [],
          "pp_lines": [
            {
              "race_date": "YYYY-MM-DD",
              "track_code": "<track>",
              "race_number": <int 1-20>,
              "distance_furlongs": <float 2.0-20.0>,
              "surface": "dirt"|"turf"|"synthetic"|"unknown",
              "condition": "<cond>"|"unknown",
              "race_type": "<type>"|"unknown",
              "claiming_price": <float> or null,
              "post_position": <int 1-24>,
              "finish_position": <int 1-30> or null,
              "lengths_behind": <float ≥ 0> or null,
              "field_size": <int 2-30> or null,
              "jockey": "<jockey>" or null,
              "weight_lbs": <float 95-145> or null,
              "odds_final": <float decimal ≥ 1.0> or null,
              "speed_figure": <float -20 to 150> or null,
              "speed_figure_source": "brisnet"|"beyer"|"equibase"|"unknown",
              "fraction_q1": <float 10-100 seconds> or null,
              "fraction_q2": <float 20-160 seconds> or null,
              "fraction_finish": <float 40-300 seconds> or null,
              "comment": "<trip note>" or null
            }
          ]
        }
      ]
    }
  ]
}

If the OCR text contains no recognizable race, emit `{"races": []}`. Do
not invent races to fill out the list.

For each race, attempt to extract every horse in the field, even when
some fields (especially PP lines) are too noisy to recover; emit the
horse with `pp_lines: []` rather than dropping it.

IMPORTANT: emit valid JSON. Use double quotes. No trailing commas. No
NaN or Infinity. If a value is unknown, use `null`, not the string
"unknown" (except for the enum fields above which have explicit
"unknown" values).
"""


# Brisnet-specific format primer; cached alongside the system prompt
# because it doesn't vary across calls. Keeps the system prompt itself
# focused on the JSON contract.
_FIELD_PRIMER = """\
Brisnet UP format hints (use these when interpreting OCR noise):

- The race header lists the track ("CD" = Churchill Downs, "BEL" =
  Belmont, etc.), date as "MM/DD/YYYY" or "Month DD, YYYY", race
  number, distance, surface in parens "(Dirt)" / "(Turf)", and
  conditions text mentioning "Claiming $X" / "Maiden" / "Allowance".

- Each horse block typically opens with:
    "  N  HORSE NAME              Jockey      120  6-1     L"
  where N is the post position (1-24), HORSE NAME is in all-caps,
  jockey is "Last, First" or "First Last", 120 is the weight in
  pounds, "6-1" is the morning line (fractional), and "L" means
  Lasix is on board.

- Each horse has up to 10 past performance lines below it, one per
  prior race. Each PP line is dense:
    "MM/DD/YY  TRK  6f  d  ft  Clm12500  3  2  Smith,J  120  4.0  92  :22.4  :45.2  1:10.4  ..."
  where surface code "d"=dirt, "t"=turf, "s"=synthetic, condition
  "ft"=fast, "sy"=sloppy, "gd"=good, "my"=muddy. Speed figures
  ("92") are Brisnet figures unless labeled otherwise.

- Workout lines appear below PP lines; ignore workouts for this
  extraction — only the horse's entry + its PP lines matter.

- OCR commonly mangles: "L" → "|" or "1"; spaces collapse so
  "Pletcher, T" reads as "PletcherT"; commas become periods;
  superscripts become digits. Be tolerant.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main parser class
# ──────────────────────────────────────────────────────────────────────────────


class LLMParser:
    """LLM-backed parser for Brisnet UP cards as OCR text.

    Usage:
        parser = LLMParser()  # uses claude-haiku-4-5, reads ANTHROPIC_API_KEY
        result = parser.parse(ocr_text)
        for race in result.races:
            print(race.header.race_number, race.field_size)

    Failure modes (none raise):
      - ANTHROPIC_API_KEY unset → empty `races`, warning logged
      - anthropic SDK not installed → empty `races`, warning logged
      - API call fails → empty `races`, warning with reason
      - Response not parseable JSON → empty `races`, warning with raw excerpt
      - JSON shape invalid → empty `races`, per-race warning
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def parse(
        self,
        ocr_text: str,
        source_format: str = "brisnet_up",
    ) -> ParsedResult:
        """Extract races from OCR text via Claude.

        Returns a `ParsedResult` (never raises). The caller is responsible
        for merging this into the final `RaceCard` and surfacing
        `result.warnings` to the user.
        """
        result = ParsedResult(model=self.model)

        if not ocr_text or not ocr_text.strip():
            result.warnings.append("llm_parser: empty OCR text; nothing to parse")
            return result

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            msg = (
                "llm_parser: ANTHROPIC_API_KEY not set; "
                "skipping LLM fallback parse"
            )
            logger.warning("llm_parser.no_api_key")
            result.warnings.append(msg)
            return result

        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError:
            msg = (
                "llm_parser: anthropic SDK not installed; "
                "run `pip install -e .[llm-parse]`"
            )
            logger.warning("llm_parser.sdk_missing")
            result.warnings.append(msg)
            return result

        # Build the request. The system prompt + field primer are two
        # cacheable blocks (one cache_control at the end of the last
        # block covers the prefix). The OCR text is the volatile per-
        # call content and goes in the user message after the cached
        # prefix — so subsequent uploads hit cache. See ADR-049.
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {"type": "text", "text": _SYSTEM_PROMPT},
                    {
                        "type": "text",
                        "text": _FIELD_PRIMER,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Source format: {source_format}\n\n"
                            f"OCR text follows. Extract every race + horse + "
                            f"PP line you can recover; emit strict JSON only.\n\n"
                            f"```\n{ocr_text}\n```"
                        ),
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 — anthropic raises many types
            msg = f"llm_parser: API call failed: {type(exc).__name__}: {exc}"
            logger.error("llm_parser.api_failed", error=str(exc))
            result.warnings.append(msg)
            return result

        # Telemetry: cache reads
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                result.cached_tokens = int(
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                )
        except Exception:  # noqa: BLE001
            pass

        # Concatenate text blocks in the response
        raw_json: str = ""
        try:
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    raw_json += getattr(block, "text", "") or ""
        except Exception as exc:  # noqa: BLE001
            msg = f"llm_parser: failed to read response content: {exc}"
            logger.error("llm_parser.response_read_failed", error=str(exc))
            result.warnings.append(msg)
            return result

        if not raw_json.strip():
            result.warnings.append("llm_parser: empty response from LLM")
            return result

        payload = _extract_json_object(raw_json)
        if payload is None:
            preview = raw_json[:240].replace("\n", " ")
            result.warnings.append(
                f"llm_parser: response was not parseable JSON. "
                f"First 240 chars: {preview!r}"
            )
            return result

        races_data = payload.get("races") if isinstance(payload, dict) else None
        if not isinstance(races_data, list):
            result.warnings.append(
                f"llm_parser: response JSON has no 'races' array (got keys "
                f"{list(payload.keys()) if isinstance(payload, dict) else 'N/A'})"
            )
            return result

        for race_idx, race_raw in enumerate(races_data):
            if not isinstance(race_raw, dict):
                result.warnings.append(
                    f"llm_parser: race[{race_idx}] is not an object; skipping"
                )
                continue
            try:
                parsed = _build_parsed_race(race_raw)
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(
                    f"llm_parser: race[{race_idx}] failed validation: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if parsed is not None:
                result.races.append(parsed)

        logger.info(
            "llm_parser.parse_complete",
            races_found=len(result.races),
            warnings=len(result.warnings),
            cached_tokens=result.cached_tokens,
            model=self.model,
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction helpers
# ──────────────────────────────────────────────────────────────────────────────


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(raw: str) -> Optional[dict[str, Any]]:
    """Best-effort extract a single top-level JSON object from `raw`.

    Tries (in order):
      1. Direct `json.loads(raw.strip())`.
      2. Markdown fenced ```json ... ``` block.
      3. First `{ ... }` block via brace balancing.

    Returns the parsed dict or None.
    """
    raw = raw.strip()
    # Direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fenced block
    m = _FENCE_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Brace-balanced first object
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic construction from LLM JSON
# ──────────────────────────────────────────────────────────────────────────────


_SURFACE_VALUES = {s.value for s in Surface}
_CONDITION_VALUES = {c.value for c in TrackCondition}
_RACETYPE_VALUES = {r.value for r in RaceType}


def _coerce_enum(raw: Any, allowed: set[str], fallback: str) -> str:
    """Map a raw LLM string onto an allowed enum value; fallback otherwise."""
    if not isinstance(raw, str):
        return fallback
    v = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if v in allowed:
        return v
    return fallback


def _coerce_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip().replace("$", "").replace(",", "")
        if not s or s.lower() in {"null", "none", "n/a", "unknown", "--"}:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_int(raw: Any) -> Optional[int]:
    f = _coerce_float(raw)
    if f is None:
        return None
    try:
        return int(round(f))
    except (TypeError, ValueError):
        return None


def _coerce_date(raw: Any) -> Optional[date]:
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _build_pp_line(raw: dict[str, Any]) -> Optional[PastPerformanceLine]:
    """Build a `PastPerformanceLine` from an LLM-emitted dict.

    Returns None if any load-bearing field (race_date, track_code,
    distance_furlongs, post_position) cannot be coerced into valid form.
    """
    race_date_val = _coerce_date(raw.get("race_date"))
    if race_date_val is None:
        return None

    track_code_val = raw.get("track_code")
    if not isinstance(track_code_val, str) or not (2 <= len(track_code_val.strip()) <= 5):
        return None
    track_code_val = track_code_val.strip().upper()

    distance_val = _coerce_float(raw.get("distance_furlongs"))
    if distance_val is None or not (2.0 <= distance_val <= 20.0):
        return None

    post_pos = _coerce_int(raw.get("post_position"))
    if post_pos is None or not (1 <= post_pos <= 24):
        return None

    race_num = _coerce_int(raw.get("race_number")) or 1
    race_num = max(1, min(race_num, 20))

    # Optional numeric fields — clamp/drop on out-of-range so the
    # schema's Pydantic validators don't reject the whole line.
    weight = _coerce_float(raw.get("weight_lbs"))
    if weight is not None and not (95.0 <= weight <= 145.0):
        weight = None

    finish_position = _coerce_int(raw.get("finish_position"))
    if finish_position is not None and not (1 <= finish_position <= 30):
        finish_position = None

    field_size = _coerce_int(raw.get("field_size"))
    if field_size is not None and not (2 <= field_size <= 30):
        field_size = None

    lengths_behind = _coerce_float(raw.get("lengths_behind"))
    if lengths_behind is not None and lengths_behind < 0:
        lengths_behind = None
    # If this horse finished first, normalize lengths_behind so the
    # PastPerformanceLine validator doesn't reject the line.
    if finish_position == 1 and lengths_behind is not None and lengths_behind > 0.01:
        lengths_behind = 0.0

    odds_final = _coerce_float(raw.get("odds_final"))
    if odds_final is not None and odds_final < 1.0:
        odds_final = None

    speed_figure = _coerce_float(raw.get("speed_figure"))
    if speed_figure is not None and not (-20.0 <= speed_figure <= 150.0):
        speed_figure = None

    def _frac(key: str, lo: float, hi: float) -> Optional[float]:
        v = _coerce_float(raw.get(key))
        if v is None:
            return None
        if not (lo <= v <= hi):
            return None
        return v

    fraction_q1 = _frac("fraction_q1", 10.0, 100.0)
    fraction_q2 = _frac("fraction_q2", 20.0, 160.0)
    fraction_finish = _frac("fraction_finish", 40.0, 300.0)

    try:
        return PastPerformanceLine(
            race_date=race_date_val,
            track_code=track_code_val,
            race_number=race_num,
            distance_furlongs=distance_val,
            surface=Surface(_coerce_enum(raw.get("surface"), _SURFACE_VALUES, "unknown")),
            condition=TrackCondition(
                _coerce_enum(raw.get("condition"), _CONDITION_VALUES, "unknown")
            ),
            race_type=RaceType(_coerce_enum(raw.get("race_type"), _RACETYPE_VALUES, "unknown")),
            claiming_price=_coerce_float(raw.get("claiming_price")),
            purse_usd=_coerce_float(raw.get("purse_usd")),
            post_position=post_pos,
            finish_position=finish_position,
            lengths_behind=lengths_behind,
            field_size=field_size,
            jockey=_clean_str(raw.get("jockey")),
            weight_lbs=weight,
            odds_final=odds_final,
            speed_figure=speed_figure,
            speed_figure_source=_clean_str(raw.get("speed_figure_source")) or "unknown",
            fraction_q1=fraction_q1,
            fraction_q2=fraction_q2,
            fraction_finish=fraction_finish,
            comment=_clean_str(raw.get("comment")),
        )
    except Exception:  # noqa: BLE001
        return None


def _clean_str(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        return s or None
    return str(raw)


def _build_horse_entry(raw: dict[str, Any]) -> Optional[HorseEntry]:
    """Build a `HorseEntry` from LLM-emitted JSON. Returns None on missing
    name / post_position (load-bearing)."""
    name = _clean_str(raw.get("horse_name"))
    if not name or len(name) > 100:
        return None

    post = _coerce_int(raw.get("post_position"))
    if post is None or not (1 <= post <= 24):
        return None

    ml = _coerce_float(raw.get("morning_line_odds"))
    if ml is not None and not (1.0 <= ml <= 500.0):
        ml = None

    weight = _coerce_float(raw.get("weight_lbs"))
    if weight is not None and not (95.0 <= weight <= 145.0):
        weight = None

    pp_raw = raw.get("pp_lines") or []
    if not isinstance(pp_raw, list):
        pp_raw = []

    pp_lines: list[PastPerformanceLine] = []
    for pp in pp_raw:
        if not isinstance(pp, dict):
            continue
        line = _build_pp_line(pp)
        if line is not None:
            pp_lines.append(line)

    meds = raw.get("medication_flags") or []
    if not isinstance(meds, list):
        meds = []
    meds = [str(m).strip() for m in meds if str(m).strip()]

    equip = raw.get("equipment_changes") or []
    if not isinstance(equip, list):
        equip = []
    equip = [str(e).strip() for e in equip if str(e).strip()]

    try:
        return HorseEntry(
            horse_name=name,
            post_position=post,
            morning_line_odds=ml,
            jockey=_clean_str(raw.get("jockey")),
            trainer=_clean_str(raw.get("trainer")),
            owner=_clean_str(raw.get("owner")),
            weight_lbs=weight,
            medication_flags=meds,
            equipment_changes=equip,
            pp_lines=pp_lines,
        )
    except Exception:  # noqa: BLE001
        return None


def _build_parsed_race(raw: dict[str, Any]) -> Optional[ParsedRace]:
    """Build a `ParsedRace` from LLM-emitted JSON. Returns None on
    missing distance / race_number / track_code."""
    race_num = _coerce_int(raw.get("race_number"))
    if race_num is None or not (1 <= race_num <= 20):
        return None

    track_code = raw.get("track_code")
    if not isinstance(track_code, str):
        return None
    track_code = track_code.strip().upper()
    if not (2 <= len(track_code) <= 5):
        return None

    race_date_val = _coerce_date(raw.get("race_date")) or date.today()

    distance_val = _coerce_float(raw.get("distance_furlongs"))
    if distance_val is None or not (2.0 <= distance_val <= 20.0):
        return None

    distance_raw = _clean_str(raw.get("distance_raw")) or f"{distance_val} Furlongs"

    grade = _coerce_int(raw.get("grade"))
    if grade is not None and not (1 <= grade <= 3):
        grade = None

    purse = _coerce_float(raw.get("purse_usd"))
    claiming = _coerce_float(raw.get("claiming_price"))

    header = RaceHeader(
        race_number=race_num,
        race_date=race_date_val,
        track_code=track_code,
        track_name=_clean_str(raw.get("track_name")),
        race_name=_clean_str(raw.get("race_name")),
        distance_furlongs=distance_val,
        distance_raw=distance_raw,
        surface=Surface(_coerce_enum(raw.get("surface"), _SURFACE_VALUES, "unknown")),
        condition=TrackCondition(
            _coerce_enum(raw.get("condition"), _CONDITION_VALUES, "unknown")
        ),
        race_type=RaceType(_coerce_enum(raw.get("race_type"), _RACETYPE_VALUES, "unknown")),
        claiming_price=claiming,
        purse_usd=purse,
        grade=grade,
        age_sex_restrictions=_clean_str(raw.get("age_sex_restrictions")),
        weight_conditions=_clean_str(raw.get("weight_conditions")),
        post_time=_clean_str(raw.get("post_time")),
        weather=_clean_str(raw.get("weather")),
    )

    entries_raw = raw.get("entries") or []
    if not isinstance(entries_raw, list):
        entries_raw = []

    entries: list[HorseEntry] = []
    for er in entries_raw:
        if not isinstance(er, dict):
            continue
        he = _build_horse_entry(er)
        if he is not None:
            entries.append(he)

    # Confidence: 0.4 baseline for parseable header + 0.6 * pp_coverage.
    # This mirrors `BrisnetParser._parse_race_block`'s formula so downstream
    # `has_enough_data` and ranking behave consistently across parsers.
    if entries:
        pp_score = sum(1 for e in entries if e.n_pp >= 1) / len(entries)
    else:
        pp_score = 0.0
    header_score = 1.0  # LLM successfully produced a header
    confidence = round(header_score * 0.4 + pp_score * 0.6, 3)

    return ParsedRace(
        header=header,
        entries=entries,
        parse_confidence=confidence,
        parse_warnings=[],
    )
