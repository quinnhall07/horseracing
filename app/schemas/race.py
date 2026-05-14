"""
app/schemas/race.py
───────────────────
Canonical Pydantic v2 schemas for every racing data structure in the system.

Hierarchy:
  PastPerformanceLine          ← one row in a horse's historical record
  HorseEntry                   ← one horse in today's race (with its PP lines)
  RaceHeader                   ← static metadata for a single race
  ParsedRace                   ← RaceHeader + list[HorseEntry]
  RaceCard                     ← list[ParsedRace] (all races on a day's card)
  IngestionResult              ← top-level API response after PDF upload

All monetary fields are stored as floats in USD.  Odds are stored as the
decimal representation of the implied probability (e.g., 3-1 morning line ⟹
odds_decimal = 4.0).  Fractional times are stored as seconds (float).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.provenance import ModelProvenance


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────


class Surface(StrEnum):
    DIRT = "dirt"
    TURF = "turf"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class TrackCondition(StrEnum):
    # Dirt conditions
    FAST = "fast"
    GOOD = "good"
    SLOPPY = "sloppy"
    MUDDY = "muddy"
    HEAVY = "heavy"
    FROZEN = "frozen"
    # Turf conditions
    FIRM = "firm"
    YIELDING = "yielding"
    SOFT = "soft"
    # Generic fallback
    UNKNOWN = "unknown"


class RaceType(StrEnum):
    MAIDEN_CLAIMING = "maiden_claiming"
    MAIDEN_SPECIAL_WEIGHT = "maiden_special_weight"
    CLAIMING = "claiming"
    ALLOWANCE = "allowance"
    ALLOWANCE_OPTIONAL_CLAIMING = "allowance_optional_claiming"
    STAKES = "stakes"
    GRADED_STAKES = "graded_stakes"
    HANDICAP = "handicap"
    UNKNOWN = "unknown"


class PaceStyle(StrEnum):
    """Assigned by the Pace Scenario Model (Layer 1b).  Populated post-parse."""
    FRONT_RUNNER = "front_runner"
    PRESSER = "presser"
    STALKER = "stalker"
    CLOSER = "closer"
    UNKNOWN = "unknown"


class BetType(StrEnum):
    WIN = "win"
    PLACE = "place"
    SHOW = "show"
    EXACTA = "exacta"
    TRIFECTA = "trifecta"
    SUPERFECTA = "superfecta"
    PICK3 = "pick3"
    PICK4 = "pick4"
    PICK6 = "pick6"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-component: one historical race result for a single horse
# ──────────────────────────────────────────────────────────────────────────────


class PastPerformanceLine(BaseModel):
    """
    One row from a horse's past performance chart.

    Source mapping (Brisnet UP column references in parentheses):
      race_date          — "Date" column
      track_code         — 3-char Equibase track abbreviation
      race_number        — card race number that day
      distance_furlongs  — race distance (stored as furlongs for consistency)
      surface            — dirt / turf / synthetic
      condition          — track condition (fast, sloppy, firm, etc.)
      race_type          — class category
      claiming_price     — relevant only for claiming/MC races; None otherwise
      purse_usd          — total purse in USD
      post_position      — gate draw (1-based)
      finish_position    — official finish (1-based); DQ/disqualification preserved
      lengths_behind     — lengths behind winner at finish; 0.0 if winner
      field_size         — number of starters
      jockey             — jockey name (Last, First format)
      weight_lbs         — weight carried in pounds
      odds_final         — decimal final post-time odds (e.g., 5.0 = 4-1)
      speed_figure       — primary speed figure (Beyer, Brisnet, Equibase; source tagged)
      speed_figure_source— "beyer" | "brisnet" | "equibase" | "unknown"
      fraction_q1        — first call split time in seconds (2f for sprints, 3f for routes)
      fraction_q2        — second call split time in seconds (4f sprints, 6f routes)
      fraction_finish    — final time in seconds
      beaten_lengths_q1  — horse's lengths behind leader at first call
      beaten_lengths_q2  — horse's lengths behind leader at second call
      days_since_prev    — days elapsed since the previous entry in this horse's PP
      comment            — brief trip note / official chart comment (may be None)
    """

    race_date: date
    track_code: Annotated[str, Field(min_length=2, max_length=5)]
    race_number: Annotated[int, Field(ge=1, le=20)]
    distance_furlongs: Annotated[float, Field(ge=2.0, le=20.0)]
    surface: Surface = Surface.UNKNOWN
    condition: TrackCondition = TrackCondition.UNKNOWN
    race_type: RaceType = RaceType.UNKNOWN
    claiming_price: Optional[float] = None
    purse_usd: Optional[float] = None

    post_position: Annotated[int, Field(ge=1, le=24)]
    finish_position: Optional[Annotated[int, Field(ge=1, le=30)]] = None
    lengths_behind: Optional[Annotated[float, Field(ge=0.0)]] = None
    field_size: Optional[Annotated[int, Field(ge=2, le=30)]] = None

    jockey: Optional[str] = None
    weight_lbs: Optional[Annotated[float, Field(ge=95.0, le=145.0)]] = None
    odds_final: Optional[Annotated[float, Field(ge=1.0)]] = None

    speed_figure: Optional[Annotated[float, Field(ge=-20.0, le=150.0)]] = None
    speed_figure_source: str = "unknown"

    # Fractional times (seconds). None when not available for that distance.
    fraction_q1: Optional[Annotated[float, Field(ge=10.0, le=100.0)]] = None
    fraction_q2: Optional[Annotated[float, Field(ge=20.0, le=160.0)]] = None
    fraction_finish: Optional[Annotated[float, Field(ge=40.0, le=300.0)]] = None

    beaten_lengths_q1: Optional[Annotated[float, Field(ge=0.0)]] = None
    beaten_lengths_q2: Optional[Annotated[float, Field(ge=0.0)]] = None

    days_since_prev: Optional[Annotated[int, Field(ge=0, le=1000)]] = None
    comment: Optional[str] = None

    @model_validator(mode="after")
    def _validate_finish_position(self) -> "PastPerformanceLine":
        """If finish_position is 1, lengths_behind must be 0 or None."""
        if self.finish_position == 1 and self.lengths_behind is not None:
            if self.lengths_behind > 0.01:
                raise ValueError(
                    f"finish_position=1 but lengths_behind={self.lengths_behind} > 0"
                )
            self.lengths_behind = 0.0
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Sub-component: one horse entered in today's race
# ──────────────────────────────────────────────────────────────────────────────


class HorseEntry(BaseModel):
    """
    A single horse entered in a specific race on today's card.

    Today's fields (not historical):
      horse_name         — registered name
      horse_id           — Equibase/Jockey Club ID (unique across career); None if unknown
      post_position      — today's gate draw
      morning_line_odds  — decimal ML odds set by track handicapper (e.g., 4.0 = 3-1)
      jockey             — today's jockey
      trainer            — today's trainer
      owner              — registered owner
      weight_lbs         — today's assigned weight
      medication_flags   — e.g., ["L", "B"] for Lasix + Blinkers
      equipment_changes  — e.g., ["blinkers_on", "bar_shoe"]
      pp_lines           — ordered list of past performances (most recent first)
      pace_style         — assigned by pace model after parsing (default UNKNOWN)

    Derived at parse-time (set to None; computed by feature engineering Phase 2):
      ml_implied_prob    — 1 / morning_line_odds (computed here for convenience)
    """

    horse_name: Annotated[str, Field(min_length=1, max_length=100)]
    horse_id: Optional[str] = None
    post_position: Annotated[int, Field(ge=1, le=24)]

    morning_line_odds: Optional[Annotated[float, Field(ge=1.0, le=500.0)]] = None
    ml_implied_prob: Optional[float] = None  # computed in model_validator below

    jockey: Optional[str] = None
    trainer: Optional[str] = None
    owner: Optional[str] = None
    weight_lbs: Optional[Annotated[float, Field(ge=95.0, le=145.0)]] = None

    # Medication and equipment — stored as sorted lists of standardised codes
    medication_flags: list[str] = Field(default_factory=list)
    equipment_changes: list[str] = Field(default_factory=list)

    # Past performance history (most recent first, up to MAX_PP_LINES)
    pp_lines: list[PastPerformanceLine] = Field(default_factory=list)

    # Set by downstream pace model; populated as UNKNOWN at parse time
    pace_style: PaceStyle = PaceStyle.UNKNOWN

    # ── Computed fields ───────────────────────────────────────────────────────
    # These are populated by the feature engineering layer (Phase 2).
    # Stored as None in the raw schema; the FE engine fills them.
    ewm_speed_figure: Optional[float] = None     # EWM(alpha=0.4) of pp speed figs
    days_since_last: Optional[int] = None        # days from most recent PP to race_date
    class_trajectory: Optional[float] = None     # normalized claiming price delta

    # ── Inference-time hydration (Stream A) ───────────────────────────────────
    # Populated by `GET /api/v1/cards/{id}` AFTER the inference pipeline runs.
    # None at parse time; filled in only on the hydrated response, never
    # persisted to the ingestion DB.
    model_prob: Optional[float] = None    # calibrated P(win) from meta-learner
    market_prob: Optional[float] = None   # 1 / morning_line_odds (or live tote)
    edge: Optional[float] = None          # model_prob - market_prob

    @model_validator(mode="after")
    def _compute_ml_implied_prob(self) -> "HorseEntry":
        if self.morning_line_odds is not None and self.ml_implied_prob is None:
            self.ml_implied_prob = 1.0 / self.morning_line_odds
        return self

    @model_validator(mode="after")
    def _sort_pp_lines(self) -> "HorseEntry":
        """Enforce most-recent-first ordering on PP lines."""
        if len(self.pp_lines) > 1:
            self.pp_lines = sorted(
                self.pp_lines, key=lambda p: p.race_date, reverse=True
            )
        return self

    @property
    def last_pp(self) -> Optional[PastPerformanceLine]:
        """Convenience: most recent past performance, or None for first-time starters."""
        return self.pp_lines[0] if self.pp_lines else None

    @property
    def n_pp(self) -> int:
        return len(self.pp_lines)


# ──────────────────────────────────────────────────────────────────────────────
# Race-level header metadata
# ──────────────────────────────────────────────────────────────────────────────


class RaceHeader(BaseModel):
    """
    Static metadata for one race on the card.

    Note on distance: stored BOTH as furlongs (float) and raw string so we
    can reconstruct the original text ("1 1/16 miles", "5½f", etc.) for UI.
    """

    race_number: Annotated[int, Field(ge=1, le=20)]
    race_date: date
    track_code: Annotated[str, Field(min_length=2, max_length=5)]
    track_name: Optional[str] = None
    race_name: Optional[str] = None  # stakes name, or None for overnight races

    distance_furlongs: Annotated[float, Field(ge=2.0, le=20.0)]
    distance_raw: str  # e.g., "1 1/16 Miles", "6 Furlongs"

    surface: Surface = Surface.UNKNOWN
    condition: TrackCondition = TrackCondition.UNKNOWN
    race_type: RaceType = RaceType.UNKNOWN

    claiming_price: Optional[float] = None  # maximum claiming price for the race
    purse_usd: Optional[float] = None
    grade: Optional[Annotated[int, Field(ge=1, le=3)]] = None  # Graded stakes only

    age_sex_restrictions: Optional[str] = None  # e.g., "3yo+", "Fillies & Mares"
    weight_conditions: Optional[str] = None     # e.g., "Weights: 3yo, 120lbs"

    post_time: Optional[str] = None  # local time string; not parsed to datetime
    weather: Optional[str] = None   # raw weather description from track

    @property
    def is_sprint(self) -> bool:
        """Sprints are ≤ 6.5 furlongs per conventional definition."""
        return self.distance_furlongs <= 6.5

    @property
    def is_route(self) -> bool:
        return not self.is_sprint


# ──────────────────────────────────────────────────────────────────────────────
# Assembled race: header + field
# ──────────────────────────────────────────────────────────────────────────────


class ParsedRace(BaseModel):
    """
    One fully assembled race: header metadata + full field of horse entries.

    parse_confidence ∈ [0.0, 1.0]:
        A quality score assigned by the PDF parser reflecting completeness.
        < 0.5  → critical fields missing; treat predictions as unreliable.
        0.5–0.8 → moderate quality; predictions valid but wider uncertainty.
        > 0.8  → high confidence parse; safe for inference.
    """

    header: RaceHeader
    entries: list[HorseEntry] = Field(default_factory=list)
    parse_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    parse_warnings: list[str] = Field(default_factory=list)

    @property
    def field_size(self) -> int:
        return len(self.entries)

    @property
    def has_enough_data(self) -> bool:
        """
        Minimum quality gate: at least 4 horses with ≥ 1 PP line each
        and a parse_confidence above 0.5.
        """
        qualified = sum(1 for e in self.entries if e.n_pp >= 1)
        return qualified >= 4 and self.parse_confidence >= 0.5

    def get_entry_by_post(self, post: int) -> Optional[HorseEntry]:
        for e in self.entries:
            if e.post_position == post:
                return e
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Full day's card
# ──────────────────────────────────────────────────────────────────────────────


class RaceCard(BaseModel):
    """
    All races from a single PDF upload.

    source_filename   — original uploaded filename (stored for audit trail)
    source_format     — detected PDF format ("brisnet_up" | "equibase" | "drf" | "unknown")
    total_pages       — PDF page count (for diagnostics)
    """

    source_filename: str
    source_format: str = "unknown"
    total_pages: int = 0
    card_date: Optional[date] = None  # inferred from races; None if inconsistent
    track_code: Optional[str] = None  # inferred from races
    races: list[ParsedRace] = Field(default_factory=list)
    model_provenance: Optional[ModelProvenance] = None

    @property
    def n_races(self) -> int:
        return len(self.races)

    @property
    def n_qualified_races(self) -> int:
        """Count races that pass the minimum quality gate."""
        return sum(1 for r in self.races if r.has_enough_data)

    @property
    def all_horses(self) -> list[tuple[int, HorseEntry]]:
        """Flat list of (race_number, HorseEntry) across the card."""
        return [
            (race.header.race_number, entry)
            for race in self.races
            for entry in race.entries
        ]


# ──────────────────────────────────────────────────────────────────────────────
# API response wrappers
# ──────────────────────────────────────────────────────────────────────────────


class IngestionResult(BaseModel):
    """
    Top-level response returned by POST /api/v1/ingest/upload.

    success          — True if at least one race was parseable
    card             — the fully structured race card (None if total parse failure)
    card_id          — DB primary key of the persisted card (Stream A); None
                       when persistence didn't run (e.g. failed parse)
    errors           — critical errors that prevented parsing
    processing_ms    — wall-clock parsing time in milliseconds
    """

    success: bool
    card: Optional[RaceCard] = None
    card_id: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    processing_ms: Optional[float] = None


class IngestionStatus(BaseModel):
    """Polling response for long-running async ingestion jobs."""

    job_id: str
    status: str  # "queued" | "parsing" | "feature_engineering" | "done" | "failed"
    progress_pct: float = 0.0
    result: Optional[IngestionResult] = None
    error: Optional[str] = None