"""Pydantic v2 canonical schemas for the Phase 0 pipeline.

These models are the typed handoff between `map_and_clean.py` (which
produces them) and `quality_gate.py` + `load_to_db.py` (which consume them).
Fields are intentionally permissive (mostly Optional) so map_and_clean does
not fail on incomplete rows; the quality gate is responsible for enforcing
data quality, not Pydantic.

Field names exactly match the SQL columns in `schema.sql`.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Canonical(BaseModel):
    """Common config for all canonical models — lenient, immutable rows."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, frozen=False)


class CanonicalHorse(_Canonical):
    name_display:      str
    name_normalized:   str
    foaling_year:      int | None    = None
    country_of_origin: str | None    = None
    sire:              str | None    = None
    dam:               str | None    = None
    dam_sire:          str | None    = None
    color:             str | None    = None
    sex:               str | None    = None
    official_id:       str | None    = None


class CanonicalPerson(_Canonical):
    """Jockeys and trainers share the same shape."""
    name_display:    str
    name_normalized: str
    jurisdiction:    str | None = None


class CanonicalRace(_Canonical):
    # Hard-required identity fields — if any of these are missing, map_and_clean
    # cannot construct a valid race row at all.
    track_code:           str
    race_date:            date
    # Soft-required for load — quality_gate is the gatekeeper. None passes
    # through here so the quality gate can score and reject downstream.
    # race_number is here (not above) because many real datasets identify
    # races by date+track+post-time rather than an explicit race number.
    # The heuristic mapper synthesizes one when missing; load_to_db requires
    # one for the dedup key. quality_gate enforces presence.
    race_number:          int | None   = None
    distance_furlongs:    float | None = None
    surface:              str | None   = None
    jurisdiction:         str | None   = None
    condition:            str | None   = None
    race_type:            str | None   = None
    claiming_price:       float | None = None
    purse_usd:            float | None = None
    grade:                int | None   = None
    field_size:           int | None   = None
    weather:              str | None   = None
    age_sex_restrictions: str | None   = None
    raw_source_id:        str | None   = None


class CanonicalRaceResult(_Canonical):
    """One horse's outcome in one race — the ML training target.

    `race` and `horse` are nested for cleaner downstream code; `load_to_db.py`
    splits them apart, dedupes, and INSERTs. The Pydantic-level shape mirrors
    DATA_PIPELINE.md §2 race_results column order otherwise.
    """
    # Race + horse + connections
    race:    CanonicalRace
    horse:   CanonicalHorse
    jockey:  CanonicalPerson | None = None
    trainer: CanonicalPerson | None = None

    # Result fields
    post_position:       int | None   = None
    finish_position:     int | None   = None
    lengths_behind:      float | None = None
    weight_lbs:          float | None = None
    odds_final:          float | None = None
    speed_figure:        float | None = None
    speed_figure_source: str | None   = None
    fraction_q1_sec:     float | None = None
    fraction_q2_sec:     float | None = None
    fraction_finish_sec: float | None = None
    beaten_lengths_q1:   float | None = None
    beaten_lengths_q2:   float | None = None
    medication_flags:    list[str]    = Field(default_factory=list)
    equipment_changes:   list[str]    = Field(default_factory=list)
    comment:             str | None   = None

    # Provenance — populated by map_and_clean.py from the dataset registry row.
    source_dataset_id:   int | None = None

    # Quality score — populated by quality_gate.py before load.
    data_quality_score:  float | None = None

    def to_parquet_dict(self) -> dict[str, Any]:
        """Flatten nested objects into a row-oriented dict for parquet writing.

        Nested fields are prefixed with their parent name (race_*, horse_*).
        load_to_db.py reads this format back into structured rows.
        """
        out: dict[str, Any] = {}
        out.update({f"race_{k}": v for k, v in self.race.model_dump().items()})
        out.update({f"horse_{k}": v for k, v in self.horse.model_dump().items()})
        if self.jockey:
            out.update({f"jockey_{k}": v for k, v in self.jockey.model_dump().items()})
        else:
            out.update({"jockey_name_display": None, "jockey_name_normalized": None,
                        "jockey_jurisdiction": None})
        if self.trainer:
            out.update({f"trainer_{k}": v for k, v in self.trainer.model_dump().items()})
        else:
            out.update({"trainer_name_display": None, "trainer_name_normalized": None,
                        "trainer_jurisdiction": None})

        for f in ("post_position", "finish_position", "lengths_behind", "weight_lbs",
                  "odds_final", "speed_figure", "speed_figure_source",
                  "fraction_q1_sec", "fraction_q2_sec", "fraction_finish_sec",
                  "beaten_lengths_q1", "beaten_lengths_q2", "comment",
                  "source_dataset_id", "data_quality_score"):
            out[f] = getattr(self, f)

        out["medication_flags"]  = list(self.medication_flags)
        out["equipment_changes"] = list(self.equipment_changes)
        return out
