"""Field mapping registry — one entry per onboarded Kaggle dataset.

Each entry tells `map_and_clean.py` how to translate a dataset's columns into
the canonical schema defined in DATA_PIPELINE.md §2.

Value semantics for race_fields / result_fields:
    "src_column"             → copy value from CSV column `src_column`
    {"const": <value>}       → use the literal value for every row
    None                     → leave the canonical field NULL

The `transformers` section is applied AFTER column renaming; each value is a
transformer name registered in `scripts.db.transformers.TRANSFORMERS`.

Source of truth: DATA_PIPELINE.md §5 (drafts) + this file (live).
When the actual CSV column names diverge from a draft (likely on first run),
update this file and re-run map_and_clean.py — not the other way around.
"""

from __future__ import annotations

from typing import Any

FieldValue = str | dict[str, Any] | None
FieldDict  = dict[str, FieldValue]


FIELD_MAPS: dict[str, dict[str, Any]] = {

    # ─────────────────────────────────────────────────────────────────────
    # US — Primary dataset for Phase 0.
    # Beyer speed figures + fraction times = strong ML signal alignment
    # with the Brisnet PDFs that drive live inference.
    # ─────────────────────────────────────────────────────────────────────
    "joebeachcapital/horse-racing": {
        "source_format": "csv",
        "jurisdiction":  "US",
        "race_fields": {
            "track_code":         "track",
            "race_date":          "date",
            "race_number":        "race",
            "distance_furlongs":  "distance",            # already furlongs
            "surface":            "surface",
            "condition":          "condition",
            "race_type":          "race_type",
            "claiming_price":     "claiming_price",
            "purse_usd":          "purse",
        },
        "result_fields": {
            "horse_name":         "horse",
            "finish_position":    "finish",
            "post_position":      "post",
            "jockey":             "jockey",
            "trainer":            "trainer",
            "odds_final":         "odds",
            "weight_lbs":         "weight",
            "speed_figure":       "speed_rating",
            "speed_figure_source": {"const": "beyer"},
            "fraction_q1_sec":    "frac1",
            "fraction_q2_sec":    "frac2",
            "fraction_finish_sec": "final_time",
        },
        "transformers": {
            "surface":             "normalize_surface",
            "condition":           "normalize_condition",
            "odds_final":          "parse_odds_to_decimal",
            "fraction_q1_sec":     "time_string_to_seconds",
            "fraction_q2_sec":     "time_string_to_seconds",
            "fraction_finish_sec": "time_string_to_seconds",
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # UK/IE — sheikhbarabas, ~744K rows, 2005-2019. Active.
    # Identifies races by date+course+post-time (no explicit race_number).
    # ─────────────────────────────────────────────────────────────────────
    "sheikhbarabas/horse-racing-results-uk-ireland-2005-to-2019": {
        "source_format": "csv",
        "jurisdiction":  "UK",
        "race_fields": {
            "track_code":        "course",
            "race_date":         "date",
            "race_number":       "time",                  # synthesized from "HH:MM"
            "distance_furlongs": "dist.f.",               # already furlongs
            "surface":           "going",                 # uk_going_to_surface
            "condition":         "going",                 # uk_going_to_condition
            "race_type":         "race_group",            # "Flat"/"Hurdle"/"Bumper"/etc.
            "purse_usd":         "prize_money",           # GBP → USD
            "field_size":        "Runners",
        },
        "result_fields": {
            "horse_name":          "horse_name",
            "finish_position":     "pos",
            "jockey":              "jockey",
            "trainer":             "trainer",
            "odds_final":          "dec",                 # already decimal odds
            "weight_lbs":          "lbs",                 # already integer lbs
            "speed_figure":        "rpr",                 # Racing Post Rating
            "fraction_finish_sec": "fin_time",
            "comment":             "comment",
        },
        "transformers": {
            "race_number":         "time_string_to_minutes",  # "5:25" → 325
            "surface":             "uk_going_to_surface",
            "condition":           "uk_going_to_condition",
            "purse_usd":           "gbp_to_usd",
            "fraction_finish_sec": "time_string_to_seconds",
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # UK — Zygmunt; drafted, not active this session.
    # ─────────────────────────────────────────────────────────────────────
    "zygmunt/horse-racing-dataset": {
        "source_format": "csv",
        "jurisdiction":  "UK",
        "race_fields": {
            "track_code":        "venue",
            "race_date":         "date",
            "race_number":       "race_num",
            "distance_furlongs": "dist",                # needs uk_distance_to_furlongs
            "surface":           "going",
            "condition":         "going",               # same source col, different transform
            "race_type":         "type",
            "purse_usd":         "prize",
        },
        "result_fields": {
            "horse_name":        "horse",
            "finish_position":   "position",
            "jockey":            "jockey",
            "trainer":           "trainer",
            "odds_final":        "sp",
            "weight_lbs":        "weight",
            "lengths_behind":    "btn",
        },
        "transformers": {
            "distance_furlongs": "uk_distance_to_furlongs",
            "odds_final":        "uk_sp_to_decimal",
            "weight_lbs":        "stones_to_lbs",
            "surface":           "uk_going_to_surface",
            "condition":         "uk_going_to_condition",
            "purse_usd":         "gbp_to_usd",
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # HK — gdaley/hkracing. Active. ~9.6K downloads (gold standard for HK).
    # Multi-CSV: races.csv joined to runs.csv on race_id.
    # NOTE: only IDs for horse/jockey/trainer (no names). We store the
    # stringified ID as the display name. The dedup_key ensures one row
    # per unique horse_id; we don't get name-based fuzzy matching but the
    # numeric IDs are stable, so cross-race horse identity is preserved.
    # ─────────────────────────────────────────────────────────────────────
    "gdaley/hkracing": {
        "source_format": "csv",
        "jurisdiction":  "HK",
        "preprocess":    "gdaley_hkracing_merge",
        "race_fields": {
            "track_code":        "venue",          # "ST" (Sha Tin) / "HV" (Happy Valley)
            "race_date":         "date",
            "race_number":       "race_no",
            "distance_furlongs": "distance",       # metres → furlongs
            "surface":           "surface",        # 0/1 numeric — needs normalize
            "condition":         "going",          # uses HK going values
            "race_type":         "race_class",
            "purse_usd":         "prize",          # HKD → USD
        },
        "result_fields": {
            "horse_name":          "horse_id",     # stringified int as name
            "finish_position":     "result",
            "post_position":       "draw",
            "jockey":              "jockey_id",    # stringified int
            "trainer":             "trainer_id",   # stringified int
            "odds_final":          "win_odds",
            "weight_lbs":          "actual_weight",  # already in lbs
            "speed_figure":        "horse_rating",
            "fraction_finish_sec": "finish_time",
            "lengths_behind":      "lengths_behind",
        },
        "transformers": {
            "distance_furlongs": "metres_to_furlongs",
            "surface":           "normalize_surface",
            "condition":         "hk_going_to_condition",
            "purse_usd":         "hkd_to_usd",
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # HK — lantanacamara. Active. ~5.4K downloads.
    # Multi-CSV: race-result-race.csv + race-result-horse.csv.
    # Has actual horse/jockey/trainer NAMES (better than gdaley).
    # ─────────────────────────────────────────────────────────────────────
    "lantanacamara/hong-kong-horse-racing": {
        "source_format": "csv",
        "jurisdiction":  "HK",
        "preprocess":    "lantanacamara_hk_merge",
        "race_fields": {
            "track_code":        "race_course",     # "Sha Tin" / "Happy Valley"
            "race_date":         "race_date",
            "race_number":       "race_number",
            "distance_furlongs": "race_distance",   # metres → furlongs
            "surface":           "track",           # 'TURF - "A" COURSE' / 'ALL WEATHER'
            "condition":         "track_condition", # "GOOD TO FIRM" etc.
            "race_type":         "race_class",
            # No purse field
        },
        "result_fields": {
            "horse_name":      "horse_name",
            "finish_position": "finishing_position",
            "post_position":   "draw",
            "jockey":          "jockey",
            "trainer":         "trainer",
            "odds_final":      "win_odds",
            "weight_lbs":      "actual_weight",     # already in lbs
            "fraction_finish_sec": "finish_time",   # HK format "M.SS.tt" — close to standard
            "lengths_behind":      "length_behind_winner",
        },
        "transformers": {
            "distance_furlongs": "metres_to_furlongs",
            "surface":           "normalize_surface",
            "condition":         "hk_going_to_condition",
            "fraction_finish_sec": "time_string_to_seconds",
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # JRA (Japan) — takamotoki. Active. 100 MB, 1986-2021.
    # Single CSV (race_result.csv); other CSVs in dataset are supplementary
    # (laptime, odds detail, corner-passing) — not joined this session.
    # All column names in Japanese.
    # ─────────────────────────────────────────────────────────────────────
    "takamotoki/jra-horse-racing-dataset": {
        "source_format": "csv",
        "jurisdiction":  "JP",
        "race_fields": {
            "track_code":        "競馬場名",          # racecourse name (札幌, etc.)
            "race_date":         "レース日付",         # race date
            "race_number":       "レース番号",         # race number
            "distance_furlongs": "距離(m)",           # distance in metres
            "surface":           "芝・ダート区分",      # turf/dirt division
            "condition":         "馬場状態1",          # track condition
            "race_type":         "競争条件",           # race conditions
            "weather":           "天候",              # weather
            "purse_usd":         "賞金(万円)",         # prize money in 10K JPY units
        },
        "result_fields": {
            "horse_name":          "馬名",            # horse name (katakana)
            "finish_position":     "着順",            # finish position
            "post_position":       "馬番",            # horse number = post position
            "jockey":              "騎手",
            "trainer":             "調教師",
            "odds_final":          "単勝",            # win odds (already decimal)
            "weight_lbs":          "斤量",            # weight carried in kg
            "fraction_finish_sec": "タイム",          # finish time "1:34.3"
        },
        "transformers": {
            "distance_furlongs":   "metres_to_furlongs",
            "surface":             "jpn_surface_to_canonical",
            "condition":           "jpn_condition_to_canonical",
            "weight_lbs":          "kg_to_lbs",
            "fraction_finish_sec": "time_string_to_seconds",
            # purse is in units of 10,000 JPY — leave raw; quality_gate doesn't validate purse
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # Argentina — felipetappata. Active. ~323K rows.
    # Spanish column conventions: nro = race number, arena = dirt (sand),
    # cesped = turf (grass). Distance in metres. Weight in kg.
    # No trainer column. Some rows may use "nro_raw" for an unprocessed
    # race-number; "nro" is post-processed and reliable.
    # ─────────────────────────────────────────────────────────────────────
    "felipetappata/thoroughbred-races-in-argentina": {
        "source_format": "csv",
        "jurisdiction":  "AR",
        "race_fields": {
            "track_code":        "venue",            # "San Isidro" / "Palermo" / etc.
            "race_date":         "date",
            "race_number":       "nro",              # already integer
            "distance_furlongs": "dist",             # metres → furlongs
            "surface":           "surface",          # arena/cesped/sintetico
            "condition":         "cond",
        },
        "result_fields": {
            "horse_name":      "horse",
            "finish_position": "pos",
            "jockey":          "jockey",
            "weight_lbs":      "jockey_weight",      # carried weight (kg → lbs)
        },                                            # NOT "weight" (that's body weight)
        "transformers": {
            "distance_furlongs": "metres_to_furlongs",
            "surface":           "ar_surface_to_canonical",
            "weight_lbs":        "kg_to_lbs",
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # HK — gdaley/horseracing-in-hk drafted in DATA_PIPELINE.md but slug
    # has been renamed to gdaley/hkracing (above). Kept for back-compat.
    # ─────────────────────────────────────────────────────────────────────
    "gdaley/horseracing-in-hk": {
        "source_format": "csv",
        "jurisdiction":  "HK",
        "preprocess":    "gdaley_horseracing_in_hk_merge",
        # Same race-shape as gdaley/hkracing but the runs CSV carries sectional
        # times + per-call lengths-behind — ADR-047 keys off these to make the
        # Pace sub-model trainable. The IDs-only schema (horse_id, jockey_id,
        # trainer_id) means we synthesise display names from the IDs.
        "race_fields": {
            "track_code":        "venue",          # "ST" / "HV"
            "race_date":         "date",
            "race_number":       "race_no",
            "distance_furlongs": "distance",       # metres in source
            "surface":           "surface",        # 0/1 numeric flag
            "condition":         "going",
            "race_type":         "race_class",
            "purse_usd":         "prize",
        },
        "result_fields": {
            "horse_name":        "horse_id",       # stringified int; HK uses IDs
            "finish_position":   "result",
            "jockey":            "jockey_id",
            "trainer":           "trainer_id",
            "odds_final":        "win_odds",
            "weight_lbs":        "actual_weight",
            "speed_figure":      "horse_rating",
            # ADR-047: pace data from the runs.csv sectional columns.
            "fraction_q1_sec":   "time1",          # cumulative seconds at 1st call
            "fraction_q2_sec":   "time2",          # cumulative seconds at 2nd call
            "fraction_finish_sec": "finish_time",
            "beaten_lengths_q1": "behind_sec1",
            "beaten_lengths_q2": "behind_sec2",
        },
        "transformers": {
            "distance_furlongs": "metres_to_furlongs",
            "surface":           "normalize_surface",
            "condition":         "hk_going_to_condition",
            "purse_usd":         "hkd_to_usd",
        },
    },
}


def get_field_map(slug: str) -> dict[str, Any]:
    """Return the field map for a Kaggle slug or raise KeyError with help text."""
    if slug not in FIELD_MAPS:
        known = ", ".join(sorted(FIELD_MAPS)) or "(none)"
        raise KeyError(
            f"No field map registered for dataset {slug!r}. "
            f"Known maps: {known}. Add a new entry to scripts/db/field_maps.py."
        )
    return FIELD_MAPS[slug]
