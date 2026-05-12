"""Score a CSV (or directory containing CSVs) before ingestion.

Implements the rubric in DATA_PIPELINE.md §6. Outputs a JSON report; threshold
is 0.70 for ingestion-eligible.

When `--slug` is given, columns are resolved via the registered field map.
Otherwise the evaluator uses heuristic column matching by name.

Usage:
    python scripts/db/evaluate_dataset.py data/staging/joebeachcapital__horse-racing/
    python scripts/db/evaluate_dataset.py path/to/file.csv --slug joebeachcapital/horse-racing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import structlog

from scripts.db.constants import DATASET_MIN_SCORE
from scripts.db.field_maps import FIELD_MAPS

log = structlog.get_logger(__name__)


# ─── column resolution ────────────────────────────────────────────────────
# Note on the boundary pattern: `\b` does NOT fire between word chars and `_`
# in Python regex (because `_` is a word char), so `\b(date)\b` would miss
# columns like `race_date`. We use lookarounds that treat letters/digits as
# the only "word" chars, so `_` and dashes act as separators.
_BOUND_L = r"(?<![a-zA-Z0-9])"
_BOUND_R = r"(?![a-zA-Z0-9])"


class _FieldSpec:
    """Resolution rule for one canonical field.

    `exact` candidates are tried first (case-insensitive). If none match, the
    `fuzzy` regex is applied to columns NOT matching any `blocklist` pattern.
    The two-phase design fixes the bug where a generic regex like `(name)`
    silently picked `race_name` before `horse_name`.
    """
    __slots__ = ("exact", "fuzzy", "blocklist")

    def __init__(self, exact: list[str],
                 fuzzy: re.Pattern | None = None,
                 blocklist: list[re.Pattern] | None = None):
        self.exact     = exact
        self.fuzzy     = fuzzy
        self.blocklist = blocklist or []


_FIELD_SPECS: dict[str, _FieldSpec] = {
    "race_date": _FieldSpec(
        exact=["race_date", "racedate", "date", "raceday", "raced_on", "raced"],
        fuzzy=re.compile(_BOUND_L + "date" + _BOUND_R, re.I),
        blocklist=[re.compile(r"(updated|modified|created|posted|inserted)", re.I)],
    ),
    "track_code": _FieldSpec(
        exact=["track_code", "trackcode", "track", "track_name", "trackname",
               "venue", "course", "racecourse", "racetrack", "course_name", "course_id"],
        fuzzy=re.compile(_BOUND_L + r"(track|venue|course)" + _BOUND_R, re.I),
        blocklist=[re.compile(r"course[_-]?(country|date|time|class|type)", re.I)],
    ),
    # horse_name: NO fuzzy — `name` alone matches sire/dam/owner/jockey/race_name.
    # Rely entirely on prioritized exact matches.
    "horse_name": _FieldSpec(
        exact=["horse_name", "horsename", "horse", "Horse", "Horse_Name",
               "horse_id", "horseid", "Name", "name"],
        fuzzy=None,
    ),
    # finish_position: prioritize finish-related names; blocklist gate-position
    # columns (post_position, start_position, draw) which are pre-race state.
    "finish_position": _FieldSpec(
        exact=["finish_position", "finishing_position", "finishposition",
               "fin_position", "finish_pos", "finishpos", "FinishPosition",
               "Finish", "finish", "position", "Position", "place", "Place",
               "pos", "Pos", "plc", "rank", "result", "official_finish"],
        fuzzy=re.compile(_BOUND_L + r"(finish|finishing)" + _BOUND_R, re.I),
        blocklist=[
            re.compile(r"post[_-]?position", re.I),
            re.compile(r"start[_-]?position", re.I),
            re.compile(r"draw[_-]?(place|pos|position)", re.I),
            re.compile(r"running[_-]?position", re.I),
        ],
    ),
    "race_number": _FieldSpec(
        exact=["race_number", "racenumber", "race_num", "racenum",
               "race_no", "raceno", "RaceNumber", "RaceNo",
               "race_index", "raceindex", "card_id", "cardid",
               "race", "Race", "race_id", "raceid"],
        fuzzy=None,  # too dangerous; race_date / race_type would match
    ),
    "distance_furlongs": _FieldSpec(
        exact=["distance_furlongs", "distance", "dist", "trip",
               "dist_f", "dist.f.", "Distance"],
        fuzzy=re.compile(_BOUND_L + r"(dist|trip)" + _BOUND_R, re.I),
        blocklist=[re.compile(r"draw[_-]?dist", re.I)],
    ),
    "odds_final": _FieldSpec(
        exact=["odds_final", "final_odds", "odds", "SP", "sp",
               "starting_price", "price", "decimal_odds",
               "dollar_odds", "win_odds", "fixed_odds"],
        fuzzy=re.compile(_BOUND_L + r"(odds|sp)" + _BOUND_R, re.I),
        blocklist=[
            re.compile(r"(morning|ml|board|opening|early|projected)", re.I),
            re.compile(r"(value|exp|expected|implied)", re.I),
            re.compile(r"price[_-]?(money|range)", re.I),
        ],
    ),
}


def _resolve_field(columns: list[str], spec: _FieldSpec) -> str | None:
    """Try exact-match candidates first, then fuzzy with blocklist."""
    lower_to_orig = {c.lower(): c for c in columns}
    for cand in spec.exact:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    if spec.fuzzy is None:
        return None
    for col in columns:
        if any(b.search(col) for b in spec.blocklist):
            continue
        if spec.fuzzy.search(col):
            return col
    return None


# Back-compat alias kept for any external callers; new code should prefer
# _resolve_field with _FIELD_SPECS.
_HEURISTICS: dict[str, list[re.Pattern]] = {
    name: [spec.fuzzy] if spec.fuzzy else []
    for name, spec in _FIELD_SPECS.items()
}


def _heuristic_match(columns: list[str]) -> dict[str, str | None]:
    """Return canonical-field → CSV column, using prioritized resolution."""
    return {name: _resolve_field(columns, spec) for name, spec in _FIELD_SPECS.items()}


def _from_field_map(slug: str) -> dict[str, str | None]:
    """Look up canonical-field → CSV column from the registered map."""
    fm = FIELD_MAPS.get(slug)
    if not fm:
        raise KeyError(f"No field map for {slug!r}")
    out: dict[str, str | None] = {}
    for canonical in ("race_date", "track_code", "distance_furlongs"):
        v = fm.get("race_fields", {}).get(canonical)
        out[canonical] = v if isinstance(v, str) else None
    for canonical in ("horse_name", "finish_position", "odds_final"):
        v = fm.get("result_fields", {}).get(canonical)
        out[canonical] = v if isinstance(v, str) else None
    return out


# ─── scoring ──────────────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "has_race_date":       0.15,
    "has_track":           0.15,
    "has_finish_position": 0.20,
    "has_horse_name":      0.15,
    "has_distance":        0.10,
    "has_odds":            0.10,
    "date_range_ok":       0.05,
    "row_count_ok":        0.05,
    "dup_rate_ok":         0.05,
}


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def evaluate_csv(csv_path: Path, slug: str | None = None) -> dict:
    """Score one CSV. Returns the report dict."""
    df = pd.read_csv(csv_path, low_memory=False)
    columns = list(df.columns)

    if slug:
        col_map = _from_field_map(slug)
    else:
        col_map = _heuristic_match(columns)

    checks: dict[str, bool] = {}
    field_coverage: dict[str, bool | str] = {}
    warnings: list[str] = []

    # Column presence checks.
    for canonical, weight_key in (
        ("race_date",         "has_race_date"),
        ("track_code",        "has_track"),
        ("finish_position",   "has_finish_position"),
        ("horse_name",        "has_horse_name"),
        ("distance_furlongs", "has_distance"),
        ("odds_final",        "has_odds"),
    ):
        src_col = col_map.get(canonical)
        present = bool(src_col) and src_col in columns and df[src_col].notna().any()
        checks[weight_key] = present
        field_coverage[canonical] = src_col if present else False
        if not present:
            warnings.append(f"missing-or-empty column for canonical field '{canonical}'")

    # Date range > 1 year.
    date_range: list[str | None] = [None, None]
    date_col = col_map.get("race_date") if col_map.get("race_date") in columns else None
    if date_col:
        dates = _safe_to_datetime(df[date_col]).dropna()
        if len(dates) > 0:
            date_range = [dates.min().date().isoformat(), dates.max().date().isoformat()]
            span_days = (dates.max() - dates.min()).days
            checks["date_range_ok"] = span_days >= 365
            if not checks["date_range_ok"]:
                warnings.append(f"date range only {span_days} days")
        else:
            checks["date_range_ok"] = False
            warnings.append("date column unparseable")
    else:
        checks["date_range_ok"] = False

    # Row count > 1,000 races.
    estimated_results = len(df)
    estimated_races: int | None = None
    track_col = col_map.get("track_code") if col_map.get("track_code") in columns else None
    if date_col and track_col:
        race_num_col = "race_num" if "race_num" in columns else ("race" if "race" in columns else None)
        if race_num_col:
            estimated_races = int(df[[date_col, track_col, race_num_col]].drop_duplicates().shape[0])
        else:
            estimated_races = int(df[[date_col, track_col]].drop_duplicates().shape[0])
    checks["row_count_ok"] = (estimated_races or estimated_results) >= 1000
    if not checks["row_count_ok"]:
        warnings.append(f"low row count: ~{estimated_races or estimated_results}")

    # Duplicate rate < 5%.
    # Treat full-row duplicates as the conservative check.
    dup_rate = 0.0
    if len(df) > 0:
        dup_rate = float(df.duplicated().sum()) / float(len(df))
    checks["dup_rate_ok"] = dup_rate < 0.05
    if not checks["dup_rate_ok"]:
        warnings.append(f"duplicate rate: {dup_rate:.1%}")

    # Compute weighted score.
    score = sum(WEIGHTS[k] for k, passed in checks.items() if passed)
    score = round(score, 4)

    # Jurisdiction guess.
    jurisdiction_guess = "UNKNOWN"
    if slug and slug in FIELD_MAPS:
        jurisdiction_guess = FIELD_MAPS[slug]["jurisdiction"]
    elif track_col:
        sample = df[track_col].dropna().astype(str).head(20).str.upper()
        if sample.str.contains("HKJC|SHA|HAP|VAL", regex=True).any():
            jurisdiction_guess = "HK"
        elif sample.str.match(r"[A-Z]{2,3}$").any():
            jurisdiction_guess = "US"

    return {
        "dataset":           str(csv_path),
        "score":             score,
        "pass":              score >= DATASET_MIN_SCORE,
        "checks":            checks,
        "field_coverage":    field_coverage,
        "sample_rows":       int(min(3, len(df))),
        "estimated_races":   estimated_races,
        "estimated_results": estimated_results,
        "date_range":        date_range,
        "jurisdiction_guess": jurisdiction_guess,
        "warnings":          warnings,
    }


def _resolve_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        if not csvs:
            raise FileNotFoundError(f"No CSV files in directory: {path}")
        return csvs[0]
    raise FileNotFoundError(path)


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a dataset before ingestion.")
    parser.add_argument("path", type=Path, help="CSV file or directory containing one")
    parser.add_argument("--slug", default=None,
                        help="Optional Kaggle slug; uses registered field map instead of heuristics")
    parser.add_argument("--output", type=Path, default=None,
                        help="If set, write JSON report here as well as stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    csv_path = _resolve_csv(args.path)
    report = evaluate_csv(csv_path, slug=args.slug)
    output = json.dumps(report, indent=2, default=str)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
