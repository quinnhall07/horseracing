"""Per-row quality scoring for the cleaned parquet.

Reads `<input>/all.parquet` (produced by `map_and_clean.py`), scores every
row per DATA_PIPELINE.md §8 (hard failures + soft penalties + range checks),
applies cross-row race-level validations, then writes:

    <input>/accepted/all.parquet   — rows with score >= ROW_MIN_SCORE
    <input>/rejected/all.parquet   — rows below threshold
    <input>/rejected/reasons.jsonl — one JSON line per rejected row

The `data_quality_score` column is set on every row before the split so
load_to_db.py can persist it.

Usage:
    python scripts/db/quality_gate.py --input data/cleaned/joebeachcapital__horse-racing/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import structlog

from scripts.db.constants import ROW_MIN_SCORE

log = structlog.get_logger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _present(v: Any) -> bool:
    return not _is_missing(v)


# ─── per-row scoring (DATA_PIPELINE.md §8) ────────────────────────────────

def score_row(row: dict) -> tuple[float, list[str]]:
    """Return (score, list_of_issues) for one race-result row.

    Hard failures short-circuit to 0.0. Otherwise start at 1.0 and subtract
    the penalties below; clamp to [0, 1].
    """
    issues: list[str] = []

    # Hard failures.
    if _is_missing(row.get("race_race_date")):
        return 0.0, ["missing race_date"]
    if _is_missing(row.get("race_track_code")):
        return 0.0, ["missing track_code"]
    if _is_missing(row.get("horse_name_normalized")):
        return 0.0, ["missing horse_name"]
    if _is_missing(row.get("finish_position")):
        return 0.0, ["missing finish_position"]

    # Load-required hard-failures (the spec calls these soft, but distance/
    # surface/jurisdiction/race_number are NOT NULL in the SQL schema and used
    # in the race dedup key — a row missing them cannot be persisted at all).
    if _is_missing(row.get("race_race_number")):
        return 0.0, ["missing race_number (required for dedup_key + SQL load)"]
    if _is_missing(row.get("race_distance_furlongs")):
        return 0.0, ["missing distance_furlongs (required for dedup_key + SQL load)"]
    if _is_missing(row.get("race_surface")):
        return 0.0, ["missing surface (required for dedup_key + SQL load)"]
    if _is_missing(row.get("race_jurisdiction")):
        return 0.0, ["missing jurisdiction (required for SQL load)"]

    score = 1.0

    # Soft failures (column-presence; per spec).
    if _is_missing(row.get("odds_final")):
        score -= 0.10; issues.append("missing odds")
    if _is_missing(row.get("speed_figure")):
        score -= 0.10; issues.append("missing speed figure")
    if _is_missing(row.get("jockey_name_normalized")):
        score -= 0.05; issues.append("missing jockey")
    if _is_missing(row.get("weight_lbs")):
        score -= 0.05; issues.append("missing weight")
    if _is_missing(row.get("fraction_finish_sec")):
        score -= 0.10; issues.append("missing final time")

    # Range validation.
    fp = row.get("finish_position")
    if _present(fp):
        try:
            fp_int = int(fp)
            if fp_int < 1 or fp_int > 30:
                score -= 0.15; issues.append(f"invalid finish_position: {fp_int}")
        except (TypeError, ValueError):
            score -= 0.15; issues.append(f"non-integer finish_position: {fp!r}")

    odds = row.get("odds_final")
    if _present(odds):
        try:
            if float(odds) < 1.0:
                score -= 0.15; issues.append(f"invalid odds: {odds}")
        except (TypeError, ValueError):
            score -= 0.15; issues.append(f"non-numeric odds: {odds!r}")

    dist = row.get("race_distance_furlongs")
    if _present(dist):
        try:
            d = float(dist)
            if not (2.0 <= d <= 20.0):
                score -= 0.20; issues.append(f"implausible distance: {d}f")
        except (TypeError, ValueError):
            score -= 0.20; issues.append(f"non-numeric distance: {dist!r}")

    weight = row.get("weight_lbs")
    if _present(weight):
        try:
            w = float(weight)
            if not (95.0 <= w <= 145.0):
                score -= 0.10; issues.append(f"implausible weight: {w}lbs")
        except (TypeError, ValueError):
            score -= 0.10; issues.append(f"non-numeric weight: {weight!r}")

    return max(0.0, round(score, 4)), issues


# ─── cross-row race-level validation ──────────────────────────────────────

def race_level_issues(df: pd.DataFrame) -> dict[tuple, list[str]]:
    """Return {race_key: [issues...]} for race-level violations.

    Race grouping must match the race_dedup_key formula
    (track, date, race_num, distance, surface) — otherwise datasets where
    the same race_number is reused across multiple physical races (e.g.,
    Argentine venues that run turf + dirt cards in parallel) get falsely
    flagged as "mixed distances" / "duplicate horses".

    Checks:
      - At most one finish_position == 1 per race
      - No horse appears twice in the same race
    Distance/surface checks were removed — they are now part of the grouping
    key, so within a group they're constant by construction.
    """
    out: dict[tuple, list[str]] = {}
    if df.empty:
        return out

    group_cols = [
        "race_race_date", "race_track_code", "race_race_number",
        "race_distance_furlongs", "race_surface",
    ]
    for col in group_cols:
        if col not in df.columns:
            return out

    for race_key, sub in df.groupby(group_cols, dropna=False):
        problems: list[str] = []

        winners = (sub["finish_position"] == 1).sum()
        if winners > 1:
            problems.append(f"multiple winners ({int(winners)})")

        if "horse_name_normalized" in sub.columns:
            dup_horses = sub["horse_name_normalized"].dropna()
            dup_counts = dup_horses.value_counts()
            dups = dup_counts[dup_counts > 1].index.tolist()
            if dups:
                problems.append(f"duplicate horses within race: {dups[:3]}")

        if problems:
            out[tuple(race_key)] = problems

    return out


# ─── orchestration ────────────────────────────────────────────────────────

def run_quality_gate(input_dir: Path) -> dict:
    """Score and split `<input>/all.parquet` into accepted/ + rejected/."""
    parquet_in = input_dir / "all.parquet"
    if not parquet_in.exists():
        raise FileNotFoundError(
            f"No all.parquet at {parquet_in}. Run map_and_clean.py first."
        )

    df = pd.read_parquet(parquet_in)
    if df.empty:
        log.warning("quality_gate.empty_input", path=str(parquet_in))
        return {"input": str(parquet_in), "rows_total": 0,
                "rows_accepted": 0, "rows_rejected": 0}

    # Per-row score.
    scores: list[float] = []
    reasons: list[list[str]] = []
    for rec in df.to_dict(orient="records"):
        s, iss = score_row(rec)
        scores.append(s)
        reasons.append(iss)

    df = df.copy()
    df["data_quality_score"] = scores
    df["_qg_issues"] = reasons

    # Cross-row issues — flag rejected rows in violating races. The grouping
    # key here MUST match the one inside race_level_issues() exactly.
    race_problems = race_level_issues(df)
    if race_problems:
        race_keys_idx = list(zip(
            df["race_race_date"], df["race_track_code"], df["race_race_number"],
            df["race_distance_furlongs"], df["race_surface"],
        ))
        for i, key in enumerate(race_keys_idx):
            if key in race_problems:
                df.at[df.index[i], "_qg_issues"] = (
                    list(df.at[df.index[i], "_qg_issues"]) + race_problems[key]
                )
                # Race-level violations zero out the score.
                df.at[df.index[i], "data_quality_score"] = 0.0
                scores[i] = 0.0

    # Split.
    accepted_mask = df["data_quality_score"] >= ROW_MIN_SCORE
    accepted_df   = df[accepted_mask].drop(columns=["_qg_issues"])
    rejected_df   = df[~accepted_mask]

    accepted_dir = input_dir / "accepted"
    rejected_dir = input_dir / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    accepted_path = accepted_dir / "all.parquet"
    rejected_path = rejected_dir / "all.parquet"
    reasons_path  = rejected_dir / "reasons.jsonl"

    if not accepted_df.empty:
        accepted_df.to_parquet(accepted_path, index=False)
    if not rejected_df.empty:
        rejected_df.drop(columns=["_qg_issues"]).to_parquet(rejected_path, index=False)
        with reasons_path.open("w", encoding="utf-8") as f:
            for _, row in rejected_df.iterrows():
                f.write(json.dumps({
                    "score":  float(row["data_quality_score"]),
                    "issues": list(row["_qg_issues"]),
                    "race_date":   str(row.get("race_race_date")),
                    "track_code":  row.get("race_track_code"),
                    "race_number": int(row["race_race_number"])
                                   if _present(row.get("race_race_number")) else None,
                    "horse":       row.get("horse_name_display"),
                }, default=str) + "\n")

    summary = {
        "input":          str(parquet_in),
        "rows_total":     int(len(df)),
        "rows_accepted":  int(len(accepted_df)),
        "rows_rejected":  int(len(rejected_df)),
        "score_mean":     round(float(df["data_quality_score"].mean()), 4),
        "score_median":   round(float(df["data_quality_score"].median()), 4),
        "race_level_violations": len(race_problems),
        "accepted_path":  str(accepted_path) if not accepted_df.empty else None,
        "rejected_path":  str(rejected_path) if not rejected_df.empty else None,
        "reasons_path":   str(reasons_path)  if not rejected_df.empty else None,
    }
    log.info("quality_gate.complete", **summary)
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-score and split a cleaned parquet.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Cleaned dataset directory (containing all.parquet)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    summary = run_quality_gate(args.input)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["rows_accepted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
