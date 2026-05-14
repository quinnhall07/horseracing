"""Export the ML-ready training DataFrame from the master DB.

Runs the SQL from DATA_PIPELINE.md §12 — joins races, race_results, horses,
jockeys, and trainers, filters by quality threshold + min field size, and
writes the result to `data/exports/training_<YYYYMMDD>.parquet`. This file
is what `backend/scripts/bootstrap_models.py` consumes to train Phase 3
baseline models.

Usage:
    python scripts/db/export_training_data.py
    python scripts/db/export_training_data.py --output data/exports/custom.parquet \\
                                              --min-score 0.70 --min-field-size 5
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import structlog

from scripts.db.constants import DB_PATH, EXPORTS_DIR, ROW_MIN_SCORE

log = structlog.get_logger(__name__)


# Lifted verbatim from DATA_PIPELINE.md §12. Parametrized for score and
# field_size only — column selection is fixed contract for downstream models.
TRAINING_QUERY_TEMPLATE = """
SELECT
    r.race_date,
    r.track_code,
    r.jurisdiction,
    r.race_number,
    r.distance_furlongs,
    r.surface,
    r.condition,
    r.race_type,
    r.claiming_price,
    r.purse_usd,
    r.field_size,
    rr.post_position,
    rr.finish_position,
    rr.weight_lbs,
    rr.odds_final,
    rr.speed_figure,
    rr.speed_figure_source,
    rr.fraction_q1_sec,
    rr.fraction_q2_sec,
    rr.fraction_finish_sec,
    rr.beaten_lengths_q1,
    rr.beaten_lengths_q2,
    rr.data_quality_score,
    h.dedup_key         AS horse_dedup_key,
    h.name_normalized   AS horse_name,
    h.foaling_year,
    h.sire,
    h.dam_sire,
    j.name_normalized   AS jockey_name,
    t.name_normalized   AS trainer_name
FROM race_results rr
JOIN races r        ON rr.race_id    = r.id
JOIN horses h       ON rr.horse_id   = h.id
LEFT JOIN jockeys j ON rr.jockey_id  = j.id
LEFT JOIN trainers t ON rr.trainer_id = t.id
WHERE rr.data_quality_score >= ?
  AND COALESCE(r.field_size, ?) >= ?
  AND rr.finish_position IS NOT NULL
  {jurisdiction_filter}
ORDER BY r.race_date ASC, r.track_code, r.race_number, rr.post_position;
"""


def export(
    db_path: Path,
    output_path: Path,
    min_score: float = ROW_MIN_SCORE,
    min_field_size: int = 4,
    jurisdictions: list[str] | None = None,
) -> dict:
    """Export training DataFrame as parquet. Returns a summary dict.

    Note: COALESCE(field_size, min_field_size) — if a source dataset didn't
    populate field_size we don't want to drop those rows; we trust them to
    pass the filter. ML training can re-derive field_size from group counts.

    `jurisdictions` is an allow-list. AR is excluded by default callers (CLI
    default is None = all) because of the known race-fragmentation issue
    documented in PROGRESS.md (avg 2.2 horses/race; nro reused across venues).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    params: list = [min_score, min_field_size, min_field_size]
    if jurisdictions:
        placeholders = ",".join("?" * len(jurisdictions))
        jurisdiction_filter = f"AND r.jurisdiction IN ({placeholders})"
        params.extend(jurisdictions)
    else:
        jurisdiction_filter = ""
    query = TRAINING_QUERY_TEMPLATE.format(jurisdiction_filter=jurisdiction_filter)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            query, conn,
            params=tuple(params),
            parse_dates=["race_date"],
        )
    finally:
        conn.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    summary = {
        "db_path":         str(db_path),
        "output_path":     str(output_path),
        "rows_exported":   int(len(df)),
        "min_score":       min_score,
        "min_field_size":  min_field_size,
        "jurisdictions":   jurisdictions or "ALL",
        "date_range":      [
            df["race_date"].min().date().isoformat() if len(df) else None,
            df["race_date"].max().date().isoformat() if len(df) else None,
        ],
        "unique_horses":   int(df["horse_name"].nunique()) if "horse_name" in df.columns else 0,
        "unique_races":    int(df.groupby(["race_date", "track_code", "race_number"]).ngroups)
                            if len(df) else 0,
        "by_jurisdiction": df["jurisdiction"].value_counts().to_dict() if len(df) else {},
    }
    log.info("export.complete", **summary)
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────

def _default_output() -> Path:
    return EXPORTS_DIR / f"training_{date.today().strftime('%Y%m%d')}.parquet"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ML training DataFrame.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help=f"Master DB (default: {DB_PATH})")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output parquet (default: {_default_output()})")
    parser.add_argument("--min-score", type=float, default=ROW_MIN_SCORE,
                        help=f"Minimum data_quality_score (default: {ROW_MIN_SCORE})")
    parser.add_argument("--min-field-size", type=int, default=4,
                        help="Minimum field_size to include (default: 4)")
    parser.add_argument("--jurisdictions", type=str, default=None,
                        help="Comma-separated allow-list (e.g. 'UK,HK,JP'). "
                             "Default: all. Recommended: exclude AR until the "
                             "race-fragmentation issue is fixed (see PROGRESS.md).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output_path = args.output or _default_output()
    jurisdictions = (
        [j.strip().upper() for j in args.jurisdictions.split(",") if j.strip()]
        if args.jurisdictions else None
    )
    summary = export(
        db_path=args.db_path,
        output_path=output_path,
        min_score=args.min_score,
        min_field_size=args.min_field_size,
        jurisdictions=jurisdictions,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["rows_exported"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
