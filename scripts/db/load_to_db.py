"""Idempotent loader from accepted parquet → master SQLite DB.

Reads `<input>/all.parquet` (the accepted output of `quality_gate.py`),
generates SHA-256 dedup keys, and INSERTs into the canonical tables in
dependency order:

    horses → jockeys → trainers → races → race_results

Conflicts on `dedup_key` are silently skipped (INSERT OR IGNORE) per
DATA_PIPELINE.md §3 "the pipeline never overwrites existing rows".

After the load completes, the corresponding row in `datasets` has its
`row_count_ingested` / `row_count_rejected` / `row_count_deduped` updated
so the audit trail stays accurate.

Usage:
    python scripts/db/load_to_db.py \
        --input data/cleaned/joebeachcapital__horse-racing/accepted/
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import structlog

from scripts.db.constants import DB_PATH
from scripts.db.dedup import (
    horse_dedup_key,
    person_dedup_key,
    race_dedup_key,
    result_dedup_key,
)

log = structlog.get_logger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────

def _to_native(v: Any) -> Any:
    """Convert pandas/numpy types to plain Python for sqlite3 binding."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return v
    return v


def _to_int(v: Any) -> int | None:
    v = _to_native(v)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:
    v = _to_native(v)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_str(v: Any) -> str | None:
    v = _to_native(v)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _to_json(v: Any) -> str | None:
    """Serialize list/dict columns; pass strings through; None stays None."""
    v = _to_native(v)
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(list(v) if isinstance(v, tuple) else v, default=str)
    return json.dumps(v, default=str)


# ─── per-table inserters ──────────────────────────────────────────────────
# Each returns (id, was_inserted). was_inserted=False means a duplicate was
# found and we returned the existing id.

def _upsert_horse(conn: sqlite3.Connection, row: dict) -> tuple[int, bool]:
    name_display = _to_str(row.get("horse_name_display"))
    name_norm    = _to_str(row.get("horse_name_normalized")) or (name_display or "").lower()
    foaling_year = _to_int(row.get("horse_foaling_year"))
    country      = _to_str(row.get("horse_country_of_origin"))
    if not name_display or not name_norm:
        raise ValueError("horse missing name")

    key = horse_dedup_key(name_display, foaling_year, country)
    cur = conn.execute("SELECT id FROM horses WHERE dedup_key = ?", (key,))
    existing = cur.fetchone()
    if existing:
        return int(existing[0]), False

    cur = conn.execute(
        """INSERT OR IGNORE INTO horses
           (dedup_key, name_normalized, name_display, foaling_year,
            country_of_origin, sire, dam, dam_sire, color, sex, official_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (key, name_norm, name_display, foaling_year, country,
         _to_str(row.get("horse_sire")), _to_str(row.get("horse_dam")),
         _to_str(row.get("horse_dam_sire")), _to_str(row.get("horse_color")),
         _to_str(row.get("horse_sex")), _to_str(row.get("horse_official_id"))),
    )
    if cur.lastrowid:
        return int(cur.lastrowid), True
    # Race condition fallback: another concurrent inserter beat us.
    cur = conn.execute("SELECT id FROM horses WHERE dedup_key = ?", (key,))
    return int(cur.fetchone()[0]), False


def _upsert_person(
    conn: sqlite3.Connection, table: str, name_display: str | None,
    name_norm: str | None, jurisdiction: str | None,
) -> int | None:
    if not name_display or not name_norm:
        return None
    key = person_dedup_key(name_display, jurisdiction)
    cur = conn.execute(f"SELECT id FROM {table} WHERE dedup_key = ?", (key,))
    existing = cur.fetchone()
    if existing:
        return int(existing[0])

    cur = conn.execute(
        f"""INSERT OR IGNORE INTO {table}
            (dedup_key, name_normalized, name_display, jurisdiction)
            VALUES (?, ?, ?, ?)""",
        (key, name_norm, name_display, jurisdiction),
    )
    if cur.lastrowid:
        return int(cur.lastrowid)
    cur = conn.execute(f"SELECT id FROM {table} WHERE dedup_key = ?", (key,))
    found = cur.fetchone()
    return int(found[0]) if found else None


def _upsert_race(conn: sqlite3.Connection, row: dict) -> tuple[int, bool] | None:
    track_code   = _to_str(row.get("race_track_code"))
    race_date    = _to_str(row.get("race_race_date"))
    race_number  = _to_int(row.get("race_race_number"))
    distance_f   = _to_float(row.get("race_distance_furlongs"))
    surface      = _to_str(row.get("race_surface"))
    jurisdiction = _to_str(row.get("race_jurisdiction"))

    if not (track_code and race_date and race_number and distance_f
            and surface and jurisdiction):
        return None

    key = race_dedup_key(track_code, race_date, race_number, distance_f, surface)
    cur = conn.execute("SELECT id FROM races WHERE dedup_key = ?", (key,))
    existing = cur.fetchone()
    if existing:
        return int(existing[0]), False

    cur = conn.execute(
        """INSERT OR IGNORE INTO races
           (dedup_key, track_code, race_date, race_number, distance_furlongs,
            surface, condition, race_type, claiming_price, purse_usd, grade,
            field_size, jurisdiction, weather, age_sex_restrictions,
            source_dataset_id, raw_source_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            key, track_code, race_date, race_number, distance_f, surface,
            _to_str(row.get("race_condition")),
            _to_str(row.get("race_race_type")),
            _to_float(row.get("race_claiming_price")),
            _to_float(row.get("race_purse_usd")),
            _to_int(row.get("race_grade")),
            _to_int(row.get("race_field_size")),
            jurisdiction,
            _to_str(row.get("race_weather")),
            _to_str(row.get("race_age_sex_restrictions")),
            _to_int(row.get("source_dataset_id")),
            _to_str(row.get("race_raw_source_id")),
        ),
    )
    if cur.lastrowid:
        return int(cur.lastrowid), True
    cur = conn.execute("SELECT id FROM races WHERE dedup_key = ?", (key,))
    return int(cur.fetchone()[0]), False


def _insert_result(
    conn: sqlite3.Connection, row: dict,
    race_id: int, horse_id: int, horse_key: str,
    jockey_id: int | None, trainer_id: int | None,
    race_key: str,
) -> bool:
    """Insert one race_result. Returns True if a new row was added."""
    key = result_dedup_key(race_key, horse_key)
    cur = conn.execute(
        """INSERT OR IGNORE INTO race_results
           (dedup_key, race_id, horse_id, jockey_id, trainer_id,
            post_position, finish_position, lengths_behind, weight_lbs,
            odds_final, speed_figure, speed_figure_source,
            fraction_q1_sec, fraction_q2_sec, fraction_finish_sec,
            beaten_lengths_q1, beaten_lengths_q2,
            medication_flags, equipment_changes, comment,
            source_dataset_id, data_quality_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            key, race_id, horse_id, jockey_id, trainer_id,
            _to_int(row.get("post_position")),
            _to_int(row.get("finish_position")),
            _to_float(row.get("lengths_behind")),
            _to_float(row.get("weight_lbs")),
            _to_float(row.get("odds_final")),
            _to_float(row.get("speed_figure")),
            _to_str(row.get("speed_figure_source")),
            _to_float(row.get("fraction_q1_sec")),
            _to_float(row.get("fraction_q2_sec")),
            _to_float(row.get("fraction_finish_sec")),
            _to_float(row.get("beaten_lengths_q1")),
            _to_float(row.get("beaten_lengths_q2")),
            _to_json(row.get("medication_flags")),
            _to_json(row.get("equipment_changes")),
            _to_str(row.get("comment")),
            _to_int(row.get("source_dataset_id")),
            _to_float(row.get("data_quality_score")),
        ),
    )
    return bool(cur.lastrowid)


# ─── orchestration ────────────────────────────────────────────────────────

def load_parquet_to_db(parquet_path: Path, db_path: Path) -> dict:
    df = pd.read_parquet(parquet_path)

    inserted_results = 0
    duplicate_results = 0
    skipped_rows      = 0
    inserted_races    = 0
    inserted_horses   = 0
    dataset_ids: set[int] = set()
    date_range: list[date | None] = [None, None]

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        for rec in df.to_dict(orient="records"):
            try:
                race_outcome = _upsert_race(conn, rec)
                if race_outcome is None:
                    skipped_rows += 1
                    continue
                race_id, race_was_new = race_outcome
                if race_was_new:
                    inserted_races += 1

                horse_id, horse_was_new = _upsert_horse(conn, rec)
                if horse_was_new:
                    inserted_horses += 1

                jurisdiction = _to_str(rec.get("race_jurisdiction"))
                jockey_id = _upsert_person(
                    conn, "jockeys",
                    _to_str(rec.get("jockey_name_display")),
                    _to_str(rec.get("jockey_name_normalized")),
                    _to_str(rec.get("jockey_jurisdiction")) or jurisdiction,
                )
                trainer_id = _upsert_person(
                    conn, "trainers",
                    _to_str(rec.get("trainer_name_display")),
                    _to_str(rec.get("trainer_name_normalized")),
                    _to_str(rec.get("trainer_jurisdiction")) or jurisdiction,
                )

                # Recompute the keys we need for the result FK.
                race_key = race_dedup_key(
                    _to_str(rec.get("race_track_code")),
                    _to_str(rec.get("race_race_date")),
                    _to_int(rec.get("race_race_number")),
                    _to_float(rec.get("race_distance_furlongs")),
                    _to_str(rec.get("race_surface")),
                )
                horse_key = horse_dedup_key(
                    _to_str(rec.get("horse_name_display")),
                    _to_int(rec.get("horse_foaling_year")),
                    _to_str(rec.get("horse_country_of_origin")),
                )

                inserted = _insert_result(
                    conn, rec, race_id, horse_id, horse_key, jockey_id, trainer_id, race_key,
                )
                if inserted:
                    inserted_results += 1
                else:
                    duplicate_results += 1

                # Track dataset + date range for the audit-trail update.
                ds_id = _to_int(rec.get("source_dataset_id"))
                if ds_id is not None:
                    dataset_ids.add(ds_id)
                rd = _to_str(rec.get("race_race_date"))
                if rd:
                    try:
                        d = date.fromisoformat(rd[:10])
                    except ValueError:
                        d = None
                    if d is not None:
                        if date_range[0] is None or d < date_range[0]:
                            date_range[0] = d
                        if date_range[1] is None or d > date_range[1]:
                            date_range[1] = d

            except (ValueError, sqlite3.IntegrityError) as e:
                skipped_rows += 1
                log.warning("load.row_failed", error=str(e), row_keys=list(rec)[:5])

        conn.commit()

        # Update audit trail in `datasets` for every contributing dataset_id.
        for ds_id in dataset_ids:
            conn.execute(
                """UPDATE datasets
                   SET row_count_ingested = COALESCE(row_count_ingested, 0) + ?,
                       row_count_deduped  = COALESCE(row_count_deduped, 0)  + ?,
                       date_range_start   = MIN(COALESCE(date_range_start, ?), ?),
                       date_range_end     = MAX(COALESCE(date_range_end,   ?), ?)
                   WHERE id = ?""",
                (
                    inserted_results, duplicate_results,
                    date_range[0].isoformat() if date_range[0] else None,
                    date_range[0].isoformat() if date_range[0] else None,
                    date_range[1].isoformat() if date_range[1] else None,
                    date_range[1].isoformat() if date_range[1] else None,
                    ds_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    summary = {
        "input":             str(parquet_path),
        "rows_total":        int(len(df)),
        "results_inserted":  inserted_results,
        "results_duplicate": duplicate_results,
        "races_inserted":    inserted_races,
        "horses_inserted":   inserted_horses,
        "rows_skipped":      skipped_rows,
        "dataset_ids":       sorted(dataset_ids),
        "date_range":        [d.isoformat() if d else None for d in date_range],
    }
    log.info("load.complete", **summary)
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load accepted parquet → master DB.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Either an accepted/ directory containing all.parquet, "
                             "or a path directly to a parquet file.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help=f"Master DB path (default: {DB_PATH})")
    return parser.parse_args(argv)


def _resolve_parquet(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "all.parquet"
        if candidate.exists():
            return candidate
        parquets = sorted(path.glob("*.parquet"))
        if parquets:
            return parquets[0]
    raise FileNotFoundError(f"No parquet file at {path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    parquet_path = _resolve_parquet(args.input)
    if not args.db_path.exists():
        log.error("load.no_db", path=str(args.db_path),
                  hint="Run scripts/db/setup_db.py first.")
        return 2
    summary = load_parquet_to_db(parquet_path, args.db_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
