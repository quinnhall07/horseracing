"""Backfill the `tracks` table from existing `races` rows.

The Phase 0 loader never populates `tracks` — it's an aspirational table
referenced by no foreign key. This script derives one row per
(track_code, jurisdiction) combo with surface_types observed on that
combination's races.

Idempotent: re-runs INSERT OR IGNORE on the existing UNIQUE(code, jurisdiction)
constraint, so adding new datasets later and re-running picks up new tracks
without disturbing existing rows.

Usage:
    python scripts/db/backfill_tracks.py
    python scripts/db/backfill_tracks.py --db-path /custom/path.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import structlog

from scripts.db.constants import DB_PATH

log = structlog.get_logger(__name__)


def backfill(db_path: Path) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT track_code, jurisdiction, surface
            FROM races
            WHERE track_code IS NOT NULL
              AND jurisdiction IS NOT NULL
            """
        ).fetchall()

        per_track: dict[tuple[str, str], set[str]] = {}
        for r in rows:
            key = (r["track_code"], r["jurisdiction"])
            per_track.setdefault(key, set()).add(r["surface"] or "unknown")

        n_inserted = 0
        for (code, jurisdiction), surfaces in per_track.items():
            surface_json = json.dumps(sorted(surfaces))
            cur.execute(
                """
                INSERT OR IGNORE INTO tracks (code, name, jurisdiction, country, surface_types)
                VALUES (?, ?, ?, ?, ?)
                """,
                (code, None, jurisdiction, None, surface_json),
            )
            if cur.rowcount > 0:
                n_inserted += 1

        # For rows that were already there (skipped by INSERT OR IGNORE), still
        # update their surface_types so they reflect the latest observation set.
        for (code, jurisdiction), surfaces in per_track.items():
            surface_json = json.dumps(sorted(surfaces))
            cur.execute(
                "UPDATE tracks SET surface_types = ? WHERE code = ? AND jurisdiction = ?",
                (surface_json, code, jurisdiction),
            )

        conn.commit()

        total = cur.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

        summary = {
            "tracks_in_table":      total,
            "distinct_combinations": len(per_track),
            "newly_inserted":        n_inserted,
        }
        log.info("backfill_tracks.complete", **summary)
        return summary
    finally:
        conn.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate tracks table from races.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    summary = backfill(args.db_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
