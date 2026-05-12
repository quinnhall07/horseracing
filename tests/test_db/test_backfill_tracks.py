"""Tests for scripts/db/backfill_tracks.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.db.backfill_tracks import backfill


def _seed_db(db_path: Path) -> None:
    """Create a minimal schema with three races on two tracks."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE tracks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                code          TEXT NOT NULL,
                name          TEXT,
                jurisdiction  TEXT NOT NULL,
                country       TEXT,
                surface_types TEXT,
                UNIQUE(code, jurisdiction)
            );
            CREATE TABLE races (
                id                INTEGER PRIMARY KEY,
                track_code        TEXT,
                jurisdiction      TEXT,
                surface           TEXT,
                race_date         TEXT
            );
            """
        )
        rows = [
            ("CD", "US", "dirt", "2025-05-01"),
            ("CD", "US", "turf", "2025-05-02"),
            ("AQU", "US", "dirt", "2025-05-03"),
        ]
        conn.executemany(
            "INSERT INTO races (track_code, jurisdiction, surface, race_date) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_backfill_inserts_distinct_track_combos(tmp_path):
    db = tmp_path / "test.db"
    _seed_db(db)
    summary = backfill(db)
    assert summary["distinct_combinations"] == 2  # CD and AQU
    assert summary["newly_inserted"] == 2
    assert summary["tracks_in_table"] == 2


def test_backfill_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    _seed_db(db)
    backfill(db)
    second = backfill(db)
    # Re-running adds zero new rows.
    assert second["newly_inserted"] == 0
    assert second["tracks_in_table"] == 2


def test_backfill_aggregates_surfaces(tmp_path):
    db = tmp_path / "test.db"
    _seed_db(db)
    backfill(db)
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT surface_types FROM tracks WHERE code='CD'"
        ).fetchone()
    finally:
        conn.close()
    # JSON-encoded list containing both dirt and turf.
    assert "dirt" in row[0] and "turf" in row[0]
