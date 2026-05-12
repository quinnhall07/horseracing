"""End-to-end Phase 0 pipeline test.

Drives a synthetic CSV through every stage:
    map_and_clean → quality_gate → load_to_db → load_to_db (again)

Asserts the four non-negotiables from DATA_PIPELINE.md §1:
    1. Idempotency: a second load adds zero new rows.
    2. Quality gate rejects bad rows.
    3. Dedup keys are stable across runs.
    4. Foreign-key cascades work (race_results → races/horses/jockeys/trainers).

This test does NOT touch Kaggle or the network — it bypasses ingest_kaggle.py
by writing a fake `_dataset_id` sidecar and a CSV directly into staging.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

from scripts.db.constants import SCHEMA_VERSION
from scripts.db.load_to_db import load_parquet_to_db
from scripts.db.map_and_clean import map_and_clean
from scripts.db.quality_gate import run_quality_gate
from scripts.db.setup_db import setup_db


# ─── synthetic dataset ────────────────────────────────────────────────────
# Columns match the joebeachcapital/horse-racing field map exactly.

_GOOD_ROWS = [
    # Race 1 at CD on 2026-05-10 — 3 horses, well-formed, all should accept.
    dict(track="CD", date="2026-05-10", race=1, distance=6.0, surface="D",
         condition="FT", race_type="claiming", claiming_price=20000, purse=40000,
         horse="Speed Demon", finish=1, post=3, jockey="Mike Smith",
         trainer="John Ortiz", odds=3.5, weight=120, speed_rating=92,
         frac1="22.40", frac2="46.10", final_time="1:11.60"),
    dict(track="CD", date="2026-05-10", race=1, distance=6.0, surface="D",
         condition="FT", race_type="claiming", claiming_price=20000, purse=40000,
         horse="Lovely Words", finish=2, post=1, jockey="Joel Rosario",
         trainer="Bob Baffert", odds=4.2, weight=121, speed_rating=88,
         frac1="22.40", frac2="46.10", final_time="1:11.85"),
    dict(track="CD", date="2026-05-10", race=1, distance=6.0, surface="D",
         condition="FT", race_type="claiming", claiming_price=20000, purse=40000,
         horse="Northern Star", finish=3, post=2, jockey="John Velazquez",
         trainer="Todd Pletcher", odds=2.5, weight=119, speed_rating=85,
         frac1="22.40", frac2="46.10", final_time="1:12.10"),

    # Race 2 at CD same day — 2 horses, both well-formed.
    dict(track="CD", date="2026-05-10", race=2, distance=8.0, surface="T",
         condition="firm", race_type="allowance", claiming_price=None, purse=80000,
         horse="Turf Master", finish=1, post=2, jockey="Irad Ortiz",
         trainer="Chad Brown", odds=2.1, weight=124, speed_rating=98,
         frac1="23.10", frac2="47.20", final_time="1:35.40"),
    dict(track="CD", date="2026-05-10", race=2, distance=8.0, surface="T",
         condition="firm", race_type="allowance", claiming_price=None, purse=80000,
         horse="Grass Hopper", finish=2, post=1, jockey="Flavien Prat",
         trainer="Brad Cox", odds=3.0, weight=124, speed_rating=95,
         frac1="23.10", frac2="47.20", final_time="1:35.65"),
]

_BAD_ROWS = [
    # Hard failure: missing finish_position → 0.0, must be rejected.
    dict(track="CD", date="2026-05-10", race=3, distance=6.0, surface="D",
         condition="FT", race_type="claiming", claiming_price=10000, purse=20000,
         horse="No Finish Horse", finish=None, post=1, jockey="Mike Smith",
         trainer="John Ortiz", odds=4.0, weight=120, speed_rating=80,
         frac1="22.40", frac2="46.10", final_time="1:11.60"),

    # Hard failure: missing horse name → 0.0.
    dict(track="CD", date="2026-05-10", race=3, distance=6.0, surface="D",
         condition="FT", race_type="claiming", claiming_price=10000, purse=20000,
         horse=None, finish=1, post=2, jockey="Mike Smith",
         trainer="John Ortiz", odds=2.0, weight=120, speed_rating=85,
         frac1="22.40", frac2="46.10", final_time="1:11.60"),
]


# ─── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def staging_dir(tmp_path: Path) -> Path:
    """Create staging dir with synthetic CSV + _dataset_id sidecar."""
    d = tmp_path / "staging" / "joebeachcapital__horse-racing"
    d.mkdir(parents=True)
    csv = d / "results.csv"
    pd.DataFrame(_GOOD_ROWS + _BAD_ROWS).to_csv(csv, index=False)
    (d / "_dataset_id").write_text("1", encoding="utf-8")
    return d


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Fresh master DB with schema applied + a fake datasets row id=1."""
    db = tmp_path / "master.db"
    setup_db(db)

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """INSERT INTO datasets
               (id, source, filename, format, jurisdiction, schema_version, notes)
               VALUES (1, 'kaggle:joebeachcapital/horse-racing',
                       'results.csv', 'csv', 'US', ?, 'test fixture')""",
            (SCHEMA_VERSION,),
        )
        conn.commit()
    finally:
        conn.close()
    return db


@pytest.fixture
def cleaned_dir(tmp_path: Path) -> Path:
    return tmp_path / "cleaned" / "joebeachcapital__horse-racing"


# ─── helpers ──────────────────────────────────────────────────────────────

def _table_count(db: Path, table: str) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def _all(db: Path, query: str) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return list(conn.execute(query).fetchall())
    finally:
        conn.close()


# ─── full pipeline test ──────────────────────────────────────────────────

def test_full_pipeline_end_to_end(staging_dir: Path, db_path: Path, cleaned_dir: Path):
    # Stage 1: map_and_clean.
    map_summary = map_and_clean(
        input_dir=staging_dir,
        slug="joebeachcapital/horse-racing",
        output_dir=cleaned_dir,
    )
    # All rows should pass Pydantic (even bad ones — quality gate is the gatekeeper).
    # The "no horse name" row will fail at Pydantic stage because horse_name is required.
    assert map_summary["rows_total"] == len(_GOOD_ROWS) + len(_BAD_ROWS)
    # 5 good + 1 missing-finish (passes Pydantic) = 6; 1 missing-horse fails Pydantic.
    assert map_summary["rows_accepted"] == 6
    assert map_summary["rows_rejected"] == 1
    assert (cleaned_dir / "all.parquet").exists()

    # Stage 2: quality_gate.
    qg_summary = run_quality_gate(cleaned_dir)
    assert qg_summary["rows_total"] == 6
    assert qg_summary["rows_accepted"] == 5  # the missing-finish row is rejected
    assert qg_summary["rows_rejected"] == 1
    assert (cleaned_dir / "accepted" / "all.parquet").exists()
    assert (cleaned_dir / "rejected" / "all.parquet").exists()
    assert (cleaned_dir / "rejected" / "reasons.jsonl").exists()

    # Stage 3: load_to_db (first run).
    load_summary = load_parquet_to_db(
        cleaned_dir / "accepted" / "all.parquet", db_path,
    )
    assert load_summary["results_inserted"] == 5
    assert load_summary["results_duplicate"] == 0
    assert load_summary["races_inserted"] == 2  # 2 distinct (CD, 2026-05-10, race 1/2)
    assert load_summary["horses_inserted"] == 5

    # Verify FK cascade: every result has race_id + horse_id.
    null_fks = _all(db_path,
        "SELECT COUNT(*) FROM race_results WHERE race_id IS NULL OR horse_id IS NULL")
    assert null_fks[0][0] == 0

    assert _table_count(db_path, "races")        == 2
    assert _table_count(db_path, "horses")       == 5
    assert _table_count(db_path, "jockeys")      == 5
    assert _table_count(db_path, "trainers")     == 5
    assert _table_count(db_path, "race_results") == 5

    # Stage 4: idempotency — re-running load_to_db must add zero new rows.
    second_summary = load_parquet_to_db(
        cleaned_dir / "accepted" / "all.parquet", db_path,
    )
    assert second_summary["results_inserted"]  == 0
    assert second_summary["results_duplicate"] == 5
    assert second_summary["races_inserted"]    == 0
    assert second_summary["horses_inserted"]   == 0

    # Counts unchanged.
    assert _table_count(db_path, "races")        == 2
    assert _table_count(db_path, "horses")       == 5
    assert _table_count(db_path, "race_results") == 5

    # Audit trail updated on the datasets row.
    ds_row = _all(db_path,
        "SELECT row_count_ingested, row_count_deduped, date_range_start, date_range_end "
        "FROM datasets WHERE id = 1")[0]
    # After two runs: 5 inserted (first run) + 0 (second), 0 deduped (first) + 5 (second).
    assert ds_row[0] == 5  # cumulative inserted
    assert ds_row[1] == 5  # cumulative duplicates seen on second run
    assert ds_row[2] == "2026-05-10"
    assert ds_row[3] == "2026-05-10"


def test_quality_score_persisted_on_results(staging_dir: Path, db_path: Path, cleaned_dir: Path):
    """Every loaded result must have data_quality_score populated."""
    map_and_clean(staging_dir, "joebeachcapital/horse-racing", cleaned_dir)
    run_quality_gate(cleaned_dir)
    load_parquet_to_db(cleaned_dir / "accepted" / "all.parquet", db_path)

    rows = _all(db_path, "SELECT data_quality_score FROM race_results")
    assert len(rows) == 5
    for (score,) in rows:
        assert score is not None
        assert 0.6 <= score <= 1.0  # accepted rows are above threshold


def test_dedup_keys_stable_across_runs(staging_dir: Path, db_path: Path, cleaned_dir: Path):
    """The dedup_key for a given race must be identical across two pipeline runs."""
    map_and_clean(staging_dir, "joebeachcapital/horse-racing", cleaned_dir)
    run_quality_gate(cleaned_dir)
    load_parquet_to_db(cleaned_dir / "accepted" / "all.parquet", db_path)

    keys_first = sorted(k for (k,) in _all(db_path, "SELECT dedup_key FROM races"))

    # Run again from scratch on a different cleaned dir but same staging.
    cleaned_dir2 = cleaned_dir.parent / "cleaned2"
    map_and_clean(staging_dir, "joebeachcapital/horse-racing", cleaned_dir2)
    run_quality_gate(cleaned_dir2)
    load_parquet_to_db(cleaned_dir2 / "accepted" / "all.parquet", db_path)

    keys_second = sorted(k for (k,) in _all(db_path, "SELECT dedup_key FROM races"))
    assert keys_first == keys_second
