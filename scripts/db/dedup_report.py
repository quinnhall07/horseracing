"""Print row counts + duplicate-key statistics for the master DB.

Reports per-table totals and quality-score distribution for race_results.
Helpful as a smoke test after every load_to_db.py run.

Because every table has UNIQUE(dedup_key), `actual` duplicates should always
be zero — a non-zero value means the dedup logic in load_to_db is broken.

Usage:
    python scripts/db/dedup_report.py
    python scripts/db/dedup_report.py --db-path data/db/master.db
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


_DEDUPED_TABLES = ("races", "horses", "jockeys", "trainers", "race_results")


def collect_report(db_path: Path) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(
            f"DB not found: {db_path}. Run scripts/db/setup_db.py first."
        )

    conn = sqlite3.connect(db_path)
    try:
        report: dict = {"db_path": str(db_path), "tables": {}, "datasets": []}

        for table in _DEDUPED_TABLES:
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            distinct_keys = conn.execute(
                f"SELECT COUNT(DISTINCT dedup_key) FROM {table}"
            ).fetchone()[0]
            report["tables"][table] = {
                "total":          int(total),
                "distinct_keys":  int(distinct_keys),
                "duplicate_keys": int(total - distinct_keys),
            }

        # race_results quality score distribution.
        rows = conn.execute(
            """SELECT
                 COUNT(*),
                 COUNT(data_quality_score),
                 AVG(data_quality_score),
                 MIN(data_quality_score),
                 MAX(data_quality_score),
                 SUM(CASE WHEN data_quality_score >= 0.90 THEN 1 ELSE 0 END),
                 SUM(CASE WHEN data_quality_score >= 0.80 AND data_quality_score < 0.90 THEN 1 ELSE 0 END),
                 SUM(CASE WHEN data_quality_score >= 0.60 AND data_quality_score < 0.80 THEN 1 ELSE 0 END),
                 SUM(CASE WHEN data_quality_score <  0.60 THEN 1 ELSE 0 END)
               FROM race_results"""
        ).fetchone()
        total, scored, mean, mn, mx, ge90, ge80, ge60, lt60 = rows
        report["quality"] = {
            "results_total":    int(total or 0),
            "results_scored":   int(scored or 0),
            "score_mean":       round(float(mean), 4) if mean is not None else None,
            "score_min":        float(mn) if mn is not None else None,
            "score_max":        float(mx) if mx is not None else None,
            "buckets": {
                "ge_0.90":      int(ge90 or 0),
                "ge_0.80":      int(ge80 or 0),
                "ge_0.60":      int(ge60 or 0),
                "lt_0.60":      int(lt60 or 0),
            },
        }

        # Dataset audit summary.
        cur = conn.execute(
            """SELECT id, source, jurisdiction, row_count_raw,
                      row_count_ingested, row_count_rejected,
                      row_count_deduped, date_range_start, date_range_end
               FROM datasets ORDER BY id"""
        )
        for r in cur.fetchall():
            report["datasets"].append({
                "id":               r[0],
                "source":           r[1],
                "jurisdiction":     r[2],
                "row_count_raw":    r[3],
                "row_count_ingested": r[4],
                "row_count_rejected": r[5],
                "row_count_deduped":  r[6],
                "date_range":       [r[7], r[8]],
            })

        return report
    finally:
        conn.close()


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master DB row + dedup report.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help=f"Master DB (default: {DB_PATH})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    report = collect_report(args.db_path)
    print(json.dumps(report, indent=2, default=str))
    # Non-zero exit if any actual duplicates leaked through.
    leaked = sum(t["duplicate_keys"] for t in report["tables"].values())
    return 1 if leaked > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
