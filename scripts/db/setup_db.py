"""Create the master training database.

Executes scripts/db/schema.sql against the configured SQLite path. Idempotent:
every CREATE statement uses IF NOT EXISTS, so re-running this script on an
existing DB is a no-op.

Usage:
    python scripts/db/setup_db.py
    python scripts/db/setup_db.py --db-path /tmp/test.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Allow direct script invocation (`python scripts/db/setup_db.py`) in addition
# to module invocation (`python -m scripts.db.setup_db`).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import structlog

from scripts.db.constants import DB_PATH, SCHEMA_PATH

log = structlog.get_logger(__name__)


def setup_db(db_path: Path = DB_PATH, schema_path: Path = SCHEMA_PATH) -> None:
    """Create or update the master DB at `db_path` using DDL at `schema_path`."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ddl = schema_path.read_text(encoding="utf-8")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(ddl)
        conn.commit()
    finally:
        conn.close()

    log.info("setup_db.complete", db_path=str(db_path), schema_path=str(schema_path))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the master training database.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help=f"SQLite DB file (default: {DB_PATH})")
    parser.add_argument("--schema-path", type=Path, default=SCHEMA_PATH,
                        help=f"Schema SQL file (default: {SCHEMA_PATH})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    setup_db(db_path=args.db_path, schema_path=args.schema_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
