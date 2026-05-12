"""Download a Kaggle dataset into data/staging/<slug>/.

After download, registers the dataset in the `datasets` table and writes a
`_dataset_id` sidecar file in the staging directory so downstream scripts
can pick up the foreign key without re-running this script.

Credential resolution (first match wins):
    1. --credentials <path>        — explicit file path
    2. KAGGLE_USERNAME + KAGGLE_KEY env vars
    3. ~/.kaggle/kaggle.json       — standard kaggle CLI location
    4. ~/.kaggle/access_token      — fallback for non-standard layouts.
                                     Either JSON ({"username","key"}) or a raw
                                     API key (requires --username or env).

Usage:
    python scripts/db/ingest_kaggle.py --dataset joebeachcapital/horse-racing
    python scripts/db/ingest_kaggle.py -d joebeachcapital/horse-racing --output data/staging/
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import structlog

from scripts.db.constants import DB_PATH, SCHEMA_VERSION, STAGING_DIR
from scripts.db.field_maps import FIELD_MAPS

log = structlog.get_logger(__name__)


# ─── credentials ──────────────────────────────────────────────────────────

class CredentialError(RuntimeError):
    """Could not resolve Kaggle credentials."""


def _set_env_from_json(creds_path: Path) -> bool:
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    user = data.get("username")
    key  = data.get("key")
    if not user or not key:
        return False
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"]      = key
    return True


def resolve_credentials(
    explicit_path: Path | None = None,
    username_override: str | None = None,
) -> None:
    """Populate KAGGLE_USERNAME and KAGGLE_KEY env vars. Raises on failure."""
    # 1. Explicit path.
    if explicit_path:
        if not explicit_path.exists():
            raise CredentialError(f"Credentials file not found: {explicit_path}")
        if _set_env_from_json(explicit_path):
            log.info("kaggle.creds", source=str(explicit_path), format="json")
            return
        # Treat as raw key.
        raw_key = explicit_path.read_text(encoding="utf-8").strip()
        if not raw_key:
            raise CredentialError(f"Credentials file is empty: {explicit_path}")
        user = username_override or os.environ.get("KAGGLE_USERNAME")
        if not user:
            raise CredentialError(
                f"{explicit_path} is not JSON; pass --username or set KAGGLE_USERNAME."
            )
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"]      = raw_key
        log.info("kaggle.creds", source=str(explicit_path), format="raw")
        return

    # 2. Env vars already set.
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        log.info("kaggle.creds", source="env")
        return

    # 3. Standard kaggle.json.
    standard = Path.home() / ".kaggle" / "kaggle.json"
    if standard.exists() and _set_env_from_json(standard):
        log.info("kaggle.creds", source=str(standard), format="json")
        return

    # 4. ~/.kaggle/access_token fallback.
    fallback = Path.home() / ".kaggle" / "access_token"
    if fallback.exists():
        if _set_env_from_json(fallback):
            log.info("kaggle.creds", source=str(fallback), format="json")
            return
        raw_key = fallback.read_text(encoding="utf-8").strip()
        if raw_key:
            user = username_override or os.environ.get("KAGGLE_USERNAME")
            if not user:
                raise CredentialError(
                    f"{fallback} is not JSON; pass --username or set KAGGLE_USERNAME."
                )
            os.environ["KAGGLE_USERNAME"] = user
            os.environ["KAGGLE_KEY"]      = raw_key
            log.info("kaggle.creds", source=str(fallback), format="raw")
            return

    raise CredentialError(
        "No Kaggle credentials found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
        "place kaggle.json at ~/.kaggle/kaggle.json, or pass --credentials."
    )


# ─── download ─────────────────────────────────────────────────────────────

def download_dataset(slug: str, output_dir: Path) -> list[Path]:
    """Download `slug` into `output_dir` and unzip. Returns the extracted files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import lazily so cred env vars are set first.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    log.info("kaggle.download.start", slug=slug, output=str(output_dir))
    api.dataset_download_files(slug, path=str(output_dir), unzip=False, quiet=False)

    # Unzip everything in output_dir.
    extracted: list[Path] = []
    for zip_path in output_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
            extracted.extend(output_dir / name for name in zf.namelist())
        zip_path.unlink()

    log.info("kaggle.download.complete", slug=slug, files=len(extracted))
    return extracted


# ─── dataset registration ─────────────────────────────────────────────────

def register_dataset(
    db_path: Path,
    slug: str,
    filename: str,
    fmt: str,
    jurisdiction: str,
    row_count_raw: int | None = None,
    notes: str = "",
) -> int:
    """Insert (or fetch existing) row in `datasets`. Returns its id."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT id FROM datasets WHERE source = ? AND filename = ? AND schema_version = ?",
            (f"kaggle:{slug}", filename, SCHEMA_VERSION),
        )
        existing = cur.fetchone()
        if existing:
            log.info("dataset.exists", id=existing[0], slug=slug)
            return int(existing[0])

        cur = conn.execute(
            """INSERT INTO datasets
               (source, filename, format, jurisdiction, row_count_raw,
                schema_version, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (f"kaggle:{slug}", filename, fmt, jurisdiction, row_count_raw,
             SCHEMA_VERSION, notes),
        )
        conn.commit()
        dataset_id = int(cur.lastrowid)
        log.info("dataset.registered", id=dataset_id, slug=slug, filename=filename)
        return dataset_id
    finally:
        conn.close()


def _pick_primary_csv(staging_dir: Path) -> Path | None:
    """Return the largest .csv file in staging_dir, or None."""
    csvs = sorted(staging_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0] if csvs else None


def _count_csv_rows(csv_path: Path) -> int | None:
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f) - 1  # subtract header
    except OSError:
        return None


# ─── orchestration ────────────────────────────────────────────────────────

def ingest(
    slug: str,
    output_root: Path = STAGING_DIR,
    db_path: Path = DB_PATH,
    credentials: Path | None = None,
    username: str | None = None,
) -> int:
    """Full ingest flow. Returns the dataset_id assigned in the DB."""
    resolve_credentials(credentials, username)

    staging_dir = output_root / slug.replace("/", "__")
    files = download_dataset(slug, staging_dir)
    primary_csv = _pick_primary_csv(staging_dir)

    if primary_csv is None:
        log.warning("kaggle.no_csv_found", slug=slug, files=[f.name for f in files])
        primary_filename = files[0].name if files else "(empty)"
        fmt = "unknown"
        row_count = None
    else:
        primary_filename = primary_csv.name
        fmt = "csv"
        row_count = _count_csv_rows(primary_csv)

    jurisdiction = FIELD_MAPS.get(slug, {}).get("jurisdiction", "UNKNOWN")

    dataset_id = register_dataset(
        db_path=db_path,
        slug=slug,
        filename=primary_filename,
        fmt=fmt,
        jurisdiction=jurisdiction,
        row_count_raw=row_count,
        notes=f"Auto-ingested from kaggle:{slug}",
    )

    # Sidecar so downstream scripts can read the FK without DB access.
    (staging_dir / "_dataset_id").write_text(str(dataset_id), encoding="utf-8")

    return dataset_id


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset.")
    parser.add_argument("-d", "--dataset", required=True,
                        help='Kaggle slug, e.g. "joebeachcapital/horse-racing"')
    parser.add_argument("--output", type=Path, default=STAGING_DIR,
                        help=f"Staging root (default: {STAGING_DIR})")
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help=f"Master DB (default: {DB_PATH})")
    parser.add_argument("--credentials", type=Path, default=None,
                        help="Path to credentials file (JSON or raw key)")
    parser.add_argument("--username", default=None,
                        help="Kaggle username (only needed if credentials file is a raw key)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        dataset_id = ingest(
            slug=args.dataset,
            output_root=args.output,
            db_path=args.db_path,
            credentials=args.credentials,
            username=args.username,
        )
    except CredentialError as e:
        log.error("kaggle.creds.error", error=str(e))
        return 2
    log.info("ingest.done", dataset_id=dataset_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
