"""Auto-discover, download, evaluate, and ingest Kaggle horse-racing datasets.

Searches Kaggle for the given keywords, deduplicates results, then for each
slug runs the full pipeline:

    download → evaluate → map+clean → quality-gate → load → audit

A slug is skipped (idempotently) if it is already present in the `datasets`
table. After a successful load, all staging/cleaned artifacts are retained
on disk for audit.

Field-map handling:
    - Strict mode (default): only datasets with a hand-written entry in
      `field_maps.FIELD_MAPS` are mapped+loaded. Others are downloaded and
      evaluated only — a "needs-map" report is logged with the actual CSV
      column names so you can write a new entry and re-run.
    - --auto-map: build a synthetic field map at runtime from the heuristic
      column matches in evaluate_dataset.py. Lossy — works for ~50% of
      unregistered datasets, silently mismaps fields on the rest. Inspect
      the rejected/reasons.jsonl after every auto-mapped load.

Usage:
    python scripts/db/auto_ingest.py
    python scripts/db/auto_ingest.py --keywords "horse racing" "horse bet" "horseracing"
    python scripts/db/auto_ingest.py --auto-map --max-per-keyword 30 --min-score 0.65
    python scripts/db/auto_ingest.py --dry-run   # search + evaluate only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import structlog

from scripts.db.constants import (
    CLEANED_DIR,
    DATASET_MIN_SCORE,
    DB_PATH,
    STAGING_DIR,
)
from scripts.db.evaluate_dataset import _FIELD_SPECS, _resolve_field, evaluate_csv
from scripts.db.field_maps import FIELD_MAPS
from scripts.db.ingest_kaggle import (
    CredentialError,
    download_dataset,
    register_dataset,
    resolve_credentials,
)
from scripts.db.load_to_db import load_parquet_to_db
from scripts.db.map_and_clean import _pick_primary_csv, map_and_clean
from scripts.db.quality_gate import run_quality_gate

log = structlog.get_logger(__name__)


DEFAULT_KEYWORDS = [
    "horse racing",
    "horse bet",
    "horseracing",
    "horse race results",
    "thoroughbred",
    "horse betting",
]


# ─── per-slug result tracking ─────────────────────────────────────────────

@dataclass
class SlugResult:
    slug:        str
    status:      str  # "loaded" | "needs_map" | "low_score" | "skipped" | "error" | "no_csv" | "already_ingested"
    score:       float | None = None
    rows_total:  int | None = None
    rows_loaded: int | None = None
    notes:       list[str]   = field(default_factory=list)
    columns:     list[str]   = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])}


# ─── kaggle search ────────────────────────────────────────────────────────

def _ds_attr(ds: object, *names: str) -> Any:
    """Return the first non-None attribute among `names` (handles camel/snake case)."""
    for n in names:
        v = getattr(ds, n, None)
        if v is not None:
            return v
    return None


def _search_paginated(api: object, keyword: str, max_results: int) -> list:
    """Page through Kaggle search results until we have `max_results` or run out."""
    out: list = []
    page = 1
    while len(out) < max_results:
        try:
            results = api.dataset_list(search=keyword, page=page)
        except Exception as e:
            log.warning("discover.page_failed", keyword=keyword, page=page, error=str(e))
            break
        if not results:
            break
        out.extend(results)
        # Kaggle returns 20 results/page; a short page means we're done.
        if len(results) < 20:
            break
        page += 1
    return out[:max_results]


def discover_slugs_with_metadata(
    keywords: list[str], max_per_keyword: int,
) -> list[dict]:
    """Return deduplicated dataset records matching any of `keywords`.

    Each record has: slug, title, size, downloads, last_updated, url, matched_keyword.
    Network-only — no downloads, no DB writes.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    seen: set[str] = set()
    ordered: list[dict] = []
    for kw in keywords:
        log.info("discover.search", keyword=kw)
        results = _search_paginated(api, kw, max_per_keyword)
        for ds in results:
            slug = _ds_attr(ds, "ref") or str(ds)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            ordered.append({
                "slug":            slug,
                "title":           _ds_attr(ds, "title"),
                "size":            _ds_attr(ds, "totalBytes", "total_bytes", "size"),
                "downloads":       _ds_attr(ds, "downloadCount", "download_count"),
                "votes":           _ds_attr(ds, "voteCount", "vote_count"),
                "last_updated":    str(_ds_attr(ds, "lastUpdated", "last_updated") or "") or None,
                "url":             _ds_attr(ds, "url"),
                "matched_keyword": kw,
            })
        log.info("discover.keyword_done", keyword=kw, results_unique_so_far=len(ordered))
    log.info("discover.complete", total_unique=len(ordered))
    return ordered


def discover_slugs(keywords: list[str], max_per_keyword: int) -> list[str]:
    """Slug-only convenience wrapper."""
    return [d["slug"] for d in discover_slugs_with_metadata(keywords, max_per_keyword)]


# ─── DB-side dedup of slugs ───────────────────────────────────────────────

def already_ingested(db_path: Path, slug: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """SELECT 1 FROM datasets
               WHERE source = ? AND row_count_ingested IS NOT NULL
                                AND row_count_ingested > 0
               LIMIT 1""",
            (f"kaggle:{slug}",),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


# ─── heuristic field-map builder ──────────────────────────────────────────

def build_heuristic_map(
    csv_columns: list[str], jurisdiction_guess: str | None,
) -> dict[str, Any]:
    """Build a field map dict from prioritized column heuristics.

    The result mirrors the schema of an entry in `FIELD_MAPS`. Default
    transformers are auto-attached based on which columns were matched, so
    the same heuristic that finds an `sp` column also wires up
    `parse_odds_to_decimal` to convert it. Quality_gate range checks remain
    the unit-mismatch safety net for everything else (e.g., distance unit).
    """
    matches = {name: _resolve_field(csv_columns, spec)
               for name, spec in _FIELD_SPECS.items()}

    surface_col   = _find_one(csv_columns, ("surface", "going", "track_type", "track_surface"))
    condition_col = _find_one(csv_columns, ("condition", "going", "track_condition"))
    race_type_col = _find_one(csv_columns, ("race_type", "type", "class", "race_class"))
    purse_col     = _find_one(csv_columns, ("purse", "prize", "prizemoney", "prize_money", "purse_usd"))
    post_col      = _find_one(csv_columns, ("post_position", "post", "barrier", "draw", "stall"))
    jockey_col    = _find_one(csv_columns, ("jockey", "jockey_name", "rider"))
    trainer_col   = _find_one(csv_columns, ("trainer", "trainer_name"))
    weight_col    = _find_one(csv_columns, ("weight_lbs", "weight", "actual_weight", "lbs"))
    speed_col     = _find_one(csv_columns, ("speed_rating", "speed_figure", "speed_fig",
                                            "rating", "rpr", "or", "official_rating"))
    fin_time_col  = _find_one(csv_columns, ("fin_time", "finish_time", "final_time", "time"))
    comment_col   = _find_one(csv_columns, ("comment", "comments", "trip_note", "in_running"))

    # Auto-attach transformers wherever we matched a column whose values are
    # known to need normalization. This is the key fix vs the previous
    # `transformers: {}` — without these, surface stays as raw "Heavy"/"Good"
    # strings and odds stay as fractional "5/2" strings, both of which break
    # the dedup key (different hash for the same logical race) or get the row
    # rejected at quality_gate.
    transformers: dict[str, str] = {}
    if matches["odds_final"]:
        transformers["odds_final"] = "parse_odds_to_decimal"
    if surface_col:
        # Auto-detect: if the surface col is also being used as condition
        # (i.e. it's a UK going-style string), use the going-aware transformer.
        is_going = surface_col == condition_col
        transformers["surface"] = (
            "uk_going_to_surface" if is_going and (jurisdiction_guess or "").upper() in ("UK","IE","UK/IE","IRE")
            else "normalize_surface"
        )
    if condition_col:
        transformers["condition"] = (
            "uk_going_to_condition" if (jurisdiction_guess or "").upper() in ("UK","IE","UK/IE","IRE")
            else "normalize_condition"
        )
    if fin_time_col:
        transformers["fraction_finish_sec"] = "time_string_to_seconds"
    if weight_col and weight_col.lower() == "weight":
        # `weight` alone is ambiguous — could be lbs or stones-lbs ('9-2').
        # If the column literally is "lbs" or "weight_lbs" the value is already
        # numeric and no transformer is needed; for the bare "weight" column
        # be defensive and try stones_to_lbs (it's a passthrough for floats).
        transformers["weight_lbs"] = "stones_to_lbs"

    return {
        "source_format": "csv",
        "jurisdiction":  jurisdiction_guess or "UNKNOWN",
        "race_fields": {
            "track_code":        matches["track_code"],
            "race_date":         matches["race_date"],
            "race_number":       matches["race_number"],
            "distance_furlongs": matches["distance_furlongs"],
            "surface":           surface_col,
            "condition":         condition_col,
            "race_type":         race_type_col,
            "purse_usd":         purse_col,
        },
        "result_fields": {
            "horse_name":          matches["horse_name"],
            "finish_position":     matches["finish_position"],
            "post_position":       post_col,
            "jockey":              jockey_col,
            "trainer":             trainer_col,
            "odds_final":          matches["odds_final"],
            "weight_lbs":          weight_col,
            "speed_figure":        speed_col,
            "fraction_finish_sec": fin_time_col,
            "comment":             comment_col,
        },
        "transformers": transformers,
    }


def _find_one(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """First exact-match (case-insensitive) of `candidates` in `columns`."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ─── per-slug pipeline driver ─────────────────────────────────────────────

def process_slug(
    slug: str,
    *,
    db_path: Path,
    staging_root: Path,
    cleaned_root: Path,
    min_score: float,
    auto_map: bool,
) -> SlugResult:
    """Run the full pipeline for one slug. Returns a structured result."""
    if already_ingested(db_path, slug):
        return SlugResult(slug=slug, status="already_ingested")

    # 1. Download.
    staging_dir = staging_root / slug.replace("/", "__")
    try:
        download_dataset(slug, staging_dir)
    except Exception as e:
        return SlugResult(slug=slug, status="error", notes=[f"download failed: {e}"])

    # 2. Pick primary CSV — bail if none.
    try:
        csv_path = _pick_primary_csv(staging_dir)
    except FileNotFoundError:
        return SlugResult(slug=slug, status="no_csv", notes=["no .csv in dataset"])

    # 3. Evaluate (with whatever we know — registered map or heuristics).
    eval_slug = slug if slug in FIELD_MAPS else None
    try:
        report = evaluate_csv(csv_path, slug=eval_slug)
    except Exception as e:
        return SlugResult(slug=slug, status="error", notes=[f"evaluate failed: {e}"])

    score = float(report["score"])
    if score < min_score:
        return SlugResult(
            slug=slug, status="low_score", score=score,
            notes=[f"score {score:.2f} < threshold {min_score}",
                   f"warnings: {report.get('warnings', [])[:3]}"],
        )

    # 4. Determine field map.
    import pandas as pd
    columns = list(pd.read_csv(csv_path, nrows=0, low_memory=False).columns)

    if slug in FIELD_MAPS:
        map_slug = slug  # use the registered one
        log.info("auto_ingest.using_registered_map", slug=slug)
    elif auto_map:
        synthetic = build_heuristic_map(columns, report.get("jurisdiction_guess"))
        FIELD_MAPS[slug] = synthetic  # in-memory registration
        map_slug = slug
        log.info("auto_ingest.using_heuristic_map", slug=slug,
                 mapped_columns={k: v for k, v in synthetic["race_fields"].items() if v})
    else:
        return SlugResult(
            slug=slug, status="needs_map", score=score, columns=columns,
            notes=["no registered field map; pass --auto-map or write one in field_maps.py"],
        )

    # 5. Register the dataset row + write the FK sidecar (mimics ingest_kaggle).
    primary_filename = csv_path.name
    row_count_raw = sum(1 for _ in csv_path.open("r", encoding="utf-8", errors="replace")) - 1
    dataset_id = register_dataset(
        db_path=db_path, slug=slug, filename=primary_filename, fmt="csv",
        jurisdiction=FIELD_MAPS[slug].get("jurisdiction", "UNKNOWN"),
        row_count_raw=row_count_raw,
        notes=f"auto-discovered via auto_ingest.py (auto_map={auto_map})",
    )
    (staging_dir / "_dataset_id").write_text(str(dataset_id), encoding="utf-8")

    # 6. Map + clean.
    cleaned_dir = cleaned_root / slug.replace("/", "__")
    try:
        map_summary = map_and_clean(staging_dir, map_slug, cleaned_dir)
    except Exception as e:
        return SlugResult(slug=slug, status="error", score=score,
                          notes=[f"map_and_clean failed: {e}"])

    if map_summary["rows_accepted"] == 0:
        return SlugResult(
            slug=slug, status="error", score=score,
            rows_total=map_summary["rows_total"],
            notes=["map+clean produced 0 valid rows — column names likely off; "
                   "check rejected_pydantic.jsonl"],
            columns=columns,
        )

    # 7. Quality gate.
    try:
        qg_summary = run_quality_gate(cleaned_dir)
    except Exception as e:
        return SlugResult(slug=slug, status="error", score=score,
                          notes=[f"quality_gate failed: {e}"])

    if qg_summary["rows_accepted"] == 0:
        return SlugResult(
            slug=slug, status="error", score=score,
            rows_total=qg_summary["rows_total"],
            notes=[f"quality gate accepted 0 rows; mean score {qg_summary['score_mean']}"],
        )

    # 8. Load to DB.
    accepted_parquet = cleaned_dir / "accepted" / "all.parquet"
    try:
        load_summary = load_parquet_to_db(accepted_parquet, db_path)
    except Exception as e:
        return SlugResult(slug=slug, status="error", score=score,
                          notes=[f"load_to_db failed: {e}"])

    return SlugResult(
        slug=slug, status="loaded", score=score,
        rows_total=qg_summary["rows_total"],
        rows_loaded=load_summary["results_inserted"],
        notes=[f"races inserted: {load_summary['races_inserted']}; "
               f"horses: {load_summary['horses_inserted']}; "
               f"duplicates: {load_summary['results_duplicate']}"],
    )


# ─── orchestration ────────────────────────────────────────────────────────

def auto_ingest(
    keywords: list[str],
    *,
    db_path: Path = DB_PATH,
    staging_root: Path = STAGING_DIR,
    cleaned_root: Path = CLEANED_DIR,
    max_per_keyword: int = 20,
    min_score: float = DATASET_MIN_SCORE,
    auto_map: bool = False,
    dry_run: bool = False,
    credentials: Path | None = None,
    username: str | None = None,
) -> dict:
    """Discover + process every slug matching any keyword. Returns a summary dict.

    With dry_run=True, only the Kaggle search runs — no downloads, no DB writes,
    no map+clean. Output annotates each found slug with whether it is already
    ingested and whether a registered field map exists.
    """
    resolve_credentials(credentials, username)

    if not db_path.exists():
        raise FileNotFoundError(
            f"DB not found: {db_path}. Run scripts/db/setup_db.py first."
        )

    if dry_run:
        previews = discover_slugs_with_metadata(keywords, max_per_keyword)
        for p in previews:
            p["already_ingested"]    = already_ingested(db_path, p["slug"])
            p["has_field_map"]       = p["slug"] in FIELD_MAPS
        # Bucket counts for the human-readable summary line.
        buckets = {
            "total":            len(previews),
            "already_ingested": sum(1 for p in previews if p["already_ingested"]),
            "has_field_map":    sum(1 for p in previews if p["has_field_map"]),
            "needs_map":        sum(1 for p in previews
                                    if not p["has_field_map"] and not p["already_ingested"]),
        }
        log.info("auto_ingest.dry_run_complete", **buckets)
        return {
            "mode":          "dry_run",
            "keywords_used": keywords,
            "buckets":       buckets,
            "previews":      previews,
        }

    slugs = discover_slugs(keywords, max_per_keyword)
    log.info("auto_ingest.start", slugs=len(slugs), auto_map=auto_map)

    results: list[SlugResult] = []
    for i, slug in enumerate(slugs, 1):
        log.info("auto_ingest.processing", slug=slug, n=i, total=len(slugs))
        try:
            r = process_slug(
                slug,
                db_path=db_path,
                staging_root=staging_root,
                cleaned_root=cleaned_root,
                min_score=min_score,
                auto_map=auto_map,
            )
        except Exception as e:
            log.exception("auto_ingest.unexpected_error", slug=slug)
            r = SlugResult(slug=slug, status="error", notes=[f"unexpected: {e}"])
        results.append(r)
        log.info("auto_ingest.slug_done", slug=slug, status=r.status,
                 rows_loaded=r.rows_loaded)

    # Summary by status.
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    summary = {
        "keywords_used":  keywords,
        "slugs_found":    len(slugs),
        "by_status":      by_status,
        "total_rows_loaded": sum(r.rows_loaded or 0 for r in results),
        "results":        [r.to_dict() for r in results],
    }
    log.info("auto_ingest.complete",
             slugs_found=len(slugs),
             by_status=by_status,
             total_rows_loaded=summary["total_rows_loaded"])
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-discover and ingest Kaggle horse-racing datasets.",
    )
    parser.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS,
                        help=f"Search keywords (default: {DEFAULT_KEYWORDS})")
    parser.add_argument("--max-per-keyword", type=int, default=20,
                        help="Max Kaggle results per keyword (default: 20)")
    parser.add_argument("--min-score", type=float, default=DATASET_MIN_SCORE,
                        help=f"Minimum evaluator score to ingest (default: {DATASET_MIN_SCORE})")
    parser.add_argument("--auto-map", action="store_true",
                        help="Build heuristic field maps for unregistered datasets "
                             "(lossy — inspect rejected reasons after).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search + evaluate only; do not map+clean or load.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--credentials", type=Path, default=None,
                        help="Kaggle credentials file (JSON or raw key)")
    parser.add_argument("--username", default=None,
                        help="Kaggle username (only if credentials file is a raw key)")
    parser.add_argument("--output", type=Path, default=None,
                        help="If set, write JSON summary here as well as stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        summary = auto_ingest(
            keywords=args.keywords,
            db_path=args.db_path,
            max_per_keyword=args.max_per_keyword,
            min_score=args.min_score,
            auto_map=args.auto_map,
            dry_run=args.dry_run,
            credentials=args.credentials,
            username=args.username,
        )
    except CredentialError as e:
        log.error("auto_ingest.creds_error", error=str(e))
        return 2

    output = json.dumps(summary, indent=2, default=str)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")

    # Dry-run: success if anything was discovered.
    if summary.get("mode") == "dry_run":
        return 0 if summary["buckets"]["total"] > 0 else 1
    # Otherwise: non-zero only if every slug failed.
    loaded     = summary["by_status"].get("loaded", 0)
    skipped_ok = summary["by_status"].get("already_ingested", 0)
    return 0 if (loaded + skipped_ok) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
