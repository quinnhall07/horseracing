"""Apply field map + transformers to a staged Kaggle dataset.

Reads the staging CSV chosen by `ingest_kaggle.py`, renames columns to the
canonical schema per `field_maps.py`, applies any registered transformers,
constructs `CanonicalRaceResult` objects, and writes the result as parquet
in `data/cleaned/<slug>/all.parquet`.

This stage is intentionally lenient — it accepts every row that has the
absolute-minimum identity fields (track_code, race_date, race_number,
horse name). Rows that fail Pydantic validation are written to
`rejected_pydantic.jsonl` for diagnostics. The proper accept/reject split
based on data quality happens in the next stage (`quality_gate.py`).

Usage:
    python scripts/db/map_and_clean.py \
        --input  data/staging/joebeachcapital__horse-racing/ \
        --map    joebeachcapital/horse-racing \
        --output data/cleaned/joebeachcapital__horse-racing/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import structlog
from pydantic import ValidationError

from scripts.db.constants import CLEANED_DIR, STAGING_DIR
from scripts.db.dedup import normalize_name
from scripts.db.field_maps import get_field_map
from scripts.db.preprocessors import PREPROCESSORS
from scripts.db.schemas import (
    CanonicalHorse,
    CanonicalPerson,
    CanonicalRace,
    CanonicalRaceResult,
)
from scripts.db.transformers import TRANSFORMERS

log = structlog.get_logger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────

def _pick_primary_csv(staging_dir: Path) -> Path:
    """Return the largest CSV in `staging_dir`."""
    csvs = sorted(staging_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV files in staging directory: {staging_dir}")
    return csvs[0]


def _read_dataset_id(staging_dir: Path) -> int | None:
    """Read the FK sidecar written by `ingest_kaggle.py`."""
    sidecar = staging_dir / "_dataset_id"
    if not sidecar.exists():
        log.warning("map.no_dataset_id_sidecar", path=str(sidecar))
        return None
    try:
        return int(sidecar.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as e:
        log.warning("map.bad_dataset_id_sidecar", error=str(e))
        return None


def _is_missing(value: Any) -> bool:
    """Treat NaN, None, empty strings, and pandas NA as missing."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN check
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _resolve_field(spec: Any, row: dict, columns: set[str]) -> Any:
    """Resolve one field-map entry against a single row.

    Spec semantics:
        - str               → copy value from CSV column `spec`; None if missing
        - {"const": value}  → use the literal value
        - None              → return None
    Unknown column references log once and return None.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        if "const" in spec:
            return spec["const"]
        return None
    if isinstance(spec, str):
        if spec not in columns:
            return None
        v = row.get(spec)
        return None if _is_missing(v) else v
    return None


def _apply_transformer(name: str | None, value: Any) -> Any:
    """Look up and apply a registered transformer. Returns value unchanged on miss."""
    if name is None:
        return value
    fn = TRANSFORMERS.get(name)
    if fn is None:
        log.warning("map.unknown_transformer", transformer=name)
        return value
    try:
        return fn(value)
    except Exception as e:  # transformers should not raise, but defend anyway
        log.warning("map.transformer_error", transformer=name, error=str(e))
        return None


def _coerce_date(raw: Any) -> date | None:
    """Best-effort parse of a date-like value into datetime.date."""
    if _is_missing(raw):
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    try:
        ts = pd.to_datetime(raw, errors="coerce", utc=False)
        if pd.isna(ts):
            return None
        return ts.date()
    except (TypeError, ValueError):
        return None


def _coerce_int(raw: Any) -> int | None:
    if _is_missing(raw):
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _coerce_float(raw: Any) -> float | None:
    if _is_missing(raw):
        return None
    try:
        v = float(raw)
        return v
    except (TypeError, ValueError):
        return None


def _coerce_str(raw: Any) -> str | None:
    if _is_missing(raw):
        return None
    s = str(raw).strip()
    return s or None


# ─── per-row construction ─────────────────────────────────────────────────

def _build_race(row: dict, columns: set[str], fm: dict, jurisdiction: str | None) -> dict:
    """Resolve+transform every race field. Returns a dict ready for CanonicalRace."""
    race_fields = fm.get("race_fields", {})
    transformers = fm.get("transformers", {})

    out: dict[str, Any] = {}
    for canonical, spec in race_fields.items():
        raw = _resolve_field(spec, row, columns)
        out[canonical] = _apply_transformer(transformers.get(canonical), raw)

    # Type coercion for the well-typed fields.
    out["race_date"]         = _coerce_date(out.get("race_date"))
    out["race_number"]       = _coerce_int(out.get("race_number"))
    out["distance_furlongs"] = _coerce_float(out.get("distance_furlongs"))
    out["track_code"]        = _coerce_str(out.get("track_code"))
    out["surface"]           = _coerce_str(out.get("surface"))
    out["condition"]         = _coerce_str(out.get("condition"))
    out["race_type"]         = _coerce_str(out.get("race_type"))
    out["claiming_price"]    = _coerce_float(out.get("claiming_price"))
    out["purse_usd"]         = _coerce_float(out.get("purse_usd"))
    out["grade"]             = _coerce_int(out.get("grade"))
    out["field_size"]        = _coerce_int(out.get("field_size"))
    out["weather"]           = _coerce_str(out.get("weather"))
    out["age_sex_restrictions"] = _coerce_str(out.get("age_sex_restrictions"))

    # Jurisdiction always comes from the field map (constant per dataset).
    out["jurisdiction"] = jurisdiction
    return out


def _build_horse(row: dict, columns: set[str], fm: dict) -> dict | None:
    result_fields = fm.get("result_fields", {})
    name_spec = result_fields.get("horse_name")
    raw_name  = _resolve_field(name_spec, row, columns)
    name      = _coerce_str(raw_name)
    if name is None:
        return None

    return {
        "name_display":      name,
        "name_normalized":   normalize_name(name),
        "foaling_year":      _coerce_int(_resolve_field(result_fields.get("foaling_year"), row, columns)),
        "country_of_origin": _coerce_str(_resolve_field(result_fields.get("country_of_origin"), row, columns)),
        "sire":              _coerce_str(_resolve_field(result_fields.get("sire"), row, columns)),
        "dam":               _coerce_str(_resolve_field(result_fields.get("dam"), row, columns)),
        "dam_sire":          _coerce_str(_resolve_field(result_fields.get("dam_sire"), row, columns)),
        "color":             _coerce_str(_resolve_field(result_fields.get("color"), row, columns)),
        "sex":               _coerce_str(_resolve_field(result_fields.get("sex"), row, columns)),
        "official_id":       _coerce_str(_resolve_field(result_fields.get("official_id"), row, columns)),
    }


def _build_person(
    row: dict, columns: set[str], fm: dict, key: str, jurisdiction: str | None
) -> dict | None:
    """Build a CanonicalPerson dict for jockey or trainer; None if name missing."""
    spec = fm.get("result_fields", {}).get(key)
    raw  = _resolve_field(spec, row, columns)
    name = _coerce_str(raw)
    if name is None:
        return None
    return {
        "name_display":    name,
        "name_normalized": normalize_name(name),
        "jurisdiction":    jurisdiction,
    }


def _build_result_fields(row: dict, columns: set[str], fm: dict) -> dict:
    rf = fm.get("result_fields", {})
    transformers = fm.get("transformers", {})

    def t(canonical: str) -> Any:
        return _apply_transformer(transformers.get(canonical), _resolve_field(rf.get(canonical), row, columns))

    return {
        "post_position":       _coerce_int(t("post_position")),
        "finish_position":     _coerce_int(t("finish_position")),
        "lengths_behind":      _coerce_float(t("lengths_behind")),
        "weight_lbs":          _coerce_float(t("weight_lbs")),
        "odds_final":          _coerce_float(t("odds_final")),
        "speed_figure":        _coerce_float(t("speed_figure")),
        "speed_figure_source": _coerce_str(t("speed_figure_source")),
        "fraction_q1_sec":     _coerce_float(t("fraction_q1_sec")),
        "fraction_q2_sec":     _coerce_float(t("fraction_q2_sec")),
        "fraction_finish_sec": _coerce_float(t("fraction_finish_sec")),
        "beaten_lengths_q1":   _coerce_float(t("beaten_lengths_q1")),
        "beaten_lengths_q2":   _coerce_float(t("beaten_lengths_q2")),
        "comment":             _coerce_str(t("comment")),
    }


def build_canonical_row(
    row: dict,
    columns: set[str],
    fm: dict,
    dataset_id: int | None,
) -> CanonicalRaceResult:
    """Construct one CanonicalRaceResult; raises ValidationError on hard failure."""
    jurisdiction = fm.get("jurisdiction")
    race_dict   = _build_race(row, columns, fm, jurisdiction)
    horse_dict  = _build_horse(row, columns, fm)
    if horse_dict is None:
        raise ValueError("missing horse_name")

    jockey_dict  = _build_person(row, columns, fm, "jockey",  jurisdiction)
    trainer_dict = _build_person(row, columns, fm, "trainer", jurisdiction)
    result_fields = _build_result_fields(row, columns, fm)

    return CanonicalRaceResult(
        race=CanonicalRace(**race_dict),
        horse=CanonicalHorse(**horse_dict),
        jockey=CanonicalPerson(**jockey_dict) if jockey_dict else None,
        trainer=CanonicalPerson(**trainer_dict) if trainer_dict else None,
        source_dataset_id=dataset_id,
        **result_fields,
    )


# ─── orchestration ────────────────────────────────────────────────────────

def map_and_clean(
    input_dir: Path,
    slug: str,
    output_dir: Path,
) -> dict:
    """Map+clean every row of the staged CSV. Returns a summary dict.

    Output files (under `output_dir`):
        all.parquet              — every successfully-validated row
        rejected_pydantic.jsonl  — rows that failed Pydantic validation, with reasons
    """
    fm = get_field_map(slug)
    dataset_id = _read_dataset_id(input_dir)

    # If the field map declares a preprocessor (multi-CSV merge, custom
    # cleanup, etc.), use it instead of picking a single CSV.
    preprocess_name = fm.get("preprocess")
    if preprocess_name:
        preproc = PREPROCESSORS.get(preprocess_name)
        if preproc is None:
            raise KeyError(
                f"Field map for {slug!r} references unknown preprocessor "
                f"{preprocess_name!r}. Register it in scripts/db/preprocessors.py."
            )
        log.info("map.start", slug=slug, preprocess=preprocess_name, dataset_id=dataset_id)
        df = preproc(input_dir)
    else:
        csv_path = _pick_primary_csv(input_dir)
        log.info("map.start", slug=slug, csv=str(csv_path), dataset_id=dataset_id)
        df = pd.read_csv(csv_path, low_memory=False)
    columns = set(df.columns)

    accepted: list[dict[str, Any]] = []
    rejects:  list[dict[str, Any]] = []

    for idx, raw_row in enumerate(_iter_records(df)):
        try:
            canonical = build_canonical_row(raw_row, columns, fm, dataset_id)
            accepted.append(canonical.to_parquet_dict())
        except (ValidationError, ValueError, TypeError) as e:
            reason = e.errors() if isinstance(e, ValidationError) else [{"msg": str(e)}]
            rejects.append({
                "row_index": idx,
                "reason":    reason,
                "raw":       {k: (None if _is_missing(v) else v) for k, v in raw_row.items()},
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "all.parquet"
    if accepted:
        pd.DataFrame(accepted).to_parquet(parquet_path, index=False)
    else:
        log.warning("map.no_accepted_rows", csv=str(csv_path))

    rejects_path = output_dir / "rejected_pydantic.jsonl"
    if rejects:
        with rejects_path.open("w", encoding="utf-8") as f:
            for r in rejects:
                f.write(json.dumps(r, default=str) + "\n")

    summary = {
        "slug":           slug,
        "source":         preprocess_name or str(_pick_primary_csv(input_dir)),
        "dataset_id":     dataset_id,
        "rows_total":     len(df),
        "rows_accepted":  len(accepted),
        "rows_rejected":  len(rejects),
        "output_parquet": str(parquet_path) if accepted else None,
        "output_rejects": str(rejects_path) if rejects else None,
    }
    log.info("map.complete", **{k: v for k, v in summary.items() if k not in ("source",)})
    return summary


def _iter_records(df: pd.DataFrame) -> Iterable[dict]:
    """Yield row dicts where NaN is preserved as float('nan') for _is_missing()."""
    for _, row in df.iterrows():
        yield row.to_dict()


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply field map + transformers to a staged CSV.")
    parser.add_argument("--input",  type=Path, required=True,
                        help="Staging directory (contains the CSV + _dataset_id sidecar)")
    parser.add_argument("--map",    required=True, dest="slug",
                        help='Kaggle slug, e.g. "joebeachcapital/horse-racing"')
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Cleaned output dir (default: {CLEANED_DIR}/<slug>/)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output_dir = args.output or (CLEANED_DIR / args.slug.replace("/", "__"))
    summary = map_and_clean(args.input, args.slug, output_dir)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["rows_accepted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
