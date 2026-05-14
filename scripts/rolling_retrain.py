"""scripts/rolling_retrain.py
─────────────────────────────
Layer-7 rolling-window retraining entry point (Phase 5b, ADR-044).

Master Reference §4 Layer 7 calls for sub-model retraining on a rolling
window (default 3 years), driven either by a drift-detector trigger or a
scheduled cadence. This is a windowed companion to
`scripts/bootstrap_models.py`:

   bootstrap_models.py     →  uses ALL parquet rows, one-shot baseline.
   rolling_retrain.py      →  uses [as_of_date − window, as_of_date) and is
                              the artifact a cron job or a drift alarm
                              calls into.

Pipeline:
    1. Load the parquet, slice the rolling window.
    2. Three-way time-ordered split (train 60% / calib 20% / test 20%) —
       reuses `validate_calibration._three_way_split` so the cutoff math
       is identical to the calibration-validation report.
    3. For each requested sub-model: build the feature matrix, fit, save
       artifact under `output_dir/<sub_model>/`. Untrained sub-models
       fall through to ADR-026's constant-0.5 stub.
    4. Re-fit the meta-learner on stacked sub-model predictions (mirrors
       `bootstrap_models.py`).
    5. Re-fit the meta-learner calibrator (Platt/isotonic auto-select)
       on the calib slice.
    6. Evaluate on the test slice: per-sub-model AUC, meta-learner ECE,
       calibrated meta-learner Brier + log-loss + ECE. Persist
       `report.json` next to the artifacts.
    7. (Optional) `--skip-if-no-drift` consults the drift-state JSON
       written by the live monitor — if `triggered: false`, log + exit
       with code 2 so cron jobs can branch on "no-op".

Per CLAUDE.md §2 the train/calib/test split is strictly time-ordered.

Usage:
    python scripts/rolling_retrain.py \\
        --parquet data/exports/training_20260512.parquet \\
        --as-of-date 2026-05-13 \\
        --window-years 3 \\
        --output-dir models/rolling/2026-05-13

    # Cron-friendly: only retrain if the drift detector fired
    python scripts/rolling_retrain.py \\
        --parquet data/exports/training_20260512.parquet \\
        --as-of-date 2026-05-13 \\
        --output-dir models/rolling/2026-05-13 \\
        --skip-if-no-drift \\
        --drift-state-path models/baseline_full/drift_state.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# `validate_calibration` is a sibling script and not part of the `app`
# package. Hook the scripts/ dir into sys.path so we can reuse its
# `_three_way_split` helper without duplicating logic.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from validate_calibration import _three_way_split  # type: ignore  # noqa: E402

from app.core.logging import configure_logging, get_logger
from app.services.calibration.calibrator import (
    Calibrator,
    CalibratorConfig,
    brier_score,
    expected_calibration_error,
)
from app.services.calibration.drift import load_drift_state
from app.services.models.connections import ConnectionsModel
from app.services.models.market import MarketModel
from app.services.models.meta_learner import MetaLearner
from app.services.models.pace_scenario import PaceScenarioModel
from app.services.models.sequence import SequenceModel
from app.services.models.speed_form import SpeedFormModel
from app.services.models.training_data import (
    load_training_parquet,
    prepare_training_features,
)

log = get_logger(__name__)

# Exit codes
EXIT_OK = 0
EXIT_SKIPPED_NO_DRIFT = 2

# The five canonical Layer-1 sub-models. Order matters only for predictable
# logging — fit order is independent.
ALL_SUB_MODELS: tuple[str, ...] = (
    "speed_form",
    "pace_scenario",
    "sequence",
    "connections",
    "market",
)


# ── Pure helpers (importable for tests) ────────────────────────────────────


@dataclass(frozen=True)
class WindowSliceResult:
    """Result of slicing a rolling window from a parquet-shaped frame."""
    df: pd.DataFrame
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_rows_in_window: int
    n_rows_original: int


def slice_rolling_window(
    df: pd.DataFrame,
    as_of_date: date,
    window_years: int,
    date_col: str = "race_date",
) -> WindowSliceResult:
    """Return the subset of `df` with `date_col ∈ [as_of - window, as_of)`.

    `as_of_date` is exclusive (half-open right) because the rows on the
    `as_of_date` itself are the candidates we'll bet on next — never train
    on them. `window_start` is inclusive.

    `window_years` is expressed as a calendar-year delta — for 3-year
    windows that's the same as 3*365 days for any realistic span.
    """
    if window_years <= 0:
        raise ValueError(f"window_years must be > 0; got {window_years}")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not in DataFrame")

    as_of_ts = pd.Timestamp(as_of_date)
    # Calendar-year delta — `replace(year=...)` is the cleanest way; falls
    # back to subtracting days when the as-of-date is Feb 29.
    try:
        window_start = as_of_ts.replace(year=as_of_ts.year - window_years)
    except ValueError:
        # Feb 29 → Feb 28 in the offset year.
        window_start = as_of_ts.replace(
            year=as_of_ts.year - window_years, day=28
        )

    dates = pd.to_datetime(df[date_col]).dt.normalize()
    mask = (dates >= window_start) & (dates < as_of_ts)
    out = df.loc[mask].reset_index(drop=True)
    return WindowSliceResult(
        df=out,
        window_start=window_start,
        window_end=as_of_ts,
        n_rows_in_window=int(len(out)),
        n_rows_original=int(len(df)),
    )


def _stack_sub_predictions(
    df: pd.DataFrame,
    *,
    speed_form: SpeedFormModel | None,
    pace: PaceScenarioModel,
    sequence: SequenceModel,
    connections: ConnectionsModel | None,
    market: MarketModel | None,
) -> pd.DataFrame:
    """Mirror of `bootstrap_models._stack_sub_predictions` — tolerant to None
    sub-models (skipped via `--sub-models`). Per ADR-026 untrained sub-models
    fall through to the constant-0.5 stub.
    """
    stacked = df.copy()
    if speed_form is not None:
        stacked["speed_form_proba"] = speed_form.predict_proba(df)
    else:
        stacked["speed_form_proba"] = 0.5
    stacked["pace_scenario_proba"] = pace.predict_proba(df)
    stacked["sequence_proba"] = sequence.predict_proba(df)
    if connections is not None:
        stacked["connections_proba"] = connections.predict_proba(df)
    else:
        stacked["connections_proba"] = 0.5
    if market is not None:
        m = market.predict_proba(df)
    else:
        m = np.full(len(df), 0.5, dtype=float)
    nan_mask = np.isnan(m)
    if nan_mask.any():
        mean = float(np.nanmean(m)) if np.isfinite(np.nanmean(m)) else 0.5
        m = np.where(nan_mask, mean, m)
    stacked["market_proba"] = m
    stacked["market_proba_was_missing"] = nan_mask.astype(int)
    return stacked


def _safe_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """ROC-AUC with the usual single-class / degenerate-input fallback."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_pred))
    except ValueError:
        return None


# ── Orchestrator ────────────────────────────────────────────────────────────


def run_rolling_retrain(
    parquet_path: Path,
    as_of_date: date,
    window_years: int,
    output_dir: Path,
    sub_models: Sequence[str],
    test_fraction: float = 0.20,
    calib_fraction: float = 0.20,
) -> dict:
    """Execute one rolling-retrain cycle and return the summary dict.

    This is the function the CLI wraps; pure-Python so tests can drive it
    directly. Writes artifacts under `output_dir`:
        speed_form/, connections/, market/   — sub-model dirs
        meta_learner/                         — stacking head
        calibrator/                           — Platt/isotonic on meta
        report.json                           — top-level summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = tuple(sub_models)
    for name in requested:
        if name not in ALL_SUB_MODELS:
            raise ValueError(
                f"Unknown sub-model '{name}'. "
                f"Choose from: {', '.join(ALL_SUB_MODELS)}"
            )

    log.info(
        "rolling_retrain.start",
        parquet=str(parquet_path),
        as_of_date=str(as_of_date),
        window_years=window_years,
        sub_models=list(requested),
        output_dir=str(output_dir),
    )

    raw = load_training_parquet(parquet_path)
    sliced = slice_rolling_window(raw, as_of_date, window_years)
    log.info(
        "rolling_retrain.windowed",
        n_in_window=sliced.n_rows_in_window,
        n_original=sliced.n_rows_original,
        window_start=str(sliced.window_start),
        window_end=str(sliced.window_end),
    )

    if sliced.n_rows_in_window == 0:
        raise RuntimeError(
            f"No rows in window [{sliced.window_start}, {sliced.window_end}). "
            f"Parquet date range may not overlap the requested window."
        )

    features = prepare_training_features(sliced.df)

    train, calib, test, calib_cutoff, test_cutoff = _three_way_split(
        features,
        test_fraction=test_fraction,
        calib_fraction=calib_fraction,
    )
    log.info(
        "rolling_retrain.split",
        n_train=len(train),
        n_calib=len(calib),
        n_test=len(test),
        calib_cutoff=str(calib_cutoff),
        test_cutoff=str(test_cutoff),
    )

    summary: dict = {
        "as_of_date": str(as_of_date),
        "window_years": window_years,
        "window_start": str(sliced.window_start.date()),
        "window_end": str(sliced.window_end.date()),
        "parquet": str(parquet_path),
        "output_dir": str(output_dir),
        "n_in_window": sliced.n_rows_in_window,
        "n_train_rows": int(len(train)),
        "n_calib_rows": int(len(calib)),
        "n_test_rows": int(len(test)),
        "calib_cutoff": str(calib_cutoff),
        "test_cutoff": str(test_cutoff),
        "sub_models_requested": list(requested),
        "sub_models_trained": [],
        "sub_models": {},
    }

    # ── Fit sub-models ────────────────────────────────────────────────────
    speed_form: Optional[SpeedFormModel] = None
    pace = PaceScenarioModel()
    sequence = SequenceModel()
    connections: Optional[ConnectionsModel] = None
    market: Optional[MarketModel] = None

    trained: list[str] = []

    if "speed_form" in requested:
        speed_form = SpeedFormModel().fit(train, val_df=calib)
        sf_dir = output_dir / "speed_form"
        speed_form.save(sf_dir)
        summary["sub_models"]["speed_form"] = {
            "path": str(sf_dir),
            "metrics": speed_form.metrics.__dict__ if speed_form.metrics else None,
        }
        trained.append("speed_form")

    if "pace_scenario" in requested:
        # ADR-047: real LightGBM fit when the parquet supplies the per-horse
        # sectional columns (e.g. the gdaley/horseracing-in-hk slug populates
        # them). Falls back to the 0.5 stub otherwise.
        pace_trainable = PaceScenarioModel.is_trainable_with(train)
        if pace_trainable:
            pace = PaceScenarioModel().fit(train, val_df=calib)
            pace_dir = output_dir / "pace_scenario"
            pace.save(pace_dir)
            summary["sub_models"]["pace_scenario"] = {
                "path": str(pace_dir),
                "trainable": True,
                "stub": False,
            }
            trained.append("pace_scenario")
        else:
            summary["sub_models"]["pace_scenario"] = {
                "trainable": False,
                "stub": True,
            }
            log.info("rolling_retrain.pace_stub", reason="fractional data missing or below row threshold")

    if "sequence" in requested:
        # ADR-046: SequenceModel is trainable when torch is importable and the
        # parquet carries a horse identifier (horse_dedup_key or the legacy
        # horse_name|jurisdiction pair). `is_trainable_with` gates both.
        sequence_trainable = SequenceModel.is_trainable_with(train)
        if sequence_trainable:
            sequence = SequenceModel().fit(train, val_df=calib)
            seq_dir = output_dir / "sequence"
            sequence.save(seq_dir)
            summary["sub_models"]["sequence"] = {
                "path": str(seq_dir),
                "trainable": True,
                "stub": False,
            }
            trained.append("sequence")
        else:
            summary["sub_models"]["sequence"] = {"trainable": False, "stub": True}
            log.info(
                "rolling_retrain.sequence_stub",
                reason="torch missing or horse identifier absent",
            )

    if "connections" in requested:
        connections = ConnectionsModel().fit(train, val_df=calib)
        conn_dir = output_dir / "connections"
        connections.save(conn_dir)
        summary["sub_models"]["connections"] = {
            "path": str(conn_dir),
            "n_jockeys": len(connections.jockey_rate),
            "n_trainers": len(connections.trainer_rate),
            "n_pairs": len(connections.pair_rate),
        }
        trained.append("connections")

    if "market" in requested:
        market = MarketModel().fit(train, val_df=calib)
        m_dir = output_dir / "market"
        market.save(m_dir)
        summary["sub_models"]["market"] = {"path": str(m_dir)}
        trained.append("market")

    summary["sub_models_trained"] = trained

    # ── Stack predictions across train / calib / test ─────────────────────
    train_stacked = _stack_sub_predictions(
        train,
        speed_form=speed_form, pace=pace, sequence=sequence,
        connections=connections, market=market,
    )
    calib_stacked = _stack_sub_predictions(
        calib,
        speed_form=speed_form, pace=pace, sequence=sequence,
        connections=connections, market=market,
    )
    test_stacked = _stack_sub_predictions(
        test,
        speed_form=speed_form, pace=pace, sequence=sequence,
        connections=connections, market=market,
    )

    # ── Meta-learner ──────────────────────────────────────────────────────
    meta = MetaLearner().fit(train_stacked, val_df=calib_stacked)
    meta_dir = output_dir / "meta_learner"
    meta.save(meta_dir)
    summary["meta_learner_path"] = str(meta_dir)

    calib_meta_p = meta.predict_proba(calib_stacked)
    test_meta_p = meta.predict_proba(test_stacked)
    test_labels = test_stacked["win"].to_numpy()
    calib_labels = calib_stacked["win"].to_numpy()

    # ── Sub-model AUC on the test slice ───────────────────────────────────
    for name, col in (
        ("speed_form", "speed_form_proba"),
        ("pace_scenario", "pace_scenario_proba"),
        ("sequence", "sequence_proba"),
        ("connections", "connections_proba"),
        ("market", "market_proba"),
    ):
        if name not in requested:
            continue
        auc = _safe_auc(test_labels, test_stacked[col].to_numpy())
        summary["sub_models"].setdefault(name, {})["test_auc"] = auc

    # ── Meta-learner pre-calibration ECE + AUC + Brier ────────────────────
    summary["meta_learner_pre_ece"] = float(
        expected_calibration_error(test_meta_p, test_labels, n_bins=15)
    )
    summary["meta_learner_pre_brier"] = float(brier_score(test_meta_p, test_labels))
    summary["meta_learner_pre_logloss"] = float(_safe_log_loss(test_labels, test_meta_p))
    summary["meta_learner_test_auc"] = _safe_auc(test_labels, test_meta_p)

    # ── Calibrator on calib-slice meta predictions ────────────────────────
    cal = Calibrator(CalibratorConfig(method="auto"))
    cal.fit(calib_meta_p, calib_labels)
    cal_dir = output_dir / "calibrator"
    cal.save(cal_dir)

    post_test = cal.predict_proba(test_meta_p)
    summary["calibrator_path"] = str(cal_dir)
    summary["calibrator_chosen_method"] = cal.chosen_method
    summary["meta_learner_ece"] = float(
        expected_calibration_error(post_test, test_labels, n_bins=15)
    )
    summary["meta_learner_brier"] = float(brier_score(post_test, test_labels))
    summary["meta_learner_logloss"] = float(_safe_log_loss(test_labels, post_test))

    # ── Persist report.json ───────────────────────────────────────────────
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    report_path = output_dir / "report.json"
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info(
        "rolling_retrain.done",
        report=str(report_path),
        meta_ece=summary["meta_learner_ece"],
        meta_brier=summary["meta_learner_brier"],
        chosen=cal.chosen_method,
    )
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Layer-7 rolling-window retraining (ADR-044).",
    )
    p.add_argument("--parquet", type=Path, required=True,
                   help="Path to training_<YYYYMMDD>.parquet")
    p.add_argument(
        "--as-of-date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        required=True,
        help="Exclusive right edge of the rolling window (YYYY-MM-DD).",
    )
    p.add_argument("--window-years", type=int, default=3,
                   help="Calendar-year window size (default 3 — see ADR-044).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write sub-model + meta artifacts.")
    p.add_argument(
        "--sub-models", nargs="+", default=list(ALL_SUB_MODELS),
        choices=list(ALL_SUB_MODELS),
        help=f"Which sub-models to (re)train. Default: {list(ALL_SUB_MODELS)}",
    )
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--calib-fraction", type=float, default=0.20)
    p.add_argument(
        "--skip-if-no-drift", action="store_true",
        help="Read the drift-state file and exit code 2 if triggered=false.",
    )
    p.add_argument(
        "--drift-state-path", type=Path,
        default=Path("models/baseline_full/drift_state.json"),
        help="Path to the JSON marker written by save_drift_state.",
    )
    return p.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.skip_if_no_drift:
        state = load_drift_state(args.drift_state_path)
        if not state.get("triggered", False):
            log.info(
                "rolling_retrain.skipped_no_drift",
                drift_state_path=str(args.drift_state_path),
                n_observations=state.get("n_observations"),
            )
            return EXIT_SKIPPED_NO_DRIFT
        log.info(
            "rolling_retrain.drift_triggered",
            drift_state_path=str(args.drift_state_path),
            alarmed_at=state.get("alarmed_at"),
            direction=state.get("direction"),
        )

    summary = run_rolling_retrain(
        parquet_path=args.parquet,
        as_of_date=args.as_of_date,
        window_years=args.window_years,
        output_dir=args.output_dir,
        sub_models=args.sub_models,
        test_fraction=args.test_fraction,
        calib_fraction=args.calib_fraction,
    )
    # Print a compact summary to stdout so cron logs surface the headline.
    headline = {
        k: summary[k] for k in (
            "as_of_date", "window_years",
            "n_train_rows", "n_calib_rows", "n_test_rows",
            "sub_models_trained",
            "meta_learner_ece", "meta_learner_brier", "meta_learner_logloss",
            "calibrator_chosen_method",
        )
    }
    print(json.dumps(headline, indent=2, default=str))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
