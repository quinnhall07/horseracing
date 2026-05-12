"""scripts/bootstrap_models.py

Train the Phase-3 baseline models from `data/exports/training_<date>.parquet`.

End-to-end flow:
    1. Load the most recent training parquet (or one passed via --input).
    2. Run `prepare_training_features` to produce a leakage-free feature matrix.
    3. Time-based train/val split (default last 10% by date).
    4. Fit the trainable sub-models:
         * SpeedFormModel  (LightGBM, primary baseline)
         * ConnectionsModel (empirical-Bayes shrinkage)
         * MarketModel     (isotonic odds calibration)
    5. Generate per-row predictions from each sub-model + the stub layers.
    6. Fit the MetaLearner on top of the stacked sub-model outputs.
    7. Save every artifact under `models/<run-timestamp>/` + a JSON summary.

Pace/Sequence layers stay as scaffolds (return a constant 0.5) — see
PROGRESS.md "Known Export Caveats" for the unblock criteria.

Usage:
    python scripts/bootstrap_models.py
    python scripts/bootstrap_models.py --input data/exports/training_20260512.parquet \
                                        --val-fraction 0.10 \
                                        --output-root models
    python scripts/bootstrap_models.py --sample-frac 0.05   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.core.logging import configure_logging, get_logger
from app.services.models.connections import ConnectionsModel
from app.services.models.market import MarketModel
from app.services.models.meta_learner import MetaLearner
from app.services.models.pace_scenario import PaceScenarioModel
from app.services.models.sequence import SequenceModel
from app.services.models.speed_form import SpeedFormModel
from app.services.models.training_data import (
    load_training_parquet,
    prepare_training_features,
    time_based_split,
)

log = get_logger(__name__)


def _latest_parquet() -> Path:
    candidates = sorted(Path("data/exports").glob("training_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "No training parquet found in data/exports/. "
            "Run scripts/db/export_training_data.py first."
        )
    return candidates[-1]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Phase 3 baseline models.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Training parquet (defaults to latest in data/exports/)")
    parser.add_argument("--output-root", type=Path, default=Path("models"),
                        help="Root dir for the run artifacts (default: ./models)")
    parser.add_argument("--val-fraction", type=float, default=0.10,
                        help="Last-X-percent-by-date held out for validation (default 0.10)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Optionally subsample the parquet for a smoke test")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the auto-generated run directory name")
    return parser.parse_args(argv)


def _stack_sub_predictions(
    df: pd.DataFrame,
    speed_form: SpeedFormModel,
    pace: PaceScenarioModel,
    sequence: SequenceModel,
    connections: ConnectionsModel,
    market: MarketModel,
) -> pd.DataFrame:
    """Append the per-row sub-model predictions to the frame."""
    stacked = df.copy()
    stacked["speed_form_proba"] = speed_form.predict_proba(df)
    stacked["pace_scenario_proba"] = pace.predict_proba(df)
    stacked["sequence_proba"] = sequence.predict_proba(df)
    stacked["connections_proba"] = connections.predict_proba(df)
    market_pred = market.predict_proba(df)
    # Replace NaN (rows with missing odds) with the global mean so the meta
    # learner has a complete feature column. Bookkeep the NaN mask as a
    # separate column the meta CAN look at if it wants to.
    nan_mask = np.isnan(market_pred)
    if nan_mask.any():
        mean = float(np.nanmean(market_pred))
        market_pred = np.where(nan_mask, mean, market_pred)
    stacked["market_proba"] = market_pred
    stacked["market_proba_was_missing"] = nan_mask.astype(int)
    return stacked


def bootstrap(
    input_path: Path,
    output_dir: Path,
    val_fraction: float = 0.10,
    sample_frac: float | None = None,
) -> dict:
    log.info("bootstrap.start", input=str(input_path), output=str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + prep ──────────────────────────────────────────────────────
    raw = load_training_parquet(input_path)
    if sample_frac is not None and 0 < sample_frac < 1:
        raw = raw.sample(frac=sample_frac, random_state=11).reset_index(drop=True)
        log.warning("bootstrap.subsampled", frac=sample_frac, rows=len(raw))

    features = prepare_training_features(raw)

    # ── 2. Time split ───────────────────────────────────────────────────────
    split = time_based_split(features, val_fraction=val_fraction)
    log.info(
        "bootstrap.split",
        train_rows=len(split.train),
        val_rows=len(split.val),
        split_date=str(split.split_date),
    )

    summary: dict = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "rows_total": int(len(features)),
        "rows_train": int(len(split.train)),
        "rows_val": int(len(split.val)),
        "split_date": str(split.split_date.date()) if hasattr(split.split_date, "date") else str(split.split_date),
        "models": {},
    }

    # ── 3. Train sub-models ─────────────────────────────────────────────────
    speed_form = SpeedFormModel().fit(split.train, val_df=split.val)
    speed_form_dir = output_dir / "speed_form"
    speed_form.save(speed_form_dir)
    summary["models"]["speed_form"] = {
        "path": str(speed_form_dir),
        "metrics": speed_form.metrics.__dict__ if speed_form.metrics else None,
    }

    pace = PaceScenarioModel()
    pace_trainable = PaceScenarioModel.is_trainable_with(split.train)
    summary["models"]["pace_scenario"] = {"trainable": pace_trainable, "stub": True}
    if not pace_trainable:
        log.warning("bootstrap.pace_skipped", reason="fractional data missing in parquet")

    sequence = SequenceModel()
    summary["models"]["sequence"] = {"trainable": False, "stub": True}
    log.warning("bootstrap.sequence_skipped", reason="pytorch dep + unique horse_id required")

    connections = ConnectionsModel().fit(split.train, val_df=split.val)
    connections_dir = output_dir / "connections"
    connections.save(connections_dir)
    summary["models"]["connections"] = {
        "path": str(connections_dir),
        "n_jockeys": len(connections.jockey_rate),
        "n_trainers": len(connections.trainer_rate),
        "n_pairs": len(connections.pair_rate),
    }

    market = MarketModel().fit(split.train, val_df=split.val)
    market_dir = output_dir / "market"
    market.save(market_dir)
    summary["models"]["market"] = {"path": str(market_dir)}

    # ── 4. Stack predictions for the meta learner ───────────────────────────
    log.info("bootstrap.stacking_predictions")
    train_stacked = _stack_sub_predictions(
        split.train, speed_form, pace, sequence, connections, market
    )
    val_stacked = _stack_sub_predictions(
        split.val, speed_form, pace, sequence, connections, market
    )

    # ── 5. Train meta-learner ───────────────────────────────────────────────
    meta = MetaLearner().fit(train_stacked, val_df=val_stacked)
    meta_dir = output_dir / "meta_learner"
    meta.save(meta_dir)
    summary["models"]["meta_learner"] = {"path": str(meta_dir)}

    # ── 6. Meta-learner evaluation on the validation set ────────────────────
    val_pred = meta.predict_proba(val_stacked)
    val_labels = val_stacked["win"].to_numpy()
    summary["meta_val_log_loss"] = float(_safe_log_loss(val_labels, val_pred))
    summary["meta_val_top1_acc"] = float(
        _race_top1_accuracy(val_pred, val_labels, val_stacked["race_id"].to_numpy())
    )

    # ── 7. Persist summary JSON ─────────────────────────────────────────────
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("bootstrap.complete", summary_path=str(summary_path))
    return summary


def _safe_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _race_top1_accuracy(
    proba: np.ndarray, labels: np.ndarray, race_ids: np.ndarray
) -> float:
    correct = 0
    total = 0
    df = pd.DataFrame({"p": proba, "y": labels, "r": race_ids})
    for _, sub in df.groupby("r", sort=False):
        if sub["y"].sum() == 0:
            continue
        total += 1
        if sub.loc[sub["p"].idxmax(), "y"] == 1:
            correct += 1
    return correct / total if total else 0.0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv or sys.argv[1:])
    input_path = args.input or _latest_parquet()

    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    output_dir = args.output_root / run_name

    summary = bootstrap(
        input_path=input_path,
        output_dir=output_dir,
        val_fraction=args.val_fraction,
        sample_frac=args.sample_frac,
    )
    # Pretty-print the top-level summary to stdout.
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
