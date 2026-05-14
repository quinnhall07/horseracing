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
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.core.logging import configure_logging, get_logger
from app.services.calibration.calibrator import Calibrator, CalibratorConfig
from app.services.models.connections import ConnectionsModel
from app.services.models.market import MarketModel
from app.services.models.meta_learner import MetaLearner
from app.services.models.pace_scenario import PaceScenarioModel
from app.services.models.sequence import SequenceModel
from app.services.models.speed_form import SpeedFormModel
from app.services.models.training_data import (
    load_training_parquet,
    prepare_training_features,
    three_way_time_split,
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
    parser.add_argument("--train-frac", type=float, default=0.60,
                        help="Fraction of rows (by date) used for training (default 0.60).")
    parser.add_argument("--calib-frac", type=float, default=0.20,
                        help="Fraction of rows for calibration slice (default 0.20). "
                             "Test slice is 1 - train_frac - calib_frac.")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Optionally subsample the parquet for a smoke test")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the auto-generated run directory name")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for any non-deterministic step (default 42).")
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
    train_frac: float = 0.60,
    calib_frac: float = 0.20,
    sample_frac: float | None = None,
    seed: int = 42,
) -> dict:
    log.info("bootstrap.start", input=str(input_path), output=str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    # ── 1. Load + prep ──────────────────────────────────────────────────────
    raw = load_training_parquet(input_path)
    if sample_frac is not None and 0 < sample_frac < 1:
        raw = raw.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        log.warning("bootstrap.subsampled", frac=sample_frac, rows=len(raw))

    features = prepare_training_features(raw)

    # ── 2. Time-based 3-way split (train / calib / test). ADR-003. ─────────
    train, calib, test = three_way_time_split(
        features, train_frac=train_frac, cal_frac=calib_frac
    )
    log.info(
        "bootstrap.split",
        train_rows=len(train),
        calib_rows=len(calib),
        test_rows=len(test),
    )

    summary: dict = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "rows_total": int(len(features)),
        "rows_train": int(len(train)),
        "rows_calib": int(len(calib)),
        "rows_test": int(len(test)),
        "train_frac": float(train_frac),
        "calib_frac": float(calib_frac),
        "seed": int(seed),
        "models": {},
    }

    # ── 3. Train sub-models on TRAIN slice (calib used as val_df for early stop) ─
    speed_form = SpeedFormModel().fit(train, val_df=calib)
    speed_form_dir = output_dir / "speed_form"
    speed_form.save(speed_form_dir)
    summary["models"]["speed_form"] = {
        "path": str(speed_form_dir),
        "metrics": speed_form.metrics.__dict__ if speed_form.metrics else None,
    }

    pace_trainable = PaceScenarioModel.is_trainable_with(train)
    if pace_trainable:
        # ADR-047: LightGBM fit on rows whose pace columns are populated.
        pace = PaceScenarioModel().fit(train, val_df=calib)
        pace_dir = output_dir / "pace_scenario"
        pace.save(pace_dir)
        summary["models"]["pace_scenario"] = {
            "path": str(pace_dir),
            "trainable": True,
            "stub": False,
            "metrics": asdict(pace.metrics) if pace.metrics else None,
        }
    else:
        pace = PaceScenarioModel()
        summary["models"]["pace_scenario"] = {"trainable": False, "stub": True}
        log.warning("bootstrap.pace_skipped", reason="fractional data missing in parquet")

    sequence_trainable = getattr(SequenceModel, "is_trainable_with", lambda _df: False)(train)
    if sequence_trainable:
        # ADR-046: real Transformer fit when torch + horse_dedup_key are present.
        sequence = SequenceModel().fit(train, val_df=calib)
        sequence_dir = output_dir / "sequence"
        sequence.save(sequence_dir)
        summary["models"]["sequence"] = {
            "path": str(sequence_dir),
            "trainable": True,
            "stub": False,
        }
    else:
        sequence = SequenceModel()
        summary["models"]["sequence"] = {"trainable": False, "stub": True}
        log.warning(
            "bootstrap.sequence_skipped",
            reason="torch missing or horse_dedup_key column not in parquet",
        )

    connections = ConnectionsModel().fit(train, val_df=calib)
    connections_dir = output_dir / "connections"
    connections.save(connections_dir)
    summary["models"]["connections"] = {
        "path": str(connections_dir),
        "n_jockeys": len(connections.jockey_rate),
        "n_trainers": len(connections.trainer_rate),
        "n_pairs": len(connections.pair_rate),
    }

    market = MarketModel().fit(train, val_df=calib)
    market_dir = output_dir / "market"
    market.save(market_dir)
    summary["models"]["market"] = {"path": str(market_dir)}

    # ── 4. Stack predictions on every slice ─────────────────────────────────
    log.info("bootstrap.stacking_predictions")
    train_stacked = _stack_sub_predictions(
        train, speed_form, pace, sequence, connections, market
    )
    calib_stacked = _stack_sub_predictions(
        calib, speed_form, pace, sequence, connections, market
    )
    test_stacked = _stack_sub_predictions(
        test, speed_form, pace, sequence, connections, market
    )

    # ── 5. Train meta-learner on TRAIN, early-stop on CALIB. ───────────────
    meta = MetaLearner().fit(train_stacked, val_df=calib_stacked)
    meta_dir = output_dir / "meta_learner"
    meta.save(meta_dir)
    summary["models"]["meta_learner"] = {"path": str(meta_dir)}

    # ── 6. Calibrator on CALIB output (auto-select Platt/iso per ADR-037/038)
    log.info("bootstrap.calibrator_fit")
    calib_pred = meta.predict_proba(calib_stacked)
    calib_labels = calib_stacked["win"].to_numpy()
    calibrator = Calibrator(CalibratorConfig(method="auto")).fit(calib_pred, calib_labels)
    cal_path = output_dir / "calibration_adr038_brier" / "meta_learner"
    calibrator.save(cal_path)
    summary["models"]["calibrator"] = {
        "path": str(cal_path),
        "chosen_method": calibrator.chosen_method,
    }

    # ── 7. Evaluate on TEST slice (post-calibration metrics are what matters)
    test_pred_raw = meta.predict_proba(test_stacked)
    test_pred_cal = calibrator.predict_proba(test_pred_raw)
    test_labels = test_stacked["win"].to_numpy()

    summary["meta_test_log_loss_pre"] = float(_safe_log_loss(test_labels, test_pred_raw))
    summary["meta_test_log_loss"] = float(_safe_log_loss(test_labels, test_pred_cal))
    summary["meta_test_brier_pre"] = float(_brier_score(test_labels, test_pred_raw))
    summary["meta_test_brier"] = float(_brier_score(test_labels, test_pred_cal))
    summary["meta_test_ece_pre"] = float(_expected_calibration_error(test_labels, test_pred_raw))
    summary["meta_test_ece"] = float(_expected_calibration_error(test_labels, test_pred_cal))
    summary["meta_test_top1_acc"] = float(
        _race_top1_accuracy(test_pred_cal, test_labels, test_stacked["race_id"].to_numpy())
    )

    # ── 8. Persist summary JSON ─────────────────────────────────────────────
    summary["elapsed_seconds"] = round(time.monotonic() - t_start, 2)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # ── 9. Provenance file (consumed by InferenceArtifacts.load) ────────────
    sub_models = ["speed_form", "connections", "market"]
    stub_sub_models = []
    if pace_trainable:
        sub_models.append("pace_scenario")
    else:
        stub_sub_models.append("pace_scenario")
    if sequence_trainable:
        sub_models.append("sequence")
    else:
        stub_sub_models.append("sequence")
    provenance = {
        "is_synthetic": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(train)),
        "n_calib_rows": int(len(calib)),
        "n_test_rows": int(len(test)),
        "sub_models": sub_models,
        "stub_sub_models": stub_sub_models,
        "meta_learner_test_ece": summary["meta_test_ece"],
        "meta_learner_test_brier": summary["meta_test_brier"],
        "bootstrap_script": "scripts/bootstrap_models.py",
        "bootstrap_seed": int(seed),
        "parquet_path": str(input_path),
        "warning": None,
    }
    with open(output_dir / "BOOTSTRAP_PROVENANCE.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    log.info(
        "bootstrap.complete",
        summary_path=str(summary_path),
        test_ece=summary["meta_test_ece"],
        test_brier=summary["meta_test_brier"],
    )
    return summary


def _brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def _expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width-bin ECE on [0, 1]. Same convention as the Calibrator's
    auto-selector criterion (ADR-037). Returns 0.0 on empty input."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, bins, right=False) - 1, 0, n_bins - 1)
    ece = 0.0
    n = float(y_pred.size)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_pred[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


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
        train_frac=args.train_frac,
        calib_frac=args.calib_frac,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )
    # Pretty-print the top-level summary to stdout.
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
