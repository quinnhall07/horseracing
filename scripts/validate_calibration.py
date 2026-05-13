"""scripts/validate_calibration.py

Fit calibrators on a held-out slice of the training parquet, evaluate ECE
and Brier on a downstream test slice, and emit a reliability diagram +
JSON report. This is the Phase-4 validation artifact called for by
CLAUDE.md §6 and ADR-008.

Pipeline:
    1. Load the most recent training parquet (or --input).
    2. prepare_training_features.
    3. 3-way time-based split: train (80%) / calib (10%) / test (10%).
    4. Load the trained Speed/Form + Meta models from --models-dir.
    5. Score the calib and test slices.
    6. Fit a Calibrator on calib for each model (Platt + isotonic auto-select).
    7. Compute pre/post-calibration metrics on the test slice.
    8. Save report (PNG + JSON) under --output-dir.

The actual calibration evaluation logic is exposed as `evaluate_calibration`
so the smoke test can exercise it on a tiny synthetic without touching the
real 2.3M-row parquet.

Usage:
    python scripts/validate_calibration.py
    python scripts/validate_calibration.py --input data/exports/training_20260512.parquet \
                                            --models-dir models/baseline_full \
                                            --output-dir models/baseline_full/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # headless PNG output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.core.logging import configure_logging, get_logger
from app.services.calibration.calibrator import (
    Calibrator,
    CalibratorConfig,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)

log = get_logger(__name__)


# ── Core evaluation routine (importable for tests) ────────────────────────


@dataclass
class CalibrationReport:
    """Result of `evaluate_calibration` for one model."""
    label: str
    chosen_method: str
    pre_ece: float
    pre_brier: float
    pre_log_loss: float
    post_ece: float
    post_brier: float
    post_log_loss: float
    n_calib: int
    n_test: int
    pre_bins: list[dict]
    post_bins: list[dict]
    # New in ADR-037: how `chosen_method` was decided.
    auto_selection_mode: Optional[str] = None
    inner_val_metrics: Optional[dict] = None

    def asdict(self) -> dict:
        return {
            **{k: v for k, v in self.__dict__.items() if k not in ("pre_bins", "post_bins")},
            "pre_bins": self.pre_bins,
            "post_bins": self.post_bins,
        }


def _safe_log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def evaluate_calibration(
    calib_scores: np.ndarray,
    calib_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    label: str,
    n_bins: int = 15,
    inner_val_indices: Optional[np.ndarray] = None,
) -> tuple[Calibrator, CalibrationReport]:
    """Fit a Calibrator on (calib_scores, calib_labels), evaluate on test.

    `inner_val_indices` (optional): time-ordered tail of the calib slice
    to use as the inner-val set for auto-selection. Recommended in the
    live pipeline since the test slice immediately follows the calib
    slice in time and the model improves over time (older calib rows
    are not exchangeable with newer test rows). See ADR-038.
    """
    cal = Calibrator(CalibratorConfig(method="auto", n_bins_for_selection=n_bins))
    cal.fit(calib_scores, calib_labels, inner_val_indices=inner_val_indices)

    pre_test = test_scores
    post_test = cal.predict_proba(test_scores)

    report = CalibrationReport(
        label=label,
        chosen_method=cal.chosen_method or "",
        pre_ece=expected_calibration_error(pre_test, test_labels, n_bins=n_bins),
        pre_brier=brier_score(pre_test, test_labels),
        pre_log_loss=_safe_log_loss(pre_test, test_labels),
        post_ece=expected_calibration_error(post_test, test_labels, n_bins=n_bins),
        post_brier=brier_score(post_test, test_labels),
        post_log_loss=_safe_log_loss(post_test, test_labels),
        n_calib=len(calib_scores),
        n_test=len(test_scores),
        pre_bins=[
            {
                "bin_lower": b.bin_lower, "bin_upper": b.bin_upper,
                "mean_predicted": b.mean_predicted, "observed_rate": b.observed_rate,
                "count": b.count,
            }
            for b in reliability_bins(pre_test, test_labels, n_bins=n_bins)
        ],
        post_bins=[
            {
                "bin_lower": b.bin_lower, "bin_upper": b.bin_upper,
                "mean_predicted": b.mean_predicted, "observed_rate": b.observed_rate,
                "count": b.count,
            }
            for b in reliability_bins(post_test, test_labels, n_bins=n_bins)
        ],
        auto_selection_mode=cal.auto_selection_mode,
        inner_val_metrics=cal.inner_val_metrics,
    )
    log.info(
        "calibration.evaluated",
        label=label,
        chosen=cal.chosen_method,
        selection_mode=cal.auto_selection_mode,
        inner_val=cal.inner_val_metrics,
        pre_ece=report.pre_ece, post_ece=report.post_ece,
        pre_brier=report.pre_brier, post_brier=report.post_brier,
    )
    return cal, report


def render_reliability_diagram(report: CalibrationReport, png_path: Path) -> None:
    """Save a 2-panel reliability diagram (pre vs post calibration) to disk."""
    fig, ax = plt.subplots(figsize=(7, 6))
    # Diagonal reference.
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="perfect")

    for tag, bins, marker in (("pre", report.pre_bins, "o"), ("post", report.post_bins, "s")):
        xs = [b["mean_predicted"] for b in bins if b["count"] > 0]
        ys = [b["observed_rate"] for b in bins if b["count"] > 0]
        ax.plot(xs, ys, marker=marker, linewidth=1.5, label=tag)

    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed win rate (bin)")
    ax.set_title(
        f"{report.label} reliability "
        f"(pre ECE={report.pre_ece:.4f} → post ECE={report.post_ece:.4f}, "
        f"method={report.chosen_method})"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


# ── Live-pipeline runner ──────────────────────────────────────────────────


def _three_way_split(
    df: pd.DataFrame,
    test_fraction: float = 0.10,
    calib_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Date-ordered 3-way split: train | calib | test."""
    if not 0 < test_fraction < 1 or not 0 < calib_fraction < 1:
        raise ValueError("fractions must lie in (0, 1)")
    if test_fraction + calib_fraction >= 1:
        raise ValueError("test_fraction + calib_fraction must be < 1")
    test_cutoff = df["race_date"].quantile(1.0 - test_fraction)
    calib_cutoff = df["race_date"].quantile(1.0 - test_fraction - calib_fraction)
    train = df[df["race_date"] <= calib_cutoff].reset_index(drop=True)
    calib = df[
        (df["race_date"] > calib_cutoff) & (df["race_date"] <= test_cutoff)
    ].reset_index(drop=True)
    test = df[df["race_date"] > test_cutoff].reset_index(drop=True)
    return train, calib, test, calib_cutoff, test_cutoff


def _stack_for_meta(
    df: pd.DataFrame,
    speed_form,
    pace,
    sequence,
    connections,
    market,
) -> pd.DataFrame:
    stacked = df.copy()
    stacked["speed_form_proba"] = speed_form.predict_proba(df)
    stacked["pace_scenario_proba"] = pace.predict_proba(df)
    stacked["sequence_proba"] = sequence.predict_proba(df)
    stacked["connections_proba"] = connections.predict_proba(df)
    m = market.predict_proba(df)
    nan_mask = np.isnan(m)
    if nan_mask.any():
        m = np.where(nan_mask, float(np.nanmean(m)), m)
    stacked["market_proba"] = m
    stacked["market_proba_was_missing"] = nan_mask.astype(int)
    return stacked


def _latest_parquet() -> Path:
    candidates = sorted(Path("data/exports").glob("training_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "No training parquet found in data/exports/. Run "
            "scripts/db/export_training_data.py first."
        )
    return candidates[-1]


def run_live(
    input_path: Path,
    models_dir: Path,
    output_dir: Path,
    test_fraction: float = 0.10,
    calib_fraction: float = 0.10,
    sample_frac: float | None = None,
) -> dict:
    """Load real trained models + parquet, run end-to-end validation."""
    from app.services.models.connections import ConnectionsModel
    from app.services.models.market import MarketModel
    from app.services.models.meta_learner import MetaLearner
    from app.services.models.pace_scenario import PaceScenarioModel
    from app.services.models.sequence import SequenceModel
    from app.services.models.speed_form import SpeedFormModel
    from app.services.models.training_data import (
        load_training_parquet, prepare_training_features,
    )

    log.info("validate_calibration.start", input=str(input_path), models=str(models_dir))

    df = load_training_parquet(input_path)
    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=11).reset_index(drop=True)
        log.warning("validate.subsampled", frac=sample_frac, rows=len(df))
    features = prepare_training_features(df)

    _, calib, test, calib_cutoff, test_cutoff = _three_way_split(
        features, test_fraction=test_fraction, calib_fraction=calib_fraction
    )
    log.info(
        "validate.split",
        n_calib=len(calib), n_test=len(test),
        calib_cutoff=str(calib_cutoff), test_cutoff=str(test_cutoff),
    )

    speed_form = SpeedFormModel.load(models_dir / "speed_form")
    pace = PaceScenarioModel()
    sequence = SequenceModel()
    connections = ConnectionsModel.load(models_dir / "connections")
    market = MarketModel.load(models_dir / "market")
    meta = MetaLearner.load(models_dir / "meta_learner")

    sf_calib_p = speed_form.predict_proba(calib)
    sf_test_p = speed_form.predict_proba(test)

    calib_stacked = _stack_for_meta(calib, speed_form, pace, sequence, connections, market)
    test_stacked = _stack_for_meta(test, speed_form, pace, sequence, connections, market)
    meta_calib_p = meta.predict_proba(calib_stacked)
    meta_test_p = meta.predict_proba(test_stacked)

    # Time-ordered inner-val indices (last 20% of the calib slice by date).
    # The model improves over time, so the calib slice and test slice are
    # not exchangeable — the tail of the calib slice is the closest
    # in-distribution proxy for the test slice. See ADR-038.
    sorted_calib_idx = calib["race_date"].argsort().to_numpy()
    inner_val_size = max(1, int(0.2 * len(sorted_calib_idx)))
    inner_val_idx = sorted_calib_idx[-inner_val_size:]
    log.info(
        "validate.inner_val",
        n_inner_val=int(len(inner_val_idx)),
        inner_val_start=str(calib["race_date"].iloc[inner_val_idx].min()),
        inner_val_end=str(calib["race_date"].iloc[inner_val_idx].max()),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "input": str(input_path),
        "models_dir": str(models_dir),
        "n_calib": int(len(calib)),
        "n_test": int(len(test)),
        "n_inner_val": int(len(inner_val_idx)),
        "calib_cutoff": str(calib_cutoff),
        "test_cutoff": str(test_cutoff),
        "models": {},
    }

    for label, calib_p, test_p in (
        ("speed_form", sf_calib_p, sf_test_p),
        ("meta_learner", meta_calib_p, meta_test_p),
    ):
        cal, report = evaluate_calibration(
            calib_scores=np.asarray(calib_p),
            calib_labels=calib["win"].to_numpy(),
            test_scores=np.asarray(test_p),
            test_labels=test["win"].to_numpy(),
            label=label,
            inner_val_indices=inner_val_idx,
        )
        cal_dir = output_dir / label
        cal.save(cal_dir)
        render_reliability_diagram(report, cal_dir / "reliability.png")
        with open(cal_dir / "report.json", "w") as fh:
            json.dump(report.asdict(), fh, indent=2, default=str)
        summary["models"][label] = report.asdict()

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("validate_calibration.done", summary_path=str(summary_path))
    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate calibration on Phase-3 baseline.")
    p.add_argument("--input", type=Path, default=None,
                   help="Training parquet (default: latest in data/exports/)")
    p.add_argument("--models-dir", type=Path, default=Path("models/baseline_full"),
                   help="Directory containing speed_form/, meta_learner/, …")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write calibration artifacts (default: models-dir/calibration)")
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--calib-fraction", type=float, default=0.10)
    p.add_argument("--sample-frac", type=float, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv or sys.argv[1:])
    input_path = args.input or _latest_parquet()
    output_dir = args.output_dir or (args.models_dir / "calibration")
    summary = run_live(
        input_path=input_path,
        models_dir=args.models_dir,
        output_dir=output_dir,
        test_fraction=args.test_fraction,
        calib_fraction=args.calib_fraction,
        sample_frac=args.sample_frac,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
