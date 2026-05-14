"""scripts/quick_bootstrap.py — DEMO-ONLY synthetic-data bootstrap.

==============================================================================
WARNING — THIS SCRIPT TRAINS MODELS ON RANDOMLY-GENERATED DATA.
==============================================================================
The artifacts produced by `quick_bootstrap.py` exist for ONE reason: to make
a fresh clone of this repository serve `/api/v1/cards/{id}` and
`/api/v1/portfolio/{id}` with 200 (not 503) before the operator has stood
up the Phase 0 master DB and run `bootstrap_models.py` on a real parquet.

The predictions returned by these models are NOT MEANINGFUL. They reflect
random noise + a small synthetic correlation injected so LightGBM has
something to learn (otherwise the booster degenerates to a single leaf
and `save()` fails).

For production-quality models, follow `DATA_PIPELINE.md` to populate
`data/exports/training_<date>.parquet`, then run:

    python scripts/bootstrap_models.py

That replaces every artifact this script writes with one trained on real
historical race results.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    # Default: ~500 races × ~8 horses (≈4,000 rows), seed=42, output
    # models/baseline_full/.
    python scripts/quick_bootstrap.py

    # Custom output location and size.
    python scripts/quick_bootstrap.py \\
        --output-dir models/demo \\
        --n-synthetic-races 500 \\
        --seed 42

Per CLAUDE.md §11 this script is STANDALONE: no FastAPI imports, no async
loops. It uses only the modules under `app/services/models/`,
`app/services/calibration/`, and the project root on `sys.path`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make `app` importable regardless of cwd. Mirrors bootstrap_models.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.services.calibration.calibrator import Calibrator, CalibratorConfig
from app.services.models.connections import ConnectionsModel
from app.services.models.market import MarketModel
from app.services.models.meta_learner import MetaLearner
from app.services.models.pace_scenario import PaceScenarioModel
from app.services.models.sequence import SequenceModel
from app.services.models.speed_form import SpeedFormModel
from app.services.models.training_data import (
    SPEED_FORM_FEATURE_COLUMNS,
    prepare_training_features,
)

DEFAULT_OUTPUT = Path("models/baseline_full")
DEFAULT_N_RACES = 500
DEFAULT_HORSES_PER_RACE = 8
DEFAULT_SEED = 42

# ── Synthetic-data generator ────────────────────────────────────────────────


def generate_synthetic_results(
    n_races: int,
    horses_per_race: int,
    seed: int,
    today: datetime | None = None,
) -> pd.DataFrame:
    """Fabricate a long-form DataFrame that mirrors the training-parquet schema.

    Columns are chosen to match what `prepare_training_features` reads. The
    `won` outcome is correlated with `speed_figure` (about 0.15 raw
    correlation) and with the jockey×trainer pair identity (some pairs are
    persistently above-average), so the trained models pick up non-trivial
    weights instead of fitting random noise to a single split.
    """
    rng = np.random.default_rng(seed)
    today = (today or datetime.now(timezone.utc)).replace(tzinfo=None)
    earliest = today - timedelta(days=730)  # 2-year window
    window_days = (today - earliest).days

    # Distinct horse / jockey / trainer pools so `prepare_training_features`
    # has multiple priors per horse_key.
    n_horses = max(60, n_races * horses_per_race // 6)
    horse_pool = [f"horse_{i:04d}" for i in range(n_horses)]
    jockey_pool = [f"jockey_{i:02d}" for i in range(24)]
    trainer_pool = [f"trainer_{i:02d}" for i in range(18)]

    # Latent per-horse ability — used to inject the signal LightGBM trains on.
    abilities = {h: rng.normal(80.0, 12.0) for h in horse_pool}
    # Some jockey/trainer pairs have a small positive lift.
    pair_lift_pool = {
        (j, t): rng.normal(0.0, 0.04) for j in jockey_pool for t in trainer_pool
    }

    surfaces = ["dirt", "turf"]
    conditions = ["fast", "good", "soft", "yielding"]
    track_codes = ["CD", "SAR", "AQU", "BEL", "DMR", "KEE", "GP"]
    race_types = ["claiming", "allowance", "stakes", "maiden"]
    jurisdictions = ["US", "UK", "HK", "JP"]

    rows: list[dict] = []
    for r in range(n_races):
        # Pick race-scoped state.
        race_day = earliest + timedelta(days=int(rng.integers(0, window_days + 1)))
        track = rng.choice(track_codes)
        race_num = int(rng.integers(1, 11))
        distance = float(rng.choice([5.0, 5.5, 6.0, 6.5, 8.0, 8.5, 9.0]))
        surface = rng.choice(surfaces)
        condition = rng.choice(conditions)
        race_type = rng.choice(race_types)
        purse = float(rng.choice([20_000, 40_000, 75_000, 150_000]))
        claiming_price = (
            float(rng.choice([10_000, 25_000, 50_000])) if race_type == "claiming" else None
        )
        jurisdiction = rng.choice(jurisdictions, p=[0.55, 0.20, 0.10, 0.15])

        # Sample horses without replacement for this race.
        starters = rng.choice(horse_pool, size=horses_per_race, replace=False)
        jockeys = rng.choice(jockey_pool, size=horses_per_race, replace=True)
        trainers = rng.choice(trainer_pool, size=horses_per_race, replace=True)

        # Latent score = ability + jockey×trainer lift + race noise.
        latents = np.empty(horses_per_race, dtype=float)
        speed_figures = np.empty(horses_per_race, dtype=float)
        for i, (h, j, t) in enumerate(zip(starters, jockeys, trainers)):
            sig = (
                abilities[h]
                + 50.0 * pair_lift_pool[(j, t)]
                + rng.normal(0.0, 6.0)
            )
            speed_figures[i] = float(np.clip(sig, 0.0, 130.0))
            latents[i] = sig + rng.normal(0.0, 2.0)

        # Winner is argmax(latent); finish positions follow the latent order.
        order = np.argsort(-latents)
        finish_positions = np.empty(horses_per_race, dtype=int)
        for rank, idx in enumerate(order, start=1):
            finish_positions[idx] = rank

        # Plausible odds: shorter on stronger latents. The shape is exponential
        # so the market reflects the same signal the model would; the rank
        # correlation with the win outcome is positive but noisy.
        odds_final = np.clip(
            2.0 + np.exp((np.max(latents) - latents) / 8.0) + rng.normal(0.0, 1.5, size=horses_per_race),
            1.2,
            99.0,
        )
        morning_line = odds_final + rng.normal(0.0, 1.0, size=horses_per_race)
        morning_line = np.clip(morning_line, 1.2, 99.0)

        for i in range(horses_per_race):
            rows.append({
                "horse_name": starters[i],
                "jurisdiction": jurisdiction,
                "jockey_name": jockeys[i],
                "trainer_name": trainers[i],
                "race_date": race_day,
                "track_code": track,
                "race_number": race_num,
                "distance_furlongs": distance,
                "surface": surface,
                "condition": condition,
                "race_type": race_type,
                "claiming_price": claiming_price,
                "purse_usd": purse,
                "field_size": horses_per_race,
                "post_position": i + 1,
                "finish_position": int(finish_positions[i]),
                "weight_lbs": float(rng.normal(120.0, 5.0)),
                "odds_final": float(odds_final[i]),
                "morning_line_odds": float(morning_line[i]),
                "speed_figure": float(speed_figures[i]),
                "speed_figure_source": "synthetic",
                "fraction_q1_sec": None,
                "fraction_q2_sec": None,
                "fraction_finish_sec": float(rng.normal(70.0, 2.0)),
                "beaten_lengths_q1": None,
                "beaten_lengths_q2": None,
                "data_quality_score": 1.0,
                "foaling_year": None,
                "sire": None,
                "dam_sire": None,
            })

    df = pd.DataFrame(rows)
    return df


# ── Time-based split (60/20/20) — hoisted to training_data module ──────────

from app.services.models.training_data import three_way_time_split  # noqa: E402,F401


# ── Sub-model stacking (mirrors bootstrap_models._stack_sub_predictions) ────


def _stack_predictions(
    df: pd.DataFrame,
    speed_form: SpeedFormModel,
    pace: PaceScenarioModel,
    sequence: SequenceModel,
    connections: ConnectionsModel,
    market: MarketModel,
) -> pd.DataFrame:
    out = df.copy()
    out["speed_form_proba"] = speed_form.predict_proba(df)
    out["pace_scenario_proba"] = pace.predict_proba(df)
    out["sequence_proba"] = sequence.predict_proba(df)
    out["connections_proba"] = connections.predict_proba(df)
    market_pred = market.predict_proba(df)
    nan_mask = np.isnan(market_pred)
    if nan_mask.any():
        mean = float(np.nanmean(market_pred)) if not np.all(nan_mask) else 0.1
        market_pred = np.where(nan_mask, mean, market_pred)
    out["market_proba"] = market_pred
    out["market_proba_was_missing"] = nan_mask.astype(int)
    return out


# ── Verification: try to use InferenceArtifacts.load if present ─────────────


def verify_artifacts_load(output_dir: Path) -> dict:
    """Best-effort verification.

    Stream X (Pareto endpoint + LP swap) is delivering
    `app/services/inference/pipeline.py::InferenceArtifacts`. If the loader
    exists by the time `quick_bootstrap.py` is run, this calls it; if not,
    it falls back to loading each component individually so we at least
    confirm every artifact round-trips.
    """
    result: dict = {"inference_artifacts_loader": False, "components": {}}
    try:
        from app.services.inference.pipeline import InferenceArtifacts  # type: ignore
    except Exception as exc:  # noqa: BLE001 — fallback path
        result["inference_artifacts_loader"] = False
        result["inference_artifacts_loader_error"] = str(exc)
    else:
        result["inference_artifacts_loader"] = True
        artifacts = InferenceArtifacts.load(output_dir)  # noqa: F841 (smoke)
        result["inference_artifacts_loaded"] = True
        return result

    # Fallback: load each component on its own. If any fails the script
    # exits non-zero so the user sees the failure immediately.
    result["components"]["speed_form"] = SpeedFormModel.load(output_dir / "speed_form") is not None
    result["components"]["connections"] = (
        ConnectionsModel.load(output_dir / "connections") is not None
    )
    result["components"]["market"] = MarketModel.load(output_dir / "market") is not None
    result["components"]["meta_learner"] = (
        MetaLearner.load(output_dir / "meta_learner") is not None
    )
    cal_dir = output_dir / "calibration_adr038_brier" / "meta_learner"
    result["components"]["calibrator"] = Calibrator.load(cal_dir) is not None
    return result


# ── Orchestrator ────────────────────────────────────────────────────────────


def quick_bootstrap(
    output_dir: Path,
    n_synthetic_races: int = DEFAULT_N_RACES,
    horses_per_race: int = DEFAULT_HORSES_PER_RACE,
    seed: int = DEFAULT_SEED,
) -> dict:
    t_start = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic data.
    print(
        f"[1/6] Generating synthetic data — n_races={n_synthetic_races}, "
        f"horses_per_race={horses_per_race}, seed={seed}"
    )
    raw = generate_synthetic_results(
        n_races=n_synthetic_races,
        horses_per_race=horses_per_race,
        seed=seed,
    )
    print(f"      → {len(raw):,} rows, {raw['horse_name'].nunique():,} unique horses,"
          f" date range {raw['race_date'].min().date()} → {raw['race_date'].max().date()}")

    # 2. Feature engineering.
    print("[2/6] Preparing training features (field-relative, leak-free)…")
    features = prepare_training_features(raw)
    # Drop rows whose features depend entirely on priors that don't exist
    # (first-time-starters keep NaN for ewm_speed_prior). Don't drop them —
    # speed_form handles NaN via LightGBM; keep all rows so the meta-learner
    # has a complete frame.
    print(f"      → {len(features):,} feature rows, {features['race_id'].nunique():,} races")

    # 3. Time-based 60/20/20 split.
    print("[3/6] Time-based 60/20/20 split (ADR-003)…")
    train, calib, test = three_way_time_split(features)
    print(f"      → train={len(train):,}  calib={len(calib):,}  test={len(test):,}")
    summary: dict = {
        "is_synthetic": True,
        "seed": seed,
        "n_synthetic_races": n_synthetic_races,
        "horses_per_race": horses_per_race,
        "rows_total": int(len(features)),
        "rows_train": int(len(train)),
        "rows_calib": int(len(calib)),
        "rows_test": int(len(test)),
        "models": {},
    }

    # 4. Train trainable sub-models. Pace + Sequence stay as stubs per
    #    ADR-026 and `InferenceArtifacts.load` is tolerant of their absence.
    print("[4/6] Training Layer-1 sub-models (SpeedForm, Connections, Market)…")
    speed_form = SpeedFormModel().fit(train, val_df=calib)
    speed_form_dir = output_dir / "speed_form"
    speed_form.save(speed_form_dir)
    summary["models"]["speed_form"] = {
        "path": str(speed_form_dir),
        "val_log_loss": float(speed_form.metrics.val_log_loss) if speed_form.metrics else None,
        "val_race_top1_accuracy": (
            float(speed_form.metrics.val_race_top1_accuracy) if speed_form.metrics else None
        ),
    }

    pace = PaceScenarioModel()
    sequence = SequenceModel()
    summary["models"]["pace_scenario"] = {"stub": True}
    summary["models"]["sequence"] = {"stub": True}

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

    # 5. Stack predictions on calib slice → train meta-learner.
    print("[5/6] Stacking sub-model outputs and training meta-learner…")
    train_stacked = _stack_predictions(train, speed_form, pace, sequence, connections, market)
    calib_stacked = _stack_predictions(calib, speed_form, pace, sequence, connections, market)
    meta = MetaLearner().fit(train_stacked, val_df=calib_stacked)
    meta_dir = output_dir / "meta_learner"
    meta.save(meta_dir)
    summary["models"]["meta_learner"] = {"path": str(meta_dir)}

    # Calibrator on meta-learner output (auto-select per ADR-037/038).
    print("       Fitting Calibrator on meta-learner calib-slice output…")
    calib_pred = meta.predict_proba(calib_stacked)
    calib_labels = calib_stacked["win"].to_numpy()
    calibrator = Calibrator(CalibratorConfig(method="auto")).fit(calib_pred, calib_labels)
    cal_path = output_dir / "calibration_adr038_brier" / "meta_learner"
    calibrator.save(cal_path)
    summary["models"]["calibrator"] = {
        "path": str(cal_path),
        "chosen_method": calibrator.chosen_method,
    }

    # 6. Marker file + summary.
    print("[6/6] Writing marker + provenance + summary…")
    warning_text = (
        "These models are trained on randomly-generated data. Their "
        "predictions are not meaningful. Re-run scripts/bootstrap_models.py "
        "on a real parquet for production."
    )
    trained_at = datetime.now(timezone.utc).isoformat()

    # Legacy marker — retained for backward compat with existing tests + tools.
    marker = {
        "is_synthetic": True,
        "generated_at": trained_at,
        "n_synthetic_rows": int(len(features)),
        "n_synthetic_races": int(n_synthetic_races),
        "seed": seed,
        "warning": warning_text,
        "produced_by": "scripts/quick_bootstrap.py",
    }
    with open(output_dir / "QUICK_BOOTSTRAP.json", "w") as fh:
        json.dump(marker, fh, indent=2)

    # Unified provenance file — matches app/schemas/provenance.py::ModelProvenance.
    provenance = {
        "is_synthetic": True,
        "trained_at": trained_at,
        "n_train_rows": int(summary.get("rows_train", 0)) or None,
        "n_calib_rows": int(summary.get("rows_calib", 0)) or None,
        "n_test_rows": int(summary.get("rows_test", 0)) or None,
        "sub_models": ["speed_form", "connections", "market"],
        "stub_sub_models": ["pace_scenario", "sequence"],
        "meta_learner_test_ece": summary.get("meta_test_ece"),
        "meta_learner_test_brier": summary.get("meta_test_brier"),
        "bootstrap_script": "scripts/quick_bootstrap.py",
        "bootstrap_seed": int(seed),
        "parquet_path": None,
        "warning": warning_text,
    }
    with open(output_dir / "BOOTSTRAP_PROVENANCE.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    summary["marker_file"] = str(output_dir / "QUICK_BOOTSTRAP.json")
    summary["provenance_file"] = str(output_dir / "BOOTSTRAP_PROVENANCE.json")
    summary["elapsed_seconds"] = round(time.monotonic() - t_start, 2)
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


def _file_size_kb(p: Path) -> float:
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / 1024.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024.0


def _print_success(output_dir: Path, summary: dict, verify: dict) -> None:
    print()
    print("=" * 70)
    print(" QUICK BOOTSTRAP COMPLETE — DEMO ARTIFACTS WRITTEN")
    print("=" * 70)
    print(f"  Output directory : {output_dir}")
    print(f"  Elapsed          : {summary.get('elapsed_seconds')}s")
    print(f"  Rows trained on  : {summary['rows_train']:,} (train)")
    print(f"                     {summary['rows_calib']:,} (calib)")
    print(f"                     {summary['rows_test']:,} (test, unused)")
    print()
    print("  Artifacts:")
    for name, sub in (
        ("speed_form", "speed_form"),
        ("connections", "connections"),
        ("market", "market"),
        ("meta_learner", "meta_learner"),
        ("calibrator", "calibration_adr038_brier/meta_learner"),
    ):
        path = output_dir / sub
        size_kb = _file_size_kb(path)
        print(f"    - {name:<14} {str(path):<60} {size_kb:>8.1f} KB")
    print()
    print("  Verification:")
    if verify.get("inference_artifacts_loader"):
        print("    OK  InferenceArtifacts.load(output_dir) succeeded.")
    else:
        print("    --  InferenceArtifacts class not present yet (Stream X work).")
        print("        Fell back to per-component round-trip checks:")
        for k, v in verify.get("components", {}).items():
            status = "OK " if v else "FAIL"
            print(f"        {status}  {k}")
    print()
    print("  REMINDER: these models are synthetic-trained. Predictions are not")
    print("  meaningful. Re-run scripts/bootstrap_models.py for production.")
    print("=" * 70)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DEMO-ONLY synthetic-data bootstrap. Produces a complete "
            "models/baseline_full/ directory in seconds. NOT FOR PRODUCTION."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the artifact directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-synthetic-races",
        type=int,
        default=DEFAULT_N_RACES,
        help=f"Number of synthetic races to generate (default: {DEFAULT_N_RACES})",
    )
    parser.add_argument(
        "--horses-per-race",
        type=int,
        default=DEFAULT_HORSES_PER_RACE,
        help=f"Horses per synthetic race (default: {DEFAULT_HORSES_PER_RACE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(
        "WARNING — scripts/quick_bootstrap.py trains models on RANDOMLY-"
        "GENERATED data. The resulting predictions are not meaningful. Use\n"
        "         scripts/bootstrap_models.py with a real parquet for "
        "production."
    )
    print()
    args = _parse_args(argv or sys.argv[1:])

    summary = quick_bootstrap(
        output_dir=args.output_dir,
        n_synthetic_races=args.n_synthetic_races,
        horses_per_race=args.horses_per_race,
        seed=args.seed,
    )
    verify = verify_artifacts_load(args.output_dir)

    _print_success(args.output_dir, summary, verify)

    # Non-zero exit if any component failed to round-trip.
    if not verify.get("inference_artifacts_loader"):
        for k, v in verify.get("components", {}).items():
            if not v:
                print(f"ERROR — component '{k}' failed to round-trip.")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
