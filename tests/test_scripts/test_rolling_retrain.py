"""Tests for scripts/rolling_retrain.py — Layer-7 rolling retrain (ADR-044).

These tests use a synthetic parquet built in tmp_path; the real
data/exports/training_*.parquet is gitignored and not touched. The
synthetic generator emits multi-horse races spread across 5 years so
the window-slicing assertion is meaningful and the downstream
sub-models actually have signal to fit.

Covered behaviour:
    1. Window slicing — only rows in [as_of - window, as_of) consumed.
    2. Three-way split is time-ordered (no future leak).
    3. --skip-if-no-drift exits 2 when triggered=false, no artifacts.
    4. --skip-if-no-drift proceeds when triggered=true, artifacts ARE written.
    5. report.json schema — required keys present.
    6. Resilience to a subset --sub-models flag — no error for unfit layers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the scripts/ directory importable so we can call run_rolling_retrain
# directly (faster than subprocess for the bulk of the tests).
_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rolling_retrain import (  # type: ignore  # noqa: E402
    ALL_SUB_MODELS,
    EXIT_OK,
    EXIT_SKIPPED_NO_DRIFT,
    run_rolling_retrain,
    slice_rolling_window,
)
from app.services.calibration.drift import (  # noqa: E402
    CUSUMConfig,
    CUSUMDetector,
    save_drift_state,
)


# ── Synthetic parquet fixture ──────────────────────────────────────────────


def _make_synthetic_parquet(
    n_races: int = 80,
    horses_per_race: int = 6,
    start_date: pd.Timestamp = pd.Timestamp("2021-01-01"),
    end_date: pd.Timestamp = pd.Timestamp("2026-01-01"),
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic parquet-shaped frame spanning ~5 years of multi-horse races.

    Returns at least 200 rows. Speed figures correlate with finish so the
    sub-models have actual signal — keeps fits stable for the report.json
    schema assertions.
    """
    rng = np.random.default_rng(seed)
    span_days = (end_date - start_date).days
    rows = []
    for r in range(n_races):
        race_date = start_date + pd.Timedelta(days=int(rng.integers(0, span_days)))
        track = str(rng.choice(["AAA", "BBB", "CCC"]))
        race_no = int(rng.integers(1, 10))
        # Each horse: latent ability → speed_figure → finish_position.
        abilities = rng.normal(80, 12, horses_per_race)
        noisy = abilities + rng.normal(0, 4, horses_per_race)
        order = np.argsort(-noisy)
        ranks = np.empty(horses_per_race, dtype=int)
        for rank, pos in enumerate(order, start=1):
            ranks[pos] = rank
        for h in range(horses_per_race):
            rows.append({
                "horse_name": f"horse_{r}_{h}",
                "jurisdiction": str(rng.choice(["UK", "HK", "JP"])),
                "race_date": race_date,
                "track_code": track,
                "race_number": race_no,
                "distance_furlongs": float(rng.choice([5.0, 6.0, 8.0])),
                "surface": str(rng.choice(["dirt", "turf"])),
                "condition": str(rng.choice(["fast", "good"])),
                "race_type": str(rng.choice(["claiming", "allowance"])),
                "claiming_price": None,
                "purse_usd": float(rng.choice([30_000.0, 60_000.0, 100_000.0])),
                "field_size": horses_per_race,
                "post_position": h + 1,
                "finish_position": int(ranks[h]),
                "weight_lbs": float(rng.normal(120, 5)),
                "odds_final": float(np.clip(rng.exponential(8.0), 1.5, 80.0)),
                "speed_figure": float(np.clip(abilities[h], 0, 130)),
                "speed_figure_source": "brisnet",
                "fraction_q1_sec": None,
                "fraction_q2_sec": None,
                "fraction_finish_sec": float(rng.normal(70.0, 2.0)),
                "beaten_lengths_q1": None,
                "beaten_lengths_q2": None,
                "data_quality_score": 0.9,
                "foaling_year": None,
                "sire": None,
                "dam_sire": None,
                "jockey_name": f"jockey_{int(rng.integers(0, 12))}",
                "trainer_name": f"trainer_{int(rng.integers(0, 8))}",
            })
    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    """Write a synthetic parquet covering Jan 2021 → Jan 2026 to tmp_path."""
    df = _make_synthetic_parquet(
        n_races=120, horses_per_race=6,
        start_date=pd.Timestamp("2021-01-01"),
        end_date=pd.Timestamp("2026-01-01"),
        seed=11,
    )
    assert len(df) >= 200, f"synthetic too small: {len(df)}"
    path = tmp_path / "training_synth.parquet"
    df.to_parquet(path)
    return path


# ── Test 1 — window slice correctness ───────────────────────────────────────


def test_window_slice_only_uses_rows_in_window(synthetic_parquet: Path):
    """as_of_date=2026-05-13, window_years=3 ⇒ rows in [2023-05-13, 2026-05-13)."""
    raw = pd.read_parquet(synthetic_parquet)
    result = slice_rolling_window(raw, date(2026, 5, 13), window_years=3)
    assert result.window_start == pd.Timestamp("2023-05-13")
    assert result.window_end == pd.Timestamp("2026-05-13")
    # Every row in the result must be inside the window
    dates = pd.to_datetime(result.df["race_date"]).dt.normalize()
    assert (dates >= result.window_start).all()
    assert (dates < result.window_end).all()
    # Rows outside the window must have been excluded
    raw_dates = pd.to_datetime(raw["race_date"]).dt.normalize()
    n_outside = int(((raw_dates < result.window_start) | (raw_dates >= result.window_end)).sum())
    assert result.n_rows_in_window + n_outside == result.n_rows_original


def test_window_slice_zero_when_no_overlap(synthetic_parquet: Path):
    """A window 100 years in the past returns zero rows."""
    raw = pd.read_parquet(synthetic_parquet)
    result = slice_rolling_window(raw, date(1900, 1, 1), window_years=1)
    assert result.n_rows_in_window == 0


def test_window_slice_rejects_zero_or_negative(synthetic_parquet: Path):
    raw = pd.read_parquet(synthetic_parquet)
    with pytest.raises(ValueError):
        slice_rolling_window(raw, date(2026, 5, 13), window_years=0)
    with pytest.raises(ValueError):
        slice_rolling_window(raw, date(2026, 5, 13), window_years=-1)


# ── Test 2 — three-way split is time-ordered ────────────────────────────────


def test_three_way_split_is_time_ordered(synthetic_parquet: Path, tmp_path: Path):
    """max(train_dates) ≤ min(calib_dates) ≤ min(test_dates) — no leakage.

    We drive this through the orchestrator and then read the report.json
    cutoff fields; the actual split lives in `validate_calibration._three_way_split`
    which has its own tests, but we want to assert the orchestrator
    consumes it correctly.
    """
    out = tmp_path / "rolling-time-order"
    summary = run_rolling_retrain(
        parquet_path=synthetic_parquet,
        as_of_date=date(2026, 1, 1),
        window_years=5,
        output_dir=out,
        sub_models=["speed_form"],  # tiny — keeps the test fast
    )
    calib_cutoff = pd.Timestamp(summary["calib_cutoff"])
    test_cutoff = pd.Timestamp(summary["test_cutoff"])
    assert calib_cutoff <= test_cutoff
    # And cutoffs sit inside the window.
    assert calib_cutoff >= pd.Timestamp(summary["window_start"])
    assert test_cutoff < pd.Timestamp(summary["window_end"])


# ── Tests 3 + 4 — --skip-if-no-drift behaviour via subprocess ───────────────


def _build_drift_state_file(path: Path, *, triggered: bool) -> None:
    """Write a drift-state JSON in the format save_drift_state produces."""
    det = CUSUMDetector(CUSUMConfig(k=0.5, h=1.0))  # h=1 so 1 obs can trigger
    if triggered:
        # Drive the detector deliberately over the threshold.
        for _ in range(8):
            det.update(0.05, 1.0)  # massive under-prediction on a longshot hit
        assert det.state.alarmed, "fixture did not trigger detector"
    save_drift_state(det, path)


def test_skip_if_no_drift_exits_2_when_not_triggered(
    synthetic_parquet: Path, tmp_path: Path,
):
    drift_state = tmp_path / "drift_state.json"
    _build_drift_state_file(drift_state, triggered=False)
    out = tmp_path / "rolling-noop"

    cli = [
        sys.executable,
        str(_SCRIPTS / "rolling_retrain.py"),
        "--parquet", str(synthetic_parquet),
        "--as-of-date", "2026-01-01",
        "--window-years", "5",
        "--output-dir", str(out),
        "--sub-models", "speed_form",
        "--skip-if-no-drift",
        "--drift-state-path", str(drift_state),
    ]
    proc = subprocess.run(cli, capture_output=True, text=True, cwd=str(_ROOT))
    assert proc.returncode == EXIT_SKIPPED_NO_DRIFT, (
        f"expected exit {EXIT_SKIPPED_NO_DRIFT}, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # No artifacts must have been written.
    assert not out.exists() or not list(out.iterdir()), (
        "no-op run wrote artifacts to output dir"
    )


def test_skip_if_no_drift_proceeds_when_triggered(
    synthetic_parquet: Path, tmp_path: Path,
):
    drift_state = tmp_path / "drift_state.json"
    _build_drift_state_file(drift_state, triggered=True)
    out = tmp_path / "rolling-proceed"

    cli = [
        sys.executable,
        str(_SCRIPTS / "rolling_retrain.py"),
        "--parquet", str(synthetic_parquet),
        "--as-of-date", "2026-01-01",
        "--window-years", "5",
        "--output-dir", str(out),
        "--sub-models", "speed_form",
        "--skip-if-no-drift",
        "--drift-state-path", str(drift_state),
    ]
    proc = subprocess.run(cli, capture_output=True, text=True, cwd=str(_ROOT))
    assert proc.returncode == EXIT_OK, (
        f"expected exit {EXIT_OK}, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # Artifacts must have been written.
    assert (out / "report.json").exists()
    assert (out / "speed_form").exists()
    assert (out / "meta_learner").exists()
    assert (out / "calibrator").exists()


# ── Test 5 — report.json schema ─────────────────────────────────────────────


def test_report_json_has_required_keys(synthetic_parquet: Path, tmp_path: Path):
    out = tmp_path / "rolling-schema"
    summary = run_rolling_retrain(
        parquet_path=synthetic_parquet,
        as_of_date=date(2026, 1, 1),
        window_years=5,
        output_dir=out,
        sub_models=["speed_form", "connections", "market"],
    )
    required = {
        "n_train_rows", "n_calib_rows", "n_test_rows",
        "sub_models_trained",
        "meta_learner_ece", "meta_learner_brier", "meta_learner_logloss",
        "as_of_date", "window_years",
    }
    assert required.issubset(summary.keys()), (
        f"missing keys: {required - set(summary.keys())}"
    )
    # And the on-disk JSON matches.
    with open(out / "report.json") as fh:
        on_disk = json.load(fh)
    assert required.issubset(on_disk.keys())
    assert on_disk["as_of_date"] == "2026-01-01"
    assert on_disk["window_years"] == 5
    # Sub-model trained list matches the requested set (minus stubs).
    assert set(on_disk["sub_models_trained"]) == {"speed_form", "connections", "market"}


# ── Test 6 — resilience to missing sub-models ───────────────────────────────


def test_partial_sub_models_does_not_error(synthetic_parquet: Path, tmp_path: Path):
    """Only --sub-models speed_form: pace/sequence/connections/market untrained.

    The orchestrator must still build the meta-learner (relying on ADR-026's
    constant-0.5 stubs) and emit a complete report.json. report.json's
    sub_models_trained must reflect just what was trained.
    """
    out = tmp_path / "rolling-partial"
    summary = run_rolling_retrain(
        parquet_path=synthetic_parquet,
        as_of_date=date(2026, 1, 1),
        window_years=5,
        output_dir=out,
        sub_models=["speed_form"],
    )
    assert summary["sub_models_trained"] == ["speed_form"]
    # Only speed_form artifact dir written
    assert (out / "speed_form").exists()
    assert not (out / "connections").exists()
    assert not (out / "market").exists()
    # Meta + calibrator still written (they always run)
    assert (out / "meta_learner").exists()
    assert (out / "calibrator").exists()
    # report.json sub_models only documents speed_form
    with open(out / "report.json") as fh:
        on_disk = json.load(fh)
    assert "speed_form" in on_disk["sub_models"]
    assert "connections" not in on_disk["sub_models"]
    assert "market" not in on_disk["sub_models"]


def test_unknown_sub_model_raises(synthetic_parquet: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown sub-model"):
        run_rolling_retrain(
            parquet_path=synthetic_parquet,
            as_of_date=date(2026, 1, 1),
            window_years=5,
            output_dir=tmp_path / "bogus",
            sub_models=["totally_made_up"],
        )


def test_all_known_sub_models_listed():
    """Future-proofs ALL_SUB_MODELS — if a new layer is added, this test
    pings to add it to the rolling retrain script too."""
    assert set(ALL_SUB_MODELS) == {
        "speed_form", "pace_scenario", "sequence", "connections", "market",
    }
