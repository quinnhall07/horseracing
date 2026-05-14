"""Smoke test for `scripts/quick_bootstrap.py`.

Runs the full DEMO-ONLY synthetic bootstrap end-to-end on a tiny race count
and asserts every artifact directory is on disk and round-trips through
its loader. The marker file's `is_synthetic` flag must be True.

Per CLAUDE.md §10: every service module gets a test. quick_bootstrap is a
script but it composes the same model loaders that production uses, so a
fail here means a Stream X / clean-clone user will see a 503.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so `import scripts.quick_bootstrap`
# resolves the same way it does when the user runs the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.calibration.calibrator import Calibrator
from app.services.models.connections import ConnectionsModel
from app.services.models.market import MarketModel
from app.services.models.meta_learner import MetaLearner
from app.services.models.speed_form import SpeedFormModel
from scripts.quick_bootstrap import (
    generate_synthetic_results,
    quick_bootstrap,
    three_way_time_split,
)


def test_generate_synthetic_results_shape_and_signal():
    """Synthetic frame has the columns prepare_training_features reads, plus a
    weak speed→win correlation so LightGBM has something to learn."""
    df = generate_synthetic_results(n_races=20, horses_per_race=6, seed=0)
    assert len(df) == 20 * 6
    # Required columns for prepare_training_features.
    for col in (
        "race_date", "track_code", "race_number", "distance_furlongs",
        "surface", "condition", "race_type", "field_size", "purse_usd",
        "weight_lbs", "odds_final", "speed_figure", "finish_position",
        "horse_name", "jurisdiction", "jockey_name", "trainer_name",
    ):
        assert col in df.columns, f"missing column {col}"

    # The winner of each race should be (mostly) the highest-speed-figure
    # horse — i.e. there must be SOME positive correlation between speed
    # figure and the binary win outcome.
    df = df.copy()
    df["won"] = (df["finish_position"] == 1).astype(int)
    corr = df["speed_figure"].corr(df["won"])
    assert corr > 0.05, f"expected positive speed→win correlation, got {corr:.4f}"


def test_three_way_time_split_is_strictly_temporal():
    """Split is by race_date; train.max < calib.min < test.min."""
    df = generate_synthetic_results(n_races=50, horses_per_race=6, seed=1)
    train, calib, test = three_way_time_split(df, train_frac=0.6, cal_frac=0.2)
    assert len(train) + len(calib) + len(test) == len(df)
    assert train["race_date"].max() <= calib["race_date"].min()
    assert calib["race_date"].max() <= test["race_date"].min()


def test_quick_bootstrap_end_to_end_produces_loadable_artifacts(tmp_path: Path):
    """The full pipeline writes every artifact in the layout the runtime needs."""
    out = tmp_path / "models" / "demo"
    summary = quick_bootstrap(
        output_dir=out,
        n_synthetic_races=80,
        horses_per_race=6,
        seed=11,
    )

    # Summary is sane.
    assert summary["is_synthetic"] is True
    assert summary["rows_total"] == 80 * 6
    assert summary["rows_train"] + summary["rows_calib"] + summary["rows_test"] == 80 * 6

    # Marker file is present, is_synthetic flag = True, warning string is there.
    marker = json.loads((out / "QUICK_BOOTSTRAP.json").read_text())
    assert marker["is_synthetic"] is True
    assert marker["n_synthetic_rows"] == 80 * 6
    assert "warning" in marker
    assert "not meaningful" in marker["warning"].lower()

    # Unified provenance file is present and matches ModelProvenance schema.
    prov = json.loads((out / "BOOTSTRAP_PROVENANCE.json").read_text())
    assert prov["is_synthetic"] is True
    assert prov["bootstrap_script"] == "scripts/quick_bootstrap.py"
    assert "speed_form" in prov["sub_models"]
    assert set(prov["stub_sub_models"]) >= {"pace_scenario", "sequence"}
    assert prov["warning"] and "not meaningful" in prov["warning"].lower()

    # Every artifact round-trips through its loader.
    SpeedFormModel.load(out / "speed_form")
    ConnectionsModel.load(out / "connections")
    MarketModel.load(out / "market")
    MetaLearner.load(out / "meta_learner")
    Calibrator.load(out / "calibration_adr038_brier" / "meta_learner")
