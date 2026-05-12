"""Smoke tests for scripts/validate_calibration.py.

The script's `evaluate_calibration` function is reused on synthetic data
to verify pre/post-calibration metrics are computed correctly and the
reliability diagram PNG is emitted.

Full end-to-end CLI is exercised manually on the real parquet — these
tests just lock in the pure-function contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Hook the scripts/ directory into sys.path so we can import the validator
# without making it a proper package.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from validate_calibration import (  # type: ignore  # noqa: E402
    CalibrationReport,
    evaluate_calibration,
    render_reliability_diagram,
)


def _biased_synthetic(n: int = 4000, seed: int = 11):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, n).astype(float)
    true_p = 1.0 / (1.0 + np.exp(-(2.0 * raw - 1.0)))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    return raw, labels


def test_evaluate_calibration_returns_pre_and_post_metrics():
    raw, labels = _biased_synthetic(n=4000)
    # 50/50 calib/test split.
    n = len(raw)
    cal_scores, cal_labels = raw[: n // 2], labels[: n // 2]
    te_scores, te_labels = raw[n // 2 :], labels[n // 2 :]

    cal, report = evaluate_calibration(
        cal_scores, cal_labels, te_scores, te_labels, label="smoke"
    )
    assert isinstance(report, CalibrationReport)
    assert report.label == "smoke"
    assert report.chosen_method in ("platt", "isotonic")
    assert report.n_calib == n // 2
    assert report.n_test == n - n // 2
    # Calibration should NOT make ECE worse on biased synthetic.
    assert report.post_ece <= report.pre_ece + 0.005
    # Sanity: bin lists have the requested length.
    assert len(report.pre_bins) == 15
    assert len(report.post_bins) == 15


def test_evaluate_calibration_dict_round_trips():
    raw, labels = _biased_synthetic(n=2000)
    n = len(raw)
    _, report = evaluate_calibration(
        raw[: n // 2], labels[: n // 2], raw[n // 2 :], labels[n // 2 :],
        label="round-trip-test",
    )
    serialised = json.dumps(report.asdict())
    payload = json.loads(serialised)
    assert payload["label"] == "round-trip-test"
    assert "pre_ece" in payload and "post_ece" in payload
    assert len(payload["pre_bins"]) == 15


def test_render_reliability_diagram_writes_png(tmp_path: Path):
    raw, labels = _biased_synthetic(n=2000)
    n = len(raw)
    _, report = evaluate_calibration(
        raw[: n // 2], labels[: n // 2], raw[n // 2 :], labels[n // 2 :],
        label="png-smoke",
    )
    png_path = tmp_path / "reliability.png"
    render_reliability_diagram(report, png_path)
    assert png_path.exists()
    # PNG header is 8 bytes.
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
