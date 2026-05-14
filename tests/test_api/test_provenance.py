"""
tests/test_api/test_provenance.py
─────────────────────────────────
Step 4 — ModelProvenance surfaces correctly through every prediction-bearing
API endpoint, and `InferenceArtifacts.load` falls back to a synthetic-flagged
ModelProvenance when BOOTSTRAP_PROVENANCE.json is missing.

The aim is not to re-verify the underlying inference pipeline (covered by
test_cards / test_portfolio / test_pareto); it's purely to assert that
`model_provenance` round-trips from the artifact directory through the API.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

# Each test module gets its own DB to avoid cross-test bleed.
_TEST_DB_PATH = f"./_test_provenance_{uuid.uuid4().hex}.db"
os.environ["HRBS_DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"
os.environ["HRBS_MODELS_DIR"] = "/nonexistent-models-dir-for-tests"

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.persistence import persist_ingestion_result  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402
from app.main import create_app  # noqa: E402
from app.schemas.provenance import ModelProvenance  # noqa: E402
from app.schemas.race import (  # noqa: E402
    HorseEntry,
    IngestionResult,
    ParsedRace,
    PastPerformanceLine,
    RaceCard,
    RaceHeader,
    RaceType,
    Surface,
    TrackCondition,
)
from app.services.calibration.calibrator import Calibrator, CalibratorConfig  # noqa: E402
from app.services.inference.pipeline import (  # noqa: E402
    InferenceArtifacts,
    _load_provenance,
)
from app.services.models.pace_scenario import PaceScenarioModel  # noqa: E402
from app.services.models.sequence import SequenceModel  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures (stubs identical in spirit to test_cards.py / test_portfolio.py)
# ──────────────────────────────────────────────────────────────────────────────


class _StubModel:
    def __init__(self, c: float = 0.5):
        self.c = float(c)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.c, dtype=float)


class _RankMeta:
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if "speed_form_proba" in df.columns:
            return np.clip(df["speed_form_proba"].to_numpy(), 0.0, 1.0)
        return np.full(len(df), 0.5, dtype=float)


def _calibrator() -> Calibrator:
    rng = np.random.default_rng(0)
    n = 2_000
    scores = rng.uniform(0.1, 0.9, size=n)
    labels = (rng.uniform(size=n) < scores).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto"))
    cal.fit(scores, labels)
    return cal


def _stub_artifacts(provenance: ModelProvenance) -> InferenceArtifacts:
    return InferenceArtifacts(
        speed_form=_StubModel(0.5),
        pace_scenario=PaceScenarioModel(),
        sequence=SequenceModel(),
        connections=_StubModel(0.5),
        market=_StubModel(0.5),
        meta_learner=_RankMeta(),
        meta_calibrator=_calibrator(),
        models_dir=Path("/stub"),
        available_sub_models=("stub",),
        provenance=provenance,
    )


def _pp(d: date, speed: float, finish: int) -> PastPerformanceLine:
    return PastPerformanceLine(
        race_date=d,
        track_code="CD",
        race_number=1,
        distance_furlongs=6.0,
        surface=Surface.DIRT,
        condition=TrackCondition.FAST,
        race_type=RaceType.ALLOWANCE,
        post_position=1,
        finish_position=finish,
        lengths_behind=0.0 if finish == 1 else float(finish),
        field_size=6,
        weight_lbs=120.0,
        odds_final=4.0,
        speed_figure=speed,
        purse_usd=50000.0,
    )


def _entry(post: int, name: str, ml: float, speed: float) -> HorseEntry:
    today = date(2026, 5, 10)
    pps = [_pp(today - timedelta(days=30 * (i + 1)), speed + i, finish=(i % 6) + 1) for i in range(3)]
    return HorseEntry(
        horse_name=name,
        post_position=post,
        morning_line_odds=ml,
        weight_lbs=120.0,
        jockey="Joe Jockey",
        trainer="Trainer A",
        pp_lines=pps,
    )


def _synthetic_card() -> RaceCard:
    header = RaceHeader(
        race_number=1,
        race_date=date(2026, 5, 10),
        track_code="CD",
        track_name="Churchill Downs",
        distance_furlongs=6.0,
        distance_raw="6 Furlongs",
        surface=Surface.DIRT,
        condition=TrackCondition.FAST,
        race_type=RaceType.ALLOWANCE,
        purse_usd=60000.0,
    )
    entries = [
        _entry(1, "Alpha", 3.0, 85.0),
        _entry(2, "Bravo", 4.0, 82.0),
        _entry(3, "Charlie", 6.0, 80.0),
        _entry(4, "Delta", 10.0, 75.0),
        _entry(5, "Echo", 15.0, 70.0),
    ]
    return RaceCard(
        source_filename="test.pdf",
        source_format="brisnet_up",
        total_pages=1,
        card_date=date(2026, 5, 10),
        track_code="CD",
        races=[ParsedRace(header=header, entries=entries, parse_confidence=0.9)],
    )


async def _persist(card: RaceCard) -> int:
    factory = get_session_factory()
    async with factory() as session:
        result = IngestionResult(success=True, card=card, processing_ms=10.0)
        pk = await persist_ingestion_result(session, result)
        await session.commit()
        assert pk is not None
        return pk


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c
    try:
        os.remove(_TEST_DB_PATH)
    except OSError:
        pass


@pytest.fixture(scope="module")
def persisted_card_id(client):
    import asyncio
    card = _synthetic_card()
    return asyncio.run(_persist(card))


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_load_provenance_missing_file_returns_synthetic_warning(tmp_path):
    """When BOOTSTRAP_PROVENANCE.json is absent from the models dir, the
    loader returns a ModelProvenance with is_synthetic=True and a warning."""
    prov = _load_provenance(tmp_path)
    assert prov.is_synthetic is True
    assert prov.warning is not None
    assert "missing" in prov.warning.lower()


def test_load_provenance_round_trips_real_payload(tmp_path):
    """A real BOOTSTRAP_PROVENANCE.json round-trips into the ModelProvenance schema."""
    payload = {
        "is_synthetic": False,
        "trained_at": "2026-05-13T22:00:00Z",
        "n_train_rows": 1_000_000,
        "n_calib_rows": 250_000,
        "n_test_rows": 250_000,
        "sub_models": ["speed_form", "connections", "market"],
        "stub_sub_models": ["pace_scenario", "sequence"],
        "meta_learner_test_ece": 0.0042,
        "meta_learner_test_brier": 0.068,
        "bootstrap_script": "scripts/bootstrap_models.py",
        "bootstrap_seed": 42,
        "parquet_path": "data/exports/training_20260513.parquet",
        "warning": None,
    }
    (tmp_path / "BOOTSTRAP_PROVENANCE.json").write_text(json.dumps(payload))
    prov = _load_provenance(tmp_path)
    assert prov.is_synthetic is False
    assert prov.n_train_rows == 1_000_000
    assert prov.meta_learner_test_ece == pytest.approx(0.0042)
    assert prov.bootstrap_seed == 42
    assert prov.warning is None


def test_load_provenance_invalid_json_falls_back(tmp_path):
    """Garbage JSON is treated as missing — yields synthetic flag + warning."""
    (tmp_path / "BOOTSTRAP_PROVENANCE.json").write_text("not-json{{")
    prov = _load_provenance(tmp_path)
    assert prov.is_synthetic is True
    assert prov.warning is not None


def test_cards_endpoint_exposes_synthetic_provenance(client, persisted_card_id):
    """GET /cards/{id} embeds the artifacts' model_provenance verbatim."""
    prov = ModelProvenance(
        is_synthetic=True,
        warning="stub synthetic warning",
        sub_models=["speed_form"],
        stub_sub_models=["pace_scenario", "sequence"],
    )
    client.app.state.artifacts = _stub_artifacts(prov)
    client.app.state.inference_cache = {}
    r = client.get(f"/api/v1/cards/{persisted_card_id}")
    assert r.status_code == 200
    body = r.json()
    assert "model_provenance" in body
    assert body["model_provenance"]["is_synthetic"] is True
    assert body["model_provenance"]["warning"] == "stub synthetic warning"
    assert body["model_provenance"]["stub_sub_models"] == ["pace_scenario", "sequence"]


def test_cards_endpoint_exposes_real_provenance(client, persisted_card_id):
    """When provenance.is_synthetic=False, the response reflects that."""
    prov = ModelProvenance(
        is_synthetic=False,
        trained_at="2026-05-13T00:00:00Z",
        n_train_rows=1234,
        meta_learner_test_ece=0.01,
        sub_models=["speed_form", "connections", "market"],
        stub_sub_models=[],
    )
    client.app.state.artifacts = _stub_artifacts(prov)
    client.app.state.inference_cache = {}
    r = client.get(f"/api/v1/cards/{persisted_card_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["model_provenance"]["is_synthetic"] is False
    assert body["model_provenance"]["n_train_rows"] == 1234
    assert body["model_provenance"]["meta_learner_test_ece"] == pytest.approx(0.01)
    assert body["model_provenance"]["stub_sub_models"] == []


def test_portfolio_endpoint_exposes_provenance(client, persisted_card_id):
    prov = ModelProvenance(is_synthetic=True, warning="synthetic", sub_models=["a"])
    client.app.state.artifacts = _stub_artifacts(prov)
    client.app.state.inference_cache = {}
    r = client.get(f"/api/v1/portfolio/{persisted_card_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["model_provenance"]["is_synthetic"] is True
    assert body["model_provenance"]["warning"] == "synthetic"


def test_pareto_endpoint_exposes_provenance(client, persisted_card_id):
    prov = ModelProvenance(
        is_synthetic=False, sub_models=["speed_form", "connections", "market"]
    )
    client.app.state.artifacts = _stub_artifacts(prov)
    client.app.state.inference_cache = {}
    r = client.get(
        f"/api/v1/portfolio/{persisted_card_id}/pareto",
        params={"risk_levels": "0.05,0.10,0.20"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_provenance"]["is_synthetic"] is False
    assert body["model_provenance"]["sub_models"] == [
        "speed_form",
        "connections",
        "market",
    ]
