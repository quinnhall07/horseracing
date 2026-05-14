"""
tests/test_api/test_cards.py
────────────────────────────
Stream A — GET /api/v1/cards/{card_id} integration tests.

Strategy
  * Use a per-test-module SQLite tmpfile (mirrors test_ingest.py pattern).
  * Stub InferenceArtifacts with a fake that returns deterministic vectors.
  * Persist a synthetic RaceCard directly via the persistence layer, then
    GET it back with hydration.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Each test module gets a fresh DB.
_TEST_DB_PATH = f"./_test_cards_{uuid.uuid4().hex}.db"
os.environ["HRBS_DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"
# Force-skip the real model directory so the app starts without artifacts.
os.environ["HRBS_MODELS_DIR"] = "/nonexistent-models-dir-for-tests"

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.persistence import persist_ingestion_result  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402
from app.main import create_app  # noqa: E402
from app.schemas.bets import BetType  # noqa: E402
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
from app.services.inference.pipeline import InferenceArtifacts  # noqa: E402
from app.services.models.pace_scenario import PaceScenarioModel  # noqa: E402
from app.services.models.sequence import SequenceModel  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
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


def _stub_artifacts() -> InferenceArtifacts:
    rng = np.random.default_rng(0)
    n = 2_000
    scores = rng.uniform(0.1, 0.9, size=n)
    labels = (rng.uniform(size=n) < scores).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto"))
    cal.fit(scores, labels)
    return InferenceArtifacts(
        speed_form=_StubModel(0.5),
        pace_scenario=PaceScenarioModel(),
        sequence=SequenceModel(),
        connections=_StubModel(0.5),
        market=_StubModel(0.5),
        meta_learner=_RankMeta(),
        meta_calibrator=cal,
        models_dir=Path("/stub"),
        available_sub_models=("stub",),
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
    """Persist a card via the persistence layer and return its DB pk."""
    import asyncio

    card = _synthetic_card()
    pk = asyncio.run(_persist(card))
    return pk


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_cards_503_when_artifacts_missing(client, persisted_card_id):
    """At startup the models dir doesn't exist → artifacts is None → 503."""
    # Defensive: clear cached artifacts.
    client.app.state.artifacts = None
    r = client.get(f"/api/v1/cards/{persisted_card_id}")
    assert r.status_code == 503
    assert "not loaded" in r.json()["detail"].lower()


def test_cards_404_unknown(client):
    client.app.state.artifacts = _stub_artifacts()
    r = client.get("/api/v1/cards/9999999")
    assert r.status_code == 404


def test_cards_404_non_integer(client):
    client.app.state.artifacts = _stub_artifacts()
    r = client.get("/api/v1/cards/not-a-number")
    assert r.status_code == 404


def test_cards_hydrates_calibrated_probs(client, persisted_card_id):
    client.app.state.artifacts = _stub_artifacts()
    client.app.state.inference_cache = {}
    r = client.get(f"/api/v1/cards/{persisted_card_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["source_filename"] == "test.pdf"
    races = body["races"]
    assert len(races) == 1
    entries = races[0]["entries"]
    probs = [e["model_prob"] for e in entries]
    # Hydration succeeded: model_prob populated everywhere.
    for p in probs:
        assert p is not None
        assert 0.0 <= p <= 1.0
    assert abs(sum(probs) - 1.0) < 1e-6
    # Edge and market_prob computed from morning_line_odds.
    for e in entries:
        assert e["market_prob"] is not None
        assert e["edge"] is not None
        assert abs(e["market_prob"] - 1.0 / e["morning_line_odds"]) < 1e-6


def test_cards_repeated_get_uses_cache(client, persisted_card_id):
    """Second GET should not hit the inference path — cache populated."""
    client.app.state.artifacts = _stub_artifacts()
    client.app.state.inference_cache = {}
    r1 = client.get(f"/api/v1/cards/{persisted_card_id}")
    assert r1.status_code == 200
    cache_after_first = dict(client.app.state.inference_cache)
    assert str(persisted_card_id) in cache_after_first
    r2 = client.get(f"/api/v1/cards/{persisted_card_id}")
    assert r2.status_code == 200
    # Cache key still present, vectors unchanged (identity preserved).
    cache_after_second = client.app.state.inference_cache
    assert cache_after_second[str(persisted_card_id)] is cache_after_first[str(persisted_card_id)]
