"""
tests/test_api/test_portfolio.py
────────────────────────────────
Stream A — GET /api/v1/portfolio/{card_id} integration tests.

Strategy mirrors test_cards.py — see that module for fixture rationale.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

_TEST_DB_PATH = f"./_test_portfolio_{uuid.uuid4().hex}.db"
os.environ["HRBS_DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"
os.environ["HRBS_MODELS_DIR"] = "/nonexistent-models-dir-for-tests"

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.persistence import persist_ingestion_result  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402
from app.main import create_app  # noqa: E402
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


class _StubModel:
    def __init__(self, c: float):
        self.c = float(c)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.c, dtype=float)


class _RankMeta:
    """Returns probs proportional to ewm_speed_prior so different horses get
    different ranks. We rely on the calibrator's softmax to renormalise."""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if "ewm_speed_prior" in df.columns:
            vals = df["ewm_speed_prior"].fillna(50.0).to_numpy()
            vals = (vals - vals.min()) / max(1.0, (vals.max() - vals.min())) * 0.8 + 0.1
            return np.clip(vals, 0.0, 1.0)
        return np.full(len(df), 0.5, dtype=float)


def _stub_artifacts() -> InferenceArtifacts:
    rng = np.random.default_rng(1)
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
    pps = [
        _pp(today - timedelta(days=30 * (i + 1)), speed + i, finish=(i % 6) + 1)
        for i in range(3)
    ]
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
    """A card with a couple of races and a clear favourite per race so the
    calibrated win prob is non-uniform — gives the EV engine something to
    bite on."""
    headers = []
    races = []
    for rn in (1, 2):
        h = RaceHeader(
            race_number=rn,
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
        headers.append(h)
        entries = [
            _entry(1, f"R{rn}-Alpha", 3.0, 95.0),
            _entry(2, f"R{rn}-Bravo", 4.0, 80.0),
            _entry(3, f"R{rn}-Charlie", 6.0, 70.0),
            _entry(4, f"R{rn}-Delta", 10.0, 60.0),
            _entry(5, f"R{rn}-Echo", 15.0, 50.0),
        ]
        races.append(ParsedRace(header=h, entries=entries, parse_confidence=0.9))
    return RaceCard(
        source_filename="test.pdf",
        source_format="brisnet_up",
        total_pages=1,
        card_date=date(2026, 5, 10),
        track_code="CD",
        races=races,
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


def test_portfolio_503_when_artifacts_missing(client, persisted_card_id):
    client.app.state.artifacts = None
    r = client.get(f"/api/v1/portfolio/{persisted_card_id}")
    assert r.status_code == 503


def test_portfolio_404_unknown(client):
    client.app.state.artifacts = _stub_artifacts()
    r = client.get("/api/v1/portfolio/9999999")
    assert r.status_code == 404


def test_portfolio_returns_aggregated_portfolio(client, persisted_card_id):
    client.app.state.artifacts = _stub_artifacts()
    r = client.get(
        f"/api/v1/portfolio/{persisted_card_id}",
        params={
            "bankroll": 10000,
            "min_edge": 0.0,  # be permissive in tests
            "n_scenarios": 200,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["card_id"] == str(persisted_card_id)
    assert body["bankroll"] == 10000.0
    assert isinstance(body["recommendations"], list)
    # Risk metrics present + within schema bounds.
    assert 0.0 <= body["total_stake_fraction"] <= 1.0
    assert isinstance(body["expected_return"], float)
    assert isinstance(body["var_95"], float)
    assert isinstance(body["cvar_95"], float)


def test_portfolio_respects_query_overrides(client, persisted_card_id):
    """Tighten min_edge so the portfolio shrinks; verify it's smaller than
    the permissive call above."""
    client.app.state.artifacts = _stub_artifacts()
    r_loose = client.get(
        f"/api/v1/portfolio/{persisted_card_id}",
        params={"bankroll": 10000, "min_edge": 0.0, "n_scenarios": 200},
    )
    r_tight = client.get(
        f"/api/v1/portfolio/{persisted_card_id}",
        params={"bankroll": 10000, "min_edge": 0.95, "n_scenarios": 200},
    )
    assert r_loose.status_code == 200
    assert r_tight.status_code == 200
    n_loose = len(r_loose.json()["recommendations"])
    n_tight = len(r_tight.json()["recommendations"])
    assert n_tight <= n_loose


def test_portfolio_bet_cap_enforced(client, persisted_card_id):
    """Per ADR-002: no single bet > 3% of bankroll."""
    client.app.state.artifacts = _stub_artifacts()
    r = client.get(
        f"/api/v1/portfolio/{persisted_card_id}",
        params={"bankroll": 10000, "min_edge": 0.0, "n_scenarios": 200},
    )
    assert r.status_code == 200
    for rec in r.json()["recommendations"]:
        assert rec["stake_fraction"] <= 0.03 + 1e-9
