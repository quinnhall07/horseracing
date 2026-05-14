"""
tests/test_api/test_pareto.py
─────────────────────────────
Stream X — GET /api/v1/portfolio/{card_id}/pareto integration tests.

Mirrors the fixture strategy in test_portfolio.py: stub InferenceArtifacts
injected into app.state, synthetic 2-race card persisted once per module.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

_TEST_DB_PATH = f"./_test_pareto_{uuid.uuid4().hex}.db"
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
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if "ewm_speed_prior" in df.columns:
            vals = df["ewm_speed_prior"].fillna(50.0).to_numpy()
            vals = (vals - vals.min()) / max(1.0, (vals.max() - vals.min())) * 0.8 + 0.1
            return np.clip(vals, 0.0, 1.0)
        return np.full(len(df), 0.5, dtype=float)


def _stub_artifacts() -> InferenceArtifacts:
    rng = np.random.default_rng(7)
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


def test_pareto_503_when_artifacts_missing(client, persisted_card_id):
    client.app.state.artifacts = None
    r = client.get(f"/api/v1/portfolio/{persisted_card_id}/pareto")
    assert r.status_code == 503


def test_pareto_404_unknown(client):
    client.app.state.artifacts = _stub_artifacts()
    r = client.get("/api/v1/portfolio/9999999/pareto")
    assert r.status_code == 404


def test_pareto_default_returns_six_points(client, persisted_card_id):
    client.app.state.artifacts = _stub_artifacts()
    r = client.get(
        f"/api/v1/portfolio/{persisted_card_id}/pareto",
        params={"bankroll": 10000, "min_edge": 0.0, "n_scenarios": 200},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["card_id"] == str(persisted_card_id)
    assert body["bankroll"] == 10000.0
    assert len(body["frontier"]) == 6
    expected_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    actual_levels = [p["max_drawdown_pct"] for p in body["frontier"]]
    assert actual_levels == pytest.approx(expected_levels, abs=1e-6)
    # Every recommendation across every frontier point respects ADR-002.
    for point in body["frontier"]:
        for rec in point["portfolio"]["recommendations"]:
            assert rec["stake_fraction"] <= 0.03 + 1e-9


def test_pareto_monotone_in_expected_return(client, persisted_card_id):
    """As risk budget loosens, expected_return must not decrease."""
    client.app.state.artifacts = _stub_artifacts()
    r = client.get(
        f"/api/v1/portfolio/{persisted_card_id}/pareto",
        params={"bankroll": 10000, "min_edge": 0.0, "n_scenarios": 200},
    )
    assert r.status_code == 200
    returns = [p["portfolio"]["expected_return"] for p in r.json()["frontier"]]
    # Allow tiny numerical slack from LP solver tolerances.
    for prev, cur in zip(returns, returns[1:]):
        assert cur >= prev - 1e-6, f"non-monotone: {prev} → {cur}"


def test_pareto_custom_risk_levels(client, persisted_card_id):
    client.app.state.artifacts = _stub_artifacts()
    r = client.get(
        f"/api/v1/portfolio/{persisted_card_id}/pareto",
        params={
            "bankroll": 10000,
            "min_edge": 0.0,
            "n_scenarios": 200,
            "risk_levels": "0.10,0.20",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["frontier"]) == 2
    assert body["frontier"][0]["max_drawdown_pct"] == pytest.approx(0.10)
    assert body["frontier"][1]["max_drawdown_pct"] == pytest.approx(0.20)


def test_pareto_malformed_risk_levels_returns_400(client, persisted_card_id):
    client.app.state.artifacts = _stub_artifacts()
    # Out of range
    r = client.get(
        f"/api/v1/portfolio/{persisted_card_id}/pareto",
        params={"risk_levels": "0.10,1.5"},
    )
    assert r.status_code == 400
    # Non-numeric
    r2 = client.get(
        f"/api/v1/portfolio/{persisted_card_id}/pareto",
        params={"risk_levels": "0.10,abc"},
    )
    assert r2.status_code == 400
    # Duplicates
    r3 = client.get(
        f"/api/v1/portfolio/{persisted_card_id}/pareto",
        params={"risk_levels": "0.10,0.10,0.20"},
    )
    assert r3.status_code == 400


def test_pareto_unit_pipeline_two_points(client):
    """Direct unit test of analyze_card_pareto (bypassing the HTTP layer)."""
    from app.services.inference.pipeline import analyze_card_pareto

    artifacts = _stub_artifacts()
    card = _synthetic_card()
    points, n_cands = analyze_card_pareto(
        card,
        artifacts,
        risk_levels=[0.05, 0.20],
        bankroll=10000.0,
        min_edge=0.0,
        n_scenarios=200,
        card_id="unit-test",
    )
    assert len(points) == 2
    assert n_cands >= 0
    # Higher risk → at-least-equal expected_return.
    low, high = points[0][1], points[1][1]
    assert high.expected_return >= low.expected_return - 1e-6
