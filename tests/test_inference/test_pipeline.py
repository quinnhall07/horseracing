"""
tests/test_inference/test_pipeline.py
─────────────────────────────────────
Stream A — pipeline-module unit tests.

Strategy
  * Build a synthetic RaceCard (2 races × 5 horses each, with PP history).
  * Mock out the sub-models and meta-learner with constant-output stubs.
  * Use a real Calibrator fit on synthetic data to validate the softmax
    step (this avoids brittle mocking of `predict_softmax`).
  * Assert: feature DataFrame schema, per-race prob normalization,
    portfolio construction sums, exception tolerance for missing sub-models.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from app.schemas.bets import BetCandidate
from app.schemas.race import (
    BetType,
    HorseEntry,
    ParsedRace,
    PastPerformanceLine,
    RaceCard,
    RaceHeader,
    RaceType,
    Surface,
    TrackCondition,
)
from app.services.calibration.calibrator import Calibrator, CalibratorConfig
from app.services.inference.pipeline import (
    DEFAULT_BANKROLL,
    InferenceArtifacts,
    analyze_card,
    build_inference_features,
    build_portfolio_from_candidates,
    infer_calibrated_win_probs,
    race_card_to_features,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _pp(d: date, speed: float, finish: int, field_size: int = 8) -> PastPerformanceLine:
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
        field_size=field_size,
        weight_lbs=120.0,
        odds_final=4.0,
        speed_figure=speed,
        purse_usd=50000.0,
    )


def _entry(
    post: int, name: str, *, ml: float = 4.0, n_pps: int = 4, base_speed: float = 70.0
) -> HorseEntry:
    today = date(2026, 5, 10)
    pps = [
        _pp(today - timedelta(days=30 * (i + 1)), base_speed + i, finish=(i % 8) + 1)
        for i in range(n_pps)
    ]
    return HorseEntry(
        horse_name=name,
        post_position=post,
        morning_line_odds=ml,
        jockey="Joe Jockey",
        trainer="Trainer A",
        owner="Owner LLC",
        weight_lbs=120.0,
        pp_lines=pps,
    )


def _race(race_number: int, n_horses: int = 5) -> ParsedRace:
    header = RaceHeader(
        race_number=race_number,
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
    # spread morning-line odds so EV calc has signal
    mls = [3.0, 4.0, 5.0, 8.0, 12.0][:n_horses]
    speeds = [85.0, 82.0, 80.0, 75.0, 70.0][:n_horses]
    entries = [
        _entry(i + 1, f"Horse R{race_number}P{i + 1}", ml=mls[i], base_speed=speeds[i])
        for i in range(n_horses)
    ]
    return ParsedRace(header=header, entries=entries, parse_confidence=0.9)


def _card() -> RaceCard:
    return RaceCard(
        source_filename="test.pdf",
        source_format="brisnet_up",
        total_pages=2,
        card_date=date(2026, 5, 10),
        track_code="CD",
        races=[_race(1), _race(2)],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mocked artifacts
# ──────────────────────────────────────────────────────────────────────────────


class _StubModel:
    """Returns a constant probability for every row."""

    def __init__(self, constant: float):
        self.constant = float(constant)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.constant, dtype=float)


class _RankBasedMetaLearner:
    """Returns scores monotonic in ewm_speed_prior so the calibrator
    sees a non-degenerate score distribution."""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Use the orthogonalised input column the real meta-learner consumes.
        # If 'speed_form_proba' isn't present, fall back to uniform.
        if "speed_form_proba" in df.columns:
            return np.clip(df["speed_form_proba"].to_numpy(), 0.0, 1.0)
        return np.full(len(df), 0.5, dtype=float)


def _fitted_calibrator() -> Calibrator:
    """Fit a Calibrator on synthetic well-calibrated data so predict_softmax
    works. We use auto with skip-when-calibrated; the input IS roughly
    calibrated so identity is likely chosen — that's fine for the test."""
    rng = np.random.default_rng(0)
    n = 4_000
    scores = rng.uniform(0.05, 0.95, size=n)
    labels = (rng.uniform(size=n) < scores).astype(int)
    cal = Calibrator(CalibratorConfig(method="auto"))
    cal.fit(scores, labels)
    return cal


def _make_artifacts(
    *,
    speed_form_const: float = 0.6,
    connections_const: float = 0.55,
    market_const: float = 0.6,
    with_speed_form: bool = True,
    with_connections: bool = True,
    with_market: bool = True,
) -> InferenceArtifacts:
    from app.services.models.pace_scenario import PaceScenarioModel
    from app.services.models.sequence import SequenceModel

    return InferenceArtifacts(
        speed_form=_StubModel(speed_form_const) if with_speed_form else None,
        pace_scenario=PaceScenarioModel(),
        sequence=SequenceModel(),
        connections=_StubModel(connections_const) if with_connections else None,
        market=_StubModel(market_const) if with_market else None,
        meta_learner=_RankBasedMetaLearner(),
        meta_calibrator=_fitted_calibrator(),
        models_dir=Path("models/none"),
        available_sub_models=("stub",),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests — feature builders
# ──────────────────────────────────────────────────────────────────────────────


def test_race_card_to_features_returns_one_row_per_horse():
    card = _card()
    fe_df = race_card_to_features(card)
    # 2 races × 5 horses = 10 rows
    assert len(fe_df) == 10
    assert "post_position" in fe_df.columns
    assert "ewm_speed_figure" in fe_df.columns


def test_build_inference_features_has_required_columns():
    card = _card()
    df = build_inference_features(card)
    for col in (
        "race_id",
        "ewm_speed_prior",
        "last_speed_prior",
        "n_prior_starts",
        "days_since_prev",
        "layoff_fitness",
        "field_size",
        "weight_lbs",
        "ewm_speed_zscore",
        "ewm_speed_rank",
        "weight_lbs_delta",
        "surface",
        "condition",
        "race_type",
        "jurisdiction",
    ):
        assert col in df.columns, f"missing column {col}"


def test_build_inference_features_race_ids_unique_per_race():
    card = _card()
    df = build_inference_features(card)
    race_ids = df.groupby("race_number")["race_id"].nunique()
    assert (race_ids == 1).all()


def test_build_inference_features_zscore_sums_to_zero_per_race():
    card = _card()
    df = build_inference_features(card)
    for _, g in df.groupby("race_id"):
        # ewm_speed_prior values differ per horse, so zscore should not all be
        # zero unless every horse has the same speed.
        z = g["ewm_speed_prior"].fillna(0)
        if z.std() > 0:
            zs = g["ewm_speed_zscore"]
            assert abs(zs.sum()) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Tests — inference scoring
# ──────────────────────────────────────────────────────────────────────────────


def test_infer_calibrated_win_probs_sums_to_one():
    card = _card()
    feats = build_inference_features(card)
    artifacts = _make_artifacts()
    race_id = feats["race_id"].iloc[0]
    one_race = feats[feats["race_id"] == race_id].sort_values("post_position").reset_index(drop=True)
    probs = infer_calibrated_win_probs(one_race, artifacts, race_id)
    assert probs.shape == (len(one_race),)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert (probs >= 0).all()


def test_infer_calibrated_win_probs_handles_empty_input():
    artifacts = _make_artifacts()
    empty = pd.DataFrame()
    out = infer_calibrated_win_probs(empty, artifacts, "race_id")
    assert out.shape == (0,)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — end-to-end orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def test_analyze_card_returns_per_race_prob_vectors():
    card = _card()
    artifacts = _make_artifacts()
    probs, candidates, portfolios = analyze_card(
        card, artifacts, bankroll=10_000.0, min_edge=0.0, optimize=False
    )
    assert len(probs) == 2  # two races on the card
    for _, p in probs.items():
        assert abs(float(p.sum()) - 1.0) < 1e-6
    assert isinstance(candidates, list)
    assert portfolios == []  # optimize=False


def test_analyze_card_with_optimize_builds_portfolios():
    card = _card()
    artifacts = _make_artifacts()
    _, candidates, portfolios = analyze_card(
        card,
        artifacts,
        bankroll=10_000.0,
        min_edge=0.0,
        optimize=True,
    )
    # Some races should produce at least one candidate at min_edge=0.0
    if candidates:
        assert len(portfolios) >= 1
        for p in portfolios:
            assert p.bankroll == 10_000.0
            assert 0.0 <= p.total_stake_fraction <= 1.0


def test_analyze_card_tolerates_missing_sub_models():
    """Per ADR-026: missing artifacts must fall back to 0.5, not crash."""
    card = _card()
    artifacts = _make_artifacts(
        with_speed_form=False, with_connections=False, with_market=False
    )
    probs, candidates, _ = analyze_card(
        card, artifacts, min_edge=0.0, optimize=False
    )
    assert len(probs) == 2
    # All sub-models constant → win probs should be uniform 1/N per race.
    for _, p in probs.items():
        assert np.allclose(p, p.mean(), atol=1e-6)


def test_analyze_card_caps_extreme_odds():
    """Per Phase 5b smoke-finding mitigation: max_decimal_odds = 100 by
    default. Inject a horse with ML odds = 500 and assert the resulting
    candidate (if any) has decimal_odds <= 100."""
    card = _card()
    # Inflate horse 5 of race 1 to a ridiculous ML — pydantic model_copy.
    card.races[0].entries[4] = card.races[0].entries[4].model_copy(
        update={"morning_line_odds": 500.0}
    )
    artifacts = _make_artifacts()
    _, candidates, _ = analyze_card(
        card, artifacts, min_edge=0.0, optimize=False, max_decimal_odds=100.0
    )
    for c in candidates:
        assert c.decimal_odds <= 100.0 + 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Tests — portfolio constructor
# ──────────────────────────────────────────────────────────────────────────────


def _make_candidate(
    race_id: str = "R1",
    horse_idx: int = 0,
    model_prob: float = 0.4,
    decimal_odds: float = 4.0,
    kelly: float = 0.02,
) -> BetCandidate:
    market_prob = 1.0 / decimal_odds
    edge = model_prob - market_prob
    return BetCandidate(
        race_id=race_id,
        bet_type=BetType.WIN,
        selection=(horse_idx,),
        model_prob=model_prob,
        decimal_odds=decimal_odds,
        market_prob=market_prob,
        edge=edge,
        expected_value=model_prob * decimal_odds - 1.0,
        kelly_fraction=kelly,
    )


def test_build_portfolio_empty_candidates():
    p = build_portfolio_from_candidates(
        card_id="cid",
        candidates=[],
        bankroll=10_000.0,
    )
    assert p.recommendations == []
    assert p.expected_return == 0.0


def test_build_portfolio_respects_bet_cap():
    # Two candidates with huge Kelly fractions should be capped at 3% each.
    cands = [
        _make_candidate("R1", 0, kelly=0.5),
        _make_candidate("R2", 1, kelly=0.5),
    ]
    p = build_portfolio_from_candidates(
        card_id="cid", candidates=cands, bankroll=10_000.0,
        n_scenarios=200,
    )
    for r in p.recommendations:
        assert r.stake_fraction <= 0.03 + 1e-9


def test_build_portfolio_total_stake_fraction_le_one():
    cands = [_make_candidate(f"R{i}", 0, kelly=0.02) for i in range(10)]
    p = build_portfolio_from_candidates(
        card_id="cid", candidates=cands, bankroll=10_000.0,
        n_scenarios=200,
    )
    assert p.total_stake_fraction <= 1.0
