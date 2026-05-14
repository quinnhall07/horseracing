"""Tests for app/services/feedback/outcomes.py.

Stream B / Layer 7 feedback-loop persistence. Locks in:
    (a) log_race_outcome is idempotent on race_dedup_key.
    (b) settle_bets correctly settles WIN bets (win/lose pnl).
    (c) settle_bets correctly settles EXACTA bets (positional prefix match).
    (d) settle_bets correctly settles TRIFECTA and SUPERFECTA.
    (e) settle_bets rejects PLACE/SHOW/PICKN bets (ADR-039).
    (f) get_settled_pnl_series returns expected DataFrame shape + sort order.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db.models import RaceOutcome, BetSettlement
from app.db.session import Base
from app.schemas.bets import BetCandidate, BetRecommendation
from app.schemas.race import BetType
from app.services.feedback.outcomes import (
    SETTLEABLE_BET_TYPES,
    get_settled_pnl_series,
    log_race_outcome,
    settle_bets,
)


pytestmark = pytest.mark.asyncio


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def session():
    """A fresh in-memory aiosqlite DB per test, with Base.metadata.create_all."""
    # Unique tmpfile-style URL per test isn't needed for async :memory: as long
    # as we keep a single connection alive. We use NullPool semantics by
    # creating a single shared engine with poolclass StaticPool? — simpler to
    # just use a file-backed unique URL like the API tests.
    import os
    import tempfile

    fd, path = tempfile.mkstemp(prefix="feedback_outcomes_", suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", future=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as s:
        yield s

    await engine.dispose()
    try:
        os.remove(path)
    except OSError:
        pass


def _make_candidate(
    *,
    race_id: str = "race-1",
    bet_type: BetType = BetType.WIN,
    selection: tuple[int, ...] = (1,),
    model_prob: float = 0.25,
    decimal_odds: float = 4.0,
    market_prob: float = 0.20,
    edge: float = 0.05,
    expected_value: float = 0.0,
    kelly_fraction: float = 0.05,
) -> BetCandidate:
    return BetCandidate(
        race_id=race_id,
        bet_type=bet_type,
        selection=selection,
        model_prob=model_prob,
        decimal_odds=decimal_odds,
        market_prob=market_prob,
        edge=edge,
        expected_value=expected_value,
        kelly_fraction=kelly_fraction,
    )


def _make_rec(cand: BetCandidate, *, stake: float = 1.0, stake_fraction: float = 0.01):
    return BetRecommendation(candidate=cand, stake=stake, stake_fraction=stake_fraction)


# ── log_race_outcome idempotency ───────────────────────────────────────────


async def test_log_race_outcome_inserts_row(session: AsyncSession):
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dedup-abc",
        finishing_order=[1, 5, 3, 2],
        payouts={"win_2.00": 8.40, "exacta_1_5": 47.20},
        source="manual",
    )
    await session.commit()

    assert outcome.id is not None
    assert outcome.winning_horse_program_number == 1
    assert outcome.place_horses == [5, 3]
    assert outcome.finishing_order == [1, 5, 3, 2]
    assert outcome.source == "manual"


async def test_log_race_outcome_is_idempotent_on_race_dedup_key(session: AsyncSession):
    """Two calls with same race_dedup_key → same row, no duplicate insert."""
    first = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dedup-abc",
        finishing_order=[1, 5, 3],
        payouts={"win_2.00": 8.40},
    )
    await session.commit()

    # Even with DIFFERENT finishing_order and payouts, the existing row wins.
    second = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dedup-abc",
        finishing_order=[2, 4, 1],
        payouts={"win_2.00": 99.99},
    )
    await session.commit()

    assert second.id == first.id
    # Original payouts are preserved (no overwrite).
    assert second.payouts == {"win_2.00": 8.40}
    assert second.finishing_order == [1, 5, 3]

    # And the table really has just one row.
    n = (await session.execute(select(RaceOutcome))).scalars().all()
    assert len(n) == 1


async def test_log_race_outcome_rejects_empty_finishing_order(session: AsyncSession):
    with pytest.raises(ValueError):
        await log_race_outcome(
            session,
            race_id="race-1",
            race_dedup_key="dedup-x",
            finishing_order=[],
            payouts={},
        )


async def test_log_race_outcome_rejects_duplicate_program_numbers(session: AsyncSession):
    with pytest.raises(ValueError):
        await log_race_outcome(
            session,
            race_id="race-1",
            race_dedup_key="dedup-x",
            finishing_order=[1, 1, 3],
            payouts={},
        )


async def test_log_race_outcome_pads_place_horses_for_short_field(session: AsyncSession):
    """Race with only 2 finishers → place_horses[1] is the -1 sentinel."""
    outcome = await log_race_outcome(
        session,
        race_id="race-2",
        race_dedup_key="dedup-2",
        finishing_order=[7, 4],
        payouts={"win_2.00": 5.20},
    )
    assert outcome.place_horses == [4, -1]


# ── settle_bets — WIN ──────────────────────────────────────────────────────


async def test_settle_win_bet_winner_matches(session: AsyncSession):
    """stake=1, decimal_odds=4, selection (3,) matches winner 3 → pnl = +3."""
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-win-hit",
        finishing_order=[3, 5, 1],
        payouts={},
    )
    cand = _make_candidate(selection=(3,), decimal_odds=4.0)
    rec = _make_rec(cand, stake=1.0)

    settlements = await settle_bets(
        session, race_id="race-1", recommendations=[rec], outcome=outcome,
    )
    await session.commit()

    assert len(settlements) == 1
    s = settlements[0]
    assert s.won is True
    assert s.payout == pytest.approx(4.0)
    assert s.pnl == pytest.approx(3.0)


async def test_settle_win_bet_loser(session: AsyncSession):
    """stake=1, selection (7,) does NOT match winner 3 → pnl = -1."""
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-win-miss",
        finishing_order=[3, 5, 1],
        payouts={},
    )
    cand = _make_candidate(selection=(7,), decimal_odds=4.0)
    rec = _make_rec(cand, stake=1.0)

    settlements = await settle_bets(
        session, race_id="race-1", recommendations=[rec], outcome=outcome,
    )
    s = settlements[0]
    assert s.won is False
    assert s.payout == pytest.approx(0.0)
    assert s.pnl == pytest.approx(-1.0)


# ── settle_bets — EXACTA / TRIFECTA / SUPERFECTA ───────────────────────────


async def test_settle_exacta_winner(session: AsyncSession):
    """stake=1, odds=8, selection (2,5) matches top-2 (2,5,...) → pnl=+7."""
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-exa-hit",
        finishing_order=[2, 5, 3, 4, 1],
        payouts={},
    )
    cand = _make_candidate(
        bet_type=BetType.EXACTA, selection=(2, 5), decimal_odds=8.0,
        model_prob=0.10,
    )
    rec = _make_rec(cand, stake=1.0)

    settlements = await settle_bets(
        session, race_id="race-1", recommendations=[rec], outcome=outcome,
    )
    s = settlements[0]
    assert s.won is True
    assert s.pnl == pytest.approx(7.0)


async def test_settle_exacta_loser_when_order_wrong(session: AsyncSession):
    """EXACTA must match order. (5,2) does NOT win when finish is (2,5,...)."""
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-exa-order",
        finishing_order=[2, 5, 3],
        payouts={},
    )
    cand = _make_candidate(
        bet_type=BetType.EXACTA, selection=(5, 2), decimal_odds=8.0,
        model_prob=0.10,
    )
    rec = _make_rec(cand, stake=1.0)
    settlements = await settle_bets(
        session, race_id="race-1", recommendations=[rec], outcome=outcome,
    )
    s = settlements[0]
    assert s.won is False
    assert s.pnl == pytest.approx(-1.0)


async def test_settle_trifecta_winner(session: AsyncSession):
    """TRI hit when top-3 program-numbers match in order."""
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-tri-hit",
        finishing_order=[1, 4, 2, 7],
        payouts={},
    )
    cand = _make_candidate(
        bet_type=BetType.TRIFECTA, selection=(1, 4, 2), decimal_odds=20.0,
        model_prob=0.03,
    )
    rec = _make_rec(cand, stake=2.0)
    settlements = await settle_bets(
        session, race_id="race-1", recommendations=[rec], outcome=outcome,
    )
    s = settlements[0]
    assert s.won is True
    assert s.pnl == pytest.approx(38.0)  # 2 * 20 - 2


async def test_settle_superfecta_winner(session: AsyncSession):
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-sup-hit",
        finishing_order=[3, 1, 7, 4, 2],
        payouts={},
    )
    cand = _make_candidate(
        bet_type=BetType.SUPERFECTA, selection=(3, 1, 7, 4), decimal_odds=100.0,
        model_prob=0.005,
    )
    rec = _make_rec(cand, stake=1.0)
    settlements = await settle_bets(
        session, race_id="race-1", recommendations=[rec], outcome=outcome,
    )
    s = settlements[0]
    assert s.won is True
    assert s.pnl == pytest.approx(99.0)


async def test_settle_exotic_when_too_few_finishers(session: AsyncSession):
    """TRI bet on a 2-horse race must lose (cannot match length-3 prefix)."""
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-short-field",
        finishing_order=[1, 2],
        payouts={},
    )
    cand = _make_candidate(
        bet_type=BetType.TRIFECTA, selection=(1, 2, 3), decimal_odds=15.0,
        model_prob=0.04,
    )
    rec = _make_rec(cand, stake=1.0)
    settlements = await settle_bets(
        session, race_id="race-1", recommendations=[rec], outcome=outcome,
    )
    s = settlements[0]
    assert s.won is False


# ── settle_bets — out-of-scope bet types reject ────────────────────────────


@pytest.mark.parametrize("bet_type", [
    BetType.PLACE,
    BetType.SHOW,
    BetType.PICK3,
    BetType.PICK4,
    BetType.PICK6,
])
async def test_settle_rejects_non_phase5a_bet_types(session: AsyncSession, bet_type: BetType):
    """Phase 5a / ADR-039 scope: only WIN/EXACTA/TRIFECTA/SUPERFECTA settle."""
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key=f"dk-reject-{bet_type.value}",
        finishing_order=[1, 2, 3, 4],
        payouts={},
    )
    # BetCandidate itself rejects these via its model_validator. We confirm
    # that — proving the rejection happens before settle_bets ever runs.
    with pytest.raises(ValueError):
        _make_candidate(
            bet_type=bet_type, selection=(1,), decimal_odds=3.0,
        )


async def test_settle_rejects_unsettleable_via_constructed_recommendation(session: AsyncSession):
    """Belt-and-braces: even if a BetCandidate somehow exists with a
    non-Phase-5a type (e.g. via .model_construct), settle_bets still rejects.
    """
    outcome = await log_race_outcome(
        session,
        race_id="race-1",
        race_dedup_key="dk-bypass",
        finishing_order=[1, 2, 3],
        payouts={},
    )
    # Bypass the Pydantic validator so we can exercise the settler's guard.
    bad = BetCandidate.model_construct(
        race_id="race-1",
        bet_type=BetType.PLACE,
        selection=(1,),
        model_prob=0.3,
        decimal_odds=3.0,
        market_prob=0.25,
        edge=0.05,
        expected_value=0.0,
        kelly_fraction=0.02,
        market_impact_applied=False,
        pool_size=None,
    )
    rec = BetRecommendation.model_construct(candidate=bad, stake=1.0, stake_fraction=0.01)
    with pytest.raises(ValueError, match="not settleable"):
        await settle_bets(
            session, race_id="race-1", recommendations=[rec], outcome=outcome,
        )


# ── get_settled_pnl_series ─────────────────────────────────────────────────


async def test_get_settled_pnl_series_empty(session: AsyncSession):
    df = await get_settled_pnl_series(session)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 0
    expected_cols = {
        "settled_at", "race_id", "bet_type", "expected_value", "pnl",
        "payout", "stake", "model_prob", "decimal_odds_at_recommendation",
        "won",
    }
    assert expected_cols.issubset(set(df.columns))


async def test_get_settled_pnl_series_returns_time_sorted_df(session: AsyncSession):
    """Settled bets across two races → DataFrame sorted by settled_at asc."""
    outcome1 = await log_race_outcome(
        session, race_id="r1", race_dedup_key="dk1",
        finishing_order=[1, 2, 3], payouts={},
    )
    cand1 = _make_candidate(race_id="r1", selection=(1,), decimal_odds=4.0)
    rec1 = _make_rec(cand1, stake=2.0)
    await settle_bets(session, race_id="r1", recommendations=[rec1], outcome=outcome1)
    await session.commit()

    outcome2 = await log_race_outcome(
        session, race_id="r2", race_dedup_key="dk2",
        finishing_order=[9, 8, 7], payouts={},
    )
    cand2 = _make_candidate(race_id="r2", selection=(5,), decimal_odds=10.0)
    rec2 = _make_rec(cand2, stake=1.0)
    await settle_bets(session, race_id="r2", recommendations=[rec2], outcome=outcome2)
    await session.commit()

    df = await get_settled_pnl_series(session)
    assert df.shape[0] == 2
    # Sort order: r1 (win, pnl=+6) came first, then r2 (loss, pnl=-1).
    assert list(df["race_id"]) == ["r1", "r2"]
    assert df["settled_at"].is_monotonic_increasing
    assert df.loc[0, "pnl"] == pytest.approx(6.0)
    assert df.loc[1, "pnl"] == pytest.approx(-1.0)
    assert bool(df.loc[0, "won"]) is True
    assert bool(df.loc[1, "won"]) is False


async def test_settleable_bet_types_constant_matches_adr_039():
    assert SETTLEABLE_BET_TYPES == frozenset({
        BetType.WIN, BetType.EXACTA, BetType.TRIFECTA, BetType.SUPERFECTA,
    })
