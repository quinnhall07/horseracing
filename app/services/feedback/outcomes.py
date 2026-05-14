"""
app/services/feedback/outcomes.py
─────────────────────────────────
Phase 5b / Layer 7 — outcomes logging + bet settlement.

The feedback loop has three stages:

    1.  log_race_outcome      — persist what actually happened (race chart).
    2.  settle_bets           — score each emitted BetRecommendation against
                                the outcome and write a BetSettlement row.
    3.  get_settled_pnl_series — pull a time-ordered DataFrame of (expected,
                                realised) pairs to feed the portfolio drift
                                CUSUM in `portfolio_drift.py`.

PnL convention (ADR-043):
    pnl = payout − stake
where `payout` is the gross dollar payout (0 if the bet lost, else
`stake × decimal_odds_at_settlement`). So a losing $1 bet has pnl = −1;
a winning $1 bet at 4.0 decimal has pnl = +3.

Settlement scope (ADR-039):
    Only WIN / EXACTA / TRIFECTA / SUPERFECTA are settled — these are the
    bet types the EV engine emits in Phase 5a. PLACE / SHOW / PICK-n raise
    ValueError so a future caller cannot silently get wrong settlements.

Selection-to-outcome matching:
    `BetCandidate.selection` is a tuple of integer identifiers chosen by the
    EV engine. By convention in Phase 5a, those integers are program numbers
    (1-indexed saddle-cloth numbers). `RaceOutcome.finishing_order` is also
    a list of program numbers in finish order. Therefore settlement reduces
    to a positional prefix-match on the finishing order:

        WIN        — selection[0] == finishing_order[0]
        EXACTA     — selection == finishing_order[:2]
        TRIFECTA   — selection == finishing_order[:3]
        SUPERFECTA — selection == finishing_order[:4]

    If the EV engine ever emits row-indices instead of program numbers,
    the caller must translate before invoking settle_bets — keeping the
    settlement function I/O-free of any RaceCard introspection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import BetSettlement, RaceOutcome
from app.schemas.bets import BetRecommendation
from app.schemas.race import BetType

log = get_logger(__name__)


# Phase 5a / ADR-039 scope. Anything outside this set raises in settle_bets.
SETTLEABLE_BET_TYPES: frozenset[BetType] = frozenset({
    BetType.WIN,
    BetType.EXACTA,
    BetType.TRIFECTA,
    BetType.SUPERFECTA,
})


_BET_TYPE_TO_PREFIX_LEN: dict[BetType, int] = {
    BetType.WIN: 1,
    BetType.EXACTA: 2,
    BetType.TRIFECTA: 3,
    BetType.SUPERFECTA: 4,
}


# ── Outcome logging ────────────────────────────────────────────────────────


async def log_race_outcome(
    session: AsyncSession,
    *,
    race_id: str,
    race_dedup_key: str,
    finishing_order: list[int],
    payouts: dict[str, float],
    source: str = "manual",
) -> RaceOutcome:
    """Insert a race outcome record. Idempotent on (race_dedup_key).

    If a row already exists for this race_dedup_key, returns it unchanged —
    the second caller does NOT overwrite payouts or finishing_order. This
    matches the Phase 0 dedup contract: "running twice on the same source
    produces the same DB state".

    Validation:
      * finishing_order must have at least one entry (the winner).
      * finishing_order entries must be distinct (a horse cannot finish
        in two positions); duplicates raise ValueError.
      * place_horses is derived from finishing_order[1:3] padded with -1 to
        length 2 when the race had fewer than three finishers (e.g. a
        scratch-heavy 2-horse race).
    """
    if not finishing_order:
        raise ValueError("finishing_order must contain at least the winner")
    if len(set(finishing_order)) != len(finishing_order):
        raise ValueError(f"finishing_order contains duplicates: {finishing_order}")
    if not isinstance(payouts, dict):
        raise ValueError(f"payouts must be a dict[str, float]; got {type(payouts)}")

    # Idempotency check: look up by race_dedup_key, return existing row if any.
    existing = await session.execute(
        select(RaceOutcome).where(RaceOutcome.race_dedup_key == race_dedup_key)
    )
    row = existing.scalar_one_or_none()
    if row is not None:
        log.info(
            "feedback.outcome.dedup_skip",
            race_id=race_id,
            race_dedup_key=race_dedup_key,
            existing_id=row.id,
        )
        return row

    winner = int(finishing_order[0])
    place_horses: list[int] = [int(x) for x in finishing_order[1:3]]
    while len(place_horses) < 2:
        place_horses.append(-1)  # padding sentinel: not enough finishers.

    outcome = RaceOutcome(
        race_id=race_id,
        race_dedup_key=race_dedup_key,
        finishing_order=[int(x) for x in finishing_order],
        winning_horse_program_number=winner,
        place_horses=place_horses,
        payouts={str(k): float(v) for k, v in payouts.items()},
        source=source,
    )
    session.add(outcome)
    await session.flush()
    log.info(
        "feedback.outcome.logged",
        race_id=race_id,
        race_dedup_key=race_dedup_key,
        winner=winner,
        source=source,
        n_finishers=len(finishing_order),
    )
    return outcome


# ── Bet settlement ─────────────────────────────────────────────────────────


def _selection_matches_outcome(
    bet_type: BetType,
    selection: tuple[int, ...],
    finishing_order: list[int],
) -> bool:
    """Return True iff the recommendation's selection wins under bet_type."""
    if bet_type not in _BET_TYPE_TO_PREFIX_LEN:
        raise ValueError(
            f"bet_type {bet_type!r} is outside Phase 5a scope (ADR-039); "
            f"supported: {sorted(SETTLEABLE_BET_TYPES)}"
        )
    prefix_len = _BET_TYPE_TO_PREFIX_LEN[bet_type]
    if len(finishing_order) < prefix_len:
        # Race has too few finishers for this exotic; bet cannot win.
        return False
    return tuple(int(x) for x in selection) == tuple(
        int(x) for x in finishing_order[:prefix_len]
    )


async def settle_bets(
    session: AsyncSession,
    *,
    race_id: str,
    recommendations: Iterable[BetRecommendation],
    outcome: RaceOutcome,
    decimal_odds_at_settlement: dict[int, float] | None = None,
) -> list[BetSettlement]:
    """Resolve a sequence of `BetRecommendation`s against a `RaceOutcome`.

    Per ADR-043:
        payout_i = stake_i × decimal_odds_at_settlement_i  if won
                 = 0                                       otherwise
        pnl_i    = payout_i − stake_i

    The settlement decimal-odds default to the recommendation-time odds
    (which is the only realistic choice when we don't yet capture
    post-time / closing tote). `decimal_odds_at_settlement` is an optional
    per-recommendation override keyed by the recommendation's index in the
    input iterable — present for future drift-decomposition use.

    Raises ValueError if any recommendation uses a non-Phase-5a bet type.
    """
    recs = list(recommendations)
    settlements: list[BetSettlement] = []

    finishing_order = [int(x) for x in outcome.finishing_order]

    for idx, rec in enumerate(recs):
        cand = rec.candidate
        bet_type = cand.bet_type

        # Reject out-of-scope bet types BEFORE any DB mutation so a bad input
        # doesn't half-settle the batch.
        if bet_type not in SETTLEABLE_BET_TYPES:
            raise ValueError(
                f"bet_type {bet_type!r} not settleable in Phase 5a "
                f"(ADR-039). Supported: {sorted(SETTLEABLE_BET_TYPES)}"
            )

        won = _selection_matches_outcome(bet_type, cand.selection, finishing_order)
        odds_settle = (
            decimal_odds_at_settlement.get(idx, cand.decimal_odds)
            if decimal_odds_at_settlement is not None
            else cand.decimal_odds
        )
        payout = float(rec.stake * odds_settle) if won else 0.0
        pnl = payout - float(rec.stake)

        settlement = BetSettlement(
            race_id=race_id,
            bet_type=str(bet_type.value),
            selection=[int(x) for x in cand.selection],
            stake=float(rec.stake),
            stake_fraction=float(rec.stake_fraction),
            decimal_odds_at_recommendation=float(cand.decimal_odds),
            decimal_odds_at_settlement=float(odds_settle),
            model_prob=float(cand.model_prob),
            expected_value=float(cand.expected_value),
            won=won,
            payout=payout,
            pnl=pnl,
        )
        session.add(settlement)
        settlements.append(settlement)

    await session.flush()
    log.info(
        "feedback.bets.settled",
        race_id=race_id,
        n_recs=len(recs),
        n_wins=sum(1 for s in settlements if s.won),
        total_pnl=float(sum(s.pnl for s in settlements)),
    )
    return settlements


# ── PnL series accessor ────────────────────────────────────────────────────


async def get_settled_pnl_series(
    session: AsyncSession,
    *,
    since: datetime | None = None,
) -> pd.DataFrame:
    """Return time-ordered settled-bet PnL stream for portfolio drift.

    Columns: [settled_at, race_id, bet_type, expected_value, pnl, payout,
              stake, model_prob, decimal_odds_at_recommendation, won].

    Sorted by settled_at ascending. When the table is empty returns an
    empty DataFrame with the expected columns (so downstream consumers
    can rely on schema invariants without explicit length-checks).
    """
    stmt = select(BetSettlement).order_by(BetSettlement.settled_at.asc())
    if since is not None:
        stmt = stmt.where(BetSettlement.settled_at >= since)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    columns = [
        "settled_at", "race_id", "bet_type", "expected_value", "pnl",
        "payout", "stake", "model_prob", "decimal_odds_at_recommendation",
        "won",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(
        [
            {
                "settled_at": r.settled_at,
                "race_id": r.race_id,
                "bet_type": r.bet_type,
                "expected_value": r.expected_value,
                "pnl": r.pnl,
                "payout": r.payout,
                "stake": r.stake,
                "model_prob": r.model_prob,
                "decimal_odds_at_recommendation": r.decimal_odds_at_recommendation,
                "won": r.won,
            }
            for r in rows
        ],
        columns=columns,
    )
    return df


__all__ = [
    "SETTLEABLE_BET_TYPES",
    "log_race_outcome",
    "settle_bets",
    "get_settled_pnl_series",
]
