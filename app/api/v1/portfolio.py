"""
app/api/v1/portfolio.py
───────────────────────
GET /api/v1/portfolio/{card_id} — return the +EV bet portfolio for
a persisted card.

Query parameters (all optional, all defaulted from
`app.services.inference.pipeline`):

    bankroll           default 10000.0      bankroll in USD
    min_edge           default 0.05         minimum edge to include
    max_decimal_odds   default 100.0        upper bound on live odds
    cvar_alpha         default 0.05         95th percentile shortfall
    max_drawdown_pct   default 0.20         per-day risk budget
    n_scenarios        default 1000         Monte-Carlo draws
    seed               default 42           RNG seed for reproducibility

Why GET (per ADR-042): inference is deterministic given (card,
artifacts, parameters). With the parameters in the query string the
response is bookmarkable / cacheable / inspectable in a browser. POST
would imply mutation which this endpoint does not perform.

Multi-race aggregation (per ADR-042): the frontend's bet-execution
ticket displays a single flat list of recommendations across the
whole card. We aggregate the per-race Portfolio objects from
`analyze_card` into a single response by:
  * concatenating `recommendations`,
  * summing `expected_return` and `total_stake_fraction` (capped at 1.0),
  * taking the worst-case VaR / CVaR across the per-race portfolios
    (most conservative for risk display).

Failure modes
─────────────
  404 — card_id not found in DB
  503 — `app.state.artifacts` is None (models not loaded at startup)
  502 — inference failed unexpectedly mid-card
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.persistence import load_card
from app.db.session import get_session
from app.schemas.bets import Portfolio
from app.services.inference.pipeline import (
    DEFAULT_BANKROLL,
    DEFAULT_CVAR_ALPHA,
    DEFAULT_MAX_DECIMAL_ODDS,
    DEFAULT_MAX_DRAWDOWN_PCT,
    DEFAULT_MIN_EDGE,
    DEFAULT_N_SCENARIOS,
    DEFAULT_SEED,
    InferenceArtifacts,
    analyze_card,
)
from app.api.v1.cards import _require_artifacts

router = APIRouter()
log = get_logger(__name__)


def _aggregate_portfolios(
    card_id: str,
    portfolios: list[Portfolio],
    bankroll: float,
) -> Portfolio:
    """Aggregate per-race Portfolio objects into a card-level Portfolio.

    See module docstring for the aggregation rules.
    """
    if not portfolios:
        return Portfolio(
            card_id=card_id,
            bankroll=bankroll,
            recommendations=[],
            expected_return=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_stake_fraction=0.0,
        )

    recommendations: list = []
    for p in portfolios:
        recommendations.extend(p.recommendations)

    total_stake_fraction = min(1.0, sum(p.total_stake_fraction for p in portfolios))
    expected_return = sum(p.expected_return for p in portfolios)
    # Conservative risk roll-up — worst across the per-race portfolios.
    var_95 = max(p.var_95 for p in portfolios)
    cvar_95 = max(p.cvar_95 for p in portfolios)

    return Portfolio(
        card_id=card_id,
        bankroll=bankroll,
        recommendations=recommendations,
        expected_return=float(expected_return),
        var_95=float(var_95),
        cvar_95=float(cvar_95),
        total_stake_fraction=float(total_stake_fraction),
    )


def _run_analyze_card_sync(
    card, artifacts: InferenceArtifacts, params: dict
) -> list[Portfolio]:
    """Synchronous wrapper invoked in a thread executor."""
    _, _, portfolios = analyze_card(card, artifacts, **params)
    return portfolios


@router.get(
    "/{card_id}", response_model=Portfolio, status_code=status.HTTP_200_OK
)
async def get_portfolio(
    card_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    bankroll: float = Query(DEFAULT_BANKROLL, gt=0.0),
    min_edge: float = Query(DEFAULT_MIN_EDGE, ge=0.0, le=1.0),
    max_decimal_odds: float = Query(DEFAULT_MAX_DECIMAL_ODDS, gt=1.0),
    cvar_alpha: float = Query(DEFAULT_CVAR_ALPHA, gt=0.0, lt=1.0),
    max_drawdown_pct: float = Query(DEFAULT_MAX_DRAWDOWN_PCT, gt=0.0, le=1.0),
    n_scenarios: int = Query(DEFAULT_N_SCENARIOS, ge=10, le=100_000),
    seed: int = Query(DEFAULT_SEED, ge=0),
) -> Portfolio:
    """Return the +EV Portfolio for the persisted card."""
    artifacts = _require_artifacts(request)

    try:
        card_pk = int(card_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"card_id {card_id!r} is not a valid integer",
        )

    card = await load_card(session, card_pk)
    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"card_id {card_id} not found",
        )

    params = dict(
        bankroll=bankroll,
        min_edge=min_edge,
        max_decimal_odds=max_decimal_odds,
        cvar_alpha=cvar_alpha,
        max_drawdown_pct=max_drawdown_pct,
        n_scenarios=n_scenarios,
        seed=seed,
        optimize=True,
        card_id=card_id,
    )

    loop = asyncio.get_running_loop()
    try:
        portfolios = await loop.run_in_executor(
            None, _run_analyze_card_sync, card, artifacts, params
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "portfolio.analyze_failed",
            error=str(exc),
            card_id=card_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"inference failed: {type(exc).__name__}: {exc}",
        )

    aggregated = _aggregate_portfolios(card_id, portfolios, bankroll)
    log.info(
        "portfolio.served",
        card_id=card_id,
        n_recommendations=len(aggregated.recommendations),
        total_stake_fraction=aggregated.total_stake_fraction,
        expected_return=aggregated.expected_return,
    )
    return aggregated


__all__ = ["router"]
