"""
app/schemas/bets.py
───────────────────
Phase 5 schemas for the EV engine and portfolio optimiser outputs.

A BetCandidate is what the EV engine emits per (race, bet_type, selection).
A BetRecommendation is what the portfolio optimiser (Phase 5b) emits — a
candidate plus the final stake. A Portfolio is the full card-level output.

Scope (ADR-039): Phase 5a supports WIN, EXACTA, TRIFECTA, SUPERFECTA.
Place/Show require live pari-mutuel pool composition; Pick 3/4/6 require
cross-race correlation modelling. Both deferred.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.race import BetType


_SUPPORTED_BET_TYPES_5A: dict[BetType, int] = {
    BetType.WIN: 1,
    BetType.EXACTA: 2,
    BetType.TRIFECTA: 3,
    BetType.SUPERFECTA: 4,
}


class BetCandidate(BaseModel):
    """A potential bet identified by the EV engine, with its computed metrics."""

    race_id: str
    bet_type: BetType
    selection: tuple[int, ...]

    model_prob: Annotated[float, Field(ge=0.0, le=1.0)]
    decimal_odds: Annotated[float, Field(ge=1.0)]
    market_prob: Annotated[float, Field(ge=0.0, le=1.0)]
    edge: float
    expected_value: float
    kelly_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    market_impact_applied: bool = False
    pool_size: Optional[Annotated[float, Field(ge=0.0)]] = None

    @model_validator(mode="after")
    def validate_selection(self) -> "BetCandidate":
        if self.bet_type not in _SUPPORTED_BET_TYPES_5A:
            raise ValueError(
                f"bet_type {self.bet_type} not supported in Phase 5a "
                f"(ADR-039). Supported: {list(_SUPPORTED_BET_TYPES_5A)}"
            )
        expected_len = _SUPPORTED_BET_TYPES_5A[self.bet_type]
        if len(self.selection) != expected_len:
            raise ValueError(
                f"selection length {len(self.selection)} != expected "
                f"{expected_len} for {self.bet_type}"
            )
        if len(set(self.selection)) != len(self.selection):
            raise ValueError(f"selection indices must be distinct; got {self.selection}")
        for idx in self.selection:
            if idx < 0:
                raise ValueError(f"selection indices must be non-negative; got {idx}")
        return self


class BetRecommendation(BaseModel):
    """A bet selected by the portfolio optimiser for placement.

    Callers (Phase 5b portfolio optimiser) are responsible for keeping
    `stake` and `stake_fraction` consistent — the schema cannot enforce
    `stake == stake_fraction * bankroll` because `bankroll` lives on
    `Portfolio`, not on the recommendation.
    """

    candidate: BetCandidate
    stake: Annotated[float, Field(ge=0.0)]
    stake_fraction: Annotated[float, Field(ge=0.0, le=1.0)]


class Portfolio(BaseModel):
    """All recommendations + risk metrics for one card."""

    card_id: str
    bankroll: Annotated[float, Field(gt=0.0)]
    recommendations: list[BetRecommendation]
    expected_return: float
    var_95: float
    cvar_95: float
    total_stake_fraction: Annotated[float, Field(ge=0.0, le=1.0)]


__all__ = ["BetCandidate", "BetRecommendation", "Portfolio"]
