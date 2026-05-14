"""
app/db/models.py
────────────────
SQLAlchemy ORM models for the live ingestion DB.

Hierarchy mirrors the Pydantic schemas in `app/schemas/race.py`:

    IngestedCard ── 1:N ── IngestedRace ── 1:N ── IngestedHorse ── 1:N ── IngestedPPLine

Each row has a system primary key (`id`) plus a logical key that lets us
look up cards by source filename or by (track, date, race#). Cascading
deletes are enabled so `DELETE FROM ingested_cards WHERE id = ?` cleanly
removes the entire tree.

JSON columns are used for variable-length fields (medication_flags,
equipment_changes, parse_warnings) — they're small enough that a JSON blob
is cheaper than a join table for the volumes the live DB will see (single-
digit cards per day per user).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class IngestedCard(Base):
    """One uploaded PDF → one card row."""

    __tablename__ = "ingested_cards"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    source_format: Mapped[str] = mapped_column(String(32), default="unknown")
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    card_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    track_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    parse_confidence_avg: Mapped[float] = mapped_column(Float, default=0.0)
    n_races: Mapped[int] = mapped_column(Integer, default=0)
    n_qualified_races: Mapped[int] = mapped_column(Integer, default=0)
    processing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    parsed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    races: Mapped[list["IngestedRace"]] = relationship(
        back_populates="card",
        cascade="all, delete-orphan",
        order_by="IngestedRace.race_number",
    )


class IngestedRace(Base):
    """One race within a card."""

    __tablename__ = "ingested_races"
    __table_args__ = (
        UniqueConstraint("card_id", "race_number", name="uq_ingested_race"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    card_id: Mapped[int] = mapped_column(
        ForeignKey("ingested_cards.id", ondelete="CASCADE"), nullable=False, index=True
    )
    race_number: Mapped[int] = mapped_column(Integer, nullable=False)
    race_date: Mapped[date] = mapped_column(Date, nullable=False)
    track_code: Mapped[str] = mapped_column(String(8), nullable=False)
    track_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    race_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    distance_furlongs: Mapped[float] = mapped_column(Float, nullable=False)
    distance_raw: Mapped[str] = mapped_column(String(64), nullable=False)
    surface: Mapped[str] = mapped_column(String(16), default="unknown")
    condition: Mapped[str] = mapped_column(String(16), default="unknown")
    race_type: Mapped[str] = mapped_column(String(32), default="unknown")

    claiming_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    purse_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade: Mapped[int | None] = mapped_column(Integer, nullable=True)

    age_sex_restrictions: Mapped[str | None] = mapped_column(String(128), nullable=True)
    weight_conditions: Mapped[str | None] = mapped_column(String(255), nullable=True)
    post_time: Mapped[str | None] = mapped_column(String(16), nullable=True)
    weather: Mapped[str | None] = mapped_column(String(64), nullable=True)

    parse_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    parse_warnings: Mapped[list] = mapped_column(JSON, default=list)

    card: Mapped[IngestedCard] = relationship(back_populates="races")
    horses: Mapped[list["IngestedHorse"]] = relationship(
        back_populates="race",
        cascade="all, delete-orphan",
        order_by="IngestedHorse.post_position",
    )


class IngestedHorse(Base):
    """One horse entered in a race."""

    __tablename__ = "ingested_horses"
    __table_args__ = (
        UniqueConstraint("race_id", "post_position", name="uq_ingested_horse"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(
        ForeignKey("ingested_races.id", ondelete="CASCADE"), nullable=False, index=True
    )

    horse_name: Mapped[str] = mapped_column(String(128), nullable=False)
    horse_id_external: Mapped[str | None] = mapped_column(String(64), nullable=True)
    post_position: Mapped[int] = mapped_column(Integer, nullable=False)

    morning_line_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_implied_prob: Mapped[float | None] = mapped_column(Float, nullable=True)

    jockey: Mapped[str | None] = mapped_column(String(128), nullable=True)
    trainer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    weight_lbs: Mapped[float | None] = mapped_column(Float, nullable=True)

    medication_flags: Mapped[list] = mapped_column(JSON, default=list)
    equipment_changes: Mapped[list] = mapped_column(JSON, default=list)

    pace_style: Mapped[str] = mapped_column(String(16), default="unknown")
    ewm_speed_figure: Mapped[float | None] = mapped_column(Float, nullable=True)
    days_since_last: Mapped[int | None] = mapped_column(Integer, nullable=True)
    class_trajectory: Mapped[float | None] = mapped_column(Float, nullable=True)

    race: Mapped[IngestedRace] = relationship(back_populates="horses")
    pp_lines: Mapped[list["IngestedPPLine"]] = relationship(
        back_populates="horse",
        cascade="all, delete-orphan",
        order_by="IngestedPPLine.race_date.desc()",
    )


class IngestedPPLine(Base):
    """One past performance line for a horse in today's race."""

    __tablename__ = "ingested_pp_lines"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    horse_id: Mapped[int] = mapped_column(
        ForeignKey("ingested_horses.id", ondelete="CASCADE"), nullable=False, index=True
    )

    race_date: Mapped[date] = mapped_column(Date, nullable=False)
    track_code: Mapped[str] = mapped_column(String(8), nullable=False)
    race_number: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_furlongs: Mapped[float] = mapped_column(Float, nullable=False)
    surface: Mapped[str] = mapped_column(String(16), default="unknown")
    condition: Mapped[str] = mapped_column(String(16), default="unknown")
    race_type: Mapped[str] = mapped_column(String(32), default="unknown")

    claiming_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    purse_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    post_position: Mapped[int] = mapped_column(Integer, nullable=False)
    finish_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lengths_behind: Mapped[float | None] = mapped_column(Float, nullable=True)
    field_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    jockey: Mapped[str | None] = mapped_column(String(128), nullable=True)
    weight_lbs: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_final: Mapped[float | None] = mapped_column(Float, nullable=True)

    speed_figure: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_figure_source: Mapped[str] = mapped_column(String(16), default="unknown")

    fraction_q1: Mapped[float | None] = mapped_column(Float, nullable=True)
    fraction_q2: Mapped[float | None] = mapped_column(Float, nullable=True)
    fraction_finish: Mapped[float | None] = mapped_column(Float, nullable=True)

    beaten_lengths_q1: Mapped[float | None] = mapped_column(Float, nullable=True)
    beaten_lengths_q2: Mapped[float | None] = mapped_column(Float, nullable=True)

    days_since_prev: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(String(512), nullable=True)

    horse: Mapped[IngestedHorse] = relationship(back_populates="pp_lines")


# ── Phase 5b/Layer 7 — Feedback loop tables ────────────────────────────────


class RaceOutcome(Base):
    """One settled race outcome (Stream B feedback loop, Layer 7).

    Persists official finishing order + payout payouts for a race so that
    recommendations from the EV engine can be settled against ground truth.

    Idempotency: `race_dedup_key` is the canonical idempotency anchor — it
    aligns with the Phase 0 master-DB dedup convention so that the same race
    settled by two sources (manual entry + official chart) collapses to one
    row. `race_id` is the inference-time identifier (e.g. the IngestedRace
    PK or a composite key from the live API); we keep it as a denormalised
    convenience for joins from BetSettlement.
    """

    __tablename__ = "race_outcomes"
    __table_args__ = (
        UniqueConstraint("race_dedup_key", name="uq_race_outcome_dedup_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    race_dedup_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Program-numbers in finish order; index 0 is the winner.
    finishing_order: Mapped[list] = mapped_column(JSON, nullable=False)
    winning_horse_program_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # [2nd, 3rd] — length 2; broken out for fast lookup on Place/Show queries.
    place_horses: Mapped[list] = mapped_column(JSON, nullable=False)
    # Free-form payout dict keyed by bet token (e.g. "win_2.00", "exacta_2_5").
    payouts: Mapped[dict] = mapped_column(JSON, nullable=False)
    settled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")


class BetSettlement(Base):
    """One settled recommendation — the realised outcome of an emitted bet.

    Captures both the prediction-time state (model_prob, EV, recommendation-
    time decimal odds) and the settlement-time state (won, payout, pnl,
    optional settlement-time decimal odds). Two snapshots are intentionally
    redundant: comparing `decimal_odds_at_recommendation` against
    `decimal_odds_at_settlement` lets us decompose realised drift into
    "model error" vs. "market move" components.

    pnl convention (see ADR-043): pnl = payout - stake. Therefore a losing
    bet's pnl == -stake (always negative). A winning bet's pnl ==
    stake * (decimal_odds_at_settlement - 1).
    """

    __tablename__ = "bet_settlements"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    bet_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # 0-indexed positions used at recommendation time (NOT program numbers —
    # the EV engine works in row-index space; the settlement function maps
    # those to program numbers via the recommendation's selection contract).
    selection: Mapped[list] = mapped_column(JSON, nullable=False)
    stake: Mapped[float] = mapped_column(Float, nullable=False)
    stake_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    decimal_odds_at_recommendation: Mapped[float] = mapped_column(Float, nullable=False)
    decimal_odds_at_settlement: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_prob: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    won: Mapped[bool] = mapped_column(Boolean, nullable=False)
    payout: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False)
    settled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Future-proof FK slot for an explicit BetRecommendation table — for now
    # nullable, so feedback can be logged before that table exists.
    bet_recommendation_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


__all__ = [
    "IngestedCard",
    "IngestedRace",
    "IngestedHorse",
    "IngestedPPLine",
    "RaceOutcome",
    "BetSettlement",
]
