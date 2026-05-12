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


__all__ = [
    "IngestedCard",
    "IngestedRace",
    "IngestedHorse",
    "IngestedPPLine",
]
