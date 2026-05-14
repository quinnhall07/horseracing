"""
app/db/persistence.py
─────────────────────
Convert ingestion results (Pydantic) to ORM rows and persist.

The ORM tree mirrors the schema 1-to-1, so this is mostly mechanical copy.
Splitting it out keeps `app/api/v1/ingest.py` focused on HTTP concerns and
makes the conversion unit-testable without spinning up FastAPI.

Stream A additions:
    * `load_card(session, card_id)` rebuilds a RaceCard pydantic object
      from the ORM rows so the analyse/portfolio endpoints can re-run
      inference without re-uploading the PDF.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import IngestedCard, IngestedHorse, IngestedPPLine, IngestedRace
from app.schemas.race import (
    HorseEntry,
    IngestionResult,
    ParsedRace,
    PastPerformanceLine,
    PaceStyle,
    RaceCard,
    RaceHeader,
    RaceType,
    Surface,
    TrackCondition,
)


def card_to_orm(card: RaceCard, processing_ms: float | None) -> IngestedCard:
    """Build a fully-populated IngestedCard ORM tree (uncommitted)."""
    avg_conf = (
        round(sum(r.parse_confidence for r in card.races) / len(card.races), 4)
        if card.races
        else 0.0
    )

    orm_card = IngestedCard(
        source_filename=card.source_filename,
        source_format=card.source_format,
        total_pages=card.total_pages,
        card_date=card.card_date,
        track_code=card.track_code,
        parse_confidence_avg=avg_conf,
        n_races=card.n_races,
        n_qualified_races=card.n_qualified_races,
        processing_ms=processing_ms,
    )

    for parsed_race in card.races:
        h = parsed_race.header
        orm_race = IngestedRace(
            race_number=h.race_number,
            race_date=h.race_date,
            track_code=h.track_code,
            track_name=h.track_name,
            race_name=h.race_name,
            distance_furlongs=h.distance_furlongs,
            distance_raw=h.distance_raw,
            surface=h.surface.value,
            condition=h.condition.value,
            race_type=h.race_type.value,
            claiming_price=h.claiming_price,
            purse_usd=h.purse_usd,
            grade=h.grade,
            age_sex_restrictions=h.age_sex_restrictions,
            weight_conditions=h.weight_conditions,
            post_time=h.post_time,
            weather=h.weather,
            parse_confidence=parsed_race.parse_confidence,
            parse_warnings=list(parsed_race.parse_warnings),
        )

        for entry in parsed_race.entries:
            orm_horse = IngestedHorse(
                horse_name=entry.horse_name,
                horse_id_external=entry.horse_id,
                post_position=entry.post_position,
                morning_line_odds=entry.morning_line_odds,
                ml_implied_prob=entry.ml_implied_prob,
                jockey=entry.jockey,
                trainer=entry.trainer,
                owner=entry.owner,
                weight_lbs=entry.weight_lbs,
                medication_flags=list(entry.medication_flags),
                equipment_changes=list(entry.equipment_changes),
                pace_style=entry.pace_style.value,
                ewm_speed_figure=entry.ewm_speed_figure,
                days_since_last=entry.days_since_last,
                class_trajectory=entry.class_trajectory,
            )
            for pp in entry.pp_lines:
                orm_horse.pp_lines.append(IngestedPPLine(
                    race_date=pp.race_date,
                    track_code=pp.track_code,
                    race_number=pp.race_number,
                    distance_furlongs=pp.distance_furlongs,
                    surface=pp.surface.value,
                    condition=pp.condition.value,
                    race_type=pp.race_type.value,
                    claiming_price=pp.claiming_price,
                    purse_usd=pp.purse_usd,
                    post_position=pp.post_position,
                    finish_position=pp.finish_position,
                    lengths_behind=pp.lengths_behind,
                    field_size=pp.field_size,
                    jockey=pp.jockey,
                    weight_lbs=pp.weight_lbs,
                    odds_final=pp.odds_final,
                    speed_figure=pp.speed_figure,
                    speed_figure_source=pp.speed_figure_source,
                    fraction_q1=pp.fraction_q1,
                    fraction_q2=pp.fraction_q2,
                    fraction_finish=pp.fraction_finish,
                    beaten_lengths_q1=pp.beaten_lengths_q1,
                    beaten_lengths_q2=pp.beaten_lengths_q2,
                    days_since_prev=pp.days_since_prev,
                    comment=pp.comment,
                ))
            orm_race.horses.append(orm_horse)

        orm_card.races.append(orm_race)

    return orm_card


async def persist_ingestion_result(
    session: AsyncSession,
    result: IngestionResult,
) -> int | None:
    """Persist an IngestionResult to the live DB. Returns the card_id or None.

    Returns None when there is no card to persist (failed ingestion).
    Caller commits via the session scope.
    """
    if result.card is None:
        return None
    orm_card = card_to_orm(result.card, processing_ms=result.processing_ms)
    session.add(orm_card)
    await session.flush()
    return orm_card.id


# ──────────────────────────────────────────────────────────────────────────────
# Stream A — DB → Pydantic reconstruction
# ──────────────────────────────────────────────────────────────────────────────


def _to_surface(value: Optional[str]) -> Surface:
    if not value:
        return Surface.UNKNOWN
    try:
        return Surface(value)
    except ValueError:
        return Surface.UNKNOWN


def _to_condition(value: Optional[str]) -> TrackCondition:
    if not value:
        return TrackCondition.UNKNOWN
    try:
        return TrackCondition(value)
    except ValueError:
        return TrackCondition.UNKNOWN


def _to_race_type(value: Optional[str]) -> RaceType:
    if not value:
        return RaceType.UNKNOWN
    try:
        return RaceType(value)
    except ValueError:
        return RaceType.UNKNOWN


def _to_pace_style(value: Optional[str]) -> PaceStyle:
    if not value:
        return PaceStyle.UNKNOWN
    try:
        return PaceStyle(value)
    except ValueError:
        return PaceStyle.UNKNOWN


def _orm_pp_to_schema(pp: IngestedPPLine) -> PastPerformanceLine:
    return PastPerformanceLine(
        race_date=pp.race_date,
        track_code=pp.track_code,
        race_number=pp.race_number,
        distance_furlongs=pp.distance_furlongs,
        surface=_to_surface(pp.surface),
        condition=_to_condition(pp.condition),
        race_type=_to_race_type(pp.race_type),
        claiming_price=pp.claiming_price,
        purse_usd=pp.purse_usd,
        post_position=pp.post_position,
        finish_position=pp.finish_position,
        lengths_behind=pp.lengths_behind,
        field_size=pp.field_size,
        jockey=pp.jockey,
        weight_lbs=pp.weight_lbs,
        odds_final=pp.odds_final,
        speed_figure=pp.speed_figure,
        speed_figure_source=pp.speed_figure_source,
        fraction_q1=pp.fraction_q1,
        fraction_q2=pp.fraction_q2,
        fraction_finish=pp.fraction_finish,
        beaten_lengths_q1=pp.beaten_lengths_q1,
        beaten_lengths_q2=pp.beaten_lengths_q2,
        days_since_prev=pp.days_since_prev,
        comment=pp.comment,
    )


def _orm_horse_to_schema(horse: IngestedHorse) -> HorseEntry:
    return HorseEntry(
        horse_name=horse.horse_name,
        horse_id=horse.horse_id_external,
        post_position=horse.post_position,
        morning_line_odds=horse.morning_line_odds,
        ml_implied_prob=horse.ml_implied_prob,
        jockey=horse.jockey,
        trainer=horse.trainer,
        owner=horse.owner,
        weight_lbs=horse.weight_lbs,
        medication_flags=list(horse.medication_flags or []),
        equipment_changes=list(horse.equipment_changes or []),
        pp_lines=[_orm_pp_to_schema(pp) for pp in horse.pp_lines],
        pace_style=_to_pace_style(horse.pace_style),
        ewm_speed_figure=horse.ewm_speed_figure,
        days_since_last=horse.days_since_last,
        class_trajectory=horse.class_trajectory,
    )


def _orm_race_to_schema(race: IngestedRace) -> ParsedRace:
    header = RaceHeader(
        race_number=race.race_number,
        race_date=race.race_date,
        track_code=race.track_code,
        track_name=race.track_name,
        race_name=race.race_name,
        distance_furlongs=race.distance_furlongs,
        distance_raw=race.distance_raw,
        surface=_to_surface(race.surface),
        condition=_to_condition(race.condition),
        race_type=_to_race_type(race.race_type),
        claiming_price=race.claiming_price,
        purse_usd=race.purse_usd,
        grade=race.grade,
        age_sex_restrictions=race.age_sex_restrictions,
        weight_conditions=race.weight_conditions,
        post_time=race.post_time,
        weather=race.weather,
    )
    return ParsedRace(
        header=header,
        entries=[_orm_horse_to_schema(h) for h in race.horses],
        parse_confidence=race.parse_confidence,
        parse_warnings=list(race.parse_warnings or []),
    )


def _orm_card_to_schema(orm_card: IngestedCard) -> RaceCard:
    return RaceCard(
        source_filename=orm_card.source_filename,
        source_format=orm_card.source_format,
        total_pages=orm_card.total_pages,
        card_date=orm_card.card_date,
        track_code=orm_card.track_code,
        races=[_orm_race_to_schema(r) for r in orm_card.races],
    )


async def load_card(session: AsyncSession, card_id: int) -> Optional[RaceCard]:
    """Reload a persisted card and return the Pydantic RaceCard.

    Eagerly loads the entire (races → horses → pp_lines) tree via
    SQLAlchemy `selectinload` to avoid lazy-load round-trips. Returns
    None when the card_id is not found.
    """
    stmt = (
        select(IngestedCard)
        .where(IngestedCard.id == card_id)
        .options(
            selectinload(IngestedCard.races)
            .selectinload(IngestedRace.horses)
            .selectinload(IngestedHorse.pp_lines)
        )
    )
    res = await session.execute(stmt)
    orm_card: Optional[IngestedCard] = res.scalar_one_or_none()
    if orm_card is None:
        return None
    return _orm_card_to_schema(orm_card)
