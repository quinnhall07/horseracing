"""
app/db/persistence.py
─────────────────────
Convert ingestion results (Pydantic) to ORM rows and persist.

The ORM tree mirrors the schema 1-to-1, so this is mostly mechanical copy.
Splitting it out keeps `app/api/v1/ingest.py` focused on HTTP concerns and
makes the conversion unit-testable without spinning up FastAPI.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IngestedCard, IngestedHorse, IngestedPPLine, IngestedRace
from app.schemas.race import IngestionResult, RaceCard


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
