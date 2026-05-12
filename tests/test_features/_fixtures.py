"""Synthetic RaceCard fixtures shared across feature-engineering tests."""

from __future__ import annotations

from datetime import date, timedelta

from app.schemas.race import (
    HorseEntry,
    ParsedRace,
    PastPerformanceLine,
    RaceCard,
    RaceHeader,
    RaceType,
    Surface,
    TrackCondition,
)

TODAY = date(2026, 5, 10)


def make_pp(
    days_ago: int = 30,
    speed_figure: float | None = 80.0,
    finish: int | None = 3,
    distance: float = 6.0,
    surface: Surface = Surface.DIRT,
    condition: TrackCondition = TrackCondition.FAST,
    claiming_price: float | None = None,
    race_type: RaceType = RaceType.ALLOWANCE,
    fraction_q1: float | None = 22.4,
    fraction_q2: float | None = 46.0,
    fraction_finish: float | None = 71.5,
    beaten_lengths_q1: float | None = 1.5,
    beaten_lengths_q2: float | None = 1.0,
    purse: float | None = 50_000.0,
    post: int = 4,
    field_size: int | None = 8,
    weight: float | None = 120.0,
    odds_final: float | None = 5.0,
    jockey: str | None = "J Doe",
    speed_figure_source: str = "brisnet",
    track_code: str = "CD",
    race_number: int = 4,
) -> PastPerformanceLine:
    return PastPerformanceLine(
        race_date=TODAY - timedelta(days=days_ago),
        track_code=track_code,
        race_number=race_number,
        distance_furlongs=distance,
        surface=surface,
        condition=condition,
        race_type=race_type,
        claiming_price=claiming_price,
        purse_usd=purse,
        post_position=post,
        finish_position=finish,
        lengths_behind=0.0 if finish == 1 else 2.0,
        field_size=field_size,
        jockey=jockey,
        weight_lbs=weight,
        odds_final=odds_final,
        speed_figure=speed_figure,
        speed_figure_source=speed_figure_source,
        fraction_q1=fraction_q1,
        fraction_q2=fraction_q2,
        fraction_finish=fraction_finish,
        beaten_lengths_q1=beaten_lengths_q1,
        beaten_lengths_q2=beaten_lengths_q2,
    )


def make_horse(
    post: int = 1,
    name: str | None = None,
    jockey: str = "J Doe",
    trainer: str = "T Smith",
    morning_line: float | None = 4.0,
    weight: float = 120.0,
    pp_lines: list[PastPerformanceLine] | None = None,
) -> HorseEntry:
    return HorseEntry(
        horse_name=name or f"HORSE_{post}",
        post_position=post,
        morning_line_odds=morning_line,
        jockey=jockey,
        trainer=trainer,
        weight_lbs=weight,
        pp_lines=pp_lines if pp_lines is not None else [make_pp()],
    )


def make_race(
    race_number: int = 4,
    n_horses: int = 6,
    distance: float = 6.0,
    surface: Surface = Surface.DIRT,
    race_type: RaceType = RaceType.ALLOWANCE,
    parse_confidence: float = 0.9,
    horses: list[HorseEntry] | None = None,
) -> ParsedRace:
    header = RaceHeader(
        race_number=race_number,
        race_date=TODAY,
        track_code="CD",
        track_name="Churchill Downs",
        distance_furlongs=distance,
        distance_raw=f"{distance:g} Furlongs",
        surface=surface,
        condition=TrackCondition.FAST,
        race_type=race_type,
        purse_usd=62_000.0,
    )
    if horses is None:
        horses = [make_horse(post=i + 1, name=f"HORSE_{i+1}") for i in range(n_horses)]
    return ParsedRace(header=header, entries=horses, parse_confidence=parse_confidence)


def make_card(races: list[ParsedRace] | None = None) -> RaceCard:
    races = races if races is not None else [make_race()]
    return RaceCard(
        source_filename="test_card.pdf",
        source_format="brisnet_up",
        total_pages=1,
        card_date=TODAY,
        track_code="CD",
        races=races,
    )
