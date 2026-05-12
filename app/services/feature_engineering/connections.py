"""
app/services/feature_engineering/connections.py
───────────────────────────────────────────────
Jockey × trainer connection features.

The full treatment of connections (Layer 1d in the master reference) is a
Bayesian hierarchical model that lives in `app/services/models/connections.py`
and is trained on the master DB. Phase 2 only needs the basic per-horse
counts and win/place rates that the model and the meta-learner consume.

Signals (computed from a horse's own PP lines — historical jockey × trainer
agreement is approximated by the recent record because the PDF parse does
NOT expose the trainer-of-record per PP line):

  * `today_jt_same_pair`   — bool: today's jockey/trainer match the most recent PP
  * `jockey_repeat_streak` — count of consecutive recent PPs with today's jockey
  * `trainer_continuity`   — fraction of recent PPs run for today's trainer (proxy)
  * `n_jockey_pps`         — total recent PPs the today-jockey was on
  * `today_jockey_win_rate_in_pps` — wins/starts for today's jockey across the
                                     horse's recent PPs (per-horse rate; not a
                                     career rate — that comes from the master DB)

The PastPerformanceLine schema (see app/schemas/race.py) carries a per-PP
`jockey` field but NO per-PP `trainer` field. So `trainer_continuity` is a
weak proxy: True only when today's jockey was the recent jockey AND today's
trainer is on the horse, since trainer typically persists across riders.
A future enhancement would parse the PP line "comment" for trainer changes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from app.schemas.race import HorseEntry, ParsedRace, PastPerformanceLine

RECENT_PP_WINDOW: int = 6
"""Recent-PP window for jockey-continuity proxies."""


def _normalize(name: Optional[str]) -> str:
    return (name or "").strip().casefold()


def _consecutive_match(values: Sequence[str], target: str) -> int:
    """Count consecutive leading elements equal to target."""
    n = 0
    for v in values:
        if v == target:
            n += 1
        else:
            break
    return n


def horse_connections_summary(
    entry: HorseEntry, window: int = RECENT_PP_WINDOW
) -> dict[str, Optional[float] | bool]:
    """Jockey/trainer continuity stats for one horse vs its recent PPs."""
    pp_window = entry.pp_lines[:window]
    today_jockey = _normalize(entry.jockey)
    today_trainer = _normalize(entry.trainer)

    pp_jockeys = [_normalize(p.jockey) for p in pp_window]
    recent_jockey = pp_jockeys[0] if pp_jockeys else ""

    today_jt_same_pair = (
        bool(today_jockey)
        and bool(recent_jockey)
        and today_jockey == recent_jockey
    )

    streak = _consecutive_match(pp_jockeys, today_jockey) if today_jockey else 0

    jockey_pps = [pj for pj in pp_jockeys if pj == today_jockey and pj]
    n_jockey_pps = len(jockey_pps)

    wins_with_today_jockey = sum(
        1
        for p in pp_window
        if _normalize(p.jockey) == today_jockey
        and today_jockey
        and p.finish_position == 1
    )
    today_jockey_win_rate_in_pps = (
        (wins_with_today_jockey / n_jockey_pps) if n_jockey_pps else None
    )

    # Trainer continuity proxy. True if today's trainer is non-empty AND
    # today's jockey matches the most recent PP. Without per-PP trainer data
    # this is the strongest evidence we can derive from the parsed schema.
    trainer_continuity: Optional[float]
    if not today_trainer:
        trainer_continuity = None
    elif not pp_window:
        trainer_continuity = None
    else:
        trainer_continuity = float(today_jt_same_pair)

    return {
        "today_jt_same_pair":             today_jt_same_pair,
        "jockey_repeat_streak":           float(streak),
        "trainer_continuity":             trainer_continuity,
        "n_jockey_pps":                   float(n_jockey_pps),
        "today_jockey_win_rate_in_pps":   today_jockey_win_rate_in_pps,
    }


def build_connection_feature_frame(race: ParsedRace) -> pd.DataFrame:
    """Per-horse connection features. No field-relative aggregation — these
    signals are absolute counts/rates per horse."""
    rows = []
    for entry in race.entries:
        summary = horse_connections_summary(entry)
        summary["post_position"] = entry.post_position
        summary["horse_name"] = entry.horse_name
        rows.append(summary)
    return pd.DataFrame(rows).set_index("post_position")


__all__ = [
    "RECENT_PP_WINDOW",
    "horse_connections_summary",
    "build_connection_feature_frame",
]
