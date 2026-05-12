"""SHA-256 deduplication keys for the master DB.

Source of truth: DATA_PIPELINE.md §3. The hashed strings below are part of the
DB contract — changing them invalidates every existing dedup_key. Do not edit
the hash inputs without bumping SCHEMA_VERSION.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

# Strip punctuation but preserve Unicode letters/digits. `\w` in Python 3 is
# Unicode-aware by default, so this keeps Japanese / Chinese / Cyrillic etc.
# names intact while still stripping `'`, `-`, `.` from English names like
# "O'Brien" or "St. Patrick".
_PUNCT_RE = re.compile(r"[^\w ]", re.UNICODE)
_WS_RE    = re.compile(r"\s+")


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Shared by horse / jockey / trainer normalization — same rule everywhere.
    """
    name = name.lower().strip()
    name = _PUNCT_RE.sub("", name)
    name = _WS_RE.sub(" ", name).strip()
    return name


# Alias kept for back-compat with DATA_PIPELINE.md §3 naming.
normalize_horse_name = normalize_name


def race_dedup_key(
    track_code: str,
    race_date: str | date,
    race_number: int,
    distance_furlongs: float,
    surface: str,
) -> str:
    """SHA256(track_code|race_date|race_num|distance_f|surface).

    `race_date` accepts ISO strings or `datetime.date`; normalized to ISO.
    `distance_furlongs` is rounded to 2 decimals so 4.5 == 4.50.
    """
    date_str = race_date.isoformat() if isinstance(race_date, date) else str(race_date).strip()
    raw = (
        f"{track_code.upper().strip()}"
        f"|{date_str}"
        f"|{int(race_number)}"
        f"|{round(float(distance_furlongs), 2)}"
        f"|{surface.lower().strip()}"
    )
    return _sha256(raw)


def horse_dedup_key(name: str, foaling_year: int | None, country: str | None) -> str:
    """SHA256(normalized_name|foaling_year_or_unknown|country_or_unknown)."""
    norm = normalize_name(name)
    year = str(int(foaling_year)) if foaling_year else "unknown"
    cty  = (country or "unknown").upper().strip()
    raw  = f"{norm}|{year}|{cty}"
    return _sha256(raw)


def person_dedup_key(name: str, jurisdiction: str | None) -> str:
    """SHA256(normalized_name|jurisdiction). Used for jockeys and trainers.

    DATA_PIPELINE.md §2 declares UNIQUE dedup_key on both tables; this is the
    canonical formula by analogy with the horse key (without foaling year).
    """
    norm = normalize_name(name)
    juris = (jurisdiction or "unknown").upper().strip()
    return _sha256(f"{norm}|{juris}")


# Explicit aliases so callers can be expressive.
jockey_dedup_key  = person_dedup_key
trainer_dedup_key = person_dedup_key


def result_dedup_key(race_key: str, horse_key: str) -> str:
    """SHA256(race_dedup_key|horse_dedup_key)."""
    return _sha256(f"{race_key}|{horse_key}")
