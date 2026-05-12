"""Unit conversion + normalization functions for the Phase 0 data pipeline.

Each Kaggle dataset uses different units (metres vs furlongs, stones vs lbs,
fractional vs decimal odds, etc.). Every column in field_maps.py that needs
conversion references a transformer name here.

Per CLAUDE.md §11, this module has no dependency on `app/`. Helpers duplicated
from `app/services/pdf_parser/cleaner.py` are intentional.

All functions return `None` on unparseable input rather than raising. That
gives the quality gate a chance to score the row instead of aborting the load.
"""

from __future__ import annotations

import re
from typing import Callable

METRES_PER_FURLONG = 201.168
YARDS_PER_FURLONG  = 220.0
LBS_PER_STONE      = 14


# ─── distance ──────────────────────────────────────────────────────────────

def metres_to_furlongs(metres: float | str | None) -> float | None:
    """Convert metres → furlongs, rounded to 4 decimals."""
    if metres is None:
        return None
    try:
        return round(float(metres) / METRES_PER_FURLONG, 4)
    except (TypeError, ValueError):
        return None


_UK_DIST_RE = re.compile(
    r"(?:(?P<m>\d+)\s*m)?\s*(?:(?P<f>\d+)\s*f)?\s*(?:(?P<y>\d+)\s*y)?",
    re.IGNORECASE,
)


def uk_distance_to_furlongs(raw: str | float | None) -> float | None:
    """Parse UK distance strings: '5f', '6f 10y', '1m', '1m2f', '1m4f 110y', '2m'.

    Conventions:
        - 1 mile = 8 furlongs
        - 1 furlong = 220 yards
        - 'm' before 'f' = miles (NOT metres) in UK racing context.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return round(float(raw), 4)

    s = str(raw).strip().lower()
    if not s:
        return None

    # If string contains no m/f/y markers, assume it is already a furlong number.
    if not re.search(r"[mfy]", s):
        try:
            return round(float(s), 4)
        except ValueError:
            return None

    match = _UK_DIST_RE.fullmatch(s.replace(" ", ""))
    if not match:
        return None

    miles    = int(match.group("m") or 0)
    furlongs = int(match.group("f") or 0)
    yards    = int(match.group("y") or 0)

    if miles == 0 and furlongs == 0 and yards == 0:
        return None

    total = miles * 8 + furlongs + yards / YARDS_PER_FURLONG
    return round(total, 4)


# ─── weight ────────────────────────────────────────────────────────────────

_KG_PER_LB = 0.45359237


def kg_to_lbs(raw: str | float | int | None) -> float | None:
    """Convert kilograms to pounds. Used for JRA weights (carried in kg)."""
    if raw is None:
        return None
    try:
        v = float(raw)
        return round(v / _KG_PER_LB, 2) if v > 0 else None
    except (TypeError, ValueError):
        return None


def stones_to_lbs(raw: str | float | int | None) -> float | None:
    """'9-0' → 126 lbs; '9-2' → 128 lbs; numeric input passes through."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    parts = s.split("-")
    if len(parts) == 2:
        try:
            return float(int(parts[0]) * LBS_PER_STONE + int(parts[1]))
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


# ─── odds ──────────────────────────────────────────────────────────────────

_ODDS_MARKER_RE = re.compile(r"\b(j-?f|co?-?f|f)\b", re.IGNORECASE)


def parse_odds_to_decimal(raw: str | float | int | None) -> float | None:
    """Generic odds parser.

    Accepts decimal floats, fractional strings ('5/2', '7/4'), 'EVS'/'EVENS',
    or US-style hyphenated ('5-2'). Returns decimal odds (e.g. 5/2 → 3.5).
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None

    s = str(raw).strip().upper()
    if not s:
        return None
    if s in {"EVS", "EVENS"}:
        return 2.0

    # Fractional: numerator/denominator or numerator-denominator.
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[/\-]\s*(\d+(?:\.\d+)?)", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den == 0:
            return None
        return round(num / den + 1.0, 4)

    # Bare number (already decimal).
    try:
        val = float(s)
        return val if val > 0 else None
    except ValueError:
        return None


def uk_sp_to_decimal(raw: str | float | int | None) -> float | None:
    """UK starting price → decimal. Strips trailing favourite markers ('F', 'J-F')."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return parse_odds_to_decimal(raw)
    cleaned = _ODDS_MARKER_RE.sub("", str(raw)).strip()
    return parse_odds_to_decimal(cleaned)


# ─── surface / condition ───────────────────────────────────────────────────

_UK_ALL_WEATHER_TRACKS = {
    "LIN", "LINGFIELD",
    "KEM", "KEMPTON",
    "WOL", "WOLVERHAMPTON",
    "SOU", "SOUTHWELL",
    "NEW", "NEWCASTLE",
    "CHE", "CHELMSFORD",
}


def uk_going_to_surface(going: str | None, track_code: str | None = None) -> str:
    """UK flat racing is turf except at all-weather tracks (Lingfield, Kempton, etc.)."""
    if track_code and track_code.upper().strip() in _UK_ALL_WEATHER_TRACKS:
        return "synthetic"
    if going and "standard" in str(going).lower():
        return "synthetic"
    return "turf"


def uk_going_to_condition(going: str | None) -> str | None:
    """UK going → canonical condition.

    UK turf labels: Firm, Good to Firm, Good, Good to Soft, Soft, Heavy.
    UK AW labels: Standard, Standard to Slow, Slow.
    """
    if going is None:
        return None
    g = str(going).lower().strip()
    if not g:
        return None
    # All-weather variants map to fast-equivalent for ML purposes.
    if g.startswith("standard"):
        return "fast"
    if "heavy" in g:
        return "heavy"
    if "soft" in g and "good" not in g:
        return "soft"
    if "good" in g and "soft" in g:
        return "good"
    if "good" in g and "firm" in g:
        return "good"
    if "firm" in g:
        return "fast"
    if "good" in g:
        return "good"
    return g  # passthrough for unknown values


# ─── Argentina (Spanish surface terms) ────────────────────────────────────

_AR_SURFACE_MAP = {
    "arena":     "dirt",       # sand
    "cesped":    "turf",       # grass (sometimes spelled "césped")
    "césped":    "turf",
    "sintetico": "synthetic",  # synthetic
    "sintético": "synthetic",
}


def ar_surface_to_canonical(raw: str | None) -> str | None:
    """Argentine racing surface → canonical surface."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    return _AR_SURFACE_MAP.get(s, s)


# ─── JRA (Japan) ──────────────────────────────────────────────────────────

# Surface: 芝 = turf, ダート = dirt, 障 = jumps (treat as turf for ML purposes)
_JRA_SURFACE_MAP = {
    "芝":     "turf",
    "ダート":  "dirt",
    "障":     "turf",
    "障害":   "turf",
}


def jpn_surface_to_canonical(raw: str | None) -> str | None:
    """JRA surface code → canonical surface."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return _JRA_SURFACE_MAP.get(s, s.lower())


# Condition: 良 = firm/fast, 稍重 = good (slightly soft), 重 = soft (heavy turf
# / muddy dirt), 不良 = heavy (very soft turf / sloppy dirt)
_JRA_CONDITION_MAP = {
    "良":   "fast",
    "稍重": "good",
    "重":   "soft",
    "不良": "heavy",
}


def jpn_condition_to_canonical(raw: str | None) -> str | None:
    """JRA track condition → canonical condition."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return _JRA_CONDITION_MAP.get(s, s.lower())


def hk_going_to_condition(going: str | None) -> str | None:
    """HK going values: Good, Good to Yielding, Yielding, Soft, Heavy."""
    if going is None:
        return None
    g = str(going).lower().strip()
    if not g:
        return None
    if "heavy" in g:
        return "heavy"
    if "soft" in g:
        return "soft"
    if "yielding" in g:
        return "good" if "good" in g else "yielding"
    if "good" in g:
        return "good"
    if "fast" in g:
        return "fast"
    return g


_SURFACE_MAP = {
    "d": "dirt",  "dirt": "dirt",  "main": "dirt",
    "t": "turf",  "turf": "turf",
    "aw": "synthetic", "syn": "synthetic", "synthetic": "synthetic",
    "all weather": "synthetic", "all-weather": "synthetic",
    "tapeta": "synthetic", "polytrack": "synthetic",
}


def normalize_surface(raw: str | None) -> str | None:
    """Map dataset-specific surface labels to {dirt, turf, synthetic}."""
    if raw is None:
        return None
    s = str(raw).lower().strip()
    if not s:
        return None
    return _SURFACE_MAP.get(s, s)


_CONDITION_MAP = {
    "ft": "fast", "fst": "fast", "fast": "fast",
    "gd": "good", "good": "good",
    "sy": "sloppy", "sly": "sloppy", "sloppy": "sloppy",
    "my": "muddy", "muddy": "muddy",
    "wf": "wet fast", "wet fast": "wet fast",
    "sl": "slow", "slow": "slow",
    "frozen": "frozen", "hard": "hard",
    "yl": "yielding", "yielding": "yielding",
    "sf": "soft", "soft": "soft",
    "hy": "heavy", "heavy": "heavy",
    "firm": "fast",
}


def normalize_condition(raw: str | None) -> str | None:
    """Map dataset-specific condition labels to canonical strings."""
    if raw is None:
        return None
    s = str(raw).lower().strip()
    if not s:
        return None
    return _CONDITION_MAP.get(s, s)


# ─── time ──────────────────────────────────────────────────────────────────

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+)(?:\.(\d+))?$")


def time_string_to_seconds(raw: str | float | int | None) -> float | None:
    """'1:10.40' → 70.40; '70.40' → 70.40; ':22.4' → 22.4."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    s = str(raw).strip().lstrip(":")
    if not s:
        return None
    m = _TIME_RE.fullmatch(s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return None
    mins  = int(m.group(1) or 0)
    secs  = int(m.group(2))
    frac  = m.group(3)
    frac_val = float(f"0.{frac}") if frac else 0.0
    total = mins * 60 + secs + frac_val
    return round(total, 4) if total > 0 else None


def time_string_to_minutes(raw: str | float | int | None) -> int | None:
    """Parse 'HH:MM' or 'H:MM' post-time into minutes-since-midnight.

    Used as a race_number proxy for datasets that don't carry an explicit
    race-number column (e.g., UK results that identify races by date+course
    +post-time). Returns None for unparseable input.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            v = int(raw)
            return v if 0 <= v < 1440 else None
        except (TypeError, ValueError):
            return None
    s = str(raw).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) == 2:
        try:
            h = int(parts[0])
            m = int(parts[1].split(".")[0])  # tolerate '5:25.30'
            if 0 <= h < 24 and 0 <= m < 60:
                return h * 60 + m
        except ValueError:
            return None
    return None


def extract_int(raw: str | float | int | None) -> int | None:
    """First contiguous integer in a string. 'RID1002-IE-05' → 1002."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    m = re.search(r"\d+", str(raw))
    return int(m.group(0)) if m else None


# ─── money ─────────────────────────────────────────────────────────────────
# For competition purposes use fixed FX rates. Mark all converted amounts in
# the source dataset's `notes` field so downstream consumers can re-convert
# if needed. Real-time FX is out of scope for Phase 0.

GBP_USD_FIXED = 1.27
HKD_USD_FIXED = 0.128


def gbp_to_usd(amount: float | int | str | None, _date: str | None = None) -> float | None:
    if amount is None:
        return None
    try:
        return round(float(amount) * GBP_USD_FIXED, 2)
    except (TypeError, ValueError):
        return None


def hkd_to_usd(amount: float | int | str | None, _date: str | None = None) -> float | None:
    if amount is None:
        return None
    try:
        return round(float(amount) * HKD_USD_FIXED, 2)
    except (TypeError, ValueError):
        return None


# ─── registry ──────────────────────────────────────────────────────────────
# Maps the string names used in field_maps.py to the actual callable.
TRANSFORMERS: dict[str, Callable] = {
    "metres_to_furlongs":      metres_to_furlongs,
    "uk_distance_to_furlongs": uk_distance_to_furlongs,
    "stones_to_lbs":           stones_to_lbs,
    "kg_to_lbs":               kg_to_lbs,
    "parse_odds_to_decimal":   parse_odds_to_decimal,
    "uk_sp_to_decimal":        uk_sp_to_decimal,
    "uk_going_to_surface":     uk_going_to_surface,
    "uk_going_to_condition":   uk_going_to_condition,
    "hk_going_to_condition":   hk_going_to_condition,
    "jpn_surface_to_canonical": jpn_surface_to_canonical,
    "jpn_condition_to_canonical": jpn_condition_to_canonical,
    "ar_surface_to_canonical": ar_surface_to_canonical,
    "normalize_surface":       normalize_surface,
    "normalize_condition":     normalize_condition,
    "time_string_to_seconds":  time_string_to_seconds,
    "time_string_to_minutes":  time_string_to_minutes,
    "extract_int":             extract_int,
    "gbp_to_usd":              gbp_to_usd,
    "hkd_to_usd":              hkd_to_usd,
}
