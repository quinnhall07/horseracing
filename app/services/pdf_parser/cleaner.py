"""
app/services/pdf_parser/cleaner.py
────────────────────────────────────
Low-level text normalisation applied to raw pdfplumber output before any
structured parsing.  These functions are stateless and pure — safe to call
in parallel across pages.

Responsibilities:
  • Strip non-ASCII garbage introduced by PDF font encoding issues
  • Normalise whitespace while preserving columnar alignment tokens
  • Standardise common abbreviations and track condition codes
  • Convert fractional odds strings ("3-1", "5/2") → decimal float
  • Convert fractional distances ("1 1/16 Miles") → furlongs float
  • Parse split times from their compact string form ("1:10.40" → seconds)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# String normalisation
# ──────────────────────────────────────────────────────────────────────────────

# Characters that are legal in racing text but can appear as multi-char ligatures
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00bd": "1/2",
    "\u00bc": "1/4",
    "\u00be": "3/4",
    # ½ / ¼ / ¾ also appear in distance strings ("5½f")
}

# Pathological space runs (≥3) collapse to exactly two spaces. This bounds
# whitespace blow-up while preserving the columnar gap that downstream parsers
# (e.g., BrisnetParser._RE_HORSE_LINE) rely on as a column separator.
_EXCESS_SPACE = re.compile(r" {3,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(raw: str) -> str:
    """
    Canonical text normalization pipeline.

    Steps:
      1. Map Unicode ligatures and typographic characters to ASCII equivalents
      2. Strip control characters (keep \\t and \\n for layout)
      3. Decompose remaining Unicode and strip combining marks
      4. Cap pathological whitespace runs (3+ spaces → 2 spaces) while
         preserving columnar alignment for downstream regex parsers
    """
    if not raw:
        return ""
    # Step 1: ligature substitution
    for ligature, replacement in _LIGATURE_MAP.items():
        raw = raw.replace(ligature, replacement)
    # Step 2: strip control characters
    raw = _CONTROL_CHARS.sub("", raw)
    # Step 3: NFC → NFKD decompose → strip combining marks → re-encode ASCII
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(c for c in raw if not unicodedata.combining(c))
    # Step 4: cap excess whitespace; preserve columnar gaps (2 spaces)
    raw = _EXCESS_SPACE.sub("  ", raw)
    return raw.strip()


def collapse_whitespace(raw: str) -> str:
    """
    Aggressively collapse all internal whitespace runs to a single space.

    Use this for cleaning short value strings (names, descriptions) where
    columnar alignment is not relevant. Distinct from `normalize_text`,
    which preserves multi-space column separators.
    """
    return re.sub(r"\s+", " ", raw).strip() if raw else ""


def clean_name(raw: str) -> str:
    """
    Normalise horse/jockey/trainer names.

    Handles:
      • All-caps names (Brisnet style) → Title Case
      • Embedded asterisks (apprentice allowance markers on jockeys)
      • Multi-space gaps (collapsed to single space — names are not columnar)
      • Leading/trailing punctuation
    """
    cleaned = collapse_whitespace(normalize_text(raw))
    # Remove apprentice marker (* or **)
    cleaned = re.sub(r"\*+$", "", cleaned).strip()
    # Brisnet prints names in ALL CAPS; convert to title case
    if cleaned.isupper():
        cleaned = cleaned.title()
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Odds conversion
# ──────────────────────────────────────────────────────────────────────────────

# Accepts: "3-1", "5/2", "3.50", "9-5", "even", "evens", "1/1"
_ODDS_FRACTIONAL = re.compile(
    r"^(?P<num>\d+(?:\.\d+)?)\s*[-/]\s*(?P<den>\d+(?:\.\d+)?)$"
)
_ODDS_DECIMAL = re.compile(r"^\d+(?:\.\d+)$")
_ODDS_EVEN = re.compile(r"^ev(?:en(?:s)?)?$", re.IGNORECASE)


def parse_odds_to_decimal(raw: str) -> Optional[float]:
    """
    Convert any odds string to decimal (European) format.

    Decimal odds = (fractional numerator / denominator) + 1.0

    Examples:
      "3-1"  → 4.0
      "5/2"  → 3.5
      "even" → 2.0
      "4.50" → 4.50 (already decimal)
      "1"    → None  (ambiguous — refuse rather than guess)
    """
    if not raw:
        return None
    raw = raw.strip()

    if _ODDS_EVEN.match(raw):
        return 2.0

    m = _ODDS_FRACTIONAL.match(raw)
    if m:
        try:
            num, den = float(m.group("num")), float(m.group("den"))
            if den == 0:
                return None
            return round(num / den + 1.0, 4)
        except (ValueError, ZeroDivisionError):
            return None

    # Already decimal
    try:
        val = float(raw)
        # Sanity: decimal odds must be ≥ 1.0 and ≤ 1001.0 (100-1 longshot)
        return val if 1.0 <= val <= 1001.0 else None
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Distance conversion
# ──────────────────────────────────────────────────────────────────────────────

# Furlong fractions that appear in distance strings.
# Coverage rationale: US racing uses 16ths and 8ths of a mile predominantly.
# Common route distances: 1 1/16m, 1 1/8m, 1 3/16m, 1 1/4m, 1 3/8m, 1 1/2m, 1 5/8m.
_FRACTION_MAP: dict[str, float] = {
    "1/16": 0.0625,
    "3/16": 0.1875,
    "5/16": 0.3125,
    "7/16": 0.4375,
    "1/8": 0.125,
    "3/8": 0.375,
    "5/8": 0.625,
    "7/8": 0.875,
    "1/4": 0.25,
    "3/4": 0.75,
    "1/2": 0.5,
    # Unicode variants (already normalised by clean_text, but belt-and-suspenders)
    "½": 0.5,
    "¼": 0.25,
    "¾": 0.75,
}

# Furlongs per unit
_DISTANCE_UNITS: dict[str, float] = {
    "furlong": 1.0,
    "furlongs": 1.0,
    "f": 1.0,
    "mile": 8.0,
    "miles": 8.0,
    "m": 8.0,
    "yard": 1.0 / 220.0,  # 1 furlong = 220 yards
    "yards": 1.0 / 220.0,
}

# Pattern: optional integer, optional fraction, unit.
# Fraction alternation lists the multi-digit denominators first so the
# engine doesn't greedily match "1/1" out of "1/16".
_DISTANCE_RE = re.compile(
    r"""
    ^\s*
    (?P<whole>\d+)?                                            # optional integer part
    \s*
    (?P<frac>1/16|3/16|5/16|7/16|1/8|3/8|5/8|7/8|1/4|3/4|1/2|½|¼|¾)?
    \s*
    (?P<unit>furlongs?|miles?|yards?|[fmF])\s*$               # required unit
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_distance_to_furlongs(raw: str) -> Optional[float]:
    """
    Convert any textual distance representation to furlongs (float).

    Examples:
      "6 Furlongs"     → 6.0
      "1 1/16 Miles"   → 8.5   (1 + 1/16 = 1.0625 miles × 8 = 8.5f)
      "5½f"            → 5.5
      "1 Mile 70 Yards"→ 8.318...
      "About 5f"       → 5.0
    """
    if not raw:
        return None

    raw = normalize_text(raw)
    # Strip "About" prefix (turf courses sometimes list approximate distances)
    raw = re.sub(r"^about\s+", "", raw, flags=re.IGNORECASE).strip()

    # Special case: "1 Mile 70 Yards" — two-segment distances
    two_seg = re.match(
        r"^(\d+)\s+miles?\s+(\d+)\s+yards?$", raw, re.IGNORECASE
    )
    if two_seg:
        miles = int(two_seg.group(1))
        yards = int(two_seg.group(2))
        return round(miles * 8.0 + yards / 220.0, 4)

    m = _DISTANCE_RE.match(raw)
    if not m:
        return None

    whole = int(m.group("whole") or 0)
    frac_str = m.group("frac") or ""
    unit_str = m.group("unit").lower().rstrip("s")  # normalise plural
    if unit_str == "m" and raw.lower().endswith("miles"):
        unit_str = "mile"

    frac_val = _FRACTION_MAP.get(frac_str, 0.0)
    unit_mult = _DISTANCE_UNITS.get(unit_str + "s", _DISTANCE_UNITS.get(unit_str, 1.0))

    total_units = whole + frac_val
    return round(total_units * unit_mult, 4) if total_units > 0 else None


# ──────────────────────────────────────────────────────────────────────────────
# Time parsing
# ──────────────────────────────────────────────────────────────────────────────

# Brisnet encodes times as ":22.2" (22.2s) for fractions or "1:10.40" for mile+
_FRACTION_TIME_RE = re.compile(r"^:?(\d+)\.(\d{1,2})$")
_FULL_TIME_RE = re.compile(r"^(\d+):(\d{2})\.(\d{1,2})$")


def parse_time_to_seconds(raw: str) -> Optional[float]:
    """
    Parse a fractional or full split time string to total seconds.

    Accepted formats:
      ":22.2"   → 22.2 seconds
      "22.20"   → 22.20 seconds
      "1:10.40" → 70.40 seconds
      "2:02.00" → 122.00 seconds
    """
    if not raw:
        return None
    raw = raw.strip()

    m = _FULL_TIME_RE.match(raw)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        hundredths = m.group(3).ljust(2, "0")  # "4" → "40"
        return round(minutes * 60 + seconds + int(hundredths) / 100, 3)

    m = _FRACTION_TIME_RE.match(raw)
    if m:
        seconds = int(m.group(1))
        hundredths = m.group(2).ljust(2, "0")
        return round(seconds + int(hundredths) / 100, 3)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Track condition and surface normalisation
# ──────────────────────────────────────────────────────────────────────────────

_SURFACE_MAP: dict[str, str] = {
    "d": "dirt", "dirt": "dirt",
    "t": "turf", "turf": "turf", "grass": "turf",
    "s": "synthetic", "syn": "synthetic", "synthetic": "synthetic",
    "a": "synthetic",  # "All-Weather" synthetic (UK/Polytrack)
}

_CONDITION_MAP: dict[str, str] = {
    "ft": "fast", "fast": "fast",
    "gd": "good", "good": "good",
    "sl": "sloppy", "sloppy": "sloppy",
    "my": "muddy", "muddy": "muddy",
    "hy": "heavy", "heavy": "heavy",
    "fr": "frozen", "frozen": "frozen",
    "fm": "firm", "firm": "firm",
    "yl": "yielding", "yielding": "yielding",
    "sf": "soft", "soft": "soft",
}


def parse_surface(raw: str) -> str:
    return _SURFACE_MAP.get(raw.strip().lower(), "unknown")


def parse_condition(raw: str) -> str:
    return _CONDITION_MAP.get(raw.strip().lower(), "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# Race type normalisation
# ──────────────────────────────────────────────────────────────────────────────

_RACE_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"maiden\s+claiming", re.I), "maiden_claiming"),
    (re.compile(r"maiden\s+special\s+weight|msw|mdn\s+sw", re.I), "maiden_special_weight"),
    (re.compile(r"\bgraded\b|\bgrade\s*[123]\b|g[123]\b", re.I), "graded_stakes"),
    (re.compile(r"\bstakes\b|\bstk\b", re.I), "stakes"),
    (re.compile(r"allowance\s+optional\s+claiming|aoc", re.I), "allowance_optional_claiming"),
    (re.compile(r"\ballowance\b|\balw\b", re.I), "allowance"),
    (re.compile(r"\bhandicap\b|\bhcp\b|\bhdcp\b", re.I), "handicap"),
    (re.compile(r"\bclaiming\b|\bclm\b", re.I), "claiming"),
]


def parse_race_type(raw: str) -> str:
    for pattern, race_type in _RACE_TYPE_PATTERNS:
        if pattern.search(raw):
            return race_type
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Utility: extract numeric value from a string (lengths, weights, claiming prices)
# ──────────────────────────────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_first_number(raw: str) -> Optional[float]:
    """Pull the first numeric value out of a mixed string.  Returns None if absent."""
    if not raw:
        return None
    m = _NUMERIC_RE.search(raw.strip())
    return float(m.group()) if m else None


def extract_claiming_price(raw: str) -> Optional[float]:
    """
    Parse a claiming price from strings like "$20,000", "20000", "Clm 15000".
    Returns the price in USD as a float.
    """
    if not raw:
        return None
    # Strip dollar signs and commas before parsing
    clean = re.sub(r"[$,]", "", raw)
    return extract_first_number(clean)