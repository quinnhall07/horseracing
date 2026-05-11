"""
app/services/pdf_parser/brisnet_parser.py
──────────────────────────────────────────
Parser for Brisnet Ultimate Past Performances (UP) PDF format.

Brisnet UP is the primary supported format for Phase 1.  The file structure:
  • Each race occupies 1–3 pages depending on field size.
  • Page header: Track name, card date, race number, distance, surface,
    conditions, purse, claiming price.
  • Horse block: horse name, post position, morning line, jockey, trainer,
    weight, medication/equipment — followed by 10 PP lines in a dense table.
  • PP line columns (approximate positions, vary by DPI):
    Date | Trk | Dst | Srf | Cond | Class | ClmPr | Purse |
    PP | Fin | Jky | Wgt | Odds | SpFig | ¼ | ½ | Fin | Comment

Because Brisnet PDFs are dense and occasionally misalign columns when
converted to text, this parser uses a two-pass strategy:
  1. Structural pass  — identify page/race boundaries with header regexes
  2. Content pass     — for each identified block, extract fields via targeted
                        column-positional slicing or line-level regex matching

All parse errors are collected as warnings rather than exceptions, so a
partially-parseable PDF still returns the races it could extract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from app.schemas.race import (
    HorseEntry,
    PastPerformanceLine,
    ParsedRace,
    RaceCard,
    RaceHeader,
    Surface,
    TrackCondition,
    RaceType,
    PaceStyle,
)
from app.services.pdf_parser.cleaner import (
    clean_name,
    extract_claiming_price,
    extract_first_number,
    normalize_text,
    parse_condition,
    parse_distance_to_furlongs,
    parse_odds_to_decimal,
    parse_race_type,
    parse_surface,
    parse_time_to_seconds,
)


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for Brisnet page-level headers
# ──────────────────────────────────────────────────────────────────────────────

# "RACE 3" or "Race 3" at the start of a page block
_RE_RACE_HEADER = re.compile(
    r"RACE\s+(?P<num>\d{1,2})", re.IGNORECASE
)

# Track code + date: "CD  05/03/2025" or "KEE   April 12, 2025"
_RE_TRACK_DATE = re.compile(
    r"""
    (?P<code>[A-Z]{2,5})          # 2-5 uppercase track code
    \s+
    (?:
        (?P<date_slash>\d{2}/\d{2}/\d{4})       # MM/DD/YYYY
        |
        (?P<date_long>[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})  # Month DD, YYYY
    )
    """,
    re.VERBOSE,
)

# Distance line: "6 Furlongs", "1 1/16 Miles", "5½f", "About 1 Mile".
# Fraction permits 1-2 digit numerator/denominator (covers 1/16, 3/16, etc.).
# Leading whitespace before the fraction is optional so post-normalize "51/2f"
# (the ½-ligature substitution output) still matches.
_RE_DISTANCE = re.compile(
    r"(?:About\s+)?(\d+(?:\s*\d{1,2}/\d{1,2})?\s*(?:Furlongs?|Miles?|[fF]))",
    re.IGNORECASE,
)

# Surface from condition line: "(Dirt)", "(Turf)", "(Synthetic)"
_RE_SURFACE_PAREN = re.compile(r"\((Dirt|Turf|Synthetic|All.Weather)\)", re.IGNORECASE)

# Purse: "Purse: $35,000" or "Purse $35,000"
_RE_PURSE = re.compile(r"Purse[:\s]+\$?([\d,]+)", re.IGNORECASE)

# Claiming price: "For 3YO+ (Claiming $20,000)" or "Clm 15000".
# Match either spelling — "Claiming" doesn't actually start with "Clm" so the
# alternation is required.
_RE_CLAIMING = re.compile(
    r"Cl(?:aiming|m)\b[\s:$]+\$?([\d,]+)", re.IGNORECASE
)

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for horse-level data
# ──────────────────────────────────────────────────────────────────────────────

# Post position + horse name line.
# Brisnet format: "  1  HORSE NAME                  Jockey Name   118  3-1"
# Post is typically at column 2-4, then name in uppercase.
_RE_HORSE_LINE = re.compile(
    r"""
    ^\s*
    (?P<post>\d{1,2})           # post position 1-24
    \s+
    (?P<name>[A-Z][A-Z\s'()\-\.]{2,40})  # horse name (uppercase in Brisnet)
    \s{2,}
    (?P<jockey>[A-Za-z,\s\.]+?) # jockey (Last, First format)
    \s+
    (?P<weight>\d{3})            # weight: 3 digits (113-126 typical)
    \s+
    (?P<ml_odds>[\d\-\/]+|even(?:s)?)   # morning line odds
    """,
    re.VERBOSE,
)

# Trainer line — always follows horse line in Brisnet
_RE_TRAINER = re.compile(
    r"^\s*Trainer[:\s]+(?P<name>[A-Za-z,\s\.]+?)\s*$", re.IGNORECASE
)

# Medication flags: "L" (Lasix), "B" (Blinkers), "LB", etc.
# In Brisnet these appear as single chars before the weight column
_RE_MEDICATION = re.compile(r"\b([LB]{1,3})\b")

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for past performance lines
# ──────────────────────────────────────────────────────────────────────────────

# PP date: "05/03/24" or "05/03/2024"
_RE_PP_DATE = re.compile(r"(\d{2}/\d{2}/\d{2,4})")

# PP line: full row pattern (positional matching after date found)
# Groups: date | track | dist | surface | condition | class_desc |
#         pp_pos | fin_pos | jockey | wt | odds | speed_fig |
#         q1 | q2 | fin_time | beaten_q1 | beaten_q2 | comment
_RE_PP_LINE = re.compile(
    r"""
    (?P<date>\d{2}/\d{2}/\d{2,4})       # race date
    \s+(?P<trk>[A-Z]{2,5})              # track code
    \s+(?P<dist>[\d\s/½¼¾]+[fmF]?)     # distance shorthand (e.g., "6f", "1m1/16")
    \s+(?P<srf>[dtsaDTSA])              # surface code (d/t/s/a)
    \s+(?P<cond>[a-zA-Z]{2,4})          # condition code (ft/gd/sl/my etc)
    \s+(?P<class_desc>[A-Za-z\d\s\-/]+?)  # race class description
    \s+(?P<pp_pos>\d{1,2})              # post position
    \s+(?P<fin_pos>\d{1,2}(?:no|dq)?)  # finish position (may have "no" or "dq" suffix)
    \s+(?P<jockey>[A-Za-z,\s\.]+?)     # jockey
    \s+(?P<weight>\d{3})               # weight
    \s+(?P<odds>[\d\.]+)               # final odds (decimal)
    \s+(?P<spfig>[-\d]+)               # speed figure (can be negative or "--")
    \s+(?P<q1>:\d+\.\d+|\d+\.\d+)     # q1 time
    \s+(?P<q2>:\d+\.\d+|\d+\.\d+)?    # q2 time (optional)
    \s+(?P<fin_time>\d+:\d+\.\d+|\d+\.\d+)  # final time
    """,
    re.VERBOSE,
)

# Simpler per-token PP pattern used as fallback when full regex fails
_RE_PP_SPEED_FIG = re.compile(r"\s+(-?\d{2,3})\s+")  # speed figure is typically 2-3 digits


# ──────────────────────────────────────────────────────────────────────────────
# Internal parse result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _RaceBlock:
    """Intermediate container accumulating raw text lines for one race."""
    race_num: int = 0
    header_lines: list[str] = field(default_factory=list)
    horse_blocks: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Main parser class
# ──────────────────────────────────────────────────────────────────────────────

class BrisnetParser:
    """
    Parse a Brisnet UP PDF (already extracted to text via pdfplumber) into a
    fully structured RaceCard.

    Usage:
        parser = BrisnetParser()
        card = parser.parse(raw_text, source_filename="card_2025_05_03.pdf")
    """

    def parse(self, raw_text: str, source_filename: str = "") -> RaceCard:
        """
        Entry point.  `raw_text` is the concatenated text of all PDF pages,
        with pages delimited by \\x0c (form feed).

        Returns a RaceCard even on partial parse failure — warnings accumulate
        per race, and parse_confidence reflects quality.
        """
        pages = raw_text.split("\x0c")
        race_blocks = self._segment_into_races(pages)

        parsed_races: list[ParsedRace] = []
        card_date: Optional[date] = None
        track_code: Optional[str] = None

        for block in race_blocks:
            parsed = self._parse_race_block(block)
            if parsed is not None:
                parsed_races.append(parsed)
                if card_date is None and parsed.header.race_date:
                    card_date = parsed.header.race_date
                if track_code is None and parsed.header.track_code:
                    track_code = parsed.header.track_code

        return RaceCard(
            source_filename=source_filename,
            source_format="brisnet_up",
            total_pages=len(pages),
            card_date=card_date,
            track_code=track_code,
            races=parsed_races,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: page segmentation into race blocks
    # ──────────────────────────────────────────────────────────────────────────

    def _segment_into_races(self, pages: list[str]) -> list[_RaceBlock]:
        """
        Split all pages into contiguous groups belonging to the same race.

        Strategy:
          • Scan each line for "RACE N" header markers.
          • When found, start a new _RaceBlock.
          • All lines until the next "RACE N" marker accumulate into the current block.
        """
        blocks: list[_RaceBlock] = []
        current: Optional[_RaceBlock] = None

        for page in pages:
            for line in page.splitlines():
                m = _RE_RACE_HEADER.search(line)
                if m:
                    if current is not None:
                        blocks.append(current)
                    current = _RaceBlock(race_num=int(m.group("num")))
                if current is not None:
                    current.header_lines.append(normalize_text(line))

        if current is not None:
            blocks.append(current)

        return blocks

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: parse one race block → ParsedRace
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_race_block(self, block: _RaceBlock) -> Optional[ParsedRace]:
        if block.race_num == 0:
            return None

        warnings: list[str] = []
        header = self._parse_race_header(block.header_lines, block.race_num, warnings)
        if header is None:
            return None
        entries = self._parse_horse_entries(block.header_lines, warnings)

        # Confidence: proportion of entries with ≥ 1 PP line, weighted by header completeness
        header_score = self._score_header(header)
        pp_score = (
            sum(1 for e in entries if e.n_pp >= 1) / max(len(entries), 1)
            if entries else 0.0
        )
        confidence = round((header_score * 0.4 + pp_score * 0.6), 3)

        return ParsedRace(
            header=header,
            entries=entries,
            parse_confidence=confidence,
            parse_warnings=warnings,
        )

    def _score_header(self, header: RaceHeader) -> float:
        """Return a completeness fraction for the race header [0, 1]."""
        scored_fields = [
            header.race_date is not None,
            header.track_code not in (None, ""),
            header.distance_furlongs > 0,
            header.surface != Surface.UNKNOWN,
            header.race_type != RaceType.UNKNOWN,
        ]
        return sum(scored_fields) / len(scored_fields)

    # ──────────────────────────────────────────────────────────────────────────
    # Header parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_race_header(
        self, lines: list[str], race_num: int, warnings: list[str]
    ) -> Optional[RaceHeader]:
        """
        Extract track code, date, distance, surface, condition, race type, purse,
        and claiming price from the first ~15 lines of a race block.

        Returns None when distance cannot be extracted — distance is the one
        truly load-bearing field for downstream feature engineering and the
        schema enforces `ge=2.0`, so we degrade gracefully rather than crash.
        """
        header_text = "\n".join(lines[:20])

        # Track code + date
        race_date: Optional[date] = None
        track_code: str = "UNK"
        m = _RE_TRACK_DATE.search(header_text)
        if m:
            track_code = m.group("code")
            raw_date = m.group("date_slash") or m.group("date_long") or ""
            race_date = _parse_date(raw_date)
        else:
            warnings.append(f"Race {race_num}: could not extract track code / date")

        # Distance — load-bearing. If missing, abandon this race.
        distance_furlongs = 0.0
        distance_raw = ""
        m_dist = _RE_DISTANCE.search(header_text)
        if m_dist:
            distance_raw = m_dist.group(1).strip()
            distance_furlongs = parse_distance_to_furlongs(distance_raw) or 0.0

        if distance_furlongs < 2.0:
            warnings.append(
                f"Race {race_num}: could not extract a valid distance "
                f"(raw='{distance_raw}'); skipping race"
            )
            return None

        # Surface
        surface_str = "unknown"
        m_srf = _RE_SURFACE_PAREN.search(header_text)
        if m_srf:
            surface_str = parse_surface(m_srf.group(1))
        else:
            # fallback: look for bare keyword
            for word in ("Dirt", "Turf", "Synthetic"):
                if re.search(rf"\b{word}\b", header_text, re.IGNORECASE):
                    surface_str = word.lower()
                    break

        # Track condition: appears near surface as "Fast", "Sloppy", etc.
        condition_str = "unknown"
        for cond_word in (
            "Fast", "Good", "Sloppy", "Muddy", "Heavy", "Frozen",
            "Firm", "Yielding", "Soft"
        ):
            if re.search(rf"\b{cond_word}\b", header_text, re.IGNORECASE):
                condition_str = cond_word.lower()
                break

        # Race type from conditions description line
        race_type_str = parse_race_type(header_text)

        # Purse
        purse: Optional[float] = None
        m_purse = _RE_PURSE.search(header_text)
        if m_purse:
            purse = extract_claiming_price(m_purse.group(1))

        # Claiming price
        claiming: Optional[float] = None
        m_clm = _RE_CLAIMING.search(header_text)
        if m_clm:
            claiming = extract_claiming_price(m_clm.group(1))

        # Grade (I/II/III) for stakes races
        grade: Optional[int] = None
        m_grade = re.search(r"Grade\s+(I{1,3}|[123])\b", header_text, re.IGNORECASE)
        if m_grade:
            raw_grade = m_grade.group(1).upper()
            grade = {"I": 1, "II": 2, "III": 3}.get(raw_grade, None)
            if grade is None and raw_grade.isdigit():
                grade = int(raw_grade)

        return RaceHeader(
            race_number=race_num,
            race_date=race_date or date.today(),  # fallback to today; flagged in warnings
            track_code=track_code,
            distance_furlongs=distance_furlongs,
            distance_raw=distance_raw or "Unknown",
            surface=Surface(surface_str),
            condition=TrackCondition(condition_str),
            race_type=RaceType(race_type_str),
            purse_usd=purse,
            claiming_price=claiming,
            grade=grade,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Horse entry parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_horse_entries(
        self, lines: list[str], warnings: list[str]
    ) -> list[HorseEntry]:
        """
        Scan through race block lines to find horse entries.

        A horse block begins with a line matching _RE_HORSE_LINE and ends
        when the next horse line (or end of block) is encountered.
        """
        entries: list[HorseEntry] = []
        current_horse_lines: list[str] = []
        in_horse_block = False

        for line in lines:
            m = _RE_HORSE_LINE.match(line)
            if m:
                # Commit previous horse block
                if in_horse_block and current_horse_lines:
                    entry = self._parse_single_horse(current_horse_lines, warnings)
                    if entry:
                        entries.append(entry)
                current_horse_lines = [line]
                in_horse_block = True
            elif in_horse_block:
                current_horse_lines.append(line)

        # Commit last horse block
        if in_horse_block and current_horse_lines:
            entry = self._parse_single_horse(current_horse_lines, warnings)
            if entry:
                entries.append(entry)

        return entries

    def _parse_single_horse(
        self, lines: list[str], warnings: list[str]
    ) -> Optional[HorseEntry]:
        """
        Extract one HorseEntry from a block of lines starting with the horse header.
        """
        if not lines:
            return None

        # ── First line: post, name, jockey, weight, ML odds ──────────────────
        m = _RE_HORSE_LINE.match(lines[0])
        if not m:
            warnings.append(f"Failed to parse horse line: {lines[0][:80]}")
            return None

        post = int(m.group("post"))
        horse_name = clean_name(m.group("name"))
        jockey = clean_name(m.group("jockey"))
        weight = extract_first_number(m.group("weight"))
        ml_decimal = parse_odds_to_decimal(m.group("ml_odds"))

        # ── Medication flags ──────────────────────────────────────────────────
        meds: list[str] = []
        m_meds = _RE_MEDICATION.search(lines[0])
        if m_meds:
            meds = list(m_meds.group(1))  # split "LB" → ["L", "B"]

        # ── Trainer (second line) ─────────────────────────────────────────────
        trainer: Optional[str] = None
        for line in lines[1:4]:
            m_tr = _RE_TRAINER.match(line)
            if m_tr:
                trainer = clean_name(m_tr.group("name"))
                break

        # ── PP lines (remaining lines) ────────────────────────────────────────
        pp_lines = self._parse_pp_lines(lines, warnings, horse_name)

        return HorseEntry(
            horse_name=horse_name,
            post_position=post,
            morning_line_odds=ml_decimal,
            jockey=jockey if jockey else None,
            trainer=trainer,
            weight_lbs=weight,
            medication_flags=sorted(set(meds)),
            pp_lines=pp_lines,
            pace_style=PaceStyle.UNKNOWN,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PP line parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_pp_lines(
        self, lines: list[str], warnings: list[str], horse_name: str
    ) -> list[PastPerformanceLine]:
        """
        Extract past performance rows from the lines following the horse header.

        Each PP line begins with a date token (MM/DD/YY).  We use the date
        as an anchor and then parse the remaining tokens positionally.
        """
        pp_lines: list[PastPerformanceLine] = []

        for line in lines[2:]:  # skip horse header + trainer line
            if not _RE_PP_DATE.search(line):
                continue
            pp = self._parse_single_pp_line(line, warnings, horse_name)
            if pp is not None:
                pp_lines.append(pp)

        # Compute days_since_prev for each PP entry
        for i in range(len(pp_lines) - 1):
            delta = (pp_lines[i].race_date - pp_lines[i + 1].race_date).days
            pp_lines[i].days_since_prev = max(delta, 0)

        return pp_lines

    def _parse_single_pp_line(
        self, line: str, warnings: list[str], horse_name: str
    ) -> Optional[PastPerformanceLine]:
        """Parse one PP table row.  Returns None if minimum required fields missing."""
        tokens = line.split()
        if len(tokens) < 8:
            return None  # too sparse; skip silently

        # ── Date ──────────────────────────────────────────────────────────────
        date_match = _RE_PP_DATE.search(line)
        if not date_match:
            return None
        pp_date = _parse_date(date_match.group(1))
        if pp_date is None:
            return None

        # ── Try full regex first, fall back to positional token extraction ────
        m = _RE_PP_LINE.match(line)
        if m:
            return self._build_pp_from_regex(m, pp_date)
        else:
            return self._build_pp_from_tokens(tokens, pp_date, warnings, horse_name)

    def _build_pp_from_regex(
        self, m: re.Match, pp_date: date
    ) -> Optional[PastPerformanceLine]:
        """Build PP from the full structured regex match."""
        try:
            fin_raw = m.group("fin_pos")
            fin_pos = int(re.match(r"\d+", fin_raw).group()) if fin_raw else None
            speed_fig_raw = m.group("spfig")
            speed_fig = float(speed_fig_raw) if speed_fig_raw and speed_fig_raw != "--" else None

            return PastPerformanceLine(
                race_date=pp_date,
                track_code=m.group("trk"),
                race_number=1,  # not reliably in line; set to 1 as placeholder
                distance_furlongs=parse_distance_to_furlongs(m.group("dist")) or 6.0,
                surface=Surface(parse_surface(m.group("srf"))),
                condition=TrackCondition(parse_condition(m.group("cond"))),
                race_type=RaceType(parse_race_type(m.group("class_desc"))),
                post_position=int(m.group("pp_pos")),
                finish_position=fin_pos,
                jockey=clean_name(m.group("jockey")),
                weight_lbs=extract_first_number(m.group("weight")),
                odds_final=parse_odds_to_decimal(m.group("odds")),
                speed_figure=speed_fig,
                speed_figure_source="brisnet",
                fraction_q1=parse_time_to_seconds(m.group("q1")),
                fraction_q2=parse_time_to_seconds(m.group("q2") or ""),
                fraction_finish=parse_time_to_seconds(m.group("fin_time")),
            )
        except Exception:
            return None

    def _build_pp_from_tokens(
        self,
        tokens: list[str],
        pp_date: date,
        warnings: list[str],
        horse_name: str,
    ) -> Optional[PastPerformanceLine]:
        """
        Fallback positional extraction when the structured regex does not match.

        Token order (approximate for Brisnet UP):
          0: date  1: track  2: dist  3: surf  4: cond  5+: class, pp, fin, ...
        This is inherently fragile — we extract what we can and mark confidence low.
        """
        if len(tokens) < 6:
            return None

        try:
            track_code = tokens[1] if len(tokens[1]) <= 5 else "UNK"
            dist_str = tokens[2] if len(tokens) > 2 else "6f"
            srf_str = tokens[3] if len(tokens) > 3 else "d"
            cond_str = tokens[4] if len(tokens) > 4 else "ft"

            # Speed figure: scan tokens for a 2-3 digit integer (possibly negative)
            speed_fig: Optional[float] = None
            for tok in tokens[5:]:
                if re.match(r"^-?\d{2,3}$", tok):
                    val = float(tok)
                    if -20 <= val <= 150:
                        speed_fig = val
                        break

            # Finish position: first 1-2 digit integer after token 4
            fin_pos: Optional[int] = None
            for tok in tokens[5:10]:
                if re.match(r"^\d{1,2}$", tok):
                    fin_pos = int(tok)
                    break

            # Post position: second such integer
            pp_pos = 1  # default; hard to distinguish from finish without full regex

            return PastPerformanceLine(
                race_date=pp_date,
                track_code=track_code,
                race_number=1,
                distance_furlongs=parse_distance_to_furlongs(dist_str) or 6.0,
                surface=Surface(parse_surface(srf_str)),
                condition=TrackCondition(parse_condition(cond_str)),
                post_position=pp_pos,
                finish_position=fin_pos,
                speed_figure=speed_fig,
                speed_figure_source="brisnet",
            )
        except Exception as exc:
            warnings.append(f"Fallback PP parse failed for {horse_name}: {exc}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Helper: date parsing
# ──────────────────────────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%m/%d/%Y",
    "%m/%d/%y",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
]


def _parse_date(raw: str) -> Optional[date]:
    raw = raw.strip().rstrip(",")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None