# PROGRESS.md — Session Log

Update this file at the end of every Claude Code session.
Format: newest session at the top.

---

## Current State

**Phase:** 1 — PDF Ingestion Pipeline
**Last completed task:** Phase 1 parser test suite (172 tests, all passing) + parser bugfixes surfaced by tests
**Next task:** Project bootstrap (`app/core/logging.py`, `app/main.py`, `app/api/v1/ingest.py`, `app/db/`, `.env.example`) — then validate end-to-end against a real Brisnet PDF

---

## Session Log

### Session: 2026-05-11 — Phase 1 parser test suite

**Completed:**
- Package scaffolding: `app/__init__.py`, `app/schemas/__init__.py`, `app/services/__init__.py`,
  `app/services/pdf_parser/__init__.py`, `app/core/__init__.py`, `tests/__init__.py`,
  `tests/test_parser/__init__.py`
- `app/core/config.py` — Pydantic v2 `BaseSettings`-based `Settings` class with all Phase 1+5
  knobs (upload size, extraction strategy, bankroll, Kelly fraction, CVaR alpha, etc.)
- `pyproject.toml` — project metadata, runtime+dev deps, pytest config with `pythonpath = ["."]`
- Local `.venv` provisioned (Python 3.14.2) with: pydantic 2.13, pydantic-settings 2.14,
  structlog 25.5, pdfplumber 0.11.9, pypdf 6.11, pytest 9.0, reportlab 4.5
- `tests/test_parser/test_cleaner.py` — 102 tests covering every public function in `cleaner.py`
- `tests/test_parser/test_brisnet_parser.py` — 45 tests against synthetic Brisnet UP text:
  card structure, race header (number/date/distance/surface/condition/race-type/purse/claiming),
  horse entries (post/name/ML/jockey/trainer/weight/ml_implied_prob), PP lines (date order,
  speed figures, fractions, finish position, track code, days-since-prev), parse_confidence,
  multi-race input, degenerate inputs
- `tests/test_parser/test_extractor.py` — 25 tests: format detection signatures, parser
  dispatch, size guard, pdfplumber text extraction via reportlab-generated PDF fixture,
  ingest_pdf shape (timing, source_filename, page count, zero-race failure, corrupt bytes)

**Bug Fixes (found by writing tests):**
- `cleaner._FRACTION_MAP` was missing `1/16` and `3/16` — the most common US route fractions.
  Added 1/16, 3/16, 5/16, 7/16, 3/8, 7/8, 1/3, 2/3. `_DISTANCE_RE` alternation updated.
- `cleaner.normalize_text` was collapsing multi-space runs to a single space, which broke
  `BrisnetParser._RE_HORSE_LINE`'s `\s{2,}` column separator (every horse line would
  silently fail to match). Behavior changed: 3+ space runs cap at 2 spaces, preserving
  columnar alignment. Added `collapse_whitespace()` for callers that genuinely want
  single-space output (used by `clean_name`).
- `brisnet_parser._RE_DISTANCE` accepted only `\d/\d` fractions (single digit), so it
  misextracted "1 1/16 Miles" as "16 Miles" → out-of-range distance → ValidationError.
  Widened to `\d{1,2}/\d{1,2}` and made leading whitespace optional.
- `brisnet_parser._RE_CLAIMING` pattern `Clm(?:aiming)?` was buggy: "Claiming" starts with
  `Cla`, not `Clm`, so the optional `(?:aiming)?` group was meaningless. The regex never
  matched the full word — only the abbreviation in PP lines. Replaced with
  `Cl(?:aiming|m)\b` plus explicit `\$?` so dollar signs aren't required.
- `BrisnetParser._parse_race_header` would crash with Pydantic ValidationError when distance
  couldn't be extracted (default `0.0` violates `ge=2.0`). Now returns `None` when distance
  is invalid; `_parse_race_block` skips the race.
- `extractor.ingest_pdf` was failing to populate `processing_ms` on early-return failure
  paths (size guard, extraction error, parse exception). Now always set via a closure helper.

**In Progress:**
- None — Phase 1 parser milestone complete and verified

**Not Started (Phase 1 remaining):**
- `app/core/logging.py` (structlog setup — currently each module calls `structlog.get_logger`
  directly with default config; explicit init lets us flip JSON vs. pretty by environment)
- `app/main.py` (FastAPI app factory)
- `app/api/v1/ingest.py` (upload endpoint; needs `run_in_executor` wrap on `ingest_pdf`)
- `app/db/models.py`, `app/db/session.py` (SQLAlchemy async; storage for ingested cards)
- `app/services/pdf_parser/equibase_parser.py` (currently falls back to BrisnetParser)
- `.env.example`
- Real-PDF validation: load an actual Brisnet UP card from a known source, sanity-check
  parse output, tune regexes to whatever shape pdfplumber actually produces

**Key Decisions Made:**
- Repo layout: keep current flat `app/` at root (no `backend/` wrapper). CLAUDE.md's
  documented `backend/` layout is aspirational — migration deferred to avoid mid-phase churn.
- Test scope split: `test_brisnet_parser.py` tests the parser against pre-extracted text
  (deterministic). `test_extractor.py` asserts only on pipeline shape for end-to-end PDF
  bytes, NOT on full parse correctness — because pdfplumber's `extract_text(x_tolerance=3,
  y_tolerance=3)` may collapse columns differently than our hand-crafted fixture, and we
  want parser tests to remain stable independent of pdfplumber version drift.
- `normalize_text` semantics changed from "single-space collapse" to "cap excess spaces at
  two" — preserves columnar gaps for downstream regex parsers. Documented in cleaner.py.
- Distance is treated as a load-bearing field for race headers: missing distance → race
  skipped from the card. Other missing fields (purse, claiming price, race type) degrade
  to None / UNKNOWN and the race still parses with lower confidence.

**Known Issues / Tech Debt:**
- `_RE_RACE_HEADER` uses `RACE\s+\d` which can false-positive on horse names like
  "DISGRACE 1" (contains "RACE 1" as substring). Need `\bRACE\b` anchor. Not blocking
  for current synthetic tests, but should fix before real-PDF validation.
- `extract_text_from_pdf` docstring promises pdfplumber layout mode then text mode then
  pypdf fallback, but the implementation only passes `x_tolerance`/`y_tolerance` (no
  `layout=True`). This may matter for real Brisnet PDFs where columns are spaced widely.
- Source-of-truth docs `Horse_Racing_System_Master_Reference.md` and
  `Horse_Racing_Betting_System_Research.pdf` referenced in CLAUDE.md do not exist in
  the repo. Working from CLAUDE.md + DECISIONS.md only.

**Blockers:**
- None

**Tests Status:**
- 172 tests passing in ~1.1s. Run with `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: [PREVIOUS — pre-test-suite scaffolding]

**Completed:**
- `app/schemas/race.py` — all Pydantic v2 schemas:
  `PastPerformanceLine`, `HorseEntry`, `RaceCard`, `ParsedRace`,
  `RaceHeader`, `IngestionResult`, `IngestionStatus`
- `app/services/pdf_parser/cleaner.py` — full normalization pipeline:
  `normalize_text`, `clean_name`, `parse_odds_to_decimal`,
  `parse_distance_to_furlongs`, `parse_time_to_seconds`,
  `parse_surface`, `parse_condition`, `parse_race_type`,
  `extract_first_number`, `extract_claiming_price`
- `app/services/pdf_parser/extractor.py` — orchestrator:
  three-pass extraction (pdfplumber layout → text → pypdf),
  format detection, parser dispatch, `ingest_pdf` entry point
- `app/services/pdf_parser/brisnet_parser.py` — Brisnet UP parser:
  page segmentation, race header parsing, horse entry parsing,
  PP line extraction (full regex + positional fallback),
  parse confidence scoring

**Key Decisions Made:**
- Brisnet UP is the Phase 1 primary format; DRF and Equibase are stubs that fall back to BrisnetParser
- Three-pass PDF extraction: pdfplumber layout → pdfplumber text → pypdf character-level
- `parse_confidence` is a weighted composite: 40% header completeness + 60% PP coverage
- `race_number` in `PastPerformanceLine` is set to 1 as a placeholder (not reliably extractable from Brisnet line format)
- Odds stored as decimal throughout (fractional strings converted on ingest)

---

## Phase Completion Checklist

### Phase 1: PDF Ingestion
- [ ] `app/core/config.py` — scaffold only; full BaseSettings present but logging/db settings unused yet
- [ ] `app/core/logging.py`
- [ ] `app/main.py`
- [ ] `app/api/v1/ingest.py`
- [ ] `app/db/models.py`
- [ ] `app/db/session.py`
- [x] `app/schemas/race.py`
- [x] `app/services/pdf_parser/cleaner.py`
- [x] `app/services/pdf_parser/extractor.py`
- [x] `app/services/pdf_parser/brisnet_parser.py`
- [ ] `app/services/pdf_parser/equibase_parser.py` (stub → implementation)
- [x] `tests/test_parser/test_cleaner.py` (102 tests)
- [x] `tests/test_parser/test_brisnet_parser.py` (45 tests)
- [x] `tests/test_parser/test_extractor.py` (25 tests)
- [x] `pyproject.toml`
- [ ] `backend/.env.example`

### Phase 2: Feature Engineering
- [ ] `app/services/feature_engineering/engine.py`
- [ ] `app/services/feature_engineering/speed_features.py`
- [ ] `app/services/feature_engineering/pace_features.py`
- [ ] `app/services/feature_engineering/class_features.py`
- [ ] `app/services/feature_engineering/connections.py`
- [ ] `app/services/feature_engineering/layoff.py`
- [ ] `tests/test_features/`

### Phase 3: Model Layer
- [ ] `scripts/bootstrap_models.py`
- [ ] `app/services/models/speed_form.py`
- [ ] `app/services/models/pace_scenario.py`
- [ ] `app/services/models/sequence.py`
- [ ] `app/services/models/connections.py`
- [ ] `app/services/models/market.py`
- [ ] `app/services/models/meta_learner.py`
- [ ] `tests/test_models/`

### Phase 4: Calibration + Ordering
- [ ] `app/services/calibration/calibrator.py`
- [ ] `app/services/calibration/drift.py`
- [ ] `app/services/ordering/plackett_luce.py`
- [ ] `app/services/ordering/stern.py`
- [ ] `app/services/ordering/copula.py`
- [ ] `scripts/validate_calibration.py`
- [ ] `tests/test_ordering/`

### Phase 5: EV Engine + Portfolio
- [ ] `app/services/ev_engine/calculator.py`
- [ ] `app/services/ev_engine/market_impact.py`
- [ ] `app/services/portfolio/optimizer.py`
- [ ] `app/services/portfolio/sizing.py`
- [ ] `app/api/v1/analyze.py`
- [ ] `app/api/v1/portfolio.py`
- [ ] `tests/test_portfolio/`

### Phase 6: Frontend
- [ ] Next.js project scaffold
- [ ] Upload flow
- [ ] Race card viewer
- [ ] Probability visualization
- [ ] Bet execution ticket
