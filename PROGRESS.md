# PROGRESS.md — Session Log

Update this file at the end of every Claude Code session.
Format: newest session at the top.

---

## Current State

**Phase:** 1 — PDF Ingestion Pipeline
**Last completed task:** Initial schema and parser scaffolding
**Next task:** Finalize BrisnetParser + write Phase 1 test suite

---

## Session Log

### Session: [DATE — fill in]

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

**In Progress:**
- BrisnetParser validation against real PDFs (not yet tested on live files)
- `equibase_parser.py` and `drf_parser.py` are stubs (BrisnetParser used as fallback)

**Not Started (Phase 1 remaining):**
- `app/core/config.py` (Pydantic BaseSettings)
- `app/core/logging.py` (structlog setup)
- `app/main.py` (FastAPI app factory)
- `app/api/v1/ingest.py` (upload endpoint)
- `app/db/` (SQLAlchemy models + session)
- `tests/test_parser/` (full test suite for cleaner + parser)
- `pyproject.toml`
- `backend/.env.example`

**Key Decisions Made:**
- Brisnet UP is the Phase 1 primary format; DRF and Equibase are stubs that fall back to BrisnetParser
- Three-pass PDF extraction: pdfplumber layout → pdfplumber text → pypdf character-level
- `parse_confidence` is a weighted composite: 40% header completeness + 60% PP coverage
- `race_number` in `PastPerformanceLine` is set to 1 as a placeholder (not reliably extractable from Brisnet line format)
- Odds stored as decimal throughout (fractional strings converted on ingest)

**Blockers:**
- None

**Tests Status:**
- No tests written yet for Phase 1

---

## Phase Completion Checklist

### Phase 1: PDF Ingestion
- [ ] `app/core/config.py`
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
- [ ] `tests/test_parser/test_cleaner.py`
- [ ] `tests/test_parser/test_brisnet_parser.py`
- [ ] `tests/test_parser/test_extractor.py`
- [ ] `pyproject.toml`
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