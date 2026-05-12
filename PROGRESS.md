# PROGRESS.md — Session Log

Update this file at the end of every Claude Code session.
Format: newest session at the top.

---

## Current State

**Phase:** 0 — Master Training Database (**LOADED**) + Phase 1 — PDF Ingestion (paused at scaffolding milestone)
**Last completed task:** Master DB populated with **2,626,284 race results** across 336,967 races, 263,209 horses, 4 jurisdictions (UK / HK / JP / AR), 1986-2023. Zero duplicate dedup keys across every deduped table. ~80% of rows scored 0.85-0.95 at quality gate.
**Next task:** Export ML training parquet (`python scripts/db/export_training_data.py`), recommended with jurisdiction filter excluding AR until the AR splitting issue is addressed. Then bootstrap Phase 3 (`backend/scripts/bootstrap_models.py`) to train baseline Speed/Form LightGBM. After that: resume Phase 1 bootstrap (`app/core/logging.py`, `app/main.py`, `app/api/v1/ingest.py`).

---

## Session Log

### Session: 2026-05-12 — Bug-fix sweep + master DB populated (2.6M results loaded)

**Context:**
The 2026-05-11 (c) auto-ingest run (`auto_ingest_run1.json`) had loaded **0 rows out of 1.5M+ downloaded**.
Six datasets registered in the DB had `row_count_ingested = NULL` because every row failed at map+clean
or quality_gate. This session diagnosed root causes, fixed them, then hand-mapped the highest-quality
datasets to populate the DB end-to-end.

**Bugs found and fixed (in priority order):**

1. **`race_number` Pydantic-required but missing in most non-US datasets.** Many datasets identify races
   by `race_id` (string), `card_id`, or post-time rather than an explicit numeric race number. Fix:
   made `race_number` `Optional` in `CanonicalRace`, added it as a load-required hard failure in
   `quality_gate` (zero score if None — same justification as distance/surface/jurisdiction per
   ADR-011 / ADR-019).

2. **Heuristic regex `(horse|name)` matched `race_name` / `Course name` before `horse_name`.**
   First-match-wins with greedy `name` was catastrophically wrong. Fix: replaced single regex per field
   with `_FieldSpec(exact=[priority list], fuzzy=optional, blocklist=[])`. Tries exact-match (case-
   insensitive) first, then fuzzy with explicit blocklist for known-wrong matches like `post_position`
   / `course_name`. See ADR-018.

3. **`normalize_name()` stripped all non-ASCII characters.** `[^a-z0-9 ]` regex turned `ワクセイ`
   (JRA katakana) into empty string, which then hit "missing horse_name" + "duplicate horses within
   race: ['']" — JRA's 1.6M rows ALL failed. Fix: regex is now `[^\w ]` with `re.UNICODE`, preserving
   CJK / Cyrillic / accented Roman names while still stripping `'`, `-`, `.` from English names. See
   ADR-017.

4. **Quality_gate's race-level grouping disagreed with `race_dedup_key`.** quality_gate grouped by
   `(date, track, race_num)`, but the dedup_key includes `(date, track, race_num, distance, surface)`.
   Argentine venues that run turf + dirt cards in parallel reuse `nro` across distinct physical races.
   Quality_gate flagged these as "mixed distances" / "duplicate horses" violations even though they
   were correctly stored as separate races by load_to_db. Fix: quality_gate now groups by all 5
   dedup_key fields. The "mixed distances within race" check was removed (it's a no-op once you group
   by distance). See ADR-016.

5. **Auto-built field maps had `transformers: {}`.** Surfaces stayed as raw "Heavy"/"Good"/"GOOD TO
   FIRM"; odds stayed as fractional "5/2"; both broke the dedup_key. Fix: `build_heuristic_map` now
   auto-attaches default transformers (parse_odds_to_decimal, normalize_surface or uk_going_to_surface,
   normalize_condition or uk_going_to_condition, time_string_to_seconds, stones_to_lbs) based on what
   columns it found.

**New infrastructure added:**

- **`scripts/db/preprocessors.py`** — registry of multi-CSV merge functions. Datasets that ship as
  `races.csv` + `runs.csv` (joined on `race_id`) declare a `preprocess` field in their FIELD_MAPS
  entry; map_and_clean.py calls the named function instead of `_pick_primary_csv`. See ADR-015.
  Two preprocessors implemented: `gdaley_hkracing_merge`, `lantanacamara_hk_merge`.

- **6 new transformers** in `transformers.py`:
  - `time_string_to_minutes` — parses "HH:MM" into minutes-since-midnight (race_number proxy when
    only post-time is available, e.g. UK sheikhbarabas)
  - `extract_int` — pulls first contiguous integer from a string (for `RID1002-IE-05` → 1002)
  - `kg_to_lbs` — for JRA carried weights (in kg)
  - `jpn_surface_to_canonical` — maps `芝` → turf, `ダート` → dirt
  - `jpn_condition_to_canonical` — maps `良` / `稍重` / `重` / `不良` to fast/good/soft/heavy
  - `ar_surface_to_canonical` — maps Spanish `arena`/`cesped`/`sintetico` to dirt/turf/synthetic

**Hand-written field maps added (5 new):**
- `sheikhbarabas/horse-racing-results-uk-ireland-2005-to-2019` (UK/IE, ~744K rows; race_number
  synthesized from post-time via `time_string_to_minutes`)
- `gdaley/hkracing` (HK, ~80K rows; multi-CSV merge; only IDs for horse/jockey/trainer)
- `lantanacamara/hong-kong-horse-racing` (HK, ~30K rows; multi-CSV merge; real names)
- `takamotoki/jra-horse-racing-dataset` (JP, ~1.6M rows; Japanese column names)
- `felipetappata/thoroughbred-races-in-argentina` (AR, ~323K rows; Spanish conventions; uses
  `jockey_weight` not `weight` to avoid body-weight-vs-carried-weight contamination)

**Master DB final state:**

| Jurisdiction | Races   | Results    | Date range          | Source                      |
|--------------|---------|------------|---------------------|-----------------------------|
| JP           | 121,785 | 1,609,930  | 1986-01 → 2021-07   | takamotoki/jra              |
| AR           | 137,344 | 306,290    | 2016-06 → 2023-10   | felipetappata               |
| UK           |  69,135 | 601,405    | 2005-04 → 2019-12   | sheikhbarabas               |
| HK           |   8,703 | 108,659    | 1997-06 → 2017-07   | gdaley + lantanacamara      |
| **Total**    | **336,967** | **2,626,284** | **1986 → 2023** | **5 hand-mapped datasets**  |

Quality score distribution across all 2.6M results:
- 0.95+ (excellent): 130,231 (5%)
- 0.85-0.95: 1,965,861 (75%)
- 0.75-0.85: 71,170 (3%)
- 0.60-0.75 (minimum pass): 459,022 (17%)

**Auto-ingest #2 results (with bug-fixes):** 0 additional rows loaded. The remaining 21 low_score
datasets had column conventions the heuristic still couldn't infer; the 7 no_csv datasets were
non-CSV; the 4 evaluate-failed had encoding/parsing errors; the others (noqcks workouts, prashant111
tipster bets, mexwell HK dividends-only) are not actually race-results data despite scoring well at
the column-presence evaluator.

**Known caveats** (documented for Phase 3 model training):
- **Argentina has structural data-quality issues.** Avg field size of 2.2 is unrealistically low
  (real races ~10 horses) — the `nro` field is reused across multiple physical races at the same
  venue/date, and even with distance+surface in the dedup_key the resulting groups are fragmented.
  AR has zero odds data. Some `jockey_weight` values are contaminated with body weight (max 1318 lbs).
  Mean score 0.66. Recommend filtering AR out of model training until a venue-specific course
  identifier can be derived.
- **HK has two non-overlapping naming conventions.** gdaley uses numeric IDs ("3875") for horses
  + "ST"/"HV" tracks (1997-2005); lantanacamara uses real names ("DOUBLE DRAGON") + "Sha Tin" /
  "Happy Valley" (2014-2017). Won't cross-dedup, but date ranges don't overlap.
- **`tracks` table is empty.** load_to_db never populates it. Aspirational table for now.
- **JRA `purse(万円)` left raw** (units of 10,000 JPY rather than USD). quality_gate doesn't validate
  purse so this isn't blocking — the field is informational for now.

**Tests Status:**
- 234 tests passing in ~3.0s. The mixed-distances test was rewritten to assert the NEW behavior
  (different distances at the same nro = separate races, not a violation).

---

### Session: 2026-05-11 (c) — Auto-discovery ingestion + bug fixes

**Completed:**
- `scripts/db/auto_ingest.py` — Kaggle-keyword-driven bulk orchestrator. Searches Kaggle
  for configurable keywords (default: "horse racing", "horse bet", "horseracing", "horse
  race results", "thoroughbred", "horse betting"), deduplicates results across keywords,
  and runs each unique slug through the full pipeline:
  download → evaluate → map+clean → quality_gate → load. Skips slugs already present
  in the `datasets` table (idempotent). Emits a structured per-slug `SlugResult` with
  status codes: `loaded` / `already_ingested` / `low_score` / `needs_map` / `no_csv` /
  `error`.
- Two field-map modes: **strict** (default) processes only datasets with a hand-written
  entry in `field_maps.FIELD_MAPS`; for unregistered slugs it logs the actual CSV column
  names so you can write a new entry and re-run. **`--auto-map`** opt-in builds a
  synthetic field map at runtime via the heuristic regex matchers from
  `evaluate_dataset.py`. Heuristic mode is intentionally lossy and the user is told to
  inspect `rejected/reasons.jsonl` after each run.
- True `--dry-run` mode — discovery only, zero downloads, zero DB writes. Returns a
  preview list with Kaggle metadata (size, downloads, votes, last_updated, url,
  matched_keyword) plus `already_ingested` and `has_field_map` flags per slug.
- No automatic cleanup — staging/cleaned/parquet artifacts are retained on disk for
  audit (per user preference; trades disk space for debuggability).

**Bug fixes (surfaced by smoke tests + dry-run):**
- `evaluate_dataset._HEURISTICS` regex `\b` boundaries did not fire on snake_case
  columns. Python's `\b` is a transition between `\w` (which includes `_`) and `\W`,
  so `\b(date)\b` cannot match `race_date` (no boundary between `e` and `_`). Fixed
  by replacing `\b` with explicit lookarounds `(?<![a-zA-Z0-9])...(?![a-zA-Z0-9])`,
  which treat `_` as a separator. This was breaking heuristic mapping for the majority
  of real Kaggle datasets.
- Initial `auto_ingest.discover_slugs` mis-used Kaggle's `dataset_list(max_size=N)` —
  `max_size` is a **byte-size filter** (max bytes per dataset), not a result-count cap.
  My first dry-run returned only 1 result because I was filtering to datasets ≤ 20
  bytes. Fixed by replacing `max_size` with proper page-by-page iteration via
  `_search_paginated` until `max_per_keyword` results are accumulated.
- Initial `--dry-run` flag was a no-op at the CLI level — it still downloaded every
  dataset and wrote `datasets` rows to the DB before bailing. Restructured so dry-run
  short-circuits at the orchestration level (in `auto_ingest()`) and returns
  discovery-only results without entering the per-slug processing loop.

**Dry-run Findings:**
- 38 unique horse-racing datasets discovered; estimated ~3.4 GB total download
- **None of the 3 datasets currently registered in `field_maps.py`
  (`joebeachcapital/horse-racing`, `zygmunt/horse-racing-dataset`,
  `gdaley/horseracing-in-hk`) appeared in the search results** — they may have been
  renamed or removed. `gdaley/hkracing` did appear and is likely the renamed HK
  dataset. The DATA_PIPELINE.md §4 dataset catalog needs updating.
- High-quality candidates by community signal (downloads):
  `gdaley/hkracing` (9628 dl), `lantanacamara/hong-kong-horse-racing` (5429 dl),
  `hwaitt/horse-racing` (5091 dl), `takamotoki/jra-horse-racing-dataset` (3159 dl),
  `deltaromeo/horse-racing-results-ukireland-2015-2025` (2780 dl, 1.1 GB),
  `eonsky/betfair-sp` (1016 dl, 1.2 GB)
- Junk that will be auto-rejected:
  `seniruepasinghe/horse-racing-player-detection-yolo11` (image dataset → no_csv),
  `thedevastator/major-us-sports-venues-usage-and-affiliations`,
  `quantumgoat/predict-horse-price`, several "winners-only" tiny datasets

**Key Decisions Made:** (full rationale in DECISIONS.md ADR-013, ADR-014)
- Auto-ingest uses a **hybrid mode**: strict by default for data-quality safety,
  `--auto-map` opt-in for bulk-build velocity. Both modes leave the DB in a clean state.
- Dry-run is **discovery-only** (no downloads, no DB writes). Two tiers of "preview"
  rejected — single tier with explicit followup is simpler.
- Heuristic auto-mapping uses **column-name regex only**; no value sniffing for unit
  detection. Distance is assumed already in furlongs; odds assumed already decimal.
  Quality gate's range checks (2.0-20.0 furlongs, 1.0+ decimal odds) are the
  unit-mismatch safety net.

**Tests Status:**
- 233 tests passing in ~2.9s (172 Phase 1 + 61 Phase 0). No new tests for `auto_ingest.py`
  itself — it's an orchestration layer over fully-tested components, and stubbing the
  Kaggle API would be more cost than value.

---

### Session: 2026-05-11 (b) — Phase 0 master DB pipeline (code complete)

**Completed:**
- `pyproject.toml` — added Phase 0 deps: `kaggle>=1.6`, `pandas>=2.2`, `pyarrow>=15.0`
- `scripts/db/schema.sql` — 7 idempotent CREATE TABLE / CREATE INDEX statements
  (datasets, tracks, races, horses, jockeys, trainers, race_results) with UNIQUE(dedup_key)
  on every dedup target
- `scripts/db/constants.py` — paths, schema version, quality thresholds, source priority dict
- `scripts/db/setup_db.py` — runs schema.sql via `executescript`; idempotent (re-runs are no-ops)
- `scripts/db/dedup.py` — SHA-256 dedup keys per DATA_PIPELINE.md §3:
  `race_dedup_key` (track|date|race#|round(distance,2)|surface),
  `horse_dedup_key` (normalized_name|year|country, with "unknown" fallbacks),
  `person_dedup_key` (jockeys/trainers — normalized_name|jurisdiction),
  `result_dedup_key` (race_key|horse_key)
- `scripts/db/transformers.py` — full unit-conversion library (metres↔furlongs, UK distance
  parser supporting `1m4f 110y`, stones↔lbs, fractional/EVS odds, UK SP with favourite-marker
  stripping, surface/condition normalization including all-weather track recognition,
  time-string parser, fixed-rate FX). All transformers return None on garbage rather than
  raising — quality gate is the gatekeeper.
- `scripts/db/field_maps.py` — registry for joebeachcapital/horse-racing (US, active),
  zygmunt/horse-racing-dataset (UK, drafted), gdaley/horseracing-in-hk (HK, drafted).
  Value semantics: `"col"` = column ref, `{"const": v}` = literal, `None` = NULL
- `scripts/db/schemas.py` — Pydantic v2 canonical models: `CanonicalRace`, `CanonicalHorse`,
  `CanonicalPerson` (jockeys/trainers share shape), `CanonicalRaceResult` with nested
  race/horse/jockey/trainer + `to_parquet_dict()` flattener (parent prefixes for nested fields)
- `scripts/db/ingest_kaggle.py` — Kaggle download + `datasets` table registration with
  4-tier credential resolution: `--credentials` path → env vars → `~/.kaggle/kaggle.json` →
  `~/.kaggle/access_token` (JSON or raw key). Writes `_dataset_id` sidecar in staging dir
  so downstream scripts can pick up the FK without re-running.
- `scripts/db/evaluate_dataset.py` — DATA_PIPELINE.md §6 rubric (weighted score, threshold
  0.70). Heuristic regex column matching when no `--slug`; field-map lookup when slug given.
  Emits JSON report (score, field_coverage, jurisdiction_guess, warnings, date range,
  estimated races/results).
- `scripts/db/map_and_clean.py` — staging CSV → CanonicalRaceResult validation → parquet.
  Writes `cleaned/<slug>/all.parquet` (every row that passes Pydantic) plus
  `rejected_pydantic.jsonl` for diagnostics. Lenient — only rejects on hard schema
  failures; quality gate handles the rest.
- `scripts/db/quality_gate.py` — DATA_PIPELINE.md §8 scoring + cross-row race-level
  validation (multiple winners, mixed distances/surfaces, duplicate horses). Splits parquet
  into `accepted/all.parquet` and `rejected/all.parquet` + `reasons.jsonl`. Sets
  `data_quality_score` on every row before split.
- `scripts/db/load_to_db.py` — idempotent loader: dependency-ordered upserts
  (horses → jockeys → trainers → races → race_results) with `INSERT OR IGNORE` on
  every dedup_key. Updates `datasets.row_count_ingested/deduped/date_range_*` after each load.
- `scripts/db/dedup_report.py` — per-table totals + duplicate-key counts (must always be 0)
  + race_results quality-score distribution + dataset audit-trail summary.
- `scripts/db/export_training_data.py` — runs DATA_PIPELINE.md §12 SQL verbatim, parametrized
  by `--min-score` and `--min-field-size`, writes `data/exports/training_<YYYYMMDD>.parquet`.

**Test Suite (61 new tests, 233 total passing):**
- `tests/test_db/test_dedup.py` — 11 tests: hash stability across `date`/string inputs,
  case-insensitivity, distance rounding, distinguishing collisions, name normalization
- `tests/test_db/test_transformers.py` — 27 tests: every transformer's happy path,
  None-on-garbage contract, registry consistency, field_maps→transformer registry validation
- `tests/test_db/test_quality_gate.py` — 20 tests: every hard failure (parametrized),
  every soft penalty (exact deduction values), every range check, score clamping at 0,
  cross-row violations (multiple winners, mixed distances, duplicate horses)
- `tests/test_db/test_pipeline.py` — 3 end-to-end tests: synthetic CSV through full
  pipeline; quality_score persisted; dedup keys stable across re-runs (idempotency)

**Schema Adjustment:**
- `CanonicalRace.distance_furlongs` / `surface` / `jurisdiction` made `Optional` so
  map_and_clean produces all rows for the quality gate to score (rather than dropping
  rows with missing soft fields at Pydantic validation time). The quality gate then
  treats these as load-required hard failures (zero score) — they're NOT NULL in SQL
  and used in the race dedup key, so a row missing them is fundamentally unloadable.

**Key Decisions Made:**
- `scripts/db/` is fully standalone — no imports from `app/`. Per CLAUDE.md §11, this
  keeps the training pipeline runnable without standing up FastAPI.
- stdlib `sqlite3` instead of SQLAlchemy in `scripts/db/` — Phase 0 has no async
  requirements and the schema is small enough that ORM overhead is pure cost.
- Bootstrap pattern in every CLI script: `if __package__ in (None, ""): sys.path.insert(0, ...)`
  so all scripts work as both `python scripts/db/foo.py` and `python -m scripts.db.foo`.
- Field-map value `{"const": value}` syntax disambiguates literal values from column
  references — small deviation from DATA_PIPELINE.md §5 sketch (which inlined raw
  values) but cleaner because it avoids clobbering valid column names like `"beyer"`.
- Source priority logic (Equibase > Brisnet > DRF > Kaggle) is enforced implicitly via
  `INSERT OR IGNORE`: whichever dataset is loaded first wins. To override, drop the
  conflicting row from the lower-priority dataset and re-run. (Future improvement: add
  an explicit `source_priority` column and a per-table `MERGE`-style upsert.)
- `quality_gate.py` adds a load-required hard-failure on missing distance/surface/
  jurisdiction even though DATA_PIPELINE.md §8 calls them soft failures — these fields
  are required for the race dedup key + SQL load, so they cannot be soft.

**Not Started (Phase 0 remaining — runtime work, not code):**
- Live run against `joebeachcapital/horse-racing`: download via `ingest_kaggle.py`,
  evaluate, map+clean (will surface CSV column-name mismatches with the field map —
  these go straight back into `field_maps.py`), quality-gate, load.
- After load: confirm `dedup_report.py` shows zero duplicate keys and a sane score
  distribution; run `export_training_data.py` and inspect the parquet.

**Tests Status:**
- 233 tests passing in ~3.0s. Run with `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

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

### Phase 0: Master Training Database
- [x] `scripts/db/schema.sql`
- [x] `scripts/db/constants.py`
- [x] `scripts/db/setup_db.py`
- [x] `scripts/db/dedup.py`
- [x] `scripts/db/transformers.py`
- [x] `scripts/db/field_maps.py`
- [x] `scripts/db/schemas.py`
- [x] `scripts/db/ingest_kaggle.py`
- [x] `scripts/db/evaluate_dataset.py`
- [x] `scripts/db/map_and_clean.py`
- [x] `scripts/db/quality_gate.py`
- [x] `scripts/db/load_to_db.py`
- [x] `scripts/db/dedup_report.py`
- [x] `scripts/db/export_training_data.py`
- [x] `scripts/db/auto_ingest.py` (Kaggle keyword-driven bulk orchestrator)
- [x] `scripts/db/preprocessors.py` (multi-CSV merge hook for map_and_clean)
- [x] `tests/test_db/test_dedup.py` (11 tests)
- [x] `tests/test_db/test_transformers.py` (27 tests)
- [x] `tests/test_db/test_quality_gate.py` (21 tests)
- [x] `tests/test_db/test_pipeline.py` (3 end-to-end tests)
- [x] **Live run: master DB populated with 2.6M results across UK / HK / JP / AR (1986-2023)**
- [ ] Export ML training parquet via `scripts/db/export_training_data.py` (recommend `--jurisdictions UK,HK,JP` until AR splitting fixed)
- [ ] Future: derive a venue-specific course identifier for AR to fix the over-split races issue
- [ ] Future: populate `tracks` table from existing `races` data (currently unused)

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
