# PROGRESS.md — Session Log

Update this file at the end of every Claude Code session.
Format: newest session at the top.

---

## Current State

**Phase:** 0 — Master Training DB **LOADED + EXPORTED** · Phase 1 — PDF Ingestion **COMPLETE** · Phase 2 — Feature Engineering **COMPLETE** · Phase 3 — Baseline Models **COMPLETE (Speed/Form + Connections + Market + Meta) · Pace/Sequence stubs**
**Last completed task:** Trained the Phase-3 baseline model stack on the full 2,317,297-row training parquet. **Speed/Form** (LightGBM, 24 features) hit `val_auc=0.745`, `val_race_top1_acc=0.257` (vs ~0.083 random for ~12-horse fields). **Meta-learner** (LightGBM stacker over orthogonalised sub-model outputs) hit `val_log_loss=0.237`, `val_race_top1_acc=0.341` — 34% winner-pick rate on a held-out 10% time-slice (2018-02-19 → 2021-07-31). Connections model fit 2,291 jockey + 2,719 trainer + 50,251 pair shrunken rates. Artifacts at `models/baseline_full/`. **343 tests passing** (was 306 → +37 new: 15 training_data + 10 speed_form + 7 connections + 5 market).
**Next task:** Phase 4 — calibration. Wrap `SpeedFormModel.predict_proba` (and the meta-learner output) in `app/services/calibration/calibrator.py` with Platt / isotonic selectors fit on a held-out slice. Build the reliability diagram + ECE report in `scripts/validate_calibration.py`. Then start Plackett-Luce MLE in `app/services/ordering/plackett_luce.py`.

### Known Export Caveats (for Phase 3 model training)

Column null rates in `training_20260512.parquet`:

| Column                                                                                          | % rows null | Cause                                                       |
|-------------------------------------------------------------------------------------------------|------------:|-------------------------------------------------------------|
| `claiming_price`, `speed_figure_source`, `beaten_lengths_q1/q2`, `fraction_q1_sec/q2_sec`, `sire`, `dam_sire`, `foaling_year` | **100%**    | Not populated by any current source dataset.                |
| `field_size`                                                                                    | 74%         | Only UK populates it. JP/HK NULL — derive via group counts. |
| `speed_figure`                                                                                  | 72%         | JP has zero speed figures (1.6M rows). UK has 96%, HK 73%.  |
| `post_position`                                                                                 | 26%         | All 598,708 UK rows. sheikhbarabas dataset omits it.        |
| `purse_usd`                                                                                     | 1.5%        | Sparse on some HK/JP rows.                                  |
| `fraction_finish_sec`                                                                           | 1.3%        | Sparse.                                                     |

UK `race_number` is a **post-time-minutes-past-midnight proxy** from `time_string_to_minutes` — within-card ordering works (sorts by post time) but won't be a contiguous 1..N sequence. Phase 3 should recompute a 1..N race number per `(track_code, race_date)` group if needed.

JP carries 1.6M rows with no speed figure. Either derive a synthetic speed figure from `(distance_furlongs, fraction_finish_sec, condition)` for JP, or restrict speed-figure-dependent models to UK+HK and rely on other features for JP.

---

## Session Log

### Session: 2026-05-12 (d) — Phase 3 baseline model stack trained

**Completed:**

*Sub-model framework (all under `app/services/models/`)*
- `training_data.py` — leakage-free feature prep. `groupby('horse_key').shift(1)`
  precedes every per-horse rolling aggregate (`ewm_speed_prior`, `last_speed_prior`,
  `best_speed_prior`, `mean_speed_prior`, `speed_delta_prior`, `mean_finish_pos_prior`,
  `win_rate_prior`, `mean_purse_prior`, `days_since_prev`, `layoff_fitness`). EWM
  uses `alpha=0.4` (CLAUDE.md §8). Adds field-relative columns
  (`ewm_speed_zscore`, `ewm_speed_rank`, `ewm_speed_pct`, `weight_lbs_delta`) and
  derives `field_size` from in-frame group counts when the source column is NULL
  (which is 74% of the parquet — see "Known Export Caveats"). Exposes
  `time_based_split(df, val_fraction)` per CLAUDE.md §2's no-random-split rule.
- `speed_form.py` — `SpeedFormModel` (Layer 1a). LightGBM binary classifier on
  `win`. Per-race softmax over RAW scores (not sigmoided) for the in-race
  probability distribution. Save/load round-trip via LightGBM's native text
  format + a JSON metadata sidecar.
- `connections.py` — `ConnectionsModel` (Layer 1d). Empirical-Bayes shrinkage
  estimator (beta-binomial pseudo-count `α=30`) over per-jurisdiction baseline
  → per-jockey → per-trainer → per-pair win rates. Inference falls back through
  the hierarchy when a pair/jockey/trainer is OOV. JSON serialisation.
- `market.py` — `MarketModel` (Layer 1e). Implied-prob from `odds_final` → in-race
  normalised → isotonic-regression calibration over historical wins. NaN for
  rows where `odds_final` is missing (caller decides fallback).
- `pace_scenario.py` + `sequence.py` — placeholder classes returning constant
  0.5 from `predict_proba`. Their `fit()` raises `NotImplementedError` until
  unblock criteria are met (pace: fractional times in parquet; sequence:
  PyTorch + globally-unique horse_id). Documented inline.
- `meta_learner.py` — `MetaLearner` (Layer 2). LightGBM head over the 5 sub-model
  outputs + meta features (`field_size`, `distance_furlongs`).
  Orthogonalisation (CLAUDE.md §2): every sub-model column except the
  anchor (`speed_form_proba`) is replaced with its residual after linear
  regression on the anchor. Cleanly separates "what the anchor already
  knows" from each layer's marginal contribution.

*Bootstrap orchestrator*
- `scripts/bootstrap_models.py` — end-to-end: load parquet → prepare features →
  time-split → fit Speed/Form + Connections + Market → stack predictions →
  fit MetaLearner → persist artifacts + summary JSON. Supports
  `--sample-frac` for smoke tests, `--run-name` for reproducible run dirs.

*Tests*
- `tests/test_models/_synth.py` — synthetic data generator producing a 480-row
  fixture where `speed_figure` predicts `finish_position` (with noise) so trained
  models have real signal to learn.
- `tests/test_models/test_training_data.py` (15 tests) — no-leakage invariant,
  field-size derivation, EWM math sanity, categorical dtype, time-split
  boundaries, parquet round-trip.
- `tests/test_models/test_speed_form.py` (10 tests) — fit/predict shape,
  val_top1_acc > random, softmax sum-to-one per race, numerically-stable softmax
  on large inputs, save/load round-trip with identical predictions.
- `tests/test_models/test_connections.py` (7 tests) — predict shape, OOV
  fallback to jurisdiction baseline, shrinkage bounded in [0, 1], save/load.
- `tests/test_models/test_market.py` (5 tests) — implied-prob math,
  non-positive odds → NaN, missing-odds rows → NaN predictions.

*Live bootstrap run on full 2.3M-row parquet*

| Model         | Train log loss | Val log loss | Val AUC | Val race top-1 |
|---------------|---------------:|-------------:|--------:|---------------:|
| Speed/Form    | 0.254          | 0.262        | 0.745   | **0.257**       |
| Meta-learner  | —              | **0.237**    | —       | **0.341**       |

Field size ≈ 12 horses on average, so random pick ≈ 0.083. Meta-learner is
**~4× better than random** at picking the winner. Stacking provided a
meaningful lift over Speed/Form alone (top-1 0.257 → 0.341, log-loss 0.262 → 0.237) —
the orthogonalised connections + market columns carry real information.

Artifacts persisted at `models/baseline_full/`:
  speed_form/{booster.txt, metadata.json}
  connections/model.json (2,291 jockeys / 2,719 trainers / 50,251 pairs)
  market/model.json
  meta_learner/{booster.txt, metadata.json}
  summary.json

**Bug fixes (during test authoring):**
- The naive cross-row AUC assertion on synthetic data was misleading — speed
  figures vary widely across small synthetic races, so cross-race AUC is
  near-chance even when the model perfectly identifies WITHIN-race winners.
  Test swapped to `val_race_top1_accuracy > 0.40`, which is the metric that
  actually matters for the wagering use case.
- Shrunken rate boundary test was too strict (`0 < v < 1`): when both the
  observed sample and the prior are uniform-1.0, the posterior mean can
  legitimately touch the boundary. Relaxed to `0 <= v <= 1` and added a
  separate "average rate close to jurisdiction baseline" check that actually
  exercises the shrinkage strength.

**Key Decisions Made:**
- **Softmax operates on raw LightGBM scores, not sigmoided probabilities.**
  Sigmoid-then-renormalise double-squashes the dynamic range and destroys the
  signal at the tails (longshots and chalk). Raw-score softmax preserves the
  additive structure the model learned and produces a calibratable
  distribution. Phase 4 calibration sits on top.
- **Orthogonalisation lives in the meta-learner module, not the sub-models.**
  Each sub-model emits its raw output. The stacker decides what's orthogonal to
  what — keeps the sub-models composable and lets us swap in a different
  meta-learner (logistic regression, MLP) without touching Layer 1.
- **Pace and Sequence are deferred via STUB classes returning 0.5.** Without
  the fractional time columns (Pace) or PyTorch + a globally-unique horse_id
  (Sequence), training them now would be busywork. The meta-learner is robust
  to constant features (the orthogonalisation step zeros them out), so wiring
  the slots in advance lets us drop trained models in without touching the
  orchestrator.
- **Horse grouping key is `(horse_name, jurisdiction)`.** Not perfect — name
  collisions across years exist — but it's the most informative key the parquet
  exposes. The cleaner fix is to surface `horses.dedup_key` in
  `export_training_data.py`; deferred to a future export.
- **Connections shrinkage prior strength = 30.** Empirically chosen — pulls
  small-sample jockey/trainer rates toward jurisdiction baseline strongly
  enough that 1-of-1 windfalls don't propagate, but lets veteran jockeys
  with hundreds of starts express their true rate. Tunable via config.

**Known limitations carried forward:**
- Pace / Sequence layers untrained — contributing 0.5 to the stacker (zeroed
  out after orthogonalisation, so no harm done, but the meta is missing
  potentially valuable signal).
- AR exclusion stands. Jurisdiction values are limited to {UK, HK, JP}.
- No calibration yet — `predict_proba` outputs are NOT calibrated; using them
  for EV calculation would be wrong. That's the Phase 4 deliverable.
- `field_size` derivation runs after the parquet load (groupby in
  `prepare_training_features`). This is correct for training but the live
  ingest path must populate the column from the parsed RaceCard itself —
  noted, not blocking.

**Tests Status:**
- **343 tests passing in ~9s** (was 306 → +37 new). Run with
  `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: 2026-05-12 (c) — Phase 0/1/2 closeout sweep

**Completed:**

*Phase 0 (loose-end cleanup)*
- `scripts/db/backfill_tracks.py` — idempotent populator for the previously-empty
  `tracks` table. Derives one row per `(track_code, jurisdiction)` from `races` and
  records the union of surfaces observed at each track as a JSON array. Ran live:
  100 distinct combos inserted. Re-running is a no-op (UNIQUE constraint).
- `tests/test_db/test_backfill_tracks.py` (3 tests) — happy path, idempotency,
  surface aggregation.

*Phase 1 (closeout)*
- `.env.example` — full HRBS_* knob template aligned with `app/core/config.py`.
- `app/core/logging.py` — idempotent `configure_logging()` honouring `LOG_LEVEL`
  and `LOG_JSON`. ConsoleRenderer (with colour when stdout is a TTY) in dev,
  JSONRenderer in prod. `get_logger()` is the public helper.
- `app/services/pdf_parser/equibase_parser.py` — `EquibaseParser(BrisnetParser)`.
  Subclass placeholder; same regex layer for now. Extractor's `_get_parser()` now
  routes the `equibase` format to it (was falling back to BrisnetParser previously).
- `app/db/session.py` — async SQLAlchemy 2.0 engine (`AsyncEngine`) + session
  factory + `session_scope()` context manager + `get_session()` FastAPI dep +
  `init_db()` / `dispose_engine()` lifespan hooks. Declarative `Base` lives here.
- `app/db/models.py` — `IngestedCard`/`IngestedRace`/`IngestedHorse`/`IngestedPPLine`
  ORM hierarchy with `cascade="all, delete-orphan"` so deleting a card cleanly
  removes the whole tree. JSON columns for variable-length fields
  (medication_flags, equipment_changes, parse_warnings).
- `app/db/persistence.py` — `card_to_orm()` + `persist_ingestion_result()`. Splits
  Pydantic-to-ORM conversion from HTTP handler logic for testability.
- `app/main.py` — `create_app()` factory with async lifespan that boots logging
  + `init_db()`. CORS allow-all (dev), `/healthz` + `/version`, mounts the
  ingest router at `/api/v1/ingest`.
- `app/api/v1/ingest.py` — POST `/upload`. Multipart PDF, content-type guard,
  size guard at HTTP boundary, then `run_in_executor(ingest_pdf, ...)` so the
  CPU-bound parser doesn't block the event loop. Successful results are
  persisted via the SQLAlchemy session dependency; failures are returned to
  the client with `processing_ms` populated.
- `tests/test_api/test_ingest.py` (6 tests) — `/healthz`, `/version`, empty body
  → 400, bad content-type → 415, corrupt PDF → success=False, valid reportlab-
  drawn fixture → 200 + structured payload.
- `pyproject.toml` — added FastAPI, uvicorn, python-multipart, SQLAlchemy 2.0,
  aiosqlite, numpy to runtime deps; httpx to dev deps.

*Phase 2 (full implementation)*
- `app/services/feature_engineering/layoff.py` — `layoff_fitness(days)` returns a
  [0,1] score; `apply_layoff_features(df)` writes the column. `DEFAULT_LAMBDA =
  ln(2)/60` so fitness halves at `recovery_threshold + 60 days` (≈ 90-day layoff).
  First-time starters mapped to the sentinel `FIRST_TIME_STARTER_FITNESS = 0.6`.
  (12 tests)
- `app/services/feature_engineering/speed_features.py` — EWM α=0.4 (CLAUDE.md §8;
  pandas convention is most-recent-LAST, so we reverse the most-recent-first
  PP list before `.ewm()`). Per-horse summary: `ewm_speed_figure`, `last_speed_figure`,
  `best_speed_figure` (6-PP window), `speed_figure_delta`. Field-relative columns:
  `ewm_speed_zscore`, `ewm_speed_rank` (1=fastest), `ewm_speed_pct`. Constant fields
  z-score to 0 (no NaN). (13 tests)
- `app/services/feature_engineering/pace_features.py` — `pace_shape_metrics(pp)`
  builds `early_speed = -beaten_lengths_q1`, `late_kick = bl_q2 - bl_finish`,
  fraction ratios. Per-horse `horse_pace_summary()` averages over the 5-PP
  window. Field-level `pace_pressure_index` = count of horses whose mean
  `beaten_lengths_q1 ≤ 1.5L` (front-runner proxy). (13 tests)
- `app/services/feature_engineering/class_features.py` — claiming-to-claiming uses
  claiming-price delta directly; otherwise falls back to purse delta. Records
  `race_type_change` as `same` or `<modal>-><today>`. Field-relative z-score
  appended. (8 tests)
- `app/services/feature_engineering/connections.py` — jockey continuity proxies:
  `today_jt_same_pair`, `jockey_repeat_streak`, `today_jockey_win_rate_in_pps`.
  `trainer_continuity` is a weak proxy (no per-PP trainer field in the schema —
  documented inline). Names normalized via casefold + strip. (8 tests)
- `app/services/feature_engineering/engine.py` — `FeatureEngine.transform(card)`
  produces a tidy long-form DataFrame keyed by `(race_number, post_position)`.
  Joins all per-module frames + base identifiers + `weight_lbs_delta` (field-
  relative). One bad race is logged + skipped without killing the card. (9 tests)
- `tests/test_features/_fixtures.py` — shared synthetic-card factory used by
  every feature test module. Centralises the construction of valid PPs / horses
  / races / cards so test files stay focused on assertions.

**Bug fixes (none — full clean closeout)**

**Key Decisions Made:**
- **Live ingestion DB is separate from master training DB.** The Pydantic schemas
  in `app/schemas/race.py` model PDF-derived race cards (today's racing data).
  The ORM in `app/db/models.py` mirrors that schema, NOT the master `race_results`
  table. Two different stores: master DB is the training corpus at
  `data/db/master.db`; the live DB defaults to `./horseracing.db` (overridable
  via `HRBS_DATABASE_URL`). Keeping them isolated prevents schema collision and
  lets ingest-time writes happen without locking read-heavy training queries.
- **EquibaseParser is a thin subclass for now.** Without real Equibase PDFs to
  test against, divergences from Brisnet are speculative — better to dispatch on
  format (so behaviour is per-format from the start) than to fork the regex layer
  prematurely. Future overrides go on the subclass.
- **Layoff curve constants live in the module, not config.** `DEFAULT_LAMBDA` and
  `DEFAULT_RECOVERY_THRESHOLD_DAYS` are pure tunables — Phase 3 will fit them
  per-surface from the master DB and pass fitted values into the engine. No
  global mutable state.
- **EWM weighting direction is documented inline.** pandas' `.ewm(alpha)` weights
  the LAST element heaviest; our `HorseEntry.pp_lines` is enforced most-recent-
  FIRST. The module reverses before applying `.ewm()` and the docstring explains
  why. Two places where the direction matters: `ewm_speed()` and any future
  rolling stats.
- **Trainer continuity is a documented weak proxy.** `PastPerformanceLine` carries
  per-PP `jockey` but no per-PP `trainer` (the Brisnet PDF format itself omits
  the per-PP trainer in the dense PP table; only today's trainer appears in the
  horse header). The connections module returns a 0/1 proxy with the limitation
  documented; the Bayesian connections model in Phase 3 will get richer trainer
  context from the master training DB instead.

**Tests Status:**
- **306 tests passing in ~7s** (was 234 → +72 new). Run with
  `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: 2026-05-12 (b) — ML training parquet exported

**Completed:**
- Added `--jurisdictions UK,HK,JP` flag to `scripts/db/export_training_data.py`. Threaded as an
  optional allow-list through to the SQL via parametrized `AND r.jurisdiction IN (?,...)` clause
  appended to the existing `TRAINING_QUERY_TEMPLATE`. Default remains "all jurisdictions".
- Ran the export. Output: `data/exports/training_20260512.parquet`. 2,317,297 rows × 29 columns,
  38.9 MB. Date range 1986-01-05 → 2021-07-31. By jurisdiction: JP 1,609,930 / UK 598,708 / HK
  108,659. Mean field size (derived from group counts) 11.67 horses/race — healthy and consistent
  with real racing.
- Updated PROGRESS.md "Current State" + added a "Known Export Caveats" table cataloguing the
  null-rate gaps Phase 3 will need to handle.

**Key Decisions Made:**
- Excluded AR from the export per PROGRESS.md recommendation (avg 2.2 horses/race due to the
  `nro`-reuse issue across same-day venues at the same track). Distance + surface in the dedup
  key don't disambiguate enough to recover the true field. Including AR would teach models that
  2-horse fields are common, which is false.
- Did not regenerate `field_size` at export time — left it as raw column value (UK populated,
  JP/HK NULL). Phase 3 can compute it via `df.groupby(['race_date','track_code','race_number']).size()`.
  Pushing that derivation downstream keeps the export contract simple.
- Did not attempt to derive a synthetic UK `post_position` or JP speed figure during export. Both
  are Phase 3 concerns — they require modelling decisions (which feature engineering approach?
  fall back vs. impute vs. exclude?) that belong in the bootstrap script, not in the SQL export.

**Tests Status:**
- 62 Phase 0 tests passing in 2.45s. No new tests added — the `--jurisdictions` flag is a thin
  CLI passthrough; the SQL parameterization path is exercised end-to-end by the live export.

---

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
- [x] **ML training parquet exported: `data/exports/training_20260512.parquet` (2.3M rows, UK+HK+JP)**
- [x] `scripts/db/backfill_tracks.py` + tests — `tracks` table now holds 100 distinct (track_code, jurisdiction) combos
- [ ] Future: derive a venue-specific course identifier for AR to fix the over-split races issue
- [ ] Future: derive JP synthetic speed figure from `(distance, fraction_finish_sec, condition)` so JP rows can train speed-dependent models

### Phase 1: PDF Ingestion
- [x] `app/core/config.py`
- [x] `app/core/logging.py`
- [x] `app/main.py`
- [x] `app/api/v1/ingest.py`
- [x] `app/db/models.py`
- [x] `app/db/session.py`
- [x] `app/db/persistence.py`
- [x] `app/schemas/race.py`
- [x] `app/services/pdf_parser/cleaner.py`
- [x] `app/services/pdf_parser/extractor.py`
- [x] `app/services/pdf_parser/brisnet_parser.py`
- [x] `app/services/pdf_parser/equibase_parser.py` (subclass placeholder)
- [x] `tests/test_parser/test_cleaner.py` (102 tests)
- [x] `tests/test_parser/test_brisnet_parser.py` (45 tests)
- [x] `tests/test_parser/test_extractor.py` (25 tests)
- [x] `tests/test_api/test_ingest.py` (6 tests)
- [x] `pyproject.toml`
- [x] `.env.example`
- [ ] Future: real-PDF validation pass against an actual Brisnet UP card; tune `_RE_RACE_HEADER` (`\bRACE\b` anchor) and `extract_text_from_pdf` (real `layout=True` mode)
- [ ] Future: DRF parser implementation (currently falls back to BrisnetParser)

### Phase 2: Feature Engineering
- [x] `app/services/feature_engineering/engine.py` (FeatureEngine orchestrator)
- [x] `app/services/feature_engineering/speed_features.py` (EWM α=0.4 + field-relative z/rank/pct)
- [x] `app/services/feature_engineering/pace_features.py` (shape, fraction ratios, pressure index)
- [x] `app/services/feature_engineering/class_features.py` (claiming/purse delta + race-type change)
- [x] `app/services/feature_engineering/connections.py` (jockey continuity + win-rate proxy)
- [x] `app/services/feature_engineering/layoff.py` (parametric exp-decay fitness curve)
- [x] `tests/test_features/test_layoff.py` (12 tests)
- [x] `tests/test_features/test_speed_features.py` (13 tests)
- [x] `tests/test_features/test_pace_features.py` (13 tests)
- [x] `tests/test_features/test_class_features.py` (8 tests)
- [x] `tests/test_features/test_connections.py` (8 tests)
- [x] `tests/test_features/test_engine.py` (9 integration tests)
- [x] `tests/test_features/_fixtures.py` (shared synthetic-card builders)

### Phase 3: Model Layer
- [x] `scripts/bootstrap_models.py` (orchestrator with --sample-frac and --run-name)
- [x] `app/services/models/training_data.py` (leakage-free feature prep + time split)
- [x] `app/services/models/speed_form.py` (LightGBM Layer 1a — trained, val_top1=0.257)
- [x] `app/services/models/pace_scenario.py` (STUB — needs fractional time columns)
- [x] `app/services/models/sequence.py` (STUB — needs PyTorch + globally-unique horse_id)
- [x] `app/services/models/connections.py` (Empirical-Bayes Layer 1d — trained, 50K pairs)
- [x] `app/services/models/market.py` (Isotonic odds calibration Layer 1e — trained)
- [x] `app/services/models/meta_learner.py` (LightGBM stacker — trained, val_top1=0.341)
- [x] `tests/test_models/test_training_data.py` (15 tests)
- [x] `tests/test_models/test_speed_form.py` (10 tests)
- [x] `tests/test_models/test_connections.py` (7 tests)
- [x] `tests/test_models/test_market.py` (5 tests)
- [x] `tests/test_models/_synth.py` (shared synthetic data generator)
- [x] **Live baseline trained: `models/baseline_full/` — Speed/Form val_top1=0.257, Meta val_top1=0.341**
- [ ] Future: train pace_scenario once fractional times are in the parquet
- [ ] Future: train sequence transformer once PyTorch + horse dedup_key in export
- [ ] Future: expose `horses.dedup_key` in `export_training_data.py` to fix horse collisions

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
