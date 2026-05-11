# CLAUDE.md — Horse Racing Betting System
# Read this file at the start of every session before writing any code.

---

## 1. PROJECT IDENTITY

**What this is:** A production-quality, end-to-end pari-mutuel wagering analytics system
built for a university finance research paper-trading competition. It ingests raw horse
racing past performance PDFs (Brisnet UP, Equibase, DRF), extracts structured data,
runs a 7-layer ensemble ML pipeline, and outputs a CVaR-optimized portfolio of +EV bets.

**Source of truth documents (read these when in doubt):**
- `Horse_Racing_System_Master_Reference.md` — architecture, math, model design, feature
  engineering specs, implementation order. **Supersedes all other sources.**
- `Horse_Racing_Betting_System_Research.pdf` — academic backing. Reference when
  implementing ordering models, calibration, or portfolio optimization math.
- `DATA_PIPELINE.md` — Phase 0 master DB build pipeline: Kaggle acquisition, schema,
  deduplication keys, field maps, quality gates, and pipeline scripts. **Read before
  touching any Phase 0 code.**

---

## 2. ABSOLUTE CONSTRAINTS — NEVER VIOLATE THESE

These are non-negotiable mathematical and design decisions. Do not rationalize exceptions.

| Constraint | Rule |
|---|---|
| Exotic ordering | **Never use Harville.** Default: Plackett-Luce. Preferred: Stern (Gamma) or Copula. |
| Feature engineering | All features must be **field-relative**, not absolute. A Beyer of 95 means nothing without the field mean. |
| Bet sizing | **1/4 Kelly only.** Never full Kelly. Never per-bet Kelly in isolation. |
| Portfolio optimization | **CVaR-constrained**, not mean-variance alone. Treat the full card as a portfolio. |
| Calibration | Raw model scores are NOT probabilities. Always apply Platt scaling or isotonic regression on held-out data before any EV calculation. |
| Validation split | Always **time-based** (train on earlier dates, validate on later). Never random split — that leaks future information. |
| Sub-model inputs | **Orthogonalize** before the meta-learner. Speed figures already incorporate pace; residualize to prevent double-counting. |

---

## 3. TECHNOLOGY STACK

**Backend / ML Pipeline**
- Python 3.11+
- FastAPI (async API layer; all PDF ingestion runs via `run_in_executor`)
- Pydantic v2 (strict validation; all schemas in `app/schemas/`)
- pandas / NumPy (vectorized feature engineering)
- LightGBM (Speed/Form Model, Pace Scenario Model)
- PyTorch (Transformer sequence model — per-horse career history)
- SciPy (Plackett-Luce MLE fitting, CVaR optimization)
- scikit-learn (Platt scaling, isotonic regression calibration)
- structlog (structured logging throughout)

**PDF Ingestion**
- pdfplumber (primary; layout mode for column preservation)
- pypdf (fallback for pdfplumber failures)
- Three-pass strategy: layout → text → pypdf character-level

**Frontend**
- Next.js 14+ (App Router)
- React + TypeScript
- Tailwind CSS
- Lucide React (icons only — no heavy component libraries)
- Recharts (probability visualization, EV charts)

**Storage**
- SQLite (local dev / Phase 0 master DB)
- PostgreSQL (production)
- SQLAlchemy 2.0 (async ORM)

**Config / Infra**
- `python-dotenv` + Pydantic `BaseSettings` for all config
- `alembic` for migrations
- `pytest` + `pytest-asyncio` for testing

---

## 4. REPOSITORY STRUCTURE

```
/
├── CLAUDE.md                          ← you are here
├── DATA_PIPELINE.md                   ← Phase 0 master DB build pipeline
├── PROGRESS.md                        ← session log; update at end of every session
├── DECISIONS.md                       ← architectural decisions with rationale
├── Horse_Racing_System_Master_Reference.md
├── Horse_Racing_Betting_System_Research.pdf
│
├── data/                              ← ALL DATA LIVES HERE; fully gitignored except logs
│   ├── staging/                       ← raw Kaggle downloads; never modify
│   │   └── <dataset-slug>/
│   ├── cleaned/                       ← after field mapping + transformer application
│   │   └── <dataset-slug>/
│   │       ├── accepted/              ← passed quality gate (score >= 0.60)
│   │       └── rejected/              ← failed quality gate; logged with reasons (COMMIT THESE)
│   ├── exports/                       ← ML-ready feature matrices exported from DB
│   └── db/
│       └── master.db                  ← SQLite master training DB (dev)
│                                         → PostgreSQL in production
│
├── scripts/
│   └── db/                            ← Phase 0 pipeline scripts (standalone; no FastAPI dep)
│       ├── setup_db.py                ← CREATE all tables; run once
│       ├── evaluate_dataset.py        ← score a dataset before ingesting
│       ├── ingest_kaggle.py           ← download + stage via Kaggle CLI
│       ├── map_and_clean.py           ← apply field maps + transformers
│       ├── quality_gate.py            ← per-row quality scoring
│       ├── load_to_db.py              ← idempotent INSERT with dedup
│       ├── dedup_report.py            ← duplicate statistics report
│       ├── export_training_data.py    ← SQL export → ML feature DataFrame
│       ├── field_maps.py              ← field mapping registry (per dataset)
│       └── transformers.py            ← unit conversion functions (metres→f, stones→lbs, etc.)
│
├── backend/
│   ├── app/
│   │   ├── main.py                    ← FastAPI app factory
│   │   ├── core/
│   │   │   ├── config.py              ← Pydantic BaseSettings
│   │   │   └── logging.py             ← structlog setup
│   │   ├── schemas/
│   │   │   ├── race.py                ← PastPerformanceLine, HorseEntry, RaceCard, etc.
│   │   │   └── bets.py                ← BetRecommendation, Portfolio, EVResult
│   │   ├── services/
│   │   │   ├── pdf_parser/
│   │   │   │   ├── extractor.py       ← orchestrator: bytes → IngestionResult
│   │   │   │   ├── cleaner.py         ← text normalization (pure functions)
│   │   │   │   ├── brisnet_parser.py  ← Brisnet UP format parser
│   │   │   │   ├── equibase_parser.py ← (Phase 1 stub → Phase 1b implementation)
│   │   │   │   └── drf_parser.py      ← (Phase 1 stub → future)
│   │   │   ├── feature_engineering/
│   │   │   │   ├── engine.py          ← FeatureEngine: RaceCard → feature DataFrame
│   │   │   │   ├── speed_features.py  ← EWM speed figs, field-relative ranks
│   │   │   │   ├── pace_features.py   ← pace shape construction, fraction ratios
│   │   │   │   ├── class_features.py  ← class trajectory, claiming price delta
│   │   │   │   ├── connections.py     ← jockey×trainer interaction features
│   │   │   │   └── layoff.py          ← parametric fitness decay curve
│   │   │   ├── models/
│   │   │   │   ├── speed_form.py      ← LightGBM Speed/Form Model (Layer 1a)
│   │   │   │   ├── pace_scenario.py   ← LightGBM Pace Model (Layer 1b)
│   │   │   │   ├── sequence.py        ← Transformer encoder (Layer 1c)
│   │   │   │   ├── connections.py     ← Bayesian hierarchical model (Layer 1d)
│   │   │   │   ├── market.py          ← Market/smart money model (Layer 1e)
│   │   │   │   └── meta_learner.py    ← Stacking meta-learner (Layer 2)
│   │   │   ├── calibration/
│   │   │   │   ├── calibrator.py      ← Platt / isotonic selector + softmax
│   │   │   │   └── drift.py           ← CUSUM change-point detection
│   │   │   ├── ordering/
│   │   │   │   ├── plackett_luce.py   ← PL MLE fitting + exotic probability enumeration
│   │   │   │   ├── stern.py           ← Gamma distribution ordering model
│   │   │   │   └── copula.py          ← Copula-based pace-correlated ordering (Layer 4)
│   │   │   ├── ev_engine/
│   │   │   │   ├── calculator.py      ← edge, EV, market impact per bet type
│   │   │   │   └── market_impact.py   ← pari-mutuel pool impact model
│   │   │   └── portfolio/
│   │   │       ├── optimizer.py       ← CVaR optimizer, correlated Kelly
│   │   │       └── sizing.py          ← 1/4 Kelly formula, bet cap enforcement
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── ingest.py          ← POST /api/v1/ingest/upload
│   │   │       ├── analyze.py         ← POST /api/v1/analyze/{card_id}
│   │   │       └── portfolio.py       ← GET /api/v1/portfolio/{card_id}
│   │   └── db/
│   │       ├── models.py              ← SQLAlchemy ORM models
│   │       └── session.py             ← async engine + session factory
│   ├── tests/
│   │   ├── test_parser/
│   │   ├── test_features/
│   │   ├── test_ordering/
│   │   └── test_portfolio/
│   ├── scripts/
│   │   ├── bootstrap_models.py        ← train baseline models from master DB export
│   │   └── validate_calibration.py    ← reliability diagram + ECE report
│   ├── pyproject.toml
│   └── .env.example
│
└── frontend/
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx                   ← upload landing
    │   ├── card/[id]/
    │   │   ├── page.tsx               ← race card viewer
    │   │   └── portfolio/page.tsx     ← bet execution ticket
    │   └── api/                       ← Next.js route handlers (proxy to FastAPI)
    ├── components/
    │   ├── RaceCard/
    │   ├── HorseTable/
    │   ├── EVGauge/
    │   ├── ProbabilityBar/
    │   └── BetTicket/
    ├── lib/
    │   └── api.ts                     ← typed fetch wrappers
    └── package.json
```

---

## 5. IMPLEMENTATION PHASES

Track current phase in `PROGRESS.md`. Build strictly in this order — each phase
is independently useful and testable before the next begins.

| Phase | Focus | Key Deliverables | Status |
|---|---|---|---|
| **0** | Master Training DB | Kaggle acquisition → field mapping → quality gate → dedup → SQLite load → ML export | ⬜ |
| **1** | PDF Ingestion Pipeline | `extractor.py`, `cleaner.py`, `brisnet_parser.py`, all Pydantic schemas | 🔄 In Progress |
| **2** | Feature Engineering Engine | `engine.py` + all feature modules; field-relative rankings, EWM, layoff curve | ⬜ |
| **3** | Model Bootstrap + Training | LightGBM Speed/Form model from master DB export; inference scaffolding for all 5 sub-models | ⬜ |
| **4** | Calibration + Ordering | Platt/isotonic calibration; Plackett-Luce MLE; Stern model; exotic prob enumeration | ⬜ |
| **5** | EV Engine + Portfolio Optimizer | Market impact model; edge calculation; CVaR optimizer; 1/4 Kelly sizing | ⬜ |
| **6** | Frontend | Upload flow; race card viewer; probability visualization; bet execution ticket | ⬜ |

Phase 0 and Phase 1 are **independent** — they can proceed in parallel once the DB schema
is created. Phase 0 feeds Phase 3 (model training). Phase 1 feeds Phase 2 (live inference).

---

## 6. PHASE 0 — MASTER TRAINING DATABASE

**Full spec lives in `DATA_PIPELINE.md`. This section is a summary only.**

### Purpose
Build a local SQLite (→ PostgreSQL in prod) master database of all findable historical race
results. This is the sole training corpus for all ML models. Its quality is the ceiling on
every model downstream.

### Non-Negotiables
- **Idempotent:** Running the pipeline twice on the same source produces the same DB state.
- **Zero duplicates:** SHA-256 dedup keys on races, horses, and results. Conflict = skip + log.
- **Audit trail:** Every row references `source_dataset_id` back to the `datasets` table.
- **Quality gate:** Rows scoring below 0.60 go to `rejected/` — never inserted.
- **Source priority** (highest wins on conflict): Equibase > Brisnet > DRF > Kaggle CSV

### Priority Kaggle Datasets (acquire in this order)
| Dataset | Slug | Jurisdiction | Priority |
|---|---|---|---|
| US Horse Racing Results | `joebeachcapital/horse-racing` | US | HIGH |
| Hong Kong HKJC | `gdaley/horseracing-in-hk` | HK | HIGH |
| UK Racing (Zygmunt) | `zygmunt/horse-racing-dataset` | UK/IRE | HIGH |
| Betfair Horse Racing | `adrianmcmahon/betfair-horse-racing` | UK | MEDIUM |
| Australian Racing | `tobycrabtree/horse-racing-australia` | AU | MEDIUM |

### Pipeline Execution Order (per dataset)
```bash
python scripts/db/setup_db.py                                    # once only
python scripts/db/ingest_kaggle.py --dataset "slug" --output data/staging/
python scripts/db/evaluate_dataset.py data/staging/slug/         # must score >= 0.70
python scripts/db/map_and_clean.py --input data/staging/slug/ --map "slug" --output data/cleaned/slug/
python scripts/db/quality_gate.py --input data/cleaned/slug/
python scripts/db/load_to_db.py --input data/cleaned/slug/accepted/
python scripts/db/dedup_report.py
```

### Dedup Keys (implement verbatim — do not change these)
```python
# Race: SHA256("CD|2026-05-10|4|4.5|dirt")
def race_dedup_key(track_code, race_date, race_number, distance_furlongs, surface) -> str

# Horse: SHA256("lovely words|2021|US")
def horse_dedup_key(name, foaling_year, country) -> str

# Result: SHA256(race_dedup_key + "|" + horse_dedup_key)
def result_dedup_key(race_dedup_key, horse_dedup_key) -> str
```

### ML Export
After DB is loaded, export the training DataFrame via:
```bash
python scripts/db/export_training_data.py
```
Output: `data/exports/training_<date>.parquet` — this is what `backend/scripts/bootstrap_models.py` consumes.

---

## 7. RACE CARD DATA FIELDS — COMPLETE EXTRACTION REFERENCE

*Derived from analysis of real Churchill Downs Brisnet UP race cards (CD-05/10/2026).*
*Every field listed here that is extractable from the PDF MUST be extracted and stored.*
*This is the ground truth for what the PDF parser must produce.*

### 7a. Race Header Fields

| Field | Example | Notes |
|---|---|---|
| `track_name` | "Churchill Downs" | Full name, not just code |
| `track_code` | "CD" | Equibase abbreviation |
| `card_date` | 2026-05-10 | Printed at bottom as "CD-05/10/2026-4" |
| `race_number` | 4 | Printed large in top-left box |
| `post_time` | "2:14PM" | "APPROX. POST:" |
| `distance_furlongs` | 4.5 | Converted from "4 1/2 FURLONGS" |
| `distance_raw` | "4 1/2 FURLONGS" | Preserve original string |
| `surface` | "dirt" | Inferred from track diagram or surface notation |
| `race_type` | "claiming" | From conditions text |
| `claiming_price` | 30000.0 | "Claiming $30,000" |
| `purse_usd` | 62000.0 | "Purse $62,000" |
| `purse_includes_ktdf` | true | When purse note includes "(Includes $X,XXX from KTDF)" |
| `ktdf_bonus_usd` | 5800.0 | Kentucky Thoroughbred Development Fund supplement |
| `age_sex_restrictions` | "Three Year Olds" | From conditions text |
| `weight_conditions` | "Non-winners Of A Race Since April 10 Allowed 2 Lbs." | Full allowance text |
| `claiming_restrictions` | "Claiming Price $30,000. Six And One-Half Furlongs" | Non-winners restriction |
| `grade` | null | Only populated for graded stakes (1/2/3) |
| `race_name` | null | Stakes name if applicable; null for overnight races |
| `track_record_holder` | "Love At Noon(3)" | From "Track Record:" line |
| `track_record_weight` | 121 | Weight carried in track record |
| `track_record_time` | "1:14.34" | Track record time string |
| `track_record_date` | "5-5-01" | Date of track record |
| `available_bet_types` | ["Daily Double","Exacta","Trifecta","Superfecta","Pick 3","Super Hi-5","Win","Place","Show"] | From top banner |
| `condition` | "fast" | Track condition; infer from condition line or track record notation |
| `weather` | null | When present on card |
| `jurisdiction` | "US" | Hardcoded for Churchill Downs |

### 7b. Horse Entry Fields (Today's Race)

| Field | Example | Notes |
|---|---|---|
| `program_number` | "1" | Saddle cloth number — may differ from post position |
| `post_position` | 1 | Gate draw; often same as program number but not always |
| `morning_line_odds` | 6.0 | "6-1" → 7.0 decimal (numerator/denominator + 1) |
| `morning_line_raw` | "6-1" | Preserve original fractional string |
| `horse_name` | "Lovely Words" | Display name; preserve original casing |
| `horse_age` | 4 | Age in years |
| `horse_sex` | "F" | M=mare, F=filly, G=gelding, C=colt, H=horse (older male) |
| `horse_color` | "Dk B/ Br" | Dark Bay or Brown; preserve raw color code |
| `foaling_year` | 2022 | Race year minus age |
| `bred_in` | "Kentucky" | "Bred in Kentucky by…" |
| `sire` | "A Thousand Words" | Sire name |
| `dam` | "Orca" | Dam name |
| `dam_sire` | "Mizzen Mast" | Broodmare sire |
| `owner` | "Rags Racing Stable, LLC (Joe Ragsdale)" | Full owner name + individual in parens |
| `silks` | "Orange, white RAGS on black ball, orange cap" | Silks description text |
| `trainer` | "John A. Ortiz" | Display name |
| `trainer_win_pct` | 0.100 | "(1-0-0-1) 100.00%" → 1st of 1 at meet |
| `trainer_meet_record` | "1-0-0-1" | Starts-1st-2nd-3rd at current meet |
| `jockey` | "Danilo Grisales Rave" | Display name |
| `jockey_career_record` | "5-0-0-0" | Career record at current meet |
| `jockey_win_pct` | 0.00 | From meet record |
| `weight_lbs` | 123.0 | Weight assigned today |
| `medication_flags` | ["L"] | L=Lasix, B=Blinkers. Parsed from superscript marker |
| `equipment_changes` | ["blinkers_on"] | From footnotes at bottom of card |
| `equibase_speed_rating` | 45 | Printed in "E Speed" column header row |
| `pace_style` | "UNKNOWN" | Assigned by pace model; UNKNOWN at parse time |
| `career_stats` | See 7c | Box in top-right of each horse entry |

### 7c. Career Statistics Box (per-horse, per-surface)

Each horse entry includes a stats box showing performance by year and surface. These are
critical ML features — extract all cells.

```python
class CareerStatLine(BaseModel):
    label: str          # "2026", "2025", "2024", "Life", "CD (Dirt)", "Turf", "Wet Turf", etc.
    starts: int
    wins: int
    places: int         # 2nd finishes
    shows: int          # 3rd finishes
    earnings_usd: float # Total earnings for this split

# Example from card: "2026: 1 0 1 1 $40,200 Turf"
# Parsed as: CareerStatLine(label="2026", starts=1, wins=0, places=1, shows=1, earnings_usd=40200.0)
```

Common `label` values seen in real cards:
- Year labels: `"2026"`, `"2025"`, `"2024"`, `"Life"` (career total)
- Surface labels: `"Turf"`, `"Wet Turf"`, `"Distance"`, `"Wet Dirt"`, `"CD (Dirt)"` (current track + surface)

**Why these matter:** Year-over-year improvement trajectory and surface/distance affinity
are strong ML signals. The career stats box provides these compactly without parsing all PP lines.

### 7d. Past Performance Line Fields (Historical)

One row per prior race start. Extract every field — missing fields reduce `parse_confidence`.

| Field | Example | Notes |
|---|---|---|
| `race_date` | 2026-04-12 | "12Apr26" → parsed date |
| `track_code` | "Kee" | Abbreviated track; normalize to Equibase code |
| `surface_symbol` | "ft" (dirt fast) / turf symbol | First symbol column after track |
| `distance_furlongs` | 6.0 | "6f" |
| `race_class` | "Clm 40000nw2/L" | Full class description string |
| `claiming_price_pp` | 40000.0 | Parse from class description |
| `purse_pp` | null | Often not shown in compact PP format |
| `post_position` | 4 | Gate for that race |
| `finish_position` | 3 | Official finish |
| `beaten_lengths_finish` | 3.25 | Lengths behind winner at finish; 0 if won |
| `beaten_lengths_q1` | 2.5 | Lengths behind leader at first call |
| `beaten_lengths_q2` | 1.0 | Lengths behind leader at second call |
| `beaten_lengths_stretch` | 0.5 | Lengths behind leader at stretch call (third call) |
| `position_q1` | 3 | Running position at first call |
| `position_q2` | 2 | Running position at second call |
| `position_stretch` | 2 | Running position at stretch |
| `jockey` | "Elliott J" | Jockey for that race |
| `weight_lbs` | 124.0 | Weight carried |
| `odds_final` | 4.0 | Final decimal odds |
| `speed_figure` | 71 | Equibase/Brisnet speed figure; "--" → null |
| `speed_figure_source` | "brisnet" | Source tag |
| `fraction_q1` | 22.40 | First call split in seconds (":22.4" → 22.40) |
| `fraction_q2` | 46.10 | Second call in seconds |
| `fraction_finish` | 71.60 | Final time in seconds ("1:11.6" → 71.60) |
| `days_since_prev` | null | Computed from date delta to next PP line |
| `field_size` | 7 | Number of starters |
| `race_type` | "claiming" | Parsed from class description |
| `condition` | "fast" | Track condition for that race |
| `surface` | "dirt" | Surface for that race |
| `claimed_by` | null | When horse was claimed; "Claimed by [Owner] from [Previous Owner]..." |
| `claimed_from` | null | Previous owner name when claimed |
| `claim_price` | null | Price at which claimed |
| `comment` | "Bided(3rd) Amazing Ascendis(4) Lovely Words" | Trip note / chart comment; preserve full text |

**Note on comment field:** These trip notes are extremely dense signals:
- Beaten horses named in the comment = implicit quality calibration of the race
- Position descriptors ("tracked 3p", "wide", "5-wide turn") = trip quality adjustment
- Effort qualifiers ("chased", "stalked", "rated off") = pace style confirmation
- These should be stored raw as text; NLP feature extraction happens in Phase 2

### 7e. Workout Lines

Workouts appear below the PP lines and are a distinct, ML-valuable signal. Extract all.

| Field | Example | Notes |
|---|---|---|
| `workout_date` | 2026-05-04 | "4 May 26" |
| `workout_track` | "CD" | Track abbreviation |
| `workout_surface` | "ft" | Same surface codes as PP lines |
| `workout_distance_furlongs` | 4.0 | "4F" |
| `workout_time_sec` | 48.20 | ":48.20" → 48.20 seconds |
| `workout_type` | "B" | B=breezing, H=handily, WO=walking over, G=galloping, D=driving |
| `workout_rank` | 5 | "5/22" → ranked 5th of 22 at that distance that day |
| `workout_total_at_distance` | 22 | Denominator |
| `workout_bullet` | false | True when workout was fastest of the day (bullet) |

**Why workouts matter:** The most recent workout pattern (timing, distance, surface,
ranking vs. peers) is one of the strongest fitness signals not captured in race history.
A horse with a bullet workout 3 days before a race at the race distance is a live contender
regardless of PP form.

### 7f. Fields Listed at Bottom of Card (Card-Level)

| Field | Example | Notes |
|---|---|---|
| `equipment_change_notes` | ["#3 - California Smoke previously trained by Hartman, Chris A."] | Trainer/equipment footnotes |
| `probable_favorites` | [6, 8, 5, 7] | Probable favorites listed as program numbers |
| `l_lasix_marker` | true | "L-Lasix" legend present = some horses on Lasix |
| `kentucky_bred_marker` | true | "- Kentucky Bred" legend present |
| `first_time_reported_gelding` | ["#6 White Whale"] | "First Time Reported Geldings" footnote |

---

## 8. KEY MATHEMATICAL SPECS

These are exact implementations required. Do not approximate or substitute.

### Fractional Kelly
```python
def kelly_fraction(edge: float, odds: float, fraction: float = 0.25) -> float:
    # edge = model_prob - market_prob
    # odds = decimal odds (e.g., 4.0 for 3-1)
    # fraction = 0.25 for quarter-Kelly
    full_kelly = (edge * odds - (1 - edge)) / odds
    return max(0.0, full_kelly * fraction)
```

### Plackett-Luce Exacta Probability
```python
# P(horse i 1st, horse j 2nd) = (s_i / S) * (s_j / (S - s_i))
# where s_i = PL strength parameter, S = sum of all strengths
# Strength params fit via MLE on historical finishing orders (scipy.optimize.minimize)
```

### Softmax Temperature for Calibration
```python
# After Platt/isotonic calibration, normalize across field:
# p_i_calibrated = softmax(logits / temperature)
# temperature is a tunable scalar fit on validation data (start at 1.0)
```

### CVaR Constraint (95th percentile shortfall)
```python
# Minimize: -E[portfolio_return]
# Subject to: CVaR_{0.05}(portfolio_loss) <= max_daily_drawdown_pct
# Solved via scipy.optimize with Monte Carlo scenario generation
```

### EWM Speed Figure
```python
# alpha = 0.4 (recent races weighted more heavily)
# ewm_speed = pd.Series(speed_figures).ewm(alpha=0.4).mean().iloc[-1]
# Computed on per-horse PP lines, most-recent-first order
```

### Layoff Decay Curve
```python
# fitness(days) = exp(-lambda * max(0, days - recovery_threshold))
# recovery_threshold ≈ 30 days (short freshening is positive)
# lambda fit empirically per surface/distance category
```

---

## 9. DATA CONVENTIONS

- **Odds storage:** Always decimal (e.g., 3-1 → 4.0). Never fractional strings in schema.
- **Times:** Always seconds as float (e.g., 1:10.40 → 70.40).
- **Distances:** Always furlongs as float (e.g., 1 1/16 miles → 8.5).
- **Money:** Always USD float. No currency symbols in schema fields.
- **Dates:** Always `datetime.date`. No strings in schema date fields.
- **Probabilities:** Always [0.0, 1.0]. Never percentages in model internals.
- **Field-relative features:** Compute rank, percentile, and z-score within the race field, not across the historical database.
- **PP line ordering:** Most recent first in `HorseEntry.pp_lines`. Enforced by model validator.
- **Program number vs post position:** Store both. They diverge when horses scratch and the card is not reprinted.
- **Speed figures:** Store both the figure and its source tag ("beyer", "brisnet", "equibase", "timeform", "derived"). Never mix sources without normalizing.
- **Career stats:** Store as a JSON array of `CareerStatLine` objects. Do not flatten into columns — the label set varies per horse.
- **Workout lines:** Store as a separate list on `HorseEntry`. Ordered most-recent-first.
- **Comment/trip notes:** Store raw text. No parsing at ingestion time — NLP extraction is a Phase 2 feature.

---

## 10. TESTING REQUIREMENTS

Every service module must have a corresponding test. Minimum coverage per phase:

- **Phase 0 tests:** Feed a known CSV fixture through the full pipeline → assert DB row count, dedup key stability (same input = same key), quality gate correctly rejects bad rows, idempotency (second run adds zero rows).
- **Parser tests:** Feed a known-format text fixture → assert schema fields match expected values exactly. Use the Churchill Downs Race 4 and Race 9 cards as test fixtures.
- **Feature tests:** Feed a synthetic `RaceCard` → assert field-relative features are correctly normalized (mean≈0, within-field rank is monotone with speed figure).
- **Ordering model tests:** Feed known win probs → assert exotic probs sum correctly; assert Plackett-Luce outperforms Harville on longshot calibration.
- **EV engine tests:** Feed synthetic probs + odds → assert edge signs are correct; assert market impact reduces EV monotonically with bet size.
- **Portfolio tests:** Assert 1/4 Kelly never exceeds max_bet_fraction; assert CVaR constraint is binding at the limit.

Run tests before ending any session:
```bash
cd backend && pytest tests/ -v --tb=short
```

---

## 11. SESSION DISCIPLINE

- **Start every session:** `Read CLAUDE.md and PROGRESS.md. Confirm current phase and last completed task, then continue.`
- **End every session:** Update `PROGRESS.md` with what was completed, what is in progress, what is next, and any key decisions made.
- **Never leave a file half-written.** If a module is started, it must be complete and importable before ending the session. Use `# STUB` comments only for functions explicitly deferred to a later phase, and log them in PROGRESS.md.
- **Never break existing tests.** Run the full test suite before committing.
- **Phase 0 is standalone.** The `scripts/db/` directory has no dependency on `backend/app/`. It imports only from `scripts/db/` itself. This keeps the training pipeline runnable without standing up FastAPI.