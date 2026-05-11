# DATA_PIPELINE.md — Master Database Build Pipeline
# Read alongside CLAUDE.md. This governs Phase 0: data acquisition before any ML work.

---

## 1. OBJECTIVE

Build a **local master database** of every horse race we can find online. This database is the
training corpus for all ML models. Its quality determines the ceiling of every model downstream.

**Non-negotiable requirements:**
- Zero duplicate races or results
- Every record maps to the canonical schema before touching the DB
- Every record passes quality gates before insertion — bad data is worse than no data
- Full audit trail: every row knows which source dataset it came from
- The pipeline must be **idempotent** — running it twice on the same source produces
  the same DB state, not doubled records

---

## 2. MASTER DATABASE SCHEMA

The canonical tables. All Kaggle datasets must be normalized into this exact structure.

### `datasets` — audit trail of every source ingested
```sql
CREATE TABLE datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,           -- "kaggle:datasets/user/name"
    filename        TEXT NOT NULL,
    format          TEXT NOT NULL,           -- "csv" | "json" | "parquet"
    jurisdiction    TEXT NOT NULL,           -- "US" | "UK" | "HK" | "JP" | "AU"
    date_range_start DATE,
    date_range_end   DATE,
    row_count_raw    INTEGER,
    row_count_ingested INTEGER,
    row_count_rejected INTEGER,
    row_count_deduped INTEGER,
    ingested_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version   TEXT NOT NULL,
    notes            TEXT
);
```

### `tracks` — reference table; one row per track
```sql
CREATE TABLE tracks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    code         TEXT NOT NULL,              -- Equibase 3-5 char code (e.g. "CD", "SA", "KEE")
    name         TEXT,
    jurisdiction TEXT NOT NULL,              -- "US" | "UK" | "HK" | "JP" | "AU"
    country      TEXT,
    surface_types TEXT,                      -- JSON array: ["dirt","turf","synthetic"]
    UNIQUE(code, jurisdiction)
);
```

### `races` — one row per race; the primary deduplication target
```sql
CREATE TABLE races (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key        TEXT NOT NULL UNIQUE,   -- SHA256(track_code|race_date|race_num|distance_f|surface)
    track_code       TEXT NOT NULL,
    race_date        DATE NOT NULL,
    race_number      INTEGER NOT NULL,
    distance_furlongs REAL NOT NULL,
    surface          TEXT NOT NULL,          -- "dirt" | "turf" | "synthetic"
    condition        TEXT,                   -- "fast" | "good" | "sloppy" | etc.
    race_type        TEXT,                   -- "claiming" | "allowance" | "stakes" | etc.
    claiming_price   REAL,
    purse_usd        REAL,
    grade            INTEGER,                -- 1/2/3 for graded stakes; NULL otherwise
    field_size       INTEGER,
    jurisdiction     TEXT NOT NULL,
    weather          TEXT,
    age_sex_restrictions TEXT,
    source_dataset_id INTEGER REFERENCES datasets(id),
    raw_source_id    TEXT                    -- original ID in source dataset if available
);

CREATE INDEX idx_races_track_date ON races(track_code, race_date);
CREATE INDEX idx_races_date ON races(race_date);
```

### `horses` — one row per unique horse
```sql
CREATE TABLE horses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key       TEXT NOT NULL UNIQUE,    -- SHA256(normalized_name|foaling_year|country_of_origin)
    name_normalized TEXT NOT NULL,           -- lowercase, stripped punctuation
    name_display    TEXT NOT NULL,           -- original display name
    foaling_year    INTEGER,
    country_of_origin TEXT,
    sire            TEXT,
    dam             TEXT,
    dam_sire        TEXT,
    color           TEXT,
    sex             TEXT,                    -- "M" | "F" | "G" | "C" | "F" (filly)
    official_id     TEXT                     -- Equibase/Jockey Club ID when available
);

CREATE INDEX idx_horses_name ON horses(name_normalized);
```

### `jockeys` — one row per unique jockey
```sql
CREATE TABLE jockeys (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key       TEXT NOT NULL UNIQUE,    -- SHA256(normalized_name|jurisdiction)
    name_normalized TEXT NOT NULL,
    name_display    TEXT NOT NULL,
    jurisdiction    TEXT
);
```

### `trainers` — one row per unique trainer
```sql
CREATE TABLE trainers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key       TEXT NOT NULL UNIQUE,
    name_normalized TEXT NOT NULL,
    name_display    TEXT NOT NULL,
    jurisdiction    TEXT
);
```

### `race_results` — one row per horse per race; the ML training target
```sql
CREATE TABLE race_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key            TEXT NOT NULL UNIQUE,  -- SHA256(race_dedup_key|horse_dedup_key)
    race_id              INTEGER NOT NULL REFERENCES races(id),
    horse_id             INTEGER NOT NULL REFERENCES horses(id),
    jockey_id            INTEGER REFERENCES jockeys(id),
    trainer_id           INTEGER REFERENCES trainers(id),
    post_position        INTEGER,
    finish_position      INTEGER,
    lengths_behind       REAL,
    weight_lbs           REAL,
    odds_final           REAL,               -- decimal odds (e.g. 4.0 = 3-1)
    speed_figure         REAL,
    speed_figure_source  TEXT,               -- "beyer" | "brisnet" | "timeform" | "derived"
    fraction_q1_sec      REAL,               -- first call time in seconds
    fraction_q2_sec      REAL,               -- second call time in seconds
    fraction_finish_sec  REAL,               -- final time in seconds
    beaten_lengths_q1    REAL,
    beaten_lengths_q2    REAL,
    medication_flags     TEXT,               -- JSON array e.g. ["L","B"]
    equipment_changes    TEXT,               -- JSON array
    comment              TEXT,               -- trip note
    source_dataset_id    INTEGER REFERENCES datasets(id),
    data_quality_score   REAL                -- 0.0-1.0; computed by quality gate
);

CREATE INDEX idx_results_race ON race_results(race_id);
CREATE INDEX idx_results_horse ON race_results(horse_id);
CREATE INDEX idx_results_horse_date ON race_results(horse_id, race_id);
```

---

## 3. DEDUPLICATION KEYS

These are exact — implement them verbatim.

### Race dedup key
```python
import hashlib

def race_dedup_key(track_code: str, race_date: str, race_number: int,
                   distance_furlongs: float, surface: str) -> str:
    raw = f"{track_code.upper().strip()}|{race_date}|{race_number}|{round(distance_furlongs, 2)}|{surface.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

### Horse dedup key
```python
import re

def normalize_horse_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", "", name)  # strip punctuation
    name = re.sub(r"\s+", " ", name).strip()
    return name

def horse_dedup_key(name: str, foaling_year: int | None, country: str | None) -> str:
    norm = normalize_horse_name(name)
    year = str(foaling_year) if foaling_year else "unknown"
    cty = (country or "unknown").upper().strip()
    raw = f"{norm}|{year}|{cty}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

### Result dedup key
```python
def result_dedup_key(race_dedup_key: str, horse_dedup_key: str) -> str:
    raw = f"{race_dedup_key}|{horse_dedup_key}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

**Rule:** On INSERT conflict on dedup_key, the pipeline logs the duplicate and skips it.
It never overwrites existing rows. Source priority order (highest to lowest):
1. Equibase (official US results)
2. Brisnet (rich PP data)
3. DRF
4. Kaggle datasets (training volume; less authoritative)

---

## 4. KNOWN KAGGLE DATASETS — CATALOG

These are the datasets to acquire first. Sorted by quality and relevance.

| Dataset | Kaggle Slug | Jurisdiction | Years | Key Fields | Priority |
|---|---|---|---|---|---|
| Horse Racing Dataset (Zygmunt) | `zygmunt/horse-racing-dataset` | UK/IRE | 2005–2016 | finish pos, odds, distance, going, prize | HIGH |
| Horse Racing (Hong Kong HKJC) | `gdaley/horseracing-in-hk` | HK | 2014–2022 | full results, sectionals, class | HIGH |
| US Horse Racing Results | `joebeachcapital/horse-racing` | US | 2000–2022 | track, date, distance, surface, odds, speed fig | HIGH |
| Betfair Horse Racing | `adrianmcmahon/betfair-horse-racing` | UK | 2014–2019 | SP odds, BSP, in-play volume | MEDIUM |
| Horse Racing Historical | `hwaitt/horserace` | UK | 2010–2020 | basic results | MEDIUM |
| Australian Racing | `tobycrabtree/horse-racing-australia` | AU | 2015–2021 | class, distance, going | MEDIUM |

**How to add new datasets:** Run the evaluation script (Section 6) before ingesting anything.

---

## 5. FIELD MAPPING REGISTRY

Each Kaggle dataset uses different column names. The mapper normalizes them to canonical fields.
Add a new entry here whenever a new dataset is onboarded.

```python
# scripts/db/field_maps.py

FIELD_MAPS = {
    "zygmunt/horse-racing-dataset": {
        "source_format": "csv",
        "jurisdiction": "UK",
        "race_fields": {
            "race_id":         None,           # construct from date+venue+race_num
            "track_code":      "venue",
            "race_date":       "date",
            "race_number":     "race_num",
            "distance_furlongs": "dist",       # needs unit conversion (chains/furlongs)
            "surface":         "going",        # needs going→surface mapping
            "condition":       "going",        # same field, different semantic
            "race_type":       "type",
            "purse_usd":       "prize",        # GBP — needs fx conversion flag
        },
        "result_fields": {
            "horse_name":      "horse",
            "finish_position": "position",
            "jockey":          "jockey",
            "trainer":         "trainer",
            "odds_final":      "sp",           # starting price, fractional → decimal
            "weight_lbs":      "weight",       # stored as stones/pounds in source
            "lengths_behind":  "btn",
        },
        "transformers": {
            "distance_furlongs": "uk_distance_to_furlongs",
            "odds_final":        "uk_sp_to_decimal",
            "weight_lbs":        "stones_to_lbs",
            "surface":           "uk_going_to_surface",
            "condition":         "uk_going_to_condition",
            "purse_usd":         "gbp_to_usd",  # use fixed historical rate or mark as GBP
        }
    },

    "gdaley/horseracing-in-hk": {
        "source_format": "csv",
        "jurisdiction": "HK",
        "race_fields": {
            "track_code":        "venue",
            "race_date":         "date",
            "race_number":       "race_no",
            "distance_furlongs": "distance",   # stored in metres — divide by 201.168
            "surface":           "surface",
            "condition":         "going",
            "race_type":         "class",
            "purse_usd":         "prize",      # HKD — flag as HKD
        },
        "result_fields": {
            "horse_name":        "horse_name",
            "finish_position":   "place",
            "jockey":            "jockey",
            "trainer":           "trainer",
            "odds_final":        "win_odds",   # already decimal
            "weight_lbs":        "actual_weight",
            "speed_figure":      "rating",
        },
        "transformers": {
            "distance_furlongs": "metres_to_furlongs",
            "condition":         "hk_going_to_condition",
            "purse_usd":         "hkd_to_usd",
        }
    },

    "joebeachcapital/horse-racing": {
        "source_format": "csv",
        "jurisdiction": "US",
        "race_fields": {
            "track_code":        "track",
            "race_date":         "date",
            "race_number":       "race",
            "distance_furlongs": "distance",   # already furlongs
            "surface":           "surface",
            "condition":         "condition",
            "race_type":         "race_type",
            "claiming_price":    "claiming_price",
            "purse_usd":         "purse",
        },
        "result_fields": {
            "horse_name":        "horse",
            "finish_position":   "finish",
            "post_position":     "post",
            "jockey":            "jockey",
            "trainer":           "trainer",
            "odds_final":        "odds",       # verify decimal vs fractional
            "weight_lbs":        "weight",
            "speed_figure":      "speed_rating",
            "speed_figure_source": "beyer",    # hardcoded — this dataset uses Beyer
            "fraction_q1_sec":   "frac1",
            "fraction_q2_sec":   "frac2",
            "fraction_finish_sec": "final_time",
        },
        "transformers": {
            "fraction_q1_sec":   "time_string_to_seconds",
            "fraction_q2_sec":   "time_string_to_seconds",
            "fraction_finish_sec": "time_string_to_seconds",
        }
    },
}
```

---

## 6. DATASET EVALUATION SCRIPT

Before ingesting any new Kaggle dataset, run this evaluation. If it scores below the threshold,
do not ingest — garbage in, garbage out.

```
scripts/db/evaluate_dataset.py <path_to_csv_or_zip>
```

The evaluator checks and scores:

| Check | Weight | Pass Condition |
|---|---|---|
| Has race date | 15% | Parseable date column present |
| Has track identifier | 15% | Track name or code present |
| Has finish position | 20% | Numeric finish column |
| Has horse name | 15% | Non-null horse name |
| Has distance | 10% | Parseable distance |
| Has odds | 10% | Some odds column (morning line or final) |
| Date range > 1 year | 5% | Covers at least 12 months |
| Row count > 1,000 races | 5% | Meaningful volume |
| Duplicate rate < 5% | 5% | Internal dedup check |

**Minimum score to ingest: 0.70**

The evaluator outputs a JSON report:
```json
{
  "dataset": "path/to/file.csv",
  "score": 0.84,
  "pass": true,
  "field_coverage": { "race_date": true, "track": true, "finish": true, ... },
  "sample_rows": 3,
  "estimated_races": 45000,
  "estimated_results": 320000,
  "date_range": ["2005-01-01", "2016-12-31"],
  "jurisdiction_guess": "UK",
  "warnings": ["purse column missing", "speed figures absent"]
}
```

---

## 7. PIPELINE SCRIPTS

All scripts live in `scripts/db/`. Run them in order for each new dataset.

```
scripts/db/
├── setup_db.py             # Create all tables; run once
├── evaluate_dataset.py     # Evaluate before ingesting
├── ingest_kaggle.py        # Download + stage a Kaggle dataset
├── map_and_clean.py        # Apply field map + all transformers
├── quality_gate.py         # Per-row quality scoring before DB insert
├── load_to_db.py           # Insert to DB with dedup logic
├── dedup_report.py         # Show duplicate stats after any load
├── field_maps.py           # Field mapping registry (Section 5)
└── transformers.py         # All unit conversion functions
```

### Full pipeline for a new dataset
```bash
# 1. Download from Kaggle (requires kaggle CLI configured)
python scripts/db/ingest_kaggle.py --dataset "user/dataset-name" --output data/staging/

# 2. Evaluate before touching DB
python scripts/db/evaluate_dataset.py data/staging/dataset-name/

# 3. If score >= 0.70, map and clean
python scripts/db/map_and_clean.py \
    --input data/staging/dataset-name/ \
    --map "user/dataset-name" \
    --output data/cleaned/dataset-name/

# 4. Run quality gate — produces data/cleaned/dataset-name/accepted/ and rejected/
python scripts/db/quality_gate.py --input data/cleaned/dataset-name/

# 5. Load to DB
python scripts/db/load_to_db.py --input data/cleaned/dataset-name/accepted/

# 6. Verify
python scripts/db/dedup_report.py
```

---

## 8. DATA QUALITY GATES

Every row is scored 0.0–1.0 before insertion. Rows below the threshold go to a
`rejected/` folder with the reason logged — they are never inserted.

**Minimum quality score to insert: 0.60**

### Per-row scoring
```python
def score_race_result(row: dict) -> tuple[float, list[str]]:
    issues = []
    score = 1.0

    # Hard failures — score = 0, do not insert
    if not row.get("race_date"):
        return 0.0, ["missing race_date"]
    if not row.get("track_code"):
        return 0.0, ["missing track_code"]
    if not row.get("horse_name"):
        return 0.0, ["missing horse_name"]
    if row.get("finish_position") is None:
        return 0.0, ["missing finish_position"]

    # Soft failures — reduce score
    if not row.get("distance_furlongs"):
        score -= 0.20; issues.append("missing distance")
    if not row.get("surface"):
        score -= 0.15; issues.append("missing surface")
    if not row.get("odds_final"):
        score -= 0.10; issues.append("missing odds")
    if not row.get("speed_figure"):
        score -= 0.10; issues.append("missing speed figure")
    if not row.get("jockey"):
        score -= 0.05; issues.append("missing jockey")
    if not row.get("weight_lbs"):
        score -= 0.05; issues.append("missing weight")
    if not row.get("fraction_finish_sec"):
        score -= 0.10; issues.append("missing final time")

    # Range validation
    if row.get("finish_position", 0) < 1 or row.get("finish_position", 0) > 30:
        score -= 0.15; issues.append(f"invalid finish_position: {row['finish_position']}")
    if row.get("odds_final") is not None and row["odds_final"] < 1.0:
        score -= 0.15; issues.append(f"invalid odds: {row['odds_final']}")
    if row.get("distance_furlongs") is not None:
        if not (2.0 <= row["distance_furlongs"] <= 20.0):
            score -= 0.20; issues.append(f"implausible distance: {row['distance_furlongs']}f")
    if row.get("weight_lbs") is not None:
        if not (95.0 <= row["weight_lbs"] <= 145.0):
            score -= 0.10; issues.append(f"implausible weight: {row['weight_lbs']}lbs")

    return max(0.0, score), issues
```

### Additional cross-row validations (race level)
- If a race has a winner (finish_position=1), exactly one horse must have it
- Field size must match number of results for that race
- All horses in same race must share the same distance, surface, and track
- No horse can finish both 1st and 2nd in the same race

---

## 9. UNIT CONVERSION FUNCTIONS

All transformers referenced in `field_maps.py`. Implement in `scripts/db/transformers.py`.

```python
METRES_PER_FURLONG = 201.168
YARDS_PER_FURLONG = 220.0

def metres_to_furlongs(metres: float) -> float:
    return round(metres / METRES_PER_FURLONG, 4)

def uk_distance_to_furlongs(raw: str) -> float:
    """UK distances: '5f', '6f 10y', '1m', '1m2f', '1m4f', '2m'"""
    # ... parse miles + furlongs + yards combinations

def stones_to_lbs(raw: str | float) -> float:
    """'9-0' = 9 stone 0 lbs = 126 lbs; '9-2' = 9 stone 2 lbs = 128 lbs"""
    if isinstance(raw, float): return raw
    parts = str(raw).split("-")
    return int(parts[0]) * 14 + int(parts[1]) if len(parts) == 2 else float(raw)

def uk_sp_to_decimal(raw: str) -> float | None:
    """UK SP: '5/2', 'EVS', '100/30', '7/4 F' (favourite marker)"""
    # strip trailing markers (F, J-F, C-F, etc.)
    # then call parse_odds_to_decimal from cleaner.py

def uk_going_to_surface(going: str) -> str:
    """UK 'Firm', 'Good to Firm', 'Soft', 'Heavy' → 'turf' (UK flat is turf)"""
    # UK flat racing is almost exclusively turf; return "turf"
    # All-weather tracks (Lingfield, Kempton, Wolverhampton) → "synthetic"

def uk_going_to_condition(going: str) -> str:
    """'Good to Firm' → 'good', 'Heavy' → 'heavy', 'Standard' → 'fast'"""

def hk_going_to_condition(going: str) -> str:
    """HK uses 'Good', 'Good to Yielding', 'Yielding', 'Soft', 'Heavy'"""

def time_string_to_seconds(raw: str) -> float | None:
    """Delegate to cleaner.parse_time_to_seconds"""

def gbp_to_usd(amount: float, date: str) -> float:
    """For competition purposes, use fixed rate of 1.27. Flag in dataset notes."""
    return round(amount * 1.27, 2)

def hkd_to_usd(amount: float, date: str) -> float:
    """HKD is pegged; use fixed rate of 0.128."""
    return round(amount * 0.128, 2)
```

---

## 10. KAGGLE API SETUP

```bash
# Install Kaggle CLI
pip install kaggle

# Place API credentials
# Windows: C:\Users\<user>\.kaggle\kaggle.json
# Mac/Linux: ~/.kaggle/kaggle.json
# Content: {"username":"your_username","key":"your_api_key"}
# Get key from: kaggle.com → Account → API → Create New API Token

# Test
kaggle datasets list --search "horse racing"

# Download a specific dataset
kaggle datasets download -d zygmunt/horse-racing-dataset -p data/staging/
```

---

## 11. DIRECTORY STRUCTURE FOR DATA

```
data/
├── staging/              # raw downloads from Kaggle; never modify these
│   └── <dataset-slug>/
├── cleaned/              # after field mapping + transformer application
│   └── <dataset-slug>/
│       ├── accepted/     # passed quality gate; ready for DB load
│       └── rejected/     # failed quality gate; logged with reasons
├── exports/              # ML-ready feature matrices exported from DB
└── db/
    └── master.db         # the SQLite master database (dev)
                          # → PostgreSQL in production
```

All `staging/` downloads are gitignored. The DB file is gitignored.
`rejected/` logs are committed — they document data quality issues.

---

## 12. ML EXPORT QUERY

After the DB is built, this query produces the training DataFrame for the Speed/Form model.
Run via `scripts/db/export_training_data.py`.

```sql
SELECT
    r.race_date,
    r.track_code,
    r.jurisdiction,
    r.race_number,
    r.distance_furlongs,
    r.surface,
    r.condition,
    r.race_type,
    r.claiming_price,
    r.purse_usd,
    r.field_size,
    rr.post_position,
    rr.finish_position,
    rr.weight_lbs,
    rr.odds_final,
    rr.speed_figure,
    rr.speed_figure_source,
    rr.fraction_q1_sec,
    rr.fraction_q2_sec,
    rr.fraction_finish_sec,
    rr.beaten_lengths_q1,
    rr.beaten_lengths_q2,
    rr.data_quality_score,
    h.name_normalized   AS horse_name,
    h.foaling_year,
    h.sire,
    h.dam_sire,
    j.name_normalized   AS jockey_name,
    t.name_normalized   AS trainer_name
FROM race_results rr
JOIN races r        ON rr.race_id    = r.id
JOIN horses h       ON rr.horse_id   = h.id
LEFT JOIN jockeys j ON rr.jockey_id  = j.id
LEFT JOIN trainers t ON rr.trainer_id = t.id
WHERE rr.data_quality_score >= 0.60
  AND r.field_size >= 4
  AND rr.finish_position IS NOT NULL
ORDER BY r.race_date ASC, r.track_code, r.race_number, rr.post_position;
```

---

## 13. ADDING THIS PHASE TO THE IMPLEMENTATION ORDER

Phase 0 runs **before Phase 1** and is independent of the PDF ingestion pipeline.
Both can proceed in parallel once the DB schema is created.

| Phase | Focus |
|---|---|
| **0** | Master DB setup → Kaggle acquisition → evaluation → cleaning → dedup → load |
| **1** | PDF ingestion pipeline (live card data → inference input) |
| **2+** | Feature engineering, models, calibration, EV, portfolio (consume from DB + live PDFs) |
