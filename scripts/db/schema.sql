-- Phase 0 master training database schema.
-- Source of truth: DATA_PIPELINE.md §2.
-- All statements are idempotent (CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS).

PRAGMA foreign_keys = ON;

-- ─── datasets ─────────────────────────────────────────────────────────────
-- Audit trail of every source ingested. Every row in races/race_results
-- references the dataset it came from via source_dataset_id.
CREATE TABLE IF NOT EXISTS datasets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source              TEXT NOT NULL,              -- "kaggle:datasets/user/name"
    filename            TEXT NOT NULL,
    format              TEXT NOT NULL,              -- "csv" | "json" | "parquet"
    jurisdiction        TEXT NOT NULL,              -- "US" | "UK" | "HK" | "JP" | "AU"
    date_range_start    DATE,
    date_range_end      DATE,
    row_count_raw       INTEGER,
    row_count_ingested  INTEGER,
    row_count_rejected  INTEGER,
    row_count_deduped   INTEGER,
    ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version      TEXT NOT NULL,
    notes               TEXT
);

-- ─── tracks ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    code          TEXT NOT NULL,
    name          TEXT,
    jurisdiction  TEXT NOT NULL,
    country       TEXT,
    surface_types TEXT,                              -- JSON array
    UNIQUE(code, jurisdiction)
);

-- ─── races ────────────────────────────────────────────────────────────────
-- Primary deduplication target.
CREATE TABLE IF NOT EXISTS races (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key             TEXT NOT NULL UNIQUE,
    track_code            TEXT NOT NULL,
    race_date             DATE NOT NULL,
    race_number           INTEGER NOT NULL,
    distance_furlongs     REAL NOT NULL,
    surface               TEXT NOT NULL,
    condition             TEXT,
    race_type             TEXT,
    claiming_price        REAL,
    purse_usd             REAL,
    grade                 INTEGER,
    field_size            INTEGER,
    jurisdiction          TEXT NOT NULL,
    weather               TEXT,
    age_sex_restrictions  TEXT,
    source_dataset_id     INTEGER REFERENCES datasets(id),
    raw_source_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_races_track_date ON races(track_code, race_date);
CREATE INDEX IF NOT EXISTS idx_races_date       ON races(race_date);

-- ─── horses ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS horses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key         TEXT NOT NULL UNIQUE,
    name_normalized   TEXT NOT NULL,
    name_display      TEXT NOT NULL,
    foaling_year      INTEGER,
    country_of_origin TEXT,
    sire              TEXT,
    dam               TEXT,
    dam_sire          TEXT,
    color             TEXT,
    sex               TEXT,
    official_id       TEXT
);

CREATE INDEX IF NOT EXISTS idx_horses_name ON horses(name_normalized);

-- ─── jockeys ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jockeys (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key       TEXT NOT NULL UNIQUE,
    name_normalized TEXT NOT NULL,
    name_display    TEXT NOT NULL,
    jurisdiction    TEXT
);

-- ─── trainers ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trainers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key       TEXT NOT NULL UNIQUE,
    name_normalized TEXT NOT NULL,
    name_display    TEXT NOT NULL,
    jurisdiction    TEXT
);

-- ─── race_results ─────────────────────────────────────────────────────────
-- One row per horse per race. The ML training target.
CREATE TABLE IF NOT EXISTS race_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key            TEXT NOT NULL UNIQUE,
    race_id              INTEGER NOT NULL REFERENCES races(id),
    horse_id             INTEGER NOT NULL REFERENCES horses(id),
    jockey_id            INTEGER REFERENCES jockeys(id),
    trainer_id           INTEGER REFERENCES trainers(id),
    post_position        INTEGER,
    finish_position      INTEGER,
    lengths_behind       REAL,
    weight_lbs           REAL,
    odds_final           REAL,
    speed_figure         REAL,
    speed_figure_source  TEXT,
    fraction_q1_sec      REAL,
    fraction_q2_sec      REAL,
    fraction_finish_sec  REAL,
    beaten_lengths_q1    REAL,
    beaten_lengths_q2    REAL,
    medication_flags     TEXT,                       -- JSON array
    equipment_changes    TEXT,                       -- JSON array
    comment              TEXT,
    source_dataset_id    INTEGER REFERENCES datasets(id),
    data_quality_score   REAL
);

CREATE INDEX IF NOT EXISTS idx_results_race       ON race_results(race_id);
CREATE INDEX IF NOT EXISTS idx_results_horse      ON race_results(horse_id);
CREATE INDEX IF NOT EXISTS idx_results_horse_date ON race_results(horse_id, race_id);
