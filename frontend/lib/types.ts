/**
 * lib/types.ts
 * ────────────
 * TypeScript mirrors of the Pydantic v2 schemas in app/schemas/race.py and
 * app/schemas/bets.py. Field names match the backend serialisation exactly
 * (snake_case) so JSON responses can be cast directly without remapping.
 *
 * Optional fields are typed `T | null` because Pydantic v2 emits `null` for
 * unset Optional fields (not undefined). `?` is reserved for genuinely
 * absent keys (rare in this schema).
 */

// ────────────────────────────────────────────────────────────────────────────
// Enumerations (StrEnum on the Python side)
// ────────────────────────────────────────────────────────────────────────────

export type Surface = "dirt" | "turf" | "synthetic" | "unknown";

export type TrackCondition =
  // dirt
  | "fast"
  | "good"
  | "sloppy"
  | "muddy"
  | "heavy"
  | "frozen"
  // turf
  | "firm"
  | "yielding"
  | "soft"
  // fallback
  | "unknown";

export type RaceType =
  | "maiden_claiming"
  | "maiden_special_weight"
  | "claiming"
  | "allowance"
  | "allowance_optional_claiming"
  | "stakes"
  | "graded_stakes"
  | "handicap"
  | "unknown";

export type PaceStyle =
  | "front_runner"
  | "presser"
  | "stalker"
  | "closer"
  | "unknown";

export type BetType =
  | "win"
  | "place"
  | "show"
  | "exacta"
  | "trifecta"
  | "superfecta"
  | "pick3"
  | "pick4"
  | "pick6";

// ────────────────────────────────────────────────────────────────────────────
// PastPerformanceLine
// ────────────────────────────────────────────────────────────────────────────

export interface PastPerformanceLine {
  race_date: string; // ISO YYYY-MM-DD
  track_code: string;
  race_number: number;
  distance_furlongs: number;
  surface: Surface;
  condition: TrackCondition;
  race_type: RaceType;
  claiming_price: number | null;
  purse_usd: number | null;

  post_position: number;
  finish_position: number | null;
  lengths_behind: number | null;
  field_size: number | null;

  jockey: string | null;
  weight_lbs: number | null;
  odds_final: number | null;

  speed_figure: number | null;
  speed_figure_source: string;

  fraction_q1: number | null;
  fraction_q2: number | null;
  fraction_finish: number | null;

  beaten_lengths_q1: number | null;
  beaten_lengths_q2: number | null;

  days_since_prev: number | null;
  comment: string | null;
}

// ────────────────────────────────────────────────────────────────────────────
// HorseEntry
// ────────────────────────────────────────────────────────────────────────────

export interface HorseEntry {
  horse_name: string;
  horse_id: string | null;
  post_position: number;

  morning_line_odds: number | null;
  ml_implied_prob: number | null;

  jockey: string | null;
  trainer: string | null;
  owner: string | null;
  weight_lbs: number | null;

  medication_flags: string[];
  equipment_changes: string[];

  pp_lines: PastPerformanceLine[];

  pace_style: PaceStyle;

  ewm_speed_figure: number | null;
  days_since_last: number | null;
  class_trajectory: number | null;

  // UI-only convenience fields. The backend currently does not populate these
  // on the raw schema; they live on derived objects from the analyze endpoint.
  // The mock payload sets them; production should fold them in via /analyze.
  program_number?: string | null;
  model_prob?: number | null;
  market_prob?: number | null;
  edge?: number | null;
}

// ────────────────────────────────────────────────────────────────────────────
// RaceHeader / ParsedRace / RaceCard
// ────────────────────────────────────────────────────────────────────────────

export interface RaceHeader {
  race_number: number;
  race_date: string;
  track_code: string;
  track_name: string | null;
  race_name: string | null;

  distance_furlongs: number;
  distance_raw: string;

  surface: Surface;
  condition: TrackCondition;
  race_type: RaceType;

  claiming_price: number | null;
  purse_usd: number | null;
  grade: number | null;

  age_sex_restrictions: string | null;
  weight_conditions: string | null;

  post_time: string | null;
  weather: string | null;
}

export interface ParsedRace {
  header: RaceHeader;
  entries: HorseEntry[];
  parse_confidence: number;
  parse_warnings: string[];
}

export interface RaceCard {
  source_filename: string;
  source_format: string;
  total_pages: number;
  card_date: string | null;
  track_code: string | null;
  races: ParsedRace[];

  // UI-only — populated by the analyze endpoint (or mock) to identify the
  // persisted card. The /upload backend returns the persisted card_id on
  // the IngestionResult; mock mode synthesises one.
  card_id?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// IngestionResult / IngestionStatus
// ────────────────────────────────────────────────────────────────────────────

export interface IngestionResult {
  success: boolean;
  card: RaceCard | null;
  errors: string[];
  processing_ms: number | null;
  // Forward-compatible field — the backend persists with a card_id but the
  // current ingest endpoint does not return it. Mock provides one and we
  // surface whatever the backend gives us under either key.
  card_id?: string | null;
}

export interface IngestionStatus {
  job_id: string;
  status: "queued" | "parsing" | "feature_engineering" | "done" | "failed";
  progress_pct: number;
  result: IngestionResult | null;
  error: string | null;
}

// ────────────────────────────────────────────────────────────────────────────
// BetCandidate / BetRecommendation / Portfolio
// ────────────────────────────────────────────────────────────────────────────

export interface BetCandidate {
  race_id: string;
  bet_type: BetType;
  /** Pydantic emits this as a list. Python tuple → JSON array → number[]. */
  selection: number[];

  model_prob: number;
  decimal_odds: number;
  market_prob: number;
  edge: number;
  expected_value: number;
  kelly_fraction: number;
  market_impact_applied: boolean;
  pool_size: number | null;
}

export interface BetRecommendation {
  candidate: BetCandidate;
  stake: number;
  stake_fraction: number;
}

export interface Portfolio {
  card_id: string;
  bankroll: number;
  recommendations: BetRecommendation[];
  expected_return: number;
  var_95: number;
  cvar_95: number;
  total_stake_fraction: number;
}

// ────────────────────────────────────────────────────────────────────────────
// ParetoPoint / ParetoFrontier
// ────────────────────────────────────────────────────────────────────────────
//
// The /pareto endpoint returns 6 (configurable) full Portfolio objects, each
// solved at a different CVaR drawdown cap. Frontend uses this to render a
// risk/return curve the user can scrub along — no extra round-trips per stop.

export interface ParetoPoint {
  /** Per-card CVaR drawdown cap used to solve this Portfolio (e.g. 0.05). */
  max_drawdown_pct: number;
  portfolio: Portfolio;
}

export interface ParetoFrontier {
  card_id: string;
  bankroll: number;
  /** Pre-filter candidate count fed into each LP solve. */
  n_candidates_total: number;
  frontier: ParetoPoint[];
}
