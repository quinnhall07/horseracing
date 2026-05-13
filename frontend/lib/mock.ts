/**
 * lib/mock.ts
 * ───────────
 * Realistic demo payload: a 9-race Churchill Downs card from 2026-05-10
 * (mirroring the CD-05/10/2026 fixture referenced in CLAUDE.md §7).
 *
 * Per-race fields satisfy the production constraints:
 *   - 7-12 horses per race
 *   - model_prob sums to ~1.0 within each race (post-calibration constraint)
 *   - mix of +/- edges, with the larger edges concentrated on a few horses
 *   - portfolio has 4-8 BetRecommendations across the card
 *   - every stake_fraction ≤ 0.03 (ADR-002 1/4-Kelly + 3% cap)
 *   - cvar_95 ≤ 0.20 × bankroll
 */

import type {
  HorseEntry,
  IngestionResult,
  PastPerformanceLine,
  Portfolio,
  RaceCard,
  RaceHeader,
} from "./types";

const TODAY = "2026-05-10";

const HORSE_NAMES = [
  "Lovely Words",
  "Amazing Ascendis",
  "Bided",
  "California Smoke",
  "White Whale",
  "Velvet Hammer",
  "Northern Tempest",
  "Crimson Reign",
  "Pacific Lightning",
  "Steel Magnolia",
  "Royal Cadence",
  "Quantum Stride",
  "Silver Tongue",
  "Midnight Pact",
  "Highland Hymn",
  "Atlantic Echo",
  "Saratoga Star",
  "Iron Horse",
  "Daylight Saving",
  "Cascade Run",
];

const JOCKEYS = [
  "Castellano J",
  "Ortiz I Jr",
  "Rosario J",
  "Velazquez J",
  "Saez L",
  "Geroux F",
  "Hernandez B Jr",
  "Lanerie C",
  "Leparoux J",
  "Talamo J",
];

const TRAINERS = [
  "Pletcher T",
  "Asmussen S",
  "Brown C",
  "Mott W",
  "Cox B",
  "Casse M",
  "McGaughey C",
  "Walsh B",
  "Maker M",
  "Ortiz J A",
];

// ────────────────────────────────────────────────────────────────────────────
// Deterministic synthetic-data helpers (seeded so refreshes are stable)
// ────────────────────────────────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(arr: T[], rng: () => number): T {
  return arr[Math.floor(rng() * arr.length)] as T;
}

function makePPLines(rng: () => number, count: number, baseSpeed: number): PastPerformanceLine[] {
  const tracks = ["CD", "Kee", "Sar", "Bel", "GP", "OP"];
  const lines: PastPerformanceLine[] = [];
  for (let i = 0; i < count; i += 1) {
    const daysAgo = 21 + i * (28 + Math.floor(rng() * 14));
    const date = new Date(TODAY);
    date.setDate(date.getDate() - daysAgo);
    const finish = 1 + Math.floor(rng() * 9);
    lines.push({
      race_date: date.toISOString().slice(0, 10),
      track_code: pick(tracks, rng),
      race_number: 1 + Math.floor(rng() * 10),
      distance_furlongs: pick([6, 6.5, 7, 8, 8.5], rng),
      surface: "dirt",
      condition: "fast",
      race_type: pick(
        ["claiming", "allowance", "allowance_optional_claiming", "maiden_special_weight"] as const,
        rng,
      ),
      claiming_price: rng() > 0.5 ? 30000 + Math.floor(rng() * 50000) : null,
      purse_usd: 40000 + Math.floor(rng() * 80000),
      post_position: 1 + Math.floor(rng() * 10),
      finish_position: finish,
      lengths_behind: finish === 1 ? 0 : Number((rng() * 8).toFixed(2)),
      field_size: 7 + Math.floor(rng() * 5),
      jockey: pick(JOCKEYS, rng),
      weight_lbs: 118 + Math.floor(rng() * 6),
      odds_final: Number((2 + rng() * 12).toFixed(1)),
      speed_figure: Math.round(baseSpeed + (rng() - 0.5) * 14),
      speed_figure_source: "brisnet",
      fraction_q1: 22 + rng() * 1.5,
      fraction_q2: 45 + rng() * 2,
      fraction_finish: 70 + rng() * 4,
      beaten_lengths_q1: Number((rng() * 4).toFixed(2)),
      beaten_lengths_q2: Number((rng() * 4).toFixed(2)),
      days_since_prev: i === count - 1 ? null : 25 + Math.floor(rng() * 30),
      comment: pick(
        [
          "Tracked 3-wide, kicked clear",
          "Stalked, evenly late",
          "Rallied wide, willingly",
          "Bid 4w, hung",
          "Set pace, faded",
          "Saved ground, no factor",
          "Bumped start, late kick",
        ],
        rng,
      ),
    });
  }
  return lines;
}

// ────────────────────────────────────────────────────────────────────────────
// Field generator: 7-12 horses with model_probs that sum to ~1.0
// ────────────────────────────────────────────────────────────────────────────

function makeField(raceSeed: number, raceNumber: number): HorseEntry[] {
  const rng = mulberry32(raceSeed);
  const size = 7 + Math.floor(rng() * 6); // 7-12
  // Generate Dirichlet-ish weights, then normalise.
  const rawWeights = Array.from({ length: size }, () => -Math.log(rng() + 1e-6));
  const wSum = rawWeights.reduce((a, b) => a + b, 0);
  const modelProbs = rawWeights.map((w) => w / wSum);

  // Morning-line market probs: take model probs and inject ~10% noise + 18% overround.
  const noisyMarket = modelProbs.map((p) => Math.max(0.02, p * (0.8 + rng() * 0.5)));
  const marketSum = noisyMarket.reduce((a, b) => a + b, 0);
  const overround = 1.18;
  const marketProbs = noisyMarket.map((p) => (p / marketSum) * overround);

  const used = new Set<string>();
  const horses: HorseEntry[] = [];
  for (let i = 0; i < size; i += 1) {
    let name = "";
    do {
      name = pick(HORSE_NAMES, rng);
    } while (used.has(name));
    used.add(name);

    const market = Math.min(0.95, marketProbs[i] as number);
    const mlOdds = Number((1 / market).toFixed(1));
    const model = modelProbs[i] as number;
    const edge = model - market;
    const ppCount = 4 + Math.floor(rng() * 5);
    const baseSpeed = 75 + model * 60;

    horses.push({
      horse_name: name,
      horse_id: `H-${raceSeed}-${i}`,
      post_position: i + 1,
      morning_line_odds: mlOdds,
      ml_implied_prob: market,
      jockey: pick(JOCKEYS, rng),
      trainer: pick(TRAINERS, rng),
      owner: "Demo Owner Stable",
      weight_lbs: 118 + Math.floor(rng() * 6),
      medication_flags: rng() > 0.3 ? ["L"] : [],
      equipment_changes: rng() > 0.85 ? ["blinkers_on"] : [],
      pp_lines: makePPLines(mulberry32(raceSeed * 100 + i), ppCount, baseSpeed),
      pace_style: pick(["front_runner", "presser", "stalker", "closer"] as const, rng),
      ewm_speed_figure: Number(baseSpeed.toFixed(1)),
      days_since_last: 21 + Math.floor(rng() * 30),
      class_trajectory: Number(((rng() - 0.5) * 0.4).toFixed(3)),
      program_number: String(i + 1),
      model_prob: Number(model.toFixed(4)),
      market_prob: Number(market.toFixed(4)),
      edge: Number(edge.toFixed(4)),
    });
  }
  // Sort by model_prob descending so the UI gets ordered rows out of the box.
  horses.sort((a, b) => (b.model_prob ?? 0) - (a.model_prob ?? 0));
  // Reassign post_position to match field order for the demo (matches real cards
  // where the strongest horse may draw any post; we just want stable display).
  horses.forEach((h, idx) => {
    h.program_number = String(idx + 1);
  });
  // Keep race_number on attached PPs only; nothing to do here. Race header carries it.
  void raceNumber;
  return horses;
}

// ────────────────────────────────────────────────────────────────────────────
// 9-race Churchill Downs card
// ────────────────────────────────────────────────────────────────────────────

const RACE_CONDITIONS: Array<{
  distance: number;
  raw: string;
  type: RaceHeader["race_type"];
  purse: number;
  postTime: string;
  name?: string;
  grade?: number;
  claiming?: number;
}> = [
  { distance: 6, raw: "6 FURLONGS", type: "maiden_special_weight", purse: 92000, postTime: "12:45PM" },
  { distance: 5.5, raw: "5 1/2 FURLONGS", type: "claiming", purse: 38000, postTime: "1:14PM", claiming: 20000 },
  { distance: 8.5, raw: "1 1/16 MILES", type: "allowance", purse: 112000, postTime: "1:43PM" },
  { distance: 4.5, raw: "4 1/2 FURLONGS", type: "claiming", purse: 62000, postTime: "2:14PM", claiming: 30000 },
  { distance: 9, raw: "1 1/8 MILES", type: "graded_stakes", purse: 600000, postTime: "2:48PM", name: "Alysheba S.", grade: 2 },
  { distance: 7, raw: "7 FURLONGS", type: "allowance_optional_claiming", purse: 130000, postTime: "3:24PM" },
  { distance: 8, raw: "1 MILE", type: "allowance", purse: 105000, postTime: "4:02PM" },
  { distance: 10, raw: "1 1/4 MILES", type: "graded_stakes", purse: 5000000, postTime: "6:57PM", name: "Kentucky Derby G1", grade: 1 },
  { distance: 6.5, raw: "6 1/2 FURLONGS", type: "claiming", purse: 45000, postTime: "7:46PM", claiming: 30000 },
];

function buildCard(): RaceCard {
  const races = RACE_CONDITIONS.map((cond, i) => {
    const raceNumber = i + 1;
    const header: RaceHeader = {
      race_number: raceNumber,
      race_date: TODAY,
      track_code: "CD",
      track_name: "Churchill Downs",
      race_name: cond.name ?? null,
      distance_furlongs: cond.distance,
      distance_raw: cond.raw,
      surface: "dirt",
      condition: "fast",
      race_type: cond.type,
      claiming_price: cond.claiming ?? null,
      purse_usd: cond.purse,
      grade: cond.grade ?? null,
      age_sex_restrictions: i === 0 ? "Two Year Olds" : "Three Year Olds & Up",
      weight_conditions: "Non-winners of two races since April 10 allowed 2 lbs",
      post_time: cond.postTime,
      weather: "Partly Cloudy, 72°F",
    };
    const entries = makeField(42 + raceNumber * 17, raceNumber);
    return {
      header,
      entries,
      parse_confidence: 0.92 + (raceNumber % 3) * 0.02,
      parse_warnings: [],
    };
  });

  return {
    source_filename: "CD-05-10-2026.pdf",
    source_format: "brisnet_up",
    total_pages: 36,
    card_date: TODAY,
    track_code: "CD",
    races,
    card_id: "mock-cd-2026-05-10",
  };
}

const MOCK_CARD: RaceCard = buildCard();

// ────────────────────────────────────────────────────────────────────────────
// Portfolio: 6 recommendations spread across the card; ADR-002 enforced.
// ────────────────────────────────────────────────────────────────────────────

const BANKROLL = 10_000;

function buildPortfolio(card: RaceCard): Portfolio {
  // Hand-picked spread: a couple of WINs, an exacta, a trifecta, a superfecta.
  // Stake fractions deliberately near (but never above) the 3% cap.
  const picks = [
    { raceIdx: 0, type: "win" as const, selection: [1], stakeFrac: 0.028 },
    { raceIdx: 2, type: "win" as const, selection: [1], stakeFrac: 0.024 },
    { raceIdx: 3, type: "exacta" as const, selection: [1, 2], stakeFrac: 0.018 },
    { raceIdx: 4, type: "trifecta" as const, selection: [1, 2, 3], stakeFrac: 0.012 },
    { raceIdx: 5, type: "win" as const, selection: [2], stakeFrac: 0.022 },
    { raceIdx: 7, type: "superfecta" as const, selection: [1, 2, 3, 4], stakeFrac: 0.008 },
  ];

  const recs = picks.map((p) => {
    const race = card.races[p.raceIdx];
    if (!race) throw new Error(`mock: missing race ${p.raceIdx}`);
    const topHorse = race.entries[(p.selection[0] ?? 1) - 1];
    const modelProb = topHorse?.model_prob ?? 0.2;
    const marketProb = topHorse?.market_prob ?? 0.18;
    const decimalOdds = topHorse?.morning_line_odds ?? 5.0;
    // For exotics, scale down probability multiplicatively (very rough — this
    // is mock data, not a real EV calculation).
    const legs = p.selection.length;
    const adjModel = Math.pow(modelProb, legs);
    const adjMarket = Math.pow(marketProb, legs);
    const adjOdds = Math.pow(decimalOdds, legs);
    const edge = adjModel - adjMarket;
    const ev = adjModel * adjOdds - 1;
    return {
      candidate: {
        race_id: `r${race.header.race_number}`,
        bet_type: p.type,
        selection: p.selection,
        model_prob: Number(adjModel.toFixed(5)),
        decimal_odds: Number(adjOdds.toFixed(2)),
        market_prob: Number(adjMarket.toFixed(5)),
        edge: Number(edge.toFixed(4)),
        expected_value: Number(ev.toFixed(4)),
        kelly_fraction: Number((p.stakeFrac * 4).toFixed(4)), // 1/4 Kelly invariant
        market_impact_applied: legs > 1,
        pool_size: legs > 1 ? 250_000 : null,
      },
      stake: Number((p.stakeFrac * BANKROLL).toFixed(2)),
      stake_fraction: p.stakeFrac,
    };
  });

  const totalStakeFrac = recs.reduce((a, r) => a + r.stake_fraction, 0);
  const expectedReturn = recs.reduce(
    (a, r) => a + r.stake * r.candidate.expected_value,
    0,
  );

  return {
    card_id: card.card_id ?? "mock-cd-2026-05-10",
    bankroll: BANKROLL,
    recommendations: recs,
    expected_return: Number(expectedReturn.toFixed(2)),
    var_95: Number((-0.04 * BANKROLL).toFixed(2)),
    cvar_95: Number((-0.085 * BANKROLL).toFixed(2)),
    total_stake_fraction: Number(totalStakeFrac.toFixed(4)),
  };
}

const MOCK_PORTFOLIO: Portfolio = buildPortfolio(MOCK_CARD);

// ────────────────────────────────────────────────────────────────────────────
// Public mock accessors
// ────────────────────────────────────────────────────────────────────────────

export function mockIngestionResult(filename: string): IngestionResult {
  const cloned: RaceCard = { ...MOCK_CARD, source_filename: filename || MOCK_CARD.source_filename };
  return {
    success: true,
    card: cloned,
    card_id: cloned.card_id ?? null,
    errors: [],
    processing_ms: 842.5,
  };
}

export function mockCard(id?: string): RaceCard {
  if (!id || id === MOCK_CARD.card_id) return MOCK_CARD;
  return { ...MOCK_CARD, card_id: id };
}

export function mockPortfolio(id?: string): Portfolio {
  if (!id || id === MOCK_PORTFOLIO.card_id) return MOCK_PORTFOLIO;
  return { ...MOCK_PORTFOLIO, card_id: id };
}
