/**
 * lib/api.ts
 * ──────────
 * Typed fetch wrappers around the FastAPI backend.
 *
 * MOCK MODE: when `NEXT_PUBLIC_MOCK_API === "true"`, every call resolves
 * from `lib/mock.ts`. This lets the entire frontend boot and demo without
 * a running backend — required because the /analyze and /portfolio
 * endpoints documented in CLAUDE.md §4 are not yet implemented in
 * `app/api/v1/` (only ingest.py exists today).
 *
 * Real-mode contracts (suggested, see README):
 *   POST /api/v1/ingest/upload       → IngestionResult                ✅ exists
 *   GET  /api/v1/cards/{card_id}     → RaceCard                       ⛔ todo
 *   GET  /api/v1/portfolio/{card_id} → Portfolio                      ⛔ todo
 */

import type {
  IngestionResult,
  ParetoFrontier,
  Portfolio,
  RaceCard,
} from "./types";
import {
  mockCard,
  mockIngestionResult,
  mockParetoFrontier,
  mockPortfolio,
} from "./mock";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const MOCK = process.env.NEXT_PUBLIC_MOCK_API === "true";

export interface ApiError extends Error {
  status: number;
  detail: string;
}

function makeError(status: number, detail: string): ApiError {
  const err = new Error(`API ${status}: ${detail}`) as ApiError;
  err.status = status;
  err.detail = detail;
  return err;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) detail = JSON.stringify(body.detail);
    } catch {
      // body was not JSON; keep the status-text detail
    }
    throw makeError(res.status, detail);
  }
  return (await res.json()) as T;
}

/** Simulates a 600-900ms async delay so mock UX matches real network feel. */
async function mockDelay<T>(value: T, ms = 700): Promise<T> {
  await new Promise((r) => setTimeout(r, ms + Math.random() * 200 - 100));
  return value;
}

// ────────────────────────────────────────────────────────────────────────────
// Public surface
// ────────────────────────────────────────────────────────────────────────────

export async function uploadCard(file: File): Promise<IngestionResult> {
  if (MOCK) return mockDelay(mockIngestionResult(file.name));
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/ingest/upload`, {
    method: "POST",
    body: form,
  });
  return jsonOrThrow<IngestionResult>(res);
}

export async function getCard(id: string): Promise<RaceCard> {
  if (MOCK) return mockDelay(mockCard(id), 350);
  const res = await fetch(`${API_BASE}/api/v1/cards/${encodeURIComponent(id)}`, {
    headers: { Accept: "application/json" },
  });
  return jsonOrThrow<RaceCard>(res);
}

export async function getPortfolio(id: string): Promise<Portfolio> {
  if (MOCK) return mockDelay(mockPortfolio(id), 350);
  const res = await fetch(
    `${API_BASE}/api/v1/portfolio/${encodeURIComponent(id)}`,
    { headers: { Accept: "application/json" } },
  );
  return jsonOrThrow<Portfolio>(res);
}

export interface ParetoFrontierOptions {
  bankroll?: number;
  minEdge?: number;
  maxDecimalOdds?: number;
  cvarAlpha?: number;
  nScenarios?: number;
  seed?: number;
  riskLevels?: number[];
}

/**
 * GET /api/v1/portfolio/{card_id}/pareto
 *
 * Returns the full risk/return frontier — one Portfolio per risk level. The
 * frontend renders the curve and lets the user slide along it without
 * additional round-trips. Query parameters mirror the optimiser knobs that
 * Stream X exposes.
 */
export async function getParetoFrontier(
  cardId: string,
  options?: ParetoFrontierOptions,
): Promise<ParetoFrontier> {
  if (MOCK) return mockDelay(mockParetoFrontier(cardId, options), 700);

  const params = new URLSearchParams();
  if (options?.bankroll != null) params.set("bankroll", String(options.bankroll));
  if (options?.minEdge != null) params.set("min_edge", String(options.minEdge));
  if (options?.maxDecimalOdds != null)
    params.set("max_decimal_odds", String(options.maxDecimalOdds));
  if (options?.cvarAlpha != null) params.set("cvar_alpha", String(options.cvarAlpha));
  if (options?.nScenarios != null) params.set("n_scenarios", String(options.nScenarios));
  if (options?.seed != null) params.set("seed", String(options.seed));
  if (options?.riskLevels?.length)
    params.set("risk_levels", options.riskLevels.join(","));

  const qs = params.toString();
  const url = `${API_BASE}/api/v1/portfolio/${encodeURIComponent(cardId)}/pareto${
    qs ? `?${qs}` : ""
  }`;
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  return jsonOrThrow<ParetoFrontier>(res);
}

export const apiConfig = { API_BASE, MOCK } as const;
