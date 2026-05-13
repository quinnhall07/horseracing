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

import type { IngestionResult, Portfolio, RaceCard } from "./types";
import { mockCard, mockIngestionResult, mockPortfolio } from "./mock";

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

export const apiConfig = { API_BASE, MOCK } as const;
