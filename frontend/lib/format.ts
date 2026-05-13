/**
 * lib/format.ts
 * ─────────────
 * Display formatters for the quant-style table cells. All input units match
 * the backend's storage conventions (CLAUDE.md §9): decimal odds, furlong
 * distances, seconds for times, USD floats for money, [0,1] probabilities.
 *
 * Every formatter is null/undefined-safe and returns a dashed placeholder
 * for missing values — UI cells should never display "NaN" or "undefined".
 */

const DASH = "—";

const moneyFmt = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const moneyFmtCompact = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

/** Decimal odds → "4.2 (3.2-1)". 1-decimal precision on both representations. */
export function formatOdds(decimal: number | null | undefined): string {
  if (decimal == null || !Number.isFinite(decimal) || decimal < 1) return DASH;
  const fractional = decimal - 1;
  return `${decimal.toFixed(1)} (${fractional.toFixed(1)}-1)`;
}

/** [0,1] probability → "23.4%" with one decimal. */
export function formatProb(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return DASH;
  return `${(p * 100).toFixed(1)}%`;
}

/** USD float → "$1,234.50". */
export function formatMoney(usd: number | null | undefined): string {
  if (usd == null || !Number.isFinite(usd)) return DASH;
  return moneyFmt.format(usd);
}

/** USD float → "$1,234" (no cents, for large bankroll/header chips). */
export function formatMoneyCompact(usd: number | null | undefined): string {
  if (usd == null || !Number.isFinite(usd)) return DASH;
  return moneyFmtCompact.format(usd);
}

/**
 * Furlongs → "6f" for sprints, "1 1/16m" for routes (≥ 8f).
 * Common route distances are rendered as familiar mile fractions.
 */
export function formatDistance(furlongs: number | null | undefined): string {
  if (furlongs == null || !Number.isFinite(furlongs)) return DASH;
  if (furlongs < 8) {
    // Sprints — print furlongs with a half if applicable.
    const whole = Math.floor(furlongs);
    const frac = furlongs - whole;
    if (Math.abs(frac) < 0.05) return `${whole}f`;
    if (Math.abs(frac - 0.5) < 0.05) return `${whole} 1/2f`;
    return `${furlongs.toFixed(1)}f`;
  }
  // Routes — express in miles where possible.
  const miles = furlongs / 8;
  const knownRoutes: Record<string, string> = {
    "8.0": "1m",
    "8.5": "1 1/16m",
    "9.0": "1 1/8m",
    "9.5": "1 3/16m",
    "10.0": "1 1/4m",
    "10.5": "1 5/16m",
    "11.0": "1 3/8m",
    "12.0": "1 1/2m",
    "14.0": "1 3/4m",
    "16.0": "2m",
  };
  const key = furlongs.toFixed(1);
  if (key in knownRoutes) return knownRoutes[key];
  return `${miles.toFixed(2)}m`;
}

/**
 * Seconds (float) → "1:11.60" (minutes:seconds.hundredths) for >= 60s,
 * "48.20" for sub-minute fractional times.
 */
export function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return DASH;
  if (seconds < 60) return seconds.toFixed(2);
  const minutes = Math.floor(seconds / 60);
  const rem = seconds - minutes * 60;
  return `${minutes}:${rem.toFixed(2).padStart(5, "0")}`;
}

/** Signed edge (in [-1, 1] units) → "+12.3%" / "-5.4%". */
export function formatEdge(edge: number | null | undefined): string {
  if (edge == null || !Number.isFinite(edge)) return DASH;
  const pct = edge * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

/** Stake fraction in [0, 1] → "1.8%" (2 decimals). */
export function formatFraction(f: number | null | undefined): string {
  if (f == null || !Number.isFinite(f)) return DASH;
  return `${(f * 100).toFixed(2)}%`;
}

/**
 * Selection tuple → "1→2→3" — used in BetTicketRow to render exotic legs.
 * For a single-element tuple (WIN), returns just the bare number.
 */
export function formatSelection(selection: number[]): string {
  if (!selection.length) return DASH;
  return selection.join("→");
}

/** ISO date string → "May 10, 2026". */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Human-readable label for a BetType. */
export function formatBetType(t: string): string {
  const map: Record<string, string> = {
    win: "WIN",
    place: "PLACE",
    show: "SHOW",
    exacta: "EXACTA",
    trifecta: "TRIFECTA",
    superfecta: "SUPERFECTA",
    pick3: "PICK 3",
    pick4: "PICK 4",
    pick6: "PICK 6",
  };
  return map[t] ?? t.toUpperCase();
}
