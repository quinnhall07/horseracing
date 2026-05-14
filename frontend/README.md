# Horse Racing Frontend (Phase 6)

Next.js 14 App Router frontend for the Pari-Mutuel Analytics system. Renders
race cards parsed by the FastAPI backend and visualises the CVaR-optimised
1/4-Kelly portfolio produced by Phase 5.

See the root [`CLAUDE.md`](../CLAUDE.md) for system architecture and
implementation phases. Phase 6 spec is in section 3 (stack) and section 4
(repo structure).

## Setup

```bash
cd frontend
npm install
cp .env.example .env.local   # adjust if your backend lives elsewhere
npm run dev                  # http://localhost:3000
```

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | FastAPI base URL |
| `NEXT_PUBLIC_MOCK_API` | unset | Set to `"true"` to bypass the network and use `lib/mock.ts` |

## URL structure

| Path | Page |
|---|---|
| `/` | PDF upload landing with demo-mode fallback |
| `/card/[id]` | Unified result view — bankroll input, pareto risk/return curve, risk slider, bet ticket that updates as the user slides along the curve, expandable race-detail accordions |
| `/card/[id]/portfolio` | Redirects to `/card/[id]#races` for legacy bookmarks |

The user flow: upload a PDF → parser extracts the card → the backend's
inference pipeline scores every horse → the optimiser solves 6 portfolios at
6 risk levels → frontend renders the frontier. Click any point on the curve
(or press a stop on the slider) to switch the bet ticket in place.

## Backend contracts (consumed)

- `POST /api/v1/ingest/upload` → `IngestionResult` — Stream A.
- `GET /api/v1/cards/{card_id}` → `RaceCard` (with hydrated `model_prob` /
  `market_prob` / `edge` per `HorseEntry`) — Stream A.
- `GET /api/v1/portfolio/{card_id}` → `Portfolio` (single-risk-level
  aggregated card portfolio) — Stream A.
- `GET /api/v1/portfolio/{card_id}/pareto?risk_levels=…` → `ParetoFrontier`
  (6 Portfolios at distinct CVaR drawdown caps) — Stream X (ADR-045).

All four endpoints are live as of the current main branch. The interfaces
in `lib/types.ts` mirror the Pydantic v2 schemas in `app/schemas/race.py` and
`app/schemas/bets.py` field-for-field.

## Local commands

```bash
npm run dev          # development server
npm run build        # production build
npm run typecheck    # tsc --noEmit
npm run lint         # next lint
```

## Demo mode

With `NEXT_PUBLIC_MOCK_API=true`:

- Upload accepts any file (or click "load a synthetic demo card") and returns
  the seeded 9-race Churchill Downs card from `lib/mock.ts`.
- `/card/[id]` renders the full pareto-driven view with deterministic
  fake data — 6 frontier points monotone in (risk, return), 6 BetRecommendations
  total spread across the points.
- Probabilities, edges, and stake fractions are deterministic (seeded PRNG),
  so screenshots stay stable across refreshes.

`NEXT_PUBLIC_MOCK_API` defaults to `false` in `.env.example` — flip it to
`true` for frontend-only iteration.

## Design

- Dark theme by default (slate-950 background, indigo accents, emerald
  positive / rose negative).
- Inter (sans) + JetBrains Mono (numbers) via `next/font/google`.
- Tailwind utility classes only — no component libraries beyond Lucide for
  icons and Recharts for the EV gauge.
- Dense, info-rich tables (13px base, `xxs` 11px for labels) — this is a
  quant tool, not a marketing site.
