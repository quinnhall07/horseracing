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
| `/card/[id]` | Race card viewer — sticky header, race tabs, horse table, EV gauge |
| `/card/[id]/portfolio` | Bet execution ticket — risk strip + recommendations |

## Backend contracts (consumed)

- `POST /api/v1/ingest/upload` → `IngestionResult` — already implemented (`app/api/v1/ingest.py`).
- `GET /api/v1/cards/{card_id}` → `RaceCard` — **not yet implemented**. Mock mode covers it.
- `GET /api/v1/portfolio/{card_id}` → `Portfolio` — **not yet implemented**. Mock mode covers it.

When wiring up the real endpoints, see `lib/types.ts` for the exact JSON
shape the frontend expects. The interfaces mirror the Pydantic v2 schemas
in `app/schemas/race.py` and `app/schemas/bets.py` field-for-field, with
two UI-only extensions on `HorseEntry`:

- `program_number`, `model_prob`, `market_prob`, `edge` — populated by
  whatever endpoint hydrates calibrated probabilities (likely `/analyze` once
  it exists).

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
- `/card/[id]` and `/card/[id]/portfolio` both work end-to-end with no
  backend running.
- Probabilities, edges, and the 6-bet portfolio are deterministic (seeded
  PRNG), so screenshots stay stable across refreshes.

## Design

- Dark theme by default (slate-950 background, indigo accents, emerald
  positive / rose negative).
- Inter (sans) + JetBrains Mono (numbers) via `next/font/google`.
- Tailwind utility classes only — no component libraries beyond Lucide for
  icons and Recharts for the EV gauge.
- Dense, info-rich tables (13px base, `xxs` 11px for labels) — this is a
  quant tool, not a marketing site.
