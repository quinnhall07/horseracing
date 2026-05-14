# Horse Racing Betting System

End-to-end pari-mutuel wagering analytics for a university finance paper-trading
competition. Ingests Brisnet UP / Equibase past-performance PDFs, runs a 7-layer
ensemble ML pipeline (orthogonalized sub-models → meta-learner → calibrator →
Plackett-Luce ordering → CVaR-constrained portfolio optimiser), and surfaces a
risk/return pareto frontier of +EV bets the user can scrub along.

Full architecture, math, and model design live in
[`Horse_Racing_System_Master_Reference.md`](Horse_Racing_System_Master_Reference.md).
Day-to-day operating instructions for AI coding sessions live in
[`CLAUDE.md`](CLAUDE.md). Architectural decisions and their rationale are in
[`DECISIONS.md`](DECISIONS.md).

## Quickstart — clean clone to working demo

```bash
git clone <repo-url>
cd horseracing

# Backend
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest pytest-asyncio httpx reportlab    # test/dev extras

python scripts/quick_bootstrap.py                    # synthetic models, ~30s
uvicorn app.main:app --reload                        # backend on :8000

# Frontend (in another shell)
cd frontend
cp .env.example .env.local
npm install
npm run dev                                          # frontend on :3000

# Open http://localhost:3000 and upload EXAMPLE_RACE_CARDS/Race 4.pdf
```

The `quick_bootstrap` script generates synthetic training data and fits baseline
models that satisfy the inference pipeline's loader contract. **Its predictions
are not meaningful** — the script exists so the API responds 200 instead of 503
on a clean clone. For real predictions, follow the data pipeline below.

## Real-data path

For production models you need historical race data. The pipeline is in
[`DATA_PIPELINE.md`](DATA_PIPELINE.md):

```bash
# Phase 0 — master training DB build
python scripts/db/setup_db.py
python scripts/db/ingest_kaggle.py --dataset <slug>
python scripts/db/evaluate_dataset.py data/staging/<slug>/
python scripts/db/map_and_clean.py --input data/staging/<slug>/ --map <slug>
python scripts/db/quality_gate.py --input data/cleaned/<slug>/
python scripts/db/load_to_db.py --input data/cleaned/<slug>/accepted/
python scripts/db/export_training_data.py            # produces parquet

# Phase 3 — full bootstrap
python scripts/bootstrap_models.py                   # consumes the parquet
```

Both `data/exports/*.parquet` and `models/baseline_full/` are gitignored —
they're rebuilt per environment.

## Architecture

Seven layers, end-to-end:

| Layer | Purpose | Modules |
|---|---|---|
| 0 | Master training DB | `scripts/db/`, `data/db/master.db` |
| 1 | PDF ingestion | `app/services/pdf_parser/`, `app/api/v1/ingest.py` |
| 2 | Feature engineering | `app/services/feature_engineering/` |
| 3 | Sub-models (5) + meta-learner | `app/services/models/` |
| 4 | Calibration + Plackett-Luce ordering | `app/services/calibration/`, `app/services/ordering/` |
| 5 | EV engine + CVaR portfolio optimiser | `app/services/ev_engine/`, `app/services/portfolio/` |
| 6 | Frontend | `frontend/` |
| 7 | Feedback loop (outcomes + drift + rolling retrain) | `app/services/feedback/`, `scripts/rolling_retrain.py` |

The frontend's pareto-driven UX is one new API call away from the optimizer:
`GET /api/v1/portfolio/{card_id}/pareto` returns 6 portfolios at different CVaR
drawdown caps; the user slides along the curve to pick a risk level.

## Running tests

```bash
.venv/bin/pytest tests/ -q
# 652 tests passing, ~13s
```

The full suite covers parser, feature engineering, sub-models, ordering models,
calibration, EV engine, portfolio optimizer, pareto frontier, inference
pipeline, outcomes logging, drift detection, and rolling-retrain script.
Frontend tests are out of scope; visual verification happens via `npm run dev`.

GitHub Actions runs the backend matrix (py 3.11 / 3.12 / 3.13) and the frontend
`typecheck` + `lint` + `build` on every push and PR — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Deployment

### Frontend → Vercel

The frontend is a stock Next.js 14 App Router app; deploys to Vercel with zero
custom server code.

1. Push the repo to GitHub.
2. Vercel → "Import Project" → connect your GitHub account → select this repo.
3. In the Vercel project settings, set **Root Directory** to `frontend/`.
   Framework auto-detects as Next.js; build / output / install commands come
   from [`frontend/vercel.json`](frontend/vercel.json).
4. Set the following environment variables in Vercel (Settings → Environment
   Variables):

   | Variable | Value | Purpose |
   |---|---|---|
   | `NEXT_PUBLIC_API_BASE` | `https://your-backend.example.com` | Backend origin |
   | `NEXT_PUBLIC_MOCK_API` | `false` (or `true` for backendless demo) | Toggle live API vs. seeded mocks |

5. Deploy. The first build is ~90 s; subsequent pushes to `main` redeploy
   automatically.

The frontend works standalone with `NEXT_PUBLIC_MOCK_API=true` if you want a
public demo without a hosted backend — `lib/mock.ts` ships a seeded Churchill
Downs card and a hand-tuned pareto frontier so the UI exercises every component.

### Backend → not Vercel

The FastAPI backend loads ML artifacts at startup, holds a process-local
inference cache, and uses SQLAlchemy async sessions. None of that maps well to
serverless function platforms. Recommended hosts:

- **[Fly.io](https://fly.io)** — `fly launch` on the repo root, point at
  `app.main:app`. Persistent volume for the SQLite DB; the included
  `pyproject.toml` is enough for `pip install -e .`.
- **[Railway](https://railway.app)** — connect repo, set start command
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- **[Render](https://render.com)** — same pattern, Render auto-detects Python
  from `pyproject.toml`.

Whichever you pick, the deploy script must run `scripts/quick_bootstrap.py`
(or a real `scripts/bootstrap_models.py` against your parquet) before the
first `uvicorn` start so `models/baseline_full/` exists. Expose the deploy URL
to Vercel via `NEXT_PUBLIC_API_BASE`.

### CORS

The backend is pre-wired with `CORSMiddleware`; the allow-list is driven by
the `HRBS_CORS_ORIGINS` env var (comma-separated). Default is `*` for dev
convenience — pin to your Vercel origin(s) in production:

```bash
HRBS_CORS_ORIGINS=https://your-app.vercel.app,https://your-app-pr-42.vercel.app
```

Setting any non-`*` value automatically enables `allow_credentials=True` so
cookie-based auth (if added later) works cross-origin.

## Configuration

All settings have working defaults; override only when deviating from dev.

| Variable | Default | Purpose |
|---|---|---|
| `HRBS_DATABASE_URL` | `sqlite+aiosqlite:///./horseracing.db` | Async ORM URL |
| `HRBS_MODELS_DIR` | `models/baseline_full` | Where the FastAPI lifespan loads trained artifacts |
| `HRBS_LOG_LEVEL` | `INFO` | structlog level |
| `HRBS_CORS_ORIGINS` | `*` | Comma-separated browser origin allow-list |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Frontend → backend base URL |
| `NEXT_PUBLIC_MOCK_API` | `false` | Set `true` for frontend-only demos |

See [`.env.example`](.env.example) (backend) and
[`frontend/.env.example`](frontend/.env.example) (frontend) for the full list
with comments.

## Project layout

```
app/                  FastAPI + service layer (ML pipeline lives in app/services/)
scripts/              Standalone CLI utilities (Phase 0 DB build, bootstrap, validators)
frontend/             Next.js 14 App Router UI
tests/                pytest suite mirroring app/ structure
data/                 Gitignored: staging/cleaned/exports/db artifacts
models/               Gitignored: trained model artifacts (baseline_full, rolling)
EXAMPLE_RACE_CARDS/   Sample Brisnet UP PDFs for end-to-end testing
```

Full tree in [`CLAUDE.md §4`](CLAUDE.md).

## Key non-negotiable constraints

These are documented as ADRs and enforced by tests. Do not relax without an
ADR amendment.

| Rule | Source |
|---|---|
| Plackett-Luce (or Stern / Copula), never Harville, for exotic ordering | [ADR-001](DECISIONS.md) |
| 1/4 Kelly with a hard 3% per-bet cap — never full Kelly | [ADR-002](DECISIONS.md) |
| Time-based train/validation split — never random | [ADR-003](DECISIONS.md) |
| Field-relative features only — absolute speed figures are diagnostic-only | [ADR-004](DECISIONS.md) |
| CVaR-constrained portfolio — not mean-variance, not per-bet Kelly | [ADR-005](DECISIONS.md) |
| Rockafellar-Uryasev LP for the CVaR optimiser | [ADR-041](DECISIONS.md) |

## Troubleshooting

**`503 Service Unavailable` from `/api/v1/cards/{id}` or `/api/v1/portfolio/{id}`**
→ `models/baseline_full/` is missing. Run `python scripts/quick_bootstrap.py`
or do a full bootstrap per the Real-data path.

**PDF parser returns warnings on upload**
→ Only digital PDFs (Brisnet UP, Equibase, DRF) are supported, not raster
scans. Add OCR if you need scan support — out of scope for this build.

**`ModuleNotFoundError: greenlet`** when running the API
→ Re-run `pip install -e .` — `greenlet` is now an explicit dep
(ADR-045). On py3.13/3.14 it sometimes drops out of `sqlalchemy[asyncio]`'s
transitive resolution.

**Tests fail with `ModuleNotFoundError: reportlab`**
→ `pip install reportlab` — it's a dev-only dep used by the parser-test
fixtures.

## License

Proprietary — see `pyproject.toml`. University finance paper-trading
competition only.
