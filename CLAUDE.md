# CLAUDE.md — Horse Racing Betting System
# Read this file at the start of every session before writing any code.

---

## 1. PROJECT IDENTITY

**What this is:** A production-quality, end-to-end pari-mutuel wagering analytics system
built for a university finance research paper-trading competition. It ingests raw horse
racing past performance PDFs (Brisnet UP, Equibase, DRF), extracts structured data,
runs a 7-layer ensemble ML pipeline, and outputs a CVaR-optimized portfolio of +EV bets.

**Source of truth documents (read these when in doubt):**
- `Horse_Racing_System_Master_Reference.md` — architecture, math, model design, feature
  engineering specs, implementation order. **Supersedes all other sources.**
- `Horse_Racing_Betting_System_Research.pdf` — academic backing. Reference when
  implementing ordering models, calibration, or portfolio optimization math.

---

## 2. ABSOLUTE CONSTRAINTS — NEVER VIOLATE THESE

These are non-negotiable mathematical and design decisions. Do not rationalize exceptions.

| Constraint | Rule |
|---|---|
| Exotic ordering | **Never use Harville.** Default: Plackett-Luce. Preferred: Stern (Gamma) or Copula. |
| Feature engineering | All features must be **field-relative**, not absolute. A Beyer of 95 means nothing without the field mean. |
| Bet sizing | **1/4 Kelly only.** Never full Kelly. Never per-bet Kelly in isolation. |
| Portfolio optimization | **CVaR-constrained**, not mean-variance alone. Treat the full card as a portfolio. |
| Calibration | Raw model scores are NOT probabilities. Always apply Platt scaling or isotonic regression on held-out data before any EV calculation. |
| Validation split | Always **time-based** (train on earlier dates, validate on later). Never random split — that leaks future information. |
| Sub-model inputs | **Orthogonalize** before the meta-learner. Speed figures already incorporate pace; residualize to prevent double-counting. |

---

## 3. TECHNOLOGY STACK

**Backend / ML Pipeline**
- Python 3.11+
- FastAPI (async API layer; all PDF ingestion runs via `run_in_executor`)
- Pydantic v2 (strict validation; all schemas in `app/schemas/`)
- pandas / NumPy (vectorized feature engineering)
- LightGBM (Speed/Form Model, Pace Scenario Model)
- PyTorch (Transformer sequence model — per-horse career history)
- SciPy (Plackett-Luce MLE fitting, CVaR optimization)
- scikit-learn (Platt scaling, isotonic regression calibration)
- structlog (structured logging throughout)

**PDF Ingestion**
- pdfplumber (primary; layout mode for column preservation)
- pypdf (fallback for pdfplumber failures)
- Three-pass strategy: layout → text → pypdf character-level

**Frontend**
- Next.js 14+ (App Router)
- React + TypeScript
- Tailwind CSS
- Lucide React (icons only — no heavy component libraries)
- Recharts (probability visualization, EV charts)

**Storage**
- SQLite (local dev)
- PostgreSQL (production)
- SQLAlchemy 2.0 (async ORM)

**Config / Infra**
- `python-dotenv` + Pydantic `BaseSettings` for all config
- `alembic` for migrations
- `pytest` + `pytest-asyncio` for testing

---

## 4. REPOSITORY STRUCTURE

```
/
├── CLAUDE.md                          ← you are here
├── PROGRESS.md                        ← session log; update at end of every session
├── DECISIONS.md                       ← architectural decisions with rationale
├── Horse_Racing_System_Master_Reference.md
├── Horse_Racing_Betting_System_Research.pdf
│
├── backend/
│   ├── app/
│   │   ├── main.py                    ← FastAPI app factory
│   │   ├── core/
│   │   │   ├── config.py              ← Pydantic BaseSettings
│   │   │   └── logging.py             ← structlog setup
│   │   ├── schemas/
│   │   │   ├── race.py                ← PastPerformanceLine, HorseEntry, RaceCard, etc.
│   │   │   └── bets.py                ← BetRecommendation, Portfolio, EVResult
│   │   ├── services/
│   │   │   ├── pdf_parser/
│   │   │   │   ├── extractor.py       ← orchestrator: bytes → IngestionResult
│   │   │   │   ├── cleaner.py         ← text normalization (pure functions)
│   │   │   │   ├── brisnet_parser.py  ← Brisnet UP format parser
│   │   │   │   ├── equibase_parser.py ← (Phase 1 stub → Phase 1b implementation)
│   │   │   │   └── drf_parser.py      ← (Phase 1 stub → future)
│   │   │   ├── feature_engineering/
│   │   │   │   ├── engine.py          ← FeatureEngine: RaceCard → feature DataFrame
│   │   │   │   ├── speed_features.py  ← EWM speed figs, field-relative ranks
│   │   │   │   ├── pace_features.py   ← pace shape construction, fraction ratios
│   │   │   │   ├── class_features.py  ← class trajectory, claiming price delta
│   │   │   │   ├── connections.py     ← jockey×trainer interaction features
│   │   │   │   └── layoff.py          ← parametric fitness decay curve
│   │   │   ├── models/
│   │   │   │   ├── speed_form.py      ← LightGBM Speed/Form Model (Layer 1a)
│   │   │   │   ├── pace_scenario.py   ← LightGBM Pace Model (Layer 1b)
│   │   │   │   ├── sequence.py        ← Transformer encoder (Layer 1c)
│   │   │   │   ├── connections.py     ← Bayesian hierarchical model (Layer 1d)
│   │   │   │   ├── market.py          ← Market/smart money model (Layer 1e)
│   │   │   │   └── meta_learner.py    ← Stacking meta-learner (Layer 2)
│   │   │   ├── calibration/
│   │   │   │   ├── calibrator.py      ← Platt / isotonic selector + softmax
│   │   │   │   └── drift.py           ← CUSUM change-point detection
│   │   │   ├── ordering/
│   │   │   │   ├── plackett_luce.py   ← PL MLE fitting + exotic probability enumeration
│   │   │   │   ├── stern.py           ← Gamma distribution ordering model
│   │   │   │   └── copula.py          ← Copula-based pace-correlated ordering (Layer 4)
│   │   │   ├── ev_engine/
│   │   │   │   ├── calculator.py      ← edge, EV, market impact per bet type
│   │   │   │   └── market_impact.py   ← pari-mutuel pool impact model
│   │   │   └── portfolio/
│   │   │       ├── optimizer.py       ← CVaR optimizer, correlated Kelly
│   │   │       └── sizing.py          ← 1/4 Kelly formula, bet cap enforcement
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── ingest.py          ← POST /api/v1/ingest/upload
│   │   │       ├── analyze.py         ← POST /api/v1/analyze/{card_id}
│   │   │       └── portfolio.py       ← GET /api/v1/portfolio/{card_id}
│   │   └── db/
│   │       ├── models.py              ← SQLAlchemy ORM models
│   │       └── session.py             ← async engine + session factory
│   ├── tests/
│   │   ├── test_parser/
│   │   ├── test_features/
│   │   ├── test_ordering/
│   │   └── test_portfolio/
│   ├── scripts/
│   │   ├── bootstrap_models.py        ← train baseline models from Kaggle CSV
│   │   └── validate_calibration.py    ← reliability diagram + ECE report
│   ├── pyproject.toml
│   └── .env.example
│
└── frontend/
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx                   ← upload landing
    │   ├── card/[id]/
    │   │   ├── page.tsx               ← race card viewer
    │   │   └── portfolio/page.tsx     ← bet execution ticket
    │   └── api/                       ← Next.js route handlers (proxy to FastAPI)
    ├── components/
    │   ├── RaceCard/
    │   ├── HorseTable/
    │   ├── EVGauge/
    │   ├── ProbabilityBar/
    │   └── BetTicket/
    ├── lib/
    │   └── api.ts                     ← typed fetch wrappers
    └── package.json
```

---

## 5. IMPLEMENTATION PHASES

Track current phase in `PROGRESS.md`. Build strictly in this order — each phase
is independently useful and testable before the next begins.

| Phase | Focus | Key Deliverables | Status |
|---|---|---|---|
| **1** | PDF Ingestion Pipeline | `extractor.py`, `cleaner.py`, `brisnet_parser.py`, all Pydantic schemas | 🔄 In Progress |
| **2** | Feature Engineering Engine | `engine.py` + all feature modules; field-relative rankings, EWM, layoff curve | ⬜ |
| **3** | Model Bootstrap + Training | LightGBM Speed/Form model from Kaggle CSV; inference scaffolding for all 5 sub-models | ⬜ |
| **4** | Calibration + Ordering | Platt/isotonic calibration; Plackett-Luce MLE; Stern model; exotic prob enumeration | ⬜ |
| **5** | EV Engine + Portfolio Optimizer | Market impact model; edge calculation; CVaR optimizer; 1/4 Kelly sizing | ⬜ |
| **6** | Frontend | Upload flow; race card viewer; probability visualization; bet execution ticket | ⬜ |

---

## 6. KEY MATHEMATICAL SPECS

These are exact implementations required. Do not approximate or substitute.

### Fractional Kelly
```python
def kelly_fraction(edge: float, odds: float, fraction: float = 0.25) -> float:
    # edge = model_prob - market_prob
    # odds = decimal odds (e.g., 4.0 for 3-1)
    # fraction = 0.25 for quarter-Kelly
    full_kelly = (edge * odds - (1 - edge)) / odds
    return max(0.0, full_kelly * fraction)
```

### Plackett-Luce Exacta Probability
```python
# P(horse i 1st, horse j 2nd) = (s_i / S) * (s_j / (S - s_i))
# where s_i = PL strength parameter, S = sum of all strengths
# Strength params fit via MLE on historical finishing orders (scipy.optimize.minimize)
```

### Softmax Temperature for Calibration
```python
# After Platt/isotonic calibration, normalize across field:
# p_i_calibrated = softmax(logits / temperature)
# temperature is a tunable scalar fit on validation data (start at 1.0)
```

### CVaR Constraint (95th percentile shortfall)
```python
# Minimize: -E[portfolio_return]
# Subject to: CVaR_{0.05}(portfolio_loss) <= max_daily_drawdown_pct
# Solved via scipy.optimize with Monte Carlo scenario generation
```

### EWM Speed Figure
```python
# alpha = 0.4 (recent races weighted more heavily)
# ewm_speed = pd.Series(speed_figures).ewm(alpha=0.4).mean().iloc[-1]
# Computed on per-horse PP lines, most-recent-first order
```

### Layoff Decay Curve
```python
# fitness(days) = exp(-lambda * max(0, days - recovery_threshold))
# recovery_threshold ≈ 30 days (short freshening is positive)
# lambda fit empirically per surface/distance category
```

---

## 7. DATA CONVENTIONS

- **Odds storage:** Always decimal (e.g., 3-1 → 4.0). Never fractional strings in schema.
- **Times:** Always seconds as float (e.g., 1:10.40 → 70.40).
- **Distances:** Always furlongs as float (e.g., 1 1/16 miles → 8.5).
- **Money:** Always USD float. No currency symbols in schema fields.
- **Dates:** Always `datetime.date`. No strings in schema date fields.
- **Probabilities:** Always [0.0, 1.0]. Never percentages in model internals.
- **Field-relative features:** Compute rank, percentile, and z-score within the race field, not across the historical database.
- **PP line ordering:** Most recent first in `HorseEntry.pp_lines`. Enforced by model validator.

---

## 8. TESTING REQUIREMENTS

Every service module must have a corresponding test. Minimum coverage per phase:

- **Parser tests:** Feed a known-format text fixture → assert schema fields match expected values exactly.
- **Feature tests:** Feed a synthetic `RaceCard` → assert field-relative features are correctly normalized (mean≈0, within-field rank is monotone with speed figure).
- **Ordering model tests:** Feed known win probs → assert exotic probs sum correctly; assert Plackett-Luce outperforms Harville on longshot calibration.
- **EV engine tests:** Feed synthetic probs + odds → assert edge signs are correct; assert market impact reduces EV monotonically with bet size.
- **Portfolio tests:** Assert 1/4 Kelly never exceeds max_bet_fraction; assert CVaR constraint is binding at the limit.

Run tests before ending any session:
```bash
cd backend && pytest tests/ -v --tb=short
```

---

## 9. SESSION DISCIPLINE

- **Start every session:** `Read CLAUDE.md and PROGRESS.md. Confirm current phase and last completed task, then continue.`
- **End every session:** Update `PROGRESS.md` with what was completed, what is in progress, what is next, and any key decisions made.
- **Never leave a file half-written.** If a module is started, it must be complete and importable before ending the session. Use `# STUB` comments only for functions explicitly deferred to a later phase, and log them in PROGRESS.md.
- **Never break existing tests.** Run the full test suite before committing.