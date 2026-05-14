# PROGRESS.md — Session Log

Update this file at the end of every Claude Code session.
Format: newest session at the top.

---

## Current State

**Phase:** Phases 0-6 **COMPLETE** · Master Reference Layer 7 **COMPLETE** · Cards + Portfolio API endpoints **LIVE** · Pareto-frontier endpoint + frontend rebuild **LIVE** · Public-ready + Vercel-deployable.
**Last completed task:** Three parallel worktree streams (A/B/C) merged into `main` in a single integration pass. **Stream A** built `app/services/inference/pipeline.py` (RaceCard → calibrated win probs → EV candidates → interim portfolio) plus `GET /api/v1/cards/{id}` (hydrates `model_prob`/`market_prob`/`edge` on each `HorseEntry`) and `GET /api/v1/portfolio/{id}` (aggregates per-race Portfolios into one card-level Portfolio). Lifespan-loads `InferenceArtifacts` from `models/baseline_full/`; missing-models case returns 503. Card persistence via existing live ingestion DB. **Stream B** added `RaceOutcome` and `BetSettlement` ORM tables (idempotent on `race_dedup_key`), `app/services/feedback/{outcomes,portfolio_drift}.py` with a two-sided CUSUM (k=0.5, h=4 — same defaults as Phase 4 calibration drift, ADR-036). **Stream C** added `scripts/rolling_retrain.py` (3-year half-open window default, drift-triggered via `--skip-if-no-drift` reading a JSON marker, standalone-script execution per CLAUDE.md §11) + `save_drift_state` / `load_drift_state` helpers on `app/services/calibration/drift.py`. **ADRs 042, 043, 044** appended. **623 tests passing** (was 557 → +66: 13 inference + 21 outcomes + 22 portfolio drift + 10 rolling retrain). The 10 new API endpoint tests also pass (`pytest tests/test_api/test_cards.py tests/test_api/test_portfolio.py`). Frontend build remains clean.
**Next task:** Push to a GitHub remote, deploy the frontend to Vercel (Root Directory = `frontend/`, set `NEXT_PUBLIC_API_BASE` + `NEXT_PUBLIC_MOCK_API` env vars), deploy the backend to Fly.io / Railway / Render. Long-term: train Pace + Sequence sub-models out of their ADR-026 stubs, optionally rewrite the proprietary license depending on the public-release plan.

### Phase 5a smoke result — flagged for Phase 5b investigation

First backtest smoke run (5% sample → 11,574 test rows / 8,717 races, dates up to 2018-02-25) produced 10,663 +EV candidates with **mean edge 0.705** and **mean EV/$ of 34.9** — both implausibly large. Artifact: `models/baseline_full/ev_engine/smoke-test-001/report.json`.

Root cause: the parquet's `odds_final` column contains extreme values (likely 99/999 placeholders for scratched or rare-payout horses) which, when fed into `1/odds`, produce near-zero market probabilities and therefore enormous nominal edges against ANY non-zero model probability. The EV engine is computing what it was asked; the data is the problem. The 3% Kelly cap caught the magnitude (total Kelly stake fraction 299 across 10,663 candidates → ~0.028 mean, right at the cap), so the optimiser will see something tractable, but it'll be allocating to noise.

Phase 5b first fix: cap `decimal_odds` at the validation-script level (suggested cap: ~100) BEFORE feeding to `compute_ev_candidates`. This is a data-cleaning concern, not an EV-engine concern.

### Phase 5a module inventory

| Module                                              | Lines | Tests | Notes                                                            |
|-----------------------------------------------------|------:|------:|------------------------------------------------------------------|
| `app/schemas/bets.py`                               |    ~80 |    11 | BetCandidate / BetRecommendation / Portfolio. ADR-039 scope.    |
| `app/services/ordering/plackett_luce.py` (new helpers) |   +60 |    +9 | `place_prob` / `show_prob` closed-form + `_DENOM_TOL` constant.  |
| `app/services/portfolio/sizing.py`                  |   ~80 |    14 | `kelly_fraction` (1/4) + `apply_bet_cap` (3%). Per ADR-002.     |
| `app/services/ev_engine/market_impact.py`           |  ~120 |    12 | Pari-mutuel `post_bet_decimal_odds` + `inferred_winning_bets`.  |
| `app/services/ev_engine/calculator.py`              |  ~270 |    17 | `compute_ev_candidates` orchestrator + `_build_candidate`.      |
| `scripts/validate_phase5a_ev_engine.py`             |  ~280 |     — | Backtest-mode end-to-end runner. Smoke ran in ~5 min @ 5%.      |

### Calibration finding — full-parquet (post-ADR-037 + ADR-038)

After hardening the auto-selector with ADR-037 (held-out inner-val
ECE) and ADR-038 (skip-when-calibrated guard + time-ordered inner-val
+ strictly-proper Brier co-criterion), the full-parquet validation
(`models/baseline_full/calibration_adr038_brier/`) produced:

| Model        | Pre ECE   | Post ECE  | Pre Brier | Post Brier | Pre LogLoss | Post LogLoss | Chosen   |
|--------------|----------:|----------:|----------:|-----------:|------------:|-------------:|----------|
| Speed/Form   | 0.00310   | 0.00310   | 0.07317   | 0.07317    | 0.26227     | 0.26227      | identity |
| Meta-learner | 0.00343   | 0.00478   | 0.06789   | 0.06788    | 0.23704     | 0.23693      | isotonic |

**Speed/Form** correctly skips (identity) because isotonic's inner-val
Brier improvement (2.8e-6) is well below the 1e-4 threshold — iso's
ECE "win" was bin redistribution, not accuracy. Post-cal metrics are
identical to pre-cal.

**Meta-learner** correctly applies isotonic. Inner-val Brier
improvement (1.6e-4) is just above the threshold; this transfers to
the test slice as a tiny but real improvement on BOTH strictly proper
scores (Brier 0.06789 → 0.06788, log-loss 0.23704 → 0.23693). Only
ECE moved the "wrong" way (0.0034 → 0.0048) — but ECE is the
bin-noisy summary the Brier co-criterion was put in place to override.
The guard is working as designed: applies calibration when proper
scores agree there is a real win, skips when they don't.

Earlier full-parquet runs (pre-fix, fit-slice criterion; post-ADR-037
but pre-Brier) both incorrectly picked isotonic on Speed/Form and
materially degraded test ECE. See `calibration/` (original) and
`calibration_adr038/` (post-ADR-037 ECE-only skip) artifact dirs for
the historical comparison.

The 5% smoke (11.5K calib / test) had reported meta-learner pre-ECE
= 0.121 / post = 0.011. That figure was a **small-sample artifact**:
with 11.5K obs in the test split, ECE bin counts are noisy enough to
inflate the estimate, AND the random 5% slice happened to land in a
slightly miscalibrated region. The full-parquet baseline is the truth.

Artifacts: `models/baseline_full/calibration_adr038_brier/{speed_form,meta_learner}/{report.json,reliability.png,platt.joblib,isotonic.joblib,metadata.json}` + `summary.json`.

### Phase 4 module inventory (final)

| Module                                              | Lines | Tests | Notes                                                            |
|-----------------------------------------------------|------:|------:|------------------------------------------------------------------|
| `app/services/calibration/calibrator.py`            |   ~430 |    21 | Platt / isotonic / auto-selector, per-race temperature softmax.  |
| `app/services/calibration/drift.py`                 |   ~290 |    26 | Standardised Bernoulli-z CUSUM (k=0.5, h=4).                     |
| `app/services/ordering/plackett_luce.py`            |   ~305 |    21 | Analytical exacta/trifecta/superfecta + vectorised MLE.          |
| `app/services/ordering/stern.py`                    |   ~340 |    29 | Gamma marginals; PL fast path at shape=1; `infer_strengths`.     |
| `app/services/ordering/copula.py`                   |   ~330 |    20 | Block-equicorrelation; PL/Stern fast paths; Cholesky+jitter MC.  |
| `scripts/validate_calibration.py`                   |   ~350 |     3 | Live pipeline runner + importable `evaluate_calibration`.        |

### Known Export Caveats (for Phase 3 model training)

Column null rates in `training_20260512.parquet`:

| Column                                                                                          | % rows null | Cause                                                       |
|-------------------------------------------------------------------------------------------------|------------:|-------------------------------------------------------------|
| `claiming_price`, `speed_figure_source`, `beaten_lengths_q1/q2`, `fraction_q1_sec/q2_sec`, `sire`, `dam_sire`, `foaling_year` | **100%**    | Not populated by any current source dataset.                |
| `field_size`                                                                                    | 74%         | Only UK populates it. JP/HK NULL — derive via group counts. |
| `speed_figure`                                                                                  | 72%         | JP has zero speed figures (1.6M rows). UK has 96%, HK 73%.  |
| `post_position`                                                                                 | 26%         | All 598,708 UK rows. sheikhbarabas dataset omits it.        |
| `purse_usd`                                                                                     | 1.5%        | Sparse on some HK/JP rows.                                  |
| `fraction_finish_sec`                                                                           | 1.3%        | Sparse.                                                     |

UK `race_number` is a **post-time-minutes-past-midnight proxy** from `time_string_to_minutes` — within-card ordering works (sorts by post time) but won't be a contiguous 1..N sequence. Phase 3 should recompute a 1..N race number per `(track_code, race_date)` group if needed.

JP carries 1.6M rows with no speed figure. Either derive a synthetic speed figure from `(distance_furlongs, fraction_finish_sec, condition)` for JP, or restrict speed-figure-dependent models to UK+HK and rely on other features for JP.

---

## Session Log

### Session: 2026-05-13 (g) — Repo hygiene + Vercel deployment readiness

**Completed:**

Pre-public polish. Goal: a clean clone can be pushed to GitHub and deployed to Vercel with one repo-import + a Root-Directory click.

*Hygiene*
- Deleted stray runtime SQLite artifacts (`_test_cards_*.db`, `horseracing.db`) from the repo root. They were always gitignored but cluttered the working tree.
- Untracked the regenerable model artifacts (`models/baseline_full/{speed_form,connections,market,meta_learner}/...`, `summary.json`, calibrator metadata). `quick_bootstrap.py` outputs are non-deterministic across LightGBM/numpy version drift, so committing them was fragile. Smoke reports (`ev_engine/*/report.json`) and historical calibration outputs (`calibration*/reliability.png`, `report.json`, `summary.json`) remain tracked as validation evidence.
- Extended `.gitignore` with `models/baseline_full/<sub_model>/` paths, `models/rolling/`, `.vercel/`, `.next/`, `node_modules/`, `*.tsbuildinfo`, `next-env.d.ts`, `.env.*.local`, `secrets.json`, `*.pem`, and `.claude/worktrees/`.
- Added `.editorconfig` (4-space Python, 2-space TS/JSON/MD, LF endings).
- Added `frontend/.nvmrc` pinning Node 20.

*CORS & env-driven config*
- `app/main.py` gained `_cors_origins()` reading `HRBS_CORS_ORIGINS` (comma-separated; default `*`). When a specific origin list is set, `allow_credentials` flips to `True` so cookie-based auth would work cross-origin. Restricted methods to `GET, POST, OPTIONS` (the surface the frontend actually uses).
- `.env.example` documents the new `HRBS_CORS_ORIGINS` and `HRBS_MODELS_DIR` variables.

*Vercel*
- `frontend/vercel.json` declares Next.js framework + build/install commands and a small set of security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, restrictive `Permissions-Policy`). `regions: ["iad1"]` keeps cold starts on the US east coast where most racing tracks live.
- No top-level `vercel.json` — the canonical monorepo deploy path is "set Root Directory to `frontend/` in the Vercel UI". Tried wrapping with a root `cd frontend && ...` build command but that fights Next.js auto-detect.

*CI*
- `.github/workflows/ci.yml` runs the backend matrix (py 3.11 / 3.12 / 3.13, `pytest tests/ -q`) and the frontend (Node 20, `typecheck` + `lint` + `build` with `NEXT_PUBLIC_MOCK_API=true` so the mock path is exercised in CI without a backend). `concurrency` cancels in-progress runs on rapid push.

*Documentation*
- `README.md` gained a comprehensive **Deployment** section: Vercel walkthrough (Root Directory = `frontend/`, env-var checklist), backend-host recommendations (Fly.io / Railway / Render with rationale for why Vercel serverless doesn't fit a model-loading stateful FastAPI app), CORS guidance. Tightened the Configuration table to include `HRBS_CORS_ORIGINS`.
- `PROGRESS.md` (this entry) + Current State header now reflect public-ready + Vercel-deployable state.

**Tests Status:**
- All **652 tests passing** (`pytest tests/ -q`) — no changes to test count.
- Frontend `npm run typecheck` ✅ — zero errors.
- Frontend `npm run lint` ✅ — zero warnings.
- Frontend `npm run build` ✅ — 4 routes, same JS sizes as before.
- `python scripts/quick_bootstrap.py` ✅ — `InferenceArtifacts.load(...)` succeeded after the run.

**Files touched this session:**
- `.gitignore` (+15 lines) — Vercel, Node, secrets, model artifacts.
- `.editorconfig` (new, 20 lines).
- `.github/workflows/ci.yml` (new, 65 lines).
- `frontend/vercel.json` (new, 22 lines).
- `frontend/.nvmrc` (new, 1 line).
- `app/main.py` (+15 lines) — `_cors_origins()` + tightened middleware.
- `.env.example` (+11 lines) — CORS + models-dir docs.
- `README.md` (+~70 lines) — Deployment section.

**Untracked from git (no content change, only the tracking status):**
- `models/baseline_full/{connections,market,meta_learner,speed_form}/...`
- `models/baseline_full/summary.json`
- `models/baseline_full/calibration_adr038_brier/meta_learner/metadata.json`

**Open follow-ups (for the first real deploy):**

1. **Pick a license.** `pyproject.toml` says "Proprietary" but the user has indicated this is going public. A `LICENSE` file (MIT, Apache-2.0, BSL, or explicit proprietary text) should land before the repo goes public.
2. **Real-data bootstrap.** Production deploy needs actual Kaggle data → `scripts/db/*` → `scripts/bootstrap_models.py`. The `quick_bootstrap.py` placeholder serves the demo but its predictions are not meaningful.
3. **Backend host selection.** README documents Fly.io / Railway / Render as candidates. Pick one, wire the deploy command, set `HRBS_CORS_ORIGINS` to the Vercel URL.
4. **Train Pace + Sequence sub-models.** Still ADR-026 stubs; the loader is tolerant but ensemble quality is 3-of-5.

---

### Session: 2026-05-13 (f) — Pareto frontier endpoint + frontend rebuild (Streams X/Y/Z)

**Completed:**

Three parallel worktree agents dispatched, each completing ~1/3 of their scope before hitting a rate-limit. Their partial commits were merged into `main`, the remaining ~2/3 of each stream was finished directly in the main session, and the result is a pareto-driven UX from upload to bet ticket.

*Stream X — Pareto endpoint + LP swap + greenlet fix*
- Swapped the interim portfolio constructor in `app/services/inference/pipeline.py::analyze_card` for Phase 5b's Rockafellar-Uryasev LP (`app/services/portfolio/optimizer.optimize_portfolio`). The interim (`build_portfolio_from_candidates`) is preserved behind `use_interim=True` for backward-compat.
- New `analyze_card_pareto(card, artifacts, *, risk_levels, ...)` — runs candidate generation ONCE, re-solves only the per-race LP at each risk level. For a 10-race card × 6 risk levels: 1 ML pass + 60 cheap LP solves.
- New schemas `ParetoPoint` + `ParetoFrontier` in `app/schemas/bets.py`.
- New endpoint `GET /api/v1/portfolio/{card_id}/pareto?risk_levels=…` — comma-separated risk levels (1–12, each in (0, 1], strictly increasing after parse; default `0.05,0.10,0.15,0.20,0.25,0.30`). Returns a card-aggregated Portfolio per risk level. Same auth/error semantics as `/portfolio/{id}`.
- Pinned `greenlet>=3.0` explicitly in `pyproject.toml` — sqlalchemy[asyncio]'s wheel resolution drops it on py3.13/3.14 sometimes.
- **+7 endpoint tests** in `tests/test_api/test_pareto.py` (503/404/default-returns-6/monotone-in-return/custom-risk-levels/malformed-400/unit-pipeline). **652 total tests passing.**

*Stream Y — Quick bootstrap + README*
- `scripts/quick_bootstrap.py` (~540 lines): generates 4000 rows of synthetic training data with a planted weak signal (~0.15 correlation between `field_relative_speed`/`jockey_trainer_interaction` and `won`), time-splits 60/20/20, trains Speed/Form + Connections + Market + Meta-learner + Calibrator (skips Pace/Sequence per ADR-026 stub policy), writes the full bundle to `models/baseline_full/` in ~1.3 s. Verification step at the end loads the bundle via `InferenceArtifacts.load()` and asserts it succeeds.
- Marker file `models/baseline_full/QUICK_BOOTSTRAP.json` carries `is_synthetic: true` so future tooling can detect placeholder bundles.
- `README.md` (new): real human-facing quickstart (clean clone → venv → pip → bootstrap → uvicorn → npm dev → upload an example PDF). Plus Real-data path, Architecture, Running tests, Configuration, Project layout, Non-negotiable constraints (linked to ADRs), Troubleshooting.
- `frontend/.env.example` flipped `NEXT_PUBLIC_MOCK_API` default from `true` to `false` — clean dev now hits the real API.

*Stream Z — Frontend rebuild around pareto*
- `frontend/lib/types.ts`: added `ParetoPoint` + `ParetoFrontier` mirroring the backend schemas.
- `frontend/lib/api.ts`: added `getParetoFrontier(cardId, options)` typed wrapper with full query-param coverage and a MOCK fallback.
- `frontend/lib/mock.ts`: added `mockParetoFrontier(...)` producing 6 deterministic points monotone in (risk, return) with realistic stake fractions (all ≤ 0.03 per ADR-002).
- Two new components:
  - `components/ParetoCurve/ParetoCurve.tsx` — Recharts `ComposedChart` (Scatter + Line overlay), custom dot renderer that grows + colours the selected point, custom tooltip, clickable points.
  - `components/ParetoCurve/RiskSlider.tsx` — six labelled stepped buttons (`conservative / moderate / balanced / aggressive / max risk`), synchronised with the curve via shared parent state.
- `frontend/app/card/[id]/page.tsx` rebuilt as a unified result view: sticky header with bankroll input (debounced 400 ms, triggers pareto re-fetch), pareto curve + risk slider side-by-side, portfolio summary strip (6 stats), bet ticket that updates in-place when the user picks a point, expandable race-detail accordions (showing horse table per race with recommendation count badges).
- `frontend/app/card/[id]/portfolio/page.tsx` reduced to a server-side `redirect` back to `/card/{id}#races` for any legacy bookmarks.
- `frontend/README.md` updated with the new URL structure + backend contract surface.
- `npm run build` ✅ — 4 routes, 0 TS errors, `/card/[id]` JS bundle 112 kB First Load (was 101 kB pre-pareto — the new components add ~11 kB).

**Key decisions made (new ADR):**

- **ADR-045:** Real R-U LP in `analyze_card` + new pareto endpoint + risk-grid default (6 levels from 0.05 to 0.30). Per-call cost dominated by `n_risk_levels × n_races` LP solves (~50 ms each, well under 5 s on a typical 8-race card × 6 levels). Returning the full Portfolio per point (not just summary stats) so the frontend slider transitions are zero-fetch.

**Tests Status:**
- Pre-session: 626 passing (the post-Streams-A/B/C state — 623 backend + 10 API tests counted separately).
- Stream X: +7 pareto endpoint tests + 13 inference-pipeline tests already accounted for.
- **Final: 652 passing** at `pytest tests/ -q` in ~11 s. No regressions.

**Surprising / non-obvious findings:**

- **All three parallel agents branched from `d425b30` (pre-Phase-5b)** even though main was `de15078` at dispatch time. Their DECISIONS.md ADRs all collided at "next-available number = 041". Streams Y and Z got `add/add` conflicts on frontend/lib/{types,api,mock}.ts during merge (their version vs. main's existing Phase 6 baseline) — resolved by keeping main's version and inserting Stream Z's pareto additions inline. Same lesson as Streams A/B/C (one round earlier): worktree base race conditions are real; ADR slots should be pre-assigned in the agent briefs and frontmatter conflicts must be resolved explicitly.
- **The pareto curve must be monotone non-decreasing in expected_return** as risk loosens, because higher `max_drawdown_pct` is a strictly looser constraint on the same LP. The test `test_pareto_monotone_in_expected_return` exercises this invariant — if it ever fails, the LP setup has a sign error or the candidate generation isn't deterministic.
- **`analyze_card(optimize=False)` was already structured perfectly** for the pareto re-use: it returns `(race_win_probs, all_candidates, [])` so the pareto function just groups candidates by race, then loops the optimizer per risk level. No refactor of the heavy inference path was needed.

**Files created/modified this session (after the 3 partial agent merges):**
- `app/schemas/bets.py` (+27 lines) — ParetoPoint, ParetoFrontier schemas.
- `app/services/inference/pipeline.py` (+140 lines) — `analyze_card_pareto`, `_aggregate_per_race_portfolios`, `DEFAULT_RISK_LEVELS` constant, updated `__all__`.
- `app/api/v1/portfolio.py` (+140 lines) — pareto route + `_parse_risk_levels` + `_run_pareto_sync` executor wrapper.
- `tests/test_api/test_pareto.py` (new, 7 tests).
- `DECISIONS.md` (+~75 lines) — ADR-045.
- `README.md` (new) — full operator quickstart.
- `frontend/.env.example` — MOCK default flip.
- `frontend/lib/{types,api,mock}.ts` — pareto types + client + mock data (from Stream Z partial).
- `frontend/components/ParetoCurve/{ParetoCurve,RiskSlider}.tsx` (new, ~250 lines combined).
- `frontend/app/card/[id]/page.tsx` (rebuilt, 290 lines).
- `frontend/app/card/[id]/portfolio/page.tsx` (reduced to a 16-line redirect).
- `frontend/README.md` (updated URL structure + backend contracts).

---



**Completed:**

Three independent worktree agents dispatched in parallel and merged sequentially into `main`. None of the three branches saw each other or saw the Phase 5b/6 merges — each branched from `d425b30`, the post-Phase-5a merge commit — so each ADR landed at "ADR-041" in its own branch and had to be re-numbered during integration. The integration kept all three.

*Stream A — `worktree-agent-a8e9a98f0b57f6f38` → ADR-042*
- `app/services/inference/__init__.py` (29) + `app/services/inference/pipeline.py` (856) — reusable inference module:
  - `InferenceArtifacts.load(models_dir)` aggregates the five sub-models + meta-learner + meta-calibrator. Tolerant of missing optional sub-models (Speed/Form / Connections / Market → 0.5 per ADR-026); strict on meta-learner + meta-calibrator (load-bearing).
  - `build_inference_features(card)` synthesises the trained-model column schema (`ewm_speed_prior`, `last_speed_prior`, `n_prior_starts`, field-relative normalisation, categoricals) directly from `HorseEntry.pp_lines`. No `groupby.shift(1)` needed (PP lines already exclude today's race).
  - `infer_calibrated_win_probs(...)` reproduces `_score_test_slice` from the validation script.
  - `analyze_card(...)` end-to-end orchestrator. Live mode emits WIN candidates only (per ADR-040 — exotic per-permutation odds unavailable pre-race). Decimal odds from `morning_line_odds`, capped at `max_decimal_odds=100.0`.
  - `build_portfolio_from_candidates(...)` — **interim CVaR-scaled portfolio constructor**. Scales 1/4-Kelly fractions by `k ∈ (0, 1]` until MC-CVaR ≤ `max_drawdown_pct × bankroll`. Phase 5b's R-U LP (`app/services/portfolio/optimizer.optimize_portfolio`) is now in main but Stream A was authored against the pre-5b base; the swap is a one-line follow-up.
- `app/api/v1/cards.py` (169) + `app/api/v1/portfolio.py` (189) — two GET endpoints, registered via `app/main.py` lifespan (loads `InferenceArtifacts` from `HRBS_MODELS_DIR` env override or `models/baseline_full/`; missing dir → endpoints respond 503). Per-card calibrated vectors memoised in `app.state.inference_cache` (process-local). Multi-race portfolio aggregation in `/portfolio` concatenates `recommendations`, sums `expected_return` and `total_stake_fraction` (capped at 1.0), takes MAX of per-race VaR/CVaR (conservative).
- `app/db/persistence.py` (+175) — `load_card(session, card_id)` rebuilds Pydantic `RaceCard` from ORM tree via `selectinload` eager joins.
- `app/api/v1/ingest.py` (+3) — sets `result.card_id` from DB PK so the frontend can roundtrip.
- `app/schemas/race.py` (+11) — `HorseEntry.{model_prob,market_prob,edge}` optional + `IngestionResult.card_id` optional.
- Tests: `tests/test_inference/test_pipeline.py` (13 unit tests) + `tests/test_api/test_cards.py` (5) + `tests/test_api/test_portfolio.py` (5).

*Stream B — `worktree-agent-a7a735541d7f27c95` → ADR-043*
- `app/db/models.py` (+82) — two new ORM tables:
  - `RaceOutcome`: idempotency anchor `race_dedup_key` (UniqueConstraint, matches Phase 0 SHA-256 dedup convention). Fields: `race_id`, `race_dedup_key`, `finishing_order` JSON, `winning_horse_program_number`, `place_horses` JSON (padded with -1 sentinel for short fields), `payouts` JSON, `settled_at`, `source`.
  - `BetSettlement`: one row per settled `BetRecommendation`. Captures BOTH recommendation-time and settlement-time decimal odds so future drift attribution can split "model error" vs. "market move". `bet_recommendation_id` is a nullable FK slot — retrofit non-null when a `BetRecommendation` table lands.
- `app/services/feedback/__init__.py` + `outcomes.py` (310) + `portfolio_drift.py` (252):
  - `log_race_outcome(...)` idempotent on `race_dedup_key` (returns existing row on dup, INFO-logs; does NOT overwrite payouts).
  - `settle_bets(...)` handles WIN/EXACTA/TRIFECTA/SUPERFECTA via positional prefix-match on `finishing_order`. Rejects PLACE/SHOW/PICK-n per ADR-039.
  - `get_settled_pnl_series(...)` returns a time-ordered DataFrame with columns `[settled_at, race_id, bet_type, expected_value, pnl, payout, stake, model_prob, decimal_odds_at_recommendation, won]`.
  - `portfolio_pnl_zscore`: Bernoulli closed-form σ = `stake · √(p(1−p)) · odds_at_settlement` when probs+odds supplied, else fallback to `|stake|`. Z-clip at ±5, eps-floor 1e-6.
  - `PortfolioDriftDetector`: two-sided CUSUM, defaults k=0.5 / h=4 (same as ADR-036; ARL₀ ≈ 168). Latches on first trigger. Supports incremental `.update(z)` and batch `.run(z_array)`.
- Tests: `tests/test_feedback/test_outcomes.py` (21) + `tests/test_feedback/test_portfolio_drift.py` (22). 43 new total.

*Stream C — `worktree-agent-a97bbbfa0128c7b25` → ADR-044*
- `scripts/rolling_retrain.py` (534) — standalone CLI mirroring `bootstrap_models.py` but windowed:
  - Slices `[as_of_date − window_years, as_of_date)` from parquet (half-open; rows on `as_of_date` excluded).
  - Reuses `validate_calibration._three_way_split` for 60/20/20 train/calib/test cut.
  - Refits requested sub-models + meta-learner + calibrator, writes `report.json` and all artifacts under `models/rolling/<as_of_date>/`.
  - `--skip-if-no-drift` reads `drift_state.json`; exits code 2 when `triggered=false` (cron no-op), code 0 on successful retrain.
  - `--sub-models` partial flag lets untrained sub-models fall through to ADR-026's constant-0.5 stubs.
- `app/services/calibration/drift.py` (+94) — `save_drift_state(detector, path)` + `load_drift_state(path)`. Idempotent file write, parent-dir creation, neutral-state return for missing/corrupt files (cron-safe — never raises).
- Tests: `tests/test_scripts/test_rolling_retrain.py` (10) — window slicing, time-ordered split, `--skip-if-no-drift` in both directions (subprocess), `report.json` schema, partial-sub-models resilience.

**Key decisions made (new ADRs):**
- **ADR-042:** Server-side inference pipeline as a reusable module + GET (not POST) endpoints + in-memory per-card cache (vs DB persistence — calibration / retraining decouples model lifecycle from card lifecycle) + multi-race portfolio aggregation rule (concat recs, max VaR/CVaR) + interim portfolio constructor placeholder for Phase 5b's R-U LP.
- **ADR-043:** Outcomes schema with `race_dedup_key` idempotency + two-sided CUSUM (k=0.5/h=4 matches ADR-036, one mental model across drift detectors) + PnL convention `pnl = payout − stake` + settlement scope = ADR-039 (WIN/EXACTA/TRIFECTA/SUPERFECTA only).
- **ADR-044:** 3-year rolling window default rationale (two full racing seasons + buffer + drops pre-COVID economics) + drift-triggered default with scheduled-cadence fallback + standalone-script execution rationale (CLAUDE.md §11) + `models/rolling/<as_of_date>/` output layout.

**Surprising / non-obvious findings:**

- **All three agents branched from the same pre-Phase-5b/6 commit (`d425b30`)** because that was main when the worktrees were created. Their DECISIONS.md ADRs all collided at "next-available number" = 041. Integration kept all three and re-tagged Stream A → 042, Stream B → 043, Stream C → 044. Lesson: when dispatching multiple worktree agents that all append to a shared sequenced file, pre-assign the slot numbers in the agent briefs (which we did — but Stream A's prompt expected `ADR-042` and built against a base where the previous ADR was `ADR-040`, so the agent's branch had `ADR-040 → ADR-042` and the merge needed to splice `ADR-041` between them).

- **Stream A built a placeholder portfolio constructor (`build_portfolio_from_candidates`) without seeing the Phase 5b LP**. The interim does scalar scaling on 1/4-Kelly fractions until MC-CVaR clears the threshold. It works (and the API returns sensible Portfolios) but it does NOT solve the Rockafellar-Uryasev LP. The swap-in is a single-line edit in `analyze_card` once we choose to do it; tests will still pass because the interface (`list[BetCandidate] → Portfolio`) is identical.

- **Stream B's CUSUM uses identical thresholds to Stream A-vintage calibration drift (k=0.5, h=4)** so the operator has ONE mental model across every drift detector in the system. The Bernoulli σ derivation is single-event; correlated baskets (e.g. exotic permutations on the same race) will eventually need a copula-derived σ — flagged in ADR-043 but explicitly out of Layer 7 scope.

- **Stream C's `--skip-if-no-drift` exit code 2 is a deliberate cron signal**, not a failure. Cron jobs can branch on it: "if exit==2 then no-op; if exit==0 then retrain proceeded; if exit==1 then bug." The JSON marker schema written by `save_drift_state` is precisely what `load_drift_state` consumes — no version drift between Stream B's CUSUM writes and Stream C's reads (Stream B's `PortfolioDriftDetector` doesn't directly serialise; the existing calibration `ChangePointDetector` does, and the Layer 7 picture is that the calibration drift detector is the live monitor whose marker file Stream C reads).

- **`models/baseline_full/` is gitignored** so the API endpoints respond 503 in a clean clone. Tests bypass this via `app.state.artifacts` injection — endpoint tests use mocked artifacts. The 503 path is the correct degenerate behaviour; the operator must train models (or copy them from an existing artifact dir) before the endpoints serve real data.

**Tests Status:**
- Pre-Streams baseline (non-API): 557 passing.
- Stream A adds 13 non-API tests (inference pipeline) + 10 API tests (cards + portfolio).
- Stream B adds 43 non-API tests (outcomes + drift).
- Stream C adds 10 non-API tests (rolling retrain).
- **Post-merge:** 623 non-API tests passing (`pytest tests/ --ignore=tests/test_api -q`) + 10 new API tests passing (`pytest tests/test_api/test_cards.py tests/test_api/test_portfolio.py`). Total verified: 633.
- The 6 pre-existing `tests/test_api/test_ingest.py` errors are environmental (py3.14 + sqlalchemy greenlet missing) — not regressions, untouched.

**Frontend status:**
- `frontend/npm run build` ✅ still clean (4 routes, zero TS/ESLint errors). Untouched by these streams; ready to point at the now-live API by flipping `NEXT_PUBLIC_MOCK_API=false`.

**Open follow-ups:**

1. **Swap Stream A's `build_portfolio_from_candidates` for Phase 5b's `optimize_portfolio`.** Stream A's interim is a scalar-scaling placeholder; the R-U LP is the canonical choice. One-line edit inside `analyze_card`. Both produce `Portfolio`; no test churn expected.
2. **`BetSettlement.bet_recommendation_id` FK retrofit** when a `BetRecommendation` table is added (Stream B left the slot nullable).
3. **Pace / Sequence rolling-retrain unblock** when the fractional-time export columns land (ADR-026) and the PyTorch + globally-unique `horse_id` is wired. Stream C's `is_trainable_with` gate is already in place — flipping these on is no-touch to the orchestrator.
4. **Empirical window-length sweep** (2y / 3y / 4y) against Stream B's outcomes log once live data accumulates (ADR-044 "When to revisit").
5. **Frontend hookup** — set `NEXT_PUBLIC_MOCK_API=false` and `NEXT_PUBLIC_API_BASE` to the running FastAPI host. Mock-mode remains the fallback for demoing without a backend.
6. **Re-export the training parquet** (Phase 0 concern). The real-data Phase 5b smoke and a real-data Stream C rolling-retrain are gated on `data/exports/training_20260512.parquet` regeneration.

**Commits on main since previous "Current State":**
- `b469b0a` Merge Stream A — inference pipeline + cards/portfolio API endpoints
- `f77f945` Merge Stream B — outcomes logging + portfolio drift CUSUM
- `c81bd26` Merge Stream C — rolling-window retraining script

Each merge required a DECISIONS.md conflict resolution (ADR ordering) — no other file conflicts across the three streams.

---

### Session: 2026-05-13 (c) — Phase 5b portfolio optimiser (parallel-worktree agent)

**Completed:**

Phase 5b built end-to-end on branch `worktree-agent-a876df5324605a63d` in a single session as one of two parallel agents working from the merged-to-main Phase 5a baseline.

*Task 1 — validation-script `--max-decimal-odds` cap*
- `scripts/validate_phase5a_ev_engine.py` gained a `--max-decimal-odds` flag (default 100). Rows with `odds_final > cap` are dropped (set to NaN, race skipped) BEFORE `compute_ev_candidates` sees them. Drop count is logged via the existing `log.info("ev_engine.validate.odds_cap.applied", …)` structlog pattern and included in the report JSON summary as `n_rows_dropped_by_odds_cap` / `max_decimal_odds`. Fixes the ADR-040-flagged 99/999 placeholder-odds issue.
- Same commit added the inert `--optimize` / `--bankroll` / `--cvar-alpha` / `--max-drawdown-pct` / `--n-scenarios` / `--seed` flags so Task 3's wiring needed only the optimizer module landing — the script's CLI surface was complete in one commit. `EVRunSummary` gained nine new fields, all with neutral defaults so the dataclass remains backward-compatible with Phase 5a's report-comparison tooling.

*Task 2 — PL Gumbel sampler + CVaR LP optimiser + tests*
- `app/services/ordering/plackett_luce.py` gained `sample_orderings_vectorised(strengths, n_samples, rng)`. Yellott / Gumbel-Max identity: `argsort(-(log_s + Gumbel(0,1)))` is distributed exactly as a single PL draw. One numpy uniform + log + log + argsort call generates the whole `(n_samples, n_horses)` matrix; ~100× faster than calling `sample_ordering` in a python loop.
- `app/services/portfolio/optimizer.py` is the Rockafellar-Uryasev (2000) CVaR LP — see ADR-041 for the full decision-vector packing and LP matrix construction. Per-race scope: each `optimize_portfolio` call operates on one race's candidates. Per-race seeding (`np.random.default_rng([seed, hash(race_id) & 0xFFFFFFFF])`) makes results invariant to candidate ordering and stable across calls.
- 6 PL sampler tests + 10 optimiser tests added. Critical assertions: per-recommendation `stake_fraction ≤ min(¼-Kelly cap, max_bet_fraction)`; `cvar_95 ≈ max_drawdown_pct × bankroll` when the CVaR constraint is binding (CLAUDE.md §10 requirement). Empty / negative-EV / out-of-range-selection / missing-race / LP-infeasible all exercised.

*Task 3 — `--optimize` wiring smoke run*
- `evaluate(...)` extended to lazy-import the optimiser and loop per race when `optimize=True`. Returns `(candidates, summary, portfolios)`; portfolios is empty when not optimising.
- `scripts/smoke_phase5b_optimizer.py` is a synthetic smoke harness — the production parquet (~2.3M rows) and trained baseline models are gitignored and not present in clean worktrees, so this script generates a 200-race / 8-horse synthetic test slice (Dirichlet win probs, planted 999-odds placeholders) and runs it through `evaluate(...)`.
- Smoke artifact: `models/baseline_full/ev_engine/phase-5b-smoke-001/report.json`. Output: 1600 synthetic rows → 348 dropped by 100-odds cap → 104 +EV candidates over 28 surviving races → 27 active portfolios with mean 2.74 recommendations and `mean_cvar_usd = $198.90` (well below the $2,000 cap on a $10K bankroll because few candidates per race).

*Task 4 — ADR-041 + this progress entry*
- ADR-041 documents (a) the Rockafellar-Uryasev LP form and decision-vector packing, (b) Gumbel-trick vectorised PL scenarios, (c) per-race scope (cross-race deferred, mirroring ADR-039), (d) the `n_scenarios=1000` default rationale, (e) USD reporting convention for VaR/CVaR.

**Key decisions made (new ADRs):**

- **ADR-041:** CVaR LP optimiser uses Rockafellar-Uryasev (2000) formulation via `scipy.optimize.linprog(method="highs")`, with per-race Monte Carlo scenarios drawn via the vectorised Gumbel-trick PL sampler. Scope is per-race (cross-race deferred until the card-level latent-state model exists, mirroring ADR-039). Default `n_scenarios=1000` chosen as the empirical sweet spot for ≲ 50 ms LP solves with ~14% Monte Carlo error at α=0.05.

**Surprising / non-obvious findings to carry forward:**

- **CLAUDE.md §8's Kelly formula is `(edge*odds − (1−edge))/odds`, NOT the classical `(p·b − q)/b`.** These are different formulas. With the project's formula, positive Kelly requires `edge > 1/(odds+1)`, a strictly tighter condition than `edge > 0` (which the EV calculator's `min_edge` filter already enforces). Consequence: many +EV candidates (with `expected_value > 0` and `edge > min_edge=0.05`) end up with `kelly_fraction = 0` and therefore zero stake fraction in the optimiser. The smoke run's first three iterations all produced 0 portfolios for this reason. Mentioned here so future Phase 5b consumers (and Phase 6 frontend developers showing "kelly cap" values to the user) know the discrepancy. ADR-002 explicitly mandates the project's formula; I did not second-guess it.

- **Per-race RNG seeding via `np.random.default_rng([seed, hash(race_id) & 0xFFFFFFFF])` rather than a single global seed.** Without this, the order in which `optimize_portfolio` is called for different races would change the per-race scenario draws (because each call would consume from the same global Generator). With per-race seeding, swap two races in the validation loop and you get the same portfolios for each — important for reproducibility of regression artefacts.

- **The training parquet is gitignored and is NOT in clean worktrees.** I expected to run the full backtest smoke against `data/exports/training_20260512.parquet`; the file doesn't exist in this worktree (or in main, post-clone). Built a synthetic smoke harness instead. The production smoke against the real parquet remains the Phase 5a `smoke-test-001` artifact — running it again is a follow-up task once the parquet is re-generated by re-running the export pipeline.

- **`VaR / CVaR` are reported in USD, not as fractions.** `Portfolio.var_95 = bankroll * v` and `Portfolio.cvar_95 = bankroll * cvar_lhs`. Phase 6 frontend should display them as currency. The schema's field type is `float`; this is a convention, not a constraint.

**Tests Status:**
- Pre-Phase-5b baseline (non-API suite): 541 passing.
- Phase 5b adds 16 tests (6 PL sampler + 10 optimiser): 557 passing.
- Pre-existing `tests/test_api/test_ingest.py` is 6 errors in this worktree's Python 3.14 + sqlalchemy env (greenlet missing). Not regressions; unrelated to Phase 5b.
- Total `pytest tests/` run: 557 passed, 6 errors (environment) — all 557 + 6 expected if the venv had a Python 3.11/3.12 fallback for sqlalchemy.

**Known limitations carried forward into Phase 6 / Phase 7:**

1. **Per-race scope only.** Cross-race correlation is deferred (ADR-041, mirrors ADR-039). When the card-level latent-state model lands, `optimize_portfolio` should be called once per CARD with all candidates rather than once per race.
2. **`max_bet_fraction` is the CLAUDE.md §2 / ADR-002 3% cap.** No way to relax it per-race for high-confidence stakes; that would require an ADR amendment.
3. **The synthetic smoke harness `scripts/smoke_phase5b_optimizer.py` is for CI / clean-clone parity, not for real validation.** The real smoke requires the training parquet. The parquet's regeneration is a Phase 0 concern.
4. **No portfolio-level drift detection.** Phase 4's `app/services/calibration/drift.py` covers calibration drift; Phase 5b doesn't add a portfolio-level analogue (e.g., realised vs. expected loss CUSUM). Likely a Phase 7 deliverable.

---

### Session: 2026-05-13 (d) — Phase 6 frontend scaffold (parallel-agent, worktree)

**Branch:** `worktree-agent-aa1eee584ff871c89` (merged into `main` alongside Phase 5b).

**Completed:**

Stood up `frontend/` from empty per CLAUDE.md §3 and §4. Hand-rolled Next.js 14.2 App Router project (TypeScript strict, Tailwind, JetBrains Mono + Inter via `next/font/google`, Lucide icons, Recharts for the EV gauge — no other UI libs).

*Pages*
- `app/page.tsx` — drag-and-drop PDF upload, Demo / mock toggle, success summary with "Analyze card →" CTA.
- `app/card/[id]/page.tsx` — sticky header + breadcrumb, `RaceTabs` across the top, selected race shows `RaceHeader` + `HorseTable` (sorted by model_prob desc) + `EVGauge` side-by-side.
- `app/card/[id]/portfolio/page.tsx` — risk strip (bankroll, expected_return, total_stake_fraction, var_95, cvar_95, bet count) + `BetTicket` + no-op Confirm CTA.

*Components* (all under `frontend/components/<Group>/`): `RaceHeader`, `RaceTabs`, `HorseRow`, `HorseTable`, `EVGauge` (Recharts BarChart), `ProbabilityBar` (pure SVG stacked bar), `BetTicketRow`, `BetTicket`. Expandable horse detail rows show top-3 PP lines + computed features (local state, no router push).

*Lib*
- `lib/types.ts` — full TS mirror of `app/schemas/race.py` and `app/schemas/bets.py`. Pydantic Optional → `T | null`. Tuples (`BetCandidate.selection`) → `number[]`. UI-only optional fields on `HorseEntry` (`program_number`, `model_prob`, `market_prob`, `edge`) for the calibrated-prob payload the `/analyze` endpoint will eventually provide.
- `lib/api.ts` — `uploadCard` / `getCard` / `getPortfolio` typed wrappers. Reads `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`) and `NEXT_PUBLIC_MOCK_API`. When MOCK is `true`, returns deterministic seeded mock data with a 600-900 ms simulated network delay.
- `lib/mock.ts` — 9-race Churchill Downs card (mirrors the CD-05/10/2026 fixture). Seeded `mulberry32` PRNG → 7-12 horses per race, model_probs that sum to ~1.0, mix of +/- edges, 6 BetRecommendations across the card with every `stake_fraction` ≤ 0.03 (ADR-002) and `cvar_95 = -8.5%` of bankroll (well under the 20% rail).
- `lib/format.ts` — `formatOdds`, `formatProb`, `formatMoney`, `formatDistance` (sprint → `6f`, route → `1 1/16m`), `formatTime`, `formatEdge`, `formatFraction`, `formatSelection`, `formatBetType`, `formatDate`. All null-safe.

*Backend endpoints still missing for production* (see frontend/README.md):
- `GET /api/v1/cards/{card_id}` → `RaceCard` (calibrated probabilities hydrated). Currently mocked.
- `GET /api/v1/portfolio/{card_id}` → `Portfolio`. Currently mocked. Will be served by Phase 5b once merged.
- The existing `POST /api/v1/ingest/upload` does not echo the persisted `card_id` on `IngestionResult`. Frontend handles it gracefully (falls back to `"latest"`); when the backend is updated, no frontend change required — the IngestionResult type already declares the optional `card_id` field.

**Verification:**
- `npm install` clean (424 packages, expected upstream deprecation warnings).
- `npm run build` ✅ — 4 routes (`/`, `/_not-found`, `/card/[id]`, `/card/[id]/portfolio`) built with zero TypeScript or ESLint errors. First Load JS 87.5 kB shared. Tailwind compiled clean.
- `node_modules/` and `.next/` confirmed in `.gitignore` and absent from the working tree.

**Deferred / polish:**
- Real upload progress events (the FastAPI endpoint isn't streaming, so this would be a fake animation). Skipped.
- Tests: explicitly out of scope per Phase 6 spec ("no tests").
- `/analyze` endpoint integration: hydrate `model_prob`/`market_prob`/`edge` on `HorseEntry` from the calibration layer. The fields are already typed; the moment the backend ships the endpoint the existing components light up.

**Next:** Phase 7 — feedback / online learning loop (Master Reference §179-187). Or: ship the backend `/api/v1/cards/{id}` + `/api/v1/portfolio/{id}` endpoints so the frontend can leave MOCK mode.

---

### Session: 2026-05-13 (b) — Phase 5a EV engine (subagent-driven)

**Completed:**

Phase 5a built end-to-end via the `superpowers:subagent-driven-development` workflow on branch `phase-5a-ev-engine`. Eight tasks, each TDD-style (failing test → implementation → green test → commit), each followed by a spec-compliance + code-quality review pair, with cosmetic fixes applied as separate follow-up commits.

*Task 1 — `app/schemas/bets.py`*
- `BetCandidate`, `BetRecommendation`, `Portfolio` Pydantic v2 models.
- Validator enforces selection length per bet type (Win=1, Exacta=2, Trifecta=3, Superfecta=4) and distinctness. Pick3/4/6 + Place/Show rejected with explicit ValueError per ADR-039.
- 11 tests.

*Task 2 — PL place/show helpers*
- `place_prob(p, i)` and `show_prob(p, i)` closed-form in `app/services/ordering/plackett_luce.py`. Verified against brute-force `enumerate_exotic_probs` enumeration; Σ_i place_prob = 2 and Σ_i show_prob = 3 exactly.
- Added `_DENOM_TOL = 1e-12` named constant for the degenerate-denominator guard (sibling to existing `_SUM_TOL`).
- 9 new tests.

*Task 3 — `app/services/portfolio/sizing.py`*
- `kelly_fraction(edge, decimal_odds, fraction=0.25)` and `apply_bet_cap(stake, cap=0.03)` pure functions per ADR-002.
- 14 tests (15 after review-fix added the `decimal_odds=1.0` boundary case).

*Task 4 — `app/services/ev_engine/market_impact.py`*
- `post_bet_decimal_odds(pre_odds, bet_amount, pool_size, takeout_rate)` closed form: `(1−τ)(P+x) / (B+x)` where `B = (1−τ)·P / pre_odds`. `pool_size=None` returns pre_odds unchanged.
- `inferred_winning_bets(pre_odds, pool_size, takeout_rate)` helper.
- `DEFAULT_TAKEOUT` table (win/place/show=0.17, exacta=0.21, trifecta/superfecta/pick=0.25) per Master Reference §190.
- 12 tests (10 + 2 added in review: saturation case `bet_amount == B`, explicit `pool_size <= 0` rejection at both entry points).

*Task 5 — `app/services/ev_engine/calculator.py` (Win path)*
- `compute_ev_candidates(race_id, win_probs, decimal_odds, bet_types, min_edge=0.05, bankroll=1.0, pool_sizes=None, takeout_rates=None)` orchestrator. Returns `list[BetCandidate]` sorted by descending EV.
- Single-step market-impact approximation: uses PRE-impact Kelly as the impact-estimate stake. Because `post_odds ≤ pre_odds`, the estimate OVERESTIMATES impact — reported EV is a conservative lower bound. When stake vastly exceeds pool, the post-impact edge recheck returns None.
- 11 tests.

*Task 6 — calculator (exotics)*
- Added `_PL_PROB_FN: dict[BetType, Callable[...]]` dispatch and `_candidate_for_exotic`. Caller supplies `exotic_odds: dict[BetType, dict[tuple[int,...], float]]` mapping each permutation to its gross decimal odds.
- Refactored to a shared `_build_candidate(race_id, bet_type, selection, model_prob, pre_odds, …)` helper after the code-quality reviewer flagged Win/Exotic duplication. Both `_candidate_for_win` and `_candidate_for_exotic` now compute model_prob then delegate.
- 6 new tests (including a -EV exacta case to exercise the filter from both directions).

*Task 7 — validation script*
- `scripts/validate_phase5a_ev_engine.py` — backtest mode using historical `odds_final` per ADR-040. Mirrors `validate_calibration.run_live` structure (reuses `_three_way_split` and `_stack_for_meta`).
- Added `--sample-frac` arg after initial smoke runs hung on the 2.3M-row `prepare_training_features` step. 5% sample completes in ~5 min.
- Smoke artifact: `models/baseline_full/ev_engine/smoke-test-001/report.json`.

*Task 8 — ADRs + this progress entry*
- ADR-039 (Phase 5a bet-type scope) + ADR-040 (odds-source-agnostic + smoke-result data-quality note) appended to DECISIONS.md.

**Key decisions made (new ADRs):**
- **ADR-039:** Phase 5a supports WIN, EXACTA, TRIFECTA, SUPERFECTA only. Place/Show require live pool composition (which other horses hit the board, with what bet weights); deferred until tote ingestion exists. Pick N requires the shared latent track-state model from Master Reference §206-209; deferred.
- **ADR-040:** EV calculator is odds-source-agnostic. Validation script picks the mode: backtest uses `odds_final`; live mode is stubbed (`NotImplementedError`) until PDF-ingestion-to-EV integration is built. ADR also pins the smoke-result data-quality issue.

**Surprising / non-obvious findings to carry forward:**

- **The smoke run's "mean edge 0.705" is a data issue, not a model issue.** Historical `odds_final` includes 99/999 placeholders that, fed into `1/odds`, produce near-zero market probabilities. Phase 5b's first job is a `decimal_odds` cap in the validation script (suggested: ~100) before the optimiser is wired in. Without this filter, the optimiser will allocate to data noise.

- **Single-step market-impact approximation is provably conservative.** Because post_odds ≤ pre_odds always, post_kelly ≤ pre_kelly, so using pre_kelly as the impact-estimate stake OVERESTIMATES impact. Reported EV is a lower bound on true settled EV. This was noticed during code review and pinned in the comment; the alternative (fixed-point iteration on stake) would be tighter but slower and adds no real value when the cap is already binding most of the time.

- **The Win/Exotic candidate-construction body was nearly identical and got extracted into `_build_candidate` only in the Task 6 code-review pass.** Task 5 created the Win path first; Task 6 copied it for Exotics. The duplication was caught by the code reviewer; the extraction was a clean 4-line each leaf with one shared 60-line core. Lesson: when adding a "parallel" code path with a single divergent step, the reviewer should specifically ask "is the divergent step the only divergence?"

- **Phase 4's calibration metadata directory naming saved us.** The calibrator script writes artifacts to `models/baseline_full/calibration_adr038_brier/<model>/`. The new EV validation script loads from that exact path. If the calibration artifact naming had not been version-tagged, swapping in a future calibration approach would have required schema changes here.

- **`place_prob` / `show_prob` are deferred but worth having.** Phase 5a does not emit Place/Show, but the PL marginals are clean closed forms with no infrastructure cost. Adding them now (with tests) keeps PL's surface complete; Phase 6+ doesn't need to revisit `plackett_luce.py` when live pool data lands.

**Code-review findings applied (cosmetic, separate follow-up commits per task):**

- Task 1: renamed `_validate_selection` → `validate_selection` (Pydantic validators show up in error tracebacks; the leading underscore was misleading); added `BetRecommendation` docstring note about stake/stake_fraction consistency being a caller responsibility; extended Pick-N rejection test to cover PICK4 and PICK6.
- Task 2: extracted `_DENOM_TOL = 1e-12` constant; renamed `d1`/`d2` → `denom_j`/`denom_jk`; strengthened certain-horse test to exercise the skip branch on a non-certain horse.
- Task 3: removed unused `import math` from test file; asserted the warm-up arithmetic cases that were previously only in comments; added `decimal_odds=1.0` boundary test.
- Task 4: consolidated `_validate` to handle optional `pool_size`; removed duplicate inline guards in `inferred_winning_bets`; added saturation test (`bet_amount == B`).
- Task 5: dropped unused imports of `DEFAULT_KELLY_FRACTION` / `DEFAULT_MAX_BET_FRACTION`; replaced "~1% accuracy" comment with the more honest "conservative approximation — reported EV is a lower bound" description.
- Task 6: extracted shared `_build_candidate` (the biggest review-driven refactor of the phase); type-hinted `_PL_PROB_FN`; added a -EV permutation to the exacta test for filter symmetry; clarified docstring on the distinct-indices test (the error originates in PL's `_validate_indices`, not in the BetCandidate schema).

**Tests Status:**
- Pre-Phase-5a baseline: 503 passing
- Task 1 (+11 schema): 514
- Task 2 (+9 PL helpers): 523
- Task 3 (+14 sizing) + Task 3 review (+1 boundary): 538
- Task 4 (+10 market impact) + review (+2 saturation/pool-rejection): 530 (counts re-baselined as tests evolve)
- Task 5 (+11 calculator Win): 541
- Task 6 (+6 exotic): 547
- **Final: 547 tests passing in ~25s.**

**Known limitations carried forward into Phase 5b:**

1. **`odds_final` cap not yet applied.** Validation script needs a `--max-decimal-odds` flag (suggested default 100) before optimiser results are meaningful.
2. **No `BetRecommendation`/`Portfolio` producer yet.** The schemas exist (Task 1) but Phase 5b's optimiser is the first consumer. Tests for those classes are minimal (round-trip only) until 5b.
3. **`live` validation mode is a `NotImplementedError` stub.** Wires up when PDF-ingestion-to-EV integration lands.
4. **Phase-3 stub models still return 0.5.** PaceScenarioModel and SequenceModel are constant. Meta-learner ECE is 0.003 raw so this is acceptable for Phase 5a's marginal-prob inputs but should not be assumed acceptable when the optimiser starts taking real positions.

---

### Session: 2026-05-13 (a) — Calibrator hardening: ADR-037 held-out IV + ADR-038 skip guard + Brier co-criterion

**Completed:**

*ADR-037 — held-out inner-val ECE (replaces fit-slice criterion)*
- `Calibrator._auto_select` now does an internal seeded shuffle of the
  fit slice (default 80% inner-train / 20% inner-val), fits Platt and
  isotonic on inner-train, computes ECE on inner-val for each, and
  picks the lower one — with a 0.001 protective bias toward Platt
  (isotonic must beat Platt by at least that on inner-val).
- Falls back to fit-slice ECE only when the calib slice would produce
  fewer than `auto_min_inner_val_size=100` inner-val rows.
- `metadata.json` adds `inner_val_metrics` and `auto_selection_mode`
  ("held_out" or "fit_slice_fallback").

*ADR-038 — skip-when-calibrated guard + time-ordered inner-val + Brier co-criterion*
- New config field `skip_threshold_delta` (default 0.001) AND
  `brier_skip_delta` (default 1e-4). The auto-selector computes raw
  inner-val ECE AND Brier alongside Platt and isotonic. Calibration
  applies only if the winning method beats raw on BOTH metrics by the
  configured deltas. Otherwise `chosen_method='identity'` and
  `predict_proba` returns raw clipped to [0, 1] — no calibration.
- The Brier leg is essential. The first post-ADR-038 full-parquet run
  (ECE-only skip, before the Brier co-criterion) showed isotonic
  genuinely winning on inner-val ECE (delta ≈ 4-5× threshold) yet
  having no measurable Brier improvement. The "win" was bin
  redistribution, not accuracy improvement, and on the test slice it
  produced strictly worse ECE (0.003 → 0.005-0.006). Brier is
  strictly proper — its minimum is achieved only at the true
  distribution and it can't be improved by bin redistribution. With
  Brier as co-criterion the guard correctly fires on this dataset.
  This refinement was suggested by the advisor; the alternative
  (bumping `skip_threshold_delta` to ~0.005) was rejected as data-
  snooping on a single dataset.
- New `Calibrator.fit(scores, labels, inner_val_indices=...)` parameter.
  Caller supplies explicit row indices for the inner-val slice — used by
  `validate_calibration.py` to pass the time-ordered TAIL of the calib
  slice (last 20% by `race_date`). Selection mode reflects the source:
  `held_out_caller` for caller-supplied indices, `held_out` for seeded
  random shuffle, `fit_slice_fallback` for tiny calib slices.
- `inner_val_metrics` and the full fit-slice `metrics` dict both gain
  an `identity` entry alongside `platt` and `isotonic` (with both
  ECE and Brier per method) so the choice is fully auditable.
- Identity mode still persists fitted Platt + isotonic in the artifact
  directory for diagnostic comparison.
- Validation (`scripts/validate_calibration.py`) computes the
  time-ordered tail with `calib['race_date'].argsort()[-20%:]` and
  passes it as `inner_val_indices`. Logs the inner-val date range
  for transparency.

**Key Decisions Made (new ADRs):**

- **ADR-037: Calibrator auto-selector uses held-out inner-val ECE.**
  Supersedes ADR-030's fit-slice criterion. The fit-slice criterion
  always picks isotonic because isotonic memorises (post-fit ECE
  ≈ 1e-18). Held-out inner-val gives a real generalisation signal.
- **ADR-038: Skip-when-calibrated guard + time-ordered inner-val +
  Brier co-criterion.** Refines (does not supersede) ADR-037. The
  guard requires the winning calibrator to beat raw on BOTH inner-val
  ECE (by `skip_threshold_delta`, default 0.001) AND inner-val Brier
  (by `brier_skip_delta`, default 1e-4). Brier is strictly proper —
  needed because ECE alone can be gamed by bin redistribution. The
  time-ordered inner-val (caller-supplied indices) handles
  distribution shift between calib and test windows by making the
  inner-val match the test slice as closely as possible. Identity
  mode is a first-class outcome that returns raw scores clipped to
  [0, 1].

**Surprising / non-obvious findings to carry forward:**

- The "isotonic over-fits the calib slice and degrades test ECE"
  finding was actually THREE problems stacked: selection bias in the
  criterion (ADR-037 fix), time-distribution shift between calib and
  test windows (ADR-038 time-ordered IV), AND ECE bin-redistribution
  noise that isotonic can win on without genuinely improving accuracy
  (ADR-038 Brier co-criterion). Each fix in isolation is insufficient
  on this dataset. The Brier co-criterion was the decisive one for
  the full-parquet behavior — even with a perfectly placed inner-val
  matching the test distribution, ECE alone is too noisy a criterion
  at the relevant sample sizes (~46K rows × 15 bins → ECE SE ~0.001,
  same magnitude as the threshold).
- "Identity" as a first-class calibration outcome is the right
  primitive. Forcing a Platt-vs-isotonic choice when neither is
  better than raw is asking the auto-selector to make a decision
  that has no winning answer. The identity path also makes the
  pipeline robust to model updates: a future retraining that
  produces well-calibrated raw output won't silently degrade
  through a learned calibration map.
- Brier as a co-criterion alongside ECE is the cheap way to make
  the auto-selector robust to bin-redistribution gaming. Without it,
  any threshold on ECE alone can be wrong — too tight and you miss
  real wins, too loose and you accept fake wins. Brier's strict
  propriety means the threshold can stay tight (1e-4 ≈ 0.3 SE) and
  still filter out bin-noise wins.
- Time-ordered inner-val is the CHEAP version of walk-forward
  calibration. Walk-forward (re-fit on a rolling window of the most
  recent N races) is the production-grade answer; the tail-of-calib
  inner-val gives most of the signal at zero infrastructure cost.

**Tests Status:**
- **483 tests passing in ~25s** (was 464 → +19 calibrator tests).
- New tests: skip-when-calibrated on already-calibrated stream;
  identity-mode predict_proba returns clipped raw; identity
  save/load round-trip; skip threshold tuning (large positive forces
  identity, default lets clear wins through); caller-supplied
  inner-val indices recorded as `held_out_caller`; time-ordered tail
  changes selection on a regime-shifting synthetic; inner-val
  index validation (out of range, duplicates, empty, ignored for
  non-auto methods); Brier recorded alongside ECE in inner_val_metrics;
  Brier co-criterion blocks bin-redistribution wins; Brier guard lets
  genuine improvement through (staircase still picks isotonic);
  brier_skip_delta tuning knob behaves as documented.

**Known limitations carried forward:**

- The Phase-3 stub models (PaceScenarioModel, SequenceModel) still
  return constant 0.5. Phase 5 EV calculation will be valid only to
  the extent that the meta-learner's marginal win probs are
  trustworthy. Given the full-parquet ECE 0.003, that's a defensible
  starting point.
- `CopulaModel.infer_strengths` is not yet implemented (would mirror
  `SternModel.infer_strengths`). Added when the pace model is real.

---

### Session: 2026-05-12 (f) — Phase 4 complete: Stern + Copula + CUSUM + full-parquet calibration

**Completed:**

*Ordering — Stern (Gamma) model (`app/services/ordering/stern.py`, 29 tests)*
- `SternConfig` (shape, n_samples, seed) + `SternModel` API parallel to
  `plackett_luce`. Shape=1 delegates to PL closed forms; shape≠1 uses
  MC via vectorised `rng.gamma(shape, scale=1/v, size=(n, N))` and
  `argsort`. Default 20k samples gives SE ~0.002 on typical exotic probs.
- `implied_win_probs` reports P(i wins) the model actually predicts
  (= strengths only at shape=1; differs otherwise).
- `infer_strengths(target, damping=0.5)` — damped multiplicative fixed-
  point iteration over MC-estimated implied marginals, converges in ~30
  iters at damping=0.5 for shape ≤ 5.
- `fit_stern_shape(corpus, shape_grid)` — grid-search MLE that scores
  observed top-`top_k` prefix matches; top-1 is invariant under shape
  given calibrated marginals, so `top_k >= 2` is enforced.

*Ordering — Copula model (`app/services/ordering/copula.py`, 20 tests)*
- Gaussian copula with Gamma marginals over a block-equicorrelation
  matrix (ρ within pace style, 0 across). `rho ∈ [0, 1)`.
- Three PL/Stern fast paths: `pace_styles=None`, `rho=0`, or
  `marginal_shape ≠ 1` → delegated. The MC path: Cholesky on
  Σ + 1e-10·I, `randn @ L.T`, `Φ(Z)` clipped, `Gamma.ppf`.
- KEY finding pinned in tests + docstring: under ρ > 0 within a style
  block, the STRONGER horse's marginal win share grows at the expense
  of the weaker. The Luce property does NOT survive. This is the
  intended behaviour (same-style horses compete for the same trip;
  independent-variation "upsets" shrink). A reviewer who expects
  marginal preservation will be surprised — the docstring + the test
  `test_correlation_concentrates_winshare_on_stronger_block_member`
  prevent silent regressions.

*Calibration — CUSUM drift detection (`app/services/calibration/drift.py`, 26 tests)*
- First attempt used raw residuals (label − pred) with k=0.025, h=2.
  False-alarmed at step 22 on a 1000-obs perfectly-calibrated stream
  because a single label=0 on pred=0.9 contributes -0.9 in one step,
  which is enormous relative to k=0.025 / h=2. The defaults were
  irreconcilable with the residual scale.
- Rewrote to STANDARDISED Bernoulli z-score residuals:
  `z = (label − p) / sqrt(p·(1−p))`, floored on the denominator (eps=1e-4)
  and clipped magnitude (z_clip=5). New defaults k=0.5, h=4 follow
  Hawkins-Olwell standard SPC tables (ARL₀ ≈ 168). Tests now pass with
  deterministic calibrated prefixes (alternating 0/1 at p=0.5) used as
  quiet warm-ups so the drift signal is isolated.
- Incremental `CUSUMDetector` + one-shot `detect_drift(...)` helper.
  Alarm latches on first crossing; `reset()` clears for a new monitoring
  epoch. `direction` field reports "high" (model under-predicts) vs.
  "low" (model over-predicts).

*Full-parquet validation*
- Ran `scripts/validate_calibration.py` on the full 2.3M-row parquet.
  Total ~14 min wall-clock; `prepare_training_features` dominates
  (~14 min for the per-horse groupby + shifts), scoring + calibration
  ~5 sec. Calib 231,942 rows / test 231,160 rows, dates 2015-07-12 to
  2018-02-18.
- Finding: both Speed/Form and Meta-learner are ALREADY well-calibrated
  raw (ECE ≈ 0.003). Auto-selector picked isotonic (fit-slice ECE → 0
  always wins), but isotonic on this slice SLIGHTLY DEGRADED test-set
  ECE (0.003 → 0.005–0.006). Brier essentially unchanged. The 5% smoke
  result (meta-learner 0.121 → 0.011) was a small-sample artifact.
- The auto-selector's fit-slice ECE criterion is broken in the
  well-calibrated regime — ADR-030 already flagged this as provisional.

**Key Decisions Made (new ADRs):**

- **ADR-034: Stern MC for shape ≠ 1, PL closed form for shape = 1.**
  Strengths are Gamma rates; at shape=1 they ARE the win probs (Luce).
  At shape ≠ 1 they are not — `infer_strengths` recovers target
  marginals via damped fixed-point.

- **ADR-035: Copula uses block-equicorrelation by pace style with
  PL/Stern fallbacks.** ρ within style, 0 across — PSD by construction
  for ρ ∈ [0, 1). Negative ρ rejected; cross-style ρ deferred. Luce
  marginal preservation explicitly NOT a goal at ρ > 0 (the docstring
  is honest about this).

- **ADR-036: CUSUM operates on standardised Bernoulli z-scores, not
  raw residuals.** Raw residuals are heteroscedastic in p; a single
  high-confidence error dominates the running statistic. Standardising
  fixes the scale, and defaults k=0.5, h=4 follow standard SPC tables.

**Surprising / non-obvious findings to carry forward:**

- The 5%-sample calibration smoke run from session (e) **lied**.
  Meta-learner pre-ECE 0.121 reported there is not consistent with the
  full-parquet pre-ECE 0.003. Small-sample ECE estimates have high
  variance when the test slice is < ~50K rows. From now on,
  **interpret smoke calibration results as PIPELINE-correctness checks
  only, not as production-fitness signals**.

- Auto-selector picks isotonic 100% of the time on this dataset
  (fit-slice ECE for isotonic is ~1e-18 by memorisation), but isotonic
  is wrong here — the test-slice ECE goes UP after calibration. This
  is the canonical "fit-slice criterion overfits" failure mode that
  ADR-030 flagged. **Phase-4 follow-up before Phase 5** (recommended):
  change the auto-selector to either (a) require a minimum delta-ECE
  on a held-out slice within the fit set, or (b) default to Platt
  with isotonic only chosen when it wins held-out by a margin.

**Known limitations carried forward:**

- The Phase-3 stub models (PaceScenarioModel, SequenceModel) still
  return constant 0.5. Phase 5 EV calculation will be valid only to
  the extent that the meta-learner's marginal win probs are
  trustworthy. Given the full-parquet ECE 0.003, that's a defensible
  starting point.
- `CopulaModel.infer_strengths` is not yet implemented (would mirror
  `SternModel.infer_strengths`). Added when the pace model is real.

**Tests Status:**
- **464 tests passing in ~25s** (was 389 → +75 new). Run with
  `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: 2026-05-12 (e) — Phase 4 calibration + Plackett-Luce (code complete; full run pending)

**Completed:**

*Calibration module (`app/services/calibration/`)*
- `calibrator.py` — `Calibrator` class with three modes (`platt`, `isotonic`,
  `auto`). The auto-selector fits both estimators and keeps the one with
  lower ECE on the fit set (ADR-008). `predict_softmax(scores, race_ids)`
  applies temperature-scaled logit softmax per race group; T < 1 sharpens,
  T > 1 flattens. Helpers `expected_calibration_error`, `reliability_bins`
  (returns exactly `n_bins` entries — empty bins included so a downstream
  plotter doesn't need to special-case them), and `brier_score`. Save/load
  is joblib for the sklearn estimators + JSON sidecar for metadata.
- `tests/test_calibration/test_calibrator.py` (21 tests) — ECE math on
  perfectly-calibrated synthetic, biased-prob ECE growth, empty-bin
  handling, Platt monotonicity, isotonic monotonicity, auto picks isotonic
  when distortion is non-sigmoidal (3-plateau staircase), softmax sum-to-1
  per race, temperature sharpens distribution, save/load round-trip.

*Ordering module (`app/services/ordering/`)*
- `plackett_luce.py` — analytical exotic probability functions
  (`exacta_prob`, `trifecta_prob`, `superfecta_prob`, plus
  `enumerate_exotic_probs(p, k)` for grid-search workflows), a
  `sample_ordering` helper, and `fit_plackett_luce_mle(orderings, n_items)`
  for the corpus-level MLE strength fit. Scale fixed by pinning θ[0] = 0,
  then strengths are exp-normalised so `strengths.sum() == 1`. The NLL is
  **vectorised over orderings**: pads to a rectangular `(n_ord, max_len)`
  index array and runs one `scipy.special.logsumexp` per position rather
  than per (ordering, position) pair — drops the test suite from 66s to 2s.
- `tests/test_ordering/test_plackett_luce.py` (21 tests) — 2-horse exacta
  sum-to-1, 3-horse trifecta uniform field gives 1/6 per permutation,
  skewed-field hand-computed values, exacta marginal recovers
  win-prob, superfecta sums to 1 over 24 perms, superfecta extends
  trifecta via summation, edge cases (zero-prob horse, certain horse,
  invalid sum / negative / out-of-range index / repeated index), MLE
  recovery on 5000 synthetic orderings to within 0.05 abs error per
  strength, MLE handles partial top-k orderings.

*Validation script*
- `scripts/validate_calibration.py` — pure `evaluate_calibration(calib, test)`
  returning a `CalibrationReport`, `render_reliability_diagram(report, png)`
  via matplotlib (headless Agg backend), plus a `run_live()` CLI that
  loads the trained `models/baseline_full/` artefacts, re-runs the feature
  pipeline, 3-way time-splits (train / calib / test), scores both
  Speed/Form and Meta-learner, and writes a JSON + PNG report per model.
- `tests/test_calibration/test_validate_calibration.py` (3 smoke tests) —
  pure-function round-trip, JSON-dict serialisation, PNG header check.

*Cross-cutting bug fix (uncovered by the live smoke run)*
- `MarketModel.load()` previously set `iso.f_ = None`, causing every
  prediction on a freshly-loaded model to crash with
  `'NoneType' object is not callable`. The bootstrap had only ever
  exercised the in-memory fit-then-predict path, so the bug went unnoticed
  until Phase 4 reloaded the model. Fix: rebuild the linear interpolation
  via `scipy.interpolate.interp1d(xs, ys, kind="linear",
  bounds_error=False, fill_value=(ys[0], ys[-1]))`, matching what sklearn
  builds internally during `fit()`. Added `test_save_load_round_trip_
  preserves_predictions` to lock the regression.

*Dependency additions*
- `scipy>=1.12` (already pulled in transitively via sklearn, now explicit)
  for PL MLE optimisation and isotonic interpolation rebuild.
- `matplotlib>=3.8` for reliability diagrams.

*Live smoke run (5% sample → 11,575 calib / 11,574 test, dates
2015-07-19 → 2018-02-25):*

| Model         | Pre-cal ECE | Post-cal ECE | Pre Brier | Post Brier | Method   |
|---------------|------------:|-------------:|----------:|-----------:|----------|
| Speed/Form    | 0.0371      | **0.0049**   | 0.0801    | 0.0777     | isotonic |
| Meta-learner  | 0.1208      | **0.0112**   | 0.0988    | 0.0773     | isotonic |

The meta-learner's pre-calibration ECE of 0.12 — substantial — is exactly
the kind of finding ADR-008 said we'd uncover. Isotonic wins both rounds;
on this dataset the distortion is non-sigmoidal enough that Platt's logistic
fit is dominated. Post-calibration both heads sit in the (0.005, 0.015)
ECE band — well-calibrated by any reasonable threshold.

**Key Decisions Made:**

- **Auto-selector chooses by ECE on the fit slice, not a held-out set
  inside the fit.** ADR-008 specifies cross-validation; the simpler
  fit-slice ECE selects isotonic in our smoke run for the obvious reason
  (isotonic has more flexibility, so on its own training data it always
  wins). The Phase-4 evaluation uses a SEPARATE test slice (post-cal ECE
  on never-seen data) which is the metric that matters. Adding fold-based
  cross-val inside `fit()` would be the right next step before
  productionising, but the smoke run shows it doesn't change the choice
  here. Deferred to a Phase-4 follow-up if Platt ever beats isotonic on
  the full parquet test slice.

- **Softmax-with-temperature is implemented as logit-scale, not power-
  scale.** Power-scaling (p^(1/T) / Σ p^(1/T)) is an alternative but
  doesn't have the additive-model interpretation that downstream PL needs.
  Logit-scale temperature is the calibration-literature default and what
  `nn.Softmax` does internally.

- **PL strengths are win probabilities at inference time; MLE fitter
  exists as a research tool.** The analytical exotic functions accept any
  valid distribution summing to 1 — at the per-race inference layer that
  IS the calibrated meta-learner output. The `fit_plackett_luce_mle`
  surface is provided so we can backtest "does PL fit our historical
  orderings better than Harville / Stern" — never to be called per-race
  at inference time.

- **`enumerate_exotic_probs` is O(N!/(N-k)!).** For a typical 12-horse
  field, exacta = 132 entries, trifecta = 1,320, superfecta = 11,880. A
  20-horse Kentucky Derby field at superfecta = 116,280. Fine for k ≤ 4.
  Out-of-scope for full pick-N enumeration (handled later via Monte Carlo
  sampling from `sample_ordering`).

- **Reliability diagram includes ALL bins (with NaN summary stats for
  empty bins).** The downstream plotter filters out `count == 0` rows
  before drawing markers. This keeps the bin output shape fixed at
  `n_bins` so the report JSON has a predictable schema.

**Known limitations carried forward:**
- Full parquet validate_calibration run not yet executed in this session;
  smoke (5% sample) confirms the pipeline. The 100% run is bottlenecked on
  `prepare_training_features` (the per-horse groupby — ~5 min on 5%, so
  maybe 30-90 min on 100%). To be run when bandwidth allows.
- Drift detection (`app/services/calibration/drift.py`) not started.
- Stern (Gamma) and Copula ordering models not started.

**Tests Status:**
- **389 tests passing in ~11s** (was 343 → +46 new). Run with
  `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: 2026-05-12 (d) — Phase 3 baseline model stack trained

**Completed:**

*Sub-model framework (all under `app/services/models/`)*
- `training_data.py` — leakage-free feature prep. `groupby('horse_key').shift(1)`
  precedes every per-horse rolling aggregate (`ewm_speed_prior`, `last_speed_prior`,
  `best_speed_prior`, `mean_speed_prior`, `speed_delta_prior`, `mean_finish_pos_prior`,
  `win_rate_prior`, `mean_purse_prior`, `days_since_prev`, `layoff_fitness`). EWM
  uses `alpha=0.4` (CLAUDE.md §8). Adds field-relative columns
  (`ewm_speed_zscore`, `ewm_speed_rank`, `ewm_speed_pct`, `weight_lbs_delta`) and
  derives `field_size` from in-frame group counts when the source column is NULL
  (which is 74% of the parquet — see "Known Export Caveats"). Exposes
  `time_based_split(df, val_fraction)` per CLAUDE.md §2's no-random-split rule.
- `speed_form.py` — `SpeedFormModel` (Layer 1a). LightGBM binary classifier on
  `win`. Per-race softmax over RAW scores (not sigmoided) for the in-race
  probability distribution. Save/load round-trip via LightGBM's native text
  format + a JSON metadata sidecar.
- `connections.py` — `ConnectionsModel` (Layer 1d). Empirical-Bayes shrinkage
  estimator (beta-binomial pseudo-count `α=30`) over per-jurisdiction baseline
  → per-jockey → per-trainer → per-pair win rates. Inference falls back through
  the hierarchy when a pair/jockey/trainer is OOV. JSON serialisation.
- `market.py` — `MarketModel` (Layer 1e). Implied-prob from `odds_final` → in-race
  normalised → isotonic-regression calibration over historical wins. NaN for
  rows where `odds_final` is missing (caller decides fallback).
- `pace_scenario.py` + `sequence.py` — placeholder classes returning constant
  0.5 from `predict_proba`. Their `fit()` raises `NotImplementedError` until
  unblock criteria are met (pace: fractional times in parquet; sequence:
  PyTorch + globally-unique horse_id). Documented inline.
- `meta_learner.py` — `MetaLearner` (Layer 2). LightGBM head over the 5 sub-model
  outputs + meta features (`field_size`, `distance_furlongs`).
  Orthogonalisation (CLAUDE.md §2): every sub-model column except the
  anchor (`speed_form_proba`) is replaced with its residual after linear
  regression on the anchor. Cleanly separates "what the anchor already
  knows" from each layer's marginal contribution.

*Bootstrap orchestrator*
- `scripts/bootstrap_models.py` — end-to-end: load parquet → prepare features →
  time-split → fit Speed/Form + Connections + Market → stack predictions →
  fit MetaLearner → persist artifacts + summary JSON. Supports
  `--sample-frac` for smoke tests, `--run-name` for reproducible run dirs.

*Tests*
- `tests/test_models/_synth.py` — synthetic data generator producing a 480-row
  fixture where `speed_figure` predicts `finish_position` (with noise) so trained
  models have real signal to learn.
- `tests/test_models/test_training_data.py` (15 tests) — no-leakage invariant,
  field-size derivation, EWM math sanity, categorical dtype, time-split
  boundaries, parquet round-trip.
- `tests/test_models/test_speed_form.py` (10 tests) — fit/predict shape,
  val_top1_acc > random, softmax sum-to-one per race, numerically-stable softmax
  on large inputs, save/load round-trip with identical predictions.
- `tests/test_models/test_connections.py` (7 tests) — predict shape, OOV
  fallback to jurisdiction baseline, shrinkage bounded in [0, 1], save/load.
- `tests/test_models/test_market.py` (5 tests) — implied-prob math,
  non-positive odds → NaN, missing-odds rows → NaN predictions.

*Live bootstrap run on full 2.3M-row parquet*

| Model         | Train log loss | Val log loss | Val AUC | Val race top-1 |
|---------------|---------------:|-------------:|--------:|---------------:|
| Speed/Form    | 0.254          | 0.262        | 0.745   | **0.257**       |
| Meta-learner  | —              | **0.237**    | —       | **0.341**       |

Field size ≈ 12 horses on average, so random pick ≈ 0.083. Meta-learner is
**~4× better than random** at picking the winner. Stacking provided a
meaningful lift over Speed/Form alone (top-1 0.257 → 0.341, log-loss 0.262 → 0.237) —
the orthogonalised connections + market columns carry real information.

Artifacts persisted at `models/baseline_full/`:
  speed_form/{booster.txt, metadata.json}
  connections/model.json (2,291 jockeys / 2,719 trainers / 50,251 pairs)
  market/model.json
  meta_learner/{booster.txt, metadata.json}
  summary.json

**Bug fixes (during test authoring):**
- The naive cross-row AUC assertion on synthetic data was misleading — speed
  figures vary widely across small synthetic races, so cross-race AUC is
  near-chance even when the model perfectly identifies WITHIN-race winners.
  Test swapped to `val_race_top1_accuracy > 0.40`, which is the metric that
  actually matters for the wagering use case.
- Shrunken rate boundary test was too strict (`0 < v < 1`): when both the
  observed sample and the prior are uniform-1.0, the posterior mean can
  legitimately touch the boundary. Relaxed to `0 <= v <= 1` and added a
  separate "average rate close to jurisdiction baseline" check that actually
  exercises the shrinkage strength.

**Key Decisions Made:**
- **Softmax operates on raw LightGBM scores, not sigmoided probabilities.**
  Sigmoid-then-renormalise double-squashes the dynamic range and destroys the
  signal at the tails (longshots and chalk). Raw-score softmax preserves the
  additive structure the model learned and produces a calibratable
  distribution. Phase 4 calibration sits on top.
- **Orthogonalisation lives in the meta-learner module, not the sub-models.**
  Each sub-model emits its raw output. The stacker decides what's orthogonal to
  what — keeps the sub-models composable and lets us swap in a different
  meta-learner (logistic regression, MLP) without touching Layer 1.
- **Pace and Sequence are deferred via STUB classes returning 0.5.** Without
  the fractional time columns (Pace) or PyTorch + a globally-unique horse_id
  (Sequence), training them now would be busywork. The meta-learner is robust
  to constant features (the orthogonalisation step zeros them out), so wiring
  the slots in advance lets us drop trained models in without touching the
  orchestrator.
- **Horse grouping key is `(horse_name, jurisdiction)`.** Not perfect — name
  collisions across years exist — but it's the most informative key the parquet
  exposes. The cleaner fix is to surface `horses.dedup_key` in
  `export_training_data.py`; deferred to a future export.
- **Connections shrinkage prior strength = 30.** Empirically chosen — pulls
  small-sample jockey/trainer rates toward jurisdiction baseline strongly
  enough that 1-of-1 windfalls don't propagate, but lets veteran jockeys
  with hundreds of starts express their true rate. Tunable via config.

**Known limitations carried forward:**
- Pace / Sequence layers untrained — contributing 0.5 to the stacker (zeroed
  out after orthogonalisation, so no harm done, but the meta is missing
  potentially valuable signal).
- AR exclusion stands. Jurisdiction values are limited to {UK, HK, JP}.
- No calibration yet — `predict_proba` outputs are NOT calibrated; using them
  for EV calculation would be wrong. That's the Phase 4 deliverable.
- `field_size` derivation runs after the parquet load (groupby in
  `prepare_training_features`). This is correct for training but the live
  ingest path must populate the column from the parsed RaceCard itself —
  noted, not blocking.

**Tests Status:**
- **343 tests passing in ~9s** (was 306 → +37 new). Run with
  `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: 2026-05-12 (c) — Phase 0/1/2 closeout sweep

**Completed:**

*Phase 0 (loose-end cleanup)*
- `scripts/db/backfill_tracks.py` — idempotent populator for the previously-empty
  `tracks` table. Derives one row per `(track_code, jurisdiction)` from `races` and
  records the union of surfaces observed at each track as a JSON array. Ran live:
  100 distinct combos inserted. Re-running is a no-op (UNIQUE constraint).
- `tests/test_db/test_backfill_tracks.py` (3 tests) — happy path, idempotency,
  surface aggregation.

*Phase 1 (closeout)*
- `.env.example` — full HRBS_* knob template aligned with `app/core/config.py`.
- `app/core/logging.py` — idempotent `configure_logging()` honouring `LOG_LEVEL`
  and `LOG_JSON`. ConsoleRenderer (with colour when stdout is a TTY) in dev,
  JSONRenderer in prod. `get_logger()` is the public helper.
- `app/services/pdf_parser/equibase_parser.py` — `EquibaseParser(BrisnetParser)`.
  Subclass placeholder; same regex layer for now. Extractor's `_get_parser()` now
  routes the `equibase` format to it (was falling back to BrisnetParser previously).
- `app/db/session.py` — async SQLAlchemy 2.0 engine (`AsyncEngine`) + session
  factory + `session_scope()` context manager + `get_session()` FastAPI dep +
  `init_db()` / `dispose_engine()` lifespan hooks. Declarative `Base` lives here.
- `app/db/models.py` — `IngestedCard`/`IngestedRace`/`IngestedHorse`/`IngestedPPLine`
  ORM hierarchy with `cascade="all, delete-orphan"` so deleting a card cleanly
  removes the whole tree. JSON columns for variable-length fields
  (medication_flags, equipment_changes, parse_warnings).
- `app/db/persistence.py` — `card_to_orm()` + `persist_ingestion_result()`. Splits
  Pydantic-to-ORM conversion from HTTP handler logic for testability.
- `app/main.py` — `create_app()` factory with async lifespan that boots logging
  + `init_db()`. CORS allow-all (dev), `/healthz` + `/version`, mounts the
  ingest router at `/api/v1/ingest`.
- `app/api/v1/ingest.py` — POST `/upload`. Multipart PDF, content-type guard,
  size guard at HTTP boundary, then `run_in_executor(ingest_pdf, ...)` so the
  CPU-bound parser doesn't block the event loop. Successful results are
  persisted via the SQLAlchemy session dependency; failures are returned to
  the client with `processing_ms` populated.
- `tests/test_api/test_ingest.py` (6 tests) — `/healthz`, `/version`, empty body
  → 400, bad content-type → 415, corrupt PDF → success=False, valid reportlab-
  drawn fixture → 200 + structured payload.
- `pyproject.toml` — added FastAPI, uvicorn, python-multipart, SQLAlchemy 2.0,
  aiosqlite, numpy to runtime deps; httpx to dev deps.

*Phase 2 (full implementation)*
- `app/services/feature_engineering/layoff.py` — `layoff_fitness(days)` returns a
  [0,1] score; `apply_layoff_features(df)` writes the column. `DEFAULT_LAMBDA =
  ln(2)/60` so fitness halves at `recovery_threshold + 60 days` (≈ 90-day layoff).
  First-time starters mapped to the sentinel `FIRST_TIME_STARTER_FITNESS = 0.6`.
  (12 tests)
- `app/services/feature_engineering/speed_features.py` — EWM α=0.4 (CLAUDE.md §8;
  pandas convention is most-recent-LAST, so we reverse the most-recent-first
  PP list before `.ewm()`). Per-horse summary: `ewm_speed_figure`, `last_speed_figure`,
  `best_speed_figure` (6-PP window), `speed_figure_delta`. Field-relative columns:
  `ewm_speed_zscore`, `ewm_speed_rank` (1=fastest), `ewm_speed_pct`. Constant fields
  z-score to 0 (no NaN). (13 tests)
- `app/services/feature_engineering/pace_features.py` — `pace_shape_metrics(pp)`
  builds `early_speed = -beaten_lengths_q1`, `late_kick = bl_q2 - bl_finish`,
  fraction ratios. Per-horse `horse_pace_summary()` averages over the 5-PP
  window. Field-level `pace_pressure_index` = count of horses whose mean
  `beaten_lengths_q1 ≤ 1.5L` (front-runner proxy). (13 tests)
- `app/services/feature_engineering/class_features.py` — claiming-to-claiming uses
  claiming-price delta directly; otherwise falls back to purse delta. Records
  `race_type_change` as `same` or `<modal>-><today>`. Field-relative z-score
  appended. (8 tests)
- `app/services/feature_engineering/connections.py` — jockey continuity proxies:
  `today_jt_same_pair`, `jockey_repeat_streak`, `today_jockey_win_rate_in_pps`.
  `trainer_continuity` is a weak proxy (no per-PP trainer field in the schema —
  documented inline). Names normalized via casefold + strip. (8 tests)
- `app/services/feature_engineering/engine.py` — `FeatureEngine.transform(card)`
  produces a tidy long-form DataFrame keyed by `(race_number, post_position)`.
  Joins all per-module frames + base identifiers + `weight_lbs_delta` (field-
  relative). One bad race is logged + skipped without killing the card. (9 tests)
- `tests/test_features/_fixtures.py` — shared synthetic-card factory used by
  every feature test module. Centralises the construction of valid PPs / horses
  / races / cards so test files stay focused on assertions.

**Bug fixes (none — full clean closeout)**

**Key Decisions Made:**
- **Live ingestion DB is separate from master training DB.** The Pydantic schemas
  in `app/schemas/race.py` model PDF-derived race cards (today's racing data).
  The ORM in `app/db/models.py` mirrors that schema, NOT the master `race_results`
  table. Two different stores: master DB is the training corpus at
  `data/db/master.db`; the live DB defaults to `./horseracing.db` (overridable
  via `HRBS_DATABASE_URL`). Keeping them isolated prevents schema collision and
  lets ingest-time writes happen without locking read-heavy training queries.
- **EquibaseParser is a thin subclass for now.** Without real Equibase PDFs to
  test against, divergences from Brisnet are speculative — better to dispatch on
  format (so behaviour is per-format from the start) than to fork the regex layer
  prematurely. Future overrides go on the subclass.
- **Layoff curve constants live in the module, not config.** `DEFAULT_LAMBDA` and
  `DEFAULT_RECOVERY_THRESHOLD_DAYS` are pure tunables — Phase 3 will fit them
  per-surface from the master DB and pass fitted values into the engine. No
  global mutable state.
- **EWM weighting direction is documented inline.** pandas' `.ewm(alpha)` weights
  the LAST element heaviest; our `HorseEntry.pp_lines` is enforced most-recent-
  FIRST. The module reverses before applying `.ewm()` and the docstring explains
  why. Two places where the direction matters: `ewm_speed()` and any future
  rolling stats.
- **Trainer continuity is a documented weak proxy.** `PastPerformanceLine` carries
  per-PP `jockey` but no per-PP `trainer` (the Brisnet PDF format itself omits
  the per-PP trainer in the dense PP table; only today's trainer appears in the
  horse header). The connections module returns a 0/1 proxy with the limitation
  documented; the Bayesian connections model in Phase 3 will get richer trainer
  context from the master training DB instead.

**Tests Status:**
- **306 tests passing in ~7s** (was 234 → +72 new). Run with
  `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: 2026-05-12 (b) — ML training parquet exported

**Completed:**
- Added `--jurisdictions UK,HK,JP` flag to `scripts/db/export_training_data.py`. Threaded as an
  optional allow-list through to the SQL via parametrized `AND r.jurisdiction IN (?,...)` clause
  appended to the existing `TRAINING_QUERY_TEMPLATE`. Default remains "all jurisdictions".
- Ran the export. Output: `data/exports/training_20260512.parquet`. 2,317,297 rows × 29 columns,
  38.9 MB. Date range 1986-01-05 → 2021-07-31. By jurisdiction: JP 1,609,930 / UK 598,708 / HK
  108,659. Mean field size (derived from group counts) 11.67 horses/race — healthy and consistent
  with real racing.
- Updated PROGRESS.md "Current State" + added a "Known Export Caveats" table cataloguing the
  null-rate gaps Phase 3 will need to handle.

**Key Decisions Made:**
- Excluded AR from the export per PROGRESS.md recommendation (avg 2.2 horses/race due to the
  `nro`-reuse issue across same-day venues at the same track). Distance + surface in the dedup
  key don't disambiguate enough to recover the true field. Including AR would teach models that
  2-horse fields are common, which is false.
- Did not regenerate `field_size` at export time — left it as raw column value (UK populated,
  JP/HK NULL). Phase 3 can compute it via `df.groupby(['race_date','track_code','race_number']).size()`.
  Pushing that derivation downstream keeps the export contract simple.
- Did not attempt to derive a synthetic UK `post_position` or JP speed figure during export. Both
  are Phase 3 concerns — they require modelling decisions (which feature engineering approach?
  fall back vs. impute vs. exclude?) that belong in the bootstrap script, not in the SQL export.

**Tests Status:**
- 62 Phase 0 tests passing in 2.45s. No new tests added — the `--jurisdictions` flag is a thin
  CLI passthrough; the SQL parameterization path is exercised end-to-end by the live export.

---

### Session: 2026-05-12 — Bug-fix sweep + master DB populated (2.6M results loaded)

**Context:**
The 2026-05-11 (c) auto-ingest run (`auto_ingest_run1.json`) had loaded **0 rows out of 1.5M+ downloaded**.
Six datasets registered in the DB had `row_count_ingested = NULL` because every row failed at map+clean
or quality_gate. This session diagnosed root causes, fixed them, then hand-mapped the highest-quality
datasets to populate the DB end-to-end.

**Bugs found and fixed (in priority order):**

1. **`race_number` Pydantic-required but missing in most non-US datasets.** Many datasets identify races
   by `race_id` (string), `card_id`, or post-time rather than an explicit numeric race number. Fix:
   made `race_number` `Optional` in `CanonicalRace`, added it as a load-required hard failure in
   `quality_gate` (zero score if None — same justification as distance/surface/jurisdiction per
   ADR-011 / ADR-019).

2. **Heuristic regex `(horse|name)` matched `race_name` / `Course name` before `horse_name`.**
   First-match-wins with greedy `name` was catastrophically wrong. Fix: replaced single regex per field
   with `_FieldSpec(exact=[priority list], fuzzy=optional, blocklist=[])`. Tries exact-match (case-
   insensitive) first, then fuzzy with explicit blocklist for known-wrong matches like `post_position`
   / `course_name`. See ADR-018.

3. **`normalize_name()` stripped all non-ASCII characters.** `[^a-z0-9 ]` regex turned `ワクセイ`
   (JRA katakana) into empty string, which then hit "missing horse_name" + "duplicate horses within
   race: ['']" — JRA's 1.6M rows ALL failed. Fix: regex is now `[^\w ]` with `re.UNICODE`, preserving
   CJK / Cyrillic / accented Roman names while still stripping `'`, `-`, `.` from English names. See
   ADR-017.

4. **Quality_gate's race-level grouping disagreed with `race_dedup_key`.** quality_gate grouped by
   `(date, track, race_num)`, but the dedup_key includes `(date, track, race_num, distance, surface)`.
   Argentine venues that run turf + dirt cards in parallel reuse `nro` across distinct physical races.
   Quality_gate flagged these as "mixed distances" / "duplicate horses" violations even though they
   were correctly stored as separate races by load_to_db. Fix: quality_gate now groups by all 5
   dedup_key fields. The "mixed distances within race" check was removed (it's a no-op once you group
   by distance). See ADR-016.

5. **Auto-built field maps had `transformers: {}`.** Surfaces stayed as raw "Heavy"/"Good"/"GOOD TO
   FIRM"; odds stayed as fractional "5/2"; both broke the dedup_key. Fix: `build_heuristic_map` now
   auto-attaches default transformers (parse_odds_to_decimal, normalize_surface or uk_going_to_surface,
   normalize_condition or uk_going_to_condition, time_string_to_seconds, stones_to_lbs) based on what
   columns it found.

**New infrastructure added:**

- **`scripts/db/preprocessors.py`** — registry of multi-CSV merge functions. Datasets that ship as
  `races.csv` + `runs.csv` (joined on `race_id`) declare a `preprocess` field in their FIELD_MAPS
  entry; map_and_clean.py calls the named function instead of `_pick_primary_csv`. See ADR-015.
  Two preprocessors implemented: `gdaley_hkracing_merge`, `lantanacamara_hk_merge`.

- **6 new transformers** in `transformers.py`:
  - `time_string_to_minutes` — parses "HH:MM" into minutes-since-midnight (race_number proxy when
    only post-time is available, e.g. UK sheikhbarabas)
  - `extract_int` — pulls first contiguous integer from a string (for `RID1002-IE-05` → 1002)
  - `kg_to_lbs` — for JRA carried weights (in kg)
  - `jpn_surface_to_canonical` — maps `芝` → turf, `ダート` → dirt
  - `jpn_condition_to_canonical` — maps `良` / `稍重` / `重` / `不良` to fast/good/soft/heavy
  - `ar_surface_to_canonical` — maps Spanish `arena`/`cesped`/`sintetico` to dirt/turf/synthetic

**Hand-written field maps added (5 new):**
- `sheikhbarabas/horse-racing-results-uk-ireland-2005-to-2019` (UK/IE, ~744K rows; race_number
  synthesized from post-time via `time_string_to_minutes`)
- `gdaley/hkracing` (HK, ~80K rows; multi-CSV merge; only IDs for horse/jockey/trainer)
- `lantanacamara/hong-kong-horse-racing` (HK, ~30K rows; multi-CSV merge; real names)
- `takamotoki/jra-horse-racing-dataset` (JP, ~1.6M rows; Japanese column names)
- `felipetappata/thoroughbred-races-in-argentina` (AR, ~323K rows; Spanish conventions; uses
  `jockey_weight` not `weight` to avoid body-weight-vs-carried-weight contamination)

**Master DB final state:**

| Jurisdiction | Races   | Results    | Date range          | Source                      |
|--------------|---------|------------|---------------------|-----------------------------|
| JP           | 121,785 | 1,609,930  | 1986-01 → 2021-07   | takamotoki/jra              |
| AR           | 137,344 | 306,290    | 2016-06 → 2023-10   | felipetappata               |
| UK           |  69,135 | 601,405    | 2005-04 → 2019-12   | sheikhbarabas               |
| HK           |   8,703 | 108,659    | 1997-06 → 2017-07   | gdaley + lantanacamara      |
| **Total**    | **336,967** | **2,626,284** | **1986 → 2023** | **5 hand-mapped datasets**  |

Quality score distribution across all 2.6M results:
- 0.95+ (excellent): 130,231 (5%)
- 0.85-0.95: 1,965,861 (75%)
- 0.75-0.85: 71,170 (3%)
- 0.60-0.75 (minimum pass): 459,022 (17%)

**Auto-ingest #2 results (with bug-fixes):** 0 additional rows loaded. The remaining 21 low_score
datasets had column conventions the heuristic still couldn't infer; the 7 no_csv datasets were
non-CSV; the 4 evaluate-failed had encoding/parsing errors; the others (noqcks workouts, prashant111
tipster bets, mexwell HK dividends-only) are not actually race-results data despite scoring well at
the column-presence evaluator.

**Known caveats** (documented for Phase 3 model training):
- **Argentina has structural data-quality issues.** Avg field size of 2.2 is unrealistically low
  (real races ~10 horses) — the `nro` field is reused across multiple physical races at the same
  venue/date, and even with distance+surface in the dedup_key the resulting groups are fragmented.
  AR has zero odds data. Some `jockey_weight` values are contaminated with body weight (max 1318 lbs).
  Mean score 0.66. Recommend filtering AR out of model training until a venue-specific course
  identifier can be derived.
- **HK has two non-overlapping naming conventions.** gdaley uses numeric IDs ("3875") for horses
  + "ST"/"HV" tracks (1997-2005); lantanacamara uses real names ("DOUBLE DRAGON") + "Sha Tin" /
  "Happy Valley" (2014-2017). Won't cross-dedup, but date ranges don't overlap.
- **`tracks` table is empty.** load_to_db never populates it. Aspirational table for now.
- **JRA `purse(万円)` left raw** (units of 10,000 JPY rather than USD). quality_gate doesn't validate
  purse so this isn't blocking — the field is informational for now.

**Tests Status:**
- 234 tests passing in ~3.0s. The mixed-distances test was rewritten to assert the NEW behavior
  (different distances at the same nro = separate races, not a violation).

---

### Session: 2026-05-11 (c) — Auto-discovery ingestion + bug fixes

**Completed:**
- `scripts/db/auto_ingest.py` — Kaggle-keyword-driven bulk orchestrator. Searches Kaggle
  for configurable keywords (default: "horse racing", "horse bet", "horseracing", "horse
  race results", "thoroughbred", "horse betting"), deduplicates results across keywords,
  and runs each unique slug through the full pipeline:
  download → evaluate → map+clean → quality_gate → load. Skips slugs already present
  in the `datasets` table (idempotent). Emits a structured per-slug `SlugResult` with
  status codes: `loaded` / `already_ingested` / `low_score` / `needs_map` / `no_csv` /
  `error`.
- Two field-map modes: **strict** (default) processes only datasets with a hand-written
  entry in `field_maps.FIELD_MAPS`; for unregistered slugs it logs the actual CSV column
  names so you can write a new entry and re-run. **`--auto-map`** opt-in builds a
  synthetic field map at runtime via the heuristic regex matchers from
  `evaluate_dataset.py`. Heuristic mode is intentionally lossy and the user is told to
  inspect `rejected/reasons.jsonl` after each run.
- True `--dry-run` mode — discovery only, zero downloads, zero DB writes. Returns a
  preview list with Kaggle metadata (size, downloads, votes, last_updated, url,
  matched_keyword) plus `already_ingested` and `has_field_map` flags per slug.
- No automatic cleanup — staging/cleaned/parquet artifacts are retained on disk for
  audit (per user preference; trades disk space for debuggability).

**Bug fixes (surfaced by smoke tests + dry-run):**
- `evaluate_dataset._HEURISTICS` regex `\b` boundaries did not fire on snake_case
  columns. Python's `\b` is a transition between `\w` (which includes `_`) and `\W`,
  so `\b(date)\b` cannot match `race_date` (no boundary between `e` and `_`). Fixed
  by replacing `\b` with explicit lookarounds `(?<![a-zA-Z0-9])...(?![a-zA-Z0-9])`,
  which treat `_` as a separator. This was breaking heuristic mapping for the majority
  of real Kaggle datasets.
- Initial `auto_ingest.discover_slugs` mis-used Kaggle's `dataset_list(max_size=N)` —
  `max_size` is a **byte-size filter** (max bytes per dataset), not a result-count cap.
  My first dry-run returned only 1 result because I was filtering to datasets ≤ 20
  bytes. Fixed by replacing `max_size` with proper page-by-page iteration via
  `_search_paginated` until `max_per_keyword` results are accumulated.
- Initial `--dry-run` flag was a no-op at the CLI level — it still downloaded every
  dataset and wrote `datasets` rows to the DB before bailing. Restructured so dry-run
  short-circuits at the orchestration level (in `auto_ingest()`) and returns
  discovery-only results without entering the per-slug processing loop.

**Dry-run Findings:**
- 38 unique horse-racing datasets discovered; estimated ~3.4 GB total download
- **None of the 3 datasets currently registered in `field_maps.py`
  (`joebeachcapital/horse-racing`, `zygmunt/horse-racing-dataset`,
  `gdaley/horseracing-in-hk`) appeared in the search results** — they may have been
  renamed or removed. `gdaley/hkracing` did appear and is likely the renamed HK
  dataset. The DATA_PIPELINE.md §4 dataset catalog needs updating.
- High-quality candidates by community signal (downloads):
  `gdaley/hkracing` (9628 dl), `lantanacamara/hong-kong-horse-racing` (5429 dl),
  `hwaitt/horse-racing` (5091 dl), `takamotoki/jra-horse-racing-dataset` (3159 dl),
  `deltaromeo/horse-racing-results-ukireland-2015-2025` (2780 dl, 1.1 GB),
  `eonsky/betfair-sp` (1016 dl, 1.2 GB)
- Junk that will be auto-rejected:
  `seniruepasinghe/horse-racing-player-detection-yolo11` (image dataset → no_csv),
  `thedevastator/major-us-sports-venues-usage-and-affiliations`,
  `quantumgoat/predict-horse-price`, several "winners-only" tiny datasets

**Key Decisions Made:** (full rationale in DECISIONS.md ADR-013, ADR-014)
- Auto-ingest uses a **hybrid mode**: strict by default for data-quality safety,
  `--auto-map` opt-in for bulk-build velocity. Both modes leave the DB in a clean state.
- Dry-run is **discovery-only** (no downloads, no DB writes). Two tiers of "preview"
  rejected — single tier with explicit followup is simpler.
- Heuristic auto-mapping uses **column-name regex only**; no value sniffing for unit
  detection. Distance is assumed already in furlongs; odds assumed already decimal.
  Quality gate's range checks (2.0-20.0 furlongs, 1.0+ decimal odds) are the
  unit-mismatch safety net.

**Tests Status:**
- 233 tests passing in ~2.9s (172 Phase 1 + 61 Phase 0). No new tests for `auto_ingest.py`
  itself — it's an orchestration layer over fully-tested components, and stubbing the
  Kaggle API would be more cost than value.

---

### Session: 2026-05-11 (b) — Phase 0 master DB pipeline (code complete)

**Completed:**
- `pyproject.toml` — added Phase 0 deps: `kaggle>=1.6`, `pandas>=2.2`, `pyarrow>=15.0`
- `scripts/db/schema.sql` — 7 idempotent CREATE TABLE / CREATE INDEX statements
  (datasets, tracks, races, horses, jockeys, trainers, race_results) with UNIQUE(dedup_key)
  on every dedup target
- `scripts/db/constants.py` — paths, schema version, quality thresholds, source priority dict
- `scripts/db/setup_db.py` — runs schema.sql via `executescript`; idempotent (re-runs are no-ops)
- `scripts/db/dedup.py` — SHA-256 dedup keys per DATA_PIPELINE.md §3:
  `race_dedup_key` (track|date|race#|round(distance,2)|surface),
  `horse_dedup_key` (normalized_name|year|country, with "unknown" fallbacks),
  `person_dedup_key` (jockeys/trainers — normalized_name|jurisdiction),
  `result_dedup_key` (race_key|horse_key)
- `scripts/db/transformers.py` — full unit-conversion library (metres↔furlongs, UK distance
  parser supporting `1m4f 110y`, stones↔lbs, fractional/EVS odds, UK SP with favourite-marker
  stripping, surface/condition normalization including all-weather track recognition,
  time-string parser, fixed-rate FX). All transformers return None on garbage rather than
  raising — quality gate is the gatekeeper.
- `scripts/db/field_maps.py` — registry for joebeachcapital/horse-racing (US, active),
  zygmunt/horse-racing-dataset (UK, drafted), gdaley/horseracing-in-hk (HK, drafted).
  Value semantics: `"col"` = column ref, `{"const": v}` = literal, `None` = NULL
- `scripts/db/schemas.py` — Pydantic v2 canonical models: `CanonicalRace`, `CanonicalHorse`,
  `CanonicalPerson` (jockeys/trainers share shape), `CanonicalRaceResult` with nested
  race/horse/jockey/trainer + `to_parquet_dict()` flattener (parent prefixes for nested fields)
- `scripts/db/ingest_kaggle.py` — Kaggle download + `datasets` table registration with
  4-tier credential resolution: `--credentials` path → env vars → `~/.kaggle/kaggle.json` →
  `~/.kaggle/access_token` (JSON or raw key). Writes `_dataset_id` sidecar in staging dir
  so downstream scripts can pick up the FK without re-running.
- `scripts/db/evaluate_dataset.py` — DATA_PIPELINE.md §6 rubric (weighted score, threshold
  0.70). Heuristic regex column matching when no `--slug`; field-map lookup when slug given.
  Emits JSON report (score, field_coverage, jurisdiction_guess, warnings, date range,
  estimated races/results).
- `scripts/db/map_and_clean.py` — staging CSV → CanonicalRaceResult validation → parquet.
  Writes `cleaned/<slug>/all.parquet` (every row that passes Pydantic) plus
  `rejected_pydantic.jsonl` for diagnostics. Lenient — only rejects on hard schema
  failures; quality gate handles the rest.
- `scripts/db/quality_gate.py` — DATA_PIPELINE.md §8 scoring + cross-row race-level
  validation (multiple winners, mixed distances/surfaces, duplicate horses). Splits parquet
  into `accepted/all.parquet` and `rejected/all.parquet` + `reasons.jsonl`. Sets
  `data_quality_score` on every row before split.
- `scripts/db/load_to_db.py` — idempotent loader: dependency-ordered upserts
  (horses → jockeys → trainers → races → race_results) with `INSERT OR IGNORE` on
  every dedup_key. Updates `datasets.row_count_ingested/deduped/date_range_*` after each load.
- `scripts/db/dedup_report.py` — per-table totals + duplicate-key counts (must always be 0)
  + race_results quality-score distribution + dataset audit-trail summary.
- `scripts/db/export_training_data.py` — runs DATA_PIPELINE.md §12 SQL verbatim, parametrized
  by `--min-score` and `--min-field-size`, writes `data/exports/training_<YYYYMMDD>.parquet`.

**Test Suite (61 new tests, 233 total passing):**
- `tests/test_db/test_dedup.py` — 11 tests: hash stability across `date`/string inputs,
  case-insensitivity, distance rounding, distinguishing collisions, name normalization
- `tests/test_db/test_transformers.py` — 27 tests: every transformer's happy path,
  None-on-garbage contract, registry consistency, field_maps→transformer registry validation
- `tests/test_db/test_quality_gate.py` — 20 tests: every hard failure (parametrized),
  every soft penalty (exact deduction values), every range check, score clamping at 0,
  cross-row violations (multiple winners, mixed distances, duplicate horses)
- `tests/test_db/test_pipeline.py` — 3 end-to-end tests: synthetic CSV through full
  pipeline; quality_score persisted; dedup keys stable across re-runs (idempotency)

**Schema Adjustment:**
- `CanonicalRace.distance_furlongs` / `surface` / `jurisdiction` made `Optional` so
  map_and_clean produces all rows for the quality gate to score (rather than dropping
  rows with missing soft fields at Pydantic validation time). The quality gate then
  treats these as load-required hard failures (zero score) — they're NOT NULL in SQL
  and used in the race dedup key, so a row missing them is fundamentally unloadable.

**Key Decisions Made:**
- `scripts/db/` is fully standalone — no imports from `app/`. Per CLAUDE.md §11, this
  keeps the training pipeline runnable without standing up FastAPI.
- stdlib `sqlite3` instead of SQLAlchemy in `scripts/db/` — Phase 0 has no async
  requirements and the schema is small enough that ORM overhead is pure cost.
- Bootstrap pattern in every CLI script: `if __package__ in (None, ""): sys.path.insert(0, ...)`
  so all scripts work as both `python scripts/db/foo.py` and `python -m scripts.db.foo`.
- Field-map value `{"const": value}` syntax disambiguates literal values from column
  references — small deviation from DATA_PIPELINE.md §5 sketch (which inlined raw
  values) but cleaner because it avoids clobbering valid column names like `"beyer"`.
- Source priority logic (Equibase > Brisnet > DRF > Kaggle) is enforced implicitly via
  `INSERT OR IGNORE`: whichever dataset is loaded first wins. To override, drop the
  conflicting row from the lower-priority dataset and re-run. (Future improvement: add
  an explicit `source_priority` column and a per-table `MERGE`-style upsert.)
- `quality_gate.py` adds a load-required hard-failure on missing distance/surface/
  jurisdiction even though DATA_PIPELINE.md §8 calls them soft failures — these fields
  are required for the race dedup key + SQL load, so they cannot be soft.

**Not Started (Phase 0 remaining — runtime work, not code):**
- Live run against `joebeachcapital/horse-racing`: download via `ingest_kaggle.py`,
  evaluate, map+clean (will surface CSV column-name mismatches with the field map —
  these go straight back into `field_maps.py`), quality-gate, load.
- After load: confirm `dedup_report.py` shows zero duplicate keys and a sane score
  distribution; run `export_training_data.py` and inspect the parquet.

**Tests Status:**
- 233 tests passing in ~3.0s. Run with `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: 2026-05-11 — Phase 1 parser test suite

**Completed:**
- Package scaffolding: `app/__init__.py`, `app/schemas/__init__.py`, `app/services/__init__.py`,
  `app/services/pdf_parser/__init__.py`, `app/core/__init__.py`, `tests/__init__.py`,
  `tests/test_parser/__init__.py`
- `app/core/config.py` — Pydantic v2 `BaseSettings`-based `Settings` class with all Phase 1+5
  knobs (upload size, extraction strategy, bankroll, Kelly fraction, CVaR alpha, etc.)
- `pyproject.toml` — project metadata, runtime+dev deps, pytest config with `pythonpath = ["."]`
- Local `.venv` provisioned (Python 3.14.2) with: pydantic 2.13, pydantic-settings 2.14,
  structlog 25.5, pdfplumber 0.11.9, pypdf 6.11, pytest 9.0, reportlab 4.5
- `tests/test_parser/test_cleaner.py` — 102 tests covering every public function in `cleaner.py`
- `tests/test_parser/test_brisnet_parser.py` — 45 tests against synthetic Brisnet UP text:
  card structure, race header (number/date/distance/surface/condition/race-type/purse/claiming),
  horse entries (post/name/ML/jockey/trainer/weight/ml_implied_prob), PP lines (date order,
  speed figures, fractions, finish position, track code, days-since-prev), parse_confidence,
  multi-race input, degenerate inputs
- `tests/test_parser/test_extractor.py` — 25 tests: format detection signatures, parser
  dispatch, size guard, pdfplumber text extraction via reportlab-generated PDF fixture,
  ingest_pdf shape (timing, source_filename, page count, zero-race failure, corrupt bytes)

**Bug Fixes (found by writing tests):**
- `cleaner._FRACTION_MAP` was missing `1/16` and `3/16` — the most common US route fractions.
  Added 1/16, 3/16, 5/16, 7/16, 3/8, 7/8, 1/3, 2/3. `_DISTANCE_RE` alternation updated.
- `cleaner.normalize_text` was collapsing multi-space runs to a single space, which broke
  `BrisnetParser._RE_HORSE_LINE`'s `\s{2,}` column separator (every horse line would
  silently fail to match). Behavior changed: 3+ space runs cap at 2 spaces, preserving
  columnar alignment. Added `collapse_whitespace()` for callers that genuinely want
  single-space output (used by `clean_name`).
- `brisnet_parser._RE_DISTANCE` accepted only `\d/\d` fractions (single digit), so it
  misextracted "1 1/16 Miles" as "16 Miles" → out-of-range distance → ValidationError.
  Widened to `\d{1,2}/\d{1,2}` and made leading whitespace optional.
- `brisnet_parser._RE_CLAIMING` pattern `Clm(?:aiming)?` was buggy: "Claiming" starts with
  `Cla`, not `Clm`, so the optional `(?:aiming)?` group was meaningless. The regex never
  matched the full word — only the abbreviation in PP lines. Replaced with
  `Cl(?:aiming|m)\b` plus explicit `\$?` so dollar signs aren't required.
- `BrisnetParser._parse_race_header` would crash with Pydantic ValidationError when distance
  couldn't be extracted (default `0.0` violates `ge=2.0`). Now returns `None` when distance
  is invalid; `_parse_race_block` skips the race.
- `extractor.ingest_pdf` was failing to populate `processing_ms` on early-return failure
  paths (size guard, extraction error, parse exception). Now always set via a closure helper.

**In Progress:**
- None — Phase 1 parser milestone complete and verified

**Not Started (Phase 1 remaining):**
- `app/core/logging.py` (structlog setup — currently each module calls `structlog.get_logger`
  directly with default config; explicit init lets us flip JSON vs. pretty by environment)
- `app/main.py` (FastAPI app factory)
- `app/api/v1/ingest.py` (upload endpoint; needs `run_in_executor` wrap on `ingest_pdf`)
- `app/db/models.py`, `app/db/session.py` (SQLAlchemy async; storage for ingested cards)
- `app/services/pdf_parser/equibase_parser.py` (currently falls back to BrisnetParser)
- `.env.example`
- Real-PDF validation: load an actual Brisnet UP card from a known source, sanity-check
  parse output, tune regexes to whatever shape pdfplumber actually produces

**Key Decisions Made:**
- Repo layout: keep current flat `app/` at root (no `backend/` wrapper). CLAUDE.md's
  documented `backend/` layout is aspirational — migration deferred to avoid mid-phase churn.
- Test scope split: `test_brisnet_parser.py` tests the parser against pre-extracted text
  (deterministic). `test_extractor.py` asserts only on pipeline shape for end-to-end PDF
  bytes, NOT on full parse correctness — because pdfplumber's `extract_text(x_tolerance=3,
  y_tolerance=3)` may collapse columns differently than our hand-crafted fixture, and we
  want parser tests to remain stable independent of pdfplumber version drift.
- `normalize_text` semantics changed from "single-space collapse" to "cap excess spaces at
  two" — preserves columnar gaps for downstream regex parsers. Documented in cleaner.py.
- Distance is treated as a load-bearing field for race headers: missing distance → race
  skipped from the card. Other missing fields (purse, claiming price, race type) degrade
  to None / UNKNOWN and the race still parses with lower confidence.

**Known Issues / Tech Debt:**
- `_RE_RACE_HEADER` uses `RACE\s+\d` which can false-positive on horse names like
  "DISGRACE 1" (contains "RACE 1" as substring). Need `\bRACE\b` anchor. Not blocking
  for current synthetic tests, but should fix before real-PDF validation.
- `extract_text_from_pdf` docstring promises pdfplumber layout mode then text mode then
  pypdf fallback, but the implementation only passes `x_tolerance`/`y_tolerance` (no
  `layout=True`). This may matter for real Brisnet PDFs where columns are spaced widely.
- Source-of-truth docs `Horse_Racing_System_Master_Reference.md` and
  `Horse_Racing_Betting_System_Research.pdf` referenced in CLAUDE.md do not exist in
  the repo. Working from CLAUDE.md + DECISIONS.md only.

**Blockers:**
- None

**Tests Status:**
- 172 tests passing in ~1.1s. Run with `.venv/Scripts/python.exe -m pytest tests/ -q`.

---

### Session: [PREVIOUS — pre-test-suite scaffolding]

**Completed:**
- `app/schemas/race.py` — all Pydantic v2 schemas:
  `PastPerformanceLine`, `HorseEntry`, `RaceCard`, `ParsedRace`,
  `RaceHeader`, `IngestionResult`, `IngestionStatus`
- `app/services/pdf_parser/cleaner.py` — full normalization pipeline:
  `normalize_text`, `clean_name`, `parse_odds_to_decimal`,
  `parse_distance_to_furlongs`, `parse_time_to_seconds`,
  `parse_surface`, `parse_condition`, `parse_race_type`,
  `extract_first_number`, `extract_claiming_price`
- `app/services/pdf_parser/extractor.py` — orchestrator:
  three-pass extraction (pdfplumber layout → text → pypdf),
  format detection, parser dispatch, `ingest_pdf` entry point
- `app/services/pdf_parser/brisnet_parser.py` — Brisnet UP parser:
  page segmentation, race header parsing, horse entry parsing,
  PP line extraction (full regex + positional fallback),
  parse confidence scoring

**Key Decisions Made:**
- Brisnet UP is the Phase 1 primary format; DRF and Equibase are stubs that fall back to BrisnetParser
- Three-pass PDF extraction: pdfplumber layout → pdfplumber text → pypdf character-level
- `parse_confidence` is a weighted composite: 40% header completeness + 60% PP coverage
- `race_number` in `PastPerformanceLine` is set to 1 as a placeholder (not reliably extractable from Brisnet line format)
- Odds stored as decimal throughout (fractional strings converted on ingest)

---

## Phase Completion Checklist

### Phase 0: Master Training Database
- [x] `scripts/db/schema.sql`
- [x] `scripts/db/constants.py`
- [x] `scripts/db/setup_db.py`
- [x] `scripts/db/dedup.py`
- [x] `scripts/db/transformers.py`
- [x] `scripts/db/field_maps.py`
- [x] `scripts/db/schemas.py`
- [x] `scripts/db/ingest_kaggle.py`
- [x] `scripts/db/evaluate_dataset.py`
- [x] `scripts/db/map_and_clean.py`
- [x] `scripts/db/quality_gate.py`
- [x] `scripts/db/load_to_db.py`
- [x] `scripts/db/dedup_report.py`
- [x] `scripts/db/export_training_data.py`
- [x] `scripts/db/auto_ingest.py` (Kaggle keyword-driven bulk orchestrator)
- [x] `scripts/db/preprocessors.py` (multi-CSV merge hook for map_and_clean)
- [x] `tests/test_db/test_dedup.py` (11 tests)
- [x] `tests/test_db/test_transformers.py` (27 tests)
- [x] `tests/test_db/test_quality_gate.py` (21 tests)
- [x] `tests/test_db/test_pipeline.py` (3 end-to-end tests)
- [x] **Live run: master DB populated with 2.6M results across UK / HK / JP / AR (1986-2023)**
- [x] **ML training parquet exported: `data/exports/training_20260512.parquet` (2.3M rows, UK+HK+JP)**
- [x] `scripts/db/backfill_tracks.py` + tests — `tracks` table now holds 100 distinct (track_code, jurisdiction) combos
- [ ] Future: derive a venue-specific course identifier for AR to fix the over-split races issue
- [ ] Future: derive JP synthetic speed figure from `(distance, fraction_finish_sec, condition)` so JP rows can train speed-dependent models

### Phase 1: PDF Ingestion
- [x] `app/core/config.py`
- [x] `app/core/logging.py`
- [x] `app/main.py`
- [x] `app/api/v1/ingest.py`
- [x] `app/db/models.py`
- [x] `app/db/session.py`
- [x] `app/db/persistence.py`
- [x] `app/schemas/race.py`
- [x] `app/services/pdf_parser/cleaner.py`
- [x] `app/services/pdf_parser/extractor.py`
- [x] `app/services/pdf_parser/brisnet_parser.py`
- [x] `app/services/pdf_parser/equibase_parser.py` (subclass placeholder)
- [x] `tests/test_parser/test_cleaner.py` (102 tests)
- [x] `tests/test_parser/test_brisnet_parser.py` (45 tests)
- [x] `tests/test_parser/test_extractor.py` (25 tests)
- [x] `tests/test_api/test_ingest.py` (6 tests)
- [x] `pyproject.toml`
- [x] `.env.example`
- [ ] Future: real-PDF validation pass against an actual Brisnet UP card; tune `_RE_RACE_HEADER` (`\bRACE\b` anchor) and `extract_text_from_pdf` (real `layout=True` mode)
- [ ] Future: DRF parser implementation (currently falls back to BrisnetParser)

### Phase 2: Feature Engineering
- [x] `app/services/feature_engineering/engine.py` (FeatureEngine orchestrator)
- [x] `app/services/feature_engineering/speed_features.py` (EWM α=0.4 + field-relative z/rank/pct)
- [x] `app/services/feature_engineering/pace_features.py` (shape, fraction ratios, pressure index)
- [x] `app/services/feature_engineering/class_features.py` (claiming/purse delta + race-type change)
- [x] `app/services/feature_engineering/connections.py` (jockey continuity + win-rate proxy)
- [x] `app/services/feature_engineering/layoff.py` (parametric exp-decay fitness curve)
- [x] `tests/test_features/test_layoff.py` (12 tests)
- [x] `tests/test_features/test_speed_features.py` (13 tests)
- [x] `tests/test_features/test_pace_features.py` (13 tests)
- [x] `tests/test_features/test_class_features.py` (8 tests)
- [x] `tests/test_features/test_connections.py` (8 tests)
- [x] `tests/test_features/test_engine.py` (9 integration tests)
- [x] `tests/test_features/_fixtures.py` (shared synthetic-card builders)

### Phase 3: Model Layer
- [x] `scripts/bootstrap_models.py` (orchestrator with --sample-frac and --run-name)
- [x] `app/services/models/training_data.py` (leakage-free feature prep + time split)
- [x] `app/services/models/speed_form.py` (LightGBM Layer 1a — trained, val_top1=0.257)
- [x] `app/services/models/pace_scenario.py` (STUB — needs fractional time columns)
- [x] `app/services/models/sequence.py` (STUB — needs PyTorch + globally-unique horse_id)
- [x] `app/services/models/connections.py` (Empirical-Bayes Layer 1d — trained, 50K pairs)
- [x] `app/services/models/market.py` (Isotonic odds calibration Layer 1e — trained)
- [x] `app/services/models/meta_learner.py` (LightGBM stacker — trained, val_top1=0.341)
- [x] `tests/test_models/test_training_data.py` (15 tests)
- [x] `tests/test_models/test_speed_form.py` (10 tests)
- [x] `tests/test_models/test_connections.py` (7 tests)
- [x] `tests/test_models/test_market.py` (5 tests)
- [x] `tests/test_models/_synth.py` (shared synthetic data generator)
- [x] **Live baseline trained: `models/baseline_full/` — Speed/Form val_top1=0.257, Meta val_top1=0.341**
- [ ] Future: train pace_scenario once fractional times are in the parquet
- [ ] Future: train sequence transformer once PyTorch + horse dedup_key in export
- [ ] Future: expose `horses.dedup_key` in `export_training_data.py` to fix horse collisions

### Phase 4: Calibration + Ordering
- [x] `app/services/calibration/calibrator.py` (Platt + isotonic + auto-selector)
- [ ] `app/services/calibration/drift.py` (CUSUM change-point — deferred)
- [x] `app/services/ordering/plackett_luce.py` (analytical exotics + MLE)
- [ ] `app/services/ordering/stern.py` (Gamma ordering — deferred)
- [ ] `app/services/ordering/copula.py` (pace-correlated ordering — deferred)
- [x] `scripts/validate_calibration.py` (CLI + importable evaluator + reliability diagram)
- [x] `tests/test_calibration/test_calibrator.py` (21 tests)
- [x] `tests/test_calibration/test_validate_calibration.py` (3 smoke tests)
- [x] `tests/test_ordering/test_plackett_luce.py` (21 tests)
- [x] **Smoke validation: 5% sample → meta-learner ECE 0.121 → 0.011 via isotonic**
- [ ] Future: full-parquet validate_calibration run (deferred for compute budget)
- [ ] Future: cross-validated calibrator selection (current selects on fit slice)

### Phase 5: EV Engine + Portfolio
- [ ] `app/services/ev_engine/calculator.py`
- [ ] `app/services/ev_engine/market_impact.py`
- [ ] `app/services/portfolio/optimizer.py`
- [ ] `app/services/portfolio/sizing.py`
- [ ] `app/api/v1/analyze.py`
- [ ] `app/api/v1/portfolio.py`
- [ ] `tests/test_portfolio/`

### Phase 6: Frontend
- [ ] Next.js project scaffold
- [ ] Upload flow
- [ ] Race card viewer
- [ ] Probability visualization
- [ ] Bet execution ticket
