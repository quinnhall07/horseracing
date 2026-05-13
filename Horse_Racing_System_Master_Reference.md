# Horse Racing Betting System — Master Reference
*All great ideas consolidated from research sessions, ready for build-out*

---

## SECTION 1: DATA ACQUISITION

### Primary Data Sources
- **Kaggle** — free historical datasets (UK, Hong Kong, US tracks); good for prototyping
- **Equibase** (US) — industry standard; paid API, free basic results; use for production
- **Racing Post** (UK/Ireland) — scrapable historical results
- **Brisnet UP (Ultimate Past Performances)** — cheap DRF files with rich feature depth

### Essential Fields to Collect
- Horse ID, jockey, trainer, owner
- Track, distance, surface (dirt/turf/synthetic), weather, track condition
- Post position, weight carried
- Past performance lines: speed figures, finishing positions, split times
- Morning line odds, early pool odds, final post-time odds
- Payout amounts for every bet type

---

## SECTION 2: EXPLORATORY DATA ANALYSIS

Run these analyses before touching any model:

- **Favorite bias baseline** — favorites win ~33% of races; this is your benchmark to beat. Confirm it on your data first.
- **Post position bias** — inner posts favor shorter distances, especially on dirt; varies heavily by track. Map this per track × distance × surface.
- **Jockey/trainer statistics** — win%, ROI by combination, performance by surface and distance
- **Speed figure trends** — is each horse improving, peaking, or declining?
- **Class change signals** — drops/rises in claiming price; one of the strongest raw signals
- **Pace analysis** — field pace scenarios (hot/honest/slow); too many front-runners = closer advantage
- **Recency curves** — days since last race vs. performance; model the fitness decay curve
- **Odds calibration check** — are crowd implied probabilities well-calibrated? Find where they're systematically wrong. That gap is where edge lives.

---

## SECTION 3: FEATURE ENGINEERING
*This is 80% of the work. Raw data must become model-ready features.*

### Static Features (per horse, don't change race-to-race)
- Horse ID **embeddings** (learned, not one-hot)
- Jockey, trainer, owner **embeddings**
- Breeding/pedigree **embeddings** — sire/dam lines encode surface and distance affinity

### Dynamic Features (computed fresh per race)
- Rolling speed figures — **exponentially weighted** (recent races matter more)
- Pace shape: early speed fraction, sustained pace fraction, closing fraction
- Class trajectory: normalized claiming price delta over last N races
- **Layoff/fitness curve**: parametric model of form decay over days since last race
- **Field-relative features**: each horse's speed figure *ranked within this specific field*, not absolute
- Jockey × trainer **interaction rate** (not just individual stats — the combination matters)
- Track-specific bias features: post position win% at this exact track + distance + surface

### Key Principle
> Most features should be *relative to the field*, not absolute. A 95 Beyer figure means nothing without knowing the field average is 91 or 102.

---

## SECTION 4: MODEL ARCHITECTURE — LAYER BY LAYER

### Philosophy
An **ensemble of specialized expert models** whose outputs feed a unified probability calibrator, which then feeds a portfolio optimizer. No single model sees the whole picture.

---

### Layer 1: Five Specialized Sub-Models

**2a. Speed/Form Model**
- Type: Gradient Boosted Trees (LightGBM or XGBoost)
- Input: speed figures, pace fractions, class changes, layoff features
- Output: raw win probability from past performance
- Why GBT: tabular racing data is where gradient boosting dominates

**2b. Pace Scenario Model**
- Type: smaller GBT or rule-based model
- Input: each horse's early speed profile, full field composition
- Output: probability of each pace scenario (hot/honest/slow) + each horse's adjusted probability under each
- Key insight: captures "lone front-runner gets a soft lead" — a massive edge the public systematically misses

**2c. Sequence Model (Per-Horse History)**
- Type: **Transformer encoder** (not LSTM — attention handles variable-length gaps between races better)
- Input: last 8–10 race performance lines as a sequence; each race is a token with: position, speed figure, distance, surface, class level
- Output: embedding representing current form trajectory
- Captures: "horse always improves second start back," "fades in back-to-back races," etc.

**2d. Jockey/Trainer/Connections Model**
- Type: **Bayesian hierarchical model**
- Input: jockey × trainer × track × surface × distance combinations; recent hot/cold streaks
- Output: connections adjustment factor
- Why Bayesian: partial pooling across sparse combinations prevents overfitting on small-sample jockey/trainer pairs
- Extension: model **trainer intent as a latent variable** — class drops after poor finishes vs. class drops after layoffs are fundamentally different signals. The *pattern of entries* is itself informative (form manipulation is common).

**2e. Market Model**
- Type: time-series / signal model
- Input: morning line odds, early pool odds movement, final post-time odds
- Output: crowd-implied probability + **smart money signal** (sharp vs. public money via odds movement pattern)
- Key insight: the crowd is a weak learner but not worthless. When your model disagrees with sharp money movement, that's a high-value signal.

---

### Layer 2: Meta-Learner (Stacking)

All sub-model outputs feed into a **stacked ensemble meta-learner** — typically a small neural net or regularized logistic regression — that learns optimal per-context weighting of each expert (e.g., weight the pace model more in sprints, sequence model more in routes).

**Critical:** **Orthogonalize inputs** before feeding the meta-learner. Speed figures already incorporate pace, so the Speed Model and Pace Model have correlated inputs. Residualize each sub-model's features against the others to ensure each expert contributes genuinely independent information and avoid double-counting.

---

### Layer 3: Calibration + Normalization

- Raw model outputs are scores, not probabilities. **Calibrate** using Platt scaling or isotonic regression trained on held-out data so that when the model says 30%, it actually wins ~30% of the time.
- After calibration: normalize probabilities to sum to 1 across the field using **softmax with a tunable temperature parameter**.
- **Online calibration**: use a sliding window of the last N races for calibration updates, *separate* from the main model retraining cycle. Calibration should update more frequently than the underlying models.
- Monitor for **calibration drift** — if your 40% horses start winning 30%, something has changed in the world.

---

### Layer 4: Ordering Probability Model (Replacing Harville)

**Do NOT use raw Harville for exotic bet probabilities.** Harville has two fundamental flaws:
1. **Independence of Irrelevant Alternatives (IIA)** — assumes remaining horses' relative probabilities stay proportional after a horse wins. False: if an elite front-runner wins, closers are more likely to place than other front-runners.
2. **Systematic bias** — overestimates place/show probability of favorites; underestimates it for longshots. Measurable and consistent.

**Use Plackett-Luce instead** (or Stern model as the theoretically ideal):
- Treats finishing order as sequential choices, each proportional to a "strength" parameter
- Parameters fit via maximum likelihood on historical finishing orders
- Handles IIA violation better than Harville
- Can incorporate model covariates directly into the strength parameter
- Also known as rank-ordered logit / exploded logit — efficient fitting algorithms exist

**Most sophisticated option: Copula-based Pace Coupling Model**
- Explicitly models that front-runners are positively correlated in bad outcomes
- If one front-runner wins on a slow pace, others are also more likely to place
- A copula over finishing positions captures these dependencies
- Research-level, but the largest remaining source of exotic probability error that no commercial system gets right

---

### Layer 5: EV Engine

For every bet type across every race:
1. Compute field win probabilities from Layer 3
2. Apply Plackett-Luce to enumerate all relevant permutations (Win, Exacta, Trifecta, Pick N)
3. For each combination compute:
   - Your implied probability (from the ordering model)
   - Market implied probability (from actual payout odds, adjusted for track take)
   - **Edge** = your probability − market probability
   - **Expected Value** = edge × net payout
4. Flag all bets where edge exceeds the minimum threshold (tune empirically; typically 5–10%)

**Market Impact Modeling (critical for larger bet sizes):**
In pari-mutuel wagering, your own bets move the odds. Heavy betting partially destroys your own edge. Model your price impact: for a given pool size, estimate how much your bet shifts the payout, and only bet up to the point where the *marginal* bet is still +EV. Larger exotic pools (Pick 6 carryovers) have the most capacity before impact matters.

---

### Layer 6: Portfolio Optimizer

Do not use Kelly per bet in isolation — treat the entire day's card as a **portfolio optimization problem**.

- **Why:** betting Exacta 1→2 and Exacta 1→3 are correlated (both lose if horse 1 loses). Naive Kelly ignores this.
- Use **CVaR (Conditional Value at Risk) optimizer** or Mean-Variance Optimization that accounts for bet correlations
- Constraints: max exposure per race, max total daily drawdown, max bet as % of bankroll
- Output: optimal bet allocation vector across all flagged opportunities

**Bet Sizing Formula (Fractional Kelly):**
```
Bet fraction = (edge × odds - (1 - edge)) / odds
```
- Use **1/4 Kelly** — full Kelly is mathematically optimal but psychologically and practically ruinous
- Set a minimum edge threshold before any bet is placed
- Cap individual bet size regardless of Kelly output

**Pareto Logic:** ~20% of races produce ~80% of profit opportunity. The system identifies and concentrates on those, passing on the rest entirely.

---

### Layer 7: Feedback + Online Learning

After each race card:
- Log all predictions vs. outcomes
- Track calibration drift continuously
- Retrain sub-models on a **rolling window** (e.g., last 3 years, dropping oldest)
- Monitor for **concept drift**: track resurfacing, medication rule changes, jockey aging, trainer strategy shifts, weather patterns
- Implement **change point detection** (CUSUM or Bayesian change point models) — automatically flags when calibration has drifted, triggers retraining or reduces bet sizing until re-validated

---

## SECTION 5: BET TYPE STRATEGY

| Bet Type | Typical Track Take | Notes |
|---|---|---|
| Win | ~17% | Easiest to model; most liquid; start here |
| Place/Show | ~17–19% | Lower variance, lower ceiling |
| Exacta | ~19–22% | Requires top-2 ordering probability |
| Trifecta | ~25% | High take, but huge payouts create +EV spots |
| Pick 3/4/6 | ~25–30% | Carryover pools can create massive +EV |

**Strategy:**
- Start with Win bets only — most transparent, easiest to validate
- Once Win model is validated, derive all exotic probabilities automatically via Plackett-Luce
- Exotics are where the real money is: the public systematically overbets favorites in exotics, creating longshot combination value
- Pick 6 carryover pools specifically offer the largest pool sizes and therefore the most market impact capacity

**Cross-Race Correlation (Pick 3/4/6):**
Multiplying win probabilities across races assumes independence — but races on the same card share weather and track condition. If the track is playing deep and tiring, it affects all races.
- Model a **shared latent track state variable** per card (hierarchical model where card-level condition is a random effect)

---

## SECTION 6: KNOWN DRAWBACKS AND MITIGATIONS

| Drawback | Mitigation |
|---|---|
| Harville IIA violation | Replace with Plackett-Luce or Stern model |
| Harville longshot bias | Plackett-Luce + empirical correction parameters |
| Market impact destroying edge | Model price impact per pool size; bet to marginal +EV only |
| Sub-model signal overlap | Orthogonalize inputs before meta-learner |
| Sparse track/distance combinations | Hierarchical Bayes with partial pooling |
| Non-stationarity (rules, surfaces, meds) | Rolling retrain + change point detection |
| Trainer intent / form manipulation | Latent variable model for trainer entry patterns |
| Calibration drift | Separate online calibration on sliding window; update frequently |
| Cross-race correlation in multi-leg bets | Shared latent track state variable per card |

---

## SECTION 7: COMPLETE SYSTEM ARCHITECTURE

```
Raw Data (past performances, odds, track conditions)
          │
          ▼
Change Point Detector → triggers retraining if drift detected
          │
          ▼
Feature Engineering
(relative features, embeddings, pace construction, orthogonalized per domain)
          │
    ┌─────┴────────────────────────────────┐
    │                                      │
Speed/Form   Pace Scenario   Sequence   Jockey/Trainer   Market
  Model        Model        Transformer  Hierarchical    Model
 (GBT)        (GBT)          (Attn)     Bayes + Intent  (Smart $)
    │                                      │
    └──────────────────┬───────────────────┘
                       ▼
          Meta-Learner (Stacking, orthogonalized inputs)
                       │
                       ▼
          Online Calibration Layer
          (sliding window, updates frequently)
                       │
                       ▼
          Plackett-Luce / Copula Ordering Model
          (replaces Harville for all exotic probabilities)
                       │
                       ▼
          EV Engine
          (market impact adjusted, edge threshold filter)
                       │
                       ▼
          Portfolio Optimizer
          (CVaR, correlated bets, bankroll constraints)
                       │
                       ▼
          Bet Allocations
                       │
                       ▼
          Outcome Logging + Drift Detection + Retraining
```

---

## SECTION 8: IMPLEMENTATION PRIORITY ORDER

Build in this sequence — each step is independently valuable:

1. **Data pipeline** — acquire, clean, store historical data (Equibase for US production)
2. **Baseline** — does betting the morning-line favorite beat the market long-term? (Usually no — establish this as your floor)
3. **Win probability model** — GBT with field-relative features; time-based validation split
4. **Calibration layer** — Platt scaling on held-out data; verify 40% = 40%
5. **Fractional Kelly on Win bets** — validate edge calculation and sizing logic
6. **Replace Harville with Plackett-Luce** — biggest accuracy gain for exotic EV, moderate effort
7. **Extend to Exacta/Trifecta** — automatic from Plackett-Luce; no new model needed
8. **Market impact modeling** — prevents destroying your own edge as bet sizes grow
9. **Change point detection** — guards against catastrophic undetected drift
10. **Copula-based pace coupling** — most complex, but largest remaining exotic probability error source
11. **Online calibration loop** — continuous improvement, relatively easy to implement

---

## SECTION 9: KEY ACADEMIC REFERENCES

| Paper | Why It Matters |
|---|---|
| **Benter (1994)** — "Computer Based Horse Race Handicapping and Wagering Systems" | The blueprint. Logit-fusion of fundamental model + public odds. Bill Benter made ~$1B with this. |
| **Harville (1973)** — JASA 68, pp. 312–316 | Original ordering probability formula. The baseline to replace. |
| **Henery (1981)** — JRSS Series B | First major Harville correction; normal running time distribution |
| **Stern (1990)** — JASA 85, pp. 558–564 | Gamma distribution model; most theoretically sound ordering model |
| **Lo, Bacon-Shone & Busche (1994)** | Empirical proof of Harville's systematic bias; longshots underestimated |
| **Hausch, Ziemba & Rubinstein (1981)** — Management Science 27(12) | Foundational market efficiency paper; proves where edge must come from |
| **Hausch & Ziemba (1985)** — Management Science 31 | Extends to exotic bets + Kelly for multiple simultaneous wagers |
| **Kelly (1956)** — Bell System Technical Journal | Original Kelly criterion derivation |
| **Thorp (2006)** — Handbook of Asset & Liability Management | Practical Kelly: fractional Kelly, growth vs. ruin tradeoffs |
| **Borowski & Chlebus (2021)** — U. Warsaw Working Paper | First ML comparison for Win + Quinella simultaneously |
| **Bolton & Chapman (1986)** — Management Science | Seminal paper that directly inspired Benter's conditional logit approach |
| **Hausch, Lo & Ziemba (Eds.) (1994)** — *Efficiency of Racetrack Betting Markets* | The canonical book; contains Benter + all seminal statistical papers |

---

## SECTION 10: THE TWO GENUINE EDGES

The two components that almost no public or commercial system implements correctly — these are the real competitive moats:

1. **Copula-based pace coupling model** — correctly modeling correlated finishing order probabilities for exotics
2. **Market impact modeling** — knowing exactly how much of your own edge you destroy with each dollar bet, and optimizing to the marginal +EV point

Everything else is table stakes. These two are the separators.
