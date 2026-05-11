# DECISIONS.md — Architectural Decision Log

Records every significant design decision, why it was made, and what was rejected.
Add entries whenever a non-obvious choice is made. Future sessions should consult
this before reconsidering any of these decisions.

---

## Decision Format

**ID:** ADR-XXX
**Date:** YYYY-MM-DD
**Status:** Accepted | Superseded by ADR-XXX
**Context:** What problem we were solving
**Decision:** What we chose
**Rationale:** Why
**Rejected Alternatives:** What we considered and discarded

---

## ADR-001: Harville Rejected for All Exotic Ordering

**Date:** Project inception
**Status:** Accepted — permanent

**Context:**
We need to compute P(exacta), P(trifecta), P(pick N) from win probabilities.
Harville (1973) is the simplest approach and is universally available in open-source libraries.

**Decision:**
Harville is prohibited throughout the codebase. Plackett-Luce is the minimum standard.
Stern (Gamma distribution) is preferred for single-race exotics. Copula-based model
is the Phase 4 target for correlated finishing orders.

**Rationale:**
Harville has two documented, systematic biases:
1. IIA violation — assumes remaining horses' relative probabilities stay proportional after a
   horse wins. False when pace style creates correlated outcomes (e.g., front-runner wins
   on slow pace → other front-runners more likely to place than closers).
2. Longshot bias — overestimates place/show probability of favorites; underestimates for
   longshots. Lo, Bacon-Shone & Busche (1994) proved this empirically across tens of
   thousands of races.
Using Harville would systematically misprice exotic EV and generate false +EV signals
on favorite-heavy combinations.

**Rejected Alternatives:**
- Harville — biased; rejected
- Raw win prob multiplication — even worse than Harville; treats positions as independent

---

## ADR-002: 1/4 Kelly, Not Full Kelly or Fixed Fraction

**Date:** Project inception
**Status:** Accepted — permanent

**Context:**
We need a bet sizing rule. Options: fixed fraction, full Kelly, fractional Kelly.

**Decision:**
1/4 Kelly universally. Formula:
  `bet_fraction = max(0, (edge * odds - (1 - edge)) / odds) * 0.25`
Hard cap: no single bet exceeds 3% of bankroll regardless of Kelly output.

**Rationale:**
Full Kelly is mathematically optimal for asymptotic growth but:
- Requires perfectly calibrated probabilities (we don't have that)
- Produces catastrophic drawdowns on probability estimation errors
- Is psychologically unmanageable
1/4 Kelly trades ~15% of asymptotic growth for ~75% reduction in drawdown severity
(Thorp 2006). The hard cap guards against Kelly blowup on high-edge exotic combinations
where the formula can output very large fractions.

**Rejected Alternatives:**
- Full Kelly — rejected (drawdown risk on miscalibration)
- 1/2 Kelly — considered; 1/4 chosen for competition context where capital preservation matters
- Fixed 1% per bet — rejected (ignores edge magnitude; leaves alpha on the table)

---

## ADR-003: Time-Based Train/Validation Split

**Date:** Project inception
**Status:** Accepted — permanent

**Context:**
How to split historical data for model training and validation.

**Decision:**
Always split chronologically. Train on [start, cutoff_date), validate on [cutoff_date, end].
Cutoff is typically 80% of the date range. Never use random split.

**Rationale:**
Random splitting leaks future information into training data. A horse's future race
outcomes are partially determined by factors that appear in its future PP lines. Random
split would produce artificially inflated validation metrics that do not generalize.
Time-based split simulates real deployment: the model always predicts on data it has
never seen in a future date.

**Rejected Alternatives:**
- Random 80/20 split — rejected (future leakage)
- K-fold cross-validation — rejected for same reason; also computationally expensive
  given the ordering of PP data

---

## ADR-004: Field-Relative Features Over Absolute Features

**Date:** Project inception
**Status:** Accepted — permanent

**Context:**
Whether to use raw speed figures (e.g., Beyer 95) or field-normalized features.

**Decision:**
All predictive features must be computed relative to the specific field in each race.
For a given race, compute: rank within field, z-score within field, percentile within
field. Raw absolute values may be stored for diagnostics but must never be primary
model inputs.

**Rationale:**
A Beyer of 95 against a field averaging 91 is a dominant horse. The same figure against
a field averaging 102 is a weak entrant. Absolute figures encode no competitive information
without field context. The crowd partially captures this — using absolute figures
would cause the model to miss the relative signal the market already prices.

**Rejected Alternatives:**
- Absolute speed figures — rejected
- Global percentile (across all horses historically) — rejected; still ignores specific field

---

## ADR-005: CVaR Portfolio Optimization, Not Per-Bet Kelly

**Date:** Project inception
**Status:** Accepted — permanent

**Context:**
How to allocate capital across all bets on a day's card.

**Decision:**
Treat the full card as a portfolio. Use CVaR-constrained optimization (expected shortfall
at 95th percentile) to allocate the bankroll across all flagged +EV bets simultaneously.
Apply 1/4 Kelly to compute maximum per-bet inputs to the optimizer, but let CVaR
enforce the final allocation.

**Rationale:**
Betting Exacta 1→2 and Exacta 1→3 are correlated: both lose if horse 1 loses.
Naive per-bet Kelly ignores this correlation and over-allocates to correlated positions.
CVaR explicitly accounts for the joint distribution of outcomes and enforces a total
drawdown ceiling, which is critical for a competition where staying in the game matters
as much as maximizing growth.

**Rejected Alternatives:**
- Per-bet Kelly, applied sequentially — rejected (ignores bet correlations)
- Mean-Variance optimization alone — rejected (penalizes upside variance equally with
  downside; inappropriate for positively skewed exotic payouts)
- Fixed fraction per race — rejected (ignores edge magnitude and correlation structure)

---

## ADR-006: Transformer Encoder for Per-Horse Career Sequence

**Date:** Project inception
**Status:** Accepted

**Context:**
How to model a horse's career trajectory across its past performance history.

**Decision:**
Transformer encoder (not LSTM). Each PP line is a token. Input sequence: last 8–10 races,
most-recent first. Output: a fixed-dimension embedding representing current form trajectory.

**Rationale:**
- Attention handles variable-length gaps between races better than LSTM's fixed time steps.
  A 6-month layoff vs. a 3-week turnaround are qualitatively different, and attention can
  learn to weight historical races based on recency and contextual similarity.
- Transformer embeddings naturally capture "always improves second start back" patterns
  that neither GBT nor LSTM handles well.
- Research (CUHK, jamesy.dev) shows 2–3% absolute accuracy improvement over MLP
  baselines on Hong Kong data.

**Rejected Alternatives:**
- LSTM — rejected (fixed time step assumption; vanishing gradient on long gaps)
- Simple recency-weighted average — rejected (no learned attention over race conditions)
- GBT on flattened PP history — rejected (loses sequential structure)

---

## ADR-007: Bayesian Hierarchical Model for Jockey/Trainer

**Date:** Project inception
**Status:** Accepted

**Context:**
How to estimate jockey × trainer × surface × distance × track interaction effects.

**Decision:**
Bayesian hierarchical model with partial pooling. Individual jockey/trainer parameters
are drawn from a global distribution. Sparse combinations (rookie jockey, new trainer
pairing) shrink toward the global mean. As data accumulates, the posterior updates.

**Rationale:**
Standard MLE on sparse jockey × trainer × surface × distance combinations overfits
dramatically. A rookie jockey winning their first race should not be modeled as having
a 100% win rate. Bayesian partial pooling is the correct statistical treatment:
high-data combinations trust their empirical record; low-data combinations trust the prior.

**Rejected Alternatives:**
- Frequency-based jockey stats (win%) — rejected (no uncertainty quantification; overfits sparse combos)
- One-hot encoding + GBT — rejected (GBT cannot generalize to unseen jockey × track combos)
- Embeddings alone — considered; Bayesian approach preferred for interpretability and
  uncertainty propagation into downstream EV calculation

---

## ADR-008: Platt Scaling + Isotonic Regression, Selected by Cross-Validation

**Date:** Project inception
**Status:** Accepted

**Context:**
How to calibrate raw model score outputs into true probabilities.

**Decision:**
Train both Platt scaling (parametric logistic) and isotonic regression (non-parametric
piecewise constant) on the held-out validation set. Select whichever achieves lower
Expected Calibration Error (ECE) via cross-validation. Apply temperature scaling to
the softmax normalization step.

**Rationale:**
Platt scaling fails when the distortion is asymmetric or non-sigmoidal. Isotonic
regression has no such constraint but requires more data to be stable. The ECE-based
selection adapts to the actual error distribution rather than assuming a shape.
If ECE is wrong (EV calculations produce false positives), it is because calibration
is wrong — this is the single most important correctness property of the pipeline.

**Rejected Alternatives:**
- No calibration — rejected (raw GBT scores are not probabilities; EV calculations
  would be completely wrong)
- Platt only — rejected (fails on asymmetric distortions)
- Isotonic only — rejected (unstable on small validation sets)