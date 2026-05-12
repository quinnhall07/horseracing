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

---

## ADR-009: Phase 0 Pipeline is Standalone — No `app/` Imports, stdlib `sqlite3`

**Date:** 2026-05-11
**Status:** Accepted

**Context:**
Phase 0 (master DB build) needs to be runnable independently of FastAPI. The training
corpus is the foundation of every downstream model — it must be reproducible from a
clean checkout without standing up the full backend.

**Decision:**
- `scripts/db/` imports only from `scripts/db/` itself. Zero dependency on `app/`.
- Database access uses stdlib `sqlite3` directly, not the SQLAlchemy async ORM that
  `app/db/` will use in production.
- All scripts work as both `python scripts/db/foo.py` (file invocation) and
  `python -m scripts.db.foo` (module invocation) via a bootstrap pattern at the top
  of every CLI script.

**Rationale:**
Phase 0 has no async requirements (it's a one-shot batch pipeline), and the schema
is small enough that ORM overhead is pure cost. Keeping it standalone means a Phase 0
re-run never blocks on a backend bug, and the training pipeline can be moved to a
separate CI job or VM without dragging the FastAPI surface along.

**Rejected Alternatives:**
- Share `app/db/models.py` SQLAlchemy schemas — rejected (couples build pipeline to
  backend startup; async-only ORM is wrong tool for a sync batch script).
- Build a third shared package — rejected (premature abstraction; two consumers don't
  justify a layer).

---

## ADR-010: Field-Map Registry With `{"const": v}` Literal Disambiguation

**Date:** 2026-05-11
**Status:** Accepted

**Context:**
Each Kaggle dataset uses different column names (`track` vs `trackname` vs `course`).
We need a registry that says "for this dataset, the canonical `track_code` field comes
from CSV column X". Some canonical fields don't have a column source at all — they're
constant per dataset (e.g., `speed_figure_source = "beyer"` for the joebeachcapital
dataset, since the entire dataset uses Beyer figures).

**Decision:**
Field-map values use a small DSL:
- `"col_name"` (string) → copy from CSV column
- `{"const": value}` (dict) → use the literal value for every row
- `None` → leave the canonical field NULL

Stored in `scripts/db/field_maps.py` as the `FIELD_MAPS` dict, keyed by Kaggle slug.

**Rationale:**
DATA_PIPELINE.md §5 sketched the registry with raw inline literals, e.g.
`"speed_figure_source": "beyer"`. That conflates two ideas: "this canonical field
comes from a column called 'beyer'" vs "this canonical field is always literally
'beyer'". The `{"const": v}` wrapper is verbose but unambiguous, and prevents
silent breakage if some future dataset has a CSV column literally named "beyer".

**Rejected Alternatives:**
- Inline literals (the DATA_PIPELINE.md sketch) — rejected (ambiguous).
- Separate `constants` dict alongside `race_fields` — rejected (introduces a third
  resolution path for one extra concept; the wrapper is cleaner).

---

## ADR-011: Quality Gate is the Sole Loadability Arbiter, With Load-Required Hard-Failures

**Date:** 2026-05-11
**Status:** Accepted

**Context:**
DATA_PIPELINE.md §8 distinguishes hard failures (missing race_date / track_code /
horse_name / finish_position → score 0.0) from soft failures (missing distance:
-0.20, missing surface: -0.15, etc.). But `distance_furlongs`, `surface`, and
`jurisdiction` are `NOT NULL` in the SQL schema and used in the race dedup key. A
row missing them cannot be persisted at all — even though the spec calls them soft.

**Decision:**
- Pydantic canonical schemas (`scripts/db/schemas.py`) are intentionally permissive:
  these fields are `Optional` so map_and_clean produces all rows for the quality gate
  to score (rather than dropping rows at validation time).
- Quality gate (`scripts/db/quality_gate.py`) treats missing
  `distance_furlongs` / `surface` / `jurisdiction` as **load-required hard
  failures** (zero score) — overriding the spec's soft classification.
- This guarantees that anything quality_gate accepts is loadable.
- `load_to_db.py` keeps a defensive belt-and-suspenders check (also skips rows with
  `None` for these three) for safety.

**Rationale:**
Treating these as soft per the spec means a row could pass the 0.60 threshold with
all three missing (1.0 - 0.20 - 0.15 = 0.65) and then fail at SQL load with an
opaque NOT NULL error. Failing them at the quality gate is semantically correct
(the row has no identity for dedup) and emits a clear rejection reason in
`rejected/reasons.jsonl`.

**Rejected Alternatives:**
- Follow spec literally; let load_to_db crash/skip — rejected (silent data loss
  with no rejection reason logged; test suite would need to assert on a SQL error).
- Default missing values to placeholders (`distance=0.0`, `surface="unknown"`) —
  rejected (would corrupt the dedup key — every "unknown" row would collide).

---

## ADR-012: SHA-256 Dedup Keys Are Frozen DB Contract

**Date:** 2026-05-11
**Status:** Accepted

**Context:**
DATA_PIPELINE.md §3 specifies exact hash inputs for race / horse / person / result
dedup keys. The exact byte-string fed into `hashlib.sha256()` is part of the database
contract — changing it produces different hashes for the same logical entity, which
silently invalidates every existing row's idempotency.

**Decision:**
- Hash input formulas in `scripts/db/dedup.py` are frozen.
- `tests/test_db/test_dedup.py` asserts hash stability across several invariants
  (date-vs-string input equivalence, case insensitivity, two-decimal distance
  rounding) so a refactor cannot silently break the contract.
- Any change to hash inputs requires bumping `SCHEMA_VERSION` in
  `scripts/db/constants.py`. Different SCHEMA_VERSION values create separate dataset
  rows, so old + new keys can coexist during a migration.

**Rationale:**
Idempotency is the single most important property of the build pipeline (DATA_PIPELINE.md
§1, "running it twice on the same source produces the same DB state"). If the hash
formula changes, a re-run sees zero matches and inserts the entire dataset again —
silently doubling every row. The frozen contract + test guard makes this a loud
failure instead.

**Rejected Alternatives:**
- Hash all available row data (more "robust") — rejected (any column with mild
  variance like timestamps would cause the same logical entity to hash differently).
- Use natural keys instead of hashes — rejected (composite UNIQUE indexes work but
  produce verbose schemas and slower equality checks at the volumes we expect; also
  doesn't help with cross-source merging).

---

## ADR-013: Auto-Ingest Uses Hybrid Strict + `--auto-map` Modes (Not One or the Other)

**Date:** 2026-05-11
**Status:** Accepted

**Context:**
`auto_ingest.py` discovers Kaggle datasets via keyword search, but most discovered
slugs have no hand-written field map in `field_maps.py`. Two extremes:
- **Strict only:** refuse to process any unmapped slug. Safest data quality, slowest
  corpus growth — every new dataset requires manual field-map authoring before any
  data lands in the DB.
- **Heuristic only:** always build a synthetic field map from column-name regexes.
  Fastest corpus growth, lowest data quality — heuristics silently mismap fields
  (e.g., picks `horse_id` as `horse_name` because both contain "horse"), and the
  user has no way to opt into strict for a careful first run.

**Decision:**
Both behaviors live behind a single `--auto-map` flag:
- Default (no flag) is **strict** — unmapped slugs return status `needs_map` with
  the actual CSV column names logged so the user can write a new entry.
- `--auto-map` activates the heuristic builder for unmapped slugs.
- In both modes, registered field maps in `FIELD_MAPS` always win over the heuristic.

**Rationale:**
The two modes serve genuinely different workflows. Early in corpus building, the
user wants `--auto-map` for bulk velocity even at the cost of some mismapped rows
(which the quality gate's range checks catch as low-scoring rejects). Later, when
the corpus is mature and adding new sources requires careful curation, strict mode
is the right default. Forcing one or the other would force the user to re-implement
the missing mode out-of-band.

**Rejected Alternatives:**
- Strict only — rejected (corpus growth would block on hand-mapping every new dataset
  found via search).
- Heuristic only — rejected (no escape hatch for the user who wants to inspect a
  novel dataset's columns before letting the heuristic guess).
- Per-keyword mode flag — rejected (overengineered; the global flag is simpler and
  the user can always run twice with different keyword sets).

---

## ADR-014: `--dry-run` is Discovery-Only — No Downloads, No DB Writes

**Date:** 2026-05-11
**Status:** Accepted

**Context:**
The first implementation of `--dry-run` short-circuited at the very end of
`process_slug` — it still downloaded every dataset (potentially many GB) and wrote
`datasets` rows to the DB before bailing. That violated user expectations of what
"dry run" means and made the flag actively dangerous.

**Decision:**
Dry-run short-circuits at the orchestration level (in `auto_ingest()`) before any
per-slug work begins. It calls `discover_slugs_with_metadata()` and returns a
discovery-only result that includes Kaggle metadata (size, downloads, votes,
last_updated, url) plus per-slug flags (`already_ingested`, `has_field_map`).
No staging directories created; no `datasets` rows written.

**Rationale:**
A dry-run that touches disk and the DB is a bug, not a feature. Users invoke
`--dry-run` precisely to preview what the real run would do without committing
side effects. The Kaggle metadata in the preview output (especially dataset size
in MB) is what enables the user to make an informed go/no-go decision before
spending the bandwidth.

**Rejected Alternatives:**
- "Preview-and-evaluate" mode that downloads but skips load — rejected (hides the
  largest cost — bandwidth — from the preview).
- Two flags: `--list-only` vs `--no-load` — rejected (the second tier of preview
  has no clear use case; the user can always run for real and inspect each
  step's parquet output if they want intermediate inspection).

---

## ADR-015: Field-Map `preprocess` Hook for Multi-CSV Datasets

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
The original `map_and_clean.py` picked the largest CSV in a staging directory and
read it as-is. Real Kaggle datasets often ship as normalized multi-CSV bundles —
a `races.csv` (per-race) joined to a `runs.csv` or `horses.csv` (per-horse) on
`race_id`. Picking only one of the two loses 50%+ of the columns and makes the
field map impossible to author.

**Decision:**
Field-map entries can declare an optional `preprocess` field naming a callable
in `scripts.db.preprocessors.PREPROCESSORS`:
```python
"gdaley/hkracing": {
    "preprocess": "gdaley_hkracing_merge",
    ...
}
```
The named callable receives the staging directory and returns a single
denormalized DataFrame. `map_and_clean.py` calls it instead of `_pick_primary_csv`
when the field map sets it.

**Rationale:**
- Keeps the merge logic in code (versionable, testable) rather than as opaque
  per-dataset preprocessing scripts the user has to remember to run.
- Each preprocessor is small (typically 4 lines of pandas merge) and easy to
  audit. Naming them in the field map keeps the dataset's full provenance
  (slug → preprocess → columns → transformers) in one place.
- Single-CSV datasets just omit the field — backward-compatible.

**Rejected Alternatives:**
- External pre-merge step that the user runs manually — rejected (breaks the
  one-command-per-stage pipeline; introduces an undocumented manual step).
- Glob-and-concat all CSVs by default — rejected (would silently merge
  schema-incompatible files like `odds.csv` + `results.csv`).
- A general-purpose JSON DSL for joins (`{"join": {...}}` in field map) —
  rejected (overengineered for the 2-3 cases we have; a function is simpler
  and more flexible).

---

## ADR-016: Quality-Gate Race-Level Grouping Must Match `race_dedup_key`

**Date:** 2026-05-12
**Status:** Accepted (corrects a bug introduced in the original quality_gate.py)

**Context:**
The original `quality_gate.race_level_issues()` grouped rows by
`(race_date, track_code, race_number)` to detect cross-row violations
(multiple winners, mixed distances/surfaces, duplicate horses). But
`race_dedup_key` includes `(track, date, race_num, distance, surface)`. Some
real datasets — Argentine venues that run turf + dirt cards in parallel —
reuse `nro` (race_number) across multiple physically distinct races on the
same date at the same venue. The dedup_key correctly stored these as separate
`races` rows, but quality_gate falsely flagged them as "mixed distances /
mixed surfaces / duplicate horses" violations and zeroed their scores. Result:
317K of 323K Argentine rows rejected on a phantom violation.

**Decision:**
The race-level grouping in `quality_gate.race_level_issues()` and the
race-key indexing in `run_quality_gate()` were updated to use all 5 fields
of the dedup_key:
`(date, track, race_num, distance_furlongs, surface)`. The "mixed distances"
and "mixed surfaces" sub-checks were removed entirely — they're a no-op once
you group by those fields, since values within a group are constant by
construction. The remaining checks (multiple winners, duplicate horses)
operate per dedup_key group.

**Rationale:**
The grouping function MUST be a strict refinement of the dedup_key, never
coarser. If quality_gate groups more aggressively than load_to_db dedupes,
quality_gate sees "violations" that aren't real. The simplest invariant is
"use the same key everywhere" — adopted.

**Rejected Alternatives:**
- Loosen the dedup_key to drop distance + surface — rejected (would cause
  Argentine venues' parallel turf+dirt races to incorrectly merge, losing
  ~half the actual race count).
- Per-dataset configurable race-key — rejected (extra config surface for no
  win; the dedup_key formula already captures everything we need).
- Keep the loose grouping but skip the mixed-distance/surface checks for
  datasets known to violate — rejected (per-dataset-special-casing is the
  exact maintenance burden the canonical pipeline was designed to avoid).

---

## ADR-017: `normalize_name` Uses Unicode-Aware Regex

**Date:** 2026-05-12
**Status:** Accepted (corrects a bug introduced in the original dedup.py)

**Context:**
The original `normalize_name()` used `re.compile(r"[^a-z0-9 ]")` to strip
punctuation. This regex is ASCII-only — it stripped every non-Roman character.
For the JRA dataset (1.6M rows of Japanese horse names like `ワクセイ` and
trainer names like `柏崎正次`), normalize_name returned the empty string for
every row. Quality_gate then rejected all 1.6M rows with
`"missing horse_name"` + `"duplicate horses within race: ['']"`.

**Decision:**
Changed the punctuation regex to `re.compile(r"[^\w ]", re.UNICODE)`. Python 3's
`\w` is Unicode-aware by default and matches CJK / Cyrillic / Greek / accented
Roman letters. Apostrophes, dashes, periods, and other punctuation are still
stripped (they're not in `\w`). Whitespace and case-folding behavior are
unchanged.

**Verified:** `normalize_name("O'Brien")` still produces `"obrien"` (apostrophe
stripped); `normalize_name("ワクセイ")` now produces `"ワクセイ"` (preserved
intact). All 234 tests pass with the change.

**Compatibility note:** This change does not violate ADR-012 (frozen dedup-key
contract). The hash *formula* is unchanged. The change to `normalize_name`
produces different hashes for non-ASCII names — but those names previously
all hashed to the same `hash("")`, so no real horses were ever distinguishable
under the old behavior. The new behavior is a strict improvement. ASCII names
hash identically before and after.

**Rejected Alternatives:**
- ASCII-fold non-Roman names (e.g., `ワクセイ` → `wakusei` via romanization)
  — rejected (requires per-language romanization tables; "wakusei" wouldn't
  match Equibase / DRF cataloging conventions which preserve katakana).
- Skip normalization for non-ASCII names — rejected (would create two
  different code paths and inconsistent dedup behavior).
- Bump SCHEMA_VERSION to invalidate old hashes — unnecessary because no
  ASCII hash changed.

---

## ADR-018: Heuristic Mapper Uses Prioritized Exact-Match Then Fuzzy

**Date:** 2026-05-12
**Status:** Accepted (replaces the original `_HEURISTICS` regex-only design)

**Context:**
The original `_HEURISTICS` resolver was a single regex per canonical field with
first-column-match-wins iteration. For `horse_name`, the regex `(horse|name)`
greedily matched `race_name` / `Course name` / `course_name` BEFORE the actual
`horse_name` column. For `finish_position`, `(finish|position|place|pos)`
matched `post_position` (gate position) and `Draw place runs` (a historical
stat) before the literal `finish_position` column. Result: the auto-mapper
catastrophically mapped wrong columns even when the right ones were obviously
named.

**Decision:**
Replaced single-regex resolution with a `_FieldSpec(exact=[...], fuzzy=...,
blocklist=[...])` per canonical field. Resolution algorithm:

1. **Exact-match phase** — try each candidate in `exact` (case-insensitive)
   against the column list; first hit wins. Order encodes priority:
   `["horse_name", "horsename", "horse", "Name", ...]` — most specific first.
2. **Fuzzy phase** (only if exact failed AND `fuzzy is not None`) — apply the
   fuzzy regex to columns NOT matching any pattern in `blocklist`.

For `horse_name`, the fuzzy phase is intentionally `None` because no generic
regex is safe (a fallback like `(horse|name)` would re-introduce the original
bug). For `finish_position`, the blocklist explicitly excludes `post_position`,
`start_position`, `draw_position`, `running_position`.

**Rationale:**
Real datasets use predictable column names ~80% of the time (`horse_name`,
`horsename`, `Name`, `horse`). For those, an ordered exact-match list works
perfectly. For the remaining ~20%, the fuzzy phase with blocklists handles
column-name variations without the false-positive disasters of the original
greedy regex.

**Rejected Alternatives:**
- Levenshtein-distance fuzzy matching — rejected (slow on 100+ column
  datasets; produces non-obvious matches that are hard to debug).
- Hand-write a column for every dataset — rejected (defeats the purpose of
  `--auto-map`; the heuristic is the fallback for datasets we don't have a
  registered map for).
- Configurable per-jurisdiction heuristics — rejected (the column conventions
  are dataset-specific, not jurisdiction-specific; a JP dataset might use
  `horse_name` while a US dataset uses `Horse`).

---

## ADR-019: `race_number` is Optional on `CanonicalRace`; Quality Gate Enforces Presence

**Date:** 2026-05-12
**Status:** Accepted (refines ADR-011)

**Context:**
The original `CanonicalRace.race_number: int` was Pydantic-required. Many real
datasets identify races by `race_id` (string ID), `card_id`, or post-time of day
rather than an explicit numeric race number. Pydantic rejected 100% of rows
from sheikhbarabas, ahmedabdulhamid, felipetappata, and noqcks at the schema
validation step — before the quality gate even saw them.

**Decision:**
Made `race_number: int | None = None` in `CanonicalRace`. Added it to the
quality_gate's load-required hard-failure block (alongside distance / surface /
jurisdiction per ADR-011): if missing, score is forced to 0.0 with reason
`"missing race_number (required for dedup_key + SQL load)"`. Field maps for
datasets without an explicit race-number column synthesize one — sheikhbarabas
uses `time_string_to_minutes` on the post-time column ("5:25" → 325 minutes
since midnight); felipetappata uses the `nro` column directly.

**Rationale:**
- Pydantic should validate identity ("can this row be parsed at all"), not
  loadability ("can this row's row be persisted in this DB"). The latter
  changes per dataset and is the quality gate's job.
- Making it Optional at the Pydantic layer means map_and_clean produces all
  rows for the quality gate to score — preserving the audit trail for missing
  race_number values that we'd otherwise silently drop.
- The error message at the quality gate is more actionable than Pydantic's
  generic `"Input should be a valid integer"` — it points the operator at
  the field map authoring as the fix.

**Rejected Alternatives:**
- Keep race_number required, require all field maps to synthesize one —
  rejected (Pydantic errors don't surface in the rejected/reasons.jsonl, so
  it's harder to diagnose new datasets).
- Drop race_number from the dedup_key entirely — rejected (would cause
  multiple races on the same date+track+distance+surface to collide; not
  acceptable for jurisdictions like UK or JP that run 8+ races per card).