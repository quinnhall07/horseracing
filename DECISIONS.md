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

---

## ADR-020: Live Ingestion DB is Separate from Master Training DB

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
Phase 1 needs durable storage for PDFs uploaded through the live API. The
Phase 0 master training DB at `data/db/master.db` already has races / horses /
race_results / jockeys / trainers tables. The naive option is to reuse those
tables and tag the new rows with a `source = "live_ingest"` column.

**Decision:**
The live ingestion DB is a SEPARATE SQLite/PostgreSQL database (default
`./horseracing.db`, overridable via `HRBS_DATABASE_URL`). It has its own ORM
tree under `app/db/models.py`: `IngestedCard` → `IngestedRace` →
`IngestedHorse` → `IngestedPPLine`, with `cascade="all, delete-orphan"` so a
card delete wipes the subtree cleanly.

**Rationale:**
- Different access patterns. Master DB is read-heavy for training queries
  (full table scans, aggregate joins). Live DB is write-heavy for individual
  card ingests, mostly point lookups for serving the UI. Mixing locks both.
- Different schemas. The master DB is normalised around dedup keys (one
  horse row across thousands of races). The live DB models a single PDF
  upload tree where each card is independent. Forcing them into one schema
  either bloats master with parse_confidence columns or makes live ingest
  unnecessarily complex with dedup logic.
- Different lifecycles. Master DB gets re-built from scratch when a new
  Kaggle dataset lands. Live DB is append-only over time. Coupling them
  means every master rebuild risks blowing away live history.

**Rejected Alternatives:**
- Single shared DB with a `source` column — rejected (schemas don't overlap
  cleanly; locks would bite under concurrent training + ingest load).
- In-memory live state, no persistence — rejected (UI needs to retrieve
  past cards; refreshing the page shouldn't lose data).

---

## ADR-021: EquibaseParser is a Thin Subclass of BrisnetParser

**Date:** 2026-05-12
**Status:** Accepted (provisional — revisit when real Equibase PDFs land)

**Context:**
Format detection routes Brisnet UP / Equibase / DRF to format-specific
parsers. We have Brisnet sample data but neither Equibase nor DRF sample
PDFs yet. Without real samples, attempting format-specific regexes for
Equibase is speculative — we'd just be re-implementing the Brisnet parser
under a different name.

**Decision:**
`EquibaseParser` is `class EquibaseParser(BrisnetParser): pass`. The
extractor dispatches on format and routes the `equibase` label to it.
When real Equibase PDFs surface format-specific divergences (different
speed-figure column header, workout-table position, fraction-time
formatting), the overrides go onto this subclass.

**Rationale:**
- Dispatch separation is cheap and has value NOW (the format label is
  carried through downstream telemetry, audit trail, etc.).
- The regex layer is the expensive thing to fork prematurely — without
  samples it's all guessing. Better to inherit the working Brisnet impl
  than to write a divergent parser that breaks on the first real PDF.
- DRF currently has no parser class — it still falls back to BrisnetParser
  via the extractor's "DRF parser not yet implemented" warning path,
  because there's not even a useful dispatch label to preserve yet.

**Rejected Alternatives:**
- Full Equibase regex layer from scratch — premature, no samples.
- Make BrisnetParser format-aware via a `_format` attribute — rejected
  (mixes responsibilities; cleaner to subclass when needed).

---

## ADR-022: EWM Direction in PP Context — Reverse Before pandas .ewm()

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
The feature engineering layer computes EWM (alpha=0.4) of speed figures
per CLAUDE.md §8. The HorseEntry schema enforces `pp_lines` are
most-recent-FIRST (validated by `_sort_pp_lines` model validator). But
pandas' `Series.ewm(alpha=0.4, adjust=True).mean()` weights the LAST
element of the input most heavily — opposite convention.

If you call `.ewm()` directly on the most-recent-first PP list, you weight
the OLDEST race heaviest, which is backwards from what we want.

**Decision:**
`ewm_speed()` in `app/services/feature_engineering/speed_features.py`
explicitly `list(reversed(figs))` before constructing the pandas Series.
The reverse step + docstring note about the direction are mandatory; any
future rolling/EWM aggregate over `pp_lines` must do the same.

The training-data path (`prepare_training_features`) avoids this because
it works on a long-form DataFrame sorted ascending by date — pandas'
natural convention then matches: most-recent rows are the LAST rows in
the group, and `groupby.shift(1).ewm(...)` correctly weights recent
history.

**Rationale:**
- Two different code paths (live inference from RaceCard vs. training
  from parquet) work with the data in opposite orders. Making the schema
  store one direction and forcing every aggregator to be aware of it is
  the simplest safe approach.
- A test asserts `ewm_speed([100, 80])` weights 100 more heavily than the
  simple mean — locks in the direction so future refactors can't silently
  flip it.

**Rejected Alternatives:**
- Change the schema to store most-recent-LAST — rejected (downstream UI
  consumes pp_lines and most-recent-first is the natural display order).
- Use a custom EWM implementation iterating in the desired direction —
  rejected (pandas .ewm() is well-tested; reinventing it for a direction
  flip is premature).

---

## ADR-023: Trainer Continuity is a Documented Weak Proxy

**Date:** 2026-05-12
**Status:** Accepted (will be superseded once richer PP data is available)

**Context:**
The connections feature module needs to know whether today's
jockey/trainer combination is new for the horse, or a continuation. The
`HorseEntry` schema carries today's trainer; `PastPerformanceLine` carries
the per-PP `jockey` but NOT a per-PP `trainer` — Brisnet's dense PP table
omits the per-PP trainer to save horizontal space.

**Decision:**
`horse_connections_summary().trainer_continuity` returns 1.0 if today's
trainer is present AND today's jockey matches the most recent PP jockey,
else 0.0; None if either today's trainer or PP history is absent. The
docstring and the module header both flag this as a proxy and identify
the unblock criterion (per-PP trainer in the schema).

**Rationale:**
- Jockey-as-trainer-proxy is empirically valid: trainers persist across
  riders far more often than they change them. So a horse with the same
  jockey today as last race is almost certainly with the same trainer.
- The full fix requires parsing the Brisnet horse-header trainer name
  AND the per-PP "comment" field for trainer changes — a non-trivial NLP
  task that belongs in Phase 2's NLP follow-up, not in the parser.
- The Phase 3 ConnectionsModel does NOT depend on this proxy — it works
  off the master DB's full per-result jockey/trainer columns. So the
  proxy only matters for live-inference feature display; it's not on
  any critical path.

**Rejected Alternatives:**
- Always return 1.0 (assume continuity) — rejected (creates false-positive
  signal in cases where the trainer actually did change).
- Always return None — rejected (loses the easy-but-correct cases where
  jockey continuity strongly implies trainer continuity).
- Extend PastPerformanceLine with optional trainer field — rejected for
  Phase 2 (deferred until the comment-line NLP work exposes it).

---

## ADR-024: Softmax Operates on Raw LightGBM Scores, Not Sigmoided Probabilities

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
After fitting the Speed/Form LightGBM binary classifier, we need a
per-race probability distribution that sums to 1 (so downstream code can
treat it as a discrete distribution over starters). LightGBM exposes two
output modes:
1. `predict(...)` → sigmoid-squashed probabilities in [0, 1].
2. `predict(..., raw_score=True)` → pre-sigmoid additive log-odds.

We could softmax either across the in-race field. The choice matters at
the tails (longshots and chalk).

**Decision:**
`SpeedFormModel.predict_softmax()` uses raw scores. The softmax operates
on the additive log-odds and produces the per-race distribution.

**Rationale:**
- Softmax is the exponential of a linear scaling of inputs. On raw scores
  it's mathematically equivalent to a normalised exp(score), which is
  what additive-model log-odds are designed for.
- Sigmoid-then-renormalise double-squashes the dynamic range: sigmoid
  compresses extreme scores toward 0 or 1, then softmax compresses them
  again. The result is a distribution flatter than the model's actual
  signal — particularly harmful at the tails where the win signal lives.
- The downstream Plackett-Luce ordering model (Phase 4) consumes
  log-odds-style strength parameters. Raw scores feed it natively.
- Calibration (Phase 4 Platt/isotonic) operates on EITHER output — using
  raw scores here doesn't preclude calibrating later.

**Rejected Alternatives:**
- Sigmoid then renormalise — rejected for tail-compression reason above.
- Per-row sigmoid output without renormalisation — rejected (per-race
  probs need to sum to 1 for downstream EV / portfolio code).

---

## ADR-025: Orthogonalisation Lives in the Meta-Learner, Not the Sub-Models

**Date:** 2026-05-12
**Status:** Accepted (implements CLAUDE.md §2 mandate)

**Context:**
CLAUDE.md §2 mandates that sub-model inputs be orthogonalised before being
fed to the meta-learner — speed figures already incorporate pace, so the
raw pace_scenario output and the raw speed_form output share signal that
the meta-learner shouldn't double-count.

The orthogonalisation could live in two places:
1. Inside each sub-model: it knows it's contributing to a stack.
2. Inside the meta-learner: it sees all inputs and residualises.

**Decision:**
Sub-models emit RAW outputs. The MetaLearner module owns the
orthogonalisation. Specifically:
- The first sub-model column (`speed_form_proba`) is the anchor.
- Every other sub-model column is replaced with its residual after a
  linear regression against the anchor.
- `_orthogonalise()` runs as the first step of `_features()`, so both
  fit and predict apply the same transformation.

**Rationale:**
- Sub-models stay composable. A `SpeedFormModel.predict_proba` is the
  best win probability the speed-form model can give — period. Useful
  on its own, not just inside a stack.
- The meta-learner is the right place to decide what's redundant: only
  it sees all 5 outputs together and knows what to anchor against.
- Swapping meta-learners (logistic regression, MLP, …) doesn't require
  touching Layer 1. The contract is "Layer 1 emits proba columns;
  Layer 2 does whatever it wants with them."
- Stub sub-models (pace, sequence) that return constant 0.5 get zeroed
  out by orthogonalisation automatically — no special-casing needed.

**Rejected Alternatives:**
- Push orthogonalisation into each sub-model — rejected (each sub-model
  would need to know about the other models' presence; tight coupling).
- Skip orthogonalisation, let LightGBM tree splits handle redundancy —
  rejected (works empirically but violates CLAUDE.md §2's explicit
  mandate; deterministic preprocessing is auditable, model-internal
  feature selection is not).

---

## ADR-026: Untrained Sub-Models Return Constant 0.5

**Date:** 2026-05-12
**Status:** Accepted (provisional — supersede when models are trained)

**Context:**
Pace and Sequence sub-models cannot be trained from the current parquet:
Pace needs fractional time columns (all NULL in source datasets);
Sequence needs PyTorch + a globally-unique horse_id (currently we collide
across years/jurisdictions). The bootstrap orchestrator needs to wire
them in for the meta-learner regardless.

Options:
1. Don't run the stacker at all until every sub-model is trained.
2. Use a placeholder column of NaN — meta-learner needs to handle missing.
3. Use a placeholder constant — meta-learner sees it as constant input.

**Decision:**
`PaceScenarioModel.predict_proba` and `SequenceModel.predict_proba`
return `np.full(n, 0.5, dtype=float)`. Their `fit()` raises
`NotImplementedError`. They have a `predict_proba` because the meta
needs a column from every slot.

**Rationale:**
- The ADR-025 orthogonalisation step regresses every sub-model column on
  the speed_form anchor. A constant column has zero variance → linear
  regression coefficient is 0 → residual is 0 across all rows → the
  meta-learner sees a constant 0 input, which has zero gain → it's
  effectively unused. Same outcome as if the slot didn't exist.
- 0.5 is the "neutral" placeholder — the expected value of an
  uninformative prior over P(win). Doesn't bias downstream tools that
  might consume the raw output before orthogonalisation (e.g., logging
  / monitoring code that asserts all proba columns are in [0, 1]).
- Plugging in a trained model later is a single-file change
  (replace the stub class). No orchestrator edits.

**Rejected Alternatives:**
- NaN placeholder — rejected (forces every consumer to NaN-guard).
- Block the stacker until all sub-models train — rejected (would
  indefinitely delay Phase 4 calibration work behind data acquisition
  that's out of our control).
- Always orthogonalise out the stub manually in the orchestrator —
  rejected (couples the orchestrator to the set of stubs).

---

## ADR-027: Horse Grouping Key is `(horse_name, jurisdiction)` (Compromise)

**Date:** 2026-05-12
**Status:** Accepted (provisional — see "Future" entry in PROGRESS.md)

**Context:**
Phase 3 leakage-free feature prep needs to group historical results by
horse so per-horse priors (EWM speed figure, win rate, days since last)
are computed over THE SAME horse's history, not across all horses sharing
a name. The training parquet ships `horse_name` (the
`horses.name_normalized` column from the master DB), `jurisdiction`,
and `foaling_year` — but foaling_year is 100% NULL.

Pure-name grouping collides: distinct horses across decades and countries
sometimes share names (especially common English-language names). The
master DB's `horses.dedup_key` is globally unique by construction (it
includes foaling year + country) but is NOT exported in the parquet.

**Decision:**
`_horse_key(df) = df["horse_name"] + "|" + df["jurisdiction"]`. Documented
inline in `training_data.py` with a TODO pointing at the proper fix
(exposing `horses.dedup_key`).

**Rationale:**
- `(name, jurisdiction)` materially reduces collisions over name-alone:
  a 2010 UK gelding and a 2015 JP filly with the same registered name
  no longer share priors. The remaining collisions (same name within the
  same jurisdiction across decades) are rare enough that they don't
  dominate training noise.
- The fix (exposing dedup_key in the export) is a one-liner in
  `export_training_data.py` — but it requires re-exporting the parquet
  AND ensuring nothing downstream depends on the current column set.
  Deferring until Phase 4 lets us land Phase 3 metrics now.

**Rejected Alternatives:**
- Pure `horse_name` grouping — rejected, too many collisions.
- Re-export the parquet right now with dedup_key — viable but deferred
  to avoid mid-phase parquet churn.
- Synthesize a per-row horse_id during prep by hashing
  (name|jurisdiction|first_seen_year) — rejected (still collides on
  shared first-seen year; would mask the bug rather than fix it).

---

## ADR-030: Calibrator Selects on Fit-Slice ECE (Provisional)

**Date:** 2026-05-12
**Status:** Superseded by ADR-037 (2026-05-12) — the full-parquet
validation surfaced the failure mode this ADR called "the case where
Platt wins on the test slice but isotonic wins on the fit slice".
The fit-slice criterion now falls back to held-out inner-val ECE.

**Context:**
ADR-008 specified cross-validated ECE selection between Platt scaling and
isotonic regression. Implementing fold-based selection inside `fit()` is
several files of CV plumbing for a question that — on our actual data so
far — has a single answer: isotonic dominates.

**Decision:**
The `Calibrator.fit(scores, labels)` path fits both candidates on the
SAME data it receives and computes ECE on that same data to decide. The
auto-selector simply picks the lower-ECE candidate. The held-out
generalisation check happens ONE LAYER UP in
`scripts/validate_calibration.py`, which evaluates the chosen calibrator
on an entirely separate test slice (`evaluate_calibration` returns both
pre- and post-cal ECE on the test slice).

**Rationale:**
- Isotonic has more degrees of freedom and will essentially always win
  on its own training data, so fit-slice ECE alone doesn't distinguish
  the methods well. But on our smoke run (5% sample, separate test slice)
  post-cal ECE remained strictly better than pre-cal ECE for BOTH models
  — isotonic was the right call, full stop.
- If we ever see a case where Platt wins on the test slice but isotonic
  wins on the fit slice, we'll have evidence that fold-based selection
  is necessary. Until then, the simpler path keeps the calibrator small
  and auditable.
- The full-blown ADR-008 spec ("Platt fails when distortion is asymmetric
  or non-sigmoidal; isotonic has no such constraint but requires more
  data to be stable") is still the operating principle — our test slice
  on the meta-learner showed exactly the asymmetric overconfidence Platt
  can't fit, and isotonic flattened it.

**Rejected Alternatives:**
- 5-fold CV inside `fit()` — deferred until evidence suggests selection
  is wrong. Extra LOC for no current behavioural difference.
- Always-isotonic (drop Platt entirely) — rejected. The smoke run shows
  isotonic wins on this corpus; that doesn't mean it always will, and
  keeping both lets the auto-selector adapt to future data without code
  changes.

---

## ADR-031: Plackett-Luce NLL is Vectorised Over Orderings, Not Looped

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
The MLE strength fitter's likelihood evaluates `Σ_{orderings} Σ_{positions}
[θ_chosen − logsumexp(θ_remaining)]`. The naive Python implementation
iterates two nested loops in pure Python — fine for hundreds of orderings
but slow at thousands. The synthetic recovery test (5000 orderings × ~50
L-BFGS iterations) took 60+ seconds with the naive loop.

**Decision:**
The NLL accepts a rectangular `(n_orderings, max_len)` integer array of
padded ordering indices plus a `(n_orderings,)` length vector. Inside the
function we loop only over POSITIONS (max length = `n_items`) and apply
the position step to all orderings simultaneously via:
1. A `(n_orderings, n_items)` boolean mask of items still remaining.
2. `logsumexp(np.where(remaining, theta, -inf), axis=1)` for per-row
   denominators.
3. Active-row gating via `pos < orderings_lens`.

This drops the test suite from 66 s to ~2 s on the same hardware.

**Rationale:**
- The position loop is bounded by `n_items` (≤ 20 in practice). The
  ordering loop was unbounded — making it the outer-Python loop is what
  was killing us.
- numpy + scipy fused-ops scale at C speed; Python loop overhead per
  ordering was the dominant cost.
- Partial orderings (top-k for k < n_items) fit naturally: pad with any
  sentinel value, gate contributions with `active = pos < lens`.

**Rejected Alternatives:**
- numba JIT — rejected (extra build dep + cold-start cost).
- scipy's analytical PL gradient via `scipy.stats` — rejected (no first-
  class PL in scipy; would need a third-party package like `choix`).
- Cython rewrite — premature; the vectorised numpy version is fast
  enough that we can fit on 100k+ orderings in seconds.

---

## ADR-032: Per-Race Temperature Softmax is Logit-Scale, Not Power-Scale

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
After per-row calibration, the per-race distribution needs to renormalise
so probabilities sum to 1 within a field. Two common forms:
1. Power scaling: `p_i^(1/T) / Σ_k p_k^(1/T)`
2. Logit-scale softmax: `softmax(logit(p_i) / T)` per race

Both produce a distribution summing to 1 per race. They differ at the
tails: power scaling is multiplicative, logit-scale softmax is additive.

**Decision:**
`Calibrator.predict_softmax(scores, race_ids)` uses logit-scale: for each
row, compute `logit(clip(p, ε, 1-ε)) / T`, then stable-softmax across
race groups.

**Rationale:**
- Plackett-Luce treats strengths as proportional to win probabilities and
  composes additively in log space. Feeding it logit-scale outputs is the
  natively-compatible form.
- The calibration literature (Guo et al. 2017, "On Calibration of Modern
  Neural Networks") uses logit-scale temperature scaling as the
  canonical form. Adopting the same shape means our calibration discussion
  is interchangeable with published reliability-diagram analyses.
- Power scaling at extreme p values is numerically unstable when p is
  near 0 or 1 — it blows up the multiplicative factor. Logit-scale with
  clipped inputs is bounded by construction.

**Rejected Alternatives:**
- Power scaling — incompatible with PL's log-space composition; numerical
  edge cases at the tails.
- No temperature (T = 1 fixed) — loses a knob the future ordering /
  portfolio layer will want for risk shaping. The current default is
  T = 1.0 (no-op), but the knob is there.

---

## ADR-033: MarketModel Save/Load Rebuilds Isotonic Interpolation

**Date:** 2026-05-12
**Status:** Accepted (corrects a pre-existing bug)

**Context:**
`MarketModel` serialises its `sklearn.isotonic.IsotonicRegression` to JSON
(thresholds + y values). The original `load()` set `iso.f_ = None`,
expecting sklearn's `_transform` to recompute the interpolation lazily.
sklearn does not in fact recompute — it calls `self.f_(T)` directly. So
every prediction on a freshly-loaded `MarketModel` raised
`TypeError: 'NoneType' object is not callable`. The Phase-3 bootstrap had
only ever exercised the fit-then-predict-in-the-same-process path; the
bug surfaced when Phase 4 `validate_calibration.py` reloaded the model.

**Decision:**
On load, rebuild `iso.f_` directly using `scipy.interpolate.interp1d(xs,
ys, kind="linear", bounds_error=False, fill_value=(ys[0], ys[-1]))`.
This matches what sklearn's `_build_f` builds at fit time for the
`out_of_bounds="clip"` configuration. A new
`test_save_load_round_trip_preserves_predictions` regression test asserts
`predict_proba` outputs are byte-identical before and after a round-trip.

**Rationale:**
- The JSON format stays portable (no joblib pickle in the model
  artefact directory — only thresholds + config), but the load path is
  now functionally complete.
- Sklearn's `_build_f` is a private method and could change; calling
  scipy's `interp1d` directly is stable across sklearn versions and uses
  the same underlying primitive sklearn itself uses.
- The regression test is the lock-in: any future change to MarketModel
  save/load must keep predictions identical or the test fails.

**Rejected Alternatives:**
- Switch entire MarketModel persistence to joblib — rejected. JSON is
  human-readable and inspectable; joblib pickles are not. The fix is one
  function call, not a serialisation-format rewrite.
- Defer the predict-time interpolation rebuild to first-call (lazy) —
  rejected. Less explicit; harder to reason about; doesn't catch the
  bug at load time when it's easiest to surface a clear error.

---

## ADR-028: ConnectionsModel Shrinkage Prior Strength α = 30

**Date:** 2026-05-12
**Status:** Accepted (tunable; default is empirically reasonable)

**Context:**
The Layer-1d ConnectionsModel uses a beta-binomial empirical-Bayes
estimator: per-jockey, per-trainer, and per-pair win rates are shrunk
toward the per-jurisdiction baseline by an additive pseudo-count `α`.
`α` is the model's central hyperparameter — too small and a 1-for-1
jockey looks like 100% win rate; too large and elite riders look average.

**Decision:**
`ConnectionsConfig.prior_strength = 30.0` by default.

**Rationale:**
- Interpretation: "act as if you've already observed 30 races at the
  baseline rate before trusting any new evidence." With baseline
  ≈ 8.5% (the actual global win rate in the training parquet), the
  prior contributes ≈ 2.55 pseudo-wins toward the shrunken numerator.
- Empirically, this puts the per-jockey rates in a believable range:
  veteran jockeys with 500+ starts express their true rate; rookies
  with 5 starts barely deviate from baseline. Mean shrunken rate across
  all jockeys is within 0.5% of the jurisdiction baseline — the
  shrinkage isn't biasing en masse, just damping outliers.
- Phase 4 calibration will surface whether the meta-learner benefits
  from a stronger or weaker prior here. Until then, α = 30 is a safe
  starting point.

**Rejected Alternatives:**
- α = 10 (less shrinkage) — rejected after spot-checks showed top-10
  jockeys with high win rates over single-digit start counts.
- α = 100 (more shrinkage) — rejected; effectively neuters the model
  by pulling everyone to baseline.
- Per-jurisdiction α — rejected for now (premature; jurisdictions have
  different start-count distributions but the constant α at this
  magnitude doesn't materially mistreat any of them).
- Full PyMC hierarchical model — deferred to Phase 4 (the
  empirical-Bayes version captures most of the signal at a fraction of
  the implementation complexity).

---

## ADR-029: Leakage-Free Feature Prep via `groupby.shift(1)` Before Every Rolling Aggregate

**Date:** 2026-05-12
**Status:** Accepted (implements CLAUDE.md §2 train/val isolation mandate)

**Context:**
CLAUDE.md §2 requires time-based train/val splits. But split-level
time-isolation alone is insufficient: if a per-row feature accidentally
includes the row's own race outcome (e.g., EWM speed of all that horse's
races including today's), the model can perfectly learn the target from
the feature within the training set, then fail on validation.

Specifically: the parquet has one row per historical horse-race result.
For each row, the per-horse rolling features must use ONLY rows with
strictly earlier dates for that horse.

**Decision:**
`prepare_training_features` enforces leakage-free aggregates uniformly:
every per-horse rolling/EWM aggregate is computed via
`df.groupby('horse_key')['col'].transform(lambda s: s.shift(1).rolling(...).foo())`.
The `shift(1)` step happens INSIDE the lambda before the rolling/EWM, so
the value at row i depends only on rows 0..i-1.

A dedicated test (`test_prepare_features_no_leakage_first_row_per_horse_has_null_priors`)
asserts that for the first start of every horse, every `*_prior` column
is NaN — which can only be true if the shift(1) is actually wired up.

**Rationale:**
- Catching leakage at the feature level is critical. By the time it
  shows up in val metrics it's too late: the entire training run
  upstream of validation is suspect.
- The `_prior` suffix on every leakage-controlled column is part of the
  public contract — any future column added to the feature set must
  follow the same suffix-and-shift discipline. The test enforces it.
- A future Phase 4 reliability-diagram report can sanity-check that
  the val log-loss is consistent with the train log-loss (they should
  differ by a few percent, not by orders of magnitude) — a large gap
  is a leakage indicator.

**Rejected Alternatives:**
- Compute features without the shift and rely on time-split alone to
  catch leakage — rejected (would only catch it on validation, and
  diagnosing post-hoc is painful).
- Use a separate "as_of" framework / library (e.g. featuretools) —
  rejected (overkill for the row-level shift this needs; adds a
  dependency for negligible gain).
- Drop the first row of every horse from training (no priors available) —
  rejected (the model should learn FTS behaviour; dropping them is a
  separate, opt-in toggle via `drop_first_starts=True`).

---

## ADR-034: Stern Model — MC for shape ≠ 1, PL Closed Form for shape = 1

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
ADR-001 names Stern (Gamma) as the preferred ordering model. The Stern
model assigns each horse a Gamma-distributed finishing time
`T_i ~ Gamma(shape=r, rate=v_i)`. For r=1 the model is exactly
Plackett-Luce (the Luce property `P(i wins) = v_i/Σv_j` holds and all
top-k probabilities have closed forms). For r ≠ 1 no closed form exists
— we need Monte Carlo over sampled finishing times.

**Decision:**
`SternModel.exacta_prob`, `.trifecta_prob`, `.superfecta_prob`, and
`.enumerate_exotic_probs` delegate to `plackett_luce.*` when
`config.shape == 1.0`. For all other shapes they invoke
`sample_orderings` (default 20,000 samples) and tabulate empirical
top-k frequencies.

The strengths input is interpreted as Gamma RATE parameters (higher =
faster). At shape=1 the rates ARE the win probabilities (Luce). At shape
≠ 1 the rates no longer match marginal win probs; the caller can recover
target marginals via `infer_strengths`, a damped multiplicative
fixed-point iteration `v ← v · (target/implied)^damping` that converges
in ~30 iterations at damping=0.5.

**Rationale:**
- The PL fast path keeps the common case (shape=1) analytic — no MC
  noise, no n_samples knob to tune in production. The Stern test suite
  asserts byte-identical results vs the PL module at shape=1.
- Monte Carlo is the only tractable path for r ≠ 1. Closed-form
  exotic probabilities under Gamma marginals involve nested incomplete
  gamma integrals; numerical quadrature is slower than 20k MC samples
  AND has its own discretisation error.
- Vectorised sampling via `rng.gamma(shape, scale=1/v, size=(n, N))` is
  fast (≈ 1ms for N=12, n=20,000 on a laptop). MC noise on a typical
  exacta probability (p ≈ 0.1) is SE ≈ 0.002 — well below the
  ~1% calibration noise floor.
- `infer_strengths` is needed because at shape > 1 the strong horse
  in any pair captures more than its Luce share; downstream EV would be
  miscalibrated unless we invert. Damping=0.5 was tuned empirically;
  undamped iteration oscillates for shape ≥ 3.

**Rejected Alternatives:**
- Closed-form exact via incomplete gamma integration — rejected
  (slower than MC for any practical accuracy target; only the exacta
  case is even reasonably tractable; trifecta/superfecta require nested
  quadrature).
- Always use MC, even at shape=1 — rejected (introduces MC noise where
  none is needed; PL closed form is exact and free).
- Treat shape as a per-race parameter — rejected for now (shape is a
  global "determinism" knob; making it per-race adds parameters without
  obvious benefit until we've established the global-shape baseline).

---

## ADR-035: Copula — Block-Equicorrelation by Pace Style, with PL/Stern Fallbacks

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
ADR-001 names a copula-based ordering model as the Phase-4 target. PL
and Stern both assume INDEPENDENT finishing times — wrong when horses
share a pace style. Two front-runners who hook up early both fade
together; two closers benefit from the same fast pace. The copula
captures this dependency between same-style horses.

**Decision:**
`CopulaModel` builds a Gaussian copula with Gamma marginals:
`Z ~ MVN(0, Σ)`, `U = Φ(Z)`, `T = Gamma_inv(U; shape, 1/v)`. The
correlation matrix is BLOCK-EQUICORRELATION: ρ within each pace-style
group, 0 across groups. PSD is guaranteed for any ρ ∈ [0, 1).

Three explicit fast paths short-circuit Monte Carlo:
1. `pace_styles=None` AND `marginal_shape=1.0` → exact PL.
2. `rho=0.0` AND `marginal_shape=1.0` → exact PL.
3. Either of the above with `marginal_shape ≠ 1.0` → delegates to
   `SternModel` (which itself goes to MC).

Otherwise: MC via Cholesky factor of Σ + jitter (1e-10), drawing
standard normals, mapping to uniforms via `scipy.stats.norm.cdf`, then
to Gamma quantiles via `scipy.stats.gamma.ppf`. Default 20k samples.

**Rationale:**
- Block-equicorrelation is the minimal pace-aware extension. Cross-
  style correlation is a research question we defer; the literature on
  exotic correlation structures is thin and the data-required-to-
  identify-cross-style-correlation is large.
- PSD safety: block-equicorrelation has analytic eigenvalues
  {1-ρ, ρ(k-1)+1} per block, all positive for ρ ∈ [0, 1). Adding
  diagonal jitter handles roundoff at ρ near 1.
- The Luce property does NOT survive same-style positive correlation:
  within a style block the stronger horse's marginal win share grows at
  the expense of the weaker. This is the entire POINT of the copula
  (same-style horses compete for the same trip; one can't "happen to be
  faster than expected" while its block-mate stays calibrated). The
  module docstring + tests pin this directionality.
- Cross-style correlation = 0 keeps the model interpretable: tuning ρ
  has a single, signed effect (more ρ → stronger same-style concentration).

**Rejected Alternatives:**
- Full unstructured correlation matrix Σ — rejected. Needs O(N²)
  parameters with no clear identification source.
- Vine copula / t-copula — rejected for now (added complexity over
  Gaussian; no evidence racing residuals are heavy-tailed enough to
  warrant t).
- Force PL marginals to survive at ρ > 0 by inverting strengths
  per-call — rejected for v1. The mathematical machinery exists
  (analogous to `SternModel.infer_strengths`) but adds runtime cost
  without a downstream consumer yet. The pace model is a stub today;
  marginal preservation can be added when there's a production pipeline
  for it.
- Negative ρ — rejected. Negative cross-style correlation isn't
  motivated by the racing literature, and validating PSD with mixed-
  sign block correlations is non-trivial.

---

## ADR-036: CUSUM Drift Detector Operates on Standardised Bernoulli Residuals

**Date:** 2026-05-12
**Status:** Accepted

**Context:**
We need an online statistic that detects when the calibrator has drifted
out of spec so the operator can refit. A raw-residual CUSUM (S+ = max(0,
S+ + r - k) where r = label - pred) is overly sensitive to individual
high-confidence observations: a single label=0 on a pred=0.9 contributes
-0.9 in one step, which can cross any reasonable h threshold by itself.
Our initial implementation with raw residuals false-alarmed on a 1000-obs
calibrated stream at step 22.

**Decision:**
The CUSUM updates on the STANDARDISED (Bernoulli z-score) residual

    z_t = (label_t − p_t) / sqrt(p_t · (1 − p_t))

with two numerical guards: `sqrt(p·(1-p))` is floored at `eps` (default
1e-4) to avoid div-by-zero at p ≈ 0 or 1, and z_t is then clipped to
`[-z_clip, +z_clip]` (default 5).

Under perfect calibration, each z_t has mean 0 and variance 1 regardless
of p_t. CUSUM defaults: `k=0.5` (σ-units), `h=4.0` (σ-units),
corresponding to ARL₀ ≈ 168 (per Hawkins-Olwell standard tables).

**Rationale:**
- The standardised residual gives each observation the same expected
  weight under calibration. A label=0 on pred=0.9 contributes
  z = -3 (capped at -5 by z_clip) — large, but bounded. A label=1 on
  pred=0.5 contributes z = +1. The accumulation is variance-stable
  across the prediction distribution.
- ARL_0 ≈ 168 at default config is a reasonable detection-vs-false-
  alarm trade-off for our use case (we refit calibration weekly at
  most; one false alarm per ~168 races is fine).
- The detector latches on first alarm (subsequent updates do not reset
  alarmed_at). This is the right behaviour for the calling code: once
  drift is detected, the operator refits and explicitly calls
  `reset()` to start a new monitoring epoch.
- A `direction` field on the state distinguishes "model under-predicts"
  (label > pred in expectation, direction='high') from
  "model over-predicts" (direction='low'). Both are diagnostically
  useful even if the response is the same (refit).

**Rejected Alternatives:**
- Raw residual CUSUM with carefully-tuned defaults — rejected.
  Initial attempt with k=0.025, h=2.0 false-alarmed at step 22 on a
  perfectly-calibrated 1000-obs stream. Any defaults that work for
  the well-calibrated case fail on the drifted case, because the
  detection signal magnitude is conflated with the prediction
  magnitude.
- EWMA of residuals + z-test alarm — rejected (no change-point
  awareness; once the model drifts the cumulative residual lingers
  forever, masking subsequent recoveries).
- Likelihood-ratio CUSUM (e.g. Wald SPRT) — rejected for v1.
  Requires an explicit "alternative" calibration distribution; we don't
  have a parameterised alternative model (drift could go any direction).
  Worth revisiting if we ever want to detect specific failure modes
  (e.g. "longshot bias growing").

---

## ADR-037: Calibrator Auto-Selector Uses Held-Out Inner-Val ECE (Supersedes ADR-030)

**Date:** 2026-05-12
**Status:** Accepted (supersedes ADR-030's "fit-slice ECE" criterion)

**Context:**
ADR-008 chose Platt-vs-isotonic auto-selection. ADR-030 (provisional)
implemented selection by ECE on the FIT SLICE itself — the same data
the calibrators were fit on. Isotonic regression has enough flexibility
to memorise the fit slice (post-fit ECE ≈ 1e-18 on every dataset), so
the fit-slice criterion always picks isotonic regardless of whether
isotonic actually generalises.

The full-parquet validate_calibration run (session 2026-05-12 f)
exposed the failure: both Speed/Form and Meta-learner were ALREADY
well-calibrated raw (test-slice ECE ≈ 0.003), the auto-selector picked
isotonic anyway, and applying isotonic slightly DEGRADED test ECE
(→ 0.005–0.006). On this dataset Platt would have been the right
choice — but the fit-slice criterion could not see that.

**Decision:**
`Calibrator.fit` under `method='auto'` does an internal split of the
fit slice (default 20% inner-val, seeded random shuffle), fits Platt
and isotonic on the inner-train slice, computes ECE on the inner-val
slice for each, and chooses based on inner-val ECE with a protective
bias toward Platt:

    chosen = isotonic if iso_val_ece + auto_min_delta_ece < platt_val_ece
             else platt

Defaults: `auto_val_fraction = 0.20`, `auto_min_delta_ece = 0.001`
(isotonic must beat Platt by ≥ 0.1% ECE), `auto_min_inner_val_size = 100`
(fall back to fit-slice ECE when the calib slice is too small for a
reliable inner-val estimate).

Both calibrators are STILL re-fit on the full calib slice after
selection so no data is wasted. The metadata persists both:
- `metrics`: fit-slice ECE/Brier/log-loss for both methods (diagnostic).
- `inner_val_metrics`: held-out ECE for both methods (drives the choice).
- `auto_selection_mode`: "held_out" or "fit_slice_fallback".

**Rationale:**
- Held-out ECE is the only valid generalisation criterion for choosing
  between calibrator flexibility levels. Fit-slice ECE is conceptually
  the same mistake as choosing model hyperparameters by training loss.
- The 20% inner-val fraction is large enough to keep variance on the
  ECE estimate low (at 200k+ calib rows, that's 40k val rows, ECE SE
  well below 0.001) and small enough that the inner-train slice
  still produces a representative calibrator.
- The 0.001 `min_delta_ece` introduces a one-sided protective bias
  toward the simpler model (Platt). On data where Platt and isotonic
  are statistically indistinguishable, picking Platt is the more
  conservative, more interpretable choice — and isotonic's
  flexibility advantage hasn't manifested in held-out evidence.
- The fit-slice fallback is necessary for unit tests and small
  research datasets (calib < 500 rows). Inner-val ECE on < 100 rows
  is too noisy to drive method selection.
- A time-based inner split was rejected in favour of random shuffle:
  time-isolation is enforced at the OUTER 3-way split (the calib slice
  is itself a contiguous time window between train_cutoff and
  test_cutoff). Choosing between two 1-D calibration maps does not
  require additional within-slice time isolation; random shuffle
  produces a more uniform validation distribution.

**Empirical Validation:**
The full-parquet re-run (session 2026-05-12 f, post-fix) will show
whether the held-out criterion picks Platt on this dataset. If yes
(expected), `auto_selection_mode='held_out'`, `chosen_method='platt'`,
and test ECE remains in the 0.003-0.005 range. If isotonic still
wins on held-out (unexpected), the finding is real (the dataset has
non-sigmoidal distortion that survives held-out evaluation) and
we should investigate per-jurisdiction calibration.

**Rejected Alternatives:**
- K-fold cross-validation within the fit slice — rejected for v1
  (cleaner statistical estimate but K× the fit cost; the simple
  inner-val gives well-bounded ECE estimates at our sample sizes,
  ECE SE ≪ 0.001 at 40k val rows).
- Default to Platt unconditionally; only allow isotonic via explicit
  `method='isotonic'` — rejected. The staircase / strongly-distorted
  case is real (early-phase models, miscalibrated by design), and
  the auto-selector should be able to switch when held-out evidence
  is unambiguous. The `min_delta_ece` knob captures this preference
  without making the choice unidirectional.
- Block bootstrap by race_id inside the calib slice — rejected for v1
  (within-race outcomes ARE correlated, so race-aware bootstrap is
  the theoretically clean inner-val; but at 200k calib rows /
  ~17k races, the IID-with-race assumption is already mild and the
  added complexity isn't worth it for an automated selection knob).

## ADR-038: Skip-When-Calibrated Guard + Time-Ordered Inner-Val (Refines ADR-037)

**Date:** 2026-05-13
**Status:** Accepted (refines ADR-037 — does NOT supersede it).
Refined the same day to add a strictly-proper Brier co-criterion to
the skip check after the post-fix full-parquet run showed isotonic
genuinely winning on inner-val ECE while only redistributing
probability across bins (post-cal test ECE > pre-cal test ECE; test
Brier essentially unchanged).

**Context:**
ADR-037 fixed the auto-selector's selection criterion (held-out
inner-val ECE instead of fit-slice ECE), but the post-fix full-parquet
run revealed two remaining problems:

1. **Selection bias is fixed; degradation persists.** On the full
   parquet, the held-out auto-selector correctly identifies isotonic
   as the inner-val winner (delta ≈ 20× the min_delta_ece threshold).
   But the chosen calibrator still degrades test-slice ECE from
   ≈ 0.003 raw to ≈ 0.005-0.006 post-cal. Root cause: time-
   distribution shift. The model improves over time, so the calib
   slice (older years) is more miscalibrated than the test slice
   (newer years). Calibration fitted on the older window adds noise
   on the newer test data.

2. **No way to skip calibration entirely.** When raw scores are
   already well-calibrated probabilities (the meta-learner's output
   on this dataset is one such case), ANY learned calibration map
   adds variance without removing bias — the optimal calibration
   map is the identity. The previous selector could only choose
   between Platt and isotonic.

**Decision:**
Two complementary refinements to `Calibrator.fit` under `method='auto'`:

1. **Skip-when-calibrated guard (`skip_threshold_delta`, default 0.001;
   `brier_skip_delta`, default 1e-4):**
   The auto-selector computes raw inner-val ECE AND Brier alongside
   Platt and isotonic. Calibration applies only if the winning method
   beats raw on BOTH metrics by the configured deltas. Otherwise
   `chosen_method = 'identity'` and `predict_proba` returns raw
   scores clipped to [0, 1].

       method_winner = isotonic_or_platt_per_min_delta_rule
       winner_ece, winner_brier = (iso or platt) per the above rule
       ece_beats   = winner_ece   + skip_threshold_delta < raw_ece
       brier_beats = winner_brier + brier_skip_delta     < raw_brier
       chosen = method_winner if (ece_beats AND brier_beats) else 'identity'

   The Brier co-criterion is essential: ECE is a binned summary that
   noise can game by redistributing probability across bins (no real
   accuracy improvement, but ECE drops). Brier is strictly proper —
   its minimum is achieved only at the true distribution. Requiring
   both rules out the case observed on the full parquet, where
   isotonic genuinely won on inner-val ECE (delta ≈ 4× threshold)
   yet had no measurable Brier improvement on either inner-val or
   test, and produced strictly worse test ECE post-cal.

2. **Time-ordered inner-val support (`inner_val_indices` parameter):**
   `Calibrator.fit(scores, labels, inner_val_indices=...)` accepts
   explicit row indices to use as the inner-val slice instead of an
   internal seeded random shuffle. Callers should pass the
   time-ordered TAIL of the calib slice (e.g. last 20% of calib by
   `race_date`). The tail is the closest in-distribution proxy for
   the test slice, which immediately follows the calib slice in
   time. `validate_calibration.py` does this automatically.

   Selection mode reflects the source: `held_out_caller` for
   caller-supplied indices, `held_out` for seeded random shuffle,
   `fit_slice_fallback` for tiny calib slices.

Both calibrators are STILL re-fit on the full calib slice and metrics
for both (plus identity) are persisted, so the selection is fully
auditable. `metadata.json` adds an `identity` entry to both `metrics`
and `inner_val_metrics`.

**Rationale:**
- The skip guard is the right answer when raw is already calibrated:
  the optimal calibration map IS the identity, and learning a Platt
  or isotonic map can only add variance. The 0.001 default threshold
  matches `auto_min_delta_ece` — both express "the differences below
  this aren't statistically meaningful at our sample sizes."
- Time-ordered inner-val is the right answer when there's drift
  between calib and test windows. Random shuffle inside the calib
  slice gives an inner-val that looks like the calib slice; the tail
  gives an inner-val that looks like the test slice. The latter is
  what we actually need to predict generalisation.
- Combining both is statistically sound: if a calibrator's apparent
  win on the random-shuffle inner-val disappears on the time-ordered
  inner-val (because that's where drift is visible), the skip guard
  catches the case and falls back to identity. Either fix alone is
  insufficient; together they are.
- Identity mode persists Platt and isotonic anyway so the same
  artifact directory can be re-loaded, inspected, and the choice
  re-litigated against newer data without re-training.

**Empirical Validation:**
First full-parquet re-run (ECE-only skip) showed isotonic still chosen
because iso beat raw on inner-val ECE by 4-5× threshold — but iso's
inner-val Brier improvement was vanishingly small. Test slice
confirmed: post-cal ECE 0.0058 / 0.0048 vs pre-cal 0.0031 / 0.0034
(speed_form / meta), and Brier essentially unchanged. The "win" was
mostly bin redistribution.

Post-Brier-co-criterion full-parquet re-run (the shipped behavior):
- `speed_form`: `chosen_method='identity'` — isotonic's Brier
  improvement on inner-val was 2.8e-6 (≪ 1e-4 threshold), so skip
  fires. Test pre-cal = post-cal exactly: ECE 0.0031, Brier 0.0732,
  log-loss 0.2623. No degradation.
- `meta_learner`: `chosen_method='isotonic'` — isotonic's inner-val
  Brier beats identity by 1.6e-4 (just above the 1e-4 threshold).
  This is a genuine (if small) accuracy improvement, and it transfers
  to the test slice: post-cal Brier 0.06788 vs pre-cal 0.06789
  (better by 1.6e-5), post-cal log-loss 0.23693 vs pre-cal 0.23704
  (better by 1.1e-4). Both strictly proper scores improve. Test ECE
  moved the "wrong" way (0.0034 → 0.0048) — but ECE is the binned
  summary that the Brier co-criterion was put in place to override.
  This is the guard working as designed: ECE alone said "apply
  isotonic, big win"; the proper scores said "tiny but real win";
  the guard correctly applied the tiny-but-real win and ignored
  the bin-summary signal.

The guard is therefore a PRECISION FILTER, not a blanket disable: it
correctly applies calibration when proper scores agree there is a
real win (meta_learner) and correctly skips when they don't
(speed_form). On future datasets where isotonic genuinely improves
both inner-val ECE AND inner-val Brier over raw by the configured
thresholds, calibration SHOULD be applied.

**Relationship to ADR-037:**
ADR-037 stays in force — held-out inner-val is still the criterion
for choosing BETWEEN Platt and isotonic. ADR-038 extends the choice
set to include identity, and lets the caller control which rows
constitute the inner-val.

**Rejected Alternatives:**
- Always force `method='identity'` for downstream models known to be
  well-calibrated (e.g. meta-learner) — rejected. The whole point of
  the auto-selector is that calibration need can change with data,
  model retraining, and jurisdiction. A static rule misses the case
  where the model becomes miscalibrated mid-season and needs a
  Platt/isotonic correction.
- Use the test slice itself as the inner-val (target leakage) —
  obviously rejected.
- Walk-forward calibration (re-fit calibrator on a rolling window of
  the most recent N races) — deferred to Phase 5 / production. This
  is the right answer for production but requires online infra.
  ADR-038's tail-of-calib inner-val is the offline approximation.


---

## ADR-039: Phase 5a Bet-Type Scope — Win + Exacta + Trifecta + Superfecta Only

**Date:** 2026-05-13
**Status:** Accepted

**Context:**
Phase 5a's `BetCandidate` schema and EV calculator support a subset of pari-mutuel bet types. The PL ordering module produces marginals for any combinatorial bet; the question is which ones the calculator should *emit* given current data and model coverage.

**Decision:**
Phase 5a supports `WIN`, `EXACTA`, `TRIFECTA`, `SUPERFECTA` only. The schema validator (`BetCandidate.validate_selection`) rejects `PLACE`, `SHOW`, `PICK3`, `PICK4`, `PICK6` with `ValueError`.

**Rationale:**
- **Place/Show payouts** are not deterministic functions of (selection, win_probs, decimal_odds). The payout depends on which OTHER horses finish in the money AND on the live pool composition (how much is bet on each potential placing horse). Without live pari-mutuel pool composition data, we can only compute the *probability* of placing/showing, not the EV. Closed-form helpers `place_prob` and `show_prob` are added to `plackett_luce.py` for future use, but Phase 5a does not emit Place/Show candidates.
- **Pick 3/4/6** are cross-race bets. Correct probability requires modelling the shared latent track state per card (Master Reference §206-209). The current ordering module treats races as independent. Adding cross-race correlation is a separate phase.

**Rejected Alternatives:**
- Emit Place/Show with a uniform-pool-composition approximation — rejected; the approximation is systematically wrong for the favourite-vs-longshot place pool, exactly where the largest +EV opportunities exist.
- Emit Pick N with multiplied marginals — rejected; same independence assumption that Harville carries, which CLAUDE.md §2 prohibits.

**When to revisit:**
When live tote ingestion is built AND when a card-level latent-state model is trained.

---

## ADR-040: EV Engine Odds Are Source-Agnostic; Validation Script Picks Mode

**Date:** 2026-05-13
**Status:** Accepted

**Context:**
The EV engine consumes `(win_probs, decimal_odds)`. The system has multiple potential odds sources:
- Morning-line from parsed PDFs (available now, biased low-information prior).
- Historical `odds_final` from PP lines (the actual closing market price for past races — most accurate possible for backtests).
- Live tote-board odds at post-time (most accurate for live, but not yet ingested).

**Decision:**
The EV calculator (`compute_ev_candidates`) accepts `decimal_odds: np.ndarray` as an input. It does not know or care how the array was produced. The validation script (`scripts/validate_phase5a_ev_engine.py`) supports two modes:
- `--mode backtest` (default): uses `odds_final` from the parquet (real closing market prices). The validation script restricts this mode to WIN bets because exotic per-permutation historical odds are not in the parquet.
- `--mode live` (stub in 5a): would use morning-line from a parsed RaceCard. Raises `NotImplementedError` until the PDF-ingestion-to-EV integration is built.

A future `live tote` mode is a third value added without changing the calculator signature.

**Rationale:**
Decoupling the calculator from the odds source has three consequences:
1. **Backtests use the most accurate available data** — closing market prices are exactly what the bet would have settled at.
2. **The live path is a one-config-flag swap** when live tote becomes available; no calculator changes needed.
3. **Tests are simpler** — synthetic odds vectors are passed directly without mocking any odds-source dependency.

**Rejected Alternatives:**
- Embed odds source selection inside the calculator with a `source` argument — rejected; it pushes ingestion concerns into a math module that should be agnostic.
- Use de-vigged morning-line as the canonical odds for both backtest and live — rejected; ML is a strictly worse estimator than `odds_final` for backtests and would hide model performance behind a noisy input.

**Note on smoke-run data quality (carried forward to Phase 5b):**
The first backtest smoke run (5% sample → 11,574 test rows / 8,717 races) produced 10,663 +EV candidates with mean edge 0.705 and mean EV/$ of 34.9 — both implausibly large. Root cause: the parquet's `odds_final` column contains extreme values (likely 99/999 placeholders for scratched or rare-payout horses) which, when fed into `1/odds`, produce near-zero market probabilities and therefore enormous nominal edges against ANY non-zero model probability. The EV engine is computing what it was asked; the data is the problem. Phase 5b should add an upper bound on `decimal_odds` at the validation-script level (suggested cap: ~100, dropping ~scratched/placeholder rows) before the portfolio optimiser will produce sensible results.

---

## ADR-041: CVaR-LP Portfolio Optimiser — Rockafellar-Uryasev with Vectorised PL Scenarios, Per-Race Scope

**Date:** 2026-05-13
**Status:** Accepted

**Context:**
Phase 5b needs to allocate capital across a list of `BetCandidate` objects under a CVaR risk constraint, per CLAUDE.md §2 ("CVaR-constrained, not mean-variance alone") and ADR-005. The optimiser's three coupled design choices are: (a) the optimisation form, (b) how scenarios are generated, and (c) the scope of a single solve.

**Decision:**

**a. Rockafellar-Uryasev (2000) CVaR LP via `scipy.optimize.linprog(method="highs")`.**
Decision vector `x = [f_0, …, f_{n-1}, v, u_0, …, u_{S-1}]`, length `n + 1 + S`, where `f_i` is the stake fraction of bankroll on candidate i, `v` is the VaR proxy (free), and `u_s ≥ 0` are per-scenario shortfall slacks. The LP is:

```
minimise   −Σ_i f_i · ev_per_dollar_i               (negate to maximise EV)
subject to
   −payoff.T · f − v − u_s ≤ 0       for s = 1..S    (scenario shortfall, S rows)
   v + (1/(α·S)) · Σ_s u_s ≤ M                       (CVaR cap, 1 row)
   Σ_i f_i ≤ 1                                       (budget, 1 row)
bounds  0 ≤ f_i ≤ min(¼-Kelly_i, max_bet_fraction)
        v free; u_s ≥ 0
```

where `α = cvar_alpha` (default 0.05), `M = max_drawdown_pct` (default 0.20), and the per-candidate upper bound uses `app.services.portfolio.sizing.kelly_fraction` and `apply_bet_cap` (ADR-002, ¼-Kelly + 3% cap). The CVaR at optimum is `v + (1/(α·S)) · Σ_s u_s`; we report it in USD as `bankroll × LHS` (i.e., `Portfolio.cvar_95` is dollar-denominated, NOT a fraction).

**b. Monte Carlo scenarios via the Gumbel-trick Plackett-Luce sampler.**
`app.services.ordering.plackett_luce.sample_orderings_vectorised(strengths, n_samples, rng)` returns an `(n_samples, n_horses)` int64 array of finishing orders. Implementation: `argsort(-(log_s + Gumbel(0,1)))`; the Yellott / Gumbel-Max identity makes a single argsort over a perturbed log-strength matrix equal in distribution to a single PL draw. One numpy call per race instead of a python loop over scenarios.

For each candidate i in scenario s, the per-$1 payoff is `+(decimal_odds_i − 1)` on a hit and `−1` on a miss. Hit logic per bet type:
- WIN: `order[0] == sel[0]`
- EXACTA: `order[0:2] == sel[0:2]`
- TRIFECTA: `order[0:3] == sel[0:3]`
- SUPERFECTA: `order[0:4] == sel[0:4]`

Per-race seeding (`np.random.default_rng([seed, hash(race_id) & 0xFFFFFFFF])`) makes the result invariant to candidate ordering and stable when callers add/remove races.

**c. Per-race scope.**
A single `optimize_portfolio(...)` call operates on candidates from one race. The validation script (`scripts/validate_phase5a_ev_engine.py --optimize`) loops over races and calls the optimiser once per race. Cross-race correlation is set aside, mirroring ADR-039's per-race exotic scope.

**d. `n_scenarios = 1000` default.**
Empirical Monte Carlo error for CVaR at α=0.05 scales as `~1/sqrt(α·S)`. With S=1000 that is `~1/sqrt(50) ≈ 14%` per-portfolio. Down-sampling to S=200 nearly doubles the error; up-sampling to S=5000 cuts it to ~6% but multiplies the linprog row count by 5 (LP rows are dominated by the S scenario rows). 1000 is the empirical sweet spot for the per-race optimiser to run in ≲ 50 ms on a 10-horse / 50-candidate field. Callers wanting tighter tails should up `n_scenarios` on important races.

**e. VaR/CVaR reported in USD.**
`Portfolio.var_95` and `Portfolio.cvar_95` are `bankroll * v` and `bankroll * cvar_lhs` respectively. The schema field type is `float` (no unit), but the convention is dollars throughout the Phase 5b code path. A future Phase 6 frontend should display them as `$1,234.56`.

**Rationale:**
- Rockafellar-Uryasev is the canonical LP form for CVaR; it converts a non-convex risk objective into a single LP that scipy can solve in milliseconds. Two other formulations were considered:
  1. Direct Monte Carlo sample-average optimisation with empirical CVaR computed per iteration: rejected because it requires a non-LP solver (`scipy.optimize.minimize`) and is non-deterministic when the simplex tie-breaks.
  2. Closed-form correlated-Kelly: rejected because PL exotic correlations have no simple closed form across bet types.
- Per-race scope is the cleanest interpretation of "correlation" given current data: WIN bets on the same race are perfectly correlated through the same finishing order; EXACTAs share the same denominator structure; cross-race bets are independent under the current independence assumption (no card-level latent state model yet — ADR-039).
- Gumbel-trick vectorisation is 50-200× faster than calling `sample_ordering` in a python loop for typical field sizes; the LP itself is microseconds, so the sampler dominated before this optimisation.
- N=1000 is conservative for production. The smoke tests use N=500-800 to stay under 1 s per race; the validation-script default is the larger value.

**Rejected Alternatives:**
- **Mean-variance** (Markowitz on portfolio variance) — rejected per CLAUDE.md §2 / ADR-005. Mean-variance penalises upside symmetric to downside, which is wrong for positively-skewed exotic payouts.
- **Per-bet Kelly applied sequentially** — rejected per ADR-005. Ignores correlation; over-allocates to correlated exactas.
- **Full-Kelly (fraction=1.0) bounds in the LP** — rejected per ADR-002. The 3% cap is unconditional; the LP's per-candidate upper bound is `min(¼-Kelly, max_bet_fraction)`.
- **Cross-race joint optimisation** — deferred. Requires a shared latent track-state model (Master Reference §206-209) to be meaningful; without it, races are independent and per-race solves give the same allocation as a single joint solve.

**When to revisit:**
- When cross-race correlation modelling lands (card-level latent state, ADR pending).
- When live pari-mutuel pool data is available — the LP should then include pool-impact constraints on `f_i` (currently delegated to the EV calculator's pre-LP single-step approximation).

---

## ADR-042: Server-Side Inference Pipeline + Cards/Portfolio API Endpoints

**Date:** 2026-05-13
**Status:** Accepted
**Phase:** Stream A (post-Phase 5a, unblocks Phase 6 frontend)

**Context:**
The Phase 6 frontend declares optional `model_prob` / `market_prob` / `edge`
fields on each `HorseEntry` and a `Portfolio` shape mirroring
`app/schemas/bets.py::Portfolio`. Without a server-side route that turns a
persisted `RaceCard` into calibrated probabilities and bet recommendations,
the frontend runs in MOCK mode. Stream A closes that gap.

**Decision:**

1. **Reusable inference module — `app/services/inference/pipeline.py`.**
   * `InferenceArtifacts` dataclass aggregates the five sub-models +
     meta-learner + meta-calibrator. `InferenceArtifacts.load(models_dir)`
     is tolerant of missing optional sub-models (Speed/Form, Connections,
     Market) — they fall back to the 0.5 neutral per ADR-026 — but is
     strict about the meta-learner and meta-calibrator since those are
     load-bearing.
   * `build_inference_features(card)` synthesises the SAME column schema
     `prepare_training_features` produces (per-horse priors + field-relative
     normalisation + categoricals). The training pipeline's `groupby.shift(1)`
     trick is unnecessary at inference time because `HorseEntry.pp_lines`
     already excludes today's race — every PP row IS a prior result.
   * `infer_calibrated_win_probs(...)` reproduces the validation-script
     scoring sequence (`_score_test_slice`): orthogonalised sub-model stack
     → meta-learner.predict_proba → meta-calibrator.predict_softmax. Per-race
     softmax guarantees the vector sums to 1.0.
   * `analyze_card(...)` is the end-to-end orchestrator. In live mode it
     emits WIN candidates only — exotic per-permutation odds are not
     available pre-race (per ADR-040). Decimal odds come from
     `HorseEntry.morning_line_odds`, capped at `max_decimal_odds=100.0`
     (the Phase 5b smoke-finding mitigation).

2. **GET endpoints, not POST.**
   * `GET /api/v1/cards/{card_id}` and `GET /api/v1/portfolio/{card_id}` both
     use GET because inference is deterministic given (card, artifacts,
     query parameters). GET is bookmarkable, cacheable by intermediaries,
     and lets the frontend swap parameters via a URL change without form
     submission. POST would imply mutation we don't perform.

3. **In-memory per-card cache for hydrated probabilities.**
   * `app.state.inference_cache` is a dict keyed by `card_id` holding the
     per-race calibrated probability vectors. Repeated GETs reuse the
     cached vectors — for a 10-race card this saves ~50 sub-model + meta
     + calibrator invocations.
   * Cache is process-local (lost on restart). Acceptable for paper-trading
     scale; production would swap in Redis with a TTL. ALTERNATIVES
     CONSIDERED: storing the probabilities in the DB alongside the parsed
     card — REJECTED because retraining invalidates them and the DB write
     path is on the ingestion hot loop. In-memory dict is the right
     latency/freshness trade.

4. **Multi-race portfolio aggregation in `/portfolio`.**
   * The frontend's bet-execution ticket displays one flat list of
     recommendations across the whole card. `analyze_card` returns one
     `Portfolio` per race; the endpoint aggregates them by concatenating
     `recommendations`, summing `expected_return` and `total_stake_fraction`
     (capped at 1.0), and taking the MAX of per-race VaR/CVaR as the
     conservative risk display. ALTERNATIVES CONSIDERED: returning the
     list of per-race Portfolios — REJECTED because it makes the frontend
     ticket render logic per-race when the user mental-model is per-card.

5. **Interim portfolio constructor stands in for the missing Phase 5b LP.**
   * `build_portfolio_from_candidates` uses each candidate's pre-capped
     1/4 Kelly fraction (ADR-002) and scales the whole vector down by a
     scalar `k ∈ (0, 1]` until the Monte-Carlo CVaR_α loss ≤
     `max_drawdown_pct × bankroll`. This is NOT the Rockafellar-Uryasev
     LP that Phase 5b will land — it's a tractable placeholder. The
     interface (`list[BetCandidate] → Portfolio`) matches what the LP
     will consume, so swapping is a single-file replacement.

6. **Card persistence reuses the existing live ingestion DB.**
   * `app/db/persistence.py` already converts an `IngestionResult` to ORM
     rows. Stream A adds `load_card(session, card_id)` which rebuilds the
     Pydantic `RaceCard` from the ORM tree via `selectinload` eager joins.
     The ingest endpoint now sets `IngestionResult.card_id` to the DB
     primary key (as string) so the frontend can roundtrip.

**Rejected Alternatives:**
* **POST /portfolio with body parameters.** Rejected — see point 2.
* **Compute probabilities at ingestion time + persist them.** Rejected —
  see point 3. Calibration / retraining decouples model lifecycle from
  card lifecycle.
* **Hold the inference path inside the validation script only and call
  out to it as a subprocess.** Rejected — fragile, slow, and the API
  needs first-class access for caching and error handling.

**Carry-forward (post-merge with Phase 5b):**
The interim portfolio constructor in `pipeline.build_portfolio_from_candidates`
was written before Stream A's worktree saw the Phase 5b CVaR LP (ADR-041).
A follow-up should swap it for `app.services.portfolio.optimizer.optimize_portfolio`
— the interface (`list[BetCandidate] → Portfolio`) matches, so this is a
one-line replacement inside `analyze_card`. Endpoint signatures and tests
do not need to change.

---

## ADR-043: Feedback Loop — Outcomes Logging + Portfolio-Level CUSUM Drift

**Date:** 2026-05-13
**Status:** Accepted

**Context:**
Master Reference §4 Layer 7 ("Feedback + Online Learning") requires a
persisted record of what actually happened on every race we bet, plus an
online drift detector that watches realised vs. expected portfolio
performance. Without this loop the model can silently miscalibrate for
weeks before we notice via P&L alone. Phase 4 already shipped a calibration-
residual CUSUM (`app/services/calibration/drift.py`, ADR-036) but at the
single-event prediction level — we still needed the portfolio-level analogue.

**Decision:**

1.  **Two new ORM tables in `app/db/models.py`:**

    `RaceOutcome` — official chart / manual result for one race.
        Idempotency anchor: `race_dedup_key` (UniqueConstraint). Same
        SHA-256 race-key convention as the Phase 0 master DB so manual
        entries and downstream official-chart imports collapse to one row.

    `BetSettlement` — one row per settled `BetRecommendation`. Stores both
        recommendation-time and settlement-time decimal odds so future drift
        attribution can split realised error into "model error" vs.
        "market move" components. No FK from `BetSettlement` to a
        recommendations table yet — `bet_recommendation_id` is a nullable
        slot reserved for when Stream A persists recommendations.

2.  **PnL convention (`pnl = payout − stake`):**
    * Lose: payout = 0  → pnl = −stake (always negative).
    * Win:  payout = stake · decimal_odds_at_settlement → pnl = stake · (odds − 1).

    Convention chosen so that "expected_value" (already a per-unit-stake
    number in `BetCandidate.expected_value`) and realised pnl live in the
    same units (dollar P&L on the stake), making the drift z-score
    straightforward: z_i = (pnl_i − E[pnl_i]) / σ_i.

3.  **Portfolio CUSUM — same defaults as calibration drift (k=0.5, h=4):**

    One set of σ-unit thresholds across every drift detector in the system
    keeps the operator's mental model simple. ARL₀ ≈ 168 (per Hawkins &
    Olwell 1998) is the same in-control regime the calibration detector
    accepts, and switching to h=6 (ARL₀ ≈ 1290) for noisier streams is a
    one-line config change.

    Per-bet z-score:
        σ_i = stake_i · √(p_i(1 − p_i)) · decimal_odds_at_settlement_i
        z_i = (pnl_i − E[pnl_i]) / σ_i

    The variance proxy is the Bernoulli closed-form stdev of (pnl | bet placed),
    which is correct for win-or-lose bets and reduces to `|stake|` as a
    conservative fallback when prob/odds are unavailable.

4.  **Two-sided alarms — both directions matter:**
    * `s_plus` crossing h  → realised exceeded expected (model under-
      confident OR a variance burst on the upside). Operationally:
      review whether stakes should be enlarged.
    * `s_minus` crossing h → realised under expected (model over-confident,
      the canonical drift failure). Operationally: trigger retrain /
      recalibration.

    Both directions latch (no re-fire after `triggered=True` until
    `reset()` is called) — matches the calibration drift behaviour.

5.  **Settlement scope = Phase 5a / ADR-039 bet types only.**
    Only WIN/EXACTA/TRIFECTA/SUPERFECTA are settleable. PLACE/SHOW/PICK-n
    raise `ValueError` so a future caller does not silently produce
    wrong settlements. Phase 5a's `BetCandidate` validator already
    enforces this at construction time; `settle_bets` carries a
    redundant guard for callers using `model_construct` bypasses.

**Rationale:**
- **Idempotency on `race_dedup_key`**: matches the Phase 0 dedup contract
  ("running twice on the same source produces the same DB state").
  Allows manual entry, official chart, and Equibase scrape to all target
  the same row without collision logic.
- **CUSUM-z over Bayesian change point / rolling-mean-test**: same
  rationale as ADR-036 (the calibration drift ADR). Bayesian CPs add a
  prior we cannot defend ("how often do we EXPECT the model to drift?")
  and rolling-mean t-tests have terrible ARL behaviour on streams with
  serial correlation. CUSUM is the standard for online change-point on
  streamed Bernoulli-derived residuals.
- **Two redundant decimal-odds snapshots** (recommendation-time +
  settlement-time): per-step they look identical (no live tote yet),
  but the second slot is the future-proof hook for when live odds land.
  Cheap to populate, expensive to retrofit if omitted.

**Rejected Alternatives:**
- One unified `outcomes_and_bets` table — rejected; outcomes are
  immutable race-level facts (one row per race), settlements are
  per-bet, and one race has 0..N settlements. Two tables keep the
  cardinalities clean.
- Bayesian online change-point detection (BOCPD) — rejected for v1
  (same reasoning as ADR-036). Worth revisiting if the operator wants
  posterior probabilities of drift rather than alarms.
- Mean-of-residual t-test in a rolling window — rejected; ignores serial
  correlation in race-to-race PnL and has unstable ARL.
- Storing pnl as `stake · (decimal_odds − 1)` directly for both win and
  loss cases (i.e. negative profit when loss) — equivalent to the chosen
  convention, but `pnl = payout − stake` is more explicit and reads off
  the BetSettlement row without computing intermediate "edge" numbers.

**Open follow-ups:**
- `BetSettlement.bet_recommendation_id` is currently nullable. When
  Stream A adds a `BetRecommendation` table, retrofit a non-null FK
  via a follow-up migration.
- The Bernoulli-σ in `portfolio_pnl_zscore` is single-event; for
  correlated baskets (e.g. exotic permutations within the same race)
  we would replace with a copula-derived σ. Out of scope for Layer 7
  but flagged here for Phase 6+.

---

## ADR-044: Rolling-Window Retraining Is a Standalone Script, Drift-Triggered by Default

**Date:** 2026-05-13
**Status:** Accepted

**Context:**
Master Reference §4 Layer 7 ("Feedback + Online Learning") calls for the sub-models to be retrained on a rolling window — for example, the last 3 years — dropping the oldest data and incorporating the newest. The retrain is coordinated with the calibration drift detector: when the live CUSUM (Phase 4, ADR-036) alarms, a retrain should be triggered, and absent that signal a scheduled cadence still keeps the model fresh.

This raises two design questions:

1. **What is the default rolling-window length?**
2. **Where does the retraining live — a long-running service or a standalone script?**

**Decision:**

### 1. Rolling-window default: **3 calendar years**

`scripts/rolling_retrain.py --window-years 3`. The window is half-open: `[as_of_date − 3y, as_of_date)`. Rows on the as-of date itself are deliberately excluded so future races (the ones we'd actually bet on) never appear in training.

Three years is the suggestion in Master Reference §183 and lands on a defensible operational compromise:
- It spans **two full racing seasons plus a one-year buffer**, so the model has seen the same horses and connections at multiple ages and the calendar-seasonal pattern (Triple Crown trail, summer turf meets, Breeders' Cup prep) appears twice.
- It is **long enough to fit the empirical-Bayes connections model** — three years of jockey-trainer pairings shrinks meaningfully toward each connection's true win rate at the per-jurisdiction `min_jockey_starts` / `min_trainer_starts` defaults.
- It is **short enough to drop pre-COVID racing economics** (purse structures, takeout rates, jockey populations) from active training as those become structurally distant — pre-2022 racing in the US is materially different from post-2022 racing, and a 3y window will roll those out within the next refit cycle.

The window length is configurable per-invocation, so a researcher can sweep it without code changes.

### 2. Retraining mode: **drift-triggered by default, scheduled cadence opt-in**

Two trigger modes are supported:

- **Drift-triggered (default operational mode):** the live monitor writes a JSON marker (`save_drift_state` in `app/services/calibration/drift.py`) every time it processes a batch of (prediction, label) pairs. The retrain CLI reads that marker via `--skip-if-no-drift`. When the CUSUM has not alarmed, the script exits with code 2 — a distinctive no-op signal that lets a cron job branch on "calibration is fine, do not spend the GPU cycles." When it has alarmed, the script proceeds normally and exits 0.
- **Scheduled cadence:** the same script run without `--skip-if-no-drift` always retrains. This is the fallback path for environments where the drift detector is not yet wired (e.g. before Stream B is integrated) and the simpler "retrain every Sunday" cron pattern.

The two-mode design keeps the alerting layer (CUSUM) decoupled from the retraining layer (sub-models + meta + calibrator). Either can change independently as long as the JSON marker schema stays stable.

### 3. Execution mode: **standalone script, not a service**

`scripts/rolling_retrain.py` follows the same pattern as `scripts/bootstrap_models.py`, `scripts/validate_calibration.py`, and `scripts/db/*`: a single-shot CLI that imports from `app/` but does not require the FastAPI app to be running.

Reasons:
- **CLAUDE.md §11** explicitly mandates the Phase-0 pipeline scripts stay standalone. The same constraint applies here for the same reason: training jobs run on the largest available box (or a fresh GPU node), not on the API host.
- **Cron-locality.** A cron job invokes a shell command; it does not negotiate with a long-running service. A standalone script is the cron-native shape.
- **No FastAPI dependency.** The training pipeline imports `app.services.models.*` and `app.services.calibration.*`, none of which require an `asyncio` event loop. Lifting this into a service would mean either a synchronous endpoint that ties up a worker for the duration of the train (bad), or an async background task with all the lifecycle/cancellation/persistence concerns that come with that.
- **Restartability.** The script writes artifacts atomically to a self-contained `output_dir`; if it crashes halfway through, the operator deletes the directory and re-runs. There is no shared state to roll back.

### 4. Output layout: `models/rolling/<as_of_date>/`

Mandated structure inside the directory:

```
models/rolling/2026-05-13/
├── speed_form/            # SpeedFormModel.save() artifact
├── connections/           # ConnectionsModel.save() artifact
├── market/                # MarketModel.save() artifact
├── meta_learner/          # MetaLearner.save() artifact
├── calibrator/            # Calibrator.save() artifact (Platt/isotonic/identity)
└── report.json            # top-level summary; see below
```

Pace / sequence sub-model directories are omitted by default — they are stubs (ADR-026) until their unblock conditions are met. Their entries still appear in `report.json` under `sub_models` with `"stub": true` so the report is self-describing.

The as-of-date is used verbatim in the directory name so that:
- The newest run is always lexicographically last under `models/rolling/`.
- Diffing two runs (drift attribution) is a directory walk, not a database query.
- The same as-of-date never produces two competing artifacts; re-running overwrites in place.

`report.json` carries the required keys consumed by Stream A's inference pipeline and Stream B's drift monitor:
`n_train_rows`, `n_calib_rows`, `n_test_rows`, `sub_models_trained`, `meta_learner_ece`, `meta_learner_brier`, `meta_learner_logloss`, `as_of_date`, `window_years`.

**Rationale:**

The drift-triggered default plus the standalone-script execution mode means a single cron entry covers both shapes:

```cron
# Every 6 hours: only spend cycles if the live monitor alarmed.
0 */6 * * * cd $REPO && python scripts/rolling_retrain.py \
    --parquet data/exports/training_latest.parquet \
    --as-of-date $(date +%Y-%m-%d) \
    --output-dir models/rolling/$(date +%Y-%m-%d) \
    --skip-if-no-drift

# Weekly belt-and-suspenders refit regardless of drift state.
0 3 * * 0 cd $REPO && python scripts/rolling_retrain.py \
    --parquet data/exports/training_latest.parquet \
    --as-of-date $(date +%Y-%m-%d) \
    --output-dir models/rolling/$(date +%Y-%m-%d)-weekly
```

Both lines call the same script with the same artifact layout; the only difference is the drift gate.

**Rejected Alternatives:**
- **5-year window:** rejected; pre-COVID racing economics distort the post-2022 distribution enough that the marginal training data is no longer drawn from the same population as the deployment data. Three years already includes two full Triple Crown cycles.
- **1-year window:** rejected; too short for the connections empirical-Bayes shrinkage to land on stable rates, and a single Northern-Hemisphere season fails to teach the model the cross-season pattern.
- **Retraining as an async FastAPI background task:** rejected; couples training cadence to the API process lifecycle. A multi-hour training job inside the request-handler container is the antipattern this project explicitly avoids (CLAUDE.md §11).
- **Always retrain on cron, ignore drift state:** rejected; wastes compute and re-introduces calibrator churn for nominal random variation. The drift detector exists exactly to gate this. The cadence-only mode survives as the opt-in for environments where the drift detector is not yet wired.

**When to revisit:**
- When live racing data becomes available and the model is in active production — at that point, the 3-year window should be empirically validated against a 2-year and 4-year sweep on the live ledger (Stream B's outcomes log).
- If GPU cost dominates (e.g. when the Sequence/Transformer model trains), revisit whether a smaller window is acceptable — sequence layers have a steeper compute curve in N than LightGBM does.
- If the parquet ever grows enough that a 3y slice is too small for stable LightGBM fits in a niche jurisdiction, switch to an explicit `--min-rows` floor and pad with older data.

---

## ADR-045: Pareto-Frontier Endpoint + Real CVaR LP in Inference Pipeline

**Date:** 2026-05-13
**Status:** Accepted

**Context:**
Stream A's `analyze_card` shipped with an interim Monte-Carlo Kelly-scaling portfolio constructor (`build_portfolio_from_candidates`) because Stream A branched from the pre-Phase-5b main and didn't have access to the Rockafellar-Uryasev LP. With Phase 5b now merged, the inference pipeline can use the real LP. Separately, the frontend's UX is designed around a *risk/return curve* — the user wants to see the whole pareto frontier (return at each CVaR cap) and pick a point, not commit to a single drawdown number up front.

**Decision:**

1. **`analyze_card` defaults to the R-U LP** (`app.services.portfolio.optimizer.optimize_portfolio`). The interim constructor (`build_portfolio_from_candidates`) is preserved and reachable via `analyze_card(..., use_interim=True)` for callers that still want the placeholder semantics, but every default call path now goes through the LP. No API or schema change.

2. **New `analyze_card_pareto(...)` function** in `app/services/inference/pipeline.py`. Runs the expensive candidate-generation pipeline (sub-models + meta-learner + calibrator + EV calculator) ONCE, then re-solves only the per-race LP at each requested risk level. For an N-race card with K risk levels: 1 ML pass + N×K cheap LP solves (≲ 50 ms each). Returns `tuple[list[(risk_level, aggregated_card_portfolio)], n_candidates_total]`.

3. **New `GET /api/v1/portfolio/{card_id}/pareto`** endpoint. Query params mirror `/portfolio/{id}` plus `risk_levels` (comma-separated CSV, default `"0.05,0.10,0.15,0.20,0.25,0.30"`). Returns `ParetoFrontier` JSON. Aggregation rule matches ADR-042: concat recommendations across races, sum return/stake, max VaR/CVaR (conservative).

4. **Risk-level grid default = 6 points from 0.05 to 0.30** in 0.05 increments. Six points is enough to draw a curve without overwhelming the LP solver budget (6×N solves) or the visualization. Configurable per request — 1 to 12 values, each in (0, 1], strictly increasing after parse. Out-of-range / non-numeric / duplicate values → 400.

5. **GET (not POST)** for `/pareto`, same rationale as ADR-042: deterministic given inputs, bookmarkable, cacheable.

6. **`greenlet` pinned in `pyproject.toml`** because sqlalchemy[asyncio] needs it but on py3.13/3.14 the wheel-selection logic shifted and it sometimes drops out of the resolved set. Explicit pin matches the pre-existing 6 `tests/test_api/test_ingest.py` errors that Stream A flagged.

7. **New schemas `ParetoPoint` and `ParetoFrontier`** in `app/schemas/bets.py` (mirrored 1:1 in `frontend/lib/types.ts` by Stream Z).

**Rationale:**

- Re-using `analyze_card(optimize=False)` to compute the once-per-call candidate set keeps `analyze_card_pareto` thin: it just groups candidates by race, then loops the optimizer. The expensive feature engineering and meta-learner inference are not duplicated.
- The endpoint solves N×K LPs serially. For a typical card (8 races, 6 risk levels = 48 solves, ≲ 50 ms each) total inference cost is well under 5 s. Parallelising the inner loops would help on giant cards but is overkill for the demo path.
- The pareto frontier MUST be monotone non-decreasing in expected_return as risk loosens, because higher `max_drawdown_pct` is a strictly looser constraint on the same LP. We test this invariant (`test_pareto_monotone_in_expected_return`).
- Returning the full Portfolio per point (not just summary stats) means the frontend slider can switch points without a round-trip. Payload is small — even 6 portfolios × ~10 recommendations × ~250 bytes JSON = ~15 KB per response.

**Rejected Alternatives:**

- **Compute one portfolio per request, frontend polls 6 times.** Rejected: 6× the network cost for the same total work, awkward loading UX (the curve renders piecewise as fetches return).
- **Return only summary stats per point, lazy-fetch the bets when the user clicks.** Rejected: the additional click-time latency would break the "slide along the curve and watch the bets update" interaction model. Payloads are small enough that returning the full portfolio per point is the right trade.
- **Use a continuous slider with the LP re-solved on every drag tick.** Rejected: the LP isn't free (50 ms/race) and the user wouldn't perceive sub-0.05 granularity in CVaR anyway. Discrete 6-stop snap is the right resolution.
- **Risk-adjusted scalar objective (mean − λ·CVaR)** instead of a pareto sweep. Rejected: the user's mental model is "show me my options across the risk-return tradeoff"; collapsing to a scalar throws away the choice.

**When to revisit:**

- When the LP solve gets slower (e.g., Sequence-Transformer-derived strengths grow scenario count), revisit whether to parallelise per-race solves with `concurrent.futures`.
- When live tote ingestion lands, the pareto endpoint should pre-fetch live odds before each LP solve. The contract doesn't change.
- If users want different default risk-grid (e.g., 0.01-0.05 conservative-only or 0.20-0.50 high-roller), the `risk_levels` query param already covers it — no schema change needed.
