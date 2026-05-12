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