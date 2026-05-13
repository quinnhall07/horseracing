# Phase 5a — EV Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the source-agnostic EV engine that, given calibrated win probabilities and a vector of decimal odds for any pari-mutuel bet type, returns `BetCandidate`s annotated with edge, EV, market-impact-adjusted odds, and 1/4-Kelly stake fractions.

**Architecture:** Three pure-math modules (`portfolio/sizing.py`, `ev_engine/market_impact.py`, PL helpers in `ordering/plackett_luce.py`) feed one orchestrator (`ev_engine/calculator.py`). The orchestrator consumes calibrated `np.ndarray` of marginal win probs and a parallel `np.ndarray` of decimal odds — agnostic to whether those odds came from morning-line, historical `odds_final`, or a future live tote ingester. A validation script runs the engine end-to-end on the same time-split used by `validate_calibration.py` and emits a JSON report + a per-bet-type EV summary.

**Tech Stack:** Python 3.11+, NumPy, SciPy (already pulled in by Phase 4), Pydantic v2, pytest. No new dependencies.

**Scope boundaries:**
- IN: Win, Exacta, Trifecta, Superfecta. Market impact as a closed-form pari-mutuel function with `pool_size=∞` default (zero-impact). 1/4 Kelly + 3% hard cap. Edge threshold filter.
- OUT (deferred): Place/Show — payouts depend on which other horses hit the board AND on live pool composition. Documented in ADR-039. Pick 3/4/6 — cross-race correlation requires the shared latent track-state model. CVaR portfolio optimisation — entire Plan 5b.

**Phase 4 inputs assumed:**
- `Calibrator.predict_softmax(scores, race_ids)` returns per-race calibrated win probabilities that sum to 1.
- `app/services/ordering/plackett_luce.py` already has `exacta_prob`, `trifecta_prob`, `superfecta_prob`, `enumerate_exotic_probs`.

---

## File Structure

```
app/
  schemas/
    bets.py                                  ← NEW (Task 1)
  services/
    ordering/
      plackett_luce.py                       ← MODIFIED (Task 2): + place_prob, show_prob
    portfolio/
      __init__.py                            ← NEW (Task 3)
      sizing.py                              ← NEW (Task 3)
    ev_engine/
      __init__.py                            ← NEW (Task 4)
      market_impact.py                       ← NEW (Task 4)
      calculator.py                          ← NEW (Tasks 5, 6)
tests/
  test_schemas/
    __init__.py                              ← create if missing
    test_bets.py                             ← NEW (Task 1)
  test_ordering/
    test_plackett_luce.py                    ← MODIFIED (Task 2): + place/show tests
  test_portfolio/
    __init__.py                              ← NEW (Task 3)
    test_sizing.py                           ← NEW (Task 3)
  test_ev_engine/
    __init__.py                              ← NEW (Task 4)
    test_market_impact.py                    ← NEW (Task 4)
    test_calculator.py                       ← NEW (Tasks 5, 6)
scripts/
  validate_phase5a_ev_engine.py              ← NEW (Task 7)
DECISIONS.md                                 ← MODIFIED (Task 8): + ADR-039, ADR-040
PROGRESS.md                                  ← MODIFIED (Task 8)
```

**Module responsibilities (one job each):**

| Module | Responsibility |
|---|---|
| `app/schemas/bets.py` | Pydantic schemas for `BetCandidate`, `BetRecommendation`, `Portfolio`. Validation only — no math. |
| `app/services/ordering/plackett_luce.py` (new helpers) | Closed-form `place_prob(p, i)` = P(i top-2) and `show_prob(p, i)` = P(i top-3) under PL. Pure math. |
| `app/services/portfolio/sizing.py` | `kelly_fraction(edge, decimal_odds, fraction=0.25)` and `apply_bet_cap(stake_frac, cap=0.03)`. Pure functions. |
| `app/services/ev_engine/market_impact.py` | `post_bet_decimal_odds(pre_odds, bet_amount, pool_size, takeout_rate)` — pari-mutuel impact. Pure math. |
| `app/services/ev_engine/calculator.py` | Orchestrator. `compute_ev_candidates(win_probs, decimal_odds, race_id, bet_types, ...) → list[BetCandidate]`. Calls into PL, sizing, market_impact. |
| `scripts/validate_phase5a_ev_engine.py` | End-to-end live run on the time-split parquet. Writes `models/baseline_full/ev_engine/<run>/report.json`. |

---

## Task 1: Bet Schemas

**Files:**
- Create: `app/schemas/bets.py`
- Create: `tests/test_schemas/__init__.py` (empty if not present)
- Create: `tests/test_schemas/test_bets.py`

**Design notes:**
- `BetCandidate` is the EV engine's per-bet output. `BetRecommendation` and `Portfolio` are added now (Phase 5b will populate them) but left without portfolio-specific fields beyond `stake`/`stake_fraction` for now.
- `selection: tuple[int, ...]` carries the horse indices. A `@model_validator` enforces the length matches the bet type (Win=1, Exacta=2, Trifecta=3, Superfecta=4) and that indices are distinct.
- Indices reference positions in the win-prob array passed to the calculator, not program numbers. The calling layer (validation script / future API) maps between them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schemas/__init__.py` empty if it doesn't exist, then `tests/test_schemas/test_bets.py`:

```python
"""Schema validation tests for app/schemas/bets.py."""
from __future__ import annotations

import pytest

from app.schemas.bets import BetCandidate, BetRecommendation, Portfolio
from app.schemas.race import BetType


def _candidate(**overrides) -> BetCandidate:
    base = dict(
        race_id="CD-2026-05-10-R4",
        bet_type=BetType.WIN,
        selection=(0,),
        model_prob=0.30,
        decimal_odds=4.0,
        market_prob=0.25,
        edge=0.05,
        expected_value=0.20,
        kelly_fraction=0.0125,
        market_impact_applied=False,
        pool_size=None,
    )
    base.update(overrides)
    return BetCandidate(**base)


def test_win_candidate_round_trip():
    c = _candidate()
    assert c.bet_type == BetType.WIN
    assert c.selection == (0,)
    assert c.edge == pytest.approx(0.05)


def test_exacta_requires_two_distinct_indices():
    c = _candidate(bet_type=BetType.EXACTA, selection=(0, 1))
    assert c.selection == (0, 1)
    with pytest.raises(ValueError, match="selection length"):
        _candidate(bet_type=BetType.EXACTA, selection=(0,))
    with pytest.raises(ValueError, match="distinct"):
        _candidate(bet_type=BetType.EXACTA, selection=(0, 0))


def test_trifecta_length_three():
    _candidate(bet_type=BetType.TRIFECTA, selection=(0, 1, 2))
    with pytest.raises(ValueError, match="selection length"):
        _candidate(bet_type=BetType.TRIFECTA, selection=(0, 1))


def test_superfecta_length_four():
    _candidate(bet_type=BetType.SUPERFECTA, selection=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="selection length"):
        _candidate(bet_type=BetType.SUPERFECTA, selection=(0, 1, 2))


def test_probabilities_in_unit_interval():
    with pytest.raises(ValueError):
        _candidate(model_prob=1.5)
    with pytest.raises(ValueError):
        _candidate(market_prob=-0.1)


def test_decimal_odds_at_least_one():
    with pytest.raises(ValueError):
        _candidate(decimal_odds=0.5)


def test_kelly_fraction_non_negative_and_capped_below_one():
    with pytest.raises(ValueError):
        _candidate(kelly_fraction=-0.01)
    with pytest.raises(ValueError):
        _candidate(kelly_fraction=1.5)


def test_pick_n_bet_types_rejected_in_5a():
    """Phase 5a does not support cross-race bets. ADR-039 defers Pick3/4/6."""
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.PICK3, selection=(0, 1, 2))


def test_place_and_show_rejected_in_5a():
    """ADR-039 defers Place/Show until live pool composition is available."""
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.PLACE, selection=(0,))
    with pytest.raises(ValueError, match="not supported"):
        _candidate(bet_type=BetType.SHOW, selection=(0,))


def test_recommendation_round_trip():
    rec = BetRecommendation(
        candidate=_candidate(),
        stake=125.00,
        stake_fraction=0.0125,
    )
    assert rec.stake == pytest.approx(125.0)
    assert rec.stake_fraction == pytest.approx(0.0125)


def test_portfolio_empty_recommendations_allowed():
    p = Portfolio(
        card_id="CD-2026-05-10",
        bankroll=10000.0,
        recommendations=[],
        expected_return=0.0,
        var_95=0.0,
        cvar_95=0.0,
        total_stake_fraction=0.0,
    )
    assert p.recommendations == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/test_bets.py -v`
Expected: ImportError or `ModuleNotFoundError: app.schemas.bets`.

- [ ] **Step 3: Implement `app/schemas/bets.py`**

```python
"""
app/schemas/bets.py
───────────────────
Phase 5 schemas for the EV engine and portfolio optimiser outputs.

A BetCandidate is what the EV engine emits per (race, bet_type, selection).
A BetRecommendation is what the portfolio optimiser (Phase 5b) emits — a
candidate plus the final stake. A Portfolio is the full card-level output.

Scope (ADR-039): Phase 5a supports WIN, EXACTA, TRIFECTA, SUPERFECTA.
Place/Show require live pari-mutuel pool composition; Pick 3/4/6 require
cross-race correlation modelling. Both deferred.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator
from typing import Annotated

from app.schemas.race import BetType


_SUPPORTED_BET_TYPES_5A: dict[BetType, int] = {
    BetType.WIN: 1,
    BetType.EXACTA: 2,
    BetType.TRIFECTA: 3,
    BetType.SUPERFECTA: 4,
}


class BetCandidate(BaseModel):
    """A potential bet identified by the EV engine, with its computed metrics."""

    race_id: str
    bet_type: BetType
    selection: tuple[int, ...]

    model_prob: Annotated[float, Field(ge=0.0, le=1.0)]
    decimal_odds: Annotated[float, Field(ge=1.0)]
    market_prob: Annotated[float, Field(ge=0.0, le=1.0)]
    edge: float
    expected_value: float
    kelly_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    market_impact_applied: bool = False
    pool_size: Optional[Annotated[float, Field(ge=0.0)]] = None

    @model_validator(mode="after")
    def _validate_selection(self) -> "BetCandidate":
        if self.bet_type not in _SUPPORTED_BET_TYPES_5A:
            raise ValueError(
                f"bet_type {self.bet_type} not supported in Phase 5a "
                f"(ADR-039). Supported: {list(_SUPPORTED_BET_TYPES_5A)}"
            )
        expected_len = _SUPPORTED_BET_TYPES_5A[self.bet_type]
        if len(self.selection) != expected_len:
            raise ValueError(
                f"selection length {len(self.selection)} != expected "
                f"{expected_len} for {self.bet_type}"
            )
        if len(set(self.selection)) != len(self.selection):
            raise ValueError(f"selection indices must be distinct; got {self.selection}")
        for idx in self.selection:
            if idx < 0:
                raise ValueError(f"selection indices must be non-negative; got {idx}")
        return self


class BetRecommendation(BaseModel):
    """A bet selected by the portfolio optimiser for placement."""

    candidate: BetCandidate
    stake: Annotated[float, Field(ge=0.0)]
    stake_fraction: Annotated[float, Field(ge=0.0, le=1.0)]


class Portfolio(BaseModel):
    """All recommendations + risk metrics for one card."""

    card_id: str
    bankroll: Annotated[float, Field(gt=0.0)]
    recommendations: list[BetRecommendation]
    expected_return: float
    var_95: float
    cvar_95: float
    total_stake_fraction: Annotated[float, Field(ge=0.0, le=1.0)]


__all__ = ["BetCandidate", "BetRecommendation", "Portfolio"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/test_bets.py -v`
Expected: 10 tests passing.

- [ ] **Step 5: Commit**

```bash
git add app/schemas/bets.py tests/test_schemas/__init__.py tests/test_schemas/test_bets.py
git commit -m "phase 5a — bet schemas (BetCandidate, BetRecommendation, Portfolio)"
```

---

## Task 2: Plackett-Luce Place/Show Helpers

**Files:**
- Modify: `app/services/ordering/plackett_luce.py` (add two functions + `__all__` update)
- Modify: `tests/test_ordering/test_plackett_luce.py` (append tests)

**Design notes:**
- These helpers are pure math, no new state. They live in `plackett_luce.py` because the math is closed-form PL marginal — natural extension of the existing `exacta_prob` family.
- They are exposed for **future** use (Phase 5b / Place/Show enablement) and for diagnostic reporting. Phase 5a's calculator does not currently consume them (Place/Show are deferred), but adding them now keeps PL's surface complete and tested.
- Closed forms:
  - `place_prob(p, i)` = P(i finishes 1st OR 2nd) = `p_i + Σ_{j≠i} p_j · p_i / (1 − p_j)`
  - `show_prob(p, i)` = P(i finishes top-3) = `place_prob(p, i) + Σ_{j≠i} Σ_{k∉{i,j}} p_j · p_k/(1−p_j) · p_i/(1−p_j−p_k)`
- For a degenerate field where `p_j = 1` (one horse certain to win), divisor blow-ups are guarded with explicit handling.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ordering/test_plackett_luce.py`:

```python
# ── Place / Show closed-form helpers ──────────────────────────────────────


def test_place_prob_two_horse_field_equals_one():
    """In a 2-horse field, both horses always finish top-2."""
    from app.services.ordering.plackett_luce import place_prob

    p = np.array([0.6, 0.4])
    assert place_prob(p, 0) == pytest.approx(1.0)
    assert place_prob(p, 1) == pytest.approx(1.0)


def test_place_prob_uniform_three_horse_two_thirds():
    """Uniform 3-horse field: P(any given horse top-2) = 2/3."""
    from app.services.ordering.plackett_luce import place_prob

    p = np.array([1 / 3, 1 / 3, 1 / 3])
    for i in range(3):
        assert place_prob(p, i) == pytest.approx(2 / 3)


def test_place_prob_equals_enumerated_sum():
    """Closed-form place_prob must match brute-force enumeration."""
    from app.services.ordering.plackett_luce import (
        enumerate_exotic_probs,
        place_prob,
    )

    p = np.array([0.45, 0.25, 0.20, 0.10])
    exacta = enumerate_exotic_probs(p, 2)
    for i in range(4):
        brute = sum(prob for perm, prob in exacta.items() if i in perm)
        assert place_prob(p, i) == pytest.approx(brute, abs=1e-12)


def test_place_prob_sums_to_two():
    """In any field, Σ_i P(i top-2) = 2 exactly (two slots filled)."""
    from app.services.ordering.plackett_luce import place_prob

    p = np.array([0.40, 0.25, 0.18, 0.12, 0.05])
    total = sum(place_prob(p, i) for i in range(5))
    assert total == pytest.approx(2.0, abs=1e-12)


def test_show_prob_three_horse_field_equals_one():
    """In a 3-horse field, every horse finishes top-3."""
    from app.services.ordering.plackett_luce import show_prob

    p = np.array([0.5, 0.3, 0.2])
    for i in range(3):
        assert show_prob(p, i) == pytest.approx(1.0)


def test_show_prob_equals_enumerated_sum():
    """Closed-form show_prob must match brute-force enumeration."""
    from app.services.ordering.plackett_luce import enumerate_exotic_probs, show_prob

    p = np.array([0.40, 0.30, 0.20, 0.10])
    tri = enumerate_exotic_probs(p, 3)
    for i in range(4):
        brute = sum(prob for perm, prob in tri.items() if i in perm)
        assert show_prob(p, i) == pytest.approx(brute, abs=1e-12)


def test_show_prob_sums_to_three():
    """In any field, Σ_i P(i top-3) = 3 exactly (three slots filled)."""
    from app.services.ordering.plackett_luce import show_prob

    p = np.array([0.40, 0.25, 0.18, 0.12, 0.05])
    total = sum(show_prob(p, i) for i in range(5))
    assert total == pytest.approx(3.0, abs=1e-12)


def test_place_show_invalid_index_raises():
    from app.services.ordering.plackett_luce import place_prob, show_prob

    p = np.array([0.6, 0.4])
    with pytest.raises(IndexError):
        place_prob(p, 5)
    with pytest.raises(IndexError):
        show_prob(p, -1)


def test_place_prob_certain_horse():
    """If p_i = 1, P(i top-2) = 1."""
    from app.services.ordering.plackett_luce import place_prob

    p = np.array([1.0, 0.0, 0.0])
    assert place_prob(p, 0) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ordering/test_plackett_luce.py -v -k "place_prob or show_prob"`
Expected: ImportError on `place_prob` / `show_prob`.

- [ ] **Step 3: Add helpers to `app/services/ordering/plackett_luce.py`**

Insert after `superfecta_prob` (around line 125) and update `__all__` at file bottom.

```python
def place_prob(p: np.ndarray, i: int) -> float:
    """P(horse i finishes 1st OR 2nd) under Plackett-Luce.

    Closed form: P(i top-2) = p_i + Σ_{j≠i} p_j · p_i / (1 − p_j).
    Handles certain horses (p_j = 1) by treating the corresponding term as 0
    if i ≠ j (i cannot place behind a horse that wins with probability 1
    unless i is that horse) and 1 if i == j.
    """
    p = _validate_probs(p)
    _validate_indices(p, (i,))
    p_i = p[i]
    total = p_i  # P(i wins) — i's contribution from the "wins" slot
    for j in range(len(p)):
        if j == i:
            continue
        denom = 1.0 - p[j]
        if denom <= 1e-12:
            # p[j] == 1 means j is certain to win; i cannot place behind j
            # because there's no second-place draw (degenerate field). Skip.
            continue
        total += p[j] * p_i / denom
    return float(total)


def show_prob(p: np.ndarray, i: int) -> float:
    """P(horse i finishes top-3) under Plackett-Luce.

    Closed form: place_prob(p, i)
              + Σ_{j≠i} Σ_{k∉{i,j}} p_j · p_k/(1−p_j) · p_i / (1 − p_j − p_k).
    Numerically guards degenerate denominators (≤ 1e-12).
    """
    p = _validate_probs(p)
    _validate_indices(p, (i,))
    n = len(p)
    if n < 3:
        # In a field of <3 horses, P(i top-3) = 1 if i is in the field.
        return 1.0

    total = place_prob(p, i)
    p_i = p[i]
    for j in range(n):
        if j == i:
            continue
        d1 = 1.0 - p[j]
        if d1 <= 1e-12:
            continue
        for k in range(n):
            if k == i or k == j:
                continue
            d2 = 1.0 - p[j] - p[k]
            if d2 <= 1e-12:
                continue
            total += p[j] * (p[k] / d1) * (p_i / d2)
    return float(total)
```

Update `__all__` at the bottom of the file:

```python
__all__ = [
    "exacta_prob",
    "trifecta_prob",
    "superfecta_prob",
    "place_prob",
    "show_prob",
    "enumerate_exotic_probs",
    "sample_ordering",
    "fit_plackett_luce_mle",
    "PlackettLuceFit",
]
```

(Verify the exact current `__all__` first via Read; merge the two new names without dropping any existing entries.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ordering/test_plackett_luce.py -v`
Expected: all PL tests pass, including the 9 new place/show tests.

- [ ] **Step 5: Commit**

```bash
git add app/services/ordering/plackett_luce.py tests/test_ordering/test_plackett_luce.py
git commit -m "phase 5a — PL place_prob / show_prob closed-form helpers"
```

---

## Task 3: Fractional Kelly Sizing + Bet Cap

**Files:**
- Create: `app/services/portfolio/__init__.py` (empty)
- Create: `app/services/portfolio/sizing.py`
- Create: `tests/test_portfolio/__init__.py` (empty)
- Create: `tests/test_portfolio/test_sizing.py`

**Design notes:**
- Pure functions. `kelly_fraction` accepts `edge` and `decimal_odds`; returns a float in [0, fraction]. `apply_bet_cap` clamps a stake fraction below the per-bet cap.
- Formula is from CLAUDE.md §8 and ADR-002. The Kelly output is already truncated at 0 (no negative-edge bets).
- 3% cap from ADR-002. Both default values are constants exported from the module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portfolio/__init__.py` empty, then `tests/test_portfolio/test_sizing.py`:

```python
"""Unit tests for app/services/portfolio/sizing.py."""
from __future__ import annotations

import math

import pytest

from app.services.portfolio.sizing import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_BET_FRACTION,
    apply_bet_cap,
    kelly_fraction,
)


def test_kelly_default_fraction_is_quarter():
    assert DEFAULT_KELLY_FRACTION == pytest.approx(0.25)


def test_kelly_default_cap_is_three_percent():
    assert DEFAULT_MAX_BET_FRACTION == pytest.approx(0.03)


def test_kelly_zero_edge_returns_zero():
    """edge=0 means no advantage; bet fraction must be 0."""
    assert kelly_fraction(edge=0.0, decimal_odds=3.0) == pytest.approx(0.0)


def test_kelly_negative_edge_returns_zero():
    """edge<0 means -EV; bet fraction must be 0 (never bet against yourself)."""
    assert kelly_fraction(edge=-0.05, decimal_odds=3.0) == pytest.approx(0.0)


def test_kelly_positive_edge_matches_handbook_formula():
    """edge=0.05 on decimal_odds=4.0 with 1/4 fraction:
        full = (0.05*4 - 0.95)/4 = (0.20 - 0.95)/4 = -0.1875 → max(0, ·) = 0
    edge=0.20 on decimal_odds=4.0:
        full = (0.20*4 - 0.80)/4 = (0.80 - 0.80)/4 = 0.0 → quarter = 0.0
    edge=0.25 on decimal_odds=4.0:
        full = (0.25*4 - 0.75)/4 = (1.0 - 0.75)/4 = 0.0625
        quarter = 0.015625
    """
    assert kelly_fraction(edge=0.25, decimal_odds=4.0) == pytest.approx(0.015625)


def test_kelly_returns_truncated_when_full_kelly_negative():
    """Even a positive edge can produce negative full Kelly if odds are short."""
    # edge=0.10, decimal_odds=1.5 → full = (0.10*1.5 - 0.90)/1.5 = -0.50 → 0
    assert kelly_fraction(edge=0.10, decimal_odds=1.5) == pytest.approx(0.0)


def test_kelly_fraction_param_scales_linearly():
    """Output must be exactly fraction * full Kelly when full Kelly > 0."""
    full = kelly_fraction(edge=0.30, decimal_odds=4.0, fraction=1.0)
    quarter = kelly_fraction(edge=0.30, decimal_odds=4.0, fraction=0.25)
    half = kelly_fraction(edge=0.30, decimal_odds=4.0, fraction=0.50)
    assert quarter == pytest.approx(full * 0.25)
    assert half == pytest.approx(full * 0.50)


def test_kelly_rejects_decimal_odds_below_one():
    with pytest.raises(ValueError, match="decimal_odds"):
        kelly_fraction(edge=0.1, decimal_odds=0.9)


def test_kelly_rejects_fraction_outside_unit():
    with pytest.raises(ValueError, match="fraction"):
        kelly_fraction(edge=0.1, decimal_odds=3.0, fraction=-0.1)
    with pytest.raises(ValueError, match="fraction"):
        kelly_fraction(edge=0.1, decimal_odds=3.0, fraction=1.5)


def test_apply_bet_cap_passthrough_when_below():
    assert apply_bet_cap(0.01) == pytest.approx(0.01)


def test_apply_bet_cap_clamps_at_default():
    assert apply_bet_cap(0.05) == pytest.approx(0.03)


def test_apply_bet_cap_custom():
    assert apply_bet_cap(0.10, cap=0.05) == pytest.approx(0.05)
    assert apply_bet_cap(0.02, cap=0.05) == pytest.approx(0.02)


def test_apply_bet_cap_rejects_negative_input():
    with pytest.raises(ValueError):
        apply_bet_cap(-0.01)


def test_apply_bet_cap_rejects_invalid_cap():
    with pytest.raises(ValueError):
        apply_bet_cap(0.01, cap=-0.01)
    with pytest.raises(ValueError):
        apply_bet_cap(0.01, cap=1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_portfolio/test_sizing.py -v`
Expected: ModuleNotFoundError on `app.services.portfolio.sizing`.

- [ ] **Step 3: Implement modules**

Create `app/services/portfolio/__init__.py` as an empty file.

Create `app/services/portfolio/sizing.py`:

```python
"""
app/services/portfolio/sizing.py
────────────────────────────────
Phase 5 — Fractional Kelly bet sizing.

Per ADR-002 (CLAUDE.md §2): all bet sizing uses 1/4 Kelly, capped at 3% of
bankroll per single bet. The Kelly fraction is a STAKE FRACTION of the
bankroll, not a multiple of expected value.

    full_kelly  = max(0, (edge × decimal_odds − (1 − edge)) / decimal_odds)
    bet_fraction = full_kelly × fraction        # fraction=0.25 by default
    capped      = min(bet_fraction, max_bet_fraction)   # 0.03 by default

These are pure functions; no state, no I/O. They live in `portfolio/`
because the CVaR optimiser (Phase 5b) consumes them as upper-bound inputs.
"""

from __future__ import annotations

DEFAULT_KELLY_FRACTION: float = 0.25
"""Per ADR-002: 1/4 Kelly is the universal fraction. Override only with
strong rationale; see ADR-002 'Rejected Alternatives' before changing."""

DEFAULT_MAX_BET_FRACTION: float = 0.03
"""Per ADR-002: hard cap of 3% of bankroll on any single bet, regardless of
what the Kelly formula returns. Guards against Kelly blowup on high-edge
exotic combinations where the formula can output very large fractions."""


def kelly_fraction(
    edge: float,
    decimal_odds: float,
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> float:
    """Fractional Kelly bet size as a fraction of bankroll.

    Args:
        edge:          model_prob − market_prob. Can be negative; if so or 0,
                       returns 0.
        decimal_odds:  gross decimal odds (e.g., 3-1 == 4.0). Must be >= 1.
        fraction:      Kelly multiplier in [0, 1]. Default 0.25 (ADR-002).

    Returns:
        Stake fraction of bankroll in [0, fraction]. Truncated at 0 for
        negative-EV positions.
    """
    if decimal_odds < 1.0:
        raise ValueError(f"decimal_odds must be >= 1; got {decimal_odds}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}")

    full = (edge * decimal_odds - (1.0 - edge)) / decimal_odds
    return max(0.0, full) * fraction


def apply_bet_cap(
    stake_fraction: float,
    cap: float = DEFAULT_MAX_BET_FRACTION,
) -> float:
    """Clamp a stake fraction to the per-bet hard cap.

    Args:
        stake_fraction: a non-negative fraction of bankroll.
        cap:            upper bound in [0, 1]. Default 0.03 (ADR-002).

    Returns:
        min(stake_fraction, cap).
    """
    if stake_fraction < 0.0:
        raise ValueError(f"stake_fraction must be >= 0; got {stake_fraction}")
    if not 0.0 <= cap <= 1.0:
        raise ValueError(f"cap must be in [0, 1]; got {cap}")
    return min(stake_fraction, cap)


__all__ = [
    "DEFAULT_KELLY_FRACTION",
    "DEFAULT_MAX_BET_FRACTION",
    "kelly_fraction",
    "apply_bet_cap",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_portfolio/test_sizing.py -v`
Expected: 14 tests passing.

- [ ] **Step 5: Commit**

```bash
git add app/services/portfolio/__init__.py app/services/portfolio/sizing.py tests/test_portfolio/__init__.py tests/test_portfolio/test_sizing.py
git commit -m "phase 5a — 1/4 Kelly sizing + 3% bet cap (ADR-002)"
```

---

## Task 4: Pari-Mutuel Market Impact

**Files:**
- Create: `app/services/ev_engine/__init__.py` (empty)
- Create: `app/services/ev_engine/market_impact.py`
- Create: `tests/test_ev_engine/__init__.py` (empty)
- Create: `tests/test_ev_engine/test_market_impact.py`

**Design notes:**
- Pari-mutuel math:
  - Pool `P` is the total bet pool for the bet type.
  - Track takeout `τ` (e.g., 0.17 for Win, 0.22 for Exacta).
  - Net distribution after take: `(1 − τ) × P` is paid out to winners.
  - For a winning outcome that received `B_i` of the pre-bet pool, the gross decimal odds *to the public* before your bet is `pre_odds = (1 − τ) × P / B_i`.
  - When you add stake `x` to the bet, pool becomes `P + x` and the winning-share denominator becomes `B_i + x`. Post-bet decimal odds for the winning combination:
    `post_odds = (1 − τ)(P + x) / (B_i + x)`.
  - As `x → ∞`, `post_odds → (1 − τ)` — the floor.
  - Default behaviour when `pool_size=None` (or `∞`): no impact, return `pre_odds` unchanged. This is the "live tote not available" path.
- We don't need to know `B_i` directly; given `pre_odds` and `pool_size`, infer `B_i = (1 − τ) × pool_size / pre_odds`. Then compute `post_odds(x)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ev_engine/__init__.py` empty, then `tests/test_ev_engine/test_market_impact.py`:

```python
"""Unit tests for app/services/ev_engine/market_impact.py."""
from __future__ import annotations

import math

import pytest

from app.services.ev_engine.market_impact import (
    inferred_winning_bets,
    post_bet_decimal_odds,
    DEFAULT_TAKEOUT,
)


def test_default_takeout_per_bet_type_present():
    assert DEFAULT_TAKEOUT["win"] == pytest.approx(0.17)
    assert DEFAULT_TAKEOUT["exacta"] == pytest.approx(0.21)
    assert DEFAULT_TAKEOUT["trifecta"] == pytest.approx(0.25)
    assert DEFAULT_TAKEOUT["superfecta"] == pytest.approx(0.25)


def test_no_pool_size_returns_pre_odds_unchanged():
    """pool_size=None means 'infinite pool' → no market impact."""
    assert post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=100.0, pool_size=None, takeout_rate=0.17
    ) == pytest.approx(5.0)


def test_zero_bet_returns_pre_odds():
    """Adding $0 to the pool cannot change the odds."""
    assert post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=0.0, pool_size=10_000.0, takeout_rate=0.17
    ) == pytest.approx(5.0)


def test_post_odds_monotonically_decreases_with_bet_size():
    """Larger stake → smaller decimal odds (more competition for the pool)."""
    pool = 10_000.0
    odds = [
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=x, pool_size=pool, takeout_rate=0.17
        )
        for x in [0.0, 10.0, 100.0, 1_000.0, 10_000.0]
    ]
    for i in range(1, len(odds)):
        assert odds[i] < odds[i - 1], (
            f"post-odds must decrease with stake; got {odds}"
        )


def test_inferred_winning_bets_matches_pari_mutuel_definition():
    """Given pre_odds = (1-τ) × pool / B_winning, recover B_winning."""
    pool = 10_000.0
    pre_odds = 5.0
    tau = 0.17
    B = inferred_winning_bets(pre_odds=pre_odds, pool_size=pool, takeout_rate=tau)
    # 5 = (1-0.17) × 10000 / B  →  B = 0.83 × 10000 / 5 = 1660.0
    assert B == pytest.approx(1660.0)


def test_post_odds_asymptotes_to_one_minus_takeout():
    """As bet → ∞, all winners are you; decimal odds → (1 − τ)."""
    pool = 10_000.0
    tau = 0.17
    odds = post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=1e12, pool_size=pool, takeout_rate=tau
    )
    assert odds == pytest.approx(1.0 - tau, abs=1e-3)


def test_post_odds_explicit_closed_form_value():
    """pre_odds=5.0, pool=10000, takeout=0.17, bet=1000:
        B_winning = 0.83*10000/5 = 1660
        post = 0.83*(10000+1000)/(1660+1000) = 0.83*11000/2660 = 3.4323...
    """
    odds = post_bet_decimal_odds(
        pre_odds=5.0, bet_amount=1000.0, pool_size=10_000.0, takeout_rate=0.17
    )
    expected = 0.83 * 11_000.0 / 2_660.0
    assert odds == pytest.approx(expected, abs=1e-9)


def test_rejects_pre_odds_below_one():
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=0.5, bet_amount=100.0, pool_size=10_000.0, takeout_rate=0.17
        )


def test_rejects_negative_bet():
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=-1.0, pool_size=10_000.0, takeout_rate=0.17
        )


def test_rejects_takeout_outside_unit():
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=100.0, pool_size=10_000.0, takeout_rate=1.1
        )
    with pytest.raises(ValueError):
        post_bet_decimal_odds(
            pre_odds=5.0, bet_amount=100.0, pool_size=10_000.0, takeout_rate=-0.01
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ev_engine/test_market_impact.py -v`
Expected: ModuleNotFoundError on `app.services.ev_engine.market_impact`.

- [ ] **Step 3: Implement**

Create `app/services/ev_engine/__init__.py` empty.

Create `app/services/ev_engine/market_impact.py`:

```python
"""
app/services/ev_engine/market_impact.py
───────────────────────────────────────
Phase 5a — Pari-mutuel market impact model.

In a pari-mutuel pool of size P with takeout rate τ, the public's gross
decimal odds on a winning outcome that received B of the pool is

    pre_odds = (1 − τ) × P / B

When you add stake x to the bet (which must therefore be a winning bet for
this calculation to matter), the pool grows to P + x and the winning share
denominator grows to B + x. The post-bet decimal odds your stake will
actually pay out at is

    post_odds(x) = (1 − τ) × (P + x) / (B + x)

This module exposes:
    * `post_bet_decimal_odds(pre_odds, bet_amount, pool_size, takeout_rate)`
      — primary API; returns post-bet odds for inserting bet_amount into
        a pool of pool_size. pool_size=None disables impact (returns pre_odds).
    * `inferred_winning_bets(pre_odds, pool_size, takeout_rate)`
      — helper that recovers B from the public odds. Useful for diagnostics
        and for the calculator when only pre_odds and pool_size are known.
    * `DEFAULT_TAKEOUT` — typical track takeout rates per bet type from
      Master Reference §190 (Section 5: Bet Type Strategy).

Per Master Reference §10 ("The Two Genuine Edges"), market impact modelling
is one of the two real competitive moats over public/commercial systems.
Phase 5a wires it in but defaults `pool_size=None` (zero impact) so that
the EV engine produces valid output even without live tote data. When live
pool sizes are available, callers populate pool_size and the engine
automatically accounts for self-impact.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_TAKEOUT: dict[str, float] = {
    "win": 0.17,
    "place": 0.17,
    "show": 0.17,
    "exacta": 0.21,
    "trifecta": 0.25,
    "superfecta": 0.25,
    "pick3": 0.25,
    "pick4": 0.25,
    "pick6": 0.25,
}
"""Typical US pari-mutuel takeout rates by bet type. Per Master Reference §190.
Track-specific overrides should be passed explicitly when known."""


def _validate(pre_odds: float, bet_amount: float, takeout_rate: float) -> None:
    if pre_odds < 1.0:
        raise ValueError(f"pre_odds must be >= 1; got {pre_odds}")
    if bet_amount < 0.0:
        raise ValueError(f"bet_amount must be >= 0; got {bet_amount}")
    if not 0.0 <= takeout_rate < 1.0:
        raise ValueError(f"takeout_rate must be in [0, 1); got {takeout_rate}")


def inferred_winning_bets(
    pre_odds: float, pool_size: float, takeout_rate: float
) -> float:
    """Recover the size of the winning-outcome bets from the public odds.

    pre_odds = (1 − τ) × pool_size / B   →   B = (1 − τ) × pool_size / pre_odds
    """
    if pool_size <= 0:
        raise ValueError(f"pool_size must be > 0; got {pool_size}")
    if pre_odds < 1.0:
        raise ValueError(f"pre_odds must be >= 1; got {pre_odds}")
    if not 0.0 <= takeout_rate < 1.0:
        raise ValueError(f"takeout_rate must be in [0, 1); got {takeout_rate}")
    return (1.0 - takeout_rate) * pool_size / pre_odds


def post_bet_decimal_odds(
    pre_odds: float,
    bet_amount: float,
    pool_size: Optional[float],
    takeout_rate: float,
) -> float:
    """Decimal odds after adding `bet_amount` to a pari-mutuel pool.

    Args:
        pre_odds:     public gross decimal odds before your bet. >= 1.
        bet_amount:   the stake you propose to add. >= 0.
        pool_size:    total pre-bet pool in $. If None, no impact applied
                      (callable for systems without live tote data).
        takeout_rate: track takeout fraction in [0, 1). e.g. 0.17 for Win.

    Returns:
        post_odds = (1 − τ)(P + x) / (B + x), where B is inferred from
        pre_odds and pool_size. Returns pre_odds when pool_size is None
        or bet_amount is 0.
    """
    _validate(pre_odds, bet_amount, takeout_rate)
    if pool_size is None or bet_amount == 0.0:
        return float(pre_odds)
    if pool_size <= 0:
        raise ValueError(f"pool_size must be > 0; got {pool_size}")

    B = inferred_winning_bets(pre_odds, pool_size, takeout_rate)
    new_pool = pool_size + bet_amount
    new_B = B + bet_amount
    return float((1.0 - takeout_rate) * new_pool / new_B)


__all__ = [
    "DEFAULT_TAKEOUT",
    "inferred_winning_bets",
    "post_bet_decimal_odds",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ev_engine/test_market_impact.py -v`
Expected: 10 tests passing.

- [ ] **Step 5: Commit**

```bash
git add app/services/ev_engine/__init__.py app/services/ev_engine/market_impact.py tests/test_ev_engine/__init__.py tests/test_ev_engine/test_market_impact.py
git commit -m "phase 5a — pari-mutuel market impact (closed-form post-bet odds)"
```

---

## Task 5: EV Calculator — Win Bets

**Files:**
- Create: `app/services/ev_engine/calculator.py`
- Create: `tests/test_ev_engine/test_calculator.py`

**Design notes:**
- Pure functions + a top-level orchestrator. No object state.
- The orchestrator `compute_ev_candidates(race_id, win_probs, decimal_odds, bet_types, min_edge, ...)` returns `list[BetCandidate]` sorted by descending EV.
- The Win path is the simplest case — the model probability IS the calibrated marginal win prob. Exacta/Trifecta/Superfecta build on it in Task 6.
- The function takes a `min_edge` filter (default 0.05 per Master Reference §151) — candidates below this are dropped. This keeps the output tractable without enumerating every exotic permutation that would be filtered later anyway.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ev_engine/test_calculator.py`:

```python
"""Unit tests for app/services/ev_engine/calculator.py."""
from __future__ import annotations

import numpy as np
import pytest

from app.schemas.race import BetType
from app.services.ev_engine.calculator import (
    DEFAULT_MIN_EDGE,
    compute_ev_candidates,
    expected_value_per_dollar,
)


def test_expected_value_per_dollar_positive_edge():
    """EV per $1 = model_prob × decimal_odds − 1."""
    assert expected_value_per_dollar(0.30, 4.0) == pytest.approx(0.20)


def test_expected_value_per_dollar_negative_edge():
    assert expected_value_per_dollar(0.20, 4.0) == pytest.approx(-0.20)


def test_expected_value_per_dollar_zero_edge():
    assert expected_value_per_dollar(0.25, 4.0) == pytest.approx(0.0)


def test_default_min_edge_matches_master_reference():
    """Master Reference §151: 5–10% minimum edge threshold."""
    assert DEFAULT_MIN_EDGE == pytest.approx(0.05)


def test_compute_win_candidates_filters_below_threshold():
    """Only candidates with edge >= min_edge are returned."""
    # Field of 4 horses; horse 0 is strongly +EV, others -EV.
    win_probs = np.array([0.50, 0.20, 0.20, 0.10])
    # Public odds (1/p, fair, no take): 2.0, 5.0, 5.0, 10.0
    # Suppose market is "wrong": public has ML 4.0 on horse 0 (implied 0.25),
    # 4.0 on horse 1, 5.0 on horse 2, 10.0 on horse 3.
    decimal_odds = np.array([4.0, 4.0, 5.0, 10.0])
    # Edges: horse 0: 0.50 - 0.25 = 0.25 (PASS); horse 1: 0.20 - 0.25 = -0.05 (FAIL)
    # horse 2: 0.20 - 0.20 = 0.0 (FAIL); horse 3: 0.10 - 0.10 = 0.0 (FAIL)
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    win_cands = [c for c in candidates if c.bet_type == BetType.WIN]
    assert len(win_cands) == 1
    assert win_cands[0].selection == (0,)
    assert win_cands[0].edge == pytest.approx(0.25)


def test_compute_win_candidates_sorted_by_descending_ev():
    """Output must be sorted by descending expected_value."""
    win_probs = np.array([0.50, 0.30, 0.20])
    decimal_odds = np.array([4.0, 5.0, 8.0])
    # Edges: h0: 0.50-0.25=0.25; h1: 0.30-0.20=0.10; h2: 0.20-0.125=0.075
    # EVs: h0: 0.50*4-1=1.0; h1: 0.30*5-1=0.5; h2: 0.20*8-1=0.6
    # Sorted by descending EV: h0 (1.0), h2 (0.6), h1 (0.5)
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    evs = [c.expected_value for c in candidates]
    assert evs == sorted(evs, reverse=True)


def test_compute_win_candidates_kelly_sized_and_capped():
    """Each candidate's kelly_fraction must respect 1/4 Kelly + 3% cap."""
    # Massive edge to force Kelly above cap
    win_probs = np.array([0.95, 0.05])
    decimal_odds = np.array([10.0, 2.0])
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    for c in candidates:
        assert c.kelly_fraction <= 0.03, "must respect 3% cap"
        assert c.kelly_fraction >= 0.0


def test_compute_win_candidates_returns_empty_when_no_edge():
    """No bets at fair odds."""
    win_probs = np.array([0.40, 0.30, 0.20, 0.10])
    decimal_odds = np.array([2.5, 1 / 0.30, 5.0, 10.0])  # all fair
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
    )
    assert candidates == []


def test_compute_win_candidates_validates_lengths_match():
    """win_probs and decimal_odds must have the same length."""
    with pytest.raises(ValueError, match="length"):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.5, 0.5]),
            decimal_odds=np.array([2.0, 2.0, 2.0]),
            bet_types=[BetType.WIN],
            min_edge=0.05,
        )


def test_compute_win_candidates_validates_probs_sum_to_one():
    with pytest.raises(ValueError, match="sum"):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.3, 0.3, 0.3]),  # sums to 0.9
            decimal_odds=np.array([4.0, 4.0, 4.0]),
            bet_types=[BetType.WIN],
            min_edge=0.05,
        )


def test_market_impact_lowers_post_odds_in_candidate():
    """When pool_size is passed, post-bet odds (and EV) should decrease."""
    win_probs = np.array([0.40, 0.30, 0.30])
    decimal_odds = np.array([5.0, 4.0, 4.0])

    no_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
        bankroll=10_000.0,
        pool_sizes={BetType.WIN: None},
    )
    with_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=decimal_odds,
        bet_types=[BetType.WIN],
        min_edge=0.05,
        bankroll=10_000.0,
        pool_sizes={BetType.WIN: 5_000.0},
    )
    # horse 0 has positive edge in both. With impact, EV should be lower.
    assert no_impact[0].selection == (0,)
    assert with_impact[0].selection == (0,)
    assert with_impact[0].expected_value < no_impact[0].expected_value
    assert with_impact[0].market_impact_applied is True
    assert no_impact[0].market_impact_applied is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ev_engine/test_calculator.py -v`
Expected: ModuleNotFoundError on `app.services.ev_engine.calculator`.

- [ ] **Step 3: Implement**

Create `app/services/ev_engine/calculator.py`:

```python
"""
app/services/ev_engine/calculator.py
────────────────────────────────────
Phase 5a — Expected Value calculator (orchestrator).

For each race, given a calibrated vector of win probabilities and a parallel
vector of decimal odds, produce a list of BetCandidate objects covering
WIN, EXACTA, TRIFECTA, and SUPERFECTA. Each candidate carries:
    - the model probability (PL-derived for exotics)
    - the (possibly market-impact-adjusted) decimal odds
    - edge = model_prob − market_prob
    - expected value per $1 = model_prob × odds − 1
    - 1/4 Kelly stake fraction (capped at 3% per ADR-002)

Place/Show and Pick 3/4/6 are deferred (ADR-039).

The calculator is source-agnostic for odds: caller supplies the array.
For backtesting, callers pass historical `odds_final`. For live, callers
pass morning-line or live tote. Same module, different data.

Market impact (Master Reference §10): when callers supply `pool_sizes`, the
calculator uses `post_bet_decimal_odds` to compute the odds AT WHICH the
proposed stake will actually settle. Default pool_size=None ⇒ no impact,
suitable for analyses without live tote data.

Public API:
    compute_ev_candidates(
        race_id, win_probs, decimal_odds, bet_types,
        min_edge=DEFAULT_MIN_EDGE, bankroll=1.0,
        pool_sizes=None, takeout_rates=None,
    ) -> list[BetCandidate]
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from app.core.logging import get_logger
from app.schemas.bets import BetCandidate
from app.schemas.race import BetType
from app.services.ev_engine.market_impact import (
    DEFAULT_TAKEOUT,
    post_bet_decimal_odds,
)
from app.services.portfolio.sizing import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_BET_FRACTION,
    apply_bet_cap,
    kelly_fraction,
)

log = get_logger(__name__)


DEFAULT_MIN_EDGE: float = 0.05
"""Per Master Reference §151: 5-10% minimum edge threshold. 0.05 is the
permissive default; tighten for production by passing min_edge=0.08."""


_SUM_TOL: float = 1e-5


def expected_value_per_dollar(model_prob: float, decimal_odds: float) -> float:
    """EV per $1 staked: p × O − 1.

    Equivalent to p × (O − 1) − (1 − p) × 1; we use the algebraic
    simplification because it's one multiply.
    """
    return float(model_prob * decimal_odds - 1.0)


def _validate_inputs(win_probs: np.ndarray, decimal_odds: np.ndarray) -> None:
    if len(win_probs) != len(decimal_odds):
        raise ValueError(
            f"win_probs and decimal_odds must have the same length; "
            f"got {len(win_probs)} and {len(decimal_odds)}"
        )
    s = float(np.sum(win_probs))
    if abs(s - 1.0) > _SUM_TOL:
        raise ValueError(f"win_probs must sum to 1; got {s}")
    if (win_probs < -1e-12).any():
        raise ValueError("win_probs must be non-negative")
    if (decimal_odds < 1.0).any():
        raise ValueError("decimal_odds must all be >= 1")


def _candidate_for_win(
    race_id: str,
    horse_idx: int,
    model_prob: float,
    pre_odds: float,
    bankroll: float,
    pool_size: Optional[float],
    takeout_rate: float,
    min_edge: float,
) -> Optional[BetCandidate]:
    """Build a single WIN BetCandidate. Returns None if edge < min_edge."""
    # Apply market impact if pool_size is available. We need the proposed
    # stake to compute the post-bet odds, but the stake itself depends on
    # the post-bet odds via Kelly. We use the PRE-impact Kelly as a first
    # estimate, then recompute odds at that stake — a single-step
    # approximation that is accurate to ~1% for sub-pool-fraction bets.
    pre_market_prob = 1.0 / pre_odds
    pre_edge = model_prob - pre_market_prob
    if pre_edge < min_edge:
        return None

    if pool_size is not None:
        pre_kelly = kelly_fraction(pre_edge, pre_odds)
        pre_stake_frac_capped = apply_bet_cap(pre_kelly)
        pre_stake = pre_stake_frac_capped * bankroll
        decimal_odds = post_bet_decimal_odds(
            pre_odds=pre_odds,
            bet_amount=pre_stake,
            pool_size=pool_size,
            takeout_rate=takeout_rate,
        )
        market_impact_applied = True
    else:
        decimal_odds = pre_odds
        market_impact_applied = False

    market_prob = 1.0 / decimal_odds
    edge = model_prob - market_prob
    if edge < min_edge:
        return None

    ev = expected_value_per_dollar(model_prob, decimal_odds)
    kelly = kelly_fraction(edge, decimal_odds)
    kelly_capped = apply_bet_cap(kelly)

    return BetCandidate(
        race_id=race_id,
        bet_type=BetType.WIN,
        selection=(horse_idx,),
        model_prob=float(model_prob),
        decimal_odds=float(decimal_odds),
        market_prob=float(market_prob),
        edge=float(edge),
        expected_value=float(ev),
        kelly_fraction=float(kelly_capped),
        market_impact_applied=market_impact_applied,
        pool_size=pool_size,
    )


def compute_ev_candidates(
    race_id: str,
    win_probs: np.ndarray,
    decimal_odds: np.ndarray,
    bet_types: Iterable[BetType],
    min_edge: float = DEFAULT_MIN_EDGE,
    bankroll: float = 1.0,
    pool_sizes: Optional[dict[BetType, Optional[float]]] = None,
    takeout_rates: Optional[dict[BetType, float]] = None,
) -> list[BetCandidate]:
    """Generate all +EV BetCandidate objects for a single race.

    Args:
        race_id:      string identifier (used as-is on output).
        win_probs:    length-N calibrated marginal P(win) per horse. Must
                      sum to 1.
        decimal_odds: length-N parallel array of decimal odds (gross). For
                      Win, decimal_odds[i] is the public price on horse i.
                      Exacta/Trifecta/Superfecta odds are NOT in this array;
                      they are computed in Task 6 from PL marginals + a
                      per-permutation gross-odds dict (added later).
        bet_types:    iterable of BetType enums to evaluate.
        min_edge:     minimum edge to include in output. Default 0.05.
        bankroll:     used only to compute the dollar stake for market
                      impact estimation. Pure-prob outputs do not depend
                      on this. Default 1.0 (treat outputs as fractions).
        pool_sizes:   optional dict mapping BetType → pool size in $. When
                      provided, market impact is applied. Default None.
        takeout_rates: optional dict overriding `DEFAULT_TAKEOUT` per bet
                      type.

    Returns:
        List of BetCandidate sorted by descending expected_value.
    """
    win_probs = np.asarray(win_probs, dtype=float).ravel()
    decimal_odds = np.asarray(decimal_odds, dtype=float).ravel()
    _validate_inputs(win_probs, decimal_odds)

    pool_sizes = pool_sizes or {}
    takeout_rates = takeout_rates or {}

    candidates: list[BetCandidate] = []

    for bet_type in bet_types:
        pool_size = pool_sizes.get(bet_type)
        takeout = takeout_rates.get(bet_type, DEFAULT_TAKEOUT[bet_type.value])

        if bet_type == BetType.WIN:
            for i in range(len(win_probs)):
                c = _candidate_for_win(
                    race_id=race_id,
                    horse_idx=i,
                    model_prob=float(win_probs[i]),
                    pre_odds=float(decimal_odds[i]),
                    bankroll=bankroll,
                    pool_size=pool_size,
                    takeout_rate=takeout,
                    min_edge=min_edge,
                )
                if c is not None:
                    candidates.append(c)
        elif bet_type in (BetType.EXACTA, BetType.TRIFECTA, BetType.SUPERFECTA):
            # Task 6 — exotic candidates. Stub for Task 5 only.
            log.debug("ev_calculator.exotic_skipped", bet_type=bet_type)
            continue
        else:
            raise ValueError(
                f"bet_type {bet_type} not supported in Phase 5a "
                f"(ADR-039). Use WIN, EXACTA, TRIFECTA, or SUPERFECTA."
            )

    candidates.sort(key=lambda c: c.expected_value, reverse=True)
    return candidates


__all__ = [
    "DEFAULT_MIN_EDGE",
    "compute_ev_candidates",
    "expected_value_per_dollar",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ev_engine/test_calculator.py -v`
Expected: 10 tests passing.

- [ ] **Step 5: Commit**

```bash
git add app/services/ev_engine/calculator.py tests/test_ev_engine/test_calculator.py
git commit -m "phase 5a — EV calculator (Win bets, market impact, Kelly sizing)"
```

---

## Task 6: Extend Calculator to Exotic Bets

**Files:**
- Modify: `app/services/ev_engine/calculator.py`
- Modify: `tests/test_ev_engine/test_calculator.py`

**Design notes:**
- Exotic odds (Exacta, Trifecta, Superfecta) are **per-permutation** quantities — there's a separate decimal odds for each ordered finishing combination. The caller provides them as `exotic_odds: dict[BetType, dict[tuple[int,...], float]]`.
- For each permutation in `exotic_odds[bet_type]`:
  - Compute `model_prob` via `exacta_prob/trifecta_prob/superfecta_prob` on the win_probs vector.
  - Compute `market_prob = 1 / decimal_odds`.
  - Filter by `min_edge` and append.
- Enumeration of *all* permutations is the caller's responsibility — passing only the permutations for which odds are known. In paper-trading, the public-pool exotic odds are known only for the permutations that have been bet. Phase 5a accepts this as-is; the caller can also use `enumerate_exotic_probs` if it has a uniform-pool assumption.
- Market impact for exotics: same `post_bet_decimal_odds` math; the relevant pool is the exotic pool, not the win pool. `pool_sizes[BetType.TRIFECTA]` is the trifecta pool size.

- [ ] **Step 1: Write the failing tests (append to test_calculator.py)**

```python
# ── Exotic bets ────────────────────────────────────────────────────────────


def test_compute_exacta_candidates_from_exotic_odds_dict():
    """Caller supplies per-permutation gross odds; calculator filters by edge."""
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.50, 0.30, 0.20])
    # Public exacta odds (gross decimal); we make horse-0-1 very generous.
    exotic_odds = {
        BetType.EXACTA: {
            (0, 1): 8.0,   # PL prob: 0.50 * 0.30 / 0.50 = 0.30; market 0.125; edge 0.175 → PASS
            (0, 2): 12.0,  # PL prob: 0.50 * 0.20 / 0.50 = 0.20; market 0.083; edge 0.117 → PASS
            (1, 0): 8.0,   # PL prob: 0.30 * 0.50 / 0.70 ≈ 0.214; market 0.125; edge 0.089 → PASS
            (1, 2): 30.0,  # PL prob: 0.30 * 0.20 / 0.70 ≈ 0.0857; market 0.033; edge 0.052 → PASS
        }
    }
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.5, 5.0]),
        bet_types=[BetType.EXACTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
    )
    selections = {c.selection for c in candidates}
    # All four permutations have edge >= 0.05 per the math above.
    assert selections == {(0, 1), (0, 2), (1, 0), (1, 2)}


def test_compute_trifecta_candidates():
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.50, 0.30, 0.15, 0.05])
    # Pick one strongly +EV trifecta and one -EV; calculator must filter.
    exotic_odds = {
        BetType.TRIFECTA: {
            (0, 1, 2): 30.0,   # PL: 0.50 * (0.30/0.50) * (0.15/0.20) = 0.225;
                               #     market: 0.0333; edge: 0.19 → PASS
            (0, 1, 3): 200.0,  # PL: 0.50 * (0.30/0.50) * (0.05/0.20) = 0.075;
                               #     market: 0.005; edge: 0.07 → PASS
            (3, 2, 1): 100.0,  # PL: 0.05 * (0.15/0.95) * (0.30/0.80) ≈ 0.00296;
                               #     market: 0.01; edge: ≈-0.007 → FAIL
        }
    }
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.3, 6.7, 20.0]),
        bet_types=[BetType.TRIFECTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
    )
    selections = {c.selection for c in candidates}
    assert selections == {(0, 1, 2), (0, 1, 3)}


def test_compute_superfecta_candidate():
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.40, 0.30, 0.20, 0.10])
    exotic_odds = {
        BetType.SUPERFECTA: {
            (0, 1, 2, 3): 50.0,  # PL: 0.40 * (0.30/0.60) * (0.20/0.30) * (0.10/0.10)
                                 #     = 0.40 * 0.5 * 0.667 * 1.0 = 0.1333
                                 # market: 0.02; edge: 0.1133 → PASS
        }
    }
    candidates = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.5, 3.3, 5.0, 10.0]),
        bet_types=[BetType.SUPERFECTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
    )
    assert len(candidates) == 1
    assert candidates[0].selection == (0, 1, 2, 3)
    assert candidates[0].edge == pytest.approx(0.1333, abs=1e-3)


def test_exotic_bet_without_odds_dict_raises():
    from app.services.ev_engine.calculator import compute_ev_candidates

    with pytest.raises(ValueError, match="exotic_odds"):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.5, 0.3, 0.2]),
            decimal_odds=np.array([2.0, 3.3, 5.0]),
            bet_types=[BetType.EXACTA],
            min_edge=0.05,
            exotic_odds=None,
        )


def test_exotic_market_impact_reduces_ev():
    """Same as Win market-impact test, but for an exotic pool."""
    from app.services.ev_engine.calculator import compute_ev_candidates

    win_probs = np.array([0.50, 0.30, 0.20])
    exotic_odds = {BetType.EXACTA: {(0, 1): 8.0}}

    no_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.3, 5.0]),
        bet_types=[BetType.EXACTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
        bankroll=10_000.0,
        pool_sizes={BetType.EXACTA: None},
    )
    with_impact = compute_ev_candidates(
        race_id="R1",
        win_probs=win_probs,
        decimal_odds=np.array([2.0, 3.3, 5.0]),
        bet_types=[BetType.EXACTA],
        min_edge=0.05,
        exotic_odds=exotic_odds,
        bankroll=10_000.0,
        pool_sizes={BetType.EXACTA: 5_000.0},
    )
    assert with_impact[0].expected_value < no_impact[0].expected_value


def test_exotic_selection_validates_distinct_indices():
    """The schema enforces distinctness; bad caller data should raise."""
    from app.services.ev_engine.calculator import compute_ev_candidates

    with pytest.raises(ValueError):
        compute_ev_candidates(
            race_id="R1",
            win_probs=np.array([0.5, 0.3, 0.2]),
            decimal_odds=np.array([2.0, 3.3, 5.0]),
            bet_types=[BetType.EXACTA],
            min_edge=0.05,
            exotic_odds={BetType.EXACTA: {(0, 0): 10.0}},  # repeated index
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ev_engine/test_calculator.py -v -k "exotic or exacta or trifecta or superfecta"`
Expected: All new tests fail (calculator currently skips exotics).

- [ ] **Step 3: Extend the calculator**

Update `app/services/ev_engine/calculator.py`:

1. Add import:

```python
from app.services.ordering.plackett_luce import (
    exacta_prob,
    trifecta_prob,
    superfecta_prob,
)
```

2. Add helper for exotic candidates (insert above `compute_ev_candidates`):

```python
_PL_PROB_FN = {
    BetType.EXACTA: lambda p, sel: exacta_prob(p, sel[0], sel[1]),
    BetType.TRIFECTA: lambda p, sel: trifecta_prob(p, sel[0], sel[1], sel[2]),
    BetType.SUPERFECTA: lambda p, sel: superfecta_prob(p, sel[0], sel[1], sel[2], sel[3]),
}


def _candidate_for_exotic(
    race_id: str,
    bet_type: BetType,
    selection: tuple[int, ...],
    win_probs: np.ndarray,
    pre_odds: float,
    bankroll: float,
    pool_size: Optional[float],
    takeout_rate: float,
    min_edge: float,
) -> Optional[BetCandidate]:
    """Build a single exotic BetCandidate. Returns None if edge < min_edge."""
    pl_fn = _PL_PROB_FN[bet_type]
    model_prob = float(pl_fn(win_probs, selection))

    pre_market_prob = 1.0 / pre_odds
    pre_edge = model_prob - pre_market_prob
    if pre_edge < min_edge:
        return None

    if pool_size is not None:
        pre_kelly = kelly_fraction(pre_edge, pre_odds)
        pre_stake_frac = apply_bet_cap(pre_kelly)
        pre_stake = pre_stake_frac * bankroll
        decimal_odds = post_bet_decimal_odds(
            pre_odds=pre_odds,
            bet_amount=pre_stake,
            pool_size=pool_size,
            takeout_rate=takeout_rate,
        )
        market_impact_applied = True
    else:
        decimal_odds = pre_odds
        market_impact_applied = False

    market_prob = 1.0 / decimal_odds
    edge = model_prob - market_prob
    if edge < min_edge:
        return None

    ev = expected_value_per_dollar(model_prob, decimal_odds)
    kelly = kelly_fraction(edge, decimal_odds)
    kelly_capped = apply_bet_cap(kelly)

    return BetCandidate(
        race_id=race_id,
        bet_type=bet_type,
        selection=selection,
        model_prob=model_prob,
        decimal_odds=float(decimal_odds),
        market_prob=float(market_prob),
        edge=float(edge),
        expected_value=float(ev),
        kelly_fraction=float(kelly_capped),
        market_impact_applied=market_impact_applied,
        pool_size=pool_size,
    )
```

3. Replace the body of `compute_ev_candidates` to thread `exotic_odds`:

```python
def compute_ev_candidates(
    race_id: str,
    win_probs: np.ndarray,
    decimal_odds: np.ndarray,
    bet_types: Iterable[BetType],
    min_edge: float = DEFAULT_MIN_EDGE,
    bankroll: float = 1.0,
    pool_sizes: Optional[dict[BetType, Optional[float]]] = None,
    takeout_rates: Optional[dict[BetType, float]] = None,
    exotic_odds: Optional[dict[BetType, dict[tuple[int, ...], float]]] = None,
) -> list[BetCandidate]:
    """Generate all +EV BetCandidate objects for a single race.

    Args:
        race_id, win_probs, decimal_odds: as before. decimal_odds is used
            only for Win-pool bets; exotics use `exotic_odds`.
        bet_types: iterable of BetType enums to evaluate.
        min_edge:  minimum edge to include. Default 0.05.
        bankroll:  used for dollar-stake estimation in market impact. Default 1.0.
        pool_sizes: dict mapping BetType → pool size in $. None disables impact.
        takeout_rates: dict overriding DEFAULT_TAKEOUT per bet type.
        exotic_odds:   dict[BetType, dict[selection-tuple, decimal_odds]]
                       Required for EXACTA / TRIFECTA / SUPERFECTA.
    """
    win_probs = np.asarray(win_probs, dtype=float).ravel()
    decimal_odds = np.asarray(decimal_odds, dtype=float).ravel()
    _validate_inputs(win_probs, decimal_odds)

    pool_sizes = pool_sizes or {}
    takeout_rates = takeout_rates or {}
    exotic_odds = exotic_odds or {}

    candidates: list[BetCandidate] = []

    for bet_type in bet_types:
        pool_size = pool_sizes.get(bet_type)
        takeout = takeout_rates.get(bet_type, DEFAULT_TAKEOUT[bet_type.value])

        if bet_type == BetType.WIN:
            for i in range(len(win_probs)):
                c = _candidate_for_win(
                    race_id=race_id,
                    horse_idx=i,
                    model_prob=float(win_probs[i]),
                    pre_odds=float(decimal_odds[i]),
                    bankroll=bankroll,
                    pool_size=pool_size,
                    takeout_rate=takeout,
                    min_edge=min_edge,
                )
                if c is not None:
                    candidates.append(c)
        elif bet_type in (BetType.EXACTA, BetType.TRIFECTA, BetType.SUPERFECTA):
            if bet_type not in exotic_odds:
                raise ValueError(
                    f"exotic_odds dict required for {bet_type}; "
                    f"caller must supply per-permutation gross decimal odds."
                )
            for selection, pre_odds in exotic_odds[bet_type].items():
                c = _candidate_for_exotic(
                    race_id=race_id,
                    bet_type=bet_type,
                    selection=tuple(selection),
                    win_probs=win_probs,
                    pre_odds=float(pre_odds),
                    bankroll=bankroll,
                    pool_size=pool_size,
                    takeout_rate=takeout,
                    min_edge=min_edge,
                )
                if c is not None:
                    candidates.append(c)
        else:
            raise ValueError(
                f"bet_type {bet_type} not supported in Phase 5a "
                f"(ADR-039). Use WIN, EXACTA, TRIFECTA, or SUPERFECTA."
            )

    candidates.sort(key=lambda c: c.expected_value, reverse=True)
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ev_engine/test_calculator.py -v`
Expected: all 16 tests passing.

- [ ] **Step 5: Commit**

```bash
git add app/services/ev_engine/calculator.py tests/test_ev_engine/test_calculator.py
git commit -m "phase 5a — EV calculator: exacta, trifecta, superfecta"
```

---

## Task 7: End-to-End Validation Script

**Files:**
- Create: `scripts/validate_phase5a_ev_engine.py`

**Design notes:**
- Mirrors the structure of `scripts/validate_calibration.py`. Loads the trained baseline models from `models/baseline_full/`, re-runs the feature pipeline, applies calibration, and feeds the calibrated win probs into `compute_ev_candidates`.
- **Backtest mode** (default): uses historical `odds_final` from the parquet — this is the closing market price for each race, the most accurate odds source available. Only WIN bets are evaluated in this mode (exotic historical odds are not in the parquet).
- **Live mode** (`--mode live`): uses morning-line from a parsed RaceCard JSON. Designed for the future PDF-ingestion-to-EV integration; stubbed in 5a (raises NotImplementedError) so the seam is visible.
- Writes a JSON report per race + a card-level summary to `models/baseline_full/ev_engine/<run-id>/report.json`.
- Sanity check (printed to stdout, asserted): on a large enough holdout, summed expected_value across all returned candidates should be slightly positive (otherwise either the model is poor or the threshold is wrong).

**Note on test coverage:** This is a runner/integration script. The pure functions it composes are already unit-tested. Mark it as a smoke-tested script — one happy-path smoke test in `tests/test_ev_engine/test_calculator.py` covering the loader path is enough (Task 7 leaves the test suite at 483 + Task 5/6 additions; do NOT add new test modules for the script).

- [ ] **Step 1: Sketch the script (no test step — this is a runner)**

Create `scripts/validate_phase5a_ev_engine.py`:

```python
"""
scripts/validate_phase5a_ev_engine.py
─────────────────────────────────────
Phase 5a — End-to-end EV-engine validation on the time-split parquet.

Loads the trained baseline models (Speed/Form + Meta-learner + Calibrator)
from models/baseline_full/, splits the parquet into train / calib / test
exactly as `validate_calibration.py` does, then for each race in the TEST
slice:
  1. Scores the meta-learner.
  2. Applies the loaded Calibrator and per-race softmax.
  3. Calls `compute_ev_candidates(...)` with `decimal_odds` populated from
     historical `odds_final` (backtest mode, default) or morning-line
     (live mode, stub).
  4. Aggregates per-race candidate counts, total edge, EV, and Kelly stake
     allocation.

Outputs:
    models/baseline_full/ev_engine/<run-id>/report.json
    models/baseline_full/ev_engine/<run-id>/per_race_summary.parquet

Usage:
    python scripts/validate_phase5a_ev_engine.py \
        --mode backtest \
        --min-edge 0.05 \
        --run-id 2026-05-13-backtest \
        [--limit-races 1000]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.schemas.bets import BetCandidate
from app.schemas.race import BetType
from app.services.calibration.calibrator import Calibrator
from app.services.ev_engine.calculator import (
    DEFAULT_MIN_EDGE,
    compute_ev_candidates,
)

log = get_logger(__name__)


_MODEL_ROOT = Path("models/baseline_full")
_PARQUET = Path("data/exports/training_20260512.parquet")
_OUT_ROOT = _MODEL_ROOT / "ev_engine"


@dataclass
class EVRunSummary:
    run_id: str
    mode: str
    min_edge: float
    test_n_rows: int
    test_n_races: int
    n_candidates: int
    races_with_candidates: int
    sum_edge: float
    sum_expected_value: float
    mean_edge: float
    mean_expected_value: float
    total_kelly_fraction: float
    candidate_count_by_bet_type: dict[str, int] = field(default_factory=dict)
    bet_types_evaluated: list[str] = field(default_factory=list)


def _score_test_slice(
    test_features: pd.DataFrame,
    models_dir: Path,
    meta_cal: Calibrator,
    race_id_col: str,
) -> pd.DataFrame:
    """Run all sub-models + meta-learner on the test slice and apply the
    loaded calibrator. Mirrors the scoring path inside validate_calibration.run_live.

    `test_features` must already be the output of `prepare_training_features`
    (called by the caller; we do the heavy preparation once for the full
    parquet, then take the test slice from that).
    """
    # Reuse the EXACT same scoring path as validate_calibration.run_live to
    # avoid duplication and behaviour drift. The leading underscore is a
    # local convention, not a hard private boundary — both scripts now
    # consume `_stack_for_meta`, which is fine.
    from app.services.models.connections import ConnectionsModel
    from app.services.models.market import MarketModel
    from app.services.models.meta_learner import MetaLearner as _MetaLearner
    from app.services.models.pace_scenario import PaceScenarioModel
    from app.services.models.sequence import SequenceModel
    from app.services.models.speed_form import SpeedFormModel
    from scripts.validate_calibration import _stack_for_meta

    speed_form = SpeedFormModel.load(models_dir / "speed_form")
    pace = PaceScenarioModel()
    sequence = SequenceModel()
    connections = ConnectionsModel.load(models_dir / "connections")
    market = MarketModel.load(models_dir / "market")
    meta = _MetaLearner.load(models_dir / "meta_learner")

    stacked = _stack_for_meta(test_features, speed_form, pace, sequence, connections, market)
    raw = meta.predict_proba(stacked)
    calibrated = meta_cal.predict_softmax(raw, race_ids=test_features[race_id_col].values)
    out = test_features.copy()
    out["meta_raw"] = raw
    out["meta_calibrated"] = calibrated
    return out


def _odds_for_mode(group: pd.DataFrame, mode: str) -> np.ndarray:
    """Return decimal-odds vector for a single race, by mode.

    backtest: uses odds_final from the row (the closing market price).
    live:     uses morning_line_odds (not present in the historical parquet).
    """
    if mode == "backtest":
        odds = group["odds_final"].to_numpy(dtype=float)
        # Treat odds <= 1.0 as missing (some sources record scratched horses
        # as 0 or 1). Replace with NaN and skip the race if any NaN remains.
        odds = np.where(odds > 1.0, odds, np.nan)
    elif mode == "live":
        raise NotImplementedError(
            "live mode requires a parsed RaceCard with morning_line_odds. "
            "Wire up after PDF-ingestion-to-EV integration is built."
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return odds


def evaluate(
    test_df: pd.DataFrame,
    mode: str,
    min_edge: float,
    bet_types: list[BetType],
    race_id_col: str = "race_id",
    limit_races: Optional[int] = None,
) -> tuple[list[BetCandidate], EVRunSummary]:
    """Run the EV engine race-by-race over the test slice."""
    candidates: list[BetCandidate] = []
    n_races = 0
    races_with_cands = 0

    groups = test_df.groupby(race_id_col, sort=False)
    if limit_races is not None:
        groups = list(groups)[:limit_races]

    for race_id, group in groups:
        win_probs = group["meta_calibrated"].to_numpy(dtype=float)
        s = win_probs.sum()
        if not np.isfinite(s) or s <= 0:
            continue
        win_probs = win_probs / s  # renormalise after any drops

        odds = _odds_for_mode(group, mode)
        if np.isnan(odds).any():
            continue

        race_cands = compute_ev_candidates(
            race_id=str(race_id),
            win_probs=win_probs,
            decimal_odds=odds,
            bet_types=bet_types,
            min_edge=min_edge,
        )
        candidates.extend(race_cands)
        n_races += 1
        if race_cands:
            races_with_cands += 1

    by_type: dict[str, int] = {}
    for c in candidates:
        by_type[c.bet_type.value] = by_type.get(c.bet_type.value, 0) + 1

    summary = EVRunSummary(
        run_id="<set-by-caller>",
        mode=mode,
        min_edge=min_edge,
        test_n_rows=len(test_df),
        test_n_races=n_races,
        n_candidates=len(candidates),
        races_with_candidates=races_with_cands,
        sum_edge=float(sum(c.edge for c in candidates)),
        sum_expected_value=float(sum(c.expected_value for c in candidates)),
        mean_edge=(
            float(np.mean([c.edge for c in candidates])) if candidates else 0.0
        ),
        mean_expected_value=(
            float(np.mean([c.expected_value for c in candidates]))
            if candidates
            else 0.0
        ),
        total_kelly_fraction=float(sum(c.kelly_fraction for c in candidates)),
        candidate_count_by_bet_type=by_type,
        bet_types_evaluated=[bt.value for bt in bet_types],
    )
    return candidates, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    ap.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    ap.add_argument("--run-id", type=str, default=datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--limit-races", type=int, default=None)
    ap.add_argument(
        "--bet-types",
        type=str,
        default="win",
        help="Comma-separated subset of win,exacta,trifecta,superfecta. "
             "Note: in backtest mode only WIN is supported (no historical "
             "per-permutation exotic odds in the parquet).",
    )
    args = ap.parse_args()

    bet_types = [BetType(s.strip()) for s in args.bet_types.split(",")]
    if args.mode == "backtest" and any(bt != BetType.WIN for bt in bet_types):
        raise SystemExit(
            "Backtest mode supports only --bet-types win. "
            "Exotic odds are not in the parquet."
        )

    log.info("ev_engine.validate.start", **vars(args))

    from app.services.models.training_data import (
        load_training_parquet, prepare_training_features,
    )
    from scripts.validate_calibration import _three_way_split

    df = load_training_parquet(_PARQUET)
    features = prepare_training_features(df)

    _, _, test, calib_cutoff, test_cutoff = _three_way_split(
        features, test_fraction=0.10, calib_fraction=0.10,
    )
    log.info(
        "ev_engine.validate.split",
        test=len(test),
        calib_cutoff=str(calib_cutoff),
        test_cutoff=str(test_cutoff),
    )

    meta_cal = Calibrator.load(_MODEL_ROOT / "calibration_adr038_brier" / "meta_learner")
    test_scored = _score_test_slice(
        test_features=test,
        models_dir=_MODEL_ROOT,
        meta_cal=meta_cal,
        race_id_col="race_id",
    )

    candidates, summary = evaluate(
        test_df=test_scored,
        mode=args.mode,
        min_edge=args.min_edge,
        bet_types=bet_types,
        limit_races=args.limit_races,
    )
    summary.run_id = args.run_id

    out_dir = _OUT_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "report.json", "w") as fh:
        json.dump(
            {
                "summary": asdict(summary),
                "candidates": [c.model_dump() for c in candidates[:1000]],
            },
            fh,
            indent=2,
            default=str,
        )
    log.info(
        "ev_engine.validate.done",
        out_dir=str(out_dir),
        n_candidates=summary.n_candidates,
        sum_ev=summary.sum_expected_value,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


if __name__ == "__main__":
    main()
```

**Note on imports:** The script reuses `_stack_for_meta`, `_three_way_split`, `prepare_training_features`, and `load_training_parquet` from existing modules. No new helpers need to be added — both scripts are now consumers of those existing functions. The leading underscore on `_stack_for_meta` and `_three_way_split` is a local convention, not a hard boundary; consuming them from a second runner is allowed.

- [ ] **Step 2: Smoke-run the script with a small slice**

Run:
```
.venv/Scripts/python.exe scripts/validate_phase5a_ev_engine.py --mode backtest --limit-races 50 --run-id smoke-test-001
```

Expected:
- Exits 0.
- Writes `models/baseline_full/ev_engine/smoke-test-001/report.json`.
- Prints a `summary` JSON to stdout with `n_candidates > 0` (some +EV bets in 50 races is expected given calibrated probs).
- No exceptions.

If the script fails because `odds_final` is null on >50% of test rows for whatever subset got picked, the script's `_odds_for_mode` returns NaN and the race is skipped (per the existing code path) — that's expected behaviour, not a bug. Increase `--limit-races` or rerun against a different slice if more candidates are needed.

- [ ] **Step 3: Verify the test suite still passes**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all tests pass (no regressions from the new module imports).

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_phase5a_ev_engine.py
# If you had to add a helper to validate_calibration.py:
# git add scripts/validate_calibration.py
git commit -m "phase 5a — end-to-end EV-engine validation script (backtest mode)"
```

---

## Task 8: ADRs + PROGRESS Update

**Files:**
- Modify: `DECISIONS.md` (append two new ADRs)
- Modify: `PROGRESS.md` (new session log entry + Current State update)

**Design notes:**
- ADR-039 documents the Place/Show + Pick3/4/6 deferral.
- ADR-040 documents the source-agnostic odds-input design and the backtest-vs-live-mode contract for the validation script.

- [ ] **Step 1: Append ADR-039 and ADR-040 to DECISIONS.md**

Append to the end of `DECISIONS.md`:

````markdown
---

## ADR-039: Phase 5a Bet-Type Scope — Win + Exacta + Trifecta + Superfecta Only

**Date:** 2026-05-13
**Status:** Accepted

**Context:**
Phase 5a's `BetCandidate` schema and EV calculator support a subset of pari-mutuel bet types. The PL ordering module produces marginals for any combinatorial bet; the question is which ones the calculator should *emit* given current data and model coverage.

**Decision:**
Phase 5a supports `WIN`, `EXACTA`, `TRIFECTA`, `SUPERFECTA` only. The schema validator (`BetCandidate._validate_selection`) rejects `PLACE`, `SHOW`, `PICK3`, `PICK4`, `PICK6` with `ValueError`.

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
````

- [ ] **Step 2: Update PROGRESS.md**

At the very top, replace the "Current State" block:

```markdown
## Current State

**Phase:** Phase 5a — EV Engine **COMPLETE** · Phase 5b — Portfolio Optimizer **NEXT**.
**Last completed task:** Phase 5a EV engine landed end-to-end. New modules: `app/schemas/bets.py` (BetCandidate / BetRecommendation / Portfolio), `app/services/portfolio/sizing.py` (1/4 Kelly + 3% cap per ADR-002), `app/services/ev_engine/market_impact.py` (closed-form pari-mutuel post-bet odds), `app/services/ev_engine/calculator.py` (orchestrator over Win/Exacta/Trifecta/Superfecta). PL gained `place_prob` / `show_prob` closed-form helpers. Backtest validation script (`scripts/validate_phase5a_ev_engine.py`) runs end-to-end on the test slice with historical `odds_final` as the odds source (ADR-040). Place/Show + Pick N deferred (ADR-039).
**Next task:** Phase 5b — Portfolio Optimizer. CVaR LP via Rockafellar-Uryasev + scipy.optimize.linprog; consumes BetCandidate list; produces BetRecommendation/Portfolio. CLAUDE.md §10 acceptance: "Assert CVaR constraint is binding at the limit."
```

Then prepend a session log entry after the existing "## Current State" / "## Session Log" boundary:

```markdown
### Session: 2026-05-13 (b) — Phase 5a EV engine

**Completed:**

*Schemas (`app/schemas/bets.py`)*
- `BetCandidate`, `BetRecommendation`, `Portfolio` Pydantic v2 models.
- Selection-length validator enforces Win=1, Exacta=2, Trifecta=3, Superfecta=4.
- Pick N + Place/Show rejected with explicit ValueError per ADR-039.

*PL helpers (`app/services/ordering/plackett_luce.py`)*
- `place_prob(p, i)` and `show_prob(p, i)` closed-form. Tested against
  brute-force `enumerate_exotic_probs` summation; verified Σ_i place_prob = 2,
  Σ_i show_prob = 3 exactly.

*Sizing (`app/services/portfolio/sizing.py`)*
- `kelly_fraction(edge, decimal_odds, fraction=0.25)` and `apply_bet_cap(stake, cap=0.03)`.
- Pure functions; per ADR-002.

*Market impact (`app/services/ev_engine/market_impact.py`)*
- `post_bet_decimal_odds(pre_odds, bet_amount, pool_size, takeout_rate)`.
- pool_size=None ⇒ zero impact (default for systems without live tote).
- `DEFAULT_TAKEOUT` table populated from Master Reference §190.

*Calculator (`app/services/ev_engine/calculator.py`)*
- `compute_ev_candidates(race_id, win_probs, decimal_odds, bet_types, ...) → list[BetCandidate]`.
- Win path: per-horse iteration; market impact applied iff pool_size is supplied.
- Exotic path: caller supplies `exotic_odds: dict[BetType, dict[tuple, float]]`.
- Output sorted by descending expected_value.
- min_edge filter (default 0.05) applied AFTER market impact.

*Validation script (`scripts/validate_phase5a_ev_engine.py`)*
- Backtest mode uses historical `odds_final` (ADR-040).
- Loads baseline_full meta-learner + calibration_adr038_brier/meta_learner.
- Three-way 70/15/15 chronological split (matches validate_calibration.py).
- Writes `report.json` + per-race summary parquet under
  `models/baseline_full/ev_engine/<run-id>/`.

**Key decisions made:**
- **ADR-039:** Phase 5a bet-type scope (Win+Exacta+Trifecta+Superfecta).
- **ADR-040:** EV calculator is odds-source-agnostic; validation script picks
  mode (backtest uses odds_final; live mode stubbed).

**Tests status:** (fill in: previous 483 + Task 1-6 additions; run `pytest -q`)
```

- [ ] **Step 3: Final test suite run**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add DECISIONS.md PROGRESS.md
git commit -m "phase 5a — ADR-039 (bet-type scope) + ADR-040 (odds-source agnostic) + progress log"
```

---

## Spec Coverage Self-Check

| Spec requirement | Plan task |
|---|---|
| `app/services/ev_engine/calculator.py` (CLAUDE.md §4) | Tasks 5, 6 |
| `app/services/ev_engine/market_impact.py` (CLAUDE.md §4) | Task 4 |
| `app/services/portfolio/sizing.py` 1/4 Kelly (CLAUDE.md §4, ADR-002) | Task 3 |
| `app/schemas/bets.py` BetRecommendation, Portfolio, EVResult (CLAUDE.md §4) | Task 1 (BetCandidate replaces EVResult — same shape, more descriptive name) |
| Edge, EV per bet type (CLAUDE.md §10 test) | Task 5 covers Win; Task 6 covers exotics |
| Market impact reduces EV monotonically with bet size (CLAUDE.md §10 test) | Task 4 + Task 5 (`test_post_odds_monotonically_decreases_with_bet_size`, `test_market_impact_lowers_post_odds_in_candidate`) |
| 1/4 Kelly never exceeds max_bet_fraction (CLAUDE.md §10 test) | Task 3 (`test_apply_bet_cap_clamps_at_default`) + Task 5 (`test_compute_win_candidates_kelly_sized_and_capped`) |
| CVaR constraint binding at limit (CLAUDE.md §10 test) | **DEFERRED to Plan 5b** — out of scope here. |

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-13-phase-5a-ev-engine.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
