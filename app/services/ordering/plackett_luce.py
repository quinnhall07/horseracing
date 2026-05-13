"""
app/services/ordering/plackett_luce.py
──────────────────────────────────────
Plackett-Luce ordering model (Phase 4).

Per CLAUDE.md §1 and ADR-001: Harville is prohibited. PL is the minimum
standard for converting win probabilities into exotic finishing-order
probabilities.

The PL model: each horse i has a strength parameter s_i ≥ 0. The probability
of a particular finishing order σ = (σ₁, σ₂, …, σₖ) is

    P(σ) = Π_{t=1..k}  s_{σ_t} / Σ_{m≥t} s_{σ_m}

where the denominator at step t is the sum of strengths of horses not yet
placed. Because both s and p are normalised within a race, win probabilities
ARE valid strength parameters. The win-prob input form is the production
path; the explicit MLE fit (`fit_plackett_luce_mle`) is provided so we can
validate PL against alternatives on historical orderings.

Public surface:
    exacta_prob(p, i, j)              → P(i 1st, j 2nd)
    trifecta_prob(p, i, j, k)         → P(i 1st, j 2nd, k 3rd)
    superfecta_prob(p, i, j, k, l)    → P(i 1st, j 2nd, k 3rd, l 4th)
    enumerate_exotic_probs(p, k=N)    → {tuple: prob, …} over all top-N
    sample_ordering(p, rng)           → random permutation from PL
    fit_plackett_luce_mle(orderings)  → MLE strength fit (scipy.optimize)

Notes:
    * Win probs are validated to sum to 1 and be non-negative. A tolerance
      of 1e-6 covers floating-point error from upstream softmax / calibration.
    * Repeated indices in exotic queries raise ValueError — repeated finishes
      are impossible.
    * `enumerate_exotic_probs` is O(N!/(N-k)!) — fine for k ≤ 4 in ≤ 20-horse
      fields. For superfecta over a 20-horse field this is 116,280 entries
      — large but tractable. Larger k or fields are out of scope.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# Tolerance for "win probs sum to 1" sanity check.
_SUM_TOL: float = 1e-5

# Tolerance for "denominator could be a degenerate 0" guard in place/show.
# Values below this come from p_j ≈ 1 (a certain winner) — a regime where the
# closed-form term is undefined and contributes 0 in the limit.
_DENOM_TOL: float = 1e-12


# ── Analytical helpers ────────────────────────────────────────────────────


def _validate_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if (p < -1e-12).any():
        raise ValueError(f"win probabilities must be non-negative; got min={p.min()}")
    s = p.sum()
    if abs(s - 1.0) > _SUM_TOL:
        raise ValueError(f"win probabilities must sum to 1; got {s}")
    # Clip tiny negatives from upstream floating-point noise.
    return np.clip(p, 0.0, None)


def _validate_indices(p: np.ndarray, indices: Sequence[int]) -> None:
    n = len(p)
    for idx in indices:
        if not (0 <= idx < n):
            raise IndexError(f"index {idx} out of bounds for field of size {n}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"indices must be distinct; got {indices}")


def _pl_step(p: np.ndarray, remaining_mask: np.ndarray, idx: int) -> float:
    """Return P(idx is next) given the set of horses still in the race
    (encoded as a boolean mask over p)."""
    if not remaining_mask[idx]:
        return 0.0
    denom = p[remaining_mask].sum()
    if denom <= 0.0:
        return 0.0
    return float(p[idx] / denom)


def exacta_prob(p: np.ndarray, i: int, j: int) -> float:
    """P(horse i wins, horse j 2nd)."""
    p = _validate_probs(p)
    _validate_indices(p, (i, j))
    mask = np.ones(len(p), dtype=bool)
    prob = _pl_step(p, mask, i)
    mask[i] = False
    prob *= _pl_step(p, mask, j)
    return prob


def trifecta_prob(p: np.ndarray, i: int, j: int, k: int) -> float:
    """P(i 1st, j 2nd, k 3rd)."""
    p = _validate_probs(p)
    _validate_indices(p, (i, j, k))
    mask = np.ones(len(p), dtype=bool)
    prob = _pl_step(p, mask, i)
    mask[i] = False
    prob *= _pl_step(p, mask, j)
    mask[j] = False
    prob *= _pl_step(p, mask, k)
    return prob


def superfecta_prob(p: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """P(i 1st, j 2nd, k 3rd, l 4th)."""
    p = _validate_probs(p)
    _validate_indices(p, (i, j, k, l))
    mask = np.ones(len(p), dtype=bool)
    prob = _pl_step(p, mask, i)
    mask[i] = False
    prob *= _pl_step(p, mask, j)
    mask[j] = False
    prob *= _pl_step(p, mask, k)
    mask[k] = False
    prob *= _pl_step(p, mask, l)
    return prob


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
        if denom <= _DENOM_TOL:
            # p[j] == 1 means j is certain to win; i cannot place behind j
            # because there's no second-place draw (degenerate field). Skip.
            continue
        total += p[j] * p_i / denom
    return float(total)


def show_prob(p: np.ndarray, i: int) -> float:
    """P(horse i finishes top-3) under Plackett-Luce.

    Closed form: place_prob(p, i)
              + Σ_{j≠i} Σ_{k∉{i,j}} p_j · p_k/(1−p_j) · p_i / (1 − p_j − p_k).
    Numerically guards degenerate denominators via `_DENOM_TOL`.
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
        denom_j = 1.0 - p[j]
        if denom_j <= _DENOM_TOL:
            continue
        for k in range(n):
            if k == i or k == j:
                continue
            denom_jk = 1.0 - p[j] - p[k]
            if denom_jk <= _DENOM_TOL:
                continue
            total += p[j] * (p[k] / denom_j) * (p_i / denom_jk)
    return float(total)


def enumerate_exotic_probs(p: np.ndarray, k: int) -> dict[tuple[int, ...], float]:
    """Return all ordered length-k tuples and their PL probabilities.

    The result has N!/(N-k)! entries summing to 1. Useful for grid search
    over an exotic pool (exacta box, trifecta wheel, superfecta key, …).
    """
    p = _validate_probs(p)
    n = len(p)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError(f"k={k} cannot exceed field size {n}")

    out: dict[tuple[int, ...], float] = {}
    for perm in itertools.permutations(range(n), k):
        mask = np.ones(n, dtype=bool)
        prob = 1.0
        for idx in perm:
            prob *= _pl_step(p, mask, idx)
            mask[idx] = False
        out[perm] = prob
    return out


def sample_ordering(p: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Draw one random complete finishing order from PL with strengths `p`."""
    p = _validate_probs(p)
    n = len(p)
    weights = p.copy()
    order: list[int] = []
    for _ in range(n):
        s = weights.sum()
        if s <= 0:
            # All remaining have zero strength — randomly assign.
            remaining = [i for i in range(n) if i not in order]
            rng.shuffle(remaining)
            order.extend(remaining)
            break
        probs = weights / s
        choice = int(rng.choice(n, p=probs))
        order.append(choice)
        weights[choice] = 0.0
    return order


# ── MLE strength estimation ───────────────────────────────────────────────


@dataclass
class PlackettLuceFit:
    """Result of `fit_plackett_luce_mle`."""
    strengths: np.ndarray            # length n_items, sums to 1
    log_likelihood: float
    n_orderings: int
    n_items: int
    converged: bool
    n_iter: int


def _neg_log_likelihood(
    theta: np.ndarray,
    orderings_padded: np.ndarray,
    orderings_lens: np.ndarray,
    n_items: int,
) -> float:
    """Negative log-likelihood under PL with log-strengths theta (length n_items),
    fixed scale via theta[0] = 0.

    `orderings_padded` has shape (n_orderings, n_items) — index entries beyond
    `orderings_lens[r]` are -1 sentinels. We compute the per-ordering log-prob
    in a vectorised inner loop over POSITIONS (max length n_items), which is
    much faster than nesting the python loop over orderings.
    """
    n_ord, max_len = orderings_padded.shape
    # Remaining-mask per ordering, updated as positions are filled.
    remaining = np.ones((n_ord, n_items), dtype=bool)
    # Theta of the chosen item at each position. -1 sentinel rows contribute 0.
    nll = 0.0
    for pos in range(max_len):
        active = pos < orderings_lens
        if not active.any():
            break
        chosen = orderings_padded[:, pos]
        # For rows that are still active, accumulate.
        # logsumexp over remaining items per row.
        masked_theta = np.where(remaining, theta[None, :], -np.inf)
        denom = logsumexp(masked_theta, axis=1)
        chosen_theta = np.where(active, theta[np.where(active, chosen, 0)], 0.0)
        denom = np.where(active, denom, 0.0)
        nll -= np.sum(chosen_theta - denom)
        # Knock out the chosen item from the remaining set for active rows.
        row_idx = np.arange(n_ord)
        # We only update remaining for active rows; for inactive rows the
        # value at remaining[row, chosen_safe] is irrelevant since `active`
        # gates contributions in subsequent positions too.
        chosen_safe = np.where(active, chosen, 0)
        remaining[row_idx, chosen_safe] = remaining[row_idx, chosen_safe] & ~active
    return float(nll)


def fit_plackett_luce_mle(
    orderings: Sequence[Sequence[int]],
    n_items: int,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> PlackettLuceFit:
    """Fit per-item PL strengths via MLE on a corpus of observed orderings.

    Each `ordering` is a sequence of item indices, winner first. Partial
    orderings (length < n_items) are supported — only the observed top-k
    positions contribute to the likelihood.

    The scale is fixed by holding θ[0] = 0 (equivalently, s[0] = 1). After
    optimisation the strengths are normalised to sum to 1.
    """
    if len(orderings) == 0:
        raise ValueError("orderings must be non-empty")
    parsed: list[list[int]] = []
    for order in orderings:
        arr = list(order)
        if not arr:
            continue
        for idx in arr:
            if not (0 <= idx < n_items):
                raise ValueError(
                    f"ordering {arr} contains index outside [0, {n_items})"
                )
        parsed.append(arr)
    if not parsed:
        raise ValueError("all provided orderings were empty")

    # Pack orderings into a rectangular int array padded with 0 (since the
    # corresponding position is gated off by `orderings_lens`).
    lens = np.array([len(o) for o in parsed], dtype=np.int64)
    max_len = int(lens.max())
    padded = np.zeros((len(parsed), max_len), dtype=np.int64)
    for r, o in enumerate(parsed):
        padded[r, : len(o)] = o

    # Free parameters: theta[1..n-1]. theta[0] is pinned to 0.
    def _unpack(free: np.ndarray) -> np.ndarray:
        theta = np.zeros(n_items)
        theta[1:] = free
        return theta

    def _objective(free: np.ndarray) -> float:
        return _neg_log_likelihood(_unpack(free), padded, lens, n_items)

    x0 = np.zeros(n_items - 1)
    result = minimize(
        _objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    theta_hat = _unpack(result.x)
    # Convert log-strengths to strengths and renormalise.
    s = np.exp(theta_hat - theta_hat.max())  # subtract max for stability
    s = s / s.sum()
    return PlackettLuceFit(
        strengths=s,
        log_likelihood=float(-result.fun),
        n_orderings=len(parsed),
        n_items=n_items,
        converged=bool(result.success),
        n_iter=int(result.nit),
    )


__all__ = [
    "PlackettLuceFit",
    "enumerate_exotic_probs",
    "exacta_prob",
    "fit_plackett_luce_mle",
    "place_prob",
    "sample_ordering",
    "show_prob",
    "superfecta_prob",
    "trifecta_prob",
]
