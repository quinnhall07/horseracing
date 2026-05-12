"""
app/services/ordering/stern.py
──────────────────────────────
Stern (Gamma) ordering model (Phase 4).

Per CLAUDE.md §1 and ADR-001: Plackett-Luce is the minimum standard, Stern
Gamma is the preferred upgrade for single-race exotics. The Stern (Henery /
Stern 1990) model places finishing times on a Gamma distribution:

    T_i ~ Gamma(shape=r, rate=v_i)         scale = 1/v_i

and the observed ordering is the rank of (T_1, …, T_N) ascending. Lowest
time wins.

Key properties:
    * When r = 1 the Gamma collapses to Exponential and the model is
      mathematically identical to Plackett-Luce: P(i wins) = v_i / Σ v_j,
      and all top-k order probabilities have the Luce closed form. In that
      case we delegate to `plackett_luce` for analytical results — Monte
      Carlo would only add noise.
    * When r > 1 finishing times have lower CV (CV = 1/√r), so the ordering
      distribution sharpens: strong horses win more often than under PL
      with the same `strengths`.
    * When 0 < r < 1 finishing times disperse, so longshots win more
      often than PL would suggest.

The shape parameter r therefore behaves as a "determinism knob" once the
caller has chosen a strength vector.

Strengths vs. win probabilities
───────────────────────────────
Strengths in `strengths` are Gamma RATE parameters. They have the same
units as PL strengths and must be non-negative; we normalise to sum to 1
to fix the scale (the model is invariant under rescaling all rates).

    * At r = 1.0 the Luce property gives P(i wins) = strengths[i] exactly,
      so the calibrated win-prob vector IS the strengths vector.
    * At r ≠ 1.0 the Luce property does NOT hold; the implied marginal
      P(i wins) differs from strengths[i]. Use `implied_win_probs(...)` to
      inspect what the model actually predicts.
    * To preserve calibration at r ≠ 1.0 use `infer_strengths(target, ...)`
      which solves for rates such that the implied marginals match the
      target win-prob vector (fixed-point iteration over Monte-Carlo
      estimates).

Compute strategy
────────────────
    * Closed form via `plackett_luce.*` when shape == 1.0.
    * Monte Carlo otherwise. Default 20,000 samples puts the standard
      error on an exacta probability below 0.005 (binomial SE on p≈0.1).
      Tune via `SternConfig.n_samples`.

Public surface mirrors `plackett_luce.py`:
    SternConfig             — frozen config (shape, n_samples, seed).
    SternModel
        .sample_ordering, .sample_orderings
        .exacta_prob, .trifecta_prob, .superfecta_prob
        .enumerate_exotic_probs
        .implied_win_probs
        .infer_strengths
    SternFit, fit_stern_shape(...)  — grid-search MLE for r given
                                       observed orderings + strengths.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from app.services.ordering import plackett_luce as pl

# Numerical floor on rates so 1/v is finite in the sampler.
_MIN_RATE: float = 1e-12
# Tolerance for "strengths sum to 1" sanity check (matches PL).
_SUM_TOL: float = 1e-5


# ── Config & validation helpers ───────────────────────────────────────────


@dataclass(frozen=True)
class SternConfig:
    """Configuration for `SternModel`.

    shape:     Gamma shape parameter r > 0. r = 1 reduces to PL.
    n_samples: Monte-Carlo samples drawn per probability query when shape != 1.
    seed:      Optional seed for the internal `np.random.Generator`. None
               builds a fresh RNG with OS entropy.
    """
    shape: float = 1.0
    n_samples: int = 20_000
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.shape <= 0:
            raise ValueError(f"shape must be > 0; got {self.shape}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be > 0; got {self.n_samples}")


def _validate_strengths(s: np.ndarray) -> np.ndarray:
    """Validate strength vector and clip tiny negatives.

    Strengths must be non-negative and sum to 1 within `_SUM_TOL`. We
    accept a small floating-point tolerance because in production these
    will arrive from a softmax / calibrator and may carry rounding noise.
    """
    s = np.asarray(s, dtype=float).ravel()
    if (s < -1e-12).any():
        raise ValueError(f"strengths must be non-negative; got min={s.min()}")
    total = s.sum()
    if abs(total - 1.0) > _SUM_TOL:
        raise ValueError(f"strengths must sum to 1; got {total}")
    return np.clip(s, 0.0, None)


def _validate_indices(n: int, indices: Sequence[int]) -> None:
    for idx in indices:
        if not (0 <= idx < n):
            raise IndexError(f"index {idx} out of bounds for field of size {n}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"indices must be distinct; got {indices}")


# ── Model ─────────────────────────────────────────────────────────────────


class SternModel:
    """Stern Gamma ordering model.

    Stateless except for the RNG: `predict`-style methods can be called
    repeatedly with different strength vectors. For deterministic outputs
    pass `seed` in the config; otherwise the RNG draws fresh entropy.
    """

    def __init__(self, config: SternConfig | None = None) -> None:
        self.config = config or SternConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # Convenience accessors.
    @property
    def shape(self) -> float:
        return self.config.shape

    @property
    def n_samples(self) -> int:
        return self.config.n_samples

    # ── Sampling ──────────────────────────────────────────────────────

    def sample_orderings(
        self,
        strengths: np.ndarray,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw `n` complete finishing orders from the model.

        Returns an `(n, N)` int array, each row a permutation of
        `range(N)` ordered by ascending Gamma-sampled finishing times.
        """
        s = _validate_strengths(strengths)
        rng = rng or self._rng
        rates = np.maximum(s, _MIN_RATE)
        scales = 1.0 / rates  # (N,)
        # numpy broadcasts (N,) scales over the (n, N) size argument.
        times = rng.gamma(self.shape, scales, size=(n, len(rates)))
        return np.argsort(times, axis=1)

    def sample_ordering(
        self, strengths: np.ndarray, rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Draw a single complete finishing order (returned as a list)."""
        ordering = self.sample_orderings(strengths, n=1, rng=rng)[0]
        return ordering.tolist()

    # ── Implied marginals & calibration ───────────────────────────────

    def implied_win_probs(self, strengths: np.ndarray) -> np.ndarray:
        """Return the model's implied P(i wins) for each horse.

        For shape == 1.0 this is the input strengths (Luce property).
        Otherwise it is Monte-Carlo estimated from `n_samples` draws.
        """
        s = _validate_strengths(strengths)
        if self.shape == 1.0:
            return s.copy()
        orderings = self.sample_orderings(s, n=self.n_samples)
        winners = orderings[:, 0]
        counts = np.bincount(winners, minlength=len(s)).astype(float)
        return counts / counts.sum()

    def infer_strengths(
        self,
        target_win_probs: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-3,
        damping: float = 0.5,
    ) -> np.ndarray:
        """Find Gamma rates `v` such that implied P(i wins) ≈ target.

        Uses a damped multiplicative fixed-point iteration:

            v ← v * (target / implied)^damping
            v ← v / Σ v

        Damping (0 < damping ≤ 1) is needed because the response of
        implied marginals to rate scaling is super-linear at large r;
        undamped updates oscillate. damping=0.5 converges robustly for
        all r ≤ 5 in our testing.

        At shape == 1.0 the answer is exactly `target_win_probs` (Luce).
        Convergence is measured in L∞ on the marginals, not the rates.
        """
        target = _validate_strengths(target_win_probs)
        if self.shape == 1.0:
            return target.copy()

        v = target.copy()
        v = np.maximum(v, _MIN_RATE)
        v = v / v.sum()
        for _ in range(max_iter):
            implied = self.implied_win_probs(v)
            err = float(np.abs(implied - target).max())
            if err < tol:
                break
            ratio = target / np.maximum(implied, _MIN_RATE)
            v = v * np.power(ratio, damping)
            v = np.maximum(v, _MIN_RATE)
            v = v / v.sum()
        return v

    # ── Exotic probabilities ──────────────────────────────────────────

    def _mc_topk_prob(self, strengths: np.ndarray, prefix: tuple[int, ...]) -> float:
        """Fraction of MC orderings whose top-`k` matches `prefix` exactly."""
        s = _validate_strengths(strengths)
        _validate_indices(len(s), prefix)
        orderings = self.sample_orderings(s, n=self.n_samples)
        k = len(prefix)
        prefix_arr = np.asarray(prefix, dtype=orderings.dtype)
        matches = np.all(orderings[:, :k] == prefix_arr[None, :], axis=1)
        return float(matches.mean())

    def exacta_prob(self, strengths: np.ndarray, i: int, j: int) -> float:
        """P(horse i wins, horse j 2nd)."""
        if self.shape == 1.0:
            return pl.exacta_prob(strengths, i, j)
        return self._mc_topk_prob(strengths, (i, j))

    def trifecta_prob(self, strengths: np.ndarray, i: int, j: int, k: int) -> float:
        """P(i 1st, j 2nd, k 3rd)."""
        if self.shape == 1.0:
            return pl.trifecta_prob(strengths, i, j, k)
        return self._mc_topk_prob(strengths, (i, j, k))

    def superfecta_prob(
        self, strengths: np.ndarray, i: int, j: int, k: int, l: int,
    ) -> float:
        """P(i 1st, j 2nd, k 3rd, l 4th)."""
        if self.shape == 1.0:
            return pl.superfecta_prob(strengths, i, j, k, l)
        return self._mc_topk_prob(strengths, (i, j, k, l))

    def enumerate_exotic_probs(
        self, strengths: np.ndarray, k: int,
    ) -> dict[tuple[int, ...], float]:
        """Return all length-`k` orderings and their model probabilities.

        At shape == 1.0 returns the analytic PL enumeration. Otherwise a
        single MC pass tabulates counts of every observed top-`k` prefix
        in `n_samples` orderings; un-observed prefixes are emitted with
        probability 0.0 so the output set is exactly `N!/(N-k)!` entries
        for parity with PL.

        Note: probabilities sum to ≤ 1 exactly under MC due to sampling
        noise; we DO NOT renormalise. Bump `n_samples` if a downstream
        consumer requires sum-to-1.
        """
        s = _validate_strengths(strengths)
        n = len(s)
        if k <= 0:
            raise ValueError("k must be >= 1")
        if k > n:
            raise ValueError(f"k={k} cannot exceed field size {n}")

        if self.shape == 1.0:
            return pl.enumerate_exotic_probs(s, k)

        orderings = self.sample_orderings(s, n=self.n_samples)
        topk = orderings[:, :k]
        # Hash each row to a tuple for counting.
        counts: dict[tuple[int, ...], int] = {}
        for row in topk:
            key = tuple(int(x) for x in row)
            counts[key] = counts.get(key, 0) + 1

        total = float(self.n_samples)
        out: dict[tuple[int, ...], float] = {}
        for perm in itertools.permutations(range(n), k):
            out[perm] = counts.get(perm, 0) / total
        return out


# ── MLE for the shape parameter ───────────────────────────────────────────


@dataclass
class SternFit:
    """Result of `fit_stern_shape`."""
    shape: float
    log_likelihood: float
    n_orderings: int
    method: str
    shape_grid: list[float]
    ll_per_shape: list[float]
    top_k: int


def fit_stern_shape(
    orderings_with_strengths: Sequence[tuple[Sequence[int], np.ndarray]],
    shape_grid: Sequence[float] | None = None,
    n_samples: int = 10_000,
    top_k: int = 2,
    seed: int = 0,
) -> SternFit:
    """Grid-search MLE for the Gamma shape parameter `r`.

    Each ordering arrives with its own race-specific strength vector
    (these are typically the calibrated win probabilities for that race).
    For each candidate shape we draw `n_samples` orderings from the model
    using the race's strengths, then estimate P(observed top-`top_k`)
    empirically and accumulate log-likelihood.

    Laplace smoothing replaces a zero-match count by 1/(2·n_samples) so
    rare events don't crash the log. This biases shapes that miss
    observations toward each other but does not affect the relative
    ranking among shapes that DO produce matches.

    Why top-`top_k`, not full ordering?
        * The full-ordering MC probability is ≈ 0 for k > 4 at any
          realistic n_samples — every sample collides with the observed
          ordering with vanishing probability, leaving only the Laplace
          floor and erasing the shape signal.
        * Top-2 (winner + place) is the smallest statistic that
          identifies the shape parameter: top-1 alone is invariant under
          shape when strengths = marginal win probs.

    Returns a `SternFit` reporting the maximising shape, its log-likelihood,
    and the full grid of (shape, LL) for inspection / plotting.
    """
    if len(orderings_with_strengths) == 0:
        raise ValueError("orderings_with_strengths must be non-empty")
    if top_k < 2:
        raise ValueError("top_k must be >= 2; top-1 is invariant under shape")
    if shape_grid is None:
        shape_grid = list(np.geomspace(0.5, 4.0, 11))
    shape_grid = [float(x) for x in shape_grid]
    for r in shape_grid:
        if r <= 0:
            raise ValueError(f"shape_grid contains non-positive value {r}")

    rng = np.random.default_rng(seed)
    ll_per_shape: list[float] = []
    laplace_eps = 1.0 / (2.0 * n_samples)

    for r in shape_grid:
        model = SternModel(SternConfig(shape=r, n_samples=n_samples, seed=seed))
        total_ll = 0.0
        for order, strengths in orderings_with_strengths:
            order_list = list(order)
            if len(order_list) < top_k:
                continue
            prefix = tuple(int(x) for x in order_list[:top_k])
            s = _validate_strengths(np.asarray(strengths))
            sampled = model.sample_orderings(s, n=n_samples, rng=rng)
            prefix_arr = np.asarray(prefix, dtype=sampled.dtype)
            matches = np.all(sampled[:, :top_k] == prefix_arr[None, :], axis=1)
            p_match = float(matches.mean())
            if p_match <= 0.0:
                p_match = laplace_eps
            total_ll += math.log(p_match)
        ll_per_shape.append(total_ll)

    best_idx = int(np.argmax(ll_per_shape))
    return SternFit(
        shape=shape_grid[best_idx],
        log_likelihood=ll_per_shape[best_idx],
        n_orderings=len(orderings_with_strengths),
        method="grid",
        shape_grid=shape_grid,
        ll_per_shape=ll_per_shape,
        top_k=top_k,
    )


__all__ = [
    "SternConfig",
    "SternFit",
    "SternModel",
    "fit_stern_shape",
]
