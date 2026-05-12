"""
app/services/ordering/copula.py
───────────────────────────────
Copula-based pace-correlated ordering model (Phase 4, Layer 4).

Per CLAUDE.md §1 and ADR-001: this is the most sophisticated of the three
ordering models. PL and Stern both assume INDEPENDENT finishing times.
That assumption is wrong when horses share a pace style — two front-
runners who hook up early both tend to fade, two closers benefit from
the same fast pace, etc. The copula model captures this dependency.

Structure
─────────
Each horse i has:
    * A strength v_i ≥ 0 (Gamma rate, mirroring Stern).
    * A pace style σ_i ∈ some discrete set (e.g. {E, EP, P, S} or
      whatever taxonomy the pace model exposes).

Finishing times are constructed via a Gaussian copula with Gamma
marginals:

    Z ~ MVN(0, Σ)              with Σ_ij = ρ if σ_i = σ_j else 0
    U_i = Φ(Z_i)               probability integral transform
    T_i = Γ_inv(U_i; shape, 1/v_i)   gamma quantile with rate v_i

The ordering is `argsort(T)` ascending — lowest time wins.

Key properties
──────────────
    * ρ = 0 → independent margins → identical to Stern model with the
      same shape. We delegate to `stern.SternModel` in that case to keep
      semantics + numerics aligned.
    * ρ > 0 within a style → finishing times correlate positively. Same-
      style horses tend to finish together (both faster than expected or
      both slower). Improves the model's calibration on within-style
      exotic pairings.
    * Cross-style correlation is held at 0 (block-equicorrelation
      structure). This is the minimal extension; cross-style dependency
      is a research question we defer.
    * The Luce property (P(i wins) = strengths[i]) does NOT survive
      under ρ > 0: within a style block, the stronger horse's win share
      grows at the expense of the weaker member(s) — same-style horses
      compete for the same scenario, so independent-variation "upsets"
      shrink. To recover target marginals at ρ > 0 the caller can refit
      strengths via a fixed-point iteration analogous to
      `SternModel.infer_strengths`; we do not provide that here yet
      because the pace model is still a stub and there is no production
      pipeline consuming this output.

Mathematical notes
──────────────────
The block-equicorrelation matrix Σ with rho ∈ [0, 1) is positive
definite for any field size because each block (1-ρ)I + ρJ has
eigenvalues {1 - ρ, ρ(k-1) + 1}, both positive for ρ ∈ [0, 1). We accept
ρ ∈ [0, 1); negative correlation would require careful PSD checks and
isn't motivated by the racing literature.

Marginals: Gamma(shape=`marginal_shape`, rate=v_i). At marginal_shape=1
the marginals are exponential — matching Stern at shape=1 (i.e. PL).

Compute strategy
────────────────
Always Monte Carlo (no closed form). Per call:
    1. Build Σ from `pace_styles`.
    2. Cholesky factor L (jitter 1e-10 on the diagonal for safety).
    3. Sample MVN: Z = randn(n, N) @ L.T.
    4. U = Φ(Z) via scipy.stats.norm.cdf.
    5. T = Gamma.ppf(U; shape, 1/v_i).
    6. orderings = argsort(T, axis=1).

Public surface mirrors `stern.py`:
    CopulaConfig            — frozen config (rho, marginal_shape, n_samples, seed).
    CopulaModel
        .sample_ordering, .sample_orderings
        .exacta_prob, .trifecta_prob, .superfecta_prob
        .enumerate_exotic_probs
        .implied_win_probs
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import scipy.stats

from app.services.ordering import plackett_luce as pl
from app.services.ordering.stern import SternConfig, SternModel

# Numerical floors.
_MIN_RATE: float = 1e-12
_SUM_TOL: float = 1e-5
# Jitter added to the diagonal of the correlation matrix before Cholesky
# — guards against numerical PSD failure when ρ ≈ 1 or fields are large.
_JITTER: float = 1e-10
# Clip on U before inverse-CDF: avoid Φ(z)=0 or 1 sending Gamma.ppf to ±∞.
_U_CLIP: float = 1e-12


# ── Config & validation ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CopulaConfig:
    """Configuration for `CopulaModel`.

    rho:             Within-style Gaussian correlation. ρ ∈ [0, 1). ρ=0
                     reduces the model to Stern.
    marginal_shape:  Gamma marginal shape parameter (mirrors Stern's r).
                     marginal_shape=1 → exponential marginals → PL at ρ=0.
    n_samples:       Monte-Carlo samples drawn per probability query.
    seed:            Optional seed for the internal RNG.
    """
    rho: float = 0.3
    marginal_shape: float = 1.0
    n_samples: int = 20_000
    seed: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.rho < 1.0):
            raise ValueError(f"rho must lie in [0, 1); got {self.rho}")
        if self.marginal_shape <= 0:
            raise ValueError(f"marginal_shape must be > 0; got {self.marginal_shape}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be > 0; got {self.n_samples}")


def _validate_strengths(s: np.ndarray) -> np.ndarray:
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


def _validate_pace_styles(
    pace_styles: Sequence[Hashable] | None, n: int,
) -> np.ndarray | None:
    """Normalise pace styles to an int array (or None for the trivial case).

    Pass-through `None` means "no pace info — fall back to Stern". Returns
    an int array of style-group labels otherwise; the actual style values
    don't matter, only the equivalence classes do.
    """
    if pace_styles is None:
        return None
    styles = list(pace_styles)
    if len(styles) != n:
        raise ValueError(
            f"pace_styles length {len(styles)} does not match strengths length {n}"
        )
    label_map: dict[Hashable, int] = {}
    out = np.empty(n, dtype=np.int64)
    for i, s in enumerate(styles):
        if s not in label_map:
            label_map[s] = len(label_map)
        out[i] = label_map[s]
    return out


def _build_correlation(style_labels: np.ndarray, rho: float) -> np.ndarray:
    """Block-equicorrelation matrix: ρ within style, 0 across, 1 on diagonal."""
    n = len(style_labels)
    same = (style_labels[:, None] == style_labels[None, :])
    corr = np.where(same, rho, 0.0).astype(float)
    np.fill_diagonal(corr, 1.0)
    return corr


# ── Model ─────────────────────────────────────────────────────────────────


class CopulaModel:
    """Gaussian-copula ordering with Gamma marginals and pace-style blocks."""

    def __init__(self, config: CopulaConfig | None = None) -> None:
        self.config = config or CopulaConfig()
        self._rng = np.random.default_rng(self.config.seed)
        # Stern fallback for ρ=0; identical shape & seed so behaviour is
        # numerically continuous in ρ at the boundary.
        self._stern = SternModel(
            SternConfig(
                shape=self.config.marginal_shape,
                n_samples=self.config.n_samples,
                seed=self.config.seed,
            )
        )

    @property
    def rho(self) -> float:
        return self.config.rho

    @property
    def marginal_shape(self) -> float:
        return self.config.marginal_shape

    @property
    def n_samples(self) -> int:
        return self.config.n_samples

    # ── Sampling ──────────────────────────────────────────────────────

    def sample_orderings(
        self,
        strengths: np.ndarray,
        pace_styles: Sequence[Hashable] | None,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw `n` orderings from the copula. Shape `(n, N)`."""
        s = _validate_strengths(strengths)
        labels = _validate_pace_styles(pace_styles, len(s))
        rng = rng or self._rng

        # Trivial fallback: no pace info → independent margins (Stern).
        if labels is None or self.rho == 0.0:
            return self._stern.sample_orderings(s, n=n, rng=rng)

        rates = np.maximum(s, _MIN_RATE)
        corr = _build_correlation(labels, self.rho)
        # Cholesky factor, with jitter for numerical PSD safety.
        L = np.linalg.cholesky(corr + _JITTER * np.eye(len(rates)))
        # Sample correlated standard normals.
        z = rng.standard_normal(size=(n, len(rates))) @ L.T  # (n, N)
        # Map to uniforms via Φ.
        u = scipy.stats.norm.cdf(z)
        # Avoid exact 0 / 1 that would send gamma.ppf to ±∞.
        np.clip(u, _U_CLIP, 1 - _U_CLIP, out=u)
        # Map to Gamma marginals: shape=r, scale = 1/v_i  (per-column).
        # scipy.stats.gamma.ppf broadcasts scale over columns.
        times = scipy.stats.gamma.ppf(u, a=self.marginal_shape, scale=1.0 / rates)
        return np.argsort(times, axis=1)

    def sample_ordering(
        self,
        strengths: np.ndarray,
        pace_styles: Sequence[Hashable] | None,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Draw a single complete finishing order."""
        ordering = self.sample_orderings(strengths, pace_styles, n=1, rng=rng)[0]
        return ordering.tolist()

    # ── Implied marginals ─────────────────────────────────────────────

    def implied_win_probs(
        self, strengths: np.ndarray, pace_styles: Sequence[Hashable] | None,
    ) -> np.ndarray:
        """Model's implied P(i wins) under the copula.

        At ρ=0 and marginal_shape=1 this equals `strengths` exactly (Luce).
        Otherwise it is MC-estimated.
        """
        s = _validate_strengths(strengths)
        labels = _validate_pace_styles(pace_styles, len(s))
        if (labels is None or self.rho == 0.0) and self.marginal_shape == 1.0:
            return s.copy()
        orderings = self.sample_orderings(s, pace_styles, n=self.n_samples)
        winners = orderings[:, 0]
        counts = np.bincount(winners, minlength=len(s)).astype(float)
        return counts / counts.sum()

    # ── Exotic probabilities ──────────────────────────────────────────

    def _mc_topk_prob(
        self,
        strengths: np.ndarray,
        pace_styles: Sequence[Hashable] | None,
        prefix: tuple[int, ...],
    ) -> float:
        s = _validate_strengths(strengths)
        _validate_indices(len(s), prefix)
        orderings = self.sample_orderings(s, pace_styles, n=self.n_samples)
        k = len(prefix)
        prefix_arr = np.asarray(prefix, dtype=orderings.dtype)
        matches = np.all(orderings[:, :k] == prefix_arr[None, :], axis=1)
        return float(matches.mean())

    def exacta_prob(
        self,
        strengths: np.ndarray,
        pace_styles: Sequence[Hashable] | None,
        i: int,
        j: int,
    ) -> float:
        """P(horse i wins, horse j 2nd)."""
        # PL fast path: ρ=0 AND marginal_shape=1 ⇒ no correlation and Luce.
        s = _validate_strengths(strengths)
        labels = _validate_pace_styles(pace_styles, len(s))
        if (labels is None or self.rho == 0.0) and self.marginal_shape == 1.0:
            return pl.exacta_prob(s, i, j)
        return self._mc_topk_prob(strengths, pace_styles, (i, j))

    def trifecta_prob(
        self,
        strengths: np.ndarray,
        pace_styles: Sequence[Hashable] | None,
        i: int,
        j: int,
        k: int,
    ) -> float:
        """P(i 1st, j 2nd, k 3rd)."""
        s = _validate_strengths(strengths)
        labels = _validate_pace_styles(pace_styles, len(s))
        if (labels is None or self.rho == 0.0) and self.marginal_shape == 1.0:
            return pl.trifecta_prob(s, i, j, k)
        return self._mc_topk_prob(strengths, pace_styles, (i, j, k))

    def superfecta_prob(
        self,
        strengths: np.ndarray,
        pace_styles: Sequence[Hashable] | None,
        i: int,
        j: int,
        k: int,
        l: int,
    ) -> float:
        """P(i 1st, j 2nd, k 3rd, l 4th)."""
        s = _validate_strengths(strengths)
        labels = _validate_pace_styles(pace_styles, len(s))
        if (labels is None or self.rho == 0.0) and self.marginal_shape == 1.0:
            return pl.superfecta_prob(s, i, j, k, l)
        return self._mc_topk_prob(strengths, pace_styles, (i, j, k, l))

    def enumerate_exotic_probs(
        self,
        strengths: np.ndarray,
        pace_styles: Sequence[Hashable] | None,
        k: int,
    ) -> dict[tuple[int, ...], float]:
        """Return all length-`k` orderings and their copula probabilities.

        Analytic PL fast path at ρ=0 + marginal_shape=1. Otherwise MC: one
        sample pass tabulates every observed top-`k` prefix; all
        `N!/(N-k)!` orderings are emitted with the un-observed ones at 0.
        """
        s = _validate_strengths(strengths)
        labels = _validate_pace_styles(pace_styles, len(s))
        n = len(s)
        if k <= 0:
            raise ValueError("k must be >= 1")
        if k > n:
            raise ValueError(f"k={k} cannot exceed field size {n}")

        if (labels is None or self.rho == 0.0) and self.marginal_shape == 1.0:
            return pl.enumerate_exotic_probs(s, k)

        orderings = self.sample_orderings(s, pace_styles, n=self.n_samples)
        topk = orderings[:, :k]
        counts: dict[tuple[int, ...], int] = {}
        for row in topk:
            key = tuple(int(x) for x in row)
            counts[key] = counts.get(key, 0) + 1

        total = float(self.n_samples)
        out: dict[tuple[int, ...], float] = {}
        for perm in itertools.permutations(range(n), k):
            out[perm] = counts.get(perm, 0) / total
        return out


__all__ = [
    "CopulaConfig",
    "CopulaModel",
]
