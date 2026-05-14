"""
app/services/inference/pipeline.py
─────────────────────────────────
Stream A — Server-side inference pipeline.

This module is the canonical "parsed RaceCard → calibrated win probs →
+EV bet candidates → (optional) Portfolio" path used by both the FastAPI
endpoints (`/api/v1/cards/{id}`, `/api/v1/portfolio/{id}`) and any future
batch scripts. It is intentionally framework-free — pure functions over
the canonical Pydantic schemas (RaceCard, BetCandidate, Portfolio) and
the trained-model artifact directory layout.

Per ADR-026 / module docstrings in `app/services/models/*.py`: the
Pace, Sequence, and (when loaded fresh) the Speed/Form scaffolding
classes default to a neutral 0.5 fallback when their artifact is
missing or unloadable. The pipeline tolerates this — only the
Speed/Form, Connections, Market, and Meta-learner artifacts are
strictly required; others degrade gracefully.

Live-inference adapter
──────────────────────
The trained models (Layer 1a Speed/Form, etc.) consume the feature
matrix produced by `app/services/models/training_data.prepare_training_features`.
That function's "shift(1) before rolling" trick produces per-row
features that depend ONLY on a horse's prior race history. For live
inference we synthesise the SAME feature columns directly from the
parsed `HorseEntry.pp_lines` (which IS the prior history). This adapter
lives in `build_inference_features` — it returns a DataFrame with the
exact column names the trained models expect.

Public API
──────────
    InferenceArtifacts                      ← container for all loaded models
    InferenceArtifacts.load(models_dir)
    race_card_to_features(card, fe=None)    ← FeatureEngine output (live FE)
    build_inference_features(card)          ← columns the trained models want
    infer_calibrated_win_probs(features,
                               artifacts,
                               race_id)     ← np.ndarray per single race
    analyze_card(card, artifacts, …)        ← (probs_by_race, candidates, portfolios)
    analyze_card_pareto(card, artifacts, …) ← (risk_level, aggregated_portfolio) pairs
    build_portfolio_from_candidates(...)    ← interim Kelly-scaling constructor
                                              (kept for backward-compat; the
                                              Phase 5b R-U LP is now the
                                              default — see ADR-045)

Per ADR-040: the EV calculator is odds-source agnostic. Live mode
extracts decimal odds from `HorseEntry.morning_line_odds`, caps them
at `max_decimal_odds` (Phase 5b smoke-finding mitigation), and drops
the race when any horse is missing odds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.schemas.bets import BetCandidate, BetRecommendation, Portfolio
from app.schemas.provenance import ModelProvenance
from app.schemas.race import BetType, HorseEntry, ParsedRace, PastPerformanceLine, RaceCard
from app.services.calibration.calibrator import Calibrator
from app.services.ev_engine.calculator import compute_ev_candidates
from app.services.feature_engineering.engine import FeatureEngine
from app.services.models.connections import ConnectionsModel
from app.services.models.market import MarketModel
from app.services.models.meta_learner import MetaLearner
from app.services.models.pace_scenario import PaceScenarioModel
from app.services.models.sequence import SequenceModel
from app.services.models.speed_form import SpeedFormModel
from app.services.models.training_data import (
    EWM_ALPHA,
    LAYOFF_DECAY_LAMBDA,
    LAYOFF_RECOVERY_THRESHOLD_DAYS,
    ROLLING_WINDOW,
    FIRST_TIME_STARTER_FITNESS,
)
from app.services.portfolio.optimizer import optimize_portfolio
from app.services.portfolio.sizing import apply_bet_cap

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_EDGE: float = 0.05
DEFAULT_MAX_DECIMAL_ODDS: float = 100.0
"""Phase 5b smoke-finding mitigation (see ADR-040 follow-up note): cap
decimal odds at this value before feeding the EV calculator. Guards against
99/999 placeholder values from incomplete data sources producing nonsense
edges. 100.0 ⇒ implied prob ≥ 1%, which is roughly the lowest live-tote
horse you'd see."""

DEFAULT_BANKROLL: float = 10_000.0
DEFAULT_CVAR_ALPHA: float = 0.05
DEFAULT_MAX_DRAWDOWN_PCT: float = 0.20
DEFAULT_N_SCENARIOS: int = 1000
DEFAULT_SEED: int = 42
DEFAULT_BET_TYPES: tuple[BetType, ...] = (
    BetType.WIN,
    BetType.EXACTA,
    BetType.TRIFECTA,
    BetType.SUPERFECTA,
)


# ──────────────────────────────────────────────────────────────────────────────
# Inference artifacts container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class InferenceArtifacts:
    """Container for all trained-model artifacts loaded at FastAPI startup.

    All five sub-models + the meta-learner + the meta-calibrator are
    held in memory. `load` is tolerant of missing artifacts (per ADR-026,
    a missing sub-model degrades to its neutral 0.5 fallback rather than
    failing the whole pipeline) — but it WILL raise if the meta-learner
    or the meta-calibrator is missing, since those are the load-bearing
    pieces that turn sub-model outputs into a calibrated win prob.
    """

    speed_form: Optional[SpeedFormModel]
    pace_scenario: PaceScenarioModel
    sequence: SequenceModel
    connections: Optional[ConnectionsModel]
    market: Optional[MarketModel]
    meta_learner: MetaLearner
    meta_calibrator: Calibrator
    models_dir: Path
    available_sub_models: tuple[str, ...] = field(default_factory=tuple)
    provenance: ModelProvenance = field(
        default_factory=lambda: ModelProvenance(
            is_synthetic=True,
            warning=(
                "BOOTSTRAP_PROVENANCE.json missing — treating loaded models as "
                "synthetic. Re-run scripts/bootstrap_models.py on a real "
                "training parquet to clear this warning."
            ),
        )
    )

    @classmethod
    def load(
        cls,
        models_dir: Path,
        calibration_subdir: str = "calibration_adr038_brier",
    ) -> "InferenceArtifacts":
        """Load every artifact under `models_dir`. Returns even if optional
        sub-models are missing — see class docstring.

        Layout (per Phase 3/4 PROGRESS notes):
            models_dir/
              speed_form/                         (LightGBM booster)
              connections/                        (Bayesian rate JSON)
              market/                             (Isotonic JSON)
              meta_learner/                       (LightGBM stacker)
              {calibration_subdir}/meta_learner/  (Platt/Iso meta-cal)
        """
        models_dir = Path(models_dir)
        if not models_dir.exists():
            raise FileNotFoundError(f"models_dir does not exist: {models_dir}")

        available: list[str] = []

        # ── Speed/Form (Layer 1a) — strictly required when present in
        # production but tolerated as missing (with neutral fallback).
        speed_form: Optional[SpeedFormModel] = None
        sf_path = models_dir / "speed_form"
        if (sf_path / "metadata.json").exists():
            try:
                speed_form = SpeedFormModel.load(sf_path)
                available.append("speed_form")
            except Exception as exc:  # noqa: BLE001
                log.warning("inference.speed_form_load_failed", error=str(exc))

        # ── Pace + Sequence: both can be either trained (ADR-046/047 — the
        # parquet provides the data) or a 0.5-fallback stub. Either way,
        # .load() returns a usable instance.
        pace = PaceScenarioModel.load(models_dir / "pace_scenario")
        sequence = SequenceModel.load(models_dir / "sequence")
        if getattr(pace, "is_fitted", False):
            available.append("pace_scenario")
        if getattr(sequence, "is_fitted", False):
            available.append("sequence")

        # ── Connections (Layer 1d)
        connections: Optional[ConnectionsModel] = None
        cx_path = models_dir / "connections"
        if (cx_path / "model.json").exists():
            try:
                connections = ConnectionsModel.load(cx_path)
                available.append("connections")
            except Exception as exc:  # noqa: BLE001
                log.warning("inference.connections_load_failed", error=str(exc))

        # ── Market (Layer 1e)
        market: Optional[MarketModel] = None
        mk_path = models_dir / "market"
        if (mk_path / "model.json").exists():
            try:
                market = MarketModel.load(mk_path)
                available.append("market")
            except Exception as exc:  # noqa: BLE001
                log.warning("inference.market_load_failed", error=str(exc))

        # ── Meta-learner (Layer 2) — REQUIRED
        meta_path = models_dir / "meta_learner"
        if not (meta_path / "metadata.json").exists():
            raise FileNotFoundError(
                f"meta_learner artifact required at {meta_path}; cannot "
                "proceed with inference without it."
            )
        meta_learner = MetaLearner.load(meta_path)
        available.append("meta_learner")

        # ── Calibrator (Phase 4) — REQUIRED
        cal_path = models_dir / calibration_subdir / "meta_learner"
        if not (cal_path / "metadata.json").exists():
            raise FileNotFoundError(
                f"meta calibrator artifact required at {cal_path}; "
                "cannot proceed with inference without it."
            )
        meta_calibrator = Calibrator.load(cal_path)
        available.append("meta_calibrator")

        provenance = _load_provenance(models_dir)

        log.info(
            "inference.artifacts_loaded",
            models_dir=str(models_dir),
            available=available,
            is_synthetic=provenance.is_synthetic,
        )

        return cls(
            speed_form=speed_form,
            pace_scenario=pace,
            sequence=sequence,
            connections=connections,
            market=market,
            meta_learner=meta_learner,
            meta_calibrator=meta_calibrator,
            models_dir=models_dir,
            available_sub_models=tuple(available),
            provenance=provenance,
        )


def _load_provenance(models_dir: Path) -> ModelProvenance:
    """Read BOOTSTRAP_PROVENANCE.json next to the loaded artifacts.

    Falls back to a synthetic-flagged ModelProvenance with a warning when the
    file is missing or unreadable — the dataclass default is identical to this
    fallback, so callers can rely on `provenance.is_synthetic` always being a
    safe bool to switch on.
    """
    path = models_dir / "BOOTSTRAP_PROVENANCE.json"
    if not path.exists():
        return ModelProvenance(
            is_synthetic=True,
            warning=(
                f"BOOTSTRAP_PROVENANCE.json missing from {models_dir} — treating "
                "loaded models as synthetic. Re-run scripts/bootstrap_models.py "
                "on a real training parquet to clear this warning."
            ),
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("inference.provenance_unreadable", path=str(path), error=str(exc))
        return ModelProvenance(
            is_synthetic=True,
            warning=f"BOOTSTRAP_PROVENANCE.json unreadable: {exc}",
        )
    try:
        return ModelProvenance.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("inference.provenance_invalid", path=str(path), error=str(exc))
        return ModelProvenance(
            is_synthetic=True,
            warning=f"BOOTSTRAP_PROVENANCE.json invalid: {exc}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────────────────────


def race_card_to_features(
    card: RaceCard,
    feature_engine: Optional[FeatureEngine] = None,
) -> pd.DataFrame:
    """Run the live FeatureEngine on a parsed RaceCard. Returns one row per
    horse (across all races on the card). The returned columns are the
    field-relative summary the UI consumes; the ML models receive the
    `build_inference_features` output instead — see that function's docstring.
    """
    fe = feature_engine or FeatureEngine()
    return fe.transform(card)


def _layoff_fitness(days: Optional[int]) -> float:
    """Twin of `training_data._layoff_fitness` for scalar inputs."""
    if days is None or (isinstance(days, float) and math.isnan(days)):
        return FIRST_TIME_STARTER_FITNESS
    if days <= LAYOFF_RECOVERY_THRESHOLD_DAYS:
        return 1.0
    return float(np.exp(-LAYOFF_DECAY_LAMBDA * (days - LAYOFF_RECOVERY_THRESHOLD_DAYS)))


def _ewm_mean(values: Sequence[float], alpha: float = EWM_ALPHA) -> Optional[float]:
    """EWM of a most-recent-first sequence (PP order). Mirrors training_data's
    `groupby.shift(1).ewm()` — since PP lines do NOT include today's race,
    no shift is needed: the most recent PP IS the prior race."""
    if not values:
        return None
    oldest_first = list(reversed(values))
    s = pd.Series(oldest_first, dtype=float)
    return float(s.ewm(alpha=alpha, adjust=True).mean().iloc[-1])


def _rolling_mean(values: Sequence[float], window: int = ROLLING_WINDOW) -> Optional[float]:
    """Mean of the most recent `window` values."""
    if not values:
        return None
    return float(np.mean(values[:window]))


def _rolling_max(values: Sequence[float], window: int = ROLLING_WINDOW) -> Optional[float]:
    if not values:
        return None
    return float(np.max(values[:window]))


def _build_horse_row(
    entry: HorseEntry,
    race: ParsedRace,
    card_track: str,
    jurisdiction: str = "US",
) -> dict:
    """One row per horse in the inference feature matrix.

    Output columns mirror `app.services.models.training_data.SPEED_FORM_FEATURE_COLUMNS`
    plus the meta-learner's `meta_feature_columns` and the stacker's
    contextual fields (race_id, jurisdiction, etc.).
    """
    pps = list(entry.pp_lines)  # most-recent-first per HorseEntry validator
    speed_figs: list[float] = [
        float(p.speed_figure) for p in pps if p.speed_figure is not None
    ]
    finish_positions: list[float] = [
        float(p.finish_position) for p in pps if p.finish_position is not None
    ]
    field_sizes: list[float] = [
        float(p.field_size) for p in pps if p.field_size is not None
    ]
    purses: list[float] = [float(p.purse_usd) for p in pps if p.purse_usd is not None]

    finish_pcts: list[float] = []
    for p in pps:
        if p.finish_position is not None and p.field_size and p.field_size > 0:
            finish_pcts.append(float(p.finish_position) / float(p.field_size))

    wins: list[float] = [
        1.0 if (p.finish_position == 1) else 0.0
        for p in pps if p.finish_position is not None
    ]

    last_speed_prior = speed_figs[0] if speed_figs else None
    speed_delta_prior = (
        (speed_figs[0] - speed_figs[1]) if len(speed_figs) >= 2 else None
    )

    # days_since_prev: gap between today's race and the most recent PP.
    if pps:
        days_since_prev: Optional[int] = (race.header.race_date - pps[0].race_date).days
    else:
        days_since_prev = None

    row: dict = {
        # ── Identifiers ────────────────────────────────────────────────────
        "race_id": _live_race_id(race, card_track),
        "race_date": pd.Timestamp(race.header.race_date),
        "race_number": race.header.race_number,
        "track_code": race.header.track_code,
        "post_position": entry.post_position,
        "horse_name": entry.horse_name,
        "horse_key": f"{entry.horse_name}|{jurisdiction}",
        "jurisdiction": jurisdiction,
        # ── Per-horse prior aggregates ────────────────────────────────────
        "ewm_speed_prior": _ewm_mean(speed_figs),
        "last_speed_prior": last_speed_prior,
        "best_speed_prior": _rolling_max(speed_figs),
        "mean_speed_prior": _rolling_mean(speed_figs),
        "speed_delta_prior": speed_delta_prior,
        "n_prior_starts": float(len(pps)),
        "mean_finish_pos_prior": _rolling_mean(finish_positions),
        "mean_finish_pct_prior": _rolling_mean(finish_pcts) if finish_pcts else None,
        "win_rate_prior": _rolling_mean(wins) if wins else None,
        "mean_purse_prior": _rolling_mean(purses),
        # ── Layoff + scheduling ──────────────────────────────────────────
        "days_since_prev": days_since_prev,
        "layoff_fitness": _layoff_fitness(days_since_prev),
        # ── Today's race scalars ─────────────────────────────────────────
        "distance_furlongs": float(race.header.distance_furlongs),
        "field_size": int(race.field_size),
        "weight_lbs": float(entry.weight_lbs) if entry.weight_lbs is not None else None,
        "purse_usd": (
            float(race.header.purse_usd) if race.header.purse_usd is not None else None
        ),
        "surface": race.header.surface.value,
        "condition": race.header.condition.value,
        "race_type": race.header.race_type.value,
        # ── Connections columns (used by ConnectionsModel) ─────────────────
        "jockey_name": entry.jockey or "",
        "trainer_name": entry.trainer or "",
        # ── Odds (used by MarketModel for backtest; here as ML odds) ──────
        "odds_final": (
            float(entry.morning_line_odds)
            if entry.morning_line_odds is not None
            else None
        ),
    }
    return row


def _live_race_id(race: ParsedRace, card_track: str) -> str:
    """Stable race identifier built from (date, track, race#)."""
    return (
        race.header.race_date.strftime("%Y%m%d")
        + "|" + str(race.header.track_code or card_track or "??")
        + "|" + str(race.header.race_number)
    )


def build_inference_features(
    card: RaceCard,
    jurisdiction: str = "US",
) -> pd.DataFrame:
    """Build the ML-ready feature DataFrame the trained sub-models consume.

    The output schema mirrors `prepare_training_features` so the loaded
    booster's `feature_columns` slot in directly. Field-relative columns
    (`ewm_speed_zscore`, `ewm_speed_rank`, `ewm_speed_pct`, `weight_lbs_delta`)
    are computed AFTER the per-horse rows are assembled, grouped by race_id.

    The categorical columns (surface/condition/race_type/jurisdiction) are
    coerced to `category` dtype so LightGBM picks them up natively.
    """
    rows: list[dict] = []
    card_track = card.track_code or ""
    for race in card.races:
        for entry in race.entries:
            rows.append(_build_horse_row(entry, race, card_track, jurisdiction))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Field-relative columns (within each race).
    grp = df.groupby("race_id", sort=False)

    df["ewm_speed_zscore"] = grp["ewm_speed_prior"].transform(_safe_zscore)
    df["ewm_speed_rank"] = grp["ewm_speed_prior"].rank(
        ascending=False, method="min", na_option="keep"
    )
    df["ewm_speed_pct"] = grp["ewm_speed_prior"].rank(pct=True, na_option="keep")
    df["weight_lbs_delta"] = (
        df["weight_lbs"] - grp["weight_lbs"].transform("mean")
    )

    # Categoricals
    for col in ("surface", "condition", "race_type", "jurisdiction"):
        df[col] = df[col].astype("category")

    return df


def _safe_zscore(s: pd.Series) -> pd.Series:
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)
    if not std or pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


# ──────────────────────────────────────────────────────────────────────────────
# Scoring (live mirror of validate_phase5a_ev_engine._score_test_slice)
# ──────────────────────────────────────────────────────────────────────────────


def _stack_for_meta(
    df: pd.DataFrame,
    artifacts: InferenceArtifacts,
) -> pd.DataFrame:
    """Append the five sub-model columns the meta-learner consumes.

    Per ADR-026 / model module docstrings: missing artifacts are tolerated
    — predict_proba falls back to a constant 0.5 vector.
    """
    stacked = df.copy()

    if artifacts.speed_form is not None:
        stacked["speed_form_proba"] = artifacts.speed_form.predict_proba(df)
    else:
        log.warning("inference.speed_form_missing_using_fallback")
        stacked["speed_form_proba"] = 0.5

    stacked["pace_scenario_proba"] = artifacts.pace_scenario.predict_proba(df)
    stacked["sequence_proba"] = artifacts.sequence.predict_proba(df)

    if artifacts.connections is not None:
        stacked["connections_proba"] = artifacts.connections.predict_proba(df)
    else:
        log.warning("inference.connections_missing_using_fallback")
        stacked["connections_proba"] = 0.5

    if artifacts.market is not None:
        m = artifacts.market.predict_proba(df)
        nan_mask = np.isnan(m)
        if nan_mask.any():
            replacement = float(np.nanmean(m)) if np.isfinite(np.nanmean(m)) else 0.5
            m = np.where(nan_mask, replacement, m)
        stacked["market_proba"] = m
        stacked["market_proba_was_missing"] = nan_mask.astype(int)
    else:
        log.warning("inference.market_missing_using_fallback")
        stacked["market_proba"] = 0.5
        stacked["market_proba_was_missing"] = 1

    return stacked


def infer_calibrated_win_probs(
    race_features: pd.DataFrame,
    artifacts: InferenceArtifacts,
    race_id: str,
) -> np.ndarray:
    """Calibrated win-prob vector summing to 1.0 for a SINGLE race.

    `race_features` must already be the `build_inference_features` output
    sliced to one race (caller groups by race_id). The race_id argument is
    used purely to pass through to the calibrator's per-race softmax.
    """
    if race_features.empty:
        return np.array([], dtype=float)

    stacked = _stack_for_meta(race_features, artifacts)
    raw = artifacts.meta_learner.predict_proba(stacked)
    race_ids = np.full(len(race_features), race_id, dtype=object)
    calibrated = artifacts.meta_calibrator.predict_softmax(raw, race_ids=race_ids)

    # predict_softmax already normalises per race; defensive renormalisation
    # against floating-point drift.
    s = float(np.sum(calibrated))
    if s > 0 and np.isfinite(s):
        calibrated = np.asarray(calibrated, dtype=float) / s
    return calibrated


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def analyze_card(
    card: RaceCard,
    artifacts: InferenceArtifacts,
    *,
    bankroll: float = DEFAULT_BANKROLL,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_decimal_odds: float = DEFAULT_MAX_DECIMAL_ODDS,
    bet_types: tuple[BetType, ...] = DEFAULT_BET_TYPES,
    optimize: bool = True,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = DEFAULT_SEED,
    card_id: Optional[str] = None,
    use_interim: bool = False,
) -> tuple[dict[str, np.ndarray], list[BetCandidate], list[Portfolio]]:
    """Full inference pipeline on a parsed RaceCard.

    Returns:
      race_win_probs:  dict[race_id, calibrated win-prob vector]
      candidates:      list[BetCandidate] across all races (sorted by EV desc)
      portfolios:      list[Portfolio], one per race when `optimize=True`,
                       else empty.

    Per ADR-045: when `optimize=True` (the default), per-race candidates
    are fed to the Phase 5b CVaR LP via `optimize_portfolio` — the
    Rockafellar-Uryasev formulation with Plackett-Luce Gumbel scenarios.
    Set `use_interim=True` to use the Stream A interim Monte-Carlo
    Kelly-scaling constructor instead (kept for backward-compat with
    earlier tests / callers; not recommended for production).

    Exotic bets (EXACTA / TRIFECTA / SUPERFECTA) are emitted only when the
    caller-supplied `exotic_odds` is non-empty. In live mode no exotic odds
    exist, so the calculator is invoked with only the WIN bet type — the
    exotic types are silently skipped. This matches the Phase 5b validation
    script's behaviour (ADR-040).
    """
    feats = build_inference_features(card)
    if feats.empty:
        return {}, [], []

    race_win_probs: dict[str, np.ndarray] = {}
    all_candidates: list[BetCandidate] = []
    portfolios: list[Portfolio] = []

    for race in card.races:
        race_id = _live_race_id(race, card.track_code or "")
        race_feats = feats[feats["race_id"] == race_id]
        if race_feats.empty:
            continue

        # Pre-sort by post_position so the calibrated win-prob vector and
        # the decimal-odds vector below share the same indexing — the EV
        # calculator interprets `i` as the in-race horse index.
        ordered = race_feats.sort_values("post_position").reset_index(drop=True)

        try:
            win_probs_ordered = infer_calibrated_win_probs(
                ordered, artifacts, race_id
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "inference.race_score_failed",
                race_id=race_id,
                error=str(exc),
            )
            continue

        if win_probs_ordered.size == 0:
            continue

        # Defensive: renormalise (calibrator's softmax should already
        # produce sum=1 per race, but FP drift can leave it slightly off).
        s = float(np.sum(win_probs_ordered))
        if s > 0:
            win_probs_ordered = win_probs_ordered / s

        race_win_probs[race_id] = win_probs_ordered

        # Decimal odds vector (live: morning line, capped).
        odds = ordered["odds_final"].to_numpy(dtype=float)
        cap_mask = odds > max_decimal_odds
        if cap_mask.any():
            odds = np.where(cap_mask, max_decimal_odds, odds)
        if np.isnan(odds).any() or (odds < 1.0).any():
            log.warning(
                "inference.race_missing_odds_skipped_ev",
                race_id=race_id,
                n=len(odds),
            )
            continue

        # Live mode: only WIN is feasible without exotic odds.
        live_bet_types = [bt for bt in bet_types if bt == BetType.WIN]
        if not live_bet_types:
            continue

        try:
            cands = compute_ev_candidates(
                race_id=race_id,
                win_probs=win_probs_ordered,
                decimal_odds=odds,
                bet_types=live_bet_types,
                min_edge=min_edge,
                bankroll=bankroll,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "inference.ev_compute_failed",
                race_id=race_id,
                error=str(exc),
            )
            continue

        all_candidates.extend(cands)

        if optimize and cands:
            if use_interim:
                portfolios.append(
                    build_portfolio_from_candidates(
                        card_id=card_id or "live",
                        candidates=cands,
                        bankroll=bankroll,
                        cvar_alpha=cvar_alpha,
                        max_drawdown_pct=max_drawdown_pct,
                        n_scenarios=n_scenarios,
                        seed=seed,
                    )
                )
            else:
                # ADR-045: per-race CVaR LP via Rockafellar-Uryasev.
                # `optimize_portfolio` is per-race; we feed it the single
                # race's candidates + that race's win-prob vector.
                portfolios.append(
                    optimize_portfolio(
                        candidates=cands,
                        race_win_probs={race_id: win_probs_ordered},
                        bankroll=bankroll,
                        cvar_alpha=cvar_alpha,
                        max_drawdown_pct=max_drawdown_pct,
                        n_scenarios=n_scenarios,
                        seed=seed,
                        card_id=card_id or "live",
                    )
                )

    all_candidates.sort(key=lambda c: c.expected_value, reverse=True)
    log.info(
        "inference.analyze_card_complete",
        n_races=len(card.races),
        n_priced=len(race_win_probs),
        n_candidates=len(all_candidates),
        n_portfolios=len(portfolios),
    )
    return race_win_probs, all_candidates, portfolios


# ──────────────────────────────────────────────────────────────────────────────
# Pareto frontier (ADR-045)
# ──────────────────────────────────────────────────────────────────────────────


DEFAULT_RISK_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
"""Six risk levels covering conservative → aggressive (per ADR-045)."""


def analyze_card_pareto(
    card: RaceCard,
    artifacts: InferenceArtifacts,
    *,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    bankroll: float = DEFAULT_BANKROLL,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_decimal_odds: float = DEFAULT_MAX_DECIMAL_ODDS,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = DEFAULT_SEED,
    card_id: Optional[str] = None,
) -> tuple[list[tuple[float, Portfolio]], int]:
    """Compute the risk/return Pareto frontier — one Portfolio per risk level.

    Returns:
      points: list of (max_drawdown_pct, aggregated_card_portfolio) tuples,
              in the same order as `risk_levels`.
      n_candidates_total: total candidate count across all races (pre-LP).

    Implementation strategy (ADR-045): run candidate generation ONCE
    (the expensive sub-model + meta-learner + calibrator + EV-calculator
    pipeline) and re-solve only the LP at each risk level. For a 10-race
    card with ~5 candidates per race and 6 risk levels, this is one ML
    pass + 60 cheap LP solves vs. 6 full pipelines.
    """
    if not risk_levels:
        raise ValueError("risk_levels must contain at least one entry")
    for rl in risk_levels:
        if not (0.0 < rl <= 1.0):
            raise ValueError(f"each risk_level must be in (0, 1]; got {rl}")

    # Candidate generation runs once. `optimize=False` skips the LP entirely.
    race_win_probs, all_candidates, _ = analyze_card(
        card,
        artifacts,
        bankroll=bankroll,
        min_edge=min_edge,
        max_decimal_odds=max_decimal_odds,
        optimize=False,
        cvar_alpha=cvar_alpha,
        n_scenarios=n_scenarios,
        seed=seed,
        card_id=card_id,
    )

    n_candidates_total = len(all_candidates)
    points: list[tuple[float, Portfolio]] = []

    if not all_candidates:
        # No candidates → empty Portfolio at every risk level.
        for rl in risk_levels:
            points.append(
                (
                    rl,
                    Portfolio(
                        card_id=card_id or "live",
                        bankroll=bankroll,
                        recommendations=[],
                        expected_return=0.0,
                        var_95=0.0,
                        cvar_95=0.0,
                        total_stake_fraction=0.0,
                    ),
                )
            )
        return points, n_candidates_total

    # Group candidates by race once.
    by_race: dict[str, list[BetCandidate]] = {}
    for cand in all_candidates:
        by_race.setdefault(cand.race_id, []).append(cand)

    for rl in risk_levels:
        per_race: list[Portfolio] = []
        for race_id, cands in by_race.items():
            if race_id not in race_win_probs:
                continue
            per_race.append(
                optimize_portfolio(
                    candidates=cands,
                    race_win_probs={race_id: race_win_probs[race_id]},
                    bankroll=bankroll,
                    cvar_alpha=cvar_alpha,
                    max_drawdown_pct=rl,
                    n_scenarios=n_scenarios,
                    seed=seed,
                    card_id=card_id or "live",
                )
            )
        aggregated = _aggregate_per_race_portfolios(
            card_id or "live", per_race, bankroll
        )
        points.append((rl, aggregated))

    log.info(
        "inference.pareto_complete",
        n_risk_levels=len(risk_levels),
        n_candidates_total=n_candidates_total,
        n_races=len(by_race),
    )
    return points, n_candidates_total


def _aggregate_per_race_portfolios(
    card_id: str, portfolios: list[Portfolio], bankroll: float
) -> Portfolio:
    """Card-level aggregation (per ADR-042): concat recs, sum return/stake,
    worst-case (max) VaR/CVaR."""
    if not portfolios:
        return Portfolio(
            card_id=card_id,
            bankroll=bankroll,
            recommendations=[],
            expected_return=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_stake_fraction=0.0,
        )
    recs: list[BetRecommendation] = []
    for p in portfolios:
        recs.extend(p.recommendations)
    total_stake_fraction = min(1.0, sum(p.total_stake_fraction for p in portfolios))
    return Portfolio(
        card_id=card_id,
        bankroll=bankroll,
        recommendations=recs,
        expected_return=float(sum(p.expected_return for p in portfolios)),
        var_95=float(max(p.var_95 for p in portfolios)),
        cvar_95=float(max(p.cvar_95 for p in portfolios)),
        total_stake_fraction=float(total_stake_fraction),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ──────────────────────────────────────────────────────────────────────────────
#
# The Phase 5b R-U LP (`app.services.portfolio.optimizer.optimize_portfolio`)
# is the default portfolio constructor since ADR-045. The interim
# Monte-Carlo Kelly-scaling constructor below (`build_portfolio_from_candidates`)
# is preserved as a fallback for backward-compat with older callers and
# tests — it is reachable through `analyze_card(..., use_interim=True)`.


def build_portfolio_from_candidates(
    card_id: str,
    candidates: list[BetCandidate],
    bankroll: float,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = DEFAULT_SEED,
) -> Portfolio:
    """Interim portfolio constructor (pre-ADR-045 fallback).

    Phase 5b's CVaR LP (`optimize_portfolio`) is the default since
    ADR-045. This function is preserved for backward-compat — callers
    can opt back into it via `analyze_card(..., use_interim=True)`.
    It constructs a portfolio by:
      * Using each candidate's pre-computed 1/4 Kelly fraction directly
        (already capped at 3% per bet via ADR-002).
      * Scaling all stakes down by a single factor `k ∈ (0, 1]` until the
        Monte-Carlo CVaR_α loss ≤ max_drawdown_pct × bankroll.
      * If CVaR is already within budget at k=1, no scaling is applied.

    The Monte-Carlo scenario generator treats per-race WIN bets as
    perfectly correlated within their race (only one horse can win) and
    independent across races. Phase 5b will replace this with the
    Rockafellar-Uryasev LP. ADR-042 documents the interim choice.

    Returns a Portfolio object that matches the `app.schemas.bets.Portfolio`
    contract field-for-field.
    """
    rng = np.random.default_rng(seed)

    if not candidates:
        return Portfolio(
            card_id=card_id,
            bankroll=bankroll,
            recommendations=[],
            expected_return=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_stake_fraction=0.0,
        )

    # ── Group candidates by race so we can sample per-race outcomes ────────
    by_race: dict[str, list[BetCandidate]] = {}
    for c in candidates:
        by_race.setdefault(c.race_id, []).append(c)

    # Initial stake fractions = pre-capped Kelly fractions.
    base_stake_fracs = np.array([c.kelly_fraction for c in candidates], dtype=float)
    total_fraction = float(base_stake_fracs.sum())
    # Hard ceiling: total exposure can't exceed bankroll.
    if total_fraction > 1.0:
        base_stake_fracs = base_stake_fracs / total_fraction
        total_fraction = 1.0

    # ── Monte-Carlo simulate the portfolio's PnL distribution ─────────────
    pnl_per_dollar = _simulate_pnl(
        candidates=candidates,
        stake_fracs=base_stake_fracs,
        by_race=by_race,
        bankroll=bankroll,
        n_scenarios=n_scenarios,
        rng=rng,
    )
    var_95 = float(-np.quantile(pnl_per_dollar, cvar_alpha))
    losses = -pnl_per_dollar
    cvar_threshold = np.quantile(losses, 1.0 - cvar_alpha)
    cvar_95 = float(np.mean(losses[losses >= cvar_threshold])) if (losses >= cvar_threshold).any() else float(cvar_threshold)
    expected_return = float(np.mean(pnl_per_dollar))

    # Scale stakes if CVaR exceeds budget.
    max_loss_budget = max_drawdown_pct * bankroll
    if cvar_95 > max_loss_budget and cvar_95 > 0:
        k = max_loss_budget / cvar_95
        k = max(0.0, min(1.0, k))
        stake_fracs = base_stake_fracs * k
        # Re-simulate to get post-scale risk metrics.
        pnl_per_dollar = _simulate_pnl(
            candidates=candidates,
            stake_fracs=stake_fracs,
            by_race=by_race,
            bankroll=bankroll,
            n_scenarios=n_scenarios,
            rng=rng,
        )
        var_95 = float(-np.quantile(pnl_per_dollar, cvar_alpha))
        losses = -pnl_per_dollar
        cvar_threshold = np.quantile(losses, 1.0 - cvar_alpha)
        cvar_95 = float(np.mean(losses[losses >= cvar_threshold])) if (losses >= cvar_threshold).any() else float(cvar_threshold)
        expected_return = float(np.mean(pnl_per_dollar))
    else:
        stake_fracs = base_stake_fracs

    recommendations: list[BetRecommendation] = []
    for c, sf in zip(candidates, stake_fracs):
        sf_capped = float(apply_bet_cap(float(sf)))
        recommendations.append(
            BetRecommendation(
                candidate=c,
                stake=sf_capped * bankroll,
                stake_fraction=sf_capped,
            )
        )

    total_stake_fraction = float(sum(r.stake_fraction for r in recommendations))
    # Pydantic schema constrains total_stake_fraction ≤ 1.0; clamp defensively.
    total_stake_fraction = min(total_stake_fraction, 1.0)

    return Portfolio(
        card_id=card_id,
        bankroll=bankroll,
        recommendations=recommendations,
        expected_return=expected_return,
        var_95=var_95,
        cvar_95=cvar_95,
        total_stake_fraction=total_stake_fraction,
    )


def _simulate_pnl(
    candidates: list[BetCandidate],
    stake_fracs: np.ndarray,
    by_race: dict[str, list[BetCandidate]],
    bankroll: float,
    n_scenarios: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte Carlo PnL (per $) for the portfolio of WIN candidates.

    Per-race outcome ≡ which horse wins. We draw the winning horse index
    for each race once per scenario. A WIN candidate on (race, horse_idx)
    pays `stake × (decimal_odds − 1)` if its horse_idx equals the winner,
    else loses `stake`. Phase 5a candidates today are WIN only — exotic
    PnL drawing requires per-permutation odds which the live path doesn't
    have. Defensive: any non-WIN candidate's pnl draw uses Bernoulli(p)
    against its model_prob as a coarse fallback.
    """
    cand_idx_by_race: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        cand_idx_by_race.setdefault(c.race_id, []).append(i)

    pnl = np.zeros(n_scenarios, dtype=float)
    for race_id, cand_indices in cand_idx_by_race.items():
        race_cands = [candidates[i] for i in cand_indices]
        win_cands = [c for c in race_cands if c.bet_type == BetType.WIN]
        other_cands = [c for c in race_cands if c.bet_type != BetType.WIN]

        if win_cands:
            # Build win-prob vector spanning the universe of horses we hold
            # candidates on plus an implicit "other horse wins" bucket.
            held_horses = [c.selection[0] for c in win_cands]
            held_probs = np.array([c.model_prob for c in win_cands], dtype=float)
            p_other = max(0.0, 1.0 - float(np.sum(held_probs)))
            probs = np.concatenate([held_probs, [p_other]])
            probs = probs / probs.sum()
            outcomes = rng.choice(
                len(probs), size=n_scenarios, p=probs
            )  # n_scenarios picks from [0..len(probs)-1]

            for k, c in enumerate(win_cands):
                i_global = cand_indices[race_cands.index(c)]
                sf = float(stake_fracs[i_global])
                stake = sf * bankroll
                hits = outcomes == k
                pnl += np.where(
                    hits,
                    stake * (c.decimal_odds - 1.0),
                    -stake,
                )

        for c in other_cands:
            i_global = cand_indices[race_cands.index(c)]
            sf = float(stake_fracs[i_global])
            stake = sf * bankroll
            hits = rng.random(n_scenarios) < c.model_prob
            pnl += np.where(
                hits,
                stake * (c.decimal_odds - 1.0),
                -stake,
            )

    if bankroll > 0:
        return pnl / bankroll
    return pnl


__all__ = [
    "DEFAULT_BANKROLL",
    "DEFAULT_BET_TYPES",
    "DEFAULT_CVAR_ALPHA",
    "DEFAULT_MAX_DECIMAL_ODDS",
    "DEFAULT_MAX_DRAWDOWN_PCT",
    "DEFAULT_MIN_EDGE",
    "DEFAULT_N_SCENARIOS",
    "DEFAULT_RISK_LEVELS",
    "DEFAULT_SEED",
    "InferenceArtifacts",
    "analyze_card",
    "analyze_card_pareto",
    "build_inference_features",
    "build_portfolio_from_candidates",
    "infer_calibrated_win_probs",
    "race_card_to_features",
]
