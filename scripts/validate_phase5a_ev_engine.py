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
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.core.logging import configure_logging, get_logger
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
    # Phase 5b additions: data-quality filter accounting + optimizer rollup.
    max_decimal_odds: float = 100.0
    n_rows_dropped_by_odds_cap: int = 0
    optimize_enabled: bool = False
    n_portfolios_with_recommendations: int = 0
    n_empty_portfolios: int = 0
    total_stake_fraction_sum: float = 0.0
    mean_cvar_usd: float = 0.0
    mean_expected_return_usd: float = 0.0
    mean_n_recommendations_per_active_portfolio: float = 0.0


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
    max_decimal_odds: float = 100.0,
    optimize: bool = False,
    bankroll: float = 1.0,
    cvar_alpha: float = 0.05,
    max_drawdown_pct: float = 0.20,
    n_scenarios: int = 1000,
    seed: int = 42,
) -> tuple[list[BetCandidate], EVRunSummary, list]:
    """Run the EV engine race-by-race over the test slice.

    Phase 5b additions:
        max_decimal_odds: drop rows with odds_final > cap BEFORE feeding to
            compute_ev_candidates. The parquet's odds_final column contains
            99/999 placeholders for scratches that produce absurd edges
            (see ADR-040 'Note on smoke-run data quality').
        optimize:  when True, runs the CVaR portfolio optimiser per race
            and returns a list of Portfolio objects alongside the flat
            BetCandidate list. When False, the third return is an empty list.
    """
    if optimize:
        # Lazy import: keeps this script importable for non-optimize callers
        # without forcing the scipy.optimize.linprog import path.
        from app.services.portfolio.optimizer import optimize_portfolio

    candidates: list[BetCandidate] = []
    portfolios: list = []
    n_races = 0
    races_with_cands = 0
    n_rows_dropped_by_odds_cap = 0

    groups = test_df.groupby(race_id_col, sort=False)
    if limit_races is not None:
        groups = list(groups)[:limit_races]

    for race_id, group in groups:
        win_probs = group["meta_calibrated"].to_numpy(dtype=float)
        s = win_probs.sum()
        if not np.isfinite(s) or s <= 0:
            continue

        odds = _odds_for_mode(group, mode)
        # Apply max-decimal-odds data-quality filter. Rows with odds above
        # the cap are treated as missing (placeholder values like 99/999).
        cap_mask = odds > max_decimal_odds
        n_dropped = int(np.sum(cap_mask & np.isfinite(odds)))
        n_rows_dropped_by_odds_cap += n_dropped
        odds = np.where(cap_mask, np.nan, odds)
        if np.isnan(odds).any():
            continue

        win_probs = win_probs / s  # renormalise

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

        if optimize and race_cands:
            portfolio = optimize_portfolio(
                candidates=race_cands,
                race_win_probs={str(race_id): win_probs},
                bankroll=bankroll,
                cvar_alpha=cvar_alpha,
                max_drawdown_pct=max_drawdown_pct,
                n_scenarios=n_scenarios,
                seed=seed,
                card_id=str(race_id),
            )
            portfolios.append(portfolio)

    if n_rows_dropped_by_odds_cap:
        log.info(
            "ev_engine.validate.odds_cap.applied",
            cap=max_decimal_odds,
            n_rows_dropped=n_rows_dropped_by_odds_cap,
        )

    by_type: dict[str, int] = {}
    for c in candidates:
        by_type[c.bet_type.value] = by_type.get(c.bet_type.value, 0) + 1

    n_active = sum(1 for p in portfolios if p.recommendations)
    n_empty = sum(1 for p in portfolios if not p.recommendations)
    if n_active:
        mean_cvar_usd = float(np.mean([p.cvar_95 for p in portfolios if p.recommendations]))
        mean_expected_return_usd = float(
            np.mean([p.expected_return for p in portfolios if p.recommendations])
        )
        mean_n_recs = float(
            np.mean([len(p.recommendations) for p in portfolios if p.recommendations])
        )
    else:
        mean_cvar_usd = 0.0
        mean_expected_return_usd = 0.0
        mean_n_recs = 0.0
    total_stake_fraction_sum = float(sum(p.total_stake_fraction for p in portfolios))

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
        max_decimal_odds=float(max_decimal_odds),
        n_rows_dropped_by_odds_cap=n_rows_dropped_by_odds_cap,
        optimize_enabled=optimize,
        n_portfolios_with_recommendations=n_active,
        n_empty_portfolios=n_empty,
        total_stake_fraction_sum=total_stake_fraction_sum,
        mean_cvar_usd=mean_cvar_usd,
        mean_expected_return_usd=mean_expected_return_usd,
        mean_n_recommendations_per_active_portfolio=mean_n_recs,
    )
    return candidates, summary, portfolios


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    ap.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    ap.add_argument("--run-id", type=str, default=datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--limit-races", type=int, default=None)
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Sub-sample the parquet to this fraction (0, 1) BEFORE feature "
             "prep. Use for smoke testing — full parquet feature prep takes "
             "~15 min on 2.3M rows; a 5%% sample completes in ~1 min.",
    )
    ap.add_argument(
        "--bet-types",
        type=str,
        default="win",
        help="Comma-separated subset of win,exacta,trifecta,superfecta. "
             "Note: in backtest mode only WIN is supported (no historical "
             "per-permutation exotic odds in the parquet).",
    )
    ap.add_argument(
        "--max-decimal-odds",
        type=float,
        default=100.0,
        help="Drop rows with odds_final > cap BEFORE feeding to the EV "
             "calculator. Defaults to 100. The parquet's odds_final column "
             "contains 99/999 placeholders for scratched horses; without "
             "this cap, 1/odds produces near-zero market probs and absurd "
             "edges (see ADR-040). Set to float('inf') to disable.",
    )
    ap.add_argument(
        "--optimize",
        action="store_true",
        help="Run the Phase 5b CVaR-constrained portfolio optimiser "
             "per race after candidate generation. Aggregates per-portfolio "
             "stats into the output report. Pure additive — when not set, "
             "behaviour is identical to Phase 5a.",
    )
    ap.add_argument(
        "--bankroll",
        type=float,
        default=1.0,
        help="Bankroll used by the portfolio optimiser (per race). "
             "Defaults to 1.0 so all reported metrics are in 'fraction of "
             "bankroll' terms.",
    )
    ap.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.05,
        help="CVaR tail probability. Defaults to 0.05 (95th-percentile "
             "shortfall).",
    )
    ap.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=0.20,
        help="CVaR upper bound as a fraction of bankroll. Defaults to 0.20.",
    )
    ap.add_argument(
        "--n-scenarios",
        type=int,
        default=1000,
        help="Number of Monte Carlo PL Gumbel scenarios per race for the "
             "CVaR LP. Defaults to 1000.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the PL Monte Carlo sampler. Defaults to 42.",
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
    if args.sample_frac and 0.0 < args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=11).reset_index(drop=True)
        log.warning("ev_engine.validate.subsampled", frac=args.sample_frac, rows=len(df))
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

    candidates, summary, portfolios = evaluate(
        test_df=test_scored,
        mode=args.mode,
        min_edge=args.min_edge,
        bet_types=bet_types,
        limit_races=args.limit_races,
        max_decimal_odds=args.max_decimal_odds,
        optimize=args.optimize,
        bankroll=args.bankroll,
        cvar_alpha=args.cvar_alpha,
        max_drawdown_pct=args.max_drawdown_pct,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
    )
    summary.run_id = args.run_id

    out_dir = _OUT_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dict = {
        "summary": asdict(summary),
        "candidates": [c.model_dump() for c in candidates[:1000]],
    }
    if args.optimize:
        # Persist a sample of portfolios for forensic inspection. Active
        # (non-empty) portfolios first so a small sample is informative.
        active = [p for p in portfolios if p.recommendations]
        empty = [p for p in portfolios if not p.recommendations]
        sample = active[:200] + empty[: max(0, 200 - len(active))]
        report_dict["portfolios"] = [p.model_dump() for p in sample]
    with open(out_dir / "report.json", "w") as fh:
        json.dump(report_dict, fh, indent=2, default=str)
    log.info(
        "ev_engine.validate.done",
        out_dir=str(out_dir),
        n_candidates=summary.n_candidates,
        sum_ev=summary.sum_expected_value,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


if __name__ == "__main__":
    main()
