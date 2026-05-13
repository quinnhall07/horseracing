"""
scripts/smoke_phase5b_optimizer.py
──────────────────────────────────
Phase 5b — Synthetic smoke harness for the CVaR-constrained optimiser.

This script exercises the same code path the validation script takes
(``evaluate(..., optimize=True)``) but on a synthetic DataFrame so it
runs without the training parquet or trained-model artifacts. Used to
verify the pipeline glues correctly when the real parquet is not
available in the worktree (which is the case for clean clones).

Outputs:
    models/baseline_full/ev_engine/<run-id>/report.json
    models/baseline_full/ev_engine/<run-id>/synthetic_inputs.json

Usage:
    python scripts/smoke_phase5b_optimizer.py --run-id phase-5b-smoke-001 \
        [--n-races 200] [--max-decimal-odds 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.core.logging import configure_logging, get_logger
from app.schemas.race import BetType
from scripts.validate_phase5a_ev_engine import _OUT_ROOT, evaluate

log = get_logger(__name__)


def synthetic_test_df(
    n_races: int,
    field_size: int,
    seed: int,
) -> pd.DataFrame:
    """Build a small synthetic version of what `validate_phase5a_ev_engine`
    feeds into `evaluate()`. Columns: race_id, meta_calibrated, odds_final.

    Per race:
      - win_probs ~ Dirichlet(alpha=2) of length `field_size`
      - odds_final ~ slightly favourable to the model so some rows clear
        the 5% min-edge bar.
      - one in ~20 rows gets a 999 placeholder odd to exercise the cap.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        # Concentrate probability on a small subset (alpha < 1) — this
        # mirrors a real race where one or two horses are clear contenders.
        win_probs = rng.dirichlet(np.full(field_size, 0.8))
        # Strong market mispricing: heavily underprice the top horse.
        # ADR-002's Kelly formula `(edge*odds - (1-edge))/odds` needs
        # `edge > 1/(odds+1)` to produce a positive stake — that's a
        # higher bar than simple p > 1/odds. We multiply decimal_odds
        # of the strongest horses by [1.5, 2.5] so edges are large.
        inflate = 1.0 + rng.uniform(0.5, 1.5, size=field_size)
        odds = (1.0 / np.maximum(win_probs, 1e-3)) * inflate
        # Plant a 999 placeholder on 5% of rows.
        placeholder_mask = rng.uniform(size=field_size) < 0.05
        odds = np.where(placeholder_mask, 999.0, odds)
        for h in range(field_size):
            rows.append(
                {
                    "race_id": f"R{r:05d}",
                    "meta_calibrated": float(win_probs[h]),
                    "odds_final": float(odds[h]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--n-races", type=int, default=200)
    ap.add_argument("--field-size", type=int, default=8)
    ap.add_argument("--max-decimal-odds", type=float, default=100.0)
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("phase5b.smoke.start", **vars(args))

    test_df = synthetic_test_df(
        n_races=args.n_races,
        field_size=args.field_size,
        seed=args.seed,
    )
    log.info("phase5b.smoke.synth_rows", rows=len(test_df))

    candidates, summary, portfolios = evaluate(
        test_df=test_df,
        mode="backtest",
        min_edge=args.min_edge,
        bet_types=[BetType.WIN],
        max_decimal_odds=args.max_decimal_odds,
        optimize=True,
        bankroll=10_000.0,
        cvar_alpha=0.05,
        max_drawdown_pct=0.20,
        n_scenarios=500,
        seed=args.seed,
    )
    summary.run_id = args.run_id

    out_dir = _OUT_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": asdict(summary),
        "candidates": [c.model_dump() for c in candidates[:200]],
        "portfolios": [p.model_dump() for p in portfolios[:50] if p.recommendations],
    }
    with open(out_dir / "report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    log.info(
        "phase5b.smoke.done",
        out_dir=str(out_dir),
        n_candidates=summary.n_candidates,
        n_portfolios_active=summary.n_portfolios_with_recommendations,
        n_rows_dropped_by_odds_cap=summary.n_rows_dropped_by_odds_cap,
        mean_cvar_usd=summary.mean_cvar_usd,
        mean_expected_return_usd=summary.mean_expected_return_usd,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


if __name__ == "__main__":
    main()
