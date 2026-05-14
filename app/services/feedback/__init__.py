"""
app/services/feedback/
──────────────────────
Phase 5b / Master Reference §4 Layer 7 — feedback loop + online learning.

Two responsibilities live here:

1. **Outcomes logging** (`outcomes.py`):
   - `log_race_outcome` — idempotent insert of an official chart / manual result.
   - `settle_bets`      — resolve `BetRecommendation`s against an outcome, emit
                          `BetSettlement` ORM rows with realised pnl.
   - `get_settled_pnl_series` — DataFrame of realised pnl ordered in time, used
                          as the input stream for portfolio drift detection.

2. **Portfolio drift** (`portfolio_drift.py`):
   - CUSUM detector analogous to `app/services/calibration/drift.py`, but
     watching realised vs. expected PnL rather than calibration residuals.
     Two-sided: under-performance flags an over-confident model; over-
     performance flags model under-confidence / upside variance bursts.

See ADR-043 for the full design rationale.
"""

from __future__ import annotations

from app.services.feedback.outcomes import (
    SETTLEABLE_BET_TYPES,
    get_settled_pnl_series,
    log_race_outcome,
    settle_bets,
)
from app.services.feedback.portfolio_drift import (
    PortfolioDriftDetector,
    PortfolioDriftState,
    portfolio_pnl_zscore,
)

__all__ = [
    "SETTLEABLE_BET_TYPES",
    "log_race_outcome",
    "settle_bets",
    "get_settled_pnl_series",
    "PortfolioDriftDetector",
    "PortfolioDriftState",
    "portfolio_pnl_zscore",
]
