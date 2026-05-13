"""Unit tests for app/services/portfolio/sizing.py."""
from __future__ import annotations

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
    assert kelly_fraction(edge=0.05, decimal_odds=4.0) == pytest.approx(0.0)
    assert kelly_fraction(edge=0.20, decimal_odds=4.0) == pytest.approx(0.0)
    assert kelly_fraction(edge=0.25, decimal_odds=4.0) == pytest.approx(0.015625)


def test_kelly_decimal_odds_at_unity_boundary():
    """decimal_odds=1.0 means the bet returns only the stake on a win (net
    payout = 0). At edge = 0.5, full Kelly hits 0 exactly: numerator
    `0.5*1 - (1-0.5) = 0`. The function accepts this boundary as valid input
    and returns 0."""
    assert kelly_fraction(edge=0.5, decimal_odds=1.0) == pytest.approx(0.0)


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
