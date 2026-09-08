"""Cost models: the economic properties, not just the arithmetic."""

from __future__ import annotations

import pandas as pd
import pytest

from etflab.config import ExperimentConfig
from etflab.costs import (
    DEFAULT_LIQUIDITY,
    UNKNOWN_LIQUIDITY,
    BpsCostModel,
    LiquidityProfile,
    SpreadImpactCostModel,
    build_cost_model,
)

TRADES = pd.Series({"SPY": 0.10, "QQQ": -0.06, "GLD": -0.04}, dtype=float)
VOL = pd.Series({"SPY": 0.010, "QQQ": 0.012, "GLD": 0.009}, dtype=float)


def _impact(notional: float, coef: float = 0.7) -> SpreadImpactCostModel:
    return SpreadImpactCostModel(LiquidityProfile.for_assets(tuple(TRADES.index)), notional, coef)


def test_flat_model_is_linear_in_traded_notional():
    single = BpsCostModel(5.0).cost(TRADES, VOL)
    doubled = BpsCostModel(5.0).cost(TRADES * 2, VOL)
    assert doubled.total == pytest.approx(2 * single.total)
    assert single.traded_notional == pytest.approx(float(TRADES.abs().sum()))


def test_flat_model_ignores_book_size_and_says_so_by_construction():
    """The documented weakness of a basis-point model, pinned as a test so the
    capacity analysis has something to contrast against."""
    assert BpsCostModel(5.0).cost(TRADES, VOL).max_participation == 0.0


def test_impact_cost_grows_with_book_size():
    totals = [_impact(n).cost(TRADES, VOL).total for n in (1e6, 1e7, 1e8, 1e9)]
    assert totals == sorted(totals)
    assert totals[-1] > totals[0]


def test_impact_is_concave_in_size():
    """A square-root law means a tenfold larger trade costs about sqrt(10) more
    per unit, not ten times more. Linear impact would fail this."""
    small = _impact(1e7).cost(TRADES, VOL)
    large = _impact(1e9).cost(TRADES, VOL)
    ratio = (large.impact / large.traded_notional) / (small.impact / small.traded_notional)
    assert 8.0 < ratio < 12.0, f"per-unit impact grew {ratio:.1f}x for a 100x book; sqrt would give 10x"


def test_spread_component_is_invariant_to_book_size():
    assert _impact(1e6).cost(TRADES, VOL).spread == pytest.approx(_impact(1e10).cost(TRADES, VOL).spread)


def test_impact_scales_with_volatility():
    quiet = _impact(1e9).cost(TRADES, VOL)
    stormy = _impact(1e9).cost(TRADES, VOL * 3)
    assert stormy.impact == pytest.approx(3 * quiet.impact, rel=1e-9)


def test_participation_is_reported_and_scales_with_size():
    assert _impact(1e9).cost(TRADES, VOL).max_participation > _impact(1e6).cost(TRADES, VOL).max_participation


def test_no_trade_costs_nothing():
    empty = pd.Series(0.0, index=TRADES.index)
    assert BpsCostModel(50.0).cost(empty, VOL).total == 0.0
    assert _impact(1e9).cost(empty, VOL).total == 0.0


def test_unknown_tickers_get_a_pessimistic_liquidity_assumption():
    profile = LiquidityProfile.for_assets(("SPY", "ZZZZ"))
    assert profile.half_spread_bps["ZZZZ"] == UNKNOWN_LIQUIDITY[0]
    assert profile.adv_usd["ZZZZ"] == UNKNOWN_LIQUIDITY[1]
    assert profile.half_spread_bps["ZZZZ"] > profile.half_spread_bps["SPY"]


def test_liquidity_overrides_are_respected():
    profile = LiquidityProfile.for_assets(("SPY",), overrides={"SPY": (9.9, 1234.0)})
    assert profile.half_spread_bps["SPY"] == 9.9
    assert profile.adv_usd["SPY"] == 1234.0


def test_cost_components_sum_to_the_total():
    cost = _impact(5e8).cost(TRADES, VOL)
    assert cost.total == pytest.approx(cost.spread + cost.impact)


def test_builder_dispatches_on_the_configured_model():
    config = ExperimentConfig()
    assert isinstance(build_cost_model(config, ("SPY", "QQQ")), BpsCostModel)
    assert isinstance(
        build_cost_model(config.with_changes(cost_model="spread_impact"), ("SPY", "QQQ")), SpreadImpactCostModel
    )


def test_the_liquidity_table_is_ordered_the_way_reality_is():
    """A sanity check on the assumptions themselves: the mega-cap ETF must not be
    assumed thinner than the niche one."""
    spy_spread, spy_adv = DEFAULT_LIQUIDITY["SPY"]
    psp_spread, psp_adv = DEFAULT_LIQUIDITY["PSP"]
    assert spy_spread < psp_spread
    assert spy_adv > psp_adv
