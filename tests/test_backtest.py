"""Walk-forward engine arithmetic.

The engine is a small amount of code with a large amount of leverage over every
number the project reports, so its arithmetic is pinned against hand-computed
values rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etflab.backtest import BacktestResult, run_backtest, run_zoo
from etflab.config import ExperimentConfig
from etflab.costs import BpsCostModel
from etflab.data.panel import PricePanel
from etflab.strategies import Fit, FitContext, build_strategy


class FixedWeights:
    """A strategy that always answers the same thing, so the engine is the only variable."""

    name = "fixed"
    label = "Fixed"

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=float)

    def fit(self, ctx: FitContext) -> Fit:
        return Fit(self.weights.copy(), "analytic")


def _panel_from_returns(returns: pd.DataFrame, assets: tuple[str, ...], target: str) -> PricePanel:
    """Build a panel whose pct_change is exactly ``returns``."""
    first = returns.index[0] - pd.tseries.offsets.BDay(1)
    prices = pd.concat(
        [
            pd.DataFrame([[100.0] * returns.shape[1]], index=[first], columns=returns.columns),
            (1.0 + returns).cumprod() * 100.0,
        ]
    )
    prices.index.name = "date"
    return PricePanel(prices[[*assets, target]], assets, target, "test")


@pytest.fixture
def toy_panel() -> PricePanel:
    dates = pd.bdate_range("2020-01-01", periods=12)
    returns = pd.DataFrame(
        {
            "AAA": [0.01, -0.01, 0.02, 0.00, 0.01, 0.02, -0.01, 0.01, 0.00, 0.01, -0.02, 0.01],
            "BBB": [0.00, 0.01, 0.01, -0.01, 0.02, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, 0.00],
            "TGT": [0.01, 0.00, 0.02, -0.01, 0.01, 0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.01],
        },
        index=dates,
    )
    return _panel_from_returns(returns, ("AAA", "BBB"), "TGT")


@pytest.fixture
def toy_config() -> ExperimentConfig:
    return ExperimentConfig(
        assets=("AAA", "BBB"),
        target="TGT",
        train_days=60,
        rebalance_days=4,
        embargo_days=0,
        max_weight=0.7,
        max_turnover=1.0,
    )


def _run_toy(panel: PricePanel, config: ExperimentConfig, cost_bps: float = 0.0) -> BacktestResult:
    # ``train_days`` is validated at >= 60, so the toy case sets it after validation.
    object.__setattr__(config, "train_days", 4)
    return run_backtest(panel, FixedWeights([0.5, 0.5]), config, BpsCostModel(cost_bps))


def test_first_rebalance_charges_the_whole_book(toy_panel, toy_config):
    """Deploying out of cash buys 100% of the book and pays a spread on all of it.

    The conventional ``sum |dw| / 2`` turnover figure reads 0.5 here because the
    trade is one-sided; charging costs on that figure would halve the cost of
    every initial deployment, which is why costs are charged on traded notional.
    """
    result = _run_toy(toy_panel, toy_config, cost_bps=10.0)
    first = result.rebalances.iloc[0]
    assert first["traded_notional"] == pytest.approx(1.0)
    assert first["one_way_turnover"] == pytest.approx(0.5)
    assert first["cost_total_bps"] == pytest.approx(10.0)
    assert result.daily["cost"].iloc[0] == pytest.approx(10.0 / 1e4)
    assert result.daily["cost"].iloc[1] == 0.0, "costs are charged once, on the trade date"


def test_both_cost_models_agree_on_a_pure_spread_trade():
    """A regression guard for the bug that started this test: the flat model once
    charged half what the spread model charged for the identical trade."""
    from etflab.costs import BpsCostModel, LiquidityProfile, SpreadImpactCostModel

    trades = pd.Series({"SPY": 0.10, "QQQ": -0.10}, dtype=float)
    volatility = pd.Series({"SPY": 0.0, "QQQ": 0.0}, dtype=float)  # no impact, spread only
    liquidity = LiquidityProfile(
        half_spread_bps=pd.Series({"SPY": 3.0, "QQQ": 3.0}),
        adv_usd=pd.Series({"SPY": 1e12, "QQQ": 1e12}),
    )
    flat = BpsCostModel(3.0).cost(trades, volatility)
    detailed = SpreadImpactCostModel(liquidity, 1e6, 0.7).cost(trades, volatility)
    assert flat.total == pytest.approx(detailed.total, rel=1e-12)
    assert flat.traded_notional == pytest.approx(0.20)


def test_gross_return_is_the_weighted_sum_on_the_rebalance_day(toy_panel, toy_config):
    result = _run_toy(toy_panel, toy_config)
    returns = toy_panel.returns
    first_date = result.daily.index[0]
    expected = 0.5 * returns.loc[first_date, "AAA"] + 0.5 * returns.loc[first_date, "BBB"]
    assert result.daily.loc[first_date, "gross"] == pytest.approx(expected, abs=1e-15)


def test_daily_weights_are_the_weights_that_earned_the_days_return(toy_panel, toy_config):
    """``daily_weights`` is start-of-day, so it reproduces ``gross`` exactly.

    The convention matters: end-of-day weights would silently break the risk
    attribution, which multiplies these weights by the same day's returns.
    """
    result = _run_toy(toy_panel, toy_config)
    np.testing.assert_allclose(result.daily_weights.sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-12)
    asset_returns = toy_panel.asset_returns.loc[result.daily_weights.index, result.daily_weights.columns]
    reconstructed = (result.daily_weights * asset_returns).sum(axis=1)
    np.testing.assert_allclose(reconstructed.to_numpy(), result.daily["gross"].to_numpy(), rtol=0, atol=1e-15)


def test_weights_drift_with_prices_between_rebalances(toy_panel, toy_config):
    result = _run_toy(toy_panel, toy_config)
    returns = toy_panel.returns
    first_date, second_date = result.daily.index[0], result.daily.index[1]
    row = returns.loc[first_date]
    gross = 0.5 * row["AAA"] + 0.5 * row["BBB"]
    expected_aaa = 0.5 * (1 + row["AAA"]) / (1 + gross)
    assert result.daily_weights.loc[second_date, "AAA"] == pytest.approx(expected_aaa, abs=1e-14)


def test_turnover_is_measured_against_drifted_holdings(toy_panel, toy_config):
    """Measuring against the previous *target* would understate what must be traded."""
    result = _run_toy(toy_panel, toy_config)
    second = result.rebalances.index[1]
    position = result.daily_weights.index.get_loc(second)
    previous_day = result.daily_weights.index[position - 1]
    # Drift the last held weights through that day's returns to get pre-trade holdings.
    start_of_day = result.daily_weights.loc[previous_day].to_numpy()
    day_returns = toy_panel.asset_returns.loc[previous_day, result.daily_weights.columns].to_numpy()
    pre_trade = start_of_day * (1 + day_returns) / (1 + float(start_of_day @ day_returns))
    expected = float(np.abs(np.array([0.5, 0.5]) - pre_trade).sum())
    assert result.rebalances.loc[second, "traded_notional"] == pytest.approx(expected, abs=1e-12)


def test_every_out_of_sample_day_appears_exactly_once(toy_panel, toy_config):
    result = _run_toy(toy_panel, toy_config)
    returns = toy_panel.returns
    expected_index = returns.index[4:]
    pd.testing.assert_index_equal(result.daily.index, expected_index)
    assert not result.daily.index.has_duplicates


def test_final_partial_period_is_included(toy_panel, toy_config):
    """12 observations, 4 training, 4-day steps: the last period is short and must still be evaluated."""
    object.__setattr__(toy_config, "rebalance_days", 5)
    result = _run_toy(toy_panel, toy_config)
    assert result.daily.index[-1] == toy_panel.returns.index[-1]


def test_active_return_identity_holds_everywhere(toy_panel, toy_config):
    result = _run_toy(toy_panel, toy_config, cost_bps=25.0)
    np.testing.assert_allclose(
        result.daily["active"].to_numpy(),
        (result.daily["net"] - result.daily["target"]).to_numpy(),
        rtol=0,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        result.daily["net"].to_numpy(),
        (result.daily["gross"] - result.daily["cost"]).to_numpy(),
        rtol=0,
        atol=1e-18,
    )


def test_zero_cost_model_produces_zero_drag(toy_panel, toy_config):
    result = _run_toy(toy_panel, toy_config, cost_bps=0.0)
    assert result.daily["cost"].sum() == 0.0
    np.testing.assert_array_equal(result.daily["net"].to_numpy(), result.daily["gross"].to_numpy())


def test_engine_refuses_a_history_shorter_than_the_training_window(toy_panel, toy_config):
    object.__setattr__(toy_config, "train_days", 500)
    with pytest.raises(ValueError, match="more than 500"):
        run_backtest(toy_panel, FixedWeights([0.5, 0.5]), toy_config)


def test_zoo_shares_one_cost_model_and_one_evaluation_window(small_panel, small_config):
    results = run_zoo(small_panel, small_config, ("tracking", "equal_weight", "static"))
    indices = [r.daily.index for r in results.values()]
    for other in indices[1:]:
        pd.testing.assert_index_equal(indices[0], other)
    assert {r.meta["cost_model"] for r in results.values()} == {small_config.cost_model}


def test_solver_failure_falls_back_to_prior_holdings_rather_than_crashing(small_panel, small_config):
    """An unsolvable window is an operational event, not the end of the run."""

    class Exploding:
        name, label = "exploding", "Exploding"

        def __init__(self) -> None:
            self.calls = 0

        def fit(self, ctx: FitContext) -> Fit:
            self.calls += 1
            if self.calls == 2:
                from etflab.strategies import _fallback_weights

                return Fit(_fallback_weights(ctx), "fallback_infeasible")
            return build_strategy("tracking", small_config).fit(ctx)

    strategy = Exploding()
    result = run_backtest(small_panel, strategy, small_config)
    assert result.degraded_rebalances == 1
    assert len(result.daily) > 0, "the run continued past the failure"
    np.testing.assert_allclose(result.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_rolling_window_uses_a_bounded_history(small_panel, small_config):
    rolling = small_config.with_changes(train_mode="rolling")
    result = run_backtest(small_panel, build_strategy("tracking", rolling), rolling)
    assert result.rebalances["train_obs"].max() <= rolling.train_days
    expanding = run_backtest(small_panel, build_strategy("tracking", small_config), small_config)
    assert expanding.rebalances["train_obs"].iloc[-1] > rolling.train_days
