"""Metrics, attribution identities, and the definitions exported alongside them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etflab.backtest import run_backtest
from etflab.metrics import (
    annualised_return,
    annualised_vol,
    attribution_by_asset,
    compute_metrics,
    cvar,
    describe_metrics,
    max_drawdown,
    recovery_metrics,
    regime_breakdown,
    rolling_tracking_error,
    up_down_capture,
)
from etflab.strategies import build_strategy


def test_annualised_return_compounds_correctly():
    daily = pd.Series([0.01] * 252)
    assert annualised_return(daily) == pytest.approx(1.01**252 - 1)


def test_annualised_return_is_nan_not_zero_on_an_empty_series():
    assert np.isnan(annualised_return(pd.Series(dtype=float)))


def test_annualised_vol_scales_by_root_252():
    series = pd.Series(np.random.default_rng(1).standard_normal(1000) * 0.01)
    assert annualised_vol(series) == pytest.approx(series.std(ddof=1) * np.sqrt(252))


def test_max_drawdown_matches_a_hand_computation():
    # 1.0 -> 1.2 -> 0.6: the trough is 50% below the 1.2 peak.
    returns = pd.Series([0.2, -0.5])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_cvar_averages_the_specified_tail():
    series = pd.Series(np.arange(-100.0, 0.0))
    assert cvar(series, 0.10) == pytest.approx(series.nsmallest(10).mean(), rel=0.15)


def test_up_down_capture_is_one_when_the_portfolio_is_the_target():
    target = pd.Series([0.01, -0.02, 0.03, -0.01])
    up, down = up_down_capture(target, target)
    assert up == pytest.approx(1.0)
    assert down == pytest.approx(1.0)


def test_rolling_tracking_error_has_the_right_window():
    active = pd.Series(np.random.default_rng(2).standard_normal(500) * 0.005)
    rolling = rolling_tracking_error(active, window=126)
    assert rolling.isna().sum() == 125
    assert rolling.dropna().iloc[0] == pytest.approx(active.iloc[:126].std(ddof=1) * np.sqrt(252))


def test_active_risk_attribution_sums_exactly_to_the_tracking_error(small_panel, small_config):
    """The identity that makes the table checkable. Omitting the target's short
    leg -- the usual mistake -- breaks it, so this test is the guard against
    that specific error returning."""
    result = run_backtest(small_panel, build_strategy("tracking", small_config), small_config)
    attribution = attribution_by_asset(result.daily_weights, small_panel.asset_returns, small_panel.target_returns)
    gross_active = result.daily["gross"] - result.daily["target"]
    assert attribution["risk_contribution"].sum() == pytest.approx(annualised_vol(gross_active), rel=1e-9)
    assert attribution["risk_share"].sum() == pytest.approx(1.0, rel=1e-9)


def test_attribution_includes_the_target_as_a_short_leg(small_panel, small_config):
    result = run_backtest(small_panel, build_strategy("tracking", small_config), small_config)
    attribution = attribution_by_asset(result.daily_weights, small_panel.asset_returns, small_panel.target_returns)
    short_leg = f"{small_panel.target} (short)"
    assert short_leg in attribution.index
    assert attribution.loc[short_leg, "avg_weight"] == -1.0


def test_metrics_identities_hold_on_a_real_run(small_panel, small_config):
    result = run_backtest(small_panel, build_strategy("tracking", small_config), small_config)
    metrics = compute_metrics(result.daily, result.weights, result.rebalances)

    assert metrics["tracking_error"] >= metrics["tracking_error_gross"] - 1e-12 or metrics["cost_drag_annual"] >= 0
    assert metrics["r_squared"] == pytest.approx(metrics["correlation"] ** 2)
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert metrics["total_cost"] >= 0.0
    assert 1.0 <= metrics["effective_positions"] <= len(small_panel.assets) + 1e-9
    assert metrics["n_rebalances"] == len(result.rebalances)


def test_annual_turnover_excludes_the_one_off_initial_deployment(small_panel, small_config):
    result = run_backtest(small_panel, build_strategy("tracking", small_config), small_config)
    metrics = compute_metrics(result.daily, result.weights, result.rebalances)
    years = len(result.daily) / 252
    including_deployment = float(result.rebalances["one_way_turnover"].sum() / years)
    assert metrics["annual_turnover"] < including_deployment
    assert metrics["initial_deployment_notional"] == pytest.approx(1.0, abs=1e-9)


def test_equal_weight_has_the_maximum_effective_position_count(small_panel, small_config):
    config = small_config.with_changes(max_weight=1.0)
    result = run_backtest(small_panel, build_strategy("equal_weight", config), config)
    metrics = compute_metrics(result.daily, result.weights, result.rebalances)
    assert metrics["effective_positions"] == pytest.approx(len(small_panel.assets), rel=0.02)


def test_recovery_metrics_are_zero_at_the_true_weights(small_panel):
    truth = small_panel.truth.true_weights
    weights = pd.DataFrame([truth, truth], columns=truth.index)
    metrics = recovery_metrics(weights, truth)
    assert metrics["recovery_l1_mean"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["recovery_max_abs_error"] == pytest.approx(0.0, abs=1e-12)


def test_regime_breakdown_partitions_the_sample(small_panel, small_config):
    result = run_backtest(small_panel, build_strategy("tracking", small_config), small_config)
    breakdown = regime_breakdown(result.daily, small_panel.truth.regimes)
    assert breakdown["days"].sum() == len(result.daily)
    assert breakdown["share"].sum() == pytest.approx(1.0)


def test_every_reported_metric_family_has_a_written_definition(small_panel, small_config):
    """Anything a reader has to guess the sign convention of is a bug in the report."""
    result = run_backtest(small_panel, build_strategy("tracking", small_config), small_config)
    metrics = compute_metrics(result.daily, result.weights, result.rebalances)
    definitions = describe_metrics()
    for key in ("tracking_error", "information_ratio", "down_capture", "annual_turnover", "effective_positions"):
        assert key in metrics
        assert key in definitions and len(definitions[key]) > 30
