"""Replication-quality metrics for backtest evaluation."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if returns.empty:
        return np.nan
    growth = float((1 + returns).prod())
    return growth ** (periods_per_year / len(returns)) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1)) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the compounded series (negative number)."""
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def max_active_drawdown(portfolio: pd.Series, target: pd.Series) -> float:
    """Largest peak-to-trough loss of portfolio wealth relative to target wealth."""
    relative = (1 + portfolio).cumprod() / (1 + target).cumprod()
    return float((relative / relative.cummax() - 1).min())


def tracking_metrics(
    portfolio: pd.Series, target: pd.Series, periods_per_year: int = TRADING_DAYS
) -> Dict[str, float]:
    """Headline replication metrics for a portfolio-vs-target return pair."""
    active = portfolio - target
    tracking_error = annualized_volatility(active, periods_per_year)
    ann_active = float(active.mean()) * periods_per_year
    target_var = float(target.var(ddof=1))
    return {
        "annualized_return": annualized_return(portfolio, periods_per_year),
        "target_annualized_return": annualized_return(target, periods_per_year),
        "annualized_active_return": ann_active,
        "tracking_error": tracking_error,
        "information_ratio": ann_active / tracking_error if tracking_error > 0 else np.nan,
        "correlation": float(portfolio.corr(target)),
        "beta": float(portfolio.cov(target) / target_var) if target_var > 0 else np.nan,
        "max_drawdown": max_drawdown(portfolio),
        "max_active_drawdown": max_active_drawdown(portfolio, target),
    }


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("ME").prod() - 1


def monthly_active_returns(portfolio: pd.Series, target: pd.Series) -> pd.Series:
    return monthly_returns(portfolio) - monthly_returns(target)


def hit_rate(monthly_active: pd.Series, tolerance: float = 0.005) -> float:
    """Share of months where the replicator tracked within +/- tolerance."""
    if monthly_active.empty:
        return np.nan
    return float((monthly_active.abs() <= tolerance).mean())


def rolling_tracking_error(
    portfolio: pd.Series, target: pd.Series, window: int = 60, periods_per_year: int = TRADING_DAYS
) -> pd.Series:
    active = portfolio - target
    return active.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_correlation(portfolio: pd.Series, target: pd.Series, window: int = 60) -> pd.Series:
    return portfolio.rolling(window).corr(target)


def group_allocation(weights: pd.DataFrame, classes_by_asset: Mapping[str, str]) -> pd.DataFrame:
    """Aggregate a weights history by asset class."""
    groups = pd.Series({a: classes_by_asset.get(a, "other") for a in weights.columns})
    return weights.T.groupby(groups).sum().T


def stress_window_metrics(
    portfolio: pd.Series,
    target: pd.Series,
    windows: Mapping[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Replication quality within named historical stress windows."""
    rows = {}
    for name, (start, end) in windows.items():
        p = portfolio.loc[start:end]
        t = target.loc[start:end]
        if len(p) < 2:
            continue
        active = p - t
        rows[name] = {
            "portfolio_return": float((1 + p).prod() - 1),
            "target_return": float((1 + t).prod() - 1),
            "active_return": float((1 + p).prod() - (1 + t).prod()),
            "tracking_error": annualized_volatility(active),
            "correlation": float(p.corr(t)),
        }
    return pd.DataFrame(rows).T


def summarize_backtest(result) -> pd.DataFrame:
    """Gross and net headline metrics for a BacktestResult, as a tidy table."""
    gross = tracking_metrics(result.gross_returns, result.target_returns)
    net = tracking_metrics(result.net_returns, result.target_returns)
    table = pd.DataFrame({"gross": gross, "net": net})
    table.loc["avg_turnover_per_rebalance"] = result.turnover.mean()
    total_gross = (1 + result.gross_returns).prod() - 1
    total_net = (1 + result.net_returns).prod() - 1
    table.loc["total_cost_drag"] = [0.0, float(total_gross - total_net)]
    return table
