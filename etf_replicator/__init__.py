"""Replicate an illiquid target ETF with a liquid basket."""

from .backtest import BacktestResult, run_expanding_backtest
from .config import ReplicatorConfig
from .data import fetch_prices, split_assets_target, to_log_returns, to_returns
from .metrics import summarize_backtest, tracking_metrics
from .optimizer import optimize_tracking_error
from .report import build_tearsheet
from .stress import beta_stress_test

__all__ = [
    "BacktestResult",
    "ReplicatorConfig",
    "beta_stress_test",
    "build_tearsheet",
    "fetch_prices",
    "optimize_tracking_error",
    "run_expanding_backtest",
    "split_assets_target",
    "summarize_backtest",
    "to_log_returns",
    "to_returns",
    "tracking_metrics",
]
