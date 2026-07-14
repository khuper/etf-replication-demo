"""Rolling ETF-to-target correlation analysis across normal and stress periods."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import STRESS_WINDOWS
from .data import fetch_prices, to_log_returns
from .plotting import plot_mean_rolling_correlation, plot_rolling_correlation_grid

logger = logging.getLogger(__name__)

DEFAULT_CORRELATION_ASSETS = ["SPY", "QQQ", "IWM", "EFA", "VWO", "AGG", "TLT", "GLD", "VNQ", "HYG"]
WINDOWS = (60, 120)


def _windows_in_range(index: pd.DatetimeIndex) -> Dict[str, Tuple[str, str]]:
    return {
        name: (start, end)
        for name, (start, end) in STRESS_WINDOWS.items()
        if pd.to_datetime(start) >= index[0] and pd.to_datetime(end) <= index[-1]
    }


def correlation_stats(
    rolling_corr: pd.DataFrame, stress_windows: Dict[str, Tuple[str, str]]
) -> pd.DataFrame:
    """Per-asset average correlation, full period and within each stress window."""
    rows = []
    for asset in rolling_corr.columns:
        corr = rolling_corr[asset].dropna()
        row = {"full_period_avg": corr.mean()}
        for name, (start, end) in stress_windows.items():
            window = corr.loc[start:end]
            key = name.split()[0].lower()
            row[f"{key}_avg"] = window.mean() if not window.empty else np.nan
            row[f"{key}_max"] = window.max() if not window.empty else np.nan
            row[f"{key}_delta"] = window.mean() - corr.mean() if not window.empty else np.nan
        rows.append(pd.Series(row, name=asset))
    return pd.DataFrame(rows)


def run_correlation_analysis(
    assets: List[str],
    target: str,
    start_date: str,
    end_date: str,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """Compute rolling correlations, save stats and charts, return the stats table."""
    os.makedirs(output_dir, exist_ok=True)

    prices = fetch_prices(assets + [target], start_date, end_date)
    log_returns = to_log_returns(prices)
    target_returns = log_returns[target]
    asset_returns = log_returns[assets]
    stress_windows = _windows_in_range(log_returns.index)

    rolling = {w: asset_returns.rolling(window=w).corr(target_returns) for w in WINDOWS}
    base_window = WINDOWS[0]

    stats = correlation_stats(rolling[base_window], stress_windows)
    stats_path = os.path.join(output_dir, "correlation_stats.csv")
    stats.to_csv(stats_path)
    logger.info("Correlation statistics saved to %s", stats_path)
    logger.info("\n%s", stats.round(3).to_string())

    plot_rolling_correlation_grid(
        rolling[base_window], target, stress_windows,
        os.path.join(output_dir, "rolling_correlation_individual.png"),
    )
    plot_mean_rolling_correlation(
        {f"{w}-day mean correlation": rolling[w].mean(axis=1) for w in WINDOWS},
        target, stress_windows,
        os.path.join(output_dir, "rolling_mean_correlation.png"),
    )

    logger.info(
        "Interpretation: if mean correlation spikes toward 1 in the shaded stress windows, "
        "diversification inside the basket disappears exactly when replication matters most; "
        "assets whose correlation is stable across windows are the reliable building blocks."
    )
    return stats
