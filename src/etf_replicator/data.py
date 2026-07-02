"""Market data download and return computation shared by all workflows."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices, one column per ticker."""
    logger.info("Fetching data for: %s", ", ".join(tickers))
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError("No data downloaded. Check your tickers and network connection.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.levels[0]:
            raise KeyError(f"Could not find 'Close' in columns: {df.columns.levels[0]}")
        prices = df["Close"].copy()
    else:
        prices = df.copy()

    prices = prices.dropna()
    if prices.empty:
        raise ValueError("Price dataframe is empty after dropping NAs. Tickers might have non-overlapping history.")
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change(fill_method=None).dropna()


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def split_assets_target(
    returns: pd.DataFrame, assets: List[str], target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a combined returns frame into basket columns and the target series."""
    return returns[assets], returns[target]
