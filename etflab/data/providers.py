"""Where prices come from, and how a ragged download becomes a rectangular panel.

Three providers share one contract: given a config, return an aligned
:class:`~etflab.data.panel.PricePanel`. Swapping ``--data-source synthetic`` for
``--data-source yfinance`` changes nothing downstream, which is the only way to
keep a research pipeline honest about what it depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from etflab.config import ExperimentConfig
from etflab.data.cache import PanelCache, cache_key
from etflab.data.panel import PricePanel
from etflab.data.synthetic import SyntheticSpec, generate_market


@dataclass(frozen=True)
class AlignmentReport:
    """What alignment cost us, stated rather than swallowed."""

    requested_start: str
    requested_end: str
    effective_start: str
    effective_end: str
    binding_ticker: str | None
    rows_dropped_leading: int
    rows_dropped_internal: int
    per_ticker_first_valid: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "effective_start": self.effective_start,
            "effective_end": self.effective_end,
            "binding_ticker": self.binding_ticker,
            "rows_dropped_leading": self.rows_dropped_leading,
            "rows_dropped_internal": self.rows_dropped_internal,
            "per_ticker_first_valid": self.per_ticker_first_valid,
        }


def align_prices(raw: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, AlignmentReport]:
    """Turn a ragged price frame into a rectangular one, and say what that cost.

    The naive version of this function is ``df.dropna()``. It is wrong in a way
    that is easy to miss: if one ticker listed five years after the others, a
    blanket dropna silently discards five years of history for *everything* and
    the run reports a shorter sample without ever mentioning why. Here the
    truncation is attributed to the ticker that caused it and surfaced in the
    quality report, so the reader can decide whether to drop the ticker instead.
    """
    missing = [c for c in columns if c not in raw.columns]
    if missing:
        raise KeyError(f"Price data is missing required columns: {missing}")
    frame = raw[columns].copy().sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    requested_start, requested_end = frame.index[0], frame.index[-1]
    first_valid = {c: frame[c].first_valid_index() for c in columns}
    if any(v is None for v in first_valid.values()):
        empty = [c for c, v in first_valid.items() if v is None]
        raise ValueError(f"No usable price history for {empty}.")

    binding_ticker = max(first_valid, key=lambda c: first_valid[c])  # latest inception
    effective_start = first_valid[binding_ticker]
    trimmed = frame.loc[effective_start:]
    rows_dropped_leading = int(len(frame) - len(trimmed))

    complete = trimmed.dropna(how="any")
    rows_dropped_internal = int(len(trimmed) - len(complete))
    if complete.empty:
        raise ValueError("Price panel is empty after alignment; tickers have no overlapping history.")

    report = AlignmentReport(
        requested_start=str(requested_start.date()),
        requested_end=str(requested_end.date()),
        effective_start=str(complete.index[0].date()),
        effective_end=str(complete.index[-1].date()),
        binding_ticker=binding_ticker if rows_dropped_leading > 0 else None,
        rows_dropped_leading=rows_dropped_leading,
        rows_dropped_internal=rows_dropped_internal,
        per_ticker_first_valid={c: str(v.date()) for c, v in first_valid.items()},
    )
    return complete, report


class Provider(Protocol):
    @property
    def name(self) -> str:  # pragma: no cover - protocol
        ...

    def fetch(self, config: ExperimentConfig) -> PricePanel:  # pragma: no cover - protocol
        ...


class SyntheticProvider:
    """Deterministic generated market. The default, and the one CI uses."""

    name = "synthetic"

    def fetch(self, config: ExperimentConfig) -> PricePanel:
        spec = SyntheticSpec(
            assets=config.assets,
            target=config.target,
            start=config.start,
            end=config.end,
            seed=config.synthetic_seed,
            leverage=config.synthetic_leverage,
        )
        return generate_market(spec)


class YFinanceProvider:
    """Real adjusted closes from Yahoo, when a network is available.

    Nothing downstream knows or cares that this provider exists, which is the
    point: the research conclusions are computed the same way regardless of where
    the prices came from.
    """

    name = "yfinance"

    def fetch(self, config: ExperimentConfig) -> PricePanel:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("yfinance is not installed; use --data-source synthetic or csv.") from exc

        tickers = list(config.universe)
        raw = yf.download(
            tickers,
            start=config.start,
            end=config.end,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise RuntimeError(
                "yfinance returned no data. This usually means no network access or a bad ticker. "
                "The default --data-source synthetic runs fully offline."
            )
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.levels[0]:
                raise KeyError(f"No 'Close' level in yfinance columns: {raw.columns.levels[0]}")
            prices = raw["Close"]
        else:
            prices = raw
        prices, report = align_prices(prices, tickers)
        prices.index.name = "date"
        return PricePanel(
            prices,
            config.assets,
            config.target,
            "yfinance",
            {"alignment": report.as_dict(), "auto_adjust": True},
        )


class CsvProvider:
    """A wide CSV of adjusted closes: ``date`` index, one column per ticker.

    The escape hatch for anyone with their own data. It is also how a reader
    reproduces the published results against real prices without this repo ever
    shipping vendor data it has no licence to redistribute.
    """

    name = "csv"

    def fetch(self, config: ExperimentConfig) -> PricePanel:
        path = Path(config.csv_path or "")
        if not path.exists():
            raise FileNotFoundError(f"csv_path does not exist: {path}")
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        prices, report = align_prices(raw, list(config.universe))
        prices = prices.loc[config.start : config.end]
        if prices.empty:
            raise ValueError(f"No rows in {path} between {config.start} and {config.end}.")
        prices.index.name = "date"
        return PricePanel(
            prices,
            config.assets,
            config.target,
            "csv",
            {"path": str(path), "alignment": report.as_dict()},
        )


PROVIDERS: dict[str, Provider] = {
    "synthetic": SyntheticProvider(),
    "yfinance": YFinanceProvider(),
    "csv": CsvProvider(),
}


def resolve_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown data source {name!r}; available: {sorted(PROVIDERS)}") from exc


def load_panel(config: ExperimentConfig, *, use_cache: bool = True) -> PricePanel:
    """Fetch (or read from cache) the price panel described by ``config``.

    Synthetic panels are never cached: regenerating them is cheaper than reading
    a file, and caching them would put a stale copy between the seed and the
    numbers, which is exactly the coupling this layer exists to prevent.
    """
    provider = resolve_provider(config.data_source)
    if config.data_source == "synthetic":
        return provider.fetch(config)

    extra = {"csv_path": config.csv_path} if config.data_source == "csv" else {}
    key = cache_key(config.data_source, config.universe, config.start, config.end, extra)
    cache = PanelCache(config.cache_dir)

    if use_cache:
        cached = cache.load(key, config.assets, config.target)
        if cached is not None:
            return cached
    if config.offline:
        raise RuntimeError(
            f"offline=True and no cache entry for {config.data_source} "
            f"({config.universe[0]}..{config.target}, {config.start}..{config.end}). "
            "Run once with --online to populate the cache, or use --data-source synthetic."
        )

    panel = provider.fetch(config)
    cache.store(key, panel)
    return panel


def realised_daily_vol(panel: PricePanel, lookback: int = 63) -> pd.Series:
    """Trailing daily volatility per asset -- an input to the market-impact model."""
    window = panel.asset_returns.tail(lookback)
    if window.empty:
        return pd.Series(0.0, index=list(panel.assets))
    return window.std(ddof=1).replace(0.0, np.nan).fillna(window.std(ddof=1).mean())
