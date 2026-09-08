"""The single in-memory representation of market data used by everything downstream.

One type, one column order, one fingerprint. Every consumer of prices in this
repository goes through :class:`PricePanel`, so there is exactly one place where
"how do we turn prices into returns" is decided.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PricePanel:
    """Aligned adjusted-close prices for a candidate universe plus its target.

    Attributes
    ----------
    prices:
        Business-day indexed frame. Columns are ``(*assets, target)`` in that
        exact order -- the order is load-bearing: weight vectors, covariance
        matrices and fingerprints all assume it.
    assets:
        Candidate proxy tickers.
    target:
        The instrument being replicated.
    source:
        Provider name (``synthetic``, ``yfinance``, ``csv``).
    meta:
        Free-form provenance from the provider (seed, cache path, download time).
    truth:
        Ground truth, when the panel came from a generator that has one. Real
        market data has no truth object; synthetic data does, which is what makes
        the recovery study in ``etflab.research.recovery`` possible.
    """

    prices: pd.DataFrame
    assets: tuple[str, ...]
    target: str
    source: str
    meta: Mapping[str, Any] = field(default_factory=dict)
    truth: MarketTruth | None = None

    def __post_init__(self) -> None:
        expected = [*self.assets, self.target]
        if list(self.prices.columns) != expected:
            raise ValueError(
                "PricePanel columns must be exactly (*assets, target) in order; "
                f"got {list(self.prices.columns)} expected {expected}."
            )
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("PricePanel requires a DatetimeIndex.")
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("PricePanel index must be sorted ascending.")
        if self.prices.index.has_duplicates:
            raise ValueError("PricePanel index contains duplicate dates.")

    # ------------------------------------------------------------------ #
    @property
    def returns(self) -> pd.DataFrame:
        """Simple daily returns.

        Simple (not log) returns because portfolio arithmetic is linear in them:
        a weighted basket's return is the weighted sum of constituent returns,
        which is exactly the quantity the optimiser and the backtester need.
        Log returns would be wrong here and are used only for diagnostics.
        """
        return self.prices.pct_change().dropna(how="any")

    @property
    def asset_returns(self) -> pd.DataFrame:
        return self.returns[list(self.assets)]

    @property
    def target_returns(self) -> pd.Series:
        return self.returns[self.target]

    @property
    def n_obs(self) -> int:
        return len(self.prices)

    @property
    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.prices.index[0], self.prices.index[-1]

    # ------------------------------------------------------------------ #
    def fingerprint(self) -> str:
        """Content hash of the actual numbers, not of the request that fetched them.

        Two runs share a fingerprint only if they saw byte-identical prices. This
        is what lets a manifest prove that a rerun used the same data rather than
        the same *query* -- a distinction that matters the moment a vendor
        restates history.
        """
        h = hashlib.sha256()
        h.update("|".join(self.prices.columns).encode("utf-8"))
        h.update(self.prices.index.astype("int64").to_numpy().tobytes())
        # Round to 1e-10 before hashing so that platform-level float noise in the
        # last ULP does not create spurious fingerprint mismatches.
        values = np.round(self.prices.to_numpy(dtype=float), 10)
        h.update(np.ascontiguousarray(values).tobytes())
        return h.hexdigest()[:16]

    def slice(self, start: pd.Timestamp | str | None = None, end: pd.Timestamp | str | None = None) -> PricePanel:
        """Return a date-restricted copy, preserving provenance."""
        sliced = self.prices.loc[start:end]
        return PricePanel(sliced, self.assets, self.target, self.source, dict(self.meta), self.truth)

    def head_obs(self, n: int) -> PricePanel:
        """First ``n`` observations -- used by the look-ahead sentinel test."""
        return PricePanel(self.prices.iloc[:n], self.assets, self.target, self.source, dict(self.meta), self.truth)

    def describe(self) -> dict[str, Any]:
        first, last = self.span
        return {
            "source": self.source,
            "target": self.target,
            "n_assets": len(self.assets),
            "n_obs": self.n_obs,
            "start": str(first.date()),
            "end": str(last.date()),
            "fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True)
class MarketTruth:
    """Ground truth for a generated market.

    Only synthetic panels carry one. It is the reason the synthetic default is a
    feature rather than an apology: with the true replicating portfolio known, the
    estimator can be scored on *recovery* (did it find the right weights?) and not
    only on fit (did it track well in-sample?).
    """

    true_weights: pd.Series
    factor_loadings: pd.DataFrame
    factor_returns: pd.DataFrame
    unspanned_returns: pd.Series
    regimes: pd.Series
    irreducible_te_annual: float
    leverage: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "true_weights": {k: float(v) for k, v in self.true_weights.items()},
            "irreducible_te_annual": float(self.irreducible_te_annual),
            "leverage": float(self.leverage),
            "factors": list(self.factor_loadings.columns),
            "regime_shares": {str(k): float(v) for k, v in self.regimes.value_counts(normalize=True).items()},
        }
