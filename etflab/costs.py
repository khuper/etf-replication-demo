"""Execution cost models.

A backtest's conclusion is usually a statement about costs in disguise. Two
models are provided and both are explicit about their assumptions:

``bps``
    A flat basis-point charge on one-way traded notional. Fine for a first pass,
    and wrong in the specific way that matters: it says trading $1bn costs the
    same per dollar as trading $1m.

``spread_impact``
    Half-spread plus a square-root market-impact term, per asset::

        cost_bps(i) = half_spread_bps(i) + k * sigma_daily_bps(i) * sqrt(Q_i / ADV_i)

    the standard concave impact form (Almgren et al., 2005; Grinold & Kahn).
    ``Q_i`` is the traded notional in asset ``i``, so cost now scales with the
    size of the book -- which is what makes the "at what AUM does this stop
    working?" question answerable instead of rhetorical.

The liquidity table below is *illustrative*, not measured. It is stated here in
one place, versioned with the code, so a reader can disagree with a number and
re-run rather than guess what was assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

BPS = 1e-4

#: (half-spread in bps, average daily dollar volume). Order-of-magnitude figures
#: for large US-listed ETFs; override via :class:`LiquidityProfile`.
DEFAULT_LIQUIDITY: dict[str, tuple[float, float]] = {
    "SPY": (0.3, 30_000_000_000.0),
    "QQQ": (0.4, 15_000_000_000.0),
    "IWM": (0.6, 4_000_000_000.0),
    "VEA": (1.0, 700_000_000.0),
    "VWO": (1.2, 900_000_000.0),
    "HYG": (1.2, 1_800_000_000.0),
    "LQD": (1.0, 1_500_000_000.0),
    "TIP": (1.5, 400_000_000.0),
    "GLD": (0.8, 2_000_000_000.0),
    "VNQ": (1.5, 500_000_000.0),
    "AGG": (1.0, 700_000_000.0),
    "TLT": (0.8, 2_500_000_000.0),
    "EFA": (1.0, 2_000_000_000.0),
    "PSP": (12.0, 15_000_000.0),
}

#: What we assume about a ticker we have never heard of. Deliberately pessimistic:
#: an unknown ETF is far more likely to be thin than to be SPY.
UNKNOWN_LIQUIDITY = (5.0, 20_000_000.0)


@dataclass(frozen=True)
class LiquidityProfile:
    """Per-asset half-spread and ADV assumptions."""

    half_spread_bps: pd.Series
    adv_usd: pd.Series

    @classmethod
    def for_assets(
        cls,
        assets: tuple[str, ...],
        overrides: Mapping[str, tuple[float, float]] | None = None,
    ) -> LiquidityProfile:
        table = {**DEFAULT_LIQUIDITY, **dict(overrides or {})}
        spreads, advs = {}, {}
        for asset in assets:
            spread, adv = table.get(asset, UNKNOWN_LIQUIDITY)
            spreads[asset] = float(spread)
            advs[asset] = float(adv)
        return cls(pd.Series(spreads, dtype=float), pd.Series(advs, dtype=float))

    def as_dict(self) -> dict[str, Any]:
        return {
            "half_spread_bps": {k: float(v) for k, v in self.half_spread_bps.items()},
            "adv_usd": {k: float(v) for k, v in self.adv_usd.items()},
        }


@dataclass(frozen=True)
class TradeCost:
    """The cost of one rebalance, decomposed so it can be argued with.

    Two turnover numbers, because conflating them is a real and expensive mistake:

    ``traded_notional``
        ``sum |dw_i|`` -- the total notional that changes hands, and therefore
        the base costs are charged on. Every leg pays, so a rebalance that sells
        10% and buys 10% has traded 20% and pays for 20%.
    ``one_way_turnover``
        ``traded_notional / 2`` -- the conventional reporting figure. It equals
        the traded notional only for a self-financing rebalance where buys match
        sells. The initial deployment out of cash is *not* self-financing: it
        buys 100% and sells nothing, so it trades 1.0 of notional while its
        conventional turnover figure reads 0.5.
    """

    total: float  #: fraction of portfolio value
    spread: float
    impact: float
    traded_notional: float
    one_way_turnover: float
    max_participation: float
    per_asset_bps: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "spread": self.spread,
            "impact": self.impact,
            "traded_notional": self.traded_notional,
            "one_way_turnover": self.one_way_turnover,
            "max_participation": self.max_participation,
        }


class CostModel(Protocol):
    @property
    def name(self) -> str:  # pragma: no cover - protocol
        ...

    def cost(self, trades: pd.Series, volatility: pd.Series) -> TradeCost:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class BpsCostModel:
    """Flat cost per unit of one-way traded notional."""

    cost_bps: float = 5.0
    name: str = "bps"

    def cost(self, trades: pd.Series, volatility: pd.Series) -> TradeCost:
        # Charged on traded notional, not on the halved reporting figure: each leg
        # of a rebalance crosses a spread, and the initial deployment out of cash
        # crosses one for the whole book.
        traded = float(trades.abs().sum())
        total = traded * self.cost_bps * BPS
        return TradeCost(
            total=total,
            spread=total,
            impact=0.0,
            traded_notional=traded,
            one_way_turnover=traded / 2.0,
            max_participation=0.0,
            per_asset_bps={k: self.cost_bps for k in trades.index},
        )


@dataclass(frozen=True)
class SpreadImpactCostModel:
    """Half-spread plus square-root impact, sized against a real book.

    ``trades`` are weight *changes*; the traded notional in asset ``i`` is
    ``|dw_i| * portfolio_notional``. Note the asymmetry with the flat model: here
    doubling the book more than doubles the cost of the same weight change.
    """

    liquidity: LiquidityProfile
    portfolio_notional: float = 50_000_000.0
    impact_coef: float = 0.7
    name: str = "spread_impact"

    def cost(self, trades: pd.Series, volatility: pd.Series) -> TradeCost:
        traded_weight = trades.abs()
        notional = traded_weight * self.portfolio_notional
        adv = self.liquidity.adv_usd.reindex(trades.index).astype(float)
        half_spread = self.liquidity.half_spread_bps.reindex(trades.index).astype(float)
        sigma_bps = volatility.reindex(trades.index).astype(float).fillna(0.0) / BPS

        participation = (notional / adv.replace(0.0, np.nan)).fillna(0.0)
        impact_bps = self.impact_coef * sigma_bps * np.sqrt(participation)

        spread_cost = float((traded_weight * half_spread * BPS).sum())
        impact_cost = float((traded_weight * impact_bps * BPS).sum())
        per_asset = (half_spread + impact_bps).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return TradeCost(
            total=spread_cost + impact_cost,
            spread=spread_cost,
            impact=impact_cost,
            traded_notional=float(traded_weight.sum()),
            one_way_turnover=float(traded_weight.sum() / 2.0),
            max_participation=float(participation.max()) if len(participation) else 0.0,
            per_asset_bps={k: float(v) for k, v in per_asset.items()},
        )


def build_cost_model(config: Any, assets: tuple[str, ...]) -> CostModel:
    """Construct the cost model named by ``config.cost_model``."""
    if config.cost_model == "bps":
        return BpsCostModel(cost_bps=config.cost_bps)
    if config.cost_model == "spread_impact":
        return SpreadImpactCostModel(
            liquidity=LiquidityProfile.for_assets(assets),
            portfolio_notional=config.portfolio_notional,
            impact_coef=config.impact_coef,
        )
    raise ValueError(f"Unknown cost model {config.cost_model!r}.")
