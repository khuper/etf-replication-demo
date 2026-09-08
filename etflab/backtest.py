"""The walk-forward engine.

This is the only place in the repository where a decision date meets a return
date, so it is the only place look-ahead bias can enter. It is written to make
that boundary explicit and testable rather than implicit and hoped-for.

The protocol, per rebalance at index ``i``:

1. **Fit** on returns ``[lo, i - embargo)``. Strictly prior. The embargo drops
   the ``embargo_days`` observations immediately before the decision, which is
   what a real desk loses to data arrival, model runtime, and getting the order
   to the market.
2. **Trade** into the fitted weights at the close of day ``i``, paying costs on
   the distance from the *drifted* holdings -- not from last rebalance's target,
   which would understate turnover.
3. **Hold** through ``[i, i + step)``, letting weights drift with prices. No
   free daily rebalancing.
4. Repeat, including the final partial period.

``tests/test_lookahead.py`` verifies point 1 by construction: truncate the panel
at any date and every ledger row before it must be bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from etflab.config import ExperimentConfig
from etflab.costs import BpsCostModel, CostModel, build_cost_model
from etflab.data.panel import PricePanel
from etflab.strategies import Constraints, FitContext, Strategy, build_strategy

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestResult:
    """Everything one strategy produced, at the granularity it was produced."""

    strategy: str
    label: str
    daily: pd.DataFrame  #: date x [gross, cost, net, target, active]
    weights: pd.DataFrame  #: rebalance_date x assets (post-trade targets)
    daily_weights: pd.DataFrame  #: date x assets, after intra-period drift
    rebalances: pd.DataFrame  #: rebalance_date x execution diagnostics
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> pd.Series:
        return self.daily["active"]

    @property
    def net(self) -> pd.Series:
        return self.daily["net"]

    @property
    def target(self) -> pd.Series:
        return self.daily["target"]

    @property
    def degraded_rebalances(self) -> int:
        return int((~self.rebalances["status"].isin(["optimal", "optimal_inaccurate", "analytic"])).sum())


def _period_bounds(n_obs: int, train_days: int, step: int) -> list[int]:
    """Indices in the return series at which we rebalance."""
    if n_obs <= train_days:
        raise ValueError(
            f"Need more than {train_days} return observations to start a walk-forward; only {n_obs} available."
        )
    return list(range(train_days, n_obs, step))


def run_backtest(
    panel: PricePanel,
    strategy: Strategy,
    config: ExperimentConfig,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Run one strategy through the walk-forward protocol."""
    returns = panel.returns
    asset_returns = returns[list(panel.assets)]
    target_returns = returns[panel.target]
    assets = list(panel.assets)
    n_obs = len(returns)

    cost_model = cost_model or build_cost_model(config, panel.assets)
    constraints_first = Constraints(config.max_weight, None)
    constraints_next = Constraints(config.max_weight, config.max_turnover)
    constraints_first.validate(len(assets))

    daily_rows: list[dict[str, Any]] = []
    daily_weight_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    rebalance_rows: list[dict[str, Any]] = []
    drifted: np.ndarray | None = None

    for i in _period_bounds(n_obs, config.train_days, config.rebalance_days):
        train_hi = i - config.embargo_days
        train_lo = 0 if config.train_mode == "expanding" else max(0, train_hi - config.train_days)
        train_assets = asset_returns.iloc[train_lo:train_hi]
        train_target = target_returns.iloc[train_lo:train_hi]

        ctx = FitContext(
            asset_returns=train_assets,
            target_returns=train_target,
            prev_weights=drifted,
            constraints=constraints_first if drifted is None else constraints_next,
        )
        fit = strategy.fit(ctx)
        target_weights = fit.weights

        rebalance_date = returns.index[i]
        # Trades are measured against what we are actually holding after drift,
        # which is the only quantity a trader could execute against.
        held = drifted if drifted is not None else np.zeros(len(assets))
        trades = pd.Series(target_weights - held, index=assets)
        # Volatility for the impact model uses trailing data only -- same rule as
        # the strategy. Using realised volatility of the period being traded into
        # would be a look-ahead in the cost model, which is a genuinely easy miss.
        trailing_vol = train_assets.tail(63).std(ddof=1)
        trade_cost = cost_model.cost(trades, trailing_vol)

        weight_rows.append(pd.Series(target_weights, index=assets, name=rebalance_date))
        rebalance_rows.append(
            {
                "rebalance_date": rebalance_date,
                "traded_notional": trade_cost.traded_notional,
                "one_way_turnover": trade_cost.one_way_turnover,
                "cost_total_bps": trade_cost.total * 1e4,
                "cost_spread_bps": trade_cost.spread * 1e4,
                "cost_impact_bps": trade_cost.impact * 1e4,
                "max_participation": trade_cost.max_participation,
                "status": fit.status,
                "solver": fit.solver,
                "train_obs": len(train_assets),
                "train_end": train_assets.index[-1],
                "max_weight_used": float(np.max(target_weights)),
                "n_positions": int((target_weights > 1e-6).sum()),
            }
        )

        current = target_weights.copy()
        period = returns.iloc[i : min(i + config.rebalance_days, n_obs)]
        for day_number, (day, row) in enumerate(period.iterrows()):
            asset_day = row[assets].to_numpy(dtype=float)
            gross = float(current @ asset_day)
            cost = trade_cost.total if day_number == 0 else 0.0
            net = gross - cost
            tgt = float(row[panel.target])
            daily_weight_rows.append(pd.Series(current, index=assets, name=day))
            daily_rows.append(
                {
                    "date": day,
                    "gross": gross,
                    "cost": cost,
                    "net": net,
                    "target": tgt,
                    "active": net - tgt,
                }
            )
            denominator = 1.0 + gross
            if denominator <= 0:
                raise RuntimeError(
                    f"Portfolio value hit zero on {day.date()}: a long-only book lost 100% in a day, "
                    "which means the input returns are wrong."
                )
            current = current * (1.0 + asset_day) / denominator
        drifted = current

    daily = pd.DataFrame(daily_rows).set_index("date")
    daily_weights = pd.DataFrame(daily_weight_rows)
    daily_weights.index.name = "date"
    weights = pd.DataFrame(weight_rows)
    weights.index.name = "rebalance_date"
    rebalances = pd.DataFrame(rebalance_rows).set_index("rebalance_date")

    return BacktestResult(
        strategy=getattr(strategy, "name", "unknown"),
        label=getattr(strategy, "label", getattr(strategy, "name", "unknown")),
        daily=daily,
        weights=weights,
        daily_weights=daily_weights,
        rebalances=rebalances,
        meta={
            "n_rebalances": len(rebalances),
            "train_mode": config.train_mode,
            "train_days": config.train_days,
            "embargo_days": config.embargo_days,
            "rebalance_days": config.rebalance_days,
            "cost_model": getattr(cost_model, "name", "unknown"),
        },
    )


def run_zoo(
    panel: PricePanel,
    config: ExperimentConfig,
    names: tuple[str, ...],
    cost_model: CostModel | None = None,
) -> dict[str, BacktestResult]:
    """Run several strategies over the identical panel, protocol and cost model.

    Same data, same windows, same constraints, same costs. Any difference in the
    results is then attributable to the weights and nothing else.
    """
    model = cost_model or build_cost_model(config, panel.assets)
    results: dict[str, BacktestResult] = {}
    for name in names:
        strategy_config = config if name != "cvar" else config.with_changes(cvar_ratio=config.cvar_ratio or 1.0)
        results[name] = run_backtest(panel, build_strategy(name, strategy_config), config, model)
    return results


def zero_cost_result(panel: PricePanel, config: ExperimentConfig, name: str) -> BacktestResult:
    """The same strategy with costs switched off -- used for cost attribution."""
    return run_backtest(panel, build_strategy(name, config), config, BpsCostModel(0.0))
