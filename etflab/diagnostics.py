"""Stress tests, cost break-evens, and the other questions a reviewer asks second.

The first question about a replication result is "how tight is the tracking?".
Every question after that is in here: what happens in the tail, what happens when
costs are worse than assumed, what happens when the book is bigger, and whether
the relationships the whole thing rests on are stable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from etflab.backtest import BacktestResult, run_backtest
from etflab.config import ExperimentConfig
from etflab.costs import BpsCostModel, LiquidityProfile, SpreadImpactCostModel
from etflab.data.panel import PricePanel
from etflab.metrics import annualised_return, annualised_vol
from etflab.strategies import build_strategy

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Stress
# --------------------------------------------------------------------------- #
def worst_window_replay(daily: pd.DataFrame, window: int = 21, top_n: int = 5) -> pd.DataFrame:
    """The target's worst rolling windows, and what the replicator did in them.

    Historical replay rather than a parametric shock: these are periods that
    actually happened in the evaluation sample, so nobody has to agree with a
    covariance assumption to accept the numbers.
    """
    target_window = daily["target"].rolling(window).sum()
    replicator_window = daily["net"].rolling(window).sum()
    ranked = target_window.nsmallest(top_n * window)

    # Keep only non-overlapping windows, worst first.
    chosen: list[pd.Timestamp] = []
    for date in ranked.index:
        if all(abs((date - other).days) > window * 1.5 for other in chosen):
            chosen.append(date)
        if len(chosen) >= top_n:
            break

    rows = []
    for end in sorted(chosen):
        rows.append(
            {
                "window_end": end,
                "target_return": float(target_window[end]),
                "replicator_return": float(replicator_window[end]),
                "active_return": float(replicator_window[end] - target_window[end]),
                "capture": float(replicator_window[end] / target_window[end]) if target_window[end] != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("window_end").sort_values("target_return")


def conditional_tail_performance(
    daily: pd.DataFrame, quantiles: tuple[float, ...] = (0.01, 0.05, 0.10)
) -> pd.DataFrame:
    """Mean active return conditioned on the target's worst days.

    This is the number that decides whether a replication is usable as a hedge.
    Average tracking error tells you nothing about whether the basket fails
    exactly when it is needed.
    """
    rows = []
    for q in quantiles:
        cutoff = daily["target"].quantile(q)
        tail = daily[daily["target"] <= cutoff]
        if tail.empty:
            continue
        rows.append(
            {
                "quantile": q,
                "n_days": len(tail),
                "target_mean": float(tail["target"].mean()),
                "replicator_mean": float(tail["net"].mean()),
                "active_mean": float(tail["active"].mean()),
                "active_worst": float(tail["active"].min()),
                "capture": float(tail["net"].mean() / tail["target"].mean()) if tail["target"].mean() != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("quantile")


def beta_shock(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    target_returns: pd.Series,
    shocks: tuple[float, ...] = (-0.30, -0.20, -0.10, 0.10, 0.20),
    lookback: int = 504,
) -> pd.DataFrame:
    """Beta-based shock propagation, estimated on trailing data only.

    The obvious implementation estimates betas on the *whole* sample and then
    reports how the portfolio would have behaved in a shock inside that sample --
    which uses the shock to predict itself. Here betas come from the last
    ``lookback`` observations before the end of the evaluation window, so the
    exercise is a forecast rather than a recollection.

    A linear shock model is a first-order approximation and says so: the
    conditional-tail table above is the empirical counterpart, and the two
    disagreeing is informative.
    """
    trailing_assets = asset_returns.tail(lookback)
    trailing_target = target_returns.reindex(trailing_assets.index)
    variance = float(trailing_target.var(ddof=1))
    if variance <= 0:
        raise ValueError("Target has zero variance over the estimation window.")

    betas = trailing_assets.apply(lambda col: float(np.cov(col, trailing_target, ddof=1)[0, 1] / variance))
    aligned = weights.reindex(betas.index).fillna(0.0)
    portfolio_beta = float((aligned * betas).sum())

    rows = []
    for shock in shocks:
        rows.append(
            {
                "target_shock": shock,
                "portfolio_impact": portfolio_beta * shock,
                "active_impact": portfolio_beta * shock - shock,
                "portfolio_beta": portfolio_beta,
            }
        )
    return pd.DataFrame(rows).set_index("target_shock")


def rolling_correlation_stability(
    asset_returns: pd.DataFrame,
    target_returns: pd.Series,
    window: int = 126,
) -> pd.DataFrame:
    """Rolling correlation of each proxy to the target, summarised.

    Replication is a bet that these relationships persist. This table is the
    evidence for or against that bet: an asset with a high mean correlation and a
    wide range is not a stable proxy, it is a proxy that was right on average.
    """
    rolling = asset_returns.rolling(window).corr(target_returns).dropna(how="all")
    summary = pd.DataFrame(
        {
            "mean": rolling.mean(),
            "std": rolling.std(ddof=1),
            "min": rolling.min(),
            "max": rolling.max(),
            "last": rolling.iloc[-1] if len(rolling) else np.nan,
        }
    )
    summary["range"] = summary["max"] - summary["min"]
    return summary.sort_values("mean", ascending=False)


# --------------------------------------------------------------------------- #
# Constraint activity
# --------------------------------------------------------------------------- #
def constraint_activity_summary(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Per strategy: how often each advertised constraint actually bound.

    A constraint that is stated in the write-up, plotted in the diagram and
    never binds is not risk control, it is presentation. This table is how the
    report tells the two apart. Shadow prices are the solver duals -- the
    marginal objective improvement per unit of relaxation -- and are reported
    only for strategies whose objective is a tracking error, where they have an
    economic meaning.
    """
    rows = []
    for name, result in results.items():
        r = result.rebalances
        n = len(r)
        if n == 0 or "cap_binding_count" not in r:
            continue
        ongoing = r.iloc[1:] if n > 1 else r.iloc[:0]
        rows.append(
            {
                "strategy": name,
                "rebalances": n,
                "cap_binds_share": float((r["cap_binding_count"] > 0).mean()),
                "cap_assets_at_limit_mean": float(r["cap_binding_count"].mean()),
                "turnover_binds_share": float(ongoing["turnover_binding"].mean()) if len(ongoing) else float("nan"),
                "turnover_used_share_of_cap": float(ongoing["turnover_share_of_cap"].mean())
                if len(ongoing)
                else float("nan"),
                "cvar_binds_share": float(r["cvar_binding"].mean()) if "cvar_binding" in r else float("nan"),
                "cap_shadow_price_max": float(r["cap_shadow_price"].max())
                if r["cap_shadow_price"].notna().any()
                else float("nan"),
                "turnover_shadow_price_max": float(r["turnover_shadow_price"].max())
                if r["turnover_shadow_price"].notna().any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("strategy") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Cost and capacity break-evens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BreakEven:
    """Where an advantage disappears, and whether it ever existed."""

    variable: str
    values: list[float]
    advantage: list[float]
    breakeven: float | None
    baseline_advantage: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "values": self.values,
            "advantage": self.advantage,
            "breakeven": self.breakeven,
            "baseline_advantage": self.baseline_advantage,
        }


def _interpolate_zero(x: list[float], y: list[float]) -> float | None:
    """First crossing of zero, linearly interpolated. ``None`` if it never crosses."""
    for i in range(1, len(y)):
        if (y[i - 1] > 0 >= y[i]) or (y[i - 1] < 0 <= y[i]):
            span = y[i] - y[i - 1]
            if span == 0:
                return x[i]
            return float(x[i - 1] + (0.0 - y[i - 1]) * (x[i] - x[i - 1]) / span)
    return None


def cost_sensitivity(
    panel: PricePanel,
    config: ExperimentConfig,
    strategy_name: str,
    benchmark_name: str,
    cost_grid_bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
) -> BreakEven:
    """What the tracking-error reduction costs, as a function of the cost assumption.

    The naive version of this analysis plots "tracking-error advantage versus
    transaction cost" and produces a flat line, because a cost charged on
    low turnover is a near-deterministic drag on the *level* of returns and
    barely touches their *dispersion*. That flat line is a real result but a
    useless chart, and mistaking it for "costs do not matter" is wrong: costs
    matter, they just show up somewhere else.

    So this measures the trade the way a practitioner states it -- how many basis
    points per year of extra trading am I paying for how many basis points of
    tracking-error reduction -- and calls the crossing point the break-even. That
    crossing compares a return drag to a risk reduction, which is a rule of thumb
    rather than a utility calculation, and is labelled as one wherever it appears.
    """
    years = None
    advantages: list[float] = []
    for bps in cost_grid_bps:
        model = BpsCostModel(bps)
        strategy = run_backtest(panel, build_strategy(strategy_name, config), config, model)
        benchmark = run_backtest(panel, build_strategy(benchmark_name, config), config, model)
        years = len(strategy.daily) / TRADING_DAYS
        te_advantage = (annualised_vol(benchmark.active) - annualised_vol(strategy.active)) * 1e4
        extra_cost = (strategy.daily["cost"].sum() - benchmark.daily["cost"].sum()) / years * 1e4
        advantages.append(float(te_advantage - extra_cost))
    return BreakEven(
        variable="cost_bps",
        values=list(cost_grid_bps),
        advantage=advantages,
        breakeven=_interpolate_zero(list(cost_grid_bps), advantages),
        baseline_advantage=advantages[0],
    )


def cost_decomposition(
    panel: PricePanel,
    config: ExperimentConfig,
    strategy_name: str,
    benchmark_name: str,
    cost_grid_bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
) -> pd.DataFrame:
    """Annual cost drag for both strategies across the cost grid, in basis points."""
    rows = []
    for bps in cost_grid_bps:
        model = BpsCostModel(bps)
        strategy = run_backtest(panel, build_strategy(strategy_name, config), config, model)
        benchmark = run_backtest(panel, build_strategy(benchmark_name, config), config, model)
        years = len(strategy.daily) / TRADING_DAYS
        rows.append(
            {
                "cost_bps": bps,
                strategy_name: float(strategy.daily["cost"].sum() / years * 1e4),
                benchmark_name: float(benchmark.daily["cost"].sum() / years * 1e4),
                "te_advantage_bp": float((annualised_vol(benchmark.active) - annualised_vol(strategy.active)) * 1e4),
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def capacity_curve(
    panel: PricePanel,
    config: ExperimentConfig,
    strategy_name: str,
    benchmark_name: str,
    notionals: tuple[float, ...] = (1e6, 1e7, 1e8, 1e9, 5e9, 2e10),
) -> BreakEven:
    """How the advantage decays with the size of the book, under square-root impact.

    A flat basis-point cost model cannot answer this at all -- it is invariant to
    size by construction. This is the practical version of the question "would
    you run this with real money, and how much?".
    """
    liquidity = LiquidityProfile.for_assets(panel.assets)
    advantages: list[float] = []
    for notional in notionals:
        model = SpreadImpactCostModel(liquidity, notional, config.impact_coef)
        strategy = run_backtest(panel, build_strategy(strategy_name, config), config, model)
        benchmark = run_backtest(panel, build_strategy(benchmark_name, config), config, model)
        advantages.append(annualised_vol(benchmark.active) - annualised_vol(strategy.active))
    return BreakEven(
        variable="portfolio_notional",
        values=list(notionals),
        advantage=advantages,
        breakeven=_interpolate_zero(list(notionals), advantages),
        baseline_advantage=advantages[0],
    )


def cost_drag_curve(
    panel: PricePanel,
    config: ExperimentConfig,
    strategy_name: str,
    notionals: tuple[float, ...] = (1e6, 1e7, 1e8, 1e9, 5e9, 2e10),
) -> pd.DataFrame:
    """Annual cost drag and peak ADV participation as a function of book size."""
    liquidity = LiquidityProfile.for_assets(panel.assets)
    rows = []
    for notional in notionals:
        model = SpreadImpactCostModel(liquidity, notional, config.impact_coef)
        result = run_backtest(panel, build_strategy(strategy_name, config), config, model)
        years = len(result.daily) / TRADING_DAYS
        rows.append(
            {
                "notional": notional,
                "cost_drag_bps_annual": float(result.daily["cost"].sum() / years * 1e4),
                "tracking_error": annualised_vol(result.active),
                "max_participation": float(result.rebalances["max_participation"].max()),
                "exceeds_participation_limit": bool(
                    result.rebalances["max_participation"].max() > config.max_participation
                ),
            }
        )
    return pd.DataFrame(rows).set_index("notional")


def sensitivity_curve(
    panel: PricePanel,
    config: ExperimentConfig,
    strategy_name: str,
    parameter: str,
    values: tuple[Any, ...],
    metric: Callable[[BacktestResult], float] = lambda r: annualised_vol(r.active),
) -> pd.DataFrame:
    """Sweep one configuration parameter and report the metric across it.

    Used for the stability plots. A result that only exists at one setting of
    ``train_days`` is a result about ``train_days``.
    """
    rows = []
    for value in values:
        candidate = config.with_changes(**{parameter: value})
        result = run_backtest(panel, build_strategy(strategy_name, candidate), candidate)
        rows.append(
            {
                parameter: value,
                "tracking_error": annualised_vol(result.active),
                "metric": metric(result),
                "annual_turnover": float(
                    result.rebalances["one_way_turnover"].sum() / (len(result.daily) / TRADING_DAYS)
                ),
                "active_return": annualised_return(result.net) - annualised_return(result.target),
            }
        )
    return pd.DataFrame(rows).set_index(parameter)
