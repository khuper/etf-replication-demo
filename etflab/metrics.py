"""Out-of-sample performance and risk metrics.

Every number here is computed from the walk-forward ledger only. There is no
in-sample metric anywhere in this module, because an in-sample tracking error is
a statement about the optimiser's arithmetic, not about the world.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return. NaN on an empty series rather than 0.0."""
    if returns.empty:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return float("nan")
    return float(growth ** (TRADING_DAYS / len(returns)) - 1.0)


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def cvar(series: pd.Series, alpha: float = 0.05) -> float:
    """Mean of the worst ``alpha`` tail. Returned as a negative number for losses."""
    if series.empty:
        return float("nan")
    cutoff = series.quantile(alpha)
    tail = series[series <= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


def up_down_capture(portfolio: pd.Series, target: pd.Series) -> tuple[float, float]:
    """Share of the target's up days and down days that the replicator captured."""
    up, down = target > 0, target < 0
    up_capture = (
        float(portfolio[up].mean() / target[up].mean()) if up.any() and target[up].mean() != 0 else float("nan")
    )
    down_capture = (
        float(portfolio[down].mean() / target[down].mean()) if down.any() and target[down].mean() != 0 else float("nan")
    )
    return up_capture, down_capture


def rolling_tracking_error(active: pd.Series, window: int = 126) -> pd.Series:
    """Rolling annualised tracking error -- the honest picture behind one number.

    A single full-sample tracking error averages a calm 2017 with a March 2020,
    and averaging those tells you about neither.
    """
    return active.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def compute_metrics(
    daily: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    rebalances: pd.DataFrame | None = None,
) -> dict[str, float]:
    """The full out-of-sample metric set for one strategy."""
    net, target, active = daily["net"], daily["target"], daily["active"]
    gross_active = daily["gross"] - target
    years = len(daily) / TRADING_DAYS

    te = annualised_vol(active)
    downside = active[active < 0]
    up_capture, down_capture = up_down_capture(net, target)

    metrics: dict[str, float] = {
        # --- replication quality ---
        "tracking_error": te,
        "tracking_error_gross": annualised_vol(gross_active),
        "downside_tracking_error": float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(downside) > 1
        else float("nan"),
        "correlation": float(net.corr(target)),
        "beta": float(np.cov(net, target, ddof=1)[0, 1] / np.var(target, ddof=1)) if target.var() > 0 else float("nan"),
        "r_squared": float(net.corr(target) ** 2),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if active.std(ddof=1) > 0
        else float("nan"),
        "active_return": annualised_return(net) - annualised_return(target),
        # --- level performance ---
        "replicator_return": annualised_return(net),
        "target_return": annualised_return(target),
        "replicator_vol": annualised_vol(net),
        "target_vol": annualised_vol(target),
        "replicator_max_drawdown": max_drawdown(net),
        "target_max_drawdown": max_drawdown(target),
        # --- tails and asymmetry ---
        "active_cvar_5pct": cvar(active, 0.05),
        "worst_day_active": float(active.min()),
        "worst_21d_active": float(active.rolling(21).sum().min()),
        "hit_rate": float((active > 0).mean()),
        "up_capture": up_capture,
        "down_capture": down_capture,
        # --- costs ---
        "cost_drag_annual": float(daily["cost"].sum() / years),
        "total_cost": float(daily["cost"].sum()),
        "n_days": float(len(daily)),
        "years": float(years),
    }

    if rebalances is not None and not rebalances.empty:
        # The first rebalance is the initial deployment out of cash: a one-off
        # that is not part of the ongoing trading rate, so including it would
        # overstate the running turnover of a short backtest and understate it
        # for a long one. It is reported separately instead.
        ongoing = rebalances.iloc[1:] if len(rebalances) > 1 else rebalances.iloc[:0]
        metrics.update(
            {
                "avg_one_way_turnover": float(ongoing["one_way_turnover"].mean()) if len(ongoing) else 0.0,
                "annual_turnover": float(ongoing["one_way_turnover"].sum() / years) if len(ongoing) else 0.0,
                "initial_deployment_notional": float(rebalances["traded_notional"].iloc[0]),
                "annual_traded_notional": float(ongoing["traded_notional"].sum() / years) if len(ongoing) else 0.0,
                "max_participation": float(rebalances["max_participation"].max()),
                "n_rebalances": float(len(rebalances)),
                "degraded_rebalances": float(
                    (~rebalances["status"].isin(["optimal", "optimal_inaccurate", "analytic"])).sum()
                ),
            }
        )
    if weights is not None and not weights.empty:
        final = weights.iloc[-1]
        hhi = float((weights**2).sum(axis=1).mean())
        metrics.update(
            {
                "effective_positions": 1.0 / hhi if hhi > 0 else float("nan"),
                "max_position": float(weights.max().max()),
                "final_max_position": float(final.max()),
                "weight_stability": float(weights.diff().abs().sum(axis=1).mean()),
            }
        )
    return metrics


def recovery_metrics(weights: pd.DataFrame, true_weights: pd.Series) -> dict[str, float]:
    """How close the estimated weights got to the ones the market was built from.

    Only computable on synthetic data, and the reason synthetic data is here: on
    real prices this table cannot exist, so nobody can tell you whether an
    estimator is finding structure or fitting noise -- only whether it fit well.
    """
    aligned = weights.reindex(columns=true_weights.index).fillna(0.0)
    truth = true_weights.to_numpy(dtype=float)
    errors = aligned.to_numpy(dtype=float) - truth
    l1 = np.abs(errors).sum(axis=1)
    return {
        "recovery_l1_mean": float(l1.mean()),
        "recovery_l1_final": float(l1[-1]),
        "recovery_l1_first": float(l1[0]),
        "recovery_l2_mean": float(np.sqrt((errors**2).sum(axis=1)).mean()),
        "recovery_max_abs_error": float(np.abs(errors).max()),
        "recovery_improvement": float(l1[0] - l1[-1]),
    }


def summary_frame(
    metrics_by_strategy: dict[str, dict[str, float]], keys: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Tidy comparison table, strategies as rows."""
    frame = pd.DataFrame(metrics_by_strategy).T
    if keys:
        frame = frame[[k for k in keys if k in frame.columns]]
    return frame


def regime_breakdown(daily: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Performance conditioned on the market regime.

    An average tracking error is a promise about a market that never happens. The
    number that matters is the one in the regime you are afraid of.
    """
    aligned = regimes.reindex(daily.index).ffill()
    rows = []
    for regime, group in daily.groupby(aligned):
        if len(group) < 2:
            continue
        rows.append(
            {
                "regime": regime,
                "days": len(group),
                "share": len(group) / len(daily),
                "tracking_error": annualised_vol(group["active"]),
                "active_return": annualised_return(group["net"]) - annualised_return(group["target"]),
                "target_return": annualised_return(group["target"]),
                "worst_day_active": float(group["active"].min()),
                "hit_rate": float((group["active"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


def attribution_by_asset(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_returns: pd.Series,
) -> pd.DataFrame:
    """Euler decomposition of active risk, including the target's short leg.

    Active return is a portfolio: long the basket, short one unit of the target.
    Decomposing it means treating the target as a position with weight ``-1``,
    which is the step that is usually skipped -- and skipping it produces
    contributions that do not sum to the tracking error, so nobody can check them.

    Here they do sum to it, exactly, and ``tests/test_metrics.py`` asserts it.
    Read the table as: the target leg is the risk you took on by having a
    liability at all, and each asset's negative contribution is how much of it
    that position hedges away.
    """
    held = weights.reindex(asset_returns.index).ffill().dropna(how="all")
    common = held.index.intersection(target_returns.index)
    held = held.loc[common]
    assets = asset_returns.loc[common, held.columns]
    target = target_returns.loc[common]

    portfolio = (assets * held).sum(axis=1)
    active = portfolio - target
    sigma = float(active.std(ddof=1))
    if sigma <= 0:
        return pd.DataFrame()
    scale = np.sqrt(TRADING_DAYS)

    rows = []
    for column in held.columns:
        avg_weight = float(held[column].mean())
        # The realised weighted return series, so time-varying weights are handled
        # exactly rather than through an average-weight approximation.
        weighted = assets[column] * held[column]
        contribution = float(np.cov(weighted, active, ddof=1)[0, 1]) / sigma
        rows.append(
            {
                "leg": column,
                "avg_weight": avg_weight,
                "risk_contribution": contribution * scale,
                "return_contribution": float(weighted.mean() * TRADING_DAYS),
                "corr_with_target": float(assets[column].corr(target)),
            }
        )
    target_contribution = float(np.cov(-target, active, ddof=1)[0, 1]) / sigma
    rows.append(
        {
            "leg": f"{target_returns.name} (short)",
            "avg_weight": -1.0,
            "risk_contribution": target_contribution * scale,
            "return_contribution": float(-target.mean() * TRADING_DAYS),
            "corr_with_target": 1.0,
        }
    )
    frame = pd.DataFrame(rows).set_index("leg")
    frame["risk_share"] = frame["risk_contribution"] / (sigma * scale)
    return frame.sort_values("risk_contribution", ascending=False)


def describe_metrics() -> dict[str, str]:
    """Human-readable definitions, exported alongside results so a reader never
    has to guess whether ``tracking_error`` was annualised or which sign
    convention ``down_capture`` uses."""
    return {
        "tracking_error": "Annualised standard deviation of daily active return (replicator net of costs minus target).",
        "tracking_error_gross": "Same, before transaction costs. The gap to tracking_error is the cost drag.",
        "downside_tracking_error": "Annualised standard deviation of negative active returns only.",
        "information_ratio": "Annualised mean active return divided by tracking error. Sign matters: negative means the replicator lagged.",
        "beta": "Slope of net replicator returns on target returns. 1.0 means matched sensitivity.",
        "active_cvar_5pct": "Mean active return on the worst 5% of days.",
        "up_capture": "Mean replicator return on the target's up days, divided by the target's mean up-day return.",
        "down_capture": "Same for down days. Below 1.0 means the replicator fell less than the target.",
        "cost_drag_annual": "Transaction costs charged per year, as a fraction of portfolio value.",
        "annual_turnover": "One-way turnover per year. 0.30 means 30% of the book is traded annually.",
        "effective_positions": "1 / average Herfindahl index of weights. A concentration measure, not a count.",
        "weight_stability": "Mean L1 distance between consecutive target weight vectors. Lower is more stable.",
        "recovery_l1_mean": "Mean L1 distance between estimated weights and the synthetic market's true weights.",
        "degraded_rebalances": "Rebalances where the optimiser failed and the engine fell back to prior holdings.",
    }
