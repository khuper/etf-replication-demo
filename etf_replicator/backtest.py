"""Expanding-window backtest with turnover-aware transaction costs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import split_assets_target
from .optimizer import optimize_tracking_error

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Out-of-sample results of an expanding-window backtest."""

    weights: pd.DataFrame  # target weights at each rebalance date
    turnover: pd.Series  # L1 turnover executed at each rebalance
    gross_returns: pd.Series  # daily out-of-sample portfolio returns
    net_returns: pd.Series  # gross minus transaction costs
    target_returns: pd.Series  # target returns over the same dates
    cost_bps: float

    @property
    def final_weights(self) -> pd.Series:
        return self.weights.iloc[-1]

    @property
    def active_returns(self) -> pd.Series:
        return self.net_returns - self.target_returns

    def net_at(self, cost_bps: float) -> pd.Series:
        """Net return stream under an alternative cost assumption."""
        net = self.gross_returns.copy()
        net.loc[self.turnover.index] -= self.turnover * (cost_bps / 1e4)
        return net


def drift_weights(weights: np.ndarray, period_returns: pd.DataFrame) -> np.ndarray:
    """Let weights drift with compounded asset returns, then renormalize."""
    compounded = (1 + period_returns).prod().values
    drifted = weights * compounded
    total = drifted.sum()
    return drifted / total if total > 0 else weights


def run_expanding_backtest(
    returns: pd.DataFrame,
    assets: List[str],
    target: str,
    initial_train_size: int = 504,
    step: int = 126,
    max_weight: float = 0.25,
    max_turnover: float = 0.20,
    cvar_ratio: float = 1.0,
    cvar_alpha: float = 0.05,
    cost_bps: float = 10.0,
    group_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    asset_classes: Optional[Dict[str, str]] = None,
) -> BacktestResult:
    """
    Re-optimize on an expanding window every ``step`` days and hold each
    allocation until the next rebalance (weights are applied at the rebalance
    target each day; turnover between rebalances is measured against
    drift-adjusted previous weights, so the turnover constraint reflects the
    trades actually needed).

    Transaction costs of ``cost_bps`` per unit of L1 turnover are charged on
    each rebalance day. The first rebalance is charged as a full deployment
    from cash (turnover of 1).
    """
    n_obs = len(returns)
    if initial_train_size >= n_obs:
        raise ValueError(
            f"initial_train_size={initial_train_size} needs more history than the {n_obs} available observations."
        )

    asset_returns, target_returns = split_assets_target(returns, assets, target)
    logger.info(
        "Expanding-window backtest: initial=%d, step=%d, max_weight=%.0f%%, max_turnover=%.0f%%, cost=%.1f bps",
        initial_train_size, step, max_weight * 100, max_turnover * 100, cost_bps,
    )

    rebalance_rows: List[pd.Series] = []
    rebalance_dates: List[pd.Timestamp] = []
    rebalance_idx: List[int] = []
    turnovers: List[float] = []
    w_prev: Optional[np.ndarray] = None
    prev_i: Optional[int] = None

    for i in range(initial_train_size, n_obs, step):
        train = returns.iloc[:i]
        asset_train, target_train = split_assets_target(train, assets, target)

        if w_prev is not None and prev_i is not None:
            w_drifted = drift_weights(w_prev, asset_returns.iloc[prev_i:i])
        else:
            w_drifted = None

        weights = optimize_tracking_error(
            asset_train,
            target_train,
            cvar_ratio=cvar_ratio,
            cvar_alpha=cvar_alpha,
            w_prev=w_drifted,
            max_weight=max_weight,
            max_turnover=max_turnover,
            group_bounds=group_bounds,
            asset_classes=asset_classes,
        )
        if weights is None:
            logger.warning("Optimization failed at %s; holding previous weights.", returns.index[i].date())
            continue

        turnover = 1.0 if w_drifted is None else float(np.abs(weights.values - w_drifted).sum())
        rebalance_rows.append(weights)
        rebalance_dates.append(returns.index[i])
        rebalance_idx.append(i)
        turnovers.append(turnover)
        w_prev = weights.values
        prev_i = i
        logger.info("Rebalanced at %s (turnover %.1f%%)", returns.index[i].date(), turnover * 100)

    if not rebalance_rows:
        raise RuntimeError("No rebalance succeeded; backtest produced no weights.")

    weights_history = pd.DataFrame(rebalance_rows, index=pd.DatetimeIndex(rebalance_dates))
    turnover_series = pd.Series(turnovers, index=weights_history.index, name="turnover")

    # Stitch out-of-sample returns: period k runs from rebalance k up to (not
    # including) rebalance k+1; the last period runs to the end of the data.
    period_returns: List[pd.Series] = []
    boundaries = rebalance_idx + [n_obs]
    for k in range(len(rebalance_idx)):
        window = asset_returns.iloc[boundaries[k]:boundaries[k + 1]]
        period_returns.append((window * weights_history.iloc[k]).sum(axis=1))
    gross = pd.concat(period_returns)
    gross.name = "gross"

    net = gross.copy()
    net.name = "net"
    costs = turnover_series * (cost_bps / 1e4)
    net.loc[costs.index] -= costs

    oos_target = target_returns.loc[gross.index]

    return BacktestResult(
        weights=weights_history,
        turnover=turnover_series,
        gross_returns=gross,
        net_returns=net,
        target_returns=oos_target,
        cost_bps=cost_bps,
    )
