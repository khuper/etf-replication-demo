"""Constrained tracking-error optimization."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import UNTAGGED_CLASS

logger = logging.getLogger(__name__)

WEIGHT_FLOOR = 1e-4


def historical_cvar(returns: np.ndarray, alpha: float) -> float:
    """Historical CVaR (expected shortfall) of a return series, as a positive loss."""
    k = max(int(alpha * len(returns)), 1)
    worst = np.sort(returns)[:k]
    return -float(np.mean(worst))


def _group_constraints(
    w: cp.Variable,
    columns: List[str],
    group_bounds: Dict[str, Tuple[float, float]],
    asset_classes: Dict[str, str],
) -> list:
    constraints = []
    for group, (lo, hi) in group_bounds.items():
        idx = [i for i, a in enumerate(columns) if asset_classes.get(a, UNTAGGED_CLASS) == group]
        if not idx:
            continue  # validated upstream; a zero-min bound on an empty group is a no-op
        group_weight = cp.sum(w[idx])
        constraints.append(group_weight >= lo)
        constraints.append(group_weight <= hi)
    return constraints


def optimize_tracking_error(
    asset_returns: pd.DataFrame,
    target_returns: pd.Series,
    cvar_ratio: float = 1.0,
    cvar_alpha: float = 0.05,
    w_prev: Optional[np.ndarray] = None,
    max_weight: float = 0.25,
    max_turnover: float = 0.20,
    group_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    asset_classes: Optional[Dict[str, str]] = None,
) -> Optional[pd.Series]:
    """
    Minimize squared tracking error versus the target subject to:

    - fully invested, long-only weights
    - per-position cap (``max_weight``)
    - portfolio CVaR no worse than ``cvar_ratio`` times the target's historical CVaR
    - L1 turnover versus ``w_prev`` capped at ``max_turnover`` (when ``w_prev`` given)
    - optional per-asset-class weight bounds (``group_bounds``)

    If the CVaR-constrained problem is infeasible, retries without the CVaR
    constraint (all other constraints kept). Returns the weights as a Series
    indexed by asset, or None if no problem could be solved.
    """
    R = asset_returns.values
    r_target = target_returns.values
    T, n_assets = R.shape

    w = cp.Variable(n_assets)
    v = cp.Variable()  # VaR level
    z = cp.Variable(T)  # CVaR auxiliary variables

    objective = cp.Minimize(cp.sum_squares(R @ w - r_target))

    base_constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    if w_prev is not None:
        base_constraints.append(cp.norm(w - w_prev, 1) <= max_turnover)
    if group_bounds:
        base_constraints += _group_constraints(w, list(asset_returns.columns), group_bounds, asset_classes or {})

    limit_cvar = historical_cvar(r_target, cvar_alpha) * cvar_ratio
    cvar_constraints = [
        z >= 0,
        z >= -(R @ w) - v,
        v + (1.0 / (T * cvar_alpha)) * cp.sum(z) <= limit_cvar,
    ]

    try:
        cp.Problem(objective, base_constraints + cvar_constraints).solve()
        if w.value is None:
            logger.warning("CVaR-constrained solve failed; retrying without the CVaR constraint.")
            cp.Problem(objective, base_constraints).solve()
    except cp.error.SolverError as exc:
        logger.error("CVXPY solver error: %s", exc)
        return None

    if w.value is None:
        logger.error("Optimization infeasible even without the CVaR constraint.")
        return None

    weights = np.where(w.value < WEIGHT_FLOOR, 0.0, w.value)
    total = weights.sum()
    if total <= 0:
        logger.error("Solver returned degenerate weights (sum <= 0).")
        return None
    return pd.Series(weights / total, index=asset_returns.columns, name="weight")
