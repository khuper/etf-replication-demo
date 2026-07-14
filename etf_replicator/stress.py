"""Beta-based regime-shift stress test."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def beta_stress_test(
    asset_returns: pd.DataFrame,
    target_returns: pd.Series,
    weights: pd.Series,
    shock: float = -0.20,
) -> Dict[str, float]:
    """
    Estimate how the basket responds if the target instantly moves by
    ``shock``, using each asset's beta to the target.
    """
    combined = pd.concat([asset_returns, target_returns], axis=1)
    cov = np.cov(combined.values, rowvar=False, ddof=1)
    target_var = cov[-1, -1]
    betas = pd.Series(cov[:-1, -1] / target_var, index=asset_returns.columns)
    portfolio_beta = float((weights * betas).sum())

    return {
        "shock": shock,
        "portfolio_beta": portfolio_beta,
        "portfolio_impact": portfolio_beta * shock,
        "target_impact": shock,
        "relative_performance": portfolio_beta * shock - shock,
    }
