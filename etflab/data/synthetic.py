"""A deterministic synthetic market whose replicating portfolio is known.

Why this exists
---------------
The default data source for this project is generated, not downloaded. That is a
deliberate design choice, not a limitation:

1. **The pipeline runs anywhere.** CI has no network and no vendor credentials,
   yet it executes the full research workflow on every commit. A research repo
   whose headline numbers cannot be reproduced by the reader is a screenshot.
2. **Ground truth exists.** The target is constructed as ``w_true . r_assets``
   plus a component that no candidate asset spans. So the population-optimal
   feasible portfolio *is* ``w_true``, the minimum attainable tracking error *is*
   the volatility of the unspanned component, and an estimator can therefore be
   scored on whether it recovers the right answer -- not merely on whether it
   fits. Real data cannot do this, because on real data nobody knows the answer.

The generator is not trying to be a market simulator. It is trying to be a market
that is *hard in the ways that matter*: fat tails, volatility clustering,
regime-dependent correlation, a flight-to-quality asset, and an irreducible
residual. Everything is seeded; the same seed gives byte-identical prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from etflab.data.panel import MarketTruth, PricePanel

TRADING_DAYS = 252

#: Latent risk factors. ``ILLIQUIDITY`` is special: no candidate asset loads on
#: it, so it is the part of the target that is structurally unreplicable.
FACTORS = ("EQUITY", "SIZE", "INTL", "CREDIT", "DURATION", "COMMODITY", "ILLIQUIDITY")

#: Annualised factor volatilities.
FACTOR_VOL = {
    "EQUITY": 0.16,
    "SIZE": 0.08,
    "INTL": 0.09,
    "CREDIT": 0.06,
    "DURATION": 0.06,
    "COMMODITY": 0.14,
    "ILLIQUIDITY": 0.040,
}

#: Annualised factor drifts. Small, and small on purpose: this project is about
#: second moments (tracking), so large first moments would only add noise to the
#: comparison between estimators.
FACTOR_DRIFT = {
    "EQUITY": 0.065,
    "SIZE": 0.010,
    "INTL": 0.005,
    "CREDIT": 0.020,
    "DURATION": 0.012,
    "COMMODITY": 0.020,
    "ILLIQUIDITY": 0.035,
}

#: How much each factor's volatility is amplified in a crisis. Duration is the
#: flight-to-quality leg: it gets a *positive* crisis drift instead of a big
#: volatility multiplier, which is what breaks naive constant-correlation models.
CRISIS_VOL_MULTIPLIER = {
    "EQUITY": 2.9,
    "SIZE": 2.6,
    "INTL": 2.7,
    "CREDIT": 3.1,
    "DURATION": 1.5,
    "COMMODITY": 2.0,
    "ILLIQUIDITY": 2.6,
}

CRISIS_DRIFT_ANNUAL = {
    "EQUITY": -0.55,
    "SIZE": -0.25,
    "INTL": -0.30,
    "CREDIT": -0.35,
    "DURATION": 0.22,
    "COMMODITY": -0.10,
    "ILLIQUIDITY": -0.60,
}

#: Factor exposures of the candidate universe. Chosen to look like the real
#: instruments the tickers name, including the collinearity that makes the
#: optimisation problem ill-conditioned (SPY/QQQ/IWM all load on EQUITY).
LOADINGS: dict[str, dict[str, float]] = {
    "SPY": {"EQUITY": 1.00},
    "QQQ": {"EQUITY": 1.08, "SIZE": -0.18},
    "IWM": {"EQUITY": 1.00, "SIZE": 1.00},
    "VEA": {"EQUITY": 0.86, "INTL": 1.00},
    "VWO": {"EQUITY": 0.88, "INTL": 1.10, "COMMODITY": 0.16},
    "HYG": {"EQUITY": 0.34, "CREDIT": 1.00, "DURATION": 0.22},
    "LQD": {"CREDIT": 0.36, "DURATION": 0.88},
    "TIP": {"DURATION": 0.62, "COMMODITY": 0.12},
    "GLD": {"COMMODITY": 1.00, "DURATION": 0.14},
    "VNQ": {"EQUITY": 0.84, "SIZE": 0.30, "DURATION": 0.45},
    "AGG": {"CREDIT": 0.18, "DURATION": 0.80},
    "TLT": {"DURATION": 1.85},
    "EFA": {"EQUITY": 0.88, "INTL": 0.95},
}

#: Annualised idiosyncratic volatility per asset.
IDIO_VOL: dict[str, float] = {
    "SPY": 0.020,
    "QQQ": 0.040,
    "IWM": 0.048,
    "VEA": 0.038,
    "VWO": 0.060,
    "HYG": 0.030,
    "LQD": 0.022,
    "TIP": 0.020,
    "GLD": 0.075,
    "VNQ": 0.055,
    "AGG": 0.016,
    "TLT": 0.035,
    "EFA": 0.036,
}

#: The portfolio the target is actually built from. Long-only, fully invested,
#: and inside a 25% position cap, so it is a feasible point of the optimisation
#: problem -- which is precisely what makes recovery testable.
TRUE_WEIGHTS: dict[str, float] = {
    "SPY": 0.22,
    "QQQ": 0.18,
    "IWM": 0.20,
    "VEA": 0.10,
    "VWO": 0.05,
    "HYG": 0.15,
    "LQD": 0.02,
    "TIP": 0.00,
    "GLD": 0.03,
    "VNQ": 0.05,
}

BASE_PRICE: dict[str, float] = {
    "SPY": 145.0,
    "QQQ": 67.0,
    "IWM": 84.0,
    "VEA": 36.0,
    "VWO": 44.0,
    "HYG": 92.0,
    "LQD": 118.0,
    "TIP": 114.0,
    "GLD": 162.0,
    "VNQ": 63.0,
    "AGG": 108.0,
    "TLT": 120.0,
    "EFA": 58.0,
    "PSP": 12.0,
}

# Regime machinery -------------------------------------------------------------
REGIME_NAMES = ("calm", "stressed", "crisis")

#: Row-stochastic transition matrix over (calm, stressed, crisis). Expected
#: sojourn times are ~1/(1-p_ii) days: roughly 3 months calm, 5 weeks stressed,
#: 2 weeks crisis, which produces a handful of distinct drawdowns per decade.
TRANSITION = np.array(
    [
        [0.9860, 0.0132, 0.0008],
        [0.0420, 0.9430, 0.0150],
        [0.0100, 0.1000, 0.8900],
    ]
)

REGIME_VOL_MULTIPLIER = {"calm": 0.88, "stressed": 1.45, "crisis": 1.0}


@dataclass(frozen=True)
class SyntheticSpec:
    """Knobs for :func:`generate_market`. Defaults are the shipped experiment."""

    assets: tuple[str, ...]
    target: str
    start: str
    end: str
    seed: int = 20_240_101
    leverage: float = 1.0
    target_idio_vol: float = 0.012
    tail_df: float = 5.0
    garch_alpha: float = 0.08
    garch_beta: float = 0.86


def _standardised_t(rng: np.random.Generator, df: float, size: tuple[int, ...]) -> np.ndarray:
    """Student-t draws rescaled to unit variance.

    Fat tails without silently inflating volatility: a raw t(5) has variance
    df/(df-2) = 1.67, so an unscaled draw would make every stated volatility a lie.
    """
    if df <= 2:
        raise ValueError("tail_df must exceed 2 for a finite variance.")
    raw = rng.standard_t(df, size=size)
    return raw / np.sqrt(df / (df - 2.0))


def _simulate_regimes(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample a Markov regime path, starting from the chain's stationary state."""
    eigvals, eigvecs = np.linalg.eig(TRANSITION.T)
    stationary = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
    stationary = stationary / stationary.sum()
    states = np.empty(n, dtype=np.int8)
    current = int(rng.choice(len(REGIME_NAMES), p=stationary))
    for t in range(n):
        states[t] = current
        current = int(rng.choice(len(REGIME_NAMES), p=TRANSITION[current]))
    return states


def _garch_path(
    rng: np.random.Generator,
    n: int,
    target_vol_daily: float,
    alpha: float,
    beta: float,
    df: float,
) -> np.ndarray:
    """GARCH(1,1) innovations with fat-tailed shocks, calibrated to ``target_vol_daily``.

    ``omega`` is pinned so the unconditional variance equals the requested one;
    without that the persistence parameters would silently set the volatility level.
    """
    if alpha + beta >= 1:
        raise ValueError("GARCH(1,1) requires alpha + beta < 1 for stationarity.")
    var_uncond = target_vol_daily**2
    omega = var_uncond * (1.0 - alpha - beta)
    z = _standardised_t(rng, df, (n,))
    out = np.empty(n)
    sigma2 = var_uncond
    for t in range(n):
        sigma = np.sqrt(sigma2)
        out[t] = sigma * z[t]
        sigma2 = omega + alpha * out[t] ** 2 + beta * sigma2
    return out


def generate_market(spec: SyntheticSpec) -> PricePanel:
    """Generate a full synthetic market and its ground truth.

    The construction is, per day ``t``:

    ``r_asset[i,t] = drift_i + sum_k L[i,k] * f[k,t] + e[i,t]``
    ``r_target[t]  = leverage * (w_true . r_asset[:,t]) + f[ILLIQUIDITY,t] + u[t]``

    ``ILLIQUIDITY`` has zero loading for every candidate asset, so the target's
    exposure to it -- along with ``u`` -- is unreplicable by construction. With
    ``leverage == 1`` the population-optimal long-only fully-invested portfolio is
    exactly ``w_true``, and the minimum attainable annualised tracking error is
    ``sqrt(var(f_ILLIQ) + var(u)) * sqrt(252)``.
    """
    unknown = [a for a in (*spec.assets, spec.target) if a not in BASE_PRICE]
    if unknown:
        raise ValueError(
            f"The synthetic market has no definition for {unknown}. "
            f"Known tickers: {sorted(BASE_PRICE)}. Use --data-source yfinance or csv for others."
        )
    missing_loadings = [a for a in spec.assets if a not in LOADINGS]
    if missing_loadings:
        raise ValueError(f"No factor loadings defined for {missing_loadings}.")

    dates = pd.bdate_range(spec.start, spec.end)
    if len(dates) < 260:
        raise ValueError(f"Synthetic market needs at least ~1 year of business days; got {len(dates)}.")
    n = len(dates)

    # Independent, reproducible substreams: adding an asset must not change the
    # factor path, and changing the factor model must not reshuffle idio noise.
    seeds = np.random.SeedSequence(spec.seed).spawn(4)
    regime_rng = np.random.default_rng(seeds[0])
    factor_rng = np.random.default_rng(seeds[1])
    idio_rng = np.random.default_rng(seeds[2])
    target_rng = np.random.default_rng(seeds[3])

    regime_idx = _simulate_regimes(regime_rng, n)
    regimes = pd.Series([REGIME_NAMES[i] for i in regime_idx], index=dates, name="regime")

    # --- factors ----------------------------------------------------------
    factor_returns = pd.DataFrame(index=dates, columns=list(FACTORS), dtype=float)
    for factor in FACTORS:
        base_daily = FACTOR_VOL[factor] / np.sqrt(TRADING_DAYS)
        shocks = _garch_path(factor_rng, n, base_daily, spec.garch_alpha, spec.garch_beta, spec.tail_df)
        multiplier = np.where(
            regime_idx == 2,
            CRISIS_VOL_MULTIPLIER[factor],
            np.where(regime_idx == 1, REGIME_VOL_MULTIPLIER["stressed"], REGIME_VOL_MULTIPLIER["calm"]),
        )
        drift = np.where(
            regime_idx == 2,
            CRISIS_DRIFT_ANNUAL[factor] / TRADING_DAYS,
            FACTOR_DRIFT[factor] / TRADING_DAYS,
        )
        factor_returns[factor] = drift + shocks * multiplier

    # --- assets -----------------------------------------------------------
    loadings = pd.DataFrame(0.0, index=list(spec.assets), columns=list(FACTORS))
    for asset in spec.assets:
        for factor, beta in LOADINGS[asset].items():
            loadings.loc[asset, factor] = beta
    # ILLIQUIDITY is unspanned by construction; assert it rather than trust it.
    if loadings["ILLIQUIDITY"].abs().sum() > 0:
        raise AssertionError("ILLIQUIDITY must have zero loading on every candidate asset.")

    systematic = factor_returns.to_numpy() @ loadings.to_numpy().T  # (n, n_assets)
    idio_vol_daily = np.array([IDIO_VOL[a] for a in spec.assets]) / np.sqrt(TRADING_DAYS)
    idio = _standardised_t(idio_rng, spec.tail_df, (n, len(spec.assets))) * idio_vol_daily
    asset_returns = pd.DataFrame(systematic + idio, index=dates, columns=list(spec.assets))

    # --- target -----------------------------------------------------------
    w_true = pd.Series({a: TRUE_WEIGHTS.get(a, 0.0) for a in spec.assets}, dtype=float)
    if w_true.sum() <= 0:
        raise ValueError("No ground-truth weights are defined for the requested asset list.")
    w_true = w_true / w_true.sum()  # renormalise if the caller used a subset

    spanned = asset_returns.to_numpy() @ w_true.to_numpy()
    unspanned_factor = factor_returns["ILLIQUIDITY"].to_numpy()
    target_idio = _standardised_t(target_rng, spec.tail_df, (n,)) * (spec.target_idio_vol / np.sqrt(TRADING_DAYS))
    unspanned = unspanned_factor + target_idio
    target_returns = spec.leverage * spanned + unspanned

    returns = asset_returns.copy()
    returns[spec.target] = target_returns

    # --- prices -----------------------------------------------------------
    prices = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
    for column in returns.columns:
        prices[column] = BASE_PRICE[column] * (1.0 + returns[column]).cumprod()
    # Prepend the base price so that pct_change() reproduces `returns` exactly and
    # no observation is lost to differencing.
    first_day = dates[0] - pd.tseries.offsets.BDay(1)
    seed_row = pd.DataFrame([[BASE_PRICE[c] for c in returns.columns]], index=[first_day], columns=returns.columns)
    prices = pd.concat([seed_row, prices])
    prices.index.name = "date"

    truth = MarketTruth(
        true_weights=w_true,
        factor_loadings=loadings,
        factor_returns=factor_returns,
        unspanned_returns=pd.Series(unspanned, index=dates, name="unspanned"),
        regimes=regimes,
        irreducible_te_annual=float(np.std(unspanned, ddof=1) * np.sqrt(TRADING_DAYS)),
        leverage=spec.leverage,
    )
    return PricePanel(
        prices=prices[[*spec.assets, spec.target]],
        assets=tuple(spec.assets),
        target=spec.target,
        source="synthetic",
        meta={
            "seed": spec.seed,
            "leverage": spec.leverage,
            "tail_df": spec.tail_df,
            "generator": "etflab.data.synthetic.generate_market",
        },
        truth=truth,
    )


def corrupt(
    panel: PricePanel,
    *,
    seed: int = 99,
    stale_runs: int = 3,
    stale_length: int = 6,
    missing_block: int = 4,
    split_jump: bool = True,
    negative_price: bool = False,
) -> PricePanel:
    """Return a deliberately damaged copy of ``panel``.

    Test fixture for the data-quality gates: a gate that has never been shown a
    broken panel is decoration, not a control.
    """
    rng = np.random.default_rng(seed)
    prices = panel.prices.copy()
    columns = list(prices.columns)

    for _ in range(stale_runs):
        col = columns[int(rng.integers(0, len(columns)))]
        start = int(rng.integers(20, max(21, len(prices) - stale_length - 1)))
        prices.iloc[start : start + stale_length, prices.columns.get_loc(col)] = prices.iloc[start - 1][col]

    if missing_block > 0:
        col = columns[int(rng.integers(0, len(columns)))]
        start = int(rng.integers(20, max(21, len(prices) - missing_block - 1)))
        prices.iloc[start : start + missing_block, prices.columns.get_loc(col)] = np.nan

    if split_jump:
        col = columns[0]
        cut = len(prices) // 2
        prices.iloc[cut:, prices.columns.get_loc(col)] = prices.iloc[cut:][col] / 2.0

    if negative_price:
        prices.iloc[10, 0] = -1.0

    return PricePanel(prices, panel.assets, panel.target, "synthetic-corrupt", {**panel.meta, "corrupt_seed": seed})
