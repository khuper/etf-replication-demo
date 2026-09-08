"""Statistical inference for backtest comparisons.

The gap between two backtested tracking errors is an estimate, and estimates have
standard errors. This module is what turns "5.73% beats 5.75%" into a statement
somebody can disagree with on the evidence.

Three problems are handled explicitly, because ignoring any one of them is how
backtests come to be believed:

1. **Serial dependence.** Daily active returns are autocorrelated and
   heteroskedastic, so an i.i.d. bootstrap or a plain t-test understates
   uncertainty. Everything here uses the stationary bootstrap of Politis & Romano
   (1994) or Newey-West HAC standard errors.
2. **Multiple comparisons.** Run nine strategies and the best one beats the
   benchmark by construction. White's Reality Check and Hansen's SPA test give a
   p-value for "the best of these nine", not for a single pre-chosen one.
3. **Selection over parameters.** A Sharpe (or information) ratio chosen as the
   best of many configurations is biased upward by a known amount. The Deflated
   Sharpe Ratio (Bailey & Lopez de Prado, 2014) prices that in.

References are given per function so a reader can check the implementation
against the source rather than trusting the docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    n_obs: int,
    n_samples: int,
    mean_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Index matrix ``(n_samples, n_obs)`` for the stationary bootstrap.

    Blocks have geometrically distributed lengths with mean ``mean_block`` and
    wrap around the end of the sample, which keeps the resampled series
    stationary -- unlike a fixed-block bootstrap, whose block boundaries are a
    deterministic function of position.

    Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap",
    *JASA* 89(428), 1303-1313.
    """
    if n_obs < 2:
        raise ValueError("Bootstrap needs at least two observations.")
    if mean_block <= 0:
        raise ValueError("mean_block must be positive.")
    p = min(1.0, 1.0 / mean_block)

    new_block = rng.random((n_samples, n_obs)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n_obs, size=(n_samples, n_obs))

    positions = np.arange(n_obs)[None, :]
    block_start_pos = np.maximum.accumulate(np.where(new_block, positions, -1), axis=1)
    base = np.take_along_axis(np.where(new_block, starts, 0), block_start_pos, axis=1)
    return (base + (positions - block_start_pos)) % n_obs


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    level: float
    n_samples: int
    std_error: float

    @property
    def excludes_zero(self) -> bool:
        return (self.lower > 0) or (self.upper < 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "std_error": self.std_error,
            "excludes_zero": self.excludes_zero,
            "n_samples": self.n_samples,
        }


def bootstrap_statistic(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_samples: int = 2000,
    mean_block: float = 21.0,
    level: float = 0.95,
    seed: int = 7,
) -> BootstrapCI:
    """Percentile confidence interval for ``statistic`` under the stationary bootstrap.

    ``data`` may be 1-D or ``(n_obs, k)``; rows are resampled jointly, which is
    what preserves the contemporaneous dependence between two strategies being
    compared. Resampling them independently would inflate the variance of their
    difference and hide real differences.
    """
    array = np.asarray(data, dtype=float)
    matrix = array[:, None] if array.ndim == 1 else array
    n_obs = matrix.shape[0]
    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(n_obs, n_samples, mean_block, rng)

    draws = np.empty(n_samples)
    for b in range(n_samples):
        resampled = matrix[indices[b]]
        draws[b] = statistic(resampled[:, 0] if array.ndim == 1 else resampled)

    alpha = (1.0 - level) / 2.0
    return BootstrapCI(
        point=float(statistic(array)),
        lower=float(np.quantile(draws, alpha)),
        upper=float(np.quantile(draws, 1.0 - alpha)),
        level=level,
        n_samples=n_samples,
        std_error=float(draws.std(ddof=1)),
    )


# --------------------------------------------------------------------------- #
# HAC variance and the Diebold-Mariano test
# --------------------------------------------------------------------------- #
def newey_west_variance(x: np.ndarray, lag: int | None = None) -> float:
    """Long-run variance of the mean of ``x`` with a Bartlett kernel.

    ``lag=None`` uses the Newey-West (1994) automatic rule
    ``floor(4 * (T/100)^(2/9))``, which is the default in most econometrics
    packages and avoids the temptation to tune the bandwidth until the answer
    comes out right.
    """
    series = np.asarray(x, dtype=float)
    n = series.size
    if n < 2:
        return float("nan")
    if lag is None:
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 1))

    centred = series - series.mean()
    variance = float(centred @ centred) / n
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        covariance = float(centred[k:] @ centred[:-k]) / n
        variance += 2.0 * weight * covariance
    return max(variance, 1e-300)


@dataclass(frozen=True)
class DieboldMarianoResult:
    statistic: float
    p_value: float
    mean_loss_difference: float
    lag: int
    n_obs: int
    favours: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "mean_loss_difference": self.mean_loss_difference,
            "lag": self.lag,
            "n_obs": self.n_obs,
            "favours": self.favours,
        }


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    lag: int | None = None,
    label_a: str = "a",
    label_b: str = "b",
    small_sample_correction: bool = True,
) -> DieboldMarianoResult:
    """Test whether two loss series have equal expected loss.

    ``H0: E[L_a - L_b] = 0``. For tracking, pass squared active returns as the
    losses; the test then asks whether one strategy's *expected squared tracking
    error* differs from the other's, using a HAC variance so daily dependence is
    priced in.

    Applies the Harvey-Leybourne-Newbold (1997) small-sample correction and uses
    a t reference distribution, both of which matter at the sample sizes a
    decade of daily data actually gives you.

    Diebold, F. X. & Mariano, R. S. (1995), *JBES* 13(3), 253-263.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Loss series must have the same shape; got {a.shape} and {b.shape}.")
    differences = a - b
    n = differences.size
    if n < 10:
        raise ValueError("Diebold-Mariano needs a meaningful sample; refusing below 10 observations.")

    used_lag = lag if lag is not None else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    variance = newey_west_variance(differences, used_lag)
    statistic = float(differences.mean() / np.sqrt(variance / n))

    if small_sample_correction:
        h = horizon
        factor = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
        statistic *= float(np.sqrt(max(factor, 1e-12)))
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))

    mean_difference = float(differences.mean())
    favours = "neither (not distinguishable)" if p_value > 0.05 else (label_b if mean_difference > 0 else label_a)
    return DieboldMarianoResult(statistic, p_value, mean_difference, used_lag, n, favours)


# --------------------------------------------------------------------------- #
# Multiple-comparison tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SuperiorPredictiveAbilityResult:
    best_model: str
    best_mean_outperformance: float
    reality_check_p: float
    spa_p: float
    n_models: int
    n_obs: int
    per_model: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.best_model,
            "best_mean_outperformance": self.best_mean_outperformance,
            "reality_check_p": self.reality_check_p,
            "spa_p": self.spa_p,
            "n_models": self.n_models,
            "n_obs": self.n_obs,
            "per_model_mean_outperformance": self.per_model,
        }


def superior_predictive_ability(
    losses: pd.DataFrame,
    benchmark_loss: pd.Series,
    *,
    n_samples: int = 2000,
    mean_block: float = 21.0,
    seed: int = 7,
) -> SuperiorPredictiveAbilityResult:
    """White's Reality Check and Hansen's SPA test over a whole model zoo.

    ``H0``: no model in ``losses`` has lower expected loss than ``benchmark_loss``.

    Why both: the Reality Check is the original and is well known, but it is
    conservative because poor models drag the null distribution around. Hansen's
    SPA studentises each model and recentres only those whose sample performance
    is not implausibly bad, which restores power. Reporting both makes the
    difference visible rather than letting the choice of test do the arguing.

    White, H. (2000), *Econometrica* 68(5), 1097-1126.
    Hansen, P. R. (2005), *JBES* 23(4), 365-380.
    """
    aligned = losses.dropna()
    benchmark = benchmark_loss.reindex(aligned.index).to_numpy(dtype=float)
    values = aligned.to_numpy(dtype=float)
    n_obs, n_models = values.shape
    if n_models == 0:
        raise ValueError("Need at least one candidate model.")

    # Positive f = the model loses less than the benchmark = the model is better.
    performance = benchmark[:, None] - values
    means = performance.mean(axis=0)
    omega = np.array([np.sqrt(newey_west_variance(performance[:, k])) for k in range(n_models)])
    omega = np.where(omega > 0, omega, 1e-12)

    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(n_obs, n_samples, mean_block, rng)
    boot_means = performance[indices].mean(axis=1)  # (n_samples, n_models)

    root_t = np.sqrt(n_obs)
    rc_observed = float(np.max(root_t * means))
    rc_draws = np.max(root_t * (boot_means - means[None, :]), axis=1)
    reality_check_p = float(np.mean(rc_draws >= rc_observed))

    # Hansen's recentring threshold: models whose sample performance is worse than
    # -A_k are treated as irrelevant rather than allowed to distort the null.
    threshold = -omega * np.sqrt(2.0 * np.log(max(np.log(n_obs), 1.0001)) / n_obs)
    recentred = np.where(means >= threshold, means, 0.0)
    spa_observed = max(0.0, float(np.max(root_t * means / omega)))
    spa_draws = np.maximum(0.0, np.max(root_t * (boot_means - recentred[None, :]) / omega[None, :], axis=1))
    spa_p = float(np.mean(spa_draws >= spa_observed))

    best_index = int(np.argmax(means))
    return SuperiorPredictiveAbilityResult(
        best_model=str(aligned.columns[best_index]),
        best_mean_outperformance=float(means[best_index]),
        reality_check_p=reality_check_p,
        spa_p=spa_p,
        n_models=n_models,
        n_obs=n_obs,
        per_model={str(c): float(m) for c, m in zip(aligned.columns, means, strict=True)},
    )


# --------------------------------------------------------------------------- #
# Selection bias in a performance ratio
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """P(true ratio > ``benchmark_sharpe``) given the estimate and its moments.

    Non-normality is not a footnote here: fat tails inflate the standard error of
    a Sharpe estimate, so a ratio computed from crisis-containing daily data is
    less informative than the same number computed from well-behaved data.

    ``kurtosis`` is raw (normal = 3), not excess.
    """
    if n_obs < 2:
        return float("nan")
    denominator = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        return float("nan")
    z = (sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(denominator)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """Expected maximum Sharpe ratio across ``n_trials`` independent null trials.

    The order-statistic approximation from Bailey & Lopez de Prado. This is the
    hurdle a "best of N configurations" result has to clear before it means
    anything: search hard enough over noise and you will find a good-looking ratio.
    """
    if n_trials < 2 or trial_variance <= 0:
        return 0.0
    gamma = EULER_MASCHERONI
    term = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trial_variance) * term)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    hurdle_sharpe: float
    deflated_probability: float
    n_trials: int
    n_obs: int

    @property
    def survives(self) -> bool:
        return self.deflated_probability > 0.95

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_sharpe": self.observed_sharpe,
            "hurdle_sharpe": self.hurdle_sharpe,
            "deflated_probability": self.deflated_probability,
            "n_trials": self.n_trials,
            "n_obs": self.n_obs,
            "survives_at_95pct": self.survives,
        }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    all_trial_sharpes: Sequence[float],
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio for the best of ``len(all_trial_sharpes)`` trials.

    ``returns`` are the per-period returns of the selected configuration;
    ``all_trial_sharpes`` are the per-period Sharpe ratios of *every* trial that
    was run, including the ones that were discarded. That list is the honest part:
    the correction is only as good as the trial count you admit to.

    Bailey, D. H. & Lopez de Prado, M. (2014), *Journal of Portfolio Management*.
    """
    series = np.asarray(returns, dtype=float)
    n_obs = series.size
    if n_obs < 10:
        raise ValueError("Deflated Sharpe needs at least 10 observations.")
    std = series.std(ddof=1)
    observed = float(series.mean() / std) if std > 0 else 0.0
    trials = np.asarray(list(all_trial_sharpes), dtype=float)
    trial_variance = float(trials.var(ddof=1)) if trials.size > 1 else 0.0
    hurdle = expected_max_sharpe(max(trials.size, 1), trial_variance)
    probability = probabilistic_sharpe_ratio(
        observed,
        n_obs,
        float(stats.skew(series)),
        float(stats.kurtosis(series, fisher=False)),
        benchmark_sharpe=hurdle,
    )
    return DeflatedSharpeResult(observed, hurdle, probability, int(trials.size), n_obs)


# --------------------------------------------------------------------------- #
# Convenience wrappers used by the horse race
# --------------------------------------------------------------------------- #
def tracking_error_difference_ci(
    active_a: pd.Series,
    active_b: pd.Series,
    *,
    n_samples: int = 2000,
    mean_block: float = 21.0,
    level: float = 0.95,
    seed: int = 7,
) -> BootstrapCI:
    """Bootstrap CI for ``TE(a) - TE(b)``, annualised, resampling both jointly.

    Negative interval entirely below zero means ``a`` genuinely tracks tighter.
    """
    joint = pd.concat([active_a.rename("a"), active_b.rename("b")], axis=1).dropna()

    def statistic(matrix: np.ndarray) -> float:
        return float((matrix[:, 0].std(ddof=1) - matrix[:, 1].std(ddof=1)) * np.sqrt(TRADING_DAYS))

    return bootstrap_statistic(
        joint.to_numpy(dtype=float),
        statistic,
        n_samples=n_samples,
        mean_block=mean_block,
        level=level,
        seed=seed,
    )
