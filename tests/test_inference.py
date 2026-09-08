"""Statistical machinery, validated by simulation rather than by inspection.

A test that merely calls ``diebold_mariano`` and checks it returns a float
verifies nothing about whether the test is a test. These check the properties
that make each procedure worth reporting: correct size under the null, power
under the alternative, and the calibration the theory promises.

Replication counts are kept small enough to run in CI. The tolerances are set
accordingly -- wide enough not to flake, tight enough that a genuinely broken
implementation cannot pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etflab.inference import (
    bootstrap_statistic,
    deflated_sharpe_ratio,
    diebold_mariano,
    expected_max_sharpe,
    newey_west_variance,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_indices,
    superior_predictive_ability,
    tracking_error_difference_ci,
)


def _ar1_from(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """AR(1) drawn from a caller-supplied generator.

    Taking the generator rather than a seed matters here: two series built from
    the same seed share their innovations, which would quietly make a
    "null is true" fixture into one where the null is false.
    """
    noise = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = noise[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    return _ar1_from(np.random.default_rng(seed), n, phi)


# --------------------------------------------------------------------------- #
# Stationary bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_indices_have_the_requested_mean_block_length():
    indices = stationary_bootstrap_indices(1000, 400, 21.0, np.random.default_rng(1))
    continuations = (np.diff(indices, axis=1) % 1000) == 1
    realised_mean_block = 1.0 / (1.0 - continuations.mean())
    assert realised_mean_block == pytest.approx(21.0, rel=0.15)


def test_bootstrap_indices_are_in_range_and_correctly_shaped():
    indices = stationary_bootstrap_indices(50, 10, 5.0, np.random.default_rng(2))
    assert indices.shape == (10, 50)
    assert indices.min() >= 0 and indices.max() < 50


def test_bootstrap_is_deterministic_given_a_seed():
    a = bootstrap_statistic(_ar1(300, 0.4, 1), lambda x: float(x.mean()), n_samples=100, seed=5)
    b = bootstrap_statistic(_ar1(300, 0.4, 1), lambda x: float(x.mean()), n_samples=100, seed=5)
    assert a.lower == b.lower and a.upper == b.upper


@pytest.mark.slow
def test_bootstrap_interval_has_approximately_nominal_coverage():
    """The stationary bootstrap slightly under-covers under strong dependence,
    which is a known property rather than a bug -- but it must be close."""
    covered = 0
    trials = 200
    for trial in range(trials):
        series = _ar1(600, 0.5, 1000 + trial)
        interval = bootstrap_statistic(
            series, lambda x: float(x.mean()), n_samples=200, mean_block=21, level=0.90, seed=trial
        )
        covered += interval.lower <= 0.0 <= interval.upper
    assert 0.80 <= covered / trials <= 0.97, f"coverage {covered / trials:.1%} is far from the nominal 90%"


def test_paired_resampling_preserves_the_link_between_two_series():
    """Resampling two strategies independently would inflate the variance of
    their difference and hide real differences. Rows must move together."""
    rng = np.random.default_rng(4)
    common = rng.standard_normal(500)
    frame = np.column_stack([common + 0.01 * rng.standard_normal(500), common])
    interval = bootstrap_statistic(frame, lambda m: float((m[:, 0] - m[:, 1]).std(ddof=1)), n_samples=200, seed=3)
    assert interval.upper < 0.1, "paired resampling should keep the difference tiny"


# --------------------------------------------------------------------------- #
# HAC variance
# --------------------------------------------------------------------------- #
def test_newey_west_matches_the_sample_variance_at_zero_lag():
    rng = np.random.default_rng(6)
    series = rng.standard_normal(500)
    assert newey_west_variance(series, lag=0) == pytest.approx(float(series.var(ddof=0)))


def test_newey_west_inflates_the_variance_of_a_persistent_series():
    """The whole point of a HAC estimator: positive autocorrelation means the
    mean is less precisely estimated than an i.i.d. calculation would suggest."""
    series = _ar1(2000, 0.7, 7)
    assert newey_west_variance(series) > 1.5 * newey_west_variance(series, lag=0)


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def test_diebold_mariano_has_correct_size_under_the_null():
    """Two strategies with identical loss distributions must be rejected as
    different about 5% of the time at the 5% level."""
    rejections, trials = 0, 300
    for trial in range(trials):
        rng = np.random.default_rng(5000 + trial)
        shared = _ar1_from(rng, 900, 0.3)
        loss_a = (shared + 0.5 * rng.standard_normal(900)) ** 2
        loss_b = (shared + 0.5 * rng.standard_normal(900)) ** 2
        rejections += diebold_mariano(loss_a, loss_b).p_value < 0.05
    size = rejections / trials
    assert 0.01 <= size <= 0.11, f"empirical size {size:.1%} is far from the nominal 5%"


def test_diebold_mariano_detects_a_genuine_difference():
    rejections, trials = 0, 60
    for trial in range(trials):
        rng = np.random.default_rng(9000 + trial)
        rejections += (
            diebold_mariano((0.8 * rng.standard_normal(900)) ** 2, (rng.standard_normal(900)) ** 2).p_value < 0.05
        )
    assert rejections / trials > 0.85


def test_diebold_mariano_reports_which_strategy_it_favours():
    rng = np.random.default_rng(21)
    better = (0.5 * rng.standard_normal(800)) ** 2
    worse = (rng.standard_normal(800)) ** 2
    assert diebold_mariano(better, worse, label_a="better", label_b="worse").favours == "better"
    assert diebold_mariano(worse, better, label_a="worse", label_b="better").favours == "better"


def test_diebold_mariano_refuses_mismatched_or_tiny_samples():
    with pytest.raises(ValueError, match="same shape"):
        diebold_mariano(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError, match=r"at least|refusing"):
        diebold_mariano(np.zeros(5), np.zeros(5))


# --------------------------------------------------------------------------- #
# Reality Check and SPA
# --------------------------------------------------------------------------- #
def test_spa_and_reality_check_have_correct_size_with_a_zoo_of_useless_models():
    rc_rejections = spa_rejections = 0
    trials = 120
    for trial in range(trials):
        rng = np.random.default_rng(200 + trial)
        benchmark = pd.Series(rng.standard_normal(600) ** 2)
        models = pd.DataFrame({f"m{k}": rng.standard_normal(600) ** 2 for k in range(6)})
        result = superior_predictive_ability(models, benchmark, n_samples=199, seed=trial)
        rc_rejections += result.reality_check_p < 0.05
        spa_rejections += result.spa_p < 0.05
    assert rc_rejections / trials <= 0.10, "the Reality Check should be conservative, not liberal"
    assert spa_rejections / trials <= 0.15


def test_spa_detects_one_genuinely_better_model_hidden_among_noise():
    detections, trials = 0, 40
    for trial in range(trials):
        rng = np.random.default_rng(400 + trial)
        benchmark = pd.Series(rng.standard_normal(600) ** 2)
        columns = {f"m{k}": rng.standard_normal(600) ** 2 for k in range(6)}
        columns["good"] = (0.8 * rng.standard_normal(600)) ** 2
        result = superior_predictive_ability(pd.DataFrame(columns), benchmark, n_samples=199, seed=trial)
        detections += result.spa_p < 0.05 and result.best_model == "good"
    assert detections / trials > 0.8


def test_spa_is_at_least_as_powerful_as_the_reality_check():
    rng = np.random.default_rng(77)
    benchmark = pd.Series(rng.standard_normal(800) ** 2)
    columns = {f"junk{k}": (2.0 * rng.standard_normal(800)) ** 2 for k in range(6)}
    columns["good"] = (0.85 * rng.standard_normal(800)) ** 2
    result = superior_predictive_ability(pd.DataFrame(columns), benchmark, n_samples=499, seed=1)
    assert result.spa_p <= result.reality_check_p + 1e-12


# --------------------------------------------------------------------------- #
# Deflated Sharpe
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_is_zero_for_a_single_trial_and_rises_with_more():
    assert expected_max_sharpe(1, 0.01) == 0.0
    hurdles = [expected_max_sharpe(n, 0.0025) for n in (5, 50, 500, 5000)]
    assert hurdles == sorted(hurdles)
    assert hurdles[0] > 0


def test_expected_max_sharpe_scales_with_the_spread_of_the_trials():
    assert expected_max_sharpe(100, 0.04) > expected_max_sharpe(100, 0.0025)


def test_probabilistic_sharpe_penalises_negative_skew_and_fat_tails():
    baseline = probabilistic_sharpe_ratio(0.05, 1000, skew=0.0, kurtosis=3.0)
    fat_tailed = probabilistic_sharpe_ratio(0.05, 1000, skew=0.0, kurtosis=10.0)
    negatively_skewed = probabilistic_sharpe_ratio(0.05, 1000, skew=-1.0, kurtosis=3.0)
    assert fat_tailed < baseline
    assert negatively_skewed < baseline


def test_deflated_sharpe_rejects_the_best_of_many_worthless_trials():
    rng = np.random.default_rng(3)
    trials = [float(rng.standard_normal(1500).mean() / rng.standard_normal(1500).std()) for _ in range(80)]
    result = deflated_sharpe_ratio(rng.standard_normal(1500) * 0.01, trials)
    assert not result.survives
    assert result.hurdle_sharpe > 0


def test_deflated_sharpe_accepts_a_strong_result_from_few_trials():
    rng = np.random.default_rng(9)
    returns = rng.standard_normal(2000) * 0.01 + 0.0025  # ~0.25 daily Sharpe
    result = deflated_sharpe_ratio(returns, [0.02, 0.03, 0.025])
    assert result.survives


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
def test_tracking_error_difference_ci_brackets_a_known_difference():
    rng = np.random.default_rng(15)
    common = rng.standard_normal(1200) * 0.005
    tight = pd.Series(common)
    loose = pd.Series(common + rng.standard_normal(1200) * 0.005)
    interval = tracking_error_difference_ci(tight, loose, n_samples=300, seed=1)
    assert interval.point < 0, "the tighter series must have the lower tracking error"
    assert interval.excludes_zero
    assert interval.lower <= interval.point <= interval.upper
