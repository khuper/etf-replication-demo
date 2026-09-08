"""CSCV and the probability of backtest overfitting, checked against known cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etflab.overfitting import (
    noise_reference_pbo,
    performance_from_active,
    probability_of_backtest_overfitting,
)


def _noise_frame(n_periods: int = 1200, n_configs: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n_periods, n_configs)), columns=[f"c{i}" for i in range(n_configs)])


def test_pure_noise_produces_a_high_pbo_on_average():
    """The PBO of a *single* noise realisation is high variance -- one
    configuration can be lucky throughout a finite sample -- so the property that
    must hold is about the average across realisations, not any one draw. This is
    exactly why the reported PBO is compared against an averaged reference."""
    values = [probability_of_backtest_overfitting(_noise_frame(seed=s), n_splits=10).pbo for s in range(5)]
    assert 0.35 < float(np.mean(values)) < 0.85, f"mean PBO across noise draws was {np.mean(values):.2f}"


def test_a_genuinely_superior_configuration_produces_a_low_pbo():
    frame = _noise_frame()
    frame["c0"] = frame["c0"] + 0.30
    result = probability_of_backtest_overfitting(frame, n_splits=10)
    assert result.pbo < 0.05
    assert result.verdict.startswith("low")
    assert pd.Series(result.selected_configs).eq("c0").mean() > 0.9


def test_a_configuration_that_flips_halfway_is_caught():
    """The classic overfitting trap: brilliant in the first half, terrible in the
    second. A single train/test split can miss it depending on where you cut;
    CSCV averages over every balanced cut, so it cannot."""
    frame = _noise_frame()
    frame.iloc[:600, 1] += 0.5
    frame.iloc[600:, 1] -= 0.5
    result = probability_of_backtest_overfitting(frame, n_splits=10)
    assert result.pbo > 0.5


def test_noise_reference_gives_the_null_for_a_grid_of_this_shape():
    """CSCV on a fixed sample does not have a clean 0.5 null, so the reported PBO
    has to be read against a matched, averaged reference."""
    reference = noise_reference_pbo(1200, 24, n_splits=10, n_replications=5)
    assert 0.3 < reference < 0.95
    signal = _noise_frame()
    signal["c0"] += 0.30
    assert probability_of_backtest_overfitting(signal, n_splits=10).pbo < reference - 0.2


def test_odd_split_counts_are_rejected():
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(_noise_frame(), n_splits=7)


def test_too_short_a_sample_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        probability_of_backtest_overfitting(_noise_frame(n_periods=50), n_splits=10)


def test_a_single_configuration_cannot_be_ranked():
    with pytest.raises(ValueError, match="at least two"):
        probability_of_backtest_overfitting(_noise_frame(n_configs=1), n_splits=10)


def test_combination_count_is_capped_and_reported():
    result = probability_of_backtest_overfitting(_noise_frame(n_periods=2000), n_splits=16, max_combinations=100)
    assert result.n_combinations == 100
    assert len(result.logits) == 100


def test_performance_matrix_rewards_tighter_tracking():
    active = pd.DataFrame({"tight": [0.001, -0.001], "loose": [0.02, -0.02]})
    performance = performance_from_active(active)
    assert performance["tight"].mean() > performance["loose"].mean()
    assert (performance <= 0).all().all(), "negative squared error is never positive"


def test_pbo_is_invariant_to_noise_below_solver_tolerance():
    """Configurations that differ only at the level of optimiser convergence
    noise must rank identically on every machine, or PBO becomes a property of
    the BLAS library rather than of the strategy."""
    base = _noise_frame(n_periods=1200, n_configs=20, seed=3)
    # Duplicate several configurations so genuine ties exist, then perturb at
    # 1e-9 relative -- an order of magnitude below the resolution.
    frame = base.copy()
    for column in ("c1", "c2", "c3", "c4"):
        frame[column] = frame["c0"]
    rng = np.random.default_rng(7)
    perturbed = frame * (1.0 + 1e-9 * rng.standard_normal(frame.shape))
    perturbed_other = frame * (1.0 + 1e-9 * rng.standard_normal(frame.shape))

    a = probability_of_backtest_overfitting(perturbed, n_splits=10)
    b = probability_of_backtest_overfitting(perturbed_other, n_splits=10)
    assert a.pbo == b.pbo
    assert a.selected_configs == b.selected_configs


def test_tie_resolution_does_not_erase_real_differences():
    frame = _noise_frame(n_periods=1200, n_configs=20, seed=3)
    frame["c0"] = frame["c0"] + 0.30
    result = probability_of_backtest_overfitting(frame, n_splits=10)
    assert result.pbo < 0.05
