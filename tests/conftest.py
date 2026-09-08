"""Shared fixtures.

Tests run against a deliberately small synthetic market. The full experiment
takes minutes; a test suite that takes minutes is a test suite that gets run
once a week, so the fixtures here are sized for seconds while exercising the
same code paths.
"""

from __future__ import annotations

import pytest

from etflab.config import ExperimentConfig
from etflab.data.synthetic import SyntheticSpec, generate_market

SMALL_ASSETS = ("SPY", "QQQ", "IWM", "HYG", "GLD")


@pytest.fixture(scope="session")
def small_config() -> ExperimentConfig:
    return ExperimentConfig(
        assets=SMALL_ASSETS,
        start="2015-01-01",
        end="2019-12-31",
        train_days=252,
        rebalance_days=63,
        max_weight=0.40,
        max_turnover=0.30,
        bootstrap_samples=200,
        output_dir="outputs-test",
    ).validate()


@pytest.fixture(scope="session")
def small_panel(small_config: ExperimentConfig):
    return generate_market(
        SyntheticSpec(
            assets=small_config.assets,
            target=small_config.target,
            start=small_config.start,
            end=small_config.end,
            seed=small_config.synthetic_seed,
        )
    )


@pytest.fixture(scope="session")
def full_config() -> ExperimentConfig:
    """The shipped default, for the handful of tests that must use it."""
    return ExperimentConfig(bootstrap_samples=200).validate()
