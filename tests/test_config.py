"""Configuration: validation, identity, and round-tripping."""

from __future__ import annotations

import dataclasses
import json

import pytest

from etflab.config import ExperimentConfig


def test_normalises_and_deduplicates_tickers():
    config = ExperimentConfig(target=" psp ", assets=("spy", " QQQ ", "SPY", ""), max_weight=0.6)
    assert config.target == "PSP"
    assert config.assets == ("SPY", "QQQ")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target": "SPY", "assets": ("SPY", "QQQ"), "max_weight": 0.6}, "cannot also"),
        ({"assets": ("SPY", "QQQ", "GLD"), "max_weight": 0.25}, "infeasible"),
        ({"start": "2020-01-01", "end": "2019-01-01"}, "strictly before"),
        ({"start": "not-a-date"}, "ISO date"),
        ({"train_days": 10}, "at least 60"),
        ({"embargo_days": -1}, "cannot be negative"),
        ({"embargo_days": 9999}, "smaller than the training window"),
        ({"max_turnover": 5.0}, "L1 weight distance"),
        ({"cvar_alpha": 0.9}, "cvar_alpha"),
        ({"cost_bps": -1}, "cannot be negative"),
        ({"portfolio_notional": 0}, "must be positive"),
        ({"max_participation": 0}, "must lie in"),
        ({"bootstrap_samples": 10}, "nobody should trust"),
        ({"data_source": "bloomberg"}, "data_source must be one of"),
        ({"data_source": "csv"}, "requires csv_path"),
        ({"train_mode": "sideways"}, "expanding"),
    ],
)
def test_validation_rejects_bad_configurations(changes, message):
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**changes).validate()


def test_string_dates_are_compared_as_dates_not_lexicographically():
    """`"2020-1-5" > "2020-12-31"` is true as strings and false as dates."""
    with pytest.raises(ValueError, match="ISO date"):
        ExperimentConfig(start="2020-1-5", end="2020-12-31").validate()


def test_semantic_hash_ignores_non_semantic_fields():
    base = ExperimentConfig().validate()
    assert base.with_changes(output_dir="elsewhere").semantic_hash() == base.semantic_hash()
    assert base.with_changes(cache_dir="/tmp/x").semantic_hash() == base.semantic_hash()
    assert base.with_changes(log_level="debug").semantic_hash() == base.semantic_hash()


@pytest.mark.parametrize(
    "changes",
    [
        {"max_weight": 0.30},
        {"train_days": 505},
        {"synthetic_seed": 1},
        {"cost_bps": 5.0001},
        {"assets": ("SPY", "QQQ", "IWM", "VEA", "VWO")},
        {"strategy": "ridge"},
    ],
)
def test_semantic_hash_changes_with_anything_that_moves_a_number(changes):
    base = ExperimentConfig().validate()
    assert base.with_changes(**changes).semantic_hash() != base.semantic_hash()


def test_hash_is_insensitive_to_field_order_but_sensitive_to_float_precision():
    base = ExperimentConfig().validate()
    assert base.with_changes(max_weight=0.25).semantic_hash() == base.semantic_hash()
    assert base.with_changes(max_weight=0.25 + 1e-12).semantic_hash() != base.semantic_hash()


def test_json_round_trip_preserves_identity():
    config = ExperimentConfig(assets=("SPY", "QQQ", "GLD"), max_weight=0.5).validate()
    restored = ExperimentConfig.from_json(config.to_json())
    assert restored == config
    assert restored.semantic_hash() == config.semantic_hash()


def test_unknown_keys_are_rejected_rather_than_ignored():
    payload = json.loads(ExperimentConfig().to_json())
    payload["sharpe_target"] = 2.0
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        ExperimentConfig.from_dict(payload)


def test_config_is_frozen():
    config = ExperimentConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_weight = 0.9  # type: ignore[misc]


def test_universe_puts_the_target_last():
    config = ExperimentConfig(assets=("SPY", "QQQ"), target="PSP", max_weight=0.6).validate()
    assert config.universe == ("SPY", "QQQ", "PSP")
