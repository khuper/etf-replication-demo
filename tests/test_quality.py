"""Data quality gates. A gate that has never seen a broken panel is not a control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etflab.data.panel import PricePanel
from etflab.data.providers import align_prices
from etflab.data.quality import DataQualityError, run_quality_gates
from etflab.data.synthetic import corrupt


def _gate(report, name):
    return next(g for g in report.gates if g.name == name)


def test_clean_synthetic_data_passes_every_blocking_gate(small_panel):
    report = run_quality_gates(small_panel, min_obs=300)
    assert report.ok
    assert report.failures == ()


def test_corrupted_data_fails(small_panel):
    report = run_quality_gates(corrupt(small_panel, seed=5), min_obs=300)
    assert not report.ok
    with pytest.raises(DataQualityError, match="gate"):
        report.raise_if_failed()


def test_stale_prices_are_detected(small_panel):
    prices = small_panel.prices.copy()
    prices.iloc[100:130, 0] = prices.iloc[99, 0]
    panel = PricePanel(prices, small_panel.assets, small_panel.target, "test")
    gate = _gate(run_quality_gates(panel), "stale_prices")
    assert gate.status == "fail"
    assert max(gate.evidence["longest_run"].values()) >= 29


def test_unadjusted_split_is_detected(small_panel):
    prices = small_panel.prices.copy()
    cut = len(prices) // 2
    prices.iloc[cut:, 1] = prices.iloc[cut:, 1] / 2.0
    panel = PricePanel(prices, small_panel.assets, small_panel.target, "test")
    assert _gate(run_quality_gates(panel), "corporate_action_jumps").status == "fail"


def test_non_positive_prices_are_fatal(small_panel):
    prices = small_panel.prices.copy()
    prices.iloc[10, 0] = -1.0
    panel = PricePanel(prices, small_panel.assets, small_panel.target, "test")
    assert _gate(run_quality_gates(panel), "non_positive_prices").status == "fail"


def test_zero_variance_series_is_fatal(small_panel):
    prices = small_panel.prices.copy()
    prices.iloc[:, 0] = 100.0
    panel = PricePanel(prices, small_panel.assets, small_panel.target, "test")
    report = run_quality_gates(panel)
    assert _gate(report, "zero_variance").status == "fail"


def test_insufficient_history_is_fatal(small_panel):
    assert _gate(run_quality_gates(small_panel, min_obs=10**6), "sample_size").status == "fail"


def test_multicollinearity_is_flagged_but_not_fatal(small_panel):
    """Ill-conditioning is a property of the problem, not an error in the data --
    it warns, because the right response is to expect unstable weights, not to
    refuse to run."""
    prices = small_panel.prices.copy()
    prices.iloc[:, 1] = prices.iloc[:, 0] * 1.0001  # a near-duplicate candidate
    panel = PricePanel(prices, small_panel.assets, small_panel.target, "test")
    gate = _gate(run_quality_gates(panel), "multicollinearity")
    assert gate.status == "warn"
    assert gate.evidence["condition_number"] > 1000


def test_report_status_is_the_worst_gate(small_panel):
    assert run_quality_gates(small_panel, min_obs=300).status in {"ok", "warn"}
    assert run_quality_gates(corrupt(small_panel, seed=5), min_obs=300).status == "fail"


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_alignment_attributes_truncation_to_the_late_listing_ticker():
    """The bug this replaces: a blanket ``dropna()`` throws away years of history
    for every ticker because one listed late, and never says so."""
    dates = pd.bdate_range("2015-01-01", periods=500)
    frame = pd.DataFrame(
        {"OLD": np.linspace(100, 150, 500), "NEW": np.linspace(50, 70, 500), "TGT": np.linspace(10, 14, 500)},
        index=dates,
    )
    frame.loc[frame.index[:300], "NEW"] = np.nan

    aligned, report = align_prices(frame, ["OLD", "NEW", "TGT"])
    assert len(aligned) == 200
    assert report.binding_ticker == "NEW"
    assert report.rows_dropped_leading == 300
    assert report.effective_start == str(dates[300].date())


def test_alignment_counts_internal_holes_separately_from_leading_truncation():
    dates = pd.bdate_range("2015-01-01", periods=200)
    frame = pd.DataFrame({"A": np.linspace(1, 2, 200), "B": np.linspace(3, 4, 200)}, index=dates)
    frame.loc[frame.index[50:55], "B"] = np.nan
    aligned, report = align_prices(frame, ["A", "B"])
    assert report.rows_dropped_leading == 0
    assert report.rows_dropped_internal == 5
    assert len(aligned) == 195


def test_alignment_refuses_a_ticker_with_no_data():
    dates = pd.bdate_range("2015-01-01", periods=100)
    frame = pd.DataFrame({"A": np.linspace(1, 2, 100), "B": [np.nan] * 100}, index=dates)
    with pytest.raises(ValueError, match="No usable price history"):
        align_prices(frame, ["A", "B"])


def test_alignment_refuses_non_overlapping_histories():
    dates = pd.bdate_range("2015-01-01", periods=100)
    frame = pd.DataFrame({"A": np.linspace(1, 2, 100), "B": np.linspace(1, 2, 100)}, index=dates)
    frame.loc[frame.index[50:], "A"] = np.nan
    frame.loc[frame.index[:50], "B"] = np.nan
    with pytest.raises(ValueError, match="no overlapping history"):
        align_prices(frame, ["A", "B"])
