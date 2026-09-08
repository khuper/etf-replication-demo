"""The kill switch: it must trip when it should, stay quiet when it should, and
never look ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from etflab.backtest import BacktestResult
from etflab.costs import BpsCostModel
from etflab.governance import HurdlePolicy, govern, payoff_ratio


def _ledger(active: np.ndarray, cost_per_day: float, seed: int) -> pd.DataFrame:
    """A daily ledger with a prescribed active return and a flat daily cost."""
    dates = pd.bdate_range("2018-01-01", periods=len(active))
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(len(active)) * 0.01
    net = target + active
    return pd.DataFrame(
        {"gross": net + cost_per_day, "cost": cost_per_day, "net": net, "target": target, "active": active},
        index=pd.Index(dates, name="date"),
    )


def _result(name: str, active: np.ndarray, cost_per_day: float, seed: int, assets=("A", "B")) -> BacktestResult:
    daily = _ledger(active, cost_per_day, seed)
    weights = pd.DataFrame({a: 1.0 / len(assets) for a in assets}, index=daily.index)
    rebalances = pd.DataFrame(
        {"one_way_turnover": [0.0], "traded_notional": [0.0], "status": ["optimal"], "max_participation": [0.0]},
        index=pd.Index([daily.index[0]], name="rebalance_date"),
    )
    return BacktestResult(name, name, daily, weights.iloc[[0]], weights, rebalances)


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=1500)
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((1500, 2)) * 0.01, index=dates, columns=["A", "B"])


def test_a_strategy_that_is_cheaper_and_better_never_trips(asset_returns):
    rng = np.random.default_rng(1)
    strategy = _result("s", rng.standard_normal(1500) * 0.003, cost_per_day=0.0, seed=1)
    benchmark = _result("b", rng.standard_normal(1500) * 0.006, cost_per_day=1e-6, seed=1)
    report = govern(strategy, benchmark, HurdlePolicy(hurdle=20), BpsCostModel(5.0), asset_returns)
    assert report.days_off == 0
    assert report.switches == 0
    assert report.undefined_share > 0.9, "cheaper-and-better should be reported as 'nothing to pay for'"


def test_a_strategy_that_tracks_worse_is_shut_off_regardless_of_cost(asset_returns):
    rng = np.random.default_rng(2)
    strategy = _result("s", rng.standard_normal(1500) * 0.008, cost_per_day=0.0, seed=2)
    benchmark = _result("b", rng.standard_normal(1500) * 0.003, cost_per_day=0.0, seed=2)
    policy = HurdlePolicy(hurdle=20, window=126, grace=21)
    report = govern(strategy, benchmark, policy, BpsCostModel(5.0), asset_returns)
    assert report.switches >= 1
    assert report.days_off > 1000
    assert report.final_state == "shut off"


def test_a_small_edge_bought_expensively_fails_the_hurdle(asset_returns):
    """Tracking 2bp tighter for 1bp a year more is a 2x payoff. The rule wants 20x."""
    rng = np.random.default_rng(3)
    common = rng.standard_normal(1500) * 0.005
    strategy = _result("s", common * 0.998, cost_per_day=1.0e-4 / 252, seed=3)  # ~1bp/yr more
    benchmark = _result("b", common, cost_per_day=0.0, seed=3)
    policy = HurdlePolicy(hurdle=20, window=126, grace=21, cost_floor_bps=0.01)
    report = govern(strategy, benchmark, policy, BpsCostModel(5.0), asset_returns)
    signal = report.daily.dropna(subset=["ratio"])
    assert not signal.empty
    assert signal["ratio"].median() < 20
    assert report.days_off > 0


def test_the_same_edge_passes_a_lower_hurdle(asset_returns):
    rng = np.random.default_rng(3)
    common = rng.standard_normal(1500) * 0.005
    strategy = _result("s", common * 0.998, cost_per_day=1.0e-4 / 252, seed=3)
    benchmark = _result("b", common, cost_per_day=0.0, seed=3)
    strict = govern(
        strategy,
        benchmark,
        HurdlePolicy(hurdle=20, window=126, grace=21, cost_floor_bps=0.01),
        BpsCostModel(5.0),
        asset_returns,
    )
    lenient = govern(
        strategy,
        benchmark,
        HurdlePolicy(hurdle=0.5, window=126, grace=21, cost_floor_bps=0.01),
        BpsCostModel(5.0),
        asset_returns,
    )
    assert lenient.days_off < strict.days_off


def test_grace_period_stops_a_single_bad_month_from_tripping(asset_returns):
    rng = np.random.default_rng(4)
    good = rng.standard_normal(1500) * 0.003
    good[600:625] *= 6.0  # one terrible month
    strategy = _result("s", good, cost_per_day=0.0, seed=4)
    benchmark = _result("b", rng.standard_normal(1500) * 0.006, cost_per_day=0.0, seed=4)
    patient = govern(
        strategy, benchmark, HurdlePolicy(hurdle=20, window=63, grace=63), BpsCostModel(5.0), asset_returns
    )
    twitchy = govern(strategy, benchmark, HurdlePolicy(hurdle=20, window=63, grace=1), BpsCostModel(5.0), asset_returns)
    assert patient.switches <= twitchy.switches


def test_switching_charges_a_trade_cost(asset_returns):
    rng = np.random.default_rng(5)
    strategy = _result("s", rng.standard_normal(1500) * 0.009, cost_per_day=0.0, seed=5, assets=("A", "B"))
    benchmark = _result("b", rng.standard_normal(1500) * 0.002, cost_per_day=0.0, seed=5, assets=("A", "B"))
    # Give the two strategies different holdings so the switch has to trade.
    object.__setattr__(strategy, "daily_weights", pd.DataFrame({"A": 1.0, "B": 0.0}, index=strategy.daily.index))
    report = govern(
        strategy, benchmark, HurdlePolicy(hurdle=20, window=126, grace=21), BpsCostModel(10.0), asset_returns
    )
    assert report.switches >= 1
    assert report.switching_cost > 0
    # The governed ledger carries the switching cost on the switch date.
    assert report.governed["cost"].max() >= 10.0 * 1e-4 * 1.0 - 1e-12


def test_the_governed_ledger_is_one_of_the_two_sources_on_every_day(asset_returns):
    rng = np.random.default_rng(6)
    strategy = _result("s", rng.standard_normal(1500) * 0.008, cost_per_day=0.0, seed=6)
    benchmark = _result("b", rng.standard_normal(1500) * 0.003, cost_per_day=0.0, seed=6)
    report = govern(
        strategy, benchmark, HurdlePolicy(hurdle=20, window=126, grace=21), BpsCostModel(0.0), asset_returns
    )
    for day, row in report.governed.iterrows():
        source = strategy.daily if report.daily.loc[day, "active"] else benchmark.daily
        assert row["gross"] == pytest.approx(source.loc[day, "gross"])
        assert row["target"] == pytest.approx(source.loc[day, "target"])


def test_the_switch_never_looks_ahead(asset_returns):
    """Truncate both ledgers; every governed row before the cut must be identical."""
    rng = np.random.default_rng(7)
    strategy = _result("s", rng.standard_normal(1500) * 0.007, cost_per_day=0.0, seed=7)
    benchmark = _result("b", rng.standard_normal(1500) * 0.004, cost_per_day=0.0, seed=7)
    policy = HurdlePolicy(hurdle=20, window=126, grace=21)
    full = govern(strategy, benchmark, policy, BpsCostModel(5.0), asset_returns)

    cut = 900

    def truncate(r: BacktestResult) -> BacktestResult:
        return BacktestResult(
            r.strategy, r.label, r.daily.iloc[:cut], r.weights, r.daily_weights.iloc[:cut], r.rebalances
        )

    partial = govern(truncate(strategy), truncate(benchmark), policy, BpsCostModel(5.0), asset_returns)
    assert_frame_equal(full.governed.iloc[:cut], partial.governed, check_exact=True)


def test_payoff_ratio_is_undefined_where_the_strategy_is_not_dearer():
    rng = np.random.default_rng(8)
    strategy = _result("s", rng.standard_normal(600) * 0.003, cost_per_day=0.0, seed=8)
    benchmark = _result("b", rng.standard_normal(600) * 0.006, cost_per_day=1e-5, seed=8)
    signal = payoff_ratio(strategy, benchmark, HurdlePolicy(window=63))
    evaluable = signal[signal["evaluable"]]
    assert evaluable["ratio"].isna().all()
    assert evaluable["free"].all()
    assert not evaluable["breaching"].any()


@pytest.mark.parametrize("bad", [{"hurdle": 0}, {"window": 5}, {"grace": 0}, {"cost_floor_bps": -1}])
def test_policy_validation(bad):
    with pytest.raises(ValueError):
        HurdlePolicy(**bad).validate()


def test_report_serialises(asset_returns):
    rng = np.random.default_rng(9)
    strategy = _result("s", rng.standard_normal(1500) * 0.003, cost_per_day=0.0, seed=9)
    benchmark = _result("b", rng.standard_normal(1500) * 0.006, cost_per_day=0.0, seed=9)
    report = govern(strategy, benchmark, HurdlePolicy(), BpsCostModel(5.0), asset_returns)
    payload = report.as_dict()
    assert payload["policy"]["hurdle"] == 20.0
    assert "verdict" in payload and len(payload["verdict"]) > 50
