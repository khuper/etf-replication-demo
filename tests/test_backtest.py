import unittest

import numpy as np
import pandas as pd

from etf_replicator.backtest import drift_weights, run_expanding_backtest


def make_market(T: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=T)
    assets = pd.DataFrame(
        {
            "AAA": rng.normal(0.0004, 0.010, T),
            "BBB": rng.normal(0.0003, 0.008, T),
            "CCC": rng.normal(0.0002, 0.005, T),
        },
        index=dates,
    )
    target = 0.5 * assets["AAA"] + 0.3 * assets["BBB"] + 0.2 * assets["CCC"] + rng.normal(0, 0.001, T)
    returns = assets.copy()
    returns["TARGET"] = target
    return returns


class RunExpandingBacktestTests(unittest.TestCase):
    ASSETS = ["AAA", "BBB", "CCC"]

    def run_backtest(self, **kwargs):
        defaults = dict(
            initial_train_size=120,
            step=60,
            max_weight=0.60,
            max_turnover=0.30,
            cvar_ratio=3.0,
            cost_bps=10.0,
        )
        defaults.update(kwargs)
        return run_expanding_backtest(make_market(), self.ASSETS, "TARGET", **defaults)

    def test_weights_history_has_one_row_per_rebalance(self):
        result = self.run_backtest()
        expected_rebalances = len(range(120, 300, 60))
        self.assertEqual(len(result.weights), expected_rebalances)
        self.assertEqual(list(result.weights.columns), self.ASSETS)
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-6)

    def test_out_of_sample_returns_cover_full_post_training_range(self):
        result = self.run_backtest()
        market = make_market()
        expected_index = market.index[120:]
        pd.testing.assert_index_equal(result.gross_returns.index, expected_index)
        pd.testing.assert_index_equal(result.net_returns.index, expected_index)
        pd.testing.assert_index_equal(result.target_returns.index, expected_index)

    def test_costs_charged_only_on_rebalance_days(self):
        result = self.run_backtest(cost_bps=10.0)
        diff = result.gross_returns - result.net_returns
        expected_costs = result.turnover * (10.0 / 1e4)

        pd.testing.assert_series_equal(
            diff.loc[result.turnover.index], expected_costs, check_names=False
        )
        off_rebalance = diff.drop(result.turnover.index)
        self.assertTrue((off_rebalance == 0).all())

    def test_first_rebalance_is_full_deployment(self):
        result = self.run_backtest()
        self.assertEqual(result.turnover.iloc[0], 1.0)
        self.assertTrue((result.turnover.iloc[1:] <= 0.30 + 1e-4).all())

    def test_net_at_zero_cost_equals_gross(self):
        result = self.run_backtest()
        pd.testing.assert_series_equal(result.net_at(0.0), result.gross_returns, check_names=False)

    def test_rejects_train_size_longer_than_history(self):
        with self.assertRaises(ValueError):
            self.run_backtest(initial_train_size=1000)


class DriftWeightsTests(unittest.TestCase):
    def test_weights_drift_with_compounded_returns(self):
        weights = np.array([0.5, 0.5])
        period = pd.DataFrame({"AAA": [0.10, 0.10], "BBB": [0.0, 0.0]})
        drifted = drift_weights(weights, period)

        # AAA compounds 21%, BBB stays flat: 0.605 / (0.605 + 0.5).
        self.assertAlmostEqual(drifted[0], 0.605 / 1.105, places=9)
        self.assertAlmostEqual(drifted.sum(), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
