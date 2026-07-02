import unittest

import numpy as np
import pandas as pd

from etf_replicator import metrics as m


def daily_index(T: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=T)


class TrackingMetricsTests(unittest.TestCase):
    def test_perfect_replication(self):
        rng = np.random.default_rng(3)
        target = pd.Series(rng.normal(0.0004, 0.01, 250), index=daily_index(250))
        stats = m.tracking_metrics(target, target)

        self.assertAlmostEqual(stats["tracking_error"], 0.0)
        self.assertAlmostEqual(stats["correlation"], 1.0)
        self.assertAlmostEqual(stats["beta"], 1.0)
        self.assertAlmostEqual(stats["annualized_active_return"], 0.0)
        self.assertAlmostEqual(stats["max_active_drawdown"], 0.0)
        self.assertTrue(np.isnan(stats["information_ratio"]))  # zero TE has no IR

    def test_constant_active_return(self):
        rng = np.random.default_rng(4)
        target = pd.Series(rng.normal(0.0002, 0.01, 250), index=daily_index(250))
        portfolio = target + 0.0001
        stats = m.tracking_metrics(portfolio, target)

        self.assertAlmostEqual(stats["tracking_error"], 0.0)
        self.assertAlmostEqual(stats["annualized_active_return"], 0.0001 * 252)
        self.assertAlmostEqual(stats["max_active_drawdown"], 0.0)

    def test_max_drawdown_on_known_path(self):
        returns = pd.Series([0.10, -0.50, 0.20], index=daily_index(3))
        # Wealth: 1.10 -> 0.55 -> 0.66; trough vs peak = 0.55/1.10 - 1 = -50%.
        self.assertAlmostEqual(m.max_drawdown(returns), -0.5)


class MonthlyConsistencyTests(unittest.TestCase):
    def test_monthly_returns_compound_within_months(self):
        index = pd.to_datetime(["2023-01-10", "2023-01-20", "2023-02-10"])
        returns = pd.Series([0.10, 0.10, 0.05], index=index)
        monthly = m.monthly_returns(returns)

        self.assertAlmostEqual(monthly.iloc[0], 1.1 * 1.1 - 1)
        self.assertAlmostEqual(monthly.iloc[1], 0.05)

    def test_hit_rate_counts_months_within_tolerance(self):
        index = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30"])
        monthly_active = pd.Series([0.001, -0.010, 0.004, -0.005], index=index)
        self.assertAlmostEqual(m.hit_rate(monthly_active, tolerance=0.005), 0.75)

    def test_hit_rate_empty_is_nan(self):
        self.assertTrue(np.isnan(m.hit_rate(pd.Series(dtype=float))))


class GroupAllocationTests(unittest.TestCase):
    def test_weights_aggregate_by_class(self):
        weights = pd.DataFrame(
            {"AAA": [0.4, 0.5], "BBB": [0.3, 0.2], "CCC": [0.3, 0.3]},
            index=pd.to_datetime(["2023-01-31", "2023-06-30"]),
        )
        groups = m.group_allocation(weights, {"AAA": "equity", "BBB": "equity", "CCC": "bond"})

        self.assertAlmostEqual(groups.loc["2023-01-31", "equity"], 0.7)
        self.assertAlmostEqual(groups.loc["2023-06-30", "bond"], 0.3)

    def test_untagged_assets_fall_into_other(self):
        weights = pd.DataFrame({"AAA": [0.6], "ZZZ": [0.4]}, index=pd.to_datetime(["2023-01-31"]))
        groups = m.group_allocation(weights, {"AAA": "equity"})
        self.assertAlmostEqual(groups.loc["2023-01-31", "other"], 0.4)


class StressWindowTests(unittest.TestCase):
    def test_metrics_computed_only_for_covered_windows(self):
        rng = np.random.default_rng(5)
        index = pd.bdate_range("2020-01-01", "2020-12-31")
        target = pd.Series(rng.normal(0, 0.02, len(index)), index=index)
        portfolio = target + rng.normal(0, 0.002, len(index))

        windows = {
            "COVID Crash": ("2020-02-19", "2020-03-23"),
            "2022 Rate Shock": ("2022-01-03", "2022-06-16"),
        }
        table = m.stress_window_metrics(portfolio, target, windows)

        self.assertIn("COVID Crash", table.index)
        self.assertNotIn("2022 Rate Shock", table.index)

        window_target = target.loc["2020-02-19":"2020-03-23"]
        expected = float((1 + window_target).prod() - 1)
        self.assertAlmostEqual(table.loc["COVID Crash", "target_return"], expected)


if __name__ == "__main__":
    unittest.main()
