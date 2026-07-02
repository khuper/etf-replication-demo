import unittest

import numpy as np
import pandas as pd

from etf_replicator.stress import beta_stress_test


class BetaStressTestTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        self.asset_returns = pd.DataFrame(
            {
                "AAA": [0.010, 0.020, -0.010, 0.015, 0.005],
                "BBB": [0.008, 0.012, -0.006, 0.009, 0.004],
            },
            index=dates,
        )
        self.target_returns = pd.Series([0.009, 0.018, -0.009, 0.013, 0.005], index=dates, name="TARGET")
        self.weights = pd.Series({"AAA": 0.6, "BBB": 0.4})

    def test_impacts_are_consistent(self):
        result = beta_stress_test(self.asset_returns, self.target_returns, self.weights, shock=-0.20)

        self.assertEqual(result["shock"], -0.20)
        self.assertAlmostEqual(result["target_impact"], -0.20)
        self.assertAlmostEqual(
            result["relative_performance"], result["portfolio_impact"] - result["target_impact"]
        )
        self.assertAlmostEqual(result["portfolio_impact"], result["portfolio_beta"] * -0.20)

    def test_portfolio_beta_matches_manual_covariance(self):
        result = beta_stress_test(self.asset_returns, self.target_returns, self.weights, shock=-0.20)

        target_var = self.target_returns.var(ddof=1)
        betas = self.asset_returns.apply(lambda col: col.cov(self.target_returns) / target_var)
        expected_beta = float((self.weights * betas).sum())
        self.assertAlmostEqual(result["portfolio_beta"], expected_beta)


if __name__ == "__main__":
    unittest.main()
