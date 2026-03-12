import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from src.replicator import SyntheticLiabilityReplicator
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    SyntheticLiabilityReplicator = None
    IMPORT_ERROR = exc


@unittest.skipIf(SyntheticLiabilityReplicator is None, f"Missing dependency: {IMPORT_ERROR}")
class SyntheticLiabilityReplicatorTests(unittest.TestCase):
    def test_fetch_data_handles_multiindex_close(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        columns = pd.MultiIndex.from_product([["Open", "Close"], ["AAA", "BBB", "TARGET"]])
        frame = pd.DataFrame(
            [
                [99, 101, 103, 100, 102, 104],
                [100, 102, 104, 101, 103, 105],
                [101, 103, 105, 102, 104, 106],
                [102, 104, 106, 103, 105, 107],
            ],
            index=dates,
            columns=columns,
        )

        with patch("src.replicator.yf.download", return_value=frame):
            replicator = SyntheticLiabilityReplicator(["AAA", "BBB"], "TARGET", "2024-01-01", "2024-01-05")
            returns = replicator.fetch_data()

        self.assertEqual(list(replicator.data.columns), ["AAA", "BBB", "TARGET"])
        self.assertEqual(returns.shape, (3, 3))

    def test_get_asset_target_split_returns_expected_columns(self):
        replicator = SyntheticLiabilityReplicator(["AAA", "BBB"], "TARGET", "2024-01-01", "2024-01-05")
        returns = pd.DataFrame(
            {
                "AAA": [0.01, 0.02],
                "BBB": [0.03, -0.01],
                "TARGET": [0.02, 0.00],
            }
        )

        asset_returns, target_returns = replicator.get_asset_target_split(returns)

        self.assertEqual(list(asset_returns.columns), ["AAA", "BBB"])
        self.assertEqual(target_returns.name, "TARGET")

    def test_optimize_tracking_error_returns_feasible_weights(self):
        replicator = SyntheticLiabilityReplicator(["AAA", "BBB", "CCC", "DDD"], "TARGET", "2024-01-01", "2024-01-10")
        asset_returns = pd.DataFrame(
            {
                "AAA": [0.010, 0.012, 0.011, 0.009, 0.013, 0.010],
                "BBB": [0.006, 0.005, 0.007, 0.006, 0.004, 0.005],
                "CCC": [-0.002, 0.000, -0.001, 0.001, 0.000, -0.001],
                "DDD": [0.004, 0.003, 0.005, 0.004, 0.003, 0.004],
            }
        )
        target_returns = pd.Series([0.0076, 0.0079, 0.0081, 0.0073, 0.0080, 0.0072], name="TARGET")

        weights = replicator.optimize_tracking_error(
            asset_returns,
            target_returns,
            cvar_constraint_ratio=2.0,
            max_weight=0.60,
        )

        self.assertIsNotNone(weights)
        self.assertTrue(np.isclose(weights["weights"].sum(), 1.0))
        self.assertTrue((weights["weights"] >= 0).all())
        self.assertTrue((weights["weights"] <= 0.60 + 1e-6).all())

    def test_stress_test_uses_portfolio_beta_against_target(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        replicator = SyntheticLiabilityReplicator(["AAA", "BBB"], "TARGET", "2024-01-01", "2024-01-10")
        replicator.returns = pd.DataFrame(
            {
                "AAA": [0.010, 0.020, -0.010, 0.015, 0.005],
                "BBB": [0.008, 0.012, -0.006, 0.009, 0.004],
                "TARGET": [0.009, 0.018, -0.009, 0.013, 0.005],
            },
            index=dates,
        )
        weights = pd.Series({"AAA": 0.6, "BBB": 0.4})

        result = replicator.stress_test(weights, shock=-0.20)

        self.assertEqual(result["Shock Size"], -0.20)
        self.assertTrue(np.isclose(result["Relative Performance"], result["Portfolio Impact"] - result["Target Impact"]))


if __name__ == "__main__":
    unittest.main()
