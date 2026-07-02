import unittest

import numpy as np
import pandas as pd

from etf_replicator.config import validate_group_bounds
from etf_replicator.optimizer import historical_cvar, optimize_tracking_error


def make_returns(seed: int = 7, T: int = 120):
    rng = np.random.default_rng(seed)
    assets = pd.DataFrame(
        {
            "AAA": rng.normal(0.0005, 0.010, T),
            "BBB": rng.normal(0.0004, 0.009, T),
            "CCC": rng.normal(0.0002, 0.004, T),
            "DDD": rng.normal(0.0002, 0.005, T),
        }
    )
    target = 0.4 * assets["AAA"] + 0.3 * assets["BBB"] + 0.3 * assets["CCC"] + rng.normal(0, 0.001, T)
    target.name = "TARGET"
    return assets, target


class OptimizeTrackingErrorTests(unittest.TestCase):
    def test_returns_feasible_weights(self):
        assets, target = make_returns()
        weights = optimize_tracking_error(assets, target, cvar_ratio=2.0, max_weight=0.60)

        self.assertIsNotNone(weights)
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights <= 0.60 + 1e-6).all())
        self.assertEqual(list(weights.index), ["AAA", "BBB", "CCC", "DDD"])

    def test_turnover_constraint_limits_trade(self):
        assets, target = make_returns()
        w_prev = np.array([0.25, 0.25, 0.25, 0.25])
        weights = optimize_tracking_error(
            assets, target, cvar_ratio=2.0, max_weight=0.60, w_prev=w_prev, max_turnover=0.10
        )

        self.assertIsNotNone(weights)
        self.assertLessEqual(np.abs(weights.values - w_prev).sum(), 0.10 + 1e-4)

    def test_infeasible_position_cap_returns_none(self):
        assets, target = make_returns()
        # Four assets capped at 10% cannot sum to 100%: both solves are infeasible.
        weights = optimize_tracking_error(assets, target, max_weight=0.10)
        self.assertIsNone(weights)

    def test_group_target_is_hit_when_min_equals_max(self):
        assets, target = make_returns()
        classes = {"AAA": "equity", "BBB": "equity", "CCC": "bond", "DDD": "bond"}
        weights = optimize_tracking_error(
            assets, target, cvar_ratio=3.0, max_weight=0.60,
            group_bounds={"equity": (0.5, 0.5)}, asset_classes=classes,
        )

        self.assertIsNotNone(weights)
        self.assertAlmostEqual(weights[["AAA", "BBB"]].sum(), 0.5, places=3)

    def test_group_can_be_excluded(self):
        assets, target = make_returns()
        classes = {"AAA": "equity", "BBB": "equity", "CCC": "bond", "DDD": "bond"}
        weights = optimize_tracking_error(
            assets, target, cvar_ratio=3.0, max_weight=0.60,
            group_bounds={"bond": (0.0, 0.0)}, asset_classes=classes,
        )

        self.assertIsNotNone(weights)
        self.assertAlmostEqual(weights[["CCC", "DDD"]].sum(), 0.0, places=6)
        self.assertAlmostEqual(weights[["AAA", "BBB"]].sum(), 1.0, places=6)

    def test_historical_cvar_matches_manual_tail_mean(self):
        returns = np.array([-0.05, -0.03, -0.01, 0.00, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04] * 4)
        cvar = historical_cvar(returns, alpha=0.05)
        # 40 observations, alpha 5% -> mean of the 2 worst returns.
        self.assertAlmostEqual(cvar, 0.05)


class ValidateGroupBoundsTests(unittest.TestCase):
    CLASSES = {"AAA": "equity", "BBB": "equity", "CCC": "bond"}

    def test_accepts_reasonable_bounds(self):
        validate_group_bounds({"equity": (0.2, 0.6), "bond": (0.0, 0.5)}, self.CLASSES, max_weight=0.5)

    def test_rejects_min_above_max(self):
        with self.assertRaisesRegex(ValueError, "min <= max"):
            validate_group_bounds({"equity": (0.7, 0.2)}, self.CLASSES, max_weight=0.5)

    def test_rejects_unknown_group_with_positive_min(self):
        with self.assertRaisesRegex(ValueError, "no basket asset is tagged"):
            validate_group_bounds({"crypto": (0.1, 0.5)}, self.CLASSES, max_weight=0.5)

    def test_rejects_unreachable_group_minimum(self):
        with self.assertRaisesRegex(ValueError, "unreachable"):
            validate_group_bounds({"bond": (0.6, 1.0)}, self.CLASSES, max_weight=0.25)

    def test_rejects_minimums_summing_above_one(self):
        with self.assertRaisesRegex(ValueError, "more than 100%"):
            validate_group_bounds({"equity": (0.7, 1.0), "bond": (0.4, 1.0)}, self.CLASSES, max_weight=0.6)

    def test_rejects_caps_that_prevent_full_investment(self):
        with self.assertRaisesRegex(ValueError, "fully invested"):
            validate_group_bounds({"equity": (0.0, 0.3), "bond": (0.0, 0.3)}, self.CLASSES, max_weight=0.3)


if __name__ == "__main__":
    unittest.main()
