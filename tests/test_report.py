import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from etf_replicator.backtest import BacktestResult
from etf_replicator.config import ReplicatorConfig
from etf_replicator.report import build_tearsheet


def synthetic_result(cost_bps: float = 10.0) -> BacktestResult:
    rng = np.random.default_rng(9)
    # Span 2019-2021 so the COVID stress window falls inside the sample.
    dates = pd.bdate_range("2019-06-03", "2021-06-30")
    T = len(dates)
    target = pd.Series(rng.normal(0.0003, 0.012, T), index=dates, name="TARGET")
    gross = target + rng.normal(0.0, 0.002, T)
    gross.name = "gross"

    rebalance_dates = dates[[0, T // 3, 2 * T // 3]]
    assets = ["AAA", "BBB", "CCC"]
    weights = pd.DataFrame(
        [[0.4, 0.35, 0.25], [0.45, 0.30, 0.25], [0.40, 0.30, 0.30]],
        index=rebalance_dates,
        columns=assets,
    )
    turnover = pd.Series([1.0, 0.15, 0.10], index=rebalance_dates, name="turnover")

    net = gross.copy()
    net.name = "net"
    net.loc[turnover.index] -= turnover * (cost_bps / 1e4)

    return BacktestResult(
        weights=weights,
        turnover=turnover,
        gross_returns=gross,
        net_returns=net,
        target_returns=target,
        cost_bps=cost_bps,
    )


class BuildTearsheetTests(unittest.TestCase):
    def test_tearsheet_contains_all_sections_and_charts(self):
        config = ReplicatorConfig(
            assets=["AAA", "BBB", "CCC"],
            target="TGT",
            start_date="2019-06-01",
            end_date="2021-06-30",
            asset_classes={"AAA": "equity", "BBB": "fixed_income", "CCC": "commodity"},
            group_bounds={"equity": (0.2, 0.6)},
        )
        result = synthetic_result()

        with tempfile.TemporaryDirectory() as out_dir:
            path = build_tearsheet(config, result, out_dir)

            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                html = fh.read()

            for section in [
                "Strategy specification",
                "Out-of-sample performance",
                "Tracking stability",
                "Monthly tracking consistency",
                "Behavior in stress windows",
                "Cost sensitivity",
                "Allocation",
            ]:
                self.assertIn(section, html)

            self.assertIn("TGT", html)
            self.assertIn("COVID Crash", html)  # stress window inside the sample
            self.assertIn("equity: 20%–60%", html)
            self.assertIn("10 bps (assumed)", html)
            self.assertGreaterEqual(html.count("data:image/png;base64,"), 5)

            for chart in [
                "cumulative_returns.png",
                "active_drawdown.png",
                "rolling_tracking.png",
                "group_allocation.png",
                "final_weights.png",
            ]:
                self.assertTrue(os.path.exists(os.path.join(out_dir, chart)), chart)


if __name__ == "__main__":
    unittest.main()
