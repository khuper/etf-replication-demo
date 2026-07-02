import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from etf_replicator.data import fetch_prices, split_assets_target, to_log_returns, to_returns


def multiindex_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    columns = pd.MultiIndex.from_product([["Open", "Close"], ["AAA", "BBB", "TARGET"]])
    return pd.DataFrame(
        [
            [99, 101, 103, 100, 102, 104],
            [100, 102, 104, 101, 103, 105],
            [101, 103, 105, 102, 104, 106],
            [102, 104, 106, 103, 105, 107],
        ],
        index=dates,
        columns=columns,
        dtype=float,
    )


class FetchPricesTests(unittest.TestCase):
    def test_extracts_close_from_multiindex(self):
        with patch("etf_replicator.data.yf.download", return_value=multiindex_frame()):
            prices = fetch_prices(["AAA", "BBB", "TARGET"], "2024-01-01", "2024-01-05")

        self.assertEqual(list(prices.columns), ["AAA", "BBB", "TARGET"])
        self.assertEqual(prices.shape, (4, 3))

    def test_raises_on_empty_download(self):
        with patch("etf_replicator.data.yf.download", return_value=pd.DataFrame()):
            with self.assertRaisesRegex(ValueError, "No data downloaded"):
                fetch_prices(["AAA"], "2024-01-01", "2024-01-05")

    def test_raises_when_no_overlapping_history(self):
        frame = multiindex_frame()
        frame.loc[:, ("Close", "AAA")] = np.nan
        with patch("etf_replicator.data.yf.download", return_value=frame):
            with self.assertRaisesRegex(ValueError, "empty after dropping NAs"):
                fetch_prices(["AAA", "BBB", "TARGET"], "2024-01-01", "2024-01-05")


class ReturnsTests(unittest.TestCase):
    PRICES = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]}, index=pd.date_range("2024-01-01", periods=3))

    def test_simple_returns(self):
        returns = to_returns(self.PRICES)
        self.assertAlmostEqual(returns.iloc[0, 0], 0.10)
        self.assertAlmostEqual(returns.iloc[1, 0], -0.10)

    def test_log_returns(self):
        returns = to_log_returns(self.PRICES)
        self.assertAlmostEqual(returns.iloc[0, 0], np.log(1.10))

    def test_split_assets_target(self):
        returns = pd.DataFrame({"AAA": [0.01], "BBB": [0.02], "TARGET": [0.015]})
        assets, target = split_assets_target(returns, ["AAA", "BBB"], "TARGET")

        self.assertEqual(list(assets.columns), ["AAA", "BBB"])
        self.assertEqual(target.name, "TARGET")


if __name__ == "__main__":
    unittest.main()
