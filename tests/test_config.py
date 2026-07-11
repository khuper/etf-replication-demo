import unittest

from src.config import ResearchConfig


class ResearchConfigTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_tickers(self):
        config = ResearchConfig(target=" psp ", assets=("spy", " QQQ ", "SPY"), max_weight=0.5)

        self.assertEqual(config.target, "PSP")
        self.assertEqual(config.assets, ("SPY", "QQQ"))

    def test_rejects_target_in_candidate_assets(self):
        config = ResearchConfig(target="SPY", assets=("SPY", "QQQ"), max_weight=0.5)

        with self.assertRaisesRegex(ValueError, "cannot also"):
            config.validate()

    def test_rejects_infeasible_max_weight(self):
        config = ResearchConfig(assets=("SPY", "QQQ", "GLD"), max_weight=0.25)

        with self.assertRaisesRegex(ValueError, "infeasible"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
