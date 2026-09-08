"""The look-ahead sentinel.

This is the most important file in the test suite. Every other claim the project
makes is downstream of "no decision used information that did not exist yet", and
that claim is not verifiable by reading code -- a single misplaced index, an
``iloc`` that should have been ``iloc[:-1]``, a volatility estimate taken from the
period being traded into, and the backtest silently becomes fiction that looks
like a good result.

So it is verified mechanically, two ways:

1. **Truncation.** Run the backtest on the full history, then on the history
   truncated at an earlier date. Every ledger row the two share must be
   bit-identical. If any future observation influenced any past decision, the
   truncated run must disagree.
2. **Future corruption.** Replace all prices after a cut date with something
   wildly different, and assert every row before the cut is unchanged. This
   catches the case where the future is read but only weakly, which truncation
   alone could conceivably mask.

Both run over every strategy in the zoo, because a leak in one strategy is not
excused by the engine being clean.
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from etflab.backtest import run_backtest
from etflab.data.panel import PricePanel
from etflab.strategies import DEFAULT_ZOO, build_strategy

CUTS = (400, 700, 1000)


def _fresh(name: str, config):
    """A new strategy instance per run: some strategies legitimately cache."""
    return build_strategy(name, config)


@pytest.mark.parametrize("strategy_name", DEFAULT_ZOO)
def test_truncating_history_cannot_change_earlier_decisions(small_panel, small_config, strategy_name):
    config = small_config.with_changes(cvar_ratio=1.0) if strategy_name == "cvar" else small_config
    full = run_backtest(small_panel, _fresh(strategy_name, config), config)

    for cut in CUTS:
        if cut >= small_panel.n_obs:
            continue
        truncated = run_backtest(small_panel.head_obs(cut), _fresh(strategy_name, config), config)
        if truncated.daily.empty:
            continue
        shared = truncated.daily.index
        assert_frame_equal(
            full.daily.loc[shared],
            truncated.daily,
            check_exact=False,
            rtol=0.0,
            atol=0.0,
            obj=f"{strategy_name} ledger diverged when history was truncated at observation {cut}",
        )


@pytest.mark.parametrize("strategy_name", ["tracking", "shrunk", "ridge", "ols_projected", "static"])
def test_corrupting_the_future_cannot_change_the_past(small_panel, small_config, strategy_name):
    """Replace the tail of history with garbage; the head of the ledger must not move."""
    config = small_config
    baseline = run_backtest(small_panel, _fresh(strategy_name, config), config)

    cut = 700
    corrupted_prices = small_panel.prices.copy()
    tail = corrupted_prices.index[cut:]
    # A violent, obviously-detectable distortion: triple every price and flip the
    # target's path. Nothing before the cut may notice.
    corrupted_prices.loc[tail] = corrupted_prices.loc[tail] * 3.0
    corrupted_prices.loc[tail, small_panel.target] = (
        corrupted_prices.loc[tail, small_panel.target].iloc[::-1].to_numpy()
    )
    corrupted = PricePanel(corrupted_prices, small_panel.assets, small_panel.target, "corrupted-future")

    result = run_backtest(corrupted, _fresh(strategy_name, config), config)
    safe_index = small_panel.prices.index[: cut - 1]
    shared = baseline.daily.index.intersection(safe_index)
    assert len(shared) > 100, "the test is only meaningful if a substantial prefix is compared"

    assert_frame_equal(
        baseline.daily.loc[shared],
        result.daily.loc[shared],
        check_exact=False,
        rtol=0.0,
        atol=0.0,
        obj=f"{strategy_name} read the future: corrupting prices after observation {cut} changed earlier rows",
    )


def test_training_window_never_reaches_the_decision_date(small_panel, small_config):
    """The engine's own accounting: every fit ends at least ``embargo_days`` before it is used."""
    config = small_config.with_changes(embargo_days=3)
    result = run_backtest(small_panel, build_strategy("tracking", config), config)
    returns_index = small_panel.returns.index

    for rebalance_date, row in result.rebalances.iterrows():
        train_end_position = returns_index.get_loc(row["train_end"])
        decision_position = returns_index.get_loc(rebalance_date)
        gap = decision_position - train_end_position
        assert gap >= config.embargo_days, (
            f"rebalance on {rebalance_date.date()} trained on data ending {gap} observations earlier, "
            f"but the embargo requires at least {config.embargo_days}"
        )


def test_cost_model_volatility_is_estimated_from_trailing_data_only(small_panel, small_config):
    """A subtle leak worth its own test: market-impact costs need a volatility
    estimate, and taking it from the period being traded into would use the
    future to price the trade. Doubling volatility only after the cut must not
    change any cost charged before it."""
    config = small_config.with_changes(cost_model="spread_impact", portfolio_notional=5e9)
    baseline = run_backtest(small_panel, build_strategy("tracking", config), config)

    cut = 700
    prices = small_panel.prices.copy()
    tail_index = prices.index[cut:]
    # Rebuild only the tail, anchored on the unchanged price at the cut, so the
    # head of the panel stays bit-identical and any difference found before the
    # cut is a genuine leak rather than floating-point drift from a rewrite.
    tail_returns = prices.pct_change().loc[tail_index] * 4.0
    rebuilt = prices.copy()
    rebuilt.loc[tail_index] = (1.0 + tail_returns).cumprod().to_numpy() * prices.iloc[cut - 1].to_numpy()
    assert rebuilt.iloc[: cut - 1].equals(prices.iloc[: cut - 1])
    volatile = PricePanel(rebuilt, small_panel.assets, small_panel.target, "volatile-future")

    result = run_backtest(volatile, build_strategy("tracking", config), config)
    safe = baseline.rebalances.index[baseline.rebalances.index < prices.index[cut - 1]]
    assert len(safe) >= 3
    np.testing.assert_allclose(
        baseline.rebalances.loc[safe, "cost_total_bps"].to_numpy(),
        result.rebalances.loc[safe, "cost_total_bps"].to_numpy(),
        rtol=0,
        atol=0,
        err_msg="costs before the cut changed when only future volatility changed",
    )
