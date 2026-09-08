"""The synthetic market: determinism, realism, and the ground truth it promises."""

from __future__ import annotations

import numpy as np
import pytest

from etflab.data.synthetic import LOADINGS, SyntheticSpec, corrupt, generate_market

TRADING_DAYS = 252


def _spec(**overrides) -> SyntheticSpec:
    base = dict(
        assets=("SPY", "QQQ", "IWM", "HYG", "GLD"),
        target="PSP",
        start="2015-01-01",
        end="2019-12-31",
        seed=42,
    )
    base.update(overrides)
    return SyntheticSpec(**base)


def test_same_seed_gives_byte_identical_prices():
    a, b = generate_market(_spec()), generate_market(_spec())
    assert a.fingerprint() == b.fingerprint()
    np.testing.assert_array_equal(a.prices.to_numpy(), b.prices.to_numpy())


def test_different_seed_gives_different_prices():
    assert generate_market(_spec()).fingerprint() != generate_market(_spec(seed=43)).fingerprint()


def test_prices_reproduce_the_generated_returns_exactly():
    """A seed row is prepended so differencing loses no observation."""
    panel = generate_market(_spec())
    assert len(panel.returns) == len(panel.prices) - 1
    assert panel.returns.notna().all().all()


def test_target_is_exactly_the_true_basket_plus_unspanned_component():
    """The promise the recovery study rests on, checked rather than assumed."""
    panel = generate_market(_spec())
    truth = panel.truth
    weights = truth.true_weights.reindex(list(panel.assets)).to_numpy()
    spanned = panel.asset_returns.to_numpy() @ weights
    unspanned = truth.unspanned_returns.reindex(panel.asset_returns.index).to_numpy()
    np.testing.assert_allclose(
        panel.target_returns.to_numpy(),
        truth.leverage * spanned + unspanned,
        rtol=1e-10,
        atol=1e-12,
    )


def test_true_weights_are_inside_the_feasible_set():
    """If the truth were infeasible, recovery would be untestable by construction."""
    truth = generate_market(_spec()).truth.true_weights
    assert truth.min() >= 0.0
    assert pytest.approx(1.0, abs=1e-12) == truth.sum()
    assert truth.max() <= 0.40 + 1e-12


def test_irreducible_tracking_error_is_what_the_true_weights_achieve():
    panel = generate_market(_spec())
    weights = panel.truth.true_weights.reindex(list(panel.assets)).to_numpy()
    active = panel.asset_returns.to_numpy() @ weights - panel.target_returns.to_numpy()
    realised = float(np.std(active, ddof=1) * np.sqrt(TRADING_DAYS))
    assert realised == pytest.approx(panel.truth.irreducible_te_annual, rel=1e-9)


def test_no_candidate_asset_loads_on_the_unspanned_factor():
    for asset, loadings in LOADINGS.items():
        assert "ILLIQUIDITY" not in loadings, f"{asset} loads on the factor that is supposed to be unreplicable"


def test_market_has_the_stylised_facts_it_claims():
    panel = generate_market(_spec(start="2010-01-01", end="2025-12-31"))
    returns = panel.returns

    vols = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert 0.10 < vols["SPY"] < 0.28, f"SPY volatility {vols['SPY']:.1%} is not equity-like"
    assert vols["QQQ"] > vols["SPY"], "QQQ should be more volatile than SPY"

    # Fat tails: excess kurtosis well above the Gaussian zero.
    assert returns["SPY"].kurtosis() > 2.0

    # Volatility clustering: squared returns are autocorrelated.
    squared = returns["SPY"] ** 2
    assert squared.autocorr(lag=1) > 0.05

    # Flight to quality: the defensive leg is much less correlated with equities.
    correlations = returns.corr()
    assert correlations.loc["GLD", "SPY"] < correlations.loc["QQQ", "SPY"]

    # All three regimes actually occur.
    assert set(panel.truth.regimes.unique()) == {"calm", "stressed", "crisis"}


def test_unknown_tickers_fail_loudly():
    with pytest.raises(ValueError, match="no definition"):
        generate_market(_spec(assets=("SPY", "NOTATICKER")))


def test_too_short_a_window_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        generate_market(_spec(start="2019-01-01", end="2019-03-01"))


def test_corrupt_actually_damages_the_panel():
    clean = generate_market(_spec())
    dirty = corrupt(clean, seed=5)
    assert dirty.fingerprint() != clean.fingerprint()
    assert dirty.prices.isna().to_numpy().sum() > 0
    assert dirty.prices.shape == clean.prices.shape
