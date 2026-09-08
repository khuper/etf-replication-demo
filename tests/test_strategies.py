"""Strategies: constraint compliance, numerical claims, and graceful degradation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etflab.strategies import (
    DEFAULT_ZOO,
    Constraints,
    FitContext,
    available_strategies,
    build_strategy,
    ledoit_wolf_shrinkage,
    project_onto_feasible,
    sample_cvar,
)


@pytest.fixture(scope="module")
def context(small_panel, small_config) -> FitContext:
    return FitContext(
        small_panel.asset_returns.iloc[:400],
        small_panel.target_returns.iloc[:400],
        None,
        Constraints(small_config.max_weight, None),
    )


@pytest.mark.parametrize("name", DEFAULT_ZOO)
def test_every_strategy_respects_every_constraint(context, small_config, name):
    config = small_config.with_changes(cvar_ratio=1.0) if name == "cvar" else small_config
    fit = build_strategy(name, config).fit(context)
    weights = fit.weights
    assert not fit.degraded, f"{name} degraded on a well-posed problem: {fit.status}"
    assert weights.sum() == pytest.approx(1.0, abs=1e-8), f"{name} is not fully invested"
    assert weights.min() >= -1e-9, f"{name} went short"
    assert weights.max() <= config.max_weight + 1e-8, f"{name} breached the position cap"


@pytest.mark.parametrize("name", DEFAULT_ZOO)
def test_every_strategy_respects_the_turnover_cap(context, small_config, name):
    config = small_config.with_changes(cvar_ratio=1.0) if name == "cvar" else small_config
    previous = np.zeros(context.n_assets)
    previous[0] = 1.0 - (context.n_assets - 1) * 0.01
    previous[1:] = 0.01
    constrained = FitContext(
        context.asset_returns,
        context.target_returns,
        previous,
        Constraints(config.max_weight, config.max_turnover),
    )
    fit = build_strategy(name, config).fit(constrained)
    distance = float(np.abs(fit.weights - previous).sum())
    assert distance <= config.max_turnover + 1e-6, (
        f"{name} traded {distance:.4f} against a cap of {config.max_turnover}"
    )


def test_projection_returns_the_input_when_it_is_already_feasible(context):
    feasible = np.full(context.n_assets, 1.0 / context.n_assets)
    fit = project_onto_feasible(feasible, context)
    np.testing.assert_allclose(fit.weights, feasible, atol=1e-7)


def test_projection_is_the_nearest_feasible_point(context):
    """A cheap but meaningful check: the projection must be closer to the target
    than any feasible point we can construct by hand."""
    desired = np.zeros(context.n_assets)
    desired[0] = 2.0
    desired[1] = -1.0
    fit = project_onto_feasible(desired, context)
    distance = float(np.linalg.norm(fit.weights - desired))
    for candidate in (
        np.full(context.n_assets, 1.0 / context.n_assets),
        np.eye(context.n_assets)[0] * 0.4 + np.full(context.n_assets, 0.6 / context.n_assets),
    ):
        candidate = candidate / candidate.sum()
        if candidate.max() <= context.constraints.max_weight + 1e-9:
            assert distance <= float(np.linalg.norm(candidate - desired)) + 1e-6


def test_sample_cvar_matches_a_hand_computation():
    """alpha * n is an exact integer: CVaR is the mean of the worst k losses."""
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert sample_cvar(losses, 0.2) == pytest.approx((10.0 + 9.0) / 2)
    assert sample_cvar(losses, 0.1) == pytest.approx(10.0)


def test_sample_cvar_interpolates_at_a_fractional_boundary():
    """The common shortcut -- averaging the worst ``int(alpha*n)`` -- silently
    evaluates CVaR at a different level. This is the case that exposes it."""
    losses = np.arange(1.0, 11.0)
    alpha = 0.15  # alpha * n = 1.5: one whole observation plus half of the next
    expected = (10.0 + 0.5 * 9.0) / 1.5
    assert sample_cvar(losses, alpha) == pytest.approx(expected)
    naive = losses[np.argsort(losses)[::-1]][: int(alpha * len(losses))].mean()
    assert naive != pytest.approx(expected), "the shortcut and the exact value must differ here"


def test_sample_cvar_equals_the_rockafellar_uryasev_optimum():
    """The estimator must agree with the linear program the constraint is built from."""
    import cvxpy as cp

    rng = np.random.default_rng(3)
    losses = rng.standard_normal(200)
    alpha = 0.05
    v = cp.Variable()
    z = cp.Variable(len(losses))
    problem = cp.Problem(
        cp.Minimize(v + (1.0 / (len(losses) * alpha)) * cp.sum(z)),
        [z >= 0, z >= losses - v],
    )
    problem.solve(solver=cp.CLARABEL)
    assert sample_cvar(losses, alpha) == pytest.approx(float(problem.value), rel=1e-6)


def test_cvar_constraint_actually_binds_when_it_is_tight(context, small_config):
    """A constraint that never binds is decoration; this proves it can bite."""
    loose = build_strategy("tracking", small_config.with_changes(cvar_ratio=5.0)).fit(context)
    tight = build_strategy("tracking", small_config.with_changes(cvar_ratio=0.70)).fit(context)
    assert not tight.degraded, f"expected a feasible tightening, got {tight.status}"
    assert float(np.abs(tight.weights - loose.weights).sum()) > 0.05

    returns = context.asset_returns.to_numpy()
    budget = sample_cvar(-context.target_returns.to_numpy(), small_config.cvar_alpha) * 0.70
    realised = sample_cvar(-(returns @ tight.weights), small_config.cvar_alpha)
    assert realised <= budget + 1e-6, "the solved portfolio must satisfy the CVaR budget it was given"


def test_an_impossible_cvar_budget_degrades_instead_of_crashing(context, small_config):
    """No long-only basket of these assets can hit a tail-risk budget this tight.
    The engine must record that and carry on, not raise."""
    fit = build_strategy("tracking", small_config.with_changes(cvar_ratio=0.30)).fit(context)
    assert fit.degraded
    assert "infeasible" in fit.status
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert fit.weights.min() >= 0.0


def test_shrinkage_is_a_convex_combination_and_stays_positive_definite():
    rng = np.random.default_rng(11)
    # Fewer observations than a stable estimate needs: exactly where shrinkage earns its keep.
    data = rng.standard_normal((40, 12))
    sigma, intensity = ledoit_wolf_shrinkage(data)
    assert 0.0 <= intensity <= 1.0
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0, "shrunk covariance must be positive definite"
    sample = np.cov(data, rowvar=False, ddof=0)
    assert np.linalg.cond(sigma) < np.linalg.cond(sample), "shrinkage must improve conditioning"


def test_shrinkage_intensity_falls_as_the_sample_grows():
    """On data with real factor structure the sample covariance eventually wins,
    so the shrinkage intensity must decay. (On i.i.d. data it correctly goes to
    1.0 instead, because there the constant-correlation target *is* the truth --
    which is why this test uses structured data.)"""
    loadings = np.random.default_rng(5).standard_normal((8, 3))

    def factor_data(n_obs: int) -> np.ndarray:
        rng = np.random.default_rng(12)
        factors = rng.standard_normal((n_obs, 3))
        return factors @ loadings.T + 0.5 * rng.standard_normal((n_obs, 8))

    intensities = [ledoit_wolf_shrinkage(factor_data(n))[1] for n in (40, 200, 2000)]
    assert intensities[0] > intensities[1] > intensities[2]
    assert intensities[-1] < 0.05


def test_infeasible_position_cap_is_caught_before_the_solver(small_panel):
    context = FitContext(
        small_panel.asset_returns.iloc[:200],
        small_panel.target_returns.iloc[:200],
        None,
        Constraints(max_weight=0.10),
    )
    with pytest.raises(ValueError, match="Infeasible"):
        context.constraints.validate(context.n_assets)


def test_top_correlation_fills_the_cap_with_the_closest_proxies(context, small_config):
    fit = build_strategy("top_correlation", small_config).fit(context)
    correlations = context.asset_returns.corrwith(context.target_returns)
    best = correlations.idxmax()
    weights = pd.Series(fit.weights, index=context.assets)
    assert weights[best] == pytest.approx(small_config.max_weight, abs=1e-4)


def test_static_strategy_reuses_its_first_solution(context, small_config):
    strategy = build_strategy("static", small_config)
    first = strategy.fit(context)
    later = FitContext(
        context.asset_returns.iloc[100:],
        context.target_returns.iloc[100:],
        first.weights,
        Constraints(small_config.max_weight, small_config.max_turnover),
    )
    second = strategy.fit(later)
    np.testing.assert_allclose(first.weights, second.weights, atol=1e-6)


def test_ols_reports_how_much_repair_its_raw_answer_needed(context, small_config):
    fit = build_strategy("ols_projected", small_config).fit(context)
    assert "repair_l1" in fit.detail
    assert fit.detail["repair_l1"] > 0, "unconstrained OLS on collinear assets should need repair"


def test_unknown_strategy_names_are_rejected(small_config):
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_strategy("magic", small_config)
    assert "tracking" in available_strategies()


def test_non_finite_training_data_is_refused(context, small_config):
    dirty = context.asset_returns.copy()
    dirty.iloc[5, 0] = np.nan
    broken = FitContext(dirty, context.target_returns, None, context.constraints)
    with pytest.raises(ValueError, match="non-finite"):
        build_strategy("tracking", small_config).fit(broken)
