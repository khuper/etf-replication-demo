"""The model zoo: every way of choosing weights that the horse race compares.

Design contract
---------------
Every strategy sees exactly the same information (a training window, the previous
holdings, the constraint set) and returns weights that satisfy exactly the same
constraints. That symmetry is the whole point. A comparison in which the fancy
model is allowed 40 assets and the naive baseline only 3, or in which the
baseline ignores the turnover cap, is not evidence -- it is a rigged demo.

So the naive strategies are not strawmen: an equal-weight or inverse-volatility
target that violates the position cap is *projected* onto the same feasible set
the optimiser lives in, using the same Euclidean projection. If the optimiser
still wins, it wins on merit.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import cvxpy as cp
import numpy as np
import pandas as pd

#: Solvers are tried in order. CLARABEL ships with cvxpy and handles both the
#: quadratic objective and the CVaR/turnover cones; the rest are fallbacks so a
#: single numerically awkward window cannot kill a whole run.
SOLVER_CHAIN: tuple[str, ...] = ("CLARABEL", "SCS", "ECOS")
_GOOD_STATUS = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
_TOL = 1e-5


@dataclass(frozen=True)
class Constraints:
    """The feasible set. Identical for every strategy, by construction."""

    max_weight: float
    max_turnover: float | None = None  # None on the first rebalance: nothing to trade from

    def validate(self, n_assets: int) -> None:
        if n_assets * self.max_weight < 1 - 1e-9:
            raise ValueError(
                f"Infeasible: {n_assets} assets capped at {self.max_weight:.1%} cannot fund a fully invested book."
            )


@dataclass(frozen=True)
class FitContext:
    """Everything a strategy is allowed to see when choosing weights.

    Nothing in here is dated later than the decision point. That is enforced by
    the backtest engine, and verified independently by the look-ahead sentinel in
    ``tests/test_lookahead.py`` -- because "we were careful" is not a control.
    """

    asset_returns: pd.DataFrame
    target_returns: pd.Series
    prev_weights: np.ndarray | None
    constraints: Constraints

    @property
    def assets(self) -> list[str]:
        return list(self.asset_returns.columns)

    @property
    def n_assets(self) -> int:
        return self.asset_returns.shape[1]


@dataclass(frozen=True)
class Fit:
    """A strategy's answer, plus enough diagnostics to audit it."""

    weights: np.ndarray
    status: str
    solver: str = "none"
    objective: float = float("nan")
    solve_seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return self.status not in {"optimal", "optimal_inaccurate", "analytic"}


class Strategy(Protocol):
    # Read-only properties rather than plain attributes: implementations are
    # frozen dataclasses, and a mutable protocol attribute would exclude them.
    @property
    def name(self) -> str:  # pragma: no cover - protocol
        ...

    @property
    def label(self) -> str:  # pragma: no cover - protocol
        ...

    def fit(self, ctx: FitContext) -> Fit:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Shared numerics
# --------------------------------------------------------------------------- #
def sample_cvar(losses: np.ndarray, alpha: float) -> float:
    """Exact sample CVaR at level ``alpha`` of a *loss* vector.

    Written to match the Rockafellar-Uryasev linear program at the same alpha,
    including the fractional weight on the boundary observation. The common
    shortcut -- "average the worst ``int(alpha*T)`` observations" -- silently
    evaluates CVaR at ``floor(alpha*T)/T`` instead of ``alpha``, so a constraint
    calibrated that way is not the constraint the write-up claims it is.
    """
    if losses.size == 0:
        raise ValueError("CVaR of an empty sample is undefined.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")
    ordered = np.sort(losses)[::-1]  # worst (largest) loss first
    n = ordered.size
    m = alpha * n
    k = int(np.floor(m))
    if k == 0:
        return float(ordered[0])
    head = ordered[:k].sum()
    remainder = m - k
    if remainder > 0 and k < n:
        head += remainder * ordered[k]
    return float(head / m)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of the sample covariance toward constant correlation.

    Returns ``(sigma, intensity)``. The sample covariance of ten highly
    correlated ETFs estimated on two years of daily data is a noisy object whose
    smallest eigenvalues are almost pure estimation error; a mean-variance style
    objective then loads precisely on those directions. Shrinking toward a
    constant-correlation target is the standard, parameter-free fix (Ledoit &
    Wolf, 2004), and the horse race is what decides whether it actually helps here.
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    if t < 2:
        raise ValueError("Shrinkage needs at least two observations.")
    x = x - x.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t

    variances = np.diag(sample)
    std = np.sqrt(np.maximum(variances, 1e-300))
    outer_std = np.outer(std, std)
    correlation = sample / outer_std
    off_diagonal = ~np.eye(n, dtype=bool)
    r_bar = float(correlation[off_diagonal].mean()) if n > 1 else 0.0

    target = r_bar * outer_std
    np.fill_diagonal(target, variances)

    # pi: asymptotic variance of the sample covariance entries.
    x2 = x**2
    pi_matrix = (x2.T @ x2) / t - sample**2
    pi_hat = float(pi_matrix.sum())

    # rho: covariance between sample entries and the shrinkage target.
    theta_ii = ((x2 * x).T @ x) / t - variances[:, None] * sample
    theta_jj = (x.T @ (x2 * x)) / t - variances[None, :] * sample
    rho_diag = float(np.diag(pi_matrix).sum())
    ratio_ji = np.divide(std[None, :], std[:, None], out=np.ones((n, n)), where=std[:, None] > 0)
    ratio_ij = np.divide(std[:, None], std[None, :], out=np.ones((n, n)), where=std[None, :] > 0)
    rho_off = (r_bar / 2.0) * (ratio_ji * theta_ii + ratio_ij * theta_jj)
    rho_hat = rho_diag + float(rho_off[off_diagonal].sum())

    gamma_hat = float(((target - sample) ** 2).sum())
    if gamma_hat <= 0:
        return sample, 0.0
    intensity = float(np.clip((pi_hat - rho_hat) / gamma_hat / t, 0.0, 1.0))
    return intensity * target + (1.0 - intensity) * sample, intensity


def _base_constraints(w: cp.Variable, ctx: FitContext) -> list[cp.Constraint]:
    cons: list[cp.Constraint] = [cp.sum(w) == 1, w >= 0, w <= ctx.constraints.max_weight]
    if ctx.prev_weights is not None and ctx.constraints.max_turnover is not None:
        cons.append(cp.norm(w - ctx.prev_weights, 1) <= ctx.constraints.max_turnover)
    return cons


def _solve(problem: cp.Problem) -> tuple[str, str]:
    """Solve with a fallback chain. Returns ``(status, solver_name)``."""
    last_status = "not_attempted"
    for solver in SOLVER_CHAIN:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=getattr(cp, solver))
        except (cp.SolverError, ValueError, ArithmeticError):
            last_status = "solver_error"
            continue
        last_status = str(problem.status)
        if problem.status in _GOOD_STATUS:
            return last_status.replace(" ", "_"), solver
    return last_status.replace(" ", "_"), "none"


def _fallback_weights(ctx: FitContext) -> np.ndarray:
    """What to hold when the optimiser cannot answer.

    Previous holdings if we have them (do nothing, incur no cost), otherwise the
    feasible equal-weight portfolio. Never an exception: an unsolvable window is
    an operational event to be recorded and survived, not a reason to lose the
    other forty-five rebalances.
    """
    if ctx.prev_weights is not None:
        return np.asarray(ctx.prev_weights, dtype=float).copy()
    n = ctx.n_assets
    return np.full(n, 1.0 / n)


def _finalise(raw: np.ndarray | None, ctx: FitContext, status: str, solver: str, objective: float, t0: float) -> Fit:
    """Validate, clean and package a solver result.

    The verification is not ceremonial. A solver that returns ``OPTIMAL`` with
    weights summing to 0.97 has told you the model was misspecified, and silently
    renormalising would hide it. We check, and we degrade loudly.
    """
    elapsed = time.perf_counter() - t0
    if raw is None or not np.isfinite(np.asarray(raw, dtype=float)).all():
        return Fit(_fallback_weights(ctx), "fallback_nonfinite", solver, objective, elapsed)
    w = np.asarray(raw, dtype=float).reshape(-1)

    violations = []
    if abs(w.sum() - 1.0) > _TOL:
        violations.append(f"sum={w.sum():.8f}")
    if w.min() < -_TOL:
        violations.append(f"min={w.min():.8f}")
    if w.max() > ctx.constraints.max_weight + _TOL:
        violations.append(f"max={w.max():.8f}>{ctx.constraints.max_weight}")
    if ctx.prev_weights is not None and ctx.constraints.max_turnover is not None:
        l1 = float(np.abs(w - ctx.prev_weights).sum())
        if l1 > ctx.constraints.max_turnover + _TOL:
            violations.append(f"L1={l1:.8f}>{ctx.constraints.max_turnover}")
    if violations:
        return Fit(
            _fallback_weights(ctx),
            "fallback_constraint_violation",
            solver,
            objective,
            elapsed,
            {"violations": violations},
        )

    w = np.clip(w, 0.0, ctx.constraints.max_weight)
    w[np.abs(w) < 1e-10] = 0.0
    total = w.sum()
    if total <= 0:
        return Fit(_fallback_weights(ctx), "fallback_degenerate", solver, objective, elapsed)
    w = w / total  # removes clipping residue only; the check above already passed
    return Fit(w, status, solver, objective, elapsed)


def project_onto_feasible(desired: np.ndarray, ctx: FitContext) -> Fit:
    """Euclidean projection of ``desired`` onto the feasible weight set.

    This is how heuristic strategies (equal weight, inverse vol, top-correlation)
    are held to the same constraints as the optimiser without being crippled by
    them: they state what they want, and we find the nearest thing they are
    allowed to hold.
    """
    t0 = time.perf_counter()
    w = cp.Variable(ctx.n_assets)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w - desired)), _base_constraints(w, ctx))
    status, solver = _solve(problem)
    if status not in {"optimal", "optimal_inaccurate"}:
        return Fit(_fallback_weights(ctx), f"fallback_{status}", solver, float("nan"), time.perf_counter() - t0)
    return _finalise(w.value, ctx, status, solver, float(problem.value), t0)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackingErrorStrategy:
    """Minimise in-sample squared tracking error, optionally under a CVaR cap.

    The headline model, and the one the rest of the zoo exists to challenge.
    ``cvar_ratio`` caps the portfolio's conditional value-at-risk at that multiple
    of the target's own CVaR over the same window -- a tail-risk budget rather
    than a variance one.
    """

    cvar_ratio: float | None = None
    cvar_alpha: float = 0.05
    name: str = "tracking"
    label: str = "Constrained tracking error"

    def fit(self, ctx: FitContext) -> Fit:
        t0 = time.perf_counter()
        r_assets = ctx.asset_returns.to_numpy(dtype=float)
        r_target = ctx.target_returns.to_numpy(dtype=float)
        if not (np.isfinite(r_assets).all() and np.isfinite(r_target).all()):
            raise ValueError("Training window contains non-finite returns; the data gates should have caught this.")

        w = cp.Variable(ctx.n_assets)
        active = r_assets @ w - r_target
        constraints = _base_constraints(w, ctx)

        if self.cvar_ratio is not None:
            periods = r_assets.shape[0]
            alpha = self.cvar_alpha
            budget = max(0.0, sample_cvar(-r_target, alpha)) * self.cvar_ratio
            v = cp.Variable()
            z = cp.Variable(periods)
            constraints += [
                z >= 0,
                z >= -(r_assets @ w) - v,
                v + (1.0 / (periods * alpha)) * cp.sum(z) <= budget,
            ]

        problem = cp.Problem(cp.Minimize(cp.sum_squares(active)), constraints)
        status, solver = _solve(problem)
        if status not in {"optimal", "optimal_inaccurate"}:
            return Fit(_fallback_weights(ctx), f"fallback_{status}", solver, float("nan"), time.perf_counter() - t0)
        return _finalise(w.value, ctx, status, solver, float(problem.value), t0)


@dataclass(frozen=True)
class RidgeTrackingStrategy:
    """Tracking error plus an L2 pull toward equal weight.

    The regulariser is the honest response to the multicollinearity gate: when
    SPY, QQQ and IWM span nearly the same direction, the unregularised solution
    picks among them on noise. ``ridge_lambda`` trades a little in-sample fit for
    weights that survive to the next window.
    """

    ridge_lambda: float = 5e-5
    name: str = "ridge"
    label: str = "Ridge-regularised tracking"

    def fit(self, ctx: FitContext) -> Fit:
        t0 = time.perf_counter()
        r_assets = ctx.asset_returns.to_numpy(dtype=float)
        r_target = ctx.target_returns.to_numpy(dtype=float)
        anchor = np.full(ctx.n_assets, 1.0 / ctx.n_assets)
        w = cp.Variable(ctx.n_assets)
        periods = r_assets.shape[0]
        objective = cp.sum_squares(r_assets @ w - r_target) / periods + self.ridge_lambda * cp.sum_squares(w - anchor)
        problem = cp.Problem(cp.Minimize(objective), _base_constraints(w, ctx))
        status, solver = _solve(problem)
        if status not in {"optimal", "optimal_inaccurate"}:
            return Fit(_fallback_weights(ctx), f"fallback_{status}", solver, float("nan"), time.perf_counter() - t0)
        return _finalise(w.value, ctx, status, solver, float(problem.value), t0)


@dataclass(frozen=True)
class ShrunkCovarianceStrategy:
    """Minimise active variance using a Ledoit-Wolf shrunk covariance.

    Algebraically the same objective as ``tracking`` -- minimise
    ``w'S_aa w - 2 w's_at`` -- but with the second-moment matrix estimated by
    shrinkage instead of raw sample moments. If shrinkage matters, this beats
    ``tracking`` out of sample while losing to it in sample. That is the test.
    """

    name: str = "shrunk"
    label: str = "Shrinkage covariance tracking"

    def fit(self, ctx: FitContext) -> Fit:
        t0 = time.perf_counter()
        joint = np.column_stack([ctx.asset_returns.to_numpy(dtype=float), ctx.target_returns.to_numpy(dtype=float)])
        sigma, intensity = ledoit_wolf_shrinkage(joint)
        sigma_aa = sigma[:-1, :-1]
        sigma_at = sigma[:-1, -1]
        w = cp.Variable(ctx.n_assets)
        objective = cp.quad_form(w, cp.psd_wrap(sigma_aa)) - 2.0 * sigma_at @ w
        problem = cp.Problem(cp.Minimize(objective), _base_constraints(w, ctx))
        status, solver = _solve(problem)
        if status not in {"optimal", "optimal_inaccurate"}:
            return Fit(_fallback_weights(ctx), f"fallback_{status}", solver, float("nan"), time.perf_counter() - t0)
        fit = _finalise(w.value, ctx, status, solver, float(problem.value), t0)
        return Fit(fit.weights, fit.status, fit.solver, fit.objective, fit.solve_seconds, {"shrinkage": intensity})


@dataclass(frozen=True)
class OLSProjectedStrategy:
    """Unconstrained least squares, then projected onto the feasible set.

    What a great many people actually do: regress the target on the candidates,
    notice the weights are negative and sum to 1.3, and "fix" them. Included
    because it is the realistic alternative to constrained optimisation, and
    because the size of the repair is itself a diagnostic.
    """

    name: str = "ols_projected"
    label: str = "OLS then repair"

    def fit(self, ctx: FitContext) -> Fit:
        x = ctx.asset_returns.to_numpy(dtype=float)
        y = ctx.target_returns.to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fit = project_onto_feasible(beta, ctx)
        repair = float(np.abs(fit.weights - beta).sum())
        return Fit(
            fit.weights,
            fit.status,
            fit.solver,
            fit.objective,
            fit.solve_seconds,
            {"raw_sum": float(beta.sum()), "raw_min": float(beta.min()), "repair_l1": repair},
        )


@dataclass(frozen=True)
class EqualWeightStrategy:
    """1/N, projected. The baseline that embarrasses more models than any other."""

    name: str = "equal_weight"
    label: str = "Equal weight"

    def fit(self, ctx: FitContext) -> Fit:
        return project_onto_feasible(np.full(ctx.n_assets, 1.0 / ctx.n_assets), ctx)


@dataclass(frozen=True)
class InverseVolStrategy:
    """Weights proportional to 1/sigma, projected. Risk-based, target-blind."""

    lookback: int = 252
    name: str = "inverse_vol"
    label: str = "Inverse volatility"

    def fit(self, ctx: FitContext) -> Fit:
        window = ctx.asset_returns.tail(self.lookback)
        vol = window.std(ddof=1).to_numpy(dtype=float)
        vol = np.where(vol > 0, vol, np.nanmedian(vol[vol > 0]) if (vol > 0).any() else 1.0)
        desired = (1.0 / vol) / (1.0 / vol).sum()
        return project_onto_feasible(desired, ctx)


@dataclass(frozen=True)
class TopCorrelationStrategy:
    """Fill the position cap with the highest-correlation proxies, then stop.

    The "no optimiser required" answer, and a surprisingly stiff benchmark: it
    captures most of the systematic exposure with none of the estimation error.
    """

    name: str = "top_correlation"
    label: str = "Top-correlation basket"

    def fit(self, ctx: FitContext) -> Fit:
        corr = ctx.asset_returns.corrwith(ctx.target_returns).fillna(0.0).to_numpy(dtype=float)
        order = np.argsort(-corr)
        cap = ctx.constraints.max_weight
        desired = np.zeros(ctx.n_assets)
        remaining = 1.0
        for idx in order:
            allocation = min(cap, remaining)
            desired[idx] = allocation
            remaining -= allocation
            if remaining <= 1e-12:
                break
        return project_onto_feasible(desired, ctx)


@dataclass(frozen=True)
class StaticTrackingStrategy:
    """Optimise once, then never trade again except to satisfy the caps.

    The zero-turnover control. If the walk-forward optimiser cannot beat a
    portfolio chosen once in 2015 and left alone, then all the rebalancing is
    paying costs for nothing -- which is a finding, and one worth publishing.
    """

    name: str = "static"
    label: str = "Optimise once, hold"
    #: Mutable state on a frozen dataclass is fine -- the binding is frozen, the
    #: dict is not. Excluded from equality so instances still compare by config.
    cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    def fit(self, ctx: FitContext) -> Fit:
        cache = self.cache
        key = "target"
        if key not in cache:
            # The buy-and-hold target is chosen once, ignoring the turnover cap,
            # because on the first call there is nothing to trade from.
            first = TrackingErrorStrategy().fit(
                FitContext(ctx.asset_returns, ctx.target_returns, None, Constraints(ctx.constraints.max_weight, None))
            )
            cache[key] = first.weights
        # Always project: if this strategy is ever handed prior holdings and a
        # turnover cap on its first call, it must respect them like everyone else.
        return project_onto_feasible(cache[key], ctx)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
StrategyFactory = Callable[[Any], Strategy]

STRATEGY_FACTORIES: dict[str, StrategyFactory] = {
    "tracking": lambda cfg: TrackingErrorStrategy(cvar_ratio=cfg.cvar_ratio, cvar_alpha=cfg.cvar_alpha),
    "cvar": lambda cfg: TrackingErrorStrategy(
        cvar_ratio=cfg.cvar_ratio or 1.0, cvar_alpha=cfg.cvar_alpha, name="cvar", label="CVaR-constrained tracking"
    ),
    "ridge": lambda cfg: RidgeTrackingStrategy(ridge_lambda=cfg.ridge_lambda or 5e-5),
    "shrunk": lambda cfg: ShrunkCovarianceStrategy(),
    "ols_projected": lambda cfg: OLSProjectedStrategy(),
    "equal_weight": lambda cfg: EqualWeightStrategy(),
    "inverse_vol": lambda cfg: InverseVolStrategy(),
    "top_correlation": lambda cfg: TopCorrelationStrategy(),
    "static": lambda cfg: StaticTrackingStrategy(),
}

#: The default horse race. Ordered from "most machinery" to "least", which is
#: also, more often than practitioners like, the order of out-of-sample results.
DEFAULT_ZOO: tuple[str, ...] = (
    "tracking",
    "cvar",
    "ridge",
    "shrunk",
    "ols_projected",
    "static",
    "top_correlation",
    "inverse_vol",
    "equal_weight",
)


def build_strategy(name: str, config: Any) -> Strategy:
    try:
        factory = STRATEGY_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy {name!r}; available: {sorted(STRATEGY_FACTORIES)}") from exc
    return factory(config)


def available_strategies() -> tuple[str, ...]:
    return tuple(sorted(STRATEGY_FACTORIES))
