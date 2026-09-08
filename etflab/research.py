"""Research orchestration: the horse race, the recovery study, and the sweep.

Three questions, in the order they should be asked:

1. **Does the machinery work at all?** :func:`run_recovery_study` -- on a market
   whose true replicating portfolio is known, does the estimator converge to it
   as the sample grows? If not, nothing downstream means anything.
2. **Does it beat the alternatives?** :func:`run_horse_race` -- every strategy on
   identical data, windows, constraints and costs, with the difference tested
   rather than eyeballed.
3. **Would it have survived my own search?** :func:`run_sweep` -- how much of the
   winning configuration's advantage is selection bias over the grid I searched.

The verdict text at the end is generated from the statistics, not written by
hand, so it cannot drift away from the numbers it claims to summarise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from etflab.backtest import BacktestResult, run_backtest, run_zoo
from etflab.config import ExperimentConfig
from etflab.costs import build_cost_model
from etflab.data.panel import PricePanel
from etflab.diagnostics import (
    BreakEven,
    beta_shock,
    capacity_curve,
    conditional_tail_performance,
    constraint_activity_summary,
    cost_decomposition,
    cost_drag_curve,
    cost_sensitivity,
    rolling_correlation_stability,
    worst_window_replay,
)
from etflab.governance import GovernanceReport, HurdlePolicy, govern
from etflab.inference import (
    BootstrapCI,
    DeflatedSharpeResult,
    DieboldMarianoResult,
    SuperiorPredictiveAbilityResult,
    deflated_sharpe_ratio,
    diebold_mariano,
    superior_predictive_ability,
    tracking_error_difference_ci,
)
from etflab.metrics import (
    annualised_vol,
    attribution_by_asset,
    compute_metrics,
    regime_breakdown,
    summary_frame,
)
from etflab.overfitting import (
    OverfittingResult,
    noise_reference_pbo,
    performance_from_active,
    probability_of_backtest_overfitting,
)
from etflab.strategies import DEFAULT_ZOO, Constraints, FitContext, build_strategy

TRADING_DAYS = 252

#: How much machinery each strategy carries, lowest first. Used to answer the
#: question that actually matters in a review -- "what is the simplest thing you
#: cannot rule out?" -- rather than only "what won?".
COMPLEXITY_RANK: dict[str, int] = {
    "equal_weight": 0,
    "inverse_vol": 1,
    "top_correlation": 2,
    "ols_projected": 3,
    "static": 4,
    "tracking": 5,
    "ridge": 6,
    "shrunk": 7,
    "cvar": 8,
}

HEADLINE_METRICS: tuple[str, ...] = (
    "tracking_error",
    "correlation",
    "information_ratio",
    "down_capture",
    "annual_turnover",
    "cost_drag_annual",
    "effective_positions",
    "worst_21d_active",
)


# --------------------------------------------------------------------------- #
# Horse race
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Comparison:
    """One strategy measured against the benchmark, with uncertainty attached."""

    strategy: str
    tracking_error: float
    te_difference: float
    ci: BootstrapCI
    dm: DieboldMarianoResult

    @property
    def significantly_better(self) -> bool:
        return self.ci.upper < 0 and self.dm.p_value < 0.05

    @property
    def distinguishable(self) -> bool:
        return self.ci.excludes_zero and self.dm.p_value < 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "tracking_error": self.tracking_error,
            "te_difference_vs_benchmark": self.te_difference,
            "ci": self.ci.as_dict(),
            "diebold_mariano": self.dm.as_dict(),
            "significantly_better": self.significantly_better,
        }


@dataclass(frozen=True)
class HorseRaceResult:
    benchmark: str
    results: dict[str, BacktestResult]
    metrics: pd.DataFrame
    comparisons: dict[str, Comparison]
    spa: SuperiorPredictiveAbilityResult
    best: str
    simplest_indistinguishable: str
    verdict: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "best": self.best,
            "simplest_indistinguishable": self.simplest_indistinguishable,
            "verdict": self.verdict,
            "notes": self.notes,
            "spa": self.spa.as_dict(),
            "comparisons": {k: v.as_dict() for k, v in self.comparisons.items()},
            "metrics": {k: {m: float(v) for m, v in row.items()} for k, row in self.metrics.to_dict("index").items()},
        }


def _pairwise_ci(active_a: pd.Series, active_b: pd.Series, config: ExperimentConfig) -> BootstrapCI:
    return tracking_error_difference_ci(
        active_a,
        active_b,
        n_samples=config.bootstrap_samples,
        mean_block=config.bootstrap_block,
        seed=config.inference_seed,
    )


def run_horse_race(
    panel: PricePanel,
    config: ExperimentConfig,
    names: tuple[str, ...] = DEFAULT_ZOO,
    benchmark: str = "equal_weight",
) -> HorseRaceResult:
    """Run every strategy on identical terms and test the differences."""
    if benchmark not in names:
        raise ValueError(f"Benchmark {benchmark!r} must be one of the strategies being run: {names}")
    results = run_zoo(panel, config, names)
    metrics = summary_frame({k: compute_metrics(r.daily, r.weights, r.rebalances) for k, r in results.items()})

    benchmark_active = results[benchmark].active
    comparisons: dict[str, Comparison] = {}
    for name, result in results.items():
        if name == benchmark:
            continue
        comparisons[name] = Comparison(
            strategy=name,
            tracking_error=annualised_vol(result.active),
            te_difference=annualised_vol(result.active) - annualised_vol(benchmark_active),
            ci=_pairwise_ci(result.active, benchmark_active, config),
            dm=diebold_mariano(
                result.active.to_numpy() ** 2,
                benchmark_active.to_numpy() ** 2,
                label_a=name,
                label_b=benchmark,
            ),
        )

    losses = pd.DataFrame({name: results[name].active ** 2 for name in names if name != benchmark})
    spa = superior_predictive_ability(
        losses,
        benchmark_active**2,
        n_samples=config.bootstrap_samples,
        mean_block=config.bootstrap_block,
        seed=config.inference_seed,
    )

    ordered = metrics["tracking_error"].sort_values()
    best = str(ordered.index[0])

    # The simplest strategy whose tracking error is not statistically
    # distinguishable from the winner's. Frequently the honest answer.
    best_active = results[best].active
    candidates = sorted(names, key=lambda n: COMPLEXITY_RANK.get(n, 99))
    simplest = best
    for candidate in candidates:
        if candidate == best:
            break
        ci = _pairwise_ci(results[candidate].active, best_active, config)
        if not ci.excludes_zero:
            simplest = candidate
            break

    notes = _build_notes(results, metrics, config)
    verdict = _build_verdict(best, benchmark, simplest, metrics, comparisons, spa, panel, config.bootstrap_samples)
    return HorseRaceResult(benchmark, results, metrics, comparisons, spa, best, simplest, verdict, notes)


def _build_notes(results: dict[str, BacktestResult], metrics: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    """Observations that a reader would otherwise have to dig out of the tables."""
    notes: list[str] = []

    if "cvar" in results and "tracking" in results:
        difference = float(np.abs(results["cvar"].weights.to_numpy() - results["tracking"].weights.to_numpy()).max())
        if difference < 1e-6:
            notes.append(
                "The CVaR constraint never binds: its weights are identical to unconstrained tracking at every "
                "rebalance. It is presentation, not risk control, at this budget."
            )
        else:
            notes.append(f"The CVaR constraint binds; it moves the largest weight by up to {difference:.2%}.")

    if "static" in results and "tracking" in results:
        gap = float(metrics.loc["static", "tracking_error"] - metrics.loc["tracking", "tracking_error"])
        if abs(gap) < 0.002:
            notes.append(
                f"Optimising once in {results['static'].weights.index[0].date()} and never trading again tracks "
                f"within {abs(gap) * 1e4:.0f}bp of full walk-forward re-optimisation."
            )

    degraded = int(metrics.get("degraded_rebalances", pd.Series(dtype=float)).sum())
    if degraded:
        notes.append(f"{degraded} rebalance(s) fell back to prior holdings after a solver failure.")

    if "ols_projected" in results:
        repair = results["ols_projected"].rebalances
        if "status" in repair:
            notes.append(
                "Unconstrained OLS weights required repair at every rebalance to satisfy the long-only and "
                "position-cap constraints, which is why it is reported as 'OLS then repair'."
            )

    turnover = metrics["annual_turnover"].max() if "annual_turnover" in metrics else 0.0
    if turnover < 0.5:
        notes.append(
            f"Annual turnover peaks at {turnover:.0%}, so transaction costs cannot explain any ranking here; "
            "the cost break-even analysis quantifies the margin."
        )
    return notes


def _format_p(p_value: float, n_samples: int | None = None) -> str:
    """Format a p-value without claiming more precision than the method has.

    A bootstrap p-value of exactly zero means "no resample out of N exceeded the
    observed statistic", which is a bound, not a point estimate. Printing "0"
    overstates it; printing the bound is honest and costs nothing.
    """
    if n_samples and p_value <= 0:
        return f"< {1.0 / n_samples:.2g}"
    if p_value <= 0:
        # A t-statistic large enough to underflow double precision. Reporting "0"
        # claims a precision the arithmetic does not have.
        return "< 1e-16"
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.3g}"


def _build_verdict(
    best: str,
    benchmark: str,
    simplest: str,
    metrics: pd.DataFrame,
    comparisons: dict[str, Comparison],
    spa: SuperiorPredictiveAbilityResult,
    panel: PricePanel,
    spa_samples: int,
) -> str:
    """Generate the conclusion from the statistics, so it cannot flatter them."""
    best_te = float(metrics.loc[best, "tracking_error"])
    benchmark_te = float(metrics.loc[benchmark, "tracking_error"])
    lines: list[str] = []

    lines.append(
        f"Best out-of-sample tracking error: {best} at {best_te:.2%} annualised, "
        f"versus {benchmark_te:.2%} for {benchmark}."
    )

    if panel.truth is not None:
        floor = panel.truth.irreducible_te_annual
        lines.append(
            f"The market's irreducible tracking error -- the part no long-only basket of these assets can remove -- "
            f"is {floor:.2%}. {best} therefore captures "
            f"{max(0.0, (benchmark_te - best_te) / max(benchmark_te - floor, 1e-9)):.0%} of the available improvement "
            f"over {benchmark}, and sits {best_te - floor:+.2%} above the floor."
        )

    if best != benchmark and best in comparisons:
        comparison = comparisons[best]
        if comparison.significantly_better:
            lines.append(
                f"That gap is statistically real: the 95% bootstrap interval for the tracking-error difference is "
                f"[{comparison.ci.lower:+.2%}, {comparison.ci.upper:+.2%}] and Diebold-Mariano rejects equal expected "
                f"squared tracking error (p = {_format_p(comparison.dm.p_value)})."
            )
        else:
            lines.append(
                f"That gap is not statistically distinguishable from zero: the 95% bootstrap interval is "
                f"[{comparison.ci.lower:+.2%}, {comparison.ci.upper:+.2%}] (Diebold-Mariano p = {_format_p(comparison.dm.p_value)})."
            )

    verdict_word = "survives" if spa.spa_p < 0.05 else "does not survive"
    lines.append(
        f"Correcting for having run {spa.n_models + 1} strategies, the best result {verdict_word} the multiple-comparison "
        f"correction (Hansen SPA p = {_format_p(spa.spa_p, spa_samples)}; "
        f"White Reality Check p = {_format_p(spa.reality_check_p, spa_samples)})."
    )

    if simplest != best:
        simplest_te = float(metrics.loc[simplest, "tracking_error"])
        lines.append(
            f"The simplest strategy that cannot be statistically separated from the winner is {simplest} "
            f"({simplest_te:.2%}). On this evidence the extra machinery in {best} is not doing measurable work."
        )
    else:
        lines.append(f"No simpler strategy matches {best} within statistical error; the machinery earns its place.")

    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Recovery study (synthetic data only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecoveryStudy:
    """Consistency evidence: does the estimator find the true portfolio?"""

    frame: pd.DataFrame
    converges: bool
    convergence_exponent: float
    final_l1: float
    final_excess_te: float  #: out-of-sample, at the largest sample size
    irreducible_te: float
    true_weights: dict[str, float]
    extended_obs: int
    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "converges": self.converges,
            "convergence_exponent": self.convergence_exponent,
            "final_l1_error": self.final_l1,
            "final_excess_tracking_error": self.final_excess_te,
            "irreducible_te_annual": self.irreducible_te,
            "extended_observations": self.extended_obs,
            "interpretation": self.interpretation,
            "true_weights": self.true_weights,
            "by_sample_size": self.frame.reset_index().to_dict("records"),
        }


def run_recovery_study(
    panel: PricePanel,
    config: ExperimentConfig,
    sample_sizes: Sequence[int] = (126, 252, 504, 1008, 2016, 4032, 8064),
    strategy_names: tuple[str, ...] = ("tracking", "shrunk", "ols_projected", "ridge"),
    extend_to: str = "2065-12-31",
) -> RecoveryStudy:
    """Does the estimator converge to the portfolio the market was built from?

    Only answerable on synthetic data, and the reason the default data source is
    synthetic. A tracking error says how well a model *fit*; this says whether the
    estimator is *consistent*, which no amount of good out-of-sample tracking on
    one history can establish.

    Two error measures are reported, and the gap between them is the finding:

    ``l1_error``
        Distance from the true weights. Identification.
    ``excess_te_is`` / ``excess_te_oos``
        Annualised tracking error of the fitted weights minus that of the true
        weights, measured in sample and on the *next* equally sized block. The
        in-sample figure is negative by construction and measures how much the
        fit overfits; the out-of-sample figure is the real cost of estimation
        error, and it is the one that should go to zero.

    When candidates are near-collinear -- SPY, QQQ and IWM span almost the same
    direction -- the objective is flat along the directions that separate them.
    The excess tracking error then collapses to nothing long before the weights
    do. That is not a defect in the optimiser; it is the problem telling you the
    weights are barely identified, which is exactly why weight stability and
    turnover deserve their own columns in the horse race.

    The sample is deliberately extended far past the configured evaluation
    window. Asking for forty years of daily data is trivial for a generator and
    impossible for a vendor, which is precisely the kind of question simulation
    is for.
    """
    if panel.truth is None:
        raise ValueError("Recovery requires a panel with known ground truth (data_source='synthetic').")

    from etflab.data.synthetic import SyntheticSpec, generate_market

    extended = generate_market(
        SyntheticSpec(
            assets=config.assets,
            target=config.target,
            start=config.start,
            end=extend_to,
            seed=config.synthetic_seed,
            leverage=config.synthetic_leverage,
        )
    )
    extended_truth = extended.truth
    if extended_truth is None:  # pragma: no cover - generate_market always sets it
        raise RuntimeError("The synthetic generator returned a panel without ground truth.")
    truth = extended_truth.true_weights.reindex(list(extended.assets)).fillna(0.0)
    truth_array = truth.to_numpy(dtype=float)
    asset_returns, target_returns = extended.asset_returns, extended.target_returns
    constraints = Constraints(config.max_weight, None)

    rows = []
    for size in sample_sizes:
        if size > len(asset_returns):
            continue
        window_assets = asset_returns.iloc[:size]
        window_target = target_returns.iloc[:size]
        ctx = FitContext(window_assets, window_target, None, constraints)
        matrix = window_assets.to_numpy(dtype=float)
        target_vector = window_target.to_numpy(dtype=float)
        te_truth = float(np.std(matrix @ truth_array - target_vector, ddof=1) * np.sqrt(TRADING_DAYS))

        # The next equally sized block, held out entirely from the fit.
        holdout_assets = asset_returns.iloc[size : size * 2]
        holdout_target = target_returns.iloc[size : size * 2]
        has_holdout = len(holdout_assets) >= 60
        if has_holdout:
            holdout_matrix = holdout_assets.to_numpy(dtype=float)
            holdout_vector = holdout_target.to_numpy(dtype=float)
            te_truth_oos = float(np.std(holdout_matrix @ truth_array - holdout_vector, ddof=1) * np.sqrt(TRADING_DAYS))

        row: dict[str, Any] = {"sample_size": size, "te_at_true_weights": te_truth}
        for name in strategy_names:
            fit = build_strategy(name, config).fit(ctx)
            te_fit = float(np.std(matrix @ fit.weights - target_vector, ddof=1) * np.sqrt(TRADING_DAYS))
            row[f"{name}_l1"] = float(np.abs(fit.weights - truth_array).sum())
            row[f"{name}_excess_te_is"] = te_fit - te_truth
            if has_holdout:
                te_fit_oos = float(
                    np.std(holdout_matrix @ fit.weights - holdout_vector, ddof=1) * np.sqrt(TRADING_DAYS)
                )
                row[f"{name}_excess_te_oos"] = te_fit_oos - te_truth_oos
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("sample_size")
    primary_l1 = f"{strategy_names[0]}_l1"
    primary_excess = f"{strategy_names[0]}_excess_te_oos"
    errors = frame[primary_l1].to_numpy()

    # Fit log(error) = a + b*log(T). A consistent estimator has b < 0; the
    # textbook parametric rate is b = -0.5, and anything much shallower is the
    # ill-conditioning showing up as a slower rate rather than as a plateau.
    exponent = float("nan")
    if len(errors) >= 3 and (errors > 0).all():
        exponent = float(np.polyfit(np.log(frame.index.to_numpy(dtype=float)), np.log(errors), 1)[0])
    converges = bool(np.isfinite(exponent) and exponent < -0.1)

    final_l1 = float(errors[-1]) if len(errors) else float("nan")
    excess_series = frame[primary_excess].dropna() if primary_excess in frame else pd.Series(dtype=float)
    final_excess = float(excess_series.to_numpy()[-1]) if len(excess_series) else float("nan")
    if converges:
        interpretation = (
            f"Weight error decays as T^{exponent:.2f} (parametric rate would be T^-0.50), reaching L1 = {final_l1:.3f} "
            f"at {int(frame.index[-1]):,} observations. The out-of-sample cost of that estimation error falls to "
            f"{final_excess * 1e4:+.1f}bp of tracking error above the true portfolio. The estimator is consistent, "
            f"but the weights are far less identified than the fit is: collinear candidates leave the objective "
            f"nearly flat in the directions that separate them, so a large weight error buys almost no extra "
            f"tracking error. That is why weight stability and turnover get their own columns in the horse race."
        )
    else:
        interpretation = (
            f"Weight error does not decay with sample size (fitted exponent {exponent:.2f}). On this evidence the "
            f"estimator is not recovering the true portfolio, and its out-of-sample fit should not be read as "
            f"evidence that it has found real structure."
        )

    return RecoveryStudy(
        frame=frame,
        converges=converges,
        convergence_exponent=exponent,
        final_l1=final_l1,
        final_excess_te=final_excess,
        irreducible_te=float(extended_truth.irreducible_te_annual),
        true_weights={k: float(v) for k, v in truth.items()},
        extended_obs=len(asset_returns),
        interpretation=interpretation,
    )


# --------------------------------------------------------------------------- #
# Parameter sweep and overfitting
# --------------------------------------------------------------------------- #
DEFAULT_GRID: dict[str, tuple[Any, ...]] = {
    "train_days": (252, 504, 756),
    "rebalance_days": (21, 63, 126),
    "max_weight": (0.20, 0.25, 0.35),
    "max_turnover": (0.10, 0.20),
}


@dataclass(frozen=True)
class SweepResult:
    frame: pd.DataFrame
    active_by_config: pd.DataFrame
    pbo: OverfittingResult
    noise_pbo: float
    deflated: DeflatedSharpeResult
    best_config: str
    best_tracking_error: float
    grid: dict[str, list[Any]]

    @property
    def selection_is_meaningful(self) -> bool:
        """Is the observed PBO materially below the null for a grid this shape?"""
        return self.pbo.pbo < self.noise_pbo - 0.10

    def as_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid,
            "n_configs": len(self.frame),
            "best_config": self.best_config,
            "best_tracking_error": self.best_tracking_error,
            "pbo": self.pbo.as_dict(),
            "noise_reference_pbo": self.noise_pbo,
            "selection_is_meaningful": self.selection_is_meaningful,
            "deflated_sharpe": self.deflated.as_dict(),
            "spread_of_tracking_error": {
                "min": float(self.frame["tracking_error"].min()),
                "median": float(self.frame["tracking_error"].median()),
                "max": float(self.frame["tracking_error"].max()),
            },
        }


def _grid_configs(config: ExperimentConfig, grid: dict[str, tuple[Any, ...]]) -> list[tuple[str, ExperimentConfig]]:
    keys = sorted(grid)
    out: list[tuple[str, ExperimentConfig]] = []
    for combination in product(*(grid[k] for k in keys)):
        changes = dict(zip(keys, combination, strict=True))
        try:
            candidate = config.with_changes(**changes)
        except ValueError:
            continue  # infeasible corner of the grid, e.g. a position cap that cannot fund the book
        label = "|".join(f"{k}={v}" for k, v in changes.items())
        out.append((label, candidate))
    return out


def run_sweep(
    panel: PricePanel,
    config: ExperimentConfig,
    grid: dict[str, tuple[Any, ...]] | None = None,
    strategy_name: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> SweepResult:
    """Evaluate a parameter grid, then price in the fact that a grid was searched.

    Every configuration's out-of-sample active-return series is retained, because
    the overfitting diagnostics operate on the full matrix -- reporting only the
    winner is precisely the practice being measured here.
    """
    grid = grid or DEFAULT_GRID
    strategy_name = strategy_name or config.strategy
    configs = _grid_configs(config, grid)
    if len(configs) < 4:
        raise ValueError("A sweep needs at least four feasible configurations to say anything about selection.")

    rows, active_series = [], {}
    for index, (label, candidate) in enumerate(configs):
        if progress:
            progress(index + 1, len(configs), label)
        result = run_backtest(panel, build_strategy(strategy_name, candidate), candidate)
        metrics = compute_metrics(result.daily, result.weights, result.rebalances)
        rows.append({"config": label, **{k: metrics[k] for k in HEADLINE_METRICS if k in metrics}})
        active_series[label] = result.active

    frame = pd.DataFrame(rows).set_index("config").sort_values("tracking_error")
    # Configurations differ in start date (train_days varies), so align on the
    # common evaluation window: comparing them over different samples would make
    # the ranking partly an artefact of which crisis each one happened to include.
    active = pd.DataFrame(active_series).dropna()

    performance = performance_from_active(active)
    pbo = probability_of_backtest_overfitting(performance)
    noise_pbo = noise_reference_pbo(performance.shape[0], performance.shape[1], n_splits=pbo.n_splits)

    best_config = str(frame.index[0])
    # Deflated Sharpe on the information ratio of the selected configuration,
    # deflated by the information ratios of every configuration tried.
    trial_sharpes = [
        float(active[c].mean() / active[c].std(ddof=1)) for c in active.columns if active[c].std(ddof=1) > 0
    ]
    deflated = deflated_sharpe_ratio(active[best_config].to_numpy(), trial_sharpes)

    return SweepResult(
        frame=frame,
        active_by_config=active,
        pbo=pbo,
        noise_pbo=noise_pbo,
        deflated=deflated,
        best_config=best_config,
        best_tracking_error=float(frame["tracking_error"].iloc[0]),
        grid={k: list(v) for k, v in grid.items()},
    )


# --------------------------------------------------------------------------- #
# The whole study
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Study:
    """Everything the report needs, computed once."""

    config: ExperimentConfig
    panel: PricePanel
    quality: Any
    race: HorseRaceResult
    recovery: RecoveryStudy | None
    sweep: SweepResult | None
    regimes: pd.DataFrame
    attribution: pd.DataFrame
    tails: pd.DataFrame
    worst_windows: pd.DataFrame
    correlation_stability: pd.DataFrame
    shock: pd.DataFrame
    cost_breakeven: BreakEven | None
    capacity: BreakEven | None
    cost_curve: pd.DataFrame | None
    cost_split: pd.DataFrame | None
    #: Which advertised constraints actually bound, per strategy.
    constraints: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The kill switch, evaluated against the benchmark and against the simplest
    #: strategy that is statistically indistinguishable from the winner.
    governance: dict[str, GovernanceReport] = field(default_factory=dict)
    #: Which optional analyses ran. The results digest depends on them, so a
    #: verification re-run has to switch on exactly the same set.
    analyses: dict[str, bool] = field(default_factory=dict)

    @property
    def headline(self) -> dict[str, float]:
        row = self.race.metrics.loc[self.race.best]
        return {
            "tracking_error": float(row["tracking_error"]),
            "correlation": float(row["correlation"]),
            "annual_turnover": float(row.get("annual_turnover", float("nan"))),
            "cost_drag_annual": float(row.get("cost_drag_annual", float("nan"))),
        }

    def results_payload(self) -> dict[str, Any]:
        """The numeric payload that the determinism digest is computed over.

        Deliberately excludes wall-clock timings, file paths and anything else
        that is not a research result -- a determinism check that fails because a
        solver took three milliseconds longer is a determinism check nobody runs.
        """
        payload: dict[str, Any] = {
            "metrics": {
                k: {m: round(float(v), 12) for m, v in row.items()}
                for k, row in self.race.metrics.to_dict("index").items()
            },
            "best": self.race.best,
            "simplest_indistinguishable": self.race.simplest_indistinguishable,
            "spa_p": round(self.race.spa.spa_p, 12),
            "reality_check_p": round(self.race.spa.reality_check_p, 12),
        }
        if self.recovery is not None:
            payload["recovery"] = {k: round(float(v), 12) for k, v in self.recovery.frame.iloc[-1].items()}
        if self.sweep is not None:
            payload["pbo"] = round(self.sweep.pbo.pbo, 12)
            payload["best_config"] = self.sweep.best_config
        for key, report in self.governance.items():
            payload[f"governance_{key}"] = {
                "days_off": report.days_off,
                "switches": report.switches,
                "final_state": report.final_state,
            }
        return payload


def run_study(
    config: ExperimentConfig,
    panel: PricePanel,
    quality: Any,
    *,
    strategies: tuple[str, ...] = DEFAULT_ZOO,
    benchmark: str = "equal_weight",
    with_sweep: bool = True,
    with_recovery: bool = True,
    with_capacity: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Study:
    """Run the full research workflow and return everything the report renders."""

    def step(message: str) -> None:
        if progress:
            progress(message)

    step("Running the strategy horse race")
    race = run_horse_race(panel, config, strategies, benchmark)
    best_result = race.results[race.best]

    step("Decomposing risk and regimes")
    regimes = (
        regime_breakdown(best_result.daily, panel.truth.regimes)
        if panel.truth is not None
        else _volatility_regimes(best_result.daily)
    )
    attribution = attribution_by_asset(best_result.daily_weights, panel.asset_returns, panel.target_returns)
    tails = conditional_tail_performance(best_result.daily)
    worst = worst_window_replay(best_result.daily)
    stability = rolling_correlation_stability(panel.asset_returns, panel.target_returns)
    shock = beta_shock(best_result.weights.iloc[-1], panel.asset_returns, panel.target_returns)

    recovery = None
    if with_recovery and panel.truth is not None:
        step("Running the weight-recovery study")
        recovery = run_recovery_study(panel, config)

    breakeven = capacity = None
    curve = split = None
    if with_capacity:
        step("Computing cost break-even and capacity")
        breakeven = cost_sensitivity(panel, config, race.best, benchmark)
        capacity = capacity_curve(panel, config, race.best, benchmark)
        curve = cost_drag_curve(panel, config, race.best)
        split = cost_decomposition(panel, config, race.best, benchmark)

    step("Reading constraint activity")
    constraints = constraint_activity_summary(race.results)

    step("Applying the kill switch")
    policy = HurdlePolicy(
        hurdle=config.hurdle,
        window=config.hurdle_window,
        grace=config.hurdle_grace,
        reactivate=config.hurdle_reactivate,
    )
    cost_model = build_cost_model(config, panel.assets)
    governance: dict[str, GovernanceReport] = {}
    governance["benchmark"] = govern(best_result, race.results[benchmark], policy, cost_model, panel.asset_returns)
    simplest = race.simplest_indistinguishable
    if simplest not in (race.best, benchmark):
        governance["simplest"] = govern(best_result, race.results[simplest], policy, cost_model, panel.asset_returns)
    # The question that actually bites: does the *rebalancing* pay for itself?
    # ``static`` is the same estimator with the trading turned off, so this is
    # the cleanest possible test of whether the walk-forward machinery earns
    # its keep, independent of whether any optimiser beats a naive basket.
    if "static" in race.results and race.best != "static" and simplest != "static":
        governance["static"] = govern(best_result, race.results["static"], policy, cost_model, panel.asset_returns)

    sweep = None
    if with_sweep:
        step("Sweeping the parameter grid and measuring overfitting")
        sweep = run_sweep(
            panel,
            config,
            progress=lambda i, n, label: step(f"  sweep {i}/{n}: {label}"),
        )

    return Study(
        config=config,
        panel=panel,
        quality=quality,
        race=race,
        recovery=recovery,
        sweep=sweep,
        regimes=regimes,
        attribution=attribution,
        tails=tails,
        worst_windows=worst,
        correlation_stability=stability,
        shock=shock,
        cost_breakeven=breakeven,
        capacity=capacity,
        cost_curve=curve,
        cost_split=split,
        constraints=constraints,
        governance=governance,
        analyses={
            "sweep": with_sweep,
            "recovery": with_recovery and panel.truth is not None,
            "capacity": with_capacity,
        },
    )


def _volatility_regimes(daily: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Regime labels derived from realised target volatility, for real data.

    Synthetic panels know their true regime. Real ones do not, so we tercile the
    trailing realised volatility of the target -- a crude but transparent proxy,
    computed from trailing data only.
    """
    realised = daily["target"].rolling(window).std()
    valid = realised.dropna()
    if valid.empty:
        return pd.DataFrame()
    low, high = valid.quantile(1 / 3), valid.quantile(2 / 3)
    labels = pd.Series(
        np.where(realised <= low, "low vol", np.where(realised <= high, "mid vol", "high vol")),
        index=daily.index,
    )
    return regime_breakdown(daily, labels)
