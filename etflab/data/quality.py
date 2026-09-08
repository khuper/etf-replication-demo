"""Data quality gates: the checks that run before anybody is allowed to optimise.

A backtest cannot tell you that its input was wrong. It will happily produce a
tracking error of 0.4% on a panel where a ticker went stale for a month, because
a flat price series is a very easy thing to track. These gates are the control
that catches that, and they run on every panel -- synthetic, cached, or live --
before a single weight is estimated.

Each gate returns ``ok`` / ``warn`` / ``fail``. A ``fail`` aborts the run unless
the operator explicitly passes ``--allow-degraded``, in which case the degraded
status is stamped into the run manifest so nobody can later mistake the result
for a clean one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from etflab.data.panel import PricePanel

Status = Literal["ok", "warn", "fail"]
_ORDER: dict[str, int] = {"ok": 0, "warn": 1, "fail": 2}

#: Ratios that a raw (unadjusted) split or reverse split would produce. A price
#: series that jumps by one of these overnight is almost certainly a corporate
#: action that the adjustment pipeline missed, not a 50% one-day move.
SPLIT_RATIOS = (0.5, 1 / 3, 0.25, 0.2, 2.0, 3.0, 4.0, 5.0)


@dataclass(frozen=True)
class GateResult:
    name: str
    status: Status
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "summary": self.summary, "evidence": self.evidence}


@dataclass(frozen=True)
class QualityReport:
    gates: tuple[GateResult, ...]
    panel_summary: dict[str, Any]

    @property
    def status(self) -> Status:
        worst = max((_ORDER[g.status] for g in self.gates), default=0)
        return ("ok", "warn", "fail")[worst]  # type: ignore[return-value]

    @property
    def ok(self) -> bool:
        return self.status != "fail"

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(g for g in self.gates if g.status == "fail")

    @property
    def warnings(self) -> tuple[GateResult, ...]:
        return tuple(g for g in self.gates if g.status == "warn")

    def raise_if_failed(self) -> None:
        if self.failures:
            lines = "\n".join(f"  - [{g.name}] {g.summary}" for g in self.failures)
            raise DataQualityError(
                f"{len(self.failures)} data quality gate(s) failed:\n{lines}\n"
                "Re-run with --allow-degraded to proceed anyway (the manifest will record it)."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "panel": self.panel_summary,
            "gates": [g.as_dict() for g in self.gates],
        }


class DataQualityError(RuntimeError):
    """Raised when a blocking data-quality gate fails."""


# --------------------------------------------------------------------------- #
# Individual gates
# --------------------------------------------------------------------------- #
def _gate_non_positive(prices: pd.DataFrame) -> GateResult:
    bad = (prices <= 0).sum()
    total = int(bad.sum())
    if total == 0:
        return GateResult("non_positive_prices", "ok", "All prices are strictly positive.")
    return GateResult(
        "non_positive_prices",
        "fail",
        f"{total} non-positive price observation(s); returns would be undefined or nonsensical.",
        {"per_ticker": {k: int(v) for k, v in bad[bad > 0].items()}},
    )


def _gate_missing(prices: pd.DataFrame) -> GateResult:
    missing = prices.isna().sum()
    total = int(missing.sum())
    if total == 0:
        return GateResult("missing_observations", "ok", "No missing prices after alignment.")
    share = total / prices.size
    status: Status = "fail" if share > 0.01 else "warn"
    return GateResult(
        "missing_observations",
        status,
        f"{total} missing price observation(s) ({share:.2%} of the panel).",
        {"per_ticker": {k: int(v) for k, v in missing[missing > 0].items()}},
    )


def _gate_stale(prices: pd.DataFrame, max_run: int = 5) -> GateResult:
    """Consecutive identical closes.

    Genuinely flat closes happen (a quiet day in a bond ETF), so a run of two is
    unremarkable. A run beyond a week is a stopped feed, and it flatters every
    risk number computed downstream.
    """
    worst: dict[str, int] = {}
    for column in prices.columns:
        series = prices[column]
        same = series.eq(series.shift())
        # Length of each consecutive True run.
        groups = (~same).cumsum()
        run_lengths = same.groupby(groups).sum()
        worst[column] = int(run_lengths.max()) if len(run_lengths) else 0
    offenders = {k: v for k, v in worst.items() if v >= max_run}
    if not offenders:
        return GateResult("stale_prices", "ok", f"No repeated-close run reached {max_run} days.")
    status: Status = "fail" if max(offenders.values()) >= 2 * max_run else "warn"
    return GateResult(
        "stale_prices",
        status,
        f"{len(offenders)} ticker(s) have a repeated-close run of at least {max_run} days.",
        {"longest_run": offenders},
    )


def _gate_split_jumps(prices: pd.DataFrame, tolerance: float = 0.02) -> GateResult:
    """Overnight ratios that look like an unadjusted corporate action."""
    ratios = prices / prices.shift()
    hits: dict[str, list[str]] = {}
    for column in prices.columns:
        series = ratios[column].dropna()
        for candidate in SPLIT_RATIOS:
            mask = (series - candidate).abs() < tolerance * candidate
            for date in series.index[mask]:
                hits.setdefault(column, []).append(f"{date.date()}:x{series[date]:.3f}")
    if not hits:
        return GateResult("corporate_action_jumps", "ok", "No overnight moves resembling an unadjusted split.")
    return GateResult(
        "corporate_action_jumps",
        "fail",
        f"{sum(len(v) for v in hits.values())} overnight move(s) match a split ratio; adjustment is likely broken.",
        {"hits": {k: v[:10] for k, v in hits.items()}},
    )


def _gate_extreme_returns(returns: pd.DataFrame, threshold: float = 0.25) -> GateResult:
    extreme = returns.abs() > threshold
    total = int(extreme.to_numpy().sum())
    if total == 0:
        return GateResult("extreme_returns", "ok", f"No daily move exceeded {threshold:.0%}.")
    where = {
        column: [f"{d.date()}:{returns.loc[d, column]:+.1%}" for d in returns.index[extreme[column]]][:5]
        for column in returns.columns
        if extreme[column].any()
    }
    return GateResult(
        "extreme_returns",
        "warn",
        f"{total} daily move(s) beyond {threshold:.0%}; verify these are real before trusting risk numbers.",
        {"examples": where},
    )


def _gate_zero_variance(returns: pd.DataFrame) -> GateResult:
    flat = returns.std(ddof=1)
    dead = flat[flat <= 1e-12]
    if dead.empty:
        return GateResult("zero_variance", "ok", "Every series has non-zero variance.")
    return GateResult(
        "zero_variance",
        "fail",
        f"{len(dead)} series have zero variance and cannot be used in a covariance estimate.",
        {"tickers": list(dead.index)},
    )


def _gate_calendar(prices: pd.DataFrame, max_gap_days: int = 10) -> GateResult:
    """Look for holes in the trading calendar.

    Roughly ten business days of nothing is either a data outage or an exchange
    closure worth knowing about (9/11 was four; the 1985 hurricane one).
    """
    gaps = prices.index.to_series().diff().dt.days.dropna()
    big = gaps[gaps > max_gap_days]
    expected = len(pd.bdate_range(prices.index[0], prices.index[-1]))
    coverage = len(prices) / expected if expected else 1.0
    if big.empty and coverage > 0.9:
        return GateResult(
            "calendar_coverage",
            "ok",
            f"{len(prices)} rows covering {coverage:.1%} of business days in the span.",
            {"coverage": round(float(coverage), 4)},
        )
    status: Status = "fail" if coverage < 0.75 else "warn"
    return GateResult(
        "calendar_coverage",
        status,
        f"Calendar coverage is {coverage:.1%} with {len(big)} gap(s) over {max_gap_days} days.",
        {
            "coverage": round(float(coverage), 4),
            "largest_gaps": [f"{d.date()}:{int(v)}d" for d, v in big.sort_values(ascending=False).head(5).items()],
        },
    )


def _gate_sample_size(panel: PricePanel, min_obs: int | None) -> GateResult:
    if min_obs is None:
        return GateResult("sample_size", "ok", "No minimum sample size requested.")
    n = len(panel.returns)
    if n >= min_obs:
        return GateResult("sample_size", "ok", f"{n} return observations, {min_obs} required.")
    return GateResult(
        "sample_size",
        "fail",
        f"Only {n} return observations; the walk-forward protocol needs at least {min_obs}.",
        {"available": n, "required": min_obs},
    )


def _gate_conditioning(returns: pd.DataFrame, assets: tuple[str, ...]) -> GateResult:
    """Condition number of the asset correlation matrix.

    Not a data error -- a warning about the *problem*. Ten highly correlated
    equity ETFs make the tracking-error objective nearly flat along some
    directions, so weights become unstable even though the fit looks fine. This
    gate is why the report talks about turnover and weight stability at all.
    """
    corr = returns[list(assets)].corr().to_numpy()
    if not np.isfinite(corr).all():
        # A constant series makes its correlations undefined. The zero-variance
        # gate has already reported that; this one has nothing to add and must
        # not take the whole report down with a LinAlgError.
        return GateResult(
            "multicollinearity",
            "warn",
            "Correlation matrix is undefined (a series has zero variance); conditioning not assessed.",
            {"condition_number": None},
        )
    eigenvalues = np.linalg.eigvalsh(corr)
    smallest = float(max(eigenvalues.min(), 1e-15))
    condition = float(eigenvalues.max() / smallest)
    if condition < 100:
        status: Status = "ok"
        summary = f"Asset correlation matrix is well conditioned (kappa = {condition:.0f})."
    elif condition < 1000:
        status = "warn"
        summary = f"Asset correlation matrix is ill conditioned (kappa = {condition:.0f}); expect unstable weights."
    else:
        status = "warn"
        summary = f"Asset correlation matrix is severely ill conditioned (kappa = {condition:.0f}); weights are close to unidentified."
    return GateResult(
        "multicollinearity",
        status,
        summary,
        {"condition_number": round(condition, 1), "min_eigenvalue": round(smallest, 6)},
    )


def run_quality_gates(panel: PricePanel, *, min_obs: int | None = None) -> QualityReport:
    """Run every gate against ``panel`` and return the combined report."""
    prices = panel.prices
    returns = prices.pct_change().dropna(how="all")
    gates = (
        _gate_non_positive(prices),
        _gate_missing(prices),
        _gate_stale(prices),
        _gate_split_jumps(prices),
        _gate_zero_variance(returns.dropna(how="any")),
        _gate_extreme_returns(returns.dropna(how="any")),
        _gate_calendar(prices),
        _gate_sample_size(panel, min_obs),
        _gate_conditioning(returns.dropna(how="any"), panel.assets),
    )
    return QualityReport(gates=gates, panel_summary=panel.describe())
