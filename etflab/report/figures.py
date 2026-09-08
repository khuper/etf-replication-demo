"""Figures for the research report.

House rules, applied to every chart here:

* One y-axis, always. Two measures on different scales become two panels.
* Categorical hues assigned in fixed slot order and never cycled; past six
  classes the tail folds into "other" rather than inventing a colour.
* Thin marks, solid hairline grid one shade off the surface, no chart junk.
* A reference line wherever an absolute benchmark exists -- most of these charts
  are meaningless without the irreducible-tracking-error floor drawn on them.
* Direct labels on any series whose colour sits below 3:1 against the surface.

The palette is the validated categorical set; the slot order is the part that
makes it colour-vision-safe, so it is not reordered for aesthetics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in CI, and figure output must not depend on one

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MUTED = "#b8b7b2"

BASE_STYLE: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "600",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "font.size": 10,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def _style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)


def _save(fig: plt.Figure, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _log_sample_ticks(ax: plt.Axes, values: Any) -> None:
    """Label a log axis at the sample sizes actually evaluated.

    Matplotlib's default minor ticks on a log axis overprint each other into an
    unreadable smear once the range spans two decades. The ticks that mean
    something here are the sample sizes, so those are the ticks.
    """
    ticks = [float(v) for v in values]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(v):,}" if v < 1000 else f"{v / 1000:g}k" for v in ticks])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelrotation=0)


def _annotate_floor(ax: plt.Axes, floor: float, label: str = "irreducible floor") -> None:
    """Draw the theoretical best attainable value. Without it a tracking-error
    chart has no scale: 5% is excellent or terrible depending on this line."""
    ax.axhline(floor, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        f"{label}  {floor:.2%}",
        xy=(0.995, floor),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK_SOFT,
    )


# --------------------------------------------------------------------------- #
def figure_cumulative(study: Any, directory: Path) -> Path:
    """Wealth paths: replicator versus target, plus the cumulative active line."""
    best = study.race.results[study.race.best]
    daily = best.daily
    replicator = (1.0 + daily["net"]).cumprod()
    target = (1.0 + daily["target"]).cumprod()
    active = replicator / target

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.14}
    )
    top.plot(replicator.index, replicator, color=SERIES[0], label=f"Replicator ({study.race.best})")
    top.plot(target.index, target, color=SERIES[1], label=f"Target ({study.config.target})")
    top.set_ylabel("growth of 1.00")
    top.set_title(
        f"Replicating {study.config.target}: out-of-sample wealth paths",
        loc="left",
        pad=12,
    )
    top.legend(loc="upper left")
    for series, colour in ((replicator, SERIES[0]), (target, SERIES[1])):
        top.annotate(
            f"{series.iloc[-1]:.2f}x",
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            va="center",
            fontweight="600",
        )
    _style_axes(top)

    bottom.fill_between(active.index, 1.0, active, color=SERIES[0], alpha=0.18, linewidth=0)
    bottom.plot(active.index, active, color=SERIES[0], linewidth=1.6)
    bottom.axhline(1.0, color=INK_SOFT, linewidth=1.0)
    bottom.set_ylabel("replicator / target")
    bottom.set_xlabel("")
    _style_axes(bottom)
    return _save(fig, directory, "cumulative_returns")


def figure_rolling_te(study: Any, directory: Path, window: int = 126) -> Path:
    """Rolling tracking error: the picture the single headline number averages away."""
    from etflab.metrics import rolling_tracking_error

    best = study.race.best
    benchmark = study.race.benchmark
    fig, ax = plt.subplots(figsize=(10, 4.6))

    for index, name in enumerate((best, benchmark)):
        series = rolling_tracking_error(study.race.results[name].active, window).dropna()
        ax.plot(series.index, series, color=SERIES[index], label=f"{name}")
        ax.annotate(
            name,
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=SERIES[index],
            fontsize=9,
            va="center",
            fontweight="600",
        )

    if study.panel.truth is not None:
        _annotate_floor(ax, study.panel.truth.irreducible_te_annual)

    ax.set_title(f"Rolling {window}-day tracking error, annualised", loc="left", pad=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    _style_axes(ax)
    return _save(fig, directory, "rolling_tracking_error")


def figure_horse_race(study: Any, directory: Path) -> Path:
    """One bar per strategy. One colour, because the story is the ordering."""
    ordered = study.race.metrics["tracking_error"].sort_values(ascending=False)
    colours = [SERIES[0] if name == study.race.best else MUTED for name in ordered.index]

    fig, ax = plt.subplots(figsize=(9, 0.46 * len(ordered) + 2.2))
    bars = ax.barh(list(ordered.index), ordered.to_numpy(), color=colours, height=0.62)
    for bar, (name, value) in zip(bars, ordered.items(), strict=True):
        ax.annotate(
            f"{value:.2%}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK if name == study.race.best else INK_SOFT,
            fontweight="600" if name == study.race.best else "normal",
        )

    if study.panel.truth is not None:
        floor = study.panel.truth.irreducible_te_annual
        ax.axvline(floor, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(
            f"irreducible floor {floor:.2%}",
            xy=(floor, len(ordered) - 0.35),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SOFT,
            va="center",
        )

    ax.set_title("Out-of-sample tracking error by strategy", loc="left", pad=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, float(ordered.max()) * 1.18)
    ax.grid(axis="y", visible=False)
    _style_axes(ax)
    return _save(fig, directory, "horse_race")


def figure_weights(study: Any, directory: Path, top_n: int = 5) -> Path:
    """Weight evolution. Six classes maximum, the tail folded into 'other'."""
    weights = study.race.results[study.race.best].daily_weights
    ranked = weights.mean().sort_values(ascending=False)
    keep = list(ranked.index[:top_n])
    stacked = weights[keep].copy()
    remainder = weights.drop(columns=keep).sum(axis=1)
    if remainder.abs().max() > 1e-9:
        stacked["other"] = remainder

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.stackplot(
        stacked.index,
        [stacked[c].to_numpy() for c in stacked.columns],
        colors=SERIES[: len(stacked.columns)],
        edgecolor=SURFACE,
        linewidth=1.2,  # the 2px surface gap between adjacent fills
    )
    # Direct labels: three of these hues sit below 3:1 on this surface, so the
    # legend alone would not carry identity.
    cumulative = stacked.cumsum(axis=1)
    for index, column in enumerate(stacked.columns):
        centre = float(cumulative[column].iloc[-1] - stacked[column].iloc[-1] / 2)
        if stacked[column].iloc[-1] < 0.03:
            continue
        ax.annotate(
            column,
            xy=(stacked.index[-1], centre),
            xytext=(8, 0),
            textcoords="offset points",
            color=SERIES[index % len(SERIES)],
            fontsize=9,
            va="center",
            fontweight="600",
        )

    ax.set_title(f"Holdings through time ({study.race.best}, after drift)", loc="left", pad=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.margins(x=0)
    _style_axes(ax)
    return _save(fig, directory, "weights")


def figure_recovery(study: Any, directory: Path) -> Path:
    """Two panels, never two axes: identification on the left, cost on the right."""
    if study.recovery is None:
        raise ValueError("No recovery study to plot.")
    frame = study.recovery.frame
    strategies = [c[: -len("_l1")] for c in frame.columns if c.endswith("_l1")]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))

    for index, name in enumerate(strategies):
        left.plot(frame.index, frame[f"{name}_l1"], color=SERIES[index], marker="o", markersize=5, label=name)
    left.set_xscale("log")
    left.set_yscale("log")
    _log_sample_ticks(left, frame.index)
    left.set_xlabel("training observations")
    left.set_ylabel("L1 distance to true weights")
    left.set_title("Weight recovery", loc="left", pad=10)
    left.legend(loc="upper right")
    _style_axes(left)

    for index, name in enumerate(strategies):
        column = f"{name}_excess_te_oos"
        if column not in frame:
            continue
        series = frame[column].dropna()
        right.plot(series.index, series * 1e4, color=SERIES[index], marker="o", markersize=5, label=name)
    right.axhline(0.0, color=INK_SOFT, linewidth=1.0)
    right.set_xscale("log")
    _log_sample_ticks(right, frame.index)
    right.set_xlabel("training observations")
    right.set_ylabel("excess tracking error (bp)")
    right.set_title("Out-of-sample cost of estimation error", loc="left", pad=10)
    _style_axes(right)

    fig.suptitle(
        "Consistency on synthetic data: the weights converge slowly, the objective converges fast",
        x=0.008,
        ha="left",
        fontsize=12,
        fontweight="600",
        color=INK,
    )
    fig.subplots_adjust(top=0.84, wspace=0.28)
    return _save(fig, directory, "recovery")


def figure_capacity(study: Any, directory: Path) -> Path:
    """Where the advantage dies: cost per unit traded, and size of the book."""
    if study.cost_breakeven is None or study.capacity is None:
        raise ValueError("No capacity analysis to plot.")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    breakeven = study.cost_breakeven
    if study.cost_split is not None:
        split = study.cost_split
        strategy_column, benchmark_column = split.columns[0], split.columns[1]
        left.plot(split.index, split[strategy_column], color=SERIES[0], marker="o", markersize=5, label=strategy_column)
        left.plot(
            split.index, split[benchmark_column], color=SERIES[1], marker="o", markersize=5, label=benchmark_column
        )
        advantage = float(split["te_advantage_bp"].mean())
        left.axhline(advantage, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
        left.annotate(
            f"tracking error reduced by {advantage:,.0f}bp",
            xy=(0.985, advantage),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=INK_SOFT,
        )
        left.set_xlabel("assumed transaction cost (bp, one way)")
        left.set_ylabel("annual cost drag (bp)")
        left.set_title("What the tracking-error reduction costs", loc="left", pad=10)
        left.legend(loc="upper left")
        if breakeven is not None and breakeven.breakeven is not None:
            left.axvline(breakeven.breakeven, color=SERIES[7], linewidth=1.4, linestyle=(0, (4, 3)))
            left.annotate(
                f"break-even {breakeven.breakeven:,.0f}bp",
                xy=(breakeven.breakeven, advantage * 0.55),
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color=SERIES[7],
                fontweight="600",
            )
    _style_axes(left)

    if study.cost_curve is not None:
        curve = study.cost_curve
        right.plot(curve.index, curve["cost_drag_bps_annual"], color=SERIES[0], marker="o", markersize=5)
        breached = curve[curve["exceeds_participation_limit"]]
        if not breached.empty:
            first = breached.index[0]
            right.axvline(first, color=SERIES[7], linewidth=1.4, linestyle=(0, (4, 3)))
            right.annotate(
                f"ADV participation limit\nbreached above ${first / 1e9:,.0f}bn",
                xy=(first, float(curve["cost_drag_bps_annual"].max()) * 0.55),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                fontsize=8.5,
                color=SERIES[7],
            )
        right.set_xscale("log")
        right.set_xlabel("book size (USD)")
        right.set_ylabel("annual cost drag (bp)")
        right.set_title("Capacity under square-root impact", loc="left", pad=10)
        right.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"${v / 1e6:,.0f}m" if v < 1e9 else f"${v / 1e9:,.0f}bn")
        )
    _style_axes(right)
    fig.subplots_adjust(wspace=0.3)
    return _save(fig, directory, "capacity")


def figure_pbo(study: Any, directory: Path) -> Path:
    """Distribution of the CSCV logit, against its own noise reference."""
    if study.sweep is None:
        raise ValueError("No sweep to plot.")
    logits = study.sweep.pbo.logits

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bins = np.linspace(float(np.min(logits)) - 0.4, float(np.max(logits)) + 0.4, 36).tolist()
    below = logits[logits < 0]
    above = logits[logits >= 0]
    ax.hist(above, bins=bins, color=SERIES[0], alpha=0.92, label="winner ranks above OOS median")
    ax.hist(below, bins=bins, color=SERIES[7], alpha=0.92, label="winner ranks below OOS median")
    ax.axvline(0.0, color=INK_SOFT, linewidth=1.2)

    ax.set_title(
        f"Backtest overfitting: PBO = {study.sweep.pbo.pbo:.0%} "
        f"(pure-noise reference for this grid: {study.sweep.noise_pbo:.0%})",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("logit of the winner's out-of-sample rank")
    ax.set_ylabel("CSCV splits")
    ax.legend(loc="upper left")
    _style_axes(ax)
    return _save(fig, directory, "overfitting")


def figure_regimes(study: Any, directory: Path) -> Path:
    """Tracking error by market regime. Ordered categories, one hue."""
    if study.regimes.empty:
        raise ValueError("No regime breakdown to plot.")
    order = [r for r in ("calm", "stressed", "crisis", "low vol", "mid vol", "high vol") if r in study.regimes.index]
    frame = study.regimes.loc[order] if order else study.regimes
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#86b6ef", "#3987e5", "#1c5cab"][: len(frame)]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(list(frame.index), frame["tracking_error"].to_numpy(), color=ramp, width=0.56)
    for bar, (_regime, row) in zip(bars, frame.iterrows(), strict=True):
        ax.annotate(
            f"{row['tracking_error']:.2%}\n{row['days']:,.0f} days",
            xy=(bar.get_x() + bar.get_width() / 2, row["tracking_error"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK_SOFT,
        )
    if study.panel.truth is not None:
        _annotate_floor(ax, study.panel.truth.irreducible_te_annual)
    ax.set_title(f"Tracking error by regime ({study.race.best})", loc="left", pad=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, float(frame["tracking_error"].max()) * 1.32)
    ax.grid(axis="x", visible=False)
    _style_axes(ax)
    return _save(fig, directory, "regimes")


FIGURES = {
    "cumulative_returns": figure_cumulative,
    "rolling_tracking_error": figure_rolling_te,
    "horse_race": figure_horse_race,
    "weights": figure_weights,
    "regimes": figure_regimes,
    "recovery": figure_recovery,
    "capacity": figure_capacity,
    "overfitting": figure_pbo,
}


def make_figures(study: Any, directory: str | Path) -> dict[str, Path]:
    """Render every figure the study supports. Missing inputs skip, never crash."""
    target = Path(directory)
    produced: dict[str, Path] = {}
    with plt.rc_context(BASE_STYLE):  # type: ignore[arg-type]  # rcParams is a TypedDict upstream
        for name, builder in FIGURES.items():
            try:
                produced[name] = builder(study, target)
            except (ValueError, KeyError, IndexError):
                continue
    return produced
