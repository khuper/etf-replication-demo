"""Chart generation for the replicator and correlation workflows.

Colors and chart chrome follow a validated light-mode palette: categorical
hues are assigned in fixed slot order, magnitude uses a single blue ramp,
and polarity (correlations, drawdowns) uses a blue/gray/red diverging map.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots, fixed order (never cycled past the list).
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _save(fig: plt.Figure, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_correlation_heatmap(returns: pd.DataFrame, path: str) -> str:
    """Pairwise return correlations on a diverging blue/gray/red scale."""
    import seaborn as sns

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            returns.corr(),
            annot=True,
            fmt=".2f",
            cmap=DIVERGING,
            vmin=-1,
            vmax=1,
            linewidths=2,
            linecolor=SURFACE,
            annot_kws={"size": 8, "color": INK},
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.grid(False)
        ax.set_title("Daily return correlations")
    return _save(fig, path)


def plot_cumulative_returns(
    net: pd.Series, target: pd.Series, target_name: str, path: str, gross: Optional[pd.Series] = None
) -> str:
    """Out-of-sample wealth curves: replicator (net of costs) versus target."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ((1 + net).cumprod() - 1).plot(ax=ax, color=SERIES[0], linewidth=2, label="Replicator (net of costs)")
        ((1 + target).cumprod() - 1).plot(ax=ax, color=SERIES[1], linewidth=2, label=f"Target ({target_name})")
        if gross is not None:
            ((1 + gross).cumprod() - 1).plot(
                ax=ax, color=SERIES[0], linewidth=1.2, alpha=0.45, linestyle="--", label="Replicator (gross)"
            )
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title("Out-of-sample cumulative return")
        ax.set_xlabel("")
        ax.legend()
    return _save(fig, path)


def plot_active_drawdown(portfolio: pd.Series, target: pd.Series, path: str) -> str:
    """Drawdown of replicator wealth relative to target wealth."""
    relative = (1 + portfolio).cumprod() / (1 + target).cumprod()
    drawdown = relative / relative.cummax() - 1
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.fill_between(drawdown.index, drawdown, 0, color="#e34948", alpha=0.25, linewidth=0)
        ax.plot(drawdown.index, drawdown.values, color="#e34948", linewidth=1.5)
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
        ax.set_title("Active drawdown (replicator wealth vs target wealth)")
        ax.set_xlabel("")
    return _save(fig, path)


def plot_rolling_tracking(rolling_te: pd.Series, rolling_corr: pd.Series, window: int, path: str) -> str:
    """Rolling tracking error and correlation as stacked panels (one axis each)."""
    with plt.rc_context(_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        rolling_te.plot(ax=ax1, color=SERIES[0], linewidth=2)
        ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax1.set_title(f"Rolling {window}-day tracking error (annualized)")

        rolling_corr.plot(ax=ax2, color=SERIES[1], linewidth=2)
        ax2.set_ylim(min(0.0, rolling_corr.min() - 0.05), 1.02)
        ax2.axhline(1.0, color=BASELINE, linewidth=1, linestyle="--")
        ax2.set_title(f"Rolling {window}-day correlation to target")
        ax2.set_xlabel("")
        fig.tight_layout()
    return _save(fig, path)


def plot_group_allocation(group_weights: pd.DataFrame, path: str) -> str:
    """Stacked area of basket allocation by instrument type at each rebalance."""
    ordered = group_weights[group_weights.mean().sort_values(ascending=False).index]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.stackplot(
            ordered.index,
            [ordered[c] for c in ordered.columns],
            labels=list(ordered.columns),
            colors=SERIES[: len(ordered.columns)],
            alpha=0.85,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title("Allocation by instrument type at each rebalance")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)


def plot_final_weights(weights: pd.Series, path: str) -> str:
    """Latest allocation as a single-hue horizontal bar chart."""
    ordered = weights[weights > 0].sort_values()
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(ordered) + 1.2))
        bars = ax.barh(ordered.index, ordered.values, color=SERIES[0], height=0.62)
        for bar, value in zip(bars, ordered.values):
            ax.text(
                value + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}", va="center", fontsize=9, color=INK_SECONDARY,
            )
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", visible=False)
        ax.set_title("Latest allocation")
    return _save(fig, path)


def plot_rolling_correlation_grid(
    rolling_corr: pd.DataFrame, target_name: str, stress_windows: Mapping[str, Tuple[str, str]], path: str
) -> str:
    """Small multiples: each basket ETF's rolling correlation to the target."""
    assets = list(rolling_corr.columns)
    ncols = 5
    nrows = -(-len(assets) // ncols)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, asset in enumerate(assets):
            ax = axes[i]
            rolling_corr[asset].plot(ax=ax, color=SERIES[0], linewidth=1.5)
            ax.axhline(1.0, color=BASELINE, linestyle="--", linewidth=1)
            _shade_stress(ax, stress_windows, label=(i == 0))
            ax.set_title(f"{asset} vs {target_name}", fontsize=10)
            ax.set_ylim(-1, 1.1)
            ax.set_xlabel("")
            if i == 0:
                ax.legend(loc="lower left", fontsize=8)
        for ax in axes[len(assets):]:
            ax.set_visible(False)
        fig.tight_layout()
    return _save(fig, path)


def plot_mean_rolling_correlation(
    mean_corrs: Dict[str, pd.Series], target_name: str, stress_windows: Mapping[str, Tuple[str, str]], path: str
) -> str:
    """Average basket correlation to the target for each rolling window length."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, (label, series) in enumerate(mean_corrs.items()):
            series.plot(ax=ax, color=SERIES[i], linewidth=2, label=label)
        _shade_stress(ax, stress_windows, label=True)
        ax.set_title(f"Average basket correlation to {target_name}")
        ax.set_xlabel("")
        ax.legend(loc="lower left")
    return _save(fig, path)


_STRESS_SHADES = ["#e34948", "#eda100", "#4a3aa7"]


def _shade_stress(ax: plt.Axes, stress_windows: Mapping[str, Tuple[str, str]], label: bool) -> None:
    for i, (name, (start, end)) in enumerate(stress_windows.items()):
        ax.axvspan(
            pd.to_datetime(start),
            pd.to_datetime(end),
            color=_STRESS_SHADES[i % len(_STRESS_SHADES)],
            alpha=0.15,
            label=name if label else None,
        )
