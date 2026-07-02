"""Self-contained HTML tearsheet for strategy review."""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from . import metrics as m
from .backtest import BacktestResult
from .config import STRESS_WINDOWS, ReplicatorConfig
from .plotting import (
    plot_active_drawdown,
    plot_cumulative_returns,
    plot_final_weights,
    plot_group_allocation,
    plot_rolling_tracking,
)

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 60
HIT_TOLERANCE = 0.005  # months tracking within +/- 50 bps count as hits
COST_SCENARIOS_BPS = [0.0, 10.0, 25.0]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_CSS = """
body { margin: 0; background: #f9f9f7; color: #0b0b0b;
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.page { max-width: 980px; margin: 0 auto; padding: 32px 24px 48px; }
.card { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
        padding: 20px 24px; margin-bottom: 20px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 12px; color: #0b0b0b; }
.sub { color: #52514e; font-size: 13px; margin-bottom: 20px; }
.spec { display: grid; grid-template-columns: 160px 1fr; gap: 4px 16px; font-size: 13px; }
.spec dt { color: #898781; } .spec dd { margin: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 12px; }
.tile { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 14px 16px; }
.tile .v { font-size: 22px; font-weight: 600; }
.tile .l { font-size: 11px; color: #898781; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
img { width: 100%; height: auto; border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th { color: #898781; font-weight: 500; text-align: right; padding: 6px 8px; border-bottom: 1px solid #e1e0d9; }
th:first-child, td:first-child { text-align: left; }
td { text-align: right; padding: 6px 8px; border-bottom: 1px solid #e1e0d9; }
.note { color: #52514e; font-size: 12.5px; margin-top: 10px; }
.footer { color: #898781; font-size: 11.5px; margin-top: 28px; line-height: 1.5; }
"""


def _img(path: str) -> str:
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{os.path.basename(path)}">'


def _pct(value: float, digits: int = 2) -> str:
    return "–" if pd.isna(value) else f"{value * 100:.{digits}f}%"


def _num(value: float, digits: int = 2) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def _tile(value: str, label: str) -> str:
    return f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div></div>'


def _monthly_table(monthly_active: pd.Series) -> str:
    """Year x month grid of active returns in bps, with a diverging cell wash."""
    frame = monthly_active.to_frame("v")
    frame["year"], frame["month"] = frame.index.year, frame.index.month
    grid = frame.pivot(index="year", columns="month", values="v")

    scale = max(float(monthly_active.abs().max()), 1e-9)
    header = "".join(f"<th>{MONTH_LABELS[c - 1]}</th>" for c in range(1, 13))
    rows = []
    for year, row in grid.iterrows():
        cells = []
        for c in range(1, 13):
            v = row.get(c)
            if v is None or pd.isna(v):
                cells.append("<td></td>")
                continue
            # Blue for tracking ahead, red for behind, fading to the surface at zero.
            alpha = min(abs(v) / scale, 1.0) * 0.30
            color = "42,120,214" if v >= 0 else "227,73,72"
            cells.append(f'<td style="background: rgba({color},{alpha:.2f})">{v * 1e4:+,.0f}</td>')
        rows.append(f"<tr><td>{year}</td>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th>Active (bps)</th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _stress_table(stress: pd.DataFrame) -> str:
    if stress.empty:
        return '<p class="note">No configured stress window falls inside the out-of-sample period.</p>'
    rows = "".join(
        f"<tr><td>{name}</td><td>{_pct(r.portfolio_return)}</td><td>{_pct(r.target_return)}</td>"
        f"<td>{_pct(r.active_return)}</td><td>{_pct(r.tracking_error)}</td><td>{_num(r.correlation)}</td></tr>"
        for name, r in stress.iterrows()
    )
    return (
        "<table><thead><tr><th>Window</th><th>Replicator</th><th>Target</th><th>Active</th>"
        "<th>Tracking error</th><th>Correlation</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _cost_sensitivity_table(result: BacktestResult) -> str:
    rows = []
    for bps in sorted(set(COST_SCENARIOS_BPS + [result.cost_bps])):
        net = result.net_at(bps)
        stats = m.tracking_metrics(net, result.target_returns)
        drag = float((1 + result.gross_returns).prod() - (1 + net).prod())
        tag = " (assumed)" if bps == result.cost_bps else ""
        rows.append(
            f"<tr><td>{bps:g} bps{tag}</td><td>{_pct(stats['annualized_active_return'])}</td>"
            f"<td>{_pct(stats['tracking_error'])}</td><td>{_num(stats['information_ratio'])}</td>"
            f"<td>{_pct(drag)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Cost per unit turnover</th><th>Ann. active return</th>"
        "<th>Tracking error</th><th>Information ratio</th><th>Total cost drag</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _spec_items(config: ReplicatorConfig, result: BacktestResult) -> List[Tuple[str, str]]:
    bounds = (
        ", ".join(f"{g}: {lo:.0%}–{hi:.0%}" for g, (lo, hi) in config.group_bounds.items())
        if config.group_bounds
        else "none"
    )
    oos = result.net_returns.index
    return [
        ("Target", config.target),
        ("Basket", ", ".join(config.assets)),
        ("Out-of-sample window", f"{oos[0].date()} to {oos[-1].date()} ({len(result.weights)} rebalances)"),
        ("Position cap", _pct(config.max_weight, 0)),
        ("Turnover cap", f"{_pct(config.max_turnover, 0)} per rebalance"),
        ("CVaR constraint", f"≤ {config.cvar_ratio:g}× target CVaR at α={config.cvar_alpha:g}"),
        ("Group bounds", bounds),
        ("Transaction costs", f"{config.cost_bps:g} bps per unit turnover"),
    ]


def build_tearsheet(config: ReplicatorConfig, result: BacktestResult, output_dir: str) -> str:
    """Render the strategy tearsheet and its charts into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    net, gross, target = result.net_returns, result.gross_returns, result.target_returns

    charts = {
        "cumulative": plot_cumulative_returns(
            net, target, config.target, os.path.join(output_dir, "cumulative_returns.png"), gross=gross
        ),
        "drawdown": plot_active_drawdown(net, target, os.path.join(output_dir, "active_drawdown.png")),
        "rolling": plot_rolling_tracking(
            m.rolling_tracking_error(net, target, ROLLING_WINDOW),
            m.rolling_correlation(net, target, ROLLING_WINDOW),
            ROLLING_WINDOW,
            os.path.join(output_dir, "rolling_tracking.png"),
        ),
        "groups": plot_group_allocation(
            m.group_allocation(result.weights, config.classes_by_asset()),
            os.path.join(output_dir, "group_allocation.png"),
        ),
        "weights": plot_final_weights(result.final_weights, os.path.join(output_dir, "final_weights.png")),
    }

    stats = m.tracking_metrics(net, target)
    monthly_active = m.monthly_active_returns(net, target)
    hit = m.hit_rate(monthly_active, HIT_TOLERANCE)
    stress = m.stress_window_metrics(net, target, STRESS_WINDOWS)
    drag = float((1 + gross).prod() - (1 + net).prod())

    tiles = "".join(
        [
            _tile(_pct(stats["tracking_error"]), "Tracking error (ann.)"),
            _tile(_num(stats["correlation"]), "Correlation"),
            _tile(_pct(stats["annualized_active_return"]), "Active return (ann., net)"),
            _tile(_num(stats["information_ratio"]), "Information ratio"),
            _tile(_pct(stats["max_active_drawdown"]), "Max active drawdown"),
            _tile(_pct(result.turnover.mean(), 0), "Avg turnover / rebalance"),
            _tile(_pct(drag), "Total cost drag"),
        ]
    )
    spec = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in _spec_items(config, result))
    best, worst = monthly_active.max(), monthly_active.min()

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Replication tearsheet — {config.target}</title>
<style>{_CSS}</style></head>
<body><div class="page">
  <h1>Synthetic replication tearsheet — {config.target}</h1>
  <div class="sub">Liquid-basket replication, expanding-window out-of-sample backtest.
  Generated {datetime.now():%Y-%m-%d}.</div>

  <div class="card"><h2>Strategy specification</h2><dl class="spec">{spec}</dl></div>

  <div class="tiles" style="margin-bottom:20px">{tiles}</div>

  <div class="card"><h2>Out-of-sample performance (net of costs)</h2>{_img(charts["cumulative"])}
    {_img(charts["drawdown"])}</div>

  <div class="card"><h2>Tracking stability</h2>{_img(charts["rolling"])}</div>

  <div class="card"><h2>Monthly tracking consistency</h2>{_monthly_table(monthly_active)}
    <p class="note">Hit rate: <b>{_pct(hit, 0)}</b> of months tracked within ±{HIT_TOLERANCE * 1e4:.0f} bps.
    Best month {best * 1e4:+,.0f} bps, worst month {worst * 1e4:+,.0f} bps.</p></div>

  <div class="card"><h2>Behavior in stress windows</h2>{_stress_table(stress)}
    {'<p class="note">Named historical windows that overlap the out-of-sample period. Tracking error is annualized within each window.</p>' if not stress.empty else ''}</div>

  <div class="card"><h2>Cost sensitivity</h2>{_cost_sensitivity_table(result)}</div>

  <div class="card"><h2>Allocation</h2>{_img(charts["groups"])}{_img(charts["weights"])}</div>

  <div class="footer">Research demo using free market data (yfinance); not investment advice.
  Backtest assumes rebalance-day fills at close, costs proportional to L1 turnover, and no slippage
  beyond the configured cost rate. Past tracking performance does not guarantee future replication quality.</div>
</div></body></html>
"""
    path = os.path.join(output_dir, "tearsheet.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("Tearsheet written to %s", path)
    return path
