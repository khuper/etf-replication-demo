"""Terminal rendering. What you see while the research is running and when it ends."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.text import Text

from etflab.config import ExperimentConfig

ACCENT = "cyan"
STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "bold red"}


def banner() -> Panel:
    title = Text("ETF REPLICATION LAB", style=f"bold {ACCENT}")
    subtitle = Text("reproducible replication research, offline by default", style="dim")
    return Panel(Align.center(Text.assemble(title, "\n", subtitle)), border_style=ACCENT, padding=(1, 4))


def config_panel(config: ExperimentConfig) -> Panel:
    table = RichTable(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=16)
    table.add_column(style="white")
    table.add_row("Target", f"[bold {ACCENT}]{config.target}[/]")
    table.add_row("Candidates", " ".join(config.assets))
    table.add_row("Period", f"{config.start} to {config.end}")
    table.add_row(
        "Data",
        f"{config.data_source}" + (f" (seed {config.synthetic_seed})" if config.data_source == "synthetic" else ""),
    )
    table.add_row("Strategy", config.strategy)
    table.add_row(
        "Protocol",
        f"{config.train_mode}, {config.train_days}d train, {config.rebalance_days}d rebalance, {config.embargo_days}d embargo",
    )
    table.add_row("Constraints", f"{config.max_weight:.0%} position cap, {config.max_turnover:.0%} L1 turnover")
    table.add_row(
        "Costs",
        f"{config.cost_model}"
        + (
            f" @ {config.cost_bps:g}bp"
            if config.cost_model == "bps"
            else f" on ${config.portfolio_notional / 1e6:,.0f}m"
        ),
    )
    table.add_row("Config hash", f"[dim]{config.semantic_hash()}[/]")
    return Panel(table, title="[bold]Experiment[/]", border_style="blue", padding=(0, 1))


def quality_panel(report: Any) -> Panel:
    table = RichTable(show_header=True, box=None, header_style="bold dim", pad_edge=False)
    table.add_column("gate", style="white", width=24)
    table.add_column("status", width=6)
    table.add_column("detail", style="dim", overflow="fold")
    for gate in report.gates:
        style = STATUS_STYLE.get(gate.status, "white")
        table.add_row(gate.name, f"[{style}]{gate.status}[/]", gate.summary)
    border = {"ok": "green", "warn": "yellow", "fail": "red"}[report.status]
    return Panel(
        table, title=f"[bold]Data quality[/] · [{border}]{report.status}[/]", border_style=border, padding=(0, 1)
    )


def _fmt(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "pct1":
        return f"{value:.1%}"
    if kind == "bp":
        return f"{value * 1e4:.1f}"
    return f"{value:.3f}"


def horse_race_table(study: Any) -> RichTable:
    metrics = study.race.metrics.sort_values("tracking_error")
    table = RichTable(title="Out-of-sample results", header_style=f"bold {ACCENT}", box=None, pad_edge=False)
    table.add_column("strategy")
    table.add_column("tracking err", justify="right")
    table.add_column("corr", justify="right")
    table.add_column("down capt", justify="right")
    table.add_column("turnover/yr", justify="right")
    table.add_column("cost bp/yr", justify="right")
    table.add_column("vs bench", justify="right")
    table.add_column("sig", justify="center")

    for name, row in metrics.iterrows():
        comparison = study.race.comparisons.get(name)
        is_best = name == study.race.best
        label = f"[bold]{name}[/]" if is_best else name
        if name == study.race.benchmark:
            label = f"[dim]{name} (benchmark)[/]"
        if comparison is None:
            delta, significance = "--", ""
        else:
            delta = _fmt(comparison.te_difference, "pct")
            if comparison.significantly_better:
                significance = "[green]yes[/]"
            elif comparison.ci.excludes_zero:
                significance = "[red]worse[/]"
            else:
                significance = "[dim]no[/]"
        table.add_row(
            label,
            f"[bold {ACCENT}]{_fmt(row['tracking_error'], 'pct')}[/]"
            if is_best
            else _fmt(row["tracking_error"], "pct"),
            _fmt(row["correlation"], "num"),
            _fmt(row.get("down_capture"), "num"),
            _fmt(row.get("annual_turnover"), "pct1"),
            _fmt(row.get("cost_drag_annual"), "bp"),
            delta,
            significance,
        )
    return table


def verdict_panel(study: Any) -> Panel:
    blocks: list[Any] = [Text(study.race.verdict, style="white")]
    if study.race.notes:
        blocks.append(Text(""))
        for note in study.race.notes:
            blocks.append(Text(f"• {note}", style="dim"))
    return Panel(Group(*blocks), title="[bold]Verdict[/]", border_style=ACCENT, padding=(1, 2))


def diagnostics_panel(study: Any) -> Panel:
    table = RichTable(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=28)
    table.add_column(style="white")
    if study.panel.truth is not None:
        table.add_row("Irreducible tracking error", f"{study.panel.truth.irreducible_te_annual:.2%}")
    if study.recovery is not None:
        table.add_row(
            "Weight recovery",
            f"L1 {study.recovery.final_l1:.3f} at {study.recovery.extended_obs:,} obs "
            f"(decays as T^{study.recovery.convergence_exponent:.2f})",
        )
    if study.sweep is not None:
        marker = "green" if study.sweep.selection_is_meaningful else "yellow"
        table.add_row(
            "Backtest overfitting (PBO)",
            f"[{marker}]{study.sweep.pbo.pbo:.0%}[/] vs {study.sweep.noise_pbo:.0%} noise reference "
            f"over {len(study.sweep.frame)} configs",
        )
        table.add_row("Deflated Sharpe", f"{study.sweep.deflated.deflated_probability:.1%} probability of a real edge")
    for report in (getattr(study, "governance", {}) or {}).values():
        colour = "red" if report.ever_shut_off else "green"
        table.add_row(
            f"Kill switch vs {report.benchmark}",
            f"[{colour}]{'shut off ' + format(report.days_off / max(len(report.daily), 1), '.0%') + ' of days' if report.ever_shut_off else 'never tripped'}[/] "
            f"at a {report.policy.hurdle:g}x hurdle",
        )
    if study.cost_breakeven is not None:
        value = study.cost_breakeven.breakeven
        table.add_row(
            "Cost break-even",
            f"{value:.0f}bp one way" if value is not None else "beyond the tested range: costs do not decide this",
        )
    if study.cost_curve is not None:
        breached = study.cost_curve[study.cost_curve["exceeds_participation_limit"]]
        table.add_row(
            "Capacity",
            f"ADV participation limit breached above ${breached.index[0] / 1e9:,.1f}bn"
            if not breached.empty
            else "within participation limits at every tested size",
        )
    return Panel(table, title="[bold]Diagnostics[/]", border_style="blue", padding=(0, 1))


def render_study(console: Console, study: Any) -> None:
    """Print the full study summary."""
    console.print(quality_panel(study.quality))
    console.print()
    console.print(horse_race_table(study))
    console.print()
    console.print(diagnostics_panel(study))
    console.print()
    console.print(verdict_panel(study))
