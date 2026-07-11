"""Rendering and export helpers for research results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import ResearchConfig
from src.replicator import BacktestResult


def metrics_table(result: BacktestResult) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right", style="bold white")
    labels = {
        "tracking_error": "Tracking error",
        "correlation": "Correlation",
        "replicator_return": "Replicator return",
        "target_return": "Target return",
        "replicator_volatility": "Replicator volatility",
        "max_drawdown": "Max drawdown",
        "average_turnover": "Average turnover",
        "total_cost": "Estimated costs",
    }
    percentages = {
        "tracking_error",
        "replicator_return",
        "target_return",
        "replicator_volatility",
        "max_drawdown",
        "average_turnover",
        "total_cost",
    }
    for key, label in labels.items():
        value = result.metrics[key]
        rendered = f"{value:.2%}" if key in percentages else f"{value:.3f}"
        table.add_row(label, rendered)
    return table


def weights_table(result: BacktestResult, limit: int = 10) -> Table:
    table = Table(title="Latest allocation", header_style="bold cyan")
    table.add_column("ETF")
    table.add_column("Weight", justify="right")
    latest = result.weights.iloc[-1].sort_values(ascending=False)
    for ticker, weight in latest.head(limit).items():
        table.add_row(str(ticker), f"{weight:.2%}")
    return table


def render_result(console: Console, result: BacktestResult) -> None:
    console.print(
        Panel(
            metrics_table(result),
            title="[bold cyan]Out-of-sample results[/]",
            subtitle=f"{result.returns.index[0].date()} → {result.returns.index[-1].date()}",
            border_style="cyan",
        )
    )
    console.print(weights_table(result))


def export_result(config: ResearchConfig, result: BacktestResult, output_dir: str | None = None) -> Path:
    root = Path(output_dir or config.output_dir)
    run_name = f"{config.target.lower()}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    result.returns.to_csv(run_dir / "returns.csv", index_label="date")
    result.weights.to_csv(run_dir / "weights.csv", index_label="rebalance_date")
    result.turnover.rename("turnover").to_csv(run_dir / "turnover.csv", index_label="rebalance_date")
    (run_dir / "config.json").write_text(json.dumps(config.as_dict(), indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
    return run_dir
