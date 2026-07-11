"""A friendly terminal interface for the ETF replication lab."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import replace

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from src.config import DEFAULT_ASSETS, ResearchConfig
from src.replicator import SyntheticLiabilityReplicator
from src.reporting import export_result, render_result


console = Console()


def banner() -> Panel:
    logo = Text("ETF REPLICATION LAB", style="bold cyan")
    subtitle = Text("A terminal playground for portfolio experiments", style="dim")
    return Panel(Align.center(Text.assemble(logo, "\n", subtitle)), border_style="cyan", padding=(1, 4))


def config_table(config: ResearchConfig) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=14)
    table.add_column(style="white")
    table.add_row("Target", f"[bold cyan]{config.target}[/]")
    table.add_row("Candidates", " ".join(config.assets))
    table.add_row("Period", f"{config.start_date} → {config.end_date}")
    table.add_row("Model", config.model.upper())
    table.add_row("Training", f"{config.initial_train_size} trading days")
    table.add_row("Rebalance", f"Every {config.rebalance_days} trading days")
    table.add_row("Constraints", f"{config.max_weight:.0%} max weight · {config.max_turnover:.0%} max L1 turnover")
    table.add_row("Trading costs", f"{config.transaction_cost_bps:g} bps")
    return table


def show_config(config: ResearchConfig) -> None:
    console.print(Panel(config_table(config), title="[bold]Experiment[/]", border_style="blue"))


def run_research(config: ResearchConfig):
    config.validate()
    replicator = SyntheticLiabilityReplicator(list(config.assets), config.target, config.start_date, config.end_date)
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching adjusted market data…", total=None)
        replicator.fetch_data()
        progress.update(task, description="Running walk-forward optimization…")
        result = replicator.run_backtest(
            initial_train_size=config.initial_train_size,
            step=config.rebalance_days,
            max_weight=config.max_weight,
            max_turnover=config.max_turnover,
            transaction_cost_bps=config.transaction_cost_bps,
            cvar_constraint_ratio=config.cvar_ratio if config.model == "cvar" else None,
        )
    return result


HELP = """\
[bold cyan]Commands[/]
  [bold]show[/]                         Show the current experiment
  [bold]target[/] TICKER                Change the replication target
  [bold]assets[/] TICKER ...            Replace candidate assets
  [bold]dates[/] YYYY-MM-DD YYYY-MM-DD  Set the research period
  [bold]set model[/] tracking|cvar      Select the optimizer
  [bold]set rebalance[/] DAYS           Set rebalance frequency
  [bold]set training[/] DAYS            Set initial training observations
  [bold]set max-weight[/] PERCENT       Set position cap (for example 25)
  [bold]set turnover[/] PERCENT         Set max L1 turnover (for example 20)
  [bold]set costs[/] BPS                Set estimated trading costs
  [bold green]run[/]                          Run and export the experiment
  [bold]help[/]                         Show this guide
  [bold]quit[/]                         Leave the lab
"""


def _updated_config(config: ResearchConfig, tokens: list[str]) -> ResearchConfig:
    command = tokens[0].lower()
    if command == "target" and len(tokens) == 2:
        return replace(config, target=tokens[1])
    if command == "assets" and len(tokens) >= 3:
        return replace(config, assets=tuple(tokens[1:]))
    if command == "dates" and len(tokens) == 3:
        return replace(config, start_date=tokens[1], end_date=tokens[2])
    if command == "set" and len(tokens) == 3:
        field, value = tokens[1].lower(), tokens[2]
        if field == "model":
            return replace(config, model=value.lower())
        if field == "rebalance":
            return replace(config, rebalance_days=int(value))
        if field == "training":
            return replace(config, initial_train_size=int(value))
        if field == "max-weight":
            return replace(config, max_weight=float(value) / 100)
        if field == "turnover":
            return replace(config, max_turnover=float(value) / 100)
        if field == "costs":
            return replace(config, transaction_cost_bps=float(value))
    raise ValueError("I don't recognize that command. Type 'help' to see the command guide.")


def interactive() -> int:
    config = ResearchConfig()
    console.print(banner())
    console.print("Type [bold]help[/] for commands, or [bold green]run[/] to start with the defaults.\n")
    show_config(config)
    while True:
        try:
            raw = Prompt.ask("\n[bold cyan]etf-lab[/]").strip()
            if not raw:
                continue
            tokens = shlex.split(raw)
            command = tokens[0].lower()
            if command in {"quit", "exit", "q"}:
                console.print("[dim]Research session closed.[/]")
                return 0
            if command == "help":
                console.print(Panel(HELP, border_style="blue"))
            elif command == "show":
                show_config(config)
            elif command == "run":
                result = run_research(config)
                render_result(console, result)
                run_dir = export_result(config, result)
                console.print(f"\n[green]◆[/] Run exported to [bold]{run_dir}[/]")
            else:
                config = _updated_config(config, tokens)
                config.validate()
                console.print("[green]◆[/] Updated")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Research session closed.[/]")
            return 0
        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etf-lab", description="Interactive ETF replication research terminal")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="Run one reproducible experiment")
    run.add_argument("--target", default="PSP")
    run.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    defaults = ResearchConfig()
    run.add_argument("--start", default=defaults.start_date)
    run.add_argument("--end", default=defaults.end_date)
    run.add_argument("--training-days", type=int, default=504)
    run.add_argument("--rebalance-days", type=int, default=126)
    run.add_argument("--max-weight", type=float, default=0.25)
    run.add_argument("--max-turnover", type=float, default=0.20)
    run.add_argument("--costs-bps", type=float, default=5.0)
    run.add_argument("--model", choices=("tracking", "cvar"), default="tracking")
    run.add_argument("--cvar-ratio", type=float, default=1.0)
    run.add_argument("--output", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        return interactive()
    config = ResearchConfig(
        target=args.target,
        assets=tuple(args.assets),
        start_date=args.start,
        end_date=args.end,
        initial_train_size=args.training_days,
        rebalance_days=args.rebalance_days,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
        transaction_cost_bps=args.costs_bps,
        model=args.model,
        cvar_ratio=args.cvar_ratio,
        output_dir=args.output,
    )
    try:
        console.print(banner())
        show_config(config)
        result = run_research(config)
        render_result(console, result)
        run_dir = export_result(config, result)
        console.print(f"\n[green]◆[/] Run exported to [bold]{run_dir}[/]")
        return 0
    except Exception as exc:
        console.print(f"\n[bold red]Experiment failed:[/] {exc}")
        return 1
