"""Command line interface.

Design rule: every command that produces a number also produces a run directory
containing the provenance for that number. There is no path through this CLI that
prints a result you cannot later reproduce or audit.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

from etflab import __version__
from etflab.config import COST_MODELS, DATA_SOURCES, DEFAULT_ASSETS, DEFAULT_TARGET, ExperimentConfig
from etflab.data import load_panel, run_quality_gates
from etflab.data.quality import DataQualityError
from etflab.registry import RunRegistry, diff_records
from etflab.report.console import banner, config_panel, horse_race_table, quality_panel, render_study, verdict_panel
from etflab.strategies import DEFAULT_ZOO, available_strategies

console = Console()


# --------------------------------------------------------------------------- #
# Argument plumbing
# --------------------------------------------------------------------------- #
def _add_experiment_args(parser: argparse.ArgumentParser) -> None:
    defaults = ExperimentConfig()
    data = parser.add_argument_group("data")
    data.add_argument("--target", default=DEFAULT_TARGET, help="ticker to replicate")
    data.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS), help="candidate proxies")
    data.add_argument("--start", default=defaults.start)
    data.add_argument("--end", default=defaults.end)
    data.add_argument("--data-source", choices=DATA_SOURCES, default=defaults.data_source)
    data.add_argument("--seed", type=int, default=defaults.synthetic_seed, help="synthetic market seed")
    data.add_argument("--leverage", type=float, default=defaults.synthetic_leverage, help="synthetic target leverage")
    data.add_argument("--csv-path", default=None, help="wide CSV of adjusted closes, for --data-source csv")
    data.add_argument("--online", action="store_true", help="allow network fetches (default is offline)")
    data.add_argument("--cache-dir", default=defaults.cache_dir)
    data.add_argument("--allow-degraded", action="store_true", help="proceed even if a data quality gate fails")

    protocol = parser.add_argument_group("walk-forward protocol")
    protocol.add_argument("--train-days", type=int, default=defaults.train_days)
    protocol.add_argument("--train-mode", choices=("expanding", "rolling"), default=defaults.train_mode)
    protocol.add_argument("--rebalance-days", type=int, default=defaults.rebalance_days)
    protocol.add_argument("--embargo-days", type=int, default=defaults.embargo_days)

    portfolio = parser.add_argument_group("portfolio")
    portfolio.add_argument("--max-weight", type=float, default=defaults.max_weight)
    portfolio.add_argument("--max-turnover", type=float, default=defaults.max_turnover)
    portfolio.add_argument("--strategy", choices=available_strategies(), default=defaults.strategy)
    portfolio.add_argument(
        "--cvar-ratio", type=float, default=None, help="cap portfolio CVaR at this multiple of the target's"
    )
    portfolio.add_argument("--cvar-alpha", type=float, default=defaults.cvar_alpha)
    portfolio.add_argument("--ridge-lambda", type=float, default=defaults.ridge_lambda)

    costs = parser.add_argument_group("costs")
    costs.add_argument("--cost-model", choices=COST_MODELS, default=defaults.cost_model)
    costs.add_argument("--cost-bps", type=float, default=defaults.cost_bps)
    costs.add_argument(
        "--notional", type=float, default=defaults.portfolio_notional, help="book size for the impact model"
    )
    costs.add_argument("--impact-coef", type=float, default=defaults.impact_coef)
    costs.add_argument("--max-participation", type=float, default=defaults.max_participation)

    inference = parser.add_argument_group("inference")
    inference.add_argument("--bootstrap-samples", type=int, default=defaults.bootstrap_samples)
    inference.add_argument("--bootstrap-block", type=int, default=defaults.bootstrap_block)
    inference.add_argument("--inference-seed", type=int, default=defaults.inference_seed)

    output = parser.add_argument_group("output")
    output.add_argument("--output", default=defaults.output_dir, help="root directory for run artefacts")
    output.add_argument("--config-file", default=None, help="load a config.json and apply the flags on top of it")
    output.add_argument("--json", action="store_true", help="print machine-readable JSON instead of tables")


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    base = ExperimentConfig()
    if getattr(args, "config_file", None):
        base = ExperimentConfig.from_json(Path(args.config_file).read_text())
    config = replace(
        base,
        target=args.target,
        assets=tuple(args.assets),
        start=args.start,
        end=args.end,
        data_source=args.data_source,
        synthetic_seed=args.seed,
        synthetic_leverage=args.leverage,
        csv_path=args.csv_path,
        offline=not args.online,
        cache_dir=args.cache_dir,
        train_days=args.train_days,
        train_mode=args.train_mode,
        rebalance_days=args.rebalance_days,
        embargo_days=args.embargo_days,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
        strategy=args.strategy,
        cvar_ratio=args.cvar_ratio,
        cvar_alpha=args.cvar_alpha,
        ridge_lambda=args.ridge_lambda,
        cost_model=args.cost_model,
        cost_bps=args.cost_bps,
        portfolio_notional=args.notional,
        impact_coef=args.impact_coef,
        max_participation=args.max_participation,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block=args.bootstrap_block,
        inference_seed=args.inference_seed,
        output_dir=args.output,
    )
    return config.validate()


def _load_checked_panel(config: ExperimentConfig, allow_degraded: bool) -> tuple[Any, Any]:
    """Load prices and run the quality gates. This is the only door into the data."""
    panel = load_panel(config)
    report = run_quality_gates(panel, min_obs=config.train_days + config.rebalance_days)
    if not allow_degraded:
        report.raise_if_failed()
    return panel, report


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_study(args: argparse.Namespace) -> int:
    """The full research workflow: horse race, recovery, sweep, costs, report."""
    from etflab.export import export_study
    from etflab.research import run_study

    config = _config_from_args(args)
    if not args.json:
        console.print(banner())
        console.print(config_panel(config))
    panel, quality = _load_checked_panel(config, args.allow_degraded)
    if not args.json and quality.status != "ok":
        console.print(quality_panel(quality))

    with console.status("[cyan]Running study…[/]", spinner="dots") as status:
        study = run_study(
            config,
            panel,
            quality,
            strategies=tuple(args.strategies) if args.strategies else DEFAULT_ZOO,
            benchmark=args.benchmark,
            with_sweep=not args.no_sweep,
            with_recovery=not args.no_recovery,
            with_capacity=not args.no_capacity,
            progress=lambda message: status.update(f"[cyan]{message}[/]"),
        )
        status.update("[cyan]Writing artefacts…[/]")
        run_dir, manifest = export_study(study, output_dir=config.output_dir, write_figures=not args.no_figures)

    if args.json:
        print(
            json.dumps(
                {"run_id": manifest.run_id, "run_dir": str(run_dir), **study.race.as_dict()}, indent=2, default=str
            )
        )
        return 0

    render_study(console, study)
    console.print()
    console.print(
        Panel(
            f"[bold]{run_dir}[/]\n"
            f"[dim]run id[/] {manifest.run_id}   [dim]digest[/] {manifest.results_digest[:16]}\n"
            f"[dim]report[/] {run_dir / 'report.html'}\n"
            f"[dim]verify[/] etf-lab verify {run_dir}",
            title="[bold green]Artefacts[/]",
            border_style="green",
            padding=(0, 2),
        )
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """The horse race alone: fast, no sweep, no report."""
    from etflab.research import run_horse_race

    config = _config_from_args(args)
    panel, quality = _load_checked_panel(config, args.allow_degraded)
    with console.status("[cyan]Running strategies…[/]", spinner="dots"):
        race = run_horse_race(panel, config, tuple(args.strategies) if args.strategies else DEFAULT_ZOO, args.benchmark)

    if args.json:
        print(json.dumps(race.as_dict(), indent=2, default=str))
        return 0

    class _Shim:  # the console renderer wants a study-shaped object
        def __init__(self) -> None:
            self.race, self.panel, self.quality = race, panel, quality

    console.print(quality_panel(quality))
    console.print()
    console.print(horse_race_table(_Shim()))
    console.print()
    console.print(verdict_panel(_Shim()))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """One strategy, one backtest. The fastest path to a number."""
    from etflab.backtest import run_backtest
    from etflab.metrics import compute_metrics
    from etflab.strategies import build_strategy

    config = _config_from_args(args)
    panel, _quality = _load_checked_panel(config, args.allow_degraded)
    result = run_backtest(panel, build_strategy(config.strategy, config), config)
    metrics = compute_metrics(result.daily, result.weights, result.rebalances)

    if args.json:
        print(json.dumps({"config_hash": config.semantic_hash(), "metrics": metrics}, indent=2, default=str))
        return 0

    table = RichTable(title=f"{config.strategy} on {config.target}", box=None, header_style="bold cyan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in (
        "tracking_error",
        "tracking_error_gross",
        "correlation",
        "beta",
        "information_ratio",
        "down_capture",
        "up_capture",
        "annual_turnover",
        "cost_drag_annual",
        "worst_21d_active",
        "replicator_max_drawdown",
        "target_max_drawdown",
        "effective_positions",
        "degraded_rebalances",
    ):
        if key not in metrics:
            continue
        value = metrics[key]
        rendered = (
            f"{value:.2%}"
            if key.endswith(("error", "turnover", "drawdown", "active", "gross", "annual"))
            else f"{value:.3f}"
        )
        table.add_row(key.replace("_", " "), rendered)
    console.print(config_panel(config))
    console.print()
    console.print(table)

    latest = result.weights.iloc[-1].sort_values(ascending=False)
    holdings = RichTable(title="Latest allocation", box=None, header_style="bold cyan")
    holdings.add_column("asset")
    holdings.add_column("weight", justify="right")
    for ticker, weight in latest[latest > 1e-6].items():
        holdings.add_row(str(ticker), f"{weight:.2%}")
    console.print()
    console.print(holdings)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Data quality gates only. Exits non-zero if a blocking gate fails."""
    config = _config_from_args(args)
    panel = load_panel(config)
    report = run_quality_gates(panel, min_obs=config.train_days + config.rebalance_days)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        console.print(quality_panel(report))
        console.print(f"\n[dim]fingerprint[/] {panel.fingerprint()}  [dim]observations[/] {panel.n_obs:,}")
    return 0 if report.ok else 2


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-run a stored run and check the results digest still matches.

    This is the reproducibility claim, made falsifiable. CI runs it on every
    commit, so "it reproduces" is a test result rather than an assurance.
    """
    import tempfile

    from etflab.export import export_study
    from etflab.provenance import RunManifest
    from etflab.research import run_study

    run_dir = Path(args.run_dir)
    stored = RunManifest.read(run_dir)
    config = ExperimentConfig.from_dict(stored.config)

    console.print(f"[dim]Re-running[/] {stored.run_id} [dim]from[/] {run_dir}")
    panel, _quality = _load_checked_panel(config, allow_degraded=True)
    if panel.fingerprint() != stored.data_fingerprint:
        console.print(
            f"[bold red]Data fingerprint changed[/]: stored {stored.data_fingerprint}, now {panel.fingerprint()}. "
            "The inputs are not the same, so a results mismatch would be uninformative."
        )
        return 2

    # The results digest depends on which optional analyses ran, so the re-run has
    # to switch on exactly the set the stored manifest recorded. Guessing produces
    # a "regression" that is really a configuration mismatch.
    analyses = dict(stored.notes.get("analyses") or {})
    with console.status("[cyan]Re-running study…[/]", spinner="dots"):
        study = run_study(
            config,
            panel,
            _quality,
            strategies=tuple(stored.notes.get("strategies", DEFAULT_ZOO)),
            benchmark=str(stored.notes.get("benchmark", "equal_weight")),
            with_sweep=bool(analyses.get("sweep", False)) or args.full,
            with_recovery=bool(analyses.get("recovery", True)),
            with_capacity=bool(analyses.get("capacity", False)) or args.full,
        )

    # Never export into the directory under test: that would overwrite the
    # manifest and artefacts that are the evidence, so the check would destroy
    # what it exists to confirm.
    with tempfile.TemporaryDirectory(prefix="etflab-verify-") as scratch:
        _, fresh = export_study(study, output_dir=scratch, write_figures=False, write_report=False)

    comparison = stored.compare(fresh)
    table = RichTable(box=None, show_header=False)
    table.add_column(style="dim", width=22)
    table.add_column()
    for key in ("same_experiment", "same_config", "same_data", "same_results"):
        ok = comparison[key]
        table.add_row(key.replace("_", " "), "[green]yes[/]" if ok else "[bold red]no[/]")
    table.add_row("code changed", "yes" if comparison["code_changed"] else "no")
    if comparison["environment_diff"]:
        table.add_row("environment diff", ", ".join(comparison["environment_diff"]))
    console.print(table)

    if comparison["same_results"]:
        console.print("\n[bold green]Reproduced.[/] Results digest is byte-identical.")
        return 0
    console.print(
        f"\n[bold red]Results changed.[/] stored {stored.results_digest[:16]} vs now {fresh.results_digest[:16]}.\n"
        "[dim]Same configuration and same data, different answer: the code changed the result. "
        "That is either a fix or a regression -- it should not be silent either way.[/]"
    )
    return 1


def cmd_runs(args: argparse.Namespace) -> int:
    """List everything in the run registry."""
    registry = RunRegistry(args.output)
    frame = registry.to_frame()
    if frame.empty:
        console.print(f"[dim]No runs recorded under {args.output}.[/]")
        return 0
    if args.json:
        print(frame.to_json(orient="records", indent=2))
        return 0
    table = RichTable(title=f"Runs in {args.output}", box=None, header_style="bold cyan")
    for column in ("run_id", "created_at", "target", "data_source", "code", "quality_status"):
        table.add_column(column)
    table.add_column("tracking_error", justify="right")
    for _, row in frame.tail(args.limit).iterrows():
        te = row.get("tracking_error")
        table.add_row(
            str(row["run_id"]),
            str(row["created_at"])[:19],
            str(row["target"]),
            str(row["data_source"]),
            str(row["code"]),
            str(row["quality_status"]),
            f"{te:.2%}" if te == te else "--",
        )
    console.print(table)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Explain the difference between two recorded runs."""
    registry = RunRegistry(args.output)
    left_matches, right_matches = registry.find(args.left), registry.find(args.right)
    if not left_matches or not right_matches:
        console.print("[bold red]Run not found.[/] Use `etf-lab runs` to list recorded runs.")
        return 2
    result = diff_records(left_matches[-1], right_matches[-1])
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    console.print(Panel(result["explanation"], title="[bold]Diagnosis[/]", border_style="cyan", padding=(0, 2)))
    if result["metrics"]:
        table = RichTable(box=None, header_style="bold cyan")
        table.add_column("metric")
        table.add_column(args.left, justify="right")
        table.add_column(args.right, justify="right")
        table.add_column("delta", justify="right")
        for metric, values in result["metrics"].items():
            table.add_row(
                metric,
                f"{values['left']:.4f}" if isinstance(values["left"], (int, float)) else str(values["left"]),
                f"{values['right']:.4f}" if isinstance(values["right"], (int, float)) else str(values["right"]),
                f"{values['delta']:+.4f}" if isinstance(values["delta"], (int, float)) else "--",
            )
        console.print()
        console.print(table)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Parameter sweep with overfitting diagnostics."""
    from etflab.research import run_sweep

    config = _config_from_args(args)
    panel, _ = _load_checked_panel(config, args.allow_degraded)
    with console.status("[cyan]Sweeping…[/]", spinner="dots") as status:
        result = run_sweep(panel, config, progress=lambda i, n, label: status.update(f"[cyan]{i}/{n} {label}[/]"))
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0
    table = RichTable(title=f"{len(result.frame)} configurations", box=None, header_style="bold cyan")
    table.add_column("configuration")
    table.add_column("tracking error", justify="right")
    table.add_column("turnover/yr", justify="right")
    for name, row in result.frame.head(args.limit).iterrows():
        table.add_row(str(name), f"{row['tracking_error']:.2%}", f"{row.get('annual_turnover', float('nan')):.1%}")
    console.print(table)
    console.print()
    console.print(
        Panel(
            f"PBO {result.pbo.pbo:.0%} against a pure-noise reference of {result.noise_pbo:.0%} "
            f"for a grid this shape -- {result.pbo.verdict}.\n"
            f"Deflated Sharpe probability {result.deflated.deflated_probability:.1%} "
            f"across {result.deflated.n_trials} trials.",
            title="[bold]Overfitting[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# Interactive terminal
# --------------------------------------------------------------------------- #
HELP = """\
[bold cyan]Experiment[/]
  [bold]show[/]                          show the current configuration
  [bold]target[/] TICKER                 change the replication target
  [bold]assets[/] TICKER ...             replace the candidate universe
  [bold]dates[/] START END                set the research period
  [bold]set[/] FIELD VALUE               any config field, e.g. [dim]set train_days 756[/]

[bold cyan]Run[/]
  [bold green]/run[/]                          one backtest of the current strategy
  [bold green]/compare[/]                      the full horse race with inference
  [bold green]/study[/]                        everything, and write the report
  [bold]/validate[/]                     data quality gates only

[bold cyan]Session[/]
  [bold]/help[/]  [bold]/quit[/]
"""

_SET_ALIASES = {
    "rebalance": "rebalance_days",
    "training": "train_days",
    "costs": "cost_bps",
    "turnover": "max_turnover",
    "max-weight": "max_weight",
    "model": "strategy",
}


def _apply_set(config: ExperimentConfig, field: str, value: str) -> ExperimentConfig:
    from dataclasses import fields as dataclass_fields

    name = _SET_ALIASES.get(field.lower(), field.lower().replace("-", "_"))
    known = {f.name: f.type for f in dataclass_fields(ExperimentConfig)}
    if name not in known:
        raise ValueError(f"Unknown field {field!r}. Known fields: {', '.join(sorted(known))}")
    current = getattr(config, name)
    if isinstance(current, bool):
        parsed: Any = value.lower() in {"1", "true", "yes", "on"}
    elif isinstance(current, int) and not isinstance(current, bool):
        parsed = int(value)
    elif isinstance(current, float) or (current is None and name in {"cvar_ratio"}):
        parsed = float(value)
    else:
        parsed = value
    return config.with_changes(**{name: parsed})


def interactive(args: argparse.Namespace) -> int:
    from rich.prompt import Prompt

    config = _config_from_args(args)
    console.print(banner())
    console.print("Type [bold]/help[/] for commands, or [bold green]/study[/] to run everything with the defaults.\n")
    console.print(config_panel(config))

    while True:
        try:
            raw = Prompt.ask("\n[bold cyan]etf-lab[/]").strip()
            if not raw:
                continue
            tokens = shlex.split(raw)
            command = tokens[0].lower().lstrip("/")

            if command in {"quit", "exit", "q"}:
                console.print("[dim]Session closed.[/]")
                return 0
            if command == "help":
                console.print(Panel(HELP, border_style="blue", padding=(1, 2)))
            elif command == "show":
                console.print(config_panel(config))
            elif command == "target" and len(tokens) == 2:
                config = config.with_changes(target=tokens[1])
                console.print("[green]updated[/]")
            elif command == "assets" and len(tokens) >= 3:
                config = config.with_changes(assets=tuple(tokens[1:]))
                console.print("[green]updated[/]")
            elif command == "dates" and len(tokens) == 3:
                config = config.with_changes(start=tokens[1], end=tokens[2])
                console.print("[green]updated[/]")
            elif command == "set" and len(tokens) == 3:
                config = _apply_set(config, tokens[1], tokens[2])
                console.print("[green]updated[/]")
            elif command in {"run", "compare", "study", "validate"}:
                sub = argparse.Namespace(**vars(args))
                for key, value in _config_to_args(config).items():
                    setattr(sub, key, value)
                sub.json = False
                {"run": cmd_run, "compare": cmd_compare, "study": cmd_study, "validate": cmd_validate}[command](sub)
            else:
                console.print("[yellow]Unrecognised.[/] Type [bold]/help[/] for the command list.")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session closed.[/]")
            return 0
        except DataQualityError as exc:
            console.print(Panel(str(exc), title="[bold red]Data quality[/]", border_style="red"))
        except Exception as exc:
            console.print(f"[bold red]{type(exc).__name__}:[/] {exc}")


def _config_to_args(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "target": config.target,
        "assets": list(config.assets),
        "start": config.start,
        "end": config.end,
        "data_source": config.data_source,
        "seed": config.synthetic_seed,
        "leverage": config.synthetic_leverage,
        "csv_path": config.csv_path,
        "online": not config.offline,
        "cache_dir": config.cache_dir,
        "train_days": config.train_days,
        "train_mode": config.train_mode,
        "rebalance_days": config.rebalance_days,
        "embargo_days": config.embargo_days,
        "max_weight": config.max_weight,
        "max_turnover": config.max_turnover,
        "strategy": config.strategy,
        "cvar_ratio": config.cvar_ratio,
        "cvar_alpha": config.cvar_alpha,
        "ridge_lambda": config.ridge_lambda,
        "cost_model": config.cost_model,
        "cost_bps": config.cost_bps,
        "notional": config.portfolio_notional,
        "impact_coef": config.impact_coef,
        "max_participation": config.max_participation,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_block": config.bootstrap_block,
        "inference_seed": config.inference_seed,
        "output": config.output_dir,
    }


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etf-lab",
        description="Reproducible ETF replication research. Runs offline by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  etf-lab study                          full workflow, writes a report\n"
            "  etf-lab compare --json                 horse race as JSON\n"
            "  etf-lab run --strategy equal_weight    single backtest\n"
            "  etf-lab verify outputs/<run-id>        prove the result reproduces\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"etf-lab {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    study = subparsers.add_parser("study", help="run the full research workflow and write a report")
    _add_experiment_args(study)
    study.add_argument("--strategies", nargs="+", choices=available_strategies(), default=None)
    study.add_argument("--benchmark", default="equal_weight", choices=available_strategies())
    study.add_argument("--no-sweep", action="store_true", help="skip the parameter sweep and PBO")
    study.add_argument("--no-recovery", action="store_true")
    study.add_argument("--no-capacity", action="store_true")
    study.add_argument("--no-figures", action="store_true")
    study.set_defaults(func=cmd_study)

    compare = subparsers.add_parser("compare", help="run the strategy horse race with inference")
    _add_experiment_args(compare)
    compare.add_argument("--strategies", nargs="+", choices=available_strategies(), default=None)
    compare.add_argument("--benchmark", default="equal_weight", choices=available_strategies())
    compare.set_defaults(func=cmd_compare)

    run = subparsers.add_parser("run", help="run a single strategy backtest")
    _add_experiment_args(run)
    run.set_defaults(func=cmd_run)

    sweep = subparsers.add_parser("sweep", help="sweep the parameter grid and measure overfitting")
    _add_experiment_args(sweep)
    sweep.add_argument("--limit", type=int, default=12)
    sweep.set_defaults(func=cmd_sweep)

    validate = subparsers.add_parser("validate", help="run data quality gates only")
    _add_experiment_args(validate)
    validate.set_defaults(func=cmd_validate)

    verify = subparsers.add_parser("verify", help="re-run a stored run and check the results digest")
    verify.add_argument("run_dir")
    verify.add_argument("--output-dir", default=None)
    verify.add_argument("--full", action="store_true", help="also re-run the sweep and capacity analysis")
    verify.set_defaults(func=cmd_verify)

    runs = subparsers.add_parser("runs", help="list recorded runs")
    runs.add_argument("--output", default="outputs")
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(func=cmd_runs)

    diff = subparsers.add_parser("diff", help="explain the difference between two runs")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.add_argument("--output", default="outputs")
    diff.add_argument("--json", action="store_true")
    diff.set_defaults(func=cmd_diff)

    shell = subparsers.add_parser("shell", help="interactive research terminal")
    _add_experiment_args(shell)
    shell.add_argument("--strategies", nargs="+", choices=available_strategies(), default=None)
    shell.add_argument("--benchmark", default="equal_weight", choices=available_strategies())
    shell.add_argument("--no-sweep", action="store_true")
    shell.add_argument("--no-recovery", action="store_true")
    shell.add_argument("--no-capacity", action="store_true")
    shell.add_argument("--no-figures", action="store_true")
    shell.set_defaults(func=interactive)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        args = parser.parse_args(["shell"])
    try:
        return int(args.func(args))
    except DataQualityError as exc:
        console.print(
            Panel(str(exc), title="[bold red]Data quality gate failed[/]", border_style="red", padding=(1, 2))
        )
        return 2
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        return 130
    except Exception as exc:
        console.print(
            Panel(f"{type(exc).__name__}: {exc}", title="[bold red]Failed[/]", border_style="red", padding=(1, 2))
        )
        if "--traceback" in (argv or sys.argv):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
