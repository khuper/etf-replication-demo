"""Command-line entry points: ``etf-replicate`` and ``etf-correlations``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

from .backtest import run_expanding_backtest
from .config import DEFAULT_ASSETS, DEFAULT_TARGET, ReplicatorConfig
from .correlations import DEFAULT_CORRELATION_ASSETS, run_correlation_analysis
from .data import fetch_prices, split_assets_target, to_returns
from .metrics import summarize_backtest
from .optimizer import optimize_tracking_error
from .plotting import plot_correlation_heatmap
from .report import build_tearsheet
from .stress import beta_stress_test

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )


def _parse_asset_classes(pairs: Optional[List[str]]) -> Dict[str, str]:
    tags = {}
    for pair in pairs or []:
        ticker, _, cls = pair.partition("=")
        if not ticker or not cls:
            raise argparse.ArgumentTypeError(f"--asset-class expects TICKER=class, got '{pair}'")
        tags[ticker.upper()] = cls.lower()
    return tags


def _parse_group_bounds(pairs: Optional[List[str]]) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for pair in pairs or []:
        group, _, span = pair.partition("=")
        lo, sep, hi = span.partition(":")
        if not group or not sep:
            raise argparse.ArgumentTypeError(f"--group-bound expects GROUP=min:max, got '{pair}'")
        try:
            bounds[group.lower()] = (float(lo), float(hi))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--group-bound expects numeric min:max, got '{pair}'") from exc
    return bounds


def _replicate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etf-replicate",
        description="Replicate an illiquid target ETF with a liquid basket via constrained tracking-error optimization.",
    )
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS), help="basket tickers")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="target ticker to replicate")
    parser.add_argument("--start", default="", help="start date YYYY-MM-DD (default: 5 years back)")
    parser.add_argument("--end", default="", help="end date YYYY-MM-DD (default: today)")
    parser.add_argument("--max-weight", type=float, default=0.25, help="per-position cap")
    parser.add_argument("--max-turnover", type=float, default=0.20, help="L1 turnover cap per rebalance")
    parser.add_argument("--cvar-ratio", type=float, default=1.0, help="portfolio CVaR limit as a multiple of target CVaR")
    parser.add_argument("--cvar-alpha", type=float, default=0.05, help="CVaR tail probability")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="transaction cost in bps per unit turnover")
    parser.add_argument("--initial-train-size", type=int, default=504, help="observations before the first rebalance")
    parser.add_argument("--step", type=int, default=126, help="observations between rebalances")
    parser.add_argument("--output-dir", default="outputs", help="directory for generated files")
    parser.add_argument(
        "--asset-class", action="append", metavar="TICKER=CLASS",
        help="tag a ticker with an instrument type, e.g. GLD=commodity (repeatable)",
    )
    parser.add_argument(
        "--group-bound", action="append", metavar="GROUP=MIN:MAX",
        help="bound total weight of an instrument type, e.g. equity=0.2:0.6 or commodity=0:0 (repeatable)",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def replicate_main(argv: Optional[List[str]] = None) -> int:
    args = _replicate_parser().parse_args(argv)
    _setup_logging(args.verbose)

    config = ReplicatorConfig(
        assets=[a.upper() for a in args.assets],
        target=args.target.upper(),
        start_date=args.start,
        end_date=args.end,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
        cvar_ratio=args.cvar_ratio,
        cvar_alpha=args.cvar_alpha,
        cost_bps=args.cost_bps,
        initial_train_size=args.initial_train_size,
        step=args.step,
        output_dir=args.output_dir,
        group_bounds=_parse_group_bounds(args.group_bound),
    )
    config.asset_classes.update(_parse_asset_classes(args.asset_class))
    try:
        config.validate()
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2

    returns = to_returns(fetch_prices(config.assets + [config.target], config.start_date, config.end_date))
    os.makedirs(config.output_dir, exist_ok=True)

    result = run_expanding_backtest(
        returns,
        config.assets,
        config.target,
        initial_train_size=config.initial_train_size,
        step=config.step,
        max_weight=config.max_weight,
        max_turnover=config.max_turnover,
        cvar_ratio=config.cvar_ratio,
        cvar_alpha=config.cvar_alpha,
        cost_bps=config.cost_bps,
        group_bounds=config.group_bounds,
        asset_classes=config.asset_classes,
    )

    summary = summarize_backtest(result)
    logger.info("Backtest summary (out-of-sample):\n%s", summary.round(4).to_string())
    summary.to_csv(os.path.join(config.output_dir, "backtest_metrics.csv"))
    result.weights.to_csv(os.path.join(config.output_dir, "weights_history.csv"))

    plot_correlation_heatmap(returns, os.path.join(config.output_dir, "correlation_heatmap.png"))
    tearsheet = build_tearsheet(config, result, config.output_dir)

    # Full-sample "ideal today" allocation and a beta stress test on it.
    asset_returns, target_returns = split_assets_target(returns, config.assets, config.target)
    ideal = optimize_tracking_error(
        asset_returns,
        target_returns,
        cvar_ratio=config.cvar_ratio,
        cvar_alpha=config.cvar_alpha,
        max_weight=config.max_weight,
        group_bounds=config.group_bounds,
        asset_classes=config.asset_classes,
    )
    if ideal is not None:
        logger.info("Full-sample allocation (no turnover constraint):\n%s", ideal.round(4).to_string())
        ideal.to_csv(os.path.join(config.output_dir, "full_sample_weights.csv"))
        stress = beta_stress_test(asset_returns, target_returns, ideal, shock=-0.20)
        logger.info(
            "Beta stress test (target %+.0f%%): portfolio impact %.2f%%, relative %.2f%%",
            stress["shock"] * 100, stress["portfolio_impact"] * 100, stress["relative_performance"] * 100,
        )
    else:
        logger.warning("Full-sample optimization failed; skipping stress test.")

    logger.info("Done. Tearsheet: %s", tearsheet)
    return 0


def _correlations_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etf-correlations",
        description="Rolling ETF-to-target correlation analysis across normal and stress periods.",
    )
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_CORRELATION_ASSETS), help="basket tickers")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="target ticker")
    parser.add_argument("--start", default="2018-01-01", help="start date YYYY-MM-DD")
    parser.add_argument("--end", default="", help="end date YYYY-MM-DD (default: today)")
    parser.add_argument("--output-dir", default="outputs", help="directory for generated files")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def correlations_main(argv: Optional[List[str]] = None) -> int:
    args = _correlations_parser().parse_args(argv)
    _setup_logging(args.verbose)

    from datetime import datetime

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    run_correlation_analysis(
        [a.upper() for a in args.assets], args.target.upper(), args.start, end, args.output_dir
    )
    return 0


if __name__ == "__main__":
    sys.exit(replicate_main())
