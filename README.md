# Synthetic ETF Replicator

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CVXPY](https://img.shields.io/badge/CVXPY-convex%20optimization-8B0000.svg)
![Status](https://img.shields.io/badge/status-research%20demo-lightgrey.svg)

This repository is a terminal playground for testing whether a basket of ETFs can replicate a target. It pulls market data with `yfinance`, estimates weights with constrained optimization, and evaluates them in a walk-forward simulation with weight drift and transaction costs.

The current default target is `PSP`, using a basket of liquid ETFs as proxies. Replication is a research problem here—not a claim that the synthetic basket is a useful trading strategy. The interesting question is whether replication works under explicit constraints and whether it beats simple alternatives after costs.

There are two front-ends over the same research question:

- **`etf-lab`** — an interactive research terminal for iterating on experiments, with reproducible run artifacts
- **`etf-replicate`** — a one-shot batch run that renders a committee-ready HTML tearsheet, with instrument-type group constraints and stress-window evidence

## Quickstart

```bash
git clone https://github.com/khuper/etf-replication-demo
cd etf-replication-demo
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -e .
```

## Research terminal

Launch the interactive terminal:

```bash
etf-lab
```

The terminal keeps an experiment configuration in memory and accepts short commands:

```text
target PSP
assets SPY QQQ VEA VWO BND LQD TIP GLD VNQ
dates 2020-01-01 2026-01-01
set rebalance 63
set costs 5
set model tracking
/run
```

Each run displays out-of-sample metrics and the latest allocation, then exports its exact configuration, metrics, returns, weights, and turnover history.

For scripts and repeatable experiments, use the one-shot interface:

```bash
etf-lab run \
  --target PSP \
  --assets SPY QQQ VEA VWO BND LQD TIP GLD VNQ \
  --start 2020-01-01 \
  --end 2026-01-01 \
  --rebalance-days 63 \
  --costs-bps 5
```

Terminal runs create a timestamped directory under `outputs/` containing:

- `config.json`
- `metrics.json`
- `returns.csv`
- `weights.csv`
- `turnover.csv`

The terminal and one-shot command expose the main research inputs without source edits: asset tickers, target ticker, start and end dates, `max_weight`, `max_turnover`, `initial_train_size`, rebalance frequency, model (`tracking` or `cvar`), and transaction costs.

## Strategy tearsheet

```bash
# Replicate PSP with the default liquid basket (10 years of history)
etf-replicate

# Rolling correlation diagnostics for the basket
etf-correlations
```

![Example tearsheet](assets/tearsheet_preview.png)

`etf-replicate` writes everything to `outputs/`:

| File | What it is |
| --- | --- |
| `tearsheet.html` | Self-contained strategy tearsheet (open in any browser, share as a file) |
| `backtest_metrics.csv` | Gross and net headline metrics |
| `weights_history.csv` | Allocation at every rebalance |
| `full_sample_weights.csv` | "Ideal today" allocation using all history |
| `*.png` | Every chart individually (cumulative returns, drawdown, rolling tracking, allocation, correlations) |

The tearsheet reports what a strategy reviewer would ask for:

- **Headline tiles** — annualized tracking error, correlation, information ratio, active return,
  max active drawdown, average turnover, total cost drag (all net of costs)
- **Consistency** — monthly active-return grid with a hit rate (share of months tracking within
  ±50 bps), best/worst months
- **Stress windows** — replication quality inside the COVID crash and the 2022 rate shock
- **Cost sensitivity** — the same headline metrics at 0 / 10 / 25 bps, so results aren't an
  artifact of the cost assumption

All parameters are flags — no code edits needed:

```bash
etf-replicate \
  --target PSP \
  --assets SPY QQQ VEA VWO BND AGG LQD TIP GLD VNQ \
  --start 2019-01-01 \
  --max-weight 0.25 \
  --max-turnover 0.20 \
  --cost-bps 10 \
  --step 126
```

### Express preferences about what you trade

Every basket ticker carries an instrument-type tag (`equity`, `fixed_income`, `commodity`,
`real_estate`, …). Group bounds constrain how much of the portfolio each type may hold, so the
basket reflects what you are actually willing to trade:

```bash
# Keep equities between 20% and 60%, hold no commodities at all
etf-replicate --group-bound equity=0.2:0.6 --group-bound commodity=0:0

# Pin fixed income to exactly 30% (min = max is a hard target)
etf-replicate --group-bound fixed_income=0.3:0.3

# Tag your own tickers, then bound the group
etf-replicate --assets SPY QQQ DBC GLD BND --asset-class DBC=commodity --group-bound commodity=0:0.15
```

Infeasible combinations (e.g. group minimums that sum past 100%) are rejected with a clear error
before any optimization runs.

## Method at a glance

```text
yfinance prices
  -> daily returns
  -> constrained optimization using prior data only
  -> walk-forward holdings with drift and trading costs
  -> metrics, allocations, and reproducible run artifacts
```

The default optimization in `src/replicator.py` focuses on:

- minimizing squared tracking error versus the target
- enforcing long-only weights
- capping single-position concentration
- limiting turnover between rebalance steps

The CVaR constraint remains available as an optional model (`set model cvar` or `--model cvar`) so its effect can be compared with plain tracking-error minimization instead of being silently imposed.

The `etf_replicator` package solves the same tracking-error problem with the CVaR constraint on by
default (`--cvar-ratio`, α via `--cvar-alpha`) plus optional instrument-type group bounds, runs an
expanding-window backtest that charges `--cost-bps` per unit of L1 turnover on each rebalance
(the first rebalance pays for a full deployment from cash), and renders the tearsheet.

## Correlation diagnostics

`etf-correlations` analyzes how each basket ETF's correlation to the target behaves over time and
inside historical stress windows — if correlations spike toward 1 in a crash, diversification
inside the basket disappears exactly when replication matters most. It saves per-asset statistics
(`correlation_stats.csv`), a small-multiples grid, and a mean-correlation chart:

![Correlation heatmap](assets/correlation_heatmap.png)

## Use it as a library

```python
from etf_replicator import fetch_prices, to_returns, run_expanding_backtest, summarize_backtest

prices = fetch_prices(["SPY", "QQQ", "GLD", "PSP"], "2020-01-01", "2025-01-01")
returns = to_returns(prices)
result = run_expanding_backtest(returns, ["SPY", "QQQ", "GLD"], "PSP", cost_bps=10)
print(summarize_backtest(result).round(4))
```

## What is in the repo

| File | Purpose |
| --- | --- |
| [`src/replicator.py`](src/replicator.py) | Main workflow for data download, optimization, walk-forward backtest, stress testing, and plot generation. |
| [`src/cli.py`](src/cli.py) | Interactive terminal and reproducible one-shot command. |
| [`src/config.py`](src/config.py) | Validated experiment configuration. |
| [`src/reporting.py`](src/reporting.py) | Rich terminal tables and run exports. |
| [`src/rolling_correlation_analysis.py`](src/rolling_correlation_analysis.py) | Separate analysis script for rolling ETF-to-target correlations across normal and stress periods. |
| [`etf_replicator/`](etf_replicator/) | Package behind `etf-replicate` / `etf-correlations`: config with instrument-type tags ([`config.py`](etf_replicator/config.py)), data ([`data.py`](etf_replicator/data.py)), CVaR/turnover/group-constrained optimizer ([`optimizer.py`](etf_replicator/optimizer.py)), backtest with costs ([`backtest.py`](etf_replicator/backtest.py)), metrics ([`metrics.py`](etf_replicator/metrics.py)), HTML tearsheet ([`report.py`](etf_replicator/report.py)), correlations ([`correlations.py`](etf_replicator/correlations.py)), stress test ([`stress.py`](etf_replicator/stress.py)), charts ([`plotting.py`](etf_replicator/plotting.py)), CLI ([`cli.py`](etf_replicator/cli.py)). |
| [`assets/`](assets/) | Example tearsheet preview and charts embedded in this README. |

## Testing

```bash
python -m unittest discover -s tests -v
```

The suite runs fully offline (market data is mocked, backtests use synthetic returns) and covers
both the terminal workflow and the package: configuration validation, optimizer constraints,
walk-forward accounting, metrics, stress test, and tearsheet generation. A GitHub Actions
workflow runs it on pushes and pull requests.

## Limitations

Still a research demo, not a production trading system:

- market data comes from `yfinance`, which is convenient but not institutional-grade
- transaction costs use a simple basis-point estimate; spread and market-impact models are not included
- the beta stress test is intentionally simple (historical stress *windows* in the tearsheet are
  the more informative view)
- there is no price-data cache yet
- the project does not yet compare the optimized result with simple regression and equal-weight baselines
- the `etf_replicator` backtest assumes weights are held at the rebalance target between rebalances

## License

Released under the [MIT License](LICENSE).
