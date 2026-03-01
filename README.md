# Synthetic Liability Replicator

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CVXPY](https://img.shields.io/badge/CVXPY-convex%20optimization-8B0000.svg)
![Status](https://img.shields.io/badge/status-learning%20project-lightgrey.svg)

---

## Summary

This engine builds **synthetic portfolios** of liquid ETFs that replicate the risk/return profile of illiquid assets (e.g., Private Equity). Instead of chasing speculative alpha, it minimizes *tracking error*—the gap between the portfolio and the target benchmark—while constraining tail risk (CVaR) and enforcing position and turnover limits. The result: institutions can approximate illiquid exposure with daily-tradable instruments, without lock-ups or steep illiquidity premiums.

---

## How It Works

A short walkthrough of the pipeline:

```
yfinance (historical prices, auto_adjust=True)
    → percentage returns
    → CVXPY optimizer (minimize TE, constrain CVaR, position limits, turnover)
    → expanding-window backtest (no look-ahead bias; last_i advances only on successful optimization)
    → stress test (vectorized beta-weighted shock simulation)
    → matplotlib / seaborn plots
```

| File | Purpose |
|------|---------|
| [`src/replicator.py`](src/replicator.py) | Main engine: data fetch, CVXPY-based tracking-error optimization with CVaR constraints, expanding-window backtest, stress testing, and visualization. |
| [`src/rolling_correlation_analysis.py`](src/rolling_correlation_analysis.py) | Rolling correlation study: 60/120-day correlations between liquid ETFs and the target, with stress-period shading (COVID crash, 2022 rate shock). |

---

## Why This Project?

Institutional portfolio management is rarely about chasing outsized returns; it is fundamentally an exercise in strict risk budgeting and liability matching. This engine is built to solve a specific, high-stakes institutional problem: how to gain the risk/return exposure of an illiquid asset class while maintaining daily liquidity and capping catastrophic downside.

Here is how this engine differentiates itself from standard academic portfolio optimizers:

### 1. Transition from VaR to CVaR (Tail Risk Focus)

Traditional Value-at-Risk (VaR) simply answers, *"What is the most we could lose on 95% of days?"* However, it ignores what happens in the worst 5% of days—the exact moments when financial systems break. By explicitly constraining **Conditional Value-at-Risk (CVaR)** via a custom CVXPY solver, this engine optimizes for the *average loss during tail-risk events*, ensuring the replicator survives market regime shifts and liquidity crunches.

### 2. Synthetic Liquidity (Replicating Illiquid Targets)

Private Equity, Pension Liabilities, and direct Real Estate often require capital lock-ups of 5 to 10 years. Rebalancing or divesting during a crisis is impossible. This engine constructs a "Synthetic Proxy" using highly liquid, daily-traded ETFs. It algorithmically weights these liquid proxies to mirror the exact correlation and behavioral profile of the illiquid target, granting institutions the flexibility to pivot without paying steep illiquidity premiums.

### 3. Tracking Error Minimization vs. Speculative Alpha

Retail optimization models usually attempt to maximize the Sharpe Ratio (return per unit of volatility). This introduces massive speculative "Look-Ahead Bias." In contrast, a replicator is agnostic to absolute returns. Its sole objective is to **Minimize Tracking Error**—ensuring the portfolio's delta relative to the benchmark approaches zero. We aren't predicting the market; we are surgically engineering exposure to a specific benchmark.

---

## Architectural Comparison

| Feature | Standard Retail/Student Project | Institutional Replicator (This Engine) |
| :--- | :--- | :--- |
| **Primary Objective** | Maximize Absolute Alpha / Sharpe Ratio | Minimize Tracking Error (Benchmark Replication) |
| **Risk Measurement** | Standard Deviation (Mean-Variance) | Conditional Value-at-Risk (Mean-CVaR) |
| **Target Asset** | Generic S&P 500 / Crypto | Illiquid Alternatives (e.g., Private Equity, Pensions) |
| **Liquidity Profile** | Trades highly liquid assets for speculation | Uses liquid proxies to synthesize illiquid profiles |
| **Solver Infrastructure** | Scipy.optimize (Unconstrained) | CVXPY (Convex Optimization with Hard Risk Boundaries) |
| **Stress Testing** | None (Relies entirely on historical mean) | Simulates systemic shocks and regime shifts |
| **Backtesting Rigor** | In-sample static weights (Look-Ahead Bias) | Expanding-window, out-of-sample forward walks |

---

## Quickstart

```bash
git clone https://github.com/khuper/etf-replication-demo
cd etf-replication-demo
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the main replicator (optimization + backtest + stress test + plots):

```bash
python src/replicator.py
```

Run the rolling correlation analysis:

```bash
python src/rolling_correlation_analysis.py
```

**Outputs:**
- `correlation_heatmap.png` — asset–target correlation matrix
- `cumulative_returns.png` — replicator vs target cumulative performance
- `outputs/` — correlation stats CSV and rolling correlation plots

---

## Configuration

All parameters are defined in the `main()` block of [`src/replicator.py`](src/replicator.py):

| Parameter | Location | Description |
|-----------|----------|-------------|
| `assets` | `main()` in `src/replicator.py` | List of liquid ETF tickers (proxies) |
| `target` | `main()` in `src/replicator.py` | Illiquid index/ETF to replicate (default: PSP) |
| `start_date` / `end_date` | `main()` in `src/replicator.py` | Data window (default: 5 years) |
| `max_weight` | `backtest_expanding_window()` / `optimize_tracking_error()` | Max single-asset weight (default: 25%) |
| `max_turnover` | `backtest_expanding_window()` | Max per-period turnover (default: 20%) |
| `initial_train_size` | `backtest_expanding_window()` args | Backtest initial training window (default: 504 trading days) |
| `step` | `backtest_expanding_window()` args | Backtest rebalance frequency (default: 126 days) |

Edit these inline to change tickers, constraints, or lookback periods.

---

## Limitations & Extension Ideas

This is a learning project. In a production setting, several practical gaps would need to be addressed. If you're forking or extending this repo, here are high-impact directions:

### Data & Infrastructure
- **Institutional data sources** — Replace `yfinance` with Bloomberg/Refinitiv or a proper market data API. Handle survivorship bias and corporate actions explicitly.
- **Robust data pipeline** — Add validation, retries, and error handling. Consider pre-computed returns or a local cache to avoid repeated API calls.
- **Transaction costs** — Model bid-ask spread, commissions, and market impact. Integrate costs into the optimization objective or at least run net-of-cost backtests.

### Methodology
- **Out-of-sample CVaR validation** — The CVaR constraint is fit in-sample. Add walk-forward checks to see how often the portfolio violates the constraint out-of-sample, or explore parametric/simulation-based CVaR.
- **Regime-adaptive windows** — Experiment with adaptive lookback lengths, regime detection (e.g., volatility regimes), or more conservative assumptions in stress periods.
- **Multiple benchmarks** — Extend to multi-target tracking (e.g., pension liabilities with different horizons). Add duration-matching or cash-flow-aware constraints.

### Target Representation
- **Stale pricing** — PSP and similar indices use lagged, quarterly NAVs. Model this explicitly (e.g., lagged target returns) or document the bias.
- **Alternative proxies** — Try other illiquid targets (REITs, hedge fund indices, infrastructure) and compare replication quality.

### Production Readiness
- **Config management** — Move tickers, dates, and hyperparameters to a config file (YAML/JSON). Version configs with the code.
- **Monitoring & alerts** — Add tracking error alerts, data staleness checks, and solver failure notifications.
- **Audit trail** — Log runs, parameters, and outputs for reproducibility and compliance.

---

## Acknowledgements

This project draws on the following libraries and references:

- [CVXPY](https://www.cvxpy.org/) — Convex optimization framework for the tracking-error and CVaR formulations
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) — Portfolio optimization reference and inspiration
- [yfinance](https://github.com/ranaroussi/yfinance) — Historical market data via Yahoo Finance (`auto_adjust=True` for adjusted prices)
<<<<<<< HEAD
=======
=======
# etf-replication-demo
Python project that builds a synthetic ETF portfolio to approximate an illiquid target, with a focus on risk controls and duration / convexity style matching goal
>>>>>>> 48d7542b4920946e8c2e48e17e4fed23aaa76584
