# Methodology

What is estimated, how it is evaluated, and which estimator each claim rests on.

## The problem

Given a target instrument `y` and a candidate universe of `n` proxies `X`, find weights `w` minimising the variance of the active return `Xw − y`, subject to constraints a real mandate would impose:

```
minimise    ‖ X w − y ‖²
subject to  1ᵀw = 1          fully invested
            0 ≤ w ≤ c        long only, position cap c
            ‖w − w₋‖₁ ≤ τ    turnover cap against current holdings
```

Optionally a tail-risk budget: portfolio CVaR at level α no greater than a multiple of the target's own CVaR over the same window, in the Rockafellar–Uryasev linear form.

Everything is solved with CVXPY, CLARABEL first, with SCS and ECOS as fallbacks. A window that no solver can answer is recorded as degraded and the engine holds its previous position rather than raising — an unsolvable window is an operational event, not a reason to lose the other forty-five rebalances.

## The walk-forward protocol

At each rebalance index `i` in the return series:

1. **Fit** on returns `[lo, i − embargo)`. Strictly prior. The embargo drops the observations immediately before the decision, standing in for data arrival, model runtime and getting the order to market. Default 1 day.
2. **Trade** into the fitted weights at the close of day `i`. Costs are charged on the distance from the *drifted* holdings — measuring against the previous target instead would understate what actually has to be traded.
3. **Hold** through `[i, i + step)`, letting weights drift with prices. No free daily rebalancing.
4. Repeat, including the final partial period.

`train_mode` selects an expanding window (default) or a rolling one of fixed length.

This is the only place in the codebase where a decision date meets a return date, which is the only place look-ahead bias can enter. `tests/test_lookahead.py` verifies point 1 by construction rather than by inspection.

## Costs

Two models, both explicit about their assumptions.

**Flat.** `cost = Σ|Δwᵢ| × bps`. Charged on traded notional, not on the halved conventional turnover figure: each leg of a rebalance crosses a spread, and deploying out of cash crosses one for the whole book.

**Spread and impact.**

```
cost_bps(i) = half_spread(i) + k · σ_daily_bps(i) · √(Qᵢ / ADVᵢ)
```

the standard concave impact form (Almgren et al. 2005; Grinold & Kahn). `Qᵢ` is the traded notional in asset `i`, so cost scales with the size of the book and the capacity question becomes answerable. The volatility term uses trailing data only — taking it from the period being traded into would price the trade with the future, and there is a test for exactly that.

The liquidity table in `etflab/costs.py` is illustrative rather than measured, stated in one versioned place so a reader can disagree with a number and re-run.

## The model zoo

Every strategy sees the same training window, the same previous holdings and the same constraint set, and returns weights inside it. Heuristic strategies state what they want and are projected onto the feasible set by Euclidean projection, so the position and turnover caps bind on the baselines exactly as they bind on the optimiser.

| Strategy | Estimator |
| --- | --- |
| `tracking` | Constrained least squares on the active return |
| `cvar` | The same, plus a Rockafellar–Uryasev CVaR budget |
| `ridge` | Tracking error plus an L2 pull toward equal weight |
| `shrunk` | Minimum active variance with a Ledoit–Wolf shrunk joint covariance |
| `ols_projected` | Unconstrained OLS, then repaired onto the feasible set |
| `static` | Optimise once, then hold — the zero-turnover control |
| `top_correlation` | Fill the position cap with the highest-correlation proxies |
| `inverse_vol` | Weights ∝ 1/σ, projected |
| `equal_weight` | 1/N, projected — the benchmark |

## Inference

**Confidence intervals.** Stationary bootstrap (Politis & Romano 1994) with geometric block lengths, mean block 21 days. Two strategies are always resampled *jointly*: resampling them independently would inflate the variance of their difference and hide real differences.

**Two strategies.** Diebold–Mariano on the loss differential of squared active returns, with a Newey–West HAC variance at the automatic bandwidth `⌊4(T/100)^(2/9)⌋`, the Harvey–Leybourne–Newbold small-sample correction, and a t reference distribution. Null: equal expected squared tracking error.

**Nine strategies.** White's Reality Check and Hansen's SPA test. Null: no strategy in the zoo has lower expected loss than the benchmark. Both are reported — the Reality Check is conservative because poor models drag its null around, and SPA recovers the power by studentising and recentring; showing both makes the difference visible instead of letting the choice of test do the arguing.

**A grid of configurations.** Combinatorially Symmetric Cross-Validation over every balanced split, giving the probability that a configuration selected in sample lands below the median out of sample. Compared against a matched pure-noise reference of the same shape rather than against 0.5, because CSCV on a fixed sample has a null above one half: the in-sample and out-of-sample halves partition one finite history, so a configuration that ran hot in one has mechanically less luck left for the other.

**The selected configuration's ratio.** Deflated Sharpe Ratio (Bailey & López de Prado 2014), which prices in the number of trials, the non-normality of the returns and the sample length. The correction is only as honest as the trial count admitted to, so every configuration evaluated is passed in, including the discarded ones.

Each of these is validated by simulation in `tests/test_inference.py` — empirical size under the null and power under the alternative — because an unvalidated test statistic is decoration.

## Recovery

On synthetic data the true replicating portfolio is known, so the estimator can be scored on consistency rather than fit. Two errors are reported, and the gap between them is the finding:

- `l1` — distance from the true weights. **Identification.**
- `excess_te_oos` — annualised tracking error above the true portfolio, on the next equally sized held-out block. **The cost of estimation error.**

Weight error decays at roughly `T^-0.41` against a parametric rate of `T^-0.5`, while the out-of-sample excess tracking error collapses to a fraction of a basis point far sooner. That gap is the ill-conditioning of the problem made visible: collinear candidates leave the objective nearly flat in the directions that separate them, so a large weight error buys almost no extra tracking error. It is why weight stability and turnover get their own columns in the horse race.

## Governance

A strategy is allowed to run only while its realised benefit covers its realised incremental cost by the hurdle multiple (default 20×), both measured on a trailing window (default 252 days). Benefit is the trailing annualised tracking error of the comparison strategy minus the strategy's own, in basis points; cost is the trailing incremental cost drag, annualised. The ratio is defined only where the strategy is dearer than the comparison by at least a floor — cheaper-and-better is the easiest possible pass, not a breach.

A breach is either a negative benefit at any cost, or a defined ratio below the hurdle. After `grace` consecutive breaching days (default 63) the switch trips, the book trades into the comparison strategy's drifted holdings — paying for that trade — and stays there until the strategy's *shadow* performance clears the hurdle for another `grace` days. The switch state on day `t` is decided from rows strictly before `t`, and `tests/test_governance.py` verifies that truncating the ledgers does not change any earlier governed row.

The rule is evaluated against the benchmark, against the simplest strategy that is statistically indistinguishable from the winner, and against `static`. The last is the cleanest test of whether the rebalancing itself earns its keep: it is the same estimator with the trading turned off.

## Constraint activity

At every rebalance, each constraint is tested for whether it binds — geometrically from the weights, so it is defined for the projected heuristics too — and, where a solve happened, the solver's dual is read as the shadow price. The report summarises the share of rebalances in which each constraint bound and the mean share of the turnover budget used. Shadow prices are reported only for tracking-family strategies, where the objective is a tracking error and the dual has an economic meaning.

The turnover convention is stated once and enforced: the cap bounds `Σ|Δw|`, which is two-way turnover, so a 0.20 cap permits 10% one-way turnover per rebalance.

## Metrics

Everything is computed from the walk-forward ledger only; there is no in-sample metric anywhere in `etflab/metrics.py`. Definitions are exported with every run in `metric_definitions.json`, so no reader has to guess whether a figure was annualised or which sign convention a capture ratio uses.

Active risk attribution is a full Euler decomposition treating the target as a position with weight −1. The contributions then sum exactly to the tracking error, which is what makes the table checkable; omitting the short leg — the usual shortcut — produces contributions that sum to nothing in particular.
