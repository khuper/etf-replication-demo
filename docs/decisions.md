# Design decisions

The choices worth arguing about, and what was rejected.

## Generated data is the default

**Decision.** `--data-source synthetic` ships as the default; yfinance and CSV are one flag away.

**Why.** Two things a downloaded panel cannot provide. The pipeline must run in CI with no network and no credentials, so the reader can regenerate every published number instead of trusting a screenshot. And the true replicating portfolio must be known, so the estimator can be scored on recovery rather than on fit.

**Rejected.** Committing a cached vendor panel — a licensing problem and a maintenance one, and it still would not give ground truth. Shipping no default at all — a repo whose first command fails without an API key is a repo nobody runs.

**Cost.** The headline numbers are about a simulated market, and the README says so in the second paragraph rather than in a footnote.

## `run_id` excludes the code version

**Decision.** `run_id = hash(semantic config ‖ data fingerprint)`. The git SHA is recorded in the manifest but not in the identity.

**Why.** If the SHA were in the identity, every commit would rename every run and "did this change move the numbers?" would be unanswerable. Keeping them separate makes it a one-line check: same `run_id`, different `results_digest` means the code changed the answer.

**Rejected.** Timestamped run directories. They are how one result quietly becomes six copies with no way to tell which is current. Re-running an experiment overwrites its directory on purpose.

## Data quality gates block by default

**Decision.** Nine gates run before any weight is estimated. A failing gate aborts unless `--allow-degraded`, which stamps the degradation into the manifest.

**Why.** A backtest cannot tell you its input was wrong. A ticker that goes stale for a month produces an *excellent* tracking error, because a flat series is very easy to track.

**Rejected.** Warn-only gates. A warning in a log nobody reads is not a control.

## Alignment reports what it cost

**Decision.** `align_prices` attributes leading truncation to the ticker that caused it and counts internal holes separately.

**Why.** The naive `df.dropna()` is wrong in a way that is easy to miss: one late-listing ticker silently discards years of history for everything else, and the run reports a shorter sample without saying why. Now the truncation is attributed and surfaced, so the reader can decide to drop the ticker instead.

## Baselines get the same constraints

**Decision.** Heuristic strategies are Euclidean-projected onto the identical feasible set the optimiser lives in.

**Why.** A comparison in which the fancy model is constrained and the baseline is not — or vice versa — is not evidence. If the optimiser wins, it should win on the weights.

**Cost.** Every baseline runs a small QP, so the "cheap" strategies are not actually cheap. Correctness beat speed.

## The verdict is generated, not written

**Decision.** The conclusion paragraph in the report is assembled from the computed statistics.

**Why.** A hand-written summary drifts from the numbers it claims to summarise the first time the model changes. A generated one cannot flatter the result, and it stays true after a refactor.

**Cost.** The prose is more mechanical than a person would write. That is the right trade for a document whose job is to be accurate.

## Costs are charged on traded notional

**Decision.** `cost = Σ|Δw| × bps`, not `(Σ|Δw|/2) × bps`.

**Why.** This was a live bug, found by a test. Each leg of a rebalance crosses a spread, and the initial deployment out of cash is one-sided — it buys 100% of the book while the conventional turnover figure reads 0.5. Charging on the halved figure understated every cost by half and the initial deployment by half again. Both cost models are now pinned against each other on an identical trade.

## PBO is compared against a matched null, not against 0.5

**Decision.** `noise_reference_pbo` generates matched noise matrices and averages their PBO.

**Why.** CSCV on a fixed sample does not have a clean one-half null: the two halves partition one finite history, which induces negative dependence and pushes the null above 0.5. And a single noise draw is itself high variance — across seeds the PBO of pure noise ranged from 0.23 to 0.79 — so the reference is averaged over replications. Comparing an observed PBO against a one-draw reference would be comparing it against a coin flip.

## A CSV cache, not parquet

**Decision.** Cached panels are CSV with a JSON sidecar.

**Why.** The cache exists to be opened by a person trying to work out why a number moved, and to avoid adding a binary dependency to a repo whose whole point is that it runs anywhere.

**Cost.** Slower and larger. Irrelevant at this scale.

## The interactive shell was kept

**Decision.** `etf-lab shell` survives from the original project.

**Why.** Exploratory work is genuinely iterative and the shell is good at it. It now shares the same config object and the same quality gates as every other entry point, so there is no path through it that produces an unreproducible number.

## The kill switch is a strategy, not a footnote

**Decision.** The governance rule is applied walk-forward to produce a governed ledger with its own tracking error and switching costs, rather than reported as a post-hoc statistic.

**Why.** "Would a committee have shut this down?" is only answerable if the shutdown is simulated with the information available at the time, including what it cost to switch and what the reverted book then did. A rule that is scored with hindsight is a different, weaker rule.

**Rejected.** Defining the payoff ratio as benefit over *total* cost. Benefit is measured against a comparison strategy, so cost must be too; a strategy that trades less than its comparison has a negative incremental cost, and a naive ratio would have reported cheaper-and-better as a catastrophic breach. That bug existed for about ten minutes.

## Constraints are audited, not assumed

**Decision.** Every constraint's binding frequency and dual value is recorded per rebalance and summarised in the report.

**Why.** The original project advertised a CVaR constraint and a turnover cap. Neither had ever been checked for doing anything. On the shipped data the position cap does all the work, the turnover cap never binds, and CVaR binds once in forty-six rebalances. Saying so is more useful than the diagram.

## What was deliberately not built

- **A web dashboard.** It would demonstrate front-end work, not research judgement, and the HTML memo already travels as a single file.
- **A database-backed run store.** The requirement is "find Tuesday's run and tell me what changed", which one JSONL file answers completely and `grep` can read when this code is gone.
- **More strategies.** Nine already saturate the multiple-comparison correction; a tenth would add trials to deflate against and nothing to the conclusion.
- **A deep-learning replicator.** With ten collinear candidates and an objective that is provably flat in the directions that separate them, the binding constraint is identification, not model capacity. It would be padding, and a reviewer would read it as such.
