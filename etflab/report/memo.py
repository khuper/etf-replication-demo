"""The research memo as structured content, rendered by markdown.py and html.py.

Writing the memo once as data rather than twice as strings means the Markdown and
the HTML cannot drift apart, and it means the narrative is assembled from the
computed results rather than typed alongside them. Nothing in here invents a
number: every sentence that states a value reads it from the study.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Paragraph:
    text: str


@dataclass(frozen=True)
class Callout:
    """A boxed statement. Kind drives the styling: finding, caution, method."""

    kind: str
    title: str
    text: str


@dataclass(frozen=True)
class Table:
    title: str
    frame: pd.DataFrame
    note: str = ""
    formats: dict[str, str] = field(default_factory=dict)
    index_label: str = ""


@dataclass(frozen=True)
class Figure:
    name: str
    caption: str


@dataclass(frozen=True)
class KeyValues:
    title: str
    items: list[tuple[str, str]]


@dataclass(frozen=True)
class CodeBlock:
    language: str
    text: str


Block = Paragraph | Callout | Table | Figure | KeyValues | CodeBlock


@dataclass(frozen=True)
class Section:
    title: str
    blocks: list[Block]
    anchor: str = ""


@dataclass(frozen=True)
class Memo:
    title: str
    subtitle: str
    sections: list[Section]
    meta: dict[str, str]


PCT = "{:.2%}"
PCT1 = "{:.1%}"
BPS = "{:+.1f}bp"
NUM3 = "{:.3f}"


def _fmt(value: Any, spec: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    if spec is None:
        if isinstance(value, float):
            return f"{value:,.4g}"
        return str(value)
    if spec.endswith("bp") and isinstance(value, (int, float)):
        return spec.format(value * 1e4)
    return spec.format(value)


def format_frame(table: Table) -> pd.DataFrame:
    """Apply per-column format strings, returning a frame of strings."""
    out = table.frame.copy()
    for column in out.columns:
        spec = table.formats.get(str(column))
        out[column] = [_fmt(v, spec) for v in out[column]]
    return out


def _reproduce_command(study: Any) -> str:
    config = study.config
    parts = [
        "etf-lab study",
        f"--target {config.target}",
        f"--assets {' '.join(config.assets)}",
        f"--start {config.start} --end {config.end}",
        f"--data-source {config.data_source}",
    ]
    if config.data_source == "synthetic":
        parts.append(f"--seed {config.synthetic_seed}")
    parts += [
        f"--train-days {config.train_days}",
        f"--rebalance-days {config.rebalance_days}",
        f"--max-weight {config.max_weight}",
        f"--max-turnover {config.max_turnover}",
        f"--cost-model {config.cost_model}",
    ]
    return " \\\n  ".join(parts)


def build_memo(study: Any, manifest: Any) -> Memo:
    """Assemble the memo from a completed :class:`~etflab.research.Study`."""
    race = study.race
    config = study.config
    metrics = race.metrics
    sections: list[Section] = []

    # ------------------------------------------------------------------ #
    headline_items = [
        ("Target", config.target),
        ("Candidates", f"{len(config.assets)} ETFs: {', '.join(config.assets)}"),
        ("Sample", f"{study.panel.span[0].date()} to {study.panel.span[1].date()} ({study.panel.n_obs:,} days)"),
        ("Evaluation", f"{len(race.results[race.best].daily):,} out-of-sample days, walk-forward"),
        ("Data source", f"{study.panel.source} (fingerprint {study.panel.fingerprint()})"),
        ("Run id", manifest.run_id),
        ("Code", manifest.code.get("git_sha", "n/a")[:12] + (" (dirty)" if manifest.code.get("dirty") else "")),
    ]
    sections.append(
        Section(
            "Summary",
            [
                Callout("finding", "Verdict", race.verdict),
                KeyValues("Experiment", headline_items),
                Paragraph(
                    "Every number in this memo was produced by the command below. It needs no API key, no vendor "
                    "credentials and no network connection, and it is the same command CI runs on every commit."
                ),
                CodeBlock("bash", _reproduce_command(study)),
            ],
            "summary",
        )
    )

    # ------------------------------------------------------------------ #
    quality_frame = pd.DataFrame(
        [{"gate": g.name, "status": g.status, "detail": g.summary} for g in study.quality.gates]
    ).set_index("gate")
    data_blocks: list[Block] = [
        Paragraph(
            f"Data quality gates run before any weight is estimated. Overall status: "
            f"**{study.quality.status}**. A failing gate aborts the run unless `--allow-degraded` is passed, "
            f"in which case the manifest records that the result is degraded."
        ),
        Table("Data quality gates", quality_frame, index_label="gate"),
    ]
    if study.panel.source == "synthetic":
        truth = study.panel.truth
        data_blocks.append(
            Callout(
                "method",
                "Why the default data is generated",
                "The shipped experiment runs on a simulated market rather than downloaded prices, for two reasons "
                "that both serve the reader. First, it reproduces exactly: same seed, same numbers, on any machine, "
                "forever, with no vendor licence to redistribute. Second -- and this is the part real data cannot "
                f"offer -- the market's true replicating portfolio is known, so an estimator can be scored on whether "
                f"it recovers the right answer rather than only on whether it fits. The irreducible tracking error "
                f"here is {truth.irreducible_te_annual:.2%} annualised: no long-only basket of these candidates can "
                "do better, which gives every tracking-error number on this page an absolute scale. "
                "`--data-source yfinance` runs the identical pipeline on real prices.",
            )
        )
    sections.append(Section("Data and quality control", data_blocks, "data"))

    # ------------------------------------------------------------------ #
    method_text = (
        f"Expanding-window walk-forward. At each rebalance the model is fitted on returns strictly before the "
        f"decision date, with a {config.embargo_days}-day embargo standing in for data arrival and execution lag. "
        f"Weights are traded at that day's close, costs are charged on the distance from the *drifted* holdings, "
        f"and the book is then held for {config.rebalance_days} trading days while weights drift with prices. "
        f"Every strategy sees the same windows, the same constraints "
        f"({config.max_weight:.0%} position cap, {config.max_turnover:.0%} L1 turnover cap, long only, fully "
        f"invested) and the same cost model, so any difference between them is attributable to the weights alone."
    )
    sections.append(
        Section(
            "Method",
            [
                Paragraph(method_text),
                Callout(
                    "method",
                    "Look-ahead is tested, not asserted",
                    "The claim that no future information reaches a decision is verified mechanically: the test "
                    "suite truncates the price panel at a series of dates and asserts that every ledger row before "
                    "the cut is bit-identical to the untruncated run. If any future observation influenced any past "
                    "decision, that test fails.",
                ),
            ],
            "method",
        )
    )

    # ------------------------------------------------------------------ #
    if study.recovery is not None:
        recovery_columns = [c for c in study.recovery.frame.columns if c.endswith(("_l1", "_excess_te_oos"))]
        sections.append(
            Section(
                "Does the estimator work at all?",
                [
                    Paragraph(
                        "Before asking whether the optimiser beats a benchmark, ask whether it converges to the "
                        "right answer when there is one. On synthetic data there is."
                    ),
                    Callout("finding", "Consistency", study.recovery.interpretation),
                    Figure("recovery", "Weight recovery and the out-of-sample cost of estimation error."),
                    Table(
                        "Recovery by training sample size",
                        study.recovery.frame[recovery_columns],
                        note="`_l1` is L1 distance to the true weights; `_excess_te_oos` is annualised tracking "
                        "error above the true portfolio on the next equally sized, entirely held-out block.",
                        formats={c: NUM3 if c.endswith("_l1") else BPS for c in recovery_columns},
                        index_label="training days",
                    ),
                ],
                "recovery",
            )
        )

    # ------------------------------------------------------------------ #
    race_columns = [
        c
        for c in (
            "tracking_error",
            "correlation",
            "down_capture",
            "annual_turnover",
            "cost_drag_annual",
            "effective_positions",
            "worst_21d_active",
        )
        if c in metrics.columns
    ]
    comparison_frame = pd.DataFrame(
        [
            {
                "strategy": name,
                "tracking_error": c.tracking_error,
                "vs benchmark": c.te_difference,
                "95% CI low": c.ci.lower,
                "95% CI high": c.ci.upper,
                "DM p-value": c.dm.p_value,
                "significant": "yes" if c.significantly_better else "no",
            }
            for name, c in sorted(race.comparisons.items(), key=lambda kv: kv[1].tracking_error)
        ]
    ).set_index("strategy")

    race_blocks: list[Block] = [
        Paragraph(
            f"Nine strategies, identical terms. `{race.benchmark}` is the benchmark: the thing a reasonable person "
            "would do without an optimiser."
        ),
        Figure("horse_race", "Out-of-sample tracking error by strategy, against the irreducible floor."),
        Table(
            "Out-of-sample performance",
            metrics[race_columns].sort_values("tracking_error"),
            formats={
                "tracking_error": PCT,
                "correlation": NUM3,
                "down_capture": NUM3,
                "annual_turnover": PCT1,
                "cost_drag_annual": BPS,
                "effective_positions": "{:.1f}",
                "worst_21d_active": PCT,
            },
            index_label="strategy",
        ),
        Paragraph(
            "The differences above are estimates. The table below tests them: a stationary-bootstrap confidence "
            "interval for the tracking-error difference against the benchmark, and a Diebold-Mariano test of equal "
            "expected squared tracking error with Newey-West standard errors. Both resample the two strategies "
            "jointly, which is what preserves the dependence between them."
        ),
        Table(
            f"Statistical comparison against {race.benchmark}",
            comparison_frame,
            formats={
                "tracking_error": PCT,
                "vs benchmark": PCT,
                "95% CI low": PCT,
                "95% CI high": PCT,
                "DM p-value": "{:.2e}",
            },
            index_label="strategy",
        ),
        Callout(
            "method",
            "Multiple comparisons",
            f"Running nine strategies and reporting the best inflates the apparent result. Hansen's SPA test "
            f"(p = {race.spa.spa_p:.3g}) and White's Reality Check (p = {race.spa.reality_check_p:.3g}) test the "
            f"null that *no* strategy beats {race.benchmark}, correcting for the fact that nine were tried.",
        ),
        Figure("rolling_tracking_error", "Rolling tracking error: the variation the headline number averages away."),
    ]
    if race.notes:
        race_blocks.append(
            Callout("caution", "What the tables do not say out loud", "\n".join(f"- {n}" for n in race.notes))
        )
    sections.append(Section("The horse race", race_blocks, "horse-race"))

    # ------------------------------------------------------------------ #
    governance = getattr(study, "governance", {}) or {}
    if governance:
        policy = next(iter(governance.values())).policy
        gov_blocks: list[Block] = [
            Paragraph(
                f"A model is allowed to run only while its realised, out-of-sample benefit covers its realised "
                f"incremental cost at least **{policy.hurdle:g} times over**, measured on a trailing "
                f"{policy.window}-day window. After {policy.grace} consecutive breaching days the book reverts to "
                f"the comparison strategy"
                + (" and may re-earn its place." if policy.reactivate else " and stays there.")
                + " The switch is applied walk-forward, so the governed book is a strategy in its own right, "
                "with its own tracking error and its own switching costs."
            ),
        ]
        labels = {
            "benchmark": "against the benchmark",
            "simplest": "against the simplest indistinguishable strategy",
            "static": "against optimise-once-and-hold",
        }
        for key, report in governance.items():
            kind = "finding" if report.ever_shut_off else "method"
            title = f"{report.strategy} {labels.get(key, key)} ({report.benchmark})"
            if report.ever_shut_off:
                title += f" -- shut off for {report.days_off / max(len(report.daily), 1):.0%} of the sample"
            gov_blocks.append(Callout(kind, title, report.verdict))
        if "static" in governance:
            gov_blocks.append(
                Figure("governance", "Rolling benefit versus the hurdle; shaded spans are where the switch was off.")
            )
            episodes = governance["static"].episodes
            if not episodes.empty:
                gov_blocks.append(
                    Table(
                        "Shut-off episodes (rebalancing versus optimise-once-and-hold)",
                        episodes,
                        formats={"days": "{:,.0f}"},
                        index_label="start",
                    )
                )
        gov_blocks.append(
            Paragraph(
                "Read the three together. The optimiser pays for itself against a naive basket trivially -- it "
                "barely trades more. The question with teeth is whether the *ongoing rebalancing* pays for itself "
                "against solving the problem once and holding, and the rule's answer to that is the one a "
                "governance committee would actually act on."
            )
        )
        sections.append(Section("Does it pay for itself?", gov_blocks, "governance"))

    # ------------------------------------------------------------------ #
    if study.sweep is not None:
        sweep = study.sweep
        sections.append(
            Section(
                "How much of this is selection bias?",
                [
                    Paragraph(
                        f"The horse race fixed one configuration. In practice a researcher tries many, and reports "
                        f"the best. This section measures what that costs. {len(sweep.frame)} configurations were "
                        f"evaluated over {' x '.join(f'{k} in {v}' for k, v in sweep.grid.items())}."
                    ),
                    Callout(
                        "finding" if sweep.selection_is_meaningful else "caution",
                        f"Probability of backtest overfitting: {sweep.pbo.pbo:.0%}",
                        f"Combinatorially symmetric cross-validation over {sweep.pbo.n_combinations:,} balanced "
                        f"splits. {sweep.pbo.verdict}. The pure-noise reference for a grid of this shape is "
                        f"{sweep.noise_pbo:.0%} -- CSCV on a fixed sample has a null above one half, because the "
                        f"in-sample and out-of-sample halves partition one finite history, so a configuration that "
                        f"ran hot in one has less luck left for the other. Compared against that reference, the "
                        f"observed PBO is "
                        f"{'materially lower, so in-sample selection here carries real information' if sweep.selection_is_meaningful else 'not materially lower, so in-sample ranking should not be trusted to survive'}.",
                    ),
                    Figure("overfitting", "Distribution of the CSCV rank logit; mass below zero is the PBO."),
                    Callout(
                        "method",
                        f"Deflated Sharpe ratio: {sweep.deflated.deflated_probability:.1%}",
                        f"The selected configuration's information ratio is "
                        f"{sweep.deflated.observed_sharpe * (252**0.5):.2f} annualised. Deflating for "
                        f"{sweep.deflated.n_trials} trials, non-normal returns and the sample length gives a "
                        f"{sweep.deflated.deflated_probability:.1%} probability that the true ratio exceeds the "
                        f"selection hurdle. "
                        f"{'It clears the 95% bar.' if sweep.deflated.survives else 'It does not clear the 95% bar, which is the expected outcome for a replication strategy: tracking tightly is not the same as earning a return.'}",
                    ),
                    Table(
                        "Best and worst configurations",
                        pd.concat([sweep.frame.head(5), sweep.frame.tail(3)])[["tracking_error", "annual_turnover"]],
                        formats={"tracking_error": PCT, "annual_turnover": PCT1},
                        index_label="configuration",
                    ),
                ],
                "overfitting",
            )
        )

    # ------------------------------------------------------------------ #
    risk_blocks: list[Block] = [
        Paragraph(
            "An average tracking error is a promise about a market that never happens. These are the conditional "
            "numbers: what the basket did when it mattered."
        ),
    ]
    if not study.regimes.empty:
        risk_blocks += [
            Figure("regimes", "Tracking error conditioned on market regime."),
            Table(
                "Performance by regime",
                study.regimes[["days", "share", "tracking_error", "active_return", "target_return", "hit_rate"]],
                formats={
                    "days": "{:,.0f}",
                    "share": PCT1,
                    "tracking_error": PCT,
                    "active_return": PCT,
                    "target_return": PCT,
                    "hit_rate": PCT1,
                },
                index_label="regime",
            ),
        ]
    risk_blocks += [
        Table(
            "Conditional tail performance",
            study.tails[["n_days", "target_mean", "replicator_mean", "active_mean", "capture"]],
            note="Mean daily returns conditioned on the target's worst days. Capture below 1.0 means the "
            "replicator fell less than the target -- which is a tracking failure and a risk benefit at the "
            "same time, and worth being explicit about.",
            formats={
                "n_days": "{:,.0f}",
                "target_mean": PCT,
                "replicator_mean": PCT,
                "active_mean": PCT,
                "capture": NUM3,
            },
            index_label="tail quantile",
        ),
        Table(
            "Worst 21-day windows for the target",
            study.worst_windows,
            formats={
                "target_return": PCT,
                "replicator_return": PCT,
                "active_return": PCT,
                "capture": NUM3,
            },
            index_label="window end",
        ),
        Table(
            "Active risk attribution",
            study.attribution[["avg_weight", "risk_contribution", "risk_share", "corr_with_target"]],
            note="Euler decomposition of annualised active risk, including the target as a short leg. "
            "Contributions sum exactly to the tracking error; the test suite asserts it.",
            formats={
                "avg_weight": PCT1,
                "risk_contribution": PCT,
                "risk_share": PCT1,
                "corr_with_target": NUM3,
            },
            index_label="leg",
        ),
        Table(
            "Rolling correlation stability",
            study.correlation_stability,
            note="126-day rolling correlation of each candidate to the target. A wide range means the proxy was "
            "right on average rather than reliably right.",
            formats={c: NUM3 for c in study.correlation_stability.columns},
            index_label="asset",
        ),
        Figure("weights", "Holdings through time, after drift."),
    ]
    sections.append(Section("Risk, regimes and attribution", risk_blocks, "risk"))

    # ------------------------------------------------------------------ #
    if study.cost_breakeven is not None:
        breakeven = study.cost_breakeven
        capacity = study.capacity
        te_gain = float(study.cost_split["te_advantage_bp"].mean()) if study.cost_split is not None else float("nan")
        if breakeven.breakeven is None:
            cost_text = (
                f"Against {race.benchmark}, the strategy buys {te_gain:,.0f}bp of annualised tracking-error "
                f"reduction. Its extra turnover never costs that much: even at "
                f"{max(breakeven.values):,.0f}bp one way -- an implausible level for large-cap ETFs -- the extra "
                f"trading is worth less than the risk it removes. Costs are not what decides this comparison, so "
                f"refining the cost model further would answer a question that is already settled. Note what this "
                f"does *not* say: costs still reduce returns, they just do not change the ranking."
            )
        else:
            cost_text = (
                f"Against {race.benchmark}, the strategy buys {te_gain:,.0f}bp of annualised tracking-error "
                f"reduction and pays for it in turnover. The two cross at roughly "
                f"{breakeven.breakeven:,.0f}bp of one-way transaction cost. Comparing a return drag against a risk "
                f"reduction is a rule of thumb rather than a utility calculation, and should be read as one."
            )
        cost_blocks: list[Block] = [
            Callout("finding", "What the tracking-error reduction costs", cost_text),
            Figure("capacity", "Cost of the tracking-error reduction, and capacity under square-root impact."),
        ]
        if study.cost_split is not None:
            cost_blocks.append(
                Table(
                    "Annual cost drag by cost assumption",
                    study.cost_split,
                    note="Basis points of portfolio value per year. The last column is the tracking-error "
                    "reduction being bought, which is almost invariant to the cost assumption because both "
                    "strategies trade little.",
                    formats={c: "{:,.2f}" for c in study.cost_split.columns},
                    index_label="assumed cost (bp)",
                )
            )
        if study.cost_curve is not None:
            breached = study.cost_curve[study.cost_curve["exceeds_participation_limit"]]
            if not breached.empty:
                cost_blocks.append(
                    Callout(
                        "caution",
                        "Capacity limit",
                        f"Above roughly ${breached.index[0] / 1e9:,.1f}bn the rebalance breaches the "
                        f"{config.max_participation:.0%} ADV participation limit in at least one asset. Past that "
                        f"point the strategy is not merely more expensive, it is not executable as modelled: the "
                        f"trade would have to be spread over multiple days, adding timing risk this backtest does "
                        f"not charge for.",
                    )
                )
            cost_blocks.append(
                Table(
                    "Cost and capacity by book size",
                    study.cost_curve[["cost_drag_bps_annual", "tracking_error", "max_participation"]],
                    formats={
                        "cost_drag_bps_annual": "{:.2f}",
                        "tracking_error": PCT,
                        "max_participation": PCT1,
                    },
                    index_label="book size (USD)",
                )
            )
        if capacity is not None:
            cost_blocks.append(
                Table(
                    "Tracking-error advantage by book size",
                    pd.DataFrame(
                        {"advantage": capacity.advantage},
                        index=pd.Index(capacity.values, name="book size (USD)"),
                    ),
                    formats={"advantage": BPS},
                    index_label="book size (USD)",
                )
            )
        sections.append(Section("Costs and capacity", cost_blocks, "costs"))

    # ------------------------------------------------------------------ #
    sections.append(
        Section(
            "Limitations",
            [
                Paragraph(
                    "The results above are conditional on assumptions that a reader should be able to attack "
                    "directly. These are the ones worth attacking."
                ),
                Callout(
                    "caution",
                    "What would break this",
                    "\n".join(f"- {item}" for item in _limitations(study)),
                ),
            ],
            "limitations",
        )
    )

    # ------------------------------------------------------------------ #
    sections.append(
        Section(
            "Provenance",
            [
                KeyValues(
                    "Reproducibility record",
                    [
                        ("Run id", manifest.run_id),
                        ("Config hash", manifest.config_hash),
                        ("Data fingerprint", manifest.data_fingerprint),
                        ("Results digest", manifest.results_digest[:16]),
                        ("Quality status", manifest.quality_status),
                        ("Degraded", "yes" if manifest.degraded else "no"),
                        ("Git", f"{manifest.code.get('git_sha', 'n/a')} ({manifest.code.get('git_branch', '?')})"),
                        ("Working tree", "dirty" if manifest.code.get("dirty") else "clean"),
                        ("Python", manifest.environment.get("python", "?")),
                        (
                            "Libraries",
                            ", ".join(
                                f"{k} {manifest.environment[k]}"
                                for k in ("numpy", "pandas", "cvxpy", "scipy")
                                if k in manifest.environment
                            ),
                        ),
                        ("Solvers", manifest.environment.get("cvxpy_solvers", "?")),
                    ],
                ),
                Paragraph(
                    "`etf-lab verify <run-directory>` re-runs this experiment and compares the results digest. "
                    "A mismatch means the code changed the answer -- which is either a fix or a regression, and "
                    "either way it should be visible rather than silent. CI runs that check on every commit."
                ),
            ],
            "provenance",
        )
    )

    return Memo(
        title=f"Replicating {config.target} with liquid ETFs",
        subtitle=race.verdict.split(".")[0] + ".",
        sections=sections,
        meta={
            "run_id": manifest.run_id,
            "generated": manifest.created_at,
            "data_source": study.panel.source,
            "fingerprint": study.panel.fingerprint(),
        },
    )


def _limitations(study: Any) -> Sequence[str]:
    """Limitations stated specifically, with the number that makes each concrete."""
    items = [
        "Transaction costs use a half-spread plus square-root impact model with an assumed liquidity table. "
        "The spreads and ADVs are illustrative order-of-magnitude figures, not measured; they are in "
        "`etflab/costs.py` so they can be replaced with real ones.",
        "The backtest charges no borrow, no financing, no creation/redemption friction, and no taxes. It is a "
        "long-only, fully invested book, so those omissions are small -- but they are omissions.",
        "Execution is modelled at the close on the rebalance date. Intraday slippage and the timing risk of "
        "spreading a large trade over several days are not charged for.",
    ]
    if study.panel.source == "synthetic":
        items.append(
            "The headline results are computed on a simulated market. Its statistical features -- fat tails, "
            "volatility clustering, regime-dependent correlation -- were chosen to be realistic, but they were "
            "chosen. Conclusions about *this estimator versus that one* transfer; conclusions about the real "
            f"{study.config.target} do not, and are not claimed."
        )
        items.append(
            "The candidate universe is fixed and hand-chosen. On real data this would carry a selection bias that "
            "the synthetic setting hides, because here the true portfolio is inside the candidate set by "
            "construction. On real prices it might not be."
        )
    else:
        items.append(
            "Prices come from a free retail data source with no point-in-time guarantee. A vendor restatement "
            "would change these numbers, which is exactly why the data fingerprint is recorded."
        )
        items.append(
            "The candidate universe is chosen with hindsight about which ETFs still exist and still trade. That is "
            "survivorship bias, and it flatters every number here."
        )
    if study.sweep is not None and not study.sweep.selection_is_meaningful:
        items.append(
            f"The parameter sweep's PBO of {study.sweep.pbo.pbo:.0%} is not meaningfully below its noise reference "
            f"of {study.sweep.noise_pbo:.0%}. Treat the ranking of configurations as unreliable, and prefer the "
            "configuration chosen on prior grounds over the one that won."
        )
    return items
