"""The kill switch: a model must pay for itself, or it is turned off.

Every desk has a version of this rule and almost no backtest implements it. A
strategy is allowed to run only while its realised, out-of-sample benefit covers
its realised cost by a stated multiple -- the *hurdle*. When it stops clearing
the hurdle for long enough to rule out noise, the book reverts to the benchmark,
and stays there until the strategy has re-earned its place.

Two things make this a research object rather than a slogan:

1. **It is evaluated walk-forward.** The decision on day ``t`` uses ledger rows
   strictly before ``t``. The governed ledger is therefore a legitimate strategy
   in its own right, with its own tracking error, its own costs -- including the
   cost of switching -- and its own place in the comparison.
2. **It answers a question the horse race cannot.** The horse race says whether
   the optimiser beat the benchmark *on average over a decade*. The kill switch
   says whether there was ever a stretch during which a reasonable governance
   committee would have shut it down, how long that stretch lasted, and what the
   shutdown would have cost or saved.

The hurdle is a multiple of *incremental* cost: benefit is measured against the
benchmark, so cost is too. A strategy that trades no more than the benchmark
has near-zero incremental cost and clears any finite hurdle -- which is the
correct answer, and the report says so rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from etflab.backtest import BacktestResult
from etflab.costs import CostModel

TRADING_DAYS = 252
BPS = 1e-4


@dataclass(frozen=True)
class HurdlePolicy:
    """The rule, stated once.

    hurdle:
        Required ratio of realised benefit to realised incremental cost.
    window:
        Trailing evaluation window in trading days. Shorter reacts faster and
        flaps more; a year is the conventional compromise.
    grace:
        Consecutive breaching days required before the switch trips. This is
        what stops a single bad month from turning a decade-long edge off.
    reactivate:
        Whether a shut-off strategy may be turned back on once its *shadow*
        performance -- what it would have done -- clears the hurdle again for
        ``grace`` days.
    cost_floor_bps:
        Incremental cost below which the ratio is treated as undefined rather
        than infinite. A ratio of "benefit divided by nothing" is not evidence.
    """

    hurdle: float = 20.0
    window: int = 252
    grace: int = 63
    reactivate: bool = True
    cost_floor_bps: float = 0.05

    def validate(self) -> HurdlePolicy:
        if self.hurdle <= 0:
            raise ValueError("hurdle must be positive.")
        if self.window < 21:
            raise ValueError("window must be at least a month of trading days.")
        if self.grace < 1:
            raise ValueError("grace must be at least one day.")
        if self.cost_floor_bps < 0:
            raise ValueError("cost_floor_bps cannot be negative.")
        return self


@dataclass(frozen=True)
class GovernanceReport:
    """What the rule did, day by day, and in summary."""

    policy: HurdlePolicy
    strategy: str
    benchmark: str
    daily: pd.DataFrame  #: date x [benefit_bp, cost_bp, ratio, breaching, active]
    governed: pd.DataFrame  #: the governed ledger: date x [gross, cost, net, target, active]
    episodes: pd.DataFrame  #: one row per shut-off episode
    switches: int
    switching_cost: float
    days_off: int
    min_ratio: float
    median_ratio: float
    final_state: str
    undefined_share: float
    verdict: str

    @property
    def ever_shut_off(self) -> bool:
        return self.days_off > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": {
                "hurdle": self.policy.hurdle,
                "window": self.policy.window,
                "grace": self.policy.grace,
                "reactivate": self.policy.reactivate,
                "cost_floor_bps": self.policy.cost_floor_bps,
            },
            "strategy": self.strategy,
            "benchmark": self.benchmark,
            "switches": self.switches,
            "switching_cost": self.switching_cost,
            "days_off": self.days_off,
            "share_off": self.days_off / max(len(self.daily), 1),
            "min_ratio": self.min_ratio,
            "median_ratio": self.median_ratio,
            "final_state": self.final_state,
            "undefined_share": self.undefined_share,
            "episodes": self.episodes.reset_index().to_dict("records") if not self.episodes.empty else [],
            "verdict": self.verdict,
        }


def payoff_ratio(
    strategy: BacktestResult,
    benchmark: BacktestResult,
    policy: HurdlePolicy,
) -> pd.DataFrame:
    """Trailing benefit, incremental cost and their ratio, aligned on shared dates.

    ``benefit_bp``: trailing annualised tracking error of the benchmark minus the
    strategy's, in basis points. Positive means the strategy tracked tighter.

    ``cost_bp``: trailing incremental cost drag, annualised, in basis points.
    Positive means the strategy traded more expensively than the benchmark.

    ``ratio``: benefit over cost, or NaN where the incremental cost is below the
    policy floor -- there the strategy is essentially free relative to the
    benchmark, and the hurdle is trivially met, but the ratio itself is not a
    meaningful number and is not reported as one.
    """
    shared = strategy.daily.index.intersection(benchmark.daily.index)
    s, b = strategy.daily.loc[shared], benchmark.daily.loc[shared]
    window = policy.window
    scale = np.sqrt(TRADING_DAYS)

    te_strategy = s["active"].rolling(window).std(ddof=1) * scale
    te_benchmark = b["active"].rolling(window).std(ddof=1) * scale
    benefit_bp = (te_benchmark - te_strategy) / BPS

    cost_strategy = s["cost"].rolling(window).sum() * (TRADING_DAYS / window)
    cost_benchmark = b["cost"].rolling(window).sum() * (TRADING_DAYS / window)
    cost_bp = (cost_strategy - cost_benchmark) / BPS

    floor = policy.cost_floor_bps
    # The ratio is only a meaningful number when the strategy costs *more* than
    # the benchmark by at least the floor. Cheaper-and-better is not a breach, it
    # is the easiest possible pass, and dividing by a negative cost would have
    # reported it as a catastrophic failure.
    priced = cost_bp >= floor
    ratio = pd.Series(np.nan, index=shared, dtype=float)
    ratio[priced] = benefit_bp[priced] / cost_bp[priced]

    evaluable = te_strategy.notna() & te_benchmark.notna()
    # Breach: tracking worse than the benchmark at any price, or tracking better
    # but not by enough to cover the incremental cost ``hurdle`` times over.
    worse = benefit_bp < 0
    under_hurdle = priced & (ratio < policy.hurdle)
    breaching = ((worse | under_hurdle) & evaluable).fillna(False)

    return pd.DataFrame(
        {
            "benefit_bp": benefit_bp,
            "cost_bp": cost_bp,
            "ratio": ratio,
            "breaching": breaching.astype(bool),
            "evaluable": evaluable,
            "free": (evaluable & ~priced & ~worse),
        }
    )


def _switch_cost(from_weights: pd.Series, to_weights: pd.Series, cost_model: CostModel, volatility: pd.Series) -> float:
    trades = (to_weights - from_weights.reindex(to_weights.index).fillna(0.0)).astype(float)
    return float(cost_model.cost(trades, volatility).total)


def govern(
    strategy: BacktestResult,
    benchmark: BacktestResult,
    policy: HurdlePolicy,
    cost_model: CostModel,
    asset_returns: pd.DataFrame,
) -> GovernanceReport:
    """Apply the kill switch walk-forward and return the governed ledger.

    On each day the switch state is decided from the payoff ratio computed on
    rows strictly before that day. When the state changes, the book trades
    from one strategy's drifted holdings into the other's, and the cost of that
    trade is charged to the governed ledger on the day of the switch.
    """
    policy = policy.validate()
    signal = payoff_ratio(strategy, benchmark, policy)
    dates = signal.index
    s, b = strategy.daily.loc[dates], benchmark.daily.loc[dates]
    ws, wb = strategy.daily_weights.reindex(dates), benchmark.daily_weights.reindex(dates)

    active = True  # the strategy starts live; it has to lose its place
    streak_breach = 0
    streak_clear = 0
    switches = 0
    switching_cost_total = 0.0
    rows: list[dict[str, Any]] = []
    state: list[bool] = []
    episode_start: pd.Timestamp | None = None
    episodes: list[dict[str, Any]] = []

    for position, day in enumerate(dates):
        # Decide from yesterday's signal only. Today's row is not yet known.
        if position > 0:
            yesterday = signal.iloc[position - 1]
            if yesterday["evaluable"]:
                if yesterday["breaching"]:
                    streak_breach += 1
                    streak_clear = 0
                else:
                    streak_clear += 1
                    streak_breach = 0

        switch_cost = 0.0
        if active and streak_breach >= policy.grace:
            active = False
            switches += 1
            streak_breach = 0
            episode_start = day
            trailing_vol = asset_returns.loc[:day].tail(63).std(ddof=1)
            switch_cost = _switch_cost(ws.iloc[position], wb.iloc[position], cost_model, trailing_vol)
        elif (not active) and policy.reactivate and streak_clear >= policy.grace:
            active = True
            switches += 1
            streak_clear = 0
            if episode_start is not None:
                episodes.append(
                    {
                        "start": episode_start,
                        "end": day,
                        "days": int(dates.get_loc(day) - dates.get_loc(episode_start)),
                    }
                )
                episode_start = None
            trailing_vol = asset_returns.loc[:day].tail(63).std(ddof=1)
            switch_cost = _switch_cost(wb.iloc[position], ws.iloc[position], cost_model, trailing_vol)
        switching_cost_total += switch_cost

        source = s if active else b
        gross = float(source.loc[day, "gross"])
        cost = float(source.loc[day, "cost"]) + switch_cost
        net = gross - cost
        target = float(source.loc[day, "target"])
        rows.append({"date": day, "gross": gross, "cost": cost, "net": net, "target": target, "active": net - target})
        state.append(active)

    if episode_start is not None:
        episodes.append(
            {"start": episode_start, "end": dates[-1], "days": int(len(dates) - dates.get_loc(episode_start))}
        )

    governed = pd.DataFrame(rows).set_index("date")
    daily = signal.copy()
    daily["active"] = state
    days_off = int((~daily["active"]).sum())
    defined = daily["ratio"].dropna()
    undefined_share = float(daily.loc[daily["evaluable"], "free"].mean()) if daily["evaluable"].any() else 0.0

    episodes_frame = pd.DataFrame(episodes).set_index("start") if episodes else pd.DataFrame(columns=["end", "days"])
    final_state = "live" if active else "shut off"
    min_ratio = float(defined.min()) if len(defined) else float("nan")
    median_ratio = float(defined.median()) if len(defined) else float("nan")

    verdict = _verdict(
        strategy.strategy,
        benchmark.strategy,
        policy,
        days_off,
        len(daily),
        switches,
        min_ratio,
        median_ratio,
        undefined_share,
        final_state,
        episodes_frame,
    )
    return GovernanceReport(
        policy=policy,
        strategy=strategy.strategy,
        benchmark=benchmark.strategy,
        daily=daily,
        governed=governed,
        episodes=episodes_frame,
        switches=switches,
        switching_cost=switching_cost_total,
        days_off=days_off,
        min_ratio=min_ratio,
        median_ratio=median_ratio,
        final_state=final_state,
        undefined_share=undefined_share,
        verdict=verdict,
    )


def _verdict(
    strategy: str,
    benchmark: str,
    policy: HurdlePolicy,
    days_off: int,
    n_days: int,
    switches: int,
    min_ratio: float,
    median_ratio: float,
    undefined_share: float,
    final_state: str,
    episodes: pd.DataFrame,
) -> str:
    rule = (
        f"Rule: {strategy} must deliver at least {policy.hurdle:g}x its incremental cost over {benchmark}, "
        f"measured on a trailing {policy.window}-day window, or after {policy.grace} consecutive breaching days "
        f"the book reverts to {benchmark}."
    )
    if undefined_share > 0.5:
        economics = (
            f" On {undefined_share:.0%} of evaluable days {strategy} cost no more to run than {benchmark} "
            f"(incremental cost under {policy.cost_floor_bps:g}bp a year) while tracking tighter, so the hurdle "
            f"was met by default rather than by margin: there was nothing to pay for."
        )
    elif np.isfinite(median_ratio):
        economics = (
            f" Where the strategy did cost more than {benchmark}, the realised payoff ratio had a median of "
            f"{median_ratio:,.0f}x and a minimum of {min_ratio:,.1f}x against a hurdle of {policy.hurdle:g}x."
        )
    else:
        economics = ""
    if days_off == 0:
        outcome = f" The switch never tripped: {strategy} was live for all {n_days:,} evaluated days."
    else:
        longest = int(episodes["days"].max()) if not episodes.empty else days_off
        outcome = (
            f" The switch tripped {switches} time(s); the book spent {days_off:,} days ({days_off / n_days:.0%}) "
            f"on {benchmark}, with the longest shutdown lasting {longest} days. Final state: {final_state}."
        )
    return rule + economics + outcome
