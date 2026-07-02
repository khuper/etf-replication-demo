"""Run configuration for the replicator workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

DEFAULT_ASSETS = ["SPY", "QQQ", "VEA", "VWO", "BND", "AGG", "LQD", "TIP", "GLD", "VNQ"]
DEFAULT_TARGET = "PSP"

# Instrument-type tags for the built-in baskets. Users can retag or add
# tickers via ``--asset-class``; untagged tickers fall in the "other" group.
DEFAULT_ASSET_CLASSES: Dict[str, str] = {
    "SPY": "equity",
    "QQQ": "equity",
    "IWM": "equity",
    "EFA": "equity",
    "VEA": "equity",
    "VWO": "equity",
    "BND": "fixed_income",
    "AGG": "fixed_income",
    "LQD": "fixed_income",
    "TIP": "fixed_income",
    "TLT": "fixed_income",
    "HYG": "fixed_income",
    "GLD": "commodity",
    "VNQ": "real_estate",
}

UNTAGGED_CLASS = "other"

# Historical stress windows used for tearsheet evidence and the rolling
# correlation analysis.
STRESS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Shock": ("2022-01-03", "2022-06-16"),
    "2018 Vol Shock": ("2018-01-26", "2018-02-08"),
}


# Long enough that the out-of-sample window (which starts after the initial
# training period) still covers the named stress windows above.
def default_date_range(lookback_years: int = 10) -> Tuple[str, str]:
    end = datetime.now()
    start = end - timedelta(days=lookback_years * 365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@dataclass
class ReplicatorConfig:
    """All knobs for one replication run."""

    assets: List[str] = field(default_factory=lambda: list(DEFAULT_ASSETS))
    target: str = DEFAULT_TARGET
    start_date: str = ""
    end_date: str = ""
    max_weight: float = 0.25
    max_turnover: float = 0.20
    cvar_ratio: float = 1.0
    cvar_alpha: float = 0.05
    cost_bps: float = 10.0
    initial_train_size: int = 504
    step: int = 126
    output_dir: str = "outputs"
    asset_classes: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ASSET_CLASSES))
    group_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.start_date or not self.end_date:
            start, end = default_date_range()
            self.start_date = self.start_date or start
            self.end_date = self.end_date or end

    def asset_class_of(self, ticker: str) -> str:
        return self.asset_classes.get(ticker, UNTAGGED_CLASS)

    def classes_by_asset(self) -> Dict[str, str]:
        return {a: self.asset_class_of(a) for a in self.assets}

    def validate(self) -> None:
        if self.target in self.assets:
            raise ValueError(f"Target {self.target} cannot also be a basket asset.")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1].")
        if self.max_turnover < 0:
            raise ValueError("max_turnover must be non-negative.")
        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be in (0, 1).")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative.")
        validate_group_bounds(self.group_bounds, self.classes_by_asset(), self.max_weight)


def validate_group_bounds(
    group_bounds: Dict[str, Tuple[float, float]],
    classes_by_asset: Dict[str, str],
    max_weight: float,
) -> None:
    """Reject group bounds that make the optimization trivially infeasible."""
    if not group_bounds:
        return

    members: Dict[str, int] = {}
    for cls in classes_by_asset.values():
        members[cls] = members.get(cls, 0) + 1

    for group, (lo, hi) in group_bounds.items():
        if not 0 <= lo <= hi <= 1:
            raise ValueError(f"Group bound for '{group}' must satisfy 0 <= min <= max <= 1, got {lo}:{hi}.")
        n = members.get(group, 0)
        if n == 0 and lo > 0:
            raise ValueError(
                f"Group '{group}' has a minimum of {lo} but no basket asset is tagged with that class "
                f"(known classes: {sorted(set(classes_by_asset.values()))})."
            )
        if lo > n * max_weight:
            raise ValueError(
                f"Group '{group}' minimum {lo} is unreachable: {n} member(s) capped at {max_weight} "
                f"can hold at most {n * max_weight:.2f}."
            )

    if sum(lo for lo, _ in group_bounds.values()) > 1 + 1e-9:
        raise ValueError("Group minimums sum to more than 100% of the portfolio.")

    # If every asset belongs to a capped group, the caps must leave room for a
    # fully invested portfolio.
    capped_classes = set(group_bounds)
    if set(classes_by_asset.values()) <= capped_classes:
        total_cap = sum(hi for _, hi in group_bounds.values())
        if total_cap < 1 - 1e-9:
            raise ValueError(
                f"Group maximums cover every asset but sum to {total_cap:.2f} < 1, "
                "so the portfolio cannot be fully invested."
            )
