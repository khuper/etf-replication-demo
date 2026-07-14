"""Configuration for reproducible ETF replication experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any


DEFAULT_ASSETS = ("SPY", "QQQ", "VEA", "VWO", "BND", "LQD", "TIP", "GLD", "VNQ")


def _default_start() -> str:
    return (date.today() - timedelta(days=5 * 365)).isoformat()


@dataclass
class ResearchConfig:
    """All inputs needed to reproduce one research run."""

    target: str = "PSP"
    assets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ASSETS)
    start_date: str = field(default_factory=_default_start)
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    initial_train_size: int = 504
    rebalance_days: int = 126
    max_weight: float = 0.25
    max_turnover: float = 0.20
    transaction_cost_bps: float = 5.0
    model: str = "tracking"
    cvar_ratio: float = 1.0
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        self.target = self.target.upper().strip()
        self.assets = tuple(dict.fromkeys(asset.upper().strip() for asset in self.assets if asset.strip()))

    def validate(self) -> None:
        if not self.target:
            raise ValueError("A target ticker is required.")
        if self.target in self.assets:
            raise ValueError("The target ticker cannot also be a candidate asset.")
        if len(self.assets) < 2:
            raise ValueError("At least two candidate assets are required.")
        if self.initial_train_size < 20:
            raise ValueError("The training window must contain at least 20 observations.")
        if self.rebalance_days < 1:
            raise ValueError("Rebalance days must be positive.")
        if not 0 < self.max_weight <= 1:
            raise ValueError("Max weight must be between 0 and 1.")
        if len(self.assets) * self.max_weight < 1 - 1e-9:
            raise ValueError(
                f"Max weight is infeasible: {len(self.assets)} assets × {self.max_weight:.1%} cannot fund 100%."
            )
        if not 0 <= self.max_turnover <= 2:
            raise ValueError("Max turnover must be between 0 and 2 (L1 weight distance).")
        if self.transaction_cost_bps < 0:
            raise ValueError("Transaction costs cannot be negative.")
        if self.model not in {"tracking", "cvar"}:
            raise ValueError("Model must be 'tracking' or 'cvar'.")
        if self.cvar_ratio <= 0:
            raise ValueError("CVaR ratio must be positive.")
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date.")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["assets"] = list(self.assets)
        return data
