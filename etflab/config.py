"""Experiment configuration: the complete, hashable description of one research run.

Every input that can change a number in the output lives here. Nothing else does.
That invariant is what makes :func:`ExperimentConfig.semantic_hash` a meaningful
identity: two configs with the same hash must produce the same results, and the
determinism test in ``tests/test_reproducibility.py`` holds us to it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import date
from typing import Any

# Candidate proxies for the default (synthetic) experiment. Real tickers are used
# as labels so the synthetic market reads like the real problem it stands in for.
DEFAULT_ASSETS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "VEA",
    "VWO",
    "HYG",
    "LQD",
    "TIP",
    "GLD",
    "VNQ",
)
DEFAULT_TARGET = "PSP"

DATA_SOURCES = ("synthetic", "yfinance", "csv")
COST_MODELS = ("bps", "spread_impact")

#: Fields that describe *where* output goes or *how loudly* we talk about it.
#: They are deliberately excluded from the semantic hash: moving an output
#: directory must not change a run's identity.
NON_SEMANTIC_FIELDS = frozenset({"output_dir", "cache_dir", "log_level"})


def _default_start() -> str:
    return "2013-01-02"


def _default_end() -> str:
    return "2025-12-31"


@dataclass(frozen=True)
class ExperimentConfig:
    """All inputs needed to reproduce one research run.

    Frozen on purpose. Mutating a config mid-run is the classic way to produce
    results whose provenance quietly stops matching the manifest; use
    :meth:`with_changes` to derive a new one instead.
    """

    # --- problem definition -------------------------------------------------
    target: str = DEFAULT_TARGET
    assets: tuple[str, ...] = DEFAULT_ASSETS
    start: str = field(default_factory=_default_start)
    end: str = field(default_factory=_default_end)

    # --- data layer ---------------------------------------------------------
    data_source: str = "synthetic"
    synthetic_seed: int = 20_240_101
    synthetic_leverage: float = 1.0
    csv_path: str | None = None
    offline: bool = True
    cache_dir: str = ".cache/etflab"

    # --- walk-forward protocol ---------------------------------------------
    train_days: int = 504
    train_mode: str = "expanding"
    rebalance_days: int = 63
    embargo_days: int = 1

    # --- portfolio constraints ---------------------------------------------
    max_weight: float = 0.25
    max_turnover: float = 0.20

    # --- strategy -----------------------------------------------------------
    strategy: str = "tracking"
    cvar_ratio: float | None = None
    cvar_alpha: float = 0.05
    ridge_lambda: float = 0.0

    # --- execution costs ----------------------------------------------------
    cost_model: str = "bps"
    cost_bps: float = 5.0
    impact_coef: float = 0.7
    portfolio_notional: float = 50_000_000.0
    max_participation: float = 0.05

    # --- governance ---------------------------------------------------------
    hurdle: float = 20.0
    hurdle_window: int = 252
    hurdle_grace: int = 63
    hurdle_reactivate: bool = True

    # --- inference ----------------------------------------------------------
    bootstrap_samples: int = 2_000
    bootstrap_block: int = 21
    inference_seed: int = 7

    # --- plumbing (non-semantic) -------------------------------------------
    output_dir: str = "outputs"
    log_level: str = "info"

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks plain assignment; normalisation goes through
        # object.__setattr__, which is the documented escape hatch for __post_init__.
        object.__setattr__(self, "target", self.target.upper().strip())
        deduped = tuple(dict.fromkeys(a.upper().strip() for a in self.assets if a and a.strip()))
        object.__setattr__(self, "assets", deduped)
        object.__setattr__(self, "strategy", self.strategy.lower().strip())
        object.__setattr__(self, "data_source", self.data_source.lower().strip())
        object.__setattr__(self, "cost_model", self.cost_model.lower().strip())

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> ExperimentConfig:
        """Raise :class:`ValueError` on any configuration that cannot be run.

        Returns ``self`` so it can be chained. Every check here exists because
        the alternative is a confusing failure deep inside the solver or, worse,
        a plausible-looking number that means nothing.
        """
        if not self.target:
            raise ValueError("A target ticker is required.")
        if self.target in self.assets:
            raise ValueError(
                f"The target {self.target!r} cannot also be a candidate asset: "
                "replicating an instrument with itself is not a research question."
            )
        if len(self.assets) < 2:
            raise ValueError("At least two candidate assets are required.")

        start, end = _parse_date(self.start, "start"), _parse_date(self.end, "end")
        if start >= end:
            raise ValueError(f"start ({self.start}) must be strictly before end ({self.end}).")

        if self.data_source not in DATA_SOURCES:
            raise ValueError(f"data_source must be one of {DATA_SOURCES}, got {self.data_source!r}.")
        if self.data_source == "csv" and not self.csv_path:
            raise ValueError("data_source='csv' requires csv_path.")
        if self.synthetic_leverage <= 0:
            raise ValueError("synthetic_leverage must be positive.")

        if self.train_days < 60:
            raise ValueError("train_days must be at least 60 observations to estimate anything stable.")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be positive.")
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative.")
        if self.embargo_days >= self.train_days:
            raise ValueError("embargo_days must be smaller than the training window.")
        if self.train_mode not in ("expanding", "rolling"):
            raise ValueError("train_mode must be 'expanding' or 'rolling'.")

        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must lie in (0, 1].")
        if len(self.assets) * self.max_weight < 1 - 1e-9:
            raise ValueError(
                f"max_weight is infeasible: {len(self.assets)} assets x {self.max_weight:.1%} "
                "cannot fund a fully invested portfolio."
            )
        if not 0 <= self.max_turnover <= 2:
            raise ValueError("max_turnover is an L1 weight distance and must lie in [0, 2].")

        if self.cvar_ratio is not None and self.cvar_ratio <= 0:
            raise ValueError("cvar_ratio must be positive when set.")
        if not 0 < self.cvar_alpha < 0.5:
            raise ValueError("cvar_alpha must lie in (0, 0.5).")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda cannot be negative.")

        if self.cost_model not in COST_MODELS:
            raise ValueError(f"cost_model must be one of {COST_MODELS}, got {self.cost_model!r}.")
        for name in ("cost_bps", "impact_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")
        if self.portfolio_notional <= 0:
            raise ValueError("portfolio_notional must be positive: costs scale with the size of the book.")
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must lie in (0, 1].")

        if self.hurdle <= 0:
            raise ValueError("hurdle must be positive: a model that need not pay for itself is not governed.")
        if self.hurdle_window < 21:
            raise ValueError("hurdle_window must be at least a month of trading days.")
        if self.hurdle_grace < 1:
            raise ValueError("hurdle_grace must be at least one day.")

        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples below 100 gives confidence intervals nobody should trust.")
        if self.bootstrap_block < 1:
            raise ValueError("bootstrap_block must be positive.")
        return self

    # ------------------------------------------------------------------ #
    # Identity and serialisation
    # ------------------------------------------------------------------ #
    def with_changes(self, **changes: Any) -> ExperimentConfig:
        """Return a validated copy with ``changes`` applied."""
        return replace(self, **changes).validate()

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    def semantic_dict(self) -> dict[str, Any]:
        """The subset of the config that can change a number in the results."""
        return {k: v for k, v in self.as_dict().items() if k not in NON_SEMANTIC_FIELDS}

    def semantic_hash(self) -> str:
        """Stable 16-hex-character identity for the *meaning* of this config.

        Keys are sorted, so the hash does not depend on field declaration order,
        and floats are serialised via ``repr`` so 0.2 and 0.2000000001 stay distinct.
        """
        payload = json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":"), default=repr)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        payload = dict(data)
        if "assets" in payload:
            payload["assets"] = tuple(payload["assets"])
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> ExperimentConfig:
        return cls.from_dict(json.loads(text))

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def universe(self) -> tuple[str, ...]:
        """Assets plus target, in the canonical column order used everywhere."""
        return (*self.assets, self.target)

    @property
    def start_date(self) -> date:
        return _parse_date(self.start, "start")

    @property
    def end_date(self) -> date:
        return _parse_date(self.end, "end")


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:  # pragma: no cover - message is the point
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}.") from exc
