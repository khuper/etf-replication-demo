"""etflab: a reproducible research harness for ETF replication.

The public surface is deliberately small. Everything a caller needs to run an
experiment, get numbers back, and prove where those numbers came from:

    >>> from etflab import ExperimentConfig, load_panel, run_horse_race
    >>> config = ExperimentConfig().validate()
    >>> panel = load_panel(config)
    >>> race = run_horse_race(panel, config)
    >>> race.verdict          # doctest: +SKIP
"""

from etflab.config import DEFAULT_ASSETS, DEFAULT_TARGET, ExperimentConfig
from etflab.data import PricePanel, load_panel, run_quality_gates

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_ASSETS",
    "DEFAULT_TARGET",
    "ExperimentConfig",
    "PricePanel",
    "__version__",
    "load_panel",
    "run_horse_race",
    "run_quality_gates",
]


def __getattr__(name: str):
    # Lazily exposed so that ``import etflab`` stays cheap and free of cycles:
    # the research layer imports config and data, not the other way round.
    if name == "run_horse_race":
        from etflab.research import run_horse_race

        return run_horse_race
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
