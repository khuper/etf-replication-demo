"""Content-addressed on-disk cache for price panels.

CSV rather than parquet on purpose: the cache is meant to be openable by a human
who is trying to work out why a number moved, and it must not add a binary
dependency to a repo whose whole point is that it runs anywhere.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from etflab.data.panel import PricePanel


def cache_key(source: str, tickers: tuple[str, ...], start: str, end: str, extra: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"source": source, "tickers": list(tickers), "start": start, "end": end, "extra": dict(extra)},
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class PanelCache:
    """A tiny, explicit cache. No TTL, no eviction, no magic.

    Market history for a closed date range does not expire, so a TTL would be a
    lie; the key includes the date range and every provider parameter, so a
    changed request is a different entry rather than a stale hit.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.csv"

    def meta_path_for(self, key: str) -> Path:
        return self.root / f"{key}.meta.json"

    def load(self, key: str, assets: tuple[str, ...], target: str) -> PricePanel | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        prices.index.name = "date"
        meta: dict[str, Any] = {}
        if self.meta_path_for(key).exists():
            meta = json.loads(self.meta_path_for(key).read_text())
        expected = [*assets, target]
        if list(prices.columns) != expected:
            # A key collision or a hand-edited cache file: refuse rather than
            # silently reorder columns, which would corrupt every weight vector.
            return None
        return PricePanel(prices, assets, target, meta.get("source", "cache"), {**meta, "cache_hit": True})

    def store(self, key: str, panel: PricePanel) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        panel.prices.to_csv(path, index_label="date")
        self.meta_path_for(key).write_text(
            json.dumps(
                {**dict(panel.meta), "source": panel.source, "fingerprint": panel.fingerprint(), **panel.describe()},
                indent=2,
                default=str,
            )
            + "\n"
        )
        return path
