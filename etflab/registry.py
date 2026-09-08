"""An append-only index of every run, so runs can be found and compared.

One JSONL file. Not a database, because a database would be a dependency and a
migration and a thing to explain, and the requirement here is "find the run I did
on Tuesday and tell me what changed" -- which a line-per-run text file answers
completely and can be read with ``grep`` when this code is gone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from etflab.provenance import RunManifest

REGISTRY_NAME = "registry.jsonl"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: str
    created_at: str
    strategy: str
    target: str
    data_source: str
    config_hash: str
    data_fingerprint: str
    results_digest: str
    code: str
    quality_status: str
    degraded: bool
    headline: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "created_at": self.created_at,
            "strategy": self.strategy,
            "target": self.target,
            "data_source": self.data_source,
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "results_digest": self.results_digest,
            "code": self.code,
            "quality_status": self.quality_status,
            "degraded": self.degraded,
            "headline": self.headline,
        }


class RunRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / REGISTRY_NAME

    def append(self, manifest: RunManifest, run_dir: Path, headline: dict[str, float]) -> RunRecord:
        record = RunRecord(
            run_id=manifest.run_id,
            run_dir=str(run_dir),
            created_at=manifest.created_at,
            strategy=str(manifest.config.get("strategy", "?")),
            target=str(manifest.config.get("target", "?")),
            data_source=str(manifest.config.get("data_source", "?")),
            config_hash=manifest.config_hash,
            data_fingerprint=manifest.data_fingerprint,
            results_digest=manifest.results_digest,
            code=str(manifest.code.get("git_sha") or "nogit")[:8] + ("-dirty" if manifest.code.get("dirty") else ""),
            quality_status=manifest.quality_status,
            degraded=manifest.degraded,
            headline={k: float(v) for k, v in headline.items()},
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record.as_dict(), default=str) + "\n")
        return record

    def __iter__(self) -> Iterator[RunRecord]:
        if not self.path.exists():
            return iter(())
        records = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(RunRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # a hand-edited registry should degrade, not crash
        return iter(records)

    def records(self) -> list[RunRecord]:
        return list(self)

    def to_frame(self) -> pd.DataFrame:
        records = self.records()
        if not records:
            return pd.DataFrame()
        rows = []
        for record in records:
            row = record.as_dict()
            headline = row.pop("headline", {})
            rows.append({**row, **headline})
        return pd.DataFrame(rows)

    def find(self, run_id_prefix: str) -> list[RunRecord]:
        return [r for r in self.records() if r.run_id.startswith(run_id_prefix)]

    def latest(self) -> RunRecord | None:
        records = self.records()
        return records[-1] if records else None


def diff_records(left: RunRecord, right: RunRecord) -> dict[str, Any]:
    """What changed between two runs, and the first plausible explanation."""
    keys = sorted(set(left.headline) | set(right.headline))
    metric_diff = {
        key: {
            "left": left.headline.get(key),
            "right": right.headline.get(key),
            "delta": (right.headline.get(key, float("nan")) - left.headline.get(key, float("nan")))
            if key in left.headline and key in right.headline
            else None,
        }
        for key in keys
        if left.headline.get(key) != right.headline.get(key)
    }
    if left.config_hash != right.config_hash:
        explanation = "configuration differs"
    elif left.data_fingerprint != right.data_fingerprint:
        explanation = "same configuration, different data"
    elif left.results_digest != right.results_digest:
        explanation = "same experiment and data, different results: the code changed the answer"
    else:
        explanation = "identical"
    return {
        "explanation": explanation,
        "same_config": left.config_hash == right.config_hash,
        "same_data": left.data_fingerprint == right.data_fingerprint,
        "same_results": left.results_digest == right.results_digest,
        "code": [left.code, right.code],
        "metrics": metric_diff,
    }
