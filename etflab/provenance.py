"""Run identity, code version, and the manifest that ties results to their inputs.

The design decision worth stating: ``run_id`` is a hash of the *experiment*
(semantic config + data fingerprint), not of the code. If it included the git
SHA, every commit would rename every run and the interesting question would
become unanswerable.

Keeping code version separate makes it answerable. Two runs sharing a ``run_id``
are the same experiment; if their ``results_digest`` differs, the code changed
the answer. That is either a bug fix or a regression, and either way it is
something you want to be told about rather than something you want hidden behind
a new identifier. ``etf-lab verify`` is exactly that check, and CI runs it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from etflab.config import ExperimentConfig

TRACKED_PACKAGES = ("numpy", "pandas", "cvxpy", "scipy")


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass(frozen=True)
class CodeVersion:
    """Where the code came from. ``dirty`` is the field that matters."""

    git_sha: str | None
    git_branch: str | None
    dirty: bool
    package_version: str

    @property
    def short(self) -> str:
        sha = (self.git_sha or "nogit")[:8]
        return f"{sha}{'-dirty' if self.dirty else ''}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def code_version() -> CodeVersion:
    from etflab import __version__

    sha = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return CodeVersion(
        git_sha=sha,
        git_branch=branch,
        dirty=bool(status),
        package_version=__version__,
    )


def environment_fingerprint() -> dict[str, str]:
    """Enough of the environment to explain a numerical difference later.

    Solver versions are included because a change in CLARABEL can move a weight
    in the sixth decimal, and six decimals is exactly where a determinism check
    fails and a person then spends an afternoon.
    """
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
    }
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "absent"
    try:
        import cvxpy as cp

        versions["cvxpy_solvers"] = ",".join(sorted(cp.installed_solvers()))
    except ImportError:
        pass
    return versions


def canonical_digest(payload: Any) -> str:
    """Stable digest of a JSON-able payload.

    Floats go through ``repr`` so that ``0.1`` and ``0.10000000000000002`` are
    different, which is the entire purpose of a determinism check.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_run_id(config: ExperimentConfig, data_fingerprint: str) -> str:
    """Identity of the experiment: what was asked, on which data."""
    return hashlib.sha256(f"{config.semantic_hash()}|{data_fingerprint}".encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RunManifest:
    """The record that makes a result auditable a year later."""

    run_id: str
    config_hash: str
    data_fingerprint: str
    results_digest: str
    code: dict[str, Any]
    environment: dict[str, str]
    config: dict[str, Any]
    data: dict[str, Any]
    quality_status: str
    degraded: bool
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str) + "\n"

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        return cls(**json.loads(text))

    def write(self, directory: Path) -> Path:
        path = Path(directory) / "manifest.json"
        path.write_text(self.to_json())
        return path

    @classmethod
    def read(cls, directory: Path) -> RunManifest:
        return cls.from_json((Path(directory) / "manifest.json").read_text())

    def compare(self, other: RunManifest) -> dict[str, Any]:
        """Structured difference against another manifest.

        Ordered by what a reader should look at first: same experiment or not,
        then whether the answer moved, then why it might have.
        """
        return {
            "same_experiment": self.run_id == other.run_id,
            "same_config": self.config_hash == other.config_hash,
            "same_data": self.data_fingerprint == other.data_fingerprint,
            "same_results": self.results_digest == other.results_digest,
            "code_changed": self.code.get("git_sha") != other.code.get("git_sha"),
            "either_dirty": bool(self.code.get("dirty")) or bool(other.code.get("dirty")),
            "environment_diff": {
                key: [self.environment.get(key), other.environment.get(key)]
                for key in sorted(set(self.environment) | set(other.environment))
                if self.environment.get(key) != other.environment.get(key)
            },
        }


def build_manifest(
    config: ExperimentConfig,
    data_fingerprint: str,
    data_summary: Mapping[str, Any],
    results_payload: Any,
    quality_status: str,
    degraded: bool,
    notes: Mapping[str, Any] | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=compute_run_id(config, data_fingerprint),
        config_hash=config.semantic_hash(),
        data_fingerprint=data_fingerprint,
        results_digest=canonical_digest(results_payload),
        code=code_version().as_dict(),
        environment=environment_fingerprint(),
        config=config.as_dict(),
        data=dict(data_summary),
        quality_status=quality_status,
        degraded=degraded,
        notes=dict(notes or {}),
    )
