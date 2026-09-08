"""Writing a run to disk: the artefact bundle a result is allowed to travel as.

A number in a terminal is not a result. A directory containing the numbers, the
configuration that produced them, the fingerprint of the data they came from, the
code version, and a digest that lets someone else check all of it -- that is a
result. This module writes that directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from etflab.config import ExperimentConfig
from etflab.metrics import describe_metrics
from etflab.provenance import RunManifest, build_manifest
from etflab.registry import RunRegistry


def _write_csv(frame: pd.DataFrame | pd.Series, path: Path, index_label: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label=index_label or None)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def export_study(
    study: Any,
    *,
    output_dir: str | Path | None = None,
    write_figures: bool = True,
    write_report: bool = True,
) -> tuple[Path, RunManifest]:
    """Write every artefact for ``study`` and return ``(run_dir, manifest)``.

    The directory is named by ``run_id``, which is a function of the experiment
    and the data -- not of the wall clock. Re-running the same experiment
    overwrites the same directory on purpose: two directories differing only in
    a timestamp are the classic way to end up with six copies of one result and
    no idea which is current.
    """
    config: ExperimentConfig = study.config
    manifest = build_manifest(
        config=config,
        data_fingerprint=study.panel.fingerprint(),
        data_summary=study.panel.describe(),
        results_payload=study.results_payload(),
        quality_status=study.quality.status,
        degraded=study.quality.status == "fail",
        notes={
            "strategies": list(study.race.results),
            "benchmark": study.race.benchmark,
            # Recorded because the results digest depends on which analyses ran:
            # verifying a run that included a sweep against a re-run that did not
            # would report a regression that is really a configuration mismatch.
            "analyses": dict(getattr(study, "analyses", {}) or {}),
        },
    )

    root = Path(output_dir or config.output_dir)
    run_dir = root / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(json.loads(config.to_json()), run_dir / "config.json")
    _write_json(study.quality.as_dict(), run_dir / "quality.json")
    _write_json(study.race.as_dict(), run_dir / "inference.json")
    _write_json(describe_metrics(), run_dir / "metric_definitions.json")

    _write_csv(study.race.metrics, run_dir / "metrics.csv", "strategy")
    if not study.regimes.empty:
        _write_csv(study.regimes, run_dir / "regimes.csv", "regime")
    _write_csv(study.attribution, run_dir / "attribution.csv", "leg")
    _write_csv(study.tails, run_dir / "conditional_tails.csv", "quantile")
    _write_csv(study.worst_windows, run_dir / "worst_windows.csv", "window_end")
    _write_csv(study.correlation_stability, run_dir / "correlation_stability.csv", "asset")
    _write_csv(study.shock, run_dir / "beta_shock.csv", "target_shock")
    if getattr(study, "constraints", None) is not None and not study.constraints.empty:
        _write_csv(study.constraints, run_dir / "constraint_activity.csv", "strategy")

    for name, result in study.race.results.items():
        _write_csv(result.daily, run_dir / "ledger" / f"{name}_daily.csv", "date")
        _write_csv(result.weights, run_dir / "ledger" / f"{name}_weights.csv", "rebalance_date")
        _write_csv(result.rebalances, run_dir / "ledger" / f"{name}_rebalances.csv", "rebalance_date")

    if study.recovery is not None:
        _write_csv(study.recovery.frame, run_dir / "recovery.csv", "sample_size")
        _write_json(study.recovery.as_dict(), run_dir / "recovery.json")
    if study.sweep is not None:
        _write_csv(study.sweep.frame, run_dir / "sweep.csv", "config")
        _write_json(study.sweep.as_dict(), run_dir / "overfitting.json")
    if study.cost_curve is not None:
        _write_csv(study.cost_curve, run_dir / "capacity.csv", "notional")
    for key, report in getattr(study, "governance", {}).items():
        _write_csv(report.daily, run_dir / "governance" / f"{key}_signal.csv", "date")
        _write_csv(report.governed, run_dir / "governance" / f"{key}_ledger.csv", "date")
        _write_json(report.as_dict(), run_dir / "governance" / f"{key}.json")
    if study.cost_breakeven is not None:
        _write_json(
            {
                "cost_breakeven": study.cost_breakeven.as_dict(),
                "capacity": study.capacity.as_dict() if study.capacity else None,
            },
            run_dir / "breakeven.json",
        )

    figure_dir = run_dir / "figures"
    if write_figures:
        from etflab.report.figures import make_figures

        make_figures(study, figure_dir)

    if write_report:
        from etflab.report.html import render_html
        from etflab.report.markdown import render_markdown

        (run_dir / "report.md").write_text(render_markdown(study, manifest))
        (run_dir / "report.html").write_text(render_html(study, manifest, figure_dir))

    manifest.write(run_dir)
    RunRegistry(root).append(manifest, run_dir, study.headline)
    return run_dir, manifest
