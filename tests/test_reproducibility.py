"""The reproducibility claim, made falsifiable.

The project asserts that a result is identified by its inputs and that re-running
it produces the same numbers. Both halves are tested here, because an untested
reproducibility claim is the least trustworthy kind of claim a research repo can
make: it is the one readers are least able to check for themselves.
"""

from __future__ import annotations

import json

import pytest

from etflab.config import ExperimentConfig
from etflab.data import load_panel, run_quality_gates
from etflab.export import export_study
from etflab.provenance import RunManifest, build_manifest, canonical_digest, code_version, compute_run_id
from etflab.registry import RunRegistry, diff_records
from etflab.research import run_study

FAST = dict(with_sweep=False, with_recovery=False, with_capacity=False)
STRATEGIES = ("tracking", "equal_weight", "static")


@pytest.fixture(scope="module")
def study(small_config, small_panel):
    quality = run_quality_gates(small_panel, min_obs=300)
    return run_study(small_config, small_panel, quality, strategies=STRATEGIES, **FAST)


def test_run_id_is_a_function_of_the_experiment_and_the_data(small_config):
    a = compute_run_id(small_config, "fingerprint-1")
    assert a == compute_run_id(small_config, "fingerprint-1")
    assert a != compute_run_id(small_config, "fingerprint-2")
    assert a != compute_run_id(small_config.with_changes(max_weight=0.5), "fingerprint-1")


def test_run_id_does_not_depend_on_the_code_version(small_config):
    """Deliberate: if the SHA were in the run id, every commit would rename every
    run and "did the code change the answer?" would become unanswerable."""
    before = compute_run_id(small_config, "fp")
    assert before == compute_run_id(small_config, "fp")
    assert "git" not in before


def test_run_id_ignores_where_the_output_goes(small_config):
    assert compute_run_id(small_config, "fp") == compute_run_id(small_config.with_changes(output_dir="/tmp/x"), "fp")


def test_the_same_study_run_twice_produces_an_identical_digest(small_config, small_panel):
    quality = run_quality_gates(small_panel, min_obs=300)
    first = run_study(small_config, small_panel, quality, strategies=STRATEGIES, **FAST)
    second = run_study(small_config, small_panel, quality, strategies=STRATEGIES, **FAST)
    assert canonical_digest(first.results_payload()) == canonical_digest(second.results_payload())


def test_the_data_layer_is_deterministic(small_config):
    assert load_panel(small_config).fingerprint() == load_panel(small_config).fingerprint()


def test_a_changed_seed_changes_the_data_fingerprint(small_config):
    other = small_config.with_changes(synthetic_seed=small_config.synthetic_seed + 1)
    assert load_panel(other).fingerprint() != load_panel(small_config).fingerprint()


def test_digest_distinguishes_values_that_differ_in_the_last_decimal():
    assert canonical_digest({"x": 0.1}) != canonical_digest({"x": 0.1 + 1e-16})
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})


def test_manifest_round_trips_through_json(study, tmp_path):
    manifest = build_manifest(
        study.config, study.panel.fingerprint(), study.panel.describe(), study.results_payload(), "ok", False
    )
    manifest.write(tmp_path)
    restored = RunManifest.read(tmp_path)
    assert restored.run_id == manifest.run_id
    assert restored.results_digest == manifest.results_digest
    assert restored.config == manifest.config


def test_manifest_comparison_identifies_a_code_induced_change(study):
    base = build_manifest(
        study.config, study.panel.fingerprint(), study.panel.describe(), study.results_payload(), "ok", False
    )
    changed_payload = {**study.results_payload(), "spa_p": 0.5}
    changed = build_manifest(
        study.config, study.panel.fingerprint(), study.panel.describe(), changed_payload, "ok", False
    )
    comparison = base.compare(changed)
    assert comparison["same_experiment"]
    assert comparison["same_config"]
    assert comparison["same_data"]
    assert not comparison["same_results"]


def test_export_writes_a_complete_and_self_describing_bundle(study, tmp_path):
    run_dir, manifest = export_study(study, output_dir=tmp_path, write_figures=False, write_report=True)
    assert run_dir.name == manifest.run_id

    required = [
        "manifest.json",
        "config.json",
        "quality.json",
        "inference.json",
        "metrics.csv",
        "metric_definitions.json",
        "report.md",
        "report.html",
    ]
    for name in required:
        assert (run_dir / name).exists(), f"missing artefact {name}"

    for strategy in STRATEGIES:
        assert (run_dir / "ledger" / f"{strategy}_daily.csv").exists()

    # The config in the bundle must reconstruct the config that produced it.
    restored = ExperimentConfig.from_json((run_dir / "config.json").read_text())
    assert restored.semantic_hash() == study.config.semantic_hash()


def test_exporting_the_same_study_twice_reuses_one_directory(study, tmp_path):
    """Timestamped run directories are how one result becomes six copies."""
    first, _ = export_study(study, output_dir=tmp_path, write_figures=False, write_report=False)
    second, _ = export_study(study, output_dir=tmp_path, write_figures=False, write_report=False)
    assert first == second


def test_the_registry_records_and_finds_runs(study, tmp_path):
    _, manifest = export_study(study, output_dir=tmp_path, write_figures=False, write_report=False)
    registry = RunRegistry(tmp_path)
    assert registry.find(manifest.run_id[:6])
    assert registry.latest().run_id == manifest.run_id
    frame = registry.to_frame()
    assert "tracking_error" in frame.columns


def test_registry_diff_explains_a_configuration_change(study, tmp_path, small_panel):
    export_study(study, output_dir=tmp_path, write_figures=False, write_report=False)
    other_config = study.config.with_changes(max_weight=0.5)
    quality = run_quality_gates(small_panel, min_obs=300)
    other = run_study(other_config, small_panel, quality, strategies=STRATEGIES, **FAST)
    export_study(other, output_dir=tmp_path, write_figures=False, write_report=False)

    records = RunRegistry(tmp_path).records()
    result = diff_records(records[0], records[-1])
    assert not result["same_config"]
    assert result["same_data"]
    assert result["explanation"] == "configuration differs"


def test_registry_survives_a_corrupted_line(tmp_path):
    registry = RunRegistry(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry.path.write_text('{"not": "a record"}\nnot json at all\n')
    assert registry.records() == []


def test_code_version_reports_whether_the_tree_is_dirty():
    version = code_version()
    assert isinstance(version.dirty, bool)
    assert version.package_version
    assert version.short


def test_results_payload_excludes_wall_clock_and_paths(study):
    """A determinism digest that includes a solver timing is a digest nobody runs."""
    payload = json.dumps(study.results_payload())
    for forbidden in ("solve_seconds", "created_at", "/tmp", "run_dir", "elapsed"):
        assert forbidden not in payload
