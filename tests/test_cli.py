"""End-to-end command line tests.

These are the tests that would have caught every "it works in the notebook"
failure. Each command is run the way a reader would run it, on the shipped
defaults, with no network.
"""

from __future__ import annotations

import json

import pytest

from etflab.cli import build_parser, main

FAST_DATA = [
    "--assets",
    "SPY",
    "QQQ",
    "IWM",
    "HYG",
    "GLD",
    "--start",
    "2015-01-01",
    "--end",
    "2019-12-31",
    "--train-days",
    "252",
    "--rebalance-days",
    "63",
    "--max-weight",
    "0.4",
    "--max-turnover",
    "0.3",
]


def _run(argv, tmp_path):
    return main([*argv, "--output", str(tmp_path)])


def test_parser_builds_and_exposes_every_command():
    parser = build_parser()
    subparsers = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    names = set()
    for action in subparsers:
        names.update(action.choices or {})
    for expected in ("study", "compare", "run", "sweep", "validate", "verify", "runs", "diff", "shell"):
        assert expected in names


def test_validate_passes_on_the_shipped_defaults(tmp_path, capsys):
    assert _run(["validate", *FAST_DATA], tmp_path) == 0


def test_validate_emits_parseable_json(tmp_path, capsys):
    _run(["validate", *FAST_DATA, "--json"], tmp_path)
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] in {"ok", "warn", "fail"}
    assert {g["name"] for g in payload["gates"]} >= {"stale_prices", "multicollinearity"}


def test_run_produces_metrics(tmp_path, capsys):
    assert _run(["run", *FAST_DATA, "--json"], tmp_path) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0 < payload["metrics"]["tracking_error"] < 1
    assert payload["config_hash"]


def test_compare_returns_a_ranked_field_and_a_verdict(tmp_path, capsys):
    assert (
        _run(
            [
                "compare",
                *FAST_DATA,
                "--strategies",
                "tracking",
                "equal_weight",
                "static",
                "--bootstrap-samples",
                "200",
                "--json",
            ],
            tmp_path,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["best"] in {"tracking", "equal_weight", "static"}
    assert len(payload["verdict"]) > 100
    assert "spa" in payload


def test_study_writes_a_complete_run_directory(tmp_path):
    assert (
        _run(
            [
                "study",
                *FAST_DATA,
                "--strategies",
                "tracking",
                "equal_weight",
                "static",
                "--bootstrap-samples",
                "200",
                "--no-sweep",
                "--no-capacity",
                "--no-figures",
            ],
            tmp_path,
        )
        == 0
    )
    runs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(runs) == 1
    for artefact in ("manifest.json", "config.json", "metrics.csv", "report.md", "report.html"):
        assert (runs[0] / artefact).exists()


def test_verify_reproduces_a_study_it_just_wrote(tmp_path):
    argv = [
        "study",
        *FAST_DATA,
        "--strategies",
        "tracking",
        "equal_weight",
        "static",
        "--bootstrap-samples",
        "200",
        "--no-sweep",
        "--no-capacity",
        "--no-figures",
    ]
    assert _run(argv, tmp_path) == 0
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert main(["verify", str(run_dir), "--output-dir", str(tmp_path)]) == 0


def test_runs_and_diff_operate_on_the_registry(tmp_path, capsys):
    base = [
        "study",
        *FAST_DATA,
        "--strategies",
        "tracking",
        "equal_weight",
        "--bootstrap-samples",
        "200",
        "--no-sweep",
        "--no-capacity",
        "--no-figures",
    ]
    _run(base, tmp_path)
    _run([*base, "--max-weight", "0.5"], tmp_path)
    capsys.readouterr()

    assert main(["runs", "--output", str(tmp_path), "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 2

    assert main(["diff", records[0]["run_id"], records[1]["run_id"], "--output", str(tmp_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["explanation"] == "configuration differs"


def test_sweep_reports_overfitting(tmp_path, capsys):
    assert _run(["sweep", *FAST_DATA, "--strategy", "tracking", "--json"], tmp_path) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0.0 <= payload["pbo"]["pbo"] <= 1.0
    assert payload["n_configs"] >= 4
    assert "noise_reference_pbo" in payload


def test_an_invalid_configuration_fails_with_a_message_not_a_traceback(tmp_path, capsys):
    assert _run(["run", "--assets", "SPY", "QQQ", "GLD", "--max-weight", "0.25"], tmp_path) == 1
    assert "infeasible" in capsys.readouterr().out.lower()


def test_offline_is_the_default_and_a_live_source_fails_cleanly(tmp_path, capsys):
    """No network in CI: asking for real prices without a cache must produce an
    explanation and an exit code, not a stack trace."""
    code = _run(["run", "--data-source", "yfinance", "--cache-dir", str(tmp_path / "empty-cache")], tmp_path)
    assert code == 1
    output = capsys.readouterr().out.lower()
    assert "offline" in output or "cache" in output


def test_unknown_strategy_is_rejected_by_the_parser(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--strategy", "wishful-thinking"])


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_verify_reproduces_a_full_study_including_the_sweep(tmp_path):
    """The regression this guards: the digest depends on which optional analyses
    ran, so verify has to re-run exactly the set the manifest recorded. Guessing
    reports a regression that is really a configuration mismatch -- and it was
    the CI research job that would have failed, not a test."""
    argv = [
        "study",
        *FAST_DATA,
        "--strategies",
        "tracking",
        "equal_weight",
        "--bootstrap-samples",
        "200",
        "--no-capacity",
        "--no-figures",
    ]
    assert _run(argv, tmp_path) == 0
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["notes"]["analyses"]["sweep"] is True
    assert main(["verify", str(run_dir)]) == 0


def test_verify_does_not_modify_the_run_it_is_checking(tmp_path):
    """A verification that rewrites its own evidence is not a verification."""
    argv = [
        "study",
        *FAST_DATA,
        "--strategies",
        "tracking",
        "equal_weight",
        "--bootstrap-samples",
        "200",
        "--no-sweep",
        "--no-capacity",
        "--no-figures",
    ]
    assert _run(argv, tmp_path) == 0
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    registry_before = (tmp_path / "registry.jsonl").read_bytes()

    assert main(["verify", str(run_dir)]) == 0

    after = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    assert after == before, "verify rewrote the artefacts it was checking"
    assert (tmp_path / "registry.jsonl").read_bytes() == registry_before


def test_verify_detects_a_changed_result(tmp_path, monkeypatch):
    """A control that has never been shown to fire is decoration."""
    argv = [
        "study",
        *FAST_DATA,
        "--strategies",
        "tracking",
        "equal_weight",
        "--bootstrap-samples",
        "200",
        "--no-sweep",
        "--no-capacity",
        "--no-figures",
    ]
    assert _run(argv, tmp_path) == 0
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())

    import etflab.metrics as metrics_module

    original = metrics_module.annualised_vol
    monkeypatch.setattr(metrics_module, "annualised_vol", lambda s: original(s) * 1.01)
    assert main(["verify", str(run_dir)]) == 1
