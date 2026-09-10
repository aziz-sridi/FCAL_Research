from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fcal_study.baseline_fixed_results import (
    GENERIC_BACKGROUND,
    GENERIC_FOREGROUND,
    METRIC_COLUMNS,
)
from fcal_study.tuned_results import (
    FIXED_DISPLAY_NAMES,
    build,
    tuning_accounting,
    tuning_attempt_table,
)

METHODS = ("fcal_v2", "rrr", "gain_ext", "camal", "selective_faster_isnet")


def _evaluation(score: float) -> list[dict]:
    rows = [{
        "variant": "clean",
        "accuracy": score,
        "balanced_accuracy": score - 0.01,
        "macro_f1": score - 0.02,
        "strict_leakage": 1.0 - score,
    }, {
        "variant": "foreground_only",
        "accuracy": score - 0.03,
    }, {
        "variant": "background_only",
        "accuracy": 0.05,
    }]
    rows.extend({"variant": name, "accuracy": score - 0.04} for name in GENERIC_BACKGROUND)
    rows.extend({"variant": name, "accuracy": score - 0.05} for name in GENERIC_FOREGROUND)
    return rows


def _manifest(path: Path) -> None:
    methods = {}
    for method in METHODS:
        methods[method] = {
            "display_name": FIXED_DISPLAY_NAMES[method],
            "coefficient": f"{method}.weight",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5],
            "supported_architectures": ["resnet50"],
            "template": "unused.yaml",
        }
    payload = {
        "study_id": "fixture-study",
        "selection": {
            "split": "validation",
            "metric": "accuracy",
            "checkpoint_rule": "best-validation.pt",
            "seed": 42,
        },
        "compute": {
            "gpus_per_job": 1,
            "max_concurrent_jobs": 1,
            "gpu_hour_accounting": "allocated_wall_clock",
        },
        "successive_halving": {
            "initial_trials": 5,
            "max_attempts_per_job": 2,
            "rungs": [
                {"total_epochs": 10, "survivors": 2},
                {"total_epochs": 25, "survivors": 1},
                {"total_epochs": 50, "survivors": 1},
            ],
        },
        "fixed_parent_split": "unused",
        "architectures": {"resnet50": {"fixed_parent_checkpoint": "unused"}},
        "methods": methods,
        "test_seal": {
            "evaluation_during_tuning": False,
            "unseal_after": "one_winner_per_supported_method_architecture_frozen",
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    manifest_path = tmp_path / "manifest.yaml"
    frozen_path = tmp_path / "frozen.yaml"
    ledger_path = tmp_path / "attempts.jsonl"
    fixed_path = tmp_path / "fixed.csv"
    _manifest(manifest_path)
    winners = []
    records = []
    fixed_rows = []
    for index, method in enumerate(METHODS):
        run_dir = tmp_path / "runs" / method
        evaluation = run_dir / "evaluation" / "summary.json"
        evaluation.parent.mkdir(parents=True)
        score = 0.70 + index / 100
        evaluation.write_text(json.dumps(_evaluation(score)), encoding="utf-8")
        checkpoint = run_dir / "best-validation.pt"
        checkpoint.write_bytes(method.encode())
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump({"training": {"epochs": 50}}), encoding="utf-8")
        trial = "trial-04"
        winners.append({
            "architecture": "resnet50",
            "method": method,
            "trial_id": trial,
            "coefficient": f"{method}.weight",
            "value": 0.5,
            "validation_accuracy": score - 0.1,
            "checkpoint": str(checkpoint),
        })
        for rung, trials in (("rung-01", range(5)), ("rung-02", (3, 4)), ("rung-03", (4,))):
            for trial_index in trials:
                trial_name = f"trial-{trial_index:02d}"
                selected_final = rung == "rung-03"
                records.append({
                    "study_id": "fixture-study",
                    "job_id": f"resnet50/{method}/{trial_name}/{rung}",
                    "attempt": 1,
                    "status": "success",
                    "elapsed_seconds": 360.0,
                    "gpu_hours": 0.1,
                    "gpu_hour_accounting": "allocated_wall_clock",
                    "validation_accuracy": score - 0.1 if selected_final else trial_index / 10,
                    "config_path": str(config_path),
                    "run_dir": str(run_dir) if selected_final else str(run_dir / rung / trial_name),
                    "exit_code": 0,
                })
        metrics = {
            key: value
            for key, value in zip(METRIC_COLUMNS, (
                score - 0.02, score - 0.03, score - 0.04, 1.02 - score,
                score - 0.06, score - 0.06, score - 0.07, score - 0.07,
                score - 0.05, 0.05,
            ))
        }
        fixed_rows.append({
            "family": "CNN",
            "architecture": "ResNet-50",
            "architecture_slug": "resnet50",
            "seed": 42,
            "method": FIXED_DISPLAY_NAMES[method],
            **metrics,
        })
    # Failed work must be retained in realized search cost.
    failed_attempt = {
        **records[0],
        "job_id": "resnet50/fcal_v2/trial-00/rung-01",
        "attempt": 1,
        "status": "failure",
        "gpu_hours": 0.25,
        "elapsed_seconds": 900.0,
        "exit_code": 1,
        "error": "fixture failure",
    }
    records[0]["attempt"] = 2
    records.insert(0, failed_attempt)
    frozen_path.write_text(yaml.safe_dump({
        "study_id": "fixture-study",
        "selection_split": "validation",
        "test_set_used": False,
        "winners": winners,
    }), encoding="utf-8")
    ledger_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(fixed_rows).to_csv(fixed_path, index=False)
    return frozen_path, ledger_path, manifest_path, fixed_path


def test_attempt_accounting_includes_failures_and_marks_selected_trial() -> None:
    winners = [{"architecture": "resnet50", "method": "fcal_v2", "trial_id": "trial-00"}]
    records = [
        {
            "study_id": "s", "job_id": "resnet50/fcal_v2/trial-00/rung-03",
            "attempt": 1, "status": "success", "gpu_hours": 1.0,
            "elapsed_seconds": 3600, "gpu_hour_accounting": "allocated_wall_clock",
        },
        {
            "study_id": "s", "job_id": "resnet50/fcal_v2/trial-04/rung-01",
            "attempt": 1, "status": "failure", "gpu_hours": 0.5,
            "elapsed_seconds": 1800, "gpu_hour_accounting": "allocated_wall_clock",
        },
    ]
    attempts = tuning_attempt_table(records, winners)
    report = tuning_accounting(attempts)
    study = report.loc[report["scope"].eq("study")].iloc[0]
    assert study["gpu_hours_including_failures"] == 1.5
    assert study["failed_gpu_hours"] == 0.5
    assert attempts.loc[attempts["selected_trial"], "trial_id"].tolist() == ["trial-00"]


def test_build_writes_post_freeze_tables_and_excludes_isnet_from_wtl(tmp_path: Path) -> None:
    frozen, ledger, manifest, fixed = _fixture(tmp_path)
    outputs = build(frozen, ledger, manifest, fixed, tmp_path / "report")
    assert len(outputs) == 7
    assert all(path.is_file() for path in outputs)
    selected = pd.read_csv(outputs[0])
    deltas = pd.read_csv(outputs[4])
    fixed_tuned_wtl = pd.read_csv(outputs[5])
    tuned_wtl = pd.read_csv(outputs[6])
    assert len(selected) == 5
    assert selected["selection_split"].eq("validation").all()
    assert len(deltas.loc[deltas["method"].eq("Faster-ISNet")]) == len(METRIC_COLUMNS) - 2
    assert "Faster-ISNet" not in set(fixed_tuned_wtl["method"])
    assert "Faster-ISNet" not in set(tuned_wtl["candidate"])
    assert fixed_tuned_wtl["architectures"].eq(1).all()


def test_build_rejects_frozen_manifest_that_claims_test_was_used(tmp_path: Path) -> None:
    frozen, ledger, manifest, fixed = _fixture(tmp_path)
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    payload["test_set_used"] = True
    frozen.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="without test-set use"):
        build(frozen, ledger, manifest, fixed, tmp_path / "report")
