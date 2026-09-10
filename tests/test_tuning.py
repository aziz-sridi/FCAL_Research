from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from fcal_study.config import PROJECT_ROOT
from fcal_study.tuning import (
    DEFAULT_MANIFEST,
    accounting,
    all_jobs,
    assert_validation_only_config,
    build_trial_config,
    dry_queue,
    format_status_text,
    load_manifest,
    main,
    pending_jobs,
    required_jobs_by_rung,
    study_status,
    tuning_status,
    validate_manifest,
)


def _success(job, score: float) -> dict:
    return {
        "job_id": job.job_id,
        "status": "success",
        "validation_accuracy": score,
        "gpu_hours": 0.25,
    }


def test_manifest_has_equal_initial_budget_and_support_aware_cells() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    jobs = all_jobs(manifest)
    initial = [job for job in jobs if job.rung_index == 0]
    counts = {}
    for job in initial:
        counts[job.cell] = counts.get(job.cell, 0) + 1
    assert len(counts) == 25
    assert set(counts.values()) == {5}
    assert {
        job.architecture for job in initial
        if job.method == "selective_faster_isnet"
    } == {"resnet50"}
    assert len(initial) == 125


def test_fcal_v2_templates_and_archived_public_v2_prerequisites_are_mapped() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    method = manifest["methods"]["fcal_v2"]
    expected_templates = {
        "resnet50": "configs/cub/resnet50_fcal_v2.yaml",
        "densenet121": "configs/cub/densenet121_fcal_v2.yaml",
        "efficientnet-b0": "configs/cub/efficientnet_b0_fcal_v2.yaml",
        "vit-tiny-p16-avg": "configs/cub/vit_tiny_avg_fcal_v2.yaml",
        "vit-small-p16-avg": "configs/cub/vit_small_avg_fcal_v2.yaml",
        "vit-base-p16-avg": "configs/cub/vit_base_avg_fcal_v2.yaml",
    }
    assert method["template_by_architecture"] == expected_templates
    for architecture, template in expected_templates.items():
        assert (PROJECT_ROOT / template).is_file()
        summary = method["fixed_baseline_summary_by_architecture"][architecture]
        assert summary == (
            "../archive/legacy_projects/adaptive_fcal_study/outputs/"
            f"{architecture}-fcal-v5-seed42/summary.json"
        )
        assert (PROJECT_ROOT / summary).resolve().is_file()


def test_trial_configs_share_selection_rules_and_remove_test_cache() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    initial = [job for job in all_jobs(manifest) if job.rung_index == 0]
    for job in initial:
        config = build_trial_config(manifest, job)
        assert config["seed"] == 42
        assert config["training"]["epochs"] == 10
        assert config["tuning"]["selection_split"] == "validation"
        assert config["tuning"]["checkpoint_rule"] == "best-validation.pt"
        assert "evaluation_cache" not in config["data"]
        assert "test" not in config["run_id"].lower()


def test_global_barrier_and_deterministic_five_to_two_promotion() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    initial = pending_jobs(manifest, [])
    incomplete = [_success(job, float(job.trial_index)) for job in initial[:-1]]
    assert pending_jobs(manifest, incomplete) == [initial[-1]]

    records = [_success(job, float(job.trial_index)) for job in initial]
    required = required_jobs_by_rung(manifest, records)
    assert len(required) == 2
    promoted = required[1]
    assert len(promoted) == 50
    for cell in {job.cell for job in promoted}:
        assert {job.trial_index for job in promoted if job.cell == cell} == {3, 4}
        assert {job.total_epochs for job in promoted if job.cell == cell} == {25}


def test_failure_attempts_are_terminal_only_at_declared_retry_limit() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    first = pending_jobs(manifest, [])[0]
    one_failure = [{"job_id": first.job_id, "status": "failure", "gpu_hours": 0.1}]
    assert first in pending_jobs(manifest, one_failure)
    two_failures = one_failure + [
        {"job_id": first.job_id, "status": "failure", "gpu_hours": 0.2},
    ]
    assert first not in pending_jobs(manifest, two_failures)


def test_gpu_accounting_includes_failures() -> None:
    report = accounting([
        {"status": "success", "gpu_hours": 1.25},
        {"status": "failure", "gpu_hours": 0.50},
    ])
    assert report == {
        "attempts": 2,
        "successful_attempts": 1,
        "failed_attempts": 1,
        "gpu_hours_including_failures": 1.75,
        "failed_gpu_hours": 0.50,
    }


def test_leakage_guard_rejects_test_cache_and_test_namespace() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    job = pending_jobs(manifest, [])[0]
    config = build_trial_config(manifest, job)
    with_cache = copy.deepcopy(config)
    with_cache["data"]["evaluation_cache"] = "cache/cub_test"
    with pytest.raises(ValueError, match="caches are forbidden"):
        assert_validation_only_config(with_cache)
    test_output = copy.deepcopy(config)
    test_output["run_id"] = "test/forbidden"
    with pytest.raises(ValueError, match="validation-only namespace"):
        assert_validation_only_config(test_output)


def test_dry_queue_is_non_evaluating_and_does_not_write_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "absent.jsonl"
    commands = dry_queue(DEFAULT_MANIFEST, ledger, output_format="jsonl")
    assert len(commands) == 125
    first = json.loads(commands[0])
    assert first["dry_run"] is True
    assert "--execute" in first["command"]
    assert "fcal_study.evaluate" not in " ".join(first["command"])
    assert not ledger.exists()


def test_manifest_rejects_any_test_unseal_shortcut() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    invalid = copy.deepcopy(manifest)
    invalid["test_seal"]["unseal_after"] = "first_rung"
    with pytest.raises(ValueError, match="unsealed only after"):
        validate_manifest(invalid)


def test_read_only_status_reports_empty_study_conservatively(tmp_path: Path) -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    frozen = tmp_path / "absent-frozen.yaml"
    status = study_status(manifest, [], frozen_winners_path=frozen)
    assert status["read_only"] is True
    assert status["declared_support_cells"] == 25
    assert status["attempts"]["attempts"] == 0
    assert status["eligible_queued_jobs"]["count"] == 125
    assert status["barrier"]["cells_awaiting_other_cells"] == []
    assert status["conservative_progress"] == {
        "completed_successful_jobs": 0,
        "total_planned_job_slots": 200,
        "fraction": 0.0,
        "percent": 0.0,
        "rule": "distinct recognized successful jobs only; failures and retries add zero",
    }
    assert [row["planned_job_slots"] for row in status["completed_jobs_by_rung"]] == [
        125, 50, 25,
    ]
    assert status["completed_jobs_by_rung"][0]["reached"] is True
    assert status["completed_jobs_by_rung"][1]["reached"] is False
    assert status["frozen"]["exists"] is False
    assert not frozen.exists()


def test_status_reports_attempts_barrier_waiters_and_gpu_hours(tmp_path: Path) -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    initial = pending_jobs(manifest, [])
    first_cell = initial[0].cell
    completed_cell = [job for job in initial if job.cell == first_cell]
    records = [_success(job, float(job.trial_index)) for job in completed_cell]
    retryable = next(job for job in initial if job.cell != first_cell)
    records.append({
        "job_id": retryable.job_id,
        "status": "failure",
        "gpu_hours": 0.5,
    })
    records.append({
        "job_id": "unknown/method/trial-00/rung-01",
        "status": "failure",
        "gpu_hours": 0.75,
    })
    status = study_status(
        manifest, records, frozen_winners_path=tmp_path / "absent.yaml",
    )
    assert status["attempts"] == {
        "attempts": 7,
        "successful_attempts": 5,
        "failed_attempts": 2,
        "gpu_hours_including_failures": 2.5,
        "failed_gpu_hours": 1.25,
    }
    assert status["unrecognized_job_attempts"] == 1
    assert status["eligible_queued_jobs"]["count"] == 120
    assert status["barrier"]["cells_awaiting_other_cells"] == [
        f"{first_cell[0]}/{first_cell[1]}"
    ]
    assert status["conservative_progress"]["completed_successful_jobs"] == 5
    text = format_status_text(status)
    assert "Progress: 5/200" in text
    assert "Barrier-waiting cells: 1" in text


def test_status_advances_rungs_and_validates_frozen_state(tmp_path: Path) -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    records = [_success(job, float(job.trial_index)) for job in pending_jobs(manifest, [])]
    second = pending_jobs(manifest, records)
    assert len(second) == 50
    records.extend(_success(job, float(job.trial_index)) for job in second)
    final = pending_jobs(manifest, records)
    assert len(final) == 25
    records.extend(_success(job, float(job.trial_index)) for job in final)
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(yaml.safe_dump({
        "study_id": manifest["study_id"],
        "selection_split": "validation",
        "test_set_used": False,
        "winners": [
            {"architecture": architecture, "method": method}
            for architecture, method in {
                job.cell for job in final
            }
        ],
    }), encoding="utf-8")
    status = study_status(manifest, records, frozen_winners_path=frozen)
    assert status["freeze_eligible"] is True
    assert status["eligible_queued_jobs"]["count"] == 0
    assert status["conservative_progress"]["completed_successful_jobs"] == 200
    assert [row["successful_jobs"] for row in status["completed_jobs_by_rung"]] == [
        125, 50, 25,
    ]
    assert status["frozen"]["valid"] is True
    assert status["frozen"]["winner_count"] == 25
    assert status["frozen"]["consistent_with_ledger"] is True


def test_status_reports_terminal_promotion_failure_without_crashing(tmp_path: Path) -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    initial = pending_jobs(manifest, [])
    first_cell = initial[0].cell
    records = []
    for job in initial:
        if job.cell == first_cell and job.trial_index > 0:
            records.extend([
                {"job_id": job.job_id, "status": "failure", "gpu_hours": 0.1},
                {"job_id": job.job_id, "status": "failure", "gpu_hours": 0.1},
            ])
        else:
            records.append(_success(job, float(job.trial_index)))
    status = study_status(
        manifest, records, frozen_winners_path=tmp_path / "absent.yaml",
    )
    assert status["barrier"]["blocked"] is True
    assert "needs 2 promotions" in status["barrier"]["error"]
    assert status["eligible_queued_jobs"]["count"] == 0
    assert status["freeze_eligible"] is False


def test_status_cli_is_json_and_does_not_modify_ledger(tmp_path: Path, monkeypatch, capsys) -> None:
    ledger = tmp_path / "attempts.jsonl"
    ledger.write_text("", encoding="utf-8")
    before = ledger.read_bytes()
    frozen = tmp_path / "absent.yaml"
    monkeypatch.setattr(sys, "argv", [
        "tuning", "status", str(DEFAULT_MANIFEST),
        "--ledger", str(ledger), "--frozen-winners", str(frozen),
    ])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["read_only"] is True
    assert payload["eligible_queued_jobs"]["count"] == 125
    assert ledger.read_bytes() == before
    assert not frozen.exists()
    assert tuning_status(DEFAULT_MANIFEST, ledger, frozen)["read_only"] is True
