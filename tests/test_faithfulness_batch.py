from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fcal_study.baseline_fixed_results import ARCHITECTURES
from fcal_study.faithfulness import PROTOCOL_VERSION, summarize
from fcal_study.faithfulness_batch import (
    ARCHITECTURE_MODEL_NAMES,
    BATCH_PROTOCOL_VERSION,
    _artifact_state,
    aggregate,
    build_manifest,
    discover_fixed_jobs,
    discover_tuned_jobs,
    job_states,
    load_batch_manifest,
    run_job,
    write_manifest,
)


def _checkpoint(root: Path, run_name: str, method: str, architecture: str) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        "method": method,
        "model": {"name": ARCHITECTURE_MODEL_NAMES[architecture]},
    }), encoding="utf-8")
    path = run_dir / "best-validation.pt"
    path.write_bytes(f"{run_name}-selected".encode())
    return path


def _fixed_matrix(archive: Path, current: Path) -> None:
    specs = (
        (archive, "ce", "ce"),
        (archive, "fcal-v5", "fcal_v5"),
        (current, "camal", "camal"),
        (current, "gain-ext", "gain"),
        (current, "rrr", "rrr"),
    )
    for architecture, _, _ in ARCHITECTURES:
        for root, suffix, method in specs:
            _checkpoint(root, f"{architecture}-{suffix}-seed42", method, architecture)
    _checkpoint(current, "resnet50-isnet-seed42", "isnet", "resnet50")


def _payload() -> dict[str, np.ndarray]:
    return {
        "image_id": np.array([10, 11], dtype=np.int64),
        "label": np.array([0, 1], dtype=np.int64),
        "mask_valid": np.array([True, True]),
        "attention_iou": np.array([0.25, 0.75], dtype=np.float32),
        "clean_target_confidence": np.array([0.8, 0.6], dtype=np.float32),
        "baseline_target_confidence": np.array([0.1, 0.2], dtype=np.float32),
        "insertion_auc_normalized": np.array([0.7, 0.5], dtype=np.float32),
        "removal_auc_normalized": np.array([0.2, 0.4], dtype=np.float32),
    }


def _write_artifact(manifest: dict, job: dict, *, changed_hash: bool = False) -> Path:
    output = Path(job["artifact_dir"])
    if not output.is_absolute():
        from fcal_study.config import PROJECT_ROOT
        output = PROJECT_ROOT / output
    output.mkdir(parents=True)
    payload = _payload()
    np.savez_compressed(output / "per_image.npz", **payload)
    summary = summarize(payload)
    binding = {
        "protocol_version": PROTOCOL_VERSION,
        "split": manifest["split"],
        "checkpoint_sha256": "changed" if changed_hash else job["checkpoint_sha256"],
        "requested_max_images": manifest["selection"]["max_images"],
        "selection_seed": manifest["selection"]["selection_seed"],
    }
    (output / "protocol.json").write_text(json.dumps(binding), encoding="utf-8")
    summary.update(binding, method=job["training_method"])
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return output


def test_fixed_discovery_is_exact_complete_matrix_and_labels_isnet(tmp_path: Path):
    archive = tmp_path / "archive"
    current = tmp_path / "current"
    _fixed_matrix(archive, current)
    jobs = discover_fixed_jobs(archive, current, tmp_path / "artifacts")
    assert len(jobs) == 31
    assert [job["job_id"] for job in jobs] == sorted(job["job_id"] for job in jobs)
    isnet = [job for job in jobs if job["training_method"] == "isnet"]
    assert len(isnet) == 1
    assert isnet[0]["support_scope"] == "resnet50_only"
    assert isnet[0]["architecture_wtl_eligible"] is False
    assert {job["architecture"] for job in jobs if job["method"] == "FCAL V2"} == {
        architecture for architecture, _, _ in ARCHITECTURES
    }


def test_tuned_discovery_requires_one_validation_only_winner_per_supported_cell(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path, "winner", "fcal_v2", "resnet50")
    tuning = tmp_path / "tuning.yaml"
    tuning.write_text(yaml.safe_dump({
        "study_id": "study",
        "architectures": {"resnet50": {}},
        "methods": {"fcal_v2": {
            "display_name": "FCAL V2",
            "coefficient": "fcal_v2.support_weight",
            "values": [0.05],
            "supported_architectures": ["resnet50"],
        }},
    }), encoding="utf-8")
    frozen = tmp_path / "frozen.yaml"
    frozen_payload = {
        "study_id": "study",
        "selection_split": "validation",
        "test_set_used": False,
        "winners": [{
            "architecture": "resnet50",
            "method": "fcal_v2",
            "trial_id": "trial-00",
            "coefficient": "fcal_v2.support_weight",
            "value": 0.05,
            "validation_accuracy": 0.8,
            "checkpoint": str(checkpoint),
        }],
    }
    frozen.write_text(yaml.safe_dump(frozen_payload), encoding="utf-8")
    jobs, sources = discover_tuned_jobs(tuning, frozen, tmp_path / "artifacts")
    assert len(jobs) == 1
    assert jobs[0]["regime"] == "tuned"
    assert jobs[0]["tuning_selection"]["validation_accuracy"] == 0.8
    assert sources["frozen_test_set_used"] is False
    frozen_payload["winners"] = []
    frozen.write_text(yaml.safe_dump(frozen_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        discover_tuned_jobs(tuning, frozen, tmp_path / "artifacts")


def test_test_manifest_stays_sealed_without_complete_frozen_winners(tmp_path: Path):
    with pytest.raises(ValueError, match="remains sealed"):
        build_manifest(
            split="test",
            archive_root=tmp_path / "archive",
            current_root=tmp_path / "fixed",
            artifact_root=tmp_path / "artifacts",
            max_images=500,
        )


def test_manifest_is_write_once_but_exact_recreation_is_idempotent(tmp_path: Path):
    manifest = {
        "batch_protocol_version": BATCH_PROTOCOL_VERSION,
        "evaluator_protocol_version": PROTOCOL_VERSION,
        "split": "validation",
        "test_unsealed": False,
        "selection": {"max_images": 2, "selection_seed": 2026, "batch_size": 1},
        "fixed_sources": {},
        "tuning_freeze": None,
        "jobs": [{"job_id": "fixed/a/b"}],
    }
    path = tmp_path / "manifest.json"
    assert write_manifest(path, manifest) == path
    assert write_manifest(path, manifest) == path
    changed = dict(manifest, split="test")
    with pytest.raises(FileExistsError, match="different"):
        write_manifest(path, changed)


def test_resume_recognizes_exact_complete_artifact_and_rejects_binding_change(
    tmp_path: Path, monkeypatch,
):
    checkpoint = _checkpoint(tmp_path, "selected", "ce", "resnet50")
    job = {
        "job_id": "fixed_default/resnet50/ce",
        "regime": "fixed_default",
        "architecture": "resnet50",
        "method": "CE",
        "training_method": "ce",
        "support_scope": "six_architecture",
        "architecture_wtl_eligible": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "artifact_dir": str(tmp_path / "sealed"),
    }
    manifest = {
        "batch_protocol_version": BATCH_PROTOCOL_VERSION,
        "evaluator_protocol_version": PROTOCOL_VERSION,
        "split": "validation",
        "test_unsealed": False,
        "selection": {"max_images": 2, "selection_seed": 2026, "batch_size": 1},
        "fixed_sources": {},
        "tuning_freeze": None,
        "jobs": [job],
    }
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    output = _write_artifact(manifest, job)
    assert _artifact_state(manifest, job) == ("complete", None)
    monkeypatch.setattr("fcal_study.faithfulness_batch.run", lambda *args, **kwargs: pytest.fail("reran"))
    assert run_job(manifest_path, job["job_id"]) == output
    assert job_states(load_batch_manifest(manifest_path))[0]["state"] == "complete"

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["checkpoint_sha256"] = "changed"
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    state, error = _artifact_state(manifest, job)
    assert state == "invalid"
    assert "checkpoint_sha256" in error
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_job(manifest_path, job["job_id"])


def test_aggregation_writes_metrics_and_support_labels_once(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path, "selected", "isnet", "resnet50")
    job = {
        "job_id": "fixed_default/resnet50/selective-faster-isnet",
        "regime": "fixed_default",
        "architecture": "resnet50",
        "method": "Selective Faster-ISNet",
        "training_method": "isnet",
        "support_scope": "resnet50_only",
        "architecture_wtl_eligible": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "artifact_dir": str(tmp_path / "artifact"),
    }
    manifest = {
        "batch_protocol_version": BATCH_PROTOCOL_VERSION,
        "evaluator_protocol_version": PROTOCOL_VERSION,
        "split": "validation",
        "test_unsealed": False,
        "selection": {"max_images": 2, "selection_seed": 2026, "batch_size": 1},
        "fixed_sources": {},
        "tuning_freeze": None,
        "jobs": [job],
    }
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    _write_artifact(manifest, job)
    csv_path, report_path = aggregate(manifest_path, tmp_path / "report")
    row = pd.read_csv(csv_path).iloc[0]
    assert row["mask_iou_mean"] == pytest.approx(0.5)
    assert row["insertion_auc_mean"] == pytest.approx(0.6)
    assert row["removal_auc_mean"] == pytest.approx(0.3)
    assert row["support_scope"] == "resnet50_only"
    assert "excluded from six-architecture W/T/L" in report_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        aggregate(manifest_path, tmp_path / "report")
