from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import yaml

from .config import PROJECT_ROOT, load_config

DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "cub" / "tuning_validation.yaml"
DEFAULT_LEDGER = PROJECT_ROOT / "outputs" / "tuning-validation" / "attempts.jsonl"
DEFAULT_FROZEN_WINNERS = PROJECT_ROOT / "outputs" / "tuning-validation" / "frozen-winners.yaml"


@dataclass(frozen=True)
class Job:
    architecture: str
    method: str
    trial_index: int
    value: float
    rung_index: int
    total_epochs: int

    @property
    def cell(self) -> tuple[str, str]:
        return self.architecture, self.method

    @property
    def trial_id(self) -> str:
        return f"trial-{self.trial_index:02d}"

    @property
    def rung_id(self) -> str:
        return f"rung-{self.rung_index + 1:02d}"

    @property
    def job_id(self) -> str:
        return f"{self.architecture}/{self.method}/{self.trial_id}/{self.rung_id}"


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    manifest_path = Path(path).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    validate_manifest(manifest)
    return manifest


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _coefficient_values(method: dict) -> list[float]:
    return [float(value) for value in method["values"]]


def _supported_cells(manifest: dict) -> list[tuple[str, str]]:
    cells = []
    for architecture in manifest["architectures"]:
        for method_name, method in manifest["methods"].items():
            if architecture in method["supported_architectures"]:
                cells.append((architecture, method_name))
    return cells


def validate_manifest(manifest: dict) -> None:
    selection = manifest.get("selection", {})
    if selection != {
        "split": "validation",
        "metric": "accuracy",
        "checkpoint_rule": "best-validation.pt",
        "seed": 42,
    }:
        raise ValueError("Tuning must use seed 42 validation accuracy and best-validation.pt")

    compute = manifest.get("compute", {})
    if compute.get("gpus_per_job") != 1 or compute.get("max_concurrent_jobs") != 1:
        raise ValueError("The RTX 5070 Ti protocol requires exactly one sequential GPU job")
    if compute.get("gpu_hour_accounting") != "allocated_wall_clock":
        raise ValueError("GPU hours must use the declared allocated-wall-clock accounting rule")

    halving = manifest.get("successive_halving", {})
    initial_trials = int(halving.get("initial_trials", 0))
    if initial_trials < 2:
        raise ValueError("Successive halving requires at least two initial trials")
    rungs = halving.get("rungs", [])
    if len(rungs) < 2:
        raise ValueError("At least two fidelity rungs are required")
    epochs = [int(rung["total_epochs"]) for rung in rungs]
    survivors = [int(rung["survivors"]) for rung in rungs]
    if epochs != sorted(set(epochs)):
        raise ValueError("Fidelity epochs must be strictly increasing")
    if survivors[0] >= initial_trials or survivors[-1] != 1:
        raise ValueError("The first rung must reduce trials and the final rung must retain one")
    if any(right > left for left, right in pairwise(survivors)):
        raise ValueError("Survivor counts cannot increase")

    architectures = set(manifest.get("architectures", {}))
    if not architectures:
        raise ValueError("No architectures declared")
    methods = manifest.get("methods", {})
    if set(methods) != {"fcal_v2", "rrr", "gain_ext", "camal", "selective_faster_isnet"}:
        raise ValueError("The tuning arm must contain exactly the five predeclared methods")
    for method_name, method in methods.items():
        supported = set(method.get("supported_architectures", []))
        if not supported or not supported <= architectures:
            raise ValueError(f"Invalid support declaration for {method_name}")
        if len(_coefficient_values(method)) != initial_trials:
            raise ValueError(f"{method_name} does not have {initial_trials} initial trials")
        if len(set(_coefficient_values(method))) != initial_trials:
            raise ValueError(f"{method_name} contains duplicate coefficient values")
    isnet_support = set(methods["selective_faster_isnet"]["supported_architectures"])
    if isnet_support != {"resnet50"}:
        raise ValueError("Selective Faster-ISNet is audited only for ResNet-50")

    seal = manifest.get("test_seal", {})
    if seal.get("unseal_after") != "one_winner_per_supported_method_architecture_frozen":
        raise ValueError("The test set may be unsealed only after all supported winners are frozen")
    if seal.get("evaluation_during_tuning") is not False:
        raise ValueError("Evaluation during tuning must be disabled")


def all_jobs(manifest: dict) -> list[Job]:
    jobs = []
    for architecture, method_name in _supported_cells(manifest):
        values = _coefficient_values(manifest["methods"][method_name])
        for trial_index, value in enumerate(values):
            for rung_index, rung in enumerate(manifest["successive_halving"]["rungs"]):
                jobs.append(Job(
                    architecture=architecture,
                    method=method_name,
                    trial_index=trial_index,
                    value=value,
                    rung_index=rung_index,
                    total_epochs=int(rung["total_epochs"]),
                ))
    return jobs


def _read_ledger(path: str | Path) -> list[dict]:
    ledger_path = Path(path)
    if not ledger_path.is_file():
        return []
    records = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid ledger JSON at line {line_number}: {ledger_path}") from error
        records.append(record)
    return records


def _attempts_by_job(records: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["job_id"]), []).append(record)
    return grouped


def _terminal(job: Job, attempts: dict[str, list[dict]], max_attempts: int) -> bool:
    records = attempts.get(job.job_id, [])
    return any(record.get("status") == "success" for record in records) or len(records) >= max_attempts


def _success(job: Job, attempts: dict[str, list[dict]]) -> dict | None:
    successes = [
        record for record in attempts.get(job.job_id, [])
        if record.get("status") == "success"
    ]
    return successes[-1] if successes else None


def _initial_jobs(manifest: dict) -> list[Job]:
    return [job for job in all_jobs(manifest) if job.rung_index == 0]


def _promote(
    previous: list[Job],
    next_rung_index: int,
    manifest: dict,
    attempts: dict[str, list[dict]],
) -> list[Job]:
    survivors = int(manifest["successive_halving"]["rungs"][next_rung_index - 1]["survivors"])
    next_epochs = int(manifest["successive_halving"]["rungs"][next_rung_index]["total_epochs"])
    promoted = []
    for cell in _supported_cells(manifest):
        candidates = [job for job in previous if job.cell == cell]
        ranked = []
        for job in candidates:
            record = _success(job, attempts)
            if record is not None:
                ranked.append((float(record["validation_accuracy"]), -job.trial_index, job))
        ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
        if len(ranked) < survivors:
            raise RuntimeError(
                f"{cell} has {len(ranked)} successful jobs but needs {survivors} promotions"
            )
        for _, _, old in ranked[:survivors]:
            promoted.append(Job(
                architecture=old.architecture,
                method=old.method,
                trial_index=old.trial_index,
                value=old.value,
                rung_index=next_rung_index,
                total_epochs=next_epochs,
            ))
    return promoted


def required_jobs_by_rung(manifest: dict, records: Iterable[dict]) -> list[list[Job]]:
    attempts = _attempts_by_job(records)
    max_attempts = int(manifest["successive_halving"]["max_attempts_per_job"])
    required = [_initial_jobs(manifest)]
    for rung_index in range(1, len(manifest["successive_halving"]["rungs"])):
        previous = required[-1]
        if not all(_terminal(job, attempts, max_attempts) for job in previous):
            break
        required.append(_promote(previous, rung_index, manifest, attempts))
    return required


def pending_jobs(manifest: dict, records: Iterable[dict]) -> list[Job]:
    records = list(records)
    attempts = _attempts_by_job(records)
    max_attempts = int(manifest["successive_halving"]["max_attempts_per_job"])
    required = required_jobs_by_rung(manifest, records)
    current = required[-1]
    if all(_terminal(job, attempts, max_attempts) for job in current):
        if len(required) == len(manifest["successive_halving"]["rungs"]):
            return []
        raise RuntimeError("A completed rung could not advance")
    return [job for job in current if not _terminal(job, attempts, max_attempts)]


def _nested_set(config: dict, dotted_path: str, value: float) -> None:
    parts = dotted_path.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _method_template(manifest: dict, job: Job) -> Path:
    method = manifest["methods"][job.method]
    template = method.get("template_by_architecture", {}).get(
        job.architecture, method.get("template"),
    )
    if template is None:
        raise ValueError(f"No template for {job.method}/{job.architecture}")
    template = template.format(architecture=job.architecture)
    return _resolve(template)


def _run_id(job: Job, attempt_number: int) -> str:
    return "/".join((
        "tuning-validation", "cub", job.architecture, job.method,
        job.trial_id, job.rung_id, f"attempt-{attempt_number:02d}",
    ))


def build_trial_config(manifest: dict, job: Job, attempt_number: int = 1) -> dict:
    config = copy.deepcopy(load_config(_method_template(manifest, job)))
    config.pop("_config_path", None)
    # The cache is named for and contains test examples.  It is irrelevant to
    # train/validation and must not be present in a tuning configuration.
    config["data"].pop("evaluation_cache", None)
    config["run_id"] = _run_id(job, attempt_number)
    config["seed"] = int(manifest["selection"]["seed"])
    config["training"]["epochs"] = job.total_epochs
    config["training"]["save_epoch5_branch"] = False
    _nested_set(config, manifest["methods"][job.method]["coefficient"], job.value)
    config["tuning"] = {
        "study_id": manifest["study_id"],
        "job_id": job.job_id,
        "selection_split": "validation",
        "selection_metric": "accuracy",
        "checkpoint_rule": "best-validation.pt",
        "promotion_restart": "fixed_parent_epoch_5",
    }
    assert_validation_only_config(config)
    return config


def assert_validation_only_config(config: dict) -> None:
    if config.get("seed") != 42:
        raise ValueError("Every tuning job must use seed 42")
    tuning = config.get("tuning", {})
    if tuning.get("selection_split") != "validation":
        raise ValueError("Only the validation split may select tuning jobs")
    if tuning.get("checkpoint_rule") != "best-validation.pt":
        raise ValueError("Unexpected checkpoint selection rule")
    if "evaluation_cache" in config.get("data", {}):
        raise ValueError("Evaluation/test caches are forbidden in tuning configs")
    run_id = str(config.get("run_id", "")).lower()
    if "test" in run_id or not run_id.startswith("tuning-validation/"):
        raise ValueError("Tuning output must remain in the validation-only namespace")


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _parent_checkpoint(manifest: dict, job: Job) -> Path:
    value = manifest["architectures"][job.architecture]["fixed_parent_checkpoint"]
    return _resolve(value)


def _parent_split(manifest: dict) -> Path:
    return _resolve(manifest["fixed_parent_split"])


def _baseline_summary(manifest: dict, job: Job) -> Path:
    method = manifest["methods"][job.method]
    pattern = method.get("fixed_baseline_summary_by_architecture", {}).get(
        job.architecture, method.get("fixed_baseline_summary"),
    )
    if pattern is None:
        raise ValueError(f"No fixed baseline prerequisite for {job.method}/{job.architecture}")
    return _resolve(pattern.format(architecture=job.architecture))


def training_argv(manifest_path: Path, ledger_path: Path, job: Job) -> list[str]:
    return [
        sys.executable, "-m", "fcal_study.tuning", "run",
        str(manifest_path), "--ledger", str(ledger_path),
        "--job-id", job.job_id, "--execute",
    ]


def _display_command(argv: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def dry_queue(manifest_path: Path, ledger_path: Path, output_format: str = "text") -> list[str]:
    manifest = load_manifest(manifest_path)
    records = _read_ledger(ledger_path)
    commands = []
    for job in pending_jobs(manifest, records):
        argv = training_argv(manifest_path.resolve(), ledger_path.resolve(), job)
        if output_format == "jsonl":
            commands.append(json.dumps({
                "job_id": job.job_id,
                "command": argv,
                "dry_run": True,
            }, sort_keys=True))
        else:
            commands.append(_display_command(argv))
    return commands


def _append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_job(manifest_path: Path, ledger_path: Path, job_id: str) -> dict:
    manifest = load_manifest(manifest_path)
    records = _read_ledger(ledger_path)
    eligible = {job.job_id: job for job in pending_jobs(manifest, records)}
    if job_id not in eligible:
        raise ValueError(f"Job is not currently eligible: {job_id}")
    job = eligible[job_id]

    baseline = _baseline_summary(manifest, job)
    parent = _parent_checkpoint(manifest, job)
    parent_split = _parent_split(manifest)
    missing = [path for path in (baseline, parent, parent_split) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Tuning starts only after its fixed baseline and frozen parent are available: "
            + ", ".join(str(path) for path in missing)
        )

    attempts = _attempts_by_job(records).get(job.job_id, [])
    attempt_number = len(attempts) + 1
    config = build_trial_config(manifest, job, attempt_number)
    config_path = (
        PROJECT_ROOT / "outputs" / "tuning-validation" / "_configs"
        / job.architecture / job.method / job.trial_id / job.rung_id
        / f"attempt-{attempt_number:02d}.yaml"
    )
    _write_yaml(config_path, config)
    run_dir = _resolve(config["output_root"]) / config["run_id"]
    argv = [
        sys.executable, "-m", "fcal_study.train", str(config_path),
        "--resume", str(parent), "--allow-data-relocation",
        "--parent-split-dir", str(parent_split),
    ]
    if "fcal_study.evaluate" in " ".join(argv) or "--split test" in " ".join(argv):
        raise RuntimeError("Test evaluation command rejected by tuning guard")

    started_utc = _utc_now()
    started = time.perf_counter()
    status = "failure"
    exit_code: int | None = None
    error: str | None = None
    validation_accuracy: float | None = None
    try:
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, check=False)
        exit_code = completed.returncode
        summary_path = run_dir / "summary.json"
        if exit_code != 0:
            error = f"training process exited with code {exit_code}"
        elif not summary_path.is_file():
            error = f"training succeeded without summary: {summary_path}"
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validation_accuracy = float(summary["best_validation_accuracy"])
            if not math.isfinite(validation_accuracy):
                error = "non-finite best_validation_accuracy"
            else:
                expected = run_dir / manifest["selection"]["checkpoint_rule"]
                if not expected.is_file():
                    error = f"missing selected validation checkpoint: {expected}"
                else:
                    status = "success"
    except (OSError, ValueError, KeyError) as caught:
        error = f"{type(caught).__name__}: {caught}"
    elapsed = time.perf_counter() - started
    record = {
        "study_id": manifest["study_id"],
        "job_id": job.job_id,
        "attempt": attempt_number,
        "status": status,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        # One GPU is exclusively allocated to every launched job.  This is
        # intentionally recorded for failed jobs as well as successful jobs.
        "gpu_hours": elapsed * int(manifest["compute"]["gpus_per_job"]) / 3600.0,
        "gpu_hour_accounting": "allocated_wall_clock",
        "exit_code": exit_code,
        "error": error,
        "validation_accuracy": validation_accuracy,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "run_dir": str(run_dir),
        "command": argv,
    }
    _append_ledger(ledger_path, record)
    if status != "success":
        raise RuntimeError(error or "tuning job failed")
    return record


def frozen_winners(manifest: dict, records: Iterable[dict]) -> dict:
    records = list(records)
    attempts = _attempts_by_job(records)
    required = required_jobs_by_rung(manifest, records)
    if len(required) != len(manifest["successive_halving"]["rungs"]):
        raise RuntimeError("All fidelity rungs must complete before winners can be frozen")
    final_jobs = required[-1]
    if any(_success(job, attempts) is None for job in final_jobs):
        raise RuntimeError("Every supported method/architecture needs a successful final rung")
    winners = []
    for job in sorted(final_jobs, key=lambda item: item.cell):
        record = _success(job, attempts)
        winners.append({
            "architecture": job.architecture,
            "method": job.method,
            "trial_id": job.trial_id,
            "coefficient": manifest["methods"][job.method]["coefficient"],
            "value": job.value,
            "validation_accuracy": record["validation_accuracy"],
            "checkpoint": str(Path(record["run_dir"]) / "best-validation.pt"),
        })
    return {
        "study_id": manifest["study_id"],
        "frozen_utc": _utc_now(),
        "selection_split": "validation",
        "test_set_used": False,
        "winners": winners,
    }


def accounting(records: Iterable[dict]) -> dict:
    records = list(records)
    return {
        "attempts": len(records),
        "successful_attempts": sum(item.get("status") == "success" for item in records),
        "failed_attempts": sum(item.get("status") != "success" for item in records),
        "gpu_hours_including_failures": sum(float(item.get("gpu_hours", 0.0)) for item in records),
        "failed_gpu_hours": sum(
            float(item.get("gpu_hours", 0.0))
            for item in records if item.get("status") != "success"
        ),
    }


def _cell_id(cell: tuple[str, str]) -> str:
    return f"{cell[0]}/{cell[1]}"


def _frozen_status(path: str | Path, manifest: dict, *, freeze_eligible: bool) -> dict:
    frozen_path = Path(path).resolve()
    result = {
        "path": str(frozen_path),
        "exists": frozen_path.is_file(),
        "valid": False,
        "winner_count": 0,
        "consistent_with_ledger": False,
        "error": None,
    }
    if not result["exists"]:
        return result
    try:
        payload = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("frozen winner artifact must be a mapping")
        if payload.get("study_id") != manifest["study_id"]:
            raise ValueError("study_id does not match the tuning manifest")
        if payload.get("selection_split") != "validation":
            raise ValueError("selection_split is not validation")
        if payload.get("test_set_used") is not False:
            raise ValueError("test_set_used must be false")
        winners = payload.get("winners")
        if not isinstance(winners, list):
            raise TypeError("winners must be a list")
        observed = [(item.get("architecture"), item.get("method")) for item in winners]
        expected = set(_supported_cells(manifest))
        if len(observed) != len(set(observed)):
            raise ValueError("winner cells are duplicated")
        if set(observed) != expected:
            raise ValueError("winner cells do not match declared support")
        result.update(
            valid=True,
            winner_count=len(winners),
            consistent_with_ledger=bool(freeze_eligible),
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def study_status(
    manifest: dict,
    records: Iterable[dict],
    *,
    frozen_winners_path: str | Path = DEFAULT_FROZEN_WINNERS,
) -> dict:
    """Return a conservative, read-only snapshot of the tuning study.

    Successful distinct required jobs count toward progress. Failed attempts,
    retries, and jobs not recognized by the manifest never inflate it.
    """
    records = list(records)
    attempts = _attempts_by_job(records)
    max_attempts = int(manifest["successive_halving"]["max_attempts_per_job"])
    cells = _supported_cells(manifest)
    cell_ids = [_cell_id(cell) for cell in cells]
    declared_jobs = {job.job_id: job for job in all_jobs(manifest)}
    recognized_records = [
        record for record in records if str(record.get("job_id")) in declared_jobs
    ]
    unknown_records = len(records) - len(recognized_records)

    required = [_initial_jobs(manifest)]
    barrier_error = None
    for rung_index in range(1, len(manifest["successive_halving"]["rungs"])):
        previous = required[-1]
        if not all(_terminal(job, attempts, max_attempts) for job in previous):
            break
        try:
            required.append(_promote(previous, rung_index, manifest, attempts))
        except RuntimeError as error:
            barrier_error = str(error)
            break

    current = required[-1]
    current_all_terminal = all(_terminal(job, attempts, max_attempts) for job in current)
    reached_final_rung = len(required) == len(manifest["successive_halving"]["rungs"])
    final_successful = reached_final_rung and all(
        _success(job, attempts) is not None for job in current
    )
    freeze_eligible = bool(reached_final_rung and current_all_terminal and final_successful)
    if barrier_error is None and current_all_terminal and not freeze_eligible:
        barrier_error = "A terminal rung cannot advance to a complete successful winner matrix"

    eligible = [] if barrier_error else [
        job for job in current if not _terminal(job, attempts, max_attempts)
    ]
    current_by_cell = {
        cell: [job for job in current if job.cell == cell] for cell in cells
    }
    global_barrier_open = not current_all_terminal
    awaiting_barrier = [
        _cell_id(cell) for cell, jobs in current_by_cell.items()
        if jobs and all(_terminal(job, attempts, max_attempts) for job in jobs)
        and global_barrier_open
    ]
    cells_with_pending = [
        _cell_id(cell) for cell, jobs in current_by_cell.items()
        if any(not _terminal(job, attempts, max_attempts) for job in jobs)
    ]

    rung_rows = []
    planned_per_cell = int(manifest["successive_halving"]["initial_trials"])
    successful_record_ids = {
        str(record["job_id"])
        for record in recognized_records if record.get("status") == "success"
    }
    reached_job_ids = {job.job_id for rung_jobs in required for job in rung_jobs}
    recognized_success_ids = successful_record_ids.intersection(reached_job_ids)
    for rung_index, rung in enumerate(manifest["successive_halving"]["rungs"]):
        if rung_index > 0:
            planned_per_cell = int(
                manifest["successive_halving"]["rungs"][rung_index - 1]["survivors"]
            )
        reached_jobs = required[rung_index] if rung_index < len(required) else []
        reached_ids = {job.job_id for job in reached_jobs}
        successful_ids = recognized_success_ids.intersection(reached_ids)
        terminal_ids = {
            job.job_id for job in reached_jobs if _terminal(job, attempts, max_attempts)
        }
        terminal_failed_ids = {
            job.job_id for job in reached_jobs
            if _terminal(job, attempts, max_attempts) and _success(job, attempts) is None
        }
        rung_rows.append({
            "rung": rung_index + 1,
            "total_epochs": int(rung["total_epochs"]),
            "planned_job_slots": len(cells) * planned_per_cell,
            "reached": rung_index < len(required),
            "required_jobs": len(reached_ids),
            "successful_jobs": len(successful_ids),
            "terminal_jobs": len(terminal_ids),
            "terminal_failed_jobs": len(terminal_failed_ids),
        })

    total_planned = sum(row["planned_job_slots"] for row in rung_rows)
    completed_successful = len(recognized_success_ids)
    account = accounting(records)
    return {
        "study_id": manifest["study_id"],
        "read_only": True,
        "declared_support_cells": len(cells),
        "support_cells": cell_ids,
        "attempts": account,
        "unrecognized_job_attempts": unknown_records,
        "completed_jobs_by_rung": rung_rows,
        "current_rung": required[-1][0].rung_index + 1,
        "eligible_queued_jobs": {
            "count": len(eligible),
            "job_ids": [job.job_id for job in eligible],
        },
        "barrier": {
            "global_barrier_open": global_barrier_open,
            "cells_awaiting_other_cells": awaiting_barrier,
            "cells_with_pending_jobs": cells_with_pending,
            "blocked": barrier_error is not None,
            "error": barrier_error,
        },
        "conservative_progress": {
            "completed_successful_jobs": completed_successful,
            "total_planned_job_slots": total_planned,
            "fraction": completed_successful / total_planned,
            "percent": 100.0 * completed_successful / total_planned,
            "rule": "distinct recognized successful jobs only; failures and retries add zero",
        },
        "freeze_eligible": freeze_eligible,
        "frozen": _frozen_status(
            frozen_winners_path, manifest, freeze_eligible=freeze_eligible,
        ),
    }


def tuning_status(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    ledger_path: str | Path = DEFAULT_LEDGER,
    frozen_winners_path: str | Path = DEFAULT_FROZEN_WINNERS,
) -> dict:
    """Load the three study artifacts without writing to any of them."""
    manifest = load_manifest(manifest_path)
    records = _read_ledger(ledger_path)
    return study_status(
        manifest, records, frozen_winners_path=frozen_winners_path,
    )


def format_status_text(status: dict) -> str:
    """Render a concise human view while JSON remains the canonical output."""
    attempts = status["attempts"]
    progress = status["conservative_progress"]
    lines = [
        f"Study: {status['study_id']}",
        (
            f"Support: {status['declared_support_cells']} cells | "
            f"Attempts: {attempts['attempts']} "
            f"({attempts['successful_attempts']} success, {attempts['failed_attempts']} failure) | "
            f"GPU-hours: {attempts['gpu_hours_including_failures']:.3f}"
        ),
        (
            f"Progress: {progress['completed_successful_jobs']}/"
            f"{progress['total_planned_job_slots']} successful jobs "
            f"({progress['percent']:.1f}%)"
        ),
    ]
    for rung in status["completed_jobs_by_rung"]:
        state = "reached" if rung["reached"] else "not reached"
        lines.append(
            f"Rung {rung['rung']} ({rung['total_epochs']} epochs, {state}): "
            f"{rung['successful_jobs']}/{rung['planned_job_slots']} successful, "
            f"{rung['terminal_failed_jobs']} terminal failures"
        )
    barrier = status["barrier"]
    lines.append(
        f"Queue: {status['eligible_queued_jobs']['count']} eligible | "
        f"Barrier-waiting cells: {len(barrier['cells_awaiting_other_cells'])}"
    )
    if barrier["blocked"]:
        lines.append(f"Blocked: {barrier['error']}")
    frozen = status["frozen"]
    frozen_label = "valid" if frozen["valid"] else ("invalid" if frozen["exists"] else "absent")
    lines.append(
        f"Freeze: eligible={status['freeze_eligible']} | artifact={frozen_label}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only equal-budget successive halving")
    subparsers = parser.add_subparsers(dest="action", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)

    queue_parser = subparsers.add_parser("queue")
    queue_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    queue_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    queue_parser.add_argument("--format", choices=("text", "jsonl"), default="text")

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    materialize_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    materialize_parser.add_argument("--directory", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    run_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    run_parser.add_argument("--job-id", required=True)
    run_parser.add_argument("--execute", action="store_true")

    account_parser = subparsers.add_parser("account")
    account_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    status_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    status_parser.add_argument(
        "--frozen-winners", type=Path, default=DEFAULT_FROZEN_WINNERS,
    )
    status_parser.add_argument("--format", choices=("json", "text"), default="json")

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    freeze_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    freeze_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.action == "validate":
        manifest = load_manifest(args.manifest)
        print(json.dumps({"valid": True, "supported_cells": len(_supported_cells(manifest))}))
    elif args.action == "queue":
        for command in dry_queue(args.manifest, args.ledger, args.format):
            print(command)
    elif args.action == "materialize":
        manifest = load_manifest(args.manifest)
        records = _read_ledger(args.ledger)
        attempts = _attempts_by_job(records)
        for job in pending_jobs(manifest, records):
            attempt_number = len(attempts.get(job.job_id, [])) + 1
            path = args.directory / Path(*job.job_id.split("/")).with_suffix(".yaml")
            _write_yaml(path, build_trial_config(manifest, job, attempt_number))
            print(path)
    elif args.action == "run":
        if not args.execute:
            argv = [
                sys.executable, "-m", "fcal_study.tuning", "run",
                str(args.manifest.resolve()), "--ledger", str(args.ledger.resolve()),
                "--job-id", args.job_id, "--execute",
            ]
            print(_display_command(argv))
        else:
            print(json.dumps(execute_job(args.manifest, args.ledger, args.job_id), sort_keys=True))
    elif args.action == "account":
        print(json.dumps(accounting(_read_ledger(args.ledger)), sort_keys=True))
    elif args.action == "status":
        payload = tuning_status(args.manifest, args.ledger, args.frozen_winners)
        if args.format == "text":
            print(format_status_text(payload))
        else:
            print(json.dumps(payload, sort_keys=True))
    else:
        manifest = load_manifest(args.manifest)
        payload = frozen_winners(manifest, _read_ledger(args.ledger))
        _write_yaml(args.output, payload)
        print(args.output)


if __name__ == "__main__":
    main()
