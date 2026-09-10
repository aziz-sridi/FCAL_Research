"""Reproducible batching and aggregation for the sealed faithfulness protocol.

This module discovers only the declared fixed CUB matrix and, optionally, the
validation-frozen tuning winners.  It does not discover checkpoints by a broad
filesystem glob: every job comes from an explicit method/architecture registry
or from a complete frozen-winner manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .baseline_fixed_results import ARCHITECTURES
from .config import PROJECT_ROOT
from .faithfulness import PROTOCOL_VERSION, run, summarize
from .utils import sha256_file

BATCH_PROTOCOL_VERSION = "fcal-faithfulness-batch-v1"
FIXED_METHODS = (
    ("CE", "ce", "archive", "ce"),
    ("FCAL V2", "fcal-v5", "archive", "fcal_v5"),
    ("CAMAL", "camal", "current", "camal"),
    ("GAIN-ext", "gain-ext", "current", "gain"),
    ("RRR", "rrr", "current", "rrr"),
)
ISNET_FIXED = ("Selective Faster-ISNet", "isnet", "current", "isnet", "resnet50")
REQUIRED_ARTIFACTS = ("protocol.json", "summary.json", "per_image.npz")
ARCHITECTURE_MODEL_NAMES = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet-b0": "efficientnet_b0",
    "vit-tiny-p16-avg": "vit_tiny_patch16_224",
    "vit-small-p16-avg": "vit_small_patch16_224",
    "vit-base-p16-avg": "vit_base_patch16_224",
}


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return Path(os.path.relpath(resolved, PROJECT_ROOT)).as_posix()
    except ValueError:  # Windows paths on different drives cannot be relative.
        return resolved.as_posix()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _slug(value: str) -> str:
    return "-".join(
        part for part in "".join(
            character.lower() if character.isalnum() else " " for character in value
        ).split() if part
    )


def _checkpoint_method(path: Path) -> tuple[str, str]:
    config_path = path.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return str(config.get("method")), str(config.get("model", {}).get("name"))


def _job(
    *,
    regime: str,
    architecture: str,
    method: str,
    training_method: str,
    checkpoint: Path,
    artifact_root: Path,
    support_scope: str,
    architecture_wtl_eligible: bool,
    selection: dict | None = None,
) -> dict:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
    observed_method, observed_architecture = _checkpoint_method(checkpoint)
    if observed_method != training_method:
        raise ValueError(
            f"Unexpected training method in {checkpoint.parent / 'config.yaml'}: "
            f"expected {training_method}, observed {observed_method}"
        )
    if observed_architecture != ARCHITECTURE_MODEL_NAMES[architecture]:
        raise ValueError(
            f"Unexpected architecture in {checkpoint.parent / 'config.yaml'}: "
            f"expected {ARCHITECTURE_MODEL_NAMES[architecture]}, observed {observed_architecture}"
        )
    job_id = f"{regime}/{architecture}/{_slug(method)}"
    result = {
        "job_id": job_id,
        "regime": regime,
        "architecture": architecture,
        "method": method,
        "training_method": training_method,
        "support_scope": support_scope,
        "architecture_wtl_eligible": architecture_wtl_eligible,
        "checkpoint": _portable_path(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "artifact_dir": _portable_path(artifact_root / Path(*job_id.split("/"))),
    }
    if selection is not None:
        result["tuning_selection"] = selection
    return result


def discover_fixed_jobs(
    archive_root: str | Path,
    current_root: str | Path,
    artifact_root: str | Path,
) -> list[dict]:
    """Discover the exact complete fixed matrix, including ResNet-only ISNet."""
    roots = {"archive": Path(archive_root).resolve(), "current": Path(current_root).resolve()}
    artifact_root = Path(artifact_root).resolve()
    jobs = []
    for architecture, _, _ in ARCHITECTURES:
        for method, suffix, source, training_method in FIXED_METHODS:
            checkpoint = roots[source] / f"{architecture}-{suffix}-seed42" / "best-validation.pt"
            jobs.append(_job(
                regime="fixed_default",
                architecture=architecture,
                method=method,
                training_method=training_method,
                checkpoint=checkpoint,
                artifact_root=artifact_root,
                support_scope="six_architecture",
                architecture_wtl_eligible=True,
            ))
    method, suffix, source, training_method, architecture = ISNET_FIXED
    jobs.append(_job(
        regime="fixed_default",
        architecture=architecture,
        method=method,
        training_method=training_method,
        checkpoint=roots[source] / f"{architecture}-{suffix}-seed42" / "best-validation.pt",
        artifact_root=artifact_root,
        support_scope="resnet50_only",
        architecture_wtl_eligible=False,
    ))
    return sorted(jobs, key=lambda item: item["job_id"])


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def discover_tuned_jobs(
    tuning_manifest_path: str | Path,
    frozen_winners_path: str | Path,
    artifact_root: str | Path,
) -> tuple[list[dict], dict]:
    """Validate and bind every supported cell in a frozen validation-only search."""
    tuning_path = Path(tuning_manifest_path).resolve()
    frozen_path = Path(frozen_winners_path).resolve()
    tuning = _load_yaml(tuning_path)
    frozen = _load_yaml(frozen_path)
    if frozen.get("study_id") != tuning.get("study_id"):
        raise ValueError("Frozen winners do not belong to the tuning study")
    if frozen.get("selection_split") != "validation" or frozen.get("test_set_used") is not False:
        raise ValueError("Tuned checkpoints must be frozen using validation only before faithfulness")

    expected = {
        (architecture, method)
        for architecture in tuning.get("architectures", {})
        for method, method_spec in tuning.get("methods", {}).items()
        if architecture in method_spec.get("supported_architectures", ())
    }
    winners = frozen.get("winners")
    if not isinstance(winners, list):
        raise TypeError("Frozen winner manifest must contain a winners list")
    indexed = {}
    for winner in winners:
        cell = (winner.get("architecture"), winner.get("method"))
        if cell in indexed:
            raise ValueError(f"Duplicate frozen winner cell: {cell}")
        indexed[cell] = winner
    if set(indexed) != expected:
        missing = sorted(expected.difference(indexed))
        extra = sorted(set(indexed).difference(expected))
        raise ValueError(f"Frozen winner matrix is incomplete; missing={missing}, extra={extra}")

    artifact_root = Path(artifact_root).resolve()
    jobs = []
    for architecture, method_key in sorted(expected):
        method_spec = tuning["methods"][method_key]
        winner = indexed[(architecture, method_key)]
        coefficient = str(method_spec["coefficient"])
        if winner.get("coefficient") != coefficient:
            raise ValueError(f"Frozen coefficient key mismatch for {(architecture, method_key)}")
        if float(winner.get("value")) not in {float(value) for value in method_spec["values"]}:
            raise ValueError(f"Frozen coefficient value is outside the search grid: {(architecture, method_key)}")
        checkpoint = _resolve(winner["checkpoint"])
        training_method, _ = _checkpoint_method(checkpoint)
        supported = tuple(method_spec["supported_architectures"])
        support_scope = "six_architecture" if len(supported) == len(ARCHITECTURES) else "resnet50_only"
        jobs.append(_job(
            regime="tuned",
            architecture=architecture,
            method=str(method_spec["display_name"]),
            training_method=training_method,
            checkpoint=checkpoint,
            artifact_root=artifact_root,
            support_scope=support_scope,
            architecture_wtl_eligible=len(supported) == len(ARCHITECTURES),
            selection={
                "trial_id": str(winner["trial_id"]),
                "coefficient": coefficient,
                "value": float(winner["value"]),
                "validation_accuracy": float(winner["validation_accuracy"]),
            },
        ))
    sources = {
        "tuning_manifest": _portable_path(tuning_path),
        "tuning_manifest_sha256": sha256_file(tuning_path),
        "frozen_winners": _portable_path(frozen_path),
        "frozen_winners_sha256": sha256_file(frozen_path),
        "frozen_study_id": frozen["study_id"],
        "frozen_selection_split": "validation",
        "frozen_test_set_used": False,
    }
    return sorted(jobs, key=lambda item: item["job_id"]), sources


def build_manifest(
    *,
    split: str,
    archive_root: str | Path,
    current_root: str | Path,
    artifact_root: str | Path,
    max_images: int | None,
    selection_seed: int = 2026,
    batch_size: int | None = None,
    tuning_manifest_path: str | Path | None = None,
    frozen_winners_path: str | Path | None = None,
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Batch faithfulness split must be validation or test")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (tuning_manifest_path is None) != (frozen_winners_path is None):
        raise ValueError("Tuning manifest and frozen winners must be supplied together")
    if split == "test" and frozen_winners_path is None:
        raise ValueError("The test split remains sealed until the complete tuned winner matrix is frozen")

    artifact_root = Path(artifact_root).resolve() / split
    jobs = discover_fixed_jobs(archive_root, current_root, artifact_root)
    tuning_sources = None
    if frozen_winners_path is not None:
        tuned, tuning_sources = discover_tuned_jobs(
            tuning_manifest_path, frozen_winners_path, artifact_root,
        )
        jobs.extend(tuned)
    jobs.sort(key=lambda item: item["job_id"])
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Faithfulness job IDs are not unique")
    return {
        "batch_protocol_version": BATCH_PROTOCOL_VERSION,
        "evaluator_protocol_version": PROTOCOL_VERSION,
        "split": split,
        "test_unsealed": split == "test",
        "selection": {
            "max_images": max_images,
            "selection_seed": int(selection_seed),
            "batch_size": batch_size,
            "outcome_independent_subset": "lowest_sha256(selection_seed:image_id)",
        },
        "fixed_sources": {
            "archive_root": _portable_path(archive_root),
            "current_root": _portable_path(current_root),
        },
        "tuning_freeze": tuning_sources,
        "jobs": jobs,
    }


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_manifest(path: str | Path, manifest: dict) -> Path:
    """Write once, while allowing an exact idempotent recreation for resume."""
    path = Path(path).resolve()
    serialized = _canonical_json(manifest)
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return path
        raise FileExistsError(f"Refusing to replace a different faithfulness manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_batch_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("batch_protocol_version") != BATCH_PROTOCOL_VERSION:
        raise ValueError("Unexpected faithfulness batch protocol version")
    if manifest.get("evaluator_protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Batch manifest targets a different evaluator protocol")
    split = manifest.get("split")
    if split not in {"validation", "test"}:
        raise ValueError("Invalid faithfulness manifest split")
    if split == "test" and (
        manifest.get("test_unsealed") is not True
        or not manifest.get("tuning_freeze")
        or manifest["tuning_freeze"].get("frozen_test_set_used") is not False
    ):
        raise ValueError("Test manifest is not bound to a validation-only tuning freeze")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Faithfulness manifest has no jobs")
    ids = [job.get("job_id") for job in jobs]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Faithfulness jobs must have unique deterministic ordering")
    return manifest


def _artifact_state(manifest: dict, job: dict) -> tuple[str, str | None]:
    output = _resolve(job["artifact_dir"])
    if not output.exists():
        return "pending", None
    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        return "invalid", f"existing artifact directory is incomplete: {', '.join(missing)}"
    try:
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        recorded = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        with np.load(output / "per_image.npz") as archive:
            payload = {key: archive[key] for key in archive.files}
        recalculated = summarize(payload)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        # Fail closed on malformed, truncated, or schema-incompatible sealed output.
        return "invalid", f"sealed artifact cannot be read: {error}"
    selection = manifest["selection"]
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "split": manifest["split"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "requested_max_images": selection["max_images"],
        "selection_seed": (
            None if selection["max_images"] is None else selection["selection_seed"]
        ),
    }
    for key, value in expected.items():
        if recorded.get(key) != value or protocol.get(key) != value:
            return "invalid", f"sealed artifact binding mismatch for {key}"
    if recorded.get("method") != job["training_method"]:
        return "invalid", "sealed artifact training method does not match its job"
    for key, value in recalculated.items():
        observed = recorded.get(key)
        if value is None:
            if observed is not None:
                return "invalid", f"summary mismatch for {key}"
        elif not np.isclose(float(observed), float(value), rtol=0.0, atol=1e-7):
            return "invalid", f"summary mismatch for {key}"
    return "complete", None


def job_states(manifest: dict) -> list[dict]:
    rows = []
    for job in manifest["jobs"]:
        state, error = _artifact_state(manifest, job)
        rows.append({"job_id": job["job_id"], "state": state, "error": error})
    return rows


def run_job(manifest_path: str | Path, job_id: str) -> Path:
    """Execute one pending job; exact completed jobs are an idempotent no-op."""
    manifest = load_batch_manifest(manifest_path)
    indexed = {job["job_id"]: job for job in manifest["jobs"]}
    if job_id not in indexed:
        raise KeyError(f"Unknown faithfulness job: {job_id}")
    job = indexed[job_id]
    state, error = _artifact_state(manifest, job)
    output = _resolve(job["artifact_dir"])
    if state == "complete":
        return output
    if state == "invalid":
        raise RuntimeError(f"Refusing to overwrite invalid sealed artifact for {job_id}: {error}")
    checkpoint = _resolve(job["checkpoint"])
    if sha256_file(checkpoint) != job["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint changed after batch manifest freeze: {checkpoint}")
    selection = manifest["selection"]
    run(
        checkpoint,
        split=manifest["split"],
        output_dir=output,
        max_images=selection["max_images"],
        selection_seed=selection["selection_seed"],
        batch_size=selection["batch_size"],
    )
    state, error = _artifact_state(manifest, job)
    if state != "complete":
        raise RuntimeError(f"New faithfulness artifact failed verification: {error}")
    return output


def aggregate(manifest_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Verify all sealed artifacts and write a write-once CSV/Markdown report."""
    manifest = load_batch_manifest(manifest_path)
    states = job_states(manifest)
    incomplete = [row for row in states if row["state"] != "complete"]
    if incomplete:
        raise RuntimeError(f"Cannot aggregate incomplete faithfulness batch: {incomplete}")
    rows = []
    for job in manifest["jobs"]:
        summary_path = _resolve(job["artifact_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "split": manifest["split"],
            "regime": job["regime"],
            "architecture": job["architecture"],
            "method": job["method"],
            "training_method": job["training_method"],
            "support_scope": job["support_scope"],
            "architecture_wtl_eligible": job["architecture_wtl_eligible"],
            "samples": summary["samples"],
            "valid_mask_samples": summary["valid_mask_samples"],
            "mask_iou_mean": summary["attention_iou_mean"],
            "insertion_auc_mean": summary["insertion_auc_normalized_mean"],
            "removal_auc_mean": summary["removal_auc_normalized_mean"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "artifact_dir": job["artifact_dir"],
        }
        selection = job.get("tuning_selection", {})
        row.update({
            "tuned_coefficient": selection.get("coefficient"),
            "tuned_value": selection.get("value"),
            "tuning_validation_accuracy": selection.get("validation_accuracy"),
        })
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["regime", "architecture", "method"])
    display = frame[[
        "regime", "architecture", "method", "support_scope", "samples",
        "valid_mask_samples", "mask_iou_mean", "insertion_auc_mean", "removal_auc_mean",
    ]].copy()
    for column in ("mask_iou_mean", "insertion_auc_mean", "removal_auc_mean"):
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{value:.6f}")
    lines = [
        f"# Faithfulness aggregation ({manifest['split']})",
        "",
        "Higher mask IoU and insertion AUC are favorable; lower removal AUC is favorable.",
        "All rows use the same frozen common-CAM evaluator and deterministic image subset.",
        "Selective Faster-ISNet is labeled `resnet50_only` and is excluded from six-architecture W/T/L.",
        "",
        display.to_markdown(index=False),
        "",
    ]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite faithfulness aggregation: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        frame.to_csv(temporary / "summary.csv", index=False)
        (temporary / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "summary.csv", output_dir / "REPORT.md"


def _queue_commands(manifest_path: Path, manifest: dict, jsonl: bool) -> list[str]:
    commands = []
    for job, status in zip(manifest["jobs"], job_states(manifest), strict=True):
        if status["state"] == "invalid":
            raise RuntimeError(f"Invalid sealed artifact for {job['job_id']}: {status['error']}")
        if status["state"] == "complete":
            continue
        argv = [
            sys.executable, "-m", "fcal_study.faithfulness_batch", "run",
            str(manifest_path), "--job-id", job["job_id"],
        ]
        if jsonl:
            commands.append(json.dumps({"job_id": job["job_id"], "command": argv}))
        else:
            commands.append(" ".join(f'"{value}"' if " " in value else value for value in argv))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--split", choices=("validation", "test"), required=True)
    manifest_parser.add_argument("--archive-root", type=Path, required=True)
    manifest_parser.add_argument("--fixed-root", type=Path, required=True)
    manifest_parser.add_argument("--artifact-root", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--max-images", type=int)
    manifest_parser.add_argument("--selection-seed", type=int, default=2026)
    manifest_parser.add_argument("--batch-size", type=int)
    manifest_parser.add_argument("--tuning-manifest", type=Path)
    manifest_parser.add_argument("--frozen-winners", type=Path)

    queue_parser = subparsers.add_parser("queue")
    queue_parser.add_argument("manifest", type=Path)
    queue_parser.add_argument("--format", choices=("command", "jsonl"), default="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("--job-id", required=True)
    run_all_parser = subparsers.add_parser("run-all")
    run_all_parser.add_argument("manifest", type=Path)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("manifest", type=Path)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "manifest":
        manifest = build_manifest(
            split=args.split,
            archive_root=args.archive_root,
            current_root=args.fixed_root,
            artifact_root=args.artifact_root,
            max_images=args.max_images,
            selection_seed=args.selection_seed,
            batch_size=args.batch_size,
            tuning_manifest_path=args.tuning_manifest,
            frozen_winners_path=args.frozen_winners,
        )
        print(write_manifest(args.output, manifest))
    elif args.action == "queue":
        manifest_path = args.manifest.resolve()
        manifest = load_batch_manifest(manifest_path)
        print(*_queue_commands(manifest_path, manifest, args.format == "jsonl"), sep="\n")
    elif args.action == "run":
        print(run_job(args.manifest, args.job_id))
    elif args.action == "run-all":
        manifest = load_batch_manifest(args.manifest)
        for job in manifest["jobs"]:
            print(run_job(args.manifest, job["job_id"]))
    else:
        print(*aggregate(args.manifest, args.output_dir), sep="\n")


if __name__ == "__main__":
    main()
