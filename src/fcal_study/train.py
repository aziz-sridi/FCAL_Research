from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from universal_fcal import AutoFCALScheduler, budgeted_objective, gradient_safe_backward

from .config import config_hash, load_config, save_resolved_config
from .data import make_loader
from .losses import adaptive_lambda, article_fcal_loss
from .models import create_model
from .regions import ramp_weight
from .utils import atomic_json, environment_manifest, seed_everything, sha256_file
from .v2 import support_aware_fcal_losses
from .ufcal import build_ufcal, calibrate_ufcal_targets, ufcal_phase


def _rng_state(loader) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader.experiment_generator.get_state(),
        "sampler_epoch": getattr(loader.sampler, "epoch", None),
    }


def _restore_rng(state: dict, loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.experiment_generator.set_state(state["loader_generator"])
    if state.get("sampler_epoch") is not None and hasattr(loader.sampler, "epoch"):
        loader.sampler.epoch = int(state["sampler_epoch"])


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _append_history(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, allow_nan=False) + "\n")
        handle.flush()


def _training_data_config(data: dict) -> dict:
    """Exclude evaluation-only paths from paired-training compatibility checks."""
    return {key: value for key, value in data.items() if key != "evaluation_cache"}


def _amp_context(device: torch.device, mode: str):
    if device.type != "cuda" or mode == "off":
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move(batch: dict, device: torch.device, channels_last: bool) -> tuple:
    images = batch["image"].to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    labels = batch["label"].to(device, non_blocking=True)
    masks = batch.get("mask")
    if masks is not None:
        masks = masks.to(device, non_blocking=True)
    return images, labels, masks


def train_epoch(
    model, loader, optimizer, config, epoch: int, device: torch.device,
    ufcal_runtime: dict | None = None,
) -> dict:
    model.train()
    method = config["method"]
    accumulation = int(config["data"]["accumulation_steps"])
    channels_last = bool(config["training"]["channels_last"])
    amp = config["training"]["amp"]
    fcal = config.get("fcal", {})
    optimizer.zero_grad(set_to_none=True)
    metric_names = (
        "ce", "effective_ce", "background_ce", "fcal", "containment", "viability",
        "logit_consistency", "evidence_consistency", "weighted_fcal", "total", "lambda",
        "outside", "coverage", "valid_mask_fraction", "zero_map_fraction",
        "gradient_ratio", "gradient_cosine", "gradient_conflict_fraction",
    )
    sums = {key: 0.0 for key in metric_names}
    correct = count = steps = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        images, labels, masks = _move(batch, device, channels_last)
        article_active = method == "fcal" and epoch >= int(fcal["warmup_epochs"])
        v2 = config.get("fcal_v2", {})
        v2_active = method == "fcal_v2" and epoch >= int(v2["warmup_epochs"])
        ufcal_active = (
            method == "ufcal"
            and ufcal_runtime is not None
            and float(ufcal_runtime["phase"]) > 0.0
        )
        with _amp_context(device, amp):
            logits = model(images, capture_attention=article_active and method == "fcal")
            ce = F.cross_entropy(logits, labels)
            effective_ce = ce
            background_ce = ce.new_zeros(())
            containment = ce.new_zeros(())
            viability = ce.new_zeros(())
            logit_consistency = ce.new_zeros(())
            evidence_consistency = ce.new_zeros(())
            outside = ce.new_zeros(())
            coverage = ce.new_zeros(())
            valid_mask_fraction = ce.new_zeros(())
            zero_map_fraction = ce.new_zeros(())
            phase = 0.0
            if article_active:
                if masks is None:
                    raise RuntimeError("FCAL batch is missing masks")
                spatial = article_fcal_loss(
                    model.training_attention(labels), masks,
                    int(fcal["dilation_cells"]), float(fcal["epsilon"]),
                )
                weight = adaptive_lambda(
                    ce, spatial, float(fcal["alpha"]), float(fcal["lambda_min"]),
                    float(fcal["lambda_max"]), float(fcal["epsilon"]),
                )
                total = ce + weight * spatial
                weighted_spatial = weight * spatial
            elif v2_active:
                if masks is None:
                    raise RuntimeError("Support-Aware FCAL V2 batch is missing masks")
                containment, viability = support_aware_fcal_losses(
                    model.raw_class_evidence(labels),
                    masks,
                    float(v2["support_temperature"]),
                    float(v2["support_margin"]),
                    float(v2["tolerance_cell_fraction"]),
                    int(v2.get("minimum_tolerance_pixels", 0)),
                    v2.get("maximum_tolerance_pixels"),
                    v2.get("fixed_tolerance_pixels"),
                    float(v2.get("epsilon", 1e-8)),
                )
                weight = adaptive_lambda(
                    ce, containment, float(fcal["alpha"]), float(fcal["lambda_min"]),
                    float(fcal["lambda_max"]), float(fcal["epsilon"]),
                )
                phase = ramp_weight(epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"]))
                weighted_spatial = phase * (
                    weight * containment + float(v2["support_weight"]) * viability
                )
                spatial = containment + viability
                total = ce + weighted_spatial
            elif ufcal_active:
                if masks is None:
                    raise RuntimeError("Universal FCAL batch is missing masks")
                result = ufcal_runtime["objective"](
                    model.raw_class_evidence(labels), masks, logits, labels,
                )
                containment = result.containment
                viability = result.support
                spatial = result.auxiliary
                outside = result.mean_outside
                coverage = result.mean_coverage
                valid_mask_fraction = result.valid_fraction
                zero_map_fraction = result.zero_map_fraction
                phase = float(ufcal_runtime["phase"])
                total, weighted_spatial, weight = budgeted_objective(
                    ce,
                    spatial,
                    phase,
                    float(ufcal_runtime["gradient_ratio"]),
                    float(config["ufcal"].get("lambda_max", 1000.0)),
                    float(config["ufcal"].get("epsilon", 1e-8)),
                )
            else:
                spatial = ce.new_zeros(())
                weight = ce.new_zeros(())
                weighted_spatial = ce.new_zeros(())
                total = ce
        gradient_ratio = gradient_cosine = conflict_fraction = 0.0
        if ufcal_active and bool(config["ufcal"].get("gradient_safe", True)):
            gradient_stats = gradient_safe_backward(
                ce,
                weighted_spatial,
                model.parameters(),
                maximum_gradient_ratio=float(ufcal_runtime["gradient_ratio"]) * phase,
                accumulation_steps=accumulation,
                project_conflicts=bool(config["ufcal"].get("project_conflicts", True)),
                epsilon=float(config["ufcal"].get("epsilon", 1e-8)),
            )
            gradient_ratio = gradient_stats.ratio_after
            gradient_cosine = gradient_stats.cosine_before
            conflict_fraction = float(gradient_stats.conflict_projected)
        else:
            (total / accumulation).backward()
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        batch_size = labels.shape[0]
        count += batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        sums["ce"] += float(ce.detach()) * batch_size
        sums["effective_ce"] += float(effective_ce.detach()) * batch_size
        sums["background_ce"] += float(background_ce.detach()) * batch_size
        sums["fcal"] += float(spatial.detach()) * batch_size
        sums["containment"] += float(containment.detach()) * batch_size
        sums["viability"] += float(viability.detach()) * batch_size
        sums["logit_consistency"] += float(logit_consistency.detach()) * batch_size
        sums["evidence_consistency"] += float(evidence_consistency.detach()) * batch_size
        sums["weighted_fcal"] += float(weighted_spatial.detach()) * batch_size
        sums["total"] += float(total.detach()) * batch_size
        sums["lambda"] += float(weight.detach()) * batch_size
        sums["outside"] += float(outside.detach()) * batch_size
        sums["coverage"] += float(coverage.detach()) * batch_size
        sums["valid_mask_fraction"] += float(valid_mask_fraction.detach()) * batch_size
        sums["zero_map_fraction"] += float(zero_map_fraction.detach()) * batch_size
        sums["gradient_ratio"] += float(gradient_ratio) * batch_size
        sums["gradient_cosine"] += float(gradient_cosine) * batch_size
        sums["gradient_conflict_fraction"] += float(conflict_fraction) * batch_size
        if (batch_index + 1) % 100 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"epoch={epoch + 1} batch={batch_index + 1}/{len(loader)} "
                f"samples_per_s={count / elapsed:.1f} ce={sums['ce'] / count:.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return {
        **{key: value / count for key, value in sums.items()},
        "accuracy": correct / count,
        "samples": count,
        "optimizer_steps": steps,
        "seconds": elapsed,
        "samples_per_second": count / elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
    }


@torch.inference_mode()
def validate(model, loader, config, device: torch.device) -> dict:
    model.eval()
    channels_last = bool(config["training"]["channels_last"])
    amp = config["training"]["amp"]
    ce_sum = 0.0
    correct = count = 0
    started = time.perf_counter()
    for batch in loader:
        images, labels, _ = _move(batch, device, channels_last)
        with _amp_context(device, amp):
            logits = model(images, capture_attention=False)
            ce = F.cross_entropy(logits, labels, reduction="sum")
        ce_sum += float(ce)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.shape[0]
    elapsed = time.perf_counter() - started
    return {
        "ce": ce_sum / count,
        "accuracy": correct / count,
        "samples": count,
        "seconds": elapsed,
        "samples_per_second": count / elapsed,
    }


def _checkpoint(
    model, optimizer, scheduler, config, epoch: int, loader, run_dir: Path, best: float,
    extra_state: dict | None = None,
) -> dict:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "config_hash": config_hash(config),
        "rng": _rng_state(loader),
        "run_dir": str(run_dir),
        "best_validation_accuracy": best,
    }
    if extra_state:
        state.update(extra_state)
    return state


def run(
    config_path: str,
    method: str | None,
    run_id: str | None,
    resume: str | None,
) -> Path:
    config = load_config(config_path)
    if method:
        config["method"] = method
    if run_id:
        config["run_id"] = run_id
    if config["method"] not in {"ce", "fcal", "fcal_v2", "ufcal"}:
        raise ValueError("method must be ce, fcal, fcal_v2, or ufcal")
    seed_everything(int(config["seed"]))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    run_dir = Path(config["output_root"]) / config["run_id"]
    if run_dir.exists() and not resume and any(
        (run_dir / name).exists() for name in ("history.jsonl", "latest.pt", "summary.json")
    ):
        raise FileExistsError(f"Run directory already contains training state: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "config.yaml")
    environment = environment_manifest()
    environment["config_hash"] = config_hash(config)
    split_dir = Path(config["data"]["split_dir"])
    candidates = (
        "train.txt", "validation.txt", "test.txt",
        "train.csv", "validation.csv", "test.csv", "manifest.json",
    )
    split_names = [name for name in candidates if (split_dir / name).is_file()]
    if "manifest.json" not in split_names or not all(
        any(f"{partition}.{suffix}" in split_names for suffix in ("txt", "csv"))
        for partition in ("train", "validation", "test")
    ):
        raise FileNotFoundError(f"Incomplete prepared split: {split_dir}")
    for name in split_names:
        path = split_dir / name
        environment[f"split_{name}_sha256"] = sha256_file(path)
    atomic_json(run_dir / "environment.json", environment)

    checkpoint = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    model = create_model(config, pretrained=checkpoint is None).to(device)
    if config["training"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    fused = bool(config["training"]["fused_adamw"] and device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]), fused=fused,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["epochs"]),
        eta_min=float(config["training"]["learning_rate"]) * float(config["training"]["min_lr_ratio"]),
    )
    need_mask = config["method"] in {"fcal", "fcal_v2", "ufcal"}
    train_loader = make_loader(config, "train", need_mask=need_mask)
    validation_loader = make_loader(config, "validation", need_mask=False, shuffle=False)
    start_epoch = 0
    best_accuracy = -math.inf
    history_path = run_dir / "history.jsonl"
    if checkpoint:
        if (
            checkpoint["config"]["model"] != config["model"]
            or _training_data_config(checkpoint["config"]["data"]) != _training_data_config(config["data"])
        ):
            raise ValueError("Resume checkpoint model/data does not match current config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        same_method_resume = checkpoint["config"].get("method") == config["method"]
        best_accuracy = (
            float(checkpoint["best_validation_accuracy"])
            if same_method_resume
            else -math.inf
        )
        _restore_rng(checkpoint["rng"], train_loader)
        parent_history = Path(checkpoint["run_dir"]) / "history.jsonl"
        if not history_path.exists():
            lines = parent_history.read_text(encoding="utf-8").splitlines()[:start_epoch]
            history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Resuming paired branch from completed epoch {start_epoch}: {resume}", flush=True)
    elif history_path.exists():
        raise FileExistsError(history_path)

    ufcal_objective = None
    ufcal_controller = None
    if config["method"] == "ufcal":
        settings = config["ufcal"]
        calibration_path = run_dir / "ufcal_calibration.json"
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        else:
            calibration_loader = make_loader(config, "train", need_mask=True, shuffle=False)
            calibration = calibrate_ufcal_targets(model, calibration_loader, config, device)
            atomic_json(calibration_path, calibration)
        settings["calibrated_allowed_outside"] = calibration["allowed_outside"]
        settings["calibrated_target_coverage"] = calibration["target_coverage"]
        ufcal_objective = build_ufcal(settings).to(device)
        automatic = settings.get("automatic", {})
        if bool(automatic.get("enabled", False)):
            ufcal_controller = AutoFCALScheduler(
                initial_gradient_ratio=float(automatic["initial_gradient_ratio"]),
                minimum_gradient_ratio=float(automatic["minimum_gradient_ratio"]),
                maximum_gradient_ratio=float(automatic["maximum_gradient_ratio"]),
                increase_factor=float(automatic.get("increase_factor", 1.25)),
                decrease_factor=float(automatic.get("decrease_factor", 0.5)),
                patience=int(automatic.get("patience", 2)),
                cooldown=int(automatic.get("cooldown", 1)),
                accuracy_drop_tolerance=float(automatic.get("accuracy_drop_tolerance", 0.005)),
                localization_plateau_tolerance=float(automatic.get("localization_plateau_tolerance", 0.001)),
                minimum_localization_loss=float(automatic.get("minimum_localization_loss", 1e-5)),
                warmup_epochs=int(settings["warmup_epochs"]),
                ramp_epochs=int(settings["ramp_epochs"]),
                release_start_epoch=automatic.get("release_start_epoch"),
                release_end_epoch=automatic.get("release_end_epoch"),
            )
            if checkpoint and checkpoint.get("ufcal_scheduler"):
                ufcal_controller.load_state_dict(checkpoint["ufcal_scheduler"])
        save_resolved_config(config, run_dir / "config.yaml")
        environment["config_hash"] = config_hash(config)
        atomic_json(run_dir / "environment.json", environment)
        print(f"U-FCAL calibration: {json.dumps(calibration, sort_keys=True)}", flush=True)

    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ufcal_runtime = None
        if config["method"] == "ufcal":
            settings = config["ufcal"]
            if ufcal_controller is not None:
                phase = ufcal_controller.phase(epoch)
                gradient_ratio = ufcal_controller.current_ratio
            else:
                phase = ufcal_phase(
                    epoch,
                    int(settings["warmup_epochs"]),
                    int(settings["ramp_epochs"]),
                    settings.get("decay_start_epoch"),
                    settings.get("decay_end_epoch"),
                )
                gradient_ratio = float(settings["target_gradient_ratio"])
            ufcal_runtime = {
                "objective": ufcal_objective,
                "phase": phase,
                "gradient_ratio": gradient_ratio,
            }
        train_metrics = train_epoch(
            model, train_loader, optimizer, config, epoch, device, ufcal_runtime,
        )
        validation_metrics = validate(model, validation_loader, config, device)
        scheduler.step()
        ufcal_event = None
        if ufcal_controller is not None:
            ufcal_event = ufcal_controller.step(
                epoch,
                validation_metrics["accuracy"],
                train_metrics["fcal"],
            )
        improved = validation_metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy = validation_metrics["accuracy"]
        item = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "best_so_far": improved,
        }
        if ufcal_runtime is not None:
            item["ufcal"] = {
                "phase": ufcal_runtime["phase"],
                "gradient_ratio_used": ufcal_runtime["gradient_ratio"],
                "automatic_event": ufcal_event,
                "scheduler": ufcal_controller.state_dict() if ufcal_controller is not None else None,
            }
        _append_history(history_path, item)
        extra_state = (
            {"ufcal_scheduler": ufcal_controller.state_dict()}
            if ufcal_controller is not None else None
        )
        state = _checkpoint(
            model, optimizer, scheduler, config, epoch + 1, train_loader, run_dir,
            best_accuracy, extra_state,
        )
        _atomic_torch_save(state, run_dir / "latest.pt")
        if epoch + 1 == 5 and bool(config["training"]["save_epoch5_branch"]):
            _atomic_torch_save(state, run_dir / "epoch-005-branch.pt")
        if improved:
            _atomic_torch_save(state, run_dir / "best-validation.pt")
        print(
            f"epoch={epoch + 1}/{int(config['training']['epochs'])} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} val_ce={validation_metrics['ce']:.4f} "
            f"seconds={train_metrics['seconds'] + validation_metrics['seconds']:.1f}",
            flush=True,
        )
    shutil.copy2(run_dir / "latest.pt", run_dir / "final.pt")
    summary = {
        "run_id": config["run_id"],
        "method": config["method"],
        "epochs": int(config["training"]["epochs"]),
        "best_validation_accuracy": best_accuracy,
        "best_checkpoint": str(run_dir / "best-validation.pt"),
        "final_checkpoint": str(run_dir / "final.pt"),
    }
    atomic_json(run_dir / "summary.json", summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--method", choices=("ce", "fcal", "fcal_v2", "ufcal"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(run(args.config, args.method, args.run_id, args.resume))


if __name__ == "__main__":
    main()
