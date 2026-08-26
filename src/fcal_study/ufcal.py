from __future__ import annotations

import math
import statistics
from contextlib import nullcontext

import torch

from universal_fcal import UniversalFCAL, measure_maps


def ufcal_phase(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    decay_start_epoch: int | None = None,
    decay_end_epoch: int | None = None,
) -> float:
    """Ramp FCAL and optionally release it for CE-only recovery."""
    human_epoch = int(epoch) + 1
    if human_epoch <= int(warmup_epochs):
        return 0.0
    ramp = min(1.0, (human_epoch - int(warmup_epochs)) / max(1, int(ramp_epochs)))
    if decay_start_epoch is None or decay_end_epoch is None:
        return ramp
    if human_epoch <= int(decay_start_epoch):
        return ramp
    if human_epoch >= int(decay_end_epoch):
        return 0.0
    progress = (human_epoch - int(decay_start_epoch)) / max(
        1, int(decay_end_epoch) - int(decay_start_epoch),
    )
    return ramp * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_ufcal(settings: dict) -> UniversalFCAL:
    return UniversalFCAL(
        allowed_outside=float(settings["calibrated_allowed_outside"]),
        target_coverage=float(settings["calibrated_target_coverage"]),
        evidence_threshold=float(settings["evidence_threshold"]),
        coverage_temperature=float(settings["coverage_temperature"]),
        confidence_gamma=float(settings["confidence_gamma"]),
        support_mix=float(settings["support_mix"]),
        mode=settings.get("mode", "combined"),
        tolerance_cell_fraction=float(settings["tolerance_cell_fraction"]),
        minimum_tolerance_pixels=int(settings.get("minimum_tolerance_pixels", 0)),
        maximum_tolerance_pixels=settings.get("maximum_tolerance_pixels"),
        maximum_tolerance_fraction=settings.get("maximum_tolerance_fraction", 0.10),
        minimum_foreground_cells=float(settings.get("minimum_foreground_cells", 1.0)),
        minimum_background_cells=float(settings.get("minimum_background_cells", 1.0)),
        epsilon=float(settings.get("epsilon", 1e-8)),
    )


@torch.no_grad()
def calibrate_ufcal_targets(model, loader, config: dict, device: torch.device) -> dict:
    """Infer resolution, valid masks, and attainable targets from the CE branch."""
    settings = config["ufcal"]
    batch_limit = int(settings.get("calibration_batches", 8))
    channels_last = bool(config["training"].get("channels_last", False))
    amp_mode = config["training"].get("amp", "off")
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16

    def autocast():
        if device.type != "cuda" or amp_mode == "off":
            return nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    was_training = model.training
    model.eval()
    outside_values: list[float] = []
    coverage_values: list[float] = []
    valid_count = sample_count = zero_count = 0
    map_sizes: set[tuple[int, int]] = set()
    tolerances: set[int] = set()
    for batch_index, batch in enumerate(loader):
        if batch_index >= batch_limit:
            break
        images = batch["image"].to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast():
            model(images, capture_attention=False)
            score_maps = model.raw_class_evidence(labels)
        for scores in score_maps:
            measured = measure_maps(
                scores,
                masks,
                tolerance_cell_fraction=float(settings["tolerance_cell_fraction"]),
                evidence_threshold=float(settings["evidence_threshold"]),
                coverage_temperature=float(settings["coverage_temperature"]),
                minimum_tolerance_pixels=int(settings.get("minimum_tolerance_pixels", 0)),
                maximum_tolerance_pixels=settings.get("maximum_tolerance_pixels"),
                maximum_tolerance_fraction=settings.get("maximum_tolerance_fraction", 0.10),
                minimum_foreground_cells=float(settings.get("minimum_foreground_cells", 1.0)),
                minimum_background_cells=float(settings.get("minimum_background_cells", 1.0)),
                epsilon=float(settings.get("epsilon", 1e-8)),
            )
            valid = measured.valid
            outside_values.extend(measured.outside[valid].float().cpu().tolist())
            coverage_values.extend(measured.coverage[valid].float().cpu().tolist())
            valid_count += int(valid.sum())
            sample_count += int(valid.numel())
            zero_count += int(measured.zero_map.sum())
            map_sizes.add((int(scores.shape[-2]), int(scores.shape[-1])))
            tolerances.add(int(measured.tolerance_pixels))
    if not outside_values or not coverage_values:
        raise RuntimeError("U-FCAL calibration found no non-degenerate masks")
    initial_outside = float(statistics.median(outside_values))
    allowed_outside = initial_outside * (
        1.0 - float(settings.get("outside_reduction_fraction", 0.25))
    )
    allowed_outside = min(
        max(allowed_outside, float(settings.get("minimum_allowed_outside", 0.02))),
        float(settings.get("maximum_allowed_outside", 0.80)),
    )
    initial_coverage = float(statistics.median(coverage_values))
    target_coverage = initial_coverage + float(settings.get("coverage_improvement", 0.10))
    target_coverage = min(
        max(target_coverage, float(settings.get("minimum_target_coverage", 0.05))),
        float(settings.get("maximum_target_coverage", 0.60)),
    )
    if was_training:
        model.train()
    return {
        "calibration_batches": batch_limit,
        "observations": sample_count,
        "valid_fraction": valid_count / max(1, sample_count),
        "zero_map_fraction": zero_count / max(1, sample_count),
        "evidence_map_sizes": [list(size) for size in sorted(map_sizes)],
        "tolerance_pixels": sorted(tolerances),
        "initial_median_outside": initial_outside,
        "allowed_outside": allowed_outside,
        "initial_median_coverage": initial_coverage,
        "target_coverage": target_coverage,
    }


__all__ = ["build_ufcal", "calibrate_ufcal_targets", "ufcal_phase"]
