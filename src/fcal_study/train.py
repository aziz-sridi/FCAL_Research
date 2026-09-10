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
from .decision_evidence import v2_training_evidence, validate_evidence_reference
from .explanation_baselines import (
    camal_losses,
    gain_losses,
    isnet_heatmap_loss,
    right_reasons_loss,
)
from .losses import adaptive_lambda, article_fcal_loss
from .models import create_model
from .regions import ramp_weight
from .ufcal import build_ufcal, calibrate_ufcal_targets, ufcal_phase
from .utils import atomic_json, environment_manifest, seed_everything, sha256_file
from .v2 import resolution_aware_tolerance_pixels, support_aware_fcal_losses
from .v2_3 import (
    foreground_coverage_hinge_loss,
    strict_positive_containment_loss,
)
from .v2_4 import seed_and_grow_foreground_loss
from .v2_next import (
    buffered_region_masks,
    local_foreground_rival_margin_loss,
    native_regional_dominance_loss,
    persistent_seed_and_grow_loss,
    regional_classification_losses,
    softmin_foreground_rival_margin_loss,
    uncancelled_channel_containment_loss,
)
from .v2_plus import (
    foreground_competitor_margin_loss,
    gated_foreground_competitor_margin_loss,
    mean_preserving_evidence_proxy,
    top_competitor_labels,
)
from .v2_relative import relative_pooled_margin_loss
from .v3 import (
    background_counterfactual,
    balanced_regional_counterfactual,
    counterfactual_support_losses,
    prediction_consistency_loss,
    strict_shell_containment,
)
from .v4 import (
    background_margin_sensitivity,
    mask_alignment_loss,
    regional_dominance_loss,
    strict_positive_containment,
)
from .v5 import regional_margin_certificate_loss


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


def _last_epoch_averages(path: Path, count: int = 5) -> dict:
    """Average numeric train/validation metrics over the final epochs."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = rows[-int(count):]
    if not selected:
        raise ValueError(f"Cannot summarize an empty history: {path}")

    def average(section: str) -> dict[str, float]:
        keys = set.intersection(*(
            {
                key for key, value in row[section].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            for row in selected
        ))
        return {
            key: sum(float(row[section][key]) for row in selected) / len(selected)
            for key in sorted(keys)
        }

    return {
        "epochs": [int(row["epoch"]) for row in selected],
        "train": average("train"),
        "validation": average("validation"),
    }


def _training_data_config(data: dict) -> dict:
    """Exclude evaluation-only paths from paired-training compatibility checks."""
    return {key: value for key, value in data.items() if key != "evaluation_cache"}


_RELOCATABLE_DATA_PATHS = {
    "cub_root", "isic_root", "pet_root", "medmnist_root", "cardiac_root",
    "segmentation_root", "split_dir", "evaluation_cache",
}


def _split_identity(data: dict) -> dict[str, str]:
    split_dir = Path(data["split_dir"])
    names = (
        "train.txt", "validation.txt", "test.txt",
        "train.csv", "validation.csv", "test.csv", "manifest.json",
    )
    return {
        name: sha256_file(split_dir / name)
        for name in names
        if (split_dir / name).is_file()
    }


def _environment_split_identity(path: str | Path) -> dict[str, str]:
    """Read prepared-split hashes recorded beside a checkpoint."""
    environment_path = Path(path)
    if not environment_path.is_file():
        return {}
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    prefix = "split_"
    suffix = "_sha256"
    return {
        key[len(prefix):-len(suffix)]: str(value)
        for key, value in environment.items()
        if key.startswith(prefix) and key.endswith(suffix)
    }


def _relocated_training_data_matches(
    parent: dict,
    child: dict,
    parent_split_dir: str | Path | None = None,
    parent_environment: str | Path | None = None,
) -> bool:
    """Verify a paired checkpoint after an explicit dataset-path relocation.

    Dataset and segmentation roots are allowed to move only when every
    non-path loader setting and every prepared split artifact are identical.
    This is intentionally opt-in at the CLI; it is used for archived FCAL
    branches whose datasets were moved without changing their contents.
    """
    stable_parent = {
        key: value for key, value in parent.items()
        if key not in _RELOCATABLE_DATA_PATHS
    }
    stable_child = {
        key: value for key, value in child.items()
        if key not in _RELOCATABLE_DATA_PATHS
    }
    child_identity = _split_identity(child)
    if parent_environment is not None:
        parent_identity = _environment_split_identity(parent_environment)
    else:
        parent_for_identity = dict(parent)
        if parent_split_dir is not None:
            parent_for_identity["split_dir"] = str(Path(parent_split_dir).resolve())
        parent_identity = _split_identity(parent_for_identity)
    if parent_split_dir is not None:
        explicit_parent = dict(parent)
        explicit_parent["split_dir"] = str(Path(parent_split_dir).resolve())
        if _split_identity(explicit_parent) != parent_identity:
            return False
    return (
        stable_parent == stable_child
        and bool(parent_identity)
        and parent_identity == child_identity
    )


def _seed_resume_history(
    resume: str | Path,
    destination: Path,
    completed_epochs: int,
) -> None:
    """Seed a child branch from history stored beside its actual checkpoint."""
    parent_history = Path(resume).resolve().parent / "history.jsonl"
    lines = parent_history.read_text(encoding="utf-8").splitlines()
    if len(lines) < completed_epochs:
        raise ValueError(
            f"Parent history has {len(lines)} records but checkpoint reports "
            f"{completed_epochs} completed epochs: {parent_history}"
        )
    destination.write_text(
        "\n".join(lines[:completed_epochs]) + "\n", encoding="utf-8",
    )


def _amp_context(device: torch.device, mode: str | bool):
    disabled = mode is False or mode is None or str(mode).lower() in {"off", "false", "none"}
    if device.type != "cuda" or disabled:
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
    evidence_reference = validate_evidence_reference(config)
    model.train()
    method = config["method"]
    accumulation = int(config["data"]["accumulation_steps"])
    channels_last = bool(config["training"]["channels_last"])
    amp = config["training"]["amp"]
    fcal = config.get("fcal", {})
    optimizer.zero_grad(set_to_none=True)
    metric_names = (
        "ce", "effective_ce", "fcal", "containment", "viability",
        "evidence_consistency", "weighted_fcal", "total", "lambda",
        "outside", "coverage", "valid_mask_fraction", "zero_map_fraction",
        "gradient_ratio", "gradient_cosine", "gradient_conflict_fraction",
        "attention_mining", "attention_supervision", "nonfinite_gradient_fraction",
        "attention_inside", "attention_outside",
        "counterfactual_ce", "prediction_consistency", "donor_mismatch_fraction",
        "foreground_intervention_fraction", "strict_shell", "strict_containment",
        "mask_alignment", "regional_dominance", "background_sensitivity",
        "certificate_loss", "certificate", "certificate_foreground_tail",
        "certificate_background_tail", "competitor_margin",
        "foreground_coverage_hinge", "seed_grow", "seed_front",
        "seed_retention", "native_dominance",
        "foreground_region_classification", "background_region_suppression",
        "regional_gamma", "weighted_regional",
        "background_feature_consistency", "weighted_feature_consistency",
        "local_foreground_rival_margin", "channel_containment",
        "relative_pooled_margin", "shortcut_gate",
        "regional_foreground_margin", "regional_background_margin",
        "regional_margin_gap", "relative_outside_evidence",
        "relative_valid_mask_fraction",
        "competitor_margin_ungated", "competitor_gate",
        "competitor_foreground_support", "competitor_target_confidence",
        "competitor_valid_mask_fraction",
    )
    sums = {key: 0.0 for key in metric_names}
    correct = count = steps = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        images, labels, masks = _move(batch, device, channels_last)
        v4 = config.get("fcal_v4", {})
        if method in {"rrr", "isnet"} or (
            method == "fcal_v4" and float(v4.get("background_sensitivity_weight", 0.0)) > 0
        ):
            images.requires_grad_(True)
        article_active = method == "fcal" and epoch >= int(fcal["warmup_epochs"])
        v2 = config.get("fcal_v2", {})
        v2_active = method == "fcal_v2" and epoch >= int(v2["warmup_epochs"])
        v2_plus = config.get("fcal_v2_plus", {})
        v2_plus_active = (
            method == "fcal_v2_plus" and epoch >= int(v2["warmup_epochs"])
        )
        v2_plus_gated = config.get("fcal_v2_plus_gated", {})
        v2_plus_gated_active = (
            method == "fcal_v2_plus_gated"
            and epoch >= int(v2["warmup_epochs"])
        )
        v2_relative = config.get("fcal_v2_relative", {})
        v2_relative_active = (
            method == "fcal_v2_relative" and epoch >= int(v2["warmup_epochs"])
        )
        v2_3 = config.get("fcal_v2_3", {})
        v2_3_active = (
            method == "fcal_v2_3" and epoch >= int(v2["warmup_epochs"])
        )
        v2_4 = config.get("fcal_v2_4", {})
        v2_4_active = (
            method == "fcal_v2_4" and epoch >= int(v2["warmup_epochs"])
        )
        v2_next = config.get("fcal_v2_next", {})
        v2_next_active = (
            method in {
                "fcal_v2_5", "fcal_v2_6", "fcal_v2_7", "fcal_v2_8",
                "fcal_v2_9", "fcal_v2_7a", "fcal_v2_7b", "fcal_v2_7c",
                "fcal_v2_7d", "fcal_v2_7_prime", "fcal_v2_feature_consistency",
            }
            and epoch >= int(v2["warmup_epochs"])
        )
        v3 = config.get("fcal_v3", {})
        v3_active = method == "fcal_v3" and epoch >= int(v2["warmup_epochs"])
        v4_active = method == "fcal_v4" and epoch >= int(v2["warmup_epochs"])
        v5 = config.get("fcal_v5_cheap", {})
        v5_active = (
            method == "fcal_v5_cheap"
            and epoch >= int(v5.get("warmup_epochs", 5))
        )
        ufcal_active = (
            method == "ufcal"
            and ufcal_runtime is not None
            and float(ufcal_runtime["phase"]) > 0.0
        )
        explanation_active = method in {"rrr", "gain", "isnet", "camal"}
        with _amp_context(device, amp):
            logits = model(
                images,
                capture_attention=(article_active and method == "fcal") or method == "isnet",
            )
            ce = F.cross_entropy(logits, labels)
            effective_ce = ce
            containment = ce.new_zeros(())
            viability = ce.new_zeros(())
            evidence_consistency = ce.new_zeros(())
            outside = ce.new_zeros(())
            coverage = ce.new_zeros(())
            valid_mask_fraction = ce.new_zeros(())
            zero_map_fraction = ce.new_zeros(())
            phase = 0.0
            attention_mining = ce.new_zeros(())
            attention_supervision = ce.new_zeros(())
            attention_inside = ce.new_zeros(())
            attention_outside = ce.new_zeros(())
            counterfactual_ce = ce.new_zeros(())
            prediction_consistency = ce.new_zeros(())
            donor_mismatch_fraction = ce.new_zeros(())
            foreground_intervention_fraction = ce.new_zeros(())
            strict_shell = ce.new_zeros(())
            strict_containment = ce.new_zeros(())
            mask_alignment = ce.new_zeros(())
            regional_dominance = ce.new_zeros(())
            background_sensitivity = ce.new_zeros(())
            certificate_loss = ce.new_zeros(())
            certificate = ce.new_zeros(())
            certificate_foreground_tail = ce.new_zeros(())
            certificate_background_tail = ce.new_zeros(())
            competitor_margin = ce.new_zeros(())
            foreground_coverage_hinge = ce.new_zeros(())
            seed_grow = ce.new_zeros(())
            seed_front = ce.new_zeros(())
            seed_retention = ce.new_zeros(())
            native_dominance = ce.new_zeros(())
            foreground_region_classification = ce.new_zeros(())
            background_region_suppression = ce.new_zeros(())
            regional_gamma = ce.new_zeros(())
            weighted_regional = ce.new_zeros(())
            background_feature_consistency = ce.new_zeros(())
            weighted_feature_consistency = ce.new_zeros(())
            local_foreground_rival_margin = ce.new_zeros(())
            channel_containment = ce.new_zeros(())
            relative_pooled_margin = ce.new_zeros(())
            shortcut_gate = ce.new_zeros(())
            regional_foreground_margin = ce.new_zeros(())
            regional_background_margin = ce.new_zeros(())
            regional_margin_gap = ce.new_zeros(())
            relative_outside_evidence = ce.new_zeros(())
            relative_valid_mask_fraction = ce.new_zeros(())
            competitor_margin_ungated = ce.new_zeros(())
            competitor_gate = ce.new_zeros(())
            competitor_foreground_support = ce.new_zeros(())
            competitor_target_confidence = ce.new_zeros(())
            competitor_valid_mask_fraction = ce.new_zeros(())
            if explanation_active:
                if masks is None:
                    raise RuntimeError(f"{method} batch is missing masks")
                if method == "rrr":
                    settings = config["rrr"]
                    spatial = right_reasons_loss(logits, images, masks)
                    weight = ce.new_tensor(float(settings["lambda"]))
                    weighted_spatial = weight * spatial
                    total = ce + weighted_spatial
                elif method == "gain":
                    settings = config["gain"]
                    attention_mining, attention_supervision = gain_losses(
                        model, images, logits, labels, masks,
                        sigma=float(settings["sigma"]),
                        omega=float(settings["omega"]),
                        epsilon=float(settings.get("epsilon", 1e-8)),
                    )
                    ce_weight = float(settings["classification_weight"])
                    mining_weight = float(settings["mining_weight"])
                    supervision_weight = float(settings["supervision_weight"])
                    effective_ce = ce_weight * ce
                    spatial = attention_mining + attention_supervision
                    weighted_spatial = (
                        mining_weight * attention_mining
                        + supervision_weight * attention_supervision
                    )
                    weight = ce.new_tensor(mining_weight + supervision_weight)
                    total = effective_ce + weighted_spatial
                elif method == "isnet":
                    settings = config["isnet"]
                    relevance = model.selective_lrp(
                        images, logits, float(settings["selective_epsilon"]),
                    )
                    spatial = isnet_heatmap_loss(
                        relevance, masks,
                        cut_low=float(settings["cut_low"]),
                        cut_high=float(settings["cut_high"]),
                        background_decay=float(settings["background_decay"]),
                        background_scale=float(settings["background_scale"]),
                        foreground_weight=float(settings["foreground_weight"]),
                        background_weight=float(settings["background_weight"]),
                        epsilon=float(settings.get("epsilon", 1e-10)),
                    )
                    weight = ce.new_tensor(float(settings["heatmap_weight"]))
                    effective_ce = (1.0 - weight) * ce
                    weighted_spatial = weight * spatial
                    total = effective_ce + weighted_spatial
                else:
                    settings = config["camal"]
                    spatial, attention_inside, attention_outside = camal_losses(
                        model, labels, masks,
                        epsilon=float(settings.get("epsilon", 1e-8)),
                    )
                    weight = ce.new_tensor(float(settings["coefficient"]))
                    weighted_spatial = weight * spatial
                    total = ce + weighted_spatial
            elif article_active:
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
                    v2_training_evidence(model, labels, evidence_reference),
                    masks,
                    support_temperature=float(v2["support_temperature"]),
                    support_margin=float(v2["support_margin"]),
                    tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                    minimum_tolerance_pixels=int(v2.get("minimum_tolerance_pixels", 0)),
                    maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                    fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                    fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
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
            elif v2_relative_active:
                if masks is None:
                    raise RuntimeError("Relative pooled-margin FCAL batch is missing masks")
                target_maps = model.raw_class_evidence(labels)
                containment, viability, background_maps = support_aware_fcal_losses(
                    target_maps,
                    masks,
                    support_temperature=float(v2["support_temperature"]),
                    support_margin=float(v2["support_margin"]),
                    tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                    minimum_tolerance_pixels=int(v2.get("minimum_tolerance_pixels", 0)),
                    maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                    fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                    fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                    return_backgrounds=True,
                )
                relative = relative_pooled_margin_loss(
                    model,
                    target_maps,
                    background_maps,
                    labels,
                    masks,
                    delta=float(v2_relative["delta"]),
                    temperature=float(v2_relative["temperature"]),
                    allowed_outside=float(v2_relative["allowed_outside"]),
                    required_foreground_margin=float(
                        v2_relative["required_foreground_margin"]
                    ),
                    outside_gate_temperature=float(
                        v2_relative["outside_gate_temperature"]
                    ),
                    margin_gate_temperature=float(
                        v2_relative["margin_gate_temperature"]
                    ),
                    background_gradient_scale=float(
                        v2_relative["background_gradient_scale"]
                    ),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                )
                relative_pooled_margin = relative.loss
                shortcut_gate = relative.shortcut_gate
                regional_foreground_margin = relative.foreground_margin
                regional_background_margin = relative.background_margin
                regional_margin_gap = relative.margin_gap
                relative_outside_evidence = relative.outside_evidence
                relative_valid_mask_fraction = relative.valid_mask_fraction
                weight = adaptive_lambda(
                    ce, containment, float(fcal["alpha"]),
                    float(fcal["lambda_min"]), float(fcal["lambda_max"]),
                    float(fcal["epsilon"]),
                )
                phase = ramp_weight(
                    epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"])
                )
                weighted_spatial = phase * (
                    weight * containment
                    + float(v2["support_weight"]) * viability
                    + float(v2_relative["weight"]) * relative_pooled_margin
                )
                spatial = containment + viability + relative_pooled_margin
                total = ce + weighted_spatial
            elif v2_plus_active or v2_plus_gated_active:
                if masks is None:
                    raise RuntimeError("FCAL V2+ batch is missing masks")
                target_maps = model.raw_class_evidence(labels)
                v2_plus_settings = (
                    v2_plus_gated if v2_plus_gated_active else v2_plus
                )
                competitors = top_competitor_labels(
                    logits,
                    labels,
                    int(v2_plus_settings.get("top_k_competitors", 2)),
                )
                rival_maps = [
                    model.raw_class_evidence(competitors[:, rank])
                    for rank in range(competitors.shape[1])
                ]
                margin_target_maps = target_maps
                margin_rival_maps = rival_maps
                if bool(v2_plus_settings.get("mean_preserving_gradient", False)):
                    margin_target_maps = [
                        mean_preserving_evidence_proxy(scores)
                        for scores in target_maps
                    ]
                    margin_rival_maps = [
                        [mean_preserving_evidence_proxy(scores) for scores in maps]
                        for maps in rival_maps
                    ]
                containment, viability = support_aware_fcal_losses(
                    target_maps,
                    masks,
                    support_temperature=float(v2["support_temperature"]),
                    support_margin=float(v2["support_margin"]),
                    tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                    minimum_tolerance_pixels=int(v2.get("minimum_tolerance_pixels", 0)),
                    maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                    fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                    fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                )
                if v2_plus_gated_active:
                    gated_margin = gated_foreground_competitor_margin_loss(
                        logits,
                        labels,
                        margin_target_maps,
                        margin_rival_maps,
                        masks,
                        evidence_temperature=float(
                            v2_plus_settings.get("evidence_temperature", 1.0)
                        ),
                        margin=float(v2_plus_settings.get("margin", 0.25)),
                        loss_temperature=float(
                            v2_plus_settings.get("loss_temperature", 0.25)
                        ),
                        confidence_floor=float(
                            v2_plus_settings["confidence_floor"]
                        ),
                        confidence_ceiling=float(
                            v2_plus_settings["confidence_ceiling"]
                        ),
                        confidence_gate_temperature=float(
                            v2_plus_settings["confidence_gate_temperature"]
                        ),
                        support_gate_temperature=float(
                            v2_plus_settings["support_gate_temperature"]
                        ),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                    competitor_margin = gated_margin.loss
                    competitor_margin_ungated = gated_margin.ungated_loss
                    competitor_gate = gated_margin.gate
                    competitor_foreground_support = (
                        gated_margin.foreground_support
                    )
                    competitor_target_confidence = gated_margin.target_confidence
                    competitor_valid_mask_fraction = (
                        gated_margin.valid_mask_fraction
                    )
                else:
                    competitor_margin = foreground_competitor_margin_loss(
                        margin_target_maps,
                        margin_rival_maps,
                        masks,
                        evidence_temperature=float(
                            v2_plus_settings.get("evidence_temperature", 1.0)
                        ),
                        margin=float(v2_plus_settings.get("margin", 0.25)),
                        loss_temperature=float(
                            v2_plus_settings.get("loss_temperature", 0.25)
                        ),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                weight = adaptive_lambda(
                    ce, containment, float(fcal["alpha"]), float(fcal["lambda_min"]),
                    float(fcal["lambda_max"]), float(fcal["epsilon"]),
                )
                phase = ramp_weight(
                    epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"])
                )
                weighted_spatial = phase * (
                    weight * containment
                    + float(v2["support_weight"]) * viability
                    + float(v2_plus_settings["coefficient"])
                    * competitor_margin
                )
                spatial = containment + viability + competitor_margin
                total = ce + weighted_spatial
            elif v2_3_active:
                if masks is None:
                    raise RuntimeError("FCAL V2.3 batch is missing masks")
                target_maps = model.raw_class_evidence(labels)
                containment, viability = support_aware_fcal_losses(
                    target_maps,
                    masks,
                    support_temperature=float(v2["support_temperature"]),
                    support_margin=float(v2["support_margin"]),
                    tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                    minimum_tolerance_pixels=int(
                        v2.get("minimum_tolerance_pixels", 0)
                    ),
                    maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                    fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                    fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                )
                strict_containment = strict_positive_containment_loss(
                    target_maps,
                    masks,
                    epsilon=float(v2.get("epsilon", 1e-8)),
                )
                foreground_coverage_hinge = foreground_coverage_hinge_loss(
                    target_maps,
                    masks,
                    target=float(v2_3["foreground_coverage_target"]),
                    power=float(v2_3.get("foreground_coverage_power", 2.0)),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                )
                weight = adaptive_lambda(
                    ce, containment, float(fcal["alpha"]), float(fcal["lambda_min"]),
                    float(fcal["lambda_max"]), float(fcal["epsilon"]),
                )
                phase = ramp_weight(
                    epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"])
                )
                weighted_spatial = phase * (
                    weight * containment
                    + float(v2["support_weight"]) * viability
                    + float(v2_3["strict_containment_weight"])
                    * strict_containment
                    + float(v2_3["foreground_coverage_weight"])
                    * foreground_coverage_hinge
                )
                spatial = (
                    containment + viability + strict_containment
                    + foreground_coverage_hinge
                )
                total = ce + weighted_spatial
            elif v2_4_active:
                if masks is None:
                    raise RuntimeError("FCAL V2.4 batch is missing masks")
                target_maps = model.raw_class_evidence(labels)
                containment, viability = support_aware_fcal_losses(
                    target_maps,
                    masks,
                    support_temperature=float(v2["support_temperature"]),
                    support_margin=float(v2["support_margin"]),
                    tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                    minimum_tolerance_pixels=int(
                        v2.get("minimum_tolerance_pixels", 0)
                    ),
                    maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                    fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                    fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                )
                strict_containment = strict_positive_containment_loss(
                    target_maps,
                    masks,
                    epsilon=float(v2.get("epsilon", 1e-8)),
                )
                seed_grow = seed_and_grow_foreground_loss(
                    target_maps,
                    masks,
                    acceptance_threshold=float(v2_4["acceptance_threshold"]),
                    growth_target=float(v2_4["growth_target"]),
                    evidence_temperature=float(
                        v2_4.get("evidence_temperature", 1.0)
                    ),
                    power=float(v2_4.get("growth_power", 2.0)),
                    foreground_threshold=float(
                        v2_4.get("foreground_threshold", 0.50)
                    ),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                )
                weight = adaptive_lambda(
                    ce, containment, float(fcal["alpha"]), float(fcal["lambda_min"]),
                    float(fcal["lambda_max"]), float(fcal["epsilon"]),
                )
                phase = ramp_weight(
                    epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"])
                )
                weighted_spatial = phase * (
                    weight * containment
                    + float(v2["support_weight"]) * viability
                    + float(v2_4["strict_containment_weight"])
                    * strict_containment
                    + float(v2_4["seed_grow_weight"]) * seed_grow
                )
                spatial = containment + viability + strict_containment + seed_grow
                total = ce + weighted_spatial
            elif v2_next_active:
                if masks is None:
                    raise RuntimeError(f"{method} batch is missing masks")
                channel_containment_weight = float(
                    v2_next.get("channel_containment_weight", 0.0)
                )
                if channel_containment_weight > 0.0:
                    target_maps, positive_weight_maps = (
                        model.raw_and_positive_weight_class_evidence(labels)
                    )
                else:
                    target_maps = model.raw_class_evidence(labels)
                    positive_weight_maps = None
                v2_losses = support_aware_fcal_losses(
                    target_maps,
                    masks,
                    support_temperature=float(v2["support_temperature"]),
                    support_margin=float(v2["support_margin"]),
                    tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                    minimum_tolerance_pixels=int(
                        v2.get("minimum_tolerance_pixels", 0)
                    ),
                    maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                    fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                    fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                    return_backgrounds=channel_containment_weight > 0.0,
                )
                if channel_containment_weight > 0.0:
                    containment, viability, v2_backgrounds = v2_losses
                else:
                    containment, viability = v2_losses
                    v2_backgrounds = None
                front_weight = float(v2_next.get("seed_front_weight", 0.0))
                retention_weight = float(
                    v2_next.get("seed_retention_weight", 0.0)
                )
                if front_weight > 0.0 or retention_weight > 0.0:
                    growth = persistent_seed_and_grow_loss(
                        target_maps,
                        masks,
                        acceptance_threshold=float(
                            v2_next.get("acceptance_threshold", 0.35)
                        ),
                        front_target=float(v2_next.get("front_target", 0.80)),
                        retention_target=float(
                            v2_next.get("retention_target", 0.90)
                        ),
                        evidence_temperature=float(
                            v2_next.get("evidence_temperature", 1.0)
                        ),
                        power=float(v2_next.get("growth_power", 2.0)),
                        foreground_threshold=float(
                            v2_next.get("foreground_threshold", 0.50)
                        ),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                    seed_front = growth.front
                    seed_retention = growth.retention
                feature_consistency_weight = float(
                    v2_next.get("feature_consistency_weight", 0.0)
                )
                if feature_consistency_weight > 0.0:
                    perturbed_logits = model.background_feature_perturbed_logits(
                        masks,
                        float(v2_next.get("background_feature_noise_sigma", 0.1)),
                    )
                    clean_log_probability = F.log_softmax(logits.float(), dim=1)
                    perturbed_log_probability = F.log_softmax(
                        perturbed_logits.float(), dim=1,
                    )
                    clean_probability = clean_log_probability.exp()
                    background_feature_consistency = (
                        clean_probability
                        * (clean_log_probability - perturbed_log_probability)
                    ).sum(dim=1).mean()
                dominance_weight = float(
                    v2_next.get("native_dominance_weight", 0.0)
                )
                if dominance_weight > 0.0:
                    native_dominance = native_regional_dominance_loss(
                        target_maps,
                        masks,
                        margin=float(v2_next.get("dominance_margin", 0.60)),
                        power=float(v2_next.get("dominance_power", 2.0)),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                foreground_region_weight = float(
                    v2_next.get("foreground_region_weight", 0.0)
                )
                background_region_weight = float(
                    v2_next.get("background_region_weight", 0.0)
                )
                adaptive_regional = bool(
                    v2_next.get("adaptive_regional_weight", False)
                )
                if (
                    foreground_region_weight > 0.0
                    or background_region_weight > 0.0
                    or adaptive_regional
                ):
                    foreground_regions = masks
                    background_regions = None
                    if adaptive_regional:
                        input_size = (int(masks.shape[-2]), int(masks.shape[-1]))
                        map_size = (
                            int(target_maps[-1].shape[-2]),
                            int(target_maps[-1].shape[-1]),
                        )
                        buffer_pixels = resolution_aware_tolerance_pixels(
                            input_size,
                            map_size,
                            cell_fraction=float(
                                v2_next.get(
                                    "regional_buffer_cell_fraction",
                                    v2["tolerance_cell_fraction"],
                                )
                            ),
                            minimum_pixels=int(
                                v2_next.get("regional_buffer_minimum_pixels", 0)
                            ),
                            maximum_pixels=v2_next.get(
                                "regional_buffer_maximum_pixels",
                                v2.get("maximum_tolerance_pixels"),
                            ),
                        )
                        foreground_regions, background_regions = buffered_region_masks(
                            masks, buffer_pixels,
                        )
                    regional = regional_classification_losses(
                        model,
                        labels,
                        foreground_regions,
                        foreground_margin=float(
                            v2_next.get("foreground_logit_margin", 0.0)
                        ),
                        maximum_background_margin=float(
                            v2_next.get("maximum_background_logit_margin", -2.0)
                        ),
                        temperature=float(
                            v2_next.get("regional_temperature", 1.0)
                        ),
                        compute_background=(
                            background_region_weight > 0.0 or adaptive_regional
                        ),
                        background_regions=background_regions,
                        exclude_empty=adaptive_regional,
                    )
                    foreground_region_classification = regional.foreground
                    background_region_suppression = regional.background
                local_margin_weight = float(
                    v2_next.get("local_foreground_rival_weight", 0.0)
                )
                if local_margin_weight > 0.0:
                    competitors = top_competitor_labels(
                        logits, labels, int(v2_next.get("local_top_k", 3))
                    )
                    rival_maps = [
                        model.raw_class_evidence(competitors[:, rank])
                        for rank in range(competitors.shape[1])
                    ]
                    local_loss_function = (
                        softmin_foreground_rival_margin_loss
                        if v2_next.get("local_selection") == "softmin"
                        else local_foreground_rival_margin_loss
                    )
                    local_options = {
                        "margin": float(v2_next.get("local_margin", 0.0)),
                        "evidence_temperature": float(
                            v2_next.get("local_evidence_temperature", 1.0)
                        ),
                        "loss_temperature": float(
                            v2_next.get("local_loss_temperature", 0.25)
                        ),
                        "epsilon": float(v2.get("epsilon", 1e-8)),
                    }
                    if local_loss_function is softmin_foreground_rival_margin_loss:
                        local_options["softmin_temperature"] = float(
                            v2_next.get("local_softmin_temperature", 0.25)
                        )
                    else:
                        local_options["lower_tail_fraction"] = float(
                            v2_next.get("local_lower_tail_fraction", 0.25)
                        )
                        local_options["foreground_threshold"] = float(
                            v2_next.get("foreground_threshold", 0.50)
                        )
                    local_foreground_rival_margin = local_loss_function(
                        target_maps,
                        rival_maps,
                        masks,
                        **local_options,
                    )
                if channel_containment_weight > 0.0:
                    channel_containment = uncancelled_channel_containment_loss(
                        positive_weight_maps,
                        masks,
                        tolerance_cell_fraction=float(
                            v2["tolerance_cell_fraction"]
                        ),
                        minimum_tolerance_pixels=int(
                            v2.get("minimum_tolerance_pixels", 0)
                        ),
                        maximum_tolerance_pixels=v2.get(
                            "maximum_tolerance_pixels"
                        ),
                        background_maps=v2_backgrounds,
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                weight = adaptive_lambda(
                    ce, containment, float(fcal["alpha"]),
                    float(fcal["lambda_min"]), float(fcal["lambda_max"]),
                    float(fcal["epsilon"]),
                )
                phase = ramp_weight(
                    epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"])
                )
                if adaptive_regional:
                    regional_objective = (
                        foreground_region_classification
                        + background_region_suppression
                    )
                    regional_gamma = adaptive_lambda(
                        ce,
                        regional_objective,
                        float(fcal["alpha"]),
                        float(fcal["lambda_min"]),
                        float(fcal["lambda_max"]),
                        float(fcal["epsilon"]),
                    )
                    regional_term = regional_gamma * regional_objective
                else:
                    regional_term = (
                        foreground_region_weight * foreground_region_classification
                        + background_region_weight * background_region_suppression
                    )
                weighted_regional = phase * regional_term
                feature_consistency_term = (
                    feature_consistency_weight * background_feature_consistency
                )
                weighted_feature_consistency = phase * feature_consistency_term
                weighted_spatial = phase * (
                    weight * containment
                    + float(v2["support_weight"]) * viability
                    + front_weight * seed_front
                    + retention_weight * seed_retention
                    + dominance_weight * native_dominance
                    + regional_term
                    + local_margin_weight * local_foreground_rival_margin
                    + channel_containment_weight * channel_containment
                    + feature_consistency_term
                )
                spatial = (
                    containment + viability + seed_front + seed_retention
                    + native_dominance + foreground_region_classification
                    + background_region_suppression
                    + local_foreground_rival_margin + channel_containment
                    + background_feature_consistency
                )
                total = ce + weighted_spatial
            elif v3_active:
                if masks is None:
                    raise RuntimeError("FCAL V3 batch is missing masks")
                clean_maps = model.raw_class_evidence(labels)
                configured_foreground_fraction = float(
                    v3.get("foreground_intervention_fraction", 0.0)
                )
                if configured_foreground_fraction > 0.0:
                    (
                        counterfactual_images,
                        evidence_consistency_samples,
                        donor_mismatch_fraction,
                    ) = balanced_regional_counterfactual(
                        images,
                        masks,
                        labels,
                        foreground_fraction=configured_foreground_fraction,
                        boundary_guard_pixels=int(v3.get("boundary_guard_pixels", 4)),
                        foreground_noise_scale=float(v3.get("foreground_noise_scale", 0.20)),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                    foreground_intervention_fraction = (
                        ~evidence_consistency_samples
                    ).float().mean()
                else:
                    counterfactual_images, _, donor_mismatch_fraction = background_counterfactual(
                        images,
                        masks,
                        labels,
                        boundary_guard_pixels=int(v3.get("boundary_guard_pixels", 4)),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                    evidence_consistency_samples = None
                counterfactual_logits = model(counterfactual_images, capture_attention=False)
                counterfactual_maps = model.raw_class_evidence(labels)
                counterfactual_ce = F.cross_entropy(counterfactual_logits, labels)
                phase = ramp_weight(epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"]))
                counterfactual_ce_weight = phase * float(v3["counterfactual_ce_weight"])
                effective_ce = (
                    ce + counterfactual_ce_weight * counterfactual_ce
                ) / (1.0 + counterfactual_ce_weight)
                containment, viability, evidence_consistency = counterfactual_support_losses(
                    clean_maps,
                    counterfactual_maps,
                    masks,
                    support_temperature=float(v2["support_temperature"]),
                    support_margin=float(v2["support_margin"]),
                    tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                    evidence_consistency_temperature=float(
                        v3.get("evidence_consistency_temperature", 1.0)
                    ),
                    evidence_consistency_samples=evidence_consistency_samples,
                    minimum_tolerance_pixels=int(v2.get("minimum_tolerance_pixels", 0)),
                    maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                )
                prediction_consistency = prediction_consistency_loss(
                    logits,
                    counterfactual_logits,
                    temperature=float(v3.get("prediction_temperature", 2.0)),
                    valid_samples=(
                        evidence_consistency_samples
                        if bool(v3.get("route_consistency_to_background", False))
                        else None
                    ),
                )
                strict_shell = 0.5 * (
                    strict_shell_containment(
                        clean_maps,
                        masks,
                        tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                        minimum_tolerance_pixels=int(
                            v2.get("minimum_tolerance_pixels", 0)
                        ),
                        maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                        fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                        fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                    + strict_shell_containment(
                        counterfactual_maps,
                        masks,
                        tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                        minimum_tolerance_pixels=int(
                            v2.get("minimum_tolerance_pixels", 0)
                        ),
                        maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                        fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                        fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                )
                weight = adaptive_lambda(
                    effective_ce,
                    containment,
                    float(fcal["alpha"]),
                    float(fcal["lambda_min"]),
                    float(fcal["lambda_max"]),
                    float(fcal["epsilon"]),
                )
                weighted_spatial = phase * (
                    weight * containment
                    + float(v2["support_weight"]) * viability
                    + float(v3["prediction_consistency_weight"]) * prediction_consistency
                    + float(v3["evidence_consistency_weight"]) * evidence_consistency
                    + float(v3.get("strict_shell_weight", 0.0)) * strict_shell
                )
                spatial = (
                    containment + viability + prediction_consistency
                    + evidence_consistency + strict_shell
                )
                total = effective_ce + weighted_spatial
            elif v4_active:
                if masks is None:
                    raise RuntimeError("FCAL V4 batch is missing masks")
                clean_maps = model.raw_class_evidence(labels)
                intervention = v4.get("intervention", {})
                intervention_enabled = bool(intervention.get("enabled", False))
                prediction_consistency = ce.new_zeros(())
                evidence_consistency = ce.new_zeros(())
                if intervention_enabled:
                    foreground_fraction = float(
                        intervention.get("foreground_intervention_fraction", 0.5)
                    )
                    (
                        counterfactual_images,
                        background_samples,
                        donor_mismatch_fraction,
                    ) = balanced_regional_counterfactual(
                        images,
                        masks,
                        labels,
                        foreground_fraction=foreground_fraction,
                        boundary_guard_pixels=int(
                            intervention.get("boundary_guard_pixels", 4)
                        ),
                        foreground_noise_scale=float(
                            intervention.get("foreground_noise_scale", 0.20)
                        ),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                    foreground_intervention_fraction = (
                        ~background_samples
                    ).float().mean()
                    counterfactual_logits = model(
                        counterfactual_images, capture_attention=False
                    )
                    counterfactual_maps = model.raw_class_evidence(labels)
                    counterfactual_ce = F.cross_entropy(counterfactual_logits, labels)
                    cf_weight = float(intervention.get("counterfactual_ce_weight", 0.25))
                    effective_ce = (ce + cf_weight * counterfactual_ce) / (1.0 + cf_weight)
                    containment, viability, evidence_consistency = (
                        counterfactual_support_losses(
                            clean_maps,
                            counterfactual_maps,
                            masks,
                            support_temperature=float(v2["support_temperature"]),
                            support_margin=float(v2["support_margin"]),
                            tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                            evidence_consistency_temperature=float(
                                intervention.get("evidence_consistency_temperature", 1.0)
                            ),
                            evidence_consistency_samples=background_samples,
                            minimum_tolerance_pixels=int(
                                v2.get("minimum_tolerance_pixels", 0)
                            ),
                            maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                            epsilon=float(v2.get("epsilon", 1e-8)),
                        )
                    )
                    prediction_consistency = prediction_consistency_loss(
                        logits,
                        counterfactual_logits,
                        temperature=float(intervention.get("prediction_temperature", 2.0)),
                        valid_samples=background_samples,
                    )
                else:
                    containment, viability = support_aware_fcal_losses(
                        clean_maps,
                        masks,
                        support_temperature=float(v2["support_temperature"]),
                        support_margin=float(v2["support_margin"]),
                        tolerance_cell_fraction=float(v2["tolerance_cell_fraction"]),
                        minimum_tolerance_pixels=int(
                            v2.get("minimum_tolerance_pixels", 0)
                        ),
                        maximum_tolerance_pixels=v2.get("maximum_tolerance_pixels"),
                        fixed_tolerance_pixels=v2.get("fixed_tolerance_pixels"),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                        fixed_tolerance_cells=v2.get("fixed_tolerance_cells"),
                    )
                strict_containment = strict_positive_containment(
                    clean_maps, masks, epsilon=float(v2.get("epsilon", 1e-8))
                )
                mask_alignment = mask_alignment_loss(
                    clean_maps,
                    masks,
                    temperature=float(v4.get("alignment_temperature", 1.0)),
                    mode=str(v4.get("alignment_mode", "rms_tanh")),
                    epsilon=float(v2.get("epsilon", 1e-8)),
                )
                if float(v4.get("regional_dominance_weight", 0.0)) > 0:
                    regional_dominance = regional_dominance_loss(
                        clean_maps,
                        masks,
                        margin=float(v4.get("regional_dominance_margin", 0.6)),
                        power=float(v4.get("regional_dominance_power", 2.0)),
                        epsilon=float(v2.get("epsilon", 1e-8)),
                    )
                if float(v4.get("background_sensitivity_weight", 0.0)) > 0:
                    background_sensitivity = background_margin_sensitivity(
                        logits,
                        labels,
                        images,
                        masks,
                        epsilon=float(v2.get("epsilon", 1e-8)),
                        power=float(v4.get("sensitivity_power", 1.0)),
                    )
                sensitivity_phase = 1.0
                if "sensitivity_warmup_epochs" in v4:
                    sensitivity_phase = ramp_weight(
                        epoch,
                        int(v4["sensitivity_warmup_epochs"]),
                        int(v4.get("sensitivity_ramp_epochs", 1)),
                    )
                phase = ramp_weight(
                    epoch, int(v2["warmup_epochs"]), int(v2["ramp_epochs"])
                )
                weight = adaptive_lambda(
                    effective_ce,
                    containment,
                    float(fcal["alpha"]),
                    float(fcal["lambda_min"]),
                    float(fcal["lambda_max"]),
                    float(fcal["epsilon"]),
                )
                weighted_spatial = phase * (
                    weight * containment
                    + float(v2["support_weight"]) * viability
                    + float(v4["strict_containment_weight"]) * strict_containment
                    + float(v4["mask_alignment_weight"]) * mask_alignment
                    + float(v4.get("regional_dominance_weight", 0.0))
                    * regional_dominance
                    + float(v4["background_sensitivity_weight"])
                    * sensitivity_phase * background_sensitivity
                    + float(intervention.get("prediction_consistency_weight", 0.0))
                    * prediction_consistency
                    + float(intervention.get("evidence_consistency_weight", 0.0))
                    * evidence_consistency
                )
                spatial = (
                    containment + viability + strict_containment + mask_alignment
                    + regional_dominance
                    + background_sensitivity + prediction_consistency
                    + evidence_consistency
                )
                total = effective_ce + weighted_spatial
            elif v5_active:
                if masks is None:
                    raise RuntimeError("FCAL V5 batch is missing masks")
                result = regional_margin_certificate_loss(
                    model,
                    logits,
                    labels,
                    masks,
                    top_k_competitors=int(v5.get("top_k_competitors", 2)),
                    foreground_temperature=float(
                        v5.get("foreground_temperature", 0.5)
                    ),
                    background_temperature=float(
                        v5.get("background_temperature", 0.5)
                    ),
                    background_scale=float(v5.get("background_scale", 0.5)),
                    margin=float(v5.get("margin", 0.0)),
                    temperature=float(v5.get("temperature", 0.25)),
                    epsilon=float(v5.get("epsilon", 1e-8)),
                )
                certificate_loss = result.loss
                certificate = result.certificate
                certificate_foreground_tail = result.foreground_tail
                certificate_background_tail = result.background_tail
                phase = ramp_weight(
                    epoch,
                    int(v5.get("warmup_epochs", 5)),
                    int(v5.get("ramp_epochs", 5)),
                )
                spatial = certificate_loss
                weight = ce.new_tensor(float(v5.get("coefficient", 0.05)))
                weighted_spatial = phase * weight * spatial
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
            maximum_gradient_ratio = float(ufcal_runtime["gradient_ratio"]) * phase
            project_conflicts = bool(config["ufcal"].get("project_conflicts", True))
            gradient_epsilon = float(config["ufcal"].get("epsilon", 1e-8))
            gradient_stats = gradient_safe_backward(
                ce,
                weighted_spatial,
                model.parameters(),
                maximum_gradient_ratio=maximum_gradient_ratio,
                accumulation_steps=accumulation,
                project_conflicts=project_conflicts,
                epsilon=gradient_epsilon,
            )
            gradient_ratio = gradient_stats.ratio_after
            gradient_cosine = gradient_stats.cosine_before
            conflict_fraction = float(gradient_stats.conflict_projected)
        else:
            (total / accumulation).backward()
        nonfinite_gradient_fraction = 0.0
        if method == "isnet" and bool(config["isnet"].get("sanitize_nonfinite_gradients", False)):
            bad = torch.zeros((), device=device, dtype=torch.long)
            gradient_elements = 0
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                finite = torch.isfinite(parameter.grad)
                bad = bad + (~finite).sum()
                gradient_elements += parameter.grad.numel()
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            nonfinite_gradient_fraction = float(bad) / max(1, gradient_elements)
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        if should_step:
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["grad_clip_norm"]),
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                nonfinite = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                ]
                raise FloatingPointError(
                    f"Non-finite gradient for method={method} epoch={epoch + 1} "
                    f"batch={batch_index + 1} ce={float(ce.detach())} "
                    f"spatial={float(spatial.detach())} total={float(total.detach())} "
                    f"parameters={nonfinite[:12]}"
                ) from error
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        batch_size = labels.shape[0]
        count += batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        sums["ce"] += float(ce.detach()) * batch_size
        sums["effective_ce"] += float(effective_ce.detach()) * batch_size
        sums["fcal"] += float(spatial.detach()) * batch_size
        sums["containment"] += float(containment.detach()) * batch_size
        sums["viability"] += float(viability.detach()) * batch_size
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
        sums["attention_mining"] += float(attention_mining.detach()) * batch_size
        sums["attention_supervision"] += float(attention_supervision.detach()) * batch_size
        sums["attention_inside"] += float(attention_inside.detach()) * batch_size
        sums["attention_outside"] += float(attention_outside.detach()) * batch_size
        sums["counterfactual_ce"] += float(counterfactual_ce.detach()) * batch_size
        sums["prediction_consistency"] += float(prediction_consistency.detach()) * batch_size
        sums["donor_mismatch_fraction"] += float(donor_mismatch_fraction.detach()) * batch_size
        sums["foreground_intervention_fraction"] += float(
            foreground_intervention_fraction.detach()
        ) * batch_size
        sums["strict_shell"] += float(strict_shell.detach()) * batch_size
        sums["strict_containment"] += float(strict_containment.detach()) * batch_size
        sums["mask_alignment"] += float(mask_alignment.detach()) * batch_size
        sums["regional_dominance"] += float(regional_dominance.detach()) * batch_size
        sums["background_sensitivity"] += float(
            background_sensitivity.detach()
        ) * batch_size
        sums["certificate_loss"] += float(certificate_loss.detach()) * batch_size
        sums["certificate"] += float(certificate.detach()) * batch_size
        sums["certificate_foreground_tail"] += float(
            certificate_foreground_tail.detach()
        ) * batch_size
        sums["certificate_background_tail"] += float(
            certificate_background_tail.detach()
        ) * batch_size
        sums["competitor_margin"] += float(competitor_margin.detach()) * batch_size
        sums["foreground_coverage_hinge"] += float(
            foreground_coverage_hinge.detach()
        ) * batch_size
        sums["seed_grow"] += float(seed_grow.detach()) * batch_size
        sums["seed_front"] += float(seed_front.detach()) * batch_size
        sums["seed_retention"] += float(seed_retention.detach()) * batch_size
        sums["native_dominance"] += float(native_dominance.detach()) * batch_size
        sums["foreground_region_classification"] += float(
            foreground_region_classification.detach()
        ) * batch_size
        sums["background_region_suppression"] += float(
            background_region_suppression.detach()
        ) * batch_size
        sums["regional_gamma"] += float(regional_gamma.detach()) * batch_size
        sums["weighted_regional"] += float(weighted_regional.detach()) * batch_size
        sums["background_feature_consistency"] += float(
            background_feature_consistency.detach()
        ) * batch_size
        sums["weighted_feature_consistency"] += float(
            weighted_feature_consistency.detach()
        ) * batch_size
        sums["local_foreground_rival_margin"] += float(
            local_foreground_rival_margin.detach()
        ) * batch_size
        sums["channel_containment"] += float(
            channel_containment.detach()
        ) * batch_size
        sums["relative_pooled_margin"] += float(
            relative_pooled_margin.detach()
        ) * batch_size
        sums["shortcut_gate"] += float(shortcut_gate.detach()) * batch_size
        sums["regional_foreground_margin"] += float(
            regional_foreground_margin.detach()
        ) * batch_size
        sums["regional_background_margin"] += float(
            regional_background_margin.detach()
        ) * batch_size
        sums["regional_margin_gap"] += float(
            regional_margin_gap.detach()
        ) * batch_size
        sums["relative_outside_evidence"] += float(
            relative_outside_evidence.detach()
        ) * batch_size
        sums["relative_valid_mask_fraction"] += float(
            relative_valid_mask_fraction.detach()
        ) * batch_size
        sums["competitor_margin_ungated"] += float(
            competitor_margin_ungated.detach()
        ) * batch_size
        sums["competitor_gate"] += float(competitor_gate.detach()) * batch_size
        sums["competitor_foreground_support"] += float(
            competitor_foreground_support.detach()
        ) * batch_size
        sums["competitor_target_confidence"] += float(
            competitor_target_confidence.detach()
        ) * batch_size
        sums["competitor_valid_mask_fraction"] += float(
            competitor_valid_mask_fraction.detach()
        ) * batch_size
        sums["nonfinite_gradient_fraction"] += nonfinite_gradient_fraction * batch_size
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
    group_logits: dict[int, torch.Tensor] = {}
    group_counts: dict[int, int] = {}
    group_labels: dict[int, int] = {}
    started = time.perf_counter()
    for batch in loader:
        images, labels, _ = _move(batch, device, channels_last)
        with _amp_context(device, amp):
            logits = model(images, capture_attention=False)
            ce = F.cross_entropy(logits, labels, reduction="sum")
        ce_sum += float(ce)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.shape[0]
        if "group_id" in batch:
            for group_id, label, item_logits in zip(
                batch["group_id"].tolist(), labels.tolist(), logits.float().cpu(), strict=True,
            ):
                group_id = int(group_id)
                if group_id in group_labels and group_labels[group_id] != int(label):
                    raise ValueError(f"Group {group_id} has inconsistent labels")
                group_labels[group_id] = int(label)
                group_logits[group_id] = group_logits.get(
                    group_id, torch.zeros_like(item_logits),
                ) + item_logits
                group_counts[group_id] = group_counts.get(group_id, 0) + 1
    elapsed = time.perf_counter() - started
    result = {
        "ce": ce_sum / count,
        "accuracy": correct / count,
        "samples": count,
        "seconds": elapsed,
        "samples_per_second": count / elapsed,
    }
    if group_logits:
        ordered = sorted(group_logits)
        averaged = torch.stack(
            [group_logits[key] / group_counts[key] for key in ordered], dim=0,
        )
        patient_labels = torch.tensor([group_labels[key] for key in ordered])
        result.update({
            "slice_ce": result["ce"],
            "slice_accuracy": result["accuracy"],
            "ce": float(F.cross_entropy(averaged, patient_labels)),
            "accuracy": float((averaged.argmax(dim=1) == patient_labels).float().mean()),
            "patients": len(ordered),
        })
    return result


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
    seed: int | None = None,
    allow_data_relocation: bool = False,
    parent_split_dir: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    if method:
        config["method"] = method
    if run_id:
        config["run_id"] = run_id
    if seed is not None:
        config["seed"] = seed
    validate_evidence_reference(config)
    if config["method"] not in {
        "ce", "fcal", "fcal_v2", "fcal_v2_plus", "fcal_v2_plus_gated", "fcal_v2_relative", "fcal_v2_3", "fcal_v2_4",
        "fcal_v2_5", "fcal_v2_6", "fcal_v2_7", "fcal_v2_8", "fcal_v2_9",
        "fcal_v2_7a", "fcal_v2_7b", "fcal_v2_7c",
        "fcal_v2_7d", "fcal_v2_7_prime", "fcal_v2_feature_consistency",
        "fcal_v3", "fcal_v4", "fcal_v5_cheap",
        "ufcal", "rrr", "gain",
        "isnet", "camal",
    }:
        raise ValueError(
            "method must be ce, fcal, fcal_v2, fcal_v2_plus, fcal_v2_plus_gated, fcal_v2_relative, fcal_v2_3, fcal_v2_4, "
            "fcal_v2_5, fcal_v2_6, fcal_v2_7, fcal_v2_8, fcal_v2_9, "
            "fcal_v2_7a, fcal_v2_7b, fcal_v2_7c, fcal_v2_7d, fcal_v2_7_prime, "
            "fcal_v2_feature_consistency, fcal_v3, fcal_v4, "
            "fcal_v5_cheap, ufcal, rrr, gain, isnet, or camal"
        )
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
    need_mask = config["method"] in {
        "fcal", "fcal_v2", "fcal_v2_plus", "fcal_v2_plus_gated", "fcal_v2_relative", "fcal_v2_3", "fcal_v2_4",
        "fcal_v2_5", "fcal_v2_6", "fcal_v2_7", "fcal_v2_8", "fcal_v2_9",
        "fcal_v2_7a", "fcal_v2_7b", "fcal_v2_7c",
        "fcal_v2_7d", "fcal_v2_7_prime", "fcal_v2_feature_consistency",
        "fcal_v3", "fcal_v4", "fcal_v5_cheap", "ufcal",
        "rrr", "gain",
        "isnet", "camal",
    }
    train_loader = make_loader(config, "train", need_mask=need_mask)
    validation_loader = make_loader(config, "validation", need_mask=False, shuffle=False)
    start_epoch = 0
    best_accuracy = -math.inf
    history_path = run_dir / "history.jsonl"
    if checkpoint:
        parent_data = checkpoint["config"]["data"]
        exact_data_match = (
            _training_data_config(parent_data) == _training_data_config(config["data"])
        )
        relocated_data_match = (
            allow_data_relocation
            and _relocated_training_data_matches(
                parent_data,
                config["data"],
                parent_split_dir,
                Path(resume).resolve().parent / "environment.json",
            )
        )
        if checkpoint["config"]["model"] != config["model"] or not (
            exact_data_match or relocated_data_match
        ):
            raise ValueError("Resume checkpoint model/data does not match current config")
        if relocated_data_match and not exact_data_match:
            print(
                "Accepted explicit data relocation after matching all loader settings "
                "and prepared split hashes.",
                flush=True,
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Optimizer moments are part of the paired branch, but implementation
        # flags are properties of the child run (ISNet requires non-fused,
        # full-precision higher-order gradients). ``load_state_dict`` restores
        # the parent's fused flag unless it is explicitly reapplied here.
        for parameter_group in optimizer.param_groups:
            parameter_group["fused"] = fused
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        same_method_resume = checkpoint["config"].get("method") == config["method"]
        best_accuracy = (
            float(checkpoint["best_validation_accuracy"])
            if same_method_resume
            else -math.inf
        )
        _restore_rng(checkpoint["rng"], train_loader)
        if not history_path.exists():
            _seed_resume_history(resume, history_path, start_epoch)
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
        "last_5_epoch_average": _last_epoch_averages(history_path, 5),
    }
    atomic_json(run_dir / "summary.json", summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--method",
        choices=(
            "ce", "fcal", "fcal_v2", "fcal_v2_plus", "fcal_v2_plus_gated", "fcal_v2_relative", "fcal_v2_3", "fcal_v2_4",
            "fcal_v2_5", "fcal_v2_6", "fcal_v2_7", "fcal_v2_8", "fcal_v2_9",
            "fcal_v2_7a", "fcal_v2_7b", "fcal_v2_7c",
            "fcal_v2_7d", "fcal_v2_7_prime", "fcal_v2_feature_consistency",
            "fcal_v3", "fcal_v4", "fcal_v5_cheap", "ufcal",
            "rrr", "gain", "isnet", "camal",
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-data-relocation",
        action="store_true",
        help="accept relocated parent data paths only after prepared split hash verification",
    )
    parser.add_argument(
        "--parent-split-dir",
        help=(
            "explicit current location of an archived parent's prepared split; "
            "used only with --allow-data-relocation and still hash-verified"
        ),
    )
    parser.add_argument("--seed", type=int, help="override the training seed in the config")
    args = parser.parse_args()
    print(run(
        args.config, args.method, args.run_id, args.resume, args.seed,
        args.allow_data_relocation,
        args.parent_split_dir,
    ))


if __name__ == "__main__":
    main()
