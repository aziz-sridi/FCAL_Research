from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import PROJECT_ROOT
from .data import make_multi_variant_loader
from .losses import article_tolerance_mask, normalized_attention, smooth_normalized_attention
from .models import ResNet50CAM, create_model
from .utils import atomic_json
from .variants import parse_variant


DEFAULT_CAM_VARIANTS = [
    "clean",
    "bg_blur:3", "bg_color_jitter:3", "bg_gaussian_noise:3", "bg_random_rgb:1",
    "fg_blur:3", "fg_color_jitter:3", "fg_gaussian_noise:3", "fg_random_rgb:1",
]


def _strict_leakage(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    tolerance = article_tolerance_mask(mask, attention.shape[-2:], 0)
    return (normalized_attention(attention) * (1.0 - tolerance)).flatten(1).sum(dim=1)


def _smooth_leakage(raw_attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    tolerance = article_tolerance_mask(mask, raw_attention.shape[-2:], 0)
    return (smooth_normalized_attention(raw_attention) * (1.0 - tolerance)).flatten(1).sum(dim=1)


@torch.inference_mode()
def run(checkpoint_path: str, variants: list[str]) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["data"].setdefault("evaluation_cache", str(PROJECT_ROOT / "cache" / "cub_test_224_center"))
    model = create_model(config, pretrained=False)
    if not isinstance(model, ResNet50CAM):
        raise TypeError("CAM diagnostics currently require ResNet50CAM")
    model.load_state_dict(checkpoint["model"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if config["training"].get("channels_last"):
        model.to(memory_format=torch.channels_last)
    specs = [parse_variant(value) for value in variants]
    loader = make_multi_variant_loader(config, "test", specs, batch_size=32)
    stores = {spec.key: [] for spec in specs}
    amp = device.type == "cuda" and config["training"].get("amp", "off") != "off"
    amp_dtype = torch.bfloat16 if config["training"].get("amp") == "bf16" else torch.float16
    for batch in loader:
        key = batch["variant_key"][0]
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if config["training"].get("channels_last"):
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            logits = model(images, capture_attention=True)
            predictions = logits.argmax(dim=1)
            raw_true = model.raw_evaluation_attention(labels).float()
            raw_predicted = model.raw_evaluation_attention(predictions).float()
        true_cam = F.relu(raw_true)
        predicted_cam = F.relu(raw_predicted)
        true_mass = true_cam.flatten(1).sum(dim=1)
        predicted_mass = predicted_cam.flatten(1).sum(dim=1)
        true_leakage = _strict_leakage(true_cam, masks)
        predicted_leakage = _strict_leakage(predicted_cam, masks)
        true_leakage = torch.where(true_mass > 1e-8, true_leakage, torch.full_like(true_leakage, torch.nan))
        predicted_leakage = torch.where(
            predicted_mass > 1e-8, predicted_leakage, torch.full_like(predicted_leakage, torch.nan)
        )
        stores[key].append({
            "image_id": batch["image_id"].numpy().astype(np.int32),
            "label": labels.cpu().numpy().astype(np.int16),
            "prediction": predictions.cpu().numpy().astype(np.int16),
            "true_cam_nonzero": (true_mass > 1e-8).cpu().numpy().astype(np.uint8),
            "predicted_cam_nonzero": (predicted_mass > 1e-8).cpu().numpy().astype(np.uint8),
            "true_cam_positive_cells": (true_cam > 0).flatten(1).sum(dim=1).cpu().numpy().astype(np.int16),
            "predicted_cam_positive_cells": (predicted_cam > 0).flatten(1).sum(dim=1).cpu().numpy().astype(np.int16),
            "true_cam_mass": true_mass.cpu().numpy().astype(np.float32),
            "predicted_cam_mass": predicted_mass.cpu().numpy().astype(np.float32),
            "true_cam_raw_max": raw_true.flatten(1).max(dim=1).values.cpu().numpy().astype(np.float32),
            "true_cam_raw_min": raw_true.flatten(1).min(dim=1).values.cpu().numpy().astype(np.float32),
            "true_cam_strict_leakage": true_leakage.cpu().numpy().astype(np.float32),
            "predicted_cam_strict_leakage": predicted_leakage.cpu().numpy().astype(np.float32),
            "smooth_true_cam_strict_leakage": _smooth_leakage(raw_true, masks).cpu().numpy().astype(np.float32),
        })
    output = Path(checkpoint["run_dir"]) / "cam_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in specs:
        payload = {key: np.concatenate([part[key] for part in stores[spec.key]]) for key in stores[spec.key][0]}
        np.savez_compressed(output / f"{spec.key}.npz", **payload)
        summary = {
            "variant": spec.key,
            "samples": int(len(payload["image_id"])),
            "accuracy": float((payload["prediction"] == payload["label"]).mean()),
            "true_cam_zero_fraction": float(1.0 - payload["true_cam_nonzero"].mean()),
            "predicted_cam_zero_fraction": float(1.0 - payload["predicted_cam_nonzero"].mean()),
            "mean_true_cam_leakage_valid_only": float(np.nanmean(payload["true_cam_strict_leakage"])),
            "mean_predicted_cam_leakage_valid_only": float(np.nanmean(payload["predicted_cam_strict_leakage"])),
            "mean_smooth_true_cam_leakage": float(payload["smooth_true_cam_strict_leakage"].mean()),
        }
        atomic_json(output / f"{spec.key}.json", summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    atomic_json(output / "summary.json", summaries)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_CAM_VARIANTS)
    args = parser.parse_args()
    print(run(args.checkpoint, args.variants))


if __name__ == "__main__":
    main()
