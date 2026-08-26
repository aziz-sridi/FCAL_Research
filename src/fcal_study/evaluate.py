from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from universal_fcal import measure_maps

from .config import PROJECT_ROOT
from .data import make_multi_variant_loader
from .losses import leakage, normalized_attention
from .models import create_model
from .utils import atomic_json
from .variants import DEFAULT_VARIANTS, parse_variant


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict:
    recalls = []
    f1_scores = []
    per_class_recall = {}
    for class_index in range(num_classes):
        truth = labels == class_index
        predicted = predictions == class_index
        true_positive = int((truth & predicted).sum())
        false_positive = int((~truth & predicted).sum())
        false_negative = int((truth & ~predicted).sum())
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class_recall[str(class_index)] = recall
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_recall": per_class_recall,
    }


@torch.inference_mode()
def evaluate_all(
    model, config: dict, variants: list[str], output_dir: Path, split: str = "test",
) -> list[dict]:
    specs = [parse_variant(value) for value in variants]
    # Evaluation has no backward graph, so a larger batch improves accelerator
    # occupancy during the 30-variant benchmark without changing predictions.
    batch_size = int(config.get("evaluation", {}).get("batch_size", 64))
    loader = make_multi_variant_loader(config, split, specs, batch_size=batch_size)
    device = next(model.parameters()).device
    amp_mode = config["training"]["amp"]
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    stores = {
        spec.key: {
            "ids": [], "labels": [], "predictions": [], "confidence": [], "probabilities": [],
            "strict": [], "tolerant": [], "nonzero": [], "ce_sum": 0.0, "correct": 0, "count": 0,
            "mask_valid": [],
        }
        for spec in specs
    }
    for batch in loader:
        key = batch["variant_key"][0]
        if any(value != key for value in batch["variant_key"]):
            raise RuntimeError("A multi-variant batch crossed a variant boundary")
        capture = key == "clean"
        store = stores[key]
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if config["training"]["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        context = torch.autocast("cuda", dtype=amp_dtype) if device.type == "cuda" and amp_mode != "off" else torch.no_grad()
        with context:
            logits = model(images, capture_attention=capture)
        logits_float = logits.float()
        probs = logits_float.softmax(dim=1)
        pred = probs.argmax(dim=1)
        store["ce_sum"] += float(F.cross_entropy(logits_float, labels, reduction="sum"))
        store["correct"] += int((pred == labels).sum())
        store["count"] += labels.shape[0]
        store["ids"].append(batch["image_id"].numpy().astype(np.int32))
        store["labels"].append(labels.cpu().numpy().astype(np.int16))
        store["predictions"].append(pred.cpu().numpy().astype(np.int16))
        store["confidence"].append(probs.max(dim=1).values.cpu().numpy().astype(np.float32))
        store["probabilities"].append(probs.cpu().numpy().astype(np.float16))
        if capture:
            flat_masks = masks.flatten(1)
            mask_valid = flat_masks.any(dim=1) & ~flat_masks.all(dim=1)
            attention = model.evaluation_attention(labels).float()
            store["strict"].append(leakage(attention, masks, 0).cpu().numpy().astype(np.float32))
            if config["method"] == "ufcal":
                settings = config["ufcal"]
                tolerant = measure_maps(
                    attention,
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
                ).outside
            else:
                tolerant = leakage(
                    attention, masks, int(config["fcal"]["dilation_cells"]),
                )
            store["tolerant"].append(tolerant.cpu().numpy().astype(np.float32))
            mass = attention.flatten(1).sum(dim=1)
            epsilon = float(
                config.get("ufcal", {}).get(
                    "epsilon", config.get("fcal", {}).get("epsilon", 1e-8),
                )
            )
            store["nonzero"].append((mass > epsilon).cpu().numpy().astype(np.uint8))
            store["mask_valid"].append(mask_valid.cpu().numpy().astype(np.uint8))
    summaries = []
    for spec in specs:
        store = stores[spec.key]
        payload = {
            "image_id": np.concatenate(store["ids"]),
            "label": np.concatenate(store["labels"]),
            "prediction": np.concatenate(store["predictions"]),
            "confidence": np.concatenate(store["confidence"]),
            "probabilities": np.concatenate(store["probabilities"]),
        }
        if spec.name == "clean":
            payload.update(
                strict_leakage=np.concatenate(store["strict"]),
                tolerant_leakage=np.concatenate(store["tolerant"]),
                nonzero_map=np.concatenate(store["nonzero"]),
                mask_valid=np.concatenate(store["mask_valid"]),
            )
        np.savez_compressed(output_dir / f"{spec.key}.npz", **payload)
        summary = {
            "variant": spec.key,
            "samples": store["count"],
            "accuracy": store["correct"] / store["count"],
            "ce": store["ce_sum"] / store["count"],
            "mean_confidence": float(np.concatenate(store["confidence"]).mean()),
        }
        summary.update(classification_metrics(
            payload["label"], payload["prediction"], int(config["model"]["num_classes"])
        ))
        if spec.name == "clean":
            valid = payload["mask_valid"].astype(bool)
            summary.update(
                strict_leakage=float(payload["strict_leakage"][valid].mean()),
                tolerant_leakage=float(payload["tolerant_leakage"][valid].mean()),
                nonzero_map_fraction=float(np.concatenate(store["nonzero"]).mean()),
                valid_mask_fraction=float(valid.mean()),
            )
        atomic_json(output_dir / f"{spec.key}.json", summary)
        # Keep long benchmark logs readable: per-class metrics remain in the JSON
        # artifact, while stdout reports only the run-level progress fields.
        progress = {
            key: summary[key]
            for key in (
                "variant", "samples", "accuracy", "balanced_accuracy", "macro_f1",
                "strict_leakage", "valid_mask_fraction",
            )
            if key in summary
        }
        print(json.dumps(progress), flush=True)
        summaries.append(summary)
    return summaries


def run(checkpoint_path: str, variants: list[str], split: str = "test") -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["data"].get("dataset", "cub") == "cub":
        config["data"].setdefault("evaluation_cache", str(PROJECT_ROOT / "cache" / "cub_test_224_center"))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = create_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if config["training"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    output_dir = Path(checkpoint["run_dir"]) / ("evaluation" if split == "test" else f"{split}_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = evaluate_all(model, config, variants, output_dir, split=split)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        merged = {item["variant"]: item for item in previous}
        merged.update({item["variant"]: item for item in summaries})
        summaries = [merged[key] for key in sorted(merged)]
    atomic_json(summary_path, summaries)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    print(run(args.checkpoint, args.variants, split=args.split))


if __name__ == "__main__":
    main()
