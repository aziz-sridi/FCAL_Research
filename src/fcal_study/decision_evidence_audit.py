"""Recompute existing evidence and test a structural repair without new training.

Run from the FCAL root: python -m fcal_study.decision_evidence_audit
No test images are read and no checkpoint is written or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from fcal_study.config import load_config
from fcal_study.data import make_loader
from fcal_study.decision_evidence import centered_class_weights, v2_training_evidence
from fcal_study.losses import adaptive_lambda, leakage
from fcal_study.models import create_model
from fcal_study.utils import seed_everything
from fcal_study.v2 import support_aware_fcal_losses


def structural_counterexample():
    features = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    weights = torch.tensor([[1., 1.], [-1., -1.]])
    labels = torch.tensor([0])
    mask = torch.tensor([[[[1., 0.]]]])
    rows = []
    for magnitude in (0., 20.):
        shifted = weights + torch.tensor([[magnitude, 0.]])
        for reference in ("target", "class_mean"):
            selected = shifted[labels] if reference == "target" else centered_class_weights(shifted, labels)
            scores = torch.einsum("bc,bchw->bhw", selected, features)
            outside, support = support_aware_fcal_losses([scores], mask, 1., 0., 0.)
            rows.append({"common_weight_shift": magnitude, "reference": reference,
                         "target_probability": F.linear(features.mean((-2, -1)), shifted).softmax(-1)[0, 0].item(),
                         "containment": outside.item(), "support_loss": support.item()})
    return rows


def historical_tables(root, output):
    sources = ["results/cub/resnet50_three_seed_metrics.csv",
               "results/v2_plus/cub/test_three_seed_metrics.csv",
               "results/tuned/cub/tuned_winner_metrics.csv"]
    metadata = {}
    for relative in sources:
        path = root / relative
        metadata[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    base = pd.read_csv(root / sources[0])
    plus = pd.read_csv(root / sources[1])
    metrics = ["clean_accuracy", "bg_mean_accuracy", "bg_worst_accuracy",
               "foreground_only_accuracy", "strict_leakage"]
    rows = []
    comparisons = [(base, "FCAL V2", "CE"), (plus, "FCAL V2+", "FCAL V2")]
    for data, candidate, reference in comparisons:
        paired = data[data.method == candidate].set_index("seed")[metrics] - data[data.method == reference].set_index("seed")[metrics]
        for metric in metrics:
            scale = 1 if metric == "strict_leakage" else 100
            rows.append({"candidate": candidate, "reference": reference, "metric": metric,
                         "n_seeds": len(paired), "mean_delta": paired[metric].mean() * scale,
                         "sample_sd_delta": paired[metric].std(ddof=1) * scale,
                         "unit": "fraction" if scale == 1 else "percentage_points"})
    pd.DataFrame(rows).to_csv(output / "paired_existing_results.csv", index=False)
    tuned = pd.read_csv(root / sources[2])
    tuned[tuned.architecture_wtl_eligible].groupby("method")[metrics].mean().to_csv(output / "tuned_existing_means.csv")
    return metadata


def checkpoint_probe(root, device):
    checkpoint_path = root.parent / "archive/legacy_projects/adaptive_fcal_study/outputs/resnet50-fcal-v5-seed42/best-validation.pt"
    config = load_config(root / "configs/cub/resnet50_fcal_v2.yaml")
    # Diagnostic loader settings, never used for retraining or result selection.
    config["data"]["num_workers"] = 0
    config["data"]["batch_size"] = 32
    model = create_model(config, pretrained=False).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    batch = next(iter(make_loader(config, "train", need_mask=True, shuffle=False)))
    with torch.no_grad():
        images, labels, masks = (batch[key].to(device) for key in ("image", "label", "mask"))
        model(images)
        raw = v2_training_evidence(model, labels, "target")[0]
        centered = v2_training_evidence(model, labels, "class_mean")[0]
        common = raw - centered
        result = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  "split": "train", "images": len(labels), "model_mode": "eval", "dtype": "float32",
                  "image_ids": batch["image_id"].tolist(),
                  "input_tensor_sha256": hashlib.sha256(batch["image"].numpy().tobytes()).hexdigest(),
                  "mask_tensor_sha256": hashlib.sha256(batch["mask"].numpy().tobytes()).hexdigest(),
                  "common_map_rms_over_raw_map_rms": (common.square().mean().sqrt() / raw.square().mean().sqrt()).item(),
                  "raw_strict_leakage": leakage(raw, masks, 0).mean().item(),
                  "centered_strict_leakage": leakage(centered, masks, 0).mean().item(),
                  "interpretation": "Same predictions, different evidence definition. These leakage values are NOT a model improvement or a benchmark comparison."}
    return result


def benchmark(root, device):
    if device.type != "cuda":
        return {"status": "skipped", "reason": "CUDA unavailable"}
    config = load_config(root / "configs/cub/resnet50_fcal_v2.yaml")
    model = create_model(config, pretrained=False).to(device).to(memory_format=torch.channels_last).train()
    images = torch.randn(32, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)
    labels = torch.arange(32, device=device)
    masks = torch.zeros(32, 1, 224, 224, device=device)
    masks[:, :, 64:160, 64:160] = 1

    def step(reference):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            ce = F.cross_entropy(logits, labels)
            outside, support = support_aware_fcal_losses(
                v2_training_evidence(model, labels, reference), masks, 1., 0., .5,
                maximum_tolerance_pixels=24)
            weight = adaptive_lambda(ce, outside, .15, .05, .3)
            loss = ce + weight * outside + .05 * support
        loss.backward()

    for reference in ("target", "class_mean"):
        for _ in range(10):
            step(reference)
    times = {"target": [], "class_mean": []}
    memory = {"target": [], "class_mean": []}
    for repeat in range(6):
        order = ("target", "class_mean") if repeat % 2 == 0 else ("class_mean", "target")
        for reference in order:
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(15):
                step(reference)
            torch.cuda.synchronize()
            times[reference].append((time.perf_counter() - start) / 15)
            memory[reference].append(torch.cuda.max_memory_allocated())
    return {"status": "complete", "gpu": torch.cuda.get_device_name(),
            "scope": "Synthetic ResNet-50 batch32 224x224 BF16 forward+loss+backward; no optimizer, data loading or validation. Not an end-to-end training cost.",
            "seconds_per_step_blocks": times, "peak_allocated_bytes_blocks": memory,
            "median_time_ratio": statistics.median(times["class_mean"]) / statistics.median(times["target"]),
            "median_peak_memory_ratio": statistics.median(memory["class_mean"]) / statistics.median(memory["target"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    seed_everything(20260909)
    root = Path(__file__).resolve().parents[2]
    output = root / "results/decision_evidence_audit"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"status": "mechanism_verified_not_trained", "torch": torch.__version__,
              "source_sha256": historical_tables(root, output),
              "counterexample": structural_counterexample(),
              "checkpoint_probe": checkpoint_probe(root, device)}
    print(json.dumps(result, indent=2), flush=True)
    if not args.skip_benchmark:
        result["benchmark"] = benchmark(root, device)
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.get("benchmark", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
