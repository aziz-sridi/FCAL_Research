from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import PROJECT_ROOT, load_config
from .data import make_loader
from .losses import adaptive_lambda, article_fcal_loss, article_fcal_map_loss
from .models import create_model
from .utils import atomic_json, environment_manifest, seed_everything


def gradient_audit() -> dict:
    attention = torch.ones((1, 2, 2), dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float64)
    loss = article_fcal_map_loss(attention, mask, dilation_cells=0)
    loss.backward()
    analytic = attention.grad.detach().clone()
    delta = 1e-6
    finite = torch.empty_like(attention)
    with torch.no_grad():
        for row in range(2):
            for column in range(2):
                plus = attention.detach().clone()
                minus = attention.detach().clone()
                plus[0, row, column] += delta
                minus[0, row, column] -= delta
                finite[0, row, column] = (
                    article_fcal_map_loss(plus, mask, 0) - article_fcal_map_loss(minus, mask, 0)
                ) / (2 * delta)
    return {
        "loss": float(loss.detach()),
        "autograd": analytic.tolist(),
        "finite_difference": finite.tolist(),
        "max_absolute_error": float((analytic - finite).abs().max()),
        "foreground_gradient_is_nonzero": bool((analytic[0, 0].abs() > 0).all()),
        "interpretation": "The exact article normalization gives negative foreground gradients and positive background gradients.",
    }


def benchmark_one(config_path: str, batches: int, pretrained: bool) -> dict:
    config = load_config(config_path)
    config["method"] = "fcal"
    config["data"]["num_workers"] = min(4, int(config["data"]["num_workers"]))
    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    model = create_model(config, pretrained=pretrained).to(device).train()
    if config["training"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    loader = make_loader(config, "train", need_mask=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    amp_dtype = torch.bfloat16
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    seen = 0
    values = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if config["training"]["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images, capture_attention=True)
            ce = F.cross_entropy(logits, labels)
            spatial = article_fcal_loss(
                model.training_attention(labels), masks,
                int(config["fcal"]["dilation_cells"]), float(config["fcal"]["epsilon"]),
            )
            weight = adaptive_lambda(
                ce, spatial, float(config["fcal"]["alpha"]), float(config["fcal"]["lambda_min"]),
                float(config["fcal"]["lambda_max"]), float(config["fcal"]["epsilon"]),
            )
            loss = ce + weight * spatial
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite benchmark loss")
        values.append((float(ce), float(spatial), float(weight)))
        seen += labels.shape[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "model": config["model"]["name"],
        "pretrained_id": config["model"]["pretrained_id"],
        "pretrained_loaded": pretrained,
        "batches": len(values),
        "samples": seen,
        "seconds": elapsed,
        "samples_per_second": seen / elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "last_ce": values[-1][0],
        "last_fcal": values[-1][1],
        "last_lambda": values[-1][2],
    }
    del model, optimizer, loader
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "phase0_benchmark.json"))
    args = parser.parse_args()
    report = {
        "environment": environment_manifest(),
        "gradient_audit": gradient_audit(),
        "benchmarks": [benchmark_one(path, args.batches, args.pretrained) for path in args.configs],
    }
    atomic_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
