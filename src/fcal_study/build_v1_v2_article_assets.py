"""Build the multi-architecture figure used by the FCAL V1/V2 article."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
REPORT = ROOT / "reports" / "v1_v2_article"
VIS_CACHE = ROOT / "reports" / "simple_visualization_cache"
EVAL_CACHE = ROOT / "cache" / "cub_test_224_center"

sys.path.insert(0, str(ROOT / "src"))
from fcal_study.variants import apply_variant, parse_variant  # noqa: E402

ARCHITECTURES = {
    "ResNet-50": ("resnet50-ce-seed42", "resnet50-fcal-seed42", "resnet50-fcal-v2-seed42"),
    "DenseNet-121": ("densenet121-ce-seed42", "densenet121-fcal-seed42", "densenet121-fcal-v2-seed42"),
    "EfficientNet-B0": ("efficientnet-b0-ce-seed42", "efficientnet-b0-fcal-seed42", "efficientnet-b0-fcal-v2-seed42"),
    "ViT-B/16 avg": ("vit-base-p16-avg-ce-seed42", "vit-base-p16-avg-fcal-seed42", "vit-base-p16-avg-fcal-v2-seed42"),
    "ViT-S/16 avg": ("vit-small-p16-avg-ce-seed42", "vit-small-p16-avg-fcal-seed42", "vit-small-p16-avg-fcal-v2-seed42"),
    "ViT-T/16 avg": ("vit-tiny-p16-avg-ce-seed42", "vit-tiny-p16-avg-fcal-seed42", "vit-tiny-p16-avg-fcal-v2-seed42"),
}
METHODS = ("CE", "V1", "V2")
COLORS = ("#66788A", "#D98932", "#2F8F64")

QUALITATIVE_TESTS = (
    ("Clean", "clean"),
    ("Blur (s3)", "bg_blur_s3"),
    ("Color jitter (s3)", "bg_color_jitter_s3"),
    ("Grayscale", "bg_grayscale_s1"),
    ("Gaussian noise (s3)", "bg_gaussian_noise_s3"),
    ("Solid mean", "bg_solid_mean_s1"),
    ("Random solid", "bg_random_solid_s1"),
    ("Random RGB", "bg_random_rgb_s1"),
)


def load_metrics(run_id: str) -> dict[str, float]:
    path = OUTPUTS / run_id / "evaluation" / "summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_name = {row["variant"]: row for row in rows}
    background = [row["accuracy"] for row in rows if row["variant"].startswith("bg_")]
    clean = by_name["clean"]
    return {
        "Clean": 100.0 * clean["accuracy"],
        "BG mean": 100.0 * float(np.mean(background)),
        "BG worst": 100.0 * float(np.min(background)),
        "FG-only": 100.0 * by_name["foreground_only"]["accuracy"],
        "Leakage": float(clean["strict_leakage"]),
    }


def class_names() -> list[str]:
    path = ROOT.parent / "cub2002011" / "versions" / "7" / "CUB_200_2011" / "classes.txt"
    return [
        line.split(maxsplit=1)[1].split(".", 1)[-1].replace("_", " ")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def cached_image(image_id: int, variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    image_ids = np.load(EVAL_CACHE / "image_ids.npy", mmap_mode="r")
    positions = np.flatnonzero(image_ids == image_id)
    if len(positions) != 1:
        raise ValueError(f"Expected one cache entry for image {image_id}")
    position = int(positions[0])
    image = torch.from_numpy(
        np.array(np.load(EVAL_CACHE / "images.npy", mmap_mode="r")[position], copy=True)
    ).permute(2, 0, 1).float() / 255.0
    mask = torch.from_numpy(
        np.array(np.load(EVAL_CACHE / "masks.npy", mmap_mode="r")[position], copy=True)
    ).unsqueeze(0).float()
    label = int(np.load(EVAL_CACHE / "labels.npy", mmap_mode="r")[position])
    if variant == "clean":
        tested = image.clone()
    else:
        name, marker, severity = variant.rpartition("_s")
        if marker != "_s" or not severity.isdigit():
            raise ValueError(f"Expected cached variant key ending in _sN, got {variant}")
        tested = apply_variant(image, mask, image_id, parse_variant(f"{name}:{severity}"), 42)
    return image, tested, mask, label


def cached_method(
    slug: str,
    method: str,
    variant: str,
    image_id: int,
) -> dict[str, np.ndarray | float | int]:
    path = VIS_CACHE / slug / variant / f"{method}.npz"
    with np.load(path) as archive:
        position = int(np.flatnonzero(archive["image_id"] == image_id)[0])
        return {
            "prediction": int(archive["prediction"][position]),
            "confidence": float(archive["confidence"][position]),
            "cam": archive["cam"][position].astype(np.float32),
        }


def normalized_cam(cam: np.ndarray) -> torch.Tensor:
    """Apply one robust normalization rule to every CNN and ViT map."""
    attention = torch.from_numpy(cam).float()[None, None]
    attention = F.interpolate(attention, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    finite = attention[torch.isfinite(attention)]
    if finite.numel() == 0:
        return torch.zeros_like(attention)
    low = torch.quantile(finite, 0.02)
    high = torch.quantile(finite, 0.98)
    if float(high - low) <= 1e-8:
        return torch.zeros_like(attention)
    return ((attention - low) / (high - low)).clamp(0, 1)


def overlay_cam(axis, image: torch.Tensor, mask: torch.Tensor, cam: np.ndarray) -> None:
    attention = normalized_cam(cam)
    axis.imshow(image.permute(1, 2, 0))
    axis.imshow(attention, cmap="jet", alpha=0.56, vmin=0, vmax=1)
    axis.contour(mask[0].numpy(), levels=[0.5], colors="white", linewidths=0.8)


def select_rescue(slug: str, variant: str) -> int:
    """Pick the clearest verified CE-wrong to V2-correct case in the saved cache."""
    baseline_path = VIS_CACHE / slug / variant / "baseline.npz"
    v2_path = VIS_CACHE / slug / variant / "v2.npz"
    with np.load(baseline_path) as baseline, np.load(v2_path) as v2:
        v2_positions = {int(image_id): i for i, image_id in enumerate(v2["image_id"])}
        candidates: list[tuple[float, int]] = []
        for baseline_position, image_id_value in enumerate(baseline["image_id"]):
            image_id = int(image_id_value)
            v2_position = v2_positions.get(image_id)
            if v2_position is None:
                continue
            _, _, mask, label = cached_image(image_id, variant)
            baseline_prediction = int(baseline["prediction"][baseline_position])
            v2_prediction = int(v2["prediction"][v2_position])
            if baseline_prediction == label or v2_prediction != label:
                continue

            background = 1.0 - mask[0]
            baseline_attention = normalized_cam(baseline["cam"][baseline_position])
            v2_attention = normalized_cam(v2["cam"][v2_position])
            baseline_leakage = float((baseline_attention * background).sum() / baseline_attention.sum().clamp_min(1e-8))
            v2_leakage = float((v2_attention * background).sum() / v2_attention.sum().clamp_min(1e-8))
            confidence = float(v2["confidence"][v2_position])
            candidates.append((baseline_leakage - v2_leakage + 0.10 * confidence, image_id))

    if not candidates:
        raise ValueError(f"No verified CE-wrong/V2-correct rescue cached for {slug}/{variant}")
    return max(candidates)[1]


def rescue_family_figure(
    model_name: str,
    slug: str,
    title: str,
    output_name: str,
) -> None:
    names = class_names()
    wrapped = lambda value: textwrap.fill(value, width=20)  # noqa: E731
    fig, axes = plt.subplots(4, 6, figsize=(13.2, 9.0), constrained_layout=True)
    for test_index, (test_name, variant) in enumerate(QUALITATIVE_TESTS):
        row = test_index // 2
        column = 3 * (test_index % 2)
        image_id = select_rescue(slug, variant)
        _, tested, mask, label = cached_image(image_id, variant)
        baseline = cached_method(slug, "baseline", variant, image_id)
        v2 = cached_method(slug, "v2", variant, image_id)
        print(f"{model_name}: {test_name}: image {image_id}")

        axes[row, column].imshow(tested.permute(1, 2, 0))
        axes[row, column].contour(mask[0].numpy(), levels=[0.5], colors="white", linewidths=0.8)
        axes[row, column].set_title(
            f"{test_name}\nInput - true:\n{wrapped(names[label])}", fontsize=7.8, weight="bold"
        )
        overlay_cam(axes[row, column + 1], tested, mask, baseline["cam"])
        axes[row, column + 1].set_title(
            f"CE - wrong:\n{wrapped(names[baseline['prediction']])}\n"
            f"confidence {100 * baseline['confidence']:.1f}%", fontsize=7.8,
        )
        overlay_cam(axes[row, column + 2], tested, mask, v2["cam"])
        axes[row, column + 2].set_title(
            f"FCAL V2 - correct\n{wrapped(names[label])}\n"
            f"confidence {100 * v2['confidence']:.1f}%",
            fontsize=7.8,
            weight="bold",
            color="#1F6B4A",
        )
        for axis in axes[row, column:column + 3]:
            axis.axis("off")

    color_scale = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="jet")
    colorbar = fig.colorbar(
        color_scale, ax=axes, orientation="horizontal", fraction=0.025, pad=0.018, aspect=55
    )
    colorbar.set_ticks([0, 1], labels=["Low evidence", "High evidence"])
    colorbar.set_label("Shared robust per-map normalization (2nd-98th percentile)")
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.savefig(REPORT / output_name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    data = {
        architecture: [load_metrics(run_id) for run_id in run_ids]
        for architecture, run_ids in ARCHITECTURES.items()
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#2D3640",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), constrained_layout=True)
    x = np.arange(len(ARCHITECTURES))
    width = 0.24
    labels = list(ARCHITECTURES)
    panels = (
        ("Clean", "Clean top-1 accuracy (%)", True),
        ("BG mean", "Mean over 13 background tests (%)", True),
        ("BG worst", "Worst background accuracy (%)", True),
        ("FG-only", "Foreground-only accuracy (%)", True),
        ("Leakage", "Strict attention leakage (lower is better)", False),
    )

    for ax, (metric, title, higher_is_better) in zip(axes.flat, panels):
        all_values: list[float] = []
        for method_index, (method, color) in enumerate(zip(METHODS, COLORS)):
            values = [data[architecture][method_index][metric] for architecture in labels]
            all_values.extend(values)
            ax.bar(
                x + (method_index - 1) * width,
                values,
                width,
                label=method,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(title, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        ax.set_axisbelow(True)
        if higher_is_better:
            lower = max(0.0, min(all_values) - 5.0)
            upper = min(100.0, max(all_values) + 3.0)
            ax.set_ylim(lower, upper)
        else:
            ax.set_ylim(0.0, 0.9)

    summary_ax = axes.flat[-1]
    summary_ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    summary_ax.legend(handles, METHODS, loc="upper left", frameon=False, ncol=3)
    summary_ax.text(
        0.0,
        0.72,
        "V2 leads all five metrics on\nResNet-50 and DenseNet-121.",
        fontsize=12,
        weight="bold",
        color="#1D2B36",
        va="top",
    )
    summary_ax.text(
        0.0,
        0.39,
        "Across all six architectures, V2 has\nthe lowest strict attention leakage.",
        fontsize=11,
        color="#1D2B36",
        va="top",
    )
    fig.suptitle(
        "FCAL V1 and V2 across six ImageNet-pretrained architectures on CUB-200-2011",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(REPORT / "multiarchitecture_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    rescue_family_figure(
        "ResNet-50 CNN",
        "resnet_50",
        "ResNet-50 CNN: CE wrong -> FCAL V2 correct across test families",
        "cnn_test_family_rescues.png",
    )
    rescue_family_figure(
        "ViT-B/16 average pool",
        "vit_base_16",
        "ViT-B/16: CE wrong -> FCAL V2 correct across test families",
        "vit_test_family_rescues.png",
    )


if __name__ == "__main__":
    main()
