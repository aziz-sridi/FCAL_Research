"""Build reproducible, prediction-aware CAM galleries for the project READMEs.

The script never invents examples: it selects test IDs from saved evaluation
arrays, reloads the corresponding best checkpoints, and renders predicted-class
CAMs for the exact same center-cropped image under every method.

Example
-------
python scripts/build_result_showcases.py \
  --outputs-root "D:/path/to/outputs" --dataset all
"""

from __future__ import annotations

import argparse
import csv
import gc
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from torch.nn import functional as F

from fcal_study.config import PROJECT_ROOT, load_config
from fcal_study.data import IMAGENET_MEAN, IMAGENET_STD, dataset_from_config
from fcal_study.models import create_model


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    family: str
    config_name: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    output_subdir: str
    config_subdir: str
    methods: tuple[tuple[str, str], ...]
    overview_model: str


MODELS = (
    ModelSpec("resnet50", "ResNet-50", "CNN", "resnet50"),
    ModelSpec("densenet121", "DenseNet-121", "CNN", "densenet121"),
    ModelSpec("efficientnet-b0", "EfficientNet-B0", "CNN", "efficientnet_b0"),
    ModelSpec("vit-tiny-p16-avg", "ViT-Tiny/16", "ViT", "vit_tiny_avg"),
    ModelSpec("vit-small-p16-avg", "ViT-Small/16", "ViT", "vit_small_avg"),
    ModelSpec("vit-base-p16-avg", "ViT-Base/16", "ViT", "vit_base_avg"),
)

DATASETS = {
    "cub": DatasetSpec(
        "cub", "CUB-200-2011", "", "cub",
        (("CE", "ce"), ("FCAL V1", "fcal"), ("FCAL V2", "fcal-v5")),
        "resnet50",
    ),
    "isic": DatasetSpec(
        "isic", "ISIC", "isic", "isic",
        (("CE", "ce"), ("FCAL V1", "fcal-v1"), ("FCAL V2", "fcal-v5")),
        "vit-small-p16-avg",
    ),
    "oxford_pet": DatasetSpec(
        "oxford_pet", "Oxford-IIIT Pet", "oxford_pet", "oxford_pet",
        (
            ("CE", "ce"),
            ("FCAL V1", "fcal-v1"),
            ("FCAL V2", "fcal-v5"),
            ("Universal", "ufcal-auto-safe"),
        ),
        "vit-base-p16-avg",
    ),
}

DISPLAY_COLORS = {
    "CE": "#94a3b8",
    "FCAL V1": "#38bdf8",
    "FCAL V2": "#34d399",
    "Universal": "#c084fc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        required=True,
        help="Root containing CUB runs and the isic/ and oxford_pet/ run folders.",
    )
    parser.add_argument(
        "--dataset", choices=("all", *DATASETS), default="all",
        help="Dataset gallery to build (default: all).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used to reconstruct CAMs.",
    )
    parser.add_argument(
        "--assets-dir", type=Path,
        default=PROJECT_ROOT / "assets" / "result_showcases",
    )
    return parser.parse_args()


def run_dir(outputs_root: Path, dataset: DatasetSpec, model: ModelSpec, suffix: str) -> Path:
    base = outputs_root / dataset.output_subdir if dataset.output_subdir else outputs_root
    return base / f"{model.key}-{suffix}-seed42"


def config_path(dataset: DatasetSpec, model: ModelSpec) -> Path:
    if dataset.key == "cub":
        return PROJECT_ROOT / "configs" / "cub" / f"{model.config_name}_ce.yaml"
    prefix = "isic" if dataset.key == "isic" else "oxford_pet"
    model_name = model.config_name if dataset.key == "isic" else model.config_name.removesuffix("_avg")
    return PROJECT_ROOT / "configs" / dataset.config_subdir / f"{prefix}_{model_name}_fcal_matrix.yaml"


def evaluation_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def aligned_arrays(
    outputs_root: Path, dataset: DatasetSpec, model: ModelSpec,
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    reference_ids = None
    for display, suffix in dataset.methods:
        path = run_dir(outputs_root, dataset, model, suffix) / "evaluation" / "clean.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        arrays = evaluation_arrays(path)
        if reference_ids is None:
            reference_ids = arrays["image_id"]
        elif not np.array_equal(reference_ids, arrays["image_id"]):
            raise ValueError(f"Mismatched test ordering for {dataset.key}/{model.key}/{display}")
        result[display] = arrays
    return result


def select_examples(arrays: dict[str, dict[str, np.ndarray]], count: int = 3) -> list[int]:
    """Choose strong, non-duplicate V2 improvements using saved true-class leakage.

    Selection requires a correct V2 prediction and first prioritizes cases where
    CE was also correct (a clean localization comparison). One classification
    rescue is included when available. The rendered metrics are recomputed from
    predicted-class CAMs and are not copied from this selection cache.
    """
    ce = arrays["CE"]
    v2 = arrays["FCAL V2"]
    improvement = ce["strict_leakage"] - v2["strict_leakage"]
    valid = (v2["prediction"] == v2["label"]) & (v2["nonzero_map"] > 0)
    if "mask_valid" in v2:
        valid &= v2["mask_valid"] > 0

    focus = valid & (ce["prediction"] == ce["label"])
    rescue = valid & (ce["prediction"] != ce["label"])
    chosen: list[int] = []

    def take(mask: np.ndarray, limit: int) -> None:
        ranked = np.argsort(np.where(mask, improvement, -np.inf))[::-1]
        for index in ranked:
            if len(chosen) >= limit or not mask[index]:
                break
            if int(index) not in chosen:
                chosen.append(int(index))

    take(focus, 1)
    if rescue.any():
        take(rescue, min(count, len(chosen) + 1))
    take(focus, count)
    take(valid, count)
    if len(chosen) < count:
        raise RuntimeError("Not enough valid examples to build the gallery")
    return chosen[:count]


def class_names(dataset: DatasetSpec, config: dict) -> dict[int, str]:
    if dataset.key == "cub":
        path = Path(config["data"]["cub_root"]) / "classes.txt"
        names = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            number, raw = line.split(maxsplit=1)
            names[int(number) - 1] = raw.split(".", 1)[-1].replace("_", " ")
        return names

    split_path = Path(config["data"]["split_dir"]) / "test.csv"
    names = {}
    with split_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            if dataset.key == "isic":
                names[label] = row["class_code"].replace("_", " ")
            else:
                raw = row["image_id"].rsplit("_", 1)[0]
                names[label] = raw.replace("_", " ")
    return names


def clean_image(normalized: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=normalized.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=normalized.dtype).view(3, 1, 1)
    return (normalized.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def checkpoint_predictions(
    checkpoint_path: Path,
    config: dict,
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> list[dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model = create_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model.to(device).eval()
    batch = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        logits = model(batch, capture_attention=True)
        probabilities = logits.float().softmax(dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        attention = model.evaluation_attention(prediction).float().unsqueeze(1)
        attention = F.interpolate(
            attention, size=masks.shape[-2:], mode="bilinear", align_corners=False,
        ).squeeze(1).clamp_min(0)
        denominator = attention.sum(dim=(-2, -1))
        inside = (attention * masks.squeeze(1)).sum(dim=(-2, -1)) / denominator.clamp_min(1e-8)

    result = []
    for index in range(images.shape[0]):
        cam = attention[index].cpu().numpy()
        minimum, maximum = float(cam.min()), float(cam.max())
        normalized = (cam - minimum) / max(maximum - minimum, 1e-8)
        result.append(
            {
                "prediction": int(prediction[index]),
                "confidence": float(confidence[index]),
                "inside": float(inside[index]) if denominator[index] > 0 else 0.0,
                "cam": normalized,
            }
        )
    del model, batch, masks, logits, probabilities, confidence, prediction, attention
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def short_label(label: str, width: int = 19) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False)[:2])


def draw_gallery(
    destination: Path,
    dataset: DatasetSpec,
    row_models: list[ModelSpec],
    row_samples: list[dict[str, object]],
    methods: tuple[str, ...],
    title: str,
) -> None:
    columns = 1 + len(methods)
    rows = len(row_samples)
    fig, axes = plt.subplots(rows, columns, figsize=(3.15 * columns, 2.85 * rows), squeeze=False)
    fig.patch.set_facecolor("#08111f")
    fig.suptitle(title, color="white", fontsize=18, fontweight="bold", y=0.995)
    cmap = colormaps["jet"]

    for row, (model, sample) in enumerate(zip(row_models, row_samples, strict=True)):
        image = sample["image"]
        mask = sample["mask"]
        true_name = sample["true_name"]
        image_id = sample["image_id"]
        axis = axes[row, 0]
        axis.imshow(image)
        axis.set_title(
            f"{model.title} · input\nTrue: {short_label(true_name)}",
            color="white", fontsize=9.5, fontweight="bold",
        )
        axis.text(
            0.02, 0.03, f"test id {image_id}", transform=axis.transAxes, fontsize=7.5,
            color="white", bbox={"facecolor": "#08111f", "alpha": 0.8, "pad": 2, "edgecolor": "none"},
        )
        axis.axis("off")

        for column, method in enumerate(methods, start=1):
            prediction = sample["methods"][method]
            # Keep the source recognizable while making the full jet palette
            # unmistakable, including for coarse 14x14 ViT patch CAMs.
            overlay = 0.25 * image + 0.75 * cmap(prediction["cam"])[..., :3]
            overlay = np.clip(overlay, 0, 1)
            axis = axes[row, column]
            axis.imshow(overlay)
            axis.contour(mask, levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.95)
            correct = prediction["prediction"] == sample["label"]
            symbol = "✓" if correct else "✗"
            color = "#4ade80" if correct else "#fb7185"
            axis.set_title(
                f"{method}\n{short_label(prediction['prediction_name'])} {symbol}\n"
                f"conf {prediction['confidence']:.0%} · inside {prediction['inside']:.0%}",
                color=color, fontsize=9.2, fontweight="bold",
            )
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(DISPLAY_COLORS[method])
                spine.set_linewidth(2.2 if method == "FCAL V2" else 1.2)
            axis.set_xticks([])
            axis.set_yticks([])

    fig.text(
        0.5, 0.006,
        "Predicted-class CAM · jet: blue = low, red = high · white contour = foreground · inside = CAM mass within contour",
        ha="center", color="#cbd5e1", fontsize=8.5,
    )
    fig.tight_layout(rect=(0.01, 0.025, 0.99, 0.965), h_pad=2.0, w_pad=0.8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=145, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def build_dataset(
    spec: DatasetSpec, outputs_root: Path, assets_root: Path, device: torch.device,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    destination = assets_root / spec.key
    destination.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    readme_examples: dict[str, dict[str, object]] = {}
    rescue_examples: dict[str, dict[str, object]] = {}

    for model in MODELS:
        print(f"[{spec.key}] selecting {model.title}", flush=True)
        arrays = aligned_arrays(outputs_root, spec, model)
        selected_indices = select_examples(arrays, count=3)
        config = load_config(config_path(spec, model))
        dataset = dataset_from_config(config, "test", need_mask=True)
        names = class_names(spec, config)
        id_to_index = {int(record.image_id): index for index, record in enumerate(dataset.records)}
        selected_ids = [int(arrays["CE"]["image_id"][index]) for index in selected_indices]
        samples = [dataset[id_to_index[image_id]] for image_id in selected_ids]
        images = torch.stack([sample["image"] for sample in samples])
        masks = torch.stack([sample["mask"] for sample in samples])
        rows = []
        for image_id, sample in zip(selected_ids, samples, strict=True):
            label = int(sample["label"])
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "true_name": names.get(label, str(label)),
                    "image": clean_image(sample["image"]),
                    "mask": sample["mask"].squeeze(0).numpy(),
                    "methods": {},
                }
            )

        for display, suffix in spec.methods:
            print(f"[{spec.key}] {model.title}: {display}", flush=True)
            checkpoint = run_dir(outputs_root, spec, model, suffix) / "best-validation.pt"
            predictions = checkpoint_predictions(checkpoint, config, images, masks, device)
            for row, prediction in zip(rows, predictions, strict=True):
                prediction["prediction_name"] = names.get(prediction["prediction"], str(prediction["prediction"]))
                row["methods"][display] = prediction

        gallery_path = destination / f"{model.key}_test_cases.png"
        draw_gallery(
            gallery_path, spec, [model] * len(rows), rows,
            tuple(display for display, _ in spec.methods),
            f"{spec.title} · {model.title} test cases",
        )
        readme_examples[model.key] = rows[0]
        rescue = next(
            (
                row for row in rows
                if row["methods"]["CE"]["prediction"] != row["label"]
                and row["methods"]["FCAL V2"]["prediction"] == row["label"]
            ),
            None,
        )
        if rescue is not None:
            rescue_examples[model.key] = rescue

        for rank, row in enumerate(rows, start=1):
            manifest = {
                "dataset": spec.title,
                "family": model.family,
                "architecture": model.title,
                "case_rank": rank,
                "image_id": row["image_id"],
                "true_class": row["true_name"],
            }
            for display, _ in spec.methods:
                prediction = row["methods"][display]
                prefix = display.lower().replace(" ", "_")
                manifest[f"{prefix}_prediction"] = prediction["prediction_name"]
                manifest[f"{prefix}_correct"] = prediction["prediction"] == row["label"]
                manifest[f"{prefix}_confidence"] = round(prediction["confidence"], 6)
                manifest[f"{prefix}_inside_fraction"] = round(prediction["inside"], 6)
            manifest_rows.append(manifest)

        del dataset, samples, images, masks, rows
        gc.collect()

    for family in ("CNN", "ViT"):
        models = [model for model in MODELS if model.family == family]
        rows = [readme_examples[model.key] for model in models]
        draw_gallery(
            destination / f"readme_{family.lower()}.png", spec, models, rows,
            tuple(display for display, _ in spec.methods),
            f"{spec.title} · {family} comparison",
        )
        rescue_models = [model for model in models if model.key in rescue_examples]
        rescue_rows = [rescue_examples[model.key] for model in rescue_models]
        if rescue_rows:
            draw_gallery(
                destination / f"readme_{family.lower()}_rescues.png",
                spec,
                rescue_models,
                rescue_rows,
                tuple(display for display, _ in spec.methods),
                f"{spec.title} · {family} CE → FCAL V2 rescues",
            )

    manifest_path = destination / "examples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_rows, readme_examples, rescue_examples


def build_overview(
    assets_root: Path,
    examples: dict[str, dict[str, dict[str, object]]],
    rescues: dict[str, dict[str, dict[str, object]]],
) -> None:
    rows = []
    models = []
    for dataset_key in ("cub", "isic", "oxford_pet"):
        spec = DATASETS[dataset_key]
        model = next(item for item in MODELS if item.key == spec.overview_model)
        sample = examples[dataset_key][model.key]
        # The overview intentionally uses the shared CE/V2 columns only.
        rows.append(sample)
        models.append(ModelSpec(model.key, f"{spec.title} · {model.title}", model.family, model.config_name))
    draw_gallery(
        assets_root / "overview.png", DATASETS["cub"], models, rows,
        ("CE", "FCAL V2"),
        "Same classifier task. Better evidence location.",
    )
    rescue_rows = []
    rescue_models = []
    for dataset_key in ("cub", "isic", "oxford_pet"):
        spec = DATASETS[dataset_key]
        model = next(item for item in MODELS if item.key == spec.overview_model)
        rescue_rows.append(rescues[dataset_key][model.key])
        rescue_models.append(
            ModelSpec(
                model.key,
                f"{spec.title} · {model.title}",
                model.family,
                model.config_name,
            )
        )
    draw_gallery(
        assets_root / "rescue_overview.png",
        DATASETS["cub"],
        rescue_models,
        rescue_rows,
        ("CE", "FCAL V2"),
        "CE missed it. FCAL V2 got it right.",
    )


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    assets_root = args.assets_dir.resolve()
    device = torch.device(args.device)
    keys = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    examples = {}
    rescues = {}
    for key in keys:
        _, examples[key], rescues[key] = build_dataset(
            DATASETS[key], outputs_root, assets_root, device,
        )
    if args.dataset == "all":
        build_overview(assets_root, examples, rescues)
    print(f"Wrote galleries to {assets_root}", flush=True)


if __name__ == "__main__":
    main()
