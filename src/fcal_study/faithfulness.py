from __future__ import annotations

import argparse
import copy
import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import make_loader
from .evaluate import relocate_missing_data_paths
from .evidence import attention_iou, faithfulness_auc, ranked_perturbation_masks
from .models import create_model
from .utils import atomic_json, sha256_file

PROTOCOL_VERSION = "fcal-faithfulness-v1"
IOU_THRESHOLD = 0.7
FRACTIONS = tuple(value / 10 for value in range(11))
SUPPORTED_METHODS = {
    "ce", "fcal", "fcal_v2", "fcal_v3", "fcal_v4", "fcal_v5",
    "fcal_v5_cheap", "ufcal", "rrr", "gain", "camal", "isnet",
}


def deterministic_subset_ids(
    image_ids: Iterable[int], max_images: int | None, selection_seed: int,
) -> set[int] | None:
    """Select an outcome-independent subset using stable hashes of image IDs."""
    if max_images is None:
        return None
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    unique = sorted({int(value) for value in image_ids})
    if max_images >= len(unique):
        return set(unique)

    def key(image_id: int) -> tuple[bytes, int]:
        payload = f"{int(selection_seed)}:{image_id}".encode()
        return hashlib.sha256(payload).digest(), image_id

    return set(sorted(unique, key=key)[:max_images])


def normalized_positive_maps(maps: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Max-normalize non-negative explanation maps, then resize to image resolution."""
    if maps.ndim != 3:
        raise ValueError("explanation maps must have shape [batch, height, width]")
    values = torch.nan_to_num(maps.float()).clamp_min(0.0).unsqueeze(1)
    flat = values.flatten(1)
    maxima = flat.amax(dim=1).view(-1, 1, 1, 1)
    values = torch.where(
        maxima > 0, values / maxima.clamp_min(1e-12), torch.zeros_like(values),
    )
    target_size = (int(size[0]), int(size[1]))
    if values.shape[-2:] != target_size:
        values = F.interpolate(values, size=target_size, mode="bilinear", align_corners=False)
    return values[:, 0]


@contextmanager
def _amp_context(device: torch.device, amp_mode: str | bool):
    enabled = device.type == "cuda" and str(amp_mode).lower() not in {"false", "off", "none", "0"}
    with torch.no_grad():
        if enabled:
            dtype = torch.bfloat16 if str(amp_mode).lower() == "bf16" else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                yield
        else:
            yield


def extract_explanation(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    """Return clean logits and the common target-class CAM used for ranking."""
    amp_mode = config["training"]["amp"]
    device = images.device
    model_config = config["model"]
    if model_config.get("family") == "vit" and model_config.get("pool") != "avg":
        raise ValueError(
            "The sealed protocol requires a target-class spatial evidence map; "
            "non-average-pooled ViT attention rollout is not class-specific"
        )
    with _amp_context(device, amp_mode):
        logits = model(images, capture_attention=False)
        maps = model.evaluation_attention(labels).float()
    return (
        logits.detach(), maps.detach(),
        "common_nonnegative_ground_truth_class_final_spatial_evidence",
        "ground_truth_class",
    )


def _target_confidence(
    model, images: torch.Tensor, labels: torch.Tensor, amp_mode: str | bool,
) -> torch.Tensor:
    with _amp_context(images.device, amp_mode):
        logits = model(images, capture_attention=False)
    return logits.float().softmax(dim=1).gather(1, labels[:, None]).squeeze(1)


def _dataset_image_ids(loader) -> list[int]:
    records = getattr(loader.dataset, "records", None)
    if records is None:
        raise TypeError("Deterministic subsampling requires a dataset with image-ID records")
    return [int(record.image_id) for record in records]


def _prepared_split_file(config: dict, split: str) -> Path:
    split_dir = Path(config["data"]["split_dir"])
    for suffix in (".txt", ".csv"):
        candidate = split_dir / f"{split}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Prepared {split} split file is missing under: {split_dir}")


def _fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def evaluate_loader(
    model,
    config: dict,
    loader,
    *,
    fractions: Sequence[float] = FRACTIONS,
    iou_threshold: float = IOU_THRESHOLD,
    selected_ids: set[int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Evaluate localization and perturbation faithfulness without updating a model."""
    method = config.get("method")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"No sealed explanation-map protocol for method: {method}")
    points = np.asarray(fractions, dtype=np.float64)
    if len(points) < 2 or points[0] != 0.0 or points[-1] != 1.0 or np.any(np.diff(points) <= 0):
        raise ValueError("fractions must increase strictly from 0 to 1")
    device = next(model.parameters()).device
    amp_mode = config["training"]["amp"]
    channels_last = bool(config["training"]["channels_last"])
    stores: dict[str, list[np.ndarray]] = {
        "image_id": [], "label": [], "mask_valid": [], "attention_iou": [],
        "clean_target_confidence": [], "baseline_target_confidence": [],
        "insertion_confidence": [], "removal_confidence": [],
    }
    map_recipe = explanation_target = ""
    for batch in loader:
        image_ids = batch["image_id"]
        keep = torch.ones(len(image_ids), dtype=torch.bool)
        if selected_ids is not None:
            keep = torch.tensor([int(value) in selected_ids for value in image_ids], dtype=torch.bool)
        if not bool(keep.any()):
            continue
        images = batch["image"][keep].to(device, non_blocking=True)
        labels = batch["label"][keep].to(device, non_blocking=True)
        masks = batch["mask"][keep].to(device, non_blocking=True).float()
        kept_ids = image_ids[keep]
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        logits, raw_maps, map_recipe, explanation_target = extract_explanation(
            model, images, labels, config,
        )
        maps = normalized_positive_maps(raw_maps, masks.shape[-2:])
        masks_2d = masks[:, 0] >= 0.5
        flat_masks = masks_2d.flatten(1)
        mask_valid = flat_masks.any(dim=1) & ~flat_masks.all(dim=1)
        iou = attention_iou(
            maps.cpu().numpy(), masks_2d.cpu().numpy(),
            threshold=float(iou_threshold), normalize_by_max=False,
        )
        iou[~mask_valid.cpu().numpy()] = np.nan

        probabilities = logits.float().softmax(dim=1)
        clean_confidence = probabilities.gather(1, labels[:, None]).squeeze(1)
        baseline_images = torch.zeros_like(images)
        baseline_confidence = _target_confidence(model, baseline_images, labels, amp_mode)
        insertion = torch.empty((len(images), len(points)), dtype=torch.float32)
        removal = torch.empty_like(insertion)
        insertion[:, 0] = baseline_confidence.cpu()
        insertion[:, -1] = clean_confidence.cpu()
        removal[:, 0] = clean_confidence.cpu()
        removal[:, -1] = baseline_confidence.cpu()
        ranked = ranked_perturbation_masks(maps.cpu().numpy(), points)
        for point_index in range(1, len(points) - 1):
            reveal = torch.from_numpy(ranked[:, point_index]).to(device=device).unsqueeze(1)
            inserted = torch.where(reveal, images, baseline_images)
            removed = torch.where(reveal, baseline_images, images)
            insertion[:, point_index] = _target_confidence(
                model, inserted, labels, amp_mode,
            ).cpu()
            removal[:, point_index] = _target_confidence(
                model, removed, labels, amp_mode,
            ).cpu()

        stores["image_id"].append(kept_ids.numpy().astype(np.int64))
        stores["label"].append(labels.cpu().numpy().astype(np.int64))
        stores["mask_valid"].append(mask_valid.cpu().numpy())
        stores["attention_iou"].append(iou.astype(np.float32))
        stores["clean_target_confidence"].append(clean_confidence.cpu().numpy().astype(np.float32))
        stores["baseline_target_confidence"].append(baseline_confidence.cpu().numpy().astype(np.float32))
        stores["insertion_confidence"].append(insertion.numpy())
        stores["removal_confidence"].append(removal.numpy())
    if not stores["image_id"]:
        raise ValueError("No images were selected for faithfulness evaluation")
    payload = {key: np.concatenate(value) for key, value in stores.items()}
    payload["fractions"] = points
    payload["insertion_auc_normalized"] = faithfulness_auc(
        points, payload["insertion_confidence"],
    ).astype(np.float32)
    payload["removal_auc_normalized"] = faithfulness_auc(
        points, payload["removal_confidence"],
    ).astype(np.float32)
    return payload, {"map_recipe": map_recipe, "explanation_target": explanation_target}


def summarize(payload: dict[str, np.ndarray]) -> dict:
    valid = payload["mask_valid"].astype(bool) & np.isfinite(payload["attention_iou"])
    return {
        "samples": len(payload["image_id"]),
        "valid_mask_samples": int(valid.sum()),
        "attention_iou_mean": float(payload["attention_iou"][valid].mean()) if valid.any() else None,
        "insertion_auc_normalized_mean": float(payload["insertion_auc_normalized"].mean()),
        "removal_auc_normalized_mean": float(payload["removal_auc_normalized"].mean()),
        "clean_target_confidence_mean": float(payload["clean_target_confidence"].mean()),
        "baseline_target_confidence_mean": float(payload["baseline_target_confidence"].mean()),
    }


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def run(
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    output_dir: str | Path | None = None,
    max_images: int | None = None,
    selection_seed: int = 2026,
    batch_size: int | None = None,
) -> Path:
    """Run the sealed protocol on validation or test data and write traceable artifacts."""
    if split not in {"validation", "test"}:
        raise ValueError("Faithfulness evaluation is sealed to validation or test splits")
    checkpoint_path = Path(checkpoint_path).resolve()
    if output_dir is None:
        output_dir = checkpoint_path.parent / "faithfulness" / split
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite sealed faithfulness artifacts: {output_dir}"
        )
    checkpoint_fingerprint = _fingerprint(checkpoint_path)
    checkpoint_digest = sha256_file(checkpoint_path)
    if _fingerprint(checkpoint_path) != checkpoint_fingerprint:
        raise RuntimeError(f"Checkpoint changed while it was being sealed: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if _fingerprint(checkpoint_path) != checkpoint_fingerprint:
        raise RuntimeError(f"Checkpoint changed while it was being loaded: {checkpoint_path}")
    config = relocate_missing_data_paths(copy.deepcopy(checkpoint["config"]))
    if config.get("method") not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {config.get('method')}")
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        config["data"]["batch_size"] = int(batch_size)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = create_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if config["training"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    split_path = _prepared_split_file(config, split)
    split_digest = sha256_file(split_path)
    loader = make_loader(config, split, need_mask=True, shuffle=False)
    selected_ids = None
    if max_images is not None:
        selected_ids = deterministic_subset_ids(
            _dataset_image_ids(loader), max_images, selection_seed,
        )
    payload, map_metadata = evaluate_loader(
        model, config, loader, selected_ids=selected_ids,
    )
    summary = summarize(payload)
    summary.update(
        protocol_version=PROTOCOL_VERSION,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_digest,
        run_id=config.get("run_id"),
        method=config["method"],
        architecture=config["model"]["name"],
        checkpoint_epoch=int(checkpoint.get("epoch", -1)),
        split=split,
        split_file_name=split_path.name,
        split_file_sha256=split_digest,
        target_confidence="ground_truth_class_softmax",
        perturbation_baseline="zero_in_normalized_model_input_space",
        fractions=list(FRACTIONS),
        iou_threshold=IOU_THRESHOLD,
        selection_rule=(
            "all_images" if max_images is None
            else "lowest_sha256(selection_seed:image_id)"
        ),
        selection_seed=None if max_images is None else int(selection_seed),
        requested_max_images=max_images,
        selected_image_ids_sha256=hashlib.sha256(
            np.sort(payload["image_id"]).astype("<i8").tobytes()
        ).hexdigest(),
        **map_metadata,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        _write_npz(temporary / "per_image.npz", payload)
        atomic_json(temporary / "summary.json", summary)
        atomic_json(
            temporary / "protocol.json",
            {
                key: summary[key]
                for key in (
                    "protocol_version", "checkpoint", "checkpoint_sha256", "split",
                    "split_file_name", "split_file_sha256",
                    "target_confidence", "explanation_target", "map_recipe",
                    "perturbation_baseline", "fractions", "iou_threshold",
                    "selection_rule", "selection_seed", "requested_max_images",
                    "selected_image_ids_sha256",
                )
            },
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sealed post-selection localization and faithfulness evaluation",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--max-images", type=int,
        help="Outcome-independent SHA-256 subset size; omit to evaluate every image.",
    )
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    print(run(
        args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        max_images=args.max_images,
        selection_seed=args.selection_seed,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
