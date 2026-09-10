from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT, load_config
from .data import make_loader
from .losses import leakage
from .models import create_model
from .prepare_acdc import prepare
from .train import _amp_context, run
from .utils import atomic_json, seed_everything


DATASET_HANDLE = "samdazel/automated-cardiac-diagnosis-challenge-miccai17"
CONFIGS = {
    "resnet50": PROJECT_ROOT / "configs" / "acdc" / "acdc_resnet50_pilot.yaml",
    "vit_tiny": PROJECT_ROOT / "configs" / "acdc" / "acdc_vit_tiny_pilot.yaml",
}


def ensure_dataset(source_root: Path, prepared_root: Path, download: bool) -> Path:
    if not any(source_root.rglob("Info.cfg")):
        if not download:
            raise FileNotFoundError(
                f"No extracted ACDC data found under {source_root}; rerun with --download"
            )
        import kagglehub

        path = kagglehub.dataset_download(DATASET_HANDLE)
        source_root = Path(path)
    manifest = prepared_root / "splits" / "manifest.json"
    if not manifest.is_file():
        prepare(source_root, prepared_root, seed=42)
    return source_root


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray, classes: int) -> dict:
    recalls = []
    f1_scores = []
    for class_index in range(classes):
        truth = labels == class_index
        predicted = predictions == class_index
        true_positive = int((truth & predicted).sum())
        false_positive = int((~truth & predicted).sum())
        false_negative = int((truth & ~predicted).sum())
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        recalls.append(recall)
        f1_scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
    }


@torch.inference_mode()
def evaluate_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(int(config["seed"]))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = create_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if config["training"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    loader = make_loader(config, "test", need_mask=True, shuffle=False)
    logits_by_patient = defaultdict(list)
    labels_by_patient = {}
    strict_values = []
    tolerant_values = []
    slice_correct = slice_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if config["training"]["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        with _amp_context(device, config["training"]["amp"]):
            logits = model(images, capture_attention=False)
        logits_float = logits.float()
        predictions = logits_float.argmax(dim=1)
        slice_correct += int((predictions == labels).sum())
        slice_count += labels.shape[0]
        attention = model.evaluation_attention(labels).float()
        strict_values.extend(leakage(attention, masks, 0).cpu().tolist())
        tolerant_values.extend(
            leakage(attention, masks, int(config["fcal"]["dilation_cells"])).cpu().tolist()
        )
        for patient_id, label, item_logits in zip(
            batch["group_id"].tolist(), labels.tolist(), logits_float.cpu(), strict=True,
        ):
            patient_id = int(patient_id)
            labels_by_patient[patient_id] = int(label)
            logits_by_patient[patient_id].append(item_logits)
    patient_ids = sorted(logits_by_patient)
    patient_logits = torch.stack([
        torch.stack(logits_by_patient[patient_id]).mean(dim=0) for patient_id in patient_ids
    ])
    patient_labels = np.asarray([labels_by_patient[patient_id] for patient_id in patient_ids])
    patient_predictions = patient_logits.argmax(dim=1).numpy()
    metrics = _classification_metrics(
        patient_labels, patient_predictions, int(config["model"]["num_classes"]),
    )
    metrics.update({
        "patient_accuracy": float((patient_predictions == patient_labels).mean()),
        "patients": len(patient_ids),
        "slice_accuracy": slice_correct / slice_count,
        "slices": slice_count,
        "strict_leakage": float(np.mean(strict_values)),
        "tolerant_leakage": float(np.mean(tolerant_values)),
    })
    return metrics


def run_pilot(source_root: Path, prepared_root: Path, download: bool) -> Path:
    ensure_dataset(source_root, prepared_root, download)
    rows = []
    for architecture, config_path in CONFIGS.items():
        config = load_config(config_path)
        base_id = config["run_id"].removesuffix("-ce-seed42")
        ce_dir = run(str(config_path), "ce", f"{base_id}-ce-seed42", None, seed=42)
        branch = ce_dir / "epoch-005-branch.pt"
        fcal_dir = run(
            str(config_path), "fcal_v2", f"{base_id}-fcal-v2-seed42", str(branch), seed=42,
        )
        for method, run_dir in (("ce", ce_dir), ("fcal_v2", fcal_dir)):
            metrics = evaluate_checkpoint(run_dir / "best-validation.pt")
            rows.append({"architecture": architecture, "method": method, **metrics})
    output_dir = PROJECT_ROOT / "results" / "acdc"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "pilot_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    comparisons = []
    for architecture in CONFIGS:
        ce = next(row for row in rows if row["architecture"] == architecture and row["method"] == "ce")
        fcal = next(
            row for row in rows if row["architecture"] == architecture and row["method"] == "fcal_v2"
        )
        comparisons.append({
            "architecture": architecture,
            "patient_accuracy_delta": fcal["patient_accuracy"] - ce["patient_accuracy"],
            "balanced_accuracy_delta": fcal["balanced_accuracy"] - ce["balanced_accuracy"],
            "strict_leakage_delta": fcal["strict_leakage"] - ce["strict_leakage"],
            "strict_leakage_relative_change": (
                (fcal["strict_leakage"] - ce["strict_leakage"]) / ce["strict_leakage"]
                if ce["strict_leakage"] else 0.0
            ),
        })
    recommendation = all(
        comparison["patient_accuracy_delta"] >= -0.05
        and comparison["strict_leakage_relative_change"] <= -0.05
        for comparison in comparisons
    ) and np.mean([item["patient_accuracy_delta"] for item in comparisons]) >= 0.0
    report = {
        "dataset": DATASET_HANDLE,
        "pilot_epochs": 12,
        "test_patients": rows[0]["patients"],
        "metrics": rows,
        "comparisons": comparisons,
        "recommend_full_matrix": bool(recommendation),
        "gate": (
            "For both architectures: FCAL patient accuracy no worse than -5 pp and strict leakage "
            "at least 5% lower; mean patient-accuracy delta must be non-negative."
        ),
        "warning": "This one-seed patient-held-out test is a screening pilot, not clinical evidence.",
    }
    atomic_json(output_dir / "pilot_summary.json", report)
    return output_dir


def main() -> None:
    dataset_root = PROJECT_ROOT.parent / "datasets"
    parser = argparse.ArgumentParser(description="Run the paired ACDC ResNet-50/ViT-Tiny pilot")
    parser.add_argument("--source-root", type=Path, default=dataset_root / "ACDC-MICCAI17")
    parser.add_argument("--prepared-root", type=Path, default=dataset_root / "ACDC-MICCAI17-prepared")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(run_pilot(args.source_root.resolve(), args.prepared_root.resolve(), args.download))


if __name__ == "__main__":
    main()
