"""Build complete, consistently defined result tables for the three benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .variants import DEFAULT_VARIANTS, TAILORED_VARIANTS, parse_variant

ARCHITECTURES = (
    ("resnet50", "ResNet-50", "CNN"),
    ("densenet121", "DenseNet-121", "CNN"),
    ("efficientnet-b0", "EfficientNet-B0", "CNN"),
    ("vit-tiny-p16-avg", "ViT-Tiny/16 avg", "ViT"),
    ("vit-small-p16-avg", "ViT-Small/16 avg", "ViT"),
    ("vit-base-p16-avg", "ViT-Base/16 avg", "ViT"),
)
METHOD_SUFFIXES = {
    "cub": (("CE", "ce"), ("FCAL V1", "fcal"), ("FCAL V2", "fcal-v5")),
    "oxford_pet": (("CE", "ce"), ("FCAL V1", "fcal-v1"), ("FCAL V2", "fcal-v5")),
    "isic": (("CE", "ce"), ("FCAL V1", "fcal-v1"), ("FCAL V2", "fcal-v5")),
}


def _key(value: str) -> str:
    return parse_variant(value).key


GENERIC_BACKGROUND = tuple(
    _key(value) for value in DEFAULT_VARIANTS if value.startswith("bg_")
)
GENERIC_FOREGROUND = tuple(
    _key(value) for value in DEFAULT_VARIANTS if value.startswith("fg_")
)


def _summary(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["variant"]: row for row in rows}


def _classification_fields(run_dir: Path, variant: str, summary: dict, num_classes: int) -> tuple[float, float]:
    if "balanced_accuracy" in summary and "macro_f1" in summary:
        return float(summary["balanced_accuracy"]), float(summary["macro_f1"])
    with np.load(run_dir / "evaluation" / f"{variant}.npz") as archive:
        labels = archive["label"]
        predictions = archive["prediction"]
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        truth = labels == class_index
        predicted = predictions == class_index
        true_positive = int((truth & predicted).sum())
        false_positive = int((~truth & predicted).sum())
        false_negative = int((truth & ~predicted).sum())
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        recalls.append(recall)
        f1_scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(recalls)), float(np.mean(f1_scores))


def build(dataset: str, outputs_root: Path, report_root: Path) -> tuple[Path, Path, Path, Path]:
    if dataset not in METHOD_SUFFIXES:
        raise ValueError(f"Unknown dataset: {dataset}")
    rows: list[dict] = []
    scenario_rows: list[dict] = []
    required = {
        "clean", "foreground_only", "background_only",
        *GENERIC_BACKGROUND, *GENERIC_FOREGROUND,
    }
    tailored = tuple(_key(value) for value in TAILORED_VARIANTS[dataset])

    for slug, architecture, family in ARCHITECTURES:
        for method, suffix in METHOD_SUFFIXES[dataset]:
            run_dir = outputs_root / f"{slug}-{suffix}-seed42"
            summary_path = run_dir / "evaluation" / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summaries = _summary(summary_path)
            missing = sorted(required.difference(summaries))
            if missing:
                raise ValueError(f"{run_dir.name} is missing: {', '.join(missing)}")

            clean = summaries["clean"]
            clean_balanced, clean_f1 = _classification_fields(
                run_dir, "clean", clean, 200 if dataset == "cub" else (37 if dataset == "oxford_pet" else 8)
            )
            background = [summaries[name]["accuracy"] for name in GENERIC_BACKGROUND]
            foreground = [summaries[name]["accuracy"] for name in GENERIC_FOREGROUND]
            rows.append({
                "family": family,
                "architecture": architecture,
                "method": method,
                "clean_accuracy": clean["accuracy"],
                "clean_balanced_accuracy": clean_balanced,
                "clean_macro_f1": clean_f1,
                "strict_leakage": clean.get("strict_leakage", np.nan),
                "bg_mean_accuracy": float(np.mean(background)),
                "bg_worst_accuracy": float(np.min(background)),
                "fg_mean_accuracy": float(np.mean(foreground)),
                "fg_worst_accuracy": float(np.min(foreground)),
                "foreground_only_accuracy": summaries["foreground_only"]["accuracy"],
                "background_only_accuracy": summaries["background_only"]["accuracy"],
            })
            for scenario in tailored:
                if scenario not in summaries:
                    raise ValueError(f"{run_dir.name} is missing tailored scenario: {scenario}")
                result = summaries[scenario]
                scenario_rows.append({
                    "family": family,
                    "architecture": architecture,
                    "method": method,
                    "scenario": scenario,
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "macro_f1": result["macro_f1"],
                })

    report_root.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    scenarios = pd.DataFrame(scenario_rows)
    metric_columns = [column for column in metrics if column.endswith("accuracy") or column == "strict_leakage"]
    means = metrics.groupby("method", sort=False)[metric_columns].mean().reset_index()
    metrics_path = report_root / "complete_metrics.csv"
    means_path = report_root / "method_means_complete.csv"
    scenarios_path = report_root / "tailored_scenarios.csv"
    scenario_means_path = report_root / "tailored_scenario_means.csv"
    metrics.to_csv(metrics_path, index=False)
    means.to_csv(means_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)
    scenarios.groupby(["scenario", "method"], sort=False)[
        ["accuracy", "balanced_accuracy", "macro_f1"]
    ].mean().reset_index().to_csv(scenario_means_path, index=False)
    return metrics_path, means_path, scenarios_path, scenario_means_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(METHOD_SUFFIXES))
    parser.add_argument("outputs_root", type=Path)
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    print(*(str(path) for path in build(args.dataset, args.outputs_root, args.report_root)), sep="\n")


if __name__ == "__main__":
    main()
