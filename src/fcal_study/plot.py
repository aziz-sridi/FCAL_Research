from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _history(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append({
            "epoch": item["epoch"],
            "train_ce": item["train"]["ce"],
            "train_total": item["train"]["total"],
            "train_fcal": item["train"]["fcal"],
            "train_weighted_fcal": item["train"]["weighted_fcal"],
            "lambda": item["train"]["lambda"],
            "validation_ce": item["validation"]["ce"],
            "validation_accuracy": item["validation"]["accuracy"],
        })
    return pd.DataFrame(rows)


def run(baseline_run: str, fcal_run: str, comparison_csv: str, output_dir: str) -> Path:
    baseline = _history(Path(baseline_run) / "history.jsonl")
    fcal = _history(Path(fcal_run) / "history.jsonl")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(baseline.epoch, baseline.train_ce, label="CE baseline")
    axes[0, 0].plot(fcal.epoch, fcal.train_ce, label="Adaptive FCAL: CE component")
    axes[0, 0].set(title="Training CE", xlabel="Epoch", ylabel="CE")
    axes[0, 0].legend()
    axes[0, 1].plot(baseline.epoch, baseline.validation_ce, label="CE baseline")
    axes[0, 1].plot(fcal.epoch, fcal.validation_ce, label="Adaptive FCAL")
    axes[0, 1].set(title="Validation CE", xlabel="Epoch", ylabel="CE")
    axes[0, 1].legend()
    axes[1, 0].plot(fcal.epoch, fcal.train_fcal, label="Raw FCAL")
    axes[1, 0].plot(fcal.epoch, fcal.train_weighted_fcal, label="Weighted FCAL")
    axes[1, 0].plot(fcal.epoch, fcal["lambda"], label="Adaptive lambda")
    axes[1, 0].set(title="FCAL components", xlabel="Epoch")
    axes[1, 0].legend()
    axes[1, 1].plot(baseline.epoch, baseline.validation_accuracy, label="CE baseline")
    axes[1, 1].plot(fcal.epoch, fcal.validation_accuracy, label="Adaptive FCAL")
    axes[1, 1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1, 1].legend()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"training_comparison.{extension}", dpi=180)
    plt.close(fig)

    transitions = pd.read_csv(comparison_csv)
    selected = transitions[transitions.variant.str.startswith("bg_")].copy()
    selected = selected.sort_values("accuracy_difference")
    fig, ax = plt.subplots(figsize=(11, max(5, len(selected) * 0.35)), constrained_layout=True)
    ax.barh(selected.variant, selected.accuracy_difference * 100)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="FCAL - baseline accuracy (percentage points)", title="Background robustness difference")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"background_accuracy_difference.{extension}", dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_run")
    parser.add_argument("fcal_run")
    parser.add_argument("comparison_csv")
    parser.add_argument("output")
    args = parser.parse_args()
    print(run(args.baseline_run, args.fcal_run, args.comparison_csv, args.output))


if __name__ == "__main__":
    main()
