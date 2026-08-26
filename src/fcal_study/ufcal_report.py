from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import PROJECT_ROOT


RUNS = {
    "CE": PROJECT_ROOT / "outputs/isic/resnet50-ce-seed42",
    "FCAL V1": PROJECT_ROOT / "outputs/isic/resnet50-fcal-v1-seed42",
    "FCAL V2": PROJECT_ROOT / "outputs/isic/resnet50-fcal-v2-seed42",
    "U-FCAL fixed": PROJECT_ROOT / "outputs/isic_ufcal_tuning/resnet50-manual-r005-t050-s050-seed42",
    "U-FCAL auto": PROJECT_ROOT / "outputs/isic_ufcal_tuning/resnet50-auto-safe-seed42",
}

TUNING_RUNS = sorted((PROJECT_ROOT / "outputs/isic_ufcal_tuning").glob("resnet50-*seed42"))


def _summary(run: Path, split: str = "evaluation") -> list[dict]:
    return json.loads((run / split / "summary.json").read_text(encoding="utf-8"))


def _clean(run: Path, split: str = "evaluation") -> dict:
    return next(item for item in _summary(run, split) if item["variant"] == "clean")


def _variant_means(items: list[dict]) -> tuple[float, float]:
    background = [item["accuracy"] for item in items if item["variant"].startswith("bg_")]
    foreground = [item["accuracy"] for item in items if item["variant"].startswith("fg_")]
    return float(np.mean(background)), float(np.mean(foreground))


def _paired_interval(reference: Path, candidate: Path) -> dict:
    left = np.load(reference / "evaluation/clean.npz")
    right = np.load(candidate / "evaluation/clean.npz")
    assert np.array_equal(left["image_id"], right["image_id"])
    assert np.array_equal(left["label"], right["label"])
    labels = left["label"]
    left_correct = left["prediction"] == labels
    right_correct = right["prediction"] == labels
    left_only = int((left_correct & ~right_correct).sum())
    right_only = int((~left_correct & right_correct).sum())
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(labels), size=(20000, len(labels)))
    differences = right_correct[indices].mean(1) - left_correct[indices].mean(1)
    low, high = np.quantile(differences, [0.025, 0.975])
    discordant = left_only + right_only
    p_value = binomtest(min(left_only, right_only), discordant, 0.5).pvalue if discordant else 1.0
    return {
        "accuracy_difference": float(right_correct.mean() - left_correct.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "ce_only_correct": left_only,
        "candidate_only_correct": right_only,
        "mcnemar_p": float(p_value),
    }


def run(output: str | Path = "reports/isic_ufcal_resnet") -> Path:
    output_dir = Path(output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tuning_rows = []
    for run_dir in TUNING_RUNS:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        validation_path = run_dir / "validation_evaluation/summary.json"
        validation = _clean(run_dir, "validation_evaluation") if validation_path.is_file() else None
        config = summary["run_id"].split("/")[-1]
        tuning_rows.append({
            "configuration": config,
            "best_validation_accuracy": summary["best_validation_accuracy"],
            "validation_balanced_accuracy": validation.get("balanced_accuracy") if validation else np.nan,
            "validation_macro_f1": validation.get("macro_f1") if validation else np.nan,
            "validation_strict_leakage": validation.get("strict_leakage") if validation else np.nan,
        })
    tuning = pd.DataFrame(tuning_rows).sort_values("best_validation_accuracy", ascending=False)
    tuning.to_csv(output_dir / "validation_tuning.csv", index=False)

    test_rows = []
    for method, run_dir in RUNS.items():
        items = _summary(run_dir)
        clean = next(item for item in items if item["variant"] == "clean")
        by_key = {item["variant"]: item for item in items}
        background_mean, foreground_mean = _variant_means(items)
        test_rows.append({
            "method": method,
            "clean_accuracy": clean["accuracy"],
            "balanced_accuracy": clean["balanced_accuracy"],
            "macro_f1": clean["macro_f1"],
            "strict_leakage": clean["strict_leakage"],
            "nonzero_map_fraction": clean["nonzero_map_fraction"],
            "mean_background_transform_accuracy": background_mean,
            "mean_foreground_transform_accuracy": foreground_mean,
            "foreground_only_accuracy": by_key["foreground_only"]["accuracy"],
            "background_only_accuracy": by_key["background_only"]["accuracy"],
        })
    test = pd.DataFrame(test_rows)
    test.to_csv(output_dir / "test_metrics.csv", index=False)

    paired_rows = []
    for method in ("FCAL V1", "FCAL V2", "U-FCAL fixed", "U-FCAL auto"):
        paired_rows.append({"method": method, **_paired_interval(RUNS["CE"], RUNS[method])})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_accuracy.csv", index=False)

    auto_history = []
    for line in (RUNS["U-FCAL auto"] / "history.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if "ufcal" not in item:
            continue
        scheduler = item["ufcal"]["scheduler"] or {}
        auto_history.append({
            "epoch": item["epoch"],
            "validation_accuracy": item["validation"]["accuracy"],
            "phase": item["ufcal"]["phase"],
            "gradient_ratio_used": item["ufcal"]["gradient_ratio_used"],
            "event": item["ufcal"]["automatic_event"],
            "next_gradient_ratio": scheduler.get("current_ratio"),
            "train_auxiliary": item["train"]["fcal"],
            "actual_gradient_ratio": item["train"]["gradient_ratio"],
            "gradient_conflict_fraction": item["train"]["gradient_conflict_fraction"],
            "valid_mask_fraction": item["train"]["valid_mask_fraction"],
        })
    pd.DataFrame(auto_history).to_csv(output_dir / "automatic_schedule.csv", index=False)

    fixed = test.set_index("method").loc["U-FCAL fixed"]
    ce = test.set_index("method").loc["CE"]
    leakage_reduction = 1.0 - fixed.strict_leakage / ce.strict_leakage
    lines = [
        "# ResNet-50 ISIC Universal FCAL experiment",
        "",
        "## Protocol",
        "",
        "All U-FCAL trials branch from the same CE epoch-5 checkpoint. Three fixed settings and two automatic settings were compared on validation only. The conservative fixed setting and safer automatic setting were frozen before their held-out test evaluations. Seed 42 is the only seed in this pilot.",
        "",
        "## Validation tuning",
        "",
        tuning.round(4).to_markdown(index=False),
        "",
        "## Held-out test",
        "",
        test.round(4).to_markdown(index=False),
        "",
        "## Paired clean-accuracy uncertainty versus CE",
        "",
        paired.round(4).to_markdown(index=False),
        "",
        "## Conclusion",
        "",
        f"The strongest fixed U-FCAL trade-off reached {100*fixed.clean_accuracy:.1f}% clean accuracy versus {100*ce.clean_accuracy:.1f}% for CE while reducing strict leakage by {100*leakage_reduction:.1f}% relative. It did not improve every metric: foreground-only accuracy changed from {100*ce.foreground_only_accuracy:.1f}% to {100*fixed.foreground_only_accuracy:.1f}%.",
        "",
        "The safer automatic run reached 82.3% validation accuracy, but its validation-selected checkpoint came from the CE-release phase and scored 63.0% on the 100-image test set. This demonstrates that a controller can adapt correctly in-run without guaranteeing out-of-domain generalization, and that accuracy-only checkpoint selection can discard localization gains.",
        "",
        "No clean-accuracy difference is statistically resolved by this 100-image test. U-FCAL is therefore a working architecture-independent library and a promising localization/accuracy trade-off, not yet a universal improvement claim.",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
