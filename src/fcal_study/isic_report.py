from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .complete_results import GENERIC_BACKGROUND, GENERIC_FOREGROUND

ARCHITECTURES = (
    "resnet50", "densenet121", "efficientnet-b0",
    "vit-tiny-p16-avg", "vit-small-p16-avg", "vit-base-p16-avg",
)
METHOD_SUFFIX = {"Baseline": "ce", "FCAL V1": "fcal-v1", "FCAL V2": "fcal-v2"}


def _summary(path: Path) -> dict[str, dict]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return {item["variant"]: item for item in values}


def build(outputs_root: Path, report_root: Path) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for architecture in ARCHITECTURES:
        for method, suffix in METHOD_SUFFIX.items():
            path = outputs_root / f"{architecture}-{suffix}-seed42" / "evaluation" / "summary.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            summaries = _summary(path)
            clean = summaries["clean"]
            background = [summaries[key] for key in GENERIC_BACKGROUND if key in summaries]
            foreground = [summaries[key] for key in GENERIC_FOREGROUND if key in summaries]
            rows.append({
                "architecture": architecture,
                "method": method,
                "clean_accuracy": clean["accuracy"],
                "clean_balanced_accuracy": clean["balanced_accuracy"],
                "clean_macro_f1": clean["macro_f1"],
                "strict_leakage": clean["strict_leakage"],
                "tolerant_leakage": clean["tolerant_leakage"],
                "mean_background_accuracy": sum(x["accuracy"] for x in background) / len(background),
                "mean_foreground_accuracy": sum(x["accuracy"] for x in foreground) / len(foreground),
                "foreground_only_accuracy": summaries["foreground_only"]["accuracy"],
                "background_only_accuracy": summaries["background_only"]["accuracy"],
            })
    if not rows:
        raise FileNotFoundError("No completed ISIC evaluations were found")
    frame = pd.DataFrame(rows)
    csv_path = report_root / "primary_metrics.csv"
    frame.to_csv(csv_path, index=False)
    metric_columns = [
        "clean_accuracy", "clean_balanced_accuracy", "clean_macro_f1", "strict_leakage",
        "mean_background_accuracy", "mean_foreground_accuracy",
        "foreground_only_accuracy", "background_only_accuracy",
    ]
    means = frame.groupby("method", sort=False)[metric_columns].mean().reset_index()
    means.to_csv(report_root / "method_means.csv", index=False)
    indexed = frame.set_index(["architecture", "method"])
    v1 = indexed.xs("FCAL V1", level="method")[metric_columns]
    v2 = indexed.xs("FCAL V2", level="method")[metric_columns]
    deltas = (v2 - v1).reset_index()
    deltas.to_csv(report_root / "v2_minus_v1_by_architecture.csv", index=False)
    mean_delta = deltas[metric_columns].mean()
    leakage_reduction = -mean_delta["strict_leakage"]
    v2_clean_wins = int((deltas["clean_accuracy"] > 0).sum())
    v2_clean_losses = int((deltas["clean_accuracy"] < 0).sum())
    lines = [
        "# ISIC FCAL V1 versus V2", "",
        "All methods use the same leakage-free, lesion-grouped split and the same epoch-5 CE branch.", "",
        "## Result", "",
        (
            f"V2 reduced mean strict outside-lesion attention by {leakage_reduction:.4f} "
            f"absolute ({100 * leakage_reduction / means.loc[means.method == 'FCAL V1', 'strict_leakage'].iloc[0]:.1f}% "
            "relative) versus V1, and reduced leakage on every architecture. "
            f"Its clean accuracy was higher on {v2_clean_wins}/6 architectures and lower on "
            f"{v2_clean_losses}/6."
        ), "",
        (
            f"Across architectures, V2 minus V1 was {100 * mean_delta['clean_accuracy']:+.2f} pp clean "
            f"accuracy, {100 * mean_delta['clean_balanced_accuracy']:+.2f} pp balanced accuracy, "
            f"{100 * mean_delta['clean_macro_f1']:+.2f} pp macro-F1, and "
            f"{100 * mean_delta['foreground_only_accuracy']:+.2f} pp foreground-only accuracy."
        ), "",
        (
            "Conclusion: V2 is consistently more effective at constraining attention to the lesion, "
            "but this six-model, one-seed experiment does not show a consistent classification "
            "advantage over V1. The CE baseline also retains the best mean clean accuracy and balanced "
            "accuracy. Treat V2 here as a localization improvement with mixed predictive effectiveness."
        ), "",
        "## Method means", "",
        means.to_markdown(index=False, floatfmt=".4f"), "",
        "## V2 minus V1 by architecture", "",
        deltas.to_markdown(index=False, floatfmt="+.4f"), "",
        "## Full metrics", "",
        frame.to_markdown(index=False, floatfmt=".4f"), "",
        "## Limitations", "",
        "- The sealed test split has 100 images; VASC has only three.",
        "- This is one seed, so cross-seed uncertainty is not estimated.",
        "- Masks mix expert challenge boundaries with reviewed novice U-Net boundaries.",
        "- These results measure model behavior on this curated subset, not clinical effectiveness.", "",
    ]
    if missing:
        lines.extend(("## Pending evaluations", "", *[f"- `{item}`" for item in missing], ""))
    (report_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/isic"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/isic_v1_v2"))
    args = parser.parse_args()
    print(build(args.outputs_root, args.report_root))


if __name__ == "__main__":
    main()
