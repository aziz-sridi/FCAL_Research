from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .complete_results import GENERIC_BACKGROUND, GENERIC_FOREGROUND

ARCHITECTURES = (
    ("resnet50", "ResNet-50", "CNN"),
    ("densenet121", "DenseNet-121", "CNN"),
    ("efficientnet-b0", "EfficientNet-B0", "CNN"),
    ("vit-tiny-p16-avg", "ViT-Tiny/16 avg", "ViT"),
    ("vit-small-p16-avg", "ViT-Small/16 avg", "ViT"),
    ("vit-base-p16-avg", "ViT-Base/16 avg", "ViT"),
)
METHODS = (
    ("CE", "ce"),
    ("FCAL V1", "fcal-v1"),
    ("FCAL V2", "fcal-v2"),
    ("Universal FCAL", "ufcal-auto-safe"),
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def _markdown(frame: pd.DataFrame, formats: dict[str, callable] | None = None) -> str:
    formats = formats or {}
    columns = list(frame.columns)
    values = []
    for _, row in frame.iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            if column in formats and pd.notna(value):
                value = formats[column](value)
            elif isinstance(value, float):
                value = f"{value:.4f}"
            rendered.append(str(value))
        values.append(rendered)
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    return "\n".join([header, divider, *("| " + " | ".join(row) + " |" for row in values)])


def _paired_clean(ce_path: Path, method_path: Path) -> tuple[int, int, float]:
    with np.load(ce_path) as ce, np.load(method_path) as method:
        if not np.array_equal(ce["image_id"], method["image_id"]):
            raise ValueError(f"Evaluation IDs are not aligned: {ce_path} and {method_path}")
        ce_correct = ce["prediction"] == ce["label"]
        method_correct = method["prediction"] == method["label"]
        rescued = int((~ce_correct & method_correct).sum())
        harmed = int((ce_correct & ~method_correct).sum())
    discordant = rescued + harmed
    p_value = float(binomtest(min(rescued, harmed), discordant, 0.5).pvalue) if discordant else 1.0
    return rescued, harmed, p_value


def build(outputs_root: Path, report_root: Path) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    primary_rows: list[dict] = []
    variant_rows: list[dict] = []
    missing: list[str] = []

    for slug, architecture, family in ARCHITECTURES:
        for method, suffix in METHODS:
            run_dir = outputs_root / f"{slug}-{suffix}-seed42"
            train_path = run_dir / "summary.json"
            eval_path = run_dir / "evaluation" / "summary.json"
            if not train_path.is_file() or not eval_path.is_file():
                missing.append(str(run_dir))
                continue
            training = _read_json(train_path)
            summaries = {item["variant"]: item for item in _read_json(eval_path)}
            required = {"clean", "foreground_only", "background_only"}
            if not required.issubset(summaries):
                missing.append(f"{run_dir} (full evaluation incomplete)")
                continue
            clean = summaries["clean"]
            background = [summaries[key] for key in GENERIC_BACKGROUND if key in summaries]
            foreground = [summaries[key] for key in GENERIC_FOREGROUND if key in summaries]
            if not background or not foreground:
                missing.append(f"{run_dir} (regional variants incomplete)")
                continue
            mean_background = float(np.mean([item["accuracy"] for item in background]))
            mean_foreground = float(np.mean([item["accuracy"] for item in foreground]))
            primary_rows.append({
                "family": family,
                "architecture": architecture,
                "method": method,
                "best_validation_accuracy": training["best_validation_accuracy"],
                "clean_accuracy": clean["accuracy"],
                "clean_balanced_accuracy": clean["balanced_accuracy"],
                "clean_macro_f1": clean["macro_f1"],
                "strict_leakage": clean["strict_leakage"],
                "valid_mask_fraction": clean["valid_mask_fraction"],
                "mean_background_accuracy": mean_background,
                "mean_foreground_accuracy": mean_foreground,
                "mean_background_drop": clean["accuracy"] - mean_background,
                "mean_foreground_drop": clean["accuracy"] - mean_foreground,
                "foreground_only_accuracy": summaries["foreground_only"]["accuracy"],
                "background_only_accuracy": summaries["background_only"]["accuracy"],
            })
            for variant, result in summaries.items():
                variant_rows.append({
                    "family": family,
                    "architecture": architecture,
                    "method": method,
                    "variant": variant,
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "macro_f1": result["macro_f1"],
                })

    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"The 24-run full benchmark is incomplete:\n{detail}")

    primary = pd.DataFrame(primary_rows)
    variants = pd.DataFrame(variant_rows)
    if len(primary) != len(ARCHITECTURES) * len(METHODS):
        raise RuntimeError(f"Expected 24 primary rows, found {len(primary)}")
    primary.to_csv(report_root / "primary_metrics.csv", index=False)
    variants.to_csv(report_root / "variant_metrics.csv", index=False)

    metric_columns = [
        "clean_accuracy", "clean_balanced_accuracy", "clean_macro_f1", "strict_leakage",
        "mean_background_accuracy", "mean_foreground_accuracy", "mean_background_drop",
        "mean_foreground_drop", "foreground_only_accuracy", "background_only_accuracy",
    ]
    method_means = primary.groupby("method", sort=False)[metric_columns].mean().reset_index()
    method_means.to_csv(report_root / "method_means.csv", index=False)

    indexed = primary.set_index(["architecture", "method"])
    delta_rows = []
    paired_rows = []
    for slug, architecture, family in ARCHITECTURES:
        ce = indexed.loc[(architecture, "CE")]
        ce_npz = outputs_root / f"{slug}-ce-seed42" / "evaluation" / "clean.npz"
        for method, suffix in METHODS[1:]:
            current = indexed.loc[(architecture, method)]
            row = {"family": family, "architecture": architecture, "method": method}
            row.update({column: current[column] - ce[column] for column in metric_columns})
            delta_rows.append(row)
            method_npz = outputs_root / f"{slug}-{suffix}-seed42" / "evaluation" / "clean.npz"
            rescued, harmed, p_value = _paired_clean(ce_npz, method_npz)
            paired_rows.append({
                "family": family,
                "architecture": architecture,
                "method": method,
                "rescued_ce_errors": rescued,
                "harmed_ce_correct": harmed,
                "mcnemar_exact_p": p_value,
            })
    deltas = pd.DataFrame(delta_rows)
    paired = pd.DataFrame(paired_rows)
    deltas.to_csv(report_root / "deltas_vs_ce.csv", index=False)
    paired.to_csv(report_root / "paired_clean_vs_ce.csv", index=False)

    display_primary = primary[[
        "family", "architecture", "method", "best_validation_accuracy", "clean_accuracy",
        "clean_balanced_accuracy", "clean_macro_f1", "strict_leakage",
        "mean_background_accuracy", "mean_foreground_accuracy", "foreground_only_accuracy",
        "background_only_accuracy",
    ]].copy()
    display_primary.columns = [
        "Family", "Architecture", "Method", "Best val", "Clean", "Balanced", "Macro-F1",
        "Strict leakage", "Mean BG corrupt.", "Mean FG corrupt.", "Foreground only",
        "Background only",
    ]
    accuracy_formats = {column: _pct for column in display_primary.columns[3:]}

    display_means = method_means[[
        "method", "clean_accuracy", "strict_leakage", "mean_background_accuracy",
        "mean_foreground_accuracy", "foreground_only_accuracy", "background_only_accuracy",
    ]].copy()
    display_means.columns = [
        "Method", "Clean", "Strict leakage", "Mean BG corrupt.", "Mean FG corrupt.",
        "Foreground only", "Background only",
    ]

    best_clean = method_means.loc[method_means["clean_accuracy"].idxmax()]
    best_leak = method_means.loc[method_means["strict_leakage"].idxmin()]
    best_bg = method_means.loc[method_means["mean_background_accuracy"].idxmax()]
    best_fg = method_means.loc[method_means["mean_foreground_accuracy"].idxmax()]
    sig = paired[paired["mcnemar_exact_p"] < 0.05]
    mean_deltas = deltas.groupby("method", sort=False)[metric_columns].mean()
    v2_delta = mean_deltas.loc["FCAL V2"]
    v1_delta = mean_deltas.loc["FCAL V1"]
    universal_delta = mean_deltas.loc["Universal FCAL"]
    clean_wins = deltas.assign(win=deltas["clean_accuracy"] > 0).groupby("method")["win"].sum()
    leakage_wins = deltas.assign(win=deltas["strict_leakage"] < 0).groupby("method")["win"].sum()
    paired_with_delta = paired.merge(
        deltas[["architecture", "method", "clean_accuracy"]],
        on=["architecture", "method"],
    )
    significant_positive = int(
        ((paired_with_delta["mcnemar_exact_p"] < 0.05) & (paired_with_delta["clean_accuracy"] > 0)).sum()
    )
    significant_negative = int(
        ((paired_with_delta["mcnemar_exact_p"] < 0.05) & (paired_with_delta["clean_accuracy"] < 0)).sum()
    )

    lines = [
        "# Oxford-IIIT Pet full FCAL benchmark", "",
        "## Protocol", "",
        "- Dataset: Oxford-IIIT Pet, with 37 breed classes and official pixel-level trimaps.",
        "- Split: 2,944 train / 736 validation from official trainval; the official 3,669-image test partition is held out for final evaluation.",
        "- Architectures: ResNet-50, DenseNet-121, EfficientNet-B0, ViT-Tiny/16, ViT-Small/16, and ViT-Base/16.",
        "- Methods: CE, FCAL V1, FCAL V2, and automatic-safe Universal FCAL; each FCAL run resumes from its architecture's exact CE epoch-5 branch.",
        "- Training: seed 42, 50 total epochs, checkpoint selected by validation accuracy.",
        "- Test suite: clean plus 29 deterministic regional perturbations, including background/foreground corruptions and region-only ablations.",
        "- Classification uses all 3,669 test images. Localization averages exclude only degenerate center-cropped masks.", "",
        "## Aggregate result", "",
        (
            f"Across the six architectures, **{best_clean['method']}** has the highest mean clean accuracy "
            f"({_pct(best_clean['clean_accuracy'])}); **{best_leak['method']}** has the lowest mean strict "
            f"leakage ({_pct(best_leak['strict_leakage'])}). **{best_bg['method']}** is strongest on mean "
            f"background corruptions ({_pct(best_bg['mean_background_accuracy'])}), while "
            f"**{best_fg['method']}** is strongest on mean foreground corruptions "
            f"({_pct(best_fg['mean_foreground_accuracy'])})."
        ), "",
        (
            f"Relative to CE, V2 averages {_pp(v2_delta['clean_accuracy'])} clean accuracy, "
            f"{_pp(v2_delta['strict_leakage'])} strict leakage, "
            f"{_pp(v2_delta['mean_background_accuracy'])} mean background-corruption accuracy, and "
            f"{_pp(v2_delta['mean_foreground_accuracy'])} mean foreground-corruption accuracy. It "
            f"improves clean accuracy on {int(clean_wins['FCAL V2'])}/6 architectures and reduces "
            f"strict leakage on {int(leakage_wins['FCAL V2'])}/6."
        ), "",
        (
            f"Universal FCAL averages {_pp(universal_delta['clean_accuracy'])} clean accuracy and "
            f"{_pp(universal_delta['mean_foreground_accuracy'])} foreground-corruption accuracy, but "
            f"only {_pp(universal_delta['strict_leakage'])} strict leakage. V1 averages "
            f"{_pp(v1_delta['clean_accuracy'])} clean accuracy and {_pp(v1_delta['strict_leakage'])} "
            f"strict leakage; its {_pp(v1_delta['foreground_only_accuracy'])} foreground-only result "
            "is driven by a large EfficientNet-B0 regression."
        ), "",
        (
            f"Exact paired clean-test McNemar tests find {len(sig)}/{len(paired)} FCAL-versus-CE "
            f"comparisons below p=0.05: {significant_positive} favor FCAL and {significant_negative} "
            "favor CE. These are single-seed results; the tests measure paired test-set disagreement, "
            "not training-seed uncertainty."
        ), "",
        "## Method means across six architectures", "",
        _markdown(display_means, {column: _pct for column in display_means.columns[1:]}), "",
        "## All 24 clean and robustness results", "",
        _markdown(display_primary, accuracy_formats), "",
        "## Interpretation notes", "",
        "- Higher accuracy and lower leakage are better. `Mean BG corrupt.` and `Mean FG corrupt.` average the 13 background-only and 13 foreground-only corruption variants, respectively.",
        "- `Foreground only` replaces the background with the ImageNet mean; `Background only` replaces the pet region.",
        "- Strict leakage is the cross-method localization metric. Tolerant leakage is intentionally omitted because legacy FCAL and Universal FCAL use different tolerance conventions.",
        "- This is a one-seed benchmark. Architecture-level differences should be confirmed across seeds before being treated as stable rankings.", "",
        "## Generated artifacts", "",
        "- `primary_metrics.csv`: one row per architecture and method.",
        "- `variant_metrics.csv`: one row per architecture, method, and perturbation.",
        "- `method_means.csv`: six-architecture method averages.",
        "- `deltas_vs_ce.csv`: architecture-level FCAL minus CE differences.",
        "- `paired_clean_vs_ce.csv`: clean paired rescue/harm counts and exact McNemar p-values.", "",
    ]
    result_path = report_root / "RESULTS.md"
    result_path.write_text("\n".join(lines), encoding="utf-8")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/oxford_pet"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/oxford_pet"))
    args = parser.parse_args()
    print(build(args.outputs_root, args.report_root))


if __name__ == "__main__":
    main()
