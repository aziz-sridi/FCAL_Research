from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .utils import atomic_json


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _correct(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["prediction"] == data["label"]


def _transitions(clean: np.ndarray, variant: np.ndarray) -> dict[str, int]:
    return {
        "true_to_true": int((clean & variant).sum()),
        "true_to_false": int((clean & ~variant).sum()),
        "false_to_true": int((~clean & variant).sum()),
        "false_to_false": int((~clean & ~variant).sum()),
    }


def _js(probability: np.ndarray, reference: np.ndarray) -> np.ndarray:
    p = probability.astype(np.float64)
    q = reference.astype(np.float64)
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    tiny = np.finfo(np.float64).tiny
    return 0.5 * (p * (np.log(p + tiny) - np.log(m + tiny))).sum(axis=1) + 0.5 * (
        q * (np.log(q + tiny) - np.log(m + tiny))
    ).sum(axis=1)


def run(baseline_dir: str, fcal_dir: str, output_dir: str) -> Path:
    baseline_dir = Path(baseline_dir)
    fcal_dir = Path(fcal_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_clean = _load(baseline_dir / "clean.npz")
    fcal_clean = _load(fcal_dir / "clean.npz")
    if not np.array_equal(baseline_clean["image_id"], fcal_clean["image_id"]):
        raise ValueError("Baseline and FCAL image IDs are not aligned")
    base_clean_correct = _correct(baseline_clean)
    fcal_clean_correct = _correct(fcal_clean)
    rows = []
    detail = {}
    variant_names = sorted(path.stem for path in baseline_dir.glob("*.npz") if (fcal_dir / path.name).is_file())
    for name in variant_names:
        baseline = _load(baseline_dir / f"{name}.npz")
        fcal = _load(fcal_dir / f"{name}.npz")
        base_variant_correct = _correct(baseline)
        fcal_variant_correct = _correct(fcal)
        rescued = int((~base_variant_correct & fcal_variant_correct).sum())
        harmed = int((base_variant_correct & ~fcal_variant_correct).sum())
        discordant = rescued + harmed
        p_value = float(binomtest(min(rescued, harmed), discordant, 0.5).pvalue) if discordant else 1.0
        robust_rescue = int((base_clean_correct & ~base_variant_correct & fcal_clean_correct & fcal_variant_correct).sum())
        robust_harm = int((base_clean_correct & base_variant_correct & fcal_clean_correct & ~fcal_variant_correct).sum())
        base_js = _js(baseline["probabilities"], baseline_clean["probabilities"])
        fcal_js = _js(fcal["probabilities"], fcal_clean["probabilities"])
        row = {
            "variant": name,
            "baseline_accuracy": float(base_variant_correct.mean()),
            "fcal_accuracy": float(fcal_variant_correct.mean()),
            "accuracy_difference": float(fcal_variant_correct.mean() - base_variant_correct.mean()),
            "baseline_clean_to_variant_true_to_false": _transitions(base_clean_correct, base_variant_correct)["true_to_false"],
            "baseline_clean_to_variant_false_to_true": _transitions(base_clean_correct, base_variant_correct)["false_to_true"],
            "fcal_clean_to_variant_true_to_false": _transitions(fcal_clean_correct, fcal_variant_correct)["true_to_false"],
            "fcal_clean_to_variant_false_to_true": _transitions(fcal_clean_correct, fcal_variant_correct)["false_to_true"],
            "baseline_false_to_fcal_true_on_variant": rescued,
            "baseline_true_to_fcal_false_on_variant": harmed,
            "fcal_robustness_rescues": robust_rescue,
            "fcal_robustness_harms": robust_harm,
            "mcnemar_exact_p": p_value,
            "baseline_mean_js_from_clean": float(base_js.mean()),
            "fcal_mean_js_from_clean": float(fcal_js.mean()),
        }
        rows.append(row)
        detail[name] = {
            "baseline_clean_to_variant": _transitions(base_clean_correct, base_variant_correct),
            "fcal_clean_to_variant": _transitions(fcal_clean_correct, fcal_variant_correct),
            "cross_method_variant": {
                "baseline_false_fcal_true": rescued,
                "baseline_true_fcal_false": harmed,
                "both_true": int((base_variant_correct & fcal_variant_correct).sum()),
                "both_false": int((~base_variant_correct & ~fcal_variant_correct).sum()),
            },
            "strict_robustness": {"fcal_rescues": robust_rescue, "fcal_harms": robust_harm},
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "transition_summary.csv", index=False)
    atomic_json(output / "transition_detail.json", detail)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_evaluation")
    parser.add_argument("fcal_evaluation")
    parser.add_argument("output")
    args = parser.parse_args()
    print(run(args.baseline_evaluation, args.fcal_evaluation, args.output))


if __name__ == "__main__":
    main()
