from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .utils import atomic_json


def paired_bootstrap_mean_interval(
    differences: np.ndarray,
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for a paired mean difference."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired bootstrap requires a non-empty vector")
    if int(samples) <= 0 or not 0.0 < float(confidence) < 1.0:
        raise ValueError("invalid bootstrap settings")
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    chunk = 1_000
    for start in range(0, int(samples), chunk):
        stop = min(start + chunk, int(samples))
        indices = generator.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(low), float(high)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise error correction."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


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


def run(
    baseline_dir: str,
    fcal_dir: str,
    output_dir: str,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> Path:
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
    for variant_index, name in enumerate(variant_names):
        baseline = _load(baseline_dir / f"{name}.npz")
        fcal = _load(fcal_dir / f"{name}.npz")
        base_variant_correct = _correct(baseline)
        fcal_variant_correct = _correct(fcal)
        rescued = int((~base_variant_correct & fcal_variant_correct).sum())
        harmed = int((base_variant_correct & ~fcal_variant_correct).sum())
        discordant = rescued + harmed
        p_value = float(binomtest(min(rescued, harmed), discordant, 0.5).pvalue) if discordant else 1.0
        accuracy_delta = fcal_variant_correct.astype(float) - base_variant_correct.astype(float)
        accuracy_ci_low, accuracy_ci_high = paired_bootstrap_mean_interval(
            accuracy_delta,
            samples=bootstrap_samples,
            seed=int(bootstrap_seed) + variant_index,
        )
        robust_rescue = int((base_clean_correct & ~base_variant_correct & fcal_clean_correct & fcal_variant_correct).sum())
        robust_harm = int((base_clean_correct & base_variant_correct & fcal_clean_correct & ~fcal_variant_correct).sum())
        base_js = _js(baseline["probabilities"], baseline_clean["probabilities"])
        fcal_js = _js(fcal["probabilities"], fcal_clean["probabilities"])
        row = {
            "variant": name,
            "baseline_accuracy": float(base_variant_correct.mean()),
            "fcal_accuracy": float(fcal_variant_correct.mean()),
            "accuracy_difference": float(fcal_variant_correct.mean() - base_variant_correct.mean()),
            "accuracy_difference_ci_low": accuracy_ci_low,
            "accuracy_difference_ci_high": accuracy_ci_high,
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
        if name == "clean" and "strict_leakage" in baseline and "strict_leakage" in fcal:
            valid = np.ones(len(baseline["strict_leakage"]), dtype=bool)
            if "mask_valid" in baseline and "mask_valid" in fcal:
                valid = baseline["mask_valid"].astype(bool) & fcal["mask_valid"].astype(bool)
            leakage_delta = (
                fcal["strict_leakage"][valid].astype(float)
                - baseline["strict_leakage"][valid].astype(float)
            )
            leakage_ci_low, leakage_ci_high = paired_bootstrap_mean_interval(
                leakage_delta,
                samples=bootstrap_samples,
                seed=int(bootstrap_seed) + len(variant_names),
            )
            row.update({
                "strict_leakage_difference": float(leakage_delta.mean()),
                "strict_leakage_difference_ci_low": leakage_ci_low,
                "strict_leakage_difference_ci_high": leakage_ci_high,
            })
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
    frame["mcnemar_exact_p_holm"] = holm_adjust(
        frame["mcnemar_exact_p"].to_numpy(dtype=float)
    )
    frame.to_csv(output / "transition_summary.csv", index=False)
    atomic_json(output / "transition_detail.json", detail)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_evaluation")
    parser.add_argument("fcal_evaluation")
    parser.add_argument("output")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    print(run(
        args.baseline_evaluation,
        args.fcal_evaluation,
        args.output,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    ))


if __name__ == "__main__":
    main()
