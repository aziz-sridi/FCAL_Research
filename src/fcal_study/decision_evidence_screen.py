"""Frozen, one-seed exploratory validation screen for class-mean FCAL V2."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .compare import holm_adjust, paired_bootstrap_mean_interval
from .config import PROJECT_ROOT
from .utils import atomic_json, sha256_file
from .variants import DEFAULT_VARIANTS, parse_variant


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("+", "_plus")).strip("_")


def _runs(protocol):
    return {protocol["candidate"]["name"]: protocol["candidate"], **protocol["references"]}


def _verify(protocol):
    for relative, expected in protocol["implementation_files_sha256"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"Frozen implementation changed: {relative}")
    if sha256_file(PROJECT_ROOT / protocol["candidate"]["config"]) != protocol["candidate"]["config_sha256"]:
        raise ValueError("Frozen candidate config changed")
    if sha256_file(protocol["branch_checkpoint"]) != protocol["branch_sha256"]:
        raise ValueError("Branch checkpoint changed")
    for name, item in protocol["references"].items():
        if sha256_file(Path(item["run_dir"]) / "best-validation.pt") != item["checkpoint_sha256"]:
            raise ValueError(f"Reference checkpoint changed: {name}")
    split = PROJECT_ROOT / "splits/cub200_2011_seed42_val0.2"
    for name, expected in protocol["split_sha256"].items():
        if sha256_file(split / name) != expected:
            raise ValueError(f"Split changed: {name}")


def evaluate(protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _verify(protocol)
    root = protocol_path.parent
    expected = {parse_variant(x).key for x in DEFAULT_VARIANTS}
    for name, item in _runs(protocol).items():
        checkpoint = Path(item["run_dir"]) / "best-validation.pt"
        directory = root / "evaluation_seed271828" / _slug(name)
        provenance = {"checkpoint_sha256": sha256_file(checkpoint),
                      "protocol_sha256": sha256_file(protocol_path),
                      "split": "validation", "variant_seed": 271828,
                      "runner_sha256": sha256_file(__file__)}
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            previous = json.loads((directory / "provenance.json").read_text())
            rows = json.loads(summary_path.read_text())
            if previous != provenance or {x["variant"] for x in rows} != expected:
                raise ValueError(f"Conflicting or incomplete existing evaluation: {directory}")
            if any(not (directory / f"{key}.npz").is_file() for key in expected):
                raise ValueError(f"Missing prediction arrays: {directory}")
            print(f"Already complete: {name}", flush=True)
            continue
        if directory.exists():
            raise FileExistsError(f"Preserve and investigate partial evaluation: {directory}")
        directory.mkdir(parents=True)
        atomic_json(directory / "provenance.json", provenance)
        print(f"Evaluating {name}: all 30 validation views", flush=True)
        with (directory / "evaluation.log").open("w", encoding="utf-8") as log:
            subprocess.run([
                sys.executable, "-m", "fcal_study.evaluate", str(checkpoint),
                "--split", "validation", "--variant-seed", "271828",
                "--output-dir", str(directory),
            ], cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = json.loads(summary_path.read_text())
        if {x["variant"] for x in rows} != expected:
            raise ValueError(f"Incomplete new evaluation: {directory}")
        clean = next(x for x in rows if x["variant"] == "clean")
        print(f"Completed {name}: clean={clean['accuracy']:.4f}", flush=True)


def _history(run_dir):
    rows = [json.loads(x) for x in (run_dir / "history.jsonl").read_text().splitlines() if x.strip()]
    active = [x for x in rows if x["epoch"] > 5]
    if [x["epoch"] for x in active] != list(range(6, 51)):
        raise ValueError(f"Incomplete training history: {run_dir}")
    return {"post_branch_seconds": sum(x["train"]["seconds"] + x["validation"]["seconds"] for x in active),
            "steady_epoch_median_seconds": float(np.median([x["train"]["seconds"] + x["validation"]["seconds"] for x in active[1:]])),
            "peak_cuda_memory_bytes": max(x["train"]["peak_cuda_memory_bytes"] for x in active),
            "last5_validation_accuracy": float(np.mean([x["validation"]["accuracy"] for x in active[-5:]]))}


def report(protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _verify(protocol)
    root = protocol_path.parent
    expected = sorted(parse_variant(x).key for x in DEFAULT_VARIANTS)
    bg = [x for x in expected if x.startswith("bg_")]
    fg = [x for x in expected if x.startswith("fg_")]
    runs = _runs(protocol)
    metrics, arrays = [], {}
    reference_ids = reference_labels = None
    for name, item in runs.items():
        directory = root / "evaluation_seed271828" / _slug(name)
        provenance = json.loads((directory / "provenance.json").read_text())
        if provenance["protocol_sha256"] != sha256_file(protocol_path):
            raise ValueError("Evaluation protocol mismatch")
        if provenance["checkpoint_sha256"] != sha256_file(Path(item["run_dir"]) / "best-validation.pt"):
            raise ValueError("Evaluation checkpoint mismatch")
        summaries = {x["variant"]: x for x in json.loads((directory / "summary.json").read_text())}
        if set(summaries) != set(expected):
            raise ValueError(f"Incomplete evaluation: {name}")
        arrays[name] = {}
        for variant in expected:
            with np.load(directory / f"{variant}.npz") as data:
                if reference_ids is None:
                    reference_ids, reference_labels = data["image_id"].copy(), data["label"].copy()
                if not np.array_equal(reference_ids, data["image_id"]) or not np.array_equal(reference_labels, data["label"]):
                    raise ValueError(f"Unpaired images or labels: {name}/{variant}")
                arrays[name][variant] = (data["prediction"] == data["label"]).astype(float)
                if not np.isclose(arrays[name][variant].mean(), summaries[variant]["accuracy"]):
                    raise ValueError("Prediction/summary disagreement")
                if variant == "clean":
                    arrays[name]["leakage"] = data["strict_leakage"].copy()
                    arrays[name]["mask_valid"] = data["mask_valid"].astype(bool)
        metrics.append({"method": name, "samples": len(reference_ids),
                        "clean_accuracy": summaries["clean"]["accuracy"],
                        "bg_mean_accuracy": float(np.mean([summaries[x]["accuracy"] for x in bg])),
                        "bg_worst_accuracy": min(summaries[x]["accuracy"] for x in bg),
                        "fg_mean_accuracy": float(np.mean([summaries[x]["accuracy"] for x in fg])),
                        "foreground_only_accuracy": summaries["foreground_only"]["accuracy"],
                        "background_only_accuracy": summaries["background_only"]["accuracy"],
                        "strict_leakage": summaries["clean"]["strict_leakage"],
                        **_history(Path(item["run_dir"]))})
    frame = pd.DataFrame(metrics)
    frame.to_csv(root / "comparison_metrics.csv", index=False)
    candidate_name = protocol["candidate"]["name"]
    candidate = arrays[candidate_name]
    paired_rows, tests = [], []
    for name in protocol["references"]:
        reference = arrays[name]
        differences = {
            "clean_accuracy": candidate["clean"] - reference["clean"],
            "foreground_only_accuracy": candidate["foreground_only"] - reference["foreground_only"],
            "bg_mean_accuracy": np.mean([candidate[x] - reference[x] for x in bg], axis=0),
        }
        valid = candidate["mask_valid"] & reference["mask_valid"]
        differences["strict_leakage"] = candidate["leakage"][valid] - reference["leakage"][valid]
        for metric, values in differences.items():
            low, high = paired_bootstrap_mean_interval(values, samples=10000, seed=20260910)
            paired_rows.append({"reference": name, "metric": metric, "delta": float(values.mean()),
                                "ci_low": low, "ci_high": high})
        # Resample images jointly across all 13 variants, and reselect the
        # worst condition for each method in every bootstrap replicate.
        cand_bg = np.stack([candidate[x] for x in bg], axis=1)
        ref_bg = np.stack([reference[x] for x in bg], axis=1)
        rng = np.random.default_rng(20260910)
        worst = []
        for _ in range(100):
            indices = rng.integers(0, len(reference_ids), size=(100, len(reference_ids)))
            worst.extend(cand_bg[indices].mean(1).min(1) - ref_bg[indices].mean(1).min(1))
        low, high = np.quantile(worst, [.025, .975])
        paired_rows.append({"reference": name, "metric": "bg_worst_accuracy",
                            "delta": float(cand_bg.mean(0).min() - ref_bg.mean(0).min()),
                            "ci_low": float(low), "ci_high": float(high)})
        for variant in expected:
            c, r = candidate[variant].astype(bool), reference[variant].astype(bool)
            rescued, harmed = int((c & ~r).sum()), int((r & ~c).sum())
            p = binomtest(rescued, rescued + harmed, .5).pvalue if rescued + harmed else 1.
            tests.append({"reference": name, "variant": variant,
                          "delta": float(c.mean() - r.mean()), "rescued": rescued,
                          "harmed": harmed, "mcnemar_p": p})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "paired_intervals.csv", index=False)
    tests = pd.DataFrame(tests)
    tests["holm_p"] = holm_adjust(tests.mcnemar_p.to_numpy())
    tests.to_csv(root / "paired_accuracy_global_holm.csv", index=False)
    indexed = frame.set_index("method")
    c, r = indexed.loc[candidate_name], indexed.loc["FCAL V2"]
    limits = protocol["promotion_gates_vs_archived_v2"]
    gates = {
        "clean": c.clean_accuracy - r.clean_accuracy >= limits["clean_delta_min"],
        "bg_mean": c.bg_mean_accuracy - r.bg_mean_accuracy >= limits["bg_mean_delta_min"],
        "bg_worst": c.bg_worst_accuracy - r.bg_worst_accuracy >= limits["bg_worst_delta_min"],
        "foreground_only": c.foreground_only_accuracy - r.foreground_only_accuracy >= limits["foreground_only_delta_min"],
        "strict_leakage": c.strict_leakage - r.strict_leakage <= limits["strict_leakage_delta_max"],
        "historical_time": c.post_branch_seconds / r.post_branch_seconds <= limits["historical_post_branch_time_ratio_max"],
        "peak_memory": c.peak_cuda_memory_bytes / r.peak_cuda_memory_bytes <= limits["peak_memory_ratio_max"],
    }
    decision = {"decision": "eligible_for_independent_replication" if all(gates.values()) else "not_promoted",
                "gates": {k: bool(v) for k, v in gates.items()},
                "caveats": [protocol["scope"], protocol["cost_caveat"]],
                "accuracy_tests": len(tests), "holm_significant": int((tests.holm_p < .05).sum()),
                "protocol_sha256": sha256_file(protocol_path)}
    atomic_json(root / "decision.json", decision)
    print(frame.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["evaluate", "report"])
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    {"evaluate": evaluate, "report": report}[args.action](args.protocol.resolve())


if __name__ == "__main__":
    main()
