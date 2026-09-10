"""Aggregate fixed explanation baselines on CUB without test-set selection.

The existing CE and FCAL V2 table is treated as a published compact index, not
as an authority for run identity.  Its rows are checked against the archived
evaluation summaries before they are combined with complete current candidate
matrices.
Training efficiency is always read from complete histories, and the common
five-epoch CE branch is verified record-for-record before post-branch costs are
reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .complete_results import (
    ARCHITECTURES,
    GENERIC_BACKGROUND,
    GENERIC_FOREGROUND,
)
from .evidence import efficiency_overhead, pareto_front, run_efficiency, win_tie_loss

SEED = 42
COMMON_BRANCH_EPOCHS = 5
EXPECTED_EPOCHS = 50
REFERENCE_SPECS = (
    ("CE", "ce", "archive", "ce"),
    ("FCAL V2", "fcal-v5", "archive", "fcal_v5"),
)
CURRENT_CANDIDATE_SPECS = (
    ("CAMAL", "camal", "current", "camal"),
    ("GAIN-ext", "gain-ext", "current", "gain"),
    ("RRR", "rrr", "current", "rrr"),
)
SUPPORT_RESTRICTED_SPECS = (
    ("Faster-ISNet", "isnet", "current", "isnet", ("resnet50",)),
)
BASE_REFERENCES = ("CE", "FCAL V2")
REQUIRED_RUN_ARTIFACTS = (
    "config.yaml",
    "environment.json",
    "history.jsonl",
    "summary.json",
    "evaluation/summary.json",
    "best-validation.pt",
)
METRIC_COLUMNS = (
    "clean_accuracy",
    "clean_balanced_accuracy",
    "clean_macro_f1",
    "strict_leakage",
    "bg_mean_accuracy",
    "bg_worst_accuracy",
    "fg_mean_accuracy",
    "fg_worst_accuracy",
    "foreground_only_accuracy",
    "background_only_accuracy",
)
WTL_METRICS = (
    "clean_accuracy",
    "bg_mean_accuracy",
    "bg_worst_accuracy",
    "fg_mean_accuracy",
    "fg_worst_accuracy",
    "foreground_only_accuracy",
    "background_only_accuracy",
    "strict_leakage",
)
LOWER_IS_BETTER = frozenset({"strict_leakage"})
PARETO_MAXIMIZE = (
    "clean_accuracy",
    "bg_mean_accuracy",
    "foreground_only_accuracy",
)
PARETO_MINIMIZE = (
    "strict_leakage",
    "post_branch_wall_seconds",
    "peak_cuda_memory_gib",
)
HISTORY_ACCOUNTING_NOTES = {
    ("efficientnet-b0", "RRR"): (
        "recorded_complete_history_only; interrupted_unrecorded_epoch46_attempt_excluded"
    ),
    ("resnet50", "Faster-ISNet"): (
        "clean_rerun_recorded_complete_history_only; "
        "interrupted_corrupt_checkpoint_directory_excluded; "
        "early_nondeterministic_divergence_observed_but_late_training_converged"
    ),
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_by_variant(path: Path) -> dict[str, dict]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"Evaluation summary must be a list: {path}")
    result = {row["variant"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate evaluation variant: {path}")
    return result


def _classification_fields(summary_path: Path, clean: dict) -> tuple[float, float]:
    """Read modern summary fields or reproduce them from archived predictions."""
    if "balanced_accuracy" in clean and "macro_f1" in clean:
        return float(clean["balanced_accuracy"]), float(clean["macro_f1"])
    prediction_path = summary_path.parent / "clean.npz"
    if not prediction_path.is_file():
        raise ValueError(
            f"{summary_path} omits balanced_accuracy/macro_f1 and {prediction_path} is missing"
        )
    with np.load(prediction_path) as archive:
        labels = archive["label"]
        predictions = archive["prediction"]
    recalls = []
    f1_scores = []
    for class_index in range(200):
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


def metrics_from_summary(path: Path) -> dict[str, float]:
    """Compute the exact metrics used by the existing complete CUB table."""
    summary = _summary_by_variant(path)
    required = {
        "clean", "foreground_only", "background_only",
        *GENERIC_BACKGROUND, *GENERIC_FOREGROUND,
    }
    missing = sorted(required.difference(summary))
    if missing:
        raise ValueError(f"{path} is missing variants: {', '.join(missing)}")
    clean = summary["clean"]
    for field in ("accuracy", "strict_leakage"):
        if field not in clean:
            raise ValueError(f"{path} clean row is missing {field}")
    clean_balanced_accuracy, clean_macro_f1 = _classification_fields(path, clean)
    background = [float(summary[name]["accuracy"]) for name in GENERIC_BACKGROUND]
    foreground = [float(summary[name]["accuracy"]) for name in GENERIC_FOREGROUND]
    return {
        "clean_accuracy": float(clean["accuracy"]),
        "clean_balanced_accuracy": clean_balanced_accuracy,
        "clean_macro_f1": clean_macro_f1,
        "strict_leakage": float(clean["strict_leakage"]),
        "bg_mean_accuracy": float(np.mean(background)),
        "bg_worst_accuracy": float(np.min(background)),
        "fg_mean_accuracy": float(np.mean(foreground)),
        "fg_worst_accuracy": float(np.min(foreground)),
        "foreground_only_accuracy": float(summary["foreground_only"]["accuracy"]),
        "background_only_accuracy": float(summary["background_only"]["accuracy"]),
    }


def validate_existing_metric_row(
    published_row: pd.Series | dict,
    recomputed: dict[str, float],
    *,
    identity: str,
    atol: float = 1e-12,
) -> None:
    """Fail if an archived run does not reproduce its compact published row."""
    mismatches = []
    for metric in METRIC_COLUMNS:
        published = float(published_row[metric])
        observed = float(recomputed[metric])
        if not np.isclose(published, observed, rtol=0.0, atol=atol, equal_nan=True):
            mismatches.append(f"{metric}: table={published!r}, archive={observed!r}")
    if mismatches:
        raise ValueError(f"Published/archive mismatch for {identity}: {'; '.join(mismatches)}")


def validate_history(history: Sequence[dict], *, identity: str) -> None:
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, EXPECTED_EPOCHS + 1)):
        raise ValueError(f"{identity} must contain epochs 1..{EXPECTED_EPOCHS}; got {epochs}")
    for row in history:
        for phase in ("train", "validation"):
            if float(row[phase]["seconds"]) <= 0 or int(row[phase]["samples"]) <= 0:
                raise ValueError(f"{identity} has invalid {phase} accounting at epoch {row['epoch']}")


def validate_common_prefix(
    reference: Sequence[dict],
    candidate: Sequence[dict],
    *,
    candidate_name: str,
    epochs: int = COMMON_BRANCH_EPOCHS,
) -> None:
    """Require literal shared-history equality before paired branch reporting."""
    if list(reference[:epochs]) != list(candidate[:epochs]):
        raise ValueError(f"{candidate_name} does not share CE's first {epochs} history records")


def selected_epoch(history: Sequence[dict], best_validation_accuracy: float) -> int:
    """Return the first epoch attaining the stored validation-selected score."""
    scores = np.asarray([float(row["validation"]["accuracy"]) for row in history])
    maximum = float(np.max(scores))
    if not np.isclose(maximum, best_validation_accuracy, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Stored best validation accuracy {best_validation_accuracy!r} "
            f"does not match history maximum {maximum!r}"
        )
    return int(history[int(np.flatnonzero(np.isclose(scores, maximum, rtol=0.0, atol=1e-12))[0])]["epoch"])


def _source_location(source_kind: str, run_dir: Path, root: Path) -> str:
    relative = run_dir.resolve().relative_to(root.resolve()).as_posix()
    return f"{source_kind}:{relative}"


def discover_current_candidates(current_outputs_root: Path) -> tuple[tuple[str, str, str, str], ...]:
    """Return supported candidates only when their six-run matrix is complete."""
    root = Path(current_outputs_root)
    completed = []
    for spec in CURRENT_CANDIDATE_SPECS:
        method, suffix, _, _ = spec
        run_dirs = [root / f"{slug}-{suffix}-seed{SEED}" for slug, _, _ in ARCHITECTURES]
        present = [run_dir.exists() for run_dir in run_dirs]
        if not any(present):
            continue
        missing = [
            str(run_dir / relative)
            for run_dir in run_dirs
            for relative in REQUIRED_RUN_ARTIFACTS
            if not (run_dir / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete fixed {method} matrix; missing artifacts: {', '.join(missing)}"
            )
        completed.append(spec)
    if not completed:
        raise ValueError(f"No complete supported fixed candidate matrix found under {root}")
    return tuple(completed)


def discover_support_restricted_candidates(
    current_outputs_root: Path,
) -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
    """Discover only explicitly registered, architecture-restricted baselines."""
    root = Path(current_outputs_root)
    completed = []
    for spec in SUPPORT_RESTRICTED_SPECS:
        method, suffix, _, _, supported_slugs = spec
        run_dirs = [root / f"{slug}-{suffix}-seed{SEED}" for slug in supported_slugs]
        if not any(run_dir.exists() for run_dir in run_dirs):
            continue
        missing = [
            str(run_dir / relative)
            for run_dir in run_dirs
            for relative in REQUIRED_RUN_ARTIFACTS
            if not (run_dir / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete support-restricted {method} run; missing artifacts: {', '.join(missing)}"
            )
        completed.append(spec)
    return tuple(completed)


def comparison_pairs(candidate_methods: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Compare each candidate to core references and preceding candidates."""
    pairs = []
    preceding = []
    for candidate in candidate_methods:
        pairs.extend((reference, candidate) for reference in (*BASE_REFERENCES, *preceding))
        preceding.append(candidate)
    return tuple(pairs)


def support_restricted_pairs(
    complete_candidate_methods: Sequence[str],
    restricted_candidate_methods: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Build descriptive pairs that must never be sent to architecture W/T/L."""
    references = (*BASE_REFERENCES, *complete_candidate_methods)
    return tuple(
        (reference, candidate)
        for candidate in restricted_candidate_methods
        for reference in references
    )


def _metric_deltas(
    metrics: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    indexed = metrics.set_index(["architecture", "method"])
    for architecture in metrics["architecture"].drop_duplicates():
        for reference_name, candidate_name in pairs:
            candidate_key = (architecture, candidate_name)
            reference_key = (architecture, reference_name)
            if candidate_key not in indexed.index:
                continue
            if reference_key not in indexed.index:
                raise ValueError(f"Missing reference {reference_name} for {architecture}")
            candidate = indexed.loc[candidate_key]
            reference = indexed.loc[reference_key]
            for metric in WTL_METRICS:
                delta = float(candidate[metric]) - float(reference[metric])
                direction = "lower" if metric in LOWER_IS_BETTER else "higher"
                rows.append({
                    "architecture": architecture,
                    "seed": SEED,
                    "metric": metric,
                    "direction": direction,
                    "reference": reference_name,
                    "candidate": candidate_name,
                    "reference_value": float(reference[metric]),
                    "candidate_value": float(candidate[metric]),
                    "candidate_minus_reference": delta,
                    "improvement": -delta if direction == "lower" else delta,
                })
    return pd.DataFrame(rows)


def build_pareto_table(metrics: pd.DataFrame, efficiency: pd.DataFrame) -> pd.DataFrame:
    costs = efficiency[[
        "architecture", "method", "post_branch_wall_seconds", "peak_cuda_memory_gib",
    ]]
    table = metrics.merge(costs, on=["architecture", "method"], validate="one_to_one")
    fronts = []
    for _, group in table.groupby("architecture", sort=False):
        fronts.append(pareto_front(group, maximize=PARETO_MAXIMIZE, minimize=PARETO_MINIMIZE))
    table["pareto_optimal"] = pd.concat(fronts).sort_index()
    table["pareto_scope"] = "within_architecture"
    table["maximize_objectives"] = ";".join(PARETO_MAXIMIZE)
    table["minimize_objectives"] = ";".join(PARETO_MINIMIZE)
    return table


def _report(
    metrics: pd.DataFrame,
    wtl: pd.DataFrame,
    efficiency: pd.DataFrame,
    overhead: pd.DataFrame,
    pareto: pd.DataFrame,
    candidate_methods: Sequence[str],
) -> str:
    lines = [
        "# Fixed explanation-baseline CUB aggregation",
        "",
        "Seed 42 only; training-run uncertainty is unestimated. All rows use the held-out CUB test split",
        "after fixed-recipe training and validation-accuracy checkpoint selection. No test metric was used",
        "to choose a method setting or checkpoint.",
    ]
    for method in candidate_methods:
        candidate = metrics.loc[metrics["method"].eq(method)]
        lines.extend([
            "",
            f"## {method} per architecture",
            "",
            "| Architecture | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Strict leakage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in candidate.itertuples():
            values = [
                row.clean_accuracy, row.bg_mean_accuracy, row.bg_worst_accuracy,
                row.fg_mean_accuracy, row.fg_worst_accuracy, row.foreground_only_accuracy,
                row.background_only_accuracy,
            ]
            lines.append(
                f"| {row.architecture} | "
                + " | ".join(f"{100 * value:.2f}%" for value in values)
                + f" | {row.strict_leakage:.3f} |"
            )
    lines.extend([
        "",
        "BG/FG means and worst cases use the same 12 generic background and 12 generic foreground",
        "interventions as `results/cub/complete_metrics.csv`. Strict leakage is lower-is-better; the",
        "accuracy columns follow the existing table's higher-is-better win/tie/loss convention. BG-only",
        "accuracy remains a descriptive stress test and should not alone be read as causal reliance.",
        "",
        "## Architecture win/tie/loss",
        "",
        "Exact equality tolerance is 0. Counts are candidate versus reference after seed averaging",
        "within architecture (trivial here because there is one seed).",
        "Faster-ISNet is support-restricted to ResNet-50 and is deliberately excluded from these",
        "six-architecture counts; its ResNet-only comparisons are retained in the delta and cost tables.",
        "",
        "| Candidate | Reference | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Leakage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    metric_order = (
        "clean_accuracy", "bg_mean_accuracy", "bg_worst_accuracy", "fg_mean_accuracy",
        "fg_worst_accuracy", "foreground_only_accuracy", "background_only_accuracy",
        "strict_leakage",
    )
    for candidate, reference in wtl[["candidate", "reference"]].drop_duplicates().itertuples(index=False):
        subset = wtl.loc[
            wtl["candidate"].eq(candidate) & wtl["reference"].eq(reference)
        ].set_index("metric")
        counts = [
            f"{int(subset.loc[metric, 'wins'])}/{int(subset.loc[metric, 'ties'])}/{int(subset.loc[metric, 'losses'])}"
            for metric in metric_order
        ]
        lines.append(f"| {candidate} | {reference} | " + " | ".join(counts) + " |")

    indexed_overhead = overhead.set_index(["architecture", "reference", "candidate"])
    lines.extend(["", "## Timing and allocated VRAM"])
    for method in candidate_methods:
        lines.extend([
            "",
            f"### {method}",
            "",
            "| Architecture | Total wall | Post-branch wall | Peak VRAM | Reference | Total overhead | Post-branch overhead | VRAM overhead |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ])
        candidate_efficiency = efficiency.loc[efficiency["method"].eq(method)]
        references = overhead.loc[overhead["candidate"].eq(method), "reference"].drop_duplicates()
        for row in candidate_efficiency.itertuples():
            for reference in references:
                comparison = indexed_overhead.loc[(row.architecture, reference, method)]
                lines.append(
                    f"| {row.architecture} | {row.wall_seconds / 3600:.3f} h | "
                    f"{row.post_branch_wall_seconds / 3600:.3f} h | {row.peak_cuda_memory_gib:.3f} GiB | "
                    f"{reference} | {comparison.wall_seconds_overhead_percent:+.1f}% | "
                    f"{comparison.post_branch_wall_seconds_overhead_percent:+.1f}% | "
                    f"{comparison.peak_cuda_memory_gib_overhead_percent:+.1f}% |"
                )
    lines.extend([
        "",
        "Total wall is summed train plus validation time for all 50 epochs. Post-branch wall covers epochs",
        "6-50 only. Peak VRAM is PyTorch peak allocated CUDA memory, not device-resident memory. The first",
        "five history records are record-for-record identical after JSON parsing within every architecture/method",
        "pair. All runs identify an NVIDIA GeForce RTX 5070 Ti; timings are observed single-run values, not",
        "uncertainty estimates, and may reflect uncontrolled system load.",
        "The interrupted RRR EfficientNet-B0 epoch-46 attempt produced no history record and is excluded;",
        "its efficiency row aggregates only the subsequently completed, recorded 50-epoch history.",
        "Faster-ISNet likewise uses only the clean `resnet50-isnet-seed42` rerun. The interrupted corrupt",
        "checkpoint directory is excluded. The clean rerun diverged nondeterministically early from the",
        "interrupted attempt but converged late; only its recorded complete history is aggregated.",
        "",
        "## Pareto audit",
        "",
        "Pareto status is computed separately within each architecture. Maximized objectives are clean",
        "accuracy, BG mean accuracy, and foreground-only accuracy; minimized objectives are strict leakage,",
        "post-branch wall time, and peak allocated VRAM. The non-dominated methods are:",
        "",
    ])
    for architecture, group in pareto.groupby("architecture", sort=False):
        methods = ", ".join(group.loc[group["pareto_optimal"], "method"])
        lines.append(f"- {architecture}: {methods}")
    lines.extend([
        "",
        "## Provenance",
        "",
        "CE and FCAL V2 metrics are independently recomputed from archived evaluation summaries and must",
        "match `results/cub/complete_metrics.csv` to 1e-12. FCAL V2 is the archived `fcal-v5` run family",
        "used by that table. General fixed candidates require complete current six-architecture matrices",
        "under `outputs/baseline-fixed/cub`. Partial candidates are never admitted by that guard;",
        "Faster-ISNet enters through a separate explicit ResNet-only support registry.",
        "`selection_provenance.csv` retains run/config/checkpoint/history/evaluation hashes, validation-selected epoch,",
        "best validation score, source location, split hashes, and the explicit fixed-default/test-evaluated",
        "flags. `metric_deltas.csv`, `architecture_win_tie_loss.csv`, `efficiency_runs.csv`,",
        "`efficiency_overhead.csv`, and `pareto.csv` retain exact machine-readable values.",
        "",
    ])
    return "\n".join(lines)


def build(
    complete_metrics_path: Path,
    archive_outputs_root: Path,
    current_outputs_root: Path,
    report_root: Path,
) -> tuple[Path, ...]:
    """Validate inputs and write all complete fixed-candidate comparisons."""
    complete_metrics_path = Path(complete_metrics_path)
    archive_outputs_root = Path(archive_outputs_root)
    current_outputs_root = Path(current_outputs_root)
    report_root = Path(report_root)
    published = pd.read_csv(complete_metrics_path)
    published = published.loc[published["method"].isin(("CE", "FCAL V2"))]
    complete_candidate_specs = discover_current_candidates(current_outputs_root)
    restricted_candidate_specs = discover_support_restricted_candidates(current_outputs_root)
    complete_candidate_methods = tuple(spec[0] for spec in complete_candidate_specs)
    restricted_candidate_methods = tuple(spec[0] for spec in restricted_candidate_specs)
    candidate_methods = (*complete_candidate_methods, *restricted_candidate_methods)
    method_specs = (*REFERENCE_SPECS, *complete_candidate_specs)
    pairs = comparison_pairs(complete_candidate_methods)
    restricted_pairs = support_restricted_pairs(
        complete_candidate_methods, restricted_candidate_methods
    )
    delta_pairs = (*pairs, *restricted_pairs)

    metric_rows: list[dict] = []
    provenance_rows: list[dict] = []
    efficiency_rows: list[dict] = []
    histories: dict[tuple[str, str], list[dict]] = {}
    split_hash_fields: set[str] | None = None

    for slug, architecture, family in ARCHITECTURES:
        architecture_method_specs = list(method_specs)
        architecture_method_specs.extend(
            spec[:4] for spec in restricted_candidate_specs if slug in spec[4]
        )
        for method, suffix, source_kind, expected_config_method in architecture_method_specs:
            root = archive_outputs_root if source_kind == "archive" else current_outputs_root
            run_dir = root / f"{slug}-{suffix}-seed{SEED}"
            paths = {
                "config": run_dir / "config.yaml",
                "environment": run_dir / "environment.json",
                "history": run_dir / "history.jsonl",
                "training_summary": run_dir / "summary.json",
                "evaluation_summary": run_dir / "evaluation" / "summary.json",
                "checkpoint": run_dir / "best-validation.pt",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing artifacts for {architecture} {method}: {', '.join(missing)}")

            config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
            environment = _read_json(paths["environment"])
            training_summary = _read_json(paths["training_summary"])
            history = _read_history(paths["history"])
            validate_history(history, identity=f"{architecture} {method}")
            histories[(slug, method)] = history
            if int(config["seed"]) != SEED:
                raise ValueError(f"Unexpected seed for {architecture} {method}: {config['seed']}")
            if config["method"] != expected_config_method:
                raise ValueError(
                    f"Unexpected method key for {architecture} {method}: {config['method']}"
                )
            best_validation = float(training_summary["best_validation_accuracy"])
            chosen_epoch = selected_epoch(history, best_validation)
            observed_metrics = metrics_from_summary(paths["evaluation_summary"])

            if source_kind == "archive":
                indexed = published.loc[
                    published["architecture"].eq(architecture) & published["method"].eq(method)
                ]
                if len(indexed) != 1:
                    raise ValueError(f"Expected one published row for {architecture} {method}; found {len(indexed)}")
                validate_existing_metric_row(
                    indexed.iloc[0], observed_metrics, identity=f"{architecture} {method}"
                )

            metric_rows.append({
                "family": family,
                "architecture": architecture,
                "architecture_slug": slug,
                "seed": SEED,
                "method": method,
                **observed_metrics,
            })

            source_location = _source_location(source_kind, run_dir, root)
            history_accounting_note = HISTORY_ACCOUNTING_NOTES.get(
                (slug, method), "recorded_complete_history_only"
            )
            support_restricted = method in restricted_candidate_methods
            support_scope = f"{slug}_only" if support_restricted else "six_architecture"
            hashes = {key: _sha256(paths[key]) for key in (
                "config", "environment", "history", "evaluation_summary", "checkpoint"
            )}
            current_split_fields = {key for key in environment if key.startswith("split_") and key.endswith("_sha256")}
            split_hash_fields = current_split_fields if split_hash_fields is None else split_hash_fields
            if current_split_fields != split_hash_fields:
                raise ValueError(f"Inconsistent split-hash fields for {architecture} {method}")
            provenance_rows.append({
                "family": family,
                "architecture": architecture,
                "architecture_slug": slug,
                "seed": SEED,
                "method": method,
                "comparison_regime": "fixed_default",
                "trial_id": environment["config_hash"],
                "selection_split": "fixed",
                "selected": True,
                "test_evaluated": True,
                "checkpoint_selection_split": "validation",
                "checkpoint_selection_metric": "accuracy",
                "selected_epoch": chosen_epoch,
                "best_validation_accuracy": best_validation,
                "run_id": config["run_id"],
                "training_method_key": config["method"],
                "source_location": source_location,
                "selected_checkpoint": f"{source_location}/best-validation.pt",
                "evaluation_binding": "colocated_with_selected_checkpoint",
                "history_accounting_note": history_accounting_note,
                "support_scope": support_scope,
                "architecture_wtl_eligible": not support_restricted,
                "excluded_artifact_location": (
                    "current:resnet50-isnet-seed42-interrupted-corrupt-checkpoints-20260828"
                    if method == "Faster-ISNet" else ""
                ),
                "cuda_device": environment.get("cuda_device"),
                "config_sha256": hashes["config"],
                "environment_sha256": hashes["environment"],
                "history_sha256": hashes["history"],
                "evaluation_summary_sha256": hashes["evaluation_summary"],
                "selected_checkpoint_sha256": hashes["checkpoint"],
                **{key: environment[key] for key in sorted(current_split_fields)},
            })

            efficiency = run_efficiency(run_dir, branch_epochs=COMMON_BRANCH_EPOCHS)
            efficiency.update({
                "family": family,
                "architecture": architecture,
                "architecture_slug": slug,
                "seed": SEED,
                "method": method,
                "source_location": source_location,
                "history_accounting_note": history_accounting_note,
                "support_scope": support_scope,
                "architecture_wtl_eligible": not support_restricted,
            })
            efficiency_rows.append(efficiency)

        ce_history = histories[(slug, "CE")]
        for method, _, _, _ in architecture_method_specs[1:]:
            validate_common_prefix(
                ce_history,
                histories[(slug, method)],
                candidate_name=f"{architecture} {method}",
            )

    provenance = pd.DataFrame(provenance_rows)
    split_columns = sorted(split_hash_fields or ())
    if split_columns and any(provenance[column].nunique(dropna=False) != 1 for column in split_columns):
        raise ValueError("Run matrix does not use one common set of prepared CUB split hashes")
    if provenance["cuda_device"].nunique(dropna=False) != 1:
        raise ValueError("Run matrix was not measured on one CUDA device model")

    metrics = pd.DataFrame(metric_rows)
    deltas = _metric_deltas(metrics, delta_pairs)
    wtl_parts = []
    for reference, candidate in pairs:
        part = win_tie_loss(
            metrics,
            reference=reference,
            candidate=candidate,
            metrics=WTL_METRICS,
            lower_is_better=LOWER_IS_BETTER,
            tolerance=0.0,
        )
        part.insert(1, "direction", part["metric"].map(
            lambda metric: "lower" if metric in LOWER_IS_BETTER else "higher"
        ))
        wtl_parts.append(part)
    wtl = pd.concat(wtl_parts, ignore_index=True)

    efficiency = pd.DataFrame(efficiency_rows)
    overhead_parts = []
    for reference, candidate in (*pairs, *restricted_pairs):
        candidate_architectures = efficiency.loc[
            efficiency["method"].eq(candidate), "architecture"
        ]
        paired_efficiency = efficiency.loc[
            efficiency["architecture"].isin(candidate_architectures)
            & efficiency["method"].isin((reference, candidate))
        ]
        part = efficiency_overhead(
            paired_efficiency, reference=reference, candidates=[candidate]
        )
        part.insert(0, "family", part["architecture"].map(
            metrics.drop_duplicates("architecture").set_index("architecture")["family"]
        ))
        overhead_parts.append(part)
    overhead = pd.concat(overhead_parts, ignore_index=True)
    pareto = build_pareto_table(metrics, efficiency)

    report_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        report_root / "comparison_metrics.csv",
        report_root / "metric_deltas.csv",
        report_root / "architecture_win_tie_loss.csv",
        report_root / "efficiency_runs.csv",
        report_root / "efficiency_overhead.csv",
        report_root / "selection_provenance.csv",
        report_root / "REPORT.md",
        report_root / "pareto.csv",
    )
    metrics.to_csv(outputs[0], index=False)
    deltas.to_csv(outputs[1], index=False)
    wtl.to_csv(outputs[2], index=False)
    efficiency.to_csv(outputs[3], index=False)
    overhead.to_csv(outputs[4], index=False)
    provenance.to_csv(outputs[5], index=False)
    outputs[6].write_text(
        _report(metrics, wtl, efficiency, overhead, pareto, candidate_methods),
        encoding="utf-8",
    )
    pareto.to_csv(outputs[7], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("complete_metrics", type=Path)
    parser.add_argument("archive_outputs", type=Path)
    parser.add_argument("current_outputs", type=Path)
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    print(*(str(path) for path in build(
        args.complete_metrics,
        args.archive_outputs,
        args.current_outputs,
        args.report_root,
    )), sep="\n")


if __name__ == "__main__":
    main()
