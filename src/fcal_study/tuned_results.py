"""Aggregate frozen, validation-selected tuning winners after sealed test evaluation.

This module is deliberately post-freeze only.  It never trains, freezes, or
evaluates a model; it consumes a frozen-winner manifest, its append-only tuning
ledger, already-written evaluation summaries, and the fixed-arm metric table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml

from .baseline_fixed_results import (
    LOWER_IS_BETTER,
    METRIC_COLUMNS,
    SEED,
    WTL_METRICS,
    metrics_from_summary,
)
from .complete_results import ARCHITECTURES
from .evidence import win_tie_loss
from .tuning import all_jobs, frozen_winners, load_manifest

GENERAL_METHODS = ("fcal_v2", "rrr", "gain_ext", "camal")
RESTRICTED_METHOD = "selective_faster_isnet"
FIXED_DISPLAY_NAMES = {
    "fcal_v2": "FCAL V2",
    "rrr": "RRR",
    "gain_ext": "GAIN-ext",
    "camal": "CAMAL",
    "selective_faster_isnet": "Faster-ISNet",
}
ARCHITECTURE_METADATA = {
    slug: {"architecture": architecture, "family": family}
    for slug, architecture, family in ARCHITECTURES
}


def _canonical_winners(rows: Iterable[dict]) -> list[dict]:
    fields = (
        "architecture", "method", "trial_id", "coefficient", "value",
        "validation_accuracy", "checkpoint",
    )
    return sorted(
        ({field: row[field] for field in fields} for row in rows),
        key=lambda row: (row["architecture"], row["method"]),
    )


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid ledger JSON at line {line_number}: {path}") from error
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_fields(job_id: str) -> tuple[str, str, str, str]:
    parts = job_id.split("/")
    if len(parts) != 4:
        raise ValueError(f"Invalid tuning job id: {job_id}")
    return parts[0], parts[1], parts[2], parts[3]


def _load_frozen(path: Path, manifest: dict) -> list[dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != manifest["study_id"]:
        raise ValueError("Frozen winners and tuning manifest have different study IDs")
    if payload.get("selection_split") != "validation" or payload.get("test_set_used") is not False:
        raise ValueError("Winner selection must be frozen from validation without test-set use")
    winners = payload.get("winners")
    if not isinstance(winners, list):
        raise TypeError("Frozen-winner manifest must contain a winners list")
    expected = {
        (architecture, method)
        for architecture in manifest["architectures"]
        for method, method_spec in manifest["methods"].items()
        if architecture in method_spec["supported_architectures"]
    }
    observed = {(row.get("architecture"), row.get("method")) for row in winners}
    if len(observed) != len(winners) or observed != expected:
        raise ValueError("Frozen winners do not exactly cover the declared support matrix")
    return winners


def _successful_final_records(records: Iterable[dict], manifest: dict) -> dict[tuple[str, str], dict]:
    final_epochs = int(manifest["successive_halving"]["rungs"][-1]["total_epochs"])
    final_rung = f"rung-{len(manifest['successive_halving']['rungs']):02d}"
    result = {}
    for record in records:
        architecture, method, trial_id, rung_id = _job_fields(str(record["job_id"]))
        if record.get("study_id") != manifest["study_id"]:
            raise ValueError("Ledger contains a record from another tuning study")
        if record.get("gpu_hour_accounting") != "allocated_wall_clock":
            raise ValueError("Every tuning attempt must use allocated-wall-clock GPU accounting")
        if rung_id != final_rung or record.get("status") != "success":
            continue
        config_path = Path(record["config_path"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if int(config["training"]["epochs"]) != final_epochs:
            raise ValueError(f"Final-rung epoch mismatch: {record['job_id']}")
        key = (architecture, method)
        candidate = dict(record, trial_id=trial_id)
        if key in result:
            raise ValueError(f"Multiple successful final-rung records for {key}")
        result[key] = candidate
    return result


def _validate_ledger(records: list[dict], manifest: dict) -> None:
    valid_job_ids = {job.job_id for job in all_jobs(manifest)}
    grouped: dict[str, list[dict]] = {}
    for record in records:
        job_id = str(record.get("job_id"))
        if job_id not in valid_job_ids:
            raise ValueError(f"Ledger contains an undeclared tuning job: {job_id}")
        if record.get("status") not in {"success", "failure"}:
            raise ValueError(f"Invalid tuning attempt status: {job_id}")
        grouped.setdefault(job_id, []).append(record)
    maximum = int(manifest["successive_halving"]["max_attempts_per_job"])
    for job_id, attempts in grouped.items():
        if len(attempts) > maximum:
            raise ValueError(f"Retry limit exceeded in tuning ledger: {job_id}")
        numbers = [int(record["attempt"]) for record in attempts]
        if numbers != list(range(1, len(attempts) + 1)):
            raise ValueError(f"Non-sequential attempt numbering in tuning ledger: {job_id}")
        successes = [index for index, record in enumerate(attempts) if record["status"] == "success"]
        if len(successes) > 1 or (successes and successes[0] != len(attempts) - 1):
            raise ValueError(f"Tuning ledger retried a successful job: {job_id}")


def tuning_attempt_table(records: Iterable[dict], winners: Iterable[dict]) -> pd.DataFrame:
    """Return every launched attempt, retaining failures in GPU-hour totals."""
    selected_jobs = {
        (row["architecture"], row["method"], row["trial_id"])
        for row in winners
    }
    rows = []
    for record in records:
        architecture, method, trial_id, rung_id = _job_fields(str(record["job_id"]))
        if method not in FIXED_DISPLAY_NAMES:
            raise ValueError(f"Unknown method in tuning ledger: {method}")
        gpu_hours = float(record.get("gpu_hours", 0.0))
        elapsed_seconds = float(record.get("elapsed_seconds", 0.0))
        if gpu_hours < 0 or elapsed_seconds < 0:
            raise ValueError(f"Negative tuning cost in ledger: {record['job_id']}")
        rows.append({
            "study_id": record.get("study_id"),
            "architecture_slug": architecture,
            "method_key": method,
            "method": FIXED_DISPLAY_NAMES[method],
            "trial_id": trial_id,
            "rung_id": rung_id,
            "attempt": int(record["attempt"]),
            "status": record["status"],
            "selected_trial": (architecture, method, trial_id) in selected_jobs,
            "validation_accuracy": record.get("validation_accuracy"),
            "elapsed_seconds": elapsed_seconds,
            "gpu_hours": gpu_hours,
            "gpu_hour_accounting": record.get("gpu_hour_accounting"),
            "exit_code": record.get("exit_code"),
            "error": record.get("error"),
            "job_id": record["job_id"],
            "run_dir": record.get("run_dir"),
            "config_path": record.get("config_path"),
        })
    return pd.DataFrame(rows)


def tuning_accounting(attempts: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized search cost by cell, method, and entire study."""
    columns = [
        "scope", "architecture_slug", "method_key", "method", "attempts",
        "successful_attempts", "failed_attempts", "elapsed_seconds_including_failures",
        "gpu_hours_including_failures", "failed_gpu_hours",
    ]
    rows = []

    def append(scope: str, frame: pd.DataFrame, architecture: str = "", method_key: str = "") -> None:
        failure = ~frame["status"].eq("success")
        rows.append({
            "scope": scope,
            "architecture_slug": architecture,
            "method_key": method_key,
            "method": FIXED_DISPLAY_NAMES.get(method_key, "ALL"),
            "attempts": len(frame),
            "successful_attempts": int((~failure).sum()),
            "failed_attempts": int(failure.sum()),
            "elapsed_seconds_including_failures": float(frame["elapsed_seconds"].sum()),
            "gpu_hours_including_failures": float(frame["gpu_hours"].sum()),
            "failed_gpu_hours": float(frame.loc[failure, "gpu_hours"].sum()),
        })

    for (architecture, method_key), group in attempts.groupby(
        ["architecture_slug", "method_key"], sort=True
    ):
        append("cell", group, architecture, method_key)
    for method_key, group in attempts.groupby("method_key", sort=True):
        append("method", group, method_key=method_key)
    append("study", attempts)
    return pd.DataFrame(rows, columns=columns)


def _fixed_tuned_deltas(fixed: pd.DataFrame, tuned: pd.DataFrame) -> pd.DataFrame:
    fixed_index = fixed.set_index(["architecture_slug", "method"])
    rows = []
    for row in tuned.itertuples(index=False):
        key = (row.architecture_slug, row.method)
        if key not in fixed_index.index:
            raise ValueError(f"Missing fixed counterpart for tuned winner: {key}")
        fixed_row = fixed_index.loc[key]
        for metric in WTL_METRICS:
            delta = float(getattr(row, metric)) - float(fixed_row[metric])
            direction = "lower" if metric in LOWER_IS_BETTER else "higher"
            rows.append({
                "family": row.family,
                "architecture": row.architecture,
                "architecture_slug": row.architecture_slug,
                "seed": SEED,
                "method": row.method,
                "support_scope": row.support_scope,
                "architecture_wtl_eligible": row.architecture_wtl_eligible,
                "metric": metric,
                "direction": direction,
                "fixed_value": float(fixed_row[metric]),
                "tuned_value": float(getattr(row, metric)),
                "tuned_minus_fixed": delta,
                "improvement": -delta if direction == "lower" else delta,
            })
    return pd.DataFrame(rows)


def _wtl_tables(fixed: pd.DataFrame, tuned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = tuned.loc[tuned["architecture_wtl_eligible"]].copy()
    fixed_eligible = fixed.merge(
        eligible[["architecture_slug", "method"]],
        on=["architecture_slug", "method"],
        how="inner",
        validate="one_to_one",
    )
    fixed_eligible["arm"] = "fixed"
    eligible["arm"] = "tuned"
    paired = pd.concat([fixed_eligible, eligible], ignore_index=True)
    fixed_tuned = []
    for method in (FIXED_DISPLAY_NAMES[key] for key in GENERAL_METHODS):
        method_frame = paired.loc[paired["method"].eq(method)]
        part = win_tie_loss(
            method_frame,
            reference="fixed",
            candidate="tuned",
            metrics=WTL_METRICS,
            method_col="arm",
            lower_is_better=LOWER_IS_BETTER,
            tolerance=0.0,
        )
        part.insert(0, "method", method)
        fixed_tuned.append(part)

    tuned_pairs = []
    reference = FIXED_DISPLAY_NAMES["fcal_v2"]
    for method_key in GENERAL_METHODS[1:]:
        candidate = FIXED_DISPLAY_NAMES[method_key]
        part = win_tie_loss(
            eligible,
            reference=reference,
            candidate=candidate,
            metrics=WTL_METRICS,
            lower_is_better=LOWER_IS_BETTER,
            tolerance=0.0,
        )
        tuned_pairs.append(part)
    return pd.concat(fixed_tuned, ignore_index=True), pd.concat(tuned_pairs, ignore_index=True)


def build(
    frozen_path: Path,
    ledger_path: Path,
    manifest_path: Path,
    fixed_metrics_path: Path,
    report_root: Path,
) -> tuple[Path, ...]:
    """Validate and aggregate already-evaluated frozen winners."""
    manifest = load_manifest(manifest_path)
    winners = _load_frozen(frozen_path, manifest)
    records = _read_jsonl(ledger_path)
    _validate_ledger(records, manifest)
    reproduced = frozen_winners(manifest, records)["winners"]
    if _canonical_winners(winners) != _canonical_winners(reproduced):
        raise ValueError("Frozen winners are not reproducible from the append-only tuning ledger")
    final_records = _successful_final_records(records, manifest)

    winner_rows = []
    metric_rows = []
    for winner in sorted(winners, key=lambda row: (row["architecture"], row["method"])):
        architecture_slug = winner["architecture"]
        method_key = winner["method"]
        key = (architecture_slug, method_key)
        final = final_records.get(key)
        if final is None or final["trial_id"] != winner["trial_id"]:
            raise ValueError(f"Frozen winner is not the successful final-rung trial: {key}")
        checkpoint = Path(winner["checkpoint"]).resolve()
        expected_checkpoint = (Path(final["run_dir"]) / "best-validation.pt").resolve()
        if checkpoint != expected_checkpoint or not checkpoint.is_file():
            raise FileNotFoundError(f"Frozen checkpoint mismatch or missing: {checkpoint}")
        evaluation_summary = checkpoint.parent / "evaluation" / "summary.json"
        if not evaluation_summary.is_file():
            raise FileNotFoundError(
                f"Frozen winner has not been test-evaluated: {evaluation_summary}"
            )
        metadata = ARCHITECTURE_METADATA[architecture_slug]
        supported = set(manifest["methods"][method_key]["supported_architectures"])
        restricted = supported != set(manifest["architectures"])
        support_scope = f"{architecture_slug}_only" if restricted else "six_architecture"
        common = {
            "family": metadata["family"],
            "architecture": metadata["architecture"],
            "architecture_slug": architecture_slug,
            "seed": SEED,
            "method_key": method_key,
            "method": FIXED_DISPLAY_NAMES[method_key],
            "support_scope": support_scope,
            "architecture_wtl_eligible": not restricted,
        }
        winner_rows.append({
            **common,
            "comparison_regime": "equal_tuned",
            "selection_split": "validation",
            "selection_metric": manifest["selection"]["metric"],
            "trial_id": winner["trial_id"],
            "coefficient": winner["coefficient"],
            "value": float(winner["value"]),
            "validation_accuracy": float(winner["validation_accuracy"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "evaluation_summary": str(evaluation_summary.resolve()),
            "evaluation_summary_sha256": _sha256(evaluation_summary),
            "test_evaluated": True,
        })
        metric_rows.append({**common, **metrics_from_summary(evaluation_summary)})

    selected = pd.DataFrame(winner_rows)
    tuned = pd.DataFrame(metric_rows)
    fixed = pd.read_csv(fixed_metrics_path)
    required_fixed = {"architecture_slug", "method", *METRIC_COLUMNS}
    missing = sorted(required_fixed.difference(fixed.columns))
    if missing:
        raise ValueError(f"Fixed metric table is missing columns: {', '.join(missing)}")
    fixed = fixed.loc[
        fixed["method"].isin(FIXED_DISPLAY_NAMES.values()),
        [column for column in fixed.columns if column in {
            "family", "architecture", "architecture_slug", "seed", "method", *METRIC_COLUMNS,
        }],
    ].copy()
    if fixed.duplicated(["architecture_slug", "method"]).any():
        raise ValueError("Fixed metric table has duplicate architecture/method rows")

    attempts = tuning_attempt_table(records, winners)
    accounting = tuning_accounting(attempts)
    deltas = _fixed_tuned_deltas(fixed, tuned)
    fixed_tuned_wtl, tuned_method_wtl = _wtl_tables(fixed, tuned)

    report_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        report_root / "validation_selected_hyperparameters.csv",
        report_root / "tuned_winner_metrics.csv",
        report_root / "tuning_attempts.csv",
        report_root / "tuning_gpu_hour_accounting.csv",
        report_root / "fixed_vs_tuned_metric_deltas.csv",
        report_root / "fixed_vs_tuned_architecture_win_tie_loss.csv",
        report_root / "tuned_architecture_win_tie_loss.csv",
    )
    for frame, output in zip(
        (selected, tuned, attempts, accounting, deltas, fixed_tuned_wtl, tuned_method_wtl),
        outputs,
    ):
        frame.to_csv(output, index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen_winners", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("fixed_metrics", type=Path)
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    print(*(str(path) for path in build(
        args.frozen_winners,
        args.ledger,
        args.manifest,
        args.fixed_metrics,
        args.report_root,
    )), sep="\n")


if __name__ == "__main__":
    main()
