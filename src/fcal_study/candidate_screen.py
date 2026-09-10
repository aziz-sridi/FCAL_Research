from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import PROJECT_ROOT
from .utils import atomic_json
from .variants import DEFAULT_VARIANTS, parse_variant

LOWER_IS_BETTER = frozenset({"strict_leakage"})
QUALITY_METRICS = (
    "clean_accuracy",
    "background_mean_accuracy",
    "background_worst_accuracy",
    "foreground_only_accuracy",
    "strict_leakage",
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _evaluation_metrics(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "validation_evaluation" / "summary.json"
    items = json.loads(summary_path.read_text(encoding="utf-8"))
    by_variant = {item["variant"]: item for item in items}
    expected = {parse_variant(value).key for value in DEFAULT_VARIANTS}
    missing = sorted(expected - set(by_variant))
    if missing:
        raise ValueError(
            f"incomplete validation suite for {run_dir}: missing {missing}"
        )
    background = [
        by_variant[key]["accuracy"]
        for key in sorted(expected)
        if key.startswith("bg_")
    ]
    foreground = [
        by_variant[key]["accuracy"]
        for key in sorted(expected)
        if key.startswith("fg_")
    ]
    clean = by_variant["clean"]
    return {
        "clean_accuracy": float(clean["accuracy"]),
        "background_mean_accuracy": sum(background) / len(background),
        "background_worst_accuracy": min(background),
        "foreground_mean_accuracy": sum(foreground) / len(foreground),
        "foreground_worst_accuracy": min(foreground),
        "foreground_only_accuracy": float(
            by_variant["foreground_only"]["accuracy"]
        ),
        "background_only_accuracy": float(
            by_variant["background_only"]["accuracy"]
        ),
        "strict_leakage": float(clean["strict_leakage"]),
    }


def _training_metrics(run_dir: Path, branch_epoch: int) -> dict[str, float | int]:
    records = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if int(record["epoch"]) > branch_epoch]
    if not records:
        raise ValueError(f"no post-branch training records for {run_dir}")
    return {
        "post_branch_seconds": sum(
            float(record["train"]["seconds"])
            + float(record["validation"]["seconds"])
            for record in records
        ),
        "peak_cuda_memory_bytes": max(
            int(record["train"]["peak_cuda_memory_bytes"])
            for record in records
        ),
    }


def _triage_metrics(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "validation_evaluation" / "summary.json"
    items = json.loads(summary_path.read_text(encoding="utf-8"))
    by_variant = {item["variant"]: item for item in items}
    required = {"clean", "bg_random_rgb_s1", "foreground_only"}
    missing = sorted(required - set(by_variant))
    if missing:
        raise ValueError(f"incomplete triage for {run_dir}: missing {missing}")
    return {
        "clean_accuracy": float(by_variant["clean"]["accuracy"]),
        "background_probe_accuracy": float(
            by_variant["bg_random_rgb_s1"]["accuracy"]
        ),
        "foreground_only_accuracy": float(
            by_variant["foreground_only"]["accuracy"]
        ),
        "strict_leakage": float(by_variant["clean"]["strict_leakage"]),
    }


def _run_metrics(run_dir: Path, branch_epoch: int) -> dict[str, float | int]:
    return {
        **_evaluation_metrics(run_dir),
        **_training_metrics(run_dir, branch_epoch),
    }


def _passes(value: float, rule: dict) -> bool:
    if "minimum" in rule:
        return value >= float(rule["minimum"])
    if "maximum" in rule:
        return value <= float(rule["maximum"])
    raise ValueError(f"gate must define minimum or maximum: {rule}")


def _dominates(
    challenger: dict[str, float | int],
    candidate: dict[str, float | int],
    metrics: list[str],
) -> bool:
    no_worse = []
    better = []
    for metric in metrics:
        challenger_value = float(challenger[metric])
        candidate_value = float(candidate[metric])
        if metric in LOWER_IS_BETTER:
            no_worse.append(challenger_value <= candidate_value)
            better.append(challenger_value < candidate_value)
        else:
            no_worse.append(challenger_value >= candidate_value)
            better.append(challenger_value > candidate_value)
    return all(no_worse) and any(better)


def _strict_win_count(
    candidate: dict[str, float | int],
    reference: dict[str, float | int],
    metrics: list[str],
) -> int:
    wins = 0
    for metric in metrics:
        candidate_value = float(candidate[metric])
        reference_value = float(reference[metric])
        wins += (
            candidate_value < reference_value
            if metric in LOWER_IS_BETTER
            else candidate_value > reference_value
        )
    return wins


def _narrow_adjustment_allowed(
    failed: list[str],
    metrics: dict[str, float | int],
    gates: dict,
    policy: dict,
) -> bool:
    if len(failed) != 1:
        return False
    metric = failed[0]
    allowance = policy.get("maximum_narrow_miss", {}).get(metric)
    if allowance is None:
        return False
    value = float(metrics[metric])
    rule = gates[metric]
    miss = (
        float(rule["minimum"]) - value
        if "minimum" in rule
        else value - float(rule["maximum"])
    )
    return 0.0 < miss <= float(allowance)


def run(protocol_path: str | Path, output_path: str | Path | None = None) -> dict:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    branch_epoch = int(protocol["branch_epoch"])
    candidate_dir = _resolve(protocol["candidate"]["run_dir"])
    anchor_dir = _resolve(protocol["anchor"]["run_dir"])
    reference_dirs = {
        name: _resolve(path) for name, path in protocol["references"].items()
    }

    missing = []
    for name, run_dir in {
        protocol["candidate"]["name"]: candidate_dir,
        protocol["anchor"]["name"]: anchor_dir,
        **reference_dirs,
    }.items():
        for relative in (
            "config.yaml",
            "history.jsonl",
            "validation_evaluation/summary.json",
        ):
            if not (run_dir / relative).is_file():
                missing.append(f"{name}:{relative}")
    if missing:
        report = {
            "protocol": protocol["protocol"],
            "decision": "incomplete",
            "missing_artifacts": missing,
        }
        if output_path is not None:
            atomic_json(_resolve(output_path), report)
        return report

    candidate = _run_metrics(candidate_dir, branch_epoch)
    anchor = _run_metrics(anchor_dir, branch_epoch)
    references = {
        name: _run_metrics(run_dir, branch_epoch)
        for name, run_dir in reference_dirs.items()
    }
    hard_gates = {
        metric: {
            "value": candidate[metric],
            **rule,
            "passed": _passes(float(candidate[metric]), rule),
        }
        for metric, rule in protocol["hard_gates"].items()
    }
    failed = [metric for metric, result in hard_gates.items() if not result["passed"]]
    material = {
        metric: _passes(float(candidate[metric]), rule)
        for metric, rule in protocol["material_improvements"]["thresholds"].items()
    }
    material_count = sum(material.values())
    required_material = list(
        protocol["material_improvements"].get("required_any", [])
    )
    required_material_passed = not required_material or any(
        material.get(metric, False) for metric in required_material
    )

    competition_metrics = list(protocol["competition"]["metrics"])
    comparison_runs = {protocol["anchor"]["name"]: anchor, **references}
    dominated_by = [
        name
        for name, metrics in comparison_runs.items()
        if _dominates(metrics, candidate, competition_metrics)
    ]
    strict_wins = {
        name: _strict_win_count(candidate, metrics, competition_metrics)
        for name, metrics in comparison_runs.items()
    }
    minimum_strict_wins = protocol["competition"].get(
        "minimum_strict_wins", {}
    )
    failed_strict_wins = [
        name
        for name, minimum in minimum_strict_wins.items()
        if strict_wins.get(name, -1) < int(minimum)
    ]
    candidate_config = yaml.safe_load(
        (candidate_dir / "config.yaml").read_text(encoding="utf-8")
    )
    anchor_config = yaml.safe_load(
        (anchor_dir / "config.yaml").read_text(encoding="utf-8")
    )
    policy_checks = {
        "expected_method": candidate_config["method"]
        == protocol["candidate"]["expected_method"],
        "unchanged_inference_model": candidate_config["model"]
        == anchor_config["model"],
        "one_untouched_input": True,
        "one_backbone_forward": True,
    }

    enough_material = (
        material_count >= int(protocol["material_improvements"]["minimum_count"])
        and required_material_passed
    )
    competition_passed = not dominated_by and not failed_strict_wins
    hard_pass = not failed and all(policy_checks.values())
    if hard_pass and enough_material and competition_passed:
        decision = "advance"
    elif (
        enough_material
        and competition_passed
        and all(policy_checks.values())
        and _narrow_adjustment_allowed(
            failed,
            candidate,
            protocol["hard_gates"],
            protocol["adjustment_policy"],
        )
    ):
        decision = "adjust_once"
    else:
        decision = "stop"

    report = {
        "protocol": protocol["protocol"],
        "decision": decision,
        "candidate": candidate,
        "anchor": anchor,
        "references": references,
        "hard_gates": hard_gates,
        "failed_hard_gates": failed,
        "material_improvements": material,
        "material_improvement_count": material_count,
        "required_material_improvement_passed": required_material_passed,
        "dominated_by": dominated_by,
        "strict_wins": strict_wins,
        "failed_strict_wins": failed_strict_wins,
        "policy_checks": policy_checks,
    }
    if output_path is not None:
        atomic_json(_resolve(output_path), report)
    return report


def run_triage(
    protocol_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    branch_epoch = int(protocol["branch_epoch"])
    candidate_dir = _resolve(protocol["candidate"]["run_dir"])
    anchor_dir = _resolve(protocol["anchor"]["run_dir"])
    reference_dirs = {
        name: _resolve(path) for name, path in protocol["references"].items()
    }
    run_dirs = {
        protocol["candidate"]["name"]: candidate_dir,
        protocol["anchor"]["name"]: anchor_dir,
        **reference_dirs,
    }
    missing = [
        f"{name}:{relative}"
        for name, run_dir in run_dirs.items()
        for relative in (
            "config.yaml",
            "history.jsonl",
            "validation_evaluation/summary.json",
        )
        if not (run_dir / relative).is_file()
    ]
    if missing:
        report = {
            "protocol": protocol["protocol"],
            "stage": "triage",
            "decision": "incomplete",
            "missing_artifacts": missing,
        }
        if output_path is not None:
            atomic_json(_resolve(output_path), report)
        return report

    candidate = {
        **_triage_metrics(candidate_dir),
        **_training_metrics(candidate_dir, branch_epoch),
    }
    anchor = {
        **_triage_metrics(anchor_dir),
        **_training_metrics(anchor_dir, branch_epoch),
    }
    references = {
        name: {
            **_triage_metrics(run_dir),
            **_training_metrics(run_dir, branch_epoch),
        }
        for name, run_dir in reference_dirs.items()
    }
    triage = protocol["triage"]
    hard_gates = {
        metric: {
            "value": candidate[metric],
            **rule,
            "passed": _passes(float(candidate[metric]), rule),
        }
        for metric, rule in triage["hard_gates"].items()
    }
    failed = [metric for metric, result in hard_gates.items() if not result["passed"]]
    material = {
        metric: _passes(float(candidate[metric]), rule)
        for metric, rule in triage["material_improvements"]["thresholds"].items()
    }
    material_count = sum(material.values())
    required_material = list(
        triage["material_improvements"].get("required_any", [])
    )
    required_material_passed = not required_material or any(
        material.get(metric, False) for metric in required_material
    )
    comparison_runs = {protocol["anchor"]["name"]: anchor, **references}
    dominated_by = [
        name
        for name, metrics in comparison_runs.items()
        if _dominates(metrics, candidate, triage["competition_metrics"])
    ]
    strict_wins = {
        name: _strict_win_count(
            candidate, metrics, triage["competition_metrics"]
        )
        for name, metrics in comparison_runs.items()
    }
    minimum_strict_wins = triage.get("minimum_strict_wins", {})
    failed_strict_wins = [
        name
        for name, minimum in minimum_strict_wins.items()
        if strict_wins.get(name, -1) < int(minimum)
    ]
    enough_material = (
        material_count >= int(triage["material_improvements"]["minimum_count"])
        and required_material_passed
    )
    competition_passed = not dominated_by and not failed_strict_wins
    report = {
        "protocol": protocol["protocol"],
        "stage": "triage",
        "decision": (
            "advance_to_full_validation"
            if not failed and enough_material and competition_passed
            else "stop"
        ),
        "candidate": candidate,
        "anchor": anchor,
        "references": references,
        "hard_gates": hard_gates,
        "failed_hard_gates": failed,
        "material_improvements": material,
        "material_improvement_count": material_count,
        "required_material_improvement_passed": required_material_passed,
        "dominated_by": dominated_by,
        "strict_wins": strict_wins,
        "failed_strict_wins": failed_strict_wins,
    }
    if output_path is not None:
        atomic_json(_resolve(output_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol")
    parser.add_argument("--output")
    parser.add_argument("--triage", action="store_true")
    args = parser.parse_args()
    screen = run_triage if args.triage else run
    print(json.dumps(screen(args.protocol, args.output), indent=2))


if __name__ == "__main__":
    main()
