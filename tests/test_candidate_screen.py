import json
from pathlib import Path

import yaml

from fcal_study.candidate_screen import run
from fcal_study.variants import DEFAULT_VARIANTS, parse_variant


def _write_run(
    root: Path,
    *,
    method: str,
    clean: float,
    background: float,
    foreground_only: float,
    leakage: float,
) -> None:
    evaluation = root / "validation_evaluation"
    evaluation.mkdir(parents=True)
    rows = []
    for value in DEFAULT_VARIANTS:
        key = parse_variant(value).key
        accuracy = background if key.startswith("bg_") else clean
        if key == "foreground_only":
            accuracy = foreground_only
        row = {"variant": key, "accuracy": accuracy}
        if key == "clean":
            row["strict_leakage"] = leakage
        rows.append(row)
    (evaluation / "summary.json").write_text(json.dumps(rows), encoding="utf-8")
    (root / "history.jsonl").write_text(
        json.dumps(
            {
                "epoch": 6,
                "train": {"seconds": 5.0, "peak_cuda_memory_bytes": 100},
                "validation": {"seconds": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump({"method": method, "model": {"name": "resnet50"}}),
        encoding="utf-8",
    )


def test_screen_advances_only_after_anchor_and_competition_checks(tmp_path: Path):
    candidate = tmp_path / "candidate"
    anchor = tmp_path / "anchor"
    competitor = tmp_path / "competitor"
    _write_run(
        candidate,
        method="fcal_v2_relative",
        clean=0.81,
        background=0.75,
        foreground_only=0.74,
        leakage=0.20,
    )
    _write_run(
        anchor,
        method="fcal_v2",
        clean=0.80,
        background=0.72,
        foreground_only=0.71,
        leakage=0.27,
    )
    _write_run(
        competitor,
        method="camal",
        clean=0.79,
        background=0.76,
        foreground_only=0.73,
        leakage=0.25,
    )
    protocol = {
        "protocol": "test",
        "branch_epoch": 5,
        "candidate": {
            "name": "candidate",
            "run_dir": str(candidate),
            "expected_method": "fcal_v2_relative",
        },
        "anchor": {"name": "anchor", "run_dir": str(anchor)},
        "references": {"CAMAL": str(competitor)},
        "hard_gates": {
            "clean_accuracy": {"minimum": 0.79},
            "background_mean_accuracy": {"minimum": 0.72},
            "background_worst_accuracy": {"minimum": 0.70},
            "foreground_only_accuracy": {"minimum": 0.70},
            "strict_leakage": {"maximum": 0.27},
            "post_branch_seconds": {"maximum": 10.0},
            "peak_cuda_memory_bytes": {"maximum": 200},
        },
        "material_improvements": {
            "minimum_count": 2,
            "required_any": ["strict_leakage"],
            "thresholds": {
                "background_mean_accuracy": {"minimum": 0.73},
                "strict_leakage": {"maximum": 0.26},
            },
        },
        "competition": {
            "metrics": [
                "clean_accuracy",
                "background_mean_accuracy",
                "background_worst_accuracy",
                "foreground_only_accuracy",
                "strict_leakage",
            ],
            "minimum_strict_wins": {"CAMAL": 3},
        },
        "adjustment_policy": {"maximum_narrow_miss": {}},
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    report = run(path)
    assert report["decision"] == "advance"
    assert report["material_improvement_count"] == 2
    assert report["required_material_improvement_passed"]
    assert report["dominated_by"] == []
    assert report["strict_wins"]["CAMAL"] == 3
    assert report["failed_strict_wins"] == []


def test_screen_is_incomplete_before_candidate_evaluation_exists(tmp_path: Path):
    protocol = {
        "protocol": "test",
        "branch_epoch": 5,
        "candidate": {
            "name": "candidate",
            "run_dir": str(tmp_path / "missing"),
            "expected_method": "fcal_v2_relative",
        },
        "anchor": {"name": "anchor", "run_dir": str(tmp_path / "missing")},
        "references": {},
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    report = run(path)
    assert report["decision"] == "incomplete"
    assert report["missing_artifacts"]
