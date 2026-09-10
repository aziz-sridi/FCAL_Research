import json
from pathlib import Path

import pandas as pd
import pytest

from fcal_study.baseline_fixed_results import (
    ARCHITECTURES,
    GENERIC_BACKGROUND,
    GENERIC_FOREGROUND,
    REQUIRED_RUN_ARTIFACTS,
    build_pareto_table,
    comparison_pairs,
    discover_current_candidates,
    discover_support_restricted_candidates,
    metrics_from_summary,
    selected_epoch,
    support_restricted_pairs,
    validate_common_prefix,
    validate_existing_metric_row,
    validate_history,
)


def _evaluation_rows() -> list[dict]:
    rows = [
        {
            "variant": "clean",
            "accuracy": 0.8,
            "balanced_accuracy": 0.7,
            "macro_f1": 0.6,
            "strict_leakage": 0.2,
        },
        {"variant": "foreground_only", "accuracy": 0.75},
        {"variant": "background_only", "accuracy": 0.05},
    ]
    rows.extend({"variant": name, "accuracy": 0.4 + index / 100} for index, name in enumerate(GENERIC_BACKGROUND))
    rows.extend({"variant": name, "accuracy": 0.2 + index / 100} for index, name in enumerate(GENERIC_FOREGROUND))
    return rows


def test_metrics_from_summary_uses_declared_generic_variant_families(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_evaluation_rows()), encoding="utf-8")
    metrics = metrics_from_summary(path)
    assert metrics["clean_accuracy"] == 0.8
    assert metrics["bg_mean_accuracy"] == pytest.approx(
        sum(0.4 + index / 100 for index in range(len(GENERIC_BACKGROUND))) / len(GENERIC_BACKGROUND)
    )
    assert metrics["bg_worst_accuracy"] == 0.4
    assert metrics["fg_worst_accuracy"] == 0.2
    assert metrics["foreground_only_accuracy"] == 0.75


def test_published_metric_validation_fails_closed_on_archive_mismatch(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_evaluation_rows()), encoding="utf-8")
    metrics = metrics_from_summary(path)
    validate_existing_metric_row(pd.Series(metrics), metrics, identity="matching")
    changed = dict(metrics, clean_accuracy=metrics["clean_accuracy"] + 1e-6)
    with pytest.raises(ValueError, match="Published/archive mismatch"):
        validate_existing_metric_row(pd.Series(metrics), changed, identity="mismatch")


def test_selection_is_first_epoch_at_stored_validation_maximum():
    history = [
        {"epoch": 1, "validation": {"accuracy": 0.2}},
        {"epoch": 2, "validation": {"accuracy": 0.4}},
        {"epoch": 3, "validation": {"accuracy": 0.4}},
    ]
    assert selected_epoch(history, 0.4) == 2
    with pytest.raises(ValueError, match="does not match history maximum"):
        selected_epoch(history, 0.3)


def test_history_and_common_branch_validation_reject_incomparable_runs():
    history = [
        {
            "epoch": epoch,
            "train": {"seconds": 1.0, "samples": 2},
            "validation": {"seconds": 0.5, "samples": 1, "accuracy": epoch / 100},
        }
        for epoch in range(1, 51)
    ]
    validate_history(history, identity="valid")
    validate_common_prefix(history, list(history), candidate_name="matching")
    changed = json.loads(json.dumps(history))
    changed[4]["train"]["seconds"] = 2.0
    with pytest.raises(ValueError, match="does not share CE"):
        validate_common_prefix(history, changed, candidate_name="changed")
    with pytest.raises(ValueError, match="epochs 1..50"):
        validate_history(history[:-1], identity="short")


def _touch_complete_matrix(root: Path, suffix: str) -> None:
    for slug, _, _ in ARCHITECTURES:
        run_dir = root / f"{slug}-{suffix}-seed42"
        for relative in REQUIRED_RUN_ARTIFACTS:
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")


def test_candidate_discovery_requires_complete_six_architecture_matrix(tmp_path: Path):
    _touch_complete_matrix(tmp_path, "camal")
    _touch_complete_matrix(tmp_path, "gain-ext")
    _touch_complete_matrix(tmp_path, "rrr")
    candidates = discover_current_candidates(tmp_path)
    assert tuple(spec[0] for spec in candidates) == ("CAMAL", "GAIN-ext", "RRR")
    (tmp_path / "vit-base-p16-avg-gain-ext-seed42" / "evaluation" / "summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete fixed GAIN-ext matrix"):
        discover_current_candidates(tmp_path)


def test_support_restricted_discovery_accepts_only_canonical_resnet_run(tmp_path: Path):
    _touch_complete_matrix(tmp_path, "camal")
    run_dir = tmp_path / "resnet50-isnet-seed42"
    for relative in REQUIRED_RUN_ARTIFACTS:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"clean-rerun")
    corrupt = tmp_path / "resnet50-isnet-seed42-interrupted-corrupt-checkpoints-20260828"
    corrupt.mkdir()
    (corrupt / "latest.pt").write_bytes(b"excluded")
    complete = discover_current_candidates(tmp_path)
    restricted = discover_support_restricted_candidates(tmp_path)
    assert tuple(spec[0] for spec in complete) == ("CAMAL",)
    assert tuple(spec[0] for spec in restricted) == ("Faster-ISNet",)
    assert restricted[0][4] == ("resnet50",)


def test_comparison_pairs_preserve_camal_and_add_preceding_candidates():
    assert comparison_pairs(("CAMAL",)) == (("CE", "CAMAL"), ("FCAL V2", "CAMAL"))
    assert comparison_pairs(("CAMAL", "GAIN-ext")) == (
        ("CE", "CAMAL"),
        ("FCAL V2", "CAMAL"),
        ("CE", "GAIN-ext"),
        ("FCAL V2", "GAIN-ext"),
        ("CAMAL", "GAIN-ext"),
    )
    assert comparison_pairs(("CAMAL", "GAIN-ext", "RRR"))[-4:] == (
        ("CE", "RRR"),
        ("FCAL V2", "RRR"),
        ("CAMAL", "RRR"),
        ("GAIN-ext", "RRR"),
    )
    assert support_restricted_pairs(
        ("CAMAL", "GAIN-ext", "RRR"), ("Faster-ISNet",)
    ) == (
        ("CE", "Faster-ISNet"),
        ("FCAL V2", "Faster-ISNet"),
        ("CAMAL", "Faster-ISNet"),
        ("GAIN-ext", "Faster-ISNet"),
        ("RRR", "Faster-ISNet"),
    )


def test_pareto_is_computed_within_architecture_and_retains_objectives():
    metrics = pd.DataFrame([
        {
            "architecture": architecture,
            "method": method,
            "clean_accuracy": value,
            "bg_mean_accuracy": value,
            "foreground_only_accuracy": value,
            "strict_leakage": 1 - value,
        }
        for architecture in ("A", "B")
        for method, value in (("dominated", 0.2), ("front", 0.8))
    ])
    efficiency = pd.DataFrame([
        {
            "architecture": architecture,
            "method": method,
            "post_branch_wall_seconds": cost,
            "peak_cuda_memory_gib": cost,
        }
        for architecture in ("A", "B")
        for method, cost in (("dominated", 2.0), ("front", 1.0))
    ])
    result = build_pareto_table(metrics, efficiency)
    assert result.groupby("architecture")["pareto_optimal"].sum().eq(1).all()
    assert result.loc[result["pareto_optimal"], "method"].eq("front").all()
    assert result["pareto_scope"].eq("within_architecture").all()
