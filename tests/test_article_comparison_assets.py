from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fcal_study.build_v1_v2_article_assets import (
    ARCHITECTURE_SLUGS,
    ARTICLE_METRICS,
    GENERAL_BASELINES,
    GENERAL_TUNED_METHODS,
    build_comparison_tables,
)


def _metric_rows(methods: tuple[str, ...], include_isnet: bool) -> pd.DataFrame:
    rows = []
    for architecture in ARCHITECTURE_SLUGS:
        for method_index, method in enumerate(methods):
            row = {
                "architecture_slug": architecture,
                "method": method,
            }
            row.update({metric: 0.5 + method_index / 100 for metric, _, _ in ARTICLE_METRICS})
            rows.append(row)
    if include_isnet:
        row = {"architecture_slug": "resnet50", "method": "Faster-ISNet"}
        row.update({metric: 0.4 for metric, _, _ in ARTICLE_METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def _wtl(reference: str, candidates: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "metric": metric,
            "reference": reference,
            "candidate": candidate,
            "architectures": 6,
            "wins": 1,
            "ties": 0,
            "losses": 5,
        }
        for candidate in candidates
        for metric, _, _ in ARTICLE_METRICS
    ])


def _write_artifacts(fixed_root: Path, tuned_root: Path) -> None:
    fixed_root.mkdir()
    tuned_root.mkdir()
    fixed_methods = ("CE", *GENERAL_TUNED_METHODS)
    fixed_metrics = _metric_rows(fixed_methods, include_isnet=True)
    fixed_metrics.to_csv(fixed_root / "comparison_metrics.csv", index=False)
    _wtl("FCAL V2", GENERAL_BASELINES).to_csv(
        fixed_root / "architecture_win_tie_loss.csv", index=False
    )
    efficiency = fixed_metrics[["architecture_slug", "method"]].copy()
    efficiency["post_branch_wall_seconds"] = 600.0
    efficiency.to_csv(fixed_root / "efficiency_runs.csv", index=False)
    overhead_rows = []
    for candidate in GENERAL_BASELINES:
        for architecture in ARCHITECTURE_SLUGS:
            overhead_rows.append({
                "architecture": architecture,
                "reference": "FCAL V2",
                "candidate": candidate,
                "post_branch_wall_seconds_overhead_percent": 25.0,
                "peak_cuda_memory_gib_overhead_percent": 10.0,
            })
    overhead_rows.append({
        "architecture": "resnet50",
        "reference": "FCAL V2",
        "candidate": "Faster-ISNet",
        "post_branch_wall_seconds_overhead_percent": 800.0,
        "peak_cuda_memory_gib_overhead_percent": 500.0,
    })
    pd.DataFrame(overhead_rows).to_csv(fixed_root / "efficiency_overhead.csv", index=False)

    tuned_metrics = _metric_rows(GENERAL_TUNED_METHODS, include_isnet=True)
    tuned_metrics.to_csv(tuned_root / "tuned_winner_metrics.csv", index=False)
    selected = tuned_metrics[["architecture_slug", "method"]].copy()
    selected["coefficient"] = selected["method"].map({
        "FCAL V2": "fcal_v2.support_weight",
        "RRR": "rrr.lambda",
        "GAIN-ext": "gain.supervision_weight",
        "CAMAL": "camal.coefficient",
        "Faster-ISNet": "isnet.heatmap_weight",
    })
    selected["value"] = selected["method"].map({
        "FCAL V2": 0.1,
        "RRR": 0.2,
        "GAIN-ext": 0.3,
        "CAMAL": 0.4,
        "Faster-ISNet": 0.5,
    })
    selected["selection_split"] = "validation"
    selected.to_csv(tuned_root / "validation_selected_hyperparameters.csv", index=False)
    _wtl("FCAL V2", GENERAL_BASELINES).to_csv(
        tuned_root / "tuned_architecture_win_tie_loss.csv", index=False
    )
    fixed_tuned = pd.DataFrame([
        {
            "method": method,
            "metric": metric,
            "architectures": 6,
            "wins": 4,
            "ties": 0,
            "losses": 2,
        }
        for method in GENERAL_TUNED_METHODS
        for metric, _, _ in ARTICLE_METRICS
    ])
    fixed_tuned.to_csv(
        tuned_root / "fixed_vs_tuned_architecture_win_tie_loss.csv", index=False
    )
    attempts = pd.DataFrame([
        {"status": "success", "gpu_hours": 1.0, "method": method}
        for method in (*GENERAL_TUNED_METHODS, "Faster-ISNet")
    ] + [{"status": "failure", "gpu_hours": 0.25, "method": "RRR"}])
    attempts.to_csv(tuned_root / "tuning_attempts.csv", index=False)
    accounting_rows = []
    for method in (*GENERAL_TUNED_METHODS, "Faster-ISNet"):
        method_attempts = attempts.loc[attempts["method"].eq(method)]
        failures = method_attempts["status"].ne("success")
        accounting_rows.append({
            "scope": "method",
            "method": method,
            "attempts": len(method_attempts),
            "failed_attempts": int(failures.sum()),
            "gpu_hours_including_failures": method_attempts["gpu_hours"].sum(),
            "failed_gpu_hours": method_attempts.loc[failures, "gpu_hours"].sum(),
        })
    accounting_rows.append({
        "scope": "study",
        "method": "ALL",
        "attempts": len(attempts),
        "failed_attempts": 1,
        "gpu_hours_including_failures": attempts["gpu_hours"].sum(),
        "failed_gpu_hours": 0.25,
    })
    pd.DataFrame(accounting_rows).to_csv(
        tuned_root / "tuning_gpu_hour_accounting.csv", index=False
    )


def test_comparison_tables_are_generated_from_complete_artifacts(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed"
    tuned = tmp_path / "tuned"
    output = tmp_path / "generated"
    _write_artifacts(fixed, tuned)
    paths = build_comparison_tables(fixed, tuned, output)
    assert len(paths) == 9
    assert all(path.is_file() for path in paths)
    fixed_table = (output / "fixed_comparison_table.tex").read_text(encoding="utf-8")
    tuned_parameters = (output / "tuned_hyperparameters_table.tex").read_text(encoding="utf-8")
    assert "Faster-ISNet$^{\\dagger}$" in fixed_table
    assert "ResNet-50 only" in fixed_table
    assert "A dash is structural non-support" in tuned_parameters
    assert "ResNet-50 & 0.1 & 0.2 & 0.3 & 0.4 & 0.5" in tuned_parameters
    assert (output / "comparison_tables_manifest.json").is_file()


def test_comparison_table_generation_fails_before_partial_final_artifacts(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed"
    tuned = tmp_path / "tuned"
    output = tmp_path / "generated"
    fixed.mkdir()
    tuned.mkdir()
    with pytest.raises(FileNotFoundError, match="artifacts are incomplete"):
        build_comparison_tables(fixed, tuned, output)
    assert not output.exists()


def test_support_restricted_isnet_cannot_be_imputed_across_architectures(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed"
    tuned = tmp_path / "tuned"
    _write_artifacts(fixed, tuned)
    metrics = pd.read_csv(tuned / "tuned_winner_metrics.csv")
    fake = metrics.loc[
        metrics["architecture_slug"].eq("densenet121") & metrics["method"].eq("FCAL V2")
    ].copy()
    fake["method"] = "Faster-ISNet"
    pd.concat([metrics, fake], ignore_index=True).to_csv(
        tuned / "tuned_winner_metrics.csv", index=False
    )
    with pytest.raises(ValueError, match="support matrix"):
        build_comparison_tables(fixed, tuned, tmp_path / "generated")
