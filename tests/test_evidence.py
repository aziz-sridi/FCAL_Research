import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fcal_study.evidence import (
    attention_iou,
    efficiency_overhead,
    faithfulness_auc,
    paired_comparison,
    paired_comparisons,
    pareto_front,
    perturbation_images,
    ranked_perturbation_masks,
    run_efficiency,
    seed_summary,
    tuning_audit,
    win_tie_loss,
)


def test_seed_summary_uses_seeds_not_repeated_rows():
    frame = pd.DataFrame({
        "method": ["A", "A", "A", "A"],
        "seed": [1, 1, 2, 2],
        "score": [1.0, 3.0, 4.0, 6.0],
    })
    row = seed_summary(frame, "method", "score").iloc[0]
    assert row["n_seeds"] == 2
    assert row["mean"] == pytest.approx(3.5)  # means 2 and 5, not n=4
    assert row["sd"] == pytest.approx(np.sqrt(4.5))
    assert row["ci_low"] < row["mean"] < row["ci_high"]


def test_paired_comparison_signs_lower_is_better_effects():
    frame = pd.DataFrame({
        "seed": [1, 1, 2, 2, 3, 3],
        "method": ["CE", "FCAL", "CE", "FCAL", "CE", "FCAL"],
        "leakage": [0.5, 0.3, 0.4, 0.3, 0.6, 0.2],
    })
    result = paired_comparison(
        frame, reference="CE", candidate="FCAL", metric="leakage",
        unit_cols="seed", lower_is_better=True,
    )
    assert result["mean_difference"] == pytest.approx(-0.2333333333)
    assert result["mean_improvement"] == pytest.approx(0.2333333333)
    assert (result["wins"], result["ties"], result["losses"]) == (3, 0, 0)
    assert result["sign_permutation_p"] == pytest.approx(0.25)


def test_paired_comparison_rejects_unmatched_or_duplicate_units():
    unmatched = pd.DataFrame({"seed": [1, 1, 2], "method": ["A", "B", "A"], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="incomplete"):
        paired_comparison(unmatched, reference="A", candidate="B", metric="x", unit_cols="seed")
    duplicated = pd.concat([unmatched.iloc[:2], unmatched.iloc[:1]])
    with pytest.raises(ValueError, match="one observation"):
        paired_comparison(duplicated, reference="A", candidate="B", metric="x", unit_cols="seed")


def test_paired_comparisons_returns_every_candidate_metric():
    frame = pd.DataFrame({
        "seed": [1, 1, 1, 2, 2, 2],
        "method": ["CE", "V1", "V2"] * 2,
        "accuracy": [0.7, 0.71, 0.72, 0.8, 0.81, 0.83],
        "leakage": [0.5, 0.4, 0.3, 0.6, 0.5, 0.2],
    })
    result = paired_comparisons(
        frame, reference="CE", candidates=["V1", "V2"],
        metrics=["accuracy", "leakage"], unit_cols="seed",
        lower_is_better=["leakage"],
    )
    assert len(result) == 4
    assert {"paired_t_p_holm", "sign_permutation_p_holm"}.issubset(result.columns)
    leakage = result[(result["candidate"] == "V2") & (result["metric"] == "leakage")].iloc[0]
    assert leakage["mean_improvement"] > 0


def test_win_tie_loss_and_pareto_direction():
    frame = pd.DataFrame({
        "architecture": ["r", "r", "v", "v"],
        "method": ["CE", "FCAL", "CE", "FCAL"],
        "seed": [1, 1, 1, 1],
        "accuracy": [0.8, 0.82, 0.7, 0.69],
    })
    row = win_tie_loss(
        frame, reference="CE", candidate="FCAL", metrics="accuracy",
    ).iloc[0]
    assert (row["wins"], row["ties"], row["losses"]) == (1, 0, 1)
    candidates = pd.DataFrame({"accuracy": [0.8, 0.9, 0.85], "hours": [1.0, 2.0, 3.0]})
    assert pareto_front(candidates, maximize=["accuracy"], minimize=["hours"]).tolist() == [True, True, False]


def test_attention_iou_uses_per_map_max_normalization():
    attention = np.array([[[0.0, 2.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])
    masks = np.array([[[0, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=bool)
    result = attention_iou(attention, masks, threshold=0.7)
    assert result[0] == 1.0
    assert np.isnan(result[1])


def test_ranked_perturbations_and_faithfulness_auc():
    attention = np.array([[[4.0, 3.0], [2.0, 1.0]]])
    fractions = [0.0, 0.5, 1.0]
    masks = ranked_perturbation_masks(attention, fractions)
    assert masks.sum(axis=(2, 3)).tolist() == [[0, 2, 4]]
    image = np.arange(4, dtype=float).reshape(1, 1, 2, 2)
    inserted = perturbation_images(image, attention, fractions, mode="insertion")
    removed = perturbation_images(image, attention, fractions, mode="removal")
    assert np.count_nonzero(inserted[0, 0]) == 0
    assert np.array_equal(inserted[0, -1], image[0])
    assert np.array_equal(removed[0, 0], image[0])
    assert np.count_nonzero(removed[0, -1]) == 0
    auc = faithfulness_auc(fractions, np.array([[0.0, 0.6, 1.0]]))
    assert auc[0] == pytest.approx(0.55)


def test_run_efficiency_weights_throughput_and_reports_vram(tmp_path: Path):
    config = {
        "run_id": "demo", "seed": 7, "method": "fcal_v2",
        "model": {"family": "cnn", "name": "r50"},
        "training": {"save_epoch5_branch": True}, "fcal": {"warmup_epochs": 1},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "environment.json").write_text(json.dumps({"cuda_available": True, "cuda_device": "GPU"}), encoding="utf-8")
    rows = [
        {"train": {"seconds": 2, "samples": 10, "peak_cuda_memory_bytes": 2**30}, "validation": {"seconds": 1}},
        {"train": {"seconds": 8, "samples": 90, "peak_cuda_memory_bytes": 2**31}, "validation": {"seconds": 1}},
    ]
    (tmp_path / "history.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = run_efficiency(tmp_path, branch_epochs=1)
    assert result["wall_seconds"] == 12
    assert result["train_samples_per_second"] == 10
    assert result["peak_cuda_memory_gib"] == 2
    assert result["gpu_hours"] == pytest.approx(12 / 3600)
    assert result["common_branch_epochs"] == 1
    assert result["post_branch_wall_seconds"] == 9


def test_efficiency_overhead_matches_units_and_directions():
    rows = []
    for seed in (1, 2):
        for method, scale in (("ce", 1.0), ("fcal", 1.5)):
            rows.append({
                "architecture": "r50", "seed": seed, "method": method,
                "train_seconds": 80 * scale, "validation_seconds": 20 * scale,
                "wall_seconds": 100 * scale, "gpu_hours": scale,
                "peak_cuda_memory_gib": 2 * scale,
                "post_branch_train_seconds": 64 * scale,
                "post_branch_validation_seconds": 16 * scale,
                "post_branch_wall_seconds": 80 * scale,
                "post_branch_gpu_hours": 0.8 * scale,
                "train_samples_per_second": 100 / scale,
                "post_branch_train_samples_per_second": 120 / scale,
            })
    result = efficiency_overhead(pd.DataFrame(rows), reference="ce", candidates=["fcal"])
    assert len(result) == 2
    assert result["wall_seconds_overhead_percent"].eq(50).all()
    assert np.allclose(result["train_samples_per_second_ratio"], 2 / 3)


def test_tuning_audit_requires_validation_and_sealed_test():
    good = pd.DataFrame({
        "method": ["CE", "CE", "FCAL", "FCAL"],
        "selection_split": ["validation"] * 4,
        "selected": [True, False, False, True],
        "test_evaluated": [True, False, False, True],
        "gpu_hours": [1.0, 1.5, 2.0, 2.5],
    })
    audit = tuning_audit(good)
    assert audit["equal_trial_budget"].all()
    assert audit.set_index("method").loc["FCAL", "tuning_gpu_hours"] == 4.5
    leaked = good.copy()
    leaked.loc[1, "test_evaluated"] = True
    with pytest.raises(ValueError, match="non-selected"):
        tuning_audit(leaked)
    leaked = good.copy()
    leaked.loc[0, "selection_split"] = "test"
    with pytest.raises(ValueError, match="validation only"):
        tuning_audit(leaked)
    strings = good.copy()
    strings["selected"] = ["true", "false", "false", "true"]
    strings["test_evaluated"] = ["true", "false", "false", "true"]
    assert len(tuning_audit(strings)) == 2
    with_default = pd.concat([
        good.assign(comparison_regime="equal_tuned"),
        pd.DataFrame({
            "method": ["CE", "FCAL"], "selection_split": ["fixed", "preregistered"],
            "selected": [True, True], "test_evaluated": [True, True],
            "gpu_hours": [1.0, 2.0], "comparison_regime": ["fixed_default"] * 2,
        }),
    ], ignore_index=True)
    audit = tuning_audit(with_default)
    assert set(audit["comparison_regime"]) == {"fixed_default", "equal_tuned"}
    assert audit.loc[audit["comparison_regime"].eq("fixed_default"), "tuning_gpu_hours"].eq(0).all()
