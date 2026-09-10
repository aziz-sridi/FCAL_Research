from pathlib import Path

import numpy as np
import pandas as pd

from fcal_study.compare import holm_adjust, paired_bootstrap_mean_interval, run


def _evaluation(path: Path, predictions: np.ndarray, leakage: np.ndarray) -> None:
    path.mkdir()
    labels = np.array([0, 0, 1, 1], dtype=np.int16)
    probabilities = np.full((4, 2), 0.5, dtype=np.float16)
    for variant in ("clean", "bg_blur_s1"):
        payload = {
            "image_id": np.arange(4),
            "label": labels,
            "prediction": predictions,
            "probabilities": probabilities,
        }
        if variant == "clean":
            payload.update(
                strict_leakage=leakage,
                mask_valid=np.ones(4, dtype=np.uint8),
            )
        np.savez(path / f"{variant}.npz", **payload)


def test_paired_bootstrap_and_holm_are_deterministic():
    first = paired_bootstrap_mean_interval(
        np.array([1.0, 0.0, -1.0, 1.0]), samples=200, seed=7,
    )
    second = paired_bootstrap_mean_interval(
        np.array([1.0, 0.0, -1.0, 1.0]), samples=200, seed=7,
    )
    assert first == second
    np.testing.assert_allclose(
        holm_adjust(np.array([0.01, 0.04, 0.03])),
        np.array([0.03, 0.06, 0.06]),
    )


def test_compare_adds_paired_intervals_leakage_and_full_family_holm(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _evaluation(
        baseline,
        np.array([0, 1, 0, 1], dtype=np.int16),
        np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
    )
    _evaluation(
        candidate,
        np.array([0, 0, 1, 1], dtype=np.int16),
        np.array([0.3, 0.2, 0.1, 0.0], dtype=np.float32),
    )
    run(
        str(baseline), str(candidate), str(output),
        bootstrap_samples=200, bootstrap_seed=9,
    )
    frame = pd.read_csv(output / "transition_summary.csv")
    assert len(frame) == 2
    assert frame["mcnemar_exact_p_holm"].notna().all()
    clean = frame.loc[frame["variant"] == "clean"].iloc[0]
    assert clean["accuracy_difference"] == 0.5
    assert clean["accuracy_difference_ci_low"] <= 0.5
    assert clean["accuracy_difference_ci_high"] >= 0.5
    np.testing.assert_allclose(clean["strict_leakage_difference"], -0.1)
