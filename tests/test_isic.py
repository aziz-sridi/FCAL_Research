from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcal_study.config import load_config
from fcal_study.data import IsicDataset
from fcal_study.evaluate import classification_metrics


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits" / "isic_skin_seed42_grouped"


def frame(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS / f"{name}.csv", dtype={"lesion_id": str}).fillna("")


def test_isic_splits_are_image_and_lesion_disjoint():
    train, validation, test = frame("train"), frame("validation"), frame("test")
    assert (len(train), len(validation), len(test)) == (1414, 356, 100)
    image_sets = [set(item.image_id) for item in (train, validation, test)]
    lesion_sets = [set(item.lesion_id) - {""} for item in (train, validation, test)]
    for left in range(3):
        for right in range(left + 1, 3):
            assert not image_sets[left] & image_sets[right]
            assert not lesion_sets[left] & lesion_sets[right]


def test_isic_sample_has_paired_deterministic_geometry():
    config = load_config(ROOT / "configs" / "isic" / "isic_resnet50_fcal_matrix.yaml")
    if not all(Path(config["data"][key]).is_dir() for key in ("isic_root", "segmentation_root")):
        pytest.skip("ISIC is not available locally")
    dataset = IsicDataset(config, "train", need_mask=True)
    first = dataset[(2, 0)]
    second = dataset[(2, 0)]
    assert first["image"].shape == (3, 224, 224)
    assert first["mask"].shape == (1, 224, 224)
    assert set(first["mask"].unique().tolist()).issubset({0.0, 1.0})
    assert first["image"].equal(second["image"])
    assert first["mask"].equal(second["mask"])


def test_skin_metrics_include_balanced_accuracy_and_macro_f1():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    predictions = np.asarray([0, 1, 1, 1, 2, 0])
    metrics = classification_metrics(labels, predictions, 3)
    assert metrics["balanced_accuracy"] == 2 / 3
    assert 0 < metrics["macro_f1"] < 1
    assert set(metrics["per_class_recall"]) == {"0", "1", "2"}


def test_isic_configs_preserve_the_six_model_matrix():
    names = [
        "isic_resnet50_fcal_matrix.yaml",
        "isic_densenet121_fcal_matrix.yaml",
        "isic_efficientnet_b0_fcal_matrix.yaml",
        "isic_vit_tiny_avg_fcal_matrix.yaml",
        "isic_vit_small_avg_fcal_matrix.yaml",
        "isic_vit_base_avg_fcal_matrix.yaml",
    ]
    configs = [load_config(ROOT / "configs" / "isic" / name) for name in names]
    assert len({item["model"]["name"] for item in configs}) == 6
    assert all(item["data"]["dataset"] == "isic" for item in configs)
    assert all(item["model"]["num_classes"] == 8 for item in configs)
    assert all("fcal_v2" in item for item in configs)
