from pathlib import Path

import pandas as pd
import pytest

from fcal_study.config import load_config
from fcal_study.data import OxfordPetDataset, dataset_from_config

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits" / "oxford_pet_seed42_val0.2"


def frame(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS / f"{name}.csv")


def test_oxford_pet_uses_official_sealed_test_and_disjoint_splits():
    train, validation, test = frame("train"), frame("validation"), frame("test")
    assert (len(train), len(validation), len(test)) == (2944, 736, 3669)
    sets = [set(item.numeric_id) for item in (train, validation, test)]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert all(set(item.label) == set(range(37)) for item in (train, validation, test))


def test_oxford_pet_sample_has_binary_paired_trimap_geometry():
    config = load_config(ROOT / "configs" / "oxford_pet" / "oxford_pet_resnet50_fcal_matrix.yaml")
    if not Path(config["data"]["pet_root"]).is_dir():
        pytest.skip("Oxford-IIIT Pet is not available locally")
    dataset = OxfordPetDataset(config, "train", need_mask=True)
    first = dataset[(2, 0)]
    second = dataset[(2, 0)]
    assert first["image"].shape == (3, 224, 224)
    assert first["mask"].shape == (1, 224, 224)
    assert set(first["mask"].unique().tolist()) == {0.0, 1.0}
    assert first["image"].equal(second["image"])
    assert first["mask"].equal(second["mask"])


def test_oxford_pet_configs_cover_full_six_model_matrix():
    names = (
        "oxford_pet_resnet50_fcal_matrix.yaml",
        "oxford_pet_densenet121_fcal_matrix.yaml",
        "oxford_pet_efficientnet_b0_fcal_matrix.yaml",
        "oxford_pet_vit_tiny_fcal_matrix.yaml",
        "oxford_pet_vit_small_fcal_matrix.yaml",
        "oxford_pet_vit_base_fcal_matrix.yaml",
    )
    configs = [load_config(ROOT / "configs" / "oxford_pet" / name) for name in names]
    assert len({item["model"]["name"] for item in configs}) == 6
    assert {item["model"]["family"] for item in configs} == {"cnn", "vit"}
    assert all(item["data"]["dataset"] == "oxford_pet" for item in configs)
    assert all(item["model"]["num_classes"] == 37 for item in configs)
    assert all("fcal_v2" in item and "ufcal" in item for item in configs)
    assert isinstance(dataset_from_config(configs[0], "validation", need_mask=False), OxfordPetDataset)
