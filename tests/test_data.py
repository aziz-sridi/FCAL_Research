from pathlib import Path

import pytest

from fcal_study.data import EpochTupleSampler, VariantBatchSampler
from fcal_study.config import load_config
from fcal_study.data import CubDataset


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits" / "cub200_2011_seed42_val0.2"


def ids(name: str) -> set[int]:
    return {int(value) for value in (SPLITS / f"{name}.txt").read_text().splitlines() if value.strip()}


def test_splits_are_disjoint_and_complete():
    train, validation, test = ids("train"), ids("validation"), ids("test")
    assert len(train) == 4794
    assert len(validation) == 1200
    assert len(test) == 5794
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == 11788


def test_epoch_sampler_can_resume_exactly():
    first = EpochTupleSampler(20, 42)
    epoch0 = list(first)
    epoch1 = list(first)
    resumed = EpochTupleSampler(20, 42)
    resumed.epoch = 1
    assert list(resumed) == epoch1
    assert epoch0 != epoch1


def test_real_cub_sample_has_paired_deterministic_geometry():
    config = load_config(ROOT / "configs" / "cub" / "resnet50_ce.yaml")
    if not all(Path(config["data"][key]).is_dir() for key in ("cub_root", "segmentation_root")):
        pytest.skip("CUB-200-2011 is not available locally")
    dataset = CubDataset(config, "train", need_mask=True)
    first = dataset[(3, 0)]
    second = dataset[(3, 0)]
    assert first["image"].shape == (3, 224, 224)
    assert first["mask"].shape == (1, 224, 224)
    assert set(first["mask"].unique().tolist()).issubset({0.0, 1.0})
    assert first["image"].equal(second["image"])
    assert first["mask"].equal(second["mask"])


def test_variant_batches_never_cross_boundaries():
    sampler = VariantBatchSampler(dataset_size=10, variant_count=3, batch_size=4)
    batches = list(sampler)
    assert len(batches) == 9
    for batch in batches:
        assert len({item[1] for item in batch}) == 1
