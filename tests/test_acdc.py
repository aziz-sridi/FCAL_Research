import csv
from pathlib import Path

import torch
from PIL import Image

from fcal_study.data import AcdcCardiacDataset
from fcal_study.prepare_acdc import (
    LABELS,
    Patient,
    _volume_paths,
    parse_info,
    stratified_patient_splits,
)
from fcal_study.train import validate


def test_parse_acdc_info_is_case_and_whitespace_tolerant(tmp_path: Path):
    info = tmp_path / "Info.cfg"
    info.write_text("ED: 1\n ES : 12\nGroup: HCM\n", encoding="utf-8")
    assert parse_info(info) == {"ed": "1", "es": "12", "group": "HCM"}


def test_acdc_patient_split_is_stratified_and_disjoint(tmp_path: Path):
    patients = []
    patient_id = 1
    for diagnosis in LABELS:
        for _ in range(20):
            patients.append(Patient(patient_id, tmp_path, diagnosis, 1, 2))
            patient_id += 1
    splits = stratified_patient_splits(patients, 42, 0.2, 0.2)
    assert {name: len(items) for name, items in splits.items()} == {
        "train": 60, "validation": 20, "test": 20,
    }
    ids = {name: {item.patient_id for item in items} for name, items in splits.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    for split, items in splits.items():
        expected = 12 if split == "train" else 4
        assert {
            diagnosis: sum(item.diagnosis == diagnosis for item in items) for diagnosis in LABELS
        } == {diagnosis: expected for diagnosis in LABELS}


def test_acdc_volume_paths_accept_uncompressed_kaggle_mirror(tmp_path: Path):
    patient_dir = tmp_path / "patient002"
    patient_dir.mkdir()
    (patient_dir / "patient002_frame01.nii").touch()
    (patient_dir / "patient002_frame01_gt.nii").touch()
    patient = Patient(2, patient_dir, "DCM", 1, 12)
    image, mask = _volume_paths(patient, 1)
    assert image.suffix == ".nii"
    assert mask.name == "patient002_frame01_gt.nii"


def test_acdc_dataset_returns_patient_group_and_paired_mask(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "splits").mkdir()
    image_path = tmp_path / "images" / "sample.png"
    mask_path = tmp_path / "masks" / "sample.png"
    Image.new("RGB", (40, 30), (10, 20, 30)).save(image_path)
    Image.new("L", (40, 30), 255).save(mask_path)
    fields = ["numeric_id", "patient_id", "diagnosis", "label", "slice_index", "image_path", "mask_path"]
    with (tmp_path / "splits" / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "numeric_id": 7,
            "patient_id": 3,
            "diagnosis": "HCM",
            "label": 3,
            "slice_index": 2,
            "image_path": "images/sample.png",
            "mask_path": "masks/sample.png",
        })
    config = {
        "seed": 42,
        "data": {
            "cardiac_root": str(tmp_path),
            "split_dir": str(tmp_path / "splits"),
            "image_size": 32,
            "resize_size": 32,
        },
    }
    sample = AcdcCardiacDataset(config, "test", need_mask=True)[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["image_id"] == 7
    assert sample["group_id"] == 3
    assert sample["label"] == 3


class _LogitModel(torch.nn.Module):
    def forward(self, images, capture_attention=False):
        return images[:, :2, 0, 0]


def test_validation_averages_slice_logits_by_patient():
    batch = {
        "image": torch.tensor([
            [[[10.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[9.0]], [[0.0]]],
            [[[0.0]], [[10.0]], [[0.0]]],
        ]),
        "label": torch.tensor([0, 0, 1]),
        "group_id": torch.tensor([1, 1, 2]),
    }
    config = {"training": {"channels_last": False, "amp": "off"}}
    metrics = validate(_LogitModel(), [batch], config, torch.device("cpu"))
    assert metrics["slice_accuracy"] == 2 / 3
    assert metrics["accuracy"] == 1.0
    assert metrics["patients"] == 2
