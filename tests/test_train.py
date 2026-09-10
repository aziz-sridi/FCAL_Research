import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from fcal_study.train import (
    _amp_context,
    _last_epoch_averages,
    _relocated_training_data_matches,
    _seed_resume_history,
)


def test_last_epoch_averages_reports_final_five_without_selection_bias(tmp_path):
    history = tmp_path / "history.jsonl"
    rows = [
        {
            "epoch": epoch,
            "train": {"ce": float(epoch), "samples": 10},
            "validation": {"accuracy": epoch / 10, "samples": 5},
        }
        for epoch in range(1, 8)
    ]
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    result = _last_epoch_averages(history)
    assert result["epochs"] == [3, 4, 5, 6, 7]
    assert result["train"]["ce"] == 5.0
    assert result["validation"]["accuracy"] == 0.5


def test_amp_context_treats_yaml_boolean_false_as_disabled():
    context = _amp_context(torch.device("cuda"), False)
    assert isinstance(context, nullcontext)


def test_amp_context_treats_off_string_as_disabled():
    context = _amp_context(torch.device("cuda"), "off")
    assert isinstance(context, nullcontext)


def test_relocated_data_requires_matching_loader_settings_and_split_hashes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for name, text in (
        ("train.txt", "1\n2\n"),
        ("validation.txt", "3\n"),
        ("test.txt", "4\n"),
        ("manifest.json", '{"seed": 42}\n'),
    ):
        (first / name).write_text(text, encoding="utf-8")
        (second / name).write_text(text, encoding="utf-8")
    common = {"batch_size": 32, "image_size": 224, "resize_size": 256}
    parent = {**common, "split_dir": str(first), "cub_root": "old"}
    child = {**common, "split_dir": str(second), "cub_root": "new"}
    assert _relocated_training_data_matches(parent, child)
    child["batch_size"] = 16
    assert not _relocated_training_data_matches(parent, child)


def test_relocated_data_accepts_explicit_hash_verified_parent_split(tmp_path):
    archived = tmp_path / "archived"
    current = tmp_path / "current"
    archived.mkdir()
    current.mkdir()
    for directory in (archived, current):
        (directory / "train.txt").write_text("1\n", encoding="utf-8")
        (directory / "validation.txt").write_text("2\n", encoding="utf-8")
        (directory / "test.txt").write_text("3\n", encoding="utf-8")
        (directory / "manifest.json").write_text('{"seed": 42}\n', encoding="utf-8")
    parent = {"batch_size": 32, "split_dir": str(tmp_path / "missing")}
    child = {"batch_size": 32, "split_dir": str(current)}
    assert not _relocated_training_data_matches(parent, child)
    assert _relocated_training_data_matches(parent, child, archived)


def test_relocated_resume_uses_actual_checkpoint_directory_for_hashes_and_history(tmp_path):
    archived_run = tmp_path / "archive" / "resnet50-ce-seed42"
    current_split = tmp_path / "current-split"
    child_run = tmp_path / "child"
    archived_run.mkdir(parents=True)
    current_split.mkdir()
    child_run.mkdir()
    split_hashes = {}
    for name, text in (
        ("train.txt", "1\n"),
        ("validation.txt", "2\n"),
        ("test.txt", "3\n"),
        ("manifest.json", '{"seed": 42}\n'),
    ):
        path = current_split / name
        path.write_text(text, encoding="utf-8")
        split_hashes[f"split_{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (archived_run / "environment.json").write_text(
        json.dumps(split_hashes), encoding="utf-8",
    )
    (archived_run / "history.jsonl").write_text(
        '{"epoch": 1}\n{"epoch": 2}\n{"epoch": 3}\n', encoding="utf-8",
    )
    resume = archived_run / "epoch-002-branch.pt"
    resume.touch()
    parent = {
        "batch_size": 32,
        "split_dir": str(tmp_path / "missing-embedded-split"),
    }
    child = {
        "batch_size": 32,
        "split_dir": str(current_split),
    }
    checkpoint = {"run_dir": str(tmp_path / "missing-embedded-run")}
    assert not Path(checkpoint["run_dir"]).exists()
    assert _relocated_training_data_matches(
        parent,
        child,
        parent_environment=archived_run / "environment.json",
    )
    bad_hashes = {**split_hashes, "split_train.txt_sha256": "0" * 64}
    (archived_run / "environment.json").write_text(
        json.dumps(bad_hashes), encoding="utf-8",
    )
    assert not _relocated_training_data_matches(
        parent,
        child,
        parent_environment=archived_run / "environment.json",
    )
    (archived_run / "environment.json").write_text(
        json.dumps(split_hashes), encoding="utf-8",
    )
    destination = child_run / "history.jsonl"
    _seed_resume_history(resume, destination, completed_epochs=2)
    assert destination.read_text(encoding="utf-8") == '{"epoch": 1}\n{"epoch": 2}\n'
    with pytest.raises(ValueError, match="3 records.*4 completed epochs"):
        _seed_resume_history(resume, destination, completed_epochs=4)
