from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from fcal_study import faithfulness
from fcal_study.faithfulness import (
    FRACTIONS,
    deterministic_subset_ids,
    evaluate_loader,
    normalized_positive_maps,
    run,
    summarize,
)


class ToyAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.images = None

    def forward(self, images, capture_attention=False):
        del capture_attention
        self.images = images
        score = images[:, 0].flatten(1).sum(dim=1) + self.anchor
        return torch.stack((score, -score), dim=1)

    def evaluation_attention(self, labels):
        del labels
        return self.images[:, 0].clamp_min(0)


def _config(method="ce"):
    return {
        "method": method,
        "model": {"family": "cnn", "name": "toy"},
        "training": {"amp": "off", "channels_last": False},
    }


def test_deterministic_subset_is_order_independent_and_validates_size():
    first = deterministic_subset_ids([7, 2, 9, 4], 2, 13)
    second = deterministic_subset_ids([4, 9, 2, 7], 2, 13)
    assert first == second
    assert len(first) == 2
    assert deterministic_subset_ids([1, 2], None, 13) is None
    with pytest.raises(ValueError, match="positive"):
        deterministic_subset_ids([1, 2], 0, 13)


def test_normalized_positive_maps_upsamples_and_handles_zero_maps():
    maps = torch.tensor([[[0.0, 2.0], [-1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])
    result = normalized_positive_maps(maps, (4, 4))
    assert result.shape == (2, 4, 4)
    assert float(result[0].max()) == pytest.approx(1.0)
    assert float(result[0].min()) >= 0.0
    assert float(result[1].sum()) == 0.0


def test_evaluate_loader_writes_mask_aware_iou_and_unit_square_aucs():
    image = torch.zeros((2, 3, 2, 2))
    image[0, 0, 0, 0] = 1.0
    image[1, 0, 1, 1] = 1.0
    loader = [{
        "image": image,
        "label": torch.tensor([0, 0]),
        "image_id": torch.tensor([10, 11]),
        "mask": torch.tensor([[[[1, 0], [0, 0]]], [[[1, 1], [1, 1]]]], dtype=torch.float32),
    }]
    payload, metadata = evaluate_loader(ToyAttentionModel(), _config(), loader)
    result = summarize(payload)
    assert metadata["explanation_target"] == "ground_truth_class"
    assert result["samples"] == 2
    assert result["valid_mask_samples"] == 1
    assert result["attention_iou_mean"] == pytest.approx(1.0)
    assert np.isnan(payload["attention_iou"][1])
    assert payload["insertion_confidence"].shape == (2, len(FRACTIONS))
    assert np.all((payload["insertion_auc_normalized"] >= 0) & (payload["insertion_auc_normalized"] <= 1))
    assert np.all((payload["removal_auc_normalized"] >= 0) & (payload["removal_auc_normalized"] <= 1))


def test_rrr_uses_common_target_class_map_for_cross_method_comparison():
    image = torch.ones((1, 3, 2, 2))
    loader = [{
        "image": image,
        "label": torch.tensor([0]),
        "image_id": torch.tensor([10]),
        "mask": torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32),
    }]
    payload, metadata = evaluate_loader(ToyAttentionModel(), _config("rrr"), loader)
    assert len(payload["image_id"]) == 1
    assert metadata["explanation_target"] == "ground_truth_class"


def test_archived_fcal_v2_method_key_uses_common_target_class_map():
    image = torch.ones((1, 3, 2, 2))
    loader = [{
        "image": image,
        "label": torch.tensor([0]),
        "image_id": torch.tensor([10]),
        "mask": torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32),
    }]
    payload, metadata = evaluate_loader(ToyAttentionModel(), _config("fcal_v5"), loader)
    assert len(payload["image_id"]) == 1
    assert metadata["explanation_target"] == "ground_truth_class"


def test_non_class_specific_vit_rollout_is_rejected():
    config = _config()
    config["model"] = {"family": "vit", "name": "toy", "pool": "cls"}
    loader = [{
        "image": torch.ones((1, 3, 2, 2)),
        "label": torch.tensor([0]),
        "image_id": torch.tensor([10]),
        "mask": torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32),
    }]
    with pytest.raises(ValueError, match="not class-specific"):
        evaluate_loader(ToyAttentionModel(), config, loader)


def test_evaluate_loader_rejects_training_only_or_unknown_method_protocols():
    with pytest.raises(ValueError, match="No sealed explanation-map protocol"):
        evaluate_loader(ToyAttentionModel(), _config("unknown"), [])


def test_run_writes_traceable_artifacts_once_and_rejects_train_split(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "best-validation.pt"
    config = {
        **_config(),
        "device": "cpu",
        "run_id": "toy-ce",
        "data": {"batch_size": 1, "split_dir": str(tmp_path)},
    }
    (tmp_path / "validation.txt").write_text("10\n", encoding="utf-8")
    torch.save(
        {"config": config, "model": ToyAttentionModel().state_dict(), "epoch": 5},
        checkpoint_path,
    )
    batch = {
        "image": torch.ones((1, 3, 2, 2)),
        "label": torch.tensor([0]),
        "image_id": torch.tensor([10]),
        "mask": torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32),
    }
    monkeypatch.setattr(faithfulness, "relocate_missing_data_paths", lambda value: value)
    monkeypatch.setattr(faithfulness, "create_model", lambda config, pretrained: ToyAttentionModel())
    monkeypatch.setattr(faithfulness, "make_loader", lambda *args, **kwargs: [batch])
    output = tmp_path / "sealed"
    assert run(checkpoint_path, split="validation", output_dir=output) == output
    assert (output / "per_image.npz").is_file()
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["split"] == "validation"
    assert protocol["checkpoint_sha256"]
    assert protocol["split_file_sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run(checkpoint_path, split="validation", output_dir=output)
    with pytest.raises(ValueError, match="validation or test"):
        run(checkpoint_path, split="train", output_dir=tmp_path / "train")
