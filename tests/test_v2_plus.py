from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fcal_study.train import train_epoch
from fcal_study.v2_plus import (
    foreground_competitor_margin_loss,
    gated_foreground_competitor_margin_loss,
    mean_preserving_evidence_proxy,
    top_competitor_labels,
)


def _foreground_mask(batch_size: int = 1) -> torch.Tensor:
    mask = torch.zeros(batch_size, 1, 8, 8)
    mask[:, :, :4] = 1.0
    return mask


def test_v2_plus_selects_top_non_target_competitors():
    logits = torch.tensor([[5.0, 4.0, 3.0], [2.0, 6.0, 5.0]])
    labels = torch.tensor([0, 1])
    competitors = top_competitor_labels(logits, labels, top_k=2)
    assert competitors.tolist() == [[1, 2], [2, 0]]
    assert not bool((competitors == labels[:, None]).any())


def test_mean_preserving_proxy_keeps_values_and_centers_auxiliary_gradient():
    scores = torch.tensor(
        [[[1.0, -2.0], [3.0, 0.5]]], requires_grad=True,
    )
    proxy = mean_preserving_evidence_proxy(scores)

    assert torch.equal(proxy, scores)

    weights = torch.tensor([[[1.0, 2.0], [-1.0, 4.0]]])
    (proxy * weights).sum().backward()
    expected = weights - weights.mean(dim=(-2, -1), keepdim=True)
    assert torch.allclose(scores.grad, expected)
    assert torch.allclose(scores.grad.sum(dim=(-2, -1)), torch.zeros(1))


def test_mean_preserving_proxy_rejects_non_spatial_evidence():
    with pytest.raises(ValueError, match="shape"):
        mean_preserving_evidence_proxy(torch.ones(2, 3))


def test_v2_plus_rewards_target_over_rival_foreground_evidence():
    target = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    rival = -target
    good = foreground_competitor_margin_loss(
        [target], [[rival]], _foreground_mask()
    )
    bad = foreground_competitor_margin_loss(
        [rival], [[target]], _foreground_mask()
    )
    assert good < bad


def test_v2_plus_margin_is_scale_invariant_and_trains_both_maps():
    target = torch.tensor(
        [[[1.0, 0.5], [-0.5, -1.0]]], requires_grad=True
    )
    rival = torch.tensor(
        [[[-0.5, -1.0], [1.0, 0.5]]], requires_grad=True
    )
    base = foreground_competitor_margin_loss(
        [target], [[rival]], _foreground_mask()
    )
    scaled = foreground_competitor_margin_loss(
        [7.0 * target], [[0.25 * rival]], _foreground_mask()
    )
    assert torch.allclose(base, scaled, atol=1e-6)
    base.backward()
    assert target.grad is not None and bool((target.grad != 0).any())
    assert rival.grad is not None and bool((rival.grad != 0).any())


def test_v2_plus_invalid_masks_return_a_differentiable_zero():
    target = torch.randn(2, 3, 3, requires_grad=True)
    rival = torch.randn(2, 3, 3, requires_grad=True)
    loss = foreground_competitor_margin_loss(
        [target], [[rival]], torch.zeros(2, 1, 8, 8)
    )
    loss.backward()
    assert loss.item() == 0.0
    assert target.grad is not None
    assert rival.grad is not None


def _gated_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    rival: torch.Tensor,
):
    return gated_foreground_competitor_margin_loss(
        logits,
        torch.zeros(logits.shape[0], dtype=torch.long),
        [target],
        [[rival]],
        _foreground_mask(logits.shape[0]),
    )


def test_gated_v2_plus_focuses_on_moderate_confidence():
    target = torch.zeros(1, 2, 2)
    rival = torch.ones(1, 2, 2)
    low = _gated_loss(torch.tensor([[-3.0, 3.0]]), target, rival)
    moderate = _gated_loss(torch.tensor([[0.0, 0.0]]), target, rival)
    high = _gated_loss(torch.tensor([[3.0, -3.0]]), target, rival)
    assert moderate.gate > low.gate
    assert moderate.gate > high.gate


def test_gated_v2_plus_focuses_on_insufficient_foreground_support():
    logits = torch.tensor([[0.0, 0.0]])
    strong_target = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    weak_target = -strong_target
    strong = _gated_loss(logits, strong_target, weak_target)
    weak = _gated_loss(logits, weak_target, strong_target)
    assert strong.foreground_support > weak.foreground_support
    assert strong.gate < weak.gate


def test_gated_v2_plus_gate_is_detached_but_maps_train():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    target = torch.zeros(1, 2, 2, requires_grad=True)
    rival = torch.ones(1, 2, 2, requires_grad=True)
    result = _gated_loss(logits, target, rival)
    result.loss.backward()
    assert logits.grad is None
    assert target.grad is not None and bool((target.grad != 0).any())
    assert rival.grad is not None and bool((rival.grad != 0).any())


def test_gated_v2_plus_invalid_masks_return_graph_connected_zero():
    logits = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 2, 2, requires_grad=True)
    rival = torch.randn(2, 2, 2, requires_grad=True)
    result = gated_foreground_competitor_margin_loss(
        logits,
        torch.tensor([0, 1]),
        [target],
        [[rival]],
        torch.zeros(2, 1, 8, 8),
    )
    result.loss.backward()
    assert result.loss == 0.0
    assert result.valid_mask_fraction == 0.0
    assert target.grad is not None
    assert rival.grad is not None


def test_gated_v2_plus_ungated_loss_matches_legacy_with_mixed_masks():
    torch.manual_seed(7)
    target_maps = [torch.randn(3, 3, 3), torch.randn(3, 2, 2)]
    rival_maps = [
        [torch.randn(3, 3, 3), torch.randn(3, 2, 2)],
        [torch.randn(3, 3, 3), torch.randn(3, 2, 2)],
    ]
    masks = _foreground_mask(3)
    masks[1] = 0.0
    legacy = foreground_competitor_margin_loss(
        target_maps, rival_maps, masks
    )
    gated = gated_foreground_competitor_margin_loss(
        torch.randn(3, 4),
        torch.tensor([0, 1, 2]),
        target_maps,
        rival_maps,
        masks,
    )
    assert torch.allclose(gated.ungated_loss, legacy, atol=1e-7)


def test_v2_plus_config_is_a_strict_v2_successor():
    root = Path(__file__).resolve().parents[1]
    v2 = yaml.safe_load((root / "configs/cub/resnet50_fcal_v2.yaml").read_text())
    candidate = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_plus.yaml").read_text()
    )
    assert candidate["method"] == "fcal_v2_plus"
    for key in ("data", "model", "training", "fcal", "fcal_v2"):
        assert candidate[key] == v2[key]
    assert "fcal_v3" not in candidate
    assert "intervention" not in candidate["fcal_v2_plus"]


def test_gated_v2_plus_config_only_adds_the_frozen_gate():
    root = Path(__file__).resolve().parents[1]
    v2 = yaml.safe_load((root / "configs/cub/resnet50_fcal_v2.yaml").read_text())
    plus = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_plus.yaml").read_text()
    )
    gated = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_plus_gated.yaml").read_text()
    )
    assert gated["method"] == "fcal_v2_plus_gated"
    for key in ("data", "model", "training", "fcal", "fcal_v2"):
        assert gated[key] == v2[key]
    settings = gated["fcal_v2_plus_gated"]
    for key, value in plus["fcal_v2_plus"].items():
        assert settings[key] == value
    assert settings | {
        "confidence_floor": 0.25,
        "confidence_ceiling": 0.80,
        "confidence_gate_temperature": 0.05,
        "support_gate_temperature": 0.10,
    } == settings


def test_v2_plus_vit_config_is_a_strict_v2_successor():
    root = Path(__file__).resolve().parents[1]
    v2 = yaml.safe_load(
        (root / "configs/cub/vit_small_avg_fcal_v2.yaml").read_text()
    )
    candidate = yaml.safe_load(
        (root / "configs/cub/vit_small_avg_fcal_v2_plus.yaml").read_text()
    )
    assert candidate["method"] == "fcal_v2_plus"
    for key in ("data", "model", "training", "fcal", "fcal_v2"):
        assert candidate[key] == v2[key]
    assert candidate["fcal_v2_plus"] == yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_plus.yaml").read_text()
    )["fcal_v2_plus"]
    assert "intervention" not in candidate["fcal_v2_plus"]


class _SingleForwardEvidenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.classifier = nn.Linear(4, 3)
        self.forward_calls = 0
        self.forward_inputs = []
        self._map = None

    def forward(self, images, capture_attention=False):
        del capture_attention
        self.forward_calls += 1
        self.forward_inputs.append(images.detach().clone())
        self._map = F.relu(self.features(images))
        return self.classifier(self._map.mean(dim=(-2, -1)))

    def raw_class_evidence(self, labels):
        weights = self.classifier.weight[labels]
        return [torch.einsum("bc,bchw->bhw", weights, self._map)]


def test_v2_plus_train_epoch_uses_original_input_and_exactly_one_forward():
    model = _SingleForwardEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(4, 3, 16, 16)
    original = images.clone()
    batch = {
        "image": images,
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": torch.zeros(4, 1, 16, 16),
    }
    batch["mask"][:, :, 4:12, 4:12] = 1.0
    config = {
        "method": "fcal_v2_plus",
        "data": {"accumulation_steps": 1},
        "training": {
            "channels_last": False,
            "amp": "off",
            "grad_clip_norm": 1.0,
        },
        "fcal": {
            "alpha": 0.15,
            "lambda_min": 0.05,
            "lambda_max": 0.30,
            "epsilon": 1e-8,
        },
        "fcal_v2": {
            "warmup_epochs": 0,
            "ramp_epochs": 1,
            "support_temperature": 1.0,
            "support_margin": 0.0,
            "support_weight": 0.05,
            "tolerance_cell_fraction": 0.5,
            "maximum_tolerance_pixels": 4,
            "epsilon": 1e-8,
        },
        "fcal_v2_plus": {
            "coefficient": 0.025,
            "top_k_competitors": 2,
            "evidence_temperature": 1.0,
            "margin": 0.25,
            "loss_temperature": 0.25,
        },
    }
    metrics = train_epoch(
        model, [batch], optimizer, config, epoch=0, device=torch.device("cpu")
    )
    assert model.forward_calls == 1
    assert len(model.forward_inputs) == 1
    assert torch.equal(model.forward_inputs[0], original)
    assert torch.equal(batch["image"], original)
    assert metrics["competitor_margin"] > 0.0
    assert metrics["samples"] == 4


def test_gated_v2_plus_train_epoch_uses_original_input_and_exactly_one_forward():
    model = _SingleForwardEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(4, 3, 16, 16)
    original = images.clone()
    batch = {
        "image": images,
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": torch.zeros(4, 1, 16, 16),
    }
    batch["mask"][:, :, 4:12, 4:12] = 1.0
    config = {
        "method": "fcal_v2_plus_gated",
        "data": {"accumulation_steps": 1},
        "training": {
            "channels_last": False,
            "amp": "off",
            "grad_clip_norm": 1.0,
        },
        "fcal": {
            "alpha": 0.15,
            "lambda_min": 0.05,
            "lambda_max": 0.30,
            "epsilon": 1e-8,
        },
        "fcal_v2": {
            "warmup_epochs": 0,
            "ramp_epochs": 1,
            "support_temperature": 1.0,
            "support_margin": 0.0,
            "support_weight": 0.05,
            "tolerance_cell_fraction": 0.5,
            "maximum_tolerance_pixels": 4,
            "epsilon": 1e-8,
        },
        "fcal_v2_plus_gated": {
            "coefficient": 0.025,
            "top_k_competitors": 2,
            "evidence_temperature": 1.0,
            "margin": 0.25,
            "loss_temperature": 0.25,
            "confidence_floor": 0.25,
            "confidence_ceiling": 0.80,
            "confidence_gate_temperature": 0.05,
            "support_gate_temperature": 0.10,
        },
    }
    metrics = train_epoch(
        model, [batch], optimizer, config, epoch=0, device=torch.device("cpu")
    )
    assert model.forward_calls == 1
    assert len(model.forward_inputs) == 1
    assert torch.equal(model.forward_inputs[0], original)
    assert torch.equal(batch["image"], original)
    assert metrics["competitor_margin"] > 0.0
    assert metrics["competitor_margin_ungated"] > 0.0
    assert 0.0 < metrics["competitor_gate"] < 1.0
    assert 0.0 < metrics["competitor_target_confidence"] < 1.0
    assert metrics["competitor_valid_mask_fraction"] == 1.0
    assert metrics["samples"] == 4
