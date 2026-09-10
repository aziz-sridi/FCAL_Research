from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fcal_study.train import train_epoch
from fcal_study.v2_relative import (
    relative_pooled_margin_loss,
    strongest_rival_margin,
)


def _mask(batch_size: int = 1, size: int = 8) -> torch.Tensor:
    masks = torch.zeros(batch_size, 1, size, size)
    masks[:, :, : size // 2] = 1.0
    return masks


class _RegionalOutputs:
    def __init__(self, foreground: torch.Tensor, background: torch.Tensor):
        self.outputs = [foreground, background]

    def regional_logits(self, regions: torch.Tensor) -> torch.Tensor:
        del regions
        return self.outputs.pop(0)


def _relative(
    foreground: torch.Tensor,
    background: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    background_gradient_scale: float = 0.1,
):
    if scores is None:
        scores = torch.ones(foreground.shape[0], 2, 2)
    background_map = torch.zeros_like(scores)
    background_map[:, 1] = 1.0
    return relative_pooled_margin_loss(
        _RegionalOutputs(foreground, background),
        [scores],
        [background_map],
        torch.zeros(foreground.shape[0], dtype=torch.long),
        _mask(foreground.shape[0]),
        background_gradient_scale=background_gradient_scale,
    )


def test_strongest_rival_margin_ignores_weaker_competitors():
    labels = torch.tensor([0])
    two_classes = strongest_rival_margin(torch.tensor([[4.0, 2.0]]), labels)
    four_classes = strongest_rival_margin(
        torch.tensor([[4.0, 2.0, -5.0, -10.0]]), labels
    )
    torch.testing.assert_close(two_classes, four_classes)


def test_relative_loss_rewards_foreground_over_background_margin():
    good = _relative(torch.tensor([[3.0, 1.0]]), torch.tensor([[1.0, 2.0]]))
    bad = _relative(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 1.0]]))
    assert good.margin_gap > bad.margin_gap
    assert good.loss < bad.loss


def test_shortcut_gate_activates_for_outside_evidence_or_weak_foreground():
    strong_foreground = torch.tensor([[5.0, 0.0]])
    neutral_background = torch.tensor([[0.0, 0.0]])
    low_outside = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    high_outside = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    safe = _relative(strong_foreground, neutral_background, scores=low_outside)
    outside_risk = _relative(
        strong_foreground, neutral_background, scores=high_outside
    )
    margin_risk = _relative(
        torch.tensor([[0.0, 5.0]]), neutral_background, scores=low_outside
    )
    assert outside_risk.shortcut_gate > safe.shortcut_gate
    assert margin_risk.shortcut_gate > safe.shortcut_gate


def test_background_gradient_attenuation_changes_gradient_not_forward_value():
    foreground_full = torch.tensor([[1.0, 0.0]], requires_grad=True)
    background_full = torch.tensor([[1.0, 0.0]], requires_grad=True)
    full = _relative(
        foreground_full, background_full, background_gradient_scale=1.0
    )
    full.loss.backward()
    full_gradient = background_full.grad.detach().clone()

    foreground_low = torch.tensor([[1.0, 0.0]], requires_grad=True)
    background_low = torch.tensor([[1.0, 0.0]], requires_grad=True)
    attenuated = _relative(
        foreground_low, background_low, background_gradient_scale=0.25
    )
    attenuated.loss.backward()
    torch.testing.assert_close(full.loss, attenuated.loss)
    torch.testing.assert_close(background_low.grad, 0.25 * full_gradient)


def test_all_invalid_masks_return_graph_connected_zero():
    scores = torch.randn(2, 2, 2, requires_grad=True)
    result = relative_pooled_margin_loss(
        _RegionalOutputs(torch.zeros(2, 2), torch.zeros(2, 2)),
        [scores],
        [torch.zeros_like(scores)],
        torch.tensor([0, 1]),
        torch.zeros(2, 1, 8, 8),
    )
    result.loss.backward()
    assert result.loss == 0.0
    assert result.valid_mask_fraction == 0.0
    assert scores.grad is not None


class _SingleForwardRegionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.classifier = nn.Linear(4, 3)
        self.forward_calls = 0
        self.regional_calls = 0
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

    def regional_logits(self, regions):
        self.regional_calls += 1
        weights = F.adaptive_avg_pool2d(regions, self._map.shape[-2:])
        pooled = (self._map * weights).flatten(2).sum(dim=2)
        pooled = pooled / weights.flatten(1).sum(dim=1, keepdim=True)
        return self.classifier(pooled)


def _training_config() -> dict:
    return {
        "method": "fcal_v2_relative",
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
        "fcal_v2_relative": {
            "weight": 0.02,
            "delta": 2.0,
            "temperature": 1.0,
            "allowed_outside": 0.2,
            "required_foreground_margin": 0.0,
            "outside_gate_temperature": 0.05,
            "margin_gate_temperature": 1.0,
            "background_gradient_scale": 0.1,
        },
    }


def test_train_epoch_uses_one_original_forward_and_reports_diagnostics():
    model = _SingleForwardRegionalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(4, 3, 8, 8)
    original = images.clone()
    batch = {
        "image": images,
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": _mask(4),
    }
    metrics = train_epoch(
        model, [batch], optimizer, _training_config(), 0, torch.device("cpu")
    )
    assert model.forward_calls == 1
    assert model.regional_calls == 2
    assert torch.equal(model.forward_inputs[0], original)
    assert torch.equal(batch["image"], original)
    assert metrics["relative_pooled_margin"] > 0.0
    assert 0.0 <= metrics["shortcut_gate"] <= 1.0
    assert metrics["relative_valid_mask_fraction"] == 1.0


def test_resnet_config_is_a_strict_v2_successor_with_frozen_parameters():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2.yaml").read_text()
    )
    candidate = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_relative.yaml").read_text()
    )
    assert candidate["method"] == "fcal_v2_relative"
    for key in ("data", "model", "training", "fcal", "fcal_v2"):
        assert candidate[key] == baseline[key]
    assert set(candidate["fcal_v2_relative"]) == {
        "weight",
        "delta",
        "temperature",
        "allowed_outside",
        "required_foreground_margin",
        "outside_gate_temperature",
        "margin_gate_temperature",
        "background_gradient_scale",
    }


@pytest.mark.parametrize("scale", [-0.1, 1.1])
def test_rejects_invalid_background_gradient_scale(scale: float):
    with pytest.raises(ValueError, match="background gradient scale"):
        _relative(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            background_gradient_scale=scale,
        )
