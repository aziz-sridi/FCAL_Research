from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fcal_study.train import train_epoch
from fcal_study.v2_3 import (
    foreground_coverage_hinge_loss,
    strict_positive_containment_loss,
)


def _mask(batch_size: int = 1) -> torch.Tensor:
    masks = torch.zeros(batch_size, 1, 16, 16)
    masks[:, :, 4:12, 4:12] = 1.0
    return masks


def test_v2_3_strict_containment_penalizes_outside_evidence():
    inside = torch.full((1, 4, 4), -1.0)
    inside[:, 1:3, 1:3] = 2.0
    outside = -inside
    assert strict_positive_containment_loss(
        [inside], _mask()
    ) < strict_positive_containment_loss([outside], _mask())


def test_v2_3_coverage_hinge_rewards_broad_foreground_not_a_hotspot():
    broad = torch.zeros(1, 4, 4)
    broad[:, 1:3, 1:3] = 1.0
    hotspot = torch.zeros(1, 4, 4)
    hotspot[:, 1, 1] = 1.0
    broad_loss = foreground_coverage_hinge_loss(
        [broad], _mask(), target=0.60
    )
    hotspot_loss = foreground_coverage_hinge_loss(
        [hotspot], _mask(), target=0.60
    )
    assert broad_loss.item() == 0.0
    assert hotspot_loss > broad_loss


def test_v2_3_losses_are_differentiable_and_ignore_invalid_masks():
    scores = torch.randn(2, 4, 4, requires_grad=True)
    valid_loss = strict_positive_containment_loss([scores], _mask(2))
    valid_loss = valid_loss + foreground_coverage_hinge_loss([scores], _mask(2))
    valid_loss.backward()
    assert scores.grad is not None and bool((scores.grad != 0).any())

    invalid_scores = torch.randn(2, 4, 4, requires_grad=True)
    invalid_masks = torch.zeros(2, 1, 16, 16)
    invalid_loss = strict_positive_containment_loss(
        [invalid_scores], invalid_masks
    ) + foreground_coverage_hinge_loss([invalid_scores], invalid_masks)
    invalid_loss.backward()
    assert invalid_loss.item() == 0.0
    assert invalid_scores.grad is not None


def test_v2_3_config_is_an_exact_v2_successor():
    root = Path(__file__).resolve().parents[1]
    v2 = yaml.safe_load((root / "configs/cub/resnet50_fcal_v2.yaml").read_text())
    candidate = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_3.yaml").read_text()
    )
    assert candidate["method"] == "fcal_v2_3"
    for key in ("data", "model", "training", "fcal", "fcal_v2"):
        assert candidate[key] == v2[key]
    assert set(candidate["fcal_v2_3"]) == {
        "strict_containment_weight",
        "foreground_coverage_weight",
        "foreground_coverage_target",
        "foreground_coverage_power",
    }


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


def test_v2_3_train_epoch_uses_original_input_and_one_forward():
    model = _SingleForwardEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(4, 3, 16, 16)
    original = images.clone()
    batch = {
        "image": images,
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": _mask(4),
    }
    config = {
        "method": "fcal_v2_3",
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
        "fcal_v2_3": {
            "strict_containment_weight": 0.025,
            "foreground_coverage_weight": 0.20,
            "foreground_coverage_target": 0.60,
            "foreground_coverage_power": 2.0,
        },
    }
    metrics = train_epoch(
        model, [batch], optimizer, config, epoch=0, device=torch.device("cpu")
    )
    assert model.forward_calls == 1
    assert len(model.forward_inputs) == 1
    assert torch.equal(model.forward_inputs[0], original)
    assert torch.equal(batch["image"], original)
    assert metrics["strict_containment"] >= 0.0
    assert metrics["foreground_coverage_hinge"] >= 0.0
    assert metrics["samples"] == 4
