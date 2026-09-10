from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fcal_study.train import train_epoch
from fcal_study.v2_4 import seed_and_grow_foreground_loss


def _mask(batch_size: int = 1, size: int = 20) -> torch.Tensor:
    masks = torch.zeros(batch_size, 1, size, size)
    masks[:, :, size // 5: 4 * size // 5, size // 5: 4 * size // 5] = 1.0
    return masks


def test_seed_and_grow_pushes_the_next_connected_foreground_ring():
    sparse = torch.zeros(1, 5, 5, requires_grad=True)
    with torch.no_grad():
        sparse[:, 2, 2] = 1.0
    loss = seed_and_grow_foreground_loss([sparse], _mask())
    assert loss > 0.0
    loss.backward()
    assert sparse.grad is not None and bool((sparse.grad != 0).any())


def test_seed_and_grow_stops_after_the_connected_foreground_is_active():
    broad = torch.zeros(1, 5, 5)
    broad[:, 1:4, 1:4] = 1.0
    loss = seed_and_grow_foreground_loss([broad], _mask())
    assert loss.item() == 0.0


def test_seed_and_grow_rejects_invalid_temperature():
    scores = torch.zeros(1, 5, 5)
    try:
        seed_and_grow_foreground_loss(
            [scores], _mask(), evidence_temperature=0.0
        )
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("expected inconsistent thresholds to fail")


def test_seed_and_grow_invalid_masks_return_differentiable_zero():
    scores = torch.randn(2, 5, 5, requires_grad=True)
    loss = seed_and_grow_foreground_loss(
        [scores], torch.zeros(2, 1, 20, 20)
    )
    loss.backward()
    assert loss.item() == 0.0
    assert scores.grad is not None


def test_v2_4_config_is_an_exact_v2_successor():
    root = Path(__file__).resolve().parents[1]
    v2 = yaml.safe_load((root / "configs/cub/resnet50_fcal_v2.yaml").read_text())
    candidate = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_4_seed_grow.yaml").read_text()
    )
    assert candidate["method"] == "fcal_v2_4"
    for key in ("data", "model", "training", "fcal", "fcal_v2"):
        assert candidate[key] == v2[key]
    assert set(candidate["fcal_v2_4"]) == {
        "strict_containment_weight",
        "seed_grow_weight",
        "acceptance_threshold",
        "growth_target",
        "evidence_temperature",
        "growth_power",
        "foreground_threshold",
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


def test_v2_4_train_epoch_uses_original_input_and_one_forward():
    model = _SingleForwardEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(4, 3, 16, 16)
    original = images.clone()
    batch = {
        "image": images,
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": _mask(4, 16),
    }
    config = {
        "method": "fcal_v2_4",
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
        "fcal_v2_4": {
            "strict_containment_weight": 0.025,
            "seed_grow_weight": 0.05,
            "acceptance_threshold": 0.35,
            "growth_target": 0.65,
            "evidence_temperature": 1.0,
            "growth_power": 2.0,
            "foreground_threshold": 0.50,
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
    assert metrics["seed_grow"] >= 0.0
    assert metrics["samples"] == 4
