from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from fcal_study.train import train_epoch
from fcal_study.v4 import (
    background_margin_sensitivity,
    mask_alignment_loss,
    regional_dominance_loss,
    strict_positive_containment,
)


def test_v4_strict_containment_distinguishes_inside_and_outside_evidence():
    masks = torch.zeros(1, 1, 32, 32)
    masks[:, :, 8:24, 8:24] = 1.0
    inside = torch.full((1, 8, 8), -1.0)
    inside[:, 2:6, 2:6] = 2.0
    outside = -inside
    inside_loss = strict_positive_containment([inside], masks)
    outside_loss = strict_positive_containment([outside], masks)
    assert inside_loss < outside_loss


def test_v4_alignment_rewards_broad_positive_foreground_evidence():
    masks = torch.zeros(1, 1, 32, 32)
    masks[:, :, 8:24, 8:24] = 1.0
    aligned = torch.full((1, 8, 8), -2.0)
    aligned[:, 2:6, 2:6] = 2.0
    reversed_map = -aligned
    assert mask_alignment_loss([aligned], masks) < mask_alignment_loss(
        [reversed_map], masks
    )


def test_v4_regional_dominance_is_one_sided_and_rewards_correct_ordering():
    masks = torch.zeros(1, 1, 32, 32)
    masks[:, :, 8:24, 8:24] = 1.0
    aligned = torch.full((1, 8, 8), -1.0)
    aligned[:, 2:6, 2:6] = 2.0
    reversed_map = -aligned
    aligned_loss = regional_dominance_loss([aligned], masks, margin=0.5)
    reversed_loss = regional_dominance_loss([reversed_map], masks, margin=0.5)
    assert aligned_loss.item() == 0.0
    assert reversed_loss > aligned_loss


def test_v4_background_sensitivity_is_zero_for_foreground_only_margin():
    masks = torch.zeros(2, 1, 8, 8)
    masks[:, :, 2:6, 2:6] = 1.0
    labels = torch.zeros(2, dtype=torch.long)
    foreground_images = torch.randn(2, 3, 8, 8, requires_grad=True)
    foreground_score = (foreground_images * masks).sum(dim=(1, 2, 3))
    foreground_logits = torch.stack((foreground_score, -foreground_score), dim=1)
    foreground_loss = background_margin_sensitivity(
        foreground_logits, labels, foreground_images, masks
    )
    background_images = torch.randn(2, 3, 8, 8, requires_grad=True)
    background_score = (background_images * (1.0 - masks)).sum(dim=(1, 2, 3))
    background_logits = torch.stack((background_score, -background_score), dim=1)
    background_loss = background_margin_sensitivity(
        background_logits, labels, background_images, masks
    )
    assert foreground_loss.item() == 0.0
    assert background_loss > foreground_loss


class _TinyEvidenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.classifier = nn.Linear(4, 3)
        self._map = None

    def forward(self, images, capture_attention=False):
        del capture_attention
        self._map = F.relu(self.features(images))
        return self.classifier(self._map.mean(dim=(-2, -1)))

    def raw_class_evidence(self, labels):
        weights = self.classifier.weight[labels]
        return [torch.einsum("bc,bchw->bhw", weights, self._map)]


def _config(intervention_enabled: bool) -> dict:
    return {
        "method": "fcal_v4",
        "data": {"accumulation_steps": 1},
        "training": {"channels_last": False, "amp": "off", "grad_clip_norm": 1.0},
        "fcal": {
            "alpha": 0.15, "lambda_min": 0.05, "lambda_max": 0.30,
            "epsilon": 1e-8,
        },
        "fcal_v2": {
            "warmup_epochs": 0, "ramp_epochs": 1, "support_temperature": 1.0,
            "support_margin": 0.0, "support_weight": 0.05,
            "tolerance_cell_fraction": 0.5, "maximum_tolerance_pixels": 4,
            "epsilon": 1e-8,
        },
        "fcal_v4": {
            "strict_containment_weight": 0.025,
            "mask_alignment_weight": 0.020,
            "alignment_temperature": 1.0,
            "regional_dominance_weight": 0.0,
            "regional_dominance_margin": 0.6,
            "regional_dominance_power": 2.0,
            "background_sensitivity_weight": 0.005,
            "intervention": {
                "enabled": intervention_enabled,
                "counterfactual_ce_weight": 0.25,
                "prediction_consistency_weight": 0.05,
                "evidence_consistency_weight": 0.025,
                "prediction_temperature": 2.0,
                "evidence_consistency_temperature": 1.0,
                "boundary_guard_pixels": 1,
                "foreground_intervention_fraction": 0.5,
                "foreground_noise_scale": 0.2,
            },
        },
    }


def _batch() -> dict:
    masks = torch.zeros(4, 1, 16, 16)
    masks[:, :, 4:12, 4:12] = 1.0
    return {
        "image": torch.randn(4, 3, 16, 16),
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": masks,
    }


def test_v4_pure_train_epoch_uses_no_manipulated_second_view():
    model = _TinyEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = train_epoch(
        model, [_batch()], optimizer, _config(False), 0, torch.device("cpu")
    )
    assert metrics["counterfactual_ce"] == 0.0
    assert metrics["foreground_intervention_fraction"] == 0.0
    assert metrics["strict_containment"] > 0.0
    assert metrics["mask_alignment"] > 0.0
    assert metrics["regional_dominance"] == 0.0
    assert metrics["background_sensitivity"] > 0.0


def test_v4_interventional_mode_is_explicitly_optional():
    config = deepcopy(_config(False))
    config["fcal_v4"]["intervention"]["enabled"] = True
    model = _TinyEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = train_epoch(
        model, [_batch()], optimizer, config, 0, torch.device("cpu")
    )
    assert metrics["counterfactual_ce"] > 0.0
    assert metrics["foreground_intervention_fraction"] == 0.5
