from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fcal_study.train import train_epoch
from fcal_study.v3 import (
    background_counterfactual,
    balanced_regional_counterfactual,
    counterfactual_support_losses,
    different_label_donors,
    foreground_corruption,
    foreground_evidence_consistency,
    prediction_consistency_loss,
    strict_shell_containment,
)


def test_v3_donors_avoid_same_class_when_a_derangement_exists():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    generator = torch.Generator().manual_seed(7)
    donors = different_label_donors(labels, generator)
    assert sorted(donors.tolist()) == list(range(len(labels)))
    assert bool((labels[donors] != labels).all())


def test_v3_counterfactual_preserves_foreground_exactly_and_changes_background():
    images = torch.stack([
        torch.full((3, 12, 12), 0.1),
        torch.full((3, 12, 12), 0.8),
    ])
    masks = torch.zeros(2, 1, 12, 12)
    masks[:, :, 4:8, 4:8] = 1.0
    images[0, :, 4:8, 4:8] = 0.3
    images[1, :, 4:8, 4:8] = 0.6
    labels = torch.tensor([0, 1])
    changed, donors, mismatch = background_counterfactual(
        images, masks, labels, boundary_guard_pixels=1,
        generator=torch.Generator().manual_seed(3),
    )
    foreground_error = ((changed - images).abs() * masks).max()
    background_change = ((changed - images).abs() * (1.0 - masks)).sum()
    assert foreground_error.item() == 0.0
    assert background_change.item() > 0.0
    assert donors.tolist() == [1, 0]
    assert mismatch.item() == 1.0


def test_v3_foreground_corruption_preserves_background_exactly():
    images = torch.randn(4, 3, 12, 12)
    masks = torch.zeros(4, 1, 12, 12)
    masks[:, :, 3:9, 3:9] = 1.0
    changed = foreground_corruption(
        images,
        masks,
        generator=torch.Generator().manual_seed(11),
    )
    background_error = ((changed - images).abs() * (1.0 - masks)).max()
    foreground_change = ((changed - images).abs() * masks).sum()
    assert background_error.item() == 0.0
    assert foreground_change.item() > 0.0


def test_v3_balanced_view_uses_one_half_foreground_interventions():
    images = torch.randn(8, 3, 12, 12)
    masks = torch.zeros(8, 1, 12, 12)
    masks[:, :, 3:9, 3:9] = 1.0
    labels = torch.arange(8)
    changed, background_samples, mismatch = balanced_regional_counterfactual(
        images,
        masks,
        labels,
        foreground_fraction=0.5,
        boundary_guard_pixels=1,
        generator=torch.Generator().manual_seed(13),
    )
    assert changed.shape == images.shape
    assert int(background_samples.sum()) == 4
    assert mismatch.item() == 0.5


def test_v3_prediction_consistency_is_zero_for_identical_logits_and_trains_student():
    clean = torch.tensor([[2.0, -1.0], [-0.5, 0.5]], requires_grad=True)
    student = clean.detach().clone().requires_grad_(True)
    identical = prediction_consistency_loss(clean, student, temperature=2.0)
    assert identical.item() < 1e-7
    shifted = torch.zeros_like(student, requires_grad=True)
    loss = prediction_consistency_loss(clean, shifted, temperature=2.0)
    loss.backward()
    assert loss.item() > 0.0
    assert shifted.grad is not None
    assert clean.grad is None


def test_v3_prediction_consistency_can_select_only_background_interventions():
    clean = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
    changed = clean.clone()
    changed[1] = torch.tensor([-2.0, 1.0])
    selected = prediction_consistency_loss(
        clean, changed, valid_samples=torch.tensor([True, False])
    )
    assert selected.item() < 1e-7


def test_v3_strict_shell_is_zero_without_tolerance_and_live_with_tolerance():
    scores = torch.ones(2, 7, 7, requires_grad=True)
    mask = torch.zeros(2, 1, 224, 224)
    mask[:, :, 64:160, 64:160] = 1.0
    no_shell = strict_shell_containment(
        [scores], mask, tolerance_cell_fraction=0.0
    )
    shell = strict_shell_containment(
        [scores], mask, tolerance_cell_fraction=0.5
    )
    shell.backward()
    assert no_shell.item() == 0.0
    assert shell.item() > 0.0
    assert scores.grad is not None and bool((scores.grad != 0).any())


def test_v3_foreground_evidence_consistency_is_scale_free_and_live():
    clean = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]], requires_grad=True)
    scaled = (7.0 * clean.detach()).requires_grad_(True)
    mask = torch.ones(1, 1, 8, 8)
    scale_free = foreground_evidence_consistency([clean], [scaled], mask)
    assert scale_free.item() < 1e-7
    changed = torch.tensor([[[-2.0, 2.0], [-1.0, 1.0]]], requires_grad=True)
    loss = foreground_evidence_consistency([clean], [changed], mask)
    loss.backward()
    assert loss.item() > 0.0
    assert clean.grad is not None and bool((clean.grad != 0).any())
    assert changed.grad is not None and bool((changed.grad != 0).any())


def test_v3_evidence_consistency_can_select_only_background_interventions():
    clean = torch.randn(2, 3, 3)
    changed = clean.clone()
    changed[1] = -changed[1]
    mask = torch.ones(2, 1, 9, 9)
    selected = foreground_evidence_consistency(
        [clean], [changed], mask, valid_samples=torch.tensor([True, False])
    )
    assert selected.item() == 0.0


def test_v3_dual_view_loss_retains_v2_terms_and_has_gradients():
    clean = torch.randn(2, 7, 7, requires_grad=True)
    counterfactual = torch.randn(2, 7, 7, requires_grad=True)
    mask = torch.zeros(2, 1, 224, 224)
    mask[:, :, 48:176, 64:160] = 1.0
    outside, support, consistency = counterfactual_support_losses(
        [clean], [counterfactual], mask,
        support_temperature=1.0,
        support_margin=0.0,
        tolerance_cell_fraction=0.5,
    )
    total = outside + support + consistency
    total.backward()
    assert all(torch.isfinite(item) for item in (outside, support, consistency))
    assert clean.grad is not None and bool((clean.grad != 0).any())
    assert counterfactual.grad is not None and bool((counterfactual.grad != 0).any())


def test_v3_configs_are_strict_v2_successors_on_all_cub_architectures():
    root = Path(__file__).resolve().parents[1]
    stems = (
        "resnet50", "densenet121", "efficientnet_b0",
        "vit_tiny_avg", "vit_small_avg", "vit_base_avg",
    )
    expected_v3_keys = {
        "counterfactual_ce_weight", "prediction_consistency_weight",
        "evidence_consistency_weight", "prediction_temperature",
        "evidence_consistency_temperature", "boundary_guard_pixels",
    }
    for stem in stems:
        v2 = yaml.safe_load((root / f"configs/cub/{stem}_fcal_v2.yaml").read_text())
        v3 = yaml.safe_load((root / f"configs/cub/{stem}_fcal_v3.yaml").read_text())
        assert v3["method"] == "fcal_v3"
        assert v3["data"] == v2["data"]
        assert v3["model"] == v2["model"]
        assert v3["training"] == v2["training"]
        assert v3["fcal"] == v2["fcal"]
        assert v3["fcal_v2"] == v2["fcal_v2"]
        assert set(v3["fcal_v3"]) == expected_v3_keys


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


def test_v3_train_epoch_exercises_complete_objective_on_cpu():
    model = _TinyEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "image": torch.randn(4, 3, 16, 16),
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": torch.zeros(4, 1, 16, 16),
    }
    batch["mask"][:, :, 4:12, 4:12] = 1.0
    config = {
        "method": "fcal_v3",
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
        "fcal_v3": {
            "counterfactual_ce_weight": 0.5,
            "prediction_consistency_weight": 0.1,
            "evidence_consistency_weight": 0.05,
            "prediction_temperature": 2.0,
            "evidence_consistency_temperature": 1.0,
            "boundary_guard_pixels": 1,
            "foreground_intervention_fraction": 0.5,
            "foreground_noise_scale": 0.2,
            "route_consistency_to_background": True,
            "strict_shell_weight": 0.025,
        },
    }
    metrics = train_epoch(
        model, [batch], optimizer, config, epoch=0, device=torch.device("cpu")
    )
    assert metrics["samples"] == 4
    assert metrics["counterfactual_ce"] > 0.0
    assert metrics["donor_mismatch_fraction"] == 0.5
    assert metrics["foreground_intervention_fraction"] == 0.5
    assert metrics["strict_shell"] > 0.0
    assert metrics["total"] > 0.0
