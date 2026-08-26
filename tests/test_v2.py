from pathlib import Path

import torch
import yaml

from fcal_study.v2 import (
    detached_rms_normalize,
    distributed_foreground_support,
    resolution_aware_tolerance_pixels,
    support_aware_fcal_losses,
)


def test_v2_tolerance_is_half_an_evidence_cell():
    assert resolution_aware_tolerance_pixels((224, 224), (7, 7), 0.5) == 16
    assert resolution_aware_tolerance_pixels((224, 224), (14, 14), 0.5) == 8
    assert resolution_aware_tolerance_pixels((224, 224), (28, 28), 0.5) == 4


def test_v2_rms_normalization_is_scale_invariant_and_keeps_gradient():
    scores = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]], requires_grad=True)
    left = detached_rms_normalize(scores)
    right = detached_rms_normalize(7.0 * scores)
    assert torch.allclose(left, right, atol=1e-6)
    left.sum().backward()
    assert scores.grad is not None
    assert bool((scores.grad != 0).any())


def test_v2_support_requires_distributed_evidence_not_one_peak():
    foreground = torch.ones(1, 2, 2)
    peak = torch.tensor([[[10.0, -1.0], [-1.0, -1.0]]])
    distributed = torch.ones(1, 2, 2)
    peak_support = distributed_foreground_support(peak, foreground)
    distributed_support = distributed_foreground_support(distributed, foreground)
    assert distributed_support > peak_support


def test_v2_dead_map_has_live_distributed_support_gradient():
    scores = torch.full((1, 7, 7), -2.0, requires_grad=True)
    mask = torch.zeros(1, 1, 224, 224)
    mask[:, :, 64:160, 64:160] = 1.0
    outside, support = support_aware_fcal_losses(
        [scores], mask, support_temperature=1.0, support_margin=0.0,
        tolerance_cell_fraction=0.5,
    )
    assert outside.item() == 0.0
    support.backward()
    assert scores.grad is not None
    assert bool((scores.grad < 0).any())


def test_v2_configs_change_only_objective_within_architecture():
    root = Path(__file__).resolve().parents[1]
    pairs = [
        ("resnet50_ce.yaml", "resnet50_fcal_v2.yaml"),
        ("densenet121_ce.yaml", "densenet121_fcal_v2.yaml"),
        ("efficientnet_b0_ce.yaml", "efficientnet_b0_fcal_v2.yaml"),
        ("vit_tiny_avg_ce.yaml", "vit_tiny_avg_fcal_v2.yaml"),
        ("vit_small_avg_ce.yaml", "vit_small_avg_fcal_v2.yaml"),
        ("vit_base_avg_ce.yaml", "vit_base_avg_fcal_v2.yaml"),
    ]
    for baseline_name, v2_name in pairs:
        baseline = yaml.safe_load((root / "configs" / "cub" / baseline_name).read_text(encoding="utf-8"))
        v2 = yaml.safe_load((root / "configs" / "cub" / v2_name).read_text(encoding="utf-8"))
        assert baseline["data"] == v2["data"]
        assert baseline["model"] == v2["model"]
        baseline_training = {**baseline["training"], "save_epoch5_branch": False}
        assert baseline_training == v2["training"]
        assert v2["method"] == "fcal_v2"
        assert "background_training" not in v2["fcal_v2"]
