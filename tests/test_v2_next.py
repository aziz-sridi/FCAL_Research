from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fcal_study.models import DenseNet121CAM, EfficientNetB0CAM, ResNet50CAM
from fcal_study.train import train_epoch
from fcal_study.v2 import support_aware_fcal_losses
from fcal_study.v2_next import (
    buffered_region_masks,
    local_foreground_rival_margin_loss,
    native_regional_dominance_loss,
    persistent_seed_and_grow_loss,
    regional_classification_losses,
    softmin_foreground_rival_margin_loss,
    uncancelled_channel_containment_loss,
)


def _mask(batch_size: int = 1, size: int = 16) -> torch.Tensor:
    masks = torch.zeros(batch_size, 1, size, size)
    masks[:, :, size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 1.0
    return masks


def test_persistent_growth_backpropagates_through_front_and_trail():
    scores = torch.zeros(1, 5, 5, requires_grad=True)
    with torch.no_grad():
        scores[:, 2, 1:4] = 0.4
        scores[:, 2, 2] = 1.0
    result = persistent_seed_and_grow_loss(
        [scores], _mask(), front_target=0.8, retention_target=0.9,
    )
    loss = result.front + result.retention
    assert result.front > 0.0
    assert result.retention > 0.0
    loss.backward()
    assert scores.grad is not None and bool((scores.grad != 0).any())


def test_native_dominance_rewards_foreground_over_background():
    mask = _mask()
    aligned = torch.zeros(1, 4, 4)
    aligned[:, 1:3, 1:3] = 1.0
    reversed_map = 1.0 - aligned
    good = native_regional_dominance_loss([aligned], mask, margin=0.6)
    bad = native_regional_dominance_loss([reversed_map], mask, margin=0.6)
    assert good < bad


def test_resnet_regional_logits_reuse_existing_features():
    model = ResNet50CAM(num_classes=3, pretrained=False)
    model._features = torch.randn(2, 2048, 2, 2, requires_grad=True)
    region = torch.ones(2, 1, 8, 8)
    actual = model.regional_logits(region)
    expected = model.model.fc(model._features.mean(dim=(-2, -1)))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model._features.grad is not None


def test_resnet_background_feature_perturbation_reconstructs_clean_at_zero_sigma():
    model = ResNet50CAM(num_classes=3, pretrained=False)
    model._features = torch.randn(2, 2048, 4, 4, requires_grad=True)
    region = _mask(batch_size=2, size=16)
    actual = model.background_feature_perturbed_logits(region, sigma=0.0)
    expected = model.model.fc(model._features.mean(dim=(-2, -1)))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual.sum().backward()
    assert model._features.grad is not None


def test_densenet_regional_logits_reuse_existing_features():
    model = DenseNet121CAM(num_classes=3, pretrained=False)
    channels = model.model.classifier.in_features
    model._features = torch.randn(2, channels, 2, 2, requires_grad=True)
    region = torch.tensor(
        [[[[1.0, 0.0], [0.5, 0.0]]], [[[0.0, 1.0], [0.0, 1.0]]]]
    )
    actual = model.regional_logits(region)
    denominator = region.flatten(1).sum(dim=1, keepdim=True)
    pooled = (model._features * region).flatten(2).sum(dim=2) / denominator
    expected = model.model.classifier(pooled)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model._features.grad is not None


def test_efficientnet_regional_logits_reuse_forward_dropout_mask():
    model = EfficientNetB0CAM(num_classes=3, pretrained=False)
    channels = model.model.classifier[-1].in_features
    model._features = torch.randn(2, channels, 2, 2, requires_grad=True)
    model._dropout_scale = torch.stack(
        (torch.ones(channels), torch.arange(channels) % 2), dim=0
    )
    region = torch.tensor(
        [[[[1.0, 0.0], [0.5, 0.0]]], [[[0.0, 1.0], [0.0, 1.0]]]]
    )
    actual = model.regional_logits(region)
    denominator = region.flatten(1).sum(dim=1, keepdim=True)
    pooled = (model._features * region).flatten(2).sum(dim=2) / denominator
    expected = model.model.classifier[-1](pooled * model._dropout_scale)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model._features.grad is not None


def test_local_rival_margin_penalizes_weak_foreground_cells():
    target = torch.ones(1, 4, 4, requires_grad=True)
    rival = torch.zeros(1, 4, 4, requires_grad=True)
    with torch.no_grad():
        target[:, 1, 1] = -1.0
        rival[:, 1, 1] = 1.0
    loss = local_foreground_rival_margin_loss(
        [target], [[rival]], _mask(), lower_tail_fraction=0.25,
    )
    assert loss > 0.0
    loss.backward()
    assert target.grad is not None and bool((target.grad != 0).any())
    assert rival.grad is not None and bool((rival.grad != 0).any())


def test_softmin_rival_margin_rewards_distributed_target_evidence():
    target = torch.ones(1, 4, 4, requires_grad=True)
    rival = torch.zeros(1, 4, 4, requires_grad=True)
    good = softmin_foreground_rival_margin_loss(
        [target], [[rival]], _mask(),
    )
    bad = softmin_foreground_rival_margin_loss(
        [rival], [[target]], _mask(),
    )
    assert good < bad
    good.backward()
    assert target.grad is not None


def test_uncancelled_channel_containment_rewards_inside_evidence():
    inside = torch.zeros(1, 4, 4)
    inside[:, 1:3, 1:3] = 1.0
    outside = 1.0 - inside
    good = uncancelled_channel_containment_loss(
        [inside], _mask(), tolerance_cell_fraction=0.0,
    )
    bad = uncancelled_channel_containment_loss(
        [outside], _mask(), tolerance_cell_fraction=0.0,
    )
    assert good < bad


def test_channel_containment_can_reuse_exact_v2_background_geometry():
    scores = torch.rand(2, 4, 4)
    masks = _mask(2)
    _, _, backgrounds = support_aware_fcal_losses(
        [scores], masks, support_temperature=1.0, support_margin=0.0,
        tolerance_cell_fraction=0.5, maximum_tolerance_pixels=4,
        return_backgrounds=True,
    )
    direct = uncancelled_channel_containment_loss(
        [scores], masks, tolerance_cell_fraction=0.5,
        maximum_tolerance_pixels=4,
    )
    reused = uncancelled_channel_containment_loss(
        [scores], masks, tolerance_cell_fraction=0.5,
        maximum_tolerance_pixels=4, background_maps=backgrounds,
    )
    torch.testing.assert_close(reused, direct)


def test_resnet_positive_weight_evidence_prevents_channel_cancellation():
    model = ResNet50CAM(num_classes=2, pretrained=False)
    model._features = torch.ones(1, 2048, 2, 2)
    with torch.no_grad():
        model.model.fc.weight[0, :1024] = 1.0
        model.model.fc.weight[0, 1024:] = -1.0
    labels = torch.tensor([0])
    assert torch.allclose(model.raw_class_evidence(labels)[0], torch.zeros(1, 2, 2))
    assert bool((model.positive_weight_class_evidence(labels)[0] > 0.0).all())
    raw, positive = model.raw_and_positive_weight_class_evidence(labels)
    torch.testing.assert_close(raw[0], model.raw_class_evidence(labels)[0])
    torch.testing.assert_close(
        positive[0], model.positive_weight_class_evidence(labels)[0]
    )


class _SingleForwardRegionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.classifier = nn.Linear(4, 3)
        self.forward_calls = 0
        self.regional_calls = 0
        self.perturbation_calls = 0
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

    def background_feature_perturbed_logits(self, foreground_regions, sigma):
        self.perturbation_calls += 1
        foreground = F.adaptive_avg_pool2d(
            foreground_regions, self._map.shape[-2:]
        )
        background = 1.0 - foreground
        cells = float(self._map.shape[-2] * self._map.shape[-1])
        foreground_fraction = foreground.flatten(1).sum(1, keepdim=True) / cells
        background_fraction = 1.0 - foreground_fraction
        foreground_pooled = (
            (self._map * foreground).flatten(2).sum(2)
            / foreground.flatten(1).sum(1, keepdim=True)
        )
        background_pooled = (
            (self._map * background).flatten(2).sum(2)
            / background.flatten(1).sum(1, keepdim=True)
        )
        perturbed = (
            foreground_fraction * foreground_pooled
            + background_fraction
            * (background_pooled + float(sigma) * torch.randn_like(background_pooled))
        )
        return self.classifier(perturbed)

    def positive_weight_class_evidence(self, labels):
        weights = F.relu(self.classifier.weight[labels])
        return [torch.einsum("bc,bchw->bhw", weights, self._map)]

    def raw_and_positive_weight_class_evidence(self, labels):
        return (
            self.raw_class_evidence(labels),
            self.positive_weight_class_evidence(labels),
        )


def _config(method: str) -> dict:
    weights = {
        "fcal_v2_5": (0.1, 0.25, 0.0, 0.0, 0.0),
        "fcal_v2_6": (0.0, 0.0, 0.08, 0.0, 0.0),
        "fcal_v2_7": (0.0, 0.0, 0.0, 0.02, 0.0),
        "fcal_v2_8": (0.0, 0.0, 0.0, 0.01, 0.001),
        "fcal_v2_9": (0.05, 0.125, 0.04, 0.005, 0.0005),
        "fcal_v2_7a": (0.0, 0.0, 0.0, 0.02, 0.0),
        "fcal_v2_7b": (0.0, 0.0, 0.0, 0.02, 0.0),
        "fcal_v2_7d": (0.0, 0.0, 0.0, 0.02, 0.0),
        "fcal_v2_7_prime": (0.0, 0.0, 0.0, 0.0, 0.0),
        "fcal_v2_feature_consistency": (0.0, 0.0, 0.0, 0.0, 0.0),
    }[method]
    return {
        "method": method,
        "data": {"accumulation_steps": 1},
        "training": {
            "channels_last": False, "amp": "off", "grad_clip_norm": 1.0,
        },
        "fcal": {
            "alpha": 0.15, "lambda_min": 0.05, "lambda_max": 0.30,
            "epsilon": 1e-8,
        },
        "fcal_v2": {
            "warmup_epochs": 0, "ramp_epochs": 1,
            "support_temperature": 1.0, "support_margin": 0.0,
            "support_weight": 0.05, "tolerance_cell_fraction": 0.5,
            "maximum_tolerance_pixels": 4, "epsilon": 1e-8,
        },
        "fcal_v2_next": {
            "seed_front_weight": weights[0],
            "seed_retention_weight": weights[1],
            "native_dominance_weight": weights[2],
            "foreground_region_weight": weights[3],
            "background_region_weight": weights[4],
            "acceptance_threshold": 0.35, "front_target": 0.8,
            "retention_target": 0.9, "evidence_temperature": 1.0,
            "growth_power": 2.0, "foreground_threshold": 0.5,
            "dominance_margin": 0.6, "dominance_power": 2.0,
            "foreground_logit_margin": 0.0,
            "maximum_background_logit_margin": (
                0.0 if method == "fcal_v2_7_prime" else -2.0
            ),
            "regional_temperature": 1.0,
            "local_foreground_rival_weight": (
                0.006 if method in {"fcal_v2_7a", "fcal_v2_7d"} else 0.0
            ),
            "local_top_k": 2,
            "local_lower_tail_fraction": 0.25,
            "local_margin": 0.0,
            "local_evidence_temperature": 1.0,
            "local_loss_temperature": 0.25,
            "local_selection": (
                "softmin" if method == "fcal_v2_7d" else "hard_tail"
            ),
            "local_softmin_temperature": 0.25,
            "channel_containment_weight": (
                0.0075 if method == "fcal_v2_7b" else 0.0
            ),
            "adaptive_regional_weight": method == "fcal_v2_7_prime",
            "regional_buffer_cell_fraction": 0.5,
            "regional_buffer_minimum_pixels": 0,
            "regional_buffer_maximum_pixels": 4,
            "feature_consistency_weight": (
                0.02 if method == "fcal_v2_feature_consistency" else 0.0
            ),
            "background_feature_noise_sigma": 0.1,
        },
    }


@pytest.mark.parametrize(
    ("method", "expected_regional_calls"),
    [
        ("fcal_v2_5", 0), ("fcal_v2_6", 0), ("fcal_v2_7", 1),
        ("fcal_v2_8", 2), ("fcal_v2_9", 2),
        ("fcal_v2_7a", 1), ("fcal_v2_7b", 1),
        ("fcal_v2_7d", 1),
        ("fcal_v2_7_prime", 2),
    ],
)
def test_next_versions_use_original_input_and_one_forward(
    method, expected_regional_calls,
):
    model = _SingleForwardRegionalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(4, 3, 16, 16)
    original = images.clone()
    batch = {
        "image": images, "label": torch.tensor([0, 1, 2, 0]),
        "mask": _mask(4),
    }
    metrics = train_epoch(
        model, [batch], optimizer, _config(method), 0, torch.device("cpu")
    )
    assert model.forward_calls == 1
    assert model.regional_calls == expected_regional_calls
    assert torch.equal(model.forward_inputs[0], original)
    assert torch.equal(batch["image"], original)
    assert metrics["samples"] == 4


def test_regional_classification_can_skip_background_pooling():
    model = _SingleForwardRegionalModel()
    images = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 1])
    model(images)
    result = regional_classification_losses(
        model, labels, _mask(2), compute_background=False,
    )
    assert model.regional_calls == 1
    assert result.foreground >= 0.0
    assert result.background == 0.0


def test_buffered_regions_are_disjoint_and_exclude_boundary_band():
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 2:7, 2:7] = 1.0
    foreground, background = buffered_region_masks(mask, radius_pixels=1)
    assert torch.equal(foreground[:, :, 3:6, 3:6], torch.ones(1, 1, 3, 3))
    assert foreground.sum() == 9
    assert background.sum() == 32
    assert not bool(((foreground > 0) & (background > 0)).any())
    assert (1.0 - foreground - background).sum() == 40


def test_prime_uses_adaptive_symmetric_regional_objective():
    model = _SingleForwardRegionalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "image": torch.randn(4, 3, 16, 16),
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": _mask(4),
    }
    metrics = train_epoch(
        model, [batch], optimizer, _config("fcal_v2_7_prime"), 0,
        torch.device("cpu"),
    )
    assert model.forward_calls == 1
    assert model.regional_calls == 2
    assert metrics["foreground_region_classification"] > 0.0
    assert metrics["background_region_suppression"] > 0.0
    assert 0.05 <= metrics["regional_gamma"] <= 0.30
    assert metrics["weighted_regional"] > 0.0


def test_feature_consistency_uses_one_backbone_forward_and_one_perturbation():
    model = _SingleForwardRegionalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(4, 3, 16, 16)
    batch = {
        "image": images,
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": _mask(4),
    }
    metrics = train_epoch(
        model, [batch], optimizer, _config("fcal_v2_feature_consistency"), 0,
        torch.device("cpu"),
    )
    assert model.forward_calls == 1
    assert model.perturbation_calls == 1
    assert model.regional_calls == 0
    assert metrics["background_feature_consistency"] >= 0.0
    assert metrics["weighted_feature_consistency"] >= 0.0


def test_five_configs_are_exact_v2_training_successors():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2.yaml").read_text()
    )
    configs = [
        path
        for path in sorted(
            (root / "configs/cub").glob("resnet50_fcal_v2_[5-9]_*.yaml")
        )
        if yaml.safe_load(path.read_text())["seed"] == 42
        and yaml.safe_load(path.read_text())["method"] in {
            "fcal_v2_5", "fcal_v2_6", "fcal_v2_7", "fcal_v2_8", "fcal_v2_9",
        }
    ]
    assert len(configs) == 5
    for index, path in enumerate(configs, start=5):
        candidate = yaml.safe_load(path.read_text())
        assert candidate["method"] == f"fcal_v2_{index}"
        for key in ("data", "model", "training", "fcal", "fcal_v2"):
            assert candidate[key] == baseline[key]
        assert set(candidate["fcal_v2_next"]) == {
            "seed_front_weight", "seed_retention_weight",
            "native_dominance_weight", "foreground_region_weight",
            "background_region_weight", "acceptance_threshold",
            "front_target", "retention_target", "evidence_temperature",
            "growth_power", "foreground_threshold", "dominance_margin",
            "dominance_power", "foreground_logit_margin",
            "maximum_background_logit_margin", "regional_temperature",
        }


def test_cross_cnn_v2_7_configs_are_exact_frozen_successors():
    root = Path(__file__).resolve().parents[1]
    reference = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_7_foreground_pool.yaml").read_text()
    )
    cases = (
        ("densenet121", "densenet121_fcal_v2.yaml"),
        ("efficientnet-b0", "efficientnet_b0_fcal_v2.yaml"),
    )
    for slug, anchor_name in cases:
        anchor = yaml.safe_load((root / "configs/cub" / anchor_name).read_text())
        candidate = yaml.safe_load(
            (root / "configs/cub" / f"{slug.replace('-', '_')}_fcal_v2_7.yaml").read_text()
        )
        assert candidate["method"] == "fcal_v2_7"
        for key in ("data", "model", "training", "fcal", "fcal_v2"):
            assert candidate[key] == anchor[key]
        assert candidate["fcal_v2_next"] == reference["fcal_v2_next"]


def test_seed7_v2_7_config_changes_only_replication_identity():
    root = Path(__file__).resolve().parents[1]
    seed42 = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_7_foreground_pool.yaml").read_text()
    )
    seed7 = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_7_seed7.yaml").read_text()
    )
    assert seed7["run_id"] == (
        "multiseed/cub/resnet50-fcal-v2-7-foreground-pool-seed7"
    )
    assert seed7["seed"] == 7
    for config in (seed42, seed7):
        config.pop("run_id")
        config.pop("seed")
    assert seed7 == seed42


def test_v2_7_prime_seed_configs_are_frozen_symmetric_successors():
    root = Path(__file__).resolve().parents[1]
    v2_7 = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_7_foreground_pool.yaml").read_text()
    )
    configs = [
        yaml.safe_load(
            (root / f"configs/cub/resnet50_fcal_v2_7_prime_seed{seed}.yaml").read_text()
        )
        for seed in (42, 7)
    ]
    for candidate, seed in zip(configs, (42, 7), strict=True):
        assert candidate["method"] == "fcal_v2_7_prime"
        assert candidate["seed"] == seed
        for key in ("data", "model", "training", "fcal", "fcal_v2"):
            assert candidate[key] == v2_7[key]
        settings = candidate["fcal_v2_next"]
        assert settings["foreground_region_weight"] == 0.0
        assert settings["background_region_weight"] == 0.0
        assert settings["adaptive_regional_weight"] is True
        assert settings["foreground_logit_margin"] == 0.0
        assert settings["maximum_background_logit_margin"] == 0.0
        assert settings["regional_buffer_cell_fraction"] == 0.5
    first, second = configs
    for config in (first, second):
        config.pop("run_id")
        config.pop("seed")
    assert first == second


def test_feature_consistency_seed_configs_add_only_the_frozen_proxy():
    root = Path(__file__).resolve().parents[1]
    v2 = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2.yaml").read_text()
    )
    configs = [
        yaml.safe_load(
            (
                root
                / f"configs/cub/resnet50_fcal_v2_feature_consistency_seed{seed}.yaml"
            ).read_text()
        )
        for seed in (42, 7)
    ]
    for candidate, seed in zip(configs, (42, 7), strict=True):
        assert candidate["method"] == "fcal_v2_feature_consistency"
        assert candidate["seed"] == seed
        for key in ("data", "model", "training", "fcal", "fcal_v2"):
            assert candidate[key] == v2[key]
        assert candidate["fcal_v2_next"] == {
            "feature_consistency_weight": 0.02,
            "background_feature_noise_sigma": 0.10,
        }
    first, second = configs
    for config in (first, second):
        config.pop("run_id")
        config.pop("seed")
    assert first == second


def test_v2_7_refinement_configs_change_only_the_declared_losses():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_7_foreground_pool.yaml").read_text()
    )
    paths = [
        root / "configs/cub/resnet50_fcal_v2_7a_local_rival.yaml",
        root / "configs/cub/resnet50_fcal_v2_7b_channel_containment.yaml",
    ]
    for method, path in zip(("fcal_v2_7a", "fcal_v2_7b"), paths, strict=True):
        candidate = yaml.safe_load(path.read_text())
        assert candidate["method"] == method
        for key in ("data", "model", "training", "fcal", "fcal_v2"):
            assert candidate[key] == baseline[key]
        for key, value in baseline["fcal_v2_next"].items():
            assert candidate["fcal_v2_next"][key] == value
    assert yaml.safe_load(paths[0].read_text())["fcal_v2_next"][
        "local_foreground_rival_weight"
    ] == 0.006
    assert yaml.safe_load(paths[1].read_text())["fcal_v2_next"][
        "channel_containment_weight"
    ] == 0.0075

    softmin = yaml.safe_load(
        (root / "configs/cub/resnet50_fcal_v2_7d_softmin_rival.yaml").read_text()
    )
    assert softmin["method"] == "fcal_v2_7d"
    for key in ("data", "model", "training", "fcal", "fcal_v2"):
        assert softmin[key] == baseline[key]
    assert softmin["fcal_v2_next"]["foreground_region_weight"] == 0.02
    assert softmin["fcal_v2_next"]["local_foreground_rival_weight"] == 0.01
    assert softmin["fcal_v2_next"]["local_top_k"] == 1
    assert softmin["fcal_v2_next"]["local_selection"] == "softmin"
