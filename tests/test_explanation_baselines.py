from pathlib import Path

import torch
import yaml

from fcal_study.baseline_matrix import (
    ARCHITECTURES,
    build_config,
    config_filename,
    iter_supported,
    supported,
)
from fcal_study.explanation_baselines import (
    camal_losses,
    isnet_heatmap_loss,
    normalized_positive_map,
    right_reasons_loss,
)
from fcal_study.models import ISNetResNet50CAM


def test_rrr_penalizes_only_annotated_background_gradients():
    images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    score = images.flatten(1).sum(dim=1)
    logits = torch.stack((score, -score), dim=1)
    foreground = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loss = right_reasons_loss(logits, images, foreground)
    assert loss > 0
    loss.backward()
    assert images.grad is not None


def test_gain_map_normalizes_each_sample_independently():
    raw = torch.tensor([
        [[0.0, 1.0], [2.0, 3.0]],
        [[0.0, 10.0], [20.0, 30.0]],
    ])
    normalized = normalized_positive_map(raw)
    torch.testing.assert_close(normalized[0], normalized[1])


class _EvidenceModel:
    def __init__(self, evidence: torch.Tensor) -> None:
        self.evidence = evidence

    def raw_class_evidence(self, labels: torch.Tensor) -> list[torch.Tensor]:
        del labels
        return [self.evidence]


def test_camal_matches_beta_minus_alpha_and_is_scale_invariant():
    masks = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    evidence = torch.tensor([[[2.0, 0.0], [1.0, 0.0]]])
    labels = torch.tensor([0])
    loss, inside, outside = camal_losses(_EvidenceModel(evidence), labels, masks)
    scaled, _, _ = camal_losses(_EvidenceModel(10.0 * evidence), labels, masks)
    torch.testing.assert_close(loss, outside - inside)
    torch.testing.assert_close(loss, scaled)
    assert inside > outside
    assert loss < 0


def test_isnet_loss_prefers_foreground_relevance():
    masks = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    foreground = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    background = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    assert isnet_heatmap_loss(foreground, masks) < isnet_heatmap_loss(background, masks)


def test_isnet_resnet_loads_ce_state_and_produces_second_order_lrp():
    regular = ISNetResNet50CAM(num_classes=3, pretrained=False).eval()
    images = torch.randn(1, 3, 32, 32, requires_grad=True)
    logits = regular(images, capture_attention=True)
    relevance = regular.selective_lrp(images, logits)
    assert relevance.shape == images.shape
    relevance.square().mean().backward()
    assert regular.model.conv1.weight.grad is not None


def test_explanation_configs_match_shared_seed7_model_and_data():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root / "configs/cub/resnet50_ce.yaml").read_text(encoding="utf-8"))
    for name, method in (
        ("resnet50_rrr_seed7.yaml", "rrr"),
        ("resnet50_gain_seed7.yaml", "gain"),
        ("resnet50_isnet_seed7.yaml", "isnet"),
    ):
        config = yaml.safe_load((root / "configs/cub" / name).read_text(encoding="utf-8"))
        assert config["seed"] == 7
        assert config["method"] == method
        assert config["model"] == baseline["model"]
        assert config["data"] == baseline["data"]


def test_fixed_baseline_matrix_has_no_mislabeled_isnet_architectures():
    assert len(iter_supported()) == 19
    assert supported("isnet", "resnet50")
    for architecture in set(ARCHITECTURES) - {"resnet50"}:
        assert not supported("isnet", architecture)


def test_fixed_recipes_use_primary_source_defaults_and_shared_training_recipe():
    rrr = build_config("rrr", "vit-base-p16-avg")
    gain = build_config("gain", "resnet50")
    camal = build_config("camal", "efficientnet-b0")
    assert rrr["rrr"]["lambda"] == 1.0e-4
    assert gain["gain"]["sigma"] == 0.5
    assert gain["gain"]["omega"] == 8.0
    assert gain["gain"]["supervision_weight"] == 10.0
    assert camal["camal"]["coefficient"] == 1.0
    assert rrr["training"]["learning_rate"] == 5.0e-5
    assert camal["training"]["learning_rate"] == 2.0e-4


def test_checked_in_fixed_configs_match_the_generator():
    root = Path(__file__).resolve().parents[1] / "configs/cub/baseline_fixed"
    for method, architecture in iter_supported():
        actual = yaml.safe_load(
            (root / config_filename(method, architecture)).read_text(encoding="utf-8")
        )
        assert actual == build_config(method, architecture)
