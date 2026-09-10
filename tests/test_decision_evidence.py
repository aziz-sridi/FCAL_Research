from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fcal_study.decision_evidence import (
    centered_class_weights,
    class_mean_evidence,
    v2_training_evidence,
    validate_evidence_reference,
)
from fcal_study.models import (
    DenseNet121CAM,
    EfficientNetB0CAM,
    ResNet50CAM,
    TimmViTAvgPoolCAM,
)
from fcal_study.train import train_epoch
from fcal_study.v2 import support_aware_fcal_losses


def test_common_classifier_shift_changes_raw_loss_but_not_predictions_or_repair():
    features = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    weight = torch.tensor([[1., 1.], [-1., -1.]], requires_grad=True)
    labels = torch.tensor([0])
    mask = torch.tensor([[[[1., 0.]]]])
    shift = torch.tensor([[20., 0.]])

    def loss(w, centered):
        selected = centered_class_weights(w, labels) if centered else w[labels]
        evidence = torch.einsum("bc,bchw->bhw", selected, features)
        return support_aware_fcal_losses([evidence], mask, 1., 0., 0.)

    original_probs = F.linear(features.mean((-2, -1)), weight).softmax(-1)
    shifted_probs = F.linear(features.mean((-2, -1)), weight + shift).softmax(-1)
    torch.testing.assert_close(original_probs, shifted_probs)
    assert loss(weight + shift, False)[0] < loss(weight, False)[0] / 10
    for before, after in zip(loss(weight, True), loss(weight + shift, True)):
        torch.testing.assert_close(before, after)
    sum(loss(weight, True)).backward()
    torch.testing.assert_close(weight.grad.sum(0), torch.zeros(2), atol=1e-7, rtol=0)


@pytest.mark.parametrize("kind", ["resnet", "dense", "efficient", "vit"])
def test_adapter_decomposes_centered_logits_and_backpropagates(kind):
    torch.manual_seed(73)
    head = nn.Linear(6, 4)
    labels = torch.tensor([1, 3])
    features = torch.randn(2, 6, 2, 2, requires_grad=True)
    # Exercise adapter methods without downloading weights or running backbones.
    classes = {"resnet": ResNet50CAM, "dense": DenseNet121CAM,
               "efficient": EfficientNetB0CAM, "vit": TimmViTAvgPoolCAM}
    model = classes[kind].__new__(classes[kind])
    nn.Module.__init__(model)
    model._features = features
    scale = torch.ones(2, 6)
    if kind == "resnet":
        model.model = SimpleNamespace(fc=head)
    elif kind == "dense":
        model.model = SimpleNamespace(classifier=head)
    elif kind == "efficient":
        scale[:, ::2] = 0
        scale[:, 1::2] = 2
        model._dropout_scale = scale
        model.model = SimpleNamespace(classifier=[nn.Dropout(), head])
    else:
        model._tokens = torch.cat((torch.zeros(2, 1, 6), features.flatten(2).transpose(1, 2)), 1)
        model.model = SimpleNamespace(head=head, head_drop=nn.Identity(), num_prefix_tokens=1)
    logits = head(features.mean((-2, -1)) * scale)
    evidence = class_mean_evidence(model, labels)[0]
    reconstructed = evidence.mean((-2, -1)) + head.bias[labels] - head.bias.mean()
    torch.testing.assert_close(reconstructed, logits.gather(1, labels[:, None])[:, 0] - logits.mean(1))
    evidence.square().sum().backward()
    assert features.grad.abs().sum() > 0
    assert head.weight.grad.abs().sum() > 0
    torch.testing.assert_close(head.weight.grad.sum(0), torch.zeros(6), atol=1e-6, rtol=0)


def test_unsupported_model_and_invalid_config_fail_explicitly():
    with pytest.raises(TypeError, match="GAP CNNs"):
        class_mean_evidence(nn.Linear(4, 2), torch.tensor([0]))
    with pytest.raises(ValueError, match="target or class_mean"):
        validate_evidence_reference({"method": "fcal_v2", "fcal_v2": {"evidence_reference": "typo"}})
    with pytest.raises(ValueError, match="only for method"):
        validate_evidence_reference({"method": "fcal_v2_plus", "fcal_v2": {"evidence_reference": "class_mean"}})


def test_default_evidence_is_exact_legacy_path():
    sentinel = [torch.randn(2, 7, 7)]
    model = SimpleNamespace(raw_class_evidence=lambda labels: sentinel)
    reference = validate_evidence_reference({"method": "fcal_v2"})
    assert v2_training_evidence(model, torch.tensor([0, 1]), reference) is sentinel


class TinyGAP(ResNet50CAM):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(3, 6, 1)
        self.model = nn.Module()
        self.model.fc = nn.Linear(6, 3)
        self.calls = 0

    def forward(self, images, capture_attention=False):
        self.calls += 1
        self._features = self.conv(images)
        return self.model.fc(self._features.mean((-2, -1)))


def test_training_option_reuses_one_unmodified_forward_and_updates_backbone():
    torch.manual_seed(73)
    model = TinyGAP()
    images = torch.randn(2, 3, 8, 8)
    original = images.clone()
    masks = torch.zeros(2, 1, 8, 8)
    masks[:, :, 2:6, 2:6] = 1
    config = {
        "method": "fcal_v2", "data": {"accumulation_steps": 1},
        "training": {"channels_last": False, "amp": "off", "grad_clip_norm": 1.0},
        "fcal": {"alpha": .15, "lambda_min": .05, "lambda_max": .3, "epsilon": 1e-8},
        "fcal_v2": {"warmup_epochs": 0, "ramp_epochs": 1, "support_temperature": 1.,
                    "support_margin": 0., "support_weight": .05, "tolerance_cell_fraction": .5,
                    "evidence_reference": "class_mean"},
    }
    before = model.conv.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    metrics = train_epoch(model, [{"image": images, "label": torch.tensor([0, 1]), "mask": masks}],
                          optimizer, config, 0, torch.device("cpu"))
    assert model.calls == 1
    torch.testing.assert_close(images, original, rtol=0, atol=0)
    assert not torch.equal(before, model.conv.weight)
    assert metrics["weighted_fcal"] > 0
