import torch
from torch import nn
from torch.nn import functional as F

from fcal_study.train import train_epoch
from fcal_study.v5 import regional_margin_certificate_from_maps


def test_certificate_rewards_foreground_sufficient_pairwise_margins():
    masks = torch.zeros(1, 1, 32, 32)
    masks[:, :, 8:24, 8:24] = 1.0
    target = torch.full((1, 8, 8), -1.0)
    target[:, 2:6, 2:6] = 2.0
    rival = -target
    aligned = regional_margin_certificate_from_maps(
        [target], [[rival]], masks,
    )
    shortcut = regional_margin_certificate_from_maps(
        [-target], [[-rival]], masks,
    )
    assert aligned.loss < shortcut.loss
    assert aligned.certificate > shortcut.certificate


def test_certificate_rejects_invalid_tail_temperature():
    masks = torch.zeros(1, 1, 16, 16)
    masks[:, :, 4:12, 4:12] = 1.0
    scores = torch.randn(1, 4, 4)
    try:
        regional_margin_certificate_from_maps(
            [scores], [], masks, foreground_temperature=0.0,
        )
    except ValueError as error:
        assert "tail temperatures" in str(error)
    else:
        raise AssertionError("invalid tail fraction was accepted")


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


def test_v5_train_epoch_is_single_view_first_order():
    config = {
        "method": "fcal_v5_cheap",
        "data": {"accumulation_steps": 1},
        "training": {"channels_last": False, "amp": "off", "grad_clip_norm": 1.0},
        "fcal_v5_cheap": {
            "warmup_epochs": 0,
            "ramp_epochs": 1,
            "coefficient": 0.05,
            "top_k_competitors": 2,
            "foreground_temperature": 0.5,
            "background_temperature": 0.5,
            "background_scale": 0.5,
            "margin": 0.0,
            "temperature": 0.25,
            "epsilon": 1e-8,
        },
    }
    masks = torch.zeros(4, 1, 16, 16)
    masks[:, :, 4:12, 4:12] = 1.0
    batch = {
        "image": torch.randn(4, 3, 16, 16),
        "label": torch.tensor([0, 1, 2, 0]),
        "mask": masks,
    }
    model = _TinyEvidenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = train_epoch(
        model, [batch], optimizer, config, 0, torch.device("cpu")
    )
    assert metrics["certificate_loss"] > 0.0
    assert metrics["counterfactual_ce"] == 0.0
    assert metrics["background_sensitivity"] == 0.0
    assert all(not parameter.grad for parameter in batch["image"])
