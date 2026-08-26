from __future__ import annotations

from collections.abc import Iterable

import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    resnet50,
)


class AttentionModel(nn.Module):
    family: str

    def forward(self, images: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    def evaluation_attention(self, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def raw_training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError(f"Raw class CAM is not available for {type(self).__name__}")

    def raw_class_evidence(self, labels: torch.Tensor) -> list[torch.Tensor]:
        """Signed, class-specific spatial scores used by FCAL V2 and Universal FCAL."""
        return self.raw_training_attention(labels)


class ResNet50CAM(AttentionModel):
    family = "cnn"

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self._features: torch.Tensor | None = None
        self._hook = self.model.layer4.register_forward_hook(self._capture)

    def _capture(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        del module, inputs
        self._features = output

    def forward(self, images: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        del capture_attention
        return self.model(images)

    def _raw_cam(self, labels: torch.Tensor) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("Forward must be called before requesting CAM")
        weights = self.model.fc.weight[labels].to(dtype=self._features.dtype)
        return torch.einsum("bc,bchw->bhw", weights, self._features)

    def _cam(self, labels: torch.Tensor) -> torch.Tensor:
        return F.relu(self._raw_cam(labels))

    def training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [self._cam(labels)]

    def raw_training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [self._raw_cam(labels)]

    def raw_evaluation_attention(self, labels: torch.Tensor) -> torch.Tensor:
        return self._raw_cam(labels)

    def evaluation_attention(self, labels: torch.Tensor) -> torch.Tensor:
        return self._cam(labels)


class DenseNet121CAM(AttentionModel):
    """DenseNet-121 with an exact final-feature class activation map."""

    family = "cnn"

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = densenet121(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self._features: torch.Tensor | None = None

    def forward(self, images: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        del capture_attention
        features = F.relu(self.model.features(images), inplace=False)
        self._features = features
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.model.classifier(pooled)

    def _raw_cam(self, labels: torch.Tensor) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("Forward must be called before requesting CAM")
        weights = self.model.classifier.weight[labels].to(dtype=self._features.dtype)
        return torch.einsum("bc,bchw->bhw", weights, self._features)

    def training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [F.relu(self._raw_cam(labels))]

    def raw_training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [self._raw_cam(labels)]

    def evaluation_attention(self, labels: torch.Tensor) -> torch.Tensor:
        return F.relu(self._raw_cam(labels))


class EfficientNetB0CAM(AttentionModel):
    """EfficientNet-B0 with dropout-aware exact class activation maps."""

    family = "cnn"

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        linear = self.model.classifier[-1]
        self.model.classifier[-1] = nn.Linear(linear.in_features, num_classes)
        self._features: torch.Tensor | None = None
        self._dropout_scale: torch.Tensor | None = None

    def forward(self, images: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        del capture_attention
        features = self.model.features(images)
        self._features = features
        pooled = self.model.avgpool(features).flatten(1)
        dropout = self.model.classifier[0]
        if not isinstance(dropout, nn.Dropout):
            raise TypeError("EfficientNet-B0 classifier no longer exposes its dropout layer")
        scale = F.dropout(torch.ones_like(pooled), p=dropout.p, training=self.training)
        self._dropout_scale = scale
        return self.model.classifier[-1](pooled * scale)

    def _raw_cam(self, labels: torch.Tensor) -> torch.Tensor:
        if self._features is None or self._dropout_scale is None:
            raise RuntimeError("Forward must be called before requesting CAM")
        weights = self.model.classifier[-1].weight[labels].to(dtype=self._features.dtype)
        weights = weights * self._dropout_scale.to(dtype=weights.dtype)
        return torch.einsum("bc,bchw->bhw", weights, self._features)

    def training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [F.relu(self._raw_cam(labels))]

    def raw_training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [self._raw_cam(labels)]

    def evaluation_attention(self, labels: torch.Tensor) -> torch.Tensor:
        return F.relu(self._raw_cam(labels))


class TimmViTAttention(AttentionModel):
    family = "vit"

    def __init__(self, model_id: str, num_classes: int, supervised_layers: int, pretrained: bool) -> None:
        super().__init__()
        self.model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
        if not hasattr(self.model, "blocks"):
            raise TypeError(f"{model_id} does not expose transformer blocks")
        self.supervised_layers = int(supervised_layers)
        self._attention: dict[int, torch.Tensor] = {}
        self._tokens: torch.Tensor | None = None
        self._capture_attention = False
        self._capture_grad = False
        self._hooks = []
        for index, block in enumerate(self.model.blocks):
            if not hasattr(block.attn, "attn_drop"):
                raise TypeError(f"Block {index} does not expose attn_drop")
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False
            self._hooks.append(block.attn.attn_drop.register_forward_hook(self._make_hook(index)))

    def _make_hook(self, index: int):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            del module, inputs
            if self._capture_attention:
                first_supervised = len(self.model.blocks) - self.supervised_layers
                keep_graph = self._capture_grad and index >= first_supervised
                self._attention[index] = output if keep_graph else output.detach()
        return hook

    def forward(self, images: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        self._attention.clear()
        self._capture_attention = bool(capture_attention)
        self._capture_grad = bool(capture_attention and torch.is_grad_enabled())
        self._tokens = self.model.forward_features(images)
        logits = self.model.forward_head(self._tokens)
        self._capture_attention = False
        self._capture_grad = False
        return logits

    def raw_class_evidence(self, labels: torch.Tensor) -> list[torch.Tensor]:
        """Project final patch tokens through the shared class head.

        Unlike head-mean self-attention, this map is class-specific and signed.
        The main prediction still comes from the normal CLS-token path.
        """
        if self._tokens is None:
            raise RuntimeError("Forward must be called before requesting patch evidence")
        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        patches = self._tokens[:, prefix_tokens:]
        head = self.model.head
        if not isinstance(head, nn.Linear):
            raise TypeError("FCAL requires a linear ViT classification head")
        weights = head.weight[labels].to(dtype=patches.dtype)
        selected = torch.einsum("bd,bnd->bn", weights, patches)
        grid = int(round(selected.shape[-1] ** 0.5))
        if grid * grid != selected.shape[-1]:
            raise ValueError(f"Patch-token count {selected.shape[-1]} is not square")
        return [selected.reshape(selected.shape[0], grid, grid)]

    @staticmethod
    def _cls_map(attention: torch.Tensor) -> torch.Tensor:
        head_mean = attention.mean(dim=1)
        patch = head_mean[:, 0, 1:]
        grid = int(round(patch.shape[-1] ** 0.5))
        if grid * grid != patch.shape[-1]:
            raise ValueError(f"Patch-token count {patch.shape[-1]} is not square")
        return patch.reshape(patch.shape[0], grid, grid)

    def training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        del labels
        count = len(self.model.blocks)
        indices = range(count - self.supervised_layers, count)
        if any(index not in self._attention for index in indices):
            raise RuntimeError("Missing ViT attention; forward with capture_attention=True")
        return [self._cls_map(self._attention[index]) for index in indices]

    def evaluation_attention(self, labels: torch.Tensor) -> torch.Tensor:
        del labels
        count = len(self.model.blocks)
        if len(self._attention) != count:
            raise RuntimeError("Rollout requires attention from every transformer block")
        rollout = None
        for index in range(count):
            attention = self._attention[index].mean(dim=1)
            identity = torch.eye(attention.shape[-1], device=attention.device, dtype=attention.dtype)
            attention = attention + identity.unsqueeze(0)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            rollout = attention if rollout is None else torch.bmm(attention, rollout)
        patch = rollout[:, 0, 1:]
        grid = int(round(patch.shape[-1] ** 0.5))
        return patch.reshape(patch.shape[0], grid, grid)


class TimmViTAvgPoolCAM(AttentionModel):
    """ViT whose prediction is exactly decomposable over normalized patch tokens."""

    family = "vit"

    def __init__(self, model_id: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_id,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool="avg",
            fc_norm=False,
        )
        self._tokens: torch.Tensor | None = None

    def forward(self, images: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        del capture_attention
        self._tokens = self.model.forward_features(images)
        return self.model.forward_head(self._tokens)

    def _raw_patch_cam(self, labels: torch.Tensor) -> torch.Tensor:
        if self._tokens is None:
            raise RuntimeError("Forward must be called before requesting patch evidence")
        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        patches = self._tokens[:, prefix_tokens:]
        head = self.model.head
        if not isinstance(head, nn.Linear):
            raise TypeError("Average-pooled ViT FCAL requires a linear classification head")
        weights = head.weight[labels].to(dtype=patches.dtype)
        selected = torch.einsum("bd,bnd->bn", weights, patches)
        grid = int(round(selected.shape[-1] ** 0.5))
        if grid * grid != selected.shape[-1]:
            raise ValueError(f"Patch-token count {selected.shape[-1]} is not square")
        return selected.reshape(selected.shape[0], grid, grid)

    def raw_class_evidence(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [self._raw_patch_cam(labels)]

    def raw_training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return self.raw_class_evidence(labels)

    def training_attention(self, labels: torch.Tensor) -> list[torch.Tensor]:
        return [F.relu(self._raw_patch_cam(labels))]

    def evaluation_attention(self, labels: torch.Tensor) -> torch.Tensor:
        return F.relu(self._raw_patch_cam(labels))


def create_model(config: dict, pretrained: bool = True) -> AttentionModel:
    model = config["model"]
    if model["name"] == "resnet50":
        return ResNet50CAM(int(model["num_classes"]), pretrained=pretrained)
    if model["name"] == "densenet121":
        return DenseNet121CAM(int(model["num_classes"]), pretrained=pretrained)
    if model["name"] == "efficientnet_b0":
        return EfficientNetB0CAM(int(model["num_classes"]), pretrained=pretrained)
    if model["family"] == "vit":
        if model.get("pool") == "avg":
            return TimmViTAvgPoolCAM(
                model["pretrained_id"],
                int(model["num_classes"]),
                pretrained=pretrained,
            )
        return TimmViTAttention(
            model["pretrained_id"],
            int(model["num_classes"]),
            int(model.get("supervised_layers", 6)),
            pretrained=pretrained,
        )
    raise ValueError(f"Unsupported model: {model}")
