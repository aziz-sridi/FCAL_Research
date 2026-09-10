from __future__ import annotations

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
from torchvision.models.resnet import Bottleneck, ResNet


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

    def regional_logits(self, regions: torch.Tensor) -> torch.Tensor:
        """Classify a soft spatial region from features of the existing forward."""
        raise NotImplementedError(
            f"Regional logits are not available for {type(self).__name__}"
        )

    def background_feature_perturbed_logits(
        self, foreground_regions: torch.Tensor, sigma: float,
    ) -> torch.Tensor:
        """Classify after perturbing only the pooled background contribution."""
        raise NotImplementedError(
            f"Feature perturbation is not available for {type(self).__name__}"
        )

    def positive_weight_class_evidence(
        self, labels: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Target evidence before positive/negative classifier channels cancel."""
        raise NotImplementedError(
            f"Channel evidence is not available for {type(self).__name__}"
        )

    def raw_and_positive_weight_class_evidence(
        self, labels: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return (
            self.raw_class_evidence(labels),
            self.positive_weight_class_evidence(labels),
        )


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

    def regional_logits(self, regions: torch.Tensor) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("Forward must be called before requesting regional logits")
        weights = F.adaptive_avg_pool2d(
            regions.float().clamp(0.0, 1.0), self._features.shape[-2:]
        ).to(dtype=self._features.dtype)
        denominator = weights.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (self._features * weights).flatten(2).sum(dim=2) / denominator
        return self.model.fc(pooled)

    def background_feature_perturbed_logits(
        self, foreground_regions: torch.Tensor, sigma: float,
    ) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("Forward must be called before perturbing pooled features")
        if float(sigma) < 0.0:
            raise ValueError("background feature noise sigma must be non-negative")
        foreground = F.adaptive_avg_pool2d(
            foreground_regions.float().clamp(0.0, 1.0), self._features.shape[-2:]
        ).to(dtype=self._features.dtype)
        background = 1.0 - foreground
        spatial_cells = float(self._features.shape[-2] * self._features.shape[-1])
        foreground_fraction = foreground.flatten(1).sum(dim=1, keepdim=True) / spatial_cells
        background_fraction = 1.0 - foreground_fraction
        foreground_pooled = (
            (self._features * foreground).flatten(2).sum(dim=2)
            / foreground.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        background_pooled = (
            (self._features * background).flatten(2).sum(dim=2)
            / background.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        noise = torch.randn(
            background_pooled.shape,
            device=background_pooled.device,
            dtype=torch.float32,
        ).to(dtype=background_pooled.dtype)
        perturbed = (
            foreground_fraction * foreground_pooled
            + background_fraction * (background_pooled + float(sigma) * noise)
        )
        return self.model.fc(perturbed)

    def positive_weight_class_evidence(
        self, labels: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self._features is None:
            raise RuntimeError("Forward must be called before requesting channel evidence")
        weights = F.relu(self.model.fc.weight[labels]).to(
            dtype=self._features.dtype
        )
        return [torch.einsum("bc,bchw->bhw", weights, self._features)]

    def raw_and_positive_weight_class_evidence(
        self, labels: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self._features is None:
            raise RuntimeError("Forward must be called before requesting channel evidence")
        raw_weights = self.model.fc.weight[labels].to(dtype=self._features.dtype)
        paired_weights = torch.stack((raw_weights, F.relu(raw_weights)), dim=1)
        paired = torch.einsum("bkc,bchw->bkhw", paired_weights, self._features)
        return [paired[:, 0]], [paired[:, 1]]


class _LRPFlexBottleneck(Bottleneck):
    """TorchVision-compatible bottleneck with one ReLU module per use."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        del self.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu3(out + identity)


class ISNetResNet50CAM(ResNet50CAM):
    """ResNet-50 CAM adapter with differentiable LRP-e explanations."""

    def __init__(self, num_classes: int, pretrained: bool, epsilon: float = 0.01) -> None:
        nn.Module.__init__(self)
        self.model = ResNet(_LRPFlexBottleneck, [3, 4, 6, 3])
        self.model.relu.inplace = False
        if pretrained:
            source = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.load_state_dict(source.state_dict())
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self._features: torch.Tensor | None = None
        self._hook = self.model.layer4.register_forward_hook(self._capture)
        self._lrp_epsilon = float(epsilon)
        self._capture_lrp = False
        self._lrp_backward = False
        self._relu_outputs: dict[int, torch.Tensor] = {}
        self._lrp_hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                key = id(module)
                self._lrp_hooks.append(module.register_forward_hook(self._lrp_forward_hook(key)))
                self._lrp_hooks.append(module.register_full_backward_hook(self._lrp_backward_hook(key)))

    def _lrp_forward_hook(self, key: int):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            del module, inputs
            if self._capture_lrp:
                self._relu_outputs[key] = output
        return hook

    def _lrp_backward_hook(self, key: int):
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple):
            del module, grad_input
            if not self._lrp_backward:
                return None
            output = self._relu_outputs[key]
            attenuated_step = output / (output + self._lrp_epsilon)
            return (torch.nan_to_num(attenuated_step * grad_output[0]),)
        return hook

    def forward(self, images: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        self._capture_lrp = bool(capture_attention)
        if self._capture_lrp:
            self._relu_outputs.clear()
        logits = self.model(images)
        self._capture_lrp = False
        return logits

    def selective_lrp(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        selective_epsilon: float = 0.01,
    ) -> torch.Tensor:
        if not images.requires_grad:
            raise ValueError("ISNet requires input images with gradients enabled")
        if not self._relu_outputs:
            raise RuntimeError("Forward must use capture_attention=True before LRP")
        indices = logits.argmin(dim=1)
        probabilities = logits.float().softmax(dim=1)
        selected = probabilities.gather(1, indices[:, None]).squeeze(1)
        quantity = (
            torch.log(selected + float(selective_epsilon))
            - torch.log(1.0 - selected + float(selective_epsilon))
        ).sum()
        self._lrp_backward = True
        try:
            gradient = torch.autograd.grad(
                quantity, images, retain_graph=True, create_graph=True,
            )[0]
        finally:
            self._lrp_backward = False
        return torch.nan_to_num(gradient) * images


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

    def regional_logits(self, regions: torch.Tensor) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("Forward must be called before requesting regional logits")
        weights = F.adaptive_avg_pool2d(
            regions.float().clamp(0.0, 1.0), self._features.shape[-2:]
        ).to(dtype=self._features.dtype)
        denominator = weights.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (self._features * weights).flatten(2).sum(dim=2) / denominator
        return self.model.classifier(pooled)


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

    def regional_logits(self, regions: torch.Tensor) -> torch.Tensor:
        if self._features is None or self._dropout_scale is None:
            raise RuntimeError("Forward must be called before requesting regional logits")
        weights = F.adaptive_avg_pool2d(
            regions.float().clamp(0.0, 1.0), self._features.shape[-2:]
        ).to(dtype=self._features.dtype)
        denominator = weights.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (self._features * weights).flatten(2).sum(dim=2) / denominator
        dropout_scale = self._dropout_scale.to(dtype=pooled.dtype)
        return self.model.classifier[-1](pooled * dropout_scale)


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
        grid = round(selected.shape[-1] ** 0.5)
        if grid * grid != selected.shape[-1]:
            raise ValueError(f"Patch-token count {selected.shape[-1]} is not square")
        return [selected.reshape(selected.shape[0], grid, grid)]

    @staticmethod
    def _cls_map(attention: torch.Tensor) -> torch.Tensor:
        head_mean = attention.mean(dim=1)
        patch = head_mean[:, 0, 1:]
        grid = round(patch.shape[-1] ** 0.5)
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
        grid = round(patch.shape[-1] ** 0.5)
        return patch.reshape(patch.shape[0], grid, grid)


class TimmViTAvgPoolCAM(AttentionModel):
    """ViT whose prediction is exactly decomposable over normalized patch tokens."""

    family = "vit"

    def __init__(
        self,
        model_id: str,
        num_classes: int,
        pretrained: bool,
        require_second_order_attention: bool = False,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_id,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool="avg",
            fc_norm=False,
        )
        if require_second_order_attention:
            # PyTorch's fused/efficient scaled-dot-product attention does not
            # implement the double backward required by RRR's input-gradient
            # penalty.  Timm's explicit attention path is mathematically the
            # same operation and supports second derivatives.
            for block in self.model.blocks:
                if hasattr(block.attn, "fused_attn"):
                    block.attn.fused_attn = False
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
        grid = round(selected.shape[-1] ** 0.5)
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
        if config.get("method") == "isnet":
            return ISNetResNet50CAM(
                int(model["num_classes"]),
                pretrained=pretrained,
                epsilon=float(config.get("isnet", {}).get("lrp_epsilon", 0.01)),
            )
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
                require_second_order_attention=config.get("method") == "rrr",
            )
        return TimmViTAttention(
            model["pretrained_id"],
            int(model["num_classes"]),
            int(model.get("supervised_layers", 6)),
            pretrained=pretrained,
        )
    raise ValueError(f"Unsupported model: {model}")
