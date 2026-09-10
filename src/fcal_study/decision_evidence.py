"""Experimental class-mean reference for V2; no additional spatial losses.

For GAP + linear heads, this removes the common-logit component to which
softmax is invariant. It is not a certificate of input-space robustness.
"""

from __future__ import annotations

import torch
from torch import nn

from .models import (
    DenseNet121CAM,
    EfficientNetB0CAM,
    ResNet50CAM,
    TimmViTAvgPoolCAM,
)


def centered_class_weights(weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Subtract the class mean BEFORE projection and nonlinear normalization.

    The mean remains differentiable: auxiliary gradients across classifier
    rows sum to zero. No [batch, classes, height, width] tensor is constructed.
    """
    if weight.ndim != 2 or weight.shape[0] < 2:
        raise ValueError("classifier weight must have shape [classes>=2, features]")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch]")
    values = weight.float()
    return values[labels] - values.mean(dim=0, keepdim=True)


def class_mean_evidence(model, labels: torch.Tensor) -> list[torch.Tensor]:
    """Reuse the exact cached classification features of supported adapters.

    EfficientNet reuses the sampled classifier dropout. CLS-token ViTs are
    intentionally unsupported: their patch means do not decompose logits.
    Bias is excluded just as in the existing raw CAM convention.
    """
    if isinstance(model, TimmViTAvgPoolCAM):
        if model._tokens is None:
            raise RuntimeError("Forward must precede class-mean evidence")
        head = model.model.head
        if not isinstance(head, nn.Linear):
            raise TypeError("class-mean evidence requires a linear head")
        if float(getattr(model.model.head_drop, "p", 0.0)) != 0.0:
            raise ValueError("class-mean ViT evidence requires zero head dropout")
        prefix = int(getattr(model.model, "num_prefix_tokens", 1))
        patches = model._tokens[:, prefix:]
        weights = centered_class_weights(head.weight, labels).to(patches.dtype)
        selected = torch.einsum("bd,bnd->bn", weights, patches)
        grid = round(selected.shape[-1] ** 0.5)
        if grid * grid != selected.shape[-1]:
            raise ValueError("patch count must be square")
        return [selected.reshape(selected.shape[0], grid, grid)]

    if isinstance(model, ResNet50CAM):
        head = model.model.fc
    elif isinstance(model, DenseNet121CAM):
        head = model.model.classifier
    elif isinstance(model, EfficientNetB0CAM):
        head = model.model.classifier[-1]
    else:
        raise TypeError("class-mean evidence supports GAP CNNs and average-pooled ViTs only")
    features = model._features
    if features is None:
        raise RuntimeError("Forward must precede class-mean evidence")
    weights = centered_class_weights(head.weight, labels).to(features.dtype)
    if isinstance(model, EfficientNetB0CAM):
        if model._dropout_scale is None:
            raise RuntimeError("Forward must capture EfficientNet dropout")
        weights = weights * model._dropout_scale.to(weights.dtype)
    return [torch.einsum("bc,bchw->bhw", weights, features)]


def validate_evidence_reference(config: dict) -> str:
    reference = config.get("fcal_v2", {}).get("evidence_reference", "target")
    if reference not in {"target", "class_mean"}:
        raise ValueError("fcal_v2.evidence_reference must be target or class_mean")
    if reference != "target" and config["method"] != "fcal_v2":
        raise ValueError("class_mean is experimental and supported only for method=fcal_v2")
    return reference


def v2_training_evidence(model, labels: torch.Tensor, reference: str) -> list[torch.Tensor]:
    if reference == "target":
        return model.raw_class_evidence(labels)
    if reference == "class_mean":
        return class_mean_evidence(model, labels)
    raise ValueError("unknown V2 evidence reference")
