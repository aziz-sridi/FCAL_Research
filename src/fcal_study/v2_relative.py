from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def strongest_rival_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return target logit minus the strongest non-target logit per sample."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits and labels must have shapes [B,C] and [B]")
    if logits.shape[1] < 2:
        raise ValueError("relative pooled margin requires at least two classes")
    float_logits = logits.float()
    target = float_logits.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        candidates = float_logits.detach().clone()
        candidates.scatter_(1, labels[:, None], float("-inf"))
        rival_labels = candidates.argmax(dim=1)
    rival = float_logits.gather(1, rival_labels[:, None]).squeeze(1)
    return target - rival


def _outside_evidence(
    raw_score_maps: list[torch.Tensor],
    background_maps: list[torch.Tensor],
    valid: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if not raw_score_maps or len(raw_score_maps) != len(background_maps):
        raise ValueError("target and background evidence depths must match and be non-empty")
    layer_values = []
    for scores, background in zip(raw_score_maps, background_maps, strict=True):
        selected = scores[valid].float()
        if background.shape[0] == scores.shape[0]:
            background = background[valid]
        elif background.shape[0] != selected.shape[0]:
            raise ValueError("background maps must contain the full or valid-only batch")
        if selected.shape != background.shape:
            raise ValueError("target evidence and V2 background maps must align")
        positive = F.relu(selected)
        probability = positive / (
            positive.flatten(1).sum(dim=1).view(-1, 1, 1) + float(epsilon)
        )
        layer_values.append(
            (probability * background.float()).flatten(1).sum(dim=1)
        )
    stacked = torch.stack(layer_values)
    if len(layer_values) == 1:
        return stacked[0]
    weights = torch.arange(
        1, len(layer_values) + 1, device=stacked.device, dtype=stacked.dtype
    )
    return (weights[:, None] / weights.sum() * stacked).sum(dim=0)


@dataclass(frozen=True)
class RelativePooledMarginLoss:
    loss: torch.Tensor
    shortcut_gate: torch.Tensor
    foreground_margin: torch.Tensor
    background_margin: torch.Tensor
    margin_gap: torch.Tensor
    outside_evidence: torch.Tensor
    valid_mask_fraction: torch.Tensor


def relative_pooled_margin_loss(
    model,
    raw_score_maps: list[torch.Tensor],
    background_maps: list[torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    delta: float = 2.0,
    temperature: float = 1.0,
    allowed_outside: float = 0.20,
    required_foreground_margin: float = 0.0,
    outside_gate_temperature: float = 0.05,
    margin_gate_temperature: float = 1.0,
    background_gradient_scale: float = 0.10,
    epsilon: float = 1e-8,
) -> RelativePooledMarginLoss:
    """Gate a foreground-versus-background class margin from cached features.

    The model's ``regional_logits`` calls reuse the feature map or patch tokens
    captured by the original image forward. The detached smooth gate focuses
    optimization on examples with excessive outside evidence or insufficient
    foreground class margin.
    """
    if float(temperature) <= 0.0:
        raise ValueError("relative-margin temperature must be positive")
    if float(outside_gate_temperature) <= 0.0 or float(margin_gate_temperature) <= 0.0:
        raise ValueError("gate temperatures must be positive")
    if not 0.0 <= float(allowed_outside) <= 1.0:
        raise ValueError("allowed outside evidence must be in [0, 1]")
    if not 0.0 <= float(background_gradient_scale) <= 1.0:
        raise ValueError("background gradient scale must be in [0, 1]")
    if masks.ndim != 4 or masks.shape[0] != labels.shape[0] or masks.shape[1] != 1:
        raise ValueError("masks must have shape [B,1,H,W] aligned with labels")

    flat_masks = masks.flatten(1)
    valid = flat_masks.any(dim=1) & ~flat_masks.all(dim=1)
    valid_fraction = valid.float().mean()
    foreground_logits = model.regional_logits(masks)
    background_logits = model.regional_logits(1.0 - masks)
    foreground_margin = strongest_rival_margin(foreground_logits, labels)[valid]
    background_margin = strongest_rival_margin(background_logits, labels)[valid]
    outside = _outside_evidence(
        raw_score_maps, background_maps, valid, float(epsilon)
    )

    with torch.no_grad():
        outside_risk = torch.sigmoid(
            (outside.detach() - float(allowed_outside))
            / float(outside_gate_temperature)
        )
        foreground_risk = torch.sigmoid(
            (float(required_foreground_margin) - foreground_margin.detach())
            / float(margin_gate_temperature)
        )
        shortcut_gate = 1.0 - (1.0 - outside_risk) * (1.0 - foreground_risk)

    valid_count = valid.float().sum().clamp_min(1.0)
    background_for_gradient = (
        float(background_gradient_scale) * background_margin
        + (1.0 - float(background_gradient_scale)) * background_margin.detach()
    )
    margin_gap = foreground_margin - background_for_gradient
    per_sample = F.softplus(
        (float(delta) - margin_gap) / float(temperature)
    ) * float(temperature)
    graph_zero = sum(scores.sum() for scores in raw_score_maps) * 0.0
    loss = (shortcut_gate * per_sample).sum() / valid_count + graph_zero
    return RelativePooledMarginLoss(
        loss=loss,
        shortcut_gate=shortcut_gate.sum() / valid_count,
        foreground_margin=foreground_margin.sum() / valid_count,
        background_margin=background_margin.sum() / valid_count,
        margin_gap=(foreground_margin - background_margin).sum() / valid_count,
        outside_evidence=outside.sum() / valid_count,
        valid_mask_fraction=valid_fraction,
    )


__all__ = [
    "RelativePooledMarginLoss",
    "relative_pooled_margin_loss",
    "strongest_rival_margin",
]
