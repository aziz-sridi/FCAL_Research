from __future__ import annotations

import torch
from torch.nn import functional as F

from .losses import article_tolerance_mask


def _valid_masks(masks: torch.Tensor) -> torch.Tensor:
    flat = masks.flatten(1)
    return flat.any(dim=1) & ~flat.all(dim=1)


def _late_weighted_mean(losses: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(losses)
    if len(losses) == 1:
        return stacked[0]
    weights = torch.arange(
        1, len(losses) + 1, device=stacked.device, dtype=stacked.dtype,
    )
    return (weights / weights.sum() * stacked).sum()


def strict_positive_containment_loss(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Penalize positive target evidence outside the undilated mask.

    V2 retains its resolution-aware tolerant containment. This complementary
    term closes the tolerated boundary halo without changing V2 itself.
    """
    if not raw_score_maps:
        raise ValueError("FCAL V2.3 requires at least one class-evidence map")
    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0

    valid_masks = masks[valid]
    losses = []
    for scores in raw_score_maps:
        scores = scores[valid]
        foreground = article_tolerance_mask(valid_masks, scores.shape[-2:], 0)
        positive = F.relu(scores.float())
        probability = positive / (
            positive.flatten(1).sum(dim=1).view(-1, 1, 1) + float(epsilon)
        )
        losses.append(
            (probability * (1.0 - foreground)).flatten(1).sum(dim=1).mean()
        )
    return _late_weighted_mean(losses)


def foreground_coverage_hinge_loss(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    target: float = 0.60,
    power: float = 2.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Require broad positive target evidence inside the foreground.

    Positive CAMs are max-normalized and resized like the common evaluator.
    The one-sided hinge becomes exactly zero after mean foreground attention
    reaches ``target``; it therefore discourages sparse hotspots without
    continually rewarding silhouette imitation.
    """
    if not raw_score_maps:
        raise ValueError("FCAL V2.3 requires at least one class-evidence map")
    if not 0.0 <= float(target) <= 1.0:
        raise ValueError("foreground coverage target must be in [0, 1]")
    if float(power) < 1.0:
        raise ValueError("foreground coverage power must be at least one")

    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0

    foreground = masks[valid].float().clamp(0.0, 1.0)
    denominator = foreground.flatten(1).sum(dim=1).clamp_min(float(epsilon))
    losses = []
    for scores in raw_score_maps:
        positive = F.relu(scores[valid].float()).unsqueeze(1)
        maxima = positive.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        attention = torch.where(
            maxima > 0.0,
            positive / maxima.clamp_min(float(epsilon)),
            torch.zeros_like(positive),
        )
        if attention.shape[-2:] != foreground.shape[-2:]:
            attention = F.interpolate(
                attention,
                size=foreground.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        inside = (
            (attention * foreground).flatten(1).sum(dim=1) / denominator
        )
        losses.append(
            F.relu(float(target) - inside).pow(float(power)).mean()
        )
    return _late_weighted_mean(losses)


__all__ = [
    "foreground_coverage_hinge_loss",
    "strict_positive_containment_loss",
]
