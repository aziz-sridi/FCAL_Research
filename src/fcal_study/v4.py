from __future__ import annotations

import torch
from torch.nn import functional as F

from .losses import article_tolerance_mask
from .v2 import detached_rms_normalize


def _valid_masks(masks: torch.Tensor) -> torch.Tensor:
    flat = masks.flatten(1)
    return flat.any(dim=1) & ~flat.all(dim=1)


def strict_positive_containment(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalized positive-evidence mass outside the undilated foreground."""
    if not raw_score_maps:
        raise ValueError("V4 requires at least one class-evidence map")
    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0
    masks = masks[valid]
    losses = []
    for scores in raw_score_maps:
        scores = scores[valid]
        foreground = article_tolerance_mask(masks, scores.shape[-2:], 0)
        positive = F.relu(scores.float())
        probability = positive / (
            positive.flatten(1).sum(dim=1).view(-1, 1, 1) + float(epsilon)
        )
        losses.append(
            (probability * (1.0 - foreground)).flatten(1).sum(dim=1).mean()
        )
    return _late_weighted_mean(losses)


def mask_alignment_loss(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    temperature: float = 1.0,
    mode: str = "rms_tanh",
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Match bounded scale-free positive evidence to the strict foreground mask."""
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if not raw_score_maps:
        raise ValueError("V4 requires at least one class-evidence map")
    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0
    masks = masks[valid]
    losses = []
    for scores in raw_score_maps:
        scores = scores[valid]
        foreground = article_tolerance_mask(masks, scores.shape[-2:], 0)
        if mode == "rms_tanh":
            activation = F.relu(
                torch.tanh(
                    detached_rms_normalize(scores, epsilon) / float(temperature)
                )
            )
        elif mode == "minmax":
            positive = F.relu(scores.float())
            flat = positive.flatten(1)
            minimum = flat.min(dim=1).values.view(-1, 1, 1)
            maximum = flat.max(dim=1).values.view(-1, 1, 1)
            activation = (positive - minimum) / (
                maximum - minimum + float(epsilon)
            )
        else:
            raise ValueError("alignment mode must be rms_tanh or minmax")
        losses.append(F.mse_loss(activation, foreground))
    return _late_weighted_mean(losses)


def regional_dominance_loss(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    margin: float = 0.6,
    power: float = 2.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Require foreground evidence to outrank background without mask imitation.

    The positive class-evidence map is normalized and resized exactly like the
    common evaluation CAM.  A one-sided margin stops contributing once mean
    foreground evidence sufficiently dominates mean background evidence, so it
    cannot keep rewarding an increasingly binary, silhouette-shaped map.
    """
    if float(margin) < 0:
        raise ValueError("regional dominance margin must be non-negative")
    if float(power) < 1:
        raise ValueError("regional dominance power must be at least one")
    if not raw_score_maps:
        raise ValueError("V4 requires at least one class-evidence map")
    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0
    masks = masks[valid].float().clamp(0.0, 1.0)
    losses = []
    for scores in raw_score_maps:
        positive = F.relu(scores[valid].float()).unsqueeze(1)
        maxima = positive.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        attention = torch.where(
            maxima > 0,
            positive / maxima.clamp_min(float(epsilon)),
            torch.zeros_like(positive),
        )
        if attention.shape[-2:] != masks.shape[-2:]:
            attention = F.interpolate(
                attention, size=masks.shape[-2:], mode="bilinear", align_corners=False,
            )
        foreground = masks
        background = 1.0 - foreground
        inside = (attention * foreground).flatten(1).sum(dim=1) / (
            foreground.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        )
        outside = (attention * background).flatten(1).sum(dim=1) / (
            background.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        )
        shortfall = F.relu(float(margin) - (inside - outside))
        losses.append(shortfall.pow(float(power)).mean())
    return _late_weighted_mean(losses)


def background_margin_sensitivity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    power: float = 1.0,
) -> torch.Tensor:
    """Scale-free squared input sensitivity of the target margin in background pixels."""
    if not images.requires_grad:
        raise ValueError("V4 sensitivity requires input images with gradients enabled")
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits and labels must have shapes [B,C] and [B]")
    target = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.float().clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    margin = target.float() - torch.logsumexp(competitors, dim=1)
    gradient = torch.autograd.grad(
        margin.sum(), images, create_graph=True, retain_graph=True,
    )[0].float()
    energy = gradient.square().mean(dim=1)
    scale = energy.mean(dim=(-2, -1), keepdim=True).add(float(epsilon)).detach()
    normalized = energy / scale
    background = 1.0 - masks[:, 0].float().clamp(0.0, 1.0)
    valid = _valid_masks(masks)
    denominator = background.flatten(1).sum(dim=1).clamp_min(float(epsilon))
    if float(power) < 1.0:
        raise ValueError("sensitivity power must be at least one")
    powered = normalized.clamp_min(0.0).pow(float(power))
    per_sample = (
        (powered * background).flatten(1).sum(dim=1) / denominator
    ).pow(1.0 / float(power))
    if not bool(valid.any()):
        return gradient.sum() * 0.0
    return per_sample[valid].mean()


def _late_weighted_mean(losses: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(losses)
    if len(losses) == 1:
        return stacked[0]
    weights = torch.arange(
        1, len(losses) + 1, device=stacked.device, dtype=stacked.dtype
    )
    return (weights / weights.sum() * stacked).sum()


__all__ = [
    "background_margin_sensitivity",
    "mask_alignment_loss",
    "regional_dominance_loss",
    "strict_positive_containment",
]
