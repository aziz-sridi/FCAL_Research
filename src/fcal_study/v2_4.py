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


def seed_and_grow_foreground_loss(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    acceptance_threshold: float = 0.35,
    growth_target: float = 0.65,
    evidence_temperature: float = 1.0,
    power: float = 2.0,
    foreground_threshold: float = 0.50,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Grow connected target evidence from its strongest foreground seed.

    Seed and connected-component selection are detached decisions. Gradients
    act only on the current in-mask growth front. The operation uses the native
    evidence grid and never constructs or forwards a modified image.
    """
    if not raw_score_maps:
        raise ValueError("FCAL V2.4 requires at least one class-evidence map")
    if not 0.0 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be in [0, 1]")
    if not 0.0 <= float(growth_target) <= 1.0:
        raise ValueError("growth target must be in [0, 1]")
    if float(evidence_temperature) <= 0.0:
        raise ValueError("evidence temperature must be positive")
    if not 0.0 <= float(foreground_threshold) <= 1.0:
        raise ValueError("foreground threshold must be in [0, 1]")
    if float(power) < 1.0:
        raise ValueError("growth power must be at least one")

    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0

    valid_masks = masks[valid]
    losses = []
    for scores in raw_score_maps:
        scores = scores[valid].float()
        projected = article_tolerance_mask(valid_masks, scores.shape[-2:], 0)
        foreground = projected >= float(foreground_threshold)

        positive = F.relu(scores)
        maxima = positive.flatten(1).amax(dim=1).view(-1, 1, 1)
        attention = torch.where(
            maxima > 0.0,
            positive / maxima.clamp_min(float(epsilon)),
            torch.zeros_like(positive),
        )
        scale = scores.square().mean(dim=(-2, -1), keepdim=True).add(
            float(epsilon)
        ).sqrt().detach()
        growable_evidence = torch.sigmoid(
            scores / scale / float(evidence_temperature)
        )

        with torch.no_grad():
            masked_attention = attention.detach().masked_fill(~foreground, -1.0)
            seed_indices = masked_attention.flatten(1).argmax(dim=1)
            active = torch.zeros_like(foreground)
            active.flatten(1).scatter_(1, seed_indices[:, None], True)
            eligible = foreground & (
                attention.detach() >= float(acceptance_threshold)
            )
            eligible = eligible | active
            for _ in range(max(scores.shape[-2:])):
                neighbors = F.max_pool2d(
                    active[:, None].float(), 3, stride=1, padding=1,
                )[:, 0].bool()
                active = active | (neighbors & eligible)
            neighbors = F.max_pool2d(
                active[:, None].float(), 3, stride=1, padding=1,
            )[:, 0].bool()
            front = neighbors & foreground & ~active

        shortfall = F.relu(
            float(growth_target) - growable_evidence
        ).pow(float(power))
        front_weight = front.float() * projected
        denominator = front_weight.flatten(1).sum(dim=1)
        per_sample = (
            (shortfall * front_weight).flatten(1).sum(dim=1)
            / denominator.clamp_min(float(epsilon))
        )
        has_front = (denominator > 0.0).to(dtype=per_sample.dtype)
        losses.append(
            (per_sample * has_front).sum() / has_front.sum().clamp_min(1.0)
        )
    return _late_weighted_mean(losses)


__all__ = ["seed_and_grow_foreground_loss"]
