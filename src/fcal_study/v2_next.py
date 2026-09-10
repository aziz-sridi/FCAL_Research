from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .losses import article_tolerance_mask
from .regions import projected_region_weights
from .v2 import detached_rms_normalize, resolution_aware_tolerance_pixels


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


def _region_shortfall(
    evidence: torch.Tensor,
    region: torch.Tensor,
    projected_mask: torch.Tensor,
    target: float,
    power: float,
    epsilon: float,
) -> torch.Tensor:
    region_weight = region.float() * projected_mask
    denominator = region_weight.flatten(1).sum(dim=1)
    shortfall = F.relu(float(target) - evidence).pow(float(power))
    per_sample = (
        (shortfall * region_weight).flatten(1).sum(dim=1)
        / denominator.clamp_min(float(epsilon))
    )
    present = (denominator > 0.0).to(dtype=per_sample.dtype)
    return (per_sample * present).sum() / present.sum().clamp_min(1.0)


@dataclass(frozen=True)
class PersistentGrowthLoss:
    front: torch.Tensor
    retention: torch.Tensor


def persistent_seed_and_grow_loss(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    acceptance_threshold: float = 0.35,
    front_target: float = 0.65,
    retention_target: float = 0.60,
    evidence_temperature: float = 1.0,
    power: float = 2.0,
    foreground_threshold: float = 0.50,
    epsilon: float = 1e-8,
) -> PersistentGrowthLoss:
    """Grow one connected ring while preserving the currently active trail."""
    if not raw_score_maps:
        raise ValueError("persistent Seed-and-Grow requires evidence maps")
    if not 0.0 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be in [0, 1]")
    if not 0.0 <= float(front_target) <= 1.0:
        raise ValueError("front target must be in [0, 1]")
    if not 0.0 <= float(retention_target) <= 1.0:
        raise ValueError("retention target must be in [0, 1]")
    if float(evidence_temperature) <= 0.0:
        raise ValueError("evidence temperature must be positive")
    if float(power) < 1.0:
        raise ValueError("growth power must be at least one")

    valid = _valid_masks(masks)
    if not bool(valid.any()):
        zero = sum(scores.sum() for scores in raw_score_maps) * 0.0
        return PersistentGrowthLoss(zero, zero)

    valid_masks = masks[valid]
    front_losses = []
    retention_losses = []
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
        evidence = torch.sigmoid(
            scores / scale / float(evidence_temperature)
        )

        with torch.no_grad():
            seed_indices = attention.detach().masked_fill(
                ~foreground, -1.0
            ).flatten(1).argmax(dim=1)
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

        front_losses.append(_region_shortfall(
            evidence, front, projected, front_target, power, epsilon,
        ))
        retention_losses.append(_region_shortfall(
            evidence, active, projected, retention_target, power, epsilon,
        ))

    return PersistentGrowthLoss(
        _late_weighted_mean(front_losses),
        _late_weighted_mean(retention_losses),
    )


def native_regional_dominance_loss(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    margin: float = 0.50,
    power: float = 2.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Require normalized target attention to dominate at the native grid."""
    if not raw_score_maps:
        raise ValueError("native regional dominance requires evidence maps")
    if float(margin) < 0.0 or float(power) < 1.0:
        raise ValueError("invalid regional-dominance margin or power")
    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0

    valid_masks = masks[valid]
    losses = []
    for scores in raw_score_maps:
        scores = scores[valid].float()
        foreground = article_tolerance_mask(
            valid_masks, scores.shape[-2:], 0
        ).clamp(0.0, 1.0)
        background = 1.0 - foreground
        positive = F.relu(scores)
        maxima = positive.flatten(1).amax(dim=1).view(-1, 1, 1)
        attention = torch.where(
            maxima > 0.0,
            positive / maxima.clamp_min(float(epsilon)),
            torch.zeros_like(positive),
        )
        inside = (attention * foreground).flatten(1).sum(dim=1) / (
            foreground.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        )
        outside = (attention * background).flatten(1).sum(dim=1) / (
            background.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        )
        losses.append(F.relu(float(margin) - (inside - outside)).pow(
            float(power)
        ).mean())
    return _late_weighted_mean(losses)


@dataclass(frozen=True)
class RegionalClassificationLoss:
    foreground: torch.Tensor
    background: torch.Tensor


def buffered_region_masks(
    masks: torch.Tensor,
    radius_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build disjoint interior-foreground and distant-background masks.

    The foreground is eroded and the background is the complement of the
    correspondingly dilated foreground. The boundary band is therefore used
    by neither regional pool. Morphology is performed at input-mask resolution
    before ``regional_logits`` projects each region to the feature grid.
    """
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError("regional masks must have shape [batch, 1, height, width]")
    radius = int(radius_pixels)
    if radius < 0:
        raise ValueError("regional buffer radius must be non-negative")
    foreground = masks.float().clamp(0.0, 1.0)
    if radius == 0:
        return foreground, 1.0 - foreground
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(
        foreground, kernel_size=kernel_size, stride=1, padding=radius,
    )
    eroded = 1.0 - F.max_pool2d(
        1.0 - foreground, kernel_size=kernel_size, stride=1, padding=radius,
    )
    return eroded, 1.0 - dilated


def local_foreground_rival_margin_loss(
    target_maps: list[torch.Tensor],
    competitor_maps: list[list[torch.Tensor]],
    masks: torch.Tensor,
    *,
    lower_tail_fraction: float = 0.25,
    margin: float = 0.0,
    evidence_temperature: float = 1.0,
    loss_temperature: float = 0.25,
    foreground_threshold: float = 0.50,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Raise the weakest target-vs-rival margins within the foreground."""
    if not target_maps or not competitor_maps:
        raise ValueError("local rival margin requires target and competitor maps")
    if any(len(maps) != len(target_maps) for maps in competitor_maps):
        raise ValueError("target and competitor evidence depths must match")
    if not 0.0 < float(lower_tail_fraction) <= 1.0:
        raise ValueError("lower-tail fraction must be in (0, 1]")
    if float(evidence_temperature) <= 0.0 or float(loss_temperature) <= 0.0:
        raise ValueError("temperatures must be positive")

    valid = _valid_masks(masks)
    if not bool(valid.any()):
        terms = [scores.sum() for scores in target_maps]
        terms.extend(scores.sum() for maps in competitor_maps for scores in maps)
        return sum(terms) * 0.0

    valid_masks = masks[valid]
    layer_losses = []
    for layer_index, target_scores in enumerate(target_maps):
        target_scores = target_scores[valid].float()
        foreground = article_tolerance_mask(
            valid_masks, target_scores.shape[-2:], 0,
        ).clamp(0.0, 1.0)
        hard_foreground = foreground >= float(foreground_threshold)
        target = detached_rms_normalize(target_scores, epsilon)
        rivals = torch.stack([
            detached_rms_normalize(maps[layer_index][valid], epsilon)
            for maps in competitor_maps
        ], dim=1)
        local_margin = torch.tanh(
            (target - rivals.amax(dim=1)) / float(evidence_temperature)
        )
        flat_margin = local_margin.flatten(1)
        flat_foreground = foreground.flatten(1)
        flat_hard = hard_foreground.flatten(1)
        counts = flat_hard.sum(dim=1)
        tail_counts = torch.ceil(
            counts.float() * float(lower_tail_fraction)
        ).long().clamp_min(1)
        order = flat_margin.detach().masked_fill(
            ~flat_hard, float("inf")
        ).argsort(dim=1)
        ranks = torch.arange(
            flat_margin.shape[1], device=flat_margin.device
        )[None]
        keep_by_rank = ranks < tail_counts[:, None]
        selected = torch.zeros_like(flat_hard).scatter(1, order, keep_by_rank)
        selected = selected & flat_hard
        weights = selected.to(dtype=flat_foreground.dtype) * flat_foreground
        shortfall = (
            float(margin) - flat_margin
        ) / float(loss_temperature)
        cell_losses = F.softplus(shortfall).mul(float(loss_temperature))
        denominator = weights.sum(dim=1)
        per_sample = (cell_losses * weights).sum(dim=1) / denominator.clamp_min(
            float(epsilon)
        )
        present = denominator > 0.0
        layer_losses.append(
            (per_sample * present).sum() / present.sum().clamp_min(1)
        )
    return _late_weighted_mean(layer_losses)


def softmin_foreground_rival_margin_loss(
    target_maps: list[torch.Tensor],
    competitor_maps: list[list[torch.Tensor]],
    masks: torch.Tensor,
    *,
    softmin_temperature: float = 0.25,
    margin: float = 0.0,
    evidence_temperature: float = 1.0,
    loss_temperature: float = 0.25,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Smoothly raise weak foreground margins without sorting spatial cells."""
    if not target_maps or not competitor_maps:
        raise ValueError("softmin rival margin requires target and competitor maps")
    if any(len(maps) != len(target_maps) for maps in competitor_maps):
        raise ValueError("target and competitor evidence depths must match")
    if min(
        float(softmin_temperature),
        float(evidence_temperature),
        float(loss_temperature),
    ) <= 0.0:
        raise ValueError("temperatures must be positive")

    valid = _valid_masks(masks)
    if not bool(valid.any()):
        terms = [scores.sum() for scores in target_maps]
        terms.extend(scores.sum() for maps in competitor_maps for scores in maps)
        return sum(terms) * 0.0

    valid_masks = masks[valid]
    losses = []
    for layer_index, target_scores in enumerate(target_maps):
        target_scores = target_scores[valid].float()
        foreground = article_tolerance_mask(
            valid_masks, target_scores.shape[-2:], 0,
        ).clamp(0.0, 1.0)
        target = detached_rms_normalize(target_scores, epsilon)
        rivals = torch.stack([
            detached_rms_normalize(maps[layer_index][valid], epsilon)
            for maps in competitor_maps
        ], dim=1)
        local_margin = torch.tanh(
            (target - rivals.amax(dim=1)) / float(evidence_temperature)
        )
        flat_weights = foreground.flatten(1)
        log_weights = flat_weights.clamp_min(float(epsilon)).log()
        scaled = -local_margin.flatten(1) / float(softmin_temperature)
        log_weighted_mean = torch.logsumexp(
            scaled + log_weights, dim=1,
        ) - flat_weights.sum(dim=1).clamp_min(float(epsilon)).log()
        soft_minimum = -float(softmin_temperature) * log_weighted_mean
        shortfall = (
            float(margin) - soft_minimum
        ) / float(loss_temperature)
        losses.append(
            F.softplus(shortfall).mul(float(loss_temperature)).mean()
        )
    return _late_weighted_mean(losses)


def uncancelled_channel_containment_loss(
    positive_weight_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    tolerance_cell_fraction: float,
    minimum_tolerance_pixels: int = 0,
    maximum_tolerance_pixels: int | None = None,
    background_maps: list[torch.Tensor] | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Contain positive classifier-channel evidence before channel cancellation."""
    if not positive_weight_maps:
        raise ValueError("channel containment requires evidence maps")
    if background_maps is not None and len(background_maps) != len(
        positive_weight_maps
    ):
        raise ValueError("channel evidence and background depths must match")
    valid = _valid_masks(masks)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in positive_weight_maps) * 0.0

    valid_masks = masks[valid]
    input_size = (int(masks.shape[-2]), int(masks.shape[-1]))
    losses = []
    for layer_index, scores in enumerate(positive_weight_maps):
        scores = scores[valid].float().clamp_min(0.0)
        map_size = (int(scores.shape[-2]), int(scores.shape[-1]))
        if background_maps is None:
            tolerance = resolution_aware_tolerance_pixels(
                input_size,
                map_size,
                tolerance_cell_fraction,
                minimum_tolerance_pixels,
                maximum_tolerance_pixels,
            )
            _, _, background = projected_region_weights(
                valid_masks, map_size, tolerance,
            )
        else:
            background = background_maps[layer_index][valid].float()
        probability = scores / (
            scores.flatten(1).sum(dim=1).view(-1, 1, 1) + float(epsilon)
        )
        losses.append(
            (probability * background).flatten(1).sum(dim=1).mean()
        )
    return _late_weighted_mean(losses)


def _target_logit_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = logits.float().gather(1, labels[:, None]).squeeze(1)
    competitors = logits.float().clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    return target - torch.logsumexp(competitors, dim=1)


def regional_classification_losses(
    model,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    foreground_margin: float = 0.0,
    maximum_background_margin: float = -2.0,
    temperature: float = 1.0,
    compute_background: bool = True,
    background_regions: torch.Tensor | None = None,
    exclude_empty: bool = False,
) -> RegionalClassificationLoss:
    """Supervise foreground sufficiency and background non-predictiveness."""
    if float(temperature) <= 0.0:
        raise ValueError("regional classification temperature must be positive")
    foreground_logits = model.regional_logits(masks)
    foreground = _target_logit_margin(foreground_logits, labels)
    foreground_per_sample = F.softplus(
        (float(foreground_margin) - foreground) / float(temperature)
    ).mul(float(temperature))
    foreground_valid = masks.flatten(1).sum(dim=1) > 0.0
    if exclude_empty:
        fg_loss = (
            foreground_per_sample[foreground_valid].mean()
            if bool(foreground_valid.any())
            else foreground_logits.sum() * 0.0
        )
    else:
        fg_loss = foreground_per_sample.mean()
    if not compute_background:
        return RegionalClassificationLoss(fg_loss, fg_loss.new_zeros(()))
    background_regions = 1.0 - masks if background_regions is None else background_regions
    if background_regions.shape != masks.shape:
        raise ValueError("foreground and background regional masks must have equal shape")
    background_logits = model.regional_logits(background_regions)
    background = _target_logit_margin(background_logits, labels)
    background_per_sample = F.softplus(
        (background - float(maximum_background_margin)) / float(temperature)
    ).mul(float(temperature))
    background_valid = background_regions.flatten(1).sum(dim=1) > 0.0
    if exclude_empty:
        bg_loss = (
            background_per_sample[background_valid].mean()
            if bool(background_valid.any())
            else background_logits.sum() * 0.0
        )
    else:
        bg_loss = background_per_sample.mean()
    return RegionalClassificationLoss(fg_loss, bg_loss)


__all__ = [
    "PersistentGrowthLoss",
    "RegionalClassificationLoss",
    "buffered_region_masks",
    "local_foreground_rival_margin_loss",
    "native_regional_dominance_loss",
    "persistent_seed_and_grow_loss",
    "regional_classification_losses",
    "softmin_foreground_rival_margin_loss",
    "uncancelled_channel_containment_loss",
]
