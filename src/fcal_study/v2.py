from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .regions import projected_region_weights, ramp_weight


def resolution_aware_tolerance_pixels(
    input_size: tuple[int, int],
    map_size: tuple[int, int],
    cell_fraction: float = 0.5,
    minimum_pixels: int = 0,
    maximum_pixels: int | None = None,
) -> int:
    """Convert a fraction of one evidence-map cell into input-space pixels."""
    if min(*input_size, *map_size) <= 0:
        raise ValueError("input and evidence-map dimensions must be positive")
    if float(cell_fraction) < 0:
        raise ValueError("cell_fraction must be non-negative")
    cell_height = float(input_size[0]) / float(map_size[0])
    cell_width = float(input_size[1]) / float(map_size[1])
    tolerance = max(int(minimum_pixels), math.ceil(float(cell_fraction) * max(cell_height, cell_width)))
    if maximum_pixels is not None:
        tolerance = min(tolerance, int(maximum_pixels))
    return tolerance


def detached_rms_normalize(scores: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize each signed map without letting the loss manipulate its scale estimate."""
    scale = scores.float().square().mean(dim=(-2, -1), keepdim=True).add(float(epsilon)).sqrt().detach()
    return scores.float() / scale


def distributed_foreground_support(
    scores: torch.Tensor,
    foreground: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Bounded mean signed evidence over the complete soft foreground region.

    Unlike log-mean-exp, a single extreme cell cannot satisfy this statistic.
    """
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    bounded = torch.tanh(detached_rms_normalize(scores, epsilon) / float(temperature))
    weights = foreground.float().clamp(0.0, 1.0)
    numerator = (bounded * weights).flatten(1).sum(dim=1)
    denominator = weights.flatten(1).sum(dim=1).clamp_min(float(epsilon))
    return numerator / denominator


def support_aware_fcal_losses(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    support_temperature: float,
    support_margin: float,
    tolerance_cell_fraction: float,
    minimum_tolerance_pixels: int = 0,
    maximum_tolerance_pixels: int | None = None,
    fixed_tolerance_pixels: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Article positive containment plus scale-free distributed signed support."""
    if not raw_score_maps:
        raise ValueError("FCAL V2 requires at least one signed spatial evidence map")

    flat_masks = masks.flatten(1)
    valid = flat_masks.any(dim=1) & ~flat_masks.all(dim=1)
    if not bool(valid.any()):
        zero = sum(scores.sum() for scores in raw_score_maps) * 0.0
        return zero, zero
    masks = masks[valid]
    raw_score_maps = [scores[valid] for scores in raw_score_maps]

    outside_losses = []
    support_losses = []
    input_size = (int(masks.shape[-2]), int(masks.shape[-1]))
    for scores in raw_score_maps:
        tolerance = (
            int(fixed_tolerance_pixels)
            if fixed_tolerance_pixels is not None
            else resolution_aware_tolerance_pixels(
                input_size,
                (int(scores.shape[-2]), int(scores.shape[-1])),
                tolerance_cell_fraction,
                minimum_tolerance_pixels,
                maximum_tolerance_pixels,
            )
        )
        foreground, _, background = projected_region_weights(masks, scores.shape[-2:], tolerance)

        positive = F.relu(scores.float())
        probability = positive / (
            positive.flatten(1).sum(dim=1).view(-1, 1, 1) + float(epsilon)
        )
        outside_losses.append((probability * background).flatten(1).sum(dim=1).mean())

        support = distributed_foreground_support(
            scores, foreground, support_temperature, epsilon,
        )
        support_losses.append(F.softplus(float(support_margin) - support).mean())

    outside = torch.stack(outside_losses)
    support = torch.stack(support_losses)
    if len(raw_score_maps) == 1:
        return outside[0], support[0]
    weights = torch.arange(1, len(raw_score_maps) + 1, device=outside.device, dtype=outside.dtype)
    weights = weights / weights.sum()
    return (weights * outside).sum(), (weights * support).sum()


__all__ = [
    "detached_rms_normalize",
    "distributed_foreground_support",
    "ramp_weight",
    "resolution_aware_tolerance_pixels",
    "support_aware_fcal_losses",
]
