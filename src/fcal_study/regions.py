from __future__ import annotations

import torch
from torch.nn import functional as F


def projected_region_weights(
    masks: torch.Tensor,
    size: tuple[int, int],
    tolerance_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project strict and tolerance-expanded foreground masks to an evidence map."""
    foreground = masks.float().clamp(0.0, 1.0)
    tolerance = int(tolerance_pixels)
    tolerant = foreground
    if tolerance > 0:
        kernel = 2 * tolerance + 1
        tolerant = F.max_pool2d(foreground, kernel_size=kernel, stride=1, padding=tolerance)
    strict_grid = F.adaptive_avg_pool2d(foreground, size)[:, 0]
    tolerant_grid = F.adaptive_avg_pool2d(tolerant, size)[:, 0]
    return strict_grid, tolerant_grid, 1.0 - tolerant_grid


def ramp_weight(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """Linearly introduce an FCAL objective after the shared CE warm-up."""
    if epoch < int(warmup_epochs):
        return 0.0
    return min(1.0, (epoch - int(warmup_epochs) + 1) / max(1, int(ramp_epochs)))


__all__ = ["projected_region_weights", "ramp_weight"]
