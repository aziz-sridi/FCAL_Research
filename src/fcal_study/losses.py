from __future__ import annotations

import torch
from torch.nn import functional as F


def article_tolerance_mask(mask: torch.Tensor, size: tuple[int, int], dilation_cells: int) -> torch.Tensor:
    """Bilinear resize followed by the article's clipped all-ones convolution."""
    resized = F.interpolate(mask.float(), size=size, mode="bilinear", align_corners=False)
    dilation = int(dilation_cells)
    if dilation > 0:
        kernel = torch.ones(
            (1, 1, 2 * dilation + 1, 2 * dilation + 1),
            device=resized.device,
            dtype=resized.dtype,
        )
        resized = F.conv2d(resized, kernel, padding=dilation).clamp_max_(1.0)
    return resized[:, 0]


def normalized_attention(attention: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    positive = F.relu(attention)
    return positive / (positive.flatten(1).sum(dim=1).view(-1, 1, 1) + epsilon)


def article_fcal_map_loss(
    attention: torch.Tensor,
    mask: torch.Tensor,
    dilation_cells: int = 3,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    flat_mask = mask.flatten(1)
    valid = flat_mask.any(dim=1) & ~flat_mask.all(dim=1)
    if not bool(valid.any()):
        return attention.sum() * 0.0
    attention = attention[valid]
    mask = mask[valid]
    tolerance = article_tolerance_mask(mask, attention.shape[-2:], dilation_cells)
    probability = normalized_attention(attention, epsilon)
    return (probability * (1.0 - tolerance)).flatten(1).sum(dim=1).mean()


def article_fcal_loss(
    attention_maps: list[torch.Tensor],
    mask: torch.Tensor,
    dilation_cells: int = 3,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    losses = torch.stack(
        [article_fcal_map_loss(item, mask, dilation_cells, epsilon) for item in attention_maps]
    )
    if len(attention_maps) == 1:
        return losses[0]
    weights = torch.arange(1, len(attention_maps) + 1, device=losses.device, dtype=losses.dtype)
    weights = weights / weights.sum()
    return (weights * losses).sum()


def smooth_normalized_attention(
    raw_attention: torch.Tensor,
    beta: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Dead-CAM-safe alternative: smooth positive activation followed by article normalization."""
    positive = F.softplus(raw_attention.float(), beta=float(beta))
    return positive / (positive.flatten(1).sum(dim=1).view(-1, 1, 1) + epsilon)


def smooth_fcal_map_loss(
    raw_attention: torch.Tensor,
    mask: torch.Tensor,
    dilation_cells: int = 3,
    beta: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    flat_mask = mask.flatten(1)
    valid = flat_mask.any(dim=1) & ~flat_mask.all(dim=1)
    if not bool(valid.any()):
        return raw_attention.sum() * 0.0
    raw_attention = raw_attention[valid]
    mask = mask[valid]
    tolerance = article_tolerance_mask(mask, raw_attention.shape[-2:], dilation_cells)
    probability = smooth_normalized_attention(raw_attention, beta, epsilon)
    return (probability * (1.0 - tolerance)).flatten(1).sum(dim=1).mean()


def smooth_fcal_loss(
    raw_attention_maps: list[torch.Tensor],
    mask: torch.Tensor,
    dilation_cells: int = 3,
    beta: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    losses = torch.stack([
        smooth_fcal_map_loss(item, mask, dilation_cells, beta, epsilon)
        for item in raw_attention_maps
    ])
    if len(raw_attention_maps) == 1:
        return losses[0]
    weights = torch.arange(1, len(raw_attention_maps) + 1, device=losses.device, dtype=losses.dtype)
    return (weights / weights.sum() * losses).sum()


def adaptive_lambda(
    ce_loss: torch.Tensor,
    fcal_loss: torch.Tensor,
    alpha: float,
    minimum: float,
    maximum: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    value = alpha * ce_loss.detach() / (fcal_loss.detach() + epsilon)
    return value.clamp(min=float(minimum), max=float(maximum))


def leakage(
    attention: torch.Tensor,
    mask: torch.Tensor,
    dilation_cells: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    tolerance = article_tolerance_mask(mask, attention.shape[-2:], dilation_cells)
    probability = normalized_attention(attention, epsilon)
    return (probability * (1.0 - tolerance)).flatten(1).sum(dim=1)
