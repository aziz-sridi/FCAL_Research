from __future__ import annotations

import torch
from torch.nn import functional as F

from .losses import article_tolerance_mask
from .regions import projected_region_weights
from .v2 import (
    detached_rms_normalize,
    resolution_aware_tolerance_pixels,
    support_aware_fcal_losses,
)


def different_label_donors(
    labels: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose a one-to-one donor assignment with as few same-label pairs as possible."""
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch]")
    batch_size = int(labels.numel())
    if batch_size < 2:
        return torch.arange(batch_size, device=labels.device)
    del generator  # Batch shuffling already randomizes which sample represents each label.
    order = torch.argsort(labels, stable=True)
    _, counts = torch.unique_consecutive(labels[order], return_counts=True)
    rotation = int(counts.max())
    rotated = order.roll(-rotation)
    donors = torch.empty_like(order)
    donors[order] = rotated
    return donors


def background_counterfactual(
    images: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    *,
    boundary_guard_pixels: int = 4,
    generator: torch.Generator | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace background context while preserving every annotated foreground pixel.

    Donor foregrounds are replaced by the donor's mean background color before
    compositing, preventing a second labeled object from entering the new image.
    A soft guard band outside the target mask reduces hard mask-edge artifacts.
    """
    if images.ndim != 4 or masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError("images and masks must be [B,C,H,W] and [B,1,H,W]")
    if images.shape[0] != masks.shape[0] or images.shape[-2:] != masks.shape[-2:]:
        raise ValueError("images and masks must have matching batch and spatial dimensions")
    if labels.shape != (images.shape[0],):
        raise ValueError("labels must have shape [batch]")
    guard = int(boundary_guard_pixels)
    if guard < 0:
        raise ValueError("boundary_guard_pixels must be non-negative")

    foreground = masks.to(dtype=images.dtype).clamp(0.0, 1.0)
    donors = different_label_donors(labels, generator)
    donor_images = images[donors]
    donor_foreground = foreground[donors]
    donor_background = 1.0 - donor_foreground
    background_count = donor_background.sum(dim=(-2, -1), keepdim=True).clamp_min(
        float(epsilon)
    )
    background_mean = (donor_images * donor_background).sum(
        dim=(-2, -1), keepdim=True
    ) / background_count
    donor_canvas = donor_images * donor_background + background_mean * donor_foreground

    protection = foreground
    if guard > 0:
        kernel = 2 * guard + 1
        expanded = F.max_pool2d(foreground, kernel, stride=1, padding=guard)
        softened = F.avg_pool2d(expanded, kernel, stride=1, padding=guard)
        protection = torch.maximum(foreground, softened).clamp(0.0, 1.0)
    counterfactual = images * protection + donor_canvas * (1.0 - protection)
    mismatch_fraction = (labels[donors] != labels).float().mean()
    return counterfactual, donors, mismatch_fraction


def foreground_corruption(
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    noise_scale: float = 0.20,
) -> torch.Tensor:
    """Apply one mild label-preserving corruption inside each foreground mask."""
    if images.ndim != 4 or masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError("images and masks must be [B,C,H,W] and [B,1,H,W]")
    foreground = masks.to(dtype=images.dtype).clamp(0.0, 1.0)
    blurred = F.avg_pool2d(images, kernel_size=7, stride=1, padding=3)
    noise = images + float(noise_scale) * torch.randn(
        images.shape,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    channel_mixed = 0.75 * images + 0.25 * images.roll(1, dims=1)
    candidates = torch.stack((blurred, noise, channel_mixed), dim=1)
    choices = torch.randint(
        0,
        candidates.shape[1],
        (images.shape[0],),
        device=images.device,
        generator=generator,
    )
    transformed = candidates[
        torch.arange(images.shape[0], device=images.device), choices
    ]
    return images * (1.0 - foreground) + transformed * foreground


def balanced_regional_counterfactual(
    images: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    *,
    foreground_fraction: float,
    boundary_guard_pixels: int = 4,
    foreground_noise_scale: float = 0.20,
    generator: torch.Generator | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix background interventions and mild foreground corruptions in one view."""
    fraction = float(foreground_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("foreground_fraction must be in [0, 1]")
    background, _, mismatch = background_counterfactual(
        images,
        masks,
        labels,
        boundary_guard_pixels=boundary_guard_pixels,
        generator=generator,
        epsilon=epsilon,
    )
    foreground = foreground_corruption(
        images,
        masks,
        generator=generator,
        noise_scale=foreground_noise_scale,
    )
    foreground_count = round(images.shape[0] * fraction)
    foreground_samples = torch.zeros(
        images.shape[0], dtype=torch.bool, device=images.device
    )
    if foreground_count:
        selected = torch.randperm(
            images.shape[0], device=images.device, generator=generator
        )[:foreground_count]
        foreground_samples[selected] = True
    mixed = torch.where(foreground_samples[:, None, None, None], foreground, background)
    background_samples = ~foreground_samples
    effective_mismatch = mismatch * background_samples.float().mean()
    return mixed, background_samples, effective_mismatch


def prediction_consistency_loss(
    clean_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    valid_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distill the clean prediction into the background-counterfactual view."""
    if clean_logits.shape != counterfactual_logits.shape:
        raise ValueError("clean and counterfactual logits must have identical shapes")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if valid_samples is not None:
        if valid_samples.shape != (clean_logits.shape[0],):
            raise ValueError("valid_samples must have shape [batch]")
        valid = valid_samples.to(device=clean_logits.device, dtype=torch.bool)
        if not bool(valid.any()):
            return (clean_logits.sum() + counterfactual_logits.sum()) * 0.0
        clean_logits = clean_logits[valid]
        counterfactual_logits = counterfactual_logits[valid]
    temperature = float(temperature)
    teacher = F.softmax(clean_logits.detach().float() / temperature, dim=1)
    student = F.log_softmax(counterfactual_logits.float() / temperature, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature**2


def strict_shell_containment(
    raw_score_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    tolerance_cell_fraction: float,
    minimum_tolerance_pixels: int = 0,
    maximum_tolerance_pixels: int | None = None,
    fixed_tolerance_pixels: int | None = None,
    fixed_tolerance_cells: int | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Penalize normalized positive evidence only in V2's tolerance shell."""
    if not raw_score_maps:
        raise ValueError("strict shell containment requires at least one evidence map")
    if fixed_tolerance_pixels is not None and fixed_tolerance_cells is not None:
        raise ValueError("Choose fixed_tolerance_pixels or fixed_tolerance_cells, not both")
    flat_masks = masks.flatten(1)
    valid = flat_masks.any(dim=1) & ~flat_masks.all(dim=1)
    if not bool(valid.any()):
        return sum(scores.sum() for scores in raw_score_maps) * 0.0
    masks = masks[valid]
    maps = [scores[valid] for scores in raw_score_maps]
    input_size = (int(masks.shape[-2]), int(masks.shape[-1]))
    losses = []
    for scores in maps:
        map_size = (int(scores.shape[-2]), int(scores.shape[-1]))
        if fixed_tolerance_cells is not None:
            strict = article_tolerance_mask(masks, map_size, 0)
            tolerant = article_tolerance_mask(
                masks, map_size, int(fixed_tolerance_cells)
            )
        else:
            tolerance = (
                int(fixed_tolerance_pixels)
                if fixed_tolerance_pixels is not None
                else resolution_aware_tolerance_pixels(
                    input_size,
                    map_size,
                    tolerance_cell_fraction,
                    minimum_tolerance_pixels,
                    maximum_tolerance_pixels,
                )
            )
            strict, tolerant, _ = projected_region_weights(
                masks, map_size, tolerance
            )
        shell = (tolerant - strict).clamp(0.0, 1.0)
        positive = F.relu(scores.float())
        probability = positive / (
            positive.flatten(1).sum(dim=1).view(-1, 1, 1) + float(epsilon)
        )
        losses.append((probability * shell).flatten(1).sum(dim=1).mean())
    stacked = torch.stack(losses)
    if len(losses) == 1:
        return stacked[0]
    weights = torch.arange(
        1, len(losses) + 1, device=stacked.device, dtype=stacked.dtype
    )
    return (weights / weights.sum() * stacked).sum()


def foreground_evidence_consistency(
    clean_maps: list[torch.Tensor],
    counterfactual_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
    valid_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match scale-free signed evidence inside unchanged foreground pixels."""
    if not clean_maps or len(clean_maps) != len(counterfactual_maps):
        raise ValueError("clean and counterfactual evidence lists must be non-empty and aligned")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    layer_losses = []
    for clean, counterfactual in zip(clean_maps, counterfactual_maps, strict=True):
        if clean.shape != counterfactual.shape or clean.ndim != 3:
            raise ValueError("paired evidence maps must have identical [B,H,W] shapes")
        foreground = F.adaptive_avg_pool2d(masks.float(), clean.shape[-2:])[:, 0]
        foreground_cells = foreground.flatten(1).sum(1)
        valid = foreground_cells > float(epsilon)
        if valid_samples is not None:
            if valid_samples.shape != valid.shape:
                raise ValueError("valid_samples must have shape [batch]")
            valid = valid & valid_samples.to(device=valid.device, dtype=torch.bool)
        if not bool(valid.any()):
            layer_losses.append((clean.sum() + counterfactual.sum()) * 0.0)
            continue
        clean_bounded = torch.tanh(
            detached_rms_normalize(clean, epsilon) / float(temperature)
        )
        counterfactual_bounded = torch.tanh(
            detached_rms_normalize(counterfactual, epsilon) / float(temperature)
        )
        difference = F.smooth_l1_loss(
            clean_bounded, counterfactual_bounded, reduction="none"
        )
        per_sample = (difference * foreground).flatten(1).sum(1) / foreground_cells.clamp_min(
            float(epsilon)
        )
        layer_losses.append(per_sample[valid].mean())
    losses = torch.stack(layer_losses)
    if len(layer_losses) == 1:
        return losses[0]
    weights = torch.arange(1, len(layer_losses) + 1, device=losses.device, dtype=losses.dtype)
    return (weights / weights.sum() * losses).sum()


def counterfactual_support_losses(
    clean_maps: list[torch.Tensor],
    counterfactual_maps: list[torch.Tensor],
    masks: torch.Tensor,
    *,
    support_temperature: float,
    support_margin: float,
    tolerance_cell_fraction: float,
    evidence_consistency_temperature: float = 1.0,
    evidence_consistency_samples: torch.Tensor | None = None,
    minimum_tolerance_pixels: int = 0,
    maximum_tolerance_pixels: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the complete V2 objective to both views and couple their foreground evidence."""
    common = {
        "support_temperature": support_temperature,
        "support_margin": support_margin,
        "tolerance_cell_fraction": tolerance_cell_fraction,
        "minimum_tolerance_pixels": minimum_tolerance_pixels,
        "maximum_tolerance_pixels": maximum_tolerance_pixels,
        "epsilon": epsilon,
    }
    clean_outside, clean_support = support_aware_fcal_losses(clean_maps, masks, **common)
    counterfactual_outside, counterfactual_support = support_aware_fcal_losses(
        counterfactual_maps, masks, **common
    )
    consistency = foreground_evidence_consistency(
        clean_maps,
        counterfactual_maps,
        masks,
        temperature=evidence_consistency_temperature,
        epsilon=epsilon,
        valid_samples=evidence_consistency_samples,
    )
    return (
        0.5 * (clean_outside + counterfactual_outside),
        0.5 * (clean_support + counterfactual_support),
        consistency,
    )


__all__ = [
    "background_counterfactual",
    "balanced_regional_counterfactual",
    "counterfactual_support_losses",
    "different_label_donors",
    "foreground_corruption",
    "foreground_evidence_consistency",
    "prediction_consistency_loss",
    "strict_shell_containment",
]
