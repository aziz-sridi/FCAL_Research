from __future__ import annotations

import torch
from torch.nn import functional as F


def right_reasons_loss(
    logits: torch.Tensor,
    images: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Ross et al. RRR penalty on annotated irrelevant (background) pixels.

    The explanation is the input gradient of the sum of all class log
    probabilities.  Squared gradients are summed over irrelevant input
    dimensions and averaged over the batch, matching Eq. 1 of the paper.
    """
    if not images.requires_grad:
        raise ValueError("RRR requires input images with gradients enabled")
    log_probability_sum = F.log_softmax(logits.float(), dim=1).sum(dim=1).sum()
    gradient = torch.autograd.grad(
        log_probability_sum,
        images,
        create_graph=True,
        retain_graph=True,
    )[0]
    background = 1.0 - masks.float()
    return (gradient.float() * background).square().flatten(1).sum(dim=1).mean()


def normalized_positive_map(
    raw_map: torch.Tensor,
    size: tuple[int, int] | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    attention = F.relu(raw_map.float()).unsqueeze(1)
    if size is not None and attention.shape[-2:] != size:
        attention = F.interpolate(attention, size=size, mode="bilinear", align_corners=False)
    flat = attention.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (attention - minimum) / (maximum - minimum).clamp_min(float(epsilon))


def gain_losses(
    model,
    images: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    sigma: float = 0.25,
    omega: float = 100.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAIN attention-mining and GAIN-ext mask-supervision losses.

    For GAP classifiers, the Grad-CAM map has the same spatial values as the
    exact class activation map up to a positive constant.  Using the closed
    form here avoids an unnecessary nested gradient without changing GAIN's
    normalized attention map.
    """
    raw_map = model.raw_class_evidence(labels)[0]
    attention = normalized_positive_map(raw_map, images.shape[-2:], epsilon)
    soft_mask = torch.sigmoid(float(omega) * (attention - float(sigma)))
    masked_logits = model(images * (1.0 - soft_mask.to(dtype=images.dtype)))
    target_probability = masked_logits.float().softmax(dim=1).gather(
        1, labels[:, None],
    ).squeeze(1)
    mining = target_probability.mean()
    supervision = F.mse_loss(attention, masks.float())
    return mining, supervision


def camal_losses(
    model,
    labels: torch.Tensor,
    masks: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CAMAL regularizer and its in/out-mask attention components.

    CAMAL minimizes ``beta - alpha``, where alpha and beta are the mean
    normalized positive Grad-CAM values inside and outside the mask.  Every
    model used in the FCAL CUB matrix has an exact class-evidence map at its
    final spatial representation.  For the GAP CNNs and average-pooled ViTs,
    this map is equal to final-layer Grad-CAM up to a positive scalar; ReLU and
    per-sample min/max normalization remove that scalar.
    """
    inside_values = []
    outside_values = []
    masks = masks.float()
    for raw_map in model.raw_class_evidence(labels):
        attention = normalized_positive_map(raw_map, masks.shape[-2:], epsilon)
        foreground = masks
        background = 1.0 - foreground
        inside_values.append(
            (attention * foreground).flatten(1).sum(dim=1)
            / foreground.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        )
        outside_values.append(
            (attention * background).flatten(1).sum(dim=1)
            / background.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        )
    inside = torch.stack(inside_values, dim=1).mean()
    outside = torch.stack(outside_values, dim=1).mean()
    return outside - inside, inside, outside


def isnet_heatmap_loss(
    relevance: torch.Tensor,
    masks: torch.Tensor,
    cut_low: float = 1.0,
    cut_high: float = 25.0,
    background_decay: float = 0.0,
    background_scale: float = 1.0,
    foreground_weight: float = 1.0,
    background_weight: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Faster-ISNet LRP heatmap loss for one explanation per sample.

    This implements the authors' abs-RoI-mean normalization, background GWRP,
    bounded foreground-relevance range, and hybrid high-relevance penalty.
    """
    if relevance.ndim != 4:
        raise ValueError("ISNet relevance must have shape [batch, channels, height, width]")
    masks = masks.float()
    if masks.shape[-2:] != relevance.shape[-2:]:
        masks = F.adaptive_avg_pool2d(masks, relevance.shape[-2:])
        masks = (masks != 0).float()
    valid = masks.flatten(1).any(dim=1) & ~masks.flatten(1).all(dim=1)
    if not bool(valid.any()):
        return relevance.sum() * 0.0
    heatmap = relevance[valid].float().abs()
    mask = masks[valid].expand(-1, heatmap.shape[1], -1, -1)
    inverse = 1.0 - mask

    roi_mean = (heatmap * mask).sum(dim=(1, 2, 3), keepdim=True) / (
        mask.sum(dim=(1, 2, 3), keepdim=True) + float(epsilon)
    )
    normalized = heatmap / (roi_mean + float(epsilon))
    background = normalized * inverse
    if float(background_decay) == 0.0:
        pooled = background.flatten(2).amax(dim=2)
    else:
        ordered = background.flatten(2).sort(dim=2, descending=True).values
        weights = torch.pow(
            ordered.new_tensor(float(background_decay)),
            torch.arange(ordered.shape[-1], device=ordered.device, dtype=ordered.dtype),
        )
        pooled = (ordered * weights).sum(dim=2) / weights.sum()
    background_loss = torch.log1p(pooled / float(background_scale)).mean(dim=1)

    foreground_relevance = (heatmap * mask).sum(dim=(1, 2, 3))
    low = F.mse_loss(
        (foreground_relevance / float(cut_low)).clamp(max=1.0),
        torch.ones_like(foreground_relevance),
        reduction="none",
    )
    high_boundary = float(cut_low) + float(cut_high)
    use_l1 = (foreground_relevance.detach() > high_boundary).float()
    high_l2 = F.mse_loss(
        (foreground_relevance / float(cut_low)).clamp(min=float(cut_high / cut_low)),
        torch.full_like(foreground_relevance, float(cut_high / cut_low)),
        reduction="none",
    )
    high_l1 = F.l1_loss(
        foreground_relevance.clamp(min=high_boundary),
        torch.full_like(foreground_relevance, high_boundary),
        reduction="none",
    ) + 1.0
    foreground_loss = low + (1.0 - use_l1) * high_l2 + use_l1 * high_l1
    return (
        float(background_weight) * background_loss
        + float(foreground_weight) * foreground_loss
    ).mean()
