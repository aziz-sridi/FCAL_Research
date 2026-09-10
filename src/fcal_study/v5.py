from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .v2 import detached_rms_normalize


@dataclass(frozen=True)
class RegionalMarginCertificate:
    loss: torch.Tensor
    certificate: torch.Tensor
    foreground_tail: torch.Tensor
    background_tail: torch.Tensor


def _valid_masks(masks: torch.Tensor) -> torch.Tensor:
    flat = masks.flatten(1)
    return flat.any(dim=1) & ~flat.all(dim=1)


def _entropic_tail(
    values: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
    *,
    largest: bool,
    epsilon: float,
) -> torch.Tensor:
    if float(temperature) <= 0:
        raise ValueError("tail temperatures must be positive")
    flat_values = values.flatten(1)
    flat_weights = weights.float().flatten(1).clamp_min(0.0)
    signed = flat_values if largest else -flat_values
    log_weights = torch.where(
        flat_weights > 0,
        flat_weights.clamp_min(float(epsilon)).log(),
        torch.full_like(flat_weights, float("-inf")),
    )
    log_total = flat_weights.sum(dim=1).clamp_min(float(epsilon)).log()
    result = float(temperature) * (
        torch.logsumexp(log_weights + signed / float(temperature), dim=1)
        - log_total
    )
    return result if largest else -result


def regional_margin_certificate_from_maps(
    target_maps: list[torch.Tensor],
    competitor_maps: list[list[torch.Tensor]],
    masks: torch.Tensor,
    *,
    foreground_temperature: float = 0.5,
    background_temperature: float = 0.5,
    background_scale: float = 0.5,
    margin: float = 0.0,
    temperature: float = 0.25,
    epsilon: float = 1e-8,
) -> RegionalMarginCertificate:
    """Certify foreground-sufficient spatial margins against null and real rivals.

    The null rival has zero spatial evidence, so its target-vs-null field
    controls class-specific leakage. Each real rival contributes a pairwise
    target-minus-rival field. The certificate is the lowest foreground tail
    minus a scaled upper tail of absolute background influence, minimized over
    all rivals.
    """
    if not target_maps:
        raise ValueError("V5 requires at least one target evidence map")
    if float(background_scale) < 0:
        raise ValueError("background scale must be non-negative")
    if float(temperature) <= 0:
        raise ValueError("certificate temperature must be positive")
    if any(len(maps) != len(target_maps) for maps in competitor_maps):
        raise ValueError("target and competitor evidence depths must match")
    valid = _valid_masks(masks)
    if not bool(valid.any()):
        zero = sum(scores.sum() for scores in target_maps) * 0.0
        return RegionalMarginCertificate(zero, zero, zero, zero)

    valid_masks = masks[valid]
    layer_losses = []
    layer_certificates = []
    layer_foreground = []
    layer_background = []
    for layer_index, target_scores in enumerate(target_maps):
        target_scores = target_scores[valid].float()
        fields = [target_scores]
        fields.extend(
            target_scores - maps[layer_index][valid].float()
            for maps in competitor_maps
        )
        foreground = F.adaptive_avg_pool2d(
            valid_masks.float(), target_scores.shape[-2:],
        )[:, 0].clamp(0.0, 1.0)
        background = 1.0 - foreground
        rival_certificates = []
        rival_foreground = []
        rival_background = []
        for field in fields:
            normalized = detached_rms_normalize(field, epsilon)
            foreground_tail = _entropic_tail(
                normalized,
                foreground,
                foreground_temperature,
                largest=False,
                epsilon=epsilon,
            )
            background_tail = _entropic_tail(
                normalized.abs(),
                background,
                background_temperature,
                largest=True,
                epsilon=epsilon,
            )
            rival_foreground.append(foreground_tail)
            rival_background.append(background_tail)
            rival_certificates.append(
                foreground_tail - float(background_scale) * background_tail
            )
        certificates = torch.stack(rival_certificates, dim=1)
        worst_rival = certificates.min(dim=1).values
        layer_losses.append(
            (F.softplus((float(margin) - worst_rival) / float(temperature))
             * float(temperature)).mean()
        )
        worst_indices = certificates.argmin(dim=1, keepdim=True)
        foreground_values = torch.stack(rival_foreground, dim=1)
        background_values = torch.stack(rival_background, dim=1)
        layer_certificates.append(worst_rival.mean())
        layer_foreground.append(
            foreground_values.gather(1, worst_indices).mean()
        )
        layer_background.append(
            background_values.gather(1, worst_indices).mean()
        )

    weights = torch.arange(
        1, len(layer_losses) + 1,
        device=layer_losses[0].device,
        dtype=layer_losses[0].dtype,
    )
    weights = weights / weights.sum()

    def weighted(values: list[torch.Tensor]) -> torch.Tensor:
        return (weights * torch.stack(values)).sum()

    return RegionalMarginCertificate(
        loss=weighted(layer_losses),
        certificate=weighted(layer_certificates),
        foreground_tail=weighted(layer_foreground),
        background_tail=weighted(layer_background),
    )


def regional_margin_certificate_loss(
    model,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    top_k_competitors: int = 2,
    foreground_temperature: float = 0.5,
    background_temperature: float = 0.5,
    background_scale: float = 0.5,
    margin: float = 0.0,
    temperature: float = 0.25,
    epsilon: float = 1e-8,
) -> RegionalMarginCertificate:
    """Build top-rival fields from one forward pass and evaluate the certificate."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits and labels must have shapes [B,C] and [B]")
    if not 1 <= int(top_k_competitors) < logits.shape[1]:
        raise ValueError("top_k_competitors must be between 1 and C-1")
    detached = logits.detach().float().clone()
    detached.scatter_(1, labels[:, None], float("-inf"))
    competitors = detached.topk(int(top_k_competitors), dim=1).indices
    target_maps = model.raw_class_evidence(labels)
    competitor_maps = [
        model.raw_class_evidence(competitors[:, rank])
        for rank in range(competitors.shape[1])
    ]
    return regional_margin_certificate_from_maps(
        target_maps,
        competitor_maps,
        masks,
        foreground_temperature=foreground_temperature,
        background_temperature=background_temperature,
        background_scale=background_scale,
        margin=margin,
        temperature=temperature,
        epsilon=epsilon,
    )


__all__ = [
    "RegionalMarginCertificate",
    "regional_margin_certificate_from_maps",
    "regional_margin_certificate_loss",
]
