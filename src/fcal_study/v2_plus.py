from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .v2 import detached_rms_normalize


def mean_preserving_evidence_proxy(scores: torch.Tensor) -> torch.Tensor:
    """Keep evidence values while removing its spatial-mean gradient.

    For a GAP classifier, the spatial mean of a class-evidence map is the
    bias-free class logit. This straight-through proxy is numerically equal
    to ``scores`` in the forward pass, but projects an auxiliary gradient onto
    the zero-spatial-mean subspace. The competitor objective can therefore
    move evidence between foreground and background without directly pushing
    the map's global class score up or down.
    """
    if scores.ndim != 3:
        raise ValueError("evidence maps must have shape [B,H,W]")
    values = scores.float()
    spatial_mean = values.mean(dim=(-2, -1), keepdim=True)
    return values + spatial_mean.detach() - spatial_mean


def top_competitor_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 2,
) -> torch.Tensor:
    """Select the strongest non-target classes without differentiating selection."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits and labels must have shapes [B,C] and [B]")
    if not 1 <= int(top_k) < logits.shape[1]:
        raise ValueError("top_k must be between 1 and C-1")
    candidates = logits.detach().float().clone()
    candidates.scatter_(1, labels[:, None], float("-inf"))
    return candidates.topk(int(top_k), dim=1).indices


def foreground_competitor_margin_loss(
    target_maps: list[torch.Tensor],
    competitor_maps: list[list[torch.Tensor]],
    masks: torch.Tensor,
    *,
    evidence_temperature: float = 1.0,
    margin: float = 0.25,
    loss_temperature: float = 0.25,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Require distributed target-over-rival evidence inside the foreground.

    This is a strict loss-only extension of V2. It consumes evidence maps from
    the existing forward pass and never creates, edits, or forwards a second
    image. V2 continues to control target evidence outside the foreground;
    this term addresses fine-grained class competition inside it.
    """
    if not target_maps:
        raise ValueError("V2+ requires at least one target evidence map")
    if not competitor_maps:
        raise ValueError("V2+ requires at least one competitor")
    if any(len(maps) != len(target_maps) for maps in competitor_maps):
        raise ValueError("target and competitor evidence depths must match")
    if float(evidence_temperature) <= 0 or float(loss_temperature) <= 0:
        raise ValueError("temperatures must be positive")

    flat_masks = masks.flatten(1)
    valid = flat_masks.any(dim=1) & ~flat_masks.all(dim=1)
    if not bool(valid.any()):
        terms = [scores.sum() for scores in target_maps]
        terms.extend(
            scores.sum() for maps in competitor_maps for scores in maps
        )
        return sum(terms) * 0.0

    valid_masks = masks[valid].float()
    layer_losses = []
    for layer_index, target_scores in enumerate(target_maps):
        target_scores = target_scores[valid]
        if target_scores.ndim != 3:
            raise ValueError("evidence maps must have shape [B,H,W]")
        foreground = F.adaptive_avg_pool2d(
            valid_masks, target_scores.shape[-2:]
        )[:, 0].clamp(0.0, 1.0)
        denominator = foreground.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        target_normalized = detached_rms_normalize(target_scores, epsilon)
        rival_losses = []
        for maps in competitor_maps:
            rival_scores = maps[layer_index][valid]
            if rival_scores.shape != target_scores.shape:
                raise ValueError("target and competitor evidence maps must align")
            rival_normalized = detached_rms_normalize(rival_scores, epsilon)
            bounded_margin = torch.tanh(
                (target_normalized - rival_normalized)
                / float(evidence_temperature)
            )
            support = (bounded_margin * foreground).flatten(1).sum(dim=1) / denominator
            shortfall = (float(margin) - support) / float(loss_temperature)
            rival_losses.append(
                (F.softplus(shortfall) * float(loss_temperature)).mean()
            )
        layer_losses.append(torch.stack(rival_losses).mean())

    losses = torch.stack(layer_losses)
    if len(layer_losses) == 1:
        return losses[0]
    weights = torch.arange(
        1, len(layer_losses) + 1, device=losses.device, dtype=losses.dtype
    )
    return (weights / weights.sum() * losses).sum()


@dataclass(frozen=True)
class GatedCompetitorMarginLoss:
    loss: torch.Tensor
    ungated_loss: torch.Tensor
    gate: torch.Tensor
    foreground_support: torch.Tensor
    target_confidence: torch.Tensor
    valid_mask_fraction: torch.Tensor


def gated_foreground_competitor_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_maps: list[torch.Tensor],
    competitor_maps: list[list[torch.Tensor]],
    masks: torch.Tensor,
    *,
    evidence_temperature: float = 1.0,
    margin: float = 0.25,
    loss_temperature: float = 0.25,
    confidence_floor: float = 0.25,
    confidence_ceiling: float = 0.80,
    confidence_gate_temperature: float = 0.05,
    support_gate_temperature: float = 0.10,
    epsilon: float = 1e-8,
) -> GatedCompetitorMarginLoss:
    """Apply V2+ only to moderate-confidence, weak-foreground examples."""
    if not target_maps:
        raise ValueError("gated V2+ requires at least one target evidence map")
    if not competitor_maps:
        raise ValueError("gated V2+ requires at least one competitor")
    if any(len(maps) != len(target_maps) for maps in competitor_maps):
        raise ValueError("target and competitor evidence depths must match")
    if (
        logits.ndim != 2
        or labels.shape != (logits.shape[0],)
        or logits.shape[0] != masks.shape[0]
    ):
        raise ValueError("logits, labels, and masks must share a batch dimension")
    if float(evidence_temperature) <= 0.0 or float(loss_temperature) <= 0.0:
        raise ValueError("loss temperatures must be positive")
    if (
        float(confidence_gate_temperature) <= 0.0
        or float(support_gate_temperature) <= 0.0
    ):
        raise ValueError("gate temperatures must be positive")
    if not 0.0 <= float(confidence_floor) < float(confidence_ceiling) <= 1.0:
        raise ValueError("confidence gate must satisfy 0 <= floor < ceiling <= 1")

    flat_masks = masks.flatten(1)
    valid = flat_masks.any(dim=1) & ~flat_masks.all(dim=1)
    valid_masks = masks[valid].float()
    layer_losses = []
    layer_supports = []
    for layer_index, target_scores in enumerate(target_maps):
        target_scores = target_scores[valid]
        if target_scores.ndim != 3:
            raise ValueError("evidence maps must have shape [B,H,W]")
        foreground = F.adaptive_avg_pool2d(
            valid_masks, target_scores.shape[-2:]
        )[:, 0].clamp(0.0, 1.0)
        denominator = foreground.flatten(1).sum(dim=1).clamp_min(float(epsilon))
        target_normalized = detached_rms_normalize(target_scores, epsilon)
        rival_supports = []
        rival_losses = []
        for maps in competitor_maps:
            rival_scores = maps[layer_index][valid]
            if rival_scores.shape != target_scores.shape:
                raise ValueError("target and competitor evidence maps must align")
            rival_normalized = detached_rms_normalize(rival_scores, epsilon)
            bounded_margin = torch.tanh(
                (target_normalized - rival_normalized)
                / float(evidence_temperature)
            )
            support = (
                (bounded_margin * foreground).flatten(1).sum(dim=1)
                / denominator
            )
            shortfall = (float(margin) - support) / float(loss_temperature)
            rival_supports.append(support)
            rival_losses.append(
                F.softplus(shortfall) * float(loss_temperature)
            )
        layer_losses.append(torch.stack(rival_losses).mean(dim=0))
        layer_supports.append(torch.stack(rival_supports).amin(dim=0))

    losses = torch.stack(layer_losses)
    supports = torch.stack(layer_supports)
    if len(layer_losses) == 1:
        per_sample_loss = losses[0]
        per_sample_support = supports[0]
    else:
        weights = torch.arange(
            1, len(layer_losses) + 1, device=losses.device, dtype=losses.dtype
        )
        weights = weights[:, None] / weights.sum()
        per_sample_loss = (weights * losses).sum(dim=0)
        per_sample_support = (weights * supports).sum(dim=0)

    with torch.no_grad():
        target_confidence = logits.float().softmax(dim=1).gather(
            1, labels[:, None]
        ).squeeze(1)[valid]
        above_floor = torch.sigmoid(
            (target_confidence - float(confidence_floor))
            / float(confidence_gate_temperature)
        )
        below_ceiling = torch.sigmoid(
            (float(confidence_ceiling) - target_confidence)
            / float(confidence_gate_temperature)
        )
        insufficient_support = torch.sigmoid(
            (float(margin) - per_sample_support.detach())
            / float(support_gate_temperature)
        )
        gate = above_floor * below_ceiling * insufficient_support

    valid_count = valid.float().sum().clamp_min(1.0)
    graph_terms = [scores.sum() for scores in target_maps]
    graph_terms.extend(
        scores.sum() for maps in competitor_maps for scores in maps
    )
    graph_zero = sum(graph_terms) * 0.0
    return GatedCompetitorMarginLoss(
        loss=(gate * per_sample_loss).sum() / valid_count + graph_zero,
        ungated_loss=per_sample_loss.sum() / valid_count + graph_zero,
        gate=gate.sum() / valid_count,
        foreground_support=per_sample_support.detach().sum() / valid_count,
        target_confidence=target_confidence.sum() / valid_count,
        valid_mask_fraction=valid.float().mean(),
    )


__all__ = [
    "GatedCompetitorMarginLoss",
    "foreground_competitor_margin_loss",
    "gated_foreground_competitor_margin_loss",
    "mean_preserving_evidence_proxy",
    "top_competitor_labels",
]
