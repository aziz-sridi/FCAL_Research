from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GradientStats:
    ce_norm: float
    auxiliary_norm_before: float
    auxiliary_norm_after: float
    ratio_before: float
    ratio_after: float
    cosine_before: float
    conflict_projected: bool
    scale: float


def _zeros_or(gradient: torch.Tensor | None, parameter: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(parameter) if gradient is None else gradient


def gradient_safe_backward(
    ce: torch.Tensor,
    weighted_auxiliary: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    maximum_gradient_ratio: float,
    accumulation_steps: int = 1,
    project_conflicts: bool = True,
    epsilon: float = 1e-12,
) -> GradientStats:
    """Backpropagate CE plus a globally capped, optionally projected FCAL gradient."""
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("No trainable parameters were supplied")
    accumulation = int(accumulation_steps)
    if accumulation <= 0:
        raise ValueError("accumulation_steps must be positive")
    ce_gradients = torch.autograd.grad(
        ce / accumulation,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        weighted_auxiliary / accumulation,
        params,
        allow_unused=True,
    )
    ce_gradients = tuple(_zeros_or(g, p) for g, p in zip(ce_gradients, params, strict=True))
    auxiliary_gradients = tuple(
        _zeros_or(g, p) for g, p in zip(auxiliary_gradients, params, strict=True)
    )
    ce_square = sum(g.float().square().sum() for g in ce_gradients)
    auxiliary_square = sum(g.float().square().sum() for g in auxiliary_gradients)
    dot = sum(
        (left.float() * right.float()).sum()
        for left, right in zip(ce_gradients, auxiliary_gradients, strict=True)
    )
    ce_norm = ce_square.sqrt()
    auxiliary_norm = auxiliary_square.sqrt()
    cosine = dot / (ce_norm * auxiliary_norm).clamp_min(float(epsilon))
    conflict = bool(project_conflicts and float(dot.detach()) < 0.0 and float(ce_square.detach()) > epsilon)
    if conflict:
        coefficient = dot / ce_square.clamp_min(float(epsilon))
        projected = tuple(
            auxiliary - coefficient.to(dtype=auxiliary.dtype) * primary
            for primary, auxiliary in zip(ce_gradients, auxiliary_gradients, strict=True)
        )
    else:
        projected = auxiliary_gradients
    projected_norm = sum(g.float().square().sum() for g in projected).sqrt()
    ratio_before = auxiliary_norm / ce_norm.clamp_min(float(epsilon))
    projected_ratio = projected_norm / ce_norm.clamp_min(float(epsilon))
    cap = max(0.0, float(maximum_gradient_ratio))
    scale = min(1.0, cap / max(float(projected_ratio.detach()), float(epsilon)))
    for parameter, primary, auxiliary in zip(params, ce_gradients, projected, strict=True):
        combined = primary + float(scale) * auxiliary
        if parameter.grad is None:
            parameter.grad = combined
        else:
            parameter.grad.add_(combined)
    final_auxiliary_norm = float(projected_norm.detach()) * scale
    ratio_after = final_auxiliary_norm / max(float(ce_norm.detach()), float(epsilon))
    return GradientStats(
        ce_norm=float(ce_norm.detach()),
        auxiliary_norm_before=float(auxiliary_norm.detach()),
        auxiliary_norm_after=final_auxiliary_norm,
        ratio_before=float(ratio_before.detach()),
        ratio_after=ratio_after,
        cosine_before=float(cosine.detach()),
        conflict_projected=conflict,
        scale=scale,
    )
