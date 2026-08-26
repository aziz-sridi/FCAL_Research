from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MapMeasurements:
    outside: torch.Tensor
    coverage: torch.Tensor
    valid: torch.Tensor
    zero_map: torch.Tensor
    tolerance_pixels: int


@dataclass
class FCALOutput:
    auxiliary: torch.Tensor
    containment: torch.Tensor
    support: torch.Tensor
    mean_outside: torch.Tensor
    mean_coverage: torch.Tensor
    valid_fraction: torch.Tensor
    zero_map_fraction: torch.Tensor


def resolution_aware_tolerance_pixels(
    input_size: tuple[int, int],
    map_size: tuple[int, int],
    cell_fraction: float = 0.5,
    minimum_pixels: int = 0,
    maximum_pixels: int | None = None,
    maximum_input_fraction: float | None = None,
) -> int:
    """Convert a dimensionless evidence-cell fraction to input-space pixels."""
    if min(*input_size, *map_size) <= 0:
        raise ValueError("input and evidence-map dimensions must be positive")
    if float(cell_fraction) < 0:
        raise ValueError("cell_fraction must be non-negative")
    cell_height = float(input_size[0]) / float(map_size[0])
    cell_width = float(input_size[1]) / float(map_size[1])
    tolerance = max(
        int(minimum_pixels),
        math.ceil(float(cell_fraction) * max(cell_height, cell_width)),
    )
    limits = []
    if maximum_pixels is not None:
        limits.append(int(maximum_pixels))
    if maximum_input_fraction is not None:
        if not 0 <= float(maximum_input_fraction) <= 1:
            raise ValueError("maximum_input_fraction must be in [0, 1]")
        limits.append(math.floor(min(input_size) * float(maximum_input_fraction)))
    if limits:
        tolerance = min(tolerance, *limits)
    return tolerance


def _project_regions(
    masks: torch.Tensor,
    size: tuple[int, int],
    tolerance_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    foreground = masks.float().clamp(0.0, 1.0)
    tolerance = int(tolerance_pixels)
    tolerant = foreground
    if tolerance > 0:
        kernel = 2 * tolerance + 1
        tolerant = F.max_pool2d(foreground, kernel, stride=1, padding=tolerance)
    strict_grid = F.adaptive_avg_pool2d(foreground, size)[:, 0]
    tolerant_grid = F.adaptive_avg_pool2d(tolerant, size)[:, 0]
    return strict_grid, 1.0 - tolerant_grid


def _detached_rms_normalize(scores: torch.Tensor, epsilon: float) -> torch.Tensor:
    scale = scores.float().square().mean(dim=(-2, -1), keepdim=True).add(float(epsilon)).sqrt().detach()
    return scores.float() / scale


def measure_maps(
    scores: torch.Tensor,
    masks: torch.Tensor,
    *,
    tolerance_cell_fraction: float,
    evidence_threshold: float,
    coverage_temperature: float,
    minimum_tolerance_pixels: int = 0,
    maximum_tolerance_pixels: int | None = None,
    maximum_tolerance_fraction: float | None = 0.10,
    minimum_foreground_cells: float = 1.0,
    minimum_background_cells: float = 1.0,
    epsilon: float = 1e-8,
) -> MapMeasurements:
    """Measure leakage and distributed support without assuming a CAM size."""
    if scores.ndim != 3 or masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError("scores must be [B,H,W] and masks must be [B,1,H,W]")
    input_size = (int(masks.shape[-2]), int(masks.shape[-1]))
    map_size = (int(scores.shape[-2]), int(scores.shape[-1]))
    tolerance = resolution_aware_tolerance_pixels(
        input_size,
        map_size,
        tolerance_cell_fraction,
        minimum_tolerance_pixels,
        maximum_tolerance_pixels,
        maximum_tolerance_fraction,
    )
    foreground, background = _project_regions(masks, map_size, tolerance)
    foreground_cells = foreground.flatten(1).sum(1)
    background_cells = background.flatten(1).sum(1)
    valid = (
        foreground_cells >= float(minimum_foreground_cells)
    ) & (
        background_cells >= float(minimum_background_cells)
    )

    positive = F.relu(scores.float())
    positive_mass = positive.flatten(1).sum(1)
    probability = positive / (positive_mass[:, None, None] + float(epsilon))
    outside = (probability * background).flatten(1).sum(1)

    temperature = float(coverage_temperature)
    if temperature <= 0:
        raise ValueError("coverage_temperature must be positive")
    normalized = _detached_rms_normalize(scores, epsilon)
    active = torch.sigmoid((normalized - float(evidence_threshold)) / temperature)
    coverage = (active * foreground).flatten(1).sum(1) / foreground_cells.clamp_min(float(epsilon))
    return MapMeasurements(
        outside=outside,
        coverage=coverage,
        valid=valid,
        zero_map=positive_mass <= float(epsilon),
        tolerance_pixels=tolerance,
    )


def _valid_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class UniversalFCAL(nn.Module):
    """Target-gated, resolution-independent FCAL with invalid-mask protection."""

    def __init__(
        self,
        *,
        allowed_outside: float,
        target_coverage: float,
        evidence_threshold: float = 2.0,
        coverage_temperature: float = 0.25,
        confidence_gamma: float = 1.0,
        support_mix: float = 0.5,
        mode: str = "combined",
        tolerance_cell_fraction: float = 0.5,
        minimum_tolerance_pixels: int = 0,
        maximum_tolerance_pixels: int | None = None,
        maximum_tolerance_fraction: float | None = 0.10,
        minimum_foreground_cells: float = 1.0,
        minimum_background_cells: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if mode not in {"outside", "support", "combined"}:
            raise ValueError("mode must be outside, support, or combined")
        self.allowed_outside = float(allowed_outside)
        self.target_coverage = float(target_coverage)
        self.evidence_threshold = float(evidence_threshold)
        self.coverage_temperature = float(coverage_temperature)
        self.confidence_gamma = float(confidence_gamma)
        self.support_mix = float(support_mix)
        self.mode = mode
        self.tolerance_cell_fraction = float(tolerance_cell_fraction)
        self.minimum_tolerance_pixels = int(minimum_tolerance_pixels)
        self.maximum_tolerance_pixels = maximum_tolerance_pixels
        self.maximum_tolerance_fraction = maximum_tolerance_fraction
        self.minimum_foreground_cells = float(minimum_foreground_cells)
        self.minimum_background_cells = float(minimum_background_cells)
        self.epsilon = float(epsilon)

    def forward(
        self,
        raw_score_maps: list[torch.Tensor],
        masks: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> FCALOutput:
        if not raw_score_maps:
            raise ValueError("UniversalFCAL requires at least one signed evidence map")
        confidence = logits.float().softmax(1).gather(1, labels[:, None]).squeeze(1)
        gate = confidence.detach().pow(self.confidence_gamma)
        layer_outputs = []
        for scores in raw_score_maps:
            measured = measure_maps(
                scores,
                masks,
                tolerance_cell_fraction=self.tolerance_cell_fraction,
                evidence_threshold=self.evidence_threshold,
                coverage_temperature=self.coverage_temperature,
                minimum_tolerance_pixels=self.minimum_tolerance_pixels,
                maximum_tolerance_pixels=self.maximum_tolerance_pixels,
                maximum_tolerance_fraction=self.maximum_tolerance_fraction,
                minimum_foreground_cells=self.minimum_foreground_cells,
                minimum_background_cells=self.minimum_background_cells,
                epsilon=self.epsilon,
            )
            containment_per_sample = gate * F.relu(
                measured.outside - self.allowed_outside,
            ).square()
            support_per_sample = gate * F.relu(
                self.target_coverage - measured.coverage,
            ).square()
            containment = _valid_mean(containment_per_sample, measured.valid)
            support = _valid_mean(support_per_sample, measured.valid)
            layer_outputs.append((containment, support, measured))

        if len(layer_outputs) == 1:
            weights = raw_score_maps[0].new_ones(1)
        else:
            weights = torch.arange(
                1, len(layer_outputs) + 1,
                device=raw_score_maps[0].device,
                dtype=raw_score_maps[0].dtype,
            )
            weights = weights / weights.sum()
        containment = sum(w * item[0] for w, item in zip(weights, layer_outputs, strict=True))
        support = sum(w * item[1] for w, item in zip(weights, layer_outputs, strict=True))
        if self.mode == "outside":
            auxiliary = containment
        elif self.mode == "support":
            auxiliary = self.support_mix * support
        else:
            auxiliary = containment + self.support_mix * support

        valid_fraction = torch.stack([
            item[2].valid.float().mean() for item in layer_outputs
        ]).mean()
        mean_outside = torch.stack([
            _valid_mean(item[2].outside, item[2].valid) for item in layer_outputs
        ]).mean()
        mean_coverage = torch.stack([
            _valid_mean(item[2].coverage, item[2].valid) for item in layer_outputs
        ]).mean()
        zero_fraction = torch.stack([
            item[2].zero_map.float().mean() for item in layer_outputs
        ]).mean()
        return FCALOutput(
            auxiliary=auxiliary,
            containment=containment,
            support=support,
            mean_outside=mean_outside,
            mean_coverage=mean_coverage,
            valid_fraction=valid_fraction,
            zero_map_fraction=zero_fraction,
        )


def budgeted_objective(
    ce: torch.Tensor,
    auxiliary: torch.Tensor,
    phase: float,
    target_loss_ratio: float,
    lambda_max: float,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a CE-relative loss budget without a positive weight floor."""
    if float(auxiliary.detach()) <= float(epsilon) or float(phase) <= 0:
        weight = ce.new_zeros(())
    else:
        weight = (
            float(target_loss_ratio) * ce.detach() / (auxiliary.detach() + float(epsilon))
        ).clamp(min=0.0, max=float(lambda_max))
    weighted = float(phase) * weight * auxiliary
    return ce + weighted, weighted, weight
