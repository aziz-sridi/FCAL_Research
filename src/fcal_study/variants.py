from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class VariantSpec:
    name: str
    severity: int = 0

    @property
    def key(self) -> str:
        return self.name if self.severity == 0 else f"{self.name}_s{self.severity}"


def parse_variant(value: str) -> VariantSpec:
    if ":" not in value:
        return VariantSpec(value, 0)
    name, severity = value.rsplit(":", 1)
    return VariantSpec(name, int(severity))


def _seed(global_seed: int, image_id: int, spec: VariantSpec) -> int:
    payload = f"{global_seed}:{image_id}:{spec.name}:{spec.severity}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def _transformed(image: torch.Tensor, spec: VariantSpec, seed: int) -> torch.Tensor:
    severity = max(1, spec.severity)
    generator = torch.Generator(device=image.device).manual_seed(seed)
    base = spec.name.removeprefix("bg_").removeprefix("fg_").removeprefix("full_")
    if base == "blur":
        kernels = {1: 9, 2: 17, 3: 31}
        sigmas = {1: 2.0, 2: 5.0, 3: 9.0}
        level = min(severity, 3)
        return TF.gaussian_blur(image, kernels[level], [sigmas[level], sigmas[level]])
    if base == "color_jitter":
        # Fixed magnitudes with seeded signs avoid hidden nondeterminism.
        magnitude = {1: 0.15, 2: 0.30, 3: 0.45}[min(severity, 3)]
        signs = torch.randint(0, 2, (3,), generator=generator, device=image.device) * 2 - 1
        result = TF.adjust_brightness(image, 1.0 + float(signs[0]) * magnitude)
        result = TF.adjust_contrast(result, 1.0 + float(signs[1]) * magnitude)
        result = TF.adjust_saturation(result, max(0.0, 1.0 + float(signs[2]) * magnitude))
        return result.clamp(0, 1)
    if base == "grayscale":
        return TF.rgb_to_grayscale(image, num_output_channels=3)
    if base == "gaussian_noise":
        sigma = {1: 0.05, 2: 0.10, 3: 0.20}[min(severity, 3)]
        noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        return (image + sigma * noise).clamp(0, 1)
    if base == "solid_mean":
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype, device=image.device)
        return mean[:, None, None].expand_as(image)
    if base == "random_solid":
        color = torch.rand((3, 1, 1), generator=generator, device=image.device, dtype=image.dtype)
        return color.expand_as(image)
    if base == "random_rgb":
        return torch.rand(image.shape, generator=generator, device=image.device, dtype=image.dtype)
    raise ValueError(f"Unknown corruption: {spec.name}")


def apply_variant(
    image: torch.Tensor,
    mask: torch.Tensor,
    image_id: int,
    spec: VariantSpec,
    global_seed: int,
) -> torch.Tensor:
    """Apply a deterministic corruption to image in [0,1], preserving the opposite region exactly."""
    if spec.name in {"clean", "sham_recompose"}:
        return image.clone()
    binary = (mask >= 0.5).to(image.dtype)
    if spec.name == "foreground_only":
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype, device=image.device)[:, None, None]
        return image * binary + mean * (1.0 - binary)
    if spec.name == "background_only":
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype, device=image.device)[:, None, None]
        return mean * binary + image * (1.0 - binary)
    transformed = _transformed(image, spec, _seed(global_seed, image_id, spec))
    if spec.name.startswith("bg_"):
        return image * binary + transformed * (1.0 - binary)
    if spec.name.startswith("fg_"):
        return transformed * binary + image * (1.0 - binary)
    if spec.name.startswith("full_"):
        return transformed
    raise ValueError(f"Unknown variant: {spec.name}")


DEFAULT_VARIANTS = [
    "clean",
    "sham_recompose",
    "bg_blur:1", "bg_blur:2", "bg_blur:3",
    "fg_blur:1", "fg_blur:2", "fg_blur:3",
    "bg_color_jitter:1", "bg_color_jitter:2", "bg_color_jitter:3",
    "fg_color_jitter:1", "fg_color_jitter:2", "fg_color_jitter:3",
    "bg_grayscale:1", "fg_grayscale:1",
    "bg_gaussian_noise:1", "bg_gaussian_noise:2", "bg_gaussian_noise:3",
    "fg_gaussian_noise:1", "fg_gaussian_noise:2", "fg_gaussian_noise:3",
    "bg_solid_mean:1", "fg_solid_mean:1",
    "bg_random_solid:1", "fg_random_solid:1",
    "bg_random_rgb:1", "fg_random_rgb:1",
    "foreground_only", "background_only",
]
