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
    height, width = image.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)[:, None]
    xx = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)[None, :]
    if base == "cub_twilight":
        # Simulate a plausible dawn/dusk habitat shift without touching the bird.
        factors = torch.tensor((0.68, 0.76, 1.05), device=image.device, dtype=image.dtype)[:, None, None]
        illumination = (0.72 + 0.18 * (1.0 - yy.square())).clamp(0.55, 0.90)
        return (image * factors * illumination).clamp(0, 1)
    if base == "pet_warm_light":
        # Simulate warm, uneven household lighting outside the annotated pet.
        factors = torch.tensor((1.12, 1.00, 0.80), device=image.device, dtype=image.dtype)[:, None, None]
        vignette = (1.0 - 0.28 * (xx.square() + yy.square()).clamp_max(1.0))[None]
        return (image * factors * vignette).clamp(0, 1)
    if base == "pet_checkerboard":
        # An intentionally extreme, high-frequency background distractor.
        cell = max(4, min(height, width) // 14)
        grid = ((torch.arange(height, device=image.device)[:, None] // cell)
                + (torch.arange(width, device=image.device)[None, :] // cell)) % 2
        colors = torch.rand((2, 3), generator=generator, device=image.device, dtype=image.dtype)
        colors = 0.15 + 0.80 * colors
        return colors[grid.long()].permute(2, 0, 1)
    if base == "isic_hair":
        # Deterministic dark hair-like curves, restricted to non-lesion skin later.
        result = image.clone()
        hair = torch.zeros((height, width), device=image.device, dtype=torch.bool)
        count = 7
        for index in range(count):
            phase = float(torch.rand((), generator=generator, device=image.device)) * 2.0 * math.pi
            slope = float(torch.empty((), device=image.device).uniform_(-0.65, 0.65, generator=generator))
            offset = float(torch.empty((), device=image.device).uniform_(-0.85, 0.85, generator=generator))
            curve = slope * xx + offset + 0.12 * torch.sin(3.0 * xx + phase)
            hair |= (yy - curve).abs() < (1.5 / max(height, 1))
        return torch.where(hair[None], result * 0.12, result)
    if base == "isic_ruler":
        # An extreme acquisition-artifact simulation: a high-contrast ruler strip.
        result = image.clone()
        strip_width = max(6, width // 12)
        start = int(torch.randint(0, max(1, width - strip_width), (1,), generator=generator))
        result[:, :, start:start + strip_width] = 0.92
        tick_gap = max(5, height // 18)
        for row in range(0, height, tick_gap):
            tick = strip_width if (row // tick_gap) % 5 == 0 else max(2, strip_width // 2)
            result[:, row:min(height, row + 2), start:start + tick] = 0.08
        return result
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
    if spec.name == "bg_cub_camouflage":
        # Extreme low-contrast habitat: background color follows the bird's mean
        # while retaining smooth deterministic texture. Bird pixels stay exact.
        foreground_count = binary.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        foreground_mean = (image * binary).sum(dim=(-2, -1), keepdim=True) / foreground_count
        generator = torch.Generator(device=image.device).manual_seed(_seed(global_seed, image_id, spec))
        noise = torch.rand(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        texture = TF.gaussian_blur(noise, 31, [9.0, 9.0])
        transformed = (foreground_mean + 0.45 * (texture - 0.5)).clamp(0, 1)
        return image * binary + transformed * (1.0 - binary)
    if spec.name == "fg_isic_low_contrast":
        # Hard diagnostic case: reduce lesion-to-skin contrast without changing
        # any background pixel.
        background = 1.0 - binary
        background_count = background.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        background_mean = (image * background).sum(dim=(-2, -1), keepdim=True) / background_count
        transformed = 0.20 * image + 0.80 * background_mean
        return transformed * binary + image * background
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


TAILORED_VARIANTS = {
    "cub": ["bg_cub_twilight:1", "bg_cub_camouflage:1"],
    "oxford_pet": ["bg_pet_warm_light:1", "bg_pet_checkerboard:1"],
    "isic": ["bg_isic_hair:1", "bg_isic_ruler:1", "fg_isic_low_contrast:1"],
}
