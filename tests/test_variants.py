import torch

from fcal_study.complete_results import GENERIC_BACKGROUND, GENERIC_FOREGROUND
from fcal_study.evaluate import with_variant_seed
from fcal_study.variants import (
    TAILORED_VARIANTS,
    VariantSpec,
    apply_variant,
    parse_variant,
)


def sample():
    image = torch.linspace(0, 1, 3 * 32 * 32).reshape(3, 32, 32)
    mask = torch.zeros((1, 32, 32))
    mask[:, 8:24, 8:24] = 1
    return image, mask


def test_background_noise_preserves_foreground_exactly_and_is_deterministic():
    image, mask = sample()
    spec = VariantSpec("bg_gaussian_noise", 2)
    first = apply_variant(image, mask, 7, spec, 42)
    second = apply_variant(image, mask, 7, spec, 42)
    foreground = mask.expand_as(image).bool()
    assert torch.equal(first[foreground], image[foreground])
    assert torch.equal(first, second)
    assert not torch.equal(first[~foreground], image[~foreground])


def test_foreground_blur_preserves_background_exactly():
    image, mask = sample()
    result = apply_variant(image, mask, 9, VariantSpec("fg_blur", 2), 42)
    foreground = mask.expand_as(image).bool()
    assert torch.equal(result[~foreground], image[~foreground])


def test_sham_is_bit_identical():
    image, mask = sample()
    result = apply_variant(image, mask, 1, VariantSpec("sham_recompose"), 42)
    assert torch.equal(result, image)


def test_random_rgb_replaces_all_background_and_preserves_foreground():
    image, mask = sample()
    result = apply_variant(image, mask, 4, VariantSpec("bg_random_rgb", 1), 42)
    foreground = mask.expand_as(image).bool()
    assert torch.equal(result[foreground], image[foreground])
    assert not torch.equal(result[~foreground], image[~foreground])
    assert torch.equal(result, apply_variant(image, mask, 4, VariantSpec("bg_random_rgb", 1), 42))


def test_random_solid_uses_one_rgb_color():
    image, mask = sample()
    result = apply_variant(image, mask, 5, VariantSpec("bg_random_solid", 1), 42)
    background = ~mask[0].bool()
    for channel in range(3):
        assert result[channel][background].unique().numel() == 1


def test_every_tailored_background_scenario_preserves_foreground_and_is_deterministic():
    image, mask = sample()
    names = (
        "bg_cub_twilight", "bg_cub_camouflage", "bg_pet_warm_light",
        "bg_pet_checkerboard", "bg_isic_hair", "bg_isic_ruler",
    )
    foreground = mask.expand_as(image).bool()
    for name in names:
        spec = VariantSpec(name, 1)
        first = apply_variant(image, mask, 19, spec, 42)
        second = apply_variant(image, mask, 19, spec, 42)
        assert torch.equal(first, second), name
        assert torch.equal(first[foreground], image[foreground]), name
        assert not torch.equal(first[~foreground], image[~foreground]), name


def test_isic_low_contrast_preserves_background_and_reduces_foreground_contrast():
    image, mask = sample()
    result = apply_variant(image, mask, 23, VariantSpec("fg_isic_low_contrast", 1), 42)
    foreground = mask.expand_as(image).bool()
    assert torch.equal(result[~foreground], image[~foreground])
    original_contrast = image[foreground].std()
    transformed_contrast = result[foreground].std()
    assert transformed_contrast < original_contrast


def test_complete_report_uses_only_the_13_generic_variants_per_region():
    tailored_cub = {parse_variant(value).key for value in TAILORED_VARIANTS["cub"]}
    assert len(GENERIC_BACKGROUND) == 13
    assert len(GENERIC_FOREGROUND) == 13
    assert not set(GENERIC_BACKGROUND).intersection(tailored_cub)


def test_evaluation_variant_seed_override_does_not_mutate_training_config():
    config = {"seed": 7, "data": {"batch_size": 32}}
    selected = with_variant_seed(config, 42)
    assert selected == {"seed": 42, "data": {"batch_size": 32}}
    assert config == {"seed": 7, "data": {"batch_size": 32}}
    assert with_variant_seed(config, None) is config

    image, mask = sample()
    spec = VariantSpec("bg_random_rgb", 1)
    reference = apply_variant(image, mask, 17, spec, 42)
    rerun = apply_variant(image, mask, 17, spec, selected["seed"])
    assert torch.equal(reference, rerun)
