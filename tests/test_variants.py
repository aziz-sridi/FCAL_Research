import torch

from fcal_study.variants import VariantSpec, apply_variant


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
