import torch

from fcal_study.losses import (
    adaptive_lambda,
    article_fcal_loss,
    article_fcal_map_loss,
    smooth_fcal_map_loss,
    smooth_normalized_attention,
)
from fcal_study.v2 import support_aware_fcal_losses


def test_article_loss_is_background_mass_fraction_without_dilation():
    attention = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    value = article_fcal_map_loss(attention, mask, dilation_cells=0)
    assert torch.allclose(value, torch.tensor(0.5))


def test_article_normalization_couples_foreground_gradient():
    attention = torch.ones((1, 2, 2), requires_grad=True)
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    article_fcal_map_loss(attention, mask, dilation_cells=0).backward()
    assert bool((attention.grad[0, 0] < 0).all())
    assert bool((attention.grad[0, 1] > 0).all())


def test_vit_layer_weights_are_linear():
    maps = [torch.ones((1, 2, 2)) * value for value in range(1, 4)]
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    # Scaling each positive map cancels under L1 normalization; every layer is 0.5.
    assert torch.allclose(article_fcal_loss(maps, mask, dilation_cells=0), torch.tensor(0.5))


def test_adaptive_lambda_uses_detached_values_and_clips():
    ce = torch.tensor(2.0, requires_grad=True)
    spatial = torch.tensor(0.5, requires_grad=True)
    weight = adaptive_lambda(ce, spatial, alpha=0.5, minimum=0.05, maximum=1.0)
    assert torch.allclose(weight, torch.tensor(1.0))
    assert not weight.requires_grad


def test_smooth_attention_stays_live_for_all_negative_raw_cam():
    raw = torch.full((1, 2, 2), -10.0, requires_grad=True)
    probability = smooth_normalized_attention(raw)
    assert bool((probability.sum(dim=(1, 2)) > 0.999).all())
    assert bool((probability > 0).all())
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    smooth_fcal_map_loss(raw, mask, dilation_cells=0).backward()
    assert raw.grad is not None
    assert bool((raw.grad.abs() > 0).all())


def test_v1_excludes_empty_and_full_masks_from_spatial_supervision():
    attention = torch.ones((2, 2, 2), requires_grad=True)
    masks = torch.stack((torch.zeros((1, 2, 2)), torch.ones((1, 2, 2))))
    loss = article_fcal_map_loss(attention, masks, dilation_cells=0)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert attention.grad is not None
    assert float(attention.grad.abs().sum()) == 0.0


def test_v2_excludes_empty_masks_but_keeps_valid_samples():
    scores = torch.ones((2, 2, 2), requires_grad=True)
    masks = torch.tensor([
        [[[0.0, 0.0], [0.0, 0.0]]],
        [[[1.0, 1.0], [0.0, 0.0]]],
    ])
    outside, support = support_aware_fcal_losses(
        [scores], masks, support_temperature=1.0, support_margin=0.0,
        tolerance_cell_fraction=0.0,
    )
    assert torch.isfinite(outside + support)
    (outside + support).backward()
    assert scores.grad is not None
    assert float(scores.grad[0].abs().sum()) == 0.0
    assert float(scores.grad[1].abs().sum()) > 0.0
