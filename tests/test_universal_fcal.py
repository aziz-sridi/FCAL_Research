import torch

from universal_fcal import (
    AutoFCALScheduler,
    UniversalFCAL,
    gradient_safe_backward,
    measure_maps,
    resolution_aware_tolerance_pixels,
)


def test_resolution_is_inferred_for_cnn_and_vit_maps():
    assert resolution_aware_tolerance_pixels((224, 224), (7, 7), 0.5) == 16
    assert resolution_aware_tolerance_pixels((224, 224), (14, 14), 0.5) == 8


def test_full_mask_is_classification_only_not_spatial_supervision():
    scores = torch.randn(2, 7, 7, requires_grad=True)
    masks = torch.ones(2, 1, 224, 224)
    logits = torch.randn(2, 3, requires_grad=True)
    labels = torch.tensor([0, 1])
    objective = UniversalFCAL(
        allowed_outside=0.2,
        target_coverage=0.3,
        maximum_tolerance_fraction=0.1,
    )
    result = objective([scores], masks, logits, labels)
    assert result.valid_fraction.item() == 0.0
    assert result.auxiliary.item() == 0.0


def test_map_measurement_keeps_non_degenerate_small_lesion():
    scores = torch.ones(1, 14, 14)
    masks = torch.zeros(1, 1, 224, 224)
    masks[:, :, 80:144, 80:144] = 1
    measured = measure_maps(
        scores,
        masks,
        tolerance_cell_fraction=0.5,
        evidence_threshold=1.0,
        coverage_temperature=0.25,
    )
    assert measured.valid.item()
    assert measured.tolerance_pixels == 8


def test_gradient_safe_backward_projects_conflict_and_caps_ratio():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    ce = parameter[0]
    auxiliary = -parameter[0] + parameter[1]
    stats = gradient_safe_backward(
        ce,
        auxiliary,
        [parameter],
        maximum_gradient_ratio=0.25,
        project_conflicts=True,
    )
    assert stats.conflict_projected
    assert stats.ratio_after <= 0.250001
    assert torch.allclose(parameter.grad, torch.tensor([1.0, 0.25]), atol=1e-6)


def test_auto_scheduler_increases_on_safe_plateau_and_backs_off_on_harm():
    scheduler = AutoFCALScheduler(
        initial_gradient_ratio=0.05,
        minimum_gradient_ratio=0.01,
        maximum_gradient_ratio=0.20,
        patience=2,
        cooldown=0,
        warmup_epochs=0,
        ramp_epochs=0,
    )
    scheduler.step(0, 0.80, 0.10)
    scheduler.step(1, 0.80, 0.10)
    event = scheduler.step(2, 0.80, 0.10)
    assert event == "increase"
    assert scheduler.current_ratio > 0.05
    scheduler.step(3, 0.78, 0.09)
    event = scheduler.step(4, 0.78, 0.08)
    assert event == "backoff"
    assert scheduler.current_ratio < 0.0625


def test_auto_scheduler_can_release_to_ce_for_final_recovery():
    scheduler = AutoFCALScheduler(
        warmup_epochs=5,
        ramp_epochs=5,
        release_start_epoch=35,
        release_end_epoch=45,
    )
    assert scheduler.phase(4) == 0.0
    assert scheduler.phase(9) == 1.0
    assert 0.0 < scheduler.phase(39) < 1.0
    assert scheduler.phase(44) == 0.0
