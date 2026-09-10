import torch

from fcal_study.models import (
    DenseNet121CAM,
    EfficientNetB0CAM,
    ISNetResNet50CAM,
    ResNet50CAM,
    TimmViTAttention,
    TimmViTAvgPoolCAM,
)


def test_isnet_resnet_adapter_preserves_resnet_logits():
    baseline = ResNet50CAM(num_classes=5, pretrained=False).eval()
    isnet = ISNetResNet50CAM(num_classes=5, pretrained=False).eval()
    isnet.load_state_dict(baseline.state_dict())
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = baseline(images)
        actual = isnet(images)
    torch.testing.assert_close(actual, expected)


def test_resnet_cam_shape_and_gradient():
    model = ResNet50CAM(num_classes=200, pretrained=False).eval()
    images = torch.randn(1, 3, 224, 224)
    labels = torch.tensor([3])
    logits = model(images, capture_attention=True)
    attention = model.training_attention(labels)[0]
    assert logits.shape == (1, 200)
    assert attention.shape == (1, 7, 7)
    attention.sum().backward()
    assert model.model.layer4[-1].conv3.weight.grad is not None


def test_resnet_exposes_pre_relu_cam_without_changing_article_cam():
    model = ResNet50CAM(num_classes=2, pretrained=False).eval()
    model._features = torch.ones(1, 2048, 2, 2)
    with torch.no_grad():
        model.model.fc.weight[0].fill_(-1.0)
    labels = torch.tensor([0])
    raw = model.raw_training_attention(labels)[0]
    article = model.training_attention(labels)[0]
    assert bool((raw < 0).all())
    assert bool((article == 0).all())


def test_resnet_cam_accepts_mixed_precision_features_outside_autocast():
    model = ResNet50CAM(num_classes=200, pretrained=False).eval()
    model._features = torch.randn(1, 2048, 7, 7, dtype=torch.bfloat16)
    attention = model.evaluation_attention(torch.tensor([3]))
    assert attention.dtype == torch.bfloat16
    assert attention.shape == (1, 7, 7)


def test_vit_attention_shapes_rollout_and_gradient():
    model = TimmViTAttention(
        "vit_base_patch16_224.augreg_in21k_ft_in1k", num_classes=200,
        supervised_layers=6, pretrained=False,
    ).eval()
    images = torch.randn(1, 3, 224, 224)
    labels = torch.tensor([3])
    logits = model(images, capture_attention=True)
    maps = model.training_attention(labels)
    assert logits.shape == (1, 200)
    assert len(maps) == 6
    assert all(item.shape == (1, 14, 14) for item in maps)
    sum(item.sum() for item in maps).backward()
    assert model.model.blocks[-1].attn.qkv.weight.grad is not None
    with torch.no_grad():
        model(images, capture_attention=True)
        rollout = model.evaluation_attention(labels)
    assert rollout.shape == (1, 14, 14)


def test_vit_exposes_signed_class_specific_patch_evidence():
    model = TimmViTAttention(
        "vit_base_patch16_224.augreg_in21k_ft_in1k", num_classes=200,
        supervised_layers=6, pretrained=False,
    ).eval()
    images = torch.randn(1, 3, 224, 224)
    labels = torch.tensor([3])
    logits = model(images, capture_attention=False)
    evidence = model.raw_class_evidence(labels)[0]
    assert logits.shape == (1, 200)
    assert evidence.shape == (1, 14, 14)
    assert bool((evidence < 0).any() | (evidence > 0).any())
    evidence.sum().backward()
    assert model.model.head.weight.grad is not None


def test_average_pooled_vit_patch_cam_exactly_decomposes_selected_logits():
    model = TimmViTAvgPoolCAM(
        "vit_tiny_patch16_224", num_classes=5, pretrained=False,
    ).eval()
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([1, 3])
    logits = model(images)
    raw = model.raw_class_evidence(labels)[0]
    bias = model.model.head.bias[labels]
    reconstructed = raw.flatten(1).mean(dim=1) + bias
    torch.testing.assert_close(logits[torch.arange(2), labels], reconstructed)
    raw.sum().backward()
    assert model.model.head.weight.grad is not None


def test_average_pooled_vit_can_disable_fused_attention_for_double_backward():
    model = TimmViTAvgPoolCAM(
        "vit_tiny_patch16_224",
        num_classes=5,
        pretrained=False,
        require_second_order_attention=True,
    )
    assert all(not block.attn.fused_attn for block in model.model.blocks)


def test_densenet_cam_exactly_decomposes_selected_logits():
    model = DenseNet121CAM(num_classes=5, pretrained=False).eval()
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([1, 3])
    logits = model(images)
    raw = model.raw_class_evidence(labels)[0]
    bias = model.model.classifier.bias[labels]
    reconstructed = raw.flatten(1).mean(dim=1) + bias
    torch.testing.assert_close(logits[torch.arange(2), labels], reconstructed, rtol=1e-5, atol=1e-5)
    raw.sum().backward()
    assert model.model.classifier.weight.grad is not None


def test_efficientnet_cam_exactly_decomposes_selected_logits_with_dropout():
    model = EfficientNetB0CAM(num_classes=5, pretrained=False).train()
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([1, 3])
    logits = model(images)
    raw = model.raw_class_evidence(labels)[0]
    bias = model.model.classifier[-1].bias[labels]
    reconstructed = raw.flatten(1).mean(dim=1) + bias
    torch.testing.assert_close(logits[torch.arange(2), labels], reconstructed, rtol=1e-5, atol=1e-5)
    raw.sum().backward()
    assert model.model.classifier[-1].weight.grad is not None
