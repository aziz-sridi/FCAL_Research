from pathlib import Path

import yaml


def test_v1_configs_change_only_objective_within_architecture():
    root = Path(__file__).resolve().parents[1]
    pairs = [
        ("resnet50_ce.yaml", "resnet50_fcal.yaml"),
        ("densenet121_ce.yaml", "densenet121_fcal.yaml"),
        ("efficientnet_b0_ce.yaml", "efficientnet_b0_fcal.yaml"),
        ("vit_tiny_avg_ce.yaml", "vit_tiny_avg_fcal.yaml"),
        ("vit_small_avg_ce.yaml", "vit_small_avg_fcal.yaml"),
        ("vit_base_avg_ce.yaml", "vit_base_avg_fcal.yaml"),
    ]
    for baseline_name, v1_name in pairs:
        baseline = yaml.safe_load((root / "configs" / "cub" / baseline_name).read_text(encoding="utf-8"))
        v1 = yaml.safe_load((root / "configs" / "cub" / v1_name).read_text(encoding="utf-8"))
        assert baseline["data"] == v1["data"]
        assert baseline["model"] == v1["model"]
        baseline_training = {**baseline["training"], "save_epoch5_branch": False}
        assert baseline_training == v1["training"]
        assert baseline["fcal"] == v1["fcal"]
        assert v1["method"] == "fcal"
