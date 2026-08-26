from __future__ import annotations

from .config import PROJECT_ROOT


RUNS = {
    "ResNet-50": {
        "CE": PROJECT_ROOT / "outputs" / "resnet50-ce-seed42",
        "FCAL V1": PROJECT_ROOT / "outputs" / "resnet50-fcal-seed42",
        "FCAL V2": PROJECT_ROOT / "outputs" / "resnet50-fcal-v2-seed42",
    },
    "ViT-B/16 average pool": {
        "CE": PROJECT_ROOT / "outputs" / "vit-base-p16-avg-ce-seed42",
        "FCAL V1": PROJECT_ROOT / "outputs" / "vit-base-p16-avg-fcal-seed42",
        "FCAL V2": PROJECT_ROOT / "outputs" / "vit-base-p16-avg-fcal-v2-seed42",
    },
}
