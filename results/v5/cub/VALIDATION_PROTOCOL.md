# FCAL V5 regional-margin certificate validation protocol

## Frozen candidate

- Configuration: `configs/cub/resnet50_fcal_v5_certificate.yaml`, run ID
  `resnet50-fcal-v5-certificate-smooth-seed42`.
- Dataset/split: CUB-200-2011 seed 42, existing prepared split.
- Backbone: ResNet-50.
- Initialization: exact archived CE epoch-5 branch.
- Budget: epochs 6--50 of the common 50-epoch schedule.
- Selection: clean validation accuracy only.
- Input policy: one original training image; no counterfactual, erasure,
  replacement, or second view.
- Objective: one regional-margin certificate over the null rival and top two
  real logit competitors. No V1, V2, CAMAL, RRR, or V4 auxiliary term is added.

The official test split remains sealed unless every performance and efficiency
gate below passes.

## Performance gates

Every regional metric must be no worse than the frozen local ResNet-50 V2
validation result, and at least three must improve strictly.

| Metric | Required |
|---|---:|
| Clean accuracy | >= 0.800833 |
| Strict leakage | <= 0.266214 |
| Background mean | >= 0.720449 |
| Background worst | >= 0.489167 |
| Foreground mean | >= 0.318269 |
| Foreground only | >= 0.716667 |

Faithfulness uses the deterministic 500-image validation subset:

| Metric | Required |
|---|---:|
| Mask IoU | >= 0.291556 |
| Insertion AUC | >= 0.696989 |
| Removal AUC | <= 0.082249 |

## Efficiency gates

The paired ResNet-50 V2 reference records 325.170 post-branch seconds and
1.7471 GiB peak allocated VRAM. V5 must satisfy both:

| Metric | Required |
|---|---:|
| Post-branch train + validation wall time | <= 390.204 seconds (1.20x V2) |
| Peak allocated VRAM | <= 1.9218 GiB (1.10x V2) |

These are single-run feasibility gates, not timing confidence intervals.

## Timing-invalid implementation attempt

The first launch used exact full-resolution top-k tails. It completed only
epoch 6 before being stopped because Python-level per-sample selection achieved
90.6 samples/s and could not satisfy the predeclared efficiency objective. No
regional or faithfulness evaluation was run. The retained
`resnet50-fcal-v5-certificate-seed42` history is excluded from model comparison.

Before any performance result, the same certificate was re-expressed with
vectorized entropic lower/upper tails at the native evidence grid. Adaptive
average pooling supplies soft foreground/background weights without empty
regions. This is a mechanism-preserving computational correction: the null and
top-two rivals, coefficient, background scale, margin, warmup, data, and all
gates remain unchanged. Tail temperatures are frozen to 0.50.
