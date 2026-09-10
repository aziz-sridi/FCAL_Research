# FCAL V3.1 routed-strict: CUB ResNet-50 seed 42

## Protocol

- Dataset: CUB-200-2011, using the existing frozen train, validation, and test
  split.
- Architecture and seed: ResNet-50, seed 42.
- Initialization: exact archived CE epoch-5 paired branch after prepared-split
  hash verification.
- Budget: 50 total epochs with the same optimizer and schedule as CE, V2, and
  the earlier V3 candidates.
- Selected checkpoint: epoch 50, using validation clean accuracy only.
- Test access: all six predeclared conditions in
  `ROUTED_STRICT_VALIDATION_PROTOCOL.md` passed before test evaluation.
- Final evaluation: all 5,794 official test images, the common 30-view suite,
  both CUB-specific scenarios, and the deterministic 500-image common-map
  faithfulness subset.

This is a one-seed screening result and does not establish training-seed
uncertainty or a general state-of-the-art claim.

## Validation gate

| Metric | Required | V3.1 | Result |
|---|---:|---:|---|
| Clean | >= 80.08% | **81.42%** | pass |
| Strict leakage | <= 0.2799 | **0.2252** | pass |
| BG mean | >= 76.72% | **77.42%** | pass |
| BG worst | >= 66.75% | **69.67%** | pass |
| FG mean | >= 42.37% | **42.45%** | pass |
| FG only | >= 76.58% | **77.08%** | pass |

## Main test comparison

Higher accuracy and lower leakage are favorable.

| Method | Clean | BG mean | BG worst | FG mean | FG only | BG only | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 77.29% | 66.92% | 28.48% | 33.00% | 68.57% | 4.85% | 0.569 |
| FCAL V2 | **78.49%** | 72.11% | 48.86% | 31.87% | 72.76% | 3.88% | 0.282 |
| CAMAL | 77.93% | 72.82% | 59.44% | 32.44% | 72.51% | 2.80% | 0.330 |
| GAIN-ext | 68.43% | 62.56% | 42.22% | 23.52% | 64.24% | **0.52%** | **0.157** |
| RRR | 66.38% | 61.64% | 45.89% | 29.33% | 61.58% | 3.35% | 0.473 |
| Faster-ISNet | 72.06% | 66.16% | 47.05% | 27.50% | 67.67% | 4.42% | 0.547 |
| V3 pilot | 77.61% | 74.91% | 64.01% | 29.04% | **76.75%** | 2.04% | 0.324 |
| Balanced V3 | 77.89% | **76.03%** | **68.24%** | **42.56%** | 76.39% | 2.73% | 0.296 |
| **V3.1 routed-strict** | 77.99% | 75.81% | 67.76% | 41.48% | 76.42% | 2.62% | **0.229** |

Bold leakage in the V3.1 row denotes the strongest result among CE, V2,
CAMAL, RRR, Faster-ISNet, and the V3 family; GAIN-ext remains lower at the cost
of 9.56 points of clean accuracy relative to V3.1.

Relative to balanced V3, V3.1 raises clean by 0.10 points and lowers leakage by
0.067 absolute. Background mean changes by -0.22 points, worst-background by
-0.48, foreground mean by -1.08, and foreground-only by +0.03. Relative to V2,
V3.1 gains 3.70 background-mean points, 18.90 worst-background points, 9.61
foreground-mean points, and 3.66 foreground-only points while lowering leakage
by 0.053; clean remains 0.50 points lower.

## Paired clean accuracy

V3.1 minus CE is +0.71 points with a deterministic 100,000-resample paired
bootstrap interval of [-0.21, +1.64] and exact McNemar `p=0.143`. V3.1 minus
V2 is -0.50 points with interval [-1.38, +0.38] and `p=0.283`. V3.1 minus
balanced V3 is +0.10 points with interval [-0.69, +0.90] and `p=0.830`. None
of the clean differences is resolved at 0.05.

## CUB-specific scenarios

| Method | Twilight | Camouflage |
|---|---:|---:|
| CE | 75.30% | 67.38% |
| FCAL V2 | 76.56% | 71.88% |
| V3 pilot | 76.58% | 74.70% |
| Balanced V3 | **77.42%** | **75.32%** |
| **V3.1 routed-strict** | 77.36% | 74.80% |

## Common-map faithfulness

Higher mask IoU and insertion AUC are favorable; lower removal AUC is
favorable.

| Method | Mask IoU | Insertion AUC | Removal AUC |
|---|---:|---:|---:|
| CE | 0.295 | 0.652 | 0.099 |
| FCAL V2 | 0.283 | 0.675 | 0.080 |
| CAMAL | **0.575** | 0.663 | 0.080 |
| GAIN-ext | 0.410 | 0.538 | **0.040** |
| RRR | 0.176 | 0.446 | 0.077 |
| Faster-ISNet | 0.257 | 0.513 | 0.063 |
| V3 pilot | 0.300 | **0.713** | 0.071 |
| Balanced V3 | 0.300 | 0.710 | 0.073 |
| **V3.1 routed-strict** | 0.244 | 0.706 | 0.073 |

V3.1 retains high insertion faithfulness, but its thresholded mask IoU drops
substantially. The strict-shell term reduces total outside evidence without
guaranteeing broad object coverage; in this run it appears to concentrate
positive evidence more narrowly inside the foreground. This is the main
remaining loss-design weakness.

## Cost and conclusion

Post-branch training plus validation took 603.89 seconds and peaked at 3.31 GB
allocated CUDA memory. Inference remains identical to CE and V2.

V3.1 is the best classification/robustness/leakage operating point tested in
the V3 family: it nearly preserves balanced V3's large regional robustness,
improves clean accuracy slightly, and cuts leakage below V2. It is not a strict
dominator because balanced V3 remains slightly stronger on aggregate
robustness and substantially stronger on mask IoU. Multi-seed replication and
a coverage-preserving refinement are required before any state-of-the-art
claim.

Machine-readable metrics are in `resnet50_seed42_comparison.csv`, paired clean
statistics in `resnet50_seed42_paired_clean.csv`, and faithfulness results in
`resnet50_seed42_faithfulness.csv`. Raw artifacts are under
`outputs/resnet50-fcal-v3-routed-strict-seed42/`.
