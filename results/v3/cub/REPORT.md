# FCAL V3 CUB ResNet-50 seed-42 pilot

## Protocol

- Dataset: CUB-200-2011, existing frozen train/validation/test split.
- Architecture: ResNet-50.
- Seed: 42.
- V3 resumed from the exact paired CE epoch-5 checkpoint after prepared-split
  hash verification.
- Training budget: 50 total epochs; checkpoint selected by validation accuracy
  only.
- Selected checkpoint: epoch 48, validation accuracy 78.75%.
- Test evaluation: all 5,794 official test images, the common 30-view regional
  suite, the two separate CUB scenarios, and the deterministic 500-image
  common-map faithfulness protocol.
- Comparator values are the existing fixed/default seed-42 ResNet-50 runs.

This is one training seed. It is a screening result, not training-seed
uncertainty and not a state-of-the-art claim.

## Main comparison

Higher accuracy and lower leakage are favorable.

| Method | Clean | BG mean | BG worst | FG mean | FG only | BG only | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 77.29% | 66.92% | 28.48% | **33.00%** | 68.57% | 4.85% | 0.569 |
| FCAL V2 | **78.49%** | 72.11% | 48.86% | 31.87% | 72.76% | 3.88% | **0.282** |
| CAMAL | 77.93% | 72.82% | 59.44% | 32.44% | 72.51% | 2.80% | 0.330 |
| GAIN-ext | 68.43% | 62.56% | 42.22% | 23.52% | 64.24% | **0.52%** | **0.157** |
| RRR | 66.38% | 61.64% | 45.89% | 29.33% | 61.58% | 3.35% | 0.473 |
| Faster-ISNet | 72.06% | 66.16% | 47.05% | 27.50% | 67.67% | 4.42% | 0.547 |
| **FCAL V3** | 77.61% | **74.91%** | **64.01%** | 29.04% | **76.75%** | 2.04% | 0.324 |

V3 is the strongest method in this comparison on BG mean, BG worst, and
foreground-only accuracy. Relative to V2, the changes are:

- clean accuracy: -0.88 percentage points;
- BG mean: +2.80 points;
- BG worst: +15.15 points;
- FG mean: -2.83 points;
- foreground-only: +3.99 points;
- strict leakage: +0.042 absolute, which is worse.

Relative to CE, V3 improves clean accuracy by 0.33 points, BG mean by 8.00
points, BG worst by 35.54 points, and foreground-only accuracy by 8.18 points,
while reducing strict leakage by 43.0% at the two means. V3 does not improve
foreground-corruption robustness.

## Paired clean accuracy

The image-level paired V3-minus-CE clean difference is +0.33 points with a
95% paired bootstrap interval of [-0.60, +1.26] points and exact McNemar
`p=0.514`. Against V2, V3 is -0.88 points with interval [-1.78, 0.00] and
`p=0.058`. Neither comparison is resolved at 0.05.

## CUB-specific scenarios

| Method | Twilight | Camouflage |
|---|---:|---:|
| CE | 75.30% | 67.38% |
| FCAL V2 | 76.56% | 71.88% |
| **FCAL V3** | **76.58%** | **74.70%** |

V3 is effectively tied with V2 on twilight and improves the extreme
camouflage scenario by 2.81 points over V2 and 7.32 points over CE.

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
| **FCAL V3** | 0.300 | **0.713** | 0.071 |

V3 has the best insertion AUC. Its removal AUC is better than CE, V2, CAMAL,
and RRR but worse than GAIN-ext and Faster-ISNet. CAMAL remains substantially
better on mask IoU.

## Training cost

V3's post-branch training-plus-validation wall time was 611.46 seconds with
3.31 GB peak allocated CUDA memory. Relative to the fixed V2 run, this is
88.0% more wall time and 76.5% more peak memory. Its wall time is approximately
the same as GAIN-ext and much lower than the recorded RRR and Faster-ISNet
runs. Inference remains identical to CE and V2.

## Decision

This configuration validates the V3 mechanism: explicit background
counterfactual supervision produces a large worst-background and
foreground-only gain without changing inference. It does **not** dominate V2:
clean accuracy, foreground-corruption robustness, and leakage regress. The
next iteration should therefore target clean/leakage recovery while preserving
the behavioral terms, and it should be chosen on validation data rather than
by revisiting these test results.

Machine-readable results are in `resnet50_seed42_comparison.csv`,
`resnet50_seed42_paired_clean.csv`, and
`resnet50_seed42_faithfulness.csv`. Raw evaluation and checkpoint artifacts are
under `outputs/resnet50-fcal-v3-seed42/`.
