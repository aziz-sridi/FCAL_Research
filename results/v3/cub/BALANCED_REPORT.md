# FCAL V3 balanced successor: CUB ResNet-50 seed 42

## Protocol

- Dataset: CUB-200-2011 with the existing frozen train, validation, and test
  split.
- Architecture and seed: ResNet-50, seed 42.
- Initialization: exact archived CE epoch-5 paired branch, accepted only after
  prepared-split hash verification.
- Budget: 50 total epochs with the same optimizer and learning-rate schedule.
- Selected checkpoint: epoch 49, using validation clean accuracy only
  (80.33%).
- Selection gate: all six thresholds in `VALIDATION_ITERATION_PROTOCOL.md`
  were fixed before training and passed before the candidate test split was
  opened.
- Test evaluation: all 5,794 official test images, the common 30-view regional
  suite, both CUB-specific stress scenarios, and the deterministic 500-image
  common-map faithfulness protocol.

This is one training seed. It is a successful screening result, not sufficient
evidence for a general or state-of-the-art claim.

## Loss revision

Balanced V3 is a direct successor to V2 and the background-only V3 pilot. It
keeps V2 containment, resolution-aware tolerance, and distributed signed
foreground support. For the extra training view, half of each minibatch gets a
different-class donor background while preserving the foreground; the other
half gets a mild foreground-only blur, noise, or channel-mixing intervention
while preserving the background. Paired classification and prediction
consistency apply to all samples. Foreground-evidence consistency applies only
to the background-intervention half, where the foreground is unchanged.

The selected coefficients are recorded in
`configs/cub/resnet50_fcal_v3_balanced.yaml`. Training changes only the loss and
second view; inference is identical to CE and V2.

## Main comparison

Higher accuracy and lower leakage are favorable.

| Method | Clean | BG mean | BG worst | FG mean | FG only | BG only | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 77.29% | 66.92% | 28.48% | 33.00% | 68.57% | 4.85% | 0.569 |
| FCAL V2 | **78.49%** | 72.11% | 48.86% | 31.87% | 72.76% | 3.88% | **0.282** |
| CAMAL | 77.93% | 72.82% | 59.44% | 32.44% | 72.51% | 2.80% | 0.330 |
| GAIN-ext | 68.43% | 62.56% | 42.22% | 23.52% | 64.24% | **0.52%** | **0.157** |
| RRR | 66.38% | 61.64% | 45.89% | 29.33% | 61.58% | 3.35% | 0.473 |
| Faster-ISNet | 72.06% | 66.16% | 47.05% | 27.50% | 67.67% | 4.42% | 0.547 |
| V3 pilot | 77.61% | 74.91% | 64.01% | 29.04% | **76.75%** | 2.04% | 0.324 |
| **Balanced V3** | 77.89% | **76.03%** | **68.24%** | **42.56%** | 76.39% | 2.73% | 0.296 |

Relative to V2, balanced V3 changes clean accuracy by -0.60 points, background
mean by +3.92 points, worst-background accuracy by +19.38 points, foreground
mean by +10.69 points, foreground-only accuracy by +3.62 points, and leakage by
+0.014 absolute (worse). Relative to the V3 pilot, it raises clean by 0.28,
background mean by 1.12, worst-background by 4.23, and foreground mean by 13.52
points while lowering leakage by 0.028. Foreground-only changes by -0.36 points.

Balanced V3 is the strongest method in this comparison on both aggregate
background metrics and foreground-corruption mean. It does not strictly
dominate V2: V2 remains 0.60 points better on clean accuracy and 0.014 better
on strict leakage. GAIN-ext remains the lowest-leakage method but has a large
clean and robustness cost.

## Paired clean accuracy

Against CE, balanced V3 is +0.60 points with a 95% paired bootstrap interval
of [-0.28, +1.48] points and exact McNemar `p=0.195`. Against V2, it is -0.60
points with interval [-1.47, +0.26] and `p=0.180`. Neither clean comparison is
resolved at 0.05. The bootstrap uses 100,000 deterministic paired resamples
with seed 2026.

## CUB-specific scenarios

| Method | Twilight | Camouflage |
|---|---:|---:|
| CE | 75.30% | 67.38% |
| FCAL V2 | 76.56% | 71.88% |
| V3 pilot | 76.58% | 74.70% |
| **Balanced V3** | **77.42%** | **75.32%** |

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
| **Balanced V3** | 0.300 | 0.710 | 0.073 |

Balanced V3 retains the pilot's strong common-map faithfulness. Its insertion
AUC remains higher than every pre-V3 baseline, while CAMAL retains the best
mask IoU and GAIN-ext the best removal AUC.

## Cost and decision

Post-branch training plus validation took 574.66 seconds and peaked at 3.31 GB
allocated CUDA memory. Inference architecture and cost are unchanged.

The balanced regional intervention solves the pilot's central failure: it
recovers clean accuracy on validation and dramatically improves held-out
foreground-corruption robustness without giving up the background gains. On
this single run it is a stronger V3 candidate, but it is not yet a new
state-of-the-art loss. The remaining gap is precise: reproduce the result over
multiple seeds and determine, without test-set tuning, whether the 0.60-point
clean and 0.014 leakage gaps to V2 can be closed.

Machine-readable metrics are in `resnet50_seed42_comparison.csv`, paired clean
statistics in `resnet50_seed42_paired_clean.csv`, and faithfulness results in
`resnet50_seed42_faithfulness.csv`. Raw artifacts are under
`outputs/resnet50-fcal-v3-balanced-seed42/`.
