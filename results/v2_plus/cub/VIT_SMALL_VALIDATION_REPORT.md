# FCAL V2+ ViT-Small validation report

## Outcome

V2+ transfers to ViT-Small/16 without any architecture-specific tuning and
improves clean accuracy and localization leakage substantially. It does not
replace the original ViT-Small V2 because it regresses all four robustness and
foreground-sufficiency metrics. The frozen six-metric gate is 2/6, so the CUB
test split remains sealed.

Against CAMAL, V2+ wins five of the six quality metrics and loses only strict
leakage. This makes V2+ a strong ViT candidate, but not a uniformly better V2.

## Validation performance

| Method | Clean | Leak down | BG mean | BG worst | FG mean | FG only |
|---|---:|---:|---:|---:|---:|---:|
| V2 original | 82.75% | 0.780 | **78.87%** | **69.00%** | **43.85%** | **78.33%** |
| **V2+** | **83.08%** | **0.540** | 77.87% | 68.08% | 41.49% | 77.08% |
| CAMAL | 82.08% | **0.486** | 76.66% | 64.67% | 37.74% | 74.08% |

Relative to original V2, V2+ gains 0.33 percentage points clean and lowers
strict leakage by 0.240. It loses 1.01 points background mean, 0.92 points
background worst, 2.35 points foreground mean, and 1.25 points foreground-only
accuracy.

Relative to CAMAL, V2+ gains 1.00 point clean, 1.21 points background mean,
3.42 points background worst, 3.75 points foreground mean, and 3.00 points
foreground-only accuracy. CAMAL retains a 0.055 advantage in strict leakage.

## Structured-only secondary view

Removing the extreme random-RGB condition uniformly does not change the
decision.

| Method | BG mean no RGB | BG worst no RGB | FG mean no RGB |
|---|---:|---:|---:|
| V2 original | **79.26%** | **69.00%** | **47.21%** |
| V2+ | 78.47% | 68.08% | 44.69% |
| CAMAL | 77.12% | 64.67% | 40.74% |

## Compute

Recorded complete-run wall time was 436.41 seconds for V2+, versus 341.15
seconds for archived V2 and 376.43 seconds for CAMAL. Those full runs were not
contemporaneous, and the machine was in a slower state during V2+.

A same-session V2 control from the identical epoch-5 branch isolates the loss
overhead. After compilation stabilized (epochs 9-10), V2+ used 12.818 training
seconds versus 12.608 for V2: 1.67% overhead. Throughput was 748.0 versus 760.5
samples/s, a 1.64% reduction. Peak allocated CUDA memory increased from 1.3824
to 1.3838 GiB, or 0.10%. The loss is therefore cheap on ViT-Small; quality, not
compute, is the reason for rejection.

## Constraint audit

- Original untouched images only.
- Exactly one backbone forward per training step.
- Masks used by the loss only; no input masking, corruption, erasure, or second
  view.
- V2+ coefficients copied unchanged from ResNet-50.
- Checkpoint selected by clean validation accuracy only.
- CUB test split unopened.
