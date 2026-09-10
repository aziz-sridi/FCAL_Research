# FCAL V5 regional-margin certificate: validation-only result

## Outcome

The first V5 certificate candidate succeeded as a compute design but failed as
a performance design. It used one original image, one forward pass, and one
ordinary backward pass. It contains no V2, CAMAL, RRR, V4, or intervention
auxiliary loss. Because it failed the frozen validation gates, the official
test split was not opened and V5 is not a state-of-the-art result.

## Efficiency

| Method | Post-branch time | Time vs V2 | Peak VRAM | VRAM vs V2 |
|---|---:|---:|---:|---:|
| FCAL V2 | 5.42 min | reference | 1.7471 GiB | reference |
| V5 certificate | 6.13 min | +13.2% | 1.7471 GiB | +0.0006% |
| CAMAL | 6.23 min | +15.0% | 1.7452 GiB | -0.1% |
| V4.2 | 29.16 min | +438.0% | 3.4674 GiB | +98.5% |

V5 passed the predeclared 1.20x wall-time and 1.10x VRAM gates. After initial
kernel setup, epochs generally took about 7.0--7.1 seconds at 730--755
samples/s.

## Validation performance

| Method | Clean | Leak down | BG mean | BG worst | FG mean | FG only | IoU | Insert | Remove down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FCAL V2 | **80.08%** | **0.266** | **72.04%** | **48.92%** | **31.83%** | **71.67%** | 0.292 | **0.697** | **0.082** |
| V5 certificate | 79.17% | 0.537 | 67.58% | 17.33% | 31.52% | 69.00% | **0.367** | 0.670 | 0.107 |

V5 passed only the IoU gate. Its high IoU with worse insertion and removal
shows that spatial mask overlap did not imply causal evidence quality.

## Interpretation

The scale-free signed margin certificate admitted a failure mode: foreground
pairwise margins could dominate its scaled background statistic while diffuse
positive target CAM mass remained in the background. The model learned
mask-plausible coverage but not background-invariant predictions. Random-RGB
background accuracy exposed this most strongly at 17.33%.

This falsifies the present certificate, not the broader goal of a cheap
original-input successor. A defensible next formulation must connect regional
supervision to the causal class probability itself rather than only normalized
spatial margin geometry. More coefficient tuning of this loss on the same
validation split is not justified.

## Implementation note

An exact full-resolution top-k implementation was stopped after epoch 6 because
it achieved only 90.6 samples/s. No regional or faithfulness result was opened.
The reported candidate uses the pre-performance vectorized entropic-tail
correction recorded in `VALIDATION_PROTOCOL.md`.
