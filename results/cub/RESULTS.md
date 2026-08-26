# Six-model FCAL V1/V2 benchmark

## Scope and fairness

This benchmark compares cross-entropy Baseline, FCAL V1, and FCAL V2 on the
same CUB-200-2011 split for ResNet-50, DenseNet-121, EfficientNet-B0,
ViT-Tiny/16, ViT-Small/16, and ViT-Base/16. All ViTs use average patch pooling,
so their patch CAM exactly decomposes the selected class logit.

Every run has the same architecture-specific data pipeline, optimizer,
schedule, initialization, and 50-epoch budget. V1 and V2 resume from their
paired Baseline's exact epoch-5 branch checkpoint. FCAL changes only the
training loss: there are no modified training images, additional inference
modules, extra inference operations, or inference-time parameters.

The notebook suite evaluates 15 views: clean, background blur (3 strengths),
background color jitter (3), background noise (3), grayscale, solid-mean,
random-solid, random-RGB, and foreground-only. Foreground-distortion and
background-only tests are excluded.

## Main results

All values are top-1 accuracy percentages. `BG mean` averages the 13
background-manipulation views; `BG worst` is their minimum.

| Model | Method | Clean | BG mean | BG worst | Foreground-only |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Baseline | 77.29 | 66.92 | 28.48 | 68.57 |
| ResNet-50 | V1 | 77.36 | 69.61 | 41.44 | 70.73 |
| ResNet-50 | V2 | **78.49** | **72.11** | **48.86** | **72.76** |
| DenseNet-121 | Baseline | 79.43 | 72.78 | 53.21 | 71.68 |
| DenseNet-121 | V1 | 78.89 | 72.81 | 54.28 | 72.26 |
| DenseNet-121 | V2 | **79.57** | **74.14** | **57.59** | **73.59** |
| EfficientNet-B0 | Baseline | 77.32 | 67.22 | 28.31 | **68.73** |
| EfficientNet-B0 | V1 | 77.72 | 65.69 | 19.76 | 67.28 |
| EfficientNet-B0 | V2 | **78.39** | **68.11** | **30.55** | 67.54 |
| ViT-Tiny/16 | Baseline | 74.30 | **66.70** | **48.93** | **65.00** |
| ViT-Tiny/16 | V1 | 74.54 | 66.46 | 48.76 | 64.79 |
| ViT-Tiny/16 | V2 | **74.82** | 66.21 | 47.88 | 64.81 |
| ViT-Small/16 | Baseline | **83.62** | **79.26** | **68.38** | 78.98 |
| ViT-Small/16 | V1 | 83.52 | 78.80 | 66.69 | **79.10** |
| ViT-Small/16 | V2 | 83.19 | 78.83 | 68.24 | 78.60 |
| ViT-Base/16 | Baseline | 84.93 | 82.34 | 75.08 | 81.38 |
| ViT-Base/16 | V1 | 85.48 | 83.01 | 77.13 | **82.76** |
| ViT-Base/16 | V2 | **85.86** | **83.09** | **77.15** | 82.59 |

## Interpretation

The user's concern is correct: the practically large result is concentrated in
ResNet-50. V2 improves it by +1.20 clean points, +5.19 BG-mean points, +20.38
BG-worst points, and +4.19 foreground-only points. V1 moves in the same
direction but less strongly, so the ResNet result is not an isolated V2
accident.

ViT-Base/16 shows a smaller but consistent benefit from both objectives. V2
adds +0.93 clean, +0.75 BG mean, and +2.07 BG worst; V1 gives the best
foreground-only score (+1.38). DenseNet-121 receives a moderate V2 robustness
benefit, but V1 is almost flat and lowers clean accuracy.

EfficientNet-B0 and the two smaller ViTs do not support a universal FCAL claim.
EfficientNet V1 is actively fragile on the strongest background tests; V2
recovers modest clean/background gains but lowers foreground-only accuracy.
ViT-Tiny and ViT-Small are essentially flat or slightly worse on robustness.

This pattern is compatible with the loss-scale measurements. For both small
ViTs, the V1 spatial penalty remains around 0.27--0.29 and adaptive lambda
quickly clips to its minimum 0.05. The CNN V1 penalty is much smaller. Thus the
same numeric FCAL settings do not imply the same effective optimization across
CAM geometry, resolution, or capacity. V2 improves scale handling, but its
fixed support weight is still not capacity-universal.

With one training seed, these data support an architecture-dependent result,
not a broad claim that FCAL always improves classification. On clean images,
V1's paired rescue/harm differences are small for every architecture. Some V2
clean differences are statistically detectable at the image level, but that
does not replace multi-seed training uncertainty. Publication-grade claims
should therefore emphasize ResNet-50, treat ViT-Base and DenseNet V2 as
supporting evidence, and label EfficientNet/Tiny/Small as boundary conditions.

## Prediction transitions

Each row below pools 86,910 predictions (5,794 images x 15 views). `Net` is
rescued minus harmed. These pooled counts describe effect size; they are not an
independence-based significance test because the same image appears in multiple
views.

| Model | Method | Rescued | Harmed | Correct in both | Wrong in both | Net |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | V1 | 6,923 | 4,764 | 54,090 | 21,133 | **+2,159** |
| ResNet-50 | V2 | 8,338 | 4,113 | 54,741 | 19,718 | **+4,225** |
| DenseNet-121 | V1 | 4,586 | 4,567 | 59,011 | 18,746 | +19 |
| DenseNet-121 | V2 | 5,509 | 4,370 | 59,208 | 17,823 | **+1,139** |
| EfficientNet-B0 | V1 | 3,206 | 4,418 | 54,674 | 24,612 | **-1,212** |
| EfficientNet-B0 | V2 | 3,884 | 3,218 | 55,874 | 23,934 | +666 |
| ViT-Tiny/16 | V1 | 1,056 | 1,235 | 57,073 | 27,546 | -179 |
| ViT-Tiny/16 | V2 | 3,310 | 3,660 | 54,648 | 25,292 | -350 |
| ViT-Small/16 | V1 | 2,728 | 3,074 | 66,050 | 15,058 | -346 |
| ViT-Small/16 | V2 | 3,083 | 3,460 | 65,664 | 14,703 | -377 |
| ViT-Base/16 | V1 | 4,172 | 3,553 | 68,104 | 11,081 | +619 |
| ViT-Base/16 | V2 | 4,387 | 3,699 | 67,958 | 10,866 | +688 |

FCAL is not a post-processing rule that preserves all Baseline-correct cases.
After the common epoch-5 state, its spatial loss changes the learned features
and class boundaries. It can remove contextual or highly localized cues that
were genuinely discriminative, especially between similar fine-grained
classes. The harmed count is therefore real and must be reported alongside
rescues and wrong-in-both cases.

## Localization diagnostics

| Model | Baseline leakage | V1 leakage | V2 leakage |
|---|---:|---:|---:|
| ResNet-50 | 0.569 | 0.567 | **0.282** |
| DenseNet-121 | 0.534 | 0.524 | **0.318** |
| EfficientNet-B0 | 0.595 | 0.585 | **0.479** |
| ViT-Tiny/16 | 0.801 | 0.799 | **0.784** |
| ViT-Small/16 | 0.804 | 0.792 | **0.789** |
| ViT-Base/16 | 0.806 | 0.694 | **0.188** |

V1 barely changes strict leakage except on ViT-Base. V2 reduces it on all six,
but better localization does not automatically produce better classification.
DenseNet V2 also increases the dead-CAM rate from 3.23% to 5.45%, which remains
an important failure mode.

Both FCAL versions add no inference parameters or operations. Their extra work
exists only during training.

## Reproducible artifacts

- [`../../configs/cub/`](../../configs/cub/): matched CE, V1, and V2 configurations.
- [`summary.csv`](summary.csv): compact ResNet-50 and ViT-Base/16 metrics.
- [`background_robustness.csv`](background_robustness.csv): the 13 background tests.
- [`../../assets/cub_multiarchitecture_summary.png`](../../assets/cub_multiarchitecture_summary.png): six-model overview.
