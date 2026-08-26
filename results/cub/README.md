# CUB-200-2011: where bird classifiers look

FCAL V2 produces the clearest CUB result on ResNet-50: compared with CE, it
adds **+1.20 percentage points clean accuracy**, **+5.19 points mean background
robustness**, and **+20.38 points on the worst background test**, while reducing
strict CAM leakage from **0.569 to 0.282** (50.5% relative).

Every panel below uses the same held-out image across methods. Green/red text
marks a correct/incorrect class prediction, the white contour is the annotated
bird, and `inside` is the fraction of positive predicted-class CAM mass within
that contour. CAM colors are normalized per panel using `jet`: blue is low
evidence and red is high evidence.

## CNN examples

One frozen test example per CNN architecture:

![CUB CNN prediction and CAM comparison](../../assets/result_showcases/cub/readme_cnn.png)

## Vision Transformer examples

One frozen test example per ViT architecture:

![CUB ViT prediction and CAM comparison](../../assets/result_showcases/cub/readme_vit.png)

## Classification rescues: CE wrong, FCAL V2 correct

The next galleries add one held-out rescue per architecture. They are selected
qualitative examples; the benchmark table below is computed on the full test
set.

![CUB CNN cases where CE is wrong and FCAL V2 is correct](../../assets/result_showcases/cub/readme_cnn_rescues.png)

![CUB ViT cases where CE is wrong and FCAL V2 is correct](../../assets/result_showcases/cub/readme_vit_rescues.png)

## What the full benchmark says

| Result | CE | FCAL V1 | FCAL V2 |
|---|---:|---:|---:|
| ResNet-50 clean accuracy | 77.29% | 77.36% | **78.49%** |
| ResNet-50 mean of 13 background tests | 66.92% | 69.61% | **72.11%** |
| ResNet-50 worst background test | 28.48% | 41.44% | **48.86%** |
| ViT-Base/16 clean accuracy | 84.93% | 85.48% | **85.86%** |
| ViT-Base/16 strict leakage ↓ | 0.806 | 0.694 | **0.188** |

V2 reduces strict leakage on all six architectures, but classification gains
are architecture-dependent: ResNet-50 is the strongest result, DenseNet-121
and ViT-Base support it, while the smaller ViTs are flat or slightly worse on
some robustness measures.

- [Open the executed test showcase](../../notebooks/cub_test_showcase.ipynb) — three cases per architecture.
- [Read the complete result tables and limitations](RESULTS.md).
- [Inspect the machine-readable metrics](summary.csv).

All runs use seed 42 and the same architecture-specific epoch-5 CE branch.
FCAL changes training only; it adds no inference parameters or operations.
