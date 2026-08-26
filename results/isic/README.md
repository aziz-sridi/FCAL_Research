# ISIC: constraining evidence to the lesion

ISIC is the clearest example of why this project reports localization and
classification separately. Across six architectures, FCAL V2 reduces mean
strict outside-lesion CAM leakage from **0.441 to 0.257** (41.6% relative), but
mean clean accuracy changes from **66.50% to 64.83%**. The result is a strong
focus improvement, not a universal clinical-accuracy claim.

Every panel below uses the same held-out dermoscopic image across methods.
Green/red text marks a correct/incorrect class prediction, the white contour is
the lesion mask, and `inside` is the fraction of positive predicted-class CAM
mass inside it. CAM colors are normalized per panel using `jet`: blue is low
evidence and red is high evidence.

## CNN examples

One frozen test example per CNN architecture:

![ISIC CNN prediction and CAM comparison](../../assets/result_showcases/isic/readme_cnn.png)

## Vision Transformer examples

One frozen test example per ViT architecture:

![ISIC ViT prediction and CAM comparison](../../assets/result_showcases/isic/readme_vit.png)

## Classification rescues: CE wrong, FCAL V2 correct

The next galleries add one held-out rescue per architecture. They are selected
qualitative examples; the full 100-image table below remains the basis for
quantitative claims.

![ISIC CNN cases where CE is wrong and FCAL V2 is correct](../../assets/result_showcases/isic/readme_cnn_rescues.png)

![ISIC ViT cases where CE is wrong and FCAL V2 is correct](../../assets/result_showcases/isic/readme_vit_rescues.png)

## What the full benchmark says

| Six-model mean | CE | FCAL V1 | FCAL V2 |
|---|---:|---:|---:|
| Clean accuracy | **66.50%** | 64.67% | 64.83% |
| Macro-F1 | **66.79%** | 64.21% | 65.47% |
| Strict leakage ↓ | 0.441 | 0.385 | **0.257** |
| Foreground-only accuracy | 42.50% | 42.83% | **46.17%** |

V2 reduces leakage on every architecture and is more accurate than V1 on three
of six. ViT-Small is the strongest classification case (69% V2 versus 67% CE),
but this sealed test contains only 100 images, including just three VASC cases.

- [Open the executed test showcase](../../notebooks/isic_test_showcase.ipynb) — three cases per architecture.
- [Read the complete result tables and limitations](RESULTS.md).
- [Inspect the machine-readable metrics](primary_metrics.csv).
- [Read the Universal FCAL ResNet-50 pilot](../universal/isic/RESULTS.md).

These are model-behavior experiments on a curated subset, not evidence of
clinical effectiveness. All completed runs use seed 42.
