# Oxford-IIIT Pet: the complete FCAL comparison

Oxford-IIIT Pet is the broadest benchmark in this repository: **3,669 held-out
test images**, six architectures, four methods, and 29 deterministic regional
perturbations. FCAL V2 improves mean clean accuracy on five of six
architectures and reduces strict leakage on all six.

Every panel below uses the same held-out image across methods. Green/red text
marks a correct/incorrect breed prediction, the white contour is the official
pet trimap, and `inside` is the fraction of positive predicted-class CAM mass
inside it. CAM colors are normalized per panel using `jet`: blue is low evidence
and red is high evidence.

## CNN examples

One frozen test example per CNN architecture:

![Oxford Pet CNN prediction and CAM comparison](../../assets/result_showcases/oxford_pet/readme_cnn.png)

## Vision Transformer examples

One frozen test example per ViT architecture:

![Oxford Pet ViT prediction and CAM comparison](../../assets/result_showcases/oxford_pet/readme_vit.png)

## Classification rescues: CE wrong, FCAL V2 correct

The next galleries add one held-out rescue per architecture. They are selected
qualitative examples; the full 3,669-image benchmark below remains the basis
for quantitative claims.

![Oxford Pet CNN cases where CE is wrong and FCAL V2 is correct](../../assets/result_showcases/oxford_pet/readme_cnn_rescues.png)

![Oxford Pet ViT cases where CE is wrong and FCAL V2 is correct](../../assets/result_showcases/oxford_pet/readme_vit_rescues.png)

## What the full benchmark says

| Six-model mean | CE | FCAL V1 | FCAL V2 | Universal FCAL |
|---|---:|---:|---:|---:|
| Clean accuracy | 89.52% | 89.79% | **90.03%** | 89.94% |
| Strict leakage ↓ | 0.292 | 0.261 | **0.191** | 0.290 |
| Mean background-corruption accuracy | 86.81% | 86.50% | **87.48%** | 87.43% |
| Mean foreground-corruption accuracy | 46.85% | 47.08% | 47.46% | **47.59%** |

Best observed V2-versus-CE single-model changes on this dataset are **+0.79 pp
clean accuracy and 51.0% lower leakage for a CNN**, and **+1.39 pp clean
accuracy and 89.9% lower leakage for a ViT**. These maxima occur on different
architectures and are not six-model averages.

Universal FCAL delivers the best mean foreground-corruption result and +0.42 pp
mean clean accuracy, but barely changes mean leakage. It remains an
architecture-independent experimental controller, not a replacement for V2.

- [Open the executed test showcase](../../notebooks/oxford_pet_test_showcase.ipynb) — three cases per architecture.
- [Read the complete 24-run result table and limitations](RESULTS.md).
- [Inspect the machine-readable metrics](primary_metrics.csv).

All runs use seed 42, the official test partition, and the same
architecture-specific epoch-5 CE branch. FCAL adds no inference overhead.
