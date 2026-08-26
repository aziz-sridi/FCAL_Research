# Oxford-IIIT Pet full FCAL benchmark

## Protocol

- Dataset: Oxford-IIIT Pet, with 37 breed classes and official pixel-level trimaps.
- Split: 2,944 train / 736 validation from official trainval; the official 3,669-image test partition is held out for final evaluation.
- Architectures: ResNet-50, DenseNet-121, EfficientNet-B0, ViT-Tiny/16, ViT-Small/16, and ViT-Base/16.
- Methods: CE, FCAL V1, FCAL V2, and automatic-safe Universal FCAL; each FCAL run resumes from its architecture's exact CE epoch-5 branch.
- Training: seed 42, 50 total epochs, checkpoint selected by validation accuracy.
- Test suite: clean plus 29 deterministic regional perturbations, including background/foreground corruptions and region-only ablations.
- Classification uses all 3,669 test images. Localization averages exclude only degenerate center-cropped masks.

## Aggregate result

Across the six architectures, **FCAL V2** has the highest mean clean accuracy (90.03%); **FCAL V2** has the lowest mean strict leakage (19.11%). **FCAL V2** is strongest on mean background corruptions (87.48%), while **Universal FCAL** is strongest on mean foreground corruptions (47.59%).

Relative to CE, V2 averages +0.50 pp clean accuracy, -10.09 pp strict leakage, +0.67 pp mean background-corruption accuracy, and +0.61 pp mean foreground-corruption accuracy. It improves clean accuracy on 5/6 architectures and reduces strict leakage on 6/6.

Universal FCAL averages +0.42 pp clean accuracy and +0.74 pp foreground-corruption accuracy, but only -0.17 pp strict leakage. V1 averages +0.27 pp clean accuracy and -3.13 pp strict leakage; its -2.08 pp foreground-only result is driven by a large EfficientNet-B0 regression.

Exact paired clean-test McNemar tests find 7/18 FCAL-versus-CE comparisons below p=0.05: 5 favor FCAL and 2 favor CE. These are single-seed results; the tests measure paired test-set disagreement, not training-seed uncertainty.

## Method means across six architectures

| Method | Clean | Strict leakage | Mean BG corrupt. | Mean FG corrupt. | Foreground only | Background only |
|---|---|---|---|---|---|---|
| CE | 89.52% | 29.20% | 86.81% | 46.85% | 86.59% | 4.03% |
| FCAL V1 | 89.79% | 26.07% | 86.50% | 47.08% | 84.51% | 4.04% |
| FCAL V2 | 90.03% | 19.11% | 87.48% | 47.46% | 88.94% | 3.54% |
| Universal FCAL | 89.94% | 29.03% | 87.43% | 47.59% | 88.77% | 4.16% |

## All 24 clean and robustness results

| Family | Architecture | Method | Best val | Clean | Balanced | Macro-F1 | Strict leakage | Mean BG corrupt. | Mean FG corrupt. | Foreground only | Background only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN | ResNet-50 | CE | 92.12% | 88.33% | 88.32% | 88.19% | 20.24% | 88.18% | 42.56% | 87.57% | 3.16% |
| CNN | ResNet-50 | FCAL V1 | 91.71% | 88.03% | 87.98% | 87.86% | 19.61% | 87.59% | 43.04% | 86.92% | 3.79% |
| CNN | ResNet-50 | FCAL V2 | 92.80% | 89.13% | 89.08% | 88.95% | 9.91% | 89.01% | 43.42% | 88.88% | 2.51% |
| CNN | ResNet-50 | Universal FCAL | 92.26% | 88.33% | 88.31% | 88.05% | 19.68% | 88.10% | 43.24% | 87.84% | 2.89% |
| CNN | DenseNet-121 | CE | 93.61% | 89.81% | 89.74% | 89.66% | 18.42% | 89.40% | 45.78% | 88.69% | 4.66% |
| CNN | DenseNet-121 | FCAL V1 | 93.48% | 88.58% | 88.52% | 88.50% | 18.34% | 87.60% | 45.22% | 86.94% | 3.54% |
| CNN | DenseNet-121 | FCAL V2 | 93.34% | 90.46% | 90.42% | 90.29% | 11.69% | 89.81% | 45.87% | 89.34% | 3.46% |
| CNN | DenseNet-121 | Universal FCAL | 94.02% | 90.62% | 90.58% | 90.51% | 18.01% | 89.62% | 45.77% | 88.85% | 4.44% |
| CNN | EfficientNet-B0 | CE | 92.93% | 90.32% | 90.26% | 90.11% | 12.19% | 76.92% | 37.15% | 77.60% | 2.94% |
| CNN | EfficientNet-B0 | FCAL V1 | 93.07% | 90.92% | 90.87% | 90.76% | 12.60% | 75.21% | 38.57% | 64.79% | 3.71% |
| CNN | EfficientNet-B0 | FCAL V2 | 93.21% | 90.54% | 90.53% | 90.34% | 8.16% | 78.00% | 38.60% | 88.03% | 3.41% |
| CNN | EfficientNet-B0 | Universal FCAL | 93.07% | 90.81% | 90.78% | 90.58% | 12.47% | 79.19% | 38.87% | 88.63% | 3.60% |
| ViT | ViT-Tiny/16 avg | CE | 92.12% | 88.23% | 88.15% | 88.00% | 40.89% | 86.90% | 44.28% | 86.43% | 3.73% |
| ViT | ViT-Tiny/16 avg | FCAL V1 | 92.39% | 88.31% | 88.24% | 88.08% | 40.15% | 86.96% | 44.98% | 86.86% | 4.01% |
| ViT | ViT-Tiny/16 avg | FCAL V2 | 92.80% | 87.46% | 87.42% | 87.23% | 39.73% | 86.70% | 45.24% | 86.73% | 3.54% |
| ViT | ViT-Tiny/16 avg | Universal FCAL | 92.26% | 87.54% | 87.50% | 87.38% | 40.80% | 86.45% | 44.80% | 86.45% | 3.38% |
| ViT | ViT-Small/16 avg | CE | 94.57% | 89.62% | 89.53% | 89.24% | 42.11% | 89.20% | 54.13% | 89.26% | 5.01% |
| ViT | ViT-Small/16 avg | FCAL V1 | 94.16% | 90.84% | 90.78% | 90.73% | 40.17% | 90.32% | 55.12% | 90.46% | 4.85% |
| ViT | ViT-Small/16 avg | FCAL V2 | 94.29% | 91.01% | 90.96% | 90.87% | 41.00% | 90.37% | 54.92% | 90.13% | 4.99% |
| ViT | ViT-Small/16 avg | Universal FCAL | 94.16% | 90.98% | 90.93% | 90.82% | 41.97% | 90.61% | 56.21% | 90.24% | 4.99% |
| ViT | ViT-Base/16 avg | CE | 94.29% | 90.84% | 90.79% | 90.68% | 41.37% | 90.28% | 57.21% | 89.97% | 4.66% |
| ViT | ViT-Base/16 avg | FCAL V1 | 94.29% | 92.07% | 92.01% | 91.91% | 25.58% | 91.35% | 55.58% | 91.09% | 4.36% |
| ViT | ViT-Base/16 avg | FCAL V2 | 93.89% | 91.58% | 91.53% | 91.41% | 4.19% | 91.01% | 56.71% | 90.54% | 3.33% |
| ViT | ViT-Base/16 avg | Universal FCAL | 93.48% | 91.36% | 91.33% | 91.31% | 41.26% | 90.63% | 56.67% | 90.57% | 5.64% |

## Interpretation notes

- Higher accuracy and lower leakage are better. `Mean BG corrupt.` and `Mean FG corrupt.` average the 13 background-only and 13 foreground-only corruption variants, respectively.
- `Foreground only` replaces the background with the ImageNet mean; `Background only` replaces the pet region.
- Strict leakage is the cross-method localization metric. Tolerant leakage is intentionally omitted because legacy FCAL and Universal FCAL use different tolerance conventions.
- This is a one-seed benchmark. Architecture-level differences should be confirmed across seeds before being treated as stable rankings.

## Generated artifacts

- `primary_metrics.csv`: one row per architecture and method.
- `variant_metrics.csv`: one row per architecture, method, and perturbation.
- `method_means.csv`: six-architecture method averages.
- `deltas_vs_ce.csv`: architecture-level FCAL minus CE differences.
- `paired_clean_vs_ce.csv`: clean paired rescue/harm counts and exact McNemar p-values.
