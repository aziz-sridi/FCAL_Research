# CUB FCAL V1/V2 evidence report

## Protocol

- Six architectures: ResNet-50, DenseNet-121, EfficientNet-B0, and
  ViT-Tiny/16, ViT-Small/16, ViT-Base/16 with average patch pooling.
- CE, V1, and V2 share the exact architecture-specific CE epoch-5 branch and
  complete the same 50-epoch budget.
- The common test protocol is now complete: clean, sham, 13 background
  corruptions, 13 foreground corruptions, foreground-only, and background-only.
- Two separate bird scenarios test twilight and an extreme camouflage
  background while bird pixels remain exact.
- The six-model matrix uses seed 42. ResNet-50 is additionally repeated with
  training seeds 7 and 123 on the same fixed split.

## Unified method means across six architectures

| Metric | CE | FCAL V1 | FCAL V2 | V2 vs CE |
|---|---:|---:|---:|---:|
| Clean | 79.48% | 79.59% | **80.05%** | +0.57 pp |
| BG mean | 72.54% | 72.73% | **73.75%** | +1.21 pp |
| BG worst | 50.40% | 51.34% | **55.05%** | +4.65 pp |
| FG mean | **37.42%** | 37.26% | 37.06% | -0.36 pp |
| FG worst | **2.49%** | 2.17% | 2.13% | -0.36 pp |
| Foreground only | 72.39% | 72.82% | **73.31%** | +0.93 pp |
| Background only | 5.25% | 5.07% | **4.50%** | -0.75 pp |
| Strict leakage | 0.685 | 0.660 | **0.473** | 30.9% lower |

The completed foreground tests materially narrow the claim. V2 improves
background robustness and foreground-only recognition, but does not improve
robustness when the bird pixels themselves are corrupted.

## ResNet-50: replicated result

| Method | Clean | BG mean | BG worst | Foreground only | Leakage |
|---|---:|---:|---:|---:|---:|
| CE | 77.44 ± 0.13% | 68.75 ± 1.70% | 38.90 ± 9.81% | 68.63 ± 0.34% | 0.571 ± 0.003 |
| FCAL V1 | 77.61 ± 0.34% | 70.00 ± 0.39% | 43.86 ± 2.46% | 70.57 ± 0.39% | 0.568 ± 0.001 |
| FCAL V2 | **78.49 ± 0.16%** | **72.40 ± 0.38%** | **51.67 ± 4.00%** | **72.51 ± 0.87%** | **0.281 ± 0.003** |

These are mean ± sample SD for training seeds 7, 42, and 123. Paired-seed V2
minus CE is +1.05 ± 0.19 pp clean, +3.65 ± 1.37 pp BG mean, +12.77 ± 6.63 pp
BG worst, and +3.87 ± 0.57 pp foreground-only. The worst-background estimate
has substantial variation, but all three seeds favor V2 on every listed metric.

## Architecture boundaries at seed 42

| Architecture | Clean V2−CE | BG mean V2−CE | BG worst V2−CE | FG-only V2−CE | Leakage reduction |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | +1.21 pp | +5.19 pp | +20.38 pp | +4.19 pp | 50.5% |
| DenseNet-121 | +0.14 pp | +1.36 pp | +4.38 pp | +1.91 pp | 40.4% |
| EfficientNet-B0 | +1.07 pp | +0.89 pp | +2.24 pp | -1.19 pp | 19.5% |
| ViT-Tiny/16 | +0.52 pp | -0.49 pp | -1.05 pp | -0.19 pp | 2.1% |
| ViT-Small/16 | -0.43 pp | -0.43 pp | -0.14 pp | -0.38 pp | 1.9% |
| ViT-Base/16 | +0.93 pp | +0.75 pp | +2.07 pp | +1.21 pp | 76.7% |

The result is capacity- and architecture-dependent. ResNet-50 is the replicated
positive result; DenseNet and ViT-Base support parts of it. The smaller ViTs do
not support a universal classification claim despite lower leakage.

## Bird-tailored scenarios

Across six architectures, twilight accuracy changes from 76.36% CE to 77.12%
V2; camouflage changes from 72.03% to 73.70%. For ResNet-50 the corresponding
gains are +1.26 pp and +4.50 pp. Both tests preserve bird pixels exactly, but
they are deterministic simulations rather than natural distribution shifts.

## V2 component ablation

This separate diagnostic uses ResNet-50, seed 42, and a 20-epoch budget. All
three rows share the first five CE epochs.

| Variant | Clean | BG mean | BG worst | FG mean | Leakage |
|---|---:|---:|---:|---:|---:|
| Signed support + fixed 3-cell tolerance | 69.80% | 61.76% | 35.07% | 26.98% | 0.564 |
| Resolution-aware tolerance only | **70.47%** | 62.40% | 34.69% | **27.76%** | **0.490** |
| Full V2 | 69.83% | **62.84%** | **43.32%** | 27.74% | 0.493 |

This table was regenerated from an archived seed-42 epoch-5 branch after a
provenance audit found that the earlier Fixed-3 row had no surviving run
configuration. `fixed_tolerance_cells: 3` now implements V1's literal
evidence-grid dilation; the 20-epoch branches retain the parent's 50-epoch
cosine schedule.

Resolution-aware tolerance is the important change: against fixed three-cell
dilation, full V2 improves BG mean by 1.08 pp, BG worst by 8.25 pp, FG mean by
0.77 pp, and leakage by 0.072 absolute at essentially identical clean
accuracy. Adding signed support to tolerance-only trades 0.64 pp clean for
0.45 pp BG mean and a substantial 8.63 pp BG-worst gain. The components are
therefore complementary specifically for background robustness, with the
combined design producing the strongest worst-case result.

## Statistical scope and matched baseline status

The three-seed ResNet result directly addresses training-run uncertainty for
one architecture. It does not turn the other five architectures into
multi-seed experiments. Pooled regional prediction counts are not treated as
independent because each image appears in multiple views.

RRR and GAIN-ext now have matched seed-7 ResNet-50 runs. FCAL V2 leads clean,
BG mean, and foreground-only accuracy; RRR narrowly leads BG worst and
GAIN-ext has the lowest CAM leakage. These are informative one-seed trade-offs,
not a completed multi-seed ranking. Selective Faster ISNet/LRP-Flex was also
attempted in two PyTorch environments but produced non-finite second-order
gradients on the first batch, so no invalid score is reported. See
[the exact comparison](explanation_baselines_seed7.md).

## Machine-readable evidence

- [Complete per-model metrics](complete_metrics.csv)
- [Unified method means](method_means_complete.csv)
- [Tailored scenarios](tailored_scenarios.csv)
- [Three-seed per-run metrics](resnet50_three_seed_metrics.csv)
- [Three-seed summary](resnet50_three_seed_summary.csv)
- [V2 component ablation](v2_component_ablation.csv)
- [Stress-test definitions](../../docs/STRESS_TESTS.md)
