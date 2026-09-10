# Fixed explanation-baseline CUB aggregation

Seed 42 only; training-run uncertainty is unestimated. All rows use the held-out CUB test split
after fixed-recipe training and validation-accuracy checkpoint selection. No test metric was used
to choose a method setting or checkpoint.

## CAMAL per architecture

| Architecture | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Strict leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 77.93% | 72.82% | 59.44% | 32.44% | 1.88% | 72.51% | 2.80% | 0.330 |
| DenseNet-121 | 78.03% | 71.86% | 53.88% | 33.47% | 1.73% | 71.33% | 4.56% | 0.315 |
| EfficientNet-B0 | 77.22% | 65.88% | 23.94% | 29.92% | 2.85% | 67.00% | 3.81% | 0.392 |
| ViT-Tiny/16 avg | 74.49% | 65.76% | 48.76% | 29.44% | 1.26% | 64.39% | 1.90% | 0.560 |
| ViT-Small/16 avg | 81.50% | 76.62% | 63.34% | 38.11% | 1.50% | 74.63% | 3.71% | 0.500 |
| ViT-Base/16 avg | 85.07% | 81.95% | 75.35% | 46.55% | 2.00% | 81.36% | 5.30% | 0.258 |

## GAIN-ext per architecture

| Architecture | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Strict leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 68.43% | 62.56% | 42.22% | 23.52% | 0.52% | 64.24% | 0.52% | 0.157 |
| DenseNet-121 | 71.00% | 63.69% | 48.90% | 30.43% | 1.00% | 58.75% | 1.00% | 0.100 |
| EfficientNet-B0 | 73.18% | 58.15% | 4.33% | 27.77% | 2.28% | 65.22% | 2.81% | 0.454 |
| ViT-Tiny/16 avg | 74.18% | 66.88% | 52.54% | 28.93% | 0.64% | 66.64% | 0.64% | 0.640 |
| ViT-Small/16 avg | 82.05% | 77.74% | 64.79% | 37.78% | 0.60% | 75.30% | 0.60% | 0.590 |
| ViT-Base/16 avg | 86.07% | 83.03% | 76.65% | 43.13% | 0.26% | 82.59% | 0.26% | 0.340 |

## RRR per architecture

| Architecture | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Strict leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 66.38% | 61.64% | 45.89% | 29.33% | 1.88% | 61.58% | 3.35% | 0.473 |
| DenseNet-121 | 69.95% | 66.02% | 55.78% | 29.51% | 2.42% | 64.57% | 3.94% | 0.546 |
| EfficientNet-B0 | 76.48% | 72.46% | 64.62% | 34.42% | 1.31% | 71.76% | 3.81% | 0.574 |
| ViT-Tiny/16 avg | 73.13% | 68.22% | 53.83% | 32.95% | 1.69% | 67.74% | 3.02% | 0.805 |
| ViT-Small/16 avg | 80.67% | 77.44% | 69.80% | 41.80% | 1.66% | 76.99% | 4.63% | 0.805 |
| ViT-Base/16 avg | 81.55% | 79.81% | 75.32% | 45.89% | 2.11% | 79.12% | 4.78% | 0.793 |

## Faster-ISNet per architecture

| Architecture | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Strict leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 72.06% | 66.16% | 47.05% | 27.50% | 1.86% | 67.67% | 4.42% | 0.547 |

BG/FG means and worst cases use the same 12 generic background and 12 generic foreground
interventions as `results/cub/complete_metrics.csv`. Strict leakage is lower-is-better; the
accuracy columns follow the existing table's higher-is-better win/tie/loss convention. BG-only
accuracy remains a descriptive stress test and should not alone be read as causal reliance.

## Architecture win/tie/loss

Exact equality tolerance is 0. Counts are candidate versus reference after seed averaging
within architecture (trivial here because there is one seed).
Faster-ISNet is support-restricted to ResNet-50 and is deliberately excluded from these
six-architecture counts; its ResNet-only comparisons are retained in the delta and cost tables.

| Candidate | Reference | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Leakage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMAL | CE | 3/0/3 | 1/0/5 | 3/0/3 | 0/0/6 | 1/0/5 | 1/0/5 | 0/0/6 | 6/0/0 |
| CAMAL | FCAL V2 | 0/0/6 | 1/0/5 | 2/0/4 | 1/0/5 | 3/0/3 | 0/0/6 | 2/0/4 | 4/0/2 |
| GAIN-ext | CE | 1/0/5 | 2/0/4 | 3/0/3 | 0/0/6 | 1/0/5 | 2/0/4 | 0/0/6 | 6/0/0 |
| GAIN-ext | FCAL V2 | 1/0/5 | 1/0/5 | 1/0/5 | 0/0/6 | 0/0/6 | 1/1/4 | 0/0/6 | 5/0/1 |
| GAIN-ext | CAMAL | 2/0/4 | 3/0/3 | 3/0/3 | 0/0/6 | 0/0/6 | 3/0/3 | 0/0/6 | 2/0/4 |
| RRR | CE | 0/0/6 | 2/0/4 | 6/0/0 | 2/0/4 | 1/0/5 | 2/0/4 | 0/0/6 | 3/0/3 |
| RRR | FCAL V2 | 0/0/6 | 2/0/4 | 3/0/3 | 2/0/4 | 2/0/4 | 2/0/4 | 1/0/5 | 0/0/6 |
| RRR | CAMAL | 0/0/6 | 3/0/3 | 4/0/2 | 3/0/3 | 4/1/1 | 3/0/3 | 3/1/2 | 0/0/6 |
| RRR | GAIN-ext | 1/0/5 | 3/0/3 | 5/0/1 | 5/0/1 | 5/0/1 | 4/0/2 | 6/0/0 | 0/0/6 |

## Timing and allocated VRAM

### CAMAL

| Architecture | Total wall | Post-branch wall | Peak VRAM | Reference | Total overhead | Post-branch overhead | VRAM overhead |
|---|---:|---:|---:|---|---:|---:|---:|
| ResNet-50 | 0.130 h | 0.104 h | 1.745 GiB | CE | +5.5% | +6.9% | +0.2% |
| ResNet-50 | 0.130 h | 0.104 h | 1.745 GiB | FCAL V2 | +11.6% | +15.0% | -0.1% |
| DenseNet-121 | 0.175 h | 0.141 h | 2.160 GiB | CE | +20.6% | +26.8% | +1.7% |
| DenseNet-121 | 0.175 h | 0.141 h | 2.160 GiB | FCAL V2 | +2.7% | +3.4% | +0.6% |
| EfficientNet-B0 | 0.143 h | 0.108 h | 1.470 GiB | CE | +35.8% | +53.8% | +2.3% |
| EfficientNet-B0 | 0.143 h | 0.108 h | 1.470 GiB | FCAL V2 | +3.6% | +4.8% | +0.9% |
| ViT-Tiny/16 avg | 0.082 h | 0.068 h | 0.660 GiB | CE | +29.1% | +37.0% | +2.1% |
| ViT-Tiny/16 avg | 0.082 h | 0.068 h | 0.660 GiB | FCAL V2 | +4.9% | +5.9% | +1.1% |
| ViT-Small/16 avg | 0.105 h | 0.089 h | 1.383 GiB | CE | +27.8% | +34.2% | +0.2% |
| ViT-Small/16 avg | 0.105 h | 0.089 h | 1.383 GiB | FCAL V2 | +10.3% | +12.3% | -0.0% |
| ViT-Base/16 avg | 0.213 h | 0.186 h | 3.256 GiB | CE | +9.9% | +11.5% | +0.1% |
| ViT-Base/16 avg | 0.213 h | 0.186 h | 3.256 GiB | FCAL V2 | +11.2% | +13.0% | -0.0% |

### GAIN-ext

| Architecture | Total wall | Post-branch wall | Peak VRAM | Reference | Total overhead | Post-branch overhead | VRAM overhead |
|---|---:|---:|---:|---|---:|---:|---:|
| ResNet-50 | 0.197 h | 0.171 h | 3.089 GiB | CE | +60.0% | +76.1% | +77.4% |
| ResNet-50 | 0.197 h | 0.171 h | 3.089 GiB | FCAL V2 | +69.3% | +89.3% | +76.8% |
| ResNet-50 | 0.197 h | 0.171 h | 3.089 GiB | CAMAL | +51.7% | +64.7% | +77.0% |
| DenseNet-121 | 0.270 h | 0.237 h | 4.151 GiB | CE | +86.7% | +113.0% | +95.4% |
| DenseNet-121 | 0.270 h | 0.237 h | 4.151 GiB | FCAL V2 | +59.1% | +73.7% | +93.3% |
| DenseNet-121 | 0.270 h | 0.237 h | 4.151 GiB | CAMAL | +54.9% | +68.0% | +92.2% |
| EfficientNet-B0 | 0.189 h | 0.153 h | 2.820 GiB | CE | +78.8% | +118.5% | +96.3% |
| EfficientNet-B0 | 0.189 h | 0.153 h | 2.820 GiB | FCAL V2 | +36.4% | +48.9% | +93.4% |
| EfficientNet-B0 | 0.189 h | 0.153 h | 2.820 GiB | CAMAL | +31.7% | +42.1% | +91.8% |
| ViT-Tiny/16 avg | 0.109 h | 0.095 h | 1.182 GiB | CE | +71.9% | +91.4% | +82.9% |
| ViT-Tiny/16 avg | 0.109 h | 0.095 h | 1.182 GiB | FCAL V2 | +39.7% | +48.0% | +81.1% |
| ViT-Tiny/16 avg | 0.109 h | 0.095 h | 1.182 GiB | CAMAL | +33.2% | +39.7% | +79.1% |
| ViT-Small/16 avg | 0.193 h | 0.177 h | 2.407 GiB | CE | +135.3% | +166.6% | +74.4% |
| ViT-Small/16 avg | 0.193 h | 0.177 h | 2.407 GiB | FCAL V2 | +103.2% | +123.2% | +74.1% |
| ViT-Small/16 avg | 0.193 h | 0.177 h | 2.407 GiB | CAMAL | +84.2% | +98.7% | +74.1% |
| ViT-Base/16 avg | 0.350 h | 0.323 h | 5.273 GiB | CE | +80.5% | +93.4% | +62.1% |
| ViT-Base/16 avg | 0.350 h | 0.323 h | 5.273 GiB | FCAL V2 | +82.6% | +96.0% | +61.9% |
| ViT-Base/16 avg | 0.350 h | 0.323 h | 5.273 GiB | CAMAL | +64.2% | +73.4% | +61.9% |

### RRR

| Architecture | Total wall | Post-branch wall | Peak VRAM | Reference | Total overhead | Post-branch overhead | VRAM overhead |
|---|---:|---:|---:|---|---:|---:|---:|
| ResNet-50 | 0.532 h | 0.506 h | 3.461 GiB | CE | +332.1% | +421.2% | +98.8% |
| ResNet-50 | 0.532 h | 0.506 h | 3.461 GiB | FCAL V2 | +357.3% | +460.3% | +98.1% |
| ResNet-50 | 0.532 h | 0.506 h | 3.461 GiB | CAMAL | +309.6% | +387.4% | +98.3% |
| ResNet-50 | 0.532 h | 0.506 h | 3.461 GiB | GAIN-ext | +170.0% | +196.0% | +12.1% |
| DenseNet-121 | 0.900 h | 0.867 h | 4.184 GiB | CE | +521.6% | +679.7% | +97.0% |
| DenseNet-121 | 0.900 h | 0.867 h | 4.184 GiB | FCAL V2 | +429.6% | +535.9% | +94.9% |
| DenseNet-121 | 0.900 h | 0.867 h | 4.184 GiB | CAMAL | +415.5% | +514.8% | +93.8% |
| DenseNet-121 | 0.900 h | 0.867 h | 4.184 GiB | GAIN-ext | +232.9% | +266.0% | +0.8% |
| EfficientNet-B0 | 3.090 h | 3.055 h | 4.717 GiB | CE | +2827.4% | +4254.4% | +228.3% |
| EfficientNet-B0 | 3.090 h | 3.055 h | 4.717 GiB | FCAL V2 | +2133.4% | +2867.0% | +223.6% |
| EfficientNet-B0 | 3.090 h | 3.055 h | 4.717 GiB | CAMAL | +2056.4% | +2731.1% | +220.8% |
| EfficientNet-B0 | 3.090 h | 3.055 h | 4.717 GiB | GAIN-ext | +1537.4% | +1892.4% | +67.3% |
| ViT-Tiny/16 avg | 0.156 h | 0.143 h | 1.740 GiB | CE | +147.2% | +186.9% | +169.2% |
| ViT-Tiny/16 avg | 0.156 h | 0.143 h | 1.740 GiB | FCAL V2 | +100.8% | +121.9% | +166.5% |
| ViT-Tiny/16 avg | 0.156 h | 0.143 h | 1.740 GiB | CAMAL | +91.5% | +109.5% | +163.6% |
| ViT-Tiny/16 avg | 0.156 h | 0.143 h | 1.740 GiB | GAIN-ext | +43.8% | +49.9% | +47.2% |
| ViT-Small/16 avg | 0.273 h | 0.258 h | 3.576 GiB | CE | +233.6% | +287.6% | +159.0% |
| ViT-Small/16 avg | 0.273 h | 0.258 h | 3.576 GiB | FCAL V2 | +188.1% | +224.5% | +158.6% |
| ViT-Small/16 avg | 0.273 h | 0.258 h | 3.576 GiB | CAMAL | +161.1% | +188.9% | +158.6% |
| ViT-Small/16 avg | 0.273 h | 0.258 h | 3.576 GiB | GAIN-ext | +41.8% | +45.4% | +48.5% |
| ViT-Base/16 avg | 0.583 h | 0.556 h | 7.640 GiB | CE | +200.8% | +232.9% | +134.9% |
| ViT-Base/16 avg | 0.583 h | 0.556 h | 7.640 GiB | FCAL V2 | +204.3% | +237.5% | +134.6% |
| ViT-Base/16 avg | 0.583 h | 0.556 h | 7.640 GiB | CAMAL | +173.6% | +198.6% | +134.6% |
| ViT-Base/16 avg | 0.583 h | 0.556 h | 7.640 GiB | GAIN-ext | +66.7% | +72.2% | +44.9% |

### Faster-ISNet

| Architecture | Total wall | Post-branch wall | Peak VRAM | Reference | Total overhead | Post-branch overhead | VRAM overhead |
|---|---:|---:|---:|---|---:|---:|---:|
| ResNet-50 | 0.879 h | 0.853 h | 10.757 GiB | CE | +613.8% | +778.5% | +517.8% |
| ResNet-50 | 0.879 h | 0.853 h | 10.757 GiB | FCAL V2 | +655.4% | +844.5% | +515.7% |
| ResNet-50 | 0.879 h | 0.853 h | 10.757 GiB | CAMAL | +576.7% | +721.5% | +516.4% |
| ResNet-50 | 0.879 h | 0.853 h | 10.757 GiB | GAIN-ext | +346.1% | +398.9% | +248.3% |
| ResNet-50 | 0.879 h | 0.853 h | 10.757 GiB | RRR | +65.2% | +68.6% | +210.8% |

Total wall is summed train plus validation time for all 50 epochs. Post-branch wall covers epochs
6-50 only. Peak VRAM is PyTorch peak allocated CUDA memory, not device-resident memory. The first
five history records are record-for-record identical after JSON parsing within every architecture/method
pair. All runs identify an NVIDIA GeForce RTX 5070 Ti; timings are observed single-run values, not
uncertainty estimates, and may reflect uncontrolled system load.
The interrupted RRR EfficientNet-B0 epoch-46 attempt produced no history record and is excluded;
its efficiency row aggregates only the subsequently completed, recorded 50-epoch history.
Faster-ISNet likewise uses only the clean `resnet50-isnet-seed42` rerun. The interrupted corrupt
checkpoint directory is excluded. The clean rerun diverged nondeterministically early from the
interrupted attempt but converged late; only its recorded complete history is aggregated.

## Pareto audit

Pareto status is computed separately within each architecture. Maximized objectives are clean
accuracy, BG mean accuracy, and foreground-only accuracy; minimized objectives are strict leakage,
post-branch wall time, and peak allocated VRAM. The non-dominated methods are:

- ResNet-50: CE, FCAL V2, CAMAL, GAIN-ext
- DenseNet-121: CE, FCAL V2, CAMAL, GAIN-ext
- EfficientNet-B0: CE, FCAL V2, CAMAL, RRR
- ViT-Tiny/16 avg: CE, FCAL V2, CAMAL, GAIN-ext, RRR
- ViT-Small/16 avg: CE, FCAL V2, CAMAL, GAIN-ext
- ViT-Base/16 avg: CE, FCAL V2, CAMAL, GAIN-ext

## Provenance

CE and FCAL V2 metrics are independently recomputed from archived evaluation summaries and must
match `results/cub/complete_metrics.csv` to 1e-12. FCAL V2 is the archived `fcal-v5` run family
used by that table. General fixed candidates require complete current six-architecture matrices
under `outputs/baseline-fixed/cub`. Partial candidates are never admitted by that guard;
Faster-ISNet enters through a separate explicit ResNet-only support registry.
`selection_provenance.csv` retains run/config/checkpoint/history/evaluation hashes, validation-selected epoch,
best validation score, source location, split hashes, and the explicit fixed-default/test-evaluated
flags. `metric_deltas.csv`, `architecture_win_tie_loss.csv`, `efficiency_runs.csv`,
`efficiency_overhead.csv`, and `pareto.csv` retain exact machine-readable values.
