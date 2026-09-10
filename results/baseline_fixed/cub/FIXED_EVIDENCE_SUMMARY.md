# Fixed/default-arm evidence summary

This is a descriptive summary of the seed-42 CUB fixed/default arm. All methods use validation-accuracy checkpoint selection, and the held-out test metrics are not used for setting or checkpoint selection. Because there is one seed per architecture, the counts below measure cross-architecture consistency, not training-run uncertainty or statistical significance.

## Six-architecture comparison with FCAL V2

Each cell is the candidate method's **wins/ties/losses against FCAL V2** across ResNet-50, DenseNet-121, EfficientNet-B0, ViT-Tiny/16 avg, ViT-Small/16 avg, and ViT-Base/16 avg. Accuracy metrics are higher-is-better; strict leakage is lower-is-better. Exact equality tolerance is zero.

| Candidate | Clean | BG mean | BG worst | FG mean | FG worst | FG only | BG only | Strict leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMAL | 0/0/6 | 1/0/5 | 2/0/4 | 1/0/5 | 3/0/3 | 0/0/6 | 2/0/4 | 4/0/2 |
| GAIN-ext | 1/0/5 | 1/0/5 | 1/0/5 | 0/0/6 | 0/0/6 | 1/1/4 | 0/0/6 | 5/0/1 |
| RRR | 0/0/6 | 2/0/4 | 3/0/3 | 2/0/4 | 2/0/4 | 2/0/4 | 1/0/5 | 0/0/6 |

FCAL V2 has higher clean accuracy than CAMAL and RRR on all six architectures and than GAIN-ext on five of six. It also has higher foreground-only accuracy than CAMAL on all six, and than GAIN-ext and RRR on four of six. Perturbation robustness is less uniform: CAMAL wins up to three architectures depending on the metric, GAIN-ext wins at most one, and RRR wins two or three on each BG/FG aggregate. The principal counter-trade-off is strict leakage: lower leakage favors CAMAL on four of six architectures and GAIN-ext on five of six, whereas FCAL V2 has lower leakage than RRR on all six. BG-only accuracy is retained as a descriptive stress test and should not by itself be interpreted as causal background reliance.

## Observed FCAL-relative training cost

Ranges are the minimum and maximum across the six architectures, calculated per architecture relative to FCAL V2. Total wall time sums training and validation over all 50 epochs; post-branch wall time covers epochs 6--50. Peak VRAM is PyTorch peak allocated CUDA memory.

| Candidate | Architectures | Total wall overhead | Post-branch wall overhead | Peak VRAM overhead |
|---|---:|---:|---:|---:|
| CAMAL | 6 | +2.75% to +11.62% | +3.43% to +14.97% | -0.11% to +1.10% |
| GAIN-ext | 6 | +36.40% to +103.24% | +48.01% to +123.20% | +61.94% to +93.43% |
| RRR | 6 | +100.85% to +2133.35% | +121.92% to +2866.96% | +94.87% to +223.59% |
| Faster-ISNet | 1 (ResNet-50 only) | +655.37% | +844.46% | +515.67% |

These are observed single-run measurements on an NVIDIA GeForce RTX 5070 Ti, not timing confidence intervals; uncontrolled system load may contribute. The RRR EfficientNet-B0 row is an extreme observed outlier and should be reported transparently rather than treated as a stable architecture-wide multiplier.

## Faster-ISNet support caveat

Faster-ISNet is audited only on ResNet-50 and is therefore excluded from the six-architecture W/T/L table. On that one architecture, FCAL V2 versus Faster-ISNet records 78.49% versus 72.06% clean accuracy, 72.11% versus 66.16% BG mean, 48.86% versus 47.05% BG worst, 31.87% versus 27.50% FG mean, 1.59% versus 1.86% FG worst, 72.76% versus 67.67% foreground-only accuracy, and 0.282 versus 0.547 strict leakage. This single supported pairing does not justify a cross-architecture Faster-ISNet claim.

## Interrupted-run accounting

- The interrupted RRR EfficientNet-B0 epoch-46 attempt produced no history record and is excluded. Its cost row uses only the subsequently completed 50 recorded epochs.
- The interrupted Faster-ISNet directory with corrupt checkpoints is excluded. Metrics and cost use only the clean, complete `resnet50-isnet-seed42` rerun. That rerun diverged nondeterministically early from the interrupted attempt but converged late; only the complete rerun is aggregated.

Machine-readable sources: `comparison_metrics.csv`, `metric_deltas.csv`, `architecture_win_tie_loss.csv`, `efficiency_runs.csv`, `efficiency_overhead.csv`, and `selection_provenance.csv` in this directory.
