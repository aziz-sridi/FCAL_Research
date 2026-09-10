# Relative pooled-margin Stage-1 triage

## Decision

**Stop the relative pooled-margin formulation after the first CUB / ResNet-50 /
seed-42 slice.** Do not launch another architecture, dataset, or seed, and do
not run the remaining validation variants.

The frozen protocol is
[`configs/research/relative_pooled_margin_stage1.yaml`](../../../configs/research/relative_pooled_margin_stage1.yaml).
The machine-readable decision is
[`resnet50_seed42_triage.json`](resnet50_seed42_triage.json).

## Triage result

All values use the same 1,200-image CUB validation partition and the
validation-selected checkpoint. The background probe is the most severe FCAL
V2 validation view, `bg_random_rgb_s1`.

| Method | Clean | Random-RGB background | Foreground only | Strict leakage | Post-branch time |
|---|---:|---:|---:|---:|---:|
| CE | 80.92% | 27.58% | 70.33% | 0.561 | 349.6 s |
| FCAL V2 | 80.08% | 48.92% | 71.67% | 0.266 | 325.2 s |
| CAMAL | 78.92% | **60.33%** | **73.50%** | 0.316 | 373.9 s |
| GAIN-ext | 70.42% | 43.33% | 63.92% | **0.149** | 615.6 s |
| RRR | 67.92% | 47.17% | 63.17% | 0.466 | 1,822.0 s |
| Faster-ISNet | 71.92% | 48.00% | 67.58% | 0.539 | 3,071.1 s |
| Relative pooled margin | 80.00% | 42.08% | 71.50% | 0.256 | 551.9 s |

The candidate passed clean accuracy, leakage, peak-memory, and implementation
policy checks. It failed three hard gates:

- random-RGB background accuracy was 6.83 percentage points below V2;
- foreground-only accuracy was 0.17 points below V2;
- post-branch wall time was 1.70x V2 instead of at most 1.15x.

It achieved zero of the four predeclared material-improvement thresholds. The
protocol permits one adjusted run only for one narrow quality miss accompanied
by at least two material gains. This result therefore does not qualify for an
adjustment.

## Implementation audit

The candidate used one untouched input, one backbone forward, cached-feature
foreground/background pooling, and the existing classifier. Initial diagnostic
runs `r1` through `r3` were interrupted while removing unintended GPU
synchronization from valid-mask counting and rival selection. `r4` is the first
complete valid run and is the only run used above. Coefficients were unchanged
between the diagnostics and `r4`.

The next permitted experiment is the separately defined gated V2+ backup on
this same CUB / ResNet-50 / seed-42 slice. It must receive its own frozen
protocol before training and must not inherit coefficients selected from this
candidate's validation result.
