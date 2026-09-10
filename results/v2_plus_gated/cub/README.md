# Gated V2+ Stage-1 triage

## Decision

**Stop gated V2+ after the first CUB / ResNet-50 / seed-42 slice.** Do not
launch another seed, architecture, or dataset, and do not evaluate the other
27 validation views.

The gate and decision thresholds were frozen before training in
[`configs/cub/resnet50_fcal_v2_plus_gated.yaml`](../../../configs/cub/resnet50_fcal_v2_plus_gated.yaml)
and
[`configs/research/gated_v2_plus_stage1.yaml`](../../../configs/research/gated_v2_plus_stage1.yaml).
The machine-readable screen is
[`resnet50_seed42_triage.json`](resnet50_seed42_triage.json).

## Triage result

All quality values use the same 1,200-image CUB validation partition and each
method's validation-selected checkpoint. The registered background probe is
`bg_random_rgb_s1`.

| Method | Clean | Random-RGB background | Foreground only | Strict leakage | Post-branch time |
|---|---:|---:|---:|---:|---:|
| CE | **80.92%** | 27.58% | 70.33% | 0.561 | 349.6 s |
| FCAL V2 | 80.08% | 48.92% | 71.67% | 0.266 | 325.2 s |
| Ungated V2+ | 79.67% | 52.67% | 73.25% | 0.244 | 377.3 s |
| CAMAL | 78.92% | **60.33%** | **73.50%** | 0.316 | 373.9 s |
| GAIN-ext | 70.42% | 43.33% | 63.92% | **0.149** | 615.6 s |
| RRR | 67.92% | 47.17% | 63.17% | 0.466 | 1,822.0 s |
| Faster-ISNet | 71.92% | 48.00% | 67.58% | 0.539 | 3,071.1 s |
| Gated V2+ | 80.25% | 44.00% | 71.83% | 0.263 | 397.3 s |

Gating recovered 0.58 percentage points of clean accuracy relative to ungated
V2+ and cleared the clean, foreground-only, leakage, and peak-memory hard
gates. It did not preserve V2+'s shortcut benefit: random-RGB background
accuracy fell 8.67 points from ungated V2+ and 4.92 points below V2. Runtime
also exceeded the frozen 1.15x-V2 ceiling by 23.36 seconds.

The candidate therefore failed two hard gates. It reached only one material
threshold (clean accuracy), reached no shortcut-related material threshold,
and missed the registered strict-win counts against RRR and Faster-ISNet. It
was Pareto non-dominated, but the full protocol is conjunctive, so that is not
enough to advance.

## Gate behavior and stop rationale

The implementation used one untouched input and one backbone forward. The
detached mean gate decreased from 0.261 at epoch 6 to 0.0006 at epoch 50 as
mean target confidence increased from 0.719 to 0.999 and foreground support
increased from 0.139 to 0.783. Thus the mechanism behaved as designed, but its
late suppression traded away the useful V2+ background effect.

This is not a narrow, single-metric miss: background robustness, runtime,
material-gain count, shortcut-gain requirement, and two competition checks all
failed. The frozen protocol permits no adjusted backup run, so retuning the
gate on this validation result would be unjustified. The candidate is marked
as a dead end for this staged screen.
