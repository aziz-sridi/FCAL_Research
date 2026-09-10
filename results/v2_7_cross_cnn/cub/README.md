# FCAL V2.7 cross-CNN seed-42 screen

## Decision

V2.7 remains a strong ResNet-50-specific candidate, but it does not currently
qualify as a general CNN successor to FCAL V2.

- ResNet-50 passed the current V2-relative, competition-aware screen.
- EfficientNet-B0 completed but failed the three-view triage.
- DenseNet-121 was stopped during epoch 10 after a decisive compute rejection.

All candidates preserve the ResNet coefficient
`foreground_region_weight: 0.02`, use one untouched input and one backbone
forward, add no parameters, and leave inference unchanged.

## EfficientNet-B0

The complete seed-42 run selected a 79.17% clean-validation checkpoint, which
improves on V2's 78.58%. The shortcut and sufficiency results regressed.

| Metric | FCAL V2 | FCAL V2.7 | Result |
|---|---:|---:|---|
| Clean | 78.58% | 79.17% | pass, material gain |
| Random-RGB background | 29.33% | 25.25% | fail |
| Foreground only | 69.92% | 62.58% | fail |
| Strict leakage | 0.46584 | 0.46602 | fail |
| Post-branch time | 370.68 s | 427.00 s | fail |
| Peak VRAM | 1.458 GiB | 1.463 GiB | pass |

The candidate achieved only one material gain and no shortcut-related gain.
It was Pareto non-dominated, but missed the frozen strict-win requirements
against CE, CAMAL, and GAIN-ext. No remaining validation variants were run.
The complete machine report is
[`efficientnet_b0_seed42_triage.json`](efficientnet_b0_seed42_triage.json).

## DenseNet-121

The DenseNet candidate completed epochs 6 through 9. Its three steady epochs
took 19.48--19.78 seconds each, after a 149.52-second first active epoch. The
frozen 1.15x-V2 budget is 564.19 seconds for all 45 post-branch epochs.

Using the observed steady median gives a projected total of 1,008.78 seconds.
After epoch 9, each remaining epoch would have needed to average at most 8.68
seconds to recover the budget, less than half the observed cost. Training was
therefore interrupted during epoch 10. Peak memory remained within its gate.

Because the schedule is incomplete, the partial checkpoint is not eligible
for quality evaluation and its partial validation accuracy must not be
compared with completed models. The machine-readable compute decision is
[`densenet121_seed42_compute_screen.json`](densenet121_seed42_compute_screen.json).

## Interpretation

Foreground-pooled classification transfers unevenly across CNN feature
extractors. It improves the ResNet-50 balance and EfficientNet clean accuracy,
but EfficientNet loses shortcut robustness and foreground sufficiency while
DenseNet incurs unacceptable training cost. No coefficient was tuned per
architecture, so these failures define transfer boundaries rather than new
hyperparameter selections.
