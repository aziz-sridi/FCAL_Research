# FCAL V2.5--V2.9 frozen validation protocol

## Common constraints

All five candidates are exact FCAL V2 successors. Each minibatch forwards the
ordinary training tensor exactly once. The mask is applied only to CAMs or the
already-computed final feature map. There is no crop, erase, replacement,
counterfactual view, augmentation change, input gradient, second backbone
forward, architecture change, or inference change.

The coefficients below were frozen before any candidate training run. They
were scale-calibrated on 16 fixed batches from the shared CE epoch-5 branch;
this calibration did not update parameters or inspect candidate outcomes.

| Version | Added mechanism | Frozen nonzero coefficients |
|---|---|---|
| V2.5 | Persistent native-grid Seed-and-Grow | front 0.10; retention 0.25 |
| V2.6 | Native foreground-over-background CAM dominance | dominance 0.08 |
| V2.7 | Foreground-pooled classification from existing features | foreground 0.02 |
| V2.8 | Foreground sufficiency plus background non-predictiveness | foreground 0.01; background 0.001 |
| V2.9 | Restrained combination of V2.5--V2.8 | front 0.05; retention 0.125; dominance 0.04; foreground 0.005; background 0.0005 |

The shared growth settings are acceptance 0.35, front target 0.80, retention
target 0.90, temperature 1.0, squared hinge, and mask threshold 0.50. Native
dominance uses margin 0.60 and a squared hinge. Regional classification uses
temperature 1.0, foreground target-vs-logsumexp-rivals margin 0.0, and maximum
background margin -2.0.

## Frozen runs

- Dataset/split: existing CUB-200-2011 seed-42 train/validation split.
- Backbone: ResNet-50; no new learned parameters.
- Initialization: exact paired CE epoch-5 branch after split-hash verification.
- Budget: epochs 6--50 of the common 50-epoch schedule.
- Selection: clean validation accuracy only within each run.
- All five run once at seed 42 before any formulation may be revised.
- Existing CUB test data remains unopened for every candidate in this screen.

## Joint promotion screen

| Metric | Frozen requirement |
|---|---:|
| Clean accuracy | >= 0.800833 (V2) |
| Strict leakage | <= 0.266214 (V2) |
| Background mean | >= 0.731603 (V2+) |
| Background worst | >= 0.526667 (V2+) |
| Foreground mean | >= 0.325705 (V2.3) |
| Foreground only | >= 0.735000 (CAMAL) |
| Post-branch wall time | <= 373.859 s (CAMAL) |
| Peak allocated CUDA memory | <= 1.9218 GiB (1.10x V2) |

A formulation must pass every row before a three-seed confirmation. Because
the current CUB test has historical development exposure from earlier work,
passing this screen supports a candidate claim only; a state-of-the-art claim
requires a newly locked confirmation split or external benchmark.
