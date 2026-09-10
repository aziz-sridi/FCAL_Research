# FCAL V4 pure original-input loss: single-seed validation report

## Outcome

V4 is implemented as a genuine original-input loss with an explicitly optional
interventional mode. Four pure ResNet-50 seed-42 candidates were trained from
the identical archived CE epoch-5 branch on the fixed CUB-200-2011 split. No
candidate passed all nine frozen validation gates, so the official test split
was not opened and these experiments do not establish a state-of-the-art
result.

V4.2 (`resnet50_fcal_v4_pure_dominance.yaml`) is the best balanced candidate.
It beats the local V2 validation reference on clean accuracy (80.50% versus
80.08%), leakage (0.164 versus 0.266), background mean (73.54% versus 72.04%),
foreground-only accuracy (74.25% versus 71.67%), insertion AUC, and removal
AUC. It fails foreground mean, worst-background accuracy, and the deliberately
strict IoU gate.

## Frozen validation results

| Candidate | Clean | Leak down | BG mean | BG worst | FG mean | FG only | IoU | Insert | Remove down | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V4.0 RMS alignment | 80.00% | **0.110** | **74.16%** | 54.50% | **32.97%** | 73.00% | 0.195 | **0.697** | 0.080 | 5/9 |
| V4.1 pixel coverage + hotspot | 77.92% | **0.105** | 71.80% | 59.33% | 31.11% | 72.25% | 0.212 | 0.641 | **0.073** | 2/9 |
| V4.2 regional dominance | **80.50%** | **0.164** | **73.54%** | 47.83% | 31.92% | **74.25%** | 0.296 | **0.693** | **0.082** | **6/9** |
| V4.3 late hotspot curriculum | 79.33% | **0.179** | 73.22% | 58.83% | 30.99% | 72.58% | 0.301 | 0.674 | **0.074** | 2/9 |
| Required | >=79.83% | <=0.266 | >=73.47% | >=60.33% | >=33.18% | >=73.50% | >=0.308 | >=0.690 | <=0.082 | 9/9 |

Bold result cells pass their corresponding frozen threshold. All faithfulness
values use the same deterministic 500-image validation subset and common
ground-truth-class CAM recipe.

## What each experiment established

- V4.0 preserved V2-level clean and mean-background performance and produced
  very low leakage, but its weak pixel alignment did not yield mask IoU.
- V4.1 showed that stronger pixelwise CAM-to-mask MSE and early power-2 input
  sensitivity can improve worst-background behavior, but they over-regularize
  clean, foreground, and insertion performance.
- V4.2 replaced silhouette imitation with a one-sided foreground-over-
  background evidence margin. This recovered clean accuracy and improved IoU
  from 0.195 to 0.296 while preserving insertion/removal faithfulness.
- V4.3 delayed the power-2 sensitivity penalty until epoch 26. It raised the
  random-RGB worst case from 47.83% to 58.83%, demonstrating that the curriculum
  works directionally, but the remaining robustness came with unacceptable
  clean and foreground costs.

## Interpretation

The single-seed evidence exposes a real Pareto boundary. Loss-only supervision
on the observed image can reduce measured background evidence and local input
sensitivity, but random-RGB background replacement is an off-manifold change.
Without observing such changes, the pure loss does not reliably learn complete
invariance to them. V3.1's stronger worst-background result is therefore not
free: its intervention supplies information unavailable to a pure per-image
objective.

The recommended V4 API keeps both scientifically distinct modes:

1. pure mode (`intervention.enabled: false`) for a regular original-input loss;
2. optional interventional mode (`true`) when maximum context-shift robustness
   is more important than the no-manipulation constraint.

The next defensible step is not more tuning on this validation split. It is a
fresh multi-seed, multi-architecture study of the frozen V4.2 mechanism, with
V4.3 retained as a predeclared robustness ablation. Until then, V4.2 is a
promising Pareto candidate, not a new state of the art.
