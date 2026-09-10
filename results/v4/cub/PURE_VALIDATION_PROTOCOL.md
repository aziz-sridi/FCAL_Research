# FCAL V4 pure-mode validation protocol

## Scope

This protocol covers `configs/cub/resnet50_fcal_v4_pure.yaml`. V4 receives the
ordinary augmented training image once and never constructs a second image:
`fcal_v4.intervention.enabled` is frozen to `false`. Masks affect losses only.

The official test split remains sealed unless both the regional-metric gate
and the validation-faithfulness gate pass completely.

## Frozen validation references

| Method | Clean | Leakage | BG mean | BG worst | FG mean | FG only |
|---|---:|---:|---:|---:|---:|---:|
| CE | 0.809167 | 0.561388 | 0.682564 | 0.275833 | 0.336410 | 0.703333 |
| FCAL V2 | 0.800833 | 0.266214 | 0.720449 | 0.489167 | 0.318269 | 0.716667 |
| CAMAL | 0.789167 | 0.315668 | 0.734744 | 0.603333 | 0.331795 | 0.735000 |
| GAIN-ext | 0.704167 | 0.148650 | 0.632051 | 0.433333 | 0.236218 | 0.639167 |
| RRR | 0.679167 | 0.465636 | 0.621282 | 0.471667 | 0.289615 | 0.631667 |

The local ResNet-50 fixed RRR checkpoint is not the cross-architecture tuned
winner. The broader article target remains RRR's 60.61% six-architecture mean
worst-background accuracy; CAMAL's 60.33% validation value supplies the local
seed-42 threshold.

## Regional-metric gate

| Metric | Required validation result | Target source |
|---|---:|---|
| Clean accuracy | >= 0.798333 | within 0.25 percentage points of V2 |
| Strict leakage | <= 0.266214 | at least V2-level leakage |
| Background mean | >= 0.734744 | at least CAMAL validation mean |
| Background worst | >= 0.603333 | at least CAMAL validation worst |
| Foreground mean | >= 0.331795 | at least CAMAL validation mean |
| Foreground only | >= 0.735000 | at least CAMAL validation value |

## Faithfulness gate

Use the frozen 500-image validation subset with selection seed 2026.

| Metric | Required validation result | Target source |
|---|---:|---|
| Mask IoU | >= 0.308455 | no worse than CE |
| Insertion AUC | >= 0.690000 | approximately V2/CE level |
| Removal AUC | <= 0.082249 | at least V2 level |

These are joint gates. Failure of any row seals the V4 test split and makes
the result a validation-only diagnostic.

## Predeclared V4.1 coverage refinement

The initial pure candidate passed clean, leakage, and background mean but
failed background worst, foreground mean/only, and mask IoU. Before any V4
test access, one refinement is admitted under the same gates:

- configuration: `configs/cub/resnet50_fcal_v4_pure_coverage.yaml`;
- alignment: min-max normalized positive CAM, matching the common-map
  evaluator, with weight 0.10;
- sensitivity: background target-margin sensitivity power mean with power 2
  and weight 0.010, emphasizing concentrated hotspots;
- intervention remains disabled;
- all other data, model, optimizer, V2, budget, and selection settings remain
  unchanged.

This refinement must pass every original regional and faithfulness threshold;
the thresholds are not relaxed in response to the first run.

## Predeclared V4.2 regional-dominance refinement

V4.1 improved strict leakage and worst-background accuracy but its pixelwise
mask MSE reduced clean, mean-background, foreground, and insertion scores while
barely improving IoU.  Before any V4 test access, one mechanism-level correction
is admitted under the same unchanged gates:

- configuration: `configs/cub/resnet50_fcal_v4_pure_dominance.yaml`;
- remove pixelwise mask alignment rather than tuning its coefficient again;
- require the evaluator-normalized mean foreground CAM to exceed mean
  background CAM by 0.60 using a squared one-sided hinge, weight 0.20;
- restore the V4.0 scale-free background-sensitivity setting (power 1,
  weight 0.005), because V4.1's hotspot emphasis over-regularized clean and
  mean-regional performance;
- intervention remains disabled and every other run condition is unchanged.

The 0.60 margin is fixed from mechanism diagnostics on validation maps: V4.0
and V4.1 had mean regional gaps near 0.36, while the independently tuned CAMAL
reference's training gap was about 0.77.  The hinge stops rewarding the loss
once the intermediate target is met.

## Predeclared V4.3 late-hotspot curriculum

V4.2 passed clean, leakage, background mean, foreground-only, insertion, and
removal gates.  It narrowly missed foreground mean and IoU, while its sole
large failure was random-RGB background accuracy.  One final validation-only
candidate is admitted before ending the single-seed design cycle:

- configuration: `configs/cub/resnet50_fcal_v4_pure_curriculum.yaml`;
- regional-dominance margin increases from 0.60 to 0.65, retaining the same
  weight and one-sided stopping behavior;
- background sensitivity uses power 2 and weight 0.012, but remains inactive
  through epoch 25 and ramps over epochs 26--35;
- intervention remains disabled and all other conditions and gates remain
  unchanged.

This tests the specific hypothesis exposed by V4.1 and V4.2: hotspot
sensitivity can address worst-background behavior if introduced only after the
clean classifier and regional evidence ordering have formed.  No further
single-seed coefficient iteration is allowed after this candidate.

## Fixed run conditions

- Dataset and split: existing CUB-200-2011 seed-42 split.
- Backbone: ResNet-50.
- Initialization: exact archived paired CE epoch-5 branch.
- Total budget: 50 epochs.
- Checkpoint selection: validation clean accuracy only.
- V4 coefficients: exactly those in the committed configuration.
- No test result from V3, V3.1, or any baseline participates in V4 checkpoint
  or coefficient selection.
