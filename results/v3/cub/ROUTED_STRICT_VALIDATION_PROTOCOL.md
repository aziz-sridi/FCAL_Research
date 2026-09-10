# FCAL V3.1 routed-strict validation protocol

## Candidate

This protocol covers the single ResNet-50 seed-42 candidate configured by
`configs/cub/resnet50_fcal_v3_routed_strict.yaml`. It differs from balanced V3
in exactly two loss behaviors:

1. prediction and foreground-evidence consistency apply only to samples whose
   backgrounds were replaced; foreground-corrupted samples retain paired CE
   and the complete V2 objective without distribution matching;
2. a weak normalized positive-evidence penalty applies to the tolerance shell
   between the strict foreground mask and V2's tolerant foreground region.

The official test split must remain unopened unless all validation conditions
below pass jointly.

## Frozen validation references

| Method | Clean | Leakage | BG mean | BG worst | FG mean | FG only |
|---|---:|---:|---:|---:|---:|---:|
| V2 | 0.800833 | 0.266214 | 0.720449 | 0.489167 | 0.318269 | 0.716667 |
| V3 pilot | 0.787500 | 0.305726 | 0.750449 | 0.654167 | 0.297564 | 0.765000 |
| Balanced V3 | 0.803333 | 0.284860 | 0.772244 | 0.687500 | 0.433718 | 0.770833 |

## Joint acceptance gate

| Metric | Required validation result | Purpose |
|---|---:|---|
| Clean accuracy | >= 0.800833 | retain at least V2 validation accuracy |
| Strict leakage | <= 0.279860 | improve balanced V3 by at least 0.005 absolute |
| Background mean | >= 0.767244 | retain balanced V3 within 0.50 percentage points |
| Background worst | >= 0.667500 | retain balanced V3 within 2.00 percentage points |
| Foreground mean | >= 0.423718 | retain balanced V3 within 1.00 percentage point |
| Foreground only | >= 0.765833 | retain balanced V3 within 0.50 percentage points |

Passing only clean accuracy or leakage is insufficient. If any condition
fails, preserve the artifacts, do not evaluate this candidate on test, and use
only validation evidence for the next decision.

## Fixed run conditions

- Dataset: CUB-200-2011.
- Split: existing seed-42 train/validation split.
- Backbone: ResNet-50.
- Initialization: archived paired CE epoch-5 branch.
- Total training budget: 50 epochs.
- Checkpoint selection: validation clean accuracy only.
- Test data do not participate in the design, weighting, or selection of this
  candidate.
