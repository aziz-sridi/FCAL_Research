# FCAL V3 balanced-iteration protocol

## Purpose

The completed ResNet-50 seed-42 V3 result is a pilot diagnosis. The balanced
V3 candidate must be selected using the frozen CUB-200-2011 validation split;
its test split remains sealed until every acceptance threshold below is met.

## Frozen references

| Method | Clean | Leakage | BG mean | BG worst | FG mean | FG only |
|---|---:|---:|---:|---:|---:|---:|
| CE | 0.809167 | 0.561388 | 0.682564 | 0.275833 | 0.336410 | 0.703333 |
| V2 | 0.800833 | 0.266214 | 0.720449 | 0.489167 | 0.318269 | 0.716667 |
| V3 pilot | 0.787500 | 0.305726 | 0.750449 | 0.654167 | 0.297564 | 0.765000 |

## Acceptance gate

The balanced candidate may be evaluated on the test split only if its single
selected validation checkpoint satisfies all of these conditions:

| Metric | Required validation result | Rationale |
|---|---:|---|
| Clean accuracy | >= 0.798333 | within 0.25 percentage points of V2 |
| Strict leakage | <= 0.286214 | within 0.02 absolute of V2 |
| Background mean | >= 0.745449 | within 0.50 percentage points of pilot V3 |
| Background worst | >= 0.634167 | within 2.00 percentage points of pilot V3 |
| Foreground mean | >= 0.313269 | within 0.50 percentage points of V2 |
| Foreground only | >= 0.760000 | within 0.50 percentage points of pilot V3 |

This is a joint gate: passing selected metrics is insufficient. If any row
fails, do not inspect the candidate's test performance. Use only validation
results to diagnose and construct the next ablation.

## Fixed run conditions

- Dataset and split: the existing seed-42 CUB-200-2011 train/validation split.
- Backbone: ResNet-50.
- Initialization: the archived paired CE epoch-5 branch.
- Candidate configuration: `configs/cub/resnet50_fcal_v3_balanced.yaml`.
- Seed: 42.
- Checkpoint selection: the configuration's validation-clean criterion.
- No test metrics participate in loss design, hyperparameter selection, or
  checkpoint selection.
