# Mean-preserving V2+ seed-42 validation report

## Decision

**Stop. Do not evaluate regional validation views or the CUB test split, and
do not tune the projection strength or V2+ coefficient on this result.**

The candidate's best clean-validation accuracy is **79.00%**, selected at
epoch 41. This is 1.08 percentage points below the matched V2 reference
(80.08%), 0.67 points below ordinary V2+ (79.67%), and 0.83 points below the
predeclared 79.83% clean gate. Because the first hard quality gate failed, the
remaining validation stress suite was not opened.

## Frozen change

Only the V2+ competitor term received the straight-through proxy

```text
S_mp = S + stopgrad(mean_spatial(S)) - mean_spatial(S).
```

`S_mp` is numerically identical to `S` in the forward pass, while its
auxiliary gradient sums to zero over each spatial map. V2 containment and
distributed support continued to use the original evidence maps. Every V2+
coefficient and all optimizer, schedule, data, and initialization settings
were unchanged.

## Compute and stability

| Quantity | Result | Gate |
|---|---:|---:|
| Post-branch training + validation | 369.26 s | <= 373.95 s |
| Aggregate training throughput | 680.1 samples/s | descriptive |
| Peak allocated VRAM | 1.7472 GiB | <= 1.85 GiB |
| Selected epoch | 41 | clean validation only |
| Last-five clean validation mean | 78.45% | descriptive |

The candidate passes the frozen time and memory limits. Its steady-state late
epochs run around 770--786 samples/s; aggregate throughput includes the first
active compilation epoch. No numerical failure occurred.

## Interpretation

The result falsifies the proposed repair. Protecting the spatial mean of the
target and rival evidence gradients is not enough to protect fine-grained
clean classification. A zero-mean auxiliary gradient can still change which
parts and textures support nearby bird classes, and shared backbone parameter
updates need not preserve the next forward pass's global feature mean.

The failure also shows why the proxy is not a true parameter-space constraint:
it is an activation-space first-order projection. A stronger clean-gradient
projection would require extra gradient calculations and would conflict with
the cheap-training objective. FCAL V2 therefore remains the supported anchor.

## Constraint and provenance audit

- One untouched training tensor and exactly one backbone forward per step.
- No image masking, corruption, erasing, replacement, or recomposition.
- No new learned parameters and no inference change.
- Exact archived seed-42 CE epoch-5 parent and split-hash relocation guard.
- Official test split not evaluated.
- Machine artifacts are under
  `outputs/v2_plus_mean_preserving/cub/resnet50-fcal-v2-plus-mean-preserving-seed42/`.
