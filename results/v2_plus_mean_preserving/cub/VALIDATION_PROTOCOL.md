# Mean-preserving V2+ seed-42 validation protocol

## Frozen hypothesis

V2+ improves strict leakage and mean worst-background accuracy but reduces
clean accuracy on all three CUB ResNet-50 test seeds. The proposed repair keeps
the complete V2 objective and the frozen V2+ coefficient and margins. It
changes only the gradient path of the V2+ competitor term:

```text
mu(S) = mean_spatial(S)
S_mp = S + stopgrad(mu(S)) - mu(S)

forward(S_mp) = S
sum_spatial(d S_mp / d S) projects auxiliary gradients to zero mean
```

The target and two selected rival maps use `S_mp` inside the competitor loss.
V2 containment and distributed support continue to use the ordinary maps.
For GAP classifiers, this removes the competitor term's direct gradient along
the bias-free clean-logit direction while retaining spatial redistribution.

## Constraint audit

- One original, untouched input tensor.
- Exactly one backbone forward and one ordinary backward.
- No masking, erasing, replacement, corruption, recomposition, or second view.
- No new parameters and no inference change.
- Same V2+ coefficient, margins, optimizer, schedule, data, and epoch-5 parent.
- Only validation data may be evaluated during this screen.

## Frozen comparison and gate

The primary comparator is the existing CUB ResNet-50 seed-42 V2 run. The
candidate advances only if all of the following hold on the 1,200-image
validation split:

| Metric | Required |
|---|---:|
| Clean accuracy | >= 79.83% |
| Strict leakage | <= 0.266 |
| Background mean | >= 72.04% |
| Background worst | >= 48.92% |
| Foreground mean | >= 31.83% |
| Foreground only | >= 71.67% |
| Post-branch wall time | <= 373.95 s (1.15x V2) |
| Peak allocated VRAM | <= 1.85 GiB |

These thresholds and the implementation are frozen before training. Failure
of any gate stops the candidate; the CUB test split remains sealed. A passing
seed-42 run permits unchanged validation replication at seeds 7 and 123 before
any test access.
