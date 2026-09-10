# FCAL V2.7 refinement validation report

## Outcome

No refinement is promoted. Exact hard-tail local supervision (V2.7a) and
uncancelled channel containment (V2.7b) failed the cheap-compute screen before
performance evaluation. Their partial histories were preserved, and neither
was evaluated on the stress suite or test set.

The cheaper smooth formulation, V2.7d, completed. It improves clean accuracy,
strict leakage, and foreground-corruption robustness, but substantially
regresses background robustness and foreground-only accuracy. It is therefore
a useful mechanistic result rather than a V2.7 successor.

## Performance

| Method | Clean | Leak down | BG mean | BG worst | FG mean | FG only | Final gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2.7 | 80.00% | 0.256 | **73.06%** | **52.42%** | 32.04% | **73.42%** | 1/6 |
| **V2.7d soft-min rival** | **80.08%** | **0.251** | 72.31% | 44.00% | **33.40%** | 72.00% | **3/6** |

V2.7d passes the final clean, leakage, and foreground-mean targets. Against
V2.7 it gains 0.08 percentage points clean, lowers leakage by 0.0047, and gains
1.36 points foreground mean. It loses 0.74 points background mean, 8.42 points
worst-background accuracy, and 1.42 points foreground-only accuracy. Because
the advancement rule requires non-inferiority on all six metrics, it fails.

## Compute

The machine entered a slower runtime state during these experiments. A
contemporaneous V2.7 control sustained roughly 533 samples/s, versus the
original V2.7 measurement of 719 samples/s. Absolute current wall clocks are
therefore not directly comparable with the earlier CAMAL gate.

- V2.7a was stopped after epoch 11; its steady epoch cost was about 19.7%
  above the contemporaneous V2.7 control.
- The final fused V2.7b implementation was stopped after epoch 8; its observed
  steady cost was about 51% above control.
- V2.7d completed with an 8.33% steady training-time overhead over the
  contemporaneous control.
- Scaling only V2.7's original training component by that paired overhead gives
  a control-normalized V2.7d wall estimate of 380.33 seconds, above the frozen
  373.859-second CAMAL limit. Raw wall time was 615.94 seconds and includes the
  machine slowdown plus a second CUDA compilation epoch caused by the explicit
  stop/control/resume audit.
- Peak allocated memory remained 1.7472 GiB.

V2.7d consequently fails the compute gate even under paired normalization.

## Interpretation

The soft-min rival loss does what it was designed to do: more foreground
locations become class-discriminative, producing a large foreground-stress
gain. But strengthening clean-image local margins does not constrain how those
features respond to an extreme background shift. It also allows the full-mask
foreground pool to be less self-sufficient, explaining the foreground-only
regression.

This rules out simply making V2.7 more spatially demanding. The next credible
loss-only direction is a *relative pooled margin*: require the foreground
target margin to exceed the background target margin by a one-sided gap, while
attenuating (not eliminating) gradients through the background branch. It can
reuse V2.8's cheap two regional pools and avoids both V2.8's absolute background
suppression and V2.7d's per-cell rival projection. That proposal must be a new,
separately frozen experiment rather than a post-hoc combination here.

## Post-hoc random-RGB exclusion audit

At the request of the follow-up analysis, the extreme per-pixel
`bg_random_rgb_s1` condition was removed from both the background mean and
background worst-case calculation. The same exclusion was applied to every
method, including the original FCAL V2 seed-42 reference. This is a secondary
structured-perturbation view only; it does not replace the frozen advancement
gate above.

| Method | BG mean without random RGB | BG worst without random RGB |
|---|---:|---:|
| FCAL V2 original | 73.97% | 63.92% |
| CAMAL | 74.57% | **65.17%** |
| V2+ | **74.87%** | 64.33% |
| V2.3 | 73.88% | 61.33% |
| V2.7 | 74.78% | 64.83% |
| V2.8 | 74.83% | 63.33% |
| V2.7d | 74.67% | 63.42% |

For original V2, the full-suite background mean/worst are 72.04%/48.92%; its
excluded `bg_random_rgb_s1` accuracy is 48.92%, and its revised worst case is
`bg_random_solid_s1` at 63.92%. V2.7d therefore exceeds original V2 by 0.70
percentage points on the revised background mean, but trails it by 0.50 points
on the revised background worst case.

If `fg_random_rgb_s1` is analogously removed from the foreground mean, original
V2 scores 34.35% and V2.7d scores 35.99%. The random-RGB results remain reported
separately rather than silently discarded.

## Verification and status

- One original, untouched image and one backbone forward per candidate.
- No masked image, intervention, second view, input gradient, or inference change.
- Full suite before training: 178 tests passed; scoped Ruff passed.
- Validation only; CUB test remained unopened.
- V2.7c combination was not created because its precondition failed.
