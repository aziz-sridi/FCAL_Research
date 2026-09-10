# FCAL V2.5--V2.9 seed-42 validation report

## Outcome

None of the five frozen loss-only candidates passes the joint promotion
screen, so none is promoted to three seeds or evaluated on the CUB test set.
V2.8 Region Contrast is the best clean/localization candidate: it reaches
80.83% clean accuracy and reduces strict leakage to 0.230 while completing
below the CAMAL compute limit. V2.7 Foreground Pool is the best balanced
candidate, but narrowly misses the background and foreground-only gates.

## Frozen comparison

| Method | Clean | Leak down | BG mean | BG worst | FG mean | FG only | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCAL V2 | 80.08% | 0.266 | 72.04% | 48.92% | 31.83% | 71.67% | reference |
| FCAL V2+ | 79.67% | 0.244 | **73.16%** | 52.67% | 31.74% | 73.25% | reference |
| FCAL V2.3 | 80.08% | 0.185 | 71.75% | 46.25% | **32.57%** | 73.25% | reference |
| FCAL V2.4 | 80.25% | **0.145** | 71.62% | 40.50% | 31.76% | **73.67%** | reference |
| CAMAL | 78.92% | 0.316 | 73.47% | **60.33%** | **33.18%** | 73.50% | stretch |
| V2.5 Persistent Growth | 79.25% | 0.351 | 72.56% | 56.83% | 31.92% | 70.83% | 1/6 |
| V2.6 Native Dominance | 79.50% | 0.269 | 69.65% | 24.58% | 31.33% | 72.33% | 0/6 |
| V2.7 Foreground Pool | 80.00% | 0.256 | 73.06% | 52.42% | 32.04% | 73.42% | 1/6 |
| **V2.8 Region Contrast** | **80.83%** | **0.230** | 72.90% | 49.75% | 32.03% | 71.92% | **2/6** |
| V2.9 Balanced | 79.08% | 0.311 | 68.94% | 30.17% | 31.29% | 71.83% | 0/6 |

The six gates are clean accuracy, strict leakage, background mean, background
worst, foreground mean, and foreground-only accuracy. Background and
foreground means each average the 13 corresponding stress variants.

## Compute

| Method | Post-branch wall time | Train throughput | Peak VRAM | Compute gate |
|---|---:|---:|---:|---:|
| FCAL V2 | 325.17 s | 796.2 samples/s | 1.7471 GiB | pass |
| CAMAL | 373.86 s | 667.6 samples/s | 1.7452 GiB | limit |
| V2.5 | 359.13 s | 703.4 samples/s | 1.7471 GiB | pass |
| V2.6 | 367.56 s | 687.9 samples/s | 1.7471 GiB | pass |
| V2.7 | 355.34 s | 718.9 samples/s | 1.7471 GiB | pass |
| **V2.8** | **349.11 s** | **728.3 samples/s** | **1.7471 GiB** | **pass** |
| V2.9 | 395.42 s | 626.2 samples/s | 1.7471 GiB | fail |

All variants retain the exact V2 parameter count and inference cost. V2.5 to
V2.8 are cheaper than CAMAL under the frozen wall-time gate. V2.9's combined
native connectivity and two regional pools exceed that limit.

## What the iteration teaches us

1. Persistent CAM growth is not enough. V2.5 raises foreground-corruption
   sensitivity and worst-case background accuracy, but simultaneously worsens
   clean accuracy, leakage, and foreground-only sufficiency.
2. A mean CAM dominance hinge is easy to satisfy without decision robustness.
   V2.6's random-RGB background collapse demonstrates that spatial CAM averages
   are not a certificate of invariance.
3. Foreground-pooled classification is the strongest cheap direction. V2.7 is
   close to V2+ on background mean/worst and reaches 73.42% foreground-only,
   while preserving clean accuracy and leakage better than V2+.
4. Background-pooled target suppression improves clean accuracy and leakage
   but hurts foreground-only and extreme-background behavior. Feature pooling
   after a clean forward does not constrain how the backbone responds when
   pixels change.
5. Combining losses is not additive. V2.9 inherits incompatible gradients and
   becomes both worse and slower.

The next loss-only candidate should therefore build on V2.7, not V2.9: use a
native per-cell target-versus-rival classification margin on foreground cells
and high class entropy on background cells. This directly supervises which
class each existing spatial feature supports, remains one original-input
forward, and costs only a `[batch, classes, 7, 7]` projection through the
existing classifier weights. It should be screened alone before combination.

## Verification and status

- Input policy: one untouched image tensor and exactly one backbone forward.
- No image masking, erasing, replacement, recomposition, or counterfactual.
- No new learned parameter and no inference change.
- Full test suite: 169 passed.
- Scoped Ruff checks: passed.
- Promotion status: all five failed; no three-seed replication and no test use.
