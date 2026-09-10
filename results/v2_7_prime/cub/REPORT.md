# FCAL V2.7-prime two-seed background-suppression audit

## Decision

**Stop. Do not run seed 123 and do not tune or ablate the components.**

V2.7-prime makes the 13-variant *mean* background delta nearly identical at
the two seeds (+0.30 percentage points at seed 42 and +0.28 at seed 7 versus
V2), but neither paired bootstrap interval excludes zero. The worst-background
effect changes sign (-1.33 points at seed 42, +0.83 at seed 7), and no accuracy
comparison survives Holm correction over the complete 60-test family. The
predeclared promotion condition—a real, seed-consistent, Holm-surviving
background gain at both seeds—is therefore not met.

The negative promotion decision coexists with one clear mechanistic result:
strict CAM leakage falls substantially at both seeds. This is evidence that the
combined objective changes spatial evidence placement, but it is not evidence
of improved background-intervention robustness.

## Frozen implementation

V2.7-prime preserves the complete V2 containment and signed-support objective.
For the final feature map, the buffer radius is V2's existing resolution-aware
half-cell radius,

```text
r_e = min(24, ceil(0.5 * max(H / h, W / w))) = 16 pixels for 224x224 / 7x7.
```

The two disjoint pooling masks are

```text
M_fg = Erode(M, r_e)
M_bg = 1 - Dilate(M, r_e)
```

so the boundary band is in neither pool. This resolves the ambiguous literal
phrase "dilate the complement": dilating `1-M` would overlap the object rather
than leave an excluded buffer.

Both regional feature vectors reuse the existing forward features and shared
classifier head:

```text
a_region[c] = sum_ij M_region[ij] A[cij] / (sum_ij M_region[ij] + epsilon)
g_region[k] = W[k]^T a_region + b[k]
d_region = g_region[y] - logsumexp_{k != y}(g_region[k])
L_fg = mean softplus(-d_fg)
L_bg = mean softplus(d_bg)                  # m_bg = 0
gamma = clip(0.15 * stopgrad(CE) /
             (stopgrad(L_fg + L_bg) + epsilon), 0.05, 0.30)
```

The complete training loss is

```text
L = CE + rho_t [lambda L_out + 0.05 L_sup + gamma (L_fg + L_bg)].
```

There are no new parameters, no masked-image forward, no second backbone
forward, and no inference change. Empty buffered pools are excluded from their
corresponding regional mean instead of being classified from an all-zero
vector. The method uses the original five-epoch CE warm-up, five-epoch ramp,
optimizer, schedule, data, split, and strict `validation_accuracy > best`
checkpoint rule.

The frozen configs are `configs/cub/resnet50_fcal_v2_7_prime_seed42.yaml` and
`configs/cub/resnet50_fcal_v2_7_prime_seed7.yaml`. Both runs resumed the exact
same-seed CE epoch-5 branches used by the earlier comparisons. The seed-42
archive relocation passed the repository's loader-setting and split-hash
guard before training.

## Side-by-side validation results

All values below come from the selected checkpoints on the same 1,200-image
validation set. Every stochastic background transformation uses common
`variant_seed=271828`, which is distinct from training seeds 42 and 7. BG mean
and worst are over all 13 background variants.

| Seed | Method | Clean | FG only | BG mean | BG worst | Strict leakage |
|---:|---|---:|---:|---:|---:|---:|
| 42 | V2 | 80.08% | 71.67% | 72.12% | 49.50% | 0.26621 |
| 42 | V2.7 | 80.00% | **73.42%** | **73.18%** | **52.75%** | 0.25562 |
| 42 | **V2.7-prime** | **80.33%** | 71.83% | 72.42% | 48.17% | **0.22986** |
| 7 | V2 | **79.83%** | 70.25% | 72.45% | 56.33% | 0.27196 |
| 7 | V2.7 | 78.75% | 71.17% | 72.15% | 56.67% | 0.26787 |
| 7 | **V2.7-prime** | 78.67% | **71.83%** | **72.73%** | **57.17%** | **0.21460** |

### V2.7-prime minus V2, paired uncertainty

| Seed | Metric | Difference | Paired-bootstrap 95% CI |
|---:|---|---:|---:|
| 42 | Clean | +0.25 pp | [-1.50, +2.00] pp |
| 42 | FG only | +0.17 pp | [-1.92, +2.25] pp |
| 42 | BG mean, 13 variants | +0.30 pp | [-0.93, +1.51] pp |
| 42 | BG worst | -1.33 pp | [-3.75, +1.08] pp |
| 42 | Deterministic BG mean, 5 variants | +0.80 pp | [-0.65, +2.25] pp |
| 42 | Strict leakage | -0.03635 | [-0.04385, -0.02869] |
| 7 | Clean | -1.17 pp | [-2.92, +0.67] pp |
| 7 | FG only | +1.58 pp | [-0.42, +3.67] pp |
| 7 | BG mean, 13 variants | +0.28 pp | [-0.85, +1.40] pp |
| 7 | BG worst | +0.83 pp | [-1.67, +3.33] pp |
| 7 | Deterministic BG mean, 5 variants | -0.10 pp | [-1.48, +1.27] pp |
| 7 | Strict leakage | -0.05736 | [-0.06383, -0.05103] |

The leakage intervals are paired percentile-bootstrap intervals over images;
lower is better. Exact McNemar tests apply to paired correctness, not continuous
leakage.

## Seed consistency and the decoupled-RNG question

The original V2.7 mean background effect versus V2 is +1.06 points at seed 42
and -0.29 at seed 7. V2.7-prime changes those point estimates to +0.30 and
+0.28. Thus the *mean delta* does move toward agreement across seeds, but the
agreement is achieved by shrinking the seed-42 effect while only slightly
raising seed 7; both uncertainty intervals contain zero. The deterministic
five-variant subset is not directionally consistent (+0.80 and -0.10 points),
and the worst-variant effect reverses sign.

For original V2.7, the seed-42 random-RGB gain survives decoupling descriptively
but shrinks from the previously reported +3.50 to +3.25 points. Its complete
13-variant mean gain is +1.06 points with 95% CI [-0.06, +2.18]. For
V2.7-prime, seed-42 random-RGB instead changes by -1.33 points relative to V2.
Therefore the original seed-42 point result does not transfer to the combined
fix.

## Multiple testing

The exact McNemar family contains every paired accuracy comparison reported by
the audit:

```text
2 training seeds x 2 candidates (V2.7 and V2.7-prime versus V2)
x (clean + foreground-only + 13 background variants) = 60 tests.
```

Holm correction was applied once across all 60, rather than separately to a
favorable subset. No comparison survives at family-wise alpha 0.05. The
smallest raw p-value is original V2.7's seed-42 random-RGB gain
(`p=0.01077`, Holm `p=0.64647`). The next-smallest V2.7-prime result is a
seed-42 Gaussian-noise-severity-3 *regression* (`p=0.02320`, Holm `p=1.0`).

## Checkpoint stability and cost

| Seed | Selected epoch | Best train-loop validation | Last-5 validation mean | Post-branch wall time |
|---:|---:|---:|---:|---:|
| 42 | 38 | 80.42% | 80.12% | 538.94 s |
| 7 | 46 | 78.67% | 78.32% | 431.32 s |

The selected-checkpoint clean values in the main table differ slightly from
the train-loop best values because the audit reloads the checkpoint through the
mask-bearing multi-variant validation loader; this same behavior is visible in
the historical V2.7 artifacts. The last-five means confirm that neither result
is solely a single final-epoch fluctuation.

Across active epochs, mean gamma was 0.05171 at seed 42 and 0.05145 at seed 7,
close to its 0.05 floor. The mean ramp-weighted regional contributions were
0.01403 and 0.01447. The two post-branch runs total 970.25 seconds versus
894.60 seconds for the two historical V2.7 runs (+8.5% across different
sessions); peak allocated memory was 1.723 GiB for both new runs. No non-finite
gradient was logged.

## Verification and artifacts

- Full test suite after implementation: **209 passed**.
- Focused Ruff checks: passed.
- One untouched input and one backbone forward: covered by unit test.
- Disjoint eroded-foreground/distant-background masks: covered by unit test.
- Shared classifier: the existing `regional_logits` methods are reused.
- Strict checkpoint rule: existing `>` comparison retained.
- Seed 123: not run.
- Component ablations: not run.

Machine-readable artifacts:

- `summary_metrics.csv`: absolute side-by-side metrics.
- `aggregate_bootstrap.csv`: background mean/worst paired intervals.
- `paired_accuracy_all.csv`: all 60 paired accuracy tests and global Holm values.
- `evaluation_seed271828/`: per-image NPZ predictions and per-variant JSON.
- `paired/`: existing comparison-harness outputs for each seed/candidate pair.

## Interpretation

The symmetric regional objective is effective at relocating the positive CAM
mass measured by strict leakage, but that relocation does not become a reliable
accuracy gain when background pixels are intervened on. Background-pooled
features from the clean forward remain an observational feature constraint,
not a causal invariance certificate. Under the frozen protocol, V2 remains the
supported anchor and V2.7-prime is retained as a rigorous negative result.
