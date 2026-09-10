# V2 background-feature consistency: two-seed audit

## Decision

**Stop. Do not run seed 123 and do not tune the noise scale or coefficient post
hoc.**

The proposed cheap feature-space perturbation does not produce a real,
seed-consistent background gain over V2. Across all 13 background variants,
the paired mean change is +0.10 percentage points at seed 42 (95% CI
[-1.02, +1.18]) and -0.39 points at seed 7 (95% CI [-1.54, +0.76]). The
worst-background change is -1.75 points at seed 42 and 0.00 at seed 7, with
both intervals crossing zero. No paired accuracy result survives Holm
correction.

The negative result is mechanistically informative: with the frozen choices
sigma=0.10 and beta=0.02, the active-epoch mean KL is only about 1.38e-4 and
its mean ramp-weighted loss contribution is about 2.22e-6. In the last five
epochs the contribution falls to about 1.7e-7. The intervention is therefore
too weak to create a measurable robustness effect under the predeclared run,
but increasing it after seeing the outcome would be post-hoc tuning.

## Mathematical definition and implementation

The formula in the proposal needs one clarification. If `g_fg = W a_fg + b`
depends only on the foreground-pooled vector, perturbing `a_bg` cannot change
`g_fg`; the two distributions are identical and the KL is exactly zero. The
implemented version is the nearest non-degenerate interpretation: decompose
the ordinary global-average-pooled vector into foreground and background
contributions, perturb only the background contribution, and compare the
ordinary clean and perturbed predictions.

For final convolutional features `A_n,c,i,j` and a foreground mask `M_n,i,j`
resampled to the feature grid,

```text
q_fg = (1 / hw) sum_ij M_ij                 q_bg = 1 - q_fg

a_fg,c = sum_ij M_ij A_cij / (sum_ij M_ij + epsilon)
a_bg,c = sum_ij (1-M_ij) A_cij / (sum_ij (1-M_ij) + epsilon)

a_c = q_fg a_fg,c + q_bg a_bg,c            # exact global-average decomposition
epsilon_c ~ Normal(0, 1)
a_pert,c = a_c + q_bg sigma epsilon_c

g = W a + b
g_pert = W a_pert + b                       # same classifier; no new parameters
p = softmax(g)                              p_pert = softmax(g_pert)

L_cons = (1/N) sum_n KL(p_n || p_pert,n)
       = (1/N) sum_n sum_k p_nk log(p_nk / p_pert,nk)
```

One Gaussian sample per image and optimizer step is a Monte Carlo estimate of
the noise-averaged consistency objective. For small sigma, the expected KL is
locally proportional to a Fisher-weighted squared prediction Jacobian,

```text
E_epsilon[KL] approximately
    (sigma^2 q_bg^2 / 2) trace(J_a^T F(p) J_a),
```

so this is consistency regularization / feature-space augmentation: it asks
the classifier to be locally smooth specifically along perturbations assigned
to the pooled background contribution. It is random smoothing, not adversarial
training, and it does not guarantee invariance to structured pixel-space
background shifts.

The complete frozen objective is V2 plus this one term:

```text
L = L_CE + rho_t [lambda L_out + 0.05 L_sup + beta L_cons]
sigma = 0.10
beta = 0.02
```

`rho_t`, adaptive `lambda`, warm-up/ramp, containment/support terms, optimizer,
schedule, data, and splits are unchanged from V2. The implementation reuses
the cached feature tensor and existing classifier `W,b`; it performs one image
forward, no pixel manipulation, no second backbone pass, and only one extra
classifier-head application. It deliberately does not include V2.7 foreground
pooling, V2.7-prime erosion/background suppression, or adaptive gamma.

## Side-by-side validation results

All selected checkpoints were evaluated on the same 1,200 validation images.
Every stochastic transformation uses `variant_seed=271828`, decoupled from
training seeds 42 and 7. BG mean/worst cover all 13 background variants. The
deterministic subset comprises blur severities 1-3, grayscale, and solid mean.

| Seed | Method | Clean | FG only | BG mean | BG worst | Deterministic BG mean | Strict leakage |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | V2 | **80.08%** | 71.67% | 72.12% | 49.50% | 75.55% | 0.26621 |
| 42 | V2.7 | 80.00% | **73.42%** | **73.18%** | **52.75%** | **76.47%** | 0.25562 |
| 42 | V2.7-prime | 80.33% | 71.83% | 72.42% | 48.17% | 76.35% | **0.22986** |
| 42 | **V2 + feature consistency** | 78.83% | 72.08% | 72.21% | 47.75% | 75.97% | 0.27220 |
| 7 | V2 | **79.83%** | 70.25% | 72.45% | 56.33% | **75.40%** | 0.27196 |
| 7 | V2.7 | 78.75% | 71.17% | 72.15% | 56.67% | 74.92% | 0.26787 |
| 7 | V2.7-prime | 78.67% | **71.83%** | **72.73%** | **57.17%** | 75.30% | **0.21460** |
| 7 | **V2 + feature consistency** | 79.50% | 71.67% | 72.06% | 56.33% | 75.38% | 0.26994 |

### Feature consistency minus V2: paired uncertainty

| Seed | Metric | Difference | Paired-bootstrap 95% CI |
|---:|---|---:|---:|
| 42 | Clean | -1.25 pp | [-3.08, +0.58] pp |
| 42 | FG only | +0.42 pp | [-1.58, +2.33] pp |
| 42 | BG mean, 13 variants | +0.10 pp | [-1.02, +1.18] pp |
| 42 | BG worst | -1.75 pp | [-4.08, +0.67] pp |
| 42 | Deterministic BG mean, 5 variants | +0.42 pp | [-0.98, +1.80] pp |
| 42 | Strict leakage | +0.00598 | [-0.00078, +0.01289] |
| 7 | Clean | -0.33 pp | [-2.17, +1.58] pp |
| 7 | FG only | +1.42 pp | [-0.58, +3.50] pp |
| 7 | BG mean, 13 variants | -0.39 pp | [-1.54, +0.76] pp |
| 7 | BG worst | +0.00 pp | [-2.50, +2.42] pp |
| 7 | Deterministic BG mean, 5 variants | -0.02 pp | [-1.45, +1.43] pp |
| 7 | Strict leakage | -0.00203 | [-0.00966, +0.00560] |

Leakage intervals are paired percentile-bootstrap intervals over images; lower
is better. Exact McNemar tests are used for paired correctness, not continuous
leakage.

## Seed consistency and multiplicity

The 13-variant mean effect changes sign across seeds (+0.10 and -0.39 points),
so the primary seed-consistency requirement fails. The five deterministic
variants are also null (+0.42 and -0.02 points), and the worst-variant result
does not improve at either seed.

For the new candidate alone, the exact McNemar family contains
`2 seeds x 15 variants = 30 tests`. Its smallest raw p-value is a seed-7
Gaussian-noise-severity-1 **regression** of -2.67 points (`p=0.01167`, Holm
`p=0.35019`); no new comparison survives.

Because this report also displays V2.7 and V2.7-prime, the final correction was
conservatively recomputed over the cumulative reported family:

```text
2 seeds x 3 candidates x
(clean + foreground-only + 13 background variants) = 90 tests.
```

No comparison survives family-wise alpha 0.05. The smallest cumulative raw
p-value remains V2.7 seed-42 random RGB (`p=0.01077`; 90-test Holm
`p=0.96971`). The feature-consistency regression above has cumulative Holm
`p=1.0`.

## Checkpoint stability and cost

Strict-improvement (`validation_accuracy > best`) selection was retained.

| Seed | Selected epoch | Best train-loop validation | Last-5 validation mean | Post-branch wall time |
|---:|---:|---:|---:|---:|
| 42 | 46 | 78.83% | 78.48% | 382.27 s |
| 7 | 49 | 79.50% | 79.12% | 356.20 s |

The last-five means are close to the selected checkpoints, so checkpoint noise
does not explain the robustness null. Peak allocated CUDA memory was 1.75 GiB
for both runs. No non-finite gradient was logged.

## Verification and artifacts

- Focused tests for sigma-zero reconstruction, one-backbone-forward behavior,
  metric logging, and config freezing pass.
- The full repository test suite passes.
- Focused Ruff checks pass.
- Both runs resumed their exact same-seed epoch-5 CE parent; seed 42's archived
  path passed split-hash and loader-setting relocation guards.
- Seed 123 was not run; sigma and beta were not retuned.

Machine-readable artifacts:

- `summary_metrics.csv`: absolute V2/V2.7/V2.7-prime/candidate results.
- `paired_summary.csv`: paired primary metric differences and intervals.
- `aggregate_bootstrap.csv`: 13-variant, deterministic-five, and worst-case
  paired intervals.
- `paired_accuracy_new.csv`: all 30 new exact McNemar tests and new-family Holm
  values.
- `comparison_seed271828/`: unchanged comparison-harness output.
- `evaluation_seed271828/`: per-image predictions and per-variant summaries.

## Interpretation

This mechanism is genuinely different from V2.7 and V2.7-prime: it targets
local prediction smoothness under stochastic feature perturbations rather than
foreground sufficiency or direct background-logit suppression. Under the
frozen cheap setting, however, the perturbation barely changes predictions and
does not improve intervention robustness. V2 remains the supported anchor;
this run is retained as a rigorous negative result.
