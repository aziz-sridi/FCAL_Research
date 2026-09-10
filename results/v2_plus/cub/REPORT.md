# FCAL V2+ foreground-competitor margin

## Outcome

V2+ is a valid loss-only V2 successor and passed all six frozen three-seed
validation gates. On the official test set it produces a real but limited
trade-off: strict leakage improves consistently and worst-background accuracy
improves at the three-seed mean, while clean and foreground-only accuracy
regress. V2+ is therefore retained as a Pareto candidate and is not promoted
as a universal replacement for V2.

## Exact change from V2

V2 itself is unchanged. V2+ adds one auxiliary term to V2 containment and
distributed foreground support:

```text
rivals = top-2 non-target classes from stopgrad(logits)
D_c = RMSNormalize(S_target) - RMSNormalize(S_c)
q_c = mean_foreground(tanh(D_c / 1.0))
L_rival = mean_c 0.25 * softplus((0.25 - q_c) / 0.25)

L_V2+ = L_V2 + ramp * 0.025 * L_rival
```

The target and rival maps come from the feature tensor already produced by the
original-image forward pass. Training uses one untouched input, one forward,
and one ordinary backward. There is no recomposition, erasing, corruption,
counterfactual view, extra model, second forward, or inference change.

## Three-seed validation

The frozen candidate passed all six mean gates before test access:

| Metric | V2 | V2+ | Paired change |
|---|---:|---:|---:|
| Clean | 79.47% | **79.64%** | +0.17 pp |
| Strict leakage | 0.269 | **0.252** | -0.017 |
| BG mean | 72.11% | **72.86%** | +0.75 pp |
| BG worst | 51.89% | **53.28%** | +1.39 pp |
| FG mean | 31.58% | **31.98%** | +0.39 pp |
| Foreground only | 71.81% | **72.72%** | +0.92 pp |

## Three-seed test result

Values are mean ± sample SD over seeds 7, 42, and 123. Paired changes are
V2+ minus V2; lower leakage is favorable.

| Metric | V2 | V2+ | Paired change |
|---|---:|---:|---:|
| Clean | **78.49 ± 0.16%** | 78.16 ± 0.25% | -0.33 ± 0.28 pp |
| Strict leakage | 0.281 ± 0.003 | **0.262 ± 0.010** | -0.019 ± 0.012 |
| BG mean | 72.40 ± 0.38% | **72.45 ± 0.19%** | +0.04 ± 0.57 pp |
| BG worst | 51.67 ± 4.00% | **53.55 ± 0.16%** | +1.88 ± 4.13 pp |
| FG mean | 31.76 ± 0.14% | **31.90 ± 0.10%** | +0.13 ± 0.21 pp |
| Foreground only | **72.51 ± 0.87%** | 72.11 ± 0.34% | -0.39 ± 0.70 pp |
| Background only | 3.64 ± 0.73% | **3.60 ± 0.24%** | -0.03 ± 0.66 pp |

Leakage improves on all three seeds. Clean accuracy regresses on all three,
although the seed-123 difference is only 0.07 points. Background-worst
accuracy improves on seeds 42 and 123 but regresses on seed 7; its mean gain is
therefore uncertain with only three seeds. The smaller V2+ standard deviation
on BG worst is descriptive and not evidence of reduced population variance.

## Efficiency

The seed-42 post-branch run took 377.27 seconds including validation and used
1.747 GiB peak allocated CUDA memory. Steady-state training was approximately
700--730 samples/s. V2+ preserves V2's one-forward training structure and
unchanged inference path.

## Decision

The foreground competitor margin is mechanistically useful for suppressing
leakage and stabilizing the worst background condition, but the current form
slightly over-constrains clean fine-grained classification. Any next revision
must be a new, predeclared loss mechanism rather than coefficient tuning on
these results. The test set must not be used to select that revision.

Machine-readable validation and test rows are in
`validation_three_seed_metrics.csv` and `test_three_seed_metrics.csv`.
