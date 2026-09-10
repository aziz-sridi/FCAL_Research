# FCAL V2.7 seed-variance audit

## Conclusion

The seed-7 failure is not explained by a configuration mismatch, wrong branch,
checkpoint selection, degenerate masks, or a grossly mis-scaled foreground
loss. The evidence instead localizes the problem to a **seed-sensitive
generalization response of the foreground-pool objective, especially on
background robustness**.

V2.7 consistently improves foreground-only accuracy on both seeds, which is
the behavior it directly optimizes. Its indirect background benefit is large
on seed 42 but weak or negative on seed 7. V2 remains the supported anchor;
V2.7 should not be promoted or extended to seed 123 under the staged protocol.

## Controlled provenance checks

- The resolved seed-42 and seed-7 V2.7 configurations differ only in `seed`
  and `run_id`.
- Python, PyTorch, CUDA, GPU, and all train/validation/test split hashes match.
- Each V2 and V2.7 run shares the first five history records exactly with its
  same-seed CE parent.
- Clean validation is an ordered deterministic center crop.
- Seed 42 selected V2 at epoch 46 and V2.7 at epoch 50. Seed 7 selected V2 at
  epoch 48 and V2.7 at epoch 47; epoch 49 tied V2.7's maximum and was correctly
  not selected under the strict-improvement rule.
- The seed-7 V2.7 deficit persists throughout the late training window, so it
  is not an unlucky single checkpoint.

The source revision was not recorded and the worktree is uncommitted. Exact
source identity therefore cannot be cryptographically proven, although active
losses, resolved configs, environment, histories, and memory behavior show no
behavioral mismatch.

## Paired prediction results

All statistics below use aligned predictions for the same 1,200 validation
images. Bootstrap intervals condition on the already-trained models; they do
not estimate training-seed uncertainty.

| Metric | Seed 42 V2.7−V2 | 95% paired bootstrap | Exact McNemar p | Seed 7 V2.7−V2 | 95% paired bootstrap | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Clean accuracy | -0.08 pp | [-1.83, +1.67] | 1.000 | -1.08 pp | [-3.08, +0.92] | 0.322 |
| BG random RGB, native draw | +3.50 pp | [+1.08, +5.92] | 0.0065 | -0.67 pp | [-3.08, +1.83] | 0.642 |
| Foreground-only | +1.75 pp | [-0.33, +3.83] | 0.110 | +0.92 pp | [-1.17, +3.00] | 0.425 |
| Strict leakage | -0.01060 | [-0.0178, -0.0034] | — | -0.00409 | [-0.0107, +0.0025] | — |

Only the seed-42 background and leakage results survive Holm correction over
the original eight within-seed tests. With only two training seeds and one
model per method/seed, population-over-seeds significance is not identifiable.

## Background localization

The cheap follow-up evaluated the twelve previously missing seed-7 background
variants without training:

| Summary | Seed 42 | Seed 7 |
|---|---:|---:|
| V2.7−V2, 13-variant BG mean | +1.02 pp | -0.55 pp |
| Variant wins / ties / losses | 12 / 1 / 0 | 2 / 1 / 10 |
| V2.7−V2, five deterministic BG variants | +0.92 pp | -0.48 pp |
| Deterministic wins / losses | 5 / 0 | 1 / 4 |

The deterministic subset is blur severities 1–3, grayscale, and solid mean;
it is unaffected by the evaluation RNG. Therefore the reversal is broader than
one random-RGB draw. Full per-variant values are in
`structured_background.csv`.

Stochastic corruptions originally inherited the checkpoint's training seed.
To remove that confound, the evaluator now supports `--variant-seed`. On the
same seed-42 random-RGB draw:

- seed-42 V2.7−V2: +3.50 pp, 135 paired wins / 93 losses, p=0.0065;
- seed-7 V2.7−V2: +0.33 pp, 116 paired wins / 112 losses, p=0.843;
- conditional interaction: -3.17 pp, paired bootstrap
  [-6.75, +0.42] pp.

The corrected result removes the apparent negative seed-7 random-RGB effect,
but confirms that the large seed-42 gain does not reproduce.

## Mechanism checks

The seed-7 CE branch is weaker at epoch 5 (62.08% versus 65.92%), but this is
not sufficient to explain the result: ordinary V2 recovers to 79.92% versus
80.08% on seed 42, whereas V2.7 reaches only 78.75% on seed 7.

The V2.7 auxiliary term behaves similarly on both seeds:

- post-branch mean raw foreground loss: 0.1150 seed 42, 0.1187 seed 7;
- mean ramp-weighted contribution: 0.00181 versus 0.00188;
- on 128 fixed validation examples at the paired epoch-5 branches, its gradient
  norm is 3.47% versus 3.32% of CE, with mean CE-gradient cosine 0.811 versus
  0.808 and zero conflicting batches;
- both runs converge to approximately 100% training accuracy with no nonfinite
  gradients.

All 47,940 training crops across epochs 6–10 and both seeds were also checked:
there were zero empty and zero all-foreground masks. Although the regional loss
does not explicitly filter degenerate masks, that edge case did not occur here.

## Diagnosis and next decision

Different seeds intentionally change the CE epoch-5 state, batch order, random
crop, and flip path. The same small, initially CE-aligned foreground objective
then reaches different generalization basins. Because the objective directly
enforces foreground sufficiency but only indirectly affects background
features, its foreground benefit transfers while its background benefit does
not.

No coefficient adjustment is justified: the loss magnitude and initial
gradient geometry are already nearly identical, and tuning seed 7 would be a
post-hoc seed-specific search. Do not run seed 123 merely to average away this
failure. Retain FCAL V2, mark standalone foreground-pool classification as a
useful but unstable ablation, and require any future candidate to constrain
background behavior directly while remaining gated and cheap.
