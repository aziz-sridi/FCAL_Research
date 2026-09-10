# FCAL selection and a concrete low-cost repair — 2026-09-09

**2026-09-10 follow-up:** the user-requested one-seed quick screen is now
complete. The class-mean repair improves clean, mean-background and
foreground-only point estimates, but fails worst-background, raw-CAM leakage
and historical timing gates. See the [full comparison](../results/decision_evidence_screen/cub_seed42/REPORT.md).
The untrained-candidate statements below describe the original September 9 audit.

**Keep FCAL V2 as the supported method. Test class-mean evidence as a narrowly
defined experimental repair; it has not yet earned a promotion.** This review
uses the completed experiments, including failed replications and later audits,
rather than treating old research proposals as untested ideas.

Constraint interpretation: no FCAL-specific edits to input images, no extra
training views or backbone forwards, low incremental training cost, unchanged
inference. The existing shared preprocessing/augmentation and training masks
are retained. If “no manipulation” also excludes those, the existing results
do not directly establish performance under that stricter protocol.

## Selection from completed results

The paired differences below were recomputed from the existing CSV files.
Uncertainty is sample SD of the three paired training-seed differences, not a
confidence interval. BG mean/worst refer to the recorded 13 background views.

| CUB ResNet-50 test metric | V2 minus CE | V2+ minus V2 |
|---|---:|---:|
| Clean accuracy | +1.05 ± 0.19 pp | −0.33 ± 0.28 pp |
| BG mean accuracy | +3.65 ± 1.37 pp | +0.04 ± 0.57 pp |
| BG worst accuracy | +12.77 ± 6.63 pp | +1.88 ± 4.13 pp |
| Foreground-only accuracy | +3.87 ± 0.57 pp | −0.39 ± 0.70 pp |
| Strict leakage (lower is better) | −0.2902 ± 0.0056 | −0.0189 ± 0.0122 |

V2 reaches 78.49% mean clean accuracy against CE's 77.44%. V2+ improves
leakage on all three seeds but loses clean accuracy on all three. Its
background-worst gain changes sign by seed. Keep it as a useful trade-off
ablation, not the default successor. See the [three-seed V2+ report](../results/v2_plus/cub/REPORT.md)
and [recomputed paired deltas](../results/decision_evidence_audit/paired_existing_results.csv).

| Formula family | Evidence that controls the decision | Decision under your constraints |
|---|---|---|
| V1 | Much smaller classification/robustness gains than V2; weaker replicated leakage result | Historical baseline |
| **V2** (archived internal name V5) | Three CUB ResNet-50 seeds, six-model matrices, positive CUB/Pet means | **Best supported foundation** |
| V2+ | Three-seed trade-off above | Secondary ablation |
| V2.7 foreground pooling | Seed-42 promise; seed-7 clean −1.08 pp; EfficientNet FG-only 69.92% → 62.58% | Failed replication/transfer |
| V2.8 region contrast | 80.83% clean validation, but fails combined robustness/sufficiency screen | Interesting one-seed result, weaker evidence than V2 |
| Relative pooling; gated V2+ | Random-RGB accuracy −6.83 and −4.92 pp versus V2; time gates fail | Already tested; do not propose again as new |
| V2.7-prime; feature consistency | Two-seed audits show no reliable background gain | Negative under their frozen protocols |
| Mean-preserving V2+ | 79.00% clean validation versus V2 80.08%; failed first quality gate | Already attempted repair |
| V2.3/2.4; V2.5/2.6/2.9; 2.7 refinements | Coverage, connectivity or local margins fail quality and/or compute gates | Preserve mechanistic lessons |
| Balanced V3 | Strong BG-worst test accuracy, 68.24%, but manipulated second training view | Excluded by input/forward constraint |
| Pure V4.2 | Some strong validation metrics, 29.16 minutes versus V2 5.42 | Excluded by cost |
| New V5 certificate | BG-worst 17.33% despite IoU 0.367 | Failed proxy objective |
| Universal FCAL | Pet mean clean +0.42 pp vs CE, near-unchanged leakage; complex gradient controller | Experimental machinery |

Supporting reports: [V2.7 replication](../results/v2_7_replication/cub/README.md),
[cross-CNN screen](../results/v2_7_cross_cnn/cub/README.md),
[V2.5–2.9](../results/v2_next/cub/VALIDATION_REPORT.md),
[relative pooling](../results/v2_relative/cub/README.md),
[gated V2+](../results/v2_plus_gated/cub/README.md),
[V2.7-prime](../results/v2_7_prime/cub/REPORT.md),
[feature consistency](../results/v2_feature_consistency/cub/REPORT.md),
[mean-preserving repair](../results/v2_plus_mean_preserving/cub/VALIDATION_REPORT.md),
[V3](../results/v3/cub/BALANCED_REPORT.md),
[V4/V5 cost](../results/v5/cub/REPORT.md),
[Universal FCAL](../results/universal/README.md).

Two corrections to old planning documents matter. `NEXT_RESEARCH_DIRECTION.md`
recommends relative pooling and gated V2+ before their subsequent failures.
Also, V4.2's +438% overhead means approximately **5.38×** V2 time, not 4.38×.
Internal archived V5 means supported public V2; the newer certificate V5 is a
different formulation.

## V2's important flaws

1. **Its attention target is not uniquely determined by classification.** The
   common-class ambiguity below gives an exact counterexample. This affects
   interpretation of raw-CAM leakage, not the validity of measured predictions.
2. **Contained features are not necessarily background-independent features.**
   Foreground spatial cells can contain background information through large
   receptive fields or attention. Clean-feature pooling does not emulate a
   changed background. The failed pooling and certificate experiments expose
   this gap, but do not prove that every cheap loss must fail.
3. **Positive target containment misses other decision mechanisms.** Negative
   evidence and rival classes can matter. V2+ adds rival pressure, but subtracts
   separately RMS-normalized maps, so its margin is not the classifier's actual
   unnormalized logit margin. Its losses can also compete with fine-grained
   discrimination. The observed trade-off does not establish a unique cause.
4. **Support keeps pushing after support is positive.** In the current code,
   `softplus(margin - mean_fg(tanh(S / stopgrad(RMS(S)))))` has no inactive
   region. Its detached scale yields a surrogate gradient rather than the
   derivative of a fully normalized objective. These are intentional design
   choices, not proven bugs; changing them should be separate ablations.
5. **Transfer and uncertainty remain limited.** V2's six-model ISIC mean loses
   1.67 clean points despite lower leakage. That test has only 100 images.
   CUB/Pet architecture averages generally have one training seed per cell.
   Repeated exploration of the same CUB validation set creates selection bias
   beyond any one experiment's within-run test sealing.
6. **Cheap training is not cheap supervision.** Masks cost annotation effort.
   Partial/noisy-mask experiments and natural context shifts are still needed.

Dataset means: [CUB](../results/cub/method_means_complete.csv),
[Oxford Pet](../results/oxford_pet/method_means_complete.csv),
[ISIC](../results/isic/method_means_complete.csv).

## Repair implemented: subtract the common-class evidence

For cached features A at spatial cell u and a linear GAP classifier,

```text
z_k = mean_u(w_k^T A_u) + b_k
S_y(u) = w_y^T A_u
```

Add any feature-weight vector v to every classifier row: `w'_k = w_k + v`.
Every logit receives the same image-dependent shift, so **all softmax
probabilities and class predictions stay identical for every image**. But
`S'_y(u) = S_y(u) + v^T A_u`, so V2's containment and support can change.
Auxiliary supervision can therefore reward a component that contributes
nothing to class discrimination.

The repair replaces the evidence map, while retaining the existing V2 loss:

```text
w_bar = mean_k(w_k)                         # differentiable
D_y(u) = (w_y - w_bar)^T A_u

L = CE + ramp * [lambda * containment(D_y, mask)
                + 0.05 * signed_support(D_y, mask)]
```

`D_y` is invariant to the common weight shift. It is proportional to the
target-versus-mean-rival evidence. Moreover,

```text
mean_u(D_y(u)) + b_y - mean_k(b_k) = z_y - mean_k(z_k).
```

Center **before** ReLU, RMS normalization or any other nonlinearity. The class
mean remains in the gradient graph; detaching it would restore unwanted
common-direction classifier gradients. This is class-axis centering of the
evidence definition, not the failed spatial-mean gradient projection in V2+.

Cost: one `[classes, channels]` mean, one subtraction per selected classifier
row, then the same spatial projection as V2. No dense per-cell class tensor,
new coefficient, extra backbone, second backward, input/feature perturbation,
learned head or inference change. EfficientNet reuses its existing dropout
sample. The adapters support ResNet-50, DenseNet-121, EfficientNet-B0 and the
average-pooled ViTs. CLS-token patch proxies are intentionally unsupported.

**Verified mechanism:** on an exact two-feature example, common-class weight
shifting lowers ordinary V2 containment from 0.500000 to 0.045455 while the
target probability stays 0.880797. The centered variant stays at 0.500000.
Tests also verify zero class-sum auxiliary weight gradients, gradient flow to
the backbone, score decomposition, the EfficientNet dropout case, and a
single unmodified input/forward through the actual training dispatch.

The saved seed-42 V2 checkpoint was additionally inspected on 32 training
images. Its common component has RMS approximately 0.87 times the raw target
CAM RMS. This is **not** a fraction of causal reliance or explained energy;
the components need not be orthogonal. Raw and centered leakage differ even
though predictions are identical. The exact values, input/mask hashes, image
IDs and checkpoint hash are in [audit.json](../results/decision_evidence_audit/audit.json).
This small training-data probe is a diagnostic, not a held-out evaluation.

The synthetic CUDA benchmark alternates six timing blocks per variant using
ResNet-50, batch 32, 224×224, BF16 and one forward/loss/backward. It records
the added cost and peak allocated memory in `audit.json`. It excludes data
loading, optimizer steps and validation; it cannot establish end-to-end
training cost or cross-architecture overhead.
The final recorded median time ratio is **1.00162×** (about **+0.16%**), with
identical measured peak allocation of **1,747,957,760 bytes**. This is effectively
near the measurement noise floor, not a precisely established population cost.

**Limits of the repair:** it removes an exact common-logit ambiguity. It does
not control the strongest rival, eliminate negative background evidence,
prevent information mixing inside the backbone, guarantee preserved clean
accuracy, or establish a robustness gain. Shared object evidence may also be
useful training supervision; discarding its common-class component could hurt.
This is a falsifiable candidate, not a new SOTA result or a claim of novelty.

## Code, validation and next experiment

Implementation: [decision_evidence.py](../src/fcal_study/decision_evidence.py),
with an opt-in dispatch in [train.py](../src/fcal_study/train.py).
Existing configs retain the exact target-CAM route. To enable the experimental
route on a separately prepared candidate configuration:

```yaml
method: fcal_v2
run_id: resnet50-v2-class-mean-<fresh-split>-<seed>
output_root: outputs/decision-evidence
fcal_v2:
  # Keep all existing V2 fields; add only this field:
  evidence_reference: class_mean
```

Use a distinct run ID/output root and keep these artifacts out of historical
V2 aggregation. The existing evaluation attention convention is unchanged.
If centered CAMs are evaluated, report them as a supplementary common
convention applied to **every comparator**, not as an apparent improvement
obtained by changing only the candidate's scoring rule.

Reproduction:

```powershell
python -m pytest -q
python -m ruff check src/fcal_study/decision_evidence.py src/fcal_study/train.py tests/test_decision_evidence.py src/fcal_study/decision_evidence_audit.py
python -m fcal_study.decision_evidence_audit
```

The full suite passed: **222 tests**. The repair has **not been trained to
convergence**. No new validation/test performance was measured, no old runs
were repeated and no existing checkpoint was changed.

The next experiment should isolate this one change on a fresh development
protocol. Freeze the partition, seeds, V2 coefficients, common corruption RNG,
checkpoint rule and endpoints before training. If the new validation images
were in the old training split, **do not reuse the old CE epoch-5 checkpoint**;
it has already trained on those images. Train a new paired CE warm-up branch.
Compare V2 and the centered variant on that exact branch, starting with
ResNet-50 and then a weak-transfer model such as ViT-Small.

A proposed screening rule, to freeze before new results: at most 0.25 pp clean
loss, at least 1 pp BG-mean gain, no decline in BG-worst or foreground-only,
at most 1.15× paired full training time and 1.10× peak memory. A screen is not
a significance claim. Replicate a passing candidate with fixed coefficients;
use paired seed uncertainty and an image-clustered analysis of the 13 views.
Do not treat views of the same image as independent samples. Failed significance
tests in earlier two-seed audits mean insufficient evidence, not proof that
their true effects are exactly zero.

## What a state-of-the-art claim still needs

The current evidence supports a useful accuracy–robustness–cost trade-off.
It does not establish overall leadership. Recomputed tuned six-architecture
CUB means illustrate why:

| Method | Clean | BG mean | BG worst | FG only | Leakage ↓ |
|---|---:|---:|---:|---:|---:|
| V2 | 79.92% | 73.19% | 52.50% | 72.98% | 0.466 |
| CAMAL | 79.33% | 73.16% | 56.96% | 71.28% | 0.437 |
| RRR | 76.66% | 72.48% | 60.61% | 72.38% | 0.672 |
| GAIN-ext | 76.06% | 68.29% | 45.38% | 68.14% | 0.303 |

V2's 0.03-point BG-mean lead over CAMAL is a descriptive near-tie, not strong
evidence of superiority. Raw CUB accuracy is another target altogether:
[HERBS](https://arxiv.org/abs/2303.06442) reports above 93% on CUB and NABirds
under a different specialized system. That is context, not a matched baseline
or an exhaustive current leaderboard. [CAMAL](https://arxiv.org/abs/2605.08325),
published as a May 2026 preprint, directly studies segmentation-guided CAM
alignment and faithfulness with no extra inference cost. Adding mask
supervision alone cannot support a novelty claim.

For a credible SOTA attempt, specify the exact task and budget; retain strong
matched CE and CAMAL controls; test fixed coefficients across seeds and an
external domain; add a natural context-shift evaluation; account for mask
annotation and tuning costs. If the target is raw classification accuracy,
establish a substantially stronger matched backbone/training baseline too.
The immediate useful advance here is an inexpensive, demonstrably well-defined
decision-evidence target whose predictive value can now be tested honestly.
