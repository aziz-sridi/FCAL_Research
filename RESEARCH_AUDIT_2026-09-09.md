# FCAL research audit — 2026-09-09

## Bottom line

Under the stated constraints—no method-specific image manipulation, one
backbone forward, low training cost, and unchanged inference—**FCAL V2 is the
most promising supported method in this workspace**.

V2 is not the numerical winner on every axis. It is the best-supported Pareto
anchor: three-seed CUB ResNet-50 gains, six-architecture matrices, positive
CUB and Oxford-IIIT Pet behavior, low cost, and no inference overhead. The
strongest robustness result, V3.1, is disqualified by the input policy because
it constructs and forwards counterfactual images. The strongest pure
original-input validation candidate, V4.2, is disqualified by the cost target
because its second-order sensitivity term is roughly 4.38x V2 wall time and
1.99x peak memory.

The newest attempted repair, mean-preserving V2+, also fails: it reaches only
79.00% clean validation accuracy versus 80.08% for V2 and fails its frozen
79.83% gate. It remains a negative result; no stress or test views were opened.

## Ranked decision table

| Rank | Method | Evidence | Constraint fit | Decision |
|---:|---|---|---|---|
| 1 | FCAL V2 | Three CUB ResNet-50 seeds; six architectures; CUB, Pet, ISIC | Full | Keep as the publication and engineering anchor |
| 2 | V2+ | Three CUB ResNet-50 seeds; one ViT validation run | Full; seed-42 time is ~1.16x V2 | Keep as a mechanistic ablation, not a replacement |
| 3 | V4.2 | One CUB ResNet-50 validation run | Original input, but far too expensive | Preserve the dominance lesson; stop the implementation |
| 4 | Universal FCAL | Full Pet matrix; small ISIC pilot | Input policy fits; controller is complex | Experimental only |
| — | V3.1 | One strong CUB ResNet-50 test run | Fails input and one-forward policies | Separate counterfactual-augmentation research line |
| — | V5 certificate | One failed validation run | Compute fits | Stop |
| — | V2.3–V2.9, relative, gated, refinements, feature consistency | Validation screens and two-seed audits | Mostly fit | Negative mechanistic results; do not revive by tuning |

## Why V2 wins the constrained comparison

The replicated CUB ResNet-50 result is the strongest trustworthy evidence in
the workspace:

| Test metric | CE | V2 | Paired V2–CE change |
|---|---:|---:|---:|
| Clean accuracy | 77.44 ± 0.13% | 78.49 ± 0.16% | +1.05 ± 0.19 pp |
| Background mean | 68.75 ± 1.70% | 72.40 ± 0.38% | +3.65 ± 1.37 pp |
| Background worst | 38.90 ± 9.81% | 51.67 ± 4.00% | +12.77 ± 6.63 pp |
| Foreground only | 68.63 ± 0.34% | 72.51 ± 0.87% | +3.87 ± 0.57 pp |
| Strict leakage | 0.571 ± 0.003 | 0.281 ± 0.003 | 50.8% lower at the means |

It also retains the simplest deployable recipe:

```text
L = CE + ramp * (lambda * containment + 0.05 * signed_support)
```

No parameters, masks, or additional operations remain at inference.

## V2’s flaws

1. **It is not universally positive.** The six-architecture ISIC mean loses
   1.67 clean points and one point on the worst background view. The ISIC test
   has only 100 images, but it is still a warning against a universal claim.
2. **Most architecture and dataset cells have one training seed.** Averaging
   architectures is not a substitute for seed uncertainty.
3. **Spatial leakage is not causal invariance.** V2.3, V2.4, V2.7-prime, and
   V5 all show that a better-looking or better-contained CAM can coexist with
   unchanged or worse intervention accuracy.
4. **The target-only loss does not explicitly control rival-class evidence.**
   V2+ addresses this, but its current form over-constrains fine-grained class
   discrimination.
5. **CAM comparability is architecture-dependent.** The exact GAP-CAM story is
   strongest for CNNs and average-pooled ViTs; other ViT pooling schemes do not
   have the same score decomposition.
6. **Masks are a real training cost.** Runtime is cheap, but dense annotation
   is not. Mask availability and mask corruption have not been stress-tested.
7. **The robustness suite is controlled and partly synthetic.** It is useful
   mechanistic evidence, not proof of natural distribution-shift robustness.
8. **Repeated CUB validation iteration creates meta-overfitting risk.** Test
   sealing prevents direct test leakage, but many formula decisions have now
   been made on the same 1,200-image validation set. New formula selection
   should move to a fresh development dataset or an outer nested split.

## What the failed formulas teach

- **V2+ competitor margin:** consistently reduces leakage, but clean accuracy
  falls on all three test seeds and ViT robustness regresses. Foreground class
  separation and background invariance are not the same objective.
- **Gated V2+:** its detached gate decays almost to zero as confidence rises,
  removing the useful late-training pressure.
- **Relative regional pooling / V2.7:** pooled clean-image features are
  observational. They do not tell the backbone how it will respond when the
  underlying background pixels change; transfer also fails on EfficientNet.
- **V2.3/V2.4 coverage and growth:** broader or more connected CAMs improve
  localization measures but not extreme background behavior, and they cost
  more.
- **V4:** input-gradient sensitivity provides some robustness pressure without
  changing pixels, but second-order differentiation breaks the cheap target.
- **V5:** a normalized regional certificate can be satisfied by the wrong
  geometry. Its 0.367 IoU accompanies 17.33% worst-background accuracy.
- **Feature consistency:** the chosen perturbation contributes only about
  2.22e-6 to the ramped loss, then vanishes further; its robustness effect is
  null and changes sign across seeds.
- **Mean-preserving V2+ repair:** zero-mean activation gradients do not imply
  clean parameter updates. Shared backbone changes can still damage the
  discriminative texture/part features used by nearby classes.

## The defensible path toward state of the art

“State of the art” must name the target. The current system is not close to
the best published raw CUB accuracy: specialized fine-grained systems report
above 93%, but they use different backbones, training recipes, and often
different computation. FCAL can instead target a more original and defensible
claim: **best accuracy–background-robustness–explanation–efficiency trade-off
among single-forward, unmodified-input, mask-supervised training losses.**

The shortest credible program is:

1. **Stop formula search on the current CUB validation split.** Freeze V2.
2. **Create a fresh outer development protocol.** Use Oxford-IIIT Pet or a new
   CUB outer split for any new loss selection; never use the existing test
   results to set coefficients.
3. **Modernize the backbone, not the deployed objective.** Test the frozen V2
   loss on a strong but reasonably priced ConvNeXt/MaxViT/Swin baseline, with
   an identically trained CE and CAMAL comparator. This separates loss quality
   from the old ResNet-50 accuracy ceiling.
4. **Run at least five seeds for the primary backbone.** Three is the minimum;
   five is better for the high-variance worst-background metric.
5. **Add one natural-shift benchmark.** Keep the controlled 30-view suite, but
   add a real background/context shift. Synthetic robustness alone cannot
   support an SOTA robustness claim.
6. **Measure supervision cost.** Train with 10%, 25%, 50%, and 100% masks and
   with eroded/dilated/noisy masks. A cheap loss with expensive annotations is
   only partially cheap.
7. **Use a predeclared multi-objective primary endpoint.** For example, require
   non-inferior clean accuracy and report the geometric mean of clean,
   foreground-only, and normalized background-worst performance, with leakage
   and training time as constrained secondary endpoints. Do not select the
   metric after seeing results.
8. **Keep direct contemporaneous baselines.** CAMAL is the main efficient
   competitor; GAIN-ext, RRR, and Faster-ISNet expose different points on the
   localization/faithfulness/cost frontier.

## Recommendation

Do not rename another unreplicated formula as the new FCAL. Keep **FCAL V2** as
the method, treat **V2+** as the strongest informative ablation, archive the
new mean-preserving repair as a failed hypothesis, and spend the next compute
budget on fresh-protocol replication, natural shift, partial-mask robustness,
and a modern backbone. That route is more likely to produce a defensible SOTA
trade-off claim than another round of CUB seed-42 loss tuning.
