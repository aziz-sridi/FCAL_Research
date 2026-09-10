# FCAL Next Research Direction

## Target claim

The research should be optimized around one precise claim:

> FCAL is a cheap, single-forward loss that uses unmodified inputs, works
> across CNNs and Vision Transformers, improves model robustness, and reduces
> shortcut learning.

This claim rules out input manipulation, additional backbone forwards,
expensive gradient procedures, and objectives that improve mask overlap
without improving prediction behavior.

## Executive recommendation

Continue three connected directions:

1. **Primary:** build a relative pooled-margin successor using the strongest
   lessons from V2.7 and V2.8.
2. **Secondary:** test a gated form of the cheap V2+ target-versus-rival
   margin.
3. **Foundation:** keep FCAL V2 as the main reference and add only lightweight
   architecture-aware loss scaling.

FCAL V2 remains the best-supported method. The relative pooled-margin
successor has the greatest potential to improve it while preserving the
intended claim.

## Why FCAL V2 remains the anchor

FCAL V2 already comes closest to satisfying the complete claim:

- one original, unmodified input;
- one backbone forward;
- no additional learned parameters;
- no inference-time loss or architecture change;
- strong replicated CUB ResNet-50 results;
- positive results on CUB and Oxford-IIIT Pet;
- strong results on CNNs and ViT-Base;
- higher fixed/default clean accuracy than CAMAL and RRR on all six CUB
  architectures;
- much lower training cost than RRR, GAIN-ext, and Faster-ISNet.

Its main limitation is uneven transfer. The same effective loss pressure does
not behave consistently across CNN 7x7 maps, ViT 14x14 patch maps, small and
large models, and datasets with different foreground statistics. ViT-Tiny,
ViT-Small, EfficientNet-B0, and ISIC expose this boundary.

## Primary direction: relative pooled class margin

### Evidence from V2.7 and V2.8

V2.7 foreground pooling produced the best cheap, balanced loss-only candidate:

| Metric | V2.7 validation result |
|---|---:|
| Clean accuracy | 80.00% |
| Background mean | 73.06% |
| Background worst | 52.42% |
| Foreground only | 73.42% |
| Strict leakage | 0.256 |

V2.8 region contrast produced the strongest clean/localization candidate:

| Metric | V2.8 validation result |
|---|---:|
| Clean accuracy | 80.83% |
| Background mean | 72.90% |
| Background worst | 49.75% |
| Foreground only | 71.92% |
| Strict leakage | 0.230 |

V2.7 is stronger on robustness and foreground sufficiency. V2.8 is stronger
on clean accuracy and leakage. The next loss should express their common idea
as one objective rather than adding their existing losses together. V2.9
already demonstrated that naive loss combination creates incompatible
gradients and unnecessary compute.

### Proposed mechanism

Use the feature map already produced by the clean model forward:

```text
foreground_logits = classifier(pool(features inside mask))
background_logits = classifier(pool(features outside mask))

foreground_margin =
    target foreground logit - strongest rival foreground logit

background_margin =
    target background logit - strongest rival background logit

L_relative =
    softplus((delta - (foreground_margin - background_margin)) / temperature)
```

This directly teaches:

> The foreground must support the correct class more strongly than the
> background supports it.

This is closer to shortcut-learning prevention than minimizing outside CAM
mass alone. It operates on the class decision and compares foreground and
background evidence explicitly.

The implementation should require only:

- one untouched image;
- one backbone forward;
- foreground and background feature pooling;
- two inexpensive uses of the existing classifier;
- no new learned parameters;
- no inference changes.

### Protecting clean accuracy

Do not apply the new margin equally to every example. Use a smooth, detached
gate that activates only when the model exhibits shortcut risk:

```text
shortcut_gate =
    outside_evidence > allowed_outside
    OR foreground_margin < required_foreground_margin

L_total = L_CE + ramp * (
    L_V2 + relative_weight * shortcut_gate * L_relative
)
```

The gate should prevent the auxiliary objective from changing examples that
already have sufficient foreground support and acceptable outside evidence.
This is particularly important for ViT-Tiny and ViT-Small, where excessive or
miscalibrated spatial pressure can reduce accuracy without improving robust
behavior.

The background branch may use attenuated gradients if full symmetric
optimization damages clean accuracy. It should not be completely detached
without an ablation, because the method must still discourage class-predictive
background features.

## Secondary direction: gated V2+

V2+ satisfies the method constraints:

- no input manipulation;
- one forward pass;
- no learned parameters;
- approximately 1.7% measured steady-state overhead on ViT-Small;
- consistent leakage improvement across three ResNet-50 seeds;
- improved mean worst-background accuracy;
- five wins out of six quality metrics against CAMAL on ViT-Small validation.

However, its three-seed CUB test result shows over-constraint:

| Metric | V2+ minus V2 |
|---|---:|
| Clean accuracy | -0.33 pp |
| Strict leakage | -0.019, favorable |
| Background mean | +0.04 pp |
| Background worst | +1.88 pp |
| Foreground mean | +0.13 pp |
| Foreground only | -0.39 pp |

The target-versus-rival term is useful, but it should be applied selectively.
A successor should activate it when confidence is moderate and the foreground
target-versus-rival margin is insufficient. Already-correct, high-margin
examples should receive little or no competitor pressure.

Gated V2+ is the backup candidate if the relative pooled-margin formulation
does not pass the cross-architecture validation screen.

## Lightweight architecture-aware scaling

The useful part of Universal FCAL is the idea of adapting auxiliary pressure.
The full gradient-projection controller should not be used for the primary
method because it is complex, can be expensive, and has produced mixed
results.

A cheap alternative is detached loss-magnitude adaptation:

```text
lambda = clip(
    target_ratio * EMA(CE_loss) / EMA(auxiliary_loss),
    lambda_min,
    lambda_max,
)
```

The exponential moving averages should be detached from autograd. This avoids
separate gradient calculations and projections while reducing scale mismatch
between architectures. Conservative limits and an auxiliary decay or stop
condition should prevent late-training drift.

This scaling should be evaluated as an ablation, not automatically combined
with every new term.

## Directions to stop or deprioritize

### V3 and V3.1

Do not continue these as part of the cheap-loss claim. Their reported runs use
manipulated background/foreground views and a second backbone forward. They
belong to a separate mask-guided counterfactual augmentation direction. Their
robustness gains cannot be attributed to a clean-input FCAL loss without a
matched augmentation-only control.

### Pure V4 family

V4.2 was the strongest pure V4 candidate, but it was approximately 438% slower
than V2 and still failed three of nine validation gates. It is not compatible
with the cheap-loss claim in its current form.

### Certificate V5

Stop this formulation. The vectorized candidate was computationally
acceptable but passed only the IoU gate:

- strict leakage: 0.537;
- background worst: 17.33%;
- worse insertion and removal faithfulness than V2;
- good mask overlap without robust class evidence.

The exact full-resolution top-k implementation was also too slow at roughly
90.6 samples per second.

### V2.3 and V2.4

Do not continue their full-resolution coverage and seed-and-grow
implementations. They were too slow and showed that broader clean-image CAM
coverage does not automatically produce robustness to changed backgrounds.

### V2.5, V2.6, and V2.9

- V2.5 did not achieve the required clean/localization/sufficiency balance.
- V2.6 showed that regional mean dominance can be satisfied while extreme
  background robustness collapses.
- V2.9 combined incompatible objectives, performed poorly, and exceeded the
  compute target.

### V2.7a, V2.7b, and V2.7d

Do not continue the expensive per-cell variants. V2.7a and V2.7b failed the
compute screen. V2.7d improved clean accuracy, leakage, and foreground
corruption but lost 8.42 pp worst-background accuracy relative to V2.7 and
also failed the compute gate. Preserve their mechanistic lessons, not their
implementations.

### Full Universal FCAL controller

Do not prioritize it as the main method. Retain it as experimental machinery
and reuse only lightweight scheduling or scaling ideas. Its Pet improvements
are modest, its ISIC result is mixed, and the older controller showed that
one-time calibration does not remain reliable as representations drift.

## Experimental sequence

### Stage 1: cheap validation screen

Evaluate the new relative pooled-margin method on three deliberately different
architectures:

1. ResNet-50: strongest established FCAL case;
2. EfficientNet-B0: unstable CNN boundary case;
3. ViT-Small/16: important weak-transfer ViT case.

Use only validation data for method selection. All runs should resume their
paired CE epoch-5 branches and preserve the architecture-specific optimizer,
schedule, batch order, augmentation, and total training budget.

The candidate advances only if it satisfies every gate:

| Gate | Requirement |
|---|---|
| Clean accuracy | no more than 0.25-0.50 pp below V2 |
| Background mean | at least V2 |
| Background worst | at least V2 |
| Foreground only | at least V2 |
| Strict leakage | no worse than V2 |
| Training wall time | no more than 1.15x V2 |
| Peak allocated memory | approximately V2, with a predeclared tolerance |
| Input policy | one original, untouched image |
| Forward policy | exactly one backbone forward |
| Inference | identical model, parameters, and computation |

The exact numerical thresholds must be frozen before launching training.

### Stage 2: complete architecture screen

If all Stage-1 gates pass, expand without changing the loss to:

- DenseNet-121;
- ViT-Tiny/16;
- ViT-Base/16.

This stage determines whether the candidate genuinely transfers across CNNs
and ViTs rather than succeeding only on ResNet-50.

### Stage 3: external-dataset confirmation

Oxford-IIIT Pet should be the first external dataset. It already provides
positive FCAL evidence and has a full test partition. Compare the frozen
candidate against:

- CE;
- FCAL V2;
- CAMAL;
- RRR;
- GAIN-ext.

At minimum, use ResNet-50 and ViT-Base/16. Prefer three seeds. The current
absence of non-CE competitors outside CUB is the largest gap in the evidence
for a general method claim.

Do not prioritize a large ISIC competitor matrix until the current 100-image
test limitation is addressed through a larger evaluation set, repeated
cross-validation, or another statistically defensible protocol.

### Stage 4: replication

After the formulation and coefficients are frozen, repeat the final method on
multiple seeds. Report paired seed-level uncertainty, failure rates, and exact
per-architecture values. Image-level tests do not replace training-seed
uncertainty.

## Required comparisons and reporting

Every final comparison should separate:

- clean classification accuracy;
- background mean and worst-case robustness;
- foreground-corruption robustness;
- foreground-only sufficiency;
- background-only behavior;
- strict leakage;
- mask IoU, insertion AUC, and removal AUC;
- training time and peak allocated memory;
- inference parameters and throughput;
- CNN and ViT results;
- fixed/default and validation-tuned settings.

The paper should not claim universal localization superiority. CAMAL currently
has stronger mask IoU, GAIN-ext often has stronger removal behavior and lower
leakage, and RRR can achieve higher worst-background accuracy. The defensible
target is the best combined accuracy-robustness-efficiency trade-off.

## Final decision

The next primary experiment should be a **relative foreground-versus-
background pooled class margin built on FCAL V2**. It has the greatest
potential to improve shortcut robustness while satisfying the exact research
claim.

The **gated V2+ rival margin** should be retained as the secondary candidate.
Ordinary **FCAL V2** remains the required anchor and publication method until
a successor passes the full CNN/ViT, efficiency, and external-dataset gates.
