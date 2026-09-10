# FCAL methods

FCAL adds spatial supervision to an ordinary classification loss. Let `S` be a signed, class-specific evidence map, `M` a foreground mask projected to the evidence-map resolution, and `CE` the cross-entropy loss. The mask is required only while training.

## FCAL V1

V1 converts positive class evidence into a spatial probability distribution:

```text
P = ReLU(S) / (sum(ReLU(S)) + epsilon)
L_out = sum(P * (1 - M_tolerant))
lambda = clip(alpha * stopgrad(CE / (L_out + epsilon)), lambda_min, lambda_max)
L = CE + lambda * L_out
```

`M_tolerant` is a resized and dilated foreground mask. CNNs use the final class activation map; ViTs use class-specific patch evidence. For multi-layer ViT evidence, later layers receive larger linear weights.

V1 directly penalizes positive attention outside the object. Its normalization couples foreground and background: moving probability mass into the foreground lowers the objective. A limitation is that all-negative or extremely sparse maps can make the containment term ineffective.

## FCAL V2

V2 is the support-aware method called V5 during internal development. It keeps V1-style positive containment and adds bounded, distributed support over the undilated foreground:

```text
S_hat = S / stopgrad(sqrt(mean(S^2) + epsilon))
q_fg = sum(M * tanh(S_hat / T)) / sum(M)
L_support = softplus(margin - q_fg)
L = CE + ramp * (lambda * L_out + support_weight * L_support)
```

Detached RMS normalization makes support comparable across evidence scales without allowing the model to satisfy the loss by changing only the scale estimate. Averaging bounded evidence across the foreground prevents one extreme cell from satisfying the objective.

The containment tolerance is defined in input pixels as half an evidence-map cell by default:

```text
delta = ceil(0.5 * max(input_height / map_height, input_width / map_width))
```

For a 224×224 image, this gives 16 pixels for a 7×7 CNN map and 8 pixels for a 14×14 ViT map.

## FCAL V2+ candidate: foreground competitor margin

V2+ is a strict loss-only extension of V2. It uses the original image once and
extracts the strongest non-target class evidence maps from the same forward
features. A bounded foreground margin requires target evidence to outrank
those competitors inside the annotated object. V2 containment and distributed
support remain unchanged and continue to control evidence placement.

V2+ performs no erasing, corruption, recomposition, counterfactual generation,
or second image forward. The mask and auxiliary loss still disappear at
inference. Its frozen validation protocol is
`results/v2_plus/cub/VALIDATION_PROTOCOL.md`.

## FCAL V2.3 candidate: balanced containment and coverage

V2.3 returns to V1/V2's target-only spatial supervision. It preserves V2's
resolution-aware tolerant containment and distributed signed foreground
support, then adds a weak undilated containment term and a one-sided
foreground-coverage hinge. The hinge uses the evaluator-normalized positive
target CAM and stops contributing after mean foreground attention reaches
0.60; it discourages a tiny discriminative hotspot without forcing pixelwise
mask reconstruction.

V2.3 uses one untouched training image, one model forward, and one ordinary
backward pass. It contains no competitor-CAM penalty, input intervention,
input-gradient regularizer, or inference change. Its frozen protocol is
`results/v2_3/cub/VALIDATION_PROTOCOL.md`.

## FCAL V2.4 candidate: Seed-and-Grow foreground evidence

V2.4 preserves exact V2 containment/support and V2.3's weak strict-containment
term, but replaces global coverage with a connected native-grid curriculum.
The strongest target-class foreground cell is a detached seed. Foreground
neighbors already above an acceptance threshold form its connected active
component, and a one-sided loss raises only the next one-cell in-mask boundary
ring. As the ring becomes active, subsequent steps move the supervision front
outward from discriminative evidence rather than targeting the whole mask at
once.

Seed-and-Grow uses the untouched image once, one model forward, and native CAM
operations only. It introduces no second view, pixel manipulation, competitor
map, full-resolution training interpolation, input gradient, or inference
change. The frozen protocol is `results/v2_4/cub/VALIDATION_PROTOCOL.md`.

## FCAL V2.5--V2.9 loss-only candidate ladder

The next five candidates preserve V2's complete objective and differ only in
small first-order losses computed after one ordinary forward:

- V2.5 extends Seed-and-Grow by supervising both the next connected foreground
  ring and retention of the already-active connected trail.
- V2.6 applies a cheap one-sided native-grid hinge requiring normalized target
  evidence inside the foreground to dominate evidence outside it.
- V2.7 mask-pools the existing final feature map and asks the unchanged
  classifier head to recognize the foreground region.
- V2.8 adds a complementary constraint that the background-pooled features
  must not retain a strong target-class margin.
- V2.9 combines half-strength growth, dominance, and regional-classification
  coefficients as a balanced candidate.

Regional pooling masks feature activations after the backbone forward; it does
not create or forward a masked image. The candidates add no parameters and use
the identical deployed ResNet-50. Their predeclared comparison is in
`results/v2_next/cub/VALIDATION_PROTOCOL.md`.

### V2.7 foreground-sufficiency refinements

V2.7a tests the lower tail of native target-versus-rival foreground margins;
V2.7b contains positive target-channel evidence before signed channel
cancellation; and V2.7d replaces exact lower-tail sorting with an entropic
soft minimum against one rival. All operate after one untouched-image forward.
V2.7a and V2.7b failed their compute screens. V2.7d improved clean accuracy,
leakage, and foreground-corruption mean but regressed background robustness and
foreground-only accuracy, so none was promoted. See
`results/v2_7_refinements/cub/VALIDATION_REPORT.md`.

## Universal FCAL

Universal FCAL separates the architecture-specific evidence extractor from an architecture-independent objective. It:

- infers image and evidence-map resolution at runtime;
- calibrates allowed outside evidence and target foreground coverage from a shared CE warm-up state;
- excludes empty, full, or spatially degenerate masks from spatial supervision while retaining CE;
- budgets the auxiliary contribution by a target CE-to-FCAL gradient ratio;
- can project gradient conflicts before the optimizer step; and
- optionally adjusts the gradient budget using validation accuracy and localization progress.

The core package is [`src/universal_fcal/`](../src/universal_fcal/). Model adapters provide signed maps shaped `[B, H, W]`; the package has no dependency on a particular CNN, ViT, or dataset.

## FCAL V3 candidate: counterfactual support consistency

V3 is a direct successor to V2, not a successor to the separate Universal
FCAL controller. It keeps V2's containment, signed distributed support, and
resolution-aware tolerance unchanged. During training, it constructs a second
view by preserving the annotated foreground exactly and replacing only the
background with a different-class donor background. The donor foreground is
removed before composition, and a soft guard band outside the target mask
reduces boundary artifacts.

For the clean image `x` and its background-counterfactual view `x_cf`, V3 uses:

```text
L_pair = (CE(x, y) + rho * CE(x_cf, y)) / (1 + rho)
L_v2_pair = mean(V2(x, M), V2(x_cf, M))
L_pred = KL(stopgrad(p(x; T)) || p(x_cf; T)) * T^2
L_evidence = mean_M SmoothL1(tanh(S_hat(x)), tanh(S_hat(x_cf)))

L_v3 = L_pair + ramp * (
    lambda * L_out_pair
    + support_weight * L_support_pair
    + prediction_weight * L_pred
    + evidence_weight * L_evidence
)
```

The paired CE trains foreground sufficiency under changed context. Prediction
distillation preserves the clean decision distribution, while foreground
evidence consistency prevents the classifier from reaching the same answer
through a different spatial shortcut. V3 changes training only and uses the
same `raw_class_evidence(labels)` adapter as V2.

### Balanced V3 regional intervention

The balanced successor addresses the pilot V3 trade-off without changing the
deployed model. Half of each second-view minibatch uses the background
counterfactual above. The other half preserves the background exactly and
applies one mild, label-preserving foreground intervention: local blur,
additive noise, or a bounded channel mixture. Paired CE and prediction
consistency supervise every second view; foreground-evidence consistency is
restricted to background interventions because only those preserve the
foreground pixels. V2 containment and distributed-support losses remain
active on both views.

This balanced training distribution prevents the objective from learning only
background invariance at the expense of sensitivity to valid foreground
appearance changes. The intervention fraction and strengths are training
hyperparameters; they do not add parameters or inference operations.

### V3.1 routed consistency and strict shell

V3.1 routes prediction and evidence consistency only to background-intervention
samples. Foreground-intervention samples retain paired cross-entropy and V2
containment/support, but are not forced to reproduce the clean image's full
class distribution. This permits a label-preserving foreground change to alter
the relative plausibility of competing fine-grained classes.

V3.1 also divides the area outside the strict foreground into V2's distant
background and its tolerated boundary shell. A weak extra term penalizes
normalized positive evidence in the shell:

```text
shell = M_tolerant - M_strict
L_shell = sum(P_positive * shell)

L_v3.1 = L_v3_routed + ramp * strict_shell_weight * L_shell_pair
```

The distant background remains governed by V2 containment. Consequently, the
new term does not double-penalize distant evidence and does not change V2's
resolution-aware tolerance definition.

## FCAL V4: pure regional evidence and sensitivity

V4 is an original-input loss by default. With `intervention.enabled: false`,
it performs one ordinary model forward and never erases, swaps, corrupts, or
otherwise constructs image pixels. It combines V2 with complementary terms
computed from that forward pass:

```text
L_strict = positive target-class evidence mass outside M_strict
L_align = optional bounded CAM-to-mask alignment
gap = mean_M(CAM_max) - mean_not_M(CAM_max)
L_dominance = ReLU(margin - gap)^power
L_sensitivity = mean_background((grad_x target_margin)^2 / stopgrad(global mean))

L_v4 = CE + ramp * (
    lambda * L_v2_containment
    + support_weight * L_v2_support
    + strict_weight * L_strict
    + alignment_weight * L_align
    + dominance_weight * L_dominance
    + sensitivity_weight * L_sensitivity
)
```

Strict containment targets leakage. Mask alignment encourages broad object
coverage rather than a small discriminative hotspot, but is optional because
pixelwise silhouette imitation can harm discriminative faithfulness. Regional
dominance instead requires evaluator-normalized foreground evidence to outrank
background evidence. Its one-sided hinge stops contributing once the margin is
met. The scale-free target-margin derivative penalizes local background
sensitivity in the spirit of RRR while retaining V2's class-specific evidence
objective. Training this derivative requires second-order gradients but no
modified input. A separate sensitivity warmup/ramp can delay this pressure
until after the classifier and spatial evidence have formed.

V4 also exposes an explicit `intervention.enabled` option. When enabled, the
V3.1 paired-view CE and routed consistency terms are added. This is a declared
interventional mode, not the primary pure-loss configuration. Both modes use
the same inference model and require no mask after training.

## FCAL V5 candidate: regional margin certification

V5 is an independent first-order objective rather than V2 plus additional
regularizers. For target class `y`, a null rival, and the top real logit rivals
`c`, it constructs scale-free spatial margin fields:

```text
D_y,null(p) = S_y(p)
D_y,c(p) = S_y(p) - S_c(p)

F_y,c = entropic lower tail of D_y,c on foreground
B_y,c = entropic upper tail of abs(D_y,c) on background
C_y,c = F_y,c - rho * B_y,c
C = min_c C_y,c
L_v5 = temperature * softplus((margin - C) / temperature)
```

The null rival makes class-specific target leakage part of the same
certificate as real-class competition. Entropic tails are vectorized smooth
worst-region statistics at the native evidence grid. V5 uses one original
image, one forward, one ordinary backward, and no V1/V2/V4, CAMAL, RRR, or
counterfactual auxiliary term.

The first frozen ResNet-50 candidate met its compute gates and improved mask
IoU, but failed the joint validation performance gate. It is therefore a
falsified candidate rather than a promoted successor; see
`results/v5/cub/REPORT.md`.

## Fair comparison protocol

For a given architecture, CE, V1, V2, V3, and Universal FCAL share the same data pipeline, initialization, optimizer, and learning-rate schedule. CE saves an epoch-5 branch. Every FCAL run resumes from that exact state and completes the same 50-epoch budget. The FCAL runs differ only in their training objective.

No FCAL method changes the deployed architecture. Inference parameters and operations are identical to the CE baseline.
