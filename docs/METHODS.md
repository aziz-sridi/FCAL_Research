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

## Universal FCAL

Universal FCAL separates the architecture-specific evidence extractor from an architecture-independent objective. It:

- infers image and evidence-map resolution at runtime;
- calibrates allowed outside evidence and target foreground coverage from a shared CE warm-up state;
- excludes empty, full, or spatially degenerate masks from spatial supervision while retaining CE;
- budgets the auxiliary contribution by a target CE-to-FCAL gradient ratio;
- can project gradient conflicts before the optimizer step; and
- optionally adjusts the gradient budget using validation accuracy and localization progress.

The core package is [`src/universal_fcal/`](../src/universal_fcal/). Model adapters provide signed maps shaped `[B, H, W]`; the package has no dependency on a particular CNN, ViT, or dataset.

## Fair comparison protocol

For a given architecture, CE, V1, V2, and Universal FCAL share the same data pipeline, initialization, optimizer, and learning-rate schedule. CE saves an epoch-5 branch. Every FCAL run resumes from that exact state and completes the same 50-epoch budget. The FCAL runs differ only in their training objective.

No FCAL method changes the deployed architecture. Inference parameters and operations are identical to the CE baseline.
