# FCAL V2+ foreground-competitor validation protocol

## Non-negotiable design constraint

V2+ is a loss-only extension of FCAL V2. Every training step uses exactly the
original image and exactly one model forward pass. It does not construct a
second view, replace a background, corrupt a foreground, erase pixels, or
modify the input tensor. Masks are consumed only by the loss, as in V2.

## Hypothesis

V2 constrains where target-class evidence appears and ensures distributed
signed support inside the object. It does not explicitly require that this
foreground evidence distinguish the true class from the model's strongest
fine-grained rivals.

V2+ retains the complete V2 objective and adds a small foreground-only margin:

```text
rivals = top non-target classes from stopgrad(logits)
D_c = RMSNormalize(S_target) - RMSNormalize(S_c)
q_c = mean_foreground(tanh(D_c / evidence_temperature))
L_rival = mean_c softplus((margin - q_c) / loss_temperature)

L_V2+ = CE + ramp * (
    lambda * L_out + support_weight * L_support
    + coefficient * L_rival
)
```

Competitor evidence is extracted from features already produced by the clean
forward pass. No second forward and no inference change are introduced.

## Frozen candidate

- Config: `configs/cub/resnet50_fcal_v2_plus.yaml`
- Dataset/split: existing CUB-200-2011 seed-42 train/validation split.
- Backbone: ResNet-50.
- Initialization: exact paired CE epoch-5 branch after split-hash verification.
- Budget and selection: 50 epochs; checkpoint selected by validation clean
  accuracy only.
- Coefficient: 0.025, chosen before training to be comparable to the observed
  weighted magnitude of V2's support term.

## Joint validation gate

The official test split remains sealed unless V2+ matches or improves every
principal V2 validation metric.

| Metric | V2 reference | V2+ requirement |
|---|---:|---:|
| Clean accuracy | 0.800833 | >= 0.800833 |
| Strict leakage | 0.266214 | <= 0.266214 |
| Background mean | 0.720449 | >= 0.720449 |
| Background worst | 0.489167 | >= 0.489167 |
| Foreground mean | 0.318269 | >= 0.318269 |
| Foreground only | 0.716667 | >= 0.716667 |

Failure on any gate keeps the result validation-only. No coefficient change is
permitted after viewing this candidate's result.

## Frozen multi-seed replication rule

The seed-42 screen improved four of six metrics but missed clean accuracy and
foreground-corruption mean. Before any further loss change, the unchanged
candidate will be replicated from the existing seed-7 and seed-123 paired CE
branches. This tests whether the small misses are training-seed noise without
tuning the method.

The three-seed V2 validation references, computed before launching either V2+
replication, are:

| Metric | Three-seed V2 mean | Three-seed V2+ requirement |
|---|---:|---:|
| Clean accuracy | 0.794722 | >= 0.794722 |
| Strict leakage | 0.268603 | <= 0.268603 |
| Background mean | 0.721090 | >= 0.721090 |
| Background worst | 0.518889 | >= 0.518889 |
| Foreground mean | 0.315833 | >= 0.315833 |
| Foreground only | 0.718056 | >= 0.718056 |

The V2+ coefficients and implementation remain frozen. The result advances
only if all six three-seed mean gates pass. Until then, the official test split
remains sealed for every V2+ seed.
