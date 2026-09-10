# FCAL V2.7a/V2.7b frozen validation protocol

## Input and compute policy

Both candidates preserve exact V2.7 and add one loss computed from the same
final feature tensor. Every minibatch forwards one ordinary, untouched image
tensor through the backbone exactly once. No image is masked, cropped, erased,
replaced, recomposed, or forwarded a second time. Neither candidate adds a
learned parameter or changes inference.

## Frozen candidates

V2.7a selects the three strongest non-target rivals from detached clean logits.
At each native foreground cell it forms a signed, RMS-normalized target-minus-
strongest-rival margin. The weakest 25% of foreground cells are selected with
detached indices and receive a temperature-0.25 softplus toward margin zero.
V2.7's foreground-pooled CE remains weighted 0.02; the local loss weight is
0.006.

V2.7b computes target evidence with negative classifier weights removed before
channels are summed. This exposes positive target-channel activation that the
ordinary signed CAM can hide through cancellation. Its V2-tolerant outside
mass is penalized at weight 0.0075, while V2.7 foreground-pooled CE remains
weighted 0.02. The implementation reuses V2's already-computed tolerance
projection; this is mathematically identical to recomputing it and avoids a
duplicate full-resolution max-pool. Its raw target CAM and positive-weight CAM
are also emitted by one batched class projection rather than two separate
projections; this changes neither tensor nor gradient.

The weights were scale-calibrated without updates on 16 fixed training batches
from the common CE epoch-5 branch. Each new weighted term was approximately
0.003 at calibration, about half V2.7's 0.0063 foreground-pooled contribution.
All values were frozen before either training run.

## Run and conditional combination rule

- CUB-200-2011 seed-42 train/validation split and ResNet-50.
- Exact paired CE epoch-5 model, optimizer, scheduler, and RNG branch.
- Common epochs 6--50; checkpoint selection by clean validation accuracy.
- Validation stress suite only; the CUB test remains unopened.
- V2.7c may be created only if both V2.7a and V2.7b are non-inferior to V2.7
  on all six metrics below and improve at least one metric strictly.

| Metric | V2.7 advancement floor | Final promotion target |
|---|---:|---:|
| Clean accuracy | >= 0.800000 | >= 0.800833 |
| Strict leakage | <= 0.255615 | <= 0.266214 |
| Background mean | >= 0.730641 | >= 0.731603 |
| Background worst | >= 0.524167 | >= 0.526667 |
| Foreground mean | >= 0.320385 | >= 0.325705 |
| Foreground only | >= 0.734167 | >= 0.735000 |

Post-branch wall time must remain at most CAMAL's 373.859 seconds and peak
allocated CUDA memory below 1.9218 GiB.
