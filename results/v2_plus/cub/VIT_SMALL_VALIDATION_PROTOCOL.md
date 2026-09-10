# FCAL V2+ ViT-Small validation protocol

## Frozen candidate

- Architecture: ViT-Small/16 with average patch pooling.
- Seed and split: CUB seed 42, existing train/validation split.
- Initialization: exact archived paired CE epoch-5 branch.
- Budget: 50 total epochs; select by clean validation accuracy only.
- V2+ hyperparameters: copied unchanged from the ResNet-50 V2+ candidate.
- Input/compute rule: one original untouched image and one backbone forward per
  training step; masks are consumed only by the loss.

The candidate is `configs/cub/vit_small_avg_fcal_v2_plus.yaml`. No
architecture-specific coefficient tuning is permitted after viewing its
result.

## Evaluation rule

The CUB test split remains sealed during this screen. The selected checkpoint
is evaluated on the validation stress suite and compared with the matching
seed-42 ViT-Small FCAL V2 reference. Promotion requires non-inferiority on all
six principal metrics: clean accuracy, strict leakage, background mean,
background worst, foreground mean, and foreground-only accuracy. Training
wall time, throughput, and peak allocated CUDA memory are reported against
the matching V2 and CAMAL runs.

The full frozen stress suite remains primary. A structured-only view excluding
the random-RGB perturbation may be reported secondarily but cannot change the
promotion decision.
