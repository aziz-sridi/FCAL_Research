# FCAL V2.3 balanced containment validation protocol

## Frozen design

V2.3 is a loss-only extension of the exact FCAL V2 configuration. Every
training step uses the ordinary training tensor once and performs one model
forward. It does not erase, replace, corrupt, recompose, or otherwise create
an additional image. Masks are consumed only by the training loss, and
inference is unchanged.

V2.3 retains V2's resolution-aware tolerant containment and distributed signed
foreground support. It adds two target-only terms:

```text
L_strict = positive target-evidence mass outside the undilated mask
A_y = ReLU(target CAM) / max(ReLU(target CAM))
L_coverage = ReLU(0.60 - mean_foreground(A_y))^2

L_V2.3 = CE + ramp * (
    adaptive_lambda * L_V2_tolerant
    + 0.05 * L_V2_support
    + 0.025 * L_strict
    + 0.20 * L_coverage
)
```

The coverage hinge stops contributing after sufficient foreground coverage;
it is not a pixelwise mask-reconstruction loss. V2+'s competitor term is not
included.

## Frozen run

- Config: `configs/cub/resnet50_fcal_v2_3.yaml`.
- Dataset/split: existing CUB-200-2011 seed-42 train/validation split.
- Backbone: ResNet-50.
- Initialization: exact paired CE epoch-5 branch after split-hash verification.
- Budget: epochs 6--50 of the common 50-epoch schedule.
- Selection: clean validation accuracy only.
- No coefficient change is permitted after viewing this candidate.
- The existing CUB test is development-exposed and is not a pristine final
  confirmation set for V2.3.

## Validation screen

The candidate is compared with the frozen seed-42 validation references.

| Metric | V2 | V2+ | CAMAL | V2.3 target |
|---|---:|---:|---:|---:|
| Clean accuracy | 0.800833 | 0.796667 | 0.789167 | >= 0.800833 |
| Strict leakage | 0.266214 | 0.243537 | 0.315668 | <= 0.266214 |
| Background mean | 0.720449 | 0.731603 | 0.734744 | >= 0.731603 |
| Background worst | 0.489167 | 0.526667 | 0.603333 | >= 0.526667 |
| Foreground mean | 0.318269 | 0.317372 | 0.331795 | >= 0.318269 |
| Foreground only | 0.716667 | 0.732500 | 0.735000 | >= 0.732500 |

Passing every target promotes the frozen formulation to a three-seed study.
CAMAL's background and foreground values remain stretch targets rather than
post-hoc requirements.

## Compute gate

The seed-42 post-branch run must use at most CAMAL's recorded 373.859 seconds
and no more than 1.10 times V2's 1.7471 GiB peak allocated CUDA memory. This is
a single-run feasibility gate, not a timing confidence interval.
