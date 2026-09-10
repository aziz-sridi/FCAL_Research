# FCAL V2.4 Seed-and-Grow validation protocol

## Frozen design

V2.4 is an original-input, loss-only successor to FCAL V2. Every step forwards
the ordinary training tensor exactly once. Masks are used after that forward
to define losses on the target-class evidence map; no input pixels are erased,
replaced, recomposed, or otherwise modified.

At each native evidence-map depth:

```text
A = ReLU(target CAM) / max(ReLU(target CAM))
B = sigmoid(RMSNormalize(target signed evidence) / 1.0)
seed = argmax_foreground(stopgrad(A))
eligible = foreground and stopgrad(A >= 0.35)
active = connected_component(eligible, seed)
front = dilate(active, one cell) and foreground and not active
L_grow = mean_front ReLU(0.65 - B)^2

L_V2.4 = CE + ramp * (
    adaptive_lambda * L_V2_tolerant
    + 0.05 * L_V2_support
    + 0.025 * L_strict
    + 0.05 * L_grow
)
```

Seed, eligibility, connectivity, and front selection are detached. The smooth
signed-evidence target gives inactive or negative front cells a nonzero
gradient. Gradients flow through target evidence only on the current front. The foreground mask
is projected to the native grid and thresholded at 0.50. Growth uses an
eight-connected 3x3 neighborhood. V2.3's full-resolution coverage loss and
V2+'s competitor loss are absent.

## Frozen run

- Config: `configs/cub/resnet50_fcal_v2_4_seed_grow.yaml`.
- Dataset/split: existing CUB-200-2011 seed-42 train/validation split.
- Backbone: ResNet-50.
- Initialization: exact paired CE epoch-5 branch after split-hash verification.
- Budget: epochs 6--50 of the common 50-epoch schedule.
- Selection: clean validation accuracy only.
- No coefficient or threshold changes are permitted after viewing this run.
- The existing CUB test remains unopened for V2.4.

## Validation screen

| Metric | Frozen requirement |
|---|---:|
| Clean accuracy | >= 0.800833 (V2) |
| Strict leakage | <= 0.266214 (V2) |
| Background mean | >= 0.731603 (V2+) |
| Background worst | >= 0.526667 (V2+) |
| Foreground mean | >= 0.325705 (V2.3) |
| Foreground only | >= 0.732500 (V2+/V2.3) |

Passing every metric promotes the frozen formulation to three seeds. CAMAL's
73.47% background mean, 60.33% background worst, 33.18% foreground mean, and
73.50% foreground-only accuracy remain stretch targets.

## Compute gate

Post-branch wall time must not exceed CAMAL's recorded 373.859 seconds, and
peak allocated CUDA memory must remain below 1.9218 GiB (1.10 times V2).
