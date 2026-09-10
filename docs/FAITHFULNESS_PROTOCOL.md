# Sealed post-selection faithfulness protocol

This evaluator is run only after a checkpoint has been selected on validation
data. It never constructs a training loader and accepts only `validation` or
`test`. The test invocation is final measurement, not another model-selection
or threshold-selection step. Output directories are write-once: an existing
artifact directory is never overwritten.

## Frozen measurements

- The scoring target is the ground-truth class softmax confidence for every
  image, including misclassified images.
- CE, FCAL, RRR, GAIN-ext, CAMAL, and the audited ResNet-50 ISNet adapter use
  the same non-negative ground-truth-class final spatial evidence map. This is
  an exact class activation map for the study's GAP CNNs and average-pooled
  ViTs. A non-average-pooled ViT whose available rollout is not class-specific
  is rejected rather than mislabeled. FCAL V2 and Universal FCAL checkpoints
  with the same exact map interface are also supported.
- Maps are clipped non-negative, max-normalized per image, and then bilinearly
  resized to the image/mask grid. Attention IoU thresholds the map strictly above
  `0.7`. Empty or full masks are invalid and excluded from the IoU mean, while
  their faithfulness curves remain in the all-image mean.
- Insertion and removal rank pixels by the same normalized map at fractions
  `0.0, 0.1, ..., 1.0`. The baseline is exactly zero in normalized model-input
  space (the ImageNet mean color before normalization). AUC is trapezoidal on
  the unit fraction axis using softmax confidence, so its normalized range is
  `[0, 1]`; there is no outcome-dependent endpoint rescaling.
- The default is every image. `--max-images N` uses the lowest SHA-256 values of
  `selection_seed:image_id`, independent of labels, predictions, masks, or
  metric outcomes. Freeze `N` and the selection seed before any test command
  and use the same values for every method.

Each run writes `protocol.json`, `summary.json`, and `per_image.npz`. The
protocol records checkpoint and selected-ID hashes, map/target conventions,
the prepared-split-file hash, perturbation grid, and subset rule. The NPZ retains per-image IoU,
AUCs, endpoints, and full confidence trajectories for paired analysis.

## Reproducible batch workflow

`fcal_study.faithfulness_batch` discovers the declared study matrix rather
than globbing arbitrary checkpoint directories. The fixed manifest contains
31 jobs: CE, FCAL V2, CAMAL, GAIN-ext, and RRR on all six architectures, plus
Selective Faster-ISNet on ResNet-50 only. The latter is labeled
`resnet50_only` and `architecture_wtl_eligible=false`; unsupported ISNet cells
are never imputed. A complete frozen tuning manifest adds 25 jobs: four methods
on six architectures and Selective Faster-ISNet on ResNet-50.

Before tuning is frozen, a fixed-only validation batch can verify the procedure
and memory envelope without opening the test split:

```powershell
python -m fcal_study.faithfulness_batch manifest `
  --split validation `
  --archive-root ../archive/legacy_projects/adaptive_fcal_study/outputs `
  --fixed-root outputs/baseline-fixed/cub `
  --artifact-root outputs/faithfulness/cub `
  --output outputs/faithfulness/cub/manifests/fixed-validation.json `
  --max-images 500 --selection-seed 2026 --batch-size 16

python -m fcal_study.faithfulness_batch queue `
  outputs/faithfulness/cub/manifests/fixed-validation.json

python -m fcal_study.faithfulness_batch run-all `
  outputs/faithfulness/cub/manifests/fixed-validation.json

python -m fcal_study.faithfulness_batch aggregate `
  outputs/faithfulness/cub/manifests/fixed-validation.json `
  --output-dir results/faithfulness/cub/fixed-validation
```

`queue` is inspection-only. `run-all` is serial and resumable: an exactly bound,
complete artifact is skipped; a missing artifact is run; an incomplete,
malformed, or mismatched existing artifact causes a hard failure. Neither
evaluation directories nor aggregation directories are overwritten. Preserve
an invalid directory for audit and use a new explicitly reviewed artifact root
instead of deleting or mutating it.

## Exact post-freeze test commands

Freeze winners only after the validation-only tuning queue is complete:

```powershell
python -m fcal_study.tuning freeze `
  configs/cub/tuning_validation.yaml `
  --ledger outputs/tuning-validation/attempts.jsonl `
  --output outputs/tuning-validation/frozen-winners.yaml
```

The batch builder verifies that the frozen file belongs to the declared study,
has `selection_split: validation` and `test_set_used: false`, contains exactly
one winner for every supported method/architecture cell, uses a coefficient
from each registered search grid, and points to an existing selected
checkpoint. Without that complete freeze, `--split test` is rejected.

Create the combined 56-job sealed-test manifest, inspect it, execute it
serially, and aggregate it exactly once:

```powershell
python -m fcal_study.faithfulness_batch manifest `
  --split test `
  --archive-root ../archive/legacy_projects/adaptive_fcal_study/outputs `
  --fixed-root outputs/baseline-fixed/cub `
  --artifact-root outputs/faithfulness/cub `
  --tuning-manifest configs/cub/tuning_validation.yaml `
  --frozen-winners outputs/tuning-validation/frozen-winners.yaml `
  --output outputs/faithfulness/cub/manifests/fixed-and-tuned-test.json `
  --max-images 500 --selection-seed 2026 --batch-size 16

python -m fcal_study.faithfulness_batch queue `
  outputs/faithfulness/cub/manifests/fixed-and-tuned-test.json `
  --format jsonl

python -m fcal_study.faithfulness_batch run-all `
  outputs/faithfulness/cub/manifests/fixed-and-tuned-test.json

python -m fcal_study.faithfulness_batch aggregate `
  outputs/faithfulness/cub/manifests/fixed-and-tuned-test.json `
  --output-dir results/faithfulness/cub/fixed-and-tuned-test
```

The manifest is deterministic and write-once, binds every checkpoint by SHA-256,
and freezes the subset size, selection seed, batch size, support scope, and
tuning provenance. The aggregator reopens each NPZ, recomputes its summary,
checks the checkpoint/split/subset bindings, and writes `summary.csv` plus
`REPORT.md`. It reports mask IoU, insertion AUC, and removal AUC; higher is
favorable for the first two and lower is favorable for removal AUC.

## Single-checkpoint commands

Use validation to verify the pre-registered procedure and resource envelope:

```powershell
python -m fcal_study.faithfulness `
  "outputs/<run>/best-validation.pt" --split validation --max-images 500
```

After all checkpoint and protocol choices are frozen, run the sealed test once:

```powershell
python -m fcal_study.faithfulness `
  "outputs/<run>/best-validation.pt" --split test --max-images 500
```

Omit `--max-images` for the all-image protocol. `--batch-size` may be reduced
for memory safety without changing selection or metric definitions. Evaluate
checkpoints serially on the single study GPU.

## Limitations

Pixel replacement can create off-manifold inputs, and a zero normalized-input
baseline is not a black image. Low-resolution CAM ties eventually fall back to
stable raster order after bilinear upsampling. AUC and IoU measure different
properties and neither establishes causal feature use by itself. Image-level
pairing does not replace seed-level uncertainty, and no test result may be used
to retune the checkpoint, threshold, grid, subset, or baseline.

This primary common-CAM analysis does not evaluate RRR's native input-gradient
map or ISNet's native selective LRP relevance. Mixing those heterogeneous maps
into the primary table would confound the map estimator with the training
method; either may be added only as a separately pre-registered supplementary
analysis and must retain its native target convention.
