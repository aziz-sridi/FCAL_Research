# Fixed-recipe explanation-baseline matrix

## Scope

The matched CUB benchmark targets the six architectures already reported for
FCAL: ResNet-50, DenseNet-121, EfficientNet-B0, and average-pooled ViT-Tiny,
ViT-Small, and ViT-Base. The fixed arm preserves each architecture's FCAL
optimizer, data split, augmentations, effective batch size, seed (42), and
50-epoch schedule. Each baseline branches from the corresponding CE checkpoint
after epoch 5, just as the matched FCAL runs do.

The active repository contains the aggregate six-architecture results but not
the original seed-42 checkpoints. The archived project contains the six CE
epoch-5 branch checkpoints at:

`D:\AI Research\archive\legacy_projects\adaptive_fcal_study\outputs\<architecture>-ce-seed42\epoch-005-branch.pt`

The archived and current train/validation/test split files and manifests have
identical SHA-256 hashes. Training accepts their relocated paths only with the
explicit `--allow-data-relocation` and `--parent-split-dir` flags, after
rechecking all non-path loader settings and every prepared split hash. The
hash authority is `environment.json` beside the actual `--resume` checkpoint;
the explicit archived split directory must agree with it. Branch history is
likewise read from `history.jsonl` beside the actual checkpoint, never from the
stale `split_dir` or `run_dir` embedded before the project was archived.

## Support matrix

| Architecture | RRR | GAIN-ext | CAMAL | Selective Faster-ISNet |
|---|---:|---:|---:|---:|
| ResNet-50 | yes | yes | yes | yes |
| DenseNet-121 | yes | yes | yes | no |
| EfficientNet-B0 | yes | yes | yes | no |
| ViT-Tiny/16 avg | yes | yes | yes | no |
| ViT-Small/16 avg | yes | yes | yes | no |
| ViT-Base/16 avg | yes | yes | yes | no |

RRR is input-gradient based and architecture-independent. GAIN-ext and CAMAL
use each FCAL model's exact final spatial class-evidence map. For the GAP CNNs
and average-pooled ViTs, this is algebraically equal to final-representation
Grad-CAM up to positive scale; ReLU and per-sample min/max normalization remove
that scale. This is an explicit adaptation for the FCAL architectures. The
GAIN-ext arm uses the available foreground mask for every training image; it
is therefore a full-mask comparison, not the original paper's limited
annotation-budget experiment. Its ViT rows are likewise an exact-CAM
architecture adaptation rather than an experiment reported in the GAIN paper.

The repository's audited LRP-Flex adapter implements Selective Faster-ISNet for
ResNet-50 only. The Faster-ISNet reference implementation supports ReLU-based
DNNs and evaluates ResNet and DenseNet, but these FCAL targets also include a
SiLU EfficientNet and GELU ViTs. Grad x Input or attention rollout must not be
substituted and reported as ISNet. Extending the audited adapter is separate
method-development work, not part of the fixed benchmark.

## Fixed method settings

- RRR: the released implementation's `l2_grads=1e-4`. The loss is the squared
  gradient of the summed class log probabilities on annotated background input
  dimensions.
- GAIN-ext: paper/released-code threshold `sigma=0.5`, sigmoid scale `8`,
  classification weight `1`, attention-mining weight `1`, and external-mask
  weight `10`.
- CAMAL: `CE + lambda * (beta - alpha)` with `lambda=1`; `alpha` and `beta`
  are normalized mean attention inside and outside the foreground mask.
- Selective Faster-ISNet: LRP epsilon `0.01`, selective epsilon `0.01`, heatmap
  weight `P=0.7`, `C1=1`, `C2=25`, and GWRP decay `d=0`, in FP32. The optional
  first-layer LRP-zB refinement is disabled: ImageNet-normalized inputs do not
  satisfy its naive pixel-bound assumptions, and the audited adapter does not
  implement a normalization-aware zB rule. This is Selective LRP-Flex without
  zB, not the complete optional reference configuration.

These fixed values are not selected using CUB test results. A later tuned arm
must give each method the same validation-only search budget and must preserve
the test set as a one-time final evaluation.

Primary sources and audited code snapshots:

- RRR paper: <https://arxiv.org/abs/1703.03717>; released code:
  <https://github.com/dtak/rrr> (`d83a16e`).
- GAIN paper: <https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Tell_Me_Where_CVPR_2018_paper.pdf>;
  implementation audited at <https://github.com/alokwhitewolf/Guided-Attention-Inference-Network>
  (`cd1cb0a`).
- Faster-ISNet paper: <https://doi.org/10.1109/ACCESS.2024.3461773>; released
  code: <https://github.com/PedroRASB/ISNet> (`7ad0a9e`).
- CAMAL v1: <https://arxiv.org/abs/2605.08325>; MIT source archive:
  <https://doi.org/10.5281/zenodo.20084662> (record `20084663`, archive MD5
  `054c421a70ad982fe49925363133e215`).

## Commands and outputs

Regenerate the 19 fixed configs and print the queue:

```powershell
python -m fcal_study.baseline_matrix write
python -m fcal_study.baseline_matrix support
python -m fcal_study.baseline_matrix commands
```

Training outputs are written under
`outputs/baseline-fixed/cub/<architecture>-<method>-seed42/`. A bounded GPU
smoke test that does not write a checkpoint is available as:

```powershell
python -m fcal_study.baseline_matrix profile camal vit-base-p16-avg --batches 2
```

The profile command is a feasibility smoke, not a paired timing result: it
loads the parent weights but does not restore the parent optimizer moments,
RNG, or sampler state. Publication timing comes from complete run histories.
In those histories, the generic fields `train.fcal`, `train.weighted_fcal`,
and `train.lambda` mean raw auxiliary objective, weighted auxiliary objective,
and total auxiliary coefficient for every explanation baseline; they are
legacy schema names and must not be described as FCAL-specific quantities.

After a run completes, evaluate the validation-selected checkpoint on the same
clean and background-intervention suite used by FCAL:

```powershell
python -m fcal_study.evaluate "outputs/baseline-fixed/cub/<run>/best-validation.pt"
```

## Hardware and feasibility audit

The available machine has one RTX 5070 Ti (16,303 MiB), an Intel i5-14600KF
(14 cores/20 threads), and 32 GiB RAM. Two-batch smoke tests on the largest
ViT-Base target passed at batch size 32 with peak allocated CUDA memory of
approximately 3.26 GiB for CAMAL, 5.28 GiB for GAIN-ext, and 7.64 GiB for RRR.
RRR requires timm's explicit attention path because PyTorch efficient/fused
attention does not implement the required double backward; the explicit path
is the same attention operation and passed the smoke test.

Completed ResNet-50 seed-7 branches provide more realistic post-branch timing
anchors than cold-start smoke timing: about 31.9 minutes for RRR, 10.0 minutes
for GAIN-ext, and 56.8 minutes for Selective Faster-ISNet over epochs 6--50.
CAMAL's exact-map implementation has FCAL-like first-order memory and compute.
Run time will vary with concurrent GPU use; the matrix should be queued
serially on this single GPU.

The canonical RRR coefficient produces a large initial auxiliary term on the
ViT-Base CE branch (about 13 after weighting in the smoke batch). This is not a
numerical failure, but it may damage accuracy. It remains in the fixed arm for
source fidelity; the tuned arm should search its coefficient on validation data.
