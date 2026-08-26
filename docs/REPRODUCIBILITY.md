# Reproducing the study

## Environment

Use Python 3.10 or newer. The reported environment is pinned in [`requirements-lock.txt`](../requirements-lock.txt); the package metadata provides a portable dependency set.

```bash
python -m venv .venv
pip install -e ".[dev]"
python -m pytest
```

The lock file records a CUDA 12.8 PyTorch build. Install the PyTorch build appropriate for your hardware and driver.

## Dataset locations

The checked-in configs expect datasets beside the repository:

```text
../datasets/CUB-200-2011/versions/7/CUB_200_2011
../datasets/CUB-200-2011/versions/7/segmentations
../datasets/ISIC/Skin cancer ISIC The International Skin Imaging Collaboration
../datasets/ISIC/fcal_v1_v5_masks
../datasets/OxfordIIITPet/oxford-iiit-pet
```

The ISIC mask directory retains its historical local name; this does not refer to the public FCAL version. Edit the YAML paths if your data lives elsewhere. Dataset files are not redistributed by this repository.

Frozen split manifests are checked into [`splits/`](../splits/):

- CUB-200-2011: seed-42 80/20 train/validation split plus the official test split.
- ISIC: lesion-grouped, image-disjoint train/validation/test partitions.
- Oxford-IIIT Pet: 2,944 train, 736 validation, and the official 3,669-image test set.

## Install and verify

```bash
pip install -e ".[dev]"
python -m pytest
```

Dataset-backed tests use the paths above when available and skip automatically otherwise. Pure objective, split, and configuration tests do not require checkpoints.

## Training protocol

Train the CE baseline first. Its configuration sets `save_epoch5_branch: true`:

```bash
python -m fcal_study.train configs/cub/resnet50_ce.yaml
```

Resume every FCAL method from the same branch:

```bash
python -m fcal_study.train configs/cub/resnet50_fcal.yaml \
  --resume outputs/resnet50-ce-seed42/epoch-005-branch.pt

python -m fcal_study.train configs/cub/resnet50_fcal_v2.yaml \
  --resume outputs/resnet50-ce-seed42/epoch-005-branch.pt
```

The supported method keys are `ce`, `fcal`, `fcal_v2`, and `ufcal`. `--method` and `--run-id` can override a matrix config without duplicating it.

## Evaluation

The deterministic evaluation suite contains clean images, 13 background perturbations, 13 foreground perturbations, foreground-only, background-only, and a sham recomposition check where supported.

```bash
python -m fcal_study.evaluate outputs/resnet50-fcal-v2-seed42/best-validation.pt
python -m fcal_study.evaluate outputs/resnet50-fcal-v2-seed42/best-validation.pt --variants clean bg_random_rgb_s1 foreground_only
```

Run directories contain the resolved config, environment manifest, history, checkpoints, and evaluation summaries. They are ignored by Git because a complete study is tens of gigabytes. Commit only compact aggregate CSVs and figures under [`results/`](../results/) and [`assets/`](../assets/).

## Reporting discipline

- Select formulas and hyperparameters on validation data before opening the sealed test results.
- Report clean accuracy, regional corruption accuracy, strict leakage, and rescue/harm counts together.
- Treat paired McNemar tests as image-level uncertainty, not training-seed uncertainty.
- Run at least three seeds before making architecture-wide or dataset-wide claims.
