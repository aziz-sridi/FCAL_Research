# Equal-budget validation-only tuning protocol

This arm compares FCAL V2, RRR, GAIN-ext, CAMAL, and Selective Faster-ISNet without using the CUB test set for model or hyperparameter selection. The validation-only search is complete: all 200 planned jobs succeeded, the 25 winners were frozen before the test split was opened, and exactly those 25 checkpoints received the standard sealed test evaluation. Machine-readable results are under `results/tuned/cub/`; immutable hashes and execution totals are recorded in `results/tuned/cub/PROVENANCE.md`.

The executable manifest is [`configs/cub/tuning_validation.yaml`](../configs/cub/tuning_validation.yaml). The runner is `python -m fcal_study.tuning`.

## Locked design

- Start only after the corresponding fixed-recipe baseline has produced its training `summary.json`. The runner checks that prerequisite without opening any evaluation directory.
- Use the same frozen seed-42 CE checkpoint at completed epoch 5 for every trial in an architecture.
- Use five initial trials for every supported method–architecture cell, one sequential GPU job at a time.
- Use total-epoch fidelities 10, 25, and 50, retaining 5→2→1 candidates. Every promoted fidelity restarts from the same epoch-5 parent. Restarting is deliberate: it gives every candidate the same full-fidelity cosine schedule instead of inheriting a scheduler whose horizon was set by a shorter rung.
- Select only by the trainer's `best_validation_accuracy`; ties go to the lower predeclared trial index. The selected artifact at every fidelity is `best-validation.pt`.
- Keep seed, split, epoch budget, optimizer, augmentation, model initialization, and checkpoint rule fixed. Only the named method coefficient changes.
- Permit two attempts per job. A failed attempt consumes its measured allocation and is never silently erased.

Each launched attempt records start/end UTC, elapsed seconds, exit code, failure text, config hash, and allocated GPU-hours (`elapsed wall-clock hours × one exclusively allocated GPU`). The accounting command sums successful and failed allocations. This is GPU-time accounting, not an invented estimate of hardware utilization.

## Coefficient grids and provenance

The grid changes one coefficient per method so the five trials have the same search dimensionality. Other method settings stay at the fixed recipe.

| Method | Tuned coefficient | Five predeclared values | Basis |
|---|---|---|---|
| FCAL V2 | `fcal_v2.support_weight` | 0.0125, 0.025, 0.05, 0.1, 0.2 | Two powers of two around 0.05. The current viability term is bounded by RMS normalization, so this is the coefficient in the repository's actual loss scale. |
| RRR | `rrr.lambda` | 1e-5, 3.16228e-5, 1e-4, 3.16228e-4, 1e-3 | Log grid around the current exact-loss fixed recipe. Ross et al. advise balancing classification and explanation terms; their reported MLP coefficient must not be copied numerically into a 224×224 pretrained model with different gradient scaling. |
| GAIN-ext | `gain.supervision_weight` | 1, 3.16228, 10, 31.6228, 100 | The paper uses external-supervision weight 10; the Faster-ISNet comparison reports 100 for dog-breed bias mitigation. Mining and classification weights remain 1. |
| CAMAL | `camal.coefficient` | 0.1, 0.316228, 1, 3.16228, 10 | One decade around the paper's coefficient 1, applied to this repository's normalized in/out-mask CAM difference. |
| Selective Faster-ISNet | `isnet.heatmap_weight` | 0.5, 0.7, 0.8, 0.9, 0.95 | This repository implements the paper's convex `P`: `(1-P) CE + P L_heatmap`. The fixed value is 0.7; Faster-ISNet applications use values in the 0.8–0.95 region. |

Primary references: [successive halving/Hyperband](https://www.jmlr.org/papers/v18/16-558.html), [RRR paper](https://arxiv.org/abs/1703.03717) and [code](https://github.com/dtak/rrr), [GAIN-ext paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Tell_Me_Where_CVPR_2018_paper.html), [CAMAL paper](https://arxiv.org/abs/2605.08325), and [Faster-ISNet paper](https://arxiv.org/abs/2401.08409) and [code](https://github.com/PedroRASB/FasterISNet).

## Test seal

Training uses `train` and `validation` loaders only. Generated configs remove `data.evaluation_cache`, because the existing CUB cache is test-specific. The training program hashes prepared split artifacts for provenance, including the sealed test partition's file hash, but it never loads test examples or metrics. Hashing split identity is not evaluation and reveals no labels or scores.

The runner never invokes `fcal_study.evaluate`, rejects test-named output namespaces and test caches, and will not create a frozen-winner manifest until every supported method–architecture cell has one successful 50-epoch winner. Do not inspect, aggregate, or use any existing test evaluation while the arm is active. After freezing, evaluate exactly that one checkpoint per supported cell once; do not return to tuning based on test outcomes.

## Support-aware comparisons

Selective Faster-ISNet is labeled only for the audited ResNet-50 LRP-Flex adapter. It is not run, imputed, or relabeled for DenseNet, EfficientNet, or ViT.

Report two strata after the test seal is legitimately opened:

1. A five-method ResNet-50 comparison: FCAL V2, RRR, GAIN-ext, CAMAL, and Selective Faster-ISNet.
2. A four-method six-architecture matrix: FCAL V2, RRR, GAIN-ext, and CAMAL. ISNet cells are “unsupported,” not missing-at-random and not losses.

Cross-architecture claims must use the four-method stratum. ISNet claims must remain ResNet-50-specific.

## Queue inspection and reproduction

Validate the protocol and inspect any jobs that remain for a fresh or interrupted ledger:

```powershell
python -m fcal_study.tuning validate
python -m fcal_study.tuning queue --format jsonl
```

Materialize the current rung's resolved configs without training:

```powershell
python -m fcal_study.tuning materialize --directory tmp/tuning-configs
```

Every queue entry is a complete command containing `--execute`. Review prerequisites and run one entry at a time. The runner enforces a global rung barrier, so no 25-epoch promotion appears until all supported 10-epoch jobs are terminal. It similarly blocks the 50-epoch rung until the 25-epoch rung is terminal.

Audit accumulated cost, including failures:

```powershell
python -m fcal_study.tuning account
```

Only after all final-rung jobs succeed, freeze the validation winners:

```powershell
python -m fcal_study.tuning freeze --output outputs/tuning-validation/frozen-winners.yaml
```

`freeze` records validation selections only. It does not evaluate the test set.

## Completed execution record

- Supported cells: 25 (four methods on six architectures, plus Selective Faster-ISNet on ResNet-50 only).
- Successful jobs: 200/200; terminal failures: 0.
- Rung totals: 125/125 at 10 epochs, 50/50 at 25 epochs, and 25/25 at 50 epochs.
- Allocated GPU time: 21.604 hours, including every attempt.
- Frozen winners: 25; selection split: validation; `test_set_used: false` at freeze time.
- Sealed standard evaluations: 25/25, each containing all 30 registered variants.
- Common-map faithfulness: 31/31 fixed validation jobs and 56/56 combined fixed-and-tuned test jobs completed and verified.

The tuned arm is descriptive single-seed CUB evidence. It does not establish a generally superior loss: tuning improved RRR broadly, produced mixed changes for CAMAL and GAIN-ext, and reduced FCAL's fixed-arm clean/BG/foreground-only values on five of six architectures. The common-map faithfulness results also retain complementary CAMAL and GAIN-ext advantages. Broader ranking claims still require multi-seed prior-method comparisons and external-dataset replication.
