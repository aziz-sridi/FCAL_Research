# FCAL next-session execution plan

Last updated: **2026-08-30 10:35 +01:00**

This file is the handoff for continuing the FCAL study. The immediate objective is to finish a fair fixed/default-versus-tuned CUB comparison without test-set leakage, then decide whether the evidence supports a CUB-focused paper or justifies the additional experiments needed for a broader "generally useful loss" claim.

## Current state

- Repository: `D:\AI Research\FCAL`
- The fixed/default CUB arm is complete for FCAL V2, RRR, GAIN-ext, and CAMAL on six architectures. Faster-ISNet is audited on ResNet-50 only.
- Fixed-arm aggregation and training-overhead artifacts are under `results/baseline_fixed/cub/`.
- The fixed evidence narrative is `results/baseline_fixed/cub/FIXED_EVIDENCE_SUMMARY.md`.
- Validation-only tuning is complete: **200/200 successful jobs, 0 failures, 21.604 allocated GPU-hours**. Rungs completed at **125/125**, **50/50**, and **25/25**.
- `outputs/tuning-validation/frozen-winners.yaml` is valid and contains exactly 25 validation-selected winners. The manifest was created only after freeze eligibility and a passing full test suite.
- Exactly **25/25** frozen checkpoints received the standard sealed test evaluation, each with all 30 registered variants. Losing trials were not evaluated.
- Tuned aggregation is complete under `results/tuned/cub/`; immutable hashes and recovery notes are in `results/tuned/cub/PROVENANCE.md`.
- Common-map faithfulness is complete: **31/31** fixed validation jobs and **56/56** combined fixed-and-tuned test jobs. Aggregates are under `results/faithfulness/cub/`.
- All eight artifact-driven comparison tables and their hash manifest were regenerated under `paper/generated/`.
- The article was updated to separate fixed and tuned arms, disclose faithfulness counter-results, and narrow the claim to a CUB-focused trade-off result. `paper/fcal_v1_v2_article.pdf` builds cleanly and all nine rendered pages passed visual inspection.
- Final automated checks: the full test suite passed, targeted Ruff passed, `git diff --check` found no whitespace errors, and the claim-language scan found no unsupported positive claim.
- The worktree contains many intentional user/research changes. Do not reset, clean, or discard unrelated files.

## Start here next session

```powershell
Set-Location "D:\AI Research\FCAL"
$env:PYTHONPATH = "src"

python -m fcal_study.tuning status `
  configs/cub/tuning_validation.yaml `
  --ledger outputs/tuning-validation/attempts.jsonl `
  --format text

nvidia-smi
git status --short
```

The completed tuning, frozen evaluation, tuned aggregation, and faithfulness batches must not be rerun or overwritten. Use the status command only as a read-only integrity check. The next decision is scope: either stop with the documented CUB-focused claim or authorize the Phase-7 multi-seed/external-dataset experiments needed for a broader method claim.

## Phase 1 - finish validation-only tuning

- [x] Complete all 125 rung-1 jobs at 10 total epochs.
- [x] Allow the global barrier to select 2/5 candidates per supported cell using validation accuracy only.
- [x] Complete all 50 rung-2 jobs at 25 total epochs.
- [x] Allow the second barrier to select 1/2 candidates per cell using validation accuracy only.
- [x] Complete all 25 rung-3 jobs at 50 total epochs.
- [x] Confirm `200/200` successful jobs or document every terminal failure and its charged GPU time.
- [x] Run the full test suite before freezing.

```powershell
python -m fcal_study.tuning status `
  configs/cub/tuning_validation.yaml `
  --ledger outputs/tuning-validation/attempts.jsonl `
  --format text

python -m fcal_study.tuning account `
  --ledger outputs/tuning-validation/attempts.jsonl

python -m pytest -q
```

Completion gate: status must report `Freeze: eligible=True`. Do not inspect or evaluate tuned checkpoints on the test split before that gate.

## Phase 2 - freeze validation-selected winners

- [x] Create the frozen-winner manifest exactly once.
- [x] Verify it contains 25 supported cells: FCAL V2, RRR, GAIN-ext, and CAMAL on all six architectures, plus Faster-ISNet on ResNet-50 only.
- [x] Preserve Faster-ISNet as `resnet50_only`; never impute unsupported architecture cells.
- [x] Record the manifest and ledger hashes in the final provenance report.

```powershell
python -m fcal_study.tuning freeze `
  configs/cub/tuning_validation.yaml `
  --ledger outputs/tuning-validation/attempts.jsonl `
  --output outputs/tuning-validation/frozen-winners.yaml

python -m fcal_study.tuning status `
  configs/cub/tuning_validation.yaml `
  --ledger outputs/tuning-validation/attempts.jsonl `
  --frozen-winners outputs/tuning-validation/frozen-winners.yaml `
  --format text
```

After freezing, test results may be measured but must never be used to change coefficients, checkpoints, thresholds, subsets, or protocol choices.

## Phase 3 - sealed test evaluation of tuned winners

- [x] Evaluate each of the 25 frozen `best-validation.pt` checkpoints exactly once with `--split test`.
- [x] Require `evaluation/summary.json` beside every selected checkpoint.
- [x] If an evaluation directory is partial or malformed, preserve it for audit and investigate before rerunning; do not silently overwrite it.
- [x] Do not evaluate losing tuning trials.

The evaluator command for each frozen checkpoint is:

```powershell
python -m fcal_study.evaluate "<frozen-checkpoint>" --split test
```

Use a manifest-driven serial loop and check for an existing valid `evaluation/summary.json` before each invocation. Keep the GPU exclusive so measured training and evaluation costs remain interpretable.

## Phase 4 - aggregate tuned performance and search cost

- [x] Generate selected-hyperparameter and tuned-metric tables.
- [x] Include every tuning attempt, including failures, in allocated GPU-hour totals.
- [x] Compare tuned versus fixed performance per method.
- [x] Compute six-architecture FCAL-versus-competitor W/T/L only for FCAL, RRR, GAIN-ext, and CAMAL.
- [x] Report Faster-ISNet only as a direct ResNet-50 comparison.
- [x] Recompute Pareto and efficiency analyses without treating unsupported cells as losses.

```powershell
python -m fcal_study.tuned_results `
  outputs/tuning-validation/frozen-winners.yaml `
  outputs/tuning-validation/attempts.jsonl `
  configs/cub/tuning_validation.yaml `
  results/baseline_fixed/cub/comparison_metrics.csv `
  results/tuned/cub
```

Expected outputs include tuned metrics, selected coefficients, attempt-level accounting, total tuning GPU-hours, fixed-versus-tuned deltas, and eligible W/T/L tables.

## Phase 5 - faithfulness evaluation

The pre-registered common-map protocol uses mask IoU, insertion AUC, and removal AUC on a deterministic 500-image subset. It uses the same ground-truth-class CAM convention for every method; native RRR gradients or ISNet LRP may only be reported as a separate supplementary analysis.

### 5A. Validation smoke/full batch

- [x] Run the fixed-only validation batch first to verify the protocol and memory envelope without opening test data.
- [x] Inspect and aggregate all 31 fixed jobs.

```powershell
python -m fcal_study.faithfulness_batch manifest `
  --split validation `
  --archive-root ../archive/legacy_projects/adaptive_fcal_study/outputs `
  --fixed-root outputs/baseline-fixed/cub `
  --artifact-root outputs/faithfulness/cub `
  --output outputs/faithfulness/cub/manifests/fixed-validation.json `
  --max-images 500 --selection-seed 2026 --batch-size 16

python -m fcal_study.faithfulness_batch run-all `
  outputs/faithfulness/cub/manifests/fixed-validation.json

python -m fcal_study.faithfulness_batch aggregate `
  outputs/faithfulness/cub/manifests/fixed-validation.json `
  --output-dir results/faithfulness/cub/fixed-validation
```

### 5B. Frozen combined test batch

- [x] Create the combined 56-job fixed-and-tuned test manifest only after winners are frozen.
- [x] Inspect the queue before execution.
- [x] Run the sealed test batch serially and aggregate it exactly once.

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

## Phase 6 - statistical and efficiency evidence

- [ ] Report exact per-architecture values rather than only averages.
- [ ] Report W/T/L consistency across the six architectures; do not call these counts seed-level significance.
- [ ] Use paired tests, effect sizes, confidence intervals, and Holm correction where repeated-seed pairing exists.
- [ ] Keep the existing ResNet-50 three-seed result qualified: with `n=3`, directional agreement alone is not strong significance evidence.
- [ ] Derive post-warm-up epoch-time distributions from histories and report median/dispersion, not only one total wall time.
- [ ] Report total training time, post-branch time, peak allocated VRAM, and the complete tuning-search GPU-hours.
- [ ] Verify that FCAL adds no inference parameters or inference-time computation; measure inference throughput with repeated controlled runs if the manuscript makes a zero-overhead claim.
- [ ] Preserve and disclose the interrupted RRR EfficientNet and Faster-ISNet attempts separately from valid runtime totals.

## Phase 7 - experiments needed for broader claims

These are required if the paper claims that FCAL is a generally useful or broadly superior loss, rather than a strong CUB result.

### Priority A: multi-seed confirmation

- [ ] Run at least three seeds on ResNet-50 for CE, FCAL V2, RRR, GAIN-ext, CAMAL, and Faster-ISNet.
- [ ] Hold tuned coefficients fixed after seed-42 selection; do not retune against the new test results.
- [ ] Prefer five seeds if compute permits, especially for clean accuracy and leakage claims.
- [ ] Report paired seed-level confidence intervals, effect sizes, corrected tests, and failure rates.

### Priority B: external datasets

- [ ] Repeat the matched fixed/default comparison on at least one additional dataset with masks; Oxford-IIIT Pet is the minimum practical extension.
- [ ] Ideally add ISIC as a second domain-shifted dataset.
- [ ] Use the same fairness structure: fixed/drop-in arm first, then equal-dimensional validation-only tuning where feasible.
- [ ] Keep dataset-specific claims separate if a baseline or architecture is unsupported.

### Priority C: supervision robustness and usability

- [ ] Test incomplete mask availability, such as 10%, 25%, 50%, and 100% annotated training samples.
- [ ] Test mask corruption: boundary noise/dilation/erosion and missing foreground regions.
- [ ] Quantify FCAL coefficient sensitivity using the completed tuning grid.
- [ ] Report annotation requirements as a limitation; FCAL is not supervision-free.

If these extensions are not completed, narrow the paper to a CUB-focused architecture and tuning study and list multi-dataset/multi-seed generalization as future work.

## Phase 8 - manuscript and repository release

- [x] Regenerate the eight artifact-driven LaTeX comparison tables.
- [x] Update `docs/TUNING_PROTOCOL.md`; its opening currently says no tuning jobs have run and will be stale after completion.
- [x] Update the root README, result READMEs, reproducibility documentation, and the manuscript from machine-readable outputs.
- [x] Include fixed/default and tuned arms as separate primary results.
- [x] Include search cost as well as final-model training cost.
- [x] Describe Faster-ISNet as ResNet-50-only everywhere.
- [x] Cite and distinguish CAMAL explicitly; do not claim novelty as merely "adding a mask-based CAM loss."
- [ ] Build manuscript assets only after tuned aggregation succeeds:

```powershell
python -m fcal_study.build_v1_v2_article_assets --tables-only
```

- [x] Rebuild the LaTeX article and bibliography.
- [x] Render the final PDF to page images and visually inspect every page for clipped tables, broken references, unreadable figures, and overflow.
- [x] Run the complete tests and targeted Ruff checks.
- [x] Verify every table number against its source CSV and record hashes/provenance.
- [x] Scan the manuscript and READMEs for unsupported words such as "always," "universally," "state of the art," or unqualified "best."

## Claim rules

The fixed arm already shows that FCAL does **not** win every metric: CAMAL and GAIN-ext often obtain lower strict leakage. Therefore, do not design the analysis around proving that FCAL "always wins."

Acceptable claim if the tuned results remain favorable:

> Across the evaluated CUB architectures, FCAL provides a consistently favorable accuracy-robustness-efficiency trade-off and is competitive as a low-tuning, drop-in training objective.

Stronger wording such as "generally useful," "best overall," or "state of the art" requires multi-seed and external-dataset evidence. Any "best" statement must name the exact metric, dataset, architecture support set, tuning arm, and uncertainty basis.

## Final completion checklist

- [x] Tuning status is complete and freeze-eligible.
- [x] Frozen manifest is valid and immutable.
- [x] Exactly 25 tuned winners received sealed standard test evaluation.
- [x] Fixed/tuned faithfulness artifacts are complete and verified.
- [x] Fixed, tuned, statistical, Pareto, and overhead tables reproduce from scripts.
- [x] Multi-seed and external-dataset scope is either completed or explicitly limited.
- [x] README and manuscript claims match the evidence, including counter-results.
- [x] PDF builds cleanly and passes visual inspection.
- [x] Full test suite and lint checks pass.
- [x] Reproduction commands, environment, hardware, seeds, failures, and artifact hashes are documented.
