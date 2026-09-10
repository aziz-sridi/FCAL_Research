# Statistical and efficiency protocol

## Experimental unit and uncertainty

A training seed is the experimental unit for claims about training methods.
Average repeated measurements (corruption variants, slices, or images) within a
seed before estimating uncertainty. Report the seed-level mean, sample SD, and
95% Student-t confidence interval. With one seed, report the point estimate and
state that training-run uncertainty is unestimated; do not manufacture an
interval from test images.

Compare methods with the same seeds, data split, architecture, initialization
branch, optimizer budget, and checkpoint-selection rule. Report candidate-minus-
reference deltas, their paired 95% t interval, paired t and sign-randomization
p-values, standardized paired effect (Hedges-corrected *dz*), and win/tie/loss
counts. With only three pairs, intervals and tests have low power and must be
described as such. Image-level McNemar or bootstrap analyses answer a different
question and cannot replace seed replication.
Pre-register a primary outcome; when testing a family of outcomes, report the
raw and Holm-adjusted p-values rather than selecting only favorable metrics.

CAMAL reports 10-fold cross-validation with a Wilcoxon signed-rank test. That is
appropriate for its fold-level design, but folds are resampled data partitions,
not independent training-seed repetitions. This study uses a fixed held-out
split, so its primary method uncertainty is instead estimated from matched
training seeds. Likewise, thousands of images give precise conditional test-set
metrics but do not measure sensitivity to initialization and training order.

Across architectures, first average matched seeds within each architecture and
then count architecture wins, ties, and losses. Define the tie tolerance before
examining results. Do not treat architectures as extra training seeds.

## Fixed defaults and equal tuning

Run two separately labeled comparisons:

1. **Fixed/default:** freeze one FCAL setting and one baseline setting before
   any target test evaluation, and reuse them across datasets/architectures.
2. **Equal-tuned:** give every method the same declared number of validation
   trials and the same scheduler, epoch, seed, and early-stopping budget. Also
   report actual tuning GPU-hours because nominally equal trial counts may have
   unequal costs.

Maintain a comparison manifest with at least `comparison_regime`, `method`, `trial_id`,
`selection_split`, `selected`, `test_evaluated`, and `gpu_hours`. Selection must
label each row `fixed_default` or `equal_tuned`; tuned selection must use
validation only, while a default is labeled `fixed` or `preregistered`. Exactly
one frozen trial per method and regime is evaluated once on
the held-out test; rejected trials must not be test-evaluated. Use nested
validation or a pre-registered rule if the validation set is repeatedly reused.
Never select a method, checkpoint, hyperparameter, threshold, perturbation, or
figure example using test performance.

## Efficiency and Pareto reporting

For every run report total train seconds, validation seconds, wall-clock seconds,
weighted training throughput (`sum(samples) / sum(train_seconds)`), peak CUDA
memory, and GPU-hours. Include the shared warm-up branch in end-to-end cost; a
separate post-branch number may be added but must be labeled. Sum all successful
and failed tuning trials when reporting tuning cost.
Pass the shared branch epoch explicitly to the audit command; a configured
warm-up duration alone is not proof that two runs resumed the same checkpoint.

Show accuracy/robustness/localization versus wall-clock, VRAM, and total tuning
GPU-hours. A setting is Pareto-optimal only when no other setting is at least as
good on every declared objective and strictly better on one. State in advance
which metrics are maximized (for example accuracy) and minimized (leakage,
wall-clock, VRAM, tuning cost).

## Attention alignment and faithfulness

In addition to leakage, report the two evaluation families used by
[CAMAL](https://arxiv.org/abs/2605.08325):

- **Attention IoU (higher is better):** max-normalize each non-negative CAM,
  bilinearly upsample it to mask resolution without changing the mask, threshold
  it at the pre-registered value 0.7, and compute intersection over
  union with the binary segmentation mask. Report the image mean within each
  seed, then the seed-level mean/CI and paired tests. Preserve NaN for empty
  unions and report the valid-image count.
- **Removal AUC (lower is better) and insertion AUC (higher is better):** rank
  pixels by the same target-class CAM. At fixed fractions from 0% to 100%, set
  the most important pixels to the declared baseline for removal, or restore
  them from that baseline for insertion. Record target-class softmax confidence
  and integrate with the trapezoidal rule. Freeze the fraction grid, baseline
  (zero in the CAMAL comparison), target-class convention, and sample-selection
  rule before test evaluation.

For a direct CAMAL comparison, reproduce its threshold and zero-baseline
settings and separately show an all-image evaluation if compute permits. The
representative faithfulness subset must be chosen deterministically without
looking at test outcomes and identically across methods. Existing FCAL summary
archives do not retain spatial CAMs or confidence trajectories, so they cannot
be used to backfill IoU or faithfulness AUC; rerun evaluation and save per-image
values. Image-level paired analyses may supplement, but never replace, the
training-seed analysis.

## Reusable audit commands

```bash
python -m fcal_study.evidence efficiency RUN_DIR [RUN_DIR ...] \
  --branch-epochs 5 --output efficiency.csv
python -m fcal_study.evidence pareto metrics.csv \
  --maximize clean_accuracy --minimize strict_leakage wall_seconds \
  --output pareto.csv
python -m fcal_study.evidence summarize per_seed.csv --groups method \
  --metrics clean_accuracy strict_leakage --output seed_summary.csv
python -m fcal_study.evidence paired per_seed.csv --reference CE \
  --candidates "FCAL V1" "FCAL V2" --metrics clean_accuracy strict_leakage \
  --units seed --lower-is-better strict_leakage --output paired.csv
python -m fcal_study.evidence efficiency-overhead efficiency.csv \
  --reference ce --candidates fcal fcal_v2 --output overhead.csv
python -m fcal_study.evidence win-tie-loss complete_metrics.csv \
  --reference CE --candidate "FCAL V2" --metrics clean_accuracy strict_leakage \
  --lower-is-better strict_leakage --output architecture_wtl.csv
```

The Python API additionally exposes `seed_summary`, `paired_comparison`,
`win_tie_loss`, and `tuning_audit`. All generated tables must retain run IDs,
seeds, and metric direction so that reported claims can be traced to artifacts.
