# Class-mean FCAL V2: one-dataset, one-seed result

**Outcome: a useful accuracy/foreground trade-off, but not a replacement for
V2 under the frozen requirements.** The repair has the highest clean and
foreground-only point estimates of these nine checkpoints and narrowly the
highest mean background accuracy. It loses worst-background robustness and
raw-CAM localization versus V2. Four of seven promotion gates pass.

This is the quick screen requested by the user: CUB-200-2011, ResNet-50,
training seed 42, the existing 4,794-image training / 1,200-image validation
partition, and saved fixed/default comparison checkpoints. The development
split has guided earlier experiments, so this is exploratory evidence.
No new test-set evaluation, coefficient tuning, second training seed or other
dataset was run. The formal setup and hashes were [frozen before training](PROTOCOL.md).

## Complete common-evaluator comparison

All accuracy values are percentages. Every model was evaluated on the same
1,200 validation images, all 30 standard views and corruption seed 271828.
BG/FG mean use their respective 13 views; BG worst is the minimum of the
13 background accuracies. Leakage uses the identical raw target-CAM convention
for every model, with lower values favorable. Training times are recorded
post-branch training + validation, excluding the shared first five epochs.

| Method | Clean | BG mean | BG worst | FG mean | FG only | Leakage ↓ | Logged minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| **V2 class-mean repair** | **81.25** | **73.51** | 45.33 | 31.73 | **74.33** | 0.391 | 7.84 |
| CE | 80.92 | 68.36 | 28.08 | **33.76** | 70.33 | 0.561 | 5.83 |
| FCAL V1 | 80.08 | 71.02 | 41.58 | 32.21 | 72.75 | 0.556 | 5.76 |
| FCAL V2 | 80.08 | 72.12 | 49.50 | 31.83 | 71.67 | 0.266 | 5.42 |
| FCAL V2+ | 79.67 | 73.14 | 52.17 | 31.64 | 73.25 | 0.244 | 6.29 |
| CAMAL | 78.92 | 73.47 | **60.08** | 33.02 | 73.50 | 0.316 | 6.23 |
| GAIN-ext | 70.42 | 63.33 | 43.42 | 23.53 | 63.92 | **0.149** | 10.26 |
| RRR | 67.92 | 62.37 | 47.67 | 29.21 | 63.17 | 0.466 | 30.37 |
| Faster-ISNet | 71.92 | 66.31 | 47.67 | 27.01 | 67.58 | 0.539 | 51.18 |

Bold indicates only the best observed point estimate, not statistical
superiority. The repair's mean BG advantage over CAMAL is just **0.038 pp**,
which should be treated as a descriptive near-tie. These are fixed/default
checkpoint comparisons, not the separate tuned arm.

Machine-readable source: [comparison_metrics.csv](comparison_metrics.csv).
It also includes background-only accuracy, peak memory, steady epoch timing
and last-five-epoch clean validation means.

## Paired changes and uncertainty

The intervals below are unadjusted paired image-bootstrap 95% intervals,
conditional on these trained checkpoints. They are not training-seed
uncertainty estimates and do not remove development-split selection bias.
Background resampling preserves all views of an image together. BG-worst
reselects the worst condition for each model in each resample.

| Repair minus V2 | Difference | Paired 95% interval |
|---|---:|---:|
| Clean | +1.17 pp | [−0.58, +2.92] pp |
| BG mean | +1.39 pp | [+0.24, +2.56] pp |
| BG worst | −4.17 pp | [−6.58, −1.67] pp |
| FG only | +2.67 pp | [+0.75, +4.58] pp |
| Raw-CAM leakage | +0.12505, worse | [+0.11472, +0.13537] |

Relative to CE, the repair gains +0.33 pp clean (interval [−1.42, +2.17]),
+5.15 pp BG mean ([+4.01, +6.33]), +17.25 pp BG worst ([+14.75, +19.83])
and +4.00 pp FG only ([+2.00, +6.00]). Its FG-corruption mean is 2.03 pp
below CE, so the robustness gain is not universal across regions.

The exact McNemar analysis contains all **240** candidate-versus-reference
accuracy tests, with one global Holm correction. **None of the 30 comparisons
against V2 survives that correction.** The raw random-RGB p-value is 0.000912
and the global Holm value is 0.135958. Overall 88 comparisons survive across
the entire family, mostly against weaker fixed/default baselines; this count
does not imply superiority over V2. The bootstrap intervals above are not
simultaneous family-corrected intervals.

Sources: [paired_intervals.csv](paired_intervals.csv) and
[paired_accuracy_global_holm.csv](paired_accuracy_global_holm.csv).

## Where the gain and failure occur

The repair improves **11 of 13 background-view point estimates** versus V2.
The largest increase is Gaussian-noise severity 3 (+3.67 pp), followed by
solid-mean replacement (+2.67 pp). Grayscale loses one image (−0.08 pp).

The important failure is **random-RGB background replacement**: 45.33% versus
V2's 49.50%. On exactly the same images, the repair rescues 85 V2 errors but
introduces 135 new errors, a net loss of 50 correct predictions. This explains
how mean background performance improves while the worst condition worsens.
Against CAMAL, the repair loses 14.75 pp on BG worst, despite its higher clean
and foreground-only scores.

The class-mean construction still removes the mathematical common-logit
ambiguity verified in the implementation audit. This run shows that fixing
that ambiguity does not guarantee robustness to a changed background.
Foreground feature locations can still encode background information.
The raw-CAM leakage regression must also be kept visible; a new centered-map
scoring convention was not substituted to hide it. These results do not prove
which internal mechanism caused the observed trade-off.

## Checkpoint, training and cost

Only `fcal_v2.evidence_reference: class_mean` was added to the V2 recipe.
The model resumed the exact CE epoch-5 weights, optimizer, RNG state and parent
50-epoch cosine schedule. The selected checkpoint is **epoch 48**. The
training-loop best is **81.33%**, and the last-five-epoch mean is **80.90%**.
The reloaded checkpoint scores **81.25%** in the common multi-view evaluator;
all table entries use that evaluator, not a mixture of train-loop and reloaded
scores. The one-image numerical discrepancy was not used to reselect it.

Logged post-branch time is **470.357 seconds**, versus archived V2's
**325.170 seconds**, a historical ratio of **1.447×**. The candidate's steady
epoch median is 9.026 seconds, versus 6.137 seconds in the archived run.
These measurements come from different sessions and cannot attribute the
44.7% increase solely to the formula. They nevertheless fail the frozen
historical time screen. A contemporaneous full-pipeline timing control would
be needed to establish actual incremental training overhead.

Peak allocated CUDA memory is **1,875,993,600 bytes** (1.7472 GiB), essentially
the same as V2's 1,875,970,560 bytes. The earlier synthetic same-session
forward/loss/backward benchmark measured about +0.16% time and unchanged
allocation, but excluded loading, optimizer steps and validation. It is not a
substitute for the logged training comparison above.

The candidate retains one original training tensor, one backbone forward,
one ordinary backward, the same model parameters and unchanged inference.
Existing shared crops/flips and training masks remain part of the common
recipe. No numerical failure was logged, and all 50 epochs are present.

## Frozen decision

| Gate | Result |
|---|---|
| Clean no worse than V2 by more than 0.25 pp | Pass |
| BG mean at least 1 pp above V2 | Pass |
| BG worst at least V2 | **Fail** |
| FG only at least V2 | Pass |
| Raw-CAM leakage no worse than V2 | **Fail** |
| Historical logged time at most 1.15× V2 | **Fail; session confounding applies** |
| Peak allocated memory at most 1.10× V2 | Pass |

**Do not promote this configuration as an overall V2 replacement.** Retain it
as a promising accuracy/foreground trade-off and a useful mechanistic result.
It is not dominated by any one comparator on the displayed quality metrics,
but it does not satisfy the original robustness/localization/cost requirements.
No second seed, retuned coefficient, hybrid formula or new dataset was launched
after seeing these results. A subsequent study should be separately specified
and should not convert this screen into a claim of independent confirmation.

The original V2 default is unchanged. Exact decision output:
[decision.json](decision.json). Configuration:
[resnet50_fcal_v2_class_mean_seed42.yaml](../../../configs/cub/resnet50_fcal_v2_class_mean_seed42.yaml).
The candidate checkpoint, comparison checkpoint hashes, source hashes and
split hashes are linked through [protocol.json](protocol.json) and the
per-method evaluation provenance. The evaluation and aggregation commands
are in [PROTOCOL.md](PROTOCOL.md).
