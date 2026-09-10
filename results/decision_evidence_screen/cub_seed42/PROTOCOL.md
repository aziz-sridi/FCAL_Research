# Class-mean V2: CUB / ResNet-50 / seed 42 exploratory screen

Frozen before candidate training on 2026-09-10. The source of truth is
`protocol.json` (SHA-256
`0bcfd6edb6f0dce12df65200955fd19b15c06b7ba185b2dc403ee9ec25a6ff77`).

The user requested one dataset and one seed, comparing the proposed repair
with CE and other losses. This first inexpensive screen uses the established
CUB development split and archived baselines. It is exploratory: the split
has guided many previous formulas. It is not the fresh-protocol confirmation
recommended in the research audit. The official test images are not evaluated.

The candidate resumes the archived seed-42 CE epoch-5 branch, with model,
loader settings and split hashes verified by the training relocation guard.
The original 50-epoch parent cosine schedule, optimizer moments, RNG state,
batch order, common preprocessing, masks and V2 coefficients are retained.
The sole objective change is `fcal_v2.evidence_reference: class_mean`.
The training tensor is not altered by FCAL; one ordinary backbone forward
and backward are used. There is no additional model or inference computation.

The eight frozen comparison checkpoints are CE, FCAL V1, FCAL V2, FCAL V2+,
CAMAL, GAIN-ext, RRR and Faster-ISNet. All are ResNet-50, training seed 42,
50 total epochs, with matching model/data settings. Method-specific numerical
settings are retained: Faster-ISNet uses FP32; the others use BF16. GAIN-ext
is retained as an external comparator with its own additional-forward method,
not as a candidate satisfying FCAL's one-forward constraint.

All nine checkpoints receive the same 30-view validation evaluation with
corruption seed 271828, independently of training seed 42. Every evaluation
is written into a new directory with checkpoint/protocol hashes; no historical
prediction arrays or summaries are overwritten. The candidate is selected
only by strictly improving clean validation accuracy. Full evaluation proceeds
even if its clean score fails, to provide the requested side-by-side picture.
No coefficient adjustment or second candidate is permitted within this screen.

All reported leakage uses the original raw target-class CAM convention for
every method. The centered training map is not substituted for the candidate
alone during evaluation. Background and foreground corruptions are evaluation
stress tests; none are passed into candidate training.

Promotion requires all of the following relative to the archived V2:

| Metric | Frozen screen threshold |
|---|---|
| Clean accuracy | Difference at least −0.25 percentage points |
| Mean accuracy across 13 background views | Difference at least +1.00 point |
| Worst background accuracy | No regression |
| Foreground-only accuracy | No regression |
| Raw-CAM strict leakage | No increase |
| Recorded post-branch training + validation time | At most 1.15× V2 |
| Recorded peak allocated memory | At most 1.10× V2 |

Historical timing is not a controlled contemporary overhead estimate; machine
state can differ across sessions. Report this limitation along with the prior
paired synthetic microbenchmark. A pass means eligibility for independent
replication, never a SOTA or population-significance claim.

Paired percentile-bootstrap intervals use 10,000 image resamples and seed
20260910. For BG mean/worst, an image's 13 views stay together; the worst
condition is reselected within each bootstrap sample for each method.
Exact McNemar tests cover candidate versus eight references on all 30 views
(240 tests), with one global Holm correction. These are image-level,
single-training-seed exploratory intervals/tests; they neither measure
training-seed uncertainty nor undo adaptive reuse of the development split.

Reproduction from the FCAL repository root:

```powershell
python -m fcal_study.train configs/cub/resnet50_fcal_v2_class_mean_seed42.yaml --resume ../archive/legacy_projects/adaptive_fcal_study/outputs/resnet50-ce-seed42/epoch-005-branch.pt --allow-data-relocation
python -m fcal_study.decision_evidence_screen evaluate results/decision_evidence_screen/cub_seed42/protocol.json
python -m fcal_study.decision_evidence_screen report results/decision_evidence_screen/cub_seed42/protocol.json
```

Do not rerun training into the existing completed output directory. Use the
saved checkpoints and evaluation provenance when reproducing the tables.
