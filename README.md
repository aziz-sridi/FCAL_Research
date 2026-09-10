# FCAL — Foreground-Constrained Attention Learning

> Teach image classifiers not only **what** to predict, but **where to look**.

FCAL is a training objective for CNNs and Vision Transformers that moves
class-specific evidence from background shortcuts onto the annotated object.
It changes training only: deployed models keep the same parameters, latency,
and prediction API.

**3 datasets · 6 architectures · complete 30-view tests · no inference overhead**

## Paper

The current 13-page manuscript presents FCAL as a single fixed training method
and reports its replicated evidence, matched baselines, computational cost,
architecture effects, and transfer limits.

**[Read the paper (PDF)](paper/FCAL_manuscript.pdf)**

This is a research manuscript, with the strongest replicated evidence on CUB
ResNet-50. It is not a claim of journal acceptance or clinical effectiveness.

## Improvements at a glance

The strongest result is now replicated across three independent CUB ResNet-50
training seeds. Values are mean ± sample SD; every FCAL run resumes its paired
CE run's exact epoch-5 branch.

| CUB ResNet-50 test metric | CE | FCAL V2 | Paired V2−CE change |
|---|---:|---:|---:|
| Clean accuracy | 77.44 ± 0.13% | **78.49 ± 0.16%** | **+1.05 ± 0.19 pp** |
| Mean of 13 background distortions | 68.75 ± 1.70% | **72.40 ± 0.38%** | **+3.65 ± 1.37 pp** |
| Worst background distortion | 38.90 ± 9.81% | **51.67 ± 4.00%** | **+12.77 ± 6.63 pp** |
| Foreground only | 68.63 ± 0.34% | **72.51 ± 0.87%** | **+3.87 ± 0.57 pp** |
| Strict leakage ↓ | 0.571 ± 0.003 | **0.281 ± 0.003** | **50.8% lower at the means** |

The completed six-architecture matrices keep all regional tests visible:

| Dataset, V2 vs CE | Clean | BG mean / worst | FG mean / only | Background only | Leakage |
|---|---:|---:|---:|---:|---:|
| CUB | +0.57 pp | +1.21 / +4.65 pp | -0.36 / +0.93 pp | -0.75 pp | 30.9% lower |
| Oxford Pet | +0.50 pp | +0.67 / +1.30 pp | +0.61 / +2.36 pp | -0.49 pp | 34.6% lower |
| ISIC subset | -1.67 pp | +0.17 / -1.00 pp | -1.24 / +3.67 pp | -1.67 pp | 41.6% lower |

The main matrices use one seed; only CUB ResNet-50 has three. ISIC is a
100-image, non-clinical subset and remains a negative classification result.

### Fixed and validation-tuned explanation-supervision baselines

The matched CUB comparison is complete for FCAL V2, RRR, GAIN-ext, and CAMAL
on all six architectures; Selective Faster-ISNet is audited on ResNet-50 only.
In the fixed/default arm, FCAL has higher clean accuracy than CAMAL and RRR on
all six architectures and than GAIN-ext on five, while CAMAL and GAIN-ext often
have lower strict leakage. The separate equal-dimensional validation-only
search completed 200/200 jobs with zero failures and consumed 21.604 allocated
GPU-hours before 25 winners were frozen and evaluated once on test.

The tuned six-architecture means are:

| Method | Clean | BG mean | BG worst | Foreground only | Strict leakage |
|---|---:|---:|---:|---:|---:|
| **FCAL V2** | **79.92%** | **73.19%** | 52.50% | **72.98%** | 0.466 |
| CAMAL | 79.33% | 73.16% | 56.96% | 71.28% | 0.437 |
| GAIN-ext | 76.06% | 68.29% | 45.38% | 68.14% | **0.303** |
| RRR | 76.66% | 72.48% | **60.61%** | 72.38% | 0.672 |

Tuning is not uniformly beneficial: FCAL's tuned setting is lower than its
fixed setting on clean, BG mean/worst, and foreground-only accuracy on five of
six architectures, whereas RRR improves broadly. On the common 500-image CAM
faithfulness subset, CAMAL leads mean mask IoU and GAIN-ext leads mean removal
AUC; FCAL therefore supports an accuracy--robustness--efficiency trade-off
claim, not a universal localization win. [Fixed evidence](results/baseline_fixed/cub/FIXED_EVIDENCE_SUMMARY.md),
[tuned provenance](results/tuned/cub/PROVENANCE.md), and
[faithfulness report](results/faithfulness/cub/fixed-and-tuned-test/REPORT.md).

## The problem in two pictures

<table>
  <tr>
    <td width="50%"><img src="assets/meme1.png" alt="A classifier confidently points in the wrong direction while naming a bird"></td>
    <td width="50%"><img src="assets/meme2.png" alt="A second person redirects the classifier toward the actual bird"></td>
  </tr>
  <tr>
    <td align="center"><strong>Cross-entropy:</strong> a correct label can still come from the wrong evidence.</td>
    <td align="center"><strong>FCAL:</strong> the mask teaches the model where class evidence should live.</td>
  </tr>
</table>

A standard classifier is rewarded for the right answer, even if it uses grass,
clinic artifacts, or scene context. FCAL adds spatial supervision to the normal
classification loss during training. At inference, the mask and auxiliary loss
disappear.

## What FCAL changes visually

![Same image with CE and FCAL V2 predicted-class CAMs across birds, skin lesions, and pets](assets/result_showcases/overview.png)

Green means the class is correct. The white contour is the annotated
foreground. `inside` is the positive predicted-class CAM mass inside it. All
CAMs use `jet`: blue is low evidence and red is high evidence.

## When better focus changes the answer

These held-out examples are classification rescues: CE predicts the wrong
class, while FCAL V2 predicts the correct one from evidence concentrated on the
foreground.

![Three held-out examples where CE is wrong and FCAL V2 is correct](assets/result_showcases/rescue_overview.png)

These examples explain the mechanism; the aggregate claims above come from
every test image, not from the selected gallery.

## Background-distortion stress tests

Here the bird pixels are preserved exactly while only the background is
changed. The suite includes blur, color jitter, Gaussian noise, grayscale,
solid-mean replacement, random-solid color, and random RGB noise. Each row is a
held-out rescue where the distorted background makes CE predict the wrong bird
but FCAL V2 remains correct.

### ResNet-50 CNN

![ResNet-50 background-distortion tests where CE is wrong and FCAL V2 is correct](assets/result_showcases/cub/background_distortion_cnn_rescues.png)

### ViT-Base/16

![ViT-Base background-distortion tests where CE is wrong and FCAL V2 is correct](assets/result_showcases/cub/background_distortion_vit_rescues.png)

These montages are selected qualitative rescues from the seed-42 pair. For
that pair, the corresponding full evaluation gives **+5.19 pp mean** and
**+20.38 pp worst-case** background-intervention improvements over all 5,794
test images and all 13 background variants. The headline table reports the
more conservative mean across three training seeds.

### Dataset-tailored background tests

The common 30-view suite is supplemented by controlled scenarios that match
each dataset. The untouched region is copied exactly.

| Six-model mean scenario | CE | FCAL V2 | Result |
|---|---:|---:|---|
| CUB twilight background | 76.36% | **77.12%** | +0.76 pp |
| CUB camouflage background | 72.03% | **73.70%** | +1.66 pp |
| Pet warm-light background | 89.19% | **89.80%** | +0.61 pp |
| Pet checkerboard background | 87.23% | **87.73%** | +0.50 pp |
| ISIC hair outside lesion | **56.83%** | 56.33% | -0.50 pp |
| ISIC ruler outside lesion | **63.17%** | 62.17% | -1.00 pp |

These are deterministic simulations, not natural-shift or clinical claims.
Their mixed outcomes are intentional evidence: V2 is stronger on the CUB/Pet
background tests but does not solve ISIC acquisition-artifact sensitivity.
[Read the exact scenario definitions](docs/STRESS_TESTS.md).

## Results that matter

| Dataset | Main result | What to conclude |
|---|---|---|
| [CUB-200-2011](results/cub/README.md) | Three-seed ResNet-50 V2: **+1.05 ± 0.19 pp clean**, **+3.65 ± 1.37 pp BG mean**, **+12.77 ± 6.63 pp BG worst**, **+3.87 ± 0.57 pp foreground-only** | Strongest replicated accuracy and background-robustness result |
| [Oxford-IIIT Pet](results/oxford_pet/README.md) | Six-model V2 mean: **+0.50 pp clean**, **+0.67 pp BG mean**, **+1.30 pp BG worst**, **+0.61 pp FG mean**, leakage **0.292 → 0.191** | Broadest one-seed benchmark |
| [ISIC](results/isic/README.md) | Leakage **0.441 → 0.257**, but clean **-1.67 pp**, BG worst **-1.00 pp**, and hair/ruler scenarios do not improve | Negative classification result; not clinical evidence |
| [Universal FCAL](results/universal/README.md) | Pet mean clean **+0.42 pp** and best mean foreground-corruption score; ISIC pilot is mixed | Working architecture-independent controller, still experimental |

The qualitative galleries are backed by frozen manifests and real checkpoint
inference. Each dataset page shows one test image per model; the executed
notebooks show three per model:

- [Bird test notebook](notebooks/cub_test_showcase.ipynb)
- [Skin-lesion test notebook](notebooks/isic_test_showcase.ipynb)
- [Pet test notebook](notebooks/oxford_pet_test_showcase.ipynb)

## Three versions, one research direction

| Method | Idea | Current role |
|---|---|---|
| **FCAL V1** | Penalize normalized positive class evidence outside a tolerance-expanded foreground | Original, simple containment baseline |
| **FCAL V2** | Add scale-normalized, distributed foreground support and resolution-aware tolerance | Main supported method; called V5 during internal ablations |
| **Universal FCAL** | Calibrate targets automatically, budget auxiliary gradients, project conflicts, and schedule the objective from validation behavior | Architecture-independent research implementation |

CNN galleries use exact final-feature class activation maps. The ViTs use
average-pooled patch tokens, so their class-specific patch CAMs decompose the
actual class score instead of substituting a generic attention visualization.

### What the corrected component ablation shows

A provenance audit found that the old Fixed-3 row could not establish whether
`3` meant input pixels or evidence-map cells. The ablation has now been fully
regenerated from an archived seed-42 epoch-5 branch with explicit
`fixed_tolerance_cells: 3` semantics and the original 50-epoch parent cosine
schedule.

The corrected result strengthens the V2 design argument. Relative to literal
three-cell dilation, full V2 improves mean background accuracy by 1.08 pp,
worst-background accuracy by 8.25 pp, foreground-corruption mean by 0.77 pp,
and strict leakage by 0.072 at essentially identical clean accuracy. Against
tolerance-only, signed support adds 8.63 pp to worst-background accuracy for a
0.64 pp clean trade-off. [Full corrected table and run IDs](results/cub/RESULTS.md#v2-component-ablation).

## Why the comparison is fair

- Each FCAL run resumes from its paired CE model's exact epoch-5 checkpoint.
- Architecture-specific data, initialization, optimizer, schedule, and
  50-epoch budget remain matched.
- Test examples, foreground masks, and deterministic corruptions are identical
  across methods.
- Classification and localization are reported separately; failures are not
  hidden behind a single score.
- Full tables and machine-readable CSVs are committed under [`results/`](results/).

The study spans ResNet-50, DenseNet-121, EfficientNet-B0, ViT-Tiny/16,
ViT-Small/16, and ViT-Base/16 on birds, skin lesions, and pet breeds.

## Repository map

```text
FCAL/
├── assets/                  README figures and frozen CAM galleries
├── configs/                 matched CUB, ISIC, and Oxford Pet experiments
├── docs/                    methods and reproduction notes
├── notebooks/               executed visual test audits
├── paper/                   current research paper PDF
├── results/                 dataset landing pages, full tables, and CSVs
├── scripts/                 experiment and showcase automation
├── splits/                  frozen train/validation/test manifests
├── src/
│   ├── fcal_study/          data, CAM adapters, training, and evaluation
│   └── universal_fcal/      standalone architecture-independent objective
└── tests/
```

Datasets live in the workspace-level [`../datasets/`](../datasets/) folder.
Checkpoints, caches, and full prediction arrays are intentionally excluded from
Git.

## Quick start

Python 3.10+ and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

The reported environment is pinned in [`requirements-lock.txt`](requirements-lock.txt).
Install the PyTorch build matching your CUDA runtime before the remaining
packages when necessary.

### Train a matched CUB ResNet-50 comparison

```bash
python -m fcal_study.train configs/cub/resnet50_ce.yaml
python -m fcal_study.train configs/cub/resnet50_fcal.yaml \
  --resume outputs/resnet50-ce-seed42/epoch-005-branch.pt
python -m fcal_study.train configs/cub/resnet50_fcal_v2.yaml \
  --resume outputs/resnet50-ce-seed42/epoch-005-branch.pt
```

Evaluate the selected checkpoint:

```bash
python -m fcal_study.evaluate outputs/resnet50-fcal-v2-seed42/best-validation.pt
```

Dataset setup, the experiment configs, and the fairness protocol are in
[Reproducing the study](docs/REPRODUCIBILITY.md). The objective equations and
design decisions are in [Methods](docs/METHODS.md).

## Universal FCAL as a library

```python
from universal_fcal import UniversalFCAL, budgeted_objective

objective = UniversalFCAL(
    allowed_outside=0.20,
    target_coverage=0.30,
    tolerance_cell_fraction=0.5,
)

result = objective(model.raw_class_evidence(labels), masks, logits, labels)
total, weighted_fcal, weight = budgeted_objective(
    ce_loss,
    result.auxiliary,
    phase=1.0,
    target_loss_ratio=0.05,
    lambda_max=1000.0,
)
```

The model adapter only needs to return one or more signed evidence maps shaped
`[batch, height, width]`.

## Research scope and next steps

- CUB ResNet-50 now uses three training seeds; the remaining architecture and
  dataset matrix still uses one. Image-level paired tests do not replace
  training-seed uncertainty.
- RRR, GAIN-ext, and Selective Faster ISNet/LRP-Flex now have matched seed-7
  runs. Expanding these comparisons to multiple seeds and architectures is the
  next statistical step.
- ISNet requires FP32 because LRP-Flex creates and differentiates gradients
  inside the forward pass. The original failure was a YAML/autocast and
  resume-backend bug, not a ResNet-50 incompatibility; the corrected run keeps
  non-finite gradients as a hard error and uses no gradient sanitization.
- CAMAL is a directly relevant 2026 attention-supervision method and is now
  included in the matched fixed and validation-tuned comparisons. Replicating
  the architecture-level method comparisons across more training seeds is the
  next statistical step.
- The ISIC sealed test has 100 images and very small rare classes. These results
  do not establish clinical effectiveness.
- Better CAM localization does not guarantee better classification; ISIC and
  the smaller ViTs make that boundary explicit.
- Universal FCAL is functional and fully benchmarked on Oxford-IIIT Pet, but
  its automatic controller remains experimental.

The support-aware objective was called **V5** while intermediate ablations were
still active. It is the second supported public method, so this repository calls
it **FCAL V2** everywhere users encounter it.
