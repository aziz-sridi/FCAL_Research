# FCAL — Foreground-Constrained Attention Learning

<table>
  <tr>
    <td width="50%"><img src="assets/meme1.png" alt="A classifier points at the landscape instead of the bird while declaring that it sees a bird"></td>
    <td width="50%"><img src="assets/meme2.png" alt="Foreground supervision redirects the classifier toward the actual bird"></td>
  </tr>
  <tr>
    <td align="center"><strong>The problem:</strong> the right label, the wrong evidence.</td>
    <td align="center"><strong>FCAL:</strong> teach the model where to look.</td>
  </tr>
</table>

A classifier can recognize a bird by its background. **FCAL adds foreground
supervision to the classification loss**, encouraging class evidence to stay
on the object. It works with CNNs and Vision Transformers; foreground masks
are needed only during training. Inference keeps the same model parameters,
latency, and prediction API.

**[Paper (PDF)](paper/FCAL_manuscript.pdf)** ·
[Results](results/README.md) ·
[Method](docs/METHODS.md) ·
[Reproduce](docs/REPRODUCIBILITY.md)

## Strongest result

**CUB-200-2011 · ResNet-50 · three paired training seeds · 5,794 test images.**
FCAL V2 improves clean accuracy and background robustness while reducing
class evidence outside the bird.

| Test metric | Cross-entropy (CE) | FCAL V2 | Paired change |
|---|---:|---:|---:|
| Clean accuracy | 77.44 ± 0.13% | **78.49 ± 0.16%** | **+1.05 ± 0.19 pp** |
| Mean accuracy over 13 background distortions | 68.75 ± 1.70% | **72.40 ± 0.38%** | **+3.65 ± 1.37 pp** |
| Worst background-distortion accuracy | 38.90 ± 9.81% | **51.67 ± 4.00%** | **+12.77 ± 6.63 pp** |
| Foreground-only accuracy | 68.63 ± 0.34% | **72.51 ± 0.87%** | **+3.87 ± 0.57 pp** |
| Strict leakage ↓ | 0.571 ± 0.003 | **0.281 ± 0.003** | **50.8% lower at the means** |

Values are mean ± sample standard deviation across seeds 7, 42, and 123 on
the same fixed split; `pp` means percentage points. Each FCAL run resumes its
paired CE run's exact epoch-5 checkpoint, with a matched 50-epoch budget.
Background tests change only the background, preserving foreground pixels.

[Full CUB results and trade-offs](results/cub/README.md) ·
[Per-seed metrics](results/cub/resnet50_three_seed_metrics.csv)

## What changes visually

![Matched CE and FCAL V2 class activation maps on birds, skin lesions, and pets, showing more evidence within the annotated foreground](assets/result_showcases/overview.png)

Each row compares the same held-out image. Both predictions are correct;
the difference is where their class activation maps (CAMs) place evidence.
Red indicates high evidence, blue low evidence, and the white contour marks
the foreground. `inside` is the fraction of positive predicted-class CAM
mass within that contour. These are selected illustrations; the headline
metrics use the complete test set.

[Cases where FCAL corrects a CE error](assets/result_showcases/rescue_overview.png) ·
[CNN background stress tests](assets/result_showcases/cub/background_distortion_cnn_rescues.png) ·
[ViT background stress tests](assets/result_showcases/cub/background_distortion_vit_rescues.png)

## How it works

FCAL V2 adds two terms to cross-entropy:

1. **Containment:** penalize positive class evidence outside a foreground
   region with resolution-aware tolerance.
2. **Foreground support:** encourage distributed, scale-normalized signed
   evidence within the object.

The mask and auxiliary objective are removed at inference. CNNs use
class-specific feature CAMs; the ViTs use class-specific patch evidence
from average-pooled tokens. [Equations and design details](docs/METHODS.md).

**FCAL V2 is the main supported method.** The repository also retains
FCAL V1, the original containment objective, and
[Universal FCAL](results/universal/README.md), an experimental adaptive
controller. V2 was called V5 during internal ablations.

## Evidence and limits

The main benchmark covers three datasets and six architectures: ResNet-50,
DenseNet-121, EfficientNet-B0, and ViT-Tiny/Small/Base with 16-pixel patches.

- **[CUB-200-2011](results/cub/README.md):** the three-seed ResNet-50 result
  above is the strongest replicated evidence. Across six architectures,
  mean foreground-corruption accuracy falls by 0.36 pp, so the robustness
  gain is specific to background interventions.
- **[Oxford-IIIT Pet](results/oxford_pet/README.md):** six-architecture means
  improve by 0.50 pp on clean accuracy and 0.67 pp on background-distortion
  accuracy, with 34.6% lower leakage.
- **[ISIC subset](results/isic/README.md):** leakage falls by 41.6%, but clean
  accuracy falls by 1.67 pp. Better localization does not guarantee better
  classification; this 100-image test is non-clinical.

Only CUB ResNet-50 has three training seeds; the broader matrix uses one.
Matched RRR, GAIN-ext, and CAMAL comparisons show an
accuracy–robustness–efficiency trade-off, with other methods leading some
localization and robustness metrics. See the
[complete results and baseline comparisons](results/README.md).

## Quick start

Use Python 3.10+ and a CUDA-capable PyTorch installation for the supplied
training configs. The reported environment is recorded in
[`requirements-lock.txt`](requirements-lock.txt).

```bash
python -m venv .venv
# Activate: .venv\Scripts\activate (Windows)
#           source .venv/bin/activate (Linux/macOS)
pip install -e ".[dev]"
python -m pytest
```

Prepare the datasets using [the setup guide](docs/REPRODUCIBILITY.md#dataset-locations).
The supplied configs expect them in `../datasets/`; update the YAML paths
for your machine. Then train CE and branch FCAL V2 from its epoch-5 checkpoint:

```bash
python -m fcal_study.train configs/cub/resnet50_ce.yaml
python -m fcal_study.train configs/cub/resnet50_fcal_v2.yaml --resume outputs/resnet50-ce-seed42/epoch-005-branch.pt
python -m fcal_study.evaluate outputs/resnet50-fcal-v2-seed42/best-validation.pt
```

Datasets and checkpoints are not bundled. The
[reproduction guide](docs/REPRODUCIBILITY.md) covers environment setup,
frozen splits, evaluation, and validation-only tuning.

## Explore the repository

| Start here | Contents |
|---|---|
| [Paper](paper/FCAL_manuscript.pdf) | Research manuscript, method, experiments, and limitations |
| [Results](results/README.md) | Dataset reports, full tables, CSVs, and baseline comparisons |
| [Notebooks](notebooks/README.md) | Executed prediction and CAM galleries |
| [Configs](configs/README.md) | Matched training configurations |
| [Source](src/) | Training, evaluation, CAM adapters, and Universal FCAL |
| [Reproduction guide](docs/REPRODUCIBILITY.md) | Dataset paths and experiment protocol |
