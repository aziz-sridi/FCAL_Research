# Matched explanation-baseline audit (CUB, ResNet-50, seed 7)

These are exploratory single-seed baselines branched from the same epoch-5
checkpoint as the existing CE and FCAL-V2 seed-7 runs. All completed methods
were trained to epoch 50 and evaluated on the same 13 background interventions.
They are evidence for manuscript revision, not a substitute for a tuned
multi-seed comparison.

| Method | Clean | BG mean | BG worst | FG-only | Strict leakage | Post-branch time |
|---|---:|---:|---:|---:|---:|---:|
| CE | 77.55% | 70.27% | 47.95% | 68.33% | 57.04% | 4.37 min |
| FCAL V2 | **78.65%** | **72.83%** | 56.25% | **71.54%** | 28.38% | 7.25 min |
| RRR | 73.58% | 68.53% | **57.34%** | 67.83% | 52.22% | 31.93 min |
| GAIN-ext | 75.11% | 69.69% | 57.27% | 69.49% | **13.79%** | 9.97 min |
| ISNet LRP-Flex | 72.56% | 67.30% | 54.54% | 66.17% | 53.74% | 56.84 min |

Interpretation:

- FCAL V2 retains the best clean, mean-background, and foreground-only results.
- RRR and GAIN-ext improve the single worst background intervention by 9.39
  and 9.32 percentage points over CE, but lose 3.97 and 2.43 clean points.
- GAIN-ext is the strongest localization result: its strict leakage is 43.26
  points below CE and 14.60 points below FCAL V2. This prevents a claim that
  FCAL uniquely suppresses background evidence; FCAL's defensible advantage is
  its clean/robustness/localization trade-off in this matched run.
- RRR is much slower because it differentiates an input-gradient penalty.
- ISNet improves the worst background intervention by 6.59 points over CE,
  but gives up 4.99 clean points. Its selective LRP-Flex objective is the most
  computationally and memory intensive baseline in this matched audit.

## ISNet numerical provenance

Selective Faster ISNet uses the reference starting values (`e=0.01`, selective
epsilon `0.01`, `P=0.7`, `C1=1`, `C2=25`, GWRP `d=0`). The failed pilot was
traced to configuration handling, not ResNet-50: YAML parsed unquoted `off` as
Boolean `false`, while the autocast helper recognized only the string `off` and
therefore enabled FP16. LRP-Flex manipulates gradients inside forward and does
not support half-precision training. The paired resume also restored the CE
checkpoint's fused-optimizer flag over the ISNet child configuration. After
fixing both runtime issues, the full epoch-6--50 run completed in FP32 with
non-finite gradients retained as a hard error; no gradient replacement or
Grad x Input substitution was used.

## Implementation labels

- RRR uses the original summed-log-probability input-gradient penalty on the
  annotated background.
- GAIN-ext uses the externally supervised attention loss and attention-mining
  forward pass. For this GAP ResNet-50, the target Grad-CAM is computed by its
  algebraically equivalent class-weighted final-feature map.
- ISNet is labeled Selective Faster ISNet / LRP-Flex and is not conflated with
  the repository's Grad x Input ablation.

Exact values and run IDs are in `explanation_baselines_seed7.csv`.
