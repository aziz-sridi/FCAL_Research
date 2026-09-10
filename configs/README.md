# Experiment configurations

Configurations are grouped by dataset. CUB has one YAML per architecture and method; ISIC and Oxford-IIIT Pet use matrix configs whose method can be selected with `--method`. ACDC contains the short ResNet-50 and ViT-Tiny screening configs used by `python -m fcal_study.acdc_pilot --download`.

| Public method | Trainer key | Config section | Run suffix |
|---|---|---|---|
| Baseline | `ce` | — | `ce` |
| FCAL V1 | `fcal` | `fcal` | `fcal` or `fcal-v1` |
| FCAL V2 | `fcal_v2` | `fcal_v2` | `fcal-v2` |
| FCAL V3 candidate | `fcal_v3` | `fcal_v2` + `fcal_v3` | `fcal-v3` |
| FCAL V4 | `fcal_v4` | `fcal_v2` + `fcal_v4` | `fcal-v4-pure` |
| FCAL V5 certificate candidate | `fcal_v5_cheap` | `fcal_v5_cheap` | `fcal-v5-certificate-smooth` |
| Universal FCAL | `ufcal` | `ufcal` | `ufcal-auto-safe` |
| Right for the Right Reasons | `rrr` | `rrr` | `rrr` |
| GAIN with external masks | `gain` | `gain` | `gain-ext` |
| Selective Faster ISNet | `isnet` | `isnet` | `isnet-selective` |

All FCAL methods should resume from the paired CE `epoch-005-branch.pt`. Dataset roots are resolved relative to the repository root, not relative to the YAML file.

The V3 candidate is anchored to CUB-200-2011. The `cub/*_fcal_v3.yaml` files
retain V2 and add a foreground-preserving background-counterfactual training
view. `cub/resnet50_fcal_v3_balanced.yaml` is the validation-selected successor:
it balances that view with foreground-only interventions. Both remain
experimental; the completed one-seed comparison does not establish a general
state-of-the-art claim.

`cub/resnet50_fcal_v3_routed_strict.yaml` is the V3.1 interventional ablation.
It routes consistency to background interventions and adds a weak strict-mask
boundary-shell penalty. Its checkpoint must be selected under the predeclared
validation protocol in `results/v3/cub/ROUTED_STRICT_VALIDATION_PROTOCOL.md`.

V4 defaults to a genuine original-input loss through
`cub/resnet50_fcal_v4_pure.yaml`. Its nested `intervention.enabled` flag is
`false`; changing it to `true` explicitly opts into the V3.1-style manipulated
second view. Pure and interventional V4 results must be reported separately.
The `pure_coverage`, `pure_dominance`, and `pure_curriculum` files are frozen
validation-only mechanism studies. Of these, `pure_dominance` is the balanced
single-seed profile; none passed every predeclared gate, so none is a confirmed
test-set or state-of-the-art result.

`cub/resnet50_fcal_v5_certificate.yaml` is the frozen first-order regional
margin-certificate experiment. The internal key avoids collision with legacy
`fcal-v5` run suffixes that refer to FCAL V2. The V5 test split remains sealed
because the candidate failed its validation gates.

The explanation-regularization configs currently cover the priority CUB
ResNet-50 seed-7 comparison. They resume from the same CE epoch-5 branch as
FCAL. ISNet uses full precision because its differentiable LRP pass does not
support AMP; `z_b_input_rule: false` records that the optional first-layer
LRP-zB refinement is not used. The reference-style ISNet run currently fails
fast on non-finite second-order gradients; the failed diagnostic is retained
instead of producing a sanitized score.

## V2 component ablation

The four `ablation_resnet50_*_seed42_20ep.yaml` files make the corrected
component experiment explicit and reproducible. The CE parent uses a 50-epoch
cosine schedule and saves its epoch-5 branch. Each V2 diagnostic resumes that
state and stops at epoch 20; the optimizer/scheduler state therefore preserves
the parent 50-epoch schedule.

`fixed_tolerance_cells: 3` means exactly three cells on the evidence grid and
uses V1's resize-then-dilate rule. `fixed_tolerance_pixels`, when used, means
input-image pixels. The trainer rejects a config that supplies both units.
