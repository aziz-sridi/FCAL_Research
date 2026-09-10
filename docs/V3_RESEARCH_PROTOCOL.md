# FCAL V3 research protocol

## Status

FCAL V3 is a state-of-the-art **candidate**, not a state-of-the-art result.
This distinction remains until the predeclared CUB-200-2011 experiment beats
the frozen V2 and explanation-supervision baselines on held-out test data.

## Why V3 follows V2

The frozen evidence identifies a specific gap. V2 improves CUB clean accuracy,
background robustness, foreground-only accuracy, and leakage, but CAMAL and
GAIN-ext often produce stronger localization and RRR obtains higher mean
worst-background accuracy after tuning. V2 supervises where class evidence is
placed; it does not directly require the prediction or foreground evidence to
remain stable when the background changes.

The archived V3 notebook tested a related behavioral idea but replaced the
ordinary classifier with a LogSumExp spatial head. Its matched GAP control
reached 72.26% clean test accuracy, while LogSumExp CE reached 67.88% and the
full method reached 67.05%. The new V3 therefore preserves the deployed
architecture and treats counterfactual behavior as a loss-only extension.

## Relation to prior methods

- V1 contributes normalized positive-evidence containment.
- V2 contributes resolution-aware tolerance and scale-free distributed signed
  foreground support.
- CAMAL aligns attention with segmentation masks and explicitly targets
  alignment and faithfulness: <https://arxiv.org/abs/2605.08325>.
- Counterfactual and invariant data generation shows that altering non-causal
  image regions can reduce spurious background dependence:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.html>.
- RRDA reports that foreground/background recombination can improve background
  robustness, while also finding that foreground-focused explanations alone
  do not guarantee robustness:
  <https://openaccess.thecvf.com/content/ICCV2023W/LXCV/html/Santos_Exploring_Image_Classification_Robustness_and_Interpretability_with_Right_for_the_ICCVW_2023_paper.html>.

V3's differentiating hypothesis is that all three constraints are needed in a
single architecture-preserving objective: spatial containment, distributed
signed support, and stability of both predictions and foreground evidence
under an explicit background intervention.

## Dataset anchor

The development dataset is CUB-200-2011 with the existing frozen split and
segmentation masks. Matching V3 configurations are provided for all six
existing CNN and ViT architectures under `configs/cub/*_fcal_v3.yaml`. All V3
runs resume from the same paired CE epoch-5 branch as V1 and V2; V3 must not
receive a longer training schedule or a different backbone.

## Validation-only ablation

Run the following progression before opening the sealed test result:

1. V2.
2. V2 plus counterfactual CE.
3. V2 plus counterfactual CE and prediction consistency.
4. Full V3 with foreground-evidence consistency.

Use validation clean accuracy, the complete 13-view background suite, strict
leakage, and common-map faithfulness. Do not select from one stochastic
background corruption or from test performance.

## Frozen domination gates

The six-architecture CUB target is to exceed the strongest currently observed
mean for each principal metric, rather than to beat only one baseline:

| Metric | V3 target | Existing reference |
|---|---:|---|
| Clean accuracy | >80.05% | fixed FCAL V2 |
| Background mean | >73.75% | fixed FCAL V2 |
| Worst-background accuracy | >60.61% | tuned RRR |
| Foreground-only accuracy | >73.31% | fixed FCAL V2 |
| Strict leakage | <0.303 | tuned GAIN-ext |

These are development targets, not achieved results. A state-of-the-art claim
also requires multi-seed uncertainty, no material architecture-specific clean
regression, and the existing common-map faithfulness evaluation. ResNet-50
should use at least seeds 7, 42, and 123; five seeds are preferred.

## Claim rule

If V3 misses any domination gate, report the result as an
accuracy--robustness--localization trade-off. Use “state of the art” only with
the exact dataset, metric set, architecture support, tuning regime, and
uncertainty basis named explicitly.
