# ACDC screening pilot

This is a one-seed screening experiment on the Kaggle mirror of the ACDC
MICCAI 2017 dataset. The 150 labeled patients are diagnosis-stratified into
90 train, 30 validation, and 30 sealed test patients. All classification
metrics below are patient-level; slice logits are averaged before prediction.

Each input stores end-diastole, end-systole, and absolute ED/ES change in its
three channels. FCAL uses the union of the supplied ED and ES anatomical masks.
Both FCAL V2 runs resume their paired CE model's exact epoch-5 checkpoint and
finish at epoch 12.

| Architecture | Method | Patient accuracy | Balanced accuracy | Macro F1 | Strict leakage ↓ | Tolerant leakage ↓ |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | CE | 66.67% | 66.67% | 0.634 | 0.915 | 0.641 |
| ResNet-50 | FCAL V2 | 66.67% | 66.67% | 0.641 | 0.882 | 0.542 |
| ViT-Tiny/16 avg | CE | 53.33% | 53.33% | 0.527 | 0.911 | 0.833 |
| ViT-Tiny/16 avg | FCAL V2 | 50.00% | 50.00% | 0.500 | 0.910 | 0.831 |

The ResNet result is directionally useful: patient accuracy is preserved and
tolerant leakage falls by 15.3%. The ViT result is negative: patient accuracy
falls by 3.33 percentage points and tolerant leakage falls by only 0.3%.

## Decision

Do **not** train the remaining architecture matrix yet. The predeclared gate
required both architectures to lose no more than 5 pp patient accuracy, lower
strict leakage by at least 5%, and have a non-negative mean accuracy delta.
The pilot fails the localization requirement for both models and the mean
accuracy requirement because of ViT-Tiny.

The next useful experiment is a small validation-only hyperparameter check for
the ViT (longer training and/or stronger FCAL schedule), without reopening the
test set. This pilot is not clinical evidence.

Machine-readable results are in `pilot_metrics.csv` and `pilot_summary.json`.
