# Faithfulness aggregation (validation)

Higher mask IoU and insertion AUC are favorable; lower removal AUC is favorable.
All rows use the same frozen common-CAM evaluator and deterministic image subset.
Selective Faster-ISNet is labeled `resnet50_only` and is excluded from six-architecture W/T/L.

| regime        | architecture      | method                 | support_scope    |   samples |   valid_mask_samples |   mask_iou_mean |   insertion_auc_mean |   removal_auc_mean |
|:--------------|:------------------|:-----------------------|:-----------------|----------:|---------------------:|----------------:|---------------------:|-------------------:|
| fixed_default | densenet121       | CAMAL                  | six_architecture |       500 |                  500 |        0.553765 |             0.673881 |           0.084097 |
| fixed_default | densenet121       | CE                     | six_architecture |       500 |                  500 |        0.323145 |             0.711784 |           0.096364 |
| fixed_default | densenet121       | FCAL V2                | six_architecture |       500 |                  500 |        0.349805 |             0.697465 |           0.088687 |
| fixed_default | densenet121       | GAIN-ext               | six_architecture |       500 |                  500 |        0.246645 |             0.406546 |           0.044427 |
| fixed_default | densenet121       | RRR                    | six_architecture |       500 |                  500 |        0.311787 |             0.514115 |           0.060195 |
| fixed_default | efficientnet-b0   | CAMAL                  | six_architecture |       500 |                  500 |        0.64372  |             0.641086 |           0.123532 |
| fixed_default | efficientnet-b0   | CE                     | six_architecture |       500 |                  500 |        0.284397 |             0.660809 |           0.127165 |
| fixed_default | efficientnet-b0   | FCAL V2                | six_architecture |       500 |                  500 |        0.337134 |             0.672824 |           0.117765 |
| fixed_default | efficientnet-b0   | GAIN-ext               | six_architecture |       500 |                  500 |        0.371077 |             0.54024  |           0.077509 |
| fixed_default | efficientnet-b0   | RRR                    | six_architecture |       500 |                  500 |        0.286757 |             0.615386 |           0.076867 |
| fixed_default | resnet50          | CAMAL                  | six_architecture |       500 |                  500 |        0.600073 |             0.688696 |           0.085469 |
| fixed_default | resnet50          | CE                     | six_architecture |       500 |                  500 |        0.308455 |             0.699177 |           0.109141 |
| fixed_default | resnet50          | FCAL V2                | six_architecture |       500 |                  500 |        0.291556 |             0.696989 |           0.082249 |
| fixed_default | resnet50          | GAIN-ext               | six_architecture |       500 |                  500 |        0.419699 |             0.567999 |           0.043125 |
| fixed_default | resnet50          | RRR                    | six_architecture |       500 |                  500 |        0.176107 |             0.454068 |           0.076859 |
| fixed_default | resnet50          | Selective Faster-ISNet | resnet50_only    |       500 |                  500 |        0.267002 |             0.5477   |           0.063599 |
| fixed_default | vit-base-p16-avg  | CAMAL                  | six_architecture |       500 |                  500 |        0.781716 |             0.7228   |           0.126459 |
| fixed_default | vit-base-p16-avg  | CE                     | six_architecture |       500 |                  500 |        0.174975 |             0.147959 |           0.681615 |
| fixed_default | vit-base-p16-avg  | FCAL V2                | six_architecture |       500 |                  500 |        0.772844 |             0.725997 |           0.115094 |
| fixed_default | vit-base-p16-avg  | GAIN-ext               | six_architecture |       500 |                  500 |        0.748766 |             0.721957 |           0.069837 |
| fixed_default | vit-base-p16-avg  | RRR                    | six_architecture |       500 |                  500 |        0.213887 |             0.203955 |           0.620096 |
| fixed_default | vit-small-p16-avg | CAMAL                  | six_architecture |       500 |                  500 |        0.766181 |             0.653067 |           0.097704 |
| fixed_default | vit-small-p16-avg | CE                     | six_architecture |       500 |                  500 |        0.199731 |             0.109129 |           0.674281 |
| fixed_default | vit-small-p16-avg | FCAL V2                | six_architecture |       500 |                  500 |        0.225382 |             0.149983 |           0.606873 |
| fixed_default | vit-small-p16-avg | GAIN-ext               | six_architecture |       500 |                  500 |        0.816834 |             0.668479 |           0.066987 |
| fixed_default | vit-small-p16-avg | RRR                    | six_architecture |       500 |                  500 |        0.201347 |             0.087049 |           0.644475 |
| fixed_default | vit-tiny-p16-avg  | CAMAL                  | six_architecture |       500 |                  500 |        0.729661 |             0.522804 |           0.078473 |
| fixed_default | vit-tiny-p16-avg  | CE                     | six_architecture |       500 |                  500 |        0.179361 |             0.084418 |           0.495469 |
| fixed_default | vit-tiny-p16-avg  | FCAL V2                | six_architecture |       500 |                  500 |        0.215962 |             0.148039 |           0.459801 |
| fixed_default | vit-tiny-p16-avg  | GAIN-ext               | six_architecture |       500 |                  500 |        0.76528  |             0.536866 |           0.054257 |
| fixed_default | vit-tiny-p16-avg  | RRR                    | six_architecture |       500 |                  500 |        0.17946  |             0.069122 |           0.460568 |
