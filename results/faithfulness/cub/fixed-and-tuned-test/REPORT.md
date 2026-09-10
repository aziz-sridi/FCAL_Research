# Faithfulness aggregation (test)

Higher mask IoU and insertion AUC are favorable; lower removal AUC is favorable.
All rows use the same frozen common-CAM evaluator and deterministic image subset.
Selective Faster-ISNet is labeled `resnet50_only` and is excluded from six-architecture W/T/L.

| regime        | architecture      | method                 | support_scope    |   samples |   valid_mask_samples |   mask_iou_mean |   insertion_auc_mean |   removal_auc_mean |
|:--------------|:------------------|:-----------------------|:-----------------|----------:|---------------------:|----------------:|---------------------:|-------------------:|
| fixed_default | densenet121       | CAMAL                  | six_architecture |       500 |                  500 |        0.537748 |             0.639254 |           0.08049  |
| fixed_default | densenet121       | CE                     | six_architecture |       500 |                  500 |        0.319555 |             0.677613 |           0.08906  |
| fixed_default | densenet121       | FCAL V2                | six_architecture |       500 |                  500 |        0.339643 |             0.663097 |           0.083547 |
| fixed_default | densenet121       | GAIN-ext               | six_architecture |       500 |                  500 |        0.236029 |             0.383707 |           0.041592 |
| fixed_default | densenet121       | RRR                    | six_architecture |       500 |                  500 |        0.309832 |             0.482854 |           0.057373 |
| fixed_default | efficientnet-b0   | CAMAL                  | six_architecture |       500 |                  500 |        0.634983 |             0.649168 |           0.122602 |
| fixed_default | efficientnet-b0   | CE                     | six_architecture |       500 |                  500 |        0.284329 |             0.669048 |           0.13318  |
| fixed_default | efficientnet-b0   | FCAL V2                | six_architecture |       500 |                  500 |        0.336877 |             0.674385 |           0.125968 |
| fixed_default | efficientnet-b0   | GAIN-ext               | six_architecture |       500 |                  500 |        0.369554 |             0.535364 |           0.080047 |
| fixed_default | efficientnet-b0   | RRR                    | six_architecture |       500 |                  500 |        0.281368 |             0.618654 |           0.077505 |
| fixed_default | resnet50          | CAMAL                  | six_architecture |       500 |                  500 |        0.575023 |             0.66326  |           0.080139 |
| fixed_default | resnet50          | CE                     | six_architecture |       500 |                  500 |        0.295412 |             0.651609 |           0.09928  |
| fixed_default | resnet50          | FCAL V2                | six_architecture |       500 |                  500 |        0.283051 |             0.674582 |           0.07963  |
| fixed_default | resnet50          | GAIN-ext               | six_architecture |       500 |                  500 |        0.410103 |             0.538306 |           0.039515 |
| fixed_default | resnet50          | RRR                    | six_architecture |       500 |                  500 |        0.176035 |             0.446287 |           0.076653 |
| fixed_default | resnet50          | Selective Faster-ISNet | resnet50_only    |       500 |                  500 |        0.257316 |             0.513385 |           0.062793 |
| fixed_default | vit-base-p16-avg  | CAMAL                  | six_architecture |       500 |                  500 |        0.776367 |             0.714274 |           0.123882 |
| fixed_default | vit-base-p16-avg  | CE                     | six_architecture |       500 |                  500 |        0.173682 |             0.145199 |           0.675609 |
| fixed_default | vit-base-p16-avg  | FCAL V2                | six_architecture |       500 |                  500 |        0.769369 |             0.702245 |           0.108308 |
| fixed_default | vit-base-p16-avg  | GAIN-ext               | six_architecture |       500 |                  500 |        0.75164  |             0.720742 |           0.069975 |
| fixed_default | vit-base-p16-avg  | RRR                    | six_architecture |       500 |                  500 |        0.206453 |             0.206596 |           0.582258 |
| fixed_default | vit-small-p16-avg | CAMAL                  | six_architecture |       500 |                  500 |        0.757473 |             0.65636  |           0.09673  |
| fixed_default | vit-small-p16-avg | CE                     | six_architecture |       500 |                  500 |        0.193192 |             0.108737 |           0.682861 |
| fixed_default | vit-small-p16-avg | FCAL V2                | six_architecture |       500 |                  500 |        0.215089 |             0.150822 |           0.620848 |
| fixed_default | vit-small-p16-avg | GAIN-ext               | six_architecture |       500 |                  500 |        0.817502 |             0.663743 |           0.066028 |
| fixed_default | vit-small-p16-avg | RRR                    | six_architecture |       500 |                  500 |        0.192546 |             0.085255 |           0.640226 |
| fixed_default | vit-tiny-p16-avg  | CAMAL                  | six_architecture |       500 |                  500 |        0.717528 |             0.524259 |           0.078026 |
| fixed_default | vit-tiny-p16-avg  | CE                     | six_architecture |       500 |                  500 |        0.166353 |             0.075325 |           0.499623 |
| fixed_default | vit-tiny-p16-avg  | FCAL V2                | six_architecture |       500 |                  500 |        0.203244 |             0.130421 |           0.464927 |
| fixed_default | vit-tiny-p16-avg  | GAIN-ext               | six_architecture |       500 |                  500 |        0.741496 |             0.536815 |           0.050854 |
| fixed_default | vit-tiny-p16-avg  | RRR                    | six_architecture |       500 |                  500 |        0.171253 |             0.066499 |           0.471656 |
| tuned         | densenet121       | CAMAL                  | six_architecture |       500 |                  500 |        0.494086 |             0.654779 |           0.080056 |
| tuned         | densenet121       | FCAL V2                | six_architecture |       500 |                  500 |        0.332241 |             0.662646 |           0.083533 |
| tuned         | densenet121       | GAIN-ext               | six_architecture |       500 |                  500 |        0.243865 |             0.16861  |           0.019313 |
| tuned         | densenet121       | RRR                    | six_architecture |       500 |                  500 |        0.298069 |             0.548777 |           0.067752 |
| tuned         | efficientnet-b0   | CAMAL                  | six_architecture |       500 |                  500 |        0.442169 |             0.644022 |           0.121909 |
| tuned         | efficientnet-b0   | FCAL V2                | six_architecture |       500 |                  500 |        0.323104 |             0.670535 |           0.124492 |
| tuned         | efficientnet-b0   | GAIN-ext               | six_architecture |       500 |                  500 |        0.327207 |             0.590014 |           0.086501 |
| tuned         | efficientnet-b0   | RRR                    | six_architecture |       500 |                  500 |        0.285725 |             0.649957 |           0.086052 |
| tuned         | resnet50          | CAMAL                  | six_architecture |       500 |                  500 |        0.620092 |             0.651198 |           0.078428 |
| tuned         | resnet50          | FCAL V2                | six_architecture |       500 |                  500 |        0.244904 |             0.661284 |           0.076897 |
| tuned         | resnet50          | GAIN-ext               | six_architecture |       500 |                  500 |        0.302927 |             0.526148 |           0.041411 |
| tuned         | resnet50          | RRR                    | six_architecture |       500 |                  500 |        0.233701 |             0.546533 |           0.082496 |
| tuned         | resnet50          | Selective Faster-ISNet | resnet50_only    |       500 |                  500 |        0.271973 |             0.585628 |           0.076436 |
| tuned         | vit-base-p16-avg  | CAMAL                  | six_architecture |       500 |                  500 |        0.783621 |             0.721521 |           0.118329 |
| tuned         | vit-base-p16-avg  | FCAL V2                | six_architecture |       500 |                  500 |        0.771746 |             0.719679 |           0.110816 |
| tuned         | vit-base-p16-avg  | GAIN-ext               | six_architecture |       500 |                  500 |        0.721104 |             0.722018 |           0.081436 |
| tuned         | vit-base-p16-avg  | RRR                    | six_architecture |       500 |                  500 |        0.193591 |             0.146085 |           0.648629 |
| tuned         | vit-small-p16-avg | CAMAL                  | six_architecture |       500 |                  500 |        0.745471 |             0.666094 |           0.096034 |
| tuned         | vit-small-p16-avg | FCAL V2                | six_architecture |       500 |                  500 |        0.338312 |             0.521128 |           0.25616  |
| tuned         | vit-small-p16-avg | GAIN-ext               | six_architecture |       500 |                  500 |        0.805838 |             0.653776 |           0.069178 |
| tuned         | vit-small-p16-avg | RRR                    | six_architecture |       500 |                  500 |        0.192924 |             0.095263 |           0.677571 |
| tuned         | vit-tiny-p16-avg  | CAMAL                  | six_architecture |       500 |                  500 |        0.652929 |             0.526711 |           0.079669 |
| tuned         | vit-tiny-p16-avg  | FCAL V2                | six_architecture |       500 |                  500 |        0.197039 |             0.10802  |           0.493674 |
| tuned         | vit-tiny-p16-avg  | GAIN-ext               | six_architecture |       500 |                  500 |        0.748338 |             0.548264 |           0.055323 |
| tuned         | vit-tiny-p16-avg  | RRR                    | six_architecture |       500 |                  500 |        0.178355 |             0.076227 |           0.559138 |
