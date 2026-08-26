# ISIC FCAL V1 versus V2

All methods use the same leakage-free, lesion-grouped split and the same epoch-5 CE branch.

## Result

V2 reduced mean strict outside-lesion attention by 0.1272 absolute (33.1% relative) versus V1, and reduced leakage on every architecture. Its clean accuracy was higher on 3/6 architectures and lower on 3/6.

Across architectures, V2 minus V1 was +0.17 pp clean accuracy, -0.31 pp balanced accuracy, +1.26 pp macro-F1, and +3.33 pp foreground-only accuracy.

Conclusion: V2 is consistently more effective at constraining attention to the lesion, but this six-model, one-seed experiment does not show a consistent classification advantage over V1. The CE baseline also retains the best mean clean accuracy and balanced accuracy. Treat V2 here as a localization improvement with mixed predictive effectiveness.

## Method means

| method   |   clean_accuracy |   clean_balanced_accuracy |   clean_macro_f1 |   strict_leakage |   mean_background_accuracy |   mean_foreground_accuracy |   foreground_only_accuracy |   background_only_accuracy |
|:---------|-----------------:|--------------------------:|-----------------:|-----------------:|---------------------------:|---------------------------:|---------------------------:|---------------------------:|
| Baseline |           0.6650 |                    0.6952 |           0.6679 |           0.4408 |                     0.4965 |                     0.3700 |                     0.4250 |                     0.2333 |
| FCAL V1  |           0.6467 |                    0.6797 |           0.6421 |           0.3846 |                     0.4886 |                     0.3560 |                     0.4283 |                     0.2150 |
| FCAL V2  |           0.6483 |                    0.6766 |           0.6547 |           0.2574 |                     0.4982 |                     0.3576 |                     0.4617 |                     0.2167 |

## V2 minus V1 by architecture

| architecture      |   clean_accuracy |   clean_balanced_accuracy |   clean_macro_f1 |   strict_leakage |   mean_background_accuracy |   mean_foreground_accuracy |   foreground_only_accuracy |   background_only_accuracy |
|:------------------|-----------------:|--------------------------:|-----------------:|-----------------:|---------------------------:|---------------------------:|---------------------------:|---------------------------:|
| resnet50          |          -0.0200 |                   -0.0187 |          +0.0031 |          -0.2021 |                    +0.0138 |                    +0.0015 |                    +0.0700 |                    +0.1000 |
| densenet121       |          -0.0100 |                   -0.0062 |          -0.0310 |          -0.1603 |                    +0.0423 |                    -0.0346 |                    +0.0800 |                    -0.1100 |
| efficientnet-b0   |          +0.0100 |                   -0.0061 |          +0.0306 |          -0.0393 |                    -0.0354 |                    +0.0100 |                    +0.0300 |                    +0.0700 |
| vit-tiny-p16-avg  |          +0.0300 |                   +0.0268 |          +0.0369 |          -0.0093 |                    +0.0423 |                    +0.0092 |                    +0.0700 |                    -0.0600 |
| vit-small-p16-avg |          +0.0400 |                   +0.0287 |          +0.0591 |          -0.2103 |                    +0.0362 |                    +0.0154 |                    +0.0400 |                    -0.0100 |
| vit-base-p16-avg  |          -0.0400 |                   -0.0431 |          -0.0231 |          -0.1419 |                    -0.0415 |                    +0.0077 |                    -0.0900 |                    +0.0200 |

## Full metrics

| architecture      | method   |   clean_accuracy |   clean_balanced_accuracy |   clean_macro_f1 |   strict_leakage |   tolerant_leakage |   mean_background_accuracy |   mean_foreground_accuracy |   foreground_only_accuracy |   background_only_accuracy |
|:------------------|:---------|-----------------:|--------------------------:|-----------------:|-----------------:|-------------------:|---------------------------:|---------------------------:|---------------------------:|---------------------------:|
| resnet50          | Baseline |           0.6800 |                    0.7016 |           0.6805 |           0.4349 |             0.0006 |                     0.4785 |                     0.3477 |                     0.4500 |                     0.2700 |
| resnet50          | FCAL V1  |           0.6500 |                    0.6788 |           0.6297 |           0.4290 |             0.0001 |                     0.4638 |                     0.3085 |                     0.4200 |                     0.1600 |
| resnet50          | FCAL V2  |           0.6300 |                    0.6600 |           0.6328 |           0.2269 |             0.0001 |                     0.4777 |                     0.3100 |                     0.4900 |                     0.2600 |
| densenet121       | Baseline |           0.6500 |                    0.6792 |           0.6624 |           0.3910 |             0.0002 |                     0.4477 |                     0.3000 |                     0.3900 |                     0.1700 |
| densenet121       | FCAL V1  |           0.6600 |                    0.6876 |           0.6707 |           0.3912 |             0.0001 |                     0.4762 |                     0.3154 |                     0.4600 |                     0.2100 |
| densenet121       | FCAL V2  |           0.6500 |                    0.6814 |           0.6397 |           0.2308 |             0.0000 |                     0.5185 |                     0.2808 |                     0.5400 |                     0.1000 |
| efficientnet-b0   | Baseline |           0.6900 |                    0.7214 |           0.6782 |           0.2650 |             0.0003 |                     0.3338 |                     0.2969 |                     0.1500 |                     0.1600 |
| efficientnet-b0   | FCAL V1  |           0.6800 |                    0.7155 |           0.6769 |           0.2585 |             0.0003 |                     0.3346 |                     0.2815 |                     0.1400 |                     0.1300 |
| efficientnet-b0   | FCAL V2  |           0.6900 |                    0.7094 |           0.7075 |           0.2192 |             0.0002 |                     0.2992 |                     0.2915 |                     0.1700 |                     0.2000 |
| vit-tiny-p16-avg  | Baseline |           0.6800 |                    0.7144 |           0.7001 |           0.5053 |             0.1117 |                     0.5931 |                     0.4069 |                     0.4800 |                     0.2300 |
| vit-tiny-p16-avg  | FCAL V1  |           0.6000 |                    0.6381 |           0.6073 |           0.5124 |             0.1125 |                     0.5192 |                     0.4000 |                     0.4700 |                     0.2600 |
| vit-tiny-p16-avg  | FCAL V2  |           0.6300 |                    0.6649 |           0.6443 |           0.5030 |             0.1068 |                     0.5615 |                     0.4092 |                     0.5400 |                     0.2000 |
| vit-small-p16-avg | Baseline |           0.6700 |                    0.7022 |           0.6863 |           0.5214 |             0.0980 |                     0.5715 |                     0.4415 |                     0.5300 |                     0.2700 |
| vit-small-p16-avg | FCAL V1  |           0.6500 |                    0.6877 |           0.6505 |           0.5005 |             0.0991 |                     0.5608 |                     0.4246 |                     0.5100 |                     0.2500 |
| vit-small-p16-avg | FCAL V2  |           0.6900 |                    0.7165 |           0.7097 |           0.2903 |             0.0269 |                     0.5969 |                     0.4400 |                     0.5500 |                     0.2400 |
| vit-base-p16-avg  | Baseline |           0.6200 |                    0.6524 |           0.5999 |           0.5276 |             0.1093 |                     0.5546 |                     0.4269 |                     0.5500 |                     0.3000 |
| vit-base-p16-avg  | FCAL V1  |           0.6400 |                    0.6706 |           0.6174 |           0.2159 |             0.0002 |                     0.5769 |                     0.4062 |                     0.5700 |                     0.2800 |
| vit-base-p16-avg  | FCAL V2  |           0.6000 |                    0.6275 |           0.5943 |           0.0740 |             0.0000 |                     0.5354 |                     0.4138 |                     0.4800 |                     0.3000 |

## Limitations

- The sealed test split has 100 images; VASC has only three.
- This is one seed, so cross-seed uncertainty is not estimated.
- Masks mix expert challenge boundaries with reviewed novice U-Net boundaries.
- These results measure model behavior on this curated subset, not clinical effectiveness.
