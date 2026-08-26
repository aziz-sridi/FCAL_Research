# ResNet-50 ISIC Universal FCAL experiment

## Protocol

All U-FCAL trials branch from the same CE epoch-5 checkpoint. Three fixed settings and two automatic settings were compared on validation only. The conservative fixed setting and safer automatic setting were frozen before their held-out test evaluations. Seed 42 is the only seed in this pilot.

## Validation tuning

| configuration                         |   best_validation_accuracy |   validation_balanced_accuracy |   validation_macro_f1 |   validation_strict_leakage |
|:--------------------------------------|---------------------------:|-------------------------------:|----------------------:|----------------------------:|
| resnet50-auto-safe-seed42             |                     0.823  |                         0.7996 |                0.8077 |                      0.4277 |
| resnet50-manual-r005-t050-s050-seed42 |                     0.8118 |                         0.7818 |                0.7935 |                      0.3509 |
| resnet50-manual-r010-t025-s025-seed42 |                     0.8034 |                         0.7891 |                0.7897 |                      0.2718 |
| resnet50-auto-seed42                  |                     0.7978 |                         0.7768 |                0.7765 |                      0.3955 |
| resnet50-manual-r010-t050-s050-seed42 |                     0.7949 |                         0.7649 |                0.7702 |                      0.2527 |

## Held-out test

| method       |   clean_accuracy |   balanced_accuracy |   macro_f1 |   strict_leakage |   nonzero_map_fraction |   mean_background_transform_accuracy |   mean_foreground_transform_accuracy |   foreground_only_accuracy |   background_only_accuracy |
|:-------------|-----------------:|--------------------:|-----------:|-----------------:|-----------------------:|-------------------------------------:|-------------------------------------:|---------------------------:|---------------------------:|
| CE           |             0.68 |              0.7016 |     0.6805 |           0.4349 |                   1    |                               0.4785 |                               0.3477 |                       0.45 |                       0.27 |
| FCAL V1      |             0.65 |              0.6788 |     0.6297 |           0.429  |                   0.98 |                               0.4638 |                               0.3085 |                       0.42 |                       0.16 |
| FCAL V2      |             0.63 |              0.66   |     0.6328 |           0.2269 |                   1    |                               0.4777 |                               0.31   |                       0.49 |                       0.26 |
| U-FCAL fixed |             0.66 |              0.6848 |     0.6792 |           0.3322 |                   0.99 |                               0.4546 |                               0.3192 |                       0.33 |                       0.22 |
| U-FCAL auto  |             0.63 |              0.6614 |     0.6383 |           0.4457 |                   1    |                               0.4754 |                               0.3262 |                       0.4  |                       0.23 |

## Paired clean-accuracy uncertainty versus CE

| method       |   accuracy_difference |   ci_low |   ci_high |   ce_only_correct |   candidate_only_correct |   mcnemar_p |
|:-------------|----------------------:|---------:|----------:|------------------:|-------------------------:|------------:|
| FCAL V1      |                 -0.03 |    -0.08 |      0.02 |                 5 |                        2 |      0.4531 |
| FCAL V2      |                 -0.05 |    -0.11 |      0.01 |                 7 |                        2 |      0.1797 |
| U-FCAL fixed |                 -0.02 |    -0.09 |      0.05 |                 8 |                        6 |      0.7905 |
| U-FCAL auto  |                 -0.05 |    -0.11 |      0    |                 7 |                        2 |      0.1797 |

## Conclusion

The strongest fixed U-FCAL trade-off reached 66.0% clean accuracy versus 68.0% for CE while reducing strict leakage by 23.6% relative. It did not improve every metric: foreground-only accuracy changed from 45.0% to 33.0%.

The safer automatic run reached 82.3% validation accuracy, but its validation-selected checkpoint came from the CE-release phase and scored 63.0% on the 100-image test set. This demonstrates that a controller can adapt correctly in-run without guaranteeing out-of-domain generalization, and that accuracy-only checkpoint selection can discard localization gains.

No clean-accuracy difference is statistically resolved by this 100-image test. U-FCAL is therefore a working architecture-independent library and a promising localization/accuracy trade-off, not yet a universal improvement claim.
