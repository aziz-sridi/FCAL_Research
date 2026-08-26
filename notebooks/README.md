# Visual test notebooks

These executed notebooks are lightweight, GitHub-readable audits of the saved
test results. They do not require checkpoints or datasets to view.

| Dataset | Notebook | Coverage |
|---|---|---|
| CUB-200-2011 | [Bird test showcase](cub_test_showcase.ipynb) | 6 architectures × 3 examples × CE/V1/V2 |
| ISIC | [Skin-lesion test showcase](isic_test_showcase.ipynb) | 6 architectures × 3 examples × CE/V1/V2 |
| Oxford-IIIT Pet | [Pet test showcase](oxford_pet_test_showcase.ipynb) | 6 architectures × 3 examples × CE/V1/V2/Universal |

Each comparison uses the same center-cropped test image for every method and
shows the predicted class, correctness, confidence, foreground contour, and
fraction of positive predicted-class CAM mass inside that contour.

The examples and their prediction metadata are frozen in
`assets/result_showcases/<dataset>/examples.csv`.
