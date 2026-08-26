# Results hub

Start with a visual dataset page, open its executed test notebook for more CAM
examples, then use the detailed report and CSVs for exact claims. Checkpoints,
full prediction arrays, and evaluation caches are intentionally excluded.

| Dataset / study | Visual summary | Test notebook | Detailed evidence |
|---|---|---|---|
| CUB-200-2011 | [Bird results](cub/README.md) | [18 test cases](../notebooks/cub_test_showcase.ipynb) | [Full report](cub/RESULTS.md) · [`cub/*.csv`](cub/) |
| ISIC | [Skin-lesion results](isic/README.md) | [18 test cases](../notebooks/isic_test_showcase.ipynb) | [Full report](isic/RESULTS.md) · [`isic/*.csv`](isic/) |
| Oxford-IIIT Pet | [Pet results](oxford_pet/README.md) | [18 test cases](../notebooks/oxford_pet_test_showcase.ipynb) | [Full report](oxford_pet/RESULTS.md) · [`oxford_pet/*.csv`](oxford_pet/) |
| Universal FCAL | [Current status](universal/README.md) | Included in the [Pet notebook](../notebooks/oxford_pet_test_showcase.ipynb) | [ISIC pilot](universal/isic/RESULTS.md) · [`universal/isic/*.csv`](universal/isic/) |

The qualitative examples are frozen in
`assets/result_showcases/<dataset>/examples.csv` and reconstructed from the real
best-validation checkpoints. All legacy method labels are normalized: the
support-aware method formerly called V5 is reported as **FCAL V2**.
