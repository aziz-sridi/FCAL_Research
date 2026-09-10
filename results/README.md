# Results hub

The CE/V1/V2 benchmark is now reported with one schema on every dataset:
clean, BG mean/worst, FG mean/worst, foreground-only, background-only, and
strict leakage. `BG` and `FG` summaries always use the same 13 common regional
interventions; dataset-tailored scenarios are kept in separate CSVs so they do
not silently change the common averages.

| Dataset | Visual result page | Executed notebook | Detailed evidence |
|---|---|---|---|
| CUB-200-2011 | [Bird results](cub/README.md) | [18 CAM cases](../notebooks/cub_test_showcase.ipynb) | [Report](cub/RESULTS.md) · [unified means](cub/method_means_complete.csv) |
| Oxford-IIIT Pet | [Pet results](oxford_pet/README.md) | [18 CAM cases](../notebooks/oxford_pet_test_showcase.ipynb) | [Report](oxford_pet/RESULTS.md) · [unified means](oxford_pet/method_means_complete.csv) |
| ISIC subset | [Skin-lesion results](isic/README.md) | [18 CAM cases](../notebooks/isic_test_showcase.ipynb) | [Report](isic/RESULTS.md) · [unified means](isic/method_means_complete.csv) |
| ACDC cardiac pilot | - | - | [Screening report](acdc/RESULTS.md) / [pilot metrics](acdc/pilot_metrics.csv) |
| Universal FCAL | [Current status](universal/README.md) | Included in the [Pet notebook](../notebooks/oxford_pet_test_showcase.ipynb) | [ISIC pilot](universal/isic/RESULTS.md) |

## CUB explanation-supervision comparison

The matched CUB fixed/default and validation-tuned arms are complete. The
fixed matrix contains 31 supported runs; the equal-dimensional successive-
halving search completed 200/200 jobs, froze 25 validation-selected winners,
and evaluated only those winners on test. Search cost was 21.604 allocated
GPU-hours with zero failed attempts.

- [Fixed/default evidence](baseline_fixed/cub/FIXED_EVIDENCE_SUMMARY.md)
- [Tuned metrics and immutable provenance](tuned/cub/PROVENANCE.md)
- [Fixed-only validation faithfulness](faithfulness/cub/fixed-validation/REPORT.md)
- [Combined fixed-and-tuned test faithfulness](faithfulness/cub/fixed-and-tuned-test/REPORT.md)

These are single-seed CUB architecture comparisons. Cross-architecture W/T/L
counts describe consistency, not seed-level significance. Selective
Faster-ISNet remains ResNet-50-only.

## Six-architecture V2 changes versus CE

| Dataset | Clean | BG mean | BG worst | FG mean | Foreground only | Leakage |
|---|---:|---:|---:|---:|---:|---:|
| CUB | +0.57 pp | +1.21 pp | +4.65 pp | -0.36 pp | +0.93 pp | 30.9% lower |
| Oxford Pet | +0.50 pp | +0.67 pp | +1.30 pp | +0.61 pp | +2.36 pp | 34.6% lower |
| ISIC subset | -1.67 pp | +0.17 pp | -1.00 pp | -1.24 pp | +3.67 pp | 41.6% lower |

These means summarize architectures, not independent repetitions. The main
matrix uses one training seed; the separate three-seed CUB ResNet-50 check is
reported in the CUB report and article. The 100-image ISIC subset is explicitly
treated as inconclusive for classification and non-clinical.

## Reading order

1. Open a visual result page to see matched source/CE/V2 predictions and `jet`
   CAMs.
2. Use the executed notebook for three examples per architecture.
3. Read the detailed report for failure cases and statistical scope.
4. Use `complete_metrics.csv`, `method_means_complete.csv`, and
   `tailored_scenarios.csv` for exact machine-readable values.

Checkpoints, full prediction arrays, and evaluation caches are intentionally
excluded. Frozen qualitative manifests live under
`assets/result_showcases/<dataset>/examples.csv`. The support-aware method that
was called V5 during development is reported publicly as **FCAL V2**.
