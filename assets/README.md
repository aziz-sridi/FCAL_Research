# Figure assets

| Path | Purpose |
|---|---|
| `meme1.png`, `meme2.png` | Two-panel motivation used in the project README |
| `cub_multiarchitecture_summary.png` | Compact numeric CUB overview retained for the detailed report |
| `result_showcases/overview.png` | Cross-dataset README hero comparison |
| `result_showcases/rescue_overview.png` | Cross-dataset CE-wrong → V2-correct comparison |
| `result_showcases/<dataset>/readme_cnn.png` | One test example per CNN for the dataset README |
| `result_showcases/<dataset>/readme_vit.png` | One test example per ViT for the dataset README |
| `result_showcases/<dataset>/readme_*_rescues.png` | One CE-wrong → V2-correct example per architecture |
| `result_showcases/<dataset>/*_test_cases.png` | Three examples per model for the executed notebook |
| `result_showcases/<dataset>/examples.csv` | Frozen IDs, predictions, correctness, confidence, and inside-foreground CAM mass |

The scientific galleries were generated from real best-validation checkpoints.
They compare each method on the exact same center-cropped test image. CAMs are
predicted-class maps rendered with `jet` (blue = low, red = high); the white
contour is the annotated foreground.
