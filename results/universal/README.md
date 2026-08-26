# Universal FCAL

Universal FCAL is the architecture-independent branch of the project. Instead
of relying on one fixed auxiliary-loss scale, it calibrates localization
targets, budgets FCAL gradients relative to cross-entropy, projects conflicting
gradients, and adjusts its schedule from validation behavior.

## Current evidence

| Benchmark | Result | Status |
|---|---|---|
| Oxford-IIIT Pet, six architectures | +0.42 pp mean clean accuracy and +0.74 pp mean foreground-corruption accuracy versus CE | Complete 24-run comparison |
| Oxford-IIIT Pet localization | Mean strict leakage 0.292 → 0.290 | Little localization benefit in the automatic-safe setting |
| ISIC ResNet-50 | Best fixed setting reduces leakage 23.6% relative with -2 pp clean accuracy | Small, one-model pilot |

On Oxford-IIIT Pet, Universal FCAL is the best method by mean
foreground-corruption accuracy, but FCAL V2 is much stronger at reducing CAM
leakage. The automatic controller is therefore useful experimental machinery,
not yet evidence of one universally superior objective.

The Oxford Pet gallery below includes Universal FCAL beside CE, V1, and V2:

![Universal FCAL beside CE, V1, and V2 on Oxford Pet CNN rescue examples](../../assets/result_showcases/oxford_pet/readme_cnn_rescues.png)

- [ViT predictions and CAMs](../../assets/result_showcases/oxford_pet/readme_vit.png)
- [Executed visual notebook](../../notebooks/oxford_pet_test_showcase.ipynb)
- [Full Oxford Pet benchmark](../oxford_pet/RESULTS.md)
- [Full ISIC pilot](isic/RESULTS.md)

The implementation lives in [`src/universal_fcal/`](../../src/universal_fcal/)
and only requires a model adapter that returns signed class-evidence maps.
