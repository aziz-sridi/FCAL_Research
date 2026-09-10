# CUB validation-only tuning provenance

Completed 2026-08-30 (Africa/Tunis).

## Execution gates

- 200/200 tuning jobs succeeded; 0 failed.
- Rung completion was 125/125, 50/50, and 25/25.
- Allocated wall-clock GPU accounting totals 21.604 hours.
- The frozen manifest contains exactly 25 validation-selected winners and records `test_set_used: false`.
- The full automated test suite passed before the frozen manifest was created.
- Exactly 25 frozen checkpoints received sealed standard test evaluation; every `evaluation/summary.json` contains all 30 registered variants.
- The fixed validation faithfulness batch completed 31/31 jobs.
- The combined fixed-and-tuned test faithfulness batch completed 56/56 jobs.
- Selective Faster-ISNet is restricted to ResNet-50 and excluded from six-architecture W/T/L calculations.

An evaluation session interruption occurred after two sealed summaries were complete and after the third job had created an empty `evaluation` directory. The empty directory was inspected, preserved as `evaluation.interrupted-20260830-empty`, and the third evaluation was then restarted. No partial metric file was overwritten.

## SHA-256 bindings

| Artifact | SHA-256 |
|---|---|
| `configs/cub/tuning_validation.yaml` | `1e103c4ddd989ee7867d831dfe95ee3c8b98d6d2e4460e7f968650b7422ff72e` |
| `outputs/tuning-validation/attempts.jsonl` | `ad1b51a70aed19c098f2b7bfa6b0c251b38d45fb96357f18003c298cbd4a5268` |
| `outputs/tuning-validation/frozen-winners.yaml` | `ed0d317038288f0c8caf0c52a15c6337a68490847c38412d35917673b2e89173` |
| `outputs/faithfulness/cub/manifests/fixed-validation.json` | `3cdc4b187303b0648319324ceb4933f44a298febfc3d738435545e670a9dfaac` |
| `outputs/faithfulness/cub/manifests/fixed-and-tuned-test.json` | `c3d5393940c2e00581f3e6ada954fcce8212510f38ecc249ee0efcdf4bc18858` |
| `results/tuned/cub/tuned_winner_metrics.csv` | `ac1153673056221c1e1f269ce121c00224e281e403ead5121bf70fe20be4c094` |
| `results/faithfulness/cub/fixed-and-tuned-test/summary.csv` | `7ce51499768cee978ff098fbaaa7ce527bd4bf9e6ed36735877c3d1120c47120` |
| `paper/generated/comparison_tables_manifest.json` | `2bcd328b7714168ca9256db5f21cec812d6f95dd90cd8735088029e651b75ded` |

The comparison-table manifest separately binds every table input and output. Recompute hashes after any intentional regeneration; do not treat this record as current if a bound artifact changes.

## Scope of inference

The fixed and tuned matrices are single-seed CUB comparisons. Architecture W/T/L counts describe cross-architecture consistency, not seed-level significance. Tuned FCAL has the highest six-architecture mean clean and foreground-only accuracy, but does not dominate strict leakage or common-map faithfulness. The evidence supports a CUB-focused accuracy--robustness--efficiency trade-off claim, not a universal or state-of-the-art claim.
