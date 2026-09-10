# FCAL V2+ three-seed validation report

## Outcome

The frozen V2+ foreground-competitor candidate passes all six predeclared
three-seed validation gates. It is eligible for one test evaluation of each
validation-selected checkpoint. No coefficient or implementation change was
made after the seed-42 screen.

V2+ is a strict loss-only successor to V2: one untouched input image, one
ordinary forward, and one ordinary backward. The loss reads the target and
top-rival evidence maps already available from that forward. It creates no
second view and performs no pixel manipulation.

## Per-seed validation metrics

Higher accuracy and lower leakage are favorable.

| Seed | Method | Clean | Leakage | BG mean | BG worst | FG mean | FG only |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | V2 | 79.83% | 0.272 | 72.73% | **57.00%** | 31.64% | 70.25% |
| 7 | V2+ | **80.42%** | **0.253** | **72.92%** | 53.67% | **32.33%** | **72.67%** |
| 42 | V2 | **80.08%** | 0.266 | 72.04% | 48.92% | **31.83%** | 71.67% |
| 42 | V2+ | 79.67% | **0.244** | **73.16%** | **52.67%** | 31.74% | **73.25%** |
| 123 | V2 | 78.50% | 0.268 | 71.55% | 49.75% | 31.28% | **73.50%** |
| 123 | V2+ | **78.83%** | **0.258** | **72.49%** | **53.50%** | **31.87%** | 72.25% |

## Frozen gate decision

| Metric | V2 mean | V2+ mean | Change | Gate |
|---|---:|---:|---:|---|
| Clean accuracy | 79.47% | **79.64%** | +0.17 pp | pass |
| Strict leakage | 0.269 | **0.252** | -0.017 | pass |
| Background mean | 72.11% | **72.86%** | +0.75 pp | pass |
| Background worst | 51.89% | **53.28%** | +1.39 pp | pass |
| Foreground mean | 31.58% | **31.98%** | +0.39 pp | pass |
| Foreground only | 71.81% | **72.72%** | +0.92 pp | pass |

The result passes 6/6 mean gates. Seed-level trade-offs remain visible: V2+
does not win every metric on every seed. The claim is therefore a three-seed
mean improvement, not per-run domination.

## Test-access decision

The joint gate in `VALIDATION_PROTOCOL.md` is satisfied. The selected seed-7,
seed-42, and seed-123 checkpoints may each receive the standard test evaluation
once, without any intervening method change.
