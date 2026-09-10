# FCAL V2.4 Seed-and-Grow seed-42 validation report

## Outcome

Seed-and-Grow improves clean accuracy, foreground-only accuracy, and strict
leakage, but fails the frozen background, foreground-corruption, and compute
gates. Its best clean-validation checkpoint reaches 80.25%, foreground-only
accuracy reaches 73.67%, and leakage falls to 0.145. Mean background accuracy
is 71.62%, while random-RGB background accuracy falls to 40.50%. The official
test was not evaluated.

## Frozen validation comparison

| Method | Clean | Leak down | BG mean | BG worst | FG mean | FG only |
|---|---:|---:|---:|---:|---:|---:|
| FCAL V2 | 80.08% | 0.266 | 72.04% | 48.92% | 31.83% | 71.67% |
| FCAL V2+ | 79.67% | 0.244 | **73.16%** | **52.67%** | 31.74% | 73.25% |
| FCAL V2.3 | 80.08% | 0.185 | 71.75% | 46.25% | **32.57%** | 73.25% |
| CAMAL | 78.92% | 0.316 | 73.47% | 60.33% | 33.18% | 73.50% |
| **FCAL V2.4** | **80.25%** | **0.145** | 71.62% | 40.50% | 31.76% | **73.67%** |

V2.4 passes three of six frozen performance gates: clean, leakage, and
foreground only. It fails background mean, background worst, and foreground
mean.

## Compute

| Method | Post-branch wall time | Aggregate throughput | Peak VRAM |
|---|---:|---:|---:|
| FCAL V2 | 325.17 s | 796.2 samples/s | 1.7471 GiB |
| CAMAL | 373.86 s | 667.6 samples/s | 1.7452 GiB |
| FCAL V2+ | 377.27 s | 661.2 samples/s | 1.7472 GiB |
| FCAL V2.3 | 470.69 s | 534.5 samples/s | 1.7471 GiB |
| **FCAL V2.4** | **414.68 s** | **609.5 samples/s** | **1.7471 GiB** |

Native-grid growth reduces V2.3's wall time by 11.9% and retains unchanged
memory, but remains 10.9% slower than CAMAL and therefore fails the frozen
compute gate. Late steady-state epochs reached approximately 710--738
samples/s; aggregate accounting includes the slow initial kernel epoch.

## Mechanism diagnostic

On the 1,200-image validation set, the learned normalized target CAM has:

- mean foreground attention: 0.406;
- mean background attention: 0.032;
- foreground pixels above 0.25 attention: 65.5%;
- foreground pixels above 0.50 attention: 40.3%;
- normalized spatial entropy: 0.539;
- connected active fraction at the frozen 0.35 threshold: 59.0%;
- a non-empty growth front for 90.4% of samples.

The method reduced outside evidence but did not accumulate a broad foreground
trail. The loss supervises only the current front; previously active cells
receive no maintenance pressure and can weaken on later updates or augmented
views. Consequently, the model keeps finding a front without producing the
broad persistent evidence intended by the hypothesis. V2's corresponding
strong-foreground coverage is 52.1%, and CAMAL's is 85.6%.

This result also reinforces that low clean-image CAM leakage is not equivalent
to prediction invariance under an off-manifold background replacement. A
future loss-only formulation needs both persistent foreground sufficiency and
a decision-level penalty on label-predictive background features.

## Verification and status

- Input policy: one untouched tensor and exactly one model forward.
- Full test suite: 159 passed.
- Scoped Ruff checks: passed.
- Promotion status: failed; no three-seed replication or test evaluation is
  justified under the frozen protocol.
