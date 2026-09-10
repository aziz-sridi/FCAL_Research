# FCAL V2.3 seed-42 validation report

## Outcome

V2.3 verifies one part of the hypothesis but fails the frozen joint promotion
gate. It preserves V2 clean validation accuracy, matches V2+ foreground-only
accuracy, improves foreground-corruption mean, and lowers strict leakage to
0.185. It does not improve background robustness: mean background accuracy is
71.75% and random-RGB background accuracy falls to 46.25%. The official test
was not evaluated.

## Frozen validation comparison

| Method | Clean | Leak down | BG mean | BG worst | FG mean | FG only |
|---|---:|---:|---:|---:|---:|---:|
| FCAL V2 | **80.08%** | 0.266 | 72.04% | 48.92% | 31.83% | 71.67% |
| FCAL V2+ | 79.67% | 0.244 | **73.16%** | **52.67%** | 31.74% | **73.25%** |
| CAMAL | 78.92% | 0.316 | 73.47% | 60.33% | 33.18% | 73.50% |
| **FCAL V2.3** | **80.08%** | **0.185** | 71.75% | 46.25% | **32.57%** | **73.25%** |

V2.3 passes four of its six frozen performance gates: clean, leakage,
foreground mean, and foreground only. It fails background mean and background
worst.

## Compute

| Method | Post-branch wall time | Aggregate throughput | Peak VRAM |
|---|---:|---:|---:|
| FCAL V2 | 325.17 s | 796.2 samples/s | 1.7471 GiB |
| CAMAL | 373.86 s | 667.6 samples/s | 1.7452 GiB |
| FCAL V2+ | 377.27 s | 661.2 samples/s | 1.7472 GiB |
| **FCAL V2.3** | **470.69 s** | **534.5 samples/s** | **1.7471 GiB** |

V2.3 passes the memory limit but fails the predeclared CAMAL wall-time limit.
The full-resolution differentiable CAM interpolation used by the coverage
hinge is the principal new work. After kernel warm-up, many epochs reached
approximately 700--730 samples/s, but aggregate post-branch timing includes
the slower initial and intermittent epochs and is the frozen accounting basis.

## Mechanism diagnostic

On the 1,200-image validation set, V2.3 produced mean normalized target
attention of 0.463 inside and 0.044 outside the foreground. Its fraction of
foreground pixels above 0.25 attention was 71.7%, but the above-0.50 fraction
was only 49.3%. Thus the loss strongly reduced outside evidence and modestly
broadened weak-to-moderate foreground evidence, but did not reproduce CAMAL's
85.6% strong foreground coverage.

The result falsifies the idea that stronger clean-image containment and CAM
coverage alone will improve the off-manifold random-RGB background case.
Those terms improve localization, clean preservation, and foreground behavior,
but do not directly constrain the class prediction under changed context.

## Verification and status

- Input policy: one untouched training tensor and exactly one forward pass.
- Full test suite: 153 passed.
- Scoped Ruff checks for the changed Python files: passed.
- Repository-wide Ruff still reports unrelated pre-existing notebook and
  research-script findings.
- Promotion status: failed; no three-seed replication or test evaluation is
  justified under the frozen protocol.
