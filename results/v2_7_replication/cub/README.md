# FCAL V2.7 ResNet-50 seed-7 replication

## Decision

**Stop.** The frozen seed-7 triage screen failed the clean-accuracy and
background-random-RGB hard gates. The remaining 27 validation views were not
run, and seed 123 was not started.

This is evidence that the strong seed-42 result does not reproduce cleanly on
seed 7. It does not overturn the seed-42 result, but V2.7 cannot yet be called a
robust replacement for V2 across seeds.

## Frozen setup

- Dataset/split: CUB-200-2011, the existing fixed validation split
  `splits/cub200_2011_seed42_val0.2`
- Architecture: ResNet-50
- Seed: 7
- Branch checkpoint: the paired seed-7 CE epoch-5 checkpoint
- V2.7 coefficient: `0.02`, unchanged from the successful seed-42 run
- Candidate checkpoint selection: clean validation accuracy
- Comparators: same-seed FCAL V2, CE, FCAL V2+, GAIN-ext, RRR, and
  Faster-ISNet. Same-seed CAMAL was unavailable and was not substituted with a
  seed-42 result.

## Triage result

| Metric | V2.7 seed 7 | V2 seed 7 | Difference | Frozen hard gate | Result |
|---|---:|---:|---:|---:|---|
| Clean accuracy | 78.75% | 79.83% | -1.08 pp | >=79.58% | fail |
| BG random RGB accuracy | 56.33% | 57.00% | -0.67 pp | >=57.00% | fail |
| Foreground-only accuracy | 71.17% | 70.25% | +0.92 pp | >=70.25% | pass |
| Strict leakage | 0.26787 | 0.27196 | -0.00409 | <=0.27196 | pass |
| Post-branch time | 539.26 s | 524.31 s | +2.9% | <=602.95 s | pass |
| Peak CUDA memory | 1,875,995,136 B | 1,873,875,456 B | +0.1% | <=2,061,263,002 B | pass |

Only foreground-only accuracy met its predefined material-improvement
threshold. The screen required at least two material improvements, including a
background/leakage improvement.

The candidate was not Pareto-dominated in the triage metric space and met every
predeclared strict-win count against the available competition. Those outcomes
do not rescue it because hard-gate and material-improvement requirements are
conjunctive.

## Resource decision

Per the staged protocol, a failed seed-7 triage ends this branch. No coefficient
tuning is performed on seed 7 because that would convert a frozen replication
into a seed-specific search. The next useful research action is to diagnose the
seed-42/seed-7 variance or test a new mechanism on one seed, rather than spend a
third full run attempting to average away the failure.

Machine-readable screen: `resnet50_seed7_triage.json`.
