# Regional stress tests

FCAL is evaluated with interventions that change one annotated region while
copying the other region bit-for-bit. This is stricter than applying a normal
whole-image corruption: a background test cannot accidentally blur, recolor,
or add noise to the object being classified.

## Common 30-view protocol

Every dataset now uses the same complete protocol:

- clean and sham-recomposition controls;
- 13 background interventions: three blur levels, three color-jitter levels,
  grayscale, three Gaussian-noise levels, solid mean, random solid, and random
  RGB;
- the same 13 interventions applied only to the foreground;
- foreground-only and background-only region ablations.

`BG mean` and `FG mean` always average exactly those 13 common interventions.
`BG worst` and `FG worst` first take the lowest of the 13 accuracies for each
architecture, then average those architecture-level worst cases. Tailored
scenarios are reported separately and never silently folded into these means.

## Dataset-tailored scenarios

These deterministic simulations probe recognizable risks, but they are not
claimed to be samples from a natural deployment distribution.

| Dataset | Scenario | Type | What remains exact | Intended question |
|---|---|---|---|---|
| CUB | Twilight habitat | Realistic | Bird | Does a dawn/dusk color and illumination shift in the scene change the species decision? |
| CUB | Camouflage background | Extreme | Bird | What happens when the background color is deliberately matched to the bird with smooth texture? |
| Oxford Pet | Warm indoor light | Realistic | Pet | Is the breed prediction stable when only household lighting and vignette change? |
| Oxford Pet | Saturated checkerboard | Extreme | Pet | Can a high-frequency background distractor override evidence from the animal? |
| ISIC | Hair artifact | Realistic | Lesion | Is the diagnosis stable when dark hair-like curves appear only on surrounding skin? |
| ISIC | Ruler artifact | Extreme | Lesion | Can a high-contrast acquisition marker outside the lesion become a shortcut? |
| ISIC | Low-contrast lesion | Extreme foreground | Surrounding skin | How does performance change when lesion contrast is strongly reduced without altering background pixels? |

All stochastic values are derived from the global seed, image identifier, and
variant name. Re-running a scenario therefore produces the same tested image.

## Interpretation limits

Regional replacement creates its own distribution shift. A strong result
shows reduced sensitivity to the tested intervention, not causal proof that a
heatmap is faithful. Clean accuracy, leakage, region-only ablations, common
corruptions, and tailored scenarios should be read together.
