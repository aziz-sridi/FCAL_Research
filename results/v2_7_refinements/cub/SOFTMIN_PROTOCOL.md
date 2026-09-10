# FCAL V2.7d-lite soft-min rival protocol

The exact hard-tail V2.7a and channel-resolved V2.7b candidates were stopped
by their frozen compute screen before performance evaluation. V2.7d-lite tests
the still-unresolved distributed-foreground hypothesis with a cheaper smooth
estimator.

After one untouched-image backbone forward, V2.7d selects the strongest
non-target class from detached clean logits and obtains its native 7x7 CAM
from the existing feature tensor. Target and rival signed CAMs are independently
RMS-normalized. Their bounded local margin is:

```text
d(p) = tanh(S_target(p) - S_rival(p))
q = -0.25 * log(mean_foreground(exp(-d(p) / 0.25)))
L_softmin = 0.25 * softplus((0 - q) / 0.25)
```

V2.7's foreground-pooled CE remains at weight 0.02. `L_softmin` has frozen
weight 0.01; its calibrated weighted magnitude was 0.00269 at the common
epoch-5 branch, less than half the 0.00629 V2.7 foreground-pool contribution.
The candidate uses one rival, no sort, no second backbone forward, no input
change, and no inference change.

Run conditions, six advancement/final performance thresholds, compute limits,
and test-data policy are identical to `VALIDATION_PROTOCOL.md`. No parameter
was selected using V2.7a/V2.7b performance because those runs ended at the
compute screen and were not stress-evaluated.
