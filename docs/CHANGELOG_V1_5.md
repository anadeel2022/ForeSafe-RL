# ForeSafe-RL V1.5

V1.5 is a corrective release based on the V1.4 600-update forensic audit.

1. Fixes premature hard-CVaR activation that caused zero admissible actions and
   repeated minimum-risk single-action fallback.
2. Adds per-action empirical support accounting in the risk replay.
3. Learned CVaR can hard-reject only sufficiently supported actions.
4. Adds a 15,000-transition minimum hard-shield warm-up.
5. Requires a configurable minimum fraction of supported native actions before
   hard-shield activation during training.
6. Adds a risk-ranked minimum operational action-set fraction to prevent
   policy-support collapse.
7. Fixes the urgency signal to use candidate-level age/own-deadline ratios.
8. Shift monitoring ignores unsupported action predictions.
9. Soft predictive penalties are support-qualified.
10. Adds shield-activation, empirical-support, risk-rejection, risk-relaxation,
    and urgency diagnostics.
11. Retains the V1.4 continuing-task and terminal-safe evaluation protocol.
12. Retains the V1.3 denominator-correct future interruption target and
    quantile-integrated CVaR definition.
