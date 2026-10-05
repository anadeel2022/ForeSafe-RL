# V1.4 600-Update Failure Diagnosis

Observed run:
`20260923_133958_train-nominal_eval-nominal+mild_minus_seed11`

## Terminal-safe evaluation

Nominal:
- pooled interruption probability: 0.033166
- pooled on-time delivery ratio: 0.537829
- pooled deadline-miss ratio: 0.462171
- mean delivered traffic: 0.642016 Mbps
- audit: passed

Mild-minus:
- pooled interruption probability: 0.040637
- pooled on-time delivery ratio: 0.539531
- pooled deadline-miss ratio: 0.460469
- mean delivered traffic: 0.646646 Mbps
- audit: passed

## Diagnosed mechanism

The hard learned shield became active at 1,500 transitions. At the first
post-warm-up rollout, the mean chosen CVaR was still approximately 0.187 while
the nominal limit was 0.10. No native feasible action passed the learned CVaR
threshold. The fallback therefore reduced policy support to one action.

Across updates 51-100, 101-200, and 201-300, the mean strict CVaR-admissible
action count was effectively zero and the forced minimum-risk fallback rate was
approximately 1.0. Only much later did a small number of actions fall below the
threshold.

At evaluation after 600 updates, only about 5.6 of the 42 resource-feasible
actions were CVaR-admissible before the urgency guard, and the resulting
deferral probability was about 0.55. This explains the combination of low
interruption and poor deadline completion.

The service guard also used a mis-scaled global age quantity in V1.4. The final
MobiSafe observation already provides candidate-level age/own-deadline ratios,
which V1.5 now uses directly.

## Consequence

The 600-update V1.4 result is retained as a diagnostic failure case. It should
not be used as the proposed-method result and it should not motivate relaxing
the reliability target. V1.5 corrects the evidence and service-control logic
before additional long simulations.
