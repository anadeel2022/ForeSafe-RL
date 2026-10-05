# ForeSafe-RL V1.4

Protocol-aligned pre-pilot release.

1. Keeps the V1.3 algorithmic risk definition and safety action rule.
2. Adds continuing-task ForeSafe training with exactly one environment reset.
3. PPO rollout boundaries are nonterminal and use critic-value bootstrap.
4. Denominator-correct H-slot risk labels span PPO rollout boundaries; only the
   final incomplete H-1 windows are dropped.
5. Replaces replay-wide predictor sweeps with a fixed 12-minibatch SGD budget
   per PPO update by default.
6. Adds final MobiSafe-aligned terminal-safe evaluation: 150-slot measurement
   cohort plus 251-slot follow-up with background arrivals active.
7. Uses the same matched held-out evaluation seed rule as the final MobiSafe
   reproducibility campaign.
8. Adds exact `mild_minus`, `standard_compound`, and `compound_severe` physical
   stress profiles from the final MobiSafe runner.
9. Separates training and evaluation profiles and allows several evaluation
   profiles from one trained checkpoint.
10. Adds automatic cohort-conservation, pending-packet, resource-budget, and
    invalid-action audits.

11. Adds `validate_backend.py` to reject older MobiSafe environment files that lack terminal-safe cohort accounting.
