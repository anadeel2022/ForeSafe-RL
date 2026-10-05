# ForeSafe-RL V1.1

This revision fixes integration with the final resource-feasible MobiSafe simulator.

1. The controller now applies the exact RB-budget feasibility mask before every action selection. For K=3 the mask contains 42 feasible actions at R=2 and 30 at R=3.
2. The learned CVaR mask is intersected with the native resource-feasibility mask.
3. Minimum-risk fallback is restricted to physically feasible actions.
4. PPO stores the rollout action mask and reapplies exactly that mask during clipped-policy updates, preserving log-probability consistency.
5. Quantile outputs are ordered to prevent quantile crossing.
6. Shift-monitor updates are aligned with the finite-horizon risk target rather than comparing a horizon prediction with an immediate one-slot outcome.
7. Evaluation resets the online shift monitor for each independent episode.
8. Diagnostics now include the native feasible-action count.
