# Robustness decision after the final evaluation campaign

The final robustness campaign supports simplifying the proposed method to the fixed-CVaR core. The adaptive shift-tightening branch is retained only as a development/ablation result, not as part of the final proposed algorithm.

Key evidence from ten training seeds:

- In the mild-to-standard stress ladder, the mean interruption probability of full adaptive ForeSafe-RL was 0.04915, 0.06388, and 0.08823 at alpha 0.25, 0.50, and 0.75. The fixed-CVaR variant achieved 0.04167, 0.05914, and 0.08076 at the same levels.
- The approximate mean reliability-envelope crossing P_int=0.05 occurs at alpha about 0.264 for the adaptive variant and alpha about 0.369 for fixed-CVaR.
- In the abrupt mild-to-standard change-point experiment, adaptive ForeSafe-RL visibly contracts the CVaR limit, but this did not reduce post-change interruption relative to fixed-CVaR.
- The nominal observation-impairment sweep produced only small changes in the frozen policies. It is therefore evidence of nominal impairment robustness, not strong evidence that temporal belief is necessary.

The final core is therefore: temporal partial-observation history + horizon risk predictor + fixed predictive CVaR shield + support-preserving action filtering + urgency protection. The shift monitor may still be reported as an explored extension whose outcome was negative, but it should not be claimed as a performance-improving contribution.
