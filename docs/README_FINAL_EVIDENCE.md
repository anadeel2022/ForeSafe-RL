# ForeSafe Final Evidence Package v1

This package freezes the manuscript-facing analysis from the completed simulation campaigns. It does not alter or rerun any policy.

## Final proposed algorithm
Temporal partial-observation belief + horizon risk prediction + fixed predictive CVaR shield + support-preserving action filtering + urgency protection.

Adaptive shift-dependent CVaR tightening is retained only as a development ablation because the completed stress-ladder and change-point experiments did not show a consistent outcome benefit.

## Evidence hierarchy
1. **Primary matched evidence:** ForeSafe (fixed CVaR) versus Belief-PPO and Belief-PPO-Lagrangian under the identical partial-observation and terminal-safe protocol.
2. **Factorial ablation:** fixed/adaptive CVaR x temporal-belief/no-belief variants.
3. **Heuristic references:** RQ-Greedy, MaxWeight, Protected-MaxWeight. Protected-MaxWeight is privileged because it uses decision-context state and must not be described as observation matched.
4. **Contextual prior work:** published MobiSafe-PPO mild-minus result. It uses a different information protocol and is not included in paired significance tests.
5. **Development-only observation sweep:** the previous observation-corruption sweep used Adaptive ForeSafe, not the now-selected fixed-CVaR final formulation. It can support development discussion but must not be used to claim fixed-CVaR observation robustness.

## Statistical unit
For learned methods, the independent unit is the training seed (n=10). Episode-level counts are pooled within each seed before across-seed statistics. Paired tests use matched seeds 11-20. Effect sizes and 95% confidence intervals are primary; p-values are secondary.

## Files
- `final_learned_seed_level.csv`: pooled seed-level metrics for all learned variants.
- `final_learned_summary_95ci.csv`: mean, SD and 95% t-CI.
- `paired_tests_vs_final_foresafe.csv`: paired tests against the final fixed-CVaR method.
- `TABLE_main_comparison.csv` / `TABLE_main_results.tex`: main comparison.
- `TABLE_factorial_ablation_long.csv` / `TABLE_factorial_ablation.tex`: ablation evidence.
- `factorial_ablation_effects.csv`: seed-paired factorial contrasts.
- `deterministic_reference_*`: deterministic references.
- `TABLE_stress_ladder_summary.csv`: intermediate severity ladder.
- `TABLE_changepoint_summary.csv`: abrupt-change response.
- `TABLE_observation_sensitivity_development_only.csv`: explicitly development-only.
- `figures/`: publication-ready PNG/PDF figures.
- `MANUSCRIPT_RESULTS_PLAN.md`: proposed Results-section structure and claim language.

No value from the development-only observation sweep should be presented as a final fixed-CVaR result.
