# Manuscript Results Plan

## R1. Verification and experimental protocol
State the 10 independent training seeds, 1,200 updates x 150 transitions = 180,000 training transitions per seed, 200 terminal-safe evaluation episodes per profile, and the packet-conservation/resource audits. Keep the reliability requirement P_int <= 0.05 distinct from the inherited deadline-reference signal.

## R2. Main comparison under increasing physical stress
Use Table `TABLE_main_results.tex` and Figures 1-2.
Primary statistical comparison: fixed-CVaR ForeSafe versus Belief-PPO and Belief-PPO-Lagrangian.
Do not call Protected-MaxWeight a matched baseline; identify it as a privileged heuristic reference.

Recommended claim:
“Across the matched recurrent learning baselines, predictive CVaR filtering lowers mean interruption probability in all four evaluation regimes while retaining comparable service delivery. The reliability target is satisfied by all seeds under nominal and mild-minus conditions, whereas the standard and severe regimes exceed the feasible reliability envelope for all evaluated schedulers.”

## R3. Component ablation
Use `TABLE_factorial_ablation.tex` and Figure 5.
The temporal-belief result should be framed as predictive discrimination, not universal metric dominance. Under fixed CVaR, belief lowers mean interruption in each profile and produces substantially stronger stress-dependent contraction of the strict risk-passing action set than the no-belief variant.

Adaptive tightening should be reported as a negative ablation:
“The shift-dependent tightening mechanism altered the admissible set as designed but did not yield a consistent end-to-end reliability advantage over the fixed-CVaR formulation. We therefore retain the fixed formulation in the final method.”

## R4. Reliability envelope
Use Figure 3 and `TABLE_stress_ladder_summary.csv`.
Report the intermediate ladder as an empirical robustness envelope, not an arbitrary-shift guarantee. Do not describe the method as distributionally robust.

## R5. Abrupt shift
Use Figure 4 and `TABLE_changepoint_summary.csv`.
The main result is diagnostic: adaptive tightening responds to the change but the response does not reduce post-change failures relative to fixed CVaR. This supports removal of the adaptive branch rather than a performance claim.

## R6. Heuristic context
Use Figure 6.
RQ-Greedy and MaxWeight provide conventional references. Protected-MaxWeight is a privileged decision-context heuristic and should be described explicitly as such. Its strong result establishes a useful reference bound, not a matched-policy comparison.

## R7. Observation corruption
Keep the existing sweep in supplementary/development discussion only because it evaluated Adaptive ForeSafe rather than the selected fixed-CVaR final method. If the paper needs a direct fixed-CVaR observation-robustness claim, run a small evaluation-only sweep using the already-trained fixed-CVaR checkpoints; no retraining is needed.

## Claims to avoid
- “ForeSafe outperforms every scheduler.”
- “Distributionally robust.”
- “Adaptive shift tightening improves reliability.”
- “Protected-MaxWeight is an observation-matched baseline.”
- Any formal deadline-miss constraint at 0.25.
- Any significance claim based on episode count instead of n=10 training seeds.
