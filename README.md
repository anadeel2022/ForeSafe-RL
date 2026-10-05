# ForeSafe-RL

Official reproducibility code for **ForeSafe-RL: Predictive CVaR-Shielded Reinforcement Learning under Partial Observability for Reliable Vehicular IoT Scheduling**.

## Final manuscript method

The proposed method used for the final manuscript is the **fixed-CVaR ForeSafe-RL** configuration implemented through `run_matched_variant.py --variant foresafe_fixed_cvar`. The final formulation combines a temporal belief representation, action-conditioned finite-horizon reliability-risk prediction, a fixed predictive CVaR safety shield, support-preserving action filtering, and urgency protection.

The adaptive shift-tightening branch is retained only as development/ablation material and should not be interpreted as the final proposed algorithm.

Key design settings include a reliability target of `0.05`, risk horizon `12`, CVaR level `0.8`, fixed CVaR threshold `0.10`, hard-shield activation after `15000` transitions, minimum matured action support `25`, required supported-action fraction `0.75`, and minimum operational action fraction `0.20`.

## Repository contents

- `foresafe_*.py`: ForeSafe configuration, observation processing, PPO implementation, and risk predictor.
- `viot_*.py`: validated RSU-assisted vehicular-IoT simulator and constraints.
- `run_matched_variant.py`: final proposed method, observation-matched learning baselines, and ablations.
- `matched_baselines.py`: Belief-PPO and Belief-PPO-Lagrangian baselines.
- `benchmark_stress_references_multiseed.py`: deterministic reference schedulers.
- `run_final_completion_*.ps1`: smoke, pilot, and final completion commands.
- `results/summary/`: compact seed-level results, confidence intervals, paired tests, and final manuscript tables.
- `figures/`: manuscript-ready figures generated from the final evidence package.
- `docs/`: development diagnostics, robustness notes, and historical changelogs.

Large raw result archives are intentionally not stored in the repository. The compact seed-level evidence needed to verify the reported manuscript statistics is included under `results/summary/`.

## Environment

The final development environment used Python 3.13 with CPU PyTorch. Install the minimum declared dependencies with:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Validation

Run the structural and backend checks before any training:

```powershell
py validate_foresafe.py
py validate_backend.py
py validate_matched_variants.py
py validate_final_completion.py
```

Expected output includes successful ForeSafe structural validation, backend compatibility, matched-experiment validation, and final-completion validation.

## Final ForeSafe-RL experiment

The final proposed method is `foresafe_fixed_cvar`.

Single-seed example:

```powershell
py run_matched_variant.py `
  --variant foresafe_fixed_cvar `
  --seed 11 `
  --train_updates 1200 `
  --eval_episodes 200 `
  --train_profile nominal `
  --eval_profiles nominal,mild_minus,standard_compound,compound_severe
```

The manuscript uses ten independent training seeds, `11` through `20`. The independent statistical unit is the training seed, not the individual evaluation episode.

## Observation-matched learning baselines

Belief-PPO:

```powershell
py run_matched_variant.py --variant belief_ppo --seed 11 --train_updates 1200 --eval_episodes 200 --train_profile nominal --eval_profiles nominal,mild_minus,standard_compound,compound_severe
```

Belief-PPO-Lagrangian:

```powershell
py run_matched_variant.py --variant belief_ppo_lagrangian --seed 11 --train_updates 1200 --eval_episodes 200 --train_profile nominal --eval_profiles nominal,mild_minus,standard_compound,compound_severe
```

## Main ablation

The temporal-belief ablation for the final fixed-CVaR formulation is:

```powershell
py run_matched_variant.py --variant foresafe_fixed_cvar_no_belief --seed 11 --train_updates 1200 --eval_episodes 200 --train_profile nominal --eval_profiles nominal,mild_minus,standard_compound,compound_severe
```

## Deterministic references

The deterministic comparison includes RQ-Greedy, MaxWeight, and Protected-MaxWeight. RQ-Greedy corresponds to the simulator's historical `oracle_action` implementation but is not clairvoyant. Protected-MaxWeight uses privileged decision context and is therefore a reference heuristic, not an observation-matched learning baseline.

Run the ten-seed deterministic reference evaluation with:

```powershell
py benchmark_stress_references_multiseed.py --seeds 11,12,13,14,15,16,17,18,19,20 --profiles nominal,mild_minus,standard_compound,compound_severe --eval_episodes 50 --out outputs_final_completion/deterministic_references
```

## Evaluation protocol

The implementation retains the final protocol used in the manuscript: uninterrupted continuing-task training, 150-transition PPO rollouts, critic bootstrap at nonterminal rollout cuts, denominator-correct finite-horizon labels spanning rollout boundaries, a 150-slot tagged measurement cohort followed by a 251-slot continuous-traffic follow-up, and matched held-out evaluation seeds.

Reliability and service metrics are reported separately. The formal reliability requirement is interruption probability `<= 0.05`. The inherited deadline-miss reference is not treated as a second formal optimization constraint.

## Statistical reporting

Main learned-method results are summarized across the ten training seeds using mean and 95% t-confidence intervals. Paired statistical comparisons use matched seeds `11` through `20`. See `results/summary/final_learned_seed_level.csv`, `final_learned_summary_95ci.csv`, and `paired_tests_vs_final_foresafe.csv`.

## Reproducibility note

The repository intentionally excludes smoke-test archives, pilot archives, intermediate checkpoints, cached Python bytecode, and large raw result folders. These are not required to inspect the implementation or reproduce the final experiment protocol.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
