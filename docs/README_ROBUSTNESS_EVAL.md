# ForeSafe-RL V1.5 Robustness Evaluation Package v1

This package performs evaluation-only experiments on the already trained, frozen 1200-update checkpoints. It does not retrain or modify the final ForeSafe-RL V1.5 algorithm or any matched baseline.

## Required local folder layout

The supplied PowerShell scripts assume these sibling folders:

```
ForeSafe_RL\
  ForeSafe_RL_V1_5\
    outputs_foresafe_v15\
  ForeSafe_RL_V1_5_Matched_Experiments_v1\
    outputs_matched_v15\
  ForeSafe_RL_V1_5_Robustness_Eval_v1\
```

The discovery code reads checkpoint metadata and accepts only one 1200-update final checkpoint per method and seed 11--20. Smoke, pilot, and 600-update midpoint checkpoints are ignored.

## Experiments

### 1. Observation impairment

Methods: full ForeSafe, ForeSafe without temporal belief (`history_len=1`), and recurrent Belief-PPO.

Physical channel: nominal. The trained observation point is delay=2 slots, Gaussian noise std=0.02, dropout=0.05. One factor is changed at a time:

- delay: 0, 2, 4, 8 slots;
- noise std: 0, 0.02, 0.05, 0.10;
- dropout: 0, 0.05, 0.15, 0.30.

The checkpoint-specific history length is preserved. Thus the no-belief ablation remains history length 1; all recurrent belief methods retain their trained history length.

### 2. Stress-severity ladder

Methods: full ForeSafe, Belief-PPO, Belief-PPO-Lagrangian, no-belief ForeSafe, and fixed-CVaR ForeSafe.

The final mild-minus and standard-compound profiles are treated as fixed endpoints. Only three new intermediate profiles are evaluated, at alpha=0.25, 0.50, and 0.75, using linear interpolation of the already frozen stress parameters. Stale CSI is rounded to the nearest integer slot. The alpha=0 and alpha=1 endpoint results are reused from the final 200-episode campaign rather than rerun.

### 3. Abrupt change point

Methods: full ForeSafe and fixed-CVaR ForeSafe.

Each episode starts under mild-minus conditions. At slot 75, the realized channel law changes abruptly to standard-compound-like shadowing, blockage, interference, and stale CSI. The environment is not reset: queues, packets, battery state, positions, sampled velocities, and observation history continue. Mobility is deliberately not resampled at the change point, so the test isolates online channel/CSI distribution-shift adaptation.

Outputs include per-episode pre/post windows and a compact per-slot time series for the CVaR limit, shift error, strict admissible action count, selected-action deferral fraction, and attempted-transmission failures.

## First run: smoke test

From this package folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_robustness_smoke.ps1
```

This evaluates only seed 11 with 5 episodes per condition. Inspect/share the smoke output before starting the final package.

## Final run

After smoke validation:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_robustness_final.ps1
```

The final secondary robustness experiments use 100 episodes per seed/condition for seeds 11--20. They are explicitly secondary sensitivity experiments; the primary final method comparison remains the 200-episode terminal-safe campaign.

## Reproducibility principles

- No training occurs in these scripts.
- No hyperparameter is selected using the robustness outcomes.
- Final checkpoint discovery requires exactly 1200 training updates.
- Evaluation seeds retain the final rule `7,000,000 + train_seed*10,000 + episode_index`.
- Packet accounting and resource-feasibility audits are run for every static evaluation.
- The frozen code files are checked against `code_sha256_manifest.csv`.
