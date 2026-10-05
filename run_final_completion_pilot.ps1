$ErrorActionPreference = "Stop"
py run_matched_variant.py `
  --variant foresafe_fixed_cvar_no_belief `
  --seed 11 `
  --train_updates 200 `
  --eval_episodes 50 `
  --train_profile nominal `
  --eval_profiles nominal,mild_minus,standard_compound,compound_severe `
  --out outputs_final_completion_pilot
if ($LASTEXITCODE -ne 0) { throw "Ablation pilot failed" }
