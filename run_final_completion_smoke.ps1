$ErrorActionPreference = "Stop"
Write-Host "========== VALIDATION =========="
py validate_foresafe.py
py validate_backend.py
py validate_matched_variants.py
py validate_final_completion.py
if ($LASTEXITCODE -ne 0) { throw "Validation failed" }

Write-Host "========== SMOKE: fixed-CVaR no-belief =========="
py run_matched_variant.py `
  --variant foresafe_fixed_cvar_no_belief `
  --seed 11 `
  --train_updates 20 `
  --eval_episodes 10 `
  --train_profile nominal `
  --eval_profiles nominal,mild_minus `
  --out outputs_final_completion_smoke/matched
if ($LASTEXITCODE -ne 0) { throw "Ablation smoke failed" }

Write-Host "========== SMOKE: deterministic references =========="
py benchmark_stress_references_multiseed.py `
  --seeds 11 `
  --profiles nominal,mild_minus `
  --eval_episodes 10 `
  --out outputs_final_completion_smoke/deterministic_references
if ($LASTEXITCODE -ne 0) { throw "Reference smoke failed" }
