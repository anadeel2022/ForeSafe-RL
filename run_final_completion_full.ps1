$ErrorActionPreference = "Stop"
foreach ($s in 11..20) {
  Write-Host "========== FINAL fixed-CVaR no-belief | seed $s =========="
  py run_matched_variant.py `
    --variant foresafe_fixed_cvar_no_belief `
    --seed $s `
    --train_updates 1200 `
    --eval_episodes 200 `
    --train_profile nominal `
    --eval_profiles nominal,mild_minus,standard_compound,compound_severe `
    --out outputs_final_completion
  if ($LASTEXITCODE -ne 0) { throw "Ablation final failed for seed $s" }
}

Write-Host "========== FINAL deterministic references =========="
py benchmark_stress_references_multiseed.py `
  --seeds 11,12,13,14,15,16,17,18,19,20 `
  --profiles nominal,mild_minus,standard_compound,compound_severe `
  --eval_episodes 50 `
  --out outputs_final_completion/deterministic_references
if ($LASTEXITCODE -ne 0) { throw "Deterministic final failed" }
