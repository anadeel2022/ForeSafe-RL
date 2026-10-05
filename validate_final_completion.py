from __future__ import annotations

from argparse import Namespace
import numpy as np

from run_matched_variant import VARIANTS, variant_configs
from viot_env import EnvConfig, VIoTEnv


def fake_args():
    return Namespace(
        variant="foresafe_fixed_cvar_no_belief",
        history_len=8, obs_delay=2, obs_noise=0.02, dropout=0.05,
        risk_horizon=12, cvar_alpha=0.80, cvar_limit=0.10,
        warmup_steps=1500, shield_warmup_steps=15000,
        min_supported_action_fraction=0.75, predictor_gradient_steps=12,
        urgency_guard_threshold=0.85, min_action_samples=25,
        min_operational_action_fraction=0.20, device="cpu",
    )


def main():
    assert "foresafe_fixed_cvar_no_belief" in VARIANTS
    po, risk, safety, ppo = variant_configs(fake_args())
    assert po.history_len == 1
    assert abs(risk.shift_tightening_gain) < 1e-12
    assert abs(risk.base_cvar_limit - 0.10) < 1e-12
    assert safety.min_operational_action_fraction == 0.20

    env=VIoTEnv(EnvConfig(seed=123,steps_per_episode=5,measurement_cohort_slots=0))
    env.reset(seed=123)
    ctx=env.decision_context()
    assert int(ctx["oracle_action"]) == int(env._build_oracle_action(priority=False))
    assert int(ctx["maxweight_action"]) == int(env._build_oracle_action(priority=True))
    assert int(ctx["protected_maxweight_action"]) == int(env._build_protected_maxweight_action())
    mask=np.asarray(env.valid_action_mask(),dtype=bool)
    assert mask.sum() > 0
    print("ForeSafe-RL final-completion validation passed.")
    print("fixed-CVaR/no-belief ablation is configured as history_len=1, shift_tightening_gain=0.")
    print("direct deterministic builders match decision_context references.")

if __name__ == "__main__":
    main()
