
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from foresafe_config import PartialObservationConfig, RiskConfig, SafetyConfig, PPOConfig
from foresafe_observation import PartialObservationWrapper
from foresafe_ppo import train_foresafe_continuing
from matched_baselines import (
    LagrangianConfig,
    train_recurrent_ppo_continuing,
    train_recurrent_ppo_lagrangian_continuing,
)
from run_foresafe import (
    make_stress,
    parse_profiles,
    evaluation_seed,
    pooled_summary,
    audit_rows,
    evaluate_profile,
    write_rows,
)
from viot_env import EnvConfig, VIoTEnv


VARIANTS = (
    "belief_ppo",
    "belief_ppo_lagrangian",
    "foresafe_no_belief",
    "foresafe_fixed_cvar",
    "foresafe_fixed_cvar_no_belief",
    "foresafe_no_support_floor",
    "foresafe_no_urgency_guard",
)


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def evaluate_basic_profile(
    policy,
    env_factory,
    episodes: int,
    train_seed: int,
    profile_name: str,
    deterministic: bool,
) -> list[dict]:
    rows = []
    for ep in range(int(episodes)):
        seed = evaluation_seed(train_seed, ep)
        env = env_factory(profile_name, seed)
        history, _ = env.reset(seed=seed)
        done = False

        while not done:
            action = policy.act(env, history, deterministic=deterministic)
            history, _, done, _ = env.step(action)

        row = dict(env.episode_metrics())
        row.update({
            "train_seed": int(train_seed),
            "evaluation_seed": int(seed),
            "evaluation_episode": int(ep),
            "evaluation_profile": str(profile_name),
            "observation_dropouts": int(env.diagnostics.dropouts),
            "observation_noisy_updates": int(env.diagnostics.noisy_observations),
            "observation_delayed_updates": int(env.diagnostics.delayed_observations),
        })
        rows.append(row)
    return rows


def variant_configs(args):
    po_cfg = PartialObservationConfig(
        history_len=int(args.history_len),
        observation_delay_slots=int(args.obs_delay),
        gaussian_noise_std=float(args.obs_noise),
        dropout_probability=float(args.dropout),
    )
    risk_cfg = RiskConfig(
        horizon=int(args.risk_horizon),
        cvar_alpha=float(args.cvar_alpha),
        base_cvar_limit=float(args.cvar_limit),
        warmup_steps=int(args.warmup_steps),
        shield_warmup_steps=int(args.shield_warmup_steps),
        min_supported_action_fraction=float(args.min_supported_action_fraction),
        predictor_gradient_steps_per_update=int(args.predictor_gradient_steps),
    )
    safety_cfg = SafetyConfig(
        urgency_guard_threshold=float(args.urgency_guard_threshold),
        min_action_samples_for_filter=int(args.min_action_samples),
        min_operational_action_fraction=float(args.min_operational_action_fraction),
    )
    ppo_cfg = PPOConfig(device=str(args.device))

    if args.variant == "foresafe_no_belief":
        po_cfg = replace(po_cfg, history_len=1)
    elif args.variant == "foresafe_fixed_cvar":
        risk_cfg = replace(risk_cfg, shift_tightening_gain=0.0)
    elif args.variant == "foresafe_fixed_cvar_no_belief":
        po_cfg = replace(po_cfg, history_len=1)
        risk_cfg = replace(risk_cfg, shift_tightening_gain=0.0)
    elif args.variant == "foresafe_no_support_floor":
        safety_cfg = replace(
            safety_cfg,
            min_operational_action_fraction=0.0,
            min_safe_actions=1,
        )
    elif args.variant == "foresafe_no_urgency_guard":
        safety_cfg = replace(safety_cfg, urgency_guard_threshold=99.0)

    return po_cfg, risk_cfg, safety_cfg, ppo_cfg


def main():
    p = argparse.ArgumentParser(
        description="Matched ForeSafe-RL V1.5 baselines and ablations"
    )
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--train_updates", type=int, default=200)
    p.add_argument("--rollout_steps", type=int, default=150)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--n_devices", type=int, default=36)
    p.add_argument("--n_channels", type=int, default=3)
    p.add_argument(
        "--train_profile",
        choices=["nominal", "mild_minus", "standard_compound", "compound_severe"],
        default="nominal",
    )
    p.add_argument(
        "--eval_profiles",
        default="nominal,mild_minus,standard_compound,compound_severe",
    )

    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--obs_delay", type=int, default=2)
    p.add_argument("--obs_noise", type=float, default=0.02)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--risk_horizon", type=int, default=12)
    p.add_argument("--cvar_alpha", type=float, default=0.80)
    p.add_argument("--cvar_limit", type=float, default=0.10)
    p.add_argument("--warmup_steps", type=int, default=1500)
    p.add_argument("--shield_warmup_steps", type=int, default=15000)
    p.add_argument("--min_supported_action_fraction", type=float, default=0.75)
    p.add_argument("--min_action_samples", type=int, default=25)
    p.add_argument("--min_operational_action_fraction", type=float, default=0.20)
    p.add_argument("--urgency_guard_threshold", type=float, default=0.85)
    p.add_argument("--predictor_gradient_steps", type=int, default=12)

    p.add_argument("--lagrangian_dual_lr", type=float, default=0.10)
    p.add_argument("--lagrangian_lambda_max", type=float, default=5.0)

    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--stochastic_eval", action="store_true")
    p.add_argument("--out", default="outputs_matched_v15")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if args.train_updates <= 0 or args.rollout_steps <= 0:
        raise ValueError("train_updates and rollout_steps must be positive.")
    if args.measurement_slots <= 0 or args.followup_slots <= 0:
        raise ValueError("measurement_slots and followup_slots must be positive.")

    eval_profiles = parse_profiles(args.eval_profiles)
    po_cfg, risk_cfg, safety_cfg, ppo_cfg = variant_configs(args)

    label = "+".join(eval_profiles)
    out = Path(args.out) / (
        f"{stamp()}_{args.variant}_train-{args.train_profile}"
        f"_eval-{label}_seed{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)

    train_horizon = (int(args.train_updates) + 1) * int(args.rollout_steps)
    train_env_cfg = EnvConfig(
        seed=int(args.seed),
        n_devices=int(args.n_devices),
        n_channels=int(args.n_channels),
        steps_per_episode=int(train_horizon),
        measurement_cohort_slots=0,
        arrival_cutoff_slots=0,
        stress=make_stress(args.train_profile),
    )
    train_env = PartialObservationWrapper(
        VIoTEnv(train_env_cfg), po_cfg, seed=int(args.seed)
    )

    foresafe_variant = args.variant.startswith("foresafe_")

    lag_cfg = None
    if args.variant == "belief_ppo":
        result = train_recurrent_ppo_continuing(
            train_env,
            ppo_cfg,
            rollout_updates=int(args.train_updates),
            rollout_steps=int(args.rollout_steps),
            seed=int(args.seed),
        )
        checkpoint = {
            "variant": args.variant,
            "actor_critic": result["actor_critic"].state_dict(),
        }
    elif args.variant == "belief_ppo_lagrangian":
        lag_cfg = LagrangianConfig(
            cost_limit=0.05,
            dual_learning_rate=float(args.lagrangian_dual_lr),
            lambda_max=float(args.lagrangian_lambda_max),
        )
        result = train_recurrent_ppo_lagrangian_continuing(
            train_env,
            ppo_cfg,
            lag_cfg,
            rollout_updates=int(args.train_updates),
            rollout_steps=int(args.rollout_steps),
            seed=int(args.seed),
        )
        checkpoint = {
            "variant": args.variant,
            "actor_critic": result["actor_critic"].state_dict(),
            "lambda_value": float(result["lambda_value"]),
            "lagrangian_cfg": asdict(lag_cfg),
        }
    else:
        result = train_foresafe_continuing(
            train_env,
            ppo_cfg,
            risk_cfg,
            safety_cfg,
            rollout_updates=int(args.train_updates),
            rollout_steps=int(args.rollout_steps),
            seed=int(args.seed),
        )
        checkpoint = {
            "variant": args.variant,
            "actor_critic": result["actor_critic"].state_dict(),
            "risk_model": result["risk_model"].state_dict(),
            "action_support_counts": result["action_support_counts"].tolist(),
        }

    checkpoint.update({
        "seed": int(args.seed),
        "train_env_cfg": train_env_cfg.to_dict(),
        "partial_observation_cfg": asdict(po_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "risk_cfg": asdict(risk_cfg),
        "safety_cfg": asdict(safety_cfg),
        "training_protocol": result["training_protocol"],
    })
    torch.save(checkpoint, out / "checkpoint.pt")

    history = result["history"]
    rollout_rows = history.get("rollout_metrics", [])
    write_rows(out / "training_rollout_metrics.csv", rollout_rows)
    diagnostics = {
        k: v for k, v in history.items()
        if k != "rollout_metrics"
    }
    with (out / "training_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    def eval_env_factory(profile_name: str, seed: int):
        cfg = EnvConfig(
            seed=int(seed),
            n_devices=int(args.n_devices),
            n_channels=int(args.n_channels),
            steps_per_episode=int(args.measurement_slots + args.followup_slots),
            measurement_cohort_slots=int(args.measurement_slots),
            arrival_cutoff_slots=0,
            stress=make_stress(profile_name),
        )
        return PartialObservationWrapper(
            VIoTEnv(cfg), po_cfg, seed=int(seed) + 991
        )

    summaries = {}
    audits = {}
    for profile in eval_profiles:
        if foresafe_variant:
            rows = evaluate_profile(
                result["policy"],
                eval_env_factory,
                int(args.eval_episodes),
                int(args.seed),
                profile,
                deterministic=not args.stochastic_eval,
            )
        else:
            rows = evaluate_basic_profile(
                result["policy"],
                eval_env_factory,
                int(args.eval_episodes),
                int(args.seed),
                profile,
                deterministic=not args.stochastic_eval,
            )

        write_rows(out / f"evaluation_{profile}_episodes.csv", rows)
        summary = pooled_summary(rows)
        audit = audit_rows(rows, int(args.n_channels))

        summary["reliability_constraint_satisfied_pooled"] = bool(
            summary["pooled_cohort_interruption_probability"] <= 0.05
        )
        summary["deadline_reference_satisfied_pooled"] = bool(
            summary["pooled_cohort_packet_deadline_miss_ratio"] <= 0.25
        )
        summary["integrity_audit_passed"] = bool(audit["passed"])

        summaries[profile] = summary
        audits[profile] = audit

    with (out / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with (out / "evaluation_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)

    manifest = {
        "experiment_package": "ForeSafe-RL V1.5 matched experiments v1",
        "frozen_foresafe_core_modified": False,
        "variant": args.variant,
        "seed": int(args.seed),
        "train_profile": args.train_profile,
        "eval_profiles": eval_profiles,
        "train_updates": int(args.train_updates),
        "rollout_steps": int(args.rollout_steps),
        "total_training_steps": int(result["total_steps"]),
        "eval_episodes_per_profile": int(args.eval_episodes),
        "measurement_cohort_slots": int(args.measurement_slots),
        "followup_slots": int(args.followup_slots),
        "background_arrivals_during_followup": True,
        "evaluation_seed_rule": "7000000 + train_seed*10000 + episode_index",
        "partial_observation_cfg": asdict(po_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "risk_cfg": asdict(risk_cfg),
        "safety_cfg": asdict(safety_cfg),
        "lagrangian_cfg": None if lag_cfg is None else asdict(lag_cfg),
        "training_protocol": result["training_protocol"],
    }
    with (out / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Completed matched V1.5 variant: {out}")
    for profile in eval_profiles:
        s = summaries[profile]
        print(f"[{profile}] terminal-safe pooled cohort")
        print(
            f"  interruption_probability: "
            f"{s['pooled_cohort_interruption_probability']:.6f}"
        )
        print(
            f"  on_time_delivery_ratio: "
            f"{s['pooled_cohort_on_time_delivery_ratio']:.6f}"
        )
        print(
            f"  packet_deadline_miss_ratio: "
            f"{s['pooled_cohort_packet_deadline_miss_ratio']:.6f}"
        )
        print(
            f"  delivered_traffic_rate_mbps: "
            f"{s['mean_cohort_delivered_traffic_rate_mbps']:.6f}"
        )
        print(
            f"  reliability_constraint_satisfied: "
            f"{s['reliability_constraint_satisfied_pooled']}; "
            f"deadline_reference_satisfied: "
            f"{s['deadline_reference_satisfied_pooled']}; "
            f"audit_passed: {s['integrity_audit_passed']}"
        )


if __name__ == "__main__":
    main()
