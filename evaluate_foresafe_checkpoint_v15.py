from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from viot_env import EnvConfig, VIoTEnv
from foresafe_config import PartialObservationConfig, RiskConfig, SafetyConfig, PPOConfig
from foresafe_observation import PartialObservationWrapper
from foresafe_ppo import BeliefActorCritic, ForeSafePolicy
from foresafe_risk import QuantileRiskPredictor
from run_foresafe import (
    parse_profiles,
    make_stress,
    evaluate_profile,
    write_rows,
    pooled_summary,
    audit_rows,
)


def load_policy(run_dir: Path, device: str):
    ckpt_path = run_dir / "foresafe_checkpoint.pt"
    diag_path = run_dir / "training_diagnostics.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if not diag_path.is_file():
        raise FileNotFoundError(diag_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    with diag_path.open("r", encoding="utf-8") as f:
        train_diag = json.load(f)

    po_cfg = PartialObservationConfig(**ckpt["partial_observation_cfg"])
    risk_cfg = RiskConfig(**ckpt["risk_cfg"])
    safety_cfg = SafetyConfig(**ckpt["safety_cfg"])
    ppo_cfg_dict = dict(ckpt["ppo_cfg"])
    ppo_cfg_dict["device"] = device
    ppo_cfg = PPOConfig(**ppo_cfg_dict)

    actor_state = ckpt["actor_critic"]
    obs_dim = int(actor_state["encoder.weight_ih_l0"].shape[1])
    n_actions = int(actor_state["actor.weight"].shape[0])

    actor_critic = BeliefActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
    actor_critic.load_state_dict(actor_state)

    risk_model = QuantileRiskPredictor(
        obs_dim,
        n_actions,
        risk_cfg.predictor_hidden_dim,
        tuple(risk_cfg.quantiles),
    ).to(device)
    risk_model.load_state_dict(ckpt["risk_model"])

    support = np.asarray(ckpt["action_support_counts"], dtype=np.int64)
    policy = ForeSafePolicy(
        actor_critic,
        risk_model,
        risk_cfg,
        safety_cfg,
        torch.device(device),
        action_support_counts=support,
    )

    shift_hist = train_diag.get("shift_error_ema", [])
    if not shift_hist:
        raise RuntimeError("training_diagnostics.json does not contain shift_error_ema.")
    policy.set_monitor_initial_state(float(shift_hist[-1]), True)

    return policy, po_cfg, ckpt


def main():
    p = argparse.ArgumentParser(
        description="Evaluate a frozen ForeSafe-RL V1.5 checkpoint without retraining."
    )
    p.add_argument("--run_dir", required=True)
    p.add_argument(
        "--profiles",
        default="standard_compound,compound_severe",
        help="Comma-separated evaluation profiles.",
    )
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--stochastic_eval", action="store_true")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    run_dir = Path(args.run_dir).resolve()
    profiles = parse_profiles(args.profiles)
    policy, po_cfg, ckpt = load_policy(run_dir, args.device)

    train_seed = int(ckpt["seed"])
    env_cfg = ckpt["train_env_cfg"]
    n_devices = int(env_cfg["n_devices"])
    n_channels = int(env_cfg["n_channels"])

    out = run_dir / ("checkpoint_eval_" + "+".join(profiles))
    out.mkdir(parents=True, exist_ok=True)

    def env_factory(profile_name: str, seed: int):
        cfg = EnvConfig(
            seed=seed,
            n_devices=n_devices,
            n_channels=n_channels,
            steps_per_episode=int(args.measurement_slots + args.followup_slots),
            measurement_cohort_slots=int(args.measurement_slots),
            arrival_cutoff_slots=0,
            stress=make_stress(profile_name),
        )
        return PartialObservationWrapper(
            VIoTEnv(cfg),
            po_cfg,
            seed=seed + 991,
        )

    summaries = {}
    audits = {}
    for profile in profiles:
        rows = evaluate_profile(
            policy,
            env_factory,
            args.eval_episodes,
            train_seed,
            profile,
            deterministic=not args.stochastic_eval,
        )
        write_rows(out / f"evaluation_{profile}_episodes.csv", rows)
        summary = pooled_summary(rows)
        audit = audit_rows(rows, n_channels)
        summary["joint_feasible_pooled"] = bool(
            summary["pooled_cohort_interruption_probability"] <= 0.05
            and summary["pooled_cohort_packet_deadline_miss_ratio"] <= 0.25
            and audit["passed"]
        )
        summaries[profile] = summary
        audits[profile] = audit

    with (out / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with (out / "evaluation_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)

    print(f"Frozen-checkpoint evaluation written to: {out}")
    for profile in profiles:
        s = summaries[profile]
        a = audits[profile]
        print(f"[{profile}]")
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
        print(f"  audit_passed: {a['passed']}")


if __name__ == "__main__":
    main()
