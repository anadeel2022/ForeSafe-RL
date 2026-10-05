from __future__ import annotations
import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from viot_env import EnvConfig, StressConfig, VIoTEnv
from foresafe_config import PartialObservationConfig, RiskConfig, SafetyConfig, PPOConfig
from foresafe_observation import PartialObservationWrapper
from foresafe_ppo import train_foresafe_continuing


PRIMARY_COHORT_METRICS = (
    "cohort_interruption_probability",
    "cohort_on_time_delivery_ratio",
    "cohort_packet_deadline_miss_ratio",
    "cohort_delivered_traffic_rate_mbps",
)


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def parse_profiles(text: str) -> list[str]:
    values = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one evaluation profile is required.")
    valid = {"nominal", "mild_minus", "standard_compound", "compound_severe"}
    unknown = [x for x in values if x not in valid]
    if unknown:
        raise ValueError(f"Unknown profiles {unknown}; valid={sorted(valid)}")
    return list(dict.fromkeys(values))


def make_stress(name: str) -> StressConfig:
    """Profiles aligned with the final published MobiSafe reproducibility code."""
    name = str(name).lower()
    if name == "nominal":
        return StressConfig(name="nominal")
    if name == "mild_minus":
        return StressConfig(
            name="mild_minus",
            actual_shadowing_std_db=4.0,
            blockage_start_probability=0.0015,
            blockage_mean_duration_slots=2.5,
            blockage_attenuation_db=4.5,
            external_interference_probability=0.015,
            external_interferer_power_dbm=6.0,
            external_interferer_distance_m=140.0,
            external_interferer_distance_jitter_m=25.0,
            stale_csi_slots=1,
            actual_speed_mean_mps=21.0,
            actual_speed_std_mps=3.5,
        )
    if name == "standard_compound":
        return StressConfig(
            name="standard_compound",
            actual_shadowing_std_db=6.0,
            blockage_start_probability=0.012,
            blockage_mean_duration_slots=8.0,
            blockage_attenuation_db=10.0,
            external_interference_probability=0.12,
            external_interferer_power_dbm=14.0,
            external_interferer_distance_m=140.0,
            external_interferer_distance_jitter_m=25.0,
            stale_csi_slots=5,
            actual_speed_mean_mps=28.0,
            actual_speed_std_mps=6.0,
        )
    if name == "compound_severe":
        return StressConfig(
            name="compound_severe",
            actual_shadowing_std_db=8.0,
            blockage_start_probability=0.04,
            blockage_mean_duration_slots=15.0,
            blockage_attenuation_db=15.0,
            external_interference_probability=0.45,
            external_interferer_power_dbm=20.0,
            external_interferer_distance_m=140.0,
            external_interferer_distance_jitter_m=25.0,
            stale_csi_slots=20,
            actual_speed_mean_mps=35.0,
            actual_speed_std_mps=8.0,
        )
    raise ValueError(f"Unknown stress profile: {name}")


def evaluation_seed(train_seed: int, episode_index: int) -> int:
    return int(7_000_000 + int(train_seed) * 10_000 + int(episode_index))


def write_rows(path: Path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pooled_summary(rows: list[dict]) -> dict:
    offered = float(sum(float(r.get("cohort_offered_packets", 0) or 0) for r in rows))
    delivered = float(sum(float(r.get("cohort_delivered_packets", 0) or 0) for r in rows))
    missed = float(sum(float(r.get("cohort_deadline_missed_packets", 0) or 0) for r in rows))
    scheduled = float(sum(float(r.get("cohort_scheduled_tx", 0) or 0) for r in rows))
    failed = float(sum(float(r.get("cohort_failed_scheduled_tx", 0) or 0) for r in rows))

    summary = {
        "n_evaluation_episodes": int(len(rows)),
        "pooled_cohort_offered_packets": int(offered),
        "pooled_cohort_delivered_packets": int(delivered),
        "pooled_cohort_deadline_missed_packets": int(missed),
        "pooled_cohort_scheduled_tx": int(scheduled),
        "pooled_cohort_failed_scheduled_tx": int(failed),
        "pooled_cohort_interruption_probability": float(failed / scheduled) if scheduled > 0 else math.nan,
        "pooled_cohort_on_time_delivery_ratio": float(delivered / offered) if offered > 0 else math.nan,
        "pooled_cohort_packet_deadline_miss_ratio": float(missed / offered) if offered > 0 else math.nan,
        "mean_cohort_delivered_traffic_rate_mbps": float(np.mean([
            float(r["cohort_delivered_traffic_rate_mbps"]) for r in rows
            if np.isfinite(float(r.get("cohort_delivered_traffic_rate_mbps", np.nan)))
        ])),
    }
    for key in PRIMARY_COHORT_METRICS:
        vals = np.asarray([
            float(r[key]) for r in rows
            if key in r and np.isfinite(float(r[key]))
        ], dtype=float)
        summary[f"episode_mean_{key}"] = float(np.mean(vals)) if vals.size else math.nan
        summary[f"episode_sd_{key}"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return summary


def audit_rows(rows: list[dict], n_channels: int) -> dict:
    required = (
        "cohort_offered_packets",
        "cohort_pending_packets_end",
        "cohort_packet_conservation_error",
        "resource_budget_violation_count",
        "invalid_action_selection_count",
    )
    missing = [k for k in required if any(k not in r for r in rows)]
    if missing:
        raise RuntimeError(f"Missing terminal-safe audit columns: {missing}")

    finite = True
    for row in rows:
        for key in PRIMARY_COHORT_METRICS:
            if key not in row or not np.isfinite(float(row[key])):
                finite = False

    checks = {
        "rows": int(len(rows)),
        "max_abs_cohort_conservation_error": float(max(
            abs(float(r.get("cohort_packet_conservation_error", 0) or 0)) for r in rows
        )),
        "max_cohort_pending_packets_end": float(max(
            float(r.get("cohort_pending_packets_end", 0) or 0) for r in rows
        )),
        "resource_budget_violations": float(sum(
            float(r.get("resource_budget_violation_count", 0) or 0) for r in rows
        )),
        "invalid_action_selections": float(sum(
            float(r.get("invalid_action_selection_count", 0) or 0) for r in rows
        )),
        "max_resource_blocks_used": float(max(
            float(r.get("max_resource_blocks_used", 0) or 0) for r in rows
        )),
        "cohort_metrics_finite": bool(finite),
    }
    checks["passed"] = bool(
        checks["max_abs_cohort_conservation_error"] < 1e-9
        and checks["max_cohort_pending_packets_end"] == 0
        and checks["resource_budget_violations"] == 0
        and checks["invalid_action_selections"] == 0
        and checks["max_resource_blocks_used"] <= int(n_channels)
        and checks["cohort_metrics_finite"]
    )
    if not checks["passed"]:
        raise RuntimeError(f"Terminal-safe evaluation audit failed: {checks}")
    return checks


def evaluate_profile(
    policy,
    env_factory,
    episodes: int,
    train_seed: int,
    profile_name: str,
    deterministic: bool,
):
    rows = []
    for ep in range(int(episodes)):
        seed = evaluation_seed(train_seed, ep)
        env = env_factory(profile_name, seed)
        policy.reset_online_monitor()
        history, _ = env.reset(seed=seed)
        done = False

        medians, cvars, native_counts = [], [], []
        supported_counts, strict_pass_counts = [], []
        cvar_counts, safe_counts = [], []
        risk_rejected, risk_relaxed = [], []
        urgency_removed, urgency_scores = [], []
        limits, forced, chosen_supported = [], [], []

        while not done:
            action, diag = policy.act(env, history, deterministic=deterministic)
            history, _, done, info = env.step(action)
            scheduled = float(info.get("scheduled", 0) or 0)
            failed = float(info.get("failed_scheduled", 0) or 0)
            policy.observe_realized_outcome(
                diag.chosen_median_risk,
                failed,
                scheduled,
                prediction_supported=diag.chosen_action_supported,
            )

            medians.append(diag.chosen_median_risk)
            cvars.append(diag.chosen_cvar)
            native_counts.append(diag.native_feasible_action_count)
            supported_counts.append(diag.supported_native_action_count)
            strict_pass_counts.append(diag.strict_supported_cvar_pass_count)
            cvar_counts.append(diag.cvar_admissible_action_count)
            safe_counts.append(diag.safe_action_count)
            risk_rejected.append(diag.risk_rejected)
            risk_relaxed.append(diag.risk_relaxed)
            urgency_removed.append(diag.urgency_guarded)
            urgency_scores.append(diag.urgency_score)
            limits.append(diag.effective_limit)
            forced.append(int(diag.forced_min_risk))
            chosen_supported.append(int(diag.chosen_action_supported))

        row = dict(env.episode_metrics())
        row.update({
            "train_seed": int(train_seed),
            "evaluation_seed": int(seed),
            "evaluation_episode": int(ep),
            "evaluation_profile": str(profile_name),
            "mean_predicted_median_risk": float(np.mean(medians)),
            "mean_predicted_cvar": float(np.mean(cvars)),
            "mean_native_feasible_action_count": float(np.mean(native_counts)),
            "mean_supported_native_action_count": float(np.mean(supported_counts)),
            "mean_strict_supported_cvar_pass_count": float(np.mean(strict_pass_counts)),
            "mean_cvar_admissible_action_count": float(np.mean(cvar_counts)),
            "mean_safe_action_count": float(np.mean(safe_counts)),
            "mean_risk_rejected_action_count": float(np.mean(risk_rejected)),
            "mean_risk_relaxed_action_count": float(np.mean(risk_relaxed)),
            "mean_urgency_removed_action_count": float(np.mean(urgency_removed)),
            "mean_urgency_score": float(np.mean(urgency_scores)),
            "mean_effective_cvar_limit": float(np.mean(limits)),
            "forced_min_risk_rate": float(np.mean(forced)),
            "chosen_supported_action_rate": float(np.mean(chosen_supported)),
            "shift_error_ema_end": float(policy.shift_monitor.error_ema),
            "observation_dropouts": int(env.diagnostics.dropouts),
            "observation_noisy_updates": int(env.diagnostics.noisy_observations),
            "observation_delayed_updates": int(env.diagnostics.delayed_observations),
        })
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(
        description="ForeSafe-RL V1.5 evidence-aware continuing-task / terminal-safe runner"
    )
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--train_updates", type=int, default=1200)
    p.add_argument("--rollout_steps", type=int, default=150)
    p.add_argument("--eval_episodes", type=int, default=200)
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
        default="nominal",
        help="Comma-separated profiles; e.g. nominal,mild_minus",
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
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--stochastic_eval", action="store_true")
    p.add_argument("--out", default="outputs_foresafe_v15")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if args.train_updates <= 0 or args.rollout_steps <= 0:
        raise ValueError("train_updates and rollout_steps must be positive.")
    if args.measurement_slots <= 0 or args.followup_slots <= 0:
        raise ValueError("measurement_slots and followup_slots must be positive.")

    eval_profiles = parse_profiles(args.eval_profiles)
    label = "+".join(eval_profiles)
    out = Path(args.out) / (
        f"{stamp()}_train-{args.train_profile}_eval-{label}_seed{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)

    po_cfg = PartialObservationConfig(
        history_len=args.history_len,
        observation_delay_slots=args.obs_delay,
        gaussian_noise_std=args.obs_noise,
        dropout_probability=args.dropout,
    )
    risk_cfg = RiskConfig(
        horizon=args.risk_horizon,
        cvar_alpha=args.cvar_alpha,
        base_cvar_limit=args.cvar_limit,
        warmup_steps=args.warmup_steps,
        shield_warmup_steps=args.shield_warmup_steps,
        min_supported_action_fraction=args.min_supported_action_fraction,
        predictor_gradient_steps_per_update=args.predictor_gradient_steps,
    )
    safety_cfg = SafetyConfig(
        urgency_guard_threshold=args.urgency_guard_threshold,
        min_action_samples_for_filter=args.min_action_samples,
        min_operational_action_fraction=args.min_operational_action_fraction,
    )
    ppo_cfg = PPOConfig(device=args.device)

    # One extra rollout of horizon prevents the simulator from becoming terminal
    # at the final optimization boundary.
    train_horizon = (int(args.train_updates) + 1) * int(args.rollout_steps)
    train_env_cfg = EnvConfig(
        seed=args.seed,
        n_devices=args.n_devices,
        n_channels=args.n_channels,
        steps_per_episode=train_horizon,
        measurement_cohort_slots=0,
        arrival_cutoff_slots=0,
        stress=make_stress(args.train_profile),
    )
    train_env = PartialObservationWrapper(
        VIoTEnv(train_env_cfg), po_cfg, seed=args.seed
    )
    result = train_foresafe_continuing(
        train_env,
        ppo_cfg,
        risk_cfg,
        safety_cfg,
        rollout_updates=args.train_updates,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
    )

    torch.save({
        "actor_critic": result["actor_critic"].state_dict(),
        "risk_model": result["risk_model"].state_dict(),
        "train_env_cfg": train_env_cfg.to_dict(),
        "partial_observation_cfg": asdict(po_cfg),
        "risk_cfg": asdict(risk_cfg),
        "safety_cfg": asdict(safety_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "seed": args.seed,
        "action_support_counts": result["action_support_counts"].tolist(),
        "training_protocol": result["training_protocol"],
    }, out / "foresafe_checkpoint.pt")

    write_rows(
        out / "training_rollout_metrics.csv",
        result["history"]["rollout_metrics"],
    )
    with (out / "training_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in result["history"].items() if k != "rollout_metrics"},
            f, indent=2
        )

    def eval_env_factory(profile_name: str, seed: int):
        cfg = EnvConfig(
            seed=seed,
            n_devices=args.n_devices,
            n_channels=args.n_channels,
            steps_per_episode=int(args.measurement_slots + args.followup_slots),
            measurement_cohort_slots=int(args.measurement_slots),
            arrival_cutoff_slots=0,
            stress=make_stress(profile_name),
        )
        return PartialObservationWrapper(
            VIoTEnv(cfg), po_cfg, seed=seed + 991
        )

    profile_summaries = {}
    profile_audits = {}
    for profile in eval_profiles:
        rows = evaluate_profile(
            result["policy"],
            eval_env_factory,
            args.eval_episodes,
            args.seed,
            profile,
            deterministic=not args.stochastic_eval,
        )
        write_rows(out / f"evaluation_{profile}_episodes.csv", rows)
        summary = pooled_summary(rows)
        audit = audit_rows(rows, args.n_channels)
        summary["joint_feasible_pooled"] = bool(
            summary["pooled_cohort_interruption_probability"] <= 0.05
            and summary["pooled_cohort_packet_deadline_miss_ratio"] <= 0.25
            and audit["passed"]
        )
        profile_summaries[profile] = summary
        profile_audits[profile] = audit

    with (out / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(profile_summaries, f, indent=2)
    with (out / "evaluation_audit.json").open("w", encoding="utf-8") as f:
        json.dump(profile_audits, f, indent=2)

    manifest = {
        "version": "ForeSafe-RL V1.5",
        "algorithm_core": "V1.5 evidence-aware predictive CVaR controller with deadline-scaled service guard",
        "seed": args.seed,
        "train_profile": args.train_profile,
        "eval_profiles": eval_profiles,
        "train_updates": args.train_updates,
        "rollout_steps": args.rollout_steps,
        "total_training_steps": result["total_steps"],
        "environment_resets_during_training": 1,
        "bootstrap_at_rollout_boundaries": True,
        "risk_windows_cross_rollout_boundaries": True,
        "pending_unlabeled_risk_windows_at_end": result[
            "pending_unlabeled_risk_windows"
        ],
        "eval_episodes_per_profile": args.eval_episodes,
        "measurement_cohort_slots": args.measurement_slots,
        "followup_slots": args.followup_slots,
        "background_arrivals_during_followup": True,
        "evaluation_seed_rule": "7000000 + train_seed*10000 + episode_index",
        "train_env_cfg": train_env_cfg.to_dict(),
        "partial_observation_cfg": asdict(po_cfg),
        "risk_cfg": asdict(risk_cfg),
        "safety_cfg": asdict(safety_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "stress_profiles": {
            name: asdict(make_stress(name))
            for name in ["nominal", "mild_minus", "standard_compound", "compound_severe"]
        },
    }
    with (out / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Completed ForeSafe-RL V1.5 run: {out}")
    for profile in eval_profiles:
        s = profile_summaries[profile]
        a = profile_audits[profile]
        print(f"[{profile}] terminal-safe pooled cohort")
        print(
            "  interruption_probability: "
            f"{s['pooled_cohort_interruption_probability']:.6f}"
        )
        print(
            "  on_time_delivery_ratio: "
            f"{s['pooled_cohort_on_time_delivery_ratio']:.6f}"
        )
        print(
            "  packet_deadline_miss_ratio: "
            f"{s['pooled_cohort_packet_deadline_miss_ratio']:.6f}"
        )
        print(
            "  delivered_traffic_rate_mbps: "
            f"{s['mean_cohort_delivered_traffic_rate_mbps']:.6f}"
        )
        print(
            f"  audit_passed: {a['passed']}; "
            f"joint_feasible_pooled: {s['joint_feasible_pooled']}"
        )


if __name__ == "__main__":
    main()
