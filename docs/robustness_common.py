from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from foresafe_config import PartialObservationConfig, RiskConfig, SafetyConfig, PPOConfig
from foresafe_observation import PartialObservationWrapper
from foresafe_ppo import BeliefActorCritic, ForeSafePolicy
from foresafe_risk import QuantileRiskPredictor
from matched_baselines import (
    RecurrentPPOPolicy,
    RecurrentConstrainedActorCritic,
    RecurrentLagrangianPolicy,
)
from run_foresafe import evaluation_seed, pooled_summary, audit_rows
from viot_env import EnvConfig, StressConfig, VIoTEnv

METHODS = (
    "foresafe",
    "belief_ppo",
    "belief_ppo_lagrangian",
    "foresafe_no_belief",
    "foresafe_fixed_cvar",
)

MATCHED_VARIANTS = {
    "belief_ppo",
    "belief_ppo_lagrangian",
    "foresafe_no_belief",
    "foresafe_fixed_cvar",
}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _training_updates(ckpt: dict) -> int:
    proto = dict(ckpt.get("training_protocol", {}) or {})
    for key in ("rollout_updates", "train_updates"):
        if key in proto:
            return int(proto[key])
    steps = proto.get("total_training_steps")
    rollout = proto.get("rollout_steps")
    if steps is not None and rollout:
        return int(steps) // int(rollout)
    return -1


def discover_final_runs(
    foresafe_root: Path,
    matched_root: Path,
    seeds: Iterable[int],
    required_updates: int = 1200,
) -> dict[str, dict[int, Path]]:
    seeds = [int(s) for s in seeds]
    found: dict[str, dict[int, Path]] = {m: {} for m in METHODS}

    for ckpt_path in foresafe_root.rglob("foresafe_checkpoint.pt"):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        seed = int(ckpt.get("seed", -1))
        if seed not in seeds or _training_updates(ckpt) != int(required_updates):
            continue
        if seed in found["foresafe"]:
            raise RuntimeError(
                f"Multiple {required_updates}-update full ForeSafe checkpoints found for seed {seed}: "
                f"{found['foresafe'][seed]} and {ckpt_path.parent}"
            )
        found["foresafe"][seed] = ckpt_path.parent

    for ckpt_path in matched_root.rglob("checkpoint.pt"):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        variant = str(ckpt.get("variant", ""))
        seed = int(ckpt.get("seed", -1))
        if variant not in MATCHED_VARIANTS or seed not in seeds:
            continue
        if _training_updates(ckpt) != int(required_updates):
            continue
        if seed in found[variant]:
            raise RuntimeError(
                f"Multiple {required_updates}-update {variant} checkpoints found for seed {seed}: "
                f"{found[variant][seed]} and {ckpt_path.parent}"
            )
        found[variant][seed] = ckpt_path.parent

    missing = {
        method: [s for s in seeds if s not in found[method]]
        for method in METHODS
    }
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise RuntimeError(
            "Missing final checkpoints. Expected one 1200-update checkpoint per method/seed. "
            f"Missing={missing}. foresafe_root={foresafe_root}; matched_root={matched_root}"
        )
    return found


def _monitor_state_from_diagnostics(run_dir: Path, risk_cfg: RiskConfig) -> tuple[float, bool]:
    diag_path = run_dir / "training_diagnostics.json"
    if not diag_path.is_file():
        return float(risk_cfg.shift_reference_error), False
    diag = load_json(diag_path)
    hist = diag.get("shift_error_ema", [])
    if isinstance(hist, list) and hist:
        return float(hist[-1]), True
    return float(risk_cfg.shift_reference_error), False


def _build_foresafe_policy(ckpt: dict, run_dir: Path, device: str):
    po_cfg = PartialObservationConfig(**ckpt["partial_observation_cfg"])
    risk_cfg = RiskConfig(**ckpt["risk_cfg"])
    safety_cfg = SafetyConfig(**ckpt["safety_cfg"])
    ppo_raw = dict(ckpt["ppo_cfg"])
    ppo_raw["device"] = str(device)
    ppo_cfg = PPOConfig(**ppo_raw)

    actor_state = ckpt["actor_critic"]
    obs_dim = int(actor_state["encoder.weight_ih_l0"].shape[1])
    n_actions = int(actor_state["actor.weight"].shape[0])
    actor = BeliefActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
    actor.load_state_dict(actor_state)

    risk_model = QuantileRiskPredictor(
        obs_dim,
        n_actions,
        risk_cfg.predictor_hidden_dim,
        tuple(risk_cfg.quantiles),
    ).to(device)
    risk_model.load_state_dict(ckpt["risk_model"])

    support = np.asarray(ckpt["action_support_counts"], dtype=np.int64)
    policy = ForeSafePolicy(
        actor,
        risk_model,
        risk_cfg,
        safety_cfg,
        torch.device(device),
        action_support_counts=support,
    )
    err, initialized = _monitor_state_from_diagnostics(run_dir, risk_cfg)
    policy.set_monitor_initial_state(err, initialized)
    return policy, po_cfg, ckpt


def load_method(method: str, run_dir: Path, device: str = "cpu"):
    method = str(method)
    run_dir = Path(run_dir)
    if method == "foresafe":
        ckpt_path = run_dir / "foresafe_checkpoint.pt"
    else:
        ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if method in {"foresafe", "foresafe_no_belief", "foresafe_fixed_cvar"}:
        return _build_foresafe_policy(ckpt, run_dir, device)

    po_cfg = PartialObservationConfig(**ckpt["partial_observation_cfg"])
    ppo_raw = dict(ckpt["ppo_cfg"])
    ppo_raw["device"] = str(device)
    ppo_cfg = PPOConfig(**ppo_raw)
    state = ckpt["actor_critic"]
    obs_dim = int(state["encoder.weight_ih_l0"].shape[1])
    n_actions = int(state["actor.weight"].shape[0])

    if method == "belief_ppo":
        model = BeliefActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
        model.load_state_dict(state)
        return RecurrentPPOPolicy(model, torch.device(device)), po_cfg, ckpt

    if method == "belief_ppo_lagrangian":
        model = RecurrentConstrainedActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
        model.load_state_dict(state)
        return RecurrentLagrangianPolicy(model, torch.device(device)), po_cfg, ckpt

    raise ValueError(f"Unknown method {method!r}")


def mild_minus() -> StressConfig:
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


def standard_compound() -> StressConfig:
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


def nominal() -> StressConfig:
    return StressConfig(name="nominal")


def interpolate_stress(alpha: float) -> StressConfig:
    """Linear ladder from frozen mild-minus (alpha=0) to standard (alpha=1)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    lo, hi = mild_minus(), standard_compound()

    def lerp(x, y):
        return float(x) + a * (float(y) - float(x))

    return StressConfig(
        name=f"ladder_{a:.2f}",
        actual_shadowing_std_db=lerp(lo.actual_shadowing_std_db, hi.actual_shadowing_std_db),
        blockage_start_probability=lerp(lo.blockage_start_probability, hi.blockage_start_probability),
        blockage_mean_duration_slots=lerp(lo.blockage_mean_duration_slots, hi.blockage_mean_duration_slots),
        blockage_attenuation_db=lerp(lo.blockage_attenuation_db, hi.blockage_attenuation_db),
        external_interference_probability=lerp(lo.external_interference_probability, hi.external_interference_probability),
        external_interferer_power_dbm=lerp(lo.external_interferer_power_dbm, hi.external_interferer_power_dbm),
        external_interferer_distance_m=lerp(lo.external_interferer_distance_m, hi.external_interferer_distance_m),
        external_interferer_distance_jitter_m=lerp(lo.external_interferer_distance_jitter_m, hi.external_interferer_distance_jitter_m),
        stale_csi_slots=int(round(lerp(lo.stale_csi_slots, hi.stale_csi_slots))),
        actual_speed_mean_mps=lerp(lo.actual_speed_mean_mps, hi.actual_speed_mean_mps),
        actual_speed_std_mps=lerp(lo.actual_speed_std_mps, hi.actual_speed_std_mps),
    )


def change_post_stress() -> StressConfig:
    """Abrupt standard-like channel/CSI shift with mobility kept continuous."""
    hi = standard_compound()
    return replace(
        hi,
        name="standard_channel_shift",
        actual_speed_mean_mps=None,
        actual_speed_std_mps=None,
    )


def apply_stress_in_place(wrapper: PartialObservationWrapper, stress: StressConfig) -> None:
    """Switch realized stress without resetting queues, packets, mobility or history."""
    base = wrapper.env
    base.cfg = replace(base.cfg, stress=stress)
    base.actual_phy = base._actual_phy_config()
    base._risk_cache.clear()
    # Existing blockage state is retained. New blockage arrivals immediately use
    # the post-change hazard, so the transition is causal rather than a reset.


def build_env(
    ckpt: dict,
    po_cfg: PartialObservationConfig,
    stress: StressConfig,
    seed: int,
    measurement_slots: int,
    followup_slots: int,
) -> PartialObservationWrapper:
    env_cfg = dict(ckpt["train_env_cfg"])
    n_devices = int(env_cfg["n_devices"])
    n_channels = int(env_cfg["n_channels"])
    cfg = EnvConfig(
        seed=int(seed),
        n_devices=n_devices,
        n_channels=n_channels,
        steps_per_episode=int(measurement_slots + followup_slots),
        measurement_cohort_slots=int(measurement_slots),
        arrival_cutoff_slots=0,
        stress=stress,
    )
    return PartialObservationWrapper(VIoTEnv(cfg), po_cfg, seed=int(seed) + 991)


def evaluate_static(
    method: str,
    policy: Any,
    ckpt: dict,
    po_cfg: PartialObservationConfig,
    stress: StressConfig,
    episodes: int,
    measurement_slots: int = 150,
    followup_slots: int = 251,
) -> tuple[list[dict], dict, dict]:
    train_seed = int(ckpt["seed"])
    rows: list[dict] = []
    foresafe_method = method.startswith("foresafe")

    for ep in range(int(episodes)):
        seed = evaluation_seed(train_seed, ep)
        env = build_env(ckpt, po_cfg, stress, seed, measurement_slots, followup_slots)
        history, _ = env.reset(seed=seed)
        if foresafe_method:
            policy.reset_online_monitor()

        medians: list[float] = []
        cvars: list[float] = []
        limits: list[float] = []
        strict: list[float] = []
        admissible: list[float] = []
        safe: list[float] = []
        forced: list[float] = []
        shift_errors: list[float] = []
        done = False

        while not done:
            if foresafe_method:
                action, diag = policy.act(env, history, deterministic=True)
                history, _, done, info = env.step(action)
                policy.observe_realized_outcome(
                    diag.chosen_median_risk,
                    float(info.get("failed_scheduled", 0) or 0),
                    float(info.get("scheduled", 0) or 0),
                    prediction_supported=diag.chosen_action_supported,
                )
                medians.append(float(diag.chosen_median_risk))
                cvars.append(float(diag.chosen_cvar))
                limits.append(float(diag.effective_limit))
                strict.append(float(diag.strict_supported_cvar_pass_count))
                admissible.append(float(diag.cvar_admissible_action_count))
                safe.append(float(diag.safe_action_count))
                forced.append(float(diag.forced_min_risk))
                shift_errors.append(float(policy.shift_monitor.error_ema))
            else:
                action = policy.act(env, history, deterministic=True)
                history, _, done, _ = env.step(action)

        row = dict(env.episode_metrics())
        row.update({
            "method": method,
            "train_seed": train_seed,
            "evaluation_seed": int(seed),
            "evaluation_episode": int(ep),
            "stress_name": str(stress.name),
            "observation_delay_slots": int(po_cfg.observation_delay_slots),
            "gaussian_noise_std": float(po_cfg.gaussian_noise_std),
            "dropout_probability": float(po_cfg.dropout_probability),
            "history_len": int(po_cfg.history_len),
            "observation_dropouts": int(env.diagnostics.dropouts),
            "observation_noisy_updates": int(env.diagnostics.noisy_observations),
            "observation_delayed_updates": int(env.diagnostics.delayed_observations),
        })
        if foresafe_method:
            row.update({
                "mean_predicted_median_risk": float(np.mean(medians)),
                "mean_predicted_cvar": float(np.mean(cvars)),
                "mean_effective_cvar_limit": float(np.mean(limits)),
                "mean_strict_supported_cvar_pass_count": float(np.mean(strict)),
                "mean_cvar_admissible_action_count": float(np.mean(admissible)),
                "mean_safe_action_count": float(np.mean(safe)),
                "forced_min_risk_rate": float(np.mean(forced)),
                "mean_shift_error_ema": float(np.mean(shift_errors)),
                "shift_error_ema_end": float(shift_errors[-1]),
            })
        rows.append(row)

    env_cfg = dict(ckpt["train_env_cfg"])
    summary = pooled_summary(rows)
    audit = audit_rows(rows, int(env_cfg["n_channels"]))
    return rows, summary, audit


def compact_summary(method: str, seed: int, condition: str, summary: dict, audit: dict, rows: list[dict]) -> dict:
    out = {
        "method": method,
        "train_seed": int(seed),
        "condition": condition,
        "episodes": int(len(rows)),
        "interruption_probability": float(summary["pooled_cohort_interruption_probability"]),
        "on_time_delivery_ratio": float(summary["pooled_cohort_on_time_delivery_ratio"]),
        "deadline_miss_ratio": float(summary["pooled_cohort_packet_deadline_miss_ratio"]),
        "throughput_mbps": float(summary["mean_cohort_delivered_traffic_rate_mbps"]),
        "audit_passed": bool(audit["passed"]),
    }
    for key in (
        "defer_probability",
        "mean_effective_cvar_limit",
        "mean_strict_supported_cvar_pass_count",
        "mean_cvar_admissible_action_count",
        "mean_safe_action_count",
        "forced_min_risk_rate",
        "mean_shift_error_ema",
        "shift_error_ema_end",
    ):
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def dump_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
