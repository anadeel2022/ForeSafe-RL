from __future__ import annotations
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate_foresafe_checkpoint_v15 import load_policy
from run_foresafe import make_stress, evaluate_profile, pooled_summary, audit_rows
from viot_env import EnvConfig, VIoTEnv
from foresafe_observation import PartialObservationWrapper


CONFIGS = {
    "baseline": {
        "shift_tightening_gain": 1.5,
        "min_cvar_limit": 0.04,
        "min_operational_action_fraction": 0.20,
    },
    "low_floor": {
        "shift_tightening_gain": 1.5,
        "min_cvar_limit": 0.04,
        "min_operational_action_fraction": 0.05,
    },
    "strong_tightening": {
        "shift_tightening_gain": 3.0,
        "min_cvar_limit": 0.02,
        "min_operational_action_fraction": 0.20,
    },
    "strong_tightening_low_floor": {
        "shift_tightening_gain": 3.0,
        "min_cvar_limit": 0.02,
        "min_operational_action_fraction": 0.05,
    },
}


def mean_col(rows, key):
    vals = [
        float(r[key]) for r in rows
        if key in r and np.isfinite(float(r[key]))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(
        description="Evaluation-only V1.5 shield sensitivity sweep."
    )
    p.add_argument("--run_dir", required=True)
    p.add_argument("--profile", default="standard_compound")
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out = run_dir / f"shield_sensitivity_{args.profile}"
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for label, override in CONFIGS.items():
        policy, po_cfg, ckpt = load_policy(run_dir, args.device)

        # Evaluation-only development overrides.
        # RiskConfig and SafetyConfig are frozen dataclasses, so construct
        # modified copies rather than mutating fields in place.
        new_risk_cfg = replace(
            policy.risk_cfg,
            shift_tightening_gain=float(override["shift_tightening_gain"]),
            min_cvar_limit=float(override["min_cvar_limit"]),
        )
        new_safety_cfg = replace(
            policy.safety_cfg,
            min_operational_action_fraction=float(
                override["min_operational_action_fraction"]
            ),
        )

        # ForeSafePolicy and its ShiftMonitor both retain config references.
        # Update both so the evaluation truly uses the requested tightening.
        policy.risk_cfg = new_risk_cfg
        policy.shift_monitor.cfg = new_risk_cfg
        policy.safety_cfg = new_safety_cfg

        train_seed = int(ckpt["seed"])
        env_cfg = ckpt["train_env_cfg"]
        n_devices = int(env_cfg["n_devices"])
        n_channels = int(env_cfg["n_channels"])

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
                VIoTEnv(cfg), po_cfg, seed=seed + 991
            )

        rows = evaluate_profile(
            policy,
            env_factory,
            args.eval_episodes,
            train_seed,
            args.profile,
            deterministic=True,
        )
        audit = audit_rows(rows, n_channels)
        pooled = pooled_summary(rows)

        row = {
            "configuration": label,
            **override,
            "pooled_interruption_probability":
                pooled["pooled_cohort_interruption_probability"],
            "pooled_on_time_delivery_ratio":
                pooled["pooled_cohort_on_time_delivery_ratio"],
            "pooled_deadline_miss_ratio":
                pooled["pooled_cohort_packet_deadline_miss_ratio"],
            "mean_traffic_rate_mbps":
                pooled["mean_cohort_delivered_traffic_rate_mbps"],
            "mean_defer_probability":
                mean_col(rows, "defer_probability"),
            "mean_shift_error_ema_end":
                mean_col(rows, "shift_error_ema_end"),
            "mean_effective_cvar_limit":
                mean_col(rows, "mean_effective_cvar_limit"),
            "mean_predicted_cvar":
                mean_col(rows, "mean_predicted_cvar"),
            "mean_strict_supported_cvar_pass_count":
                mean_col(rows, "mean_strict_supported_cvar_pass_count"),
            "mean_cvar_admissible_action_count":
                mean_col(rows, "mean_cvar_admissible_action_count"),
            "mean_risk_relaxed_action_count":
                mean_col(rows, "mean_risk_relaxed_action_count"),
            "forced_min_risk_rate":
                mean_col(rows, "forced_min_risk_rate"),
            "audit_passed": bool(audit["passed"]),
        }
        summary_rows.append(row)

        # Save episode-level data for auditability.
        episode_path = out / f"{label}_episodes.csv"
        keys = sorted({k for r in rows for k in r})
        with episode_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # Guard against a silent config-linkage failure. Under the same episodes,
    # stronger tightening must not produce a larger mean effective limit than
    # baseline when the shift monitor sees positive excess error.
    by_name = {r["configuration"]: r for r in summary_rows}
    if (
        "baseline" in by_name
        and "strong_tightening" in by_name
        and np.isfinite(by_name["baseline"]["mean_effective_cvar_limit"])
        and np.isfinite(by_name["strong_tightening"]["mean_effective_cvar_limit"])
        and by_name["strong_tightening"]["mean_effective_cvar_limit"]
            > by_name["baseline"]["mean_effective_cvar_limit"] + 1e-12
    ):
        raise RuntimeError(
            "Strong-tightening configuration produced a larger effective CVaR "
            "limit than baseline; diagnostic configuration was not applied correctly."
        )

    summary_path = out / "shield_sensitivity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"V1.5 shield sensitivity written to: {out}")
    for r in summary_rows:
        print(
            f"[{r['configuration']}] "
            f"intr={r['pooled_interruption_probability']:.6f}; "
            f"OTD={r['pooled_on_time_delivery_ratio']:.6f}; "
            f"miss={r['pooled_deadline_miss_ratio']:.6f}; "
            f"thr={r['mean_traffic_rate_mbps']:.6f}; "
            f"defer={r['mean_defer_probability']:.4f}; "
            f"strict={r['mean_strict_supported_cvar_pass_count']:.2f}; "
            f"admissible={r['mean_cvar_admissible_action_count']:.2f}; "
            f"limit={r['mean_effective_cvar_limit']:.4f}; "
            f"gain={r['shift_tightening_gain']:.1f}; "
            f"min_limit={r['min_cvar_limit']:.2f}; "
            f"floor={r['min_operational_action_fraction']:.2f}; "
            f"audit={r['audit_passed']}"
        )


if __name__ == "__main__":
    main()
