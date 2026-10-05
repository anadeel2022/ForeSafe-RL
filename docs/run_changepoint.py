from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from robustness_common import (
    apply_stress_in_place,
    build_env,
    change_post_stress,
    compact_summary,
    discover_final_runs,
    dump_manifest,
    load_method,
    mild_minus,
    write_rows,
)
from run_foresafe import evaluation_seed, pooled_summary, audit_rows

METHODS = ("foresafe", "foresafe_fixed_cvar")


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def window_stats(slot_rows: list[dict], start: int, stop: int) -> dict:
    use = [r for r in slot_rows if start <= int(r["slot"]) < stop]
    sched = float(sum(float(r["scheduled"]) for r in use))
    fail = float(sum(float(r["failed"]) for r in use))
    return {
        "start": int(start),
        "stop": int(stop),
        "scheduled": sched,
        "failed": fail,
        "interruption_probability": ratio(fail, sched),
        "mean_cvar_limit": float(np.mean([r["effective_cvar_limit"] for r in use])) if use else float("nan"),
        "mean_shift_error": float(np.mean([r["shift_error_ema"] for r in use])) if use else float("nan"),
        "mean_strict_pass_count": float(np.mean([r["strict_pass_count"] for r in use])) if use else float("nan"),
        "mean_action_defer_fraction": float(np.mean([r["action_defer_fraction"] for r in use])) if use else float("nan"),
    }


def run_episode(method, policy, ckpt, po_cfg, seed: int, change_slot: int, measurement_slots: int, followup_slots: int):
    env = build_env(ckpt, po_cfg, mild_minus(), seed, measurement_slots, followup_slots)
    history, _ = env.reset(seed=seed)
    policy.reset_online_monitor()
    done = False
    slot_rows = []

    while not done:
        slot = int(env.env.t)
        if slot == int(change_slot):
            apply_stress_in_place(env, change_post_stress())

        action, diag = policy.act(env, history, deterministic=True)
        defer_fraction = float(env.decoded_defer_fraction(action))
        history, _, done, info = env.step(action)
        policy.observe_realized_outcome(
            diag.chosen_median_risk,
            float(info.get("failed_scheduled", 0) or 0),
            float(info.get("scheduled", 0) or 0),
            prediction_supported=diag.chosen_action_supported,
        )
        slot_rows.append({
            "slot": slot,
            "scheduled": int(info.get("scheduled", 0) or 0),
            "failed": int(info.get("failed_scheduled", 0) or 0),
            "effective_cvar_limit": float(diag.effective_limit),
            "shift_error_ema": float(policy.shift_monitor.error_ema),
            "strict_pass_count": float(diag.strict_supported_cvar_pass_count),
            "admissible_count": float(diag.cvar_admissible_action_count),
            "safe_count": float(diag.safe_action_count),
            "action_defer_fraction": defer_fraction,
            "chosen_cvar": float(diag.chosen_cvar),
            "chosen_median_risk": float(diag.chosen_median_risk),
        })

    episode = dict(env.episode_metrics())
    episode.update({
        "evaluation_seed": int(seed),
        "method": method,
        "change_slot": int(change_slot),
    })
    return episode, slot_rows


def main():
    p = argparse.ArgumentParser(description="Abrupt mild-to-standard channel change-point evaluation")
    p.add_argument("--foresafe_root", required=True)
    p.add_argument("--matched_root", required=True)
    p.add_argument("--seeds", default="11,12,13,14,15,16,17,18,19,20")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--methods", default=",".join(METHODS), help="Comma-separated subset of methods")
    p.add_argument("--change_slot", type=int, default=75)
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", default="outputs_robustness_v1/changepoint")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if not 12 <= args.change_slot < args.measurement_slots:
        raise ValueError("change_slot must allow at least the 12-slot risk horizon before and after the change.")

    seeds = parse_seeds(args.seeds)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = [x for x in methods if x not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    roots = discover_final_runs(Path(args.foresafe_root), Path(args.matched_root), seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global_summary = []

    for method in methods:
        for train_seed in seeds:
            policy, po_cfg, ckpt = load_method(method, roots[method][train_seed], args.device)
            episode_rows = []
            accum = defaultdict(lambda: defaultdict(float))
            n_per_slot = defaultdict(int)

            for ep in range(int(args.episodes)):
                seed = evaluation_seed(train_seed, ep)
                episode, slots = run_episode(
                    method, policy, ckpt, po_cfg, seed, args.change_slot,
                    args.measurement_slots, args.followup_slots,
                )
                episode["train_seed"] = int(train_seed)
                episode["evaluation_episode"] = int(ep)
                for label, start, stop in (
                    ("pre25", args.change_slot - 25, args.change_slot),
                    ("post25", args.change_slot, args.change_slot + 25),
                    ("post50", args.change_slot, args.change_slot + 50),
                    ("post75", args.change_slot, args.change_slot + 75),
                ):
                    ws = window_stats(slots, start, stop)
                    for k, v in ws.items():
                        if k not in {"start", "stop"}:
                            episode[f"{label}_{k}"] = v
                episode_rows.append(episode)

                for r in slots:
                    t = int(r["slot"])
                    n_per_slot[t] += 1
                    for k, v in r.items():
                        if k == "slot":
                            continue
                        accum[t][k] += float(v)

            n_channels = int(ckpt["train_env_cfg"]["n_channels"])
            audit = audit_rows(episode_rows, n_channels)
            summary = pooled_summary(episode_rows)
            summary_row = compact_summary(
                method, train_seed, "mild_to_standard_channel_change", summary, audit, episode_rows
            )
            for label in ("pre25", "post25", "post50", "post75"):
                for metric in (
                    "interruption_probability", "mean_cvar_limit", "mean_shift_error",
                    "mean_strict_pass_count", "mean_action_defer_fraction",
                ):
                    vals = [
                        float(r[f"{label}_{metric}"]) for r in episode_rows
                        if np.isfinite(float(r.get(f"{label}_{metric}", np.nan)))
                    ]
                    if vals:
                        summary_row[f"{label}_{metric}"] = float(np.mean(vals))
            global_summary.append(summary_row)

            write_rows(out / method / f"seed{train_seed}" / "episodes.csv", episode_rows)
            time_rows = []
            for t in sorted(n_per_slot):
                n = float(n_per_slot[t])
                sched = accum[t]["scheduled"]
                fail = accum[t]["failed"]
                row = {
                    "method": method,
                    "train_seed": int(train_seed),
                    "slot": int(t),
                    "relative_to_change": int(t - args.change_slot),
                    "episodes": int(n),
                    "pooled_slot_interruption_probability": ratio(fail, sched),
                }
                for k in (
                    "effective_cvar_limit", "shift_error_ema", "strict_pass_count",
                    "admissible_count", "safe_count", "action_defer_fraction",
                    "chosen_cvar", "chosen_median_risk",
                ):
                    row[f"mean_{k}"] = float(accum[t][k] / n)
                time_rows.append(row)
            write_rows(out / method / f"seed{train_seed}" / "timeseries.csv", time_rows)

            print(
                f"[{method} seed={train_seed}] "
                f"whole_intr={summary['pooled_cohort_interruption_probability']:.6f}; "
                f"post25_intr={summary_row.get('post25_interruption_probability', float('nan')):.6f}; "
                f"pre_limit={summary_row.get('pre25_mean_cvar_limit', float('nan')):.4f}; "
                f"post50_limit={summary_row.get('post50_mean_cvar_limit', float('nan')):.4f}; "
                f"audit={audit['passed']}"
            )

    write_rows(out / "changepoint_summary.csv", global_summary)
    dump_manifest(out / "manifest.json", {
        "experiment": "abrupt_distribution_shift_change_point",
        "evaluation_only": True,
        "methods": methods,
        "seeds": seeds,
        "episodes_per_method_seed": int(args.episodes),
        "change_slot": int(args.change_slot),
        "pre_change_stress": asdict(mild_minus()),
        "post_change_stress": asdict(change_post_stress()),
        "mobility_transition": "positions, queues, batteries, observation history, and sampled speeds continue without reset; post-change mobility distribution is not resampled",
        "shifted_components": ["realized shadowing", "blockage process", "external interference", "CSI staleness"],
        "measurement_slots": int(args.measurement_slots),
        "followup_slots": int(args.followup_slots),
    })
    print(f"Change-point experiment written to: {out.resolve()}")


if __name__ == "__main__":
    main()
