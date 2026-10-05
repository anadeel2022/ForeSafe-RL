from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from viot_env import EnvConfig, VIoTEnv
from run_foresafe import make_stress, evaluation_seed


SCHEMES = (
    "oracle",
    "maxweight",
    "protected_maxweight",
)


def choose_action(env: VIoTEnv, scheme: str) -> int:
    ctx = env.decision_context()
    if scheme == "oracle":
        return int(ctx["oracle_action"])
    if scheme == "maxweight":
        return int(ctx["maxweight_action"])
    if scheme == "protected_maxweight":
        return int(ctx["protected_maxweight_action"])
    raise ValueError(f"unknown scheme: {scheme}")


def pooled(rows: list[dict]) -> dict:
    failed = float(sum(float(r.get("cohort_failed_scheduled_tx", 0) or 0) for r in rows))
    scheduled = float(sum(float(r.get("cohort_scheduled_tx", 0) or 0) for r in rows))
    offered = float(sum(float(r.get("cohort_offered_packets", 0) or 0) for r in rows))
    delivered = float(sum(float(r.get("cohort_delivered_packets", 0) or 0) for r in rows))
    missed = float(sum(float(r.get("cohort_deadline_missed_packets", 0) or 0) for r in rows))
    throughput = [
        float(r["cohort_delivered_traffic_rate_mbps"])
        for r in rows
        if np.isfinite(float(r.get("cohort_delivered_traffic_rate_mbps", np.nan)))
    ]
    defer = [
        float(r["defer_probability"])
        for r in rows
        if np.isfinite(float(r.get("defer_probability", np.nan)))
    ]
    return {
        "pooled_interruption_probability": failed / scheduled if scheduled > 0 else float("nan"),
        "pooled_on_time_delivery_ratio": delivered / offered if offered > 0 else float("nan"),
        "pooled_deadline_miss_ratio": missed / offered if offered > 0 else float("nan"),
        "mean_traffic_rate_mbps": float(np.mean(throughput)) if throughput else float("nan"),
        "mean_defer_probability": float(np.mean(defer)) if defer else float("nan"),
        "cohort_failed": int(failed),
        "cohort_scheduled": int(scheduled),
        "cohort_offered": int(offered),
    }


def audit(rows: list[dict], n_channels: int) -> dict:
    max_pending = max(int(r.get("cohort_pending_packets_end", 0) or 0) for r in rows)
    max_conservation = max(abs(int(r.get("cohort_packet_conservation_error", 0) or 0)) for r in rows)
    invalid = sum(int(r.get("invalid_action_selection_count", 0) or 0) for r in rows)
    resource = sum(int(r.get("resource_budget_violation_count", 0) or 0) for r in rows)
    max_rb = max(int(r.get("max_resource_blocks_used", 0) or 0) for r in rows)
    passed = (
        max_pending == 0
        and max_conservation == 0
        and invalid == 0
        and resource == 0
        and max_rb <= int(n_channels)
    )
    return {
        "passed": bool(passed),
        "max_pending": max_pending,
        "max_conservation_error": max_conservation,
        "invalid_action_count": invalid,
        "resource_violation_count": resource,
        "max_resource_blocks_used": max_rb,
    }


def run_scheme(
    scheme: str,
    profile: str,
    train_seed: int,
    episodes: int,
    n_devices: int,
    n_channels: int,
    measurement_slots: int,
    followup_slots: int,
) -> list[dict]:
    rows = []

    for ep in range(int(episodes)):
        seed = evaluation_seed(train_seed, ep)
        cfg = EnvConfig(
            seed=seed,
            n_devices=int(n_devices),
            n_channels=int(n_channels),
            steps_per_episode=int(measurement_slots + followup_slots),
            measurement_cohort_slots=int(measurement_slots),
            arrival_cutoff_slots=0,
            stress=make_stress(profile),
        )
        env = VIoTEnv(cfg)
        _, _ = env.reset(seed=seed)
        done = False

        while not done:
            action = choose_action(env, scheme)
            _, _, done, _ = env.step(action)

        row = dict(env.episode_metrics())
        row.update({
            "scheme": scheme,
            "profile": profile,
            "evaluation_episode": int(ep),
            "evaluation_seed": int(seed),
        })
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict]):
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(
        description="Terminal-safe stress benchmark for built-in deterministic V-IoT references."
    )
    p.add_argument("--profiles", default="nominal,standard_compound,compound_severe")
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--train_seed", type=int, default=11)
    p.add_argument("--n_devices", type=int, default=36)
    p.add_argument("--n_channels", type=int, default=3)
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--out", default="stress_reference_benchmark")
    args = p.parse_args()

    profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []

    for profile in profiles:
        for scheme in SCHEMES:
            rows = run_scheme(
                scheme=scheme,
                profile=profile,
                train_seed=args.train_seed,
                episodes=args.eval_episodes,
                n_devices=args.n_devices,
                n_channels=args.n_channels,
                measurement_slots=args.measurement_slots,
                followup_slots=args.followup_slots,
            )
            write_csv(out / f"{profile}_{scheme}_episodes.csv", rows)

            s = pooled(rows)
            a = audit(rows, args.n_channels)
            result = {
                "profile": profile,
                "scheme": scheme,
                **s,
                "audit_passed": bool(a["passed"]),
            }
            summaries.append(result)

            print(
                f"[{profile} | {scheme}] "
                f"intr={s['pooled_interruption_probability']:.6f}; "
                f"OTD={s['pooled_on_time_delivery_ratio']:.6f}; "
                f"miss={s['pooled_deadline_miss_ratio']:.6f}; "
                f"thr={s['mean_traffic_rate_mbps']:.6f}; "
                f"defer={s['mean_defer_probability']:.4f}; "
                f"audit={a['passed']}"
            )

    write_csv(out / "stress_reference_summary.csv", summaries)
    with (out / "stress_reference_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "profiles": profiles,
            "schemes": list(SCHEMES),
            "eval_episodes": int(args.eval_episodes),
            "train_seed_for_matched_evaluation_namespace": int(args.train_seed),
            "measurement_slots": int(args.measurement_slots),
            "followup_slots": int(args.followup_slots),
            "n_devices": int(args.n_devices),
            "n_channels": int(args.n_channels),
            "note": (
                "These are built-in deterministic reference schedulers using the "
                "environment's nominal risk/priority information. They are not a "
                "clairvoyant upper bound and are used only to characterize stress difficulty."
            ),
        }, f, indent=2)

    print(f"Results written to: {out.resolve()}")


if __name__ == "__main__":
    main()
