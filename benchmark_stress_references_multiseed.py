from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from viot_env import EnvConfig, VIoTEnv
from run_foresafe import make_stress, evaluation_seed

SCHEMES = ("rq_greedy", "maxweight", "protected_maxweight")


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_action(env: VIoTEnv, scheme: str) -> int:
    # Use the environment's deterministic reference builders directly.
    # This avoids constructing the full decision_context dictionary at every
    # slot while preserving exactly the same reference actions.
    if scheme == "rq_greedy":
        return int(env._build_oracle_action(priority=False))
    if scheme == "maxweight":
        return int(env._build_oracle_action(priority=True))
    if scheme == "protected_maxweight":
        return int(env._build_protected_maxweight_action())
    raise ValueError(scheme)


def pooled(rows: list[dict]) -> dict:
    failed = sum(float(r.get("cohort_failed_scheduled_tx", 0) or 0) for r in rows)
    scheduled = sum(float(r.get("cohort_scheduled_tx", 0) or 0) for r in rows)
    offered = sum(float(r.get("cohort_offered_packets", 0) or 0) for r in rows)
    delivered = sum(float(r.get("cohort_delivered_packets", 0) or 0) for r in rows)
    missed = sum(float(r.get("cohort_deadline_missed_packets", 0) or 0) for r in rows)
    thr = [float(r["cohort_delivered_traffic_rate_mbps"]) for r in rows]
    defer = [float(r["defer_probability"]) for r in rows]
    return {
        "interruption_probability": failed / scheduled if scheduled else np.nan,
        "on_time_delivery_ratio": delivered / offered if offered else np.nan,
        "deadline_miss_ratio": missed / offered if offered else np.nan,
        "throughput_mbps": float(np.mean(thr)),
        "defer_probability": float(np.mean(defer)),
        "cohort_failed": int(failed),
        "cohort_scheduled": int(scheduled),
        "cohort_offered": int(offered),
    }


def audit(rows: list[dict], n_channels: int) -> bool:
    return bool(
        max(int(r.get("cohort_pending_packets_end", 0) or 0) for r in rows) == 0
        and max(abs(int(r.get("cohort_packet_conservation_error", 0) or 0)) for r in rows) == 0
        and sum(int(r.get("invalid_action_selection_count", 0) or 0) for r in rows) == 0
        and sum(int(r.get("resource_budget_violation_count", 0) or 0) for r in rows) == 0
        and max(int(r.get("max_resource_blocks_used", 0) or 0) for r in rows) <= int(n_channels)
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def run_one(scheme, profile, train_seed, episodes, n_devices, n_channels, measurement_slots, followup_slots):
    rows=[]
    for ep in range(int(episodes)):
        seed=evaluation_seed(train_seed, ep)
        cfg=EnvConfig(seed=seed,n_devices=n_devices,n_channels=n_channels,
                      steps_per_episode=measurement_slots+followup_slots,
                      measurement_cohort_slots=measurement_slots,arrival_cutoff_slots=0,
                      stress=make_stress(profile))
        env=VIoTEnv(cfg)
        env.reset(seed=seed)
        done=False
        while not done:
            action=choose_action(env,scheme)
            _,_,done,_=env.step(action)
        row=dict(env.episode_metrics())
        row.update({"scheme":scheme,"profile":profile,"train_seed":train_seed,
                    "evaluation_seed":seed,"evaluation_episode":ep})
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default="11,12,13,14,15,16,17,18,19,20")
    p.add_argument("--profiles",default="nominal,mild_minus,standard_compound,compound_severe")
    p.add_argument("--eval_episodes",type=int,default=200)
    p.add_argument("--n_devices",type=int,default=36)
    p.add_argument("--n_channels",type=int,default=3)
    p.add_argument("--measurement_slots",type=int,default=150)
    p.add_argument("--followup_slots",type=int,default=251)
    p.add_argument("--out",default="outputs_final_completion/deterministic_references")
    args=p.parse_args()
    seeds=parse_seeds(args.seeds); profiles=[x.strip() for x in args.profiles.split(",") if x.strip()]
    out=Path(args.out); summaries=[]
    for seed in seeds:
        for profile in profiles:
            for scheme in SCHEMES:
                rows=run_one(scheme,profile,seed,args.eval_episodes,args.n_devices,args.n_channels,args.measurement_slots,args.followup_slots)
                ok=audit(rows,args.n_channels)
                s=pooled(rows)
                s.update({"scheme":scheme,"profile":profile,"train_seed":seed,"episodes":len(rows),"audit_passed":ok})
                summaries.append(s)
                write_csv(out/scheme/f"seed{seed}"/f"{profile}_episodes.csv",rows)
                print(f"[{scheme} seed={seed} {profile}] intr={s['interruption_probability']:.6f}; OTD={s['on_time_delivery_ratio']:.6f}; thr={s['throughput_mbps']:.6f}; audit={ok}")
                if not ok: raise RuntimeError("deterministic reference audit failed")
    write_csv(out/"deterministic_reference_summary.csv",summaries)
    (out/"manifest.json").write_text(json.dumps({
        "schemes":list(SCHEMES),"seeds":seeds,"profiles":profiles,"eval_episodes":args.eval_episodes,
        "measurement_slots":args.measurement_slots,"followup_slots":args.followup_slots,
        "note":"RQ-Greedy is the former environment oracle_action label. It is not clairvoyant; it uses available nominal risk information."
    },indent=2),encoding="utf-8")

if __name__=="__main__": main()
