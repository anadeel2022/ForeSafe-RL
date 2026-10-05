from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from robustness_common import (
    compact_summary,
    discover_final_runs,
    dump_manifest,
    evaluate_static,
    load_method,
    nominal,
    write_rows,
)

METHODS = ("foresafe", "foresafe_no_belief", "belief_ppo")

# One-factor-at-a-time conditions around the frozen training point d=2, sigma=.02, p=.05.
CONDITIONS = (
    ("trained", 2, 0.02, 0.05),
    ("delay_0", 0, 0.02, 0.05),
    ("delay_4", 4, 0.02, 0.05),
    ("delay_8", 8, 0.02, 0.05),
    ("noise_0", 2, 0.00, 0.05),
    ("noise_0p05", 2, 0.05, 0.05),
    ("noise_0p10", 2, 0.10, 0.05),
    ("dropout_0", 2, 0.02, 0.00),
    ("dropout_0p15", 2, 0.02, 0.15),
    ("dropout_0p30", 2, 0.02, 0.30),
)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Evaluation-only observation impairment sweep")
    p.add_argument("--foresafe_root", required=True)
    p.add_argument("--matched_root", required=True)
    p.add_argument("--seeds", default="11,12,13,14,15,16,17,18,19,20")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--methods", default=",".join(METHODS), help="Comma-separated subset of methods")
    p.add_argument("--conditions", default=",".join(x[0] for x in CONDITIONS), help="Comma-separated subset of condition names")
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", default="outputs_robustness_v1/observation_sweep")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    seeds = parse_seeds(args.seeds)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = [x for x in methods if x not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    condition_names = [x.strip() for x in args.conditions.split(",") if x.strip()]
    condition_map = {x[0]: x for x in CONDITIONS}
    unknown_conditions = [x for x in condition_names if x not in condition_map]
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    selected_conditions = [condition_map[x] for x in condition_names]
    roots = discover_final_runs(Path(args.foresafe_root), Path(args.matched_root), seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []

    for method in methods:
        for seed in seeds:
            policy, trained_po, ckpt = load_method(method, roots[method][seed], args.device)
            for name, delay, noise, dropout in selected_conditions:
                po = replace(
                    trained_po,
                    observation_delay_slots=int(delay),
                    gaussian_noise_std=float(noise),
                    dropout_probability=float(dropout),
                )
                rows, summary, audit = evaluate_static(
                    method, policy, ckpt, po, nominal(), args.episodes,
                    args.measurement_slots, args.followup_slots,
                )
                for row in rows:
                    row["condition"] = name
                write_rows(out / method / f"seed{seed}" / f"{name}_episodes.csv", rows)
                summaries.append(compact_summary(method, seed, name, summary, audit, rows))
                print(
                    f"[{method} seed={seed} {name}] "
                    f"intr={summary['pooled_cohort_interruption_probability']:.6f}; "
                    f"OTD={summary['pooled_cohort_on_time_delivery_ratio']:.6f}; "
                    f"thr={summary['mean_cohort_delivered_traffic_rate_mbps']:.6f}; "
                    f"audit={audit['passed']}"
                )

    write_rows(out / "observation_sweep_summary.csv", summaries)
    dump_manifest(out / "manifest.json", {
        "experiment": "observation_impairment_one_factor_at_a_time",
        "evaluation_only": True,
        "physical_stress": asdict(nominal()),
        "episodes_per_method_seed_condition": int(args.episodes),
        "measurement_slots": int(args.measurement_slots),
        "followup_slots": int(args.followup_slots),
        "methods": methods,
        "seeds": seeds,
        "conditions": [
            {"name": n, "observation_delay_slots": d, "gaussian_noise_std": z, "dropout_probability": q}
            for n, d, z, q in selected_conditions
        ],
        "note": "history_len is preserved from each trained checkpoint; only the named observation impairment is changed.",
    })
    print(f"Observation sweep written to: {out.resolve()}")


if __name__ == "__main__":
    main()
