from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from robustness_common import (
    METHODS,
    compact_summary,
    discover_final_runs,
    dump_manifest,
    evaluate_static,
    interpolate_stress,
    load_method,
    write_rows,
)

LEVELS = (0.25, 0.50, 0.75)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Evaluation-only mild-to-standard stress ladder")
    p.add_argument("--foresafe_root", required=True)
    p.add_argument("--matched_root", required=True)
    p.add_argument("--seeds", default="11,12,13,14,15,16,17,18,19,20")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--methods", default=",".join(METHODS), help="Comma-separated subset of methods")
    p.add_argument("--levels", default=",".join(str(x) for x in LEVELS), help="Comma-separated interpolation levels")
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", default="outputs_robustness_v1/stress_ladder")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    seeds = parse_seeds(args.seeds)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = [x for x in methods if x not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    levels = [float(x.strip()) for x in args.levels.split(",") if x.strip()]
    if any(all(abs(x-y) > 1e-9 for y in LEVELS) for x in levels):
        raise ValueError(f"Levels must be selected from {LEVELS}")
    roots = discover_final_runs(Path(args.foresafe_root), Path(args.matched_root), seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []

    for method in methods:
        for seed in seeds:
            policy, po_cfg, ckpt = load_method(method, roots[method][seed], args.device)
            for level in levels:
                stress = interpolate_stress(level)
                condition = f"ladder_{int(round(level * 100)):03d}"
                rows, summary, audit = evaluate_static(
                    method, policy, ckpt, po_cfg, stress, args.episodes,
                    args.measurement_slots, args.followup_slots,
                )
                for row in rows:
                    row["condition"] = condition
                    row["ladder_alpha"] = float(level)
                write_rows(out / method / f"seed{seed}" / f"{condition}_episodes.csv", rows)
                summaries.append(compact_summary(method, seed, condition, summary, audit, rows))
                print(
                    f"[{method} seed={seed} alpha={level:.2f}] "
                    f"intr={summary['pooled_cohort_interruption_probability']:.6f}; "
                    f"OTD={summary['pooled_cohort_on_time_delivery_ratio']:.6f}; "
                    f"thr={summary['mean_cohort_delivered_traffic_rate_mbps']:.6f}; "
                    f"audit={audit['passed']}"
                )

    write_rows(out / "stress_ladder_summary.csv", summaries)
    stress_configs = {f"alpha_{x:.2f}": asdict(interpolate_stress(x)) for x in levels}
    dump_manifest(out / "manifest.json", {
        "experiment": "mild_minus_to_standard_compound_interpolation_ladder",
        "evaluation_only": True,
        "episodes_per_method_seed_level": int(args.episodes),
        "measurement_slots": int(args.measurement_slots),
        "followup_slots": int(args.followup_slots),
        "methods": methods,
        "seeds": seeds,
        "intermediate_levels": levels,
        "stress_configs": stress_configs,
        "endpoint_reuse": "alpha=0 mild_minus and alpha=1 standard_compound are already available from the frozen final 200-episode campaign and are not rerun here.",
    })
    print(f"Stress ladder written to: {out.resolve()}")


if __name__ == "__main__":
    main()
