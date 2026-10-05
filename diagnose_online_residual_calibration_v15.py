from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from evaluate_foresafe_checkpoint_v15 import load_policy
from viot_env import EnvConfig, VIoTEnv
from foresafe_observation import PartialObservationWrapper
from foresafe_ppo import _combined_mask
from foresafe_risk import upper_cvar_from_quantiles, ShiftMonitor
from run_foresafe import (
    parse_profiles,
    make_stress,
    evaluate_profile,
    pooled_summary,
    audit_rows,
    write_rows,
    evaluation_seed,
)


def calibration_seed(train_seed: int, episode_index: int) -> int:
    return 6_000_000 + int(train_seed) * 10_000 + int(episode_index)


def ema_trace(values, alpha: float) -> list[float]:
    out = []
    state = None
    for x in values:
        x = float(x)
        state = x if state is None else (1.0 - alpha) * state + alpha * x
        out.append(float(state))
    return out


def collect_nominal_calibration(policy, env_factory, episodes: int, train_seed: int):
    h = max(1, int(policy.risk_cfg.horizon))
    alpha = float(policy.risk_cfg.shift_ema_alpha)

    median_abs_residuals = []
    cvar_under_residuals = []

    for ep in range(int(episodes)):
        seed = calibration_seed(train_seed, ep)
        env = env_factory("nominal", seed)
        policy.reset_online_monitor()
        history, _ = env.reset(seed=seed)
        done = False

        q_action = deque()
        q_median = deque()
        q_cvar = deque()
        q_failed = deque()
        q_scheduled = deque()
        q_supported = deque()

        while not done:
            action, diag = policy.act(env, history, deterministic=True)
            history, _, done, info = env.step(action)

            q_action.append(int(action))
            q_median.append(float(diag.chosen_median_risk))
            q_cvar.append(float(diag.chosen_cvar))
            q_failed.append(float(info.get("failed_scheduled", 0) or 0))
            q_scheduled.append(float(info.get("scheduled", 0) or 0))
            q_supported.append(bool(diag.chosen_action_supported))

            if len(q_action) >= h:
                den = float(np.sum(list(q_scheduled)[:h]))
                num = float(np.sum(list(q_failed)[:h]))
                target = num / den if den > 0.0 else 0.0

                if bool(q_supported[0]):
                    median_abs_residuals.append(abs(float(target) - float(q_median[0])))
                    cvar_under_residuals.append(max(0.0, float(target) - float(q_cvar[0])))

                q_action.popleft()
                q_median.popleft()
                q_cvar.popleft()
                q_failed.popleft()
                q_scheduled.popleft()
                q_supported.popleft()

    if not median_abs_residuals or not cvar_under_residuals:
        raise RuntimeError("Nominal calibration produced no supported horizon residuals.")

    median_ema = ema_trace(median_abs_residuals, alpha)
    under_ema = ema_trace(cvar_under_residuals, alpha)

    return {
        "n_horizon_samples": int(len(median_abs_residuals)),
        "median_error_ema_reference_q95": float(np.quantile(median_ema, 0.95)),
        "cvar_underprediction_ema_reference_q95": float(np.quantile(under_ema, 0.95)),
        "median_abs_residual_mean": float(np.mean(median_abs_residuals)),
        "cvar_underprediction_mean": float(np.mean(cvar_under_residuals)),
    }


class ResidualCalibratedPolicy:
    def __init__(self, base_policy, median_error_ref: float, cvar_under_ref: float):
        self.actor_critic = base_policy.actor_critic
        self.risk_model = base_policy.risk_model
        self.risk_cfg = replace(
            base_policy.risk_cfg,
            shift_reference_error=max(float(median_error_ref), 1e-6),
        )
        self.safety_cfg = base_policy.safety_cfg
        self.device = base_policy.device
        self.action_support_counts = (
            None
            if base_policy.action_support_counts is None
            else np.asarray(base_policy.action_support_counts, dtype=np.int64).copy()
        )

        self.shift_monitor = ShiftMonitor(self.risk_cfg)
        self.median_error_ref = max(float(median_error_ref), 1e-6)
        self.cvar_under_ref = max(float(cvar_under_ref), 0.0)

        self._h = max(1, int(self.risk_cfg.horizon))
        self._alpha = float(self.risk_cfg.shift_ema_alpha)
        self._shrinkage_k = 5.0
        self.reset_online_monitor()

    def reset_online_monitor(self):
        self.shift_monitor.reset(self.median_error_ref, True)

        self.global_under_ema = float(self.cvar_under_ref)
        self.action_under_ema = np.full(
            int(self.risk_model.n_actions),
            float(self.cvar_under_ref),
            dtype=np.float64,
        )
        self.action_online_counts = np.zeros(
            int(self.risk_model.n_actions), dtype=np.int64
        )

        self._actions = deque()
        self._medians = deque()
        self._raw_cvars = deque()
        self._failed = deque()
        self._scheduled = deque()
        self._supported = deque()

        self.last_global_excess = 0.0
        self.last_mean_action_excess = 0.0

    def _adjusted_cvar(self, raw_cvar: torch.Tensor) -> torch.Tensor:
        global_excess = max(
            0.0, float(self.global_under_ema) - float(self.cvar_under_ref)
        )

        counts = self.action_online_counts.astype(np.float64)
        shrink = counts / (counts + self._shrinkage_k)
        action_excess_np = np.maximum(
            0.0, self.action_under_ema - float(self.cvar_under_ref)
        ) * shrink

        self.last_global_excess = float(global_excess)
        self.last_mean_action_excess = float(np.mean(action_excess_np))

        margin = torch.as_tensor(
            global_excess + action_excess_np,
            dtype=raw_cvar.dtype,
            device=raw_cvar.device,
        )
        return torch.clamp(raw_cvar + margin, 0.0, 1.0)

    @torch.no_grad()
    def act(self, env, history: np.ndarray, deterministic: bool = True):
        h_t = torch.as_tensor(
            history[None, ...], dtype=torch.float32, device=self.device
        )
        logits, _ = self.actor_critic(h_t)
        all_q = self.risk_model.all_action_quantiles(h_t)
        raw_cvar = upper_cvar_from_quantiles(
            all_q, self.risk_cfg.quantiles, self.risk_cfg.cvar_alpha
        )
        calibrated_cvar = self._adjusted_cvar(raw_cvar)

        median_idx = min(
            range(len(self.risk_cfg.quantiles)),
            key=lambda i: abs(float(self.risk_cfg.quantiles[i]) - 0.50),
        )
        limit = self.shift_monitor.effective_limit()

        (
            mask,
            native,
            forced,
            cvar_count,
            _legacy_removed,
            urgency_guarded,
            supported_native_count,
            strict_supported_pass,
            risk_rejected,
            risk_relaxed,
            urgency_score,
        ) = _combined_mask(
            env,
            calibrated_cvar,
            limit,
            True,
            self.safety_cfg,
            history,
            self.device,
            support_counts=self.action_support_counts,
            return_details=True,
        )

        masked_logits = logits[0].masked_fill(~mask, -1e9)
        action = (
            int(torch.argmax(masked_logits).item())
            if deterministic
            else int(Categorical(logits=masked_logits).sample().item())
        )

        chosen_supported = True
        if self.action_support_counts is not None:
            chosen_supported = bool(
                int(self.action_support_counts[action])
                >= int(self.safety_cfg.min_action_samples_for_filter)
            )

        feasible_cal = calibrated_cvar[native]

        class D:
            pass

        d = D()
        d.native_feasible_action_count = int(native.sum().item())
        d.supported_native_action_count = int(supported_native_count)
        d.strict_supported_cvar_pass_count = int(strict_supported_pass)
        d.cvar_admissible_action_count = int(cvar_count)
        d.safe_action_count = int(mask.sum().item())
        d.chosen_action_supported = bool(chosen_supported)
        d.chosen_median_risk = float(all_q[action, median_idx].item())
        d.chosen_raw_cvar = float(raw_cvar[action].item())
        d.chosen_cvar = float(calibrated_cvar[action].item())
        d.min_feasible_cvar = float(feasible_cal.min().item())
        d.effective_limit = float(limit)
        d.shield_ready = True
        d.forced_min_risk = bool(forced)
        d.risk_rejected = int(risk_rejected)
        d.risk_relaxed = int(risk_relaxed)
        d.urgency_guarded = int(urgency_guarded)
        d.urgency_score = float(urgency_score)
        d.global_calibration_excess = float(self.last_global_excess)
        d.mean_action_calibration_excess = float(self.last_mean_action_excess)
        d.action = int(action)
        return action, d

    def observe_realized_outcome(
        self,
        action: int,
        predicted_median: float,
        predicted_raw_cvar: float,
        failed_scheduled: float,
        scheduled: float,
        prediction_supported: bool = True,
    ):
        self._actions.append(int(action))
        self._medians.append(float(predicted_median))
        self._raw_cvars.append(float(predicted_raw_cvar))
        self._failed.append(float(failed_scheduled))
        self._scheduled.append(float(scheduled))
        self._supported.append(bool(prediction_supported))

        if len(self._actions) < self._h:
            return

        den = float(np.sum(list(self._scheduled)[: self._h]))
        num = float(np.sum(list(self._failed)[: self._h]))
        target = num / den if den > 0.0 else 0.0

        action0 = int(self._actions[0])
        median0 = float(self._medians[0])
        cvar0 = float(self._raw_cvars[0])
        supported0 = bool(self._supported[0])

        if supported0:
            self.shift_monitor.update(median0, target)

            under = max(0.0, float(target) - cvar0)
            a = self._alpha
            self.global_under_ema = (
                (1.0 - a) * float(self.global_under_ema) + a * under
            )

            n = int(self.action_online_counts[action0])
            if n == 0:
                self.action_under_ema[action0] = under
            else:
                self.action_under_ema[action0] = (
                    (1.0 - a) * float(self.action_under_ema[action0]) + a * under
                )
            self.action_online_counts[action0] += 1

        self._actions.popleft()
        self._medians.popleft()
        self._raw_cvars.popleft()
        self._failed.popleft()
        self._scheduled.popleft()
        self._supported.popleft()


def evaluate_calibrated(policy, env_factory, episodes: int, train_seed: int, profile_name: str):
    rows = []
    for ep in range(int(episodes)):
        seed = evaluation_seed(train_seed, ep)
        env = env_factory(profile_name, seed)
        policy.reset_online_monitor()
        history, _ = env.reset(seed=seed)
        done = False

        raw_cvars, cal_cvars = [], []
        global_excesses, action_excesses = [], []
        strict, admissible, safe = [], [], []
        limits, forced = [], []
        medians = []

        while not done:
            action, diag = policy.act(env, history, deterministic=True)
            history, _, done, info = env.step(action)

            policy.observe_realized_outcome(
                action=action,
                predicted_median=diag.chosen_median_risk,
                predicted_raw_cvar=diag.chosen_raw_cvar,
                failed_scheduled=float(info.get("failed_scheduled", 0) or 0),
                scheduled=float(info.get("scheduled", 0) or 0),
                prediction_supported=diag.chosen_action_supported,
            )

            medians.append(diag.chosen_median_risk)
            raw_cvars.append(diag.chosen_raw_cvar)
            cal_cvars.append(diag.chosen_cvar)
            global_excesses.append(diag.global_calibration_excess)
            action_excesses.append(diag.mean_action_calibration_excess)
            strict.append(diag.strict_supported_cvar_pass_count)
            admissible.append(diag.cvar_admissible_action_count)
            safe.append(diag.safe_action_count)
            limits.append(diag.effective_limit)
            forced.append(int(diag.forced_min_risk))

        row = dict(env.episode_metrics())
        row.update({
            "train_seed": int(train_seed),
            "evaluation_seed": int(seed),
            "evaluation_episode": int(ep),
            "evaluation_profile": str(profile_name),
            "mean_predicted_median_risk": float(np.mean(medians)),
            "mean_raw_predicted_cvar": float(np.mean(raw_cvars)),
            "mean_calibrated_cvar": float(np.mean(cal_cvars)),
            "mean_global_calibration_excess": float(np.mean(global_excesses)),
            "mean_action_calibration_excess": float(np.mean(action_excesses)),
            "mean_strict_supported_cvar_pass_count": float(np.mean(strict)),
            "mean_cvar_admissible_action_count": float(np.mean(admissible)),
            "mean_safe_action_count": float(np.mean(safe)),
            "mean_effective_cvar_limit": float(np.mean(limits)),
            "forced_min_risk_rate": float(np.mean(forced)),
            "shift_error_ema_end": float(policy.shift_monitor.error_ema),
        })
        rows.append(row)
    return rows


def mean_col(rows, key):
    vals = [
        float(r[key]) for r in rows
        if key in r and np.isfinite(float(r[key]))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(
        description="Frozen-checkpoint nominal residual calibration diagnostic for ForeSafe V1.5."
    )
    p.add_argument("--run_dir", required=True)
    p.add_argument(
        "--profiles",
        default="nominal,standard_compound,compound_severe",
    )
    p.add_argument("--calibration_episodes", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--measurement_slots", type=int, default=150)
    p.add_argument("--followup_slots", type=int, default=251)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    profiles = parse_profiles(args.profiles)

    base_policy, po_cfg, ckpt = load_policy(run_dir, args.device)
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

    calibration = collect_nominal_calibration(
        base_policy,
        env_factory,
        args.calibration_episodes,
        train_seed,
    )

    calibrated_policy = ResidualCalibratedPolicy(
        base_policy,
        median_error_ref=calibration["median_error_ema_reference_q95"],
        cvar_under_ref=calibration["cvar_underprediction_ema_reference_q95"],
    )

    out = run_dir / "online_residual_calibration_diagnostic"
    out.mkdir(parents=True, exist_ok=True)

    with (out / "nominal_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    summary_rows = []

    for profile in profiles:
        baseline_policy, _, _ = load_policy(run_dir, args.device)
        base_rows = evaluate_profile(
            baseline_policy,
            env_factory,
            args.eval_episodes,
            train_seed,
            profile,
            deterministic=True,
        )
        base_audit = audit_rows(base_rows, n_channels)
        base_pool = pooled_summary(base_rows)

        cal_rows = evaluate_calibrated(
            calibrated_policy,
            env_factory,
            args.eval_episodes,
            train_seed,
            profile,
        )
        cal_audit = audit_rows(cal_rows, n_channels)
        cal_pool = pooled_summary(cal_rows)

        write_rows(out / f"baseline_{profile}_episodes.csv", base_rows)
        write_rows(out / f"calibrated_{profile}_episodes.csv", cal_rows)

        for label, rows, pool, audit in [
            ("baseline", base_rows, base_pool, base_audit),
            ("residual_calibrated", cal_rows, cal_pool, cal_audit),
        ]:
            summary_rows.append({
                "profile": profile,
                "method": label,
                "pooled_interruption_probability":
                    pool["pooled_cohort_interruption_probability"],
                "pooled_on_time_delivery_ratio":
                    pool["pooled_cohort_on_time_delivery_ratio"],
                "pooled_deadline_miss_ratio":
                    pool["pooled_cohort_packet_deadline_miss_ratio"],
                "mean_traffic_rate_mbps":
                    pool["mean_cohort_delivered_traffic_rate_mbps"],
                "mean_defer_probability":
                    mean_col(rows, "defer_probability"),
                "mean_effective_cvar_limit":
                    mean_col(rows, "mean_effective_cvar_limit"),
                "mean_strict_supported_cvar_pass_count":
                    mean_col(rows, "mean_strict_supported_cvar_pass_count"),
                "mean_cvar_admissible_action_count":
                    mean_col(rows, "mean_cvar_admissible_action_count"),
                "mean_raw_predicted_cvar":
                    mean_col(rows, "mean_raw_predicted_cvar")
                    if label == "residual_calibrated"
                    else mean_col(rows, "mean_predicted_cvar"),
                "mean_calibrated_cvar":
                    mean_col(rows, "mean_calibrated_cvar")
                    if label == "residual_calibrated"
                    else float("nan"),
                "mean_global_calibration_excess":
                    mean_col(rows, "mean_global_calibration_excess")
                    if label == "residual_calibrated"
                    else float("nan"),
                "mean_action_calibration_excess":
                    mean_col(rows, "mean_action_calibration_excess")
                    if label == "residual_calibrated"
                    else float("nan"),
                "audit_passed": bool(audit["passed"]),
            })

    summary_path = out / "diagnostic_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("Nominal calibration:")
    print(
        f"  horizon samples: {calibration['n_horizon_samples']}; "
        f"median-error EMA q95={calibration['median_error_ema_reference_q95']:.6f}; "
        f"CVaR-underprediction EMA q95="
        f"{calibration['cvar_underprediction_ema_reference_q95']:.6f}"
    )
    print(f"Diagnostic written to: {out}")

    for r in summary_rows:
        print(
            f"[{r['profile']} | {r['method']}] "
            f"intr={r['pooled_interruption_probability']:.6f}; "
            f"OTD={r['pooled_on_time_delivery_ratio']:.6f}; "
            f"miss={r['pooled_deadline_miss_ratio']:.6f}; "
            f"thr={r['mean_traffic_rate_mbps']:.6f}; "
            f"defer={r['mean_defer_probability']:.4f}; "
            f"strict={r['mean_strict_supported_cvar_pass_count']:.2f}; "
            f"admissible={r['mean_cvar_admissible_action_count']:.2f}; "
            f"limit={r['mean_effective_cvar_limit']:.4f}; "
            f"audit={r['audit_passed']}"
        )


if __name__ == "__main__":
    main()
