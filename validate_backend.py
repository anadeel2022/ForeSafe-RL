from __future__ import annotations
import numpy as np

from viot_env import EnvConfig, StressConfig, VIoTEnv
from foresafe_config import PartialObservationConfig
from foresafe_observation import PartialObservationWrapper


REQUIRED_ENV_FIELDS = {
    "measurement_cohort_slots",
    "arrival_cutoff_slots",
    "steps_per_episode",
    "n_devices",
    "n_channels",
}
REQUIRED_STRESS_FIELDS = {
    "actual_shadowing_std_db",
    "blockage_start_probability",
    "blockage_mean_duration_slots",
    "blockage_attenuation_db",
    "external_interference_probability",
    "external_interferer_power_dbm",
    "external_interferer_distance_m",
    "external_interferer_distance_jitter_m",
    "stale_csi_slots",
    "actual_speed_mean_mps",
    "actual_speed_std_mps",
}
REQUIRED_COHORT_METRICS = {
    "cohort_offered_packets",
    "cohort_delivered_packets",
    "cohort_deadline_missed_packets",
    "cohort_pending_packets_end",
    "cohort_packet_conservation_error",
    "cohort_interruption_probability",
    "cohort_scheduled_tx",
    "cohort_failed_scheduled_tx",
    "cohort_on_time_delivery_ratio",
    "cohort_packet_deadline_miss_ratio",
    "cohort_delivered_traffic_rate_mbps",
}


def main():
    env_fields = set(getattr(EnvConfig, "__dataclass_fields__", {}))
    stress_fields = set(getattr(StressConfig, "__dataclass_fields__", {}))

    missing_env = sorted(REQUIRED_ENV_FIELDS - env_fields)
    missing_stress = sorted(REQUIRED_STRESS_FIELDS - stress_fields)
    if missing_env:
        raise RuntimeError(
            "viot_env.py is not the required terminal-safe final backend; "
            f"missing EnvConfig fields: {missing_env}"
        )
    if missing_stress:
        raise RuntimeError(
            "viot_env.py is not the required final stress backend; "
            f"missing StressConfig fields: {missing_stress}"
        )

    cfg = EnvConfig(
        seed=123,
        n_devices=6,
        n_channels=3,
        steps_per_episode=8,
        measurement_cohort_slots=2,
        arrival_cutoff_slots=0,
        stress=StressConfig(name="nominal"),
    )
    wrapped = PartialObservationWrapper(
        VIoTEnv(cfg),
        PartialObservationConfig(
            history_len=4,
            observation_delay_slots=1,
            gaussian_noise_std=0.0,
            dropout_probability=0.0,
        ),
        seed=123,
    )

    history, _ = wrapped.reset(seed=123)
    if history.ndim != 2:
        raise RuntimeError(f"Unexpected wrapped observation shape: {history.shape}")

    mask = wrapped.resource_feasible_action_mask()
    if int(mask.sum()) != 42:
        raise RuntimeError(
            f"Expected 42 resource-feasible K=3,R=2 actions; got {int(mask.sum())}."
        )

    done = False
    while not done:
        history, _reward, done, _info = wrapped.step(0)

    metrics = wrapped.episode_metrics()
    missing_metrics = sorted(REQUIRED_COHORT_METRICS - set(metrics))
    if missing_metrics:
        raise RuntimeError(
            "Backend does not expose required terminal-safe cohort metrics: "
            f"{missing_metrics}"
        )

    for key in ("resource_budget_violation_count", "invalid_action_selection_count"):
        if key not in metrics:
            raise RuntimeError(f"Backend missing required audit metric: {key}")

    if float(metrics["resource_budget_violation_count"]) != 0.0:
        raise RuntimeError("Unexpected resource-budget violation in backend smoke test.")
    if float(metrics["invalid_action_selection_count"]) != 0.0:
        raise RuntimeError("Unexpected invalid action in backend smoke test.")

    print("ForeSafe-RL final MobiSafe backend compatibility passed.")


if __name__ == "__main__":
    main()
