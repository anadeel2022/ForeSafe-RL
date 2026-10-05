
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from foresafe_config import PartialObservationConfig, PPOConfig
from foresafe_observation import PartialObservationWrapper
from matched_baselines import (
    LagrangianConfig,
    train_recurrent_ppo_continuing,
    train_recurrent_ppo_lagrangian_continuing,
)
from viot_env import EnvConfig, VIoTEnv


def _verify_frozen_hash_manifest(root: Path) -> None:
    manifest = root / "code_sha256_manifest.csv"
    if not manifest.is_file():
        raise RuntimeError("Missing code_sha256_manifest.csv")

    with manifest.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Empty frozen hash manifest.")

    mismatches = []
    for row in rows:
        raw_path = str(row["Path"]).replace("\\", "/")
        name = raw_path.split("/")[-1]
        path = root / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        expected = str(row["Hash"]).strip().upper()
        if actual != expected:
            mismatches.append(f"{name}: SHA256 mismatch")

    if mismatches:
        raise RuntimeError(
            "Frozen ForeSafe source changed:\n  " + "\n  ".join(mismatches)
        )


def _tiny_training_checks() -> None:
    po_cfg = PartialObservationConfig(
        history_len=8,
        observation_delay_slots=2,
        gaussian_noise_std=0.02,
        dropout_probability=0.05,
    )
    ppo_cfg = PPOConfig(device="cpu")

    cfg = EnvConfig(
        seed=991,
        n_devices=36,
        n_channels=3,
        steps_per_episode=40,
        measurement_cohort_slots=0,
        arrival_cutoff_slots=0,
    )
    env = PartialObservationWrapper(VIoTEnv(cfg), po_cfg, seed=991)
    history, _ = env.reset(seed=991)

    mask = env.resource_feasible_action_mask()
    if int(np.sum(mask)) != 42:
        raise RuntimeError(
            f"Expected 42 feasible K=3,R=2 actions; found {int(np.sum(mask))}."
        )
    if history.shape[0] != 8:
        raise RuntimeError(f"Expected history length 8; found {history.shape}.")

    result = train_recurrent_ppo_continuing(
        env,
        ppo_cfg,
        rollout_updates=2,
        rollout_steps=8,
        seed=991,
    )
    if int(result["total_steps"]) != 16:
        raise RuntimeError("Belief-PPO tiny continuing training length mismatch.")

    cfg2 = EnvConfig(
        seed=992,
        n_devices=36,
        n_channels=3,
        steps_per_episode=40,
        measurement_cohort_slots=0,
        arrival_cutoff_slots=0,
    )
    env2 = PartialObservationWrapper(VIoTEnv(cfg2), po_cfg, seed=992)
    result2 = train_recurrent_ppo_lagrangian_continuing(
        env2,
        ppo_cfg,
        LagrangianConfig(),
        rollout_updates=2,
        rollout_steps=8,
        seed=992,
    )
    if int(result2["total_steps"]) != 16:
        raise RuntimeError("Belief-PPO-Lagrangian tiny training length mismatch.")
    if not np.isfinite(float(result2["lambda_value"])):
        raise RuntimeError("Non-finite Lagrange multiplier.")


def main() -> None:
    root = Path(__file__).resolve().parent
    _verify_frozen_hash_manifest(root)
    _tiny_training_checks()
    print("ForeSafe-RL V1.5 matched-experiment validation passed.")
    print("Frozen ForeSafe hashes match the pre-final manifest.")
    print("Belief-PPO and Belief-PPO-Lagrangian tiny continuing-task checks passed.")


if __name__ == "__main__":
    main()
