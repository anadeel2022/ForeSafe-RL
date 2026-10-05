"""V28 resource-feasibility and reliability-constraint utilities.

This module centralizes the two reviewer-critical invariants:

1. A raw base-4 joint action is executable only when the resource-block
   requirement of all selected modes does not exceed the available budget.
2. The cost critic, episode controller, feasibility test, and reported
   interruption probability are all derived from the same additive numerator
   F - d S, where F and S are failed and scheduled transmissions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

DEFER = 0
DEDICATED_GRANT = 1
PRIORITY_GRANT = 2
PROTECTED_GRANT = 3
N_MODES = 4

MODE_NAMES = (
    "defer",
    "dedicated_grant",
    "priority_grant",
    "protected_grant",
)


@dataclass(frozen=True)
class ConstraintAudit:
    failed: int
    scheduled: int
    cost_limit: float
    n_channels: int
    steps: int

    @property
    def numerator(self) -> float:
        return float(self.failed - self.cost_limit * self.scheduled)

    @property
    def mean_slot_residual(self) -> float:
        denom = max(int(self.n_channels) * int(self.steps), 1)
        return float(self.numerator / denom)

    @property
    def interruption_probability(self) -> float:
        if self.scheduled <= 0:
            return float("nan")
        return float(self.failed / self.scheduled)

    @property
    def ratio_feasible(self) -> bool:
        return bool(self.scheduled > 0 and self.numerator <= 1e-12)


def decode_joint_action(action: int, n_channels: int, n_modes: int = N_MODES) -> tuple[int, ...]:
    action = int(action)
    n_actions = int(n_modes ** int(n_channels))
    if not 0 <= action < n_actions:
        raise ValueError(f"invalid action {action}; expected 0..{n_actions - 1}")
    modes: list[int] = []
    x = action
    for _ in range(int(n_channels)):
        modes.append(int(x % n_modes))
        x //= n_modes
    return tuple(modes)


def mode_resource_cost(mode: int, protection_repetitions: int) -> int:
    mode = int(mode)
    if mode == DEFER:
        return 0
    if mode in (DEDICATED_GRANT, PRIORITY_GRANT):
        return 1
    if mode == PROTECTED_GRANT:
        return max(int(protection_repetitions), 1)
    raise ValueError(f"unknown mode {mode}")


def joint_resource_cost(modes: Sequence[int], protection_repetitions: int) -> int:
    return int(sum(mode_resource_cost(mode, protection_repetitions) for mode in modes))


def is_resource_feasible(
    modes: Sequence[int],
    n_channels: int,
    protection_repetitions: int,
) -> bool:
    if len(tuple(modes)) != int(n_channels):
        return False
    return bool(joint_resource_cost(modes, protection_repetitions) <= int(n_channels))


def valid_action_mask(
    n_channels: int,
    protection_repetitions: int,
    n_modes: int = N_MODES,
) -> np.ndarray:
    n_channels = int(n_channels)
    n_actions = int(n_modes ** n_channels)
    mask = np.zeros(n_actions, dtype=bool)
    for action in range(n_actions):
        modes = decode_joint_action(action, n_channels, n_modes=n_modes)
        mask[action] = is_resource_feasible(modes, n_channels, protection_repetitions)
    if not np.any(mask):
        raise RuntimeError("resource-feasibility mask contains no valid action")
    return mask


def context_action_mask(context: dict | None, n_actions: int) -> np.ndarray:
    """Read and validate a Boolean action mask from an environment context.

    Older contexts without a mask remain usable: all actions are treated as
    available. V28 environments always provide the mask.
    """
    if not context or "action_mask" not in context:
        return np.ones(int(n_actions), dtype=bool)
    mask = np.asarray(context["action_mask"], dtype=bool).reshape(-1)
    if mask.size != int(n_actions):
        raise ValueError(f"action mask has length {mask.size}; expected {n_actions}")
    if not np.any(mask):
        raise ValueError("action mask contains no feasible action")
    return mask


def feasible_action_indices(context: dict | None, n_actions: int) -> np.ndarray:
    return np.flatnonzero(context_action_mask(context, n_actions))


def slot_constraint_residual(
    failed: int | float,
    scheduled: int | float,
    cost_limit: float,
    n_channels: int,
) -> float:
    """Per-slot additive residual g_t = (F_t - d S_t) / K.

    It is exactly zero when no packet is scheduled. Summing this residual over
    an episode preserves the sign of F - d S and is therefore equivalent to
    the ratio constraint F/S <= d whenever S > 0.
    """
    k = max(int(n_channels), 1)
    return float((float(failed) - float(cost_limit) * float(scheduled)) / k)


def constraint_residual_from_info(info: dict, cost_limit: float, n_channels: int | None = None) -> float:
    failed = float(info.get("failed_scheduled", 0) or 0)
    scheduled = float(info.get("scheduled", 0) or 0)
    k = int(n_channels or info.get("n_channels", 1) or 1)
    return slot_constraint_residual(failed, scheduled, cost_limit, k)


def episode_constraint_audit(
    metrics: dict,
    cost_limit: float,
    n_channels: int | None = None,
    steps: int | None = None,
) -> ConstraintAudit:
    failed = int(metrics.get("failed_scheduled_tx", 0) or 0)
    scheduled = int(metrics.get("scheduled_tx", 0) or 0)
    k = int(n_channels or metrics.get("n_channels", 1) or 1)
    t = int(steps or metrics.get("steps_per_episode", 1) or 1)
    return ConstraintAudit(
        failed=failed,
        scheduled=scheduled,
        cost_limit=float(cost_limit),
        n_channels=max(k, 1),
        steps=max(t, 1),
    )


def episode_constraint_residual(
    metrics: dict,
    cost_limit: float,
    n_channels: int | None = None,
    steps: int | None = None,
) -> float:
    return episode_constraint_audit(metrics, cost_limit, n_channels, steps).mean_slot_residual


def assert_allocation(
    resource_blocks: Iterable[int],
    n_channels: int,
) -> tuple[int, ...]:
    blocks = tuple(int(x) for x in resource_blocks)
    if any(block < 0 or block >= int(n_channels) for block in blocks):
        raise AssertionError(f"resource block outside 0..{int(n_channels) - 1}: {blocks}")
    if len(set(blocks)) != len(blocks):
        raise AssertionError(f"resource block reused in one slot: {blocks}")
    if len(blocks) > int(n_channels):
        raise AssertionError(f"resource budget exceeded: {len(blocks)} > {n_channels}")
    return blocks
