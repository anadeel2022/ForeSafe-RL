
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from foresafe_config import PPOConfig
from foresafe_ppo import BeliefActorCritic, _gae_bootstrap
from viot_v28_constraints import constraint_residual_from_info


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


class RecurrentPPOPolicy:
    """Deterministic/stochastic recurrent PPO policy using only native RB feasibility."""

    def __init__(self, actor_critic: BeliefActorCritic, device: torch.device):
        self.actor_critic = actor_critic.eval()
        self.device = device

    @torch.no_grad()
    def act(self, env: Any, history: np.ndarray, deterministic: bool = True) -> int:
        x = torch.as_tensor(history[None, ...], dtype=torch.float32, device=self.device)
        logits, _ = self.actor_critic(x)
        native = torch.as_tensor(
            env.resource_feasible_action_mask(), dtype=torch.bool, device=self.device
        )
        masked_logits = logits[0].masked_fill(~native, -1e9)
        if deterministic:
            return int(torch.argmax(masked_logits).item())
        return int(Categorical(logits=masked_logits).sample().item())


class RecurrentConstrainedActorCritic(nn.Module):
    """GRU actor with matched reward and reliability-cost critics."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: PPOConfig):
        super().__init__()
        self.encoder = nn.GRU(obs_dim, cfg.gru_hidden_dim, batch_first=True)
        self.body = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.hidden_dim), nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(),
        )
        self.actor = nn.Linear(cfg.hidden_dim, n_actions)
        self.reward_value = nn.Linear(cfg.hidden_dim, 1)
        self.cost_value = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, history: torch.Tensor):
        _, h = self.encoder(history)
        z = self.body(h[-1])
        return (
            self.actor(z),
            self.reward_value(z).squeeze(-1),
            self.cost_value(z).squeeze(-1),
        )


class RecurrentLagrangianPolicy:
    def __init__(self, model: RecurrentConstrainedActorCritic, device: torch.device):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def act(self, env: Any, history: np.ndarray, deterministic: bool = True) -> int:
        x = torch.as_tensor(history[None, ...], dtype=torch.float32, device=self.device)
        logits, _, _ = self.model(x)
        native = torch.as_tensor(
            env.resource_feasible_action_mask(), dtype=torch.bool, device=self.device
        )
        masked_logits = logits[0].masked_fill(~native, -1e9)
        if deterministic:
            return int(torch.argmax(masked_logits).item())
        return int(Categorical(logits=masked_logits).sample().item())


@dataclass(frozen=True)
class LagrangianConfig:
    cost_limit: float = 0.05
    dual_learning_rate: float = 0.10
    lambda_init: float = 0.0
    lambda_max: float = 5.0
    cost_value_coef: float = 0.5


def _native_mask_tensor(env: Any, device: torch.device) -> torch.Tensor:
    mask = np.asarray(env.resource_feasible_action_mask(), dtype=bool)
    if mask.ndim != 1 or mask.size != int(env.n_actions):
        raise RuntimeError("Invalid native resource-feasibility mask.")
    if not bool(mask.any()):
        raise RuntimeError("Native resource-feasibility mask is empty.")
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def train_recurrent_ppo_continuing(
    env: Any,
    ppo_cfg: PPOConfig,
    rollout_updates: int,
    rollout_steps: int,
    seed: int,
) -> dict:
    """Matched recurrent PPO under the same continuing-task protocol as ForeSafe.

    The actor architecture, PPO hyperparameters, partial-observation history,
    rollout length, training budget, and native resource mask match ForeSafe.
    The predictive risk model, CVaR shield, shift monitor, urgency guard, and
    predictive risk penalty are absent.
    """
    set_seed(seed)
    device = torch.device(ppo_cfg.device)

    history, _ = env.reset(seed=seed)
    obs_dim = int(history.shape[-1])
    n_actions = int(env.n_actions)

    ac = BeliefActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=ppo_cfg.learning_rate)

    hist = {
        "rollout_reward": [],
        "rollout_interruption_probability": [],
        "bootstrap_value": [],
        "mean_native_feasible_action_count": [],
        "rollout_metrics": [],
    }

    total_steps = 0
    done = False

    for update_idx in range(int(rollout_updates)):
        states, actions, old_logps, rewards, values, dones, masks = (
            [], [], [], [], [], [], []
        )
        scheduled_counts, failed_counts = [], []
        raw_rollout_reward = 0.0

        for _ in range(int(rollout_steps)):
            x = torch.as_tensor(
                history[None, ...], dtype=torch.float32, device=device
            )
            native = _native_mask_tensor(env, device)
            with torch.no_grad():
                logits, value = ac(x)
                masked_logits = logits[0].masked_fill(~native, -1e9)
                dist = Categorical(logits=masked_logits)
                action = int(dist.sample().item())
                logp = float(
                    dist.log_prob(torch.tensor(action, device=device)).item()
                )

            history_before = np.asarray(history, dtype=np.float32).copy()
            next_history, reward, done, info = env.step(action)
            info = dict(info or {})

            states.append(history_before)
            actions.append(action)
            old_logps.append(logp)
            rewards.append(float(reward))
            values.append(float(value.item()))
            dones.append(float(done))
            masks.append(native.detach().cpu().numpy().astype(bool))
            scheduled_counts.append(float(info.get("scheduled", 0) or 0))
            failed_counts.append(float(info.get("failed_scheduled", 0) or 0))

            history = next_history
            raw_rollout_reward += float(reward)
            total_steps += 1
            if done:
                break

        if not states:
            continue

        bootstrap_value = 0.0
        if not done:
            with torch.no_grad():
                x_next = torch.as_tensor(
                    history[None, ...], dtype=torch.float32, device=device
                )
                _, next_value = ac(x_next)
                bootstrap_value = float(next_value.item())

        returns, adv = _gae_bootstrap(
            rewards,
            values,
            dones,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
            bootstrap_value,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_logps_t = torch.as_tensor(old_logps, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        masks_t = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)

        n = len(states)
        batch_size = min(max(8, int(ppo_cfg.minibatch_size)), n)
        for _ in range(int(ppo_cfg.train_epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                logits, value_pred = ac(states_t[idx])
                logits = logits.masked_fill(~masks_t[idx], -1e9)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(actions_t[idx])
                ratio = torch.exp(lp - old_logps_t[idx])
                obj1 = ratio * adv_t[idx]
                obj2 = torch.clamp(
                    ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio
                ) * adv_t[idx]
                policy_loss = -torch.min(obj1, obj2).mean()
                value_loss = torch.mean((value_pred - returns_t[idx]) ** 2)
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

        scheduled_sum = float(np.sum(scheduled_counts))
        failed_sum = float(np.sum(failed_counts))
        rollout_intr = (
            failed_sum / scheduled_sum if scheduled_sum > 0.0 else 0.0
        )
        metrics = dict(env.episode_metrics())
        metrics.update({
            "continuing_rollout_index": int(update_idx),
            "continuing_rollout_steps": int(len(states)),
            "rollout_reward": float(raw_rollout_reward),
            "rollout_scheduled_tx": float(scheduled_sum),
            "rollout_failed_scheduled_tx": float(failed_sum),
            "rollout_interruption_probability": float(rollout_intr),
            "bootstrap_value": float(bootstrap_value),
        })

        hist["rollout_reward"].append(float(raw_rollout_reward))
        hist["rollout_interruption_probability"].append(float(rollout_intr))
        hist["bootstrap_value"].append(float(bootstrap_value))
        hist["mean_native_feasible_action_count"].append(
            float(np.sum(env.resource_feasible_action_mask()))
        )
        hist["rollout_metrics"].append(metrics)

        if done and update_idx + 1 < int(rollout_updates):
            raise RuntimeError(
                "Continuing-task environment terminated before all requested "
                "rollouts. Increase the configured training horizon."
            )

    return {
        "policy": RecurrentPPOPolicy(ac, device),
        "actor_critic": ac,
        "history": hist,
        "total_steps": int(total_steps),
        "training_protocol": {
            "mode": "continuing_task",
            "environment_resets": 1,
            "rollout_updates": int(rollout_updates),
            "rollout_steps": int(rollout_steps),
            "total_training_steps": int(total_steps),
            "bootstrap_at_rollout_boundaries": True,
            "risk_model": False,
            "cvar_filter": False,
        },
    }


def train_recurrent_ppo_lagrangian_continuing(
    env: Any,
    ppo_cfg: PPOConfig,
    lag_cfg: LagrangianConfig,
    rollout_updates: int,
    rollout_steps: int,
    seed: int,
) -> dict:
    """Matched recurrent PPO-Lagrangian baseline.

    The reliability cost uses the same centered per-slot residual
    (F_t - d S_t)/K with d=0.05. The dual update is rollout-local and uses
    projected gradient ascent. The default dual learning rate 0.10 matches
    the previously selected PPO-Lagrangian setting in the MobiSafe study.
    """
    set_seed(seed)
    device = torch.device(ppo_cfg.device)

    history, _ = env.reset(seed=seed)
    obs_dim = int(history.shape[-1])
    n_actions = int(env.n_actions)

    model = RecurrentConstrainedActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)
    lambda_value = float(np.clip(lag_cfg.lambda_init, 0.0, lag_cfg.lambda_max))

    hist = {
        "rollout_reward": [],
        "rollout_interruption_probability": [],
        "rollout_constraint_residual": [],
        "lambda_value": [],
        "bootstrap_reward_value": [],
        "bootstrap_cost_value": [],
        "mean_native_feasible_action_count": [],
        "rollout_metrics": [],
    }

    total_steps = 0
    done = False

    for update_idx in range(int(rollout_updates)):
        states, actions, old_logps = [], [], []
        rewards, costs = [], []
        reward_values, cost_values, dones, masks = [], [], [], []
        scheduled_counts, failed_counts = [], []
        raw_rollout_reward = 0.0

        for _ in range(int(rollout_steps)):
            x = torch.as_tensor(
                history[None, ...], dtype=torch.float32, device=device
            )
            native = _native_mask_tensor(env, device)
            with torch.no_grad():
                logits, reward_value, cost_value = model(x)
                masked_logits = logits[0].masked_fill(~native, -1e9)
                dist = Categorical(logits=masked_logits)
                action = int(dist.sample().item())
                logp = float(
                    dist.log_prob(torch.tensor(action, device=device)).item()
                )

            history_before = np.asarray(history, dtype=np.float32).copy()
            next_history, reward, done, info = env.step(action)
            info = dict(info or {})
            cost = constraint_residual_from_info(
                info,
                cost_limit=float(lag_cfg.cost_limit),
                n_channels=int(env.n_channels),
            )

            states.append(history_before)
            actions.append(action)
            old_logps.append(logp)
            rewards.append(float(reward))
            costs.append(float(cost))
            reward_values.append(float(reward_value.item()))
            cost_values.append(float(cost_value.item()))
            dones.append(float(done))
            masks.append(native.detach().cpu().numpy().astype(bool))
            scheduled_counts.append(float(info.get("scheduled", 0) or 0))
            failed_counts.append(float(info.get("failed_scheduled", 0) or 0))

            history = next_history
            raw_rollout_reward += float(reward)
            total_steps += 1
            if done:
                break

        if not states:
            continue

        bootstrap_reward = 0.0
        bootstrap_cost = 0.0
        if not done:
            with torch.no_grad():
                x_next = torch.as_tensor(
                    history[None, ...], dtype=torch.float32, device=device
                )
                _, next_reward_value, next_cost_value = model(x_next)
                bootstrap_reward = float(next_reward_value.item())
                bootstrap_cost = float(next_cost_value.item())

        reward_returns, reward_adv = _gae_bootstrap(
            rewards,
            reward_values,
            dones,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
            bootstrap_reward,
        )
        cost_returns, cost_adv = _gae_bootstrap(
            costs,
            cost_values,
            dones,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
            bootstrap_cost,
        )

        safe_adv = reward_adv - float(lambda_value) * cost_adv
        safe_adv = (safe_adv - safe_adv.mean()) / (safe_adv.std() + 1e-8)

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_logps_t = torch.as_tensor(old_logps, dtype=torch.float32, device=device)
        reward_returns_t = torch.as_tensor(
            reward_returns, dtype=torch.float32, device=device
        )
        cost_returns_t = torch.as_tensor(
            cost_returns, dtype=torch.float32, device=device
        )
        adv_t = torch.as_tensor(safe_adv, dtype=torch.float32, device=device)
        masks_t = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)

        n = len(states)
        batch_size = min(max(8, int(ppo_cfg.minibatch_size)), n)
        for _ in range(int(ppo_cfg.train_epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                logits, reward_pred, cost_pred = model(states_t[idx])
                logits = logits.masked_fill(~masks_t[idx], -1e9)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(actions_t[idx])
                ratio = torch.exp(lp - old_logps_t[idx])
                obj1 = ratio * adv_t[idx]
                obj2 = torch.clamp(
                    ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio
                ) * adv_t[idx]
                policy_loss = -torch.min(obj1, obj2).mean()
                reward_value_loss = torch.mean(
                    (reward_pred - reward_returns_t[idx]) ** 2
                )
                cost_value_loss = torch.mean(
                    (cost_pred - cost_returns_t[idx]) ** 2
                )
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * reward_value_loss
                    + float(lag_cfg.cost_value_coef) * cost_value_loss
                    - ppo_cfg.entropy_coef * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ppo_cfg.max_grad_norm
                )
                optimizer.step()

        rollout_cost = float(np.mean(costs)) if costs else 0.0
        lambda_value = float(np.clip(
            lambda_value
            + float(lag_cfg.dual_learning_rate) * rollout_cost,
            0.0,
            float(lag_cfg.lambda_max),
        ))

        scheduled_sum = float(np.sum(scheduled_counts))
        failed_sum = float(np.sum(failed_counts))
        rollout_intr = (
            failed_sum / scheduled_sum if scheduled_sum > 0.0 else 0.0
        )
        metrics = dict(env.episode_metrics())
        metrics.update({
            "continuing_rollout_index": int(update_idx),
            "continuing_rollout_steps": int(len(states)),
            "rollout_reward": float(raw_rollout_reward),
            "rollout_constraint_residual": float(rollout_cost),
            "lambda_value": float(lambda_value),
            "rollout_scheduled_tx": float(scheduled_sum),
            "rollout_failed_scheduled_tx": float(failed_sum),
            "rollout_interruption_probability": float(rollout_intr),
            "bootstrap_reward_value": float(bootstrap_reward),
            "bootstrap_cost_value": float(bootstrap_cost),
        })

        hist["rollout_reward"].append(float(raw_rollout_reward))
        hist["rollout_interruption_probability"].append(float(rollout_intr))
        hist["rollout_constraint_residual"].append(float(rollout_cost))
        hist["lambda_value"].append(float(lambda_value))
        hist["bootstrap_reward_value"].append(float(bootstrap_reward))
        hist["bootstrap_cost_value"].append(float(bootstrap_cost))
        hist["mean_native_feasible_action_count"].append(
            float(np.sum(env.resource_feasible_action_mask()))
        )
        hist["rollout_metrics"].append(metrics)

        if done and update_idx + 1 < int(rollout_updates):
            raise RuntimeError(
                "Continuing-task environment terminated before all requested "
                "rollouts. Increase the configured training horizon."
            )

    return {
        "policy": RecurrentLagrangianPolicy(model, device),
        "actor_critic": model,
        "history": hist,
        "total_steps": int(total_steps),
        "lambda_value": float(lambda_value),
        "training_protocol": {
            "mode": "continuing_task",
            "environment_resets": 1,
            "rollout_updates": int(rollout_updates),
            "rollout_steps": int(rollout_steps),
            "total_training_steps": int(total_steps),
            "bootstrap_at_rollout_boundaries": True,
            "cost_limit": float(lag_cfg.cost_limit),
            "dual_learning_rate": float(lag_cfg.dual_learning_rate),
            "lambda_max": float(lag_cfg.lambda_max),
        },
    }
