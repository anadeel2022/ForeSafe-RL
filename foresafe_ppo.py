from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from foresafe_config import PPOConfig, RiskConfig, SafetyConfig
from foresafe_risk import (
    QuantileRiskPredictor, RiskReplay, RiskSample, ShiftMonitor, HorizonRiskLabeler,
    finite_horizon_failure_ratio_targets, train_risk_predictor, upper_cvar_from_quantiles,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def failure_cost(info: dict) -> float:
    scheduled = float(info.get("scheduled", 0) or 0)
    failed = float(info.get("failed_scheduled", 0) or 0)
    return float(failed / scheduled) if scheduled > 0 else 0.0

def _gae(rewards, values, dones, gamma, lam):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    adv = np.zeros_like(rewards)
    last = 0.0
    ext = np.concatenate([values, np.asarray([0.0], dtype=np.float32)])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * ext[t + 1] * (1.0 - dones[t]) - ext[t]
        last = delta + gamma * lam * (1.0 - dones[t]) * last
        adv[t] = last
    return adv + values, adv


def _gae_bootstrap(rewards, values, dones, gamma, lam, bootstrap_value):
    """GAE for a nonterminal rollout cut with an explicit critic bootstrap."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    adv = np.zeros_like(rewards)
    last = 0.0
    ext = np.concatenate([
        values,
        np.asarray([float(bootstrap_value)], dtype=np.float32),
    ])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * ext[t + 1] * (1.0 - dones[t]) - ext[t]
        last = delta + gamma * lam * (1.0 - dones[t]) * last
        adv[t] = last
    return adv + values, adv

class BeliefActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, cfg: PPOConfig):
        super().__init__()
        self.encoder = nn.GRU(obs_dim, cfg.gru_hidden_dim, batch_first=True)
        self.body = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.hidden_dim), nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(),
        )
        self.actor = nn.Linear(cfg.hidden_dim, n_actions)
        self.value = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, history: torch.Tensor):
        _, h = self.encoder(history)
        z = self.body(h[-1])
        return self.actor(z), self.value(z).squeeze(-1)

@dataclass
class ActionDiagnostics:
    native_feasible_action_count: int
    supported_native_action_count: int
    strict_supported_cvar_pass_count: int
    cvar_admissible_action_count: int
    safe_action_count: int
    chosen_action_supported: bool
    chosen_median_risk: float
    chosen_cvar: float
    min_feasible_cvar: float
    effective_limit: float
    shield_ready: bool
    forced_min_risk: bool
    risk_rejected: int
    risk_relaxed: int
    urgency_guarded: int
    urgency_score: float


def _combined_mask(
    env,
    cvar: torch.Tensor,
    limit: float,
    risk_active: bool,
    safety_cfg: SafetyConfig,
    history: np.ndarray,
    device: torch.device,
    support_counts: np.ndarray | None = None,
    return_details: bool = False,
):
    """Combine physical feasibility, evidence-aware CVaR filtering, and service protection.

    Learned counterfactual CVaR estimates are allowed to hard-reject an action
    only after that action has accumulated enough realized training support.
    A minimum operational action-set fraction prevents the learned shield from
    collapsing PPO support to a single fallback action.
    """
    native_np = env.resource_feasible_action_mask()
    native = torch.as_tensor(native_np, dtype=torch.bool, device=device)
    native_count = int(native.sum().item())

    if support_counts is None:
        support_np = np.full(int(env.n_actions), 10**9, dtype=np.int64)
    else:
        support_np = np.asarray(support_counts, dtype=np.int64).reshape(-1)
        if support_np.size != int(env.n_actions):
            raise ValueError("support_counts must have one entry per action.")

    supported_np = support_np >= int(safety_cfg.min_action_samples_for_filter)
    supported = torch.as_tensor(supported_np, dtype=torch.bool, device=device)
    supported_native_count = int((native & supported).sum().item())

    mask = native.clone()
    strict_supported_pass = supported_native_count
    risk_rejected = 0

    if bool(risk_active) and bool(safety_cfg.enable_cvar_filter):
        reject = native & supported & (cvar > float(limit))
        mask &= ~reject
        risk_rejected = int(reject.sum().item())
        strict_supported_pass = int(
            (native & supported & (cvar <= float(limit))).sum().item()
        )

    # Preserve a nondegenerate operational support set. Re-admission is always
    # risk ranked and never violates the native resource-feasibility mask.
    floor_fraction = float(np.clip(
        safety_cfg.min_operational_action_fraction, 0.0, 1.0
    ))
    operational_floor = max(
        int(safety_cfg.min_safe_actions),
        int(np.ceil(floor_fraction * native_count)),
    )
    risk_relaxed = 0
    if bool(risk_active) and int(mask.sum().item()) < operational_floor:
        feasible_idx = torch.where(native)[0]
        order = feasible_idx[torch.argsort(cvar[feasible_idx])]
        need = operational_floor - int(mask.sum().item())
        for idx in order.tolist():
            if need <= 0:
                break
            if not bool(mask[int(idx)]):
                mask[int(idx)] = True
                risk_relaxed += 1
                need -= 1

    cvar_count = int(mask.sum().item())

    urgency_score = float(env.urgency_from_history(history))
    urgency_guarded = 0
    if urgency_score >= float(safety_cfg.urgency_guard_threshold):
        for a in range(int(env.n_actions)):
            if bool(mask[a]) and env.decoded_defer_fraction(a) > float(
                safety_cfg.max_defer_fraction_when_urgent
            ):
                mask[a] = False
                urgency_guarded += 1

    forced = False
    if int(mask.sum().item()) < int(safety_cfg.min_safe_actions):
        feasible_idx = torch.where(native)[0]
        if feasible_idx.numel() == 0:
            raise RuntimeError("The native resource mask contains no feasible action.")

        # Under deadline pressure, prefer a service-capable fallback.
        candidates = []
        if urgency_score >= float(safety_cfg.urgency_guard_threshold):
            candidates = [
                int(a) for a in feasible_idx.tolist()
                if env.decoded_defer_fraction(int(a))
                <= float(safety_cfg.max_defer_fraction_when_urgent)
            ]
        if not candidates:
            candidates = [int(a) for a in feasible_idx.tolist()]

        candidate_t = torch.as_tensor(candidates, dtype=torch.long, device=device)
        best = int(candidate_t[int(torch.argmin(cvar[candidate_t]).item())].item())
        mask[:] = False
        mask[best] = True
        forced = True

    if return_details:
        return (
            mask,
            native,
            forced,
            cvar_count,
            int(native_count - cvar_count),
            urgency_guarded,
            supported_native_count,
            strict_supported_pass,
            risk_rejected,
            risk_relaxed,
            urgency_score,
        )
    return (
        mask,
        native,
        forced,
        cvar_count,
        int(native_count - cvar_count),
        urgency_guarded,
    )


class ForeSafePolicy:
    def __init__(
        self,
        actor_critic,
        risk_model,
        risk_cfg,
        safety_cfg,
        device,
        action_support_counts: np.ndarray | None = None,
    ):
        self.actor_critic = actor_critic.eval()
        self.risk_model = risk_model.eval()
        self.risk_cfg = risk_cfg
        self.safety_cfg = safety_cfg
        self.device = device
        self.shift_monitor = ShiftMonitor(risk_cfg)
        self.monitor_initial_error = float(risk_cfg.shift_reference_error)
        self.monitor_initial_initialized = False
        self.action_support_counts = (
            None
            if action_support_counts is None
            else np.asarray(action_support_counts, dtype=np.int64).copy()
        )
        self._median_pred_queue: list[float] = []
        self._failed_queue: list[float] = []
        self._scheduled_queue: list[float] = []
        self._support_queue: list[bool] = []

    def set_monitor_initial_state(self, error_ema: float, initialized: bool):
        self.monitor_initial_error = float(error_ema)
        self.monitor_initial_initialized = bool(initialized)
        self.reset_online_monitor()

    def set_action_support_counts(self, counts: np.ndarray | None):
        self.action_support_counts = (
            None if counts is None else np.asarray(counts, dtype=np.int64).copy()
        )

    def reset_online_monitor(self):
        self.shift_monitor.reset(
            self.monitor_initial_error, self.monitor_initial_initialized
        )
        self._median_pred_queue = []
        self._failed_queue = []
        self._scheduled_queue = []
        self._support_queue = []

    def observe_realized_outcome(
        self,
        predicted_median: float,
        failed_scheduled: float,
        scheduled: float,
        prediction_supported: bool = True,
    ):
        """Causal shift update using supported, horizon-aligned predictions."""
        self._median_pred_queue.append(float(predicted_median))
        self._failed_queue.append(float(failed_scheduled))
        self._scheduled_queue.append(float(scheduled))
        self._support_queue.append(bool(prediction_supported))
        h = max(1, int(self.risk_cfg.horizon))
        if len(self._scheduled_queue) >= h:
            den = float(np.sum(self._scheduled_queue[-h:]))
            num = float(np.sum(self._failed_queue[-h:]))
            target = num / den if den > 0.0 else 0.0
            pred = float(self._median_pred_queue[-h])
            supported = bool(self._support_queue[-h])
            if supported:
                self.shift_monitor.update(pred, target)

    @torch.no_grad()
    def act(self, env, history: np.ndarray, deterministic: bool = True):
        h = torch.as_tensor(
            history[None, ...], dtype=torch.float32, device=self.device
        )
        logits, _ = self.actor_critic(h)
        all_q = self.risk_model.all_action_quantiles(h)
        cvar = upper_cvar_from_quantiles(
            all_q,
            self.risk_cfg.quantiles,
            self.risk_cfg.cvar_alpha,
        )
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
            cvar,
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
        feasible_cvar = cvar[native]

        chosen_supported = True
        if self.action_support_counts is not None:
            chosen_supported = bool(
                int(self.action_support_counts[action])
                >= int(self.safety_cfg.min_action_samples_for_filter)
            )

        diag = ActionDiagnostics(
            native_feasible_action_count=int(native.sum().item()),
            supported_native_action_count=int(supported_native_count),
            strict_supported_cvar_pass_count=int(strict_supported_pass),
            cvar_admissible_action_count=int(cvar_count),
            safe_action_count=int(mask.sum().item()),
            chosen_action_supported=bool(chosen_supported),
            chosen_median_risk=float(all_q[action, median_idx].item()),
            chosen_cvar=float(cvar[action].item()),
            min_feasible_cvar=float(feasible_cvar.min().item()),
            effective_limit=float(limit),
            shield_ready=True,
            forced_min_risk=bool(forced),
            risk_rejected=int(risk_rejected),
            risk_relaxed=int(risk_relaxed),
            urgency_guarded=int(urgency_guarded),
            urgency_score=float(urgency_score),
        )
        return action, diag


def train_foresafe(env: Any, ppo_cfg: PPOConfig, risk_cfg: RiskConfig, safety_cfg: SafetyConfig,
                   episodes: int, seed: int) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(int(seed))
    device = torch.device(ppo_cfg.device)

    reset_out = env.reset(seed=seed)
    history = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    obs_dim = int(history.shape[-1])
    n_actions = int(env.n_actions)

    ac = BeliefActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
    risk_model = QuantileRiskPredictor(
        obs_dim, n_actions, risk_cfg.predictor_hidden_dim, risk_cfg.quantiles
    ).to(device)
    ac_opt = optim.Adam(ac.parameters(), lr=ppo_cfg.learning_rate)
    risk_opt = optim.Adam(risk_model.parameters(), lr=risk_cfg.predictor_lr)
    replay = RiskReplay(risk_cfg.replay_capacity)
    shift = ShiftMonitor(risk_cfg)

    total_steps = 0
    hist_out = {
        "episode_reward": [], "episode_failure_cost": [], "risk_predictor_loss": [],
        "mean_chosen_median_risk": [], "mean_chosen_cvar": [],
        "mean_native_feasible_action_count": [],
        "mean_cvar_admissible_action_count": [], "mean_safe_action_count": [],
        "mean_cvar_removed_action_count": [], "mean_urgency_removed_action_count": [],
        "forced_min_risk_rate": [],
        "effective_cvar_limit": [], "shift_error_ema": [], "episode_metrics": [],
    }

    for ep in range(int(episodes)):
        history, _ = env.reset(seed=seed * 100000 + ep + 1)
        states, actions, logps, rewards, values, dones = [], [], [], [], [], []
        rollout_masks = []
        costs, predicted_medians, predicted_cvars = [], [], []
        failed_counts, scheduled_counts = [], []
        native_counts, cvar_counts, safe_counts, cvar_removed_counts = [], [], [], []
        urgency_removed_counts, forced_flags, limits = [], [], []
        ep_reward = 0.0

        for _ in range(int(env.steps_per_ep)):
            ht = torch.as_tensor(history[None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = ac(ht)
                all_q = risk_model.all_action_quantiles(ht)
                cvar = upper_cvar_from_quantiles(all_q, risk_cfg.quantiles, risk_cfg.cvar_alpha)
                median_idx = min(
                    range(len(risk_cfg.quantiles)),
                    key=lambda i: abs(float(risk_cfg.quantiles[i]) - 0.50),
                )

            limit = shift.effective_limit()
            risk_active = total_steps >= int(risk_cfg.warmup_steps)
            mask, native, forced, cvar_count, cvar_removed, urgency_removed = _combined_mask(
                env, cvar, limit, risk_active, safety_cfg, history, device
            )

            masked_logits = logits[0].masked_fill(~mask, -1e9)
            dist = Categorical(logits=masked_logits)
            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action, device=device)).item())
            pred_median = float(all_q[action, median_idx].item())
            pred_cvar = float(cvar[action].item())

            next_history, reward, done, info = env.step(action)
            cost = failure_cost(info)
            scheduled_now = float(info.get("scheduled", 0) or 0)
            failed_now = float(info.get("failed_scheduled", 0) or 0)

            states.append(np.asarray(history, dtype=np.float32).copy())
            actions.append(action)
            logps.append(logp)
            predictive_penalty = (
                float(ppo_cfg.risk_penalty_coef) * max(0.0, pred_cvar - limit)
                if risk_active else 0.0
            )
            rewards.append(float(reward) - predictive_penalty)
            values.append(float(value.item()))
            dones.append(float(done))
            rollout_masks.append(mask.detach().cpu().numpy().astype(bool))
            costs.append(cost)
            predicted_medians.append(pred_median)
            predicted_cvars.append(pred_cvar)
            failed_counts.append(failed_now)
            scheduled_counts.append(scheduled_now)
            native_counts.append(int(native.sum().item()))
            cvar_counts.append(int(cvar_count))
            safe_counts.append(int(mask.sum().item()))
            cvar_removed_counts.append(int(cvar_removed))
            urgency_removed_counts.append(int(urgency_removed))
            forced_flags.append(int(forced))
            limits.append(limit)

            # Causal, horizon-aligned shift update. Prediction at t-H+1 is
            # compared with the denominator-correct realized H-slot
            # interruption ratio once that label becomes available. The q50
            # predictor drives residual monitoring; CVaR is reserved for safety.
            h = max(1, int(risk_cfg.horizon))
            if len(scheduled_counts) >= h:
                den = float(np.sum(scheduled_counts[-h:]))
                num = float(np.sum(failed_counts[-h:]))
                target_now = num / den if den > 0.0 else 0.0
                pred_old = float(predicted_medians[-h])
                shift.update(pred_old, target_now)

            history = next_history
            ep_reward += float(reward)
            total_steps += 1
            if done:
                break

        if not states:
            continue

        risk_targets = finite_horizon_failure_ratio_targets(
            failed_counts, scheduled_counts, risk_cfg.horizon
        )
        replay.add_many(
            RiskSample(history=s, action=a, target=float(t))
            for s, a, t in zip(states, actions, risk_targets)
        )
        risk_loss = train_risk_predictor(risk_model, replay, risk_cfg, risk_opt, device, rng)

        returns, adv = _gae(rewards, values, dones, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        s_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        a_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_lp_t = torch.as_tensor(logps, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(np.stack(rollout_masks), dtype=torch.bool, device=device)

        n = len(states)
        for _ in range(int(ppo_cfg.train_epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, int(ppo_cfg.minibatch_size)):
                idx = perm[start:start + int(ppo_cfg.minibatch_size)]
                logits, v = ac(s_t[idx])
                logits = logits.masked_fill(~mask_t[idx], -1e9)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(a_t[idx])
                ratio = torch.exp(lp - old_lp_t[idx])
                obj1 = ratio * adv_t[idx]
                obj2 = torch.clamp(
                    ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio
                ) * adv_t[idx]
                p_loss = -torch.min(obj1, obj2).mean()
                v_loss = torch.mean((v - ret_t[idx]) ** 2)
                ent = dist.entropy().mean()
                loss = p_loss + ppo_cfg.value_coef * v_loss - ppo_cfg.entropy_coef * ent
                ac_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), ppo_cfg.max_grad_norm)
                ac_opt.step()

        metrics = env.episode_metrics() if hasattr(env, "episode_metrics") else {}
        hist_out["episode_reward"].append(float(ep_reward))
        hist_out["episode_failure_cost"].append(float(np.mean(costs)))
        hist_out["risk_predictor_loss"].append(float(risk_loss))
        hist_out["mean_chosen_median_risk"].append(float(np.mean(predicted_medians)))
        hist_out["mean_chosen_cvar"].append(float(np.mean(predicted_cvars)))
        hist_out["mean_native_feasible_action_count"].append(float(np.mean(native_counts)))
        hist_out["mean_cvar_admissible_action_count"].append(float(np.mean(cvar_counts)))
        hist_out["mean_safe_action_count"].append(float(np.mean(safe_counts)))
        hist_out["mean_cvar_removed_action_count"].append(float(np.mean(cvar_removed_counts)))
        hist_out["mean_urgency_removed_action_count"].append(float(np.mean(urgency_removed_counts)))
        hist_out["forced_min_risk_rate"].append(float(np.mean(forced_flags)))
        hist_out["effective_cvar_limit"].append(float(np.mean(limits)))
        hist_out["shift_error_ema"].append(float(shift.error_ema))
        hist_out["episode_metrics"].append(dict(metrics))

    policy = ForeSafePolicy(ac, risk_model, risk_cfg, safety_cfg, device)
    policy.set_monitor_initial_state(shift.error_ema, shift.initialized)
    return {
        "policy": policy,
        "actor_critic": ac,
        "risk_model": risk_model,
        "history": hist_out,
        "replay_size": len(replay),
        "total_steps": total_steps,
    }



def train_foresafe_continuing(
    env: Any,
    ppo_cfg: PPOConfig,
    risk_cfg: RiskConfig,
    safety_cfg: SafetyConfig,
    rollout_updates: int,
    rollout_steps: int,
    seed: int,
) -> dict:
    """Train ForeSafe as a continuing task with evidence-aware predictive shielding.

    The recurrent PPO stream is uninterrupted. The quantile-risk model learns
    throughout training, but the hard learned CVaR filter is enabled only after
    a minimum temporal warm-up *and* adequate empirical action support. Before
    that point, the published resource/chance-qualified simulator remains the
    hard safety floor and the learned risk model can contribute only a
    support-qualified soft penalty.
    """
    set_seed(seed)
    rng = np.random.default_rng(int(seed))
    device = torch.device(ppo_cfg.device)

    reset_out = env.reset(seed=seed)
    history = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    obs_dim = int(history.shape[-1])
    n_actions = int(env.n_actions)

    ac = BeliefActorCritic(obs_dim, n_actions, ppo_cfg).to(device)
    risk_model = QuantileRiskPredictor(
        obs_dim, n_actions, risk_cfg.predictor_hidden_dim, risk_cfg.quantiles
    ).to(device)
    ac_opt = optim.Adam(ac.parameters(), lr=ppo_cfg.learning_rate)
    risk_opt = optim.Adam(risk_model.parameters(), lr=risk_cfg.predictor_lr)
    replay = RiskReplay(risk_cfg.replay_capacity)
    labeler = HorizonRiskLabeler(risk_cfg.horizon)
    shift = ShiftMonitor(risk_cfg)

    total_steps = 0
    median_idx = min(
        range(len(risk_cfg.quantiles)),
        key=lambda i: abs(float(risk_cfg.quantiles[i]) - 0.50),
    )

    hist_out = {
        "rollout_reward": [],
        "rollout_interruption_probability": [],
        "risk_predictor_loss": [],
        "mean_chosen_median_risk": [],
        "mean_chosen_cvar": [],
        "mean_native_feasible_action_count": [],
        "mean_supported_native_action_count": [],
        "mean_strict_supported_cvar_pass_count": [],
        "mean_cvar_admissible_action_count": [],
        "mean_safe_action_count": [],
        "mean_risk_rejected_action_count": [],
        "mean_risk_relaxed_action_count": [],
        "mean_urgency_removed_action_count": [],
        "mean_urgency_score": [],
        "forced_min_risk_rate": [],
        "shield_activation_rate": [],
        "supported_action_fraction": [],
        "effective_cvar_limit": [],
        "shift_error_ema": [],
        "bootstrap_value": [],
        "risk_replay_size": [],
        "pending_unlabeled_risk_windows": [],
        "rollout_metrics": [],
    }

    done = False
    for update_idx in range(int(rollout_updates)):
        states, actions, logps, rewards, values, dones = [], [], [], [], [], []
        rollout_masks = []
        predicted_medians, predicted_cvars = [], []
        failed_counts, scheduled_counts = [], []
        native_counts, supported_counts_roll = [], []
        strict_pass_counts, cvar_counts, safe_counts = [], [], []
        risk_rejected_counts, risk_relaxed_counts = [], []
        urgency_removed_counts, urgency_scores = [], []
        forced_flags, shield_flags, limits = [], [], []
        raw_rollout_reward = 0.0

        # Support is evaluated from already matured, realized labels. It is
        # frozen within each PPO rollout to avoid changing the action mask for
        # identical data merely because one more transition has matured.
        action_support_counts = replay.action_counts(n_actions)
        native_np_for_support = env.resource_feasible_action_mask()
        native_count_for_support = max(int(np.sum(native_np_for_support)), 1)
        supported_native = int(np.sum(
            native_np_for_support
            & (
                action_support_counts
                >= int(safety_cfg.min_action_samples_for_filter)
            )
        ))
        supported_fraction = float(
            supported_native / native_count_for_support
        )
        shield_ready_rollout = bool(
            total_steps >= int(risk_cfg.shield_warmup_steps)
            and supported_fraction >= float(
                risk_cfg.min_supported_action_fraction
            )
        )

        for _ in range(int(rollout_steps)):
            ht = torch.as_tensor(
                history[None, ...], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                logits, value = ac(ht)
                all_q = risk_model.all_action_quantiles(ht)
                cvar = upper_cvar_from_quantiles(
                    all_q, risk_cfg.quantiles, risk_cfg.cvar_alpha
                )

            limit = shift.effective_limit()
            (
                mask,
                native,
                forced,
                cvar_count,
                _legacy_removed,
                urgency_removed,
                supported_native_count,
                strict_supported_pass,
                risk_rejected,
                risk_relaxed,
                urgency_score,
            ) = _combined_mask(
                env,
                cvar,
                limit,
                shield_ready_rollout,
                safety_cfg,
                history,
                device,
                support_counts=action_support_counts,
                return_details=True,
            )

            masked_logits = logits[0].masked_fill(~mask, -1e9)
            dist = Categorical(logits=masked_logits)
            action = int(dist.sample().item())
            logp = float(
                dist.log_prob(torch.tensor(action, device=device)).item()
            )
            pred_median = float(all_q[action, median_idx].item())
            pred_cvar = float(cvar[action].item())

            chosen_supported = bool(
                int(action_support_counts[action])
                >= int(safety_cfg.min_action_samples_for_filter)
            )
            predictor_ready = bool(
                total_steps >= int(risk_cfg.warmup_steps)
                and len(replay) >= int(risk_cfg.min_replay_size)
            )

            history_before = np.asarray(history, dtype=np.float32).copy()
            next_history, env_reward, done, info = env.step(action)
            info = dict(info or {})
            scheduled_now = float(info.get("scheduled", 0) or 0)
            failed_now = float(info.get("failed_scheduled", 0) or 0)

            states.append(history_before)
            actions.append(action)
            logps.append(logp)

            predictive_penalty = (
                float(ppo_cfg.risk_penalty_coef)
                * max(0.0, pred_cvar - limit)
                if predictor_ready and chosen_supported
                else 0.0
            )
            rewards.append(float(env_reward) - predictive_penalty)
            values.append(float(value.item()))
            dones.append(float(done))
            rollout_masks.append(
                mask.detach().cpu().numpy().astype(bool)
            )

            predicted_medians.append(pred_median)
            predicted_cvars.append(pred_cvar)
            failed_counts.append(failed_now)
            scheduled_counts.append(scheduled_now)
            native_counts.append(int(native.sum().item()))
            supported_counts_roll.append(int(supported_native_count))
            strict_pass_counts.append(int(strict_supported_pass))
            cvar_counts.append(int(cvar_count))
            safe_counts.append(int(mask.sum().item()))
            risk_rejected_counts.append(int(risk_rejected))
            risk_relaxed_counts.append(int(risk_relaxed))
            urgency_removed_counts.append(int(urgency_removed))
            urgency_scores.append(float(urgency_score))
            forced_flags.append(int(forced))
            shield_flags.append(int(shield_ready_rollout))
            limits.append(float(limit))

            matured = labeler.push(
                history_before,
                action,
                pred_median,
                failed_now,
                scheduled_now,
            )
            if matured is not None:
                sample, predicted_for_shift, realized_target = matured
                replay.add_many([sample])
                updated_count = int(
                    replay.action_counts(n_actions)[int(sample.action)]
                )
                if (
                    total_steps >= int(risk_cfg.warmup_steps)
                    and updated_count
                    >= int(safety_cfg.min_action_samples_for_filter)
                ):
                    shift.update(
                        predicted_for_shift, realized_target
                    )

            history = next_history
            raw_rollout_reward += float(env_reward)
            total_steps += 1
            if done:
                break

        if not states:
            continue

        bootstrap_value = 0.0
        if not done:
            with torch.no_grad():
                ht_next = torch.as_tensor(
                    history[None, ...],
                    dtype=torch.float32,
                    device=device,
                )
                _next_logits, next_value = ac(ht_next)
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

        s_t = torch.as_tensor(
            np.stack(states), dtype=torch.float32, device=device
        )
        a_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_lp_t = torch.as_tensor(
            logps, dtype=torch.float32, device=device
        )
        ret_t = torch.as_tensor(
            returns, dtype=torch.float32, device=device
        )
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(
            np.stack(rollout_masks), dtype=torch.bool, device=device
        )

        n = len(states)
        batch_size = min(max(8, int(ppo_cfg.minibatch_size)), n)
        for _ in range(int(ppo_cfg.train_epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                logits, value_pred = ac(s_t[idx])
                logits = logits.masked_fill(~mask_t[idx], -1e9)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(a_t[idx])
                ratio = torch.exp(lp - old_lp_t[idx])
                obj1 = ratio * adv_t[idx]
                obj2 = torch.clamp(
                    ratio,
                    1.0 - ppo_cfg.clip_ratio,
                    1.0 + ppo_cfg.clip_ratio,
                ) * adv_t[idx]
                policy_loss = -torch.min(obj1, obj2).mean()
                value_loss = torch.mean(
                    (value_pred - ret_t[idx]) ** 2
                )
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy
                )
                ac_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    ac.parameters(), ppo_cfg.max_grad_norm
                )
                ac_opt.step()

        risk_loss = train_risk_predictor(
            risk_model, replay, risk_cfg, risk_opt, device, rng
        )

        scheduled_sum = float(np.sum(scheduled_counts))
        failed_sum = float(np.sum(failed_counts))
        rollout_intr = (
            failed_sum / scheduled_sum
            if scheduled_sum > 0.0
            else 0.0
        )
        metrics = (
            dict(env.episode_metrics())
            if hasattr(env, "episode_metrics")
            else {}
        )
        metrics.update({
            "continuing_rollout_index": int(update_idx),
            "continuing_rollout_steps": int(len(states)),
            "rollout_reward": float(raw_rollout_reward),
            "rollout_scheduled_tx": float(scheduled_sum),
            "rollout_failed_scheduled_tx": float(failed_sum),
            "rollout_interruption_probability": float(rollout_intr),
            "bootstrap_value": float(bootstrap_value),
            "risk_replay_size": int(len(replay)),
            "pending_unlabeled_risk_windows": int(len(labeler)),
            "supported_action_fraction": float(supported_fraction),
            "shield_ready_rollout": int(shield_ready_rollout),
        })

        hist_out["rollout_reward"].append(float(raw_rollout_reward))
        hist_out["rollout_interruption_probability"].append(
            float(rollout_intr)
        )
        hist_out["risk_predictor_loss"].append(float(risk_loss))
        hist_out["mean_chosen_median_risk"].append(
            float(np.mean(predicted_medians))
        )
        hist_out["mean_chosen_cvar"].append(
            float(np.mean(predicted_cvars))
        )
        hist_out["mean_native_feasible_action_count"].append(
            float(np.mean(native_counts))
        )
        hist_out["mean_supported_native_action_count"].append(
            float(np.mean(supported_counts_roll))
        )
        hist_out["mean_strict_supported_cvar_pass_count"].append(
            float(np.mean(strict_pass_counts))
        )
        hist_out["mean_cvar_admissible_action_count"].append(
            float(np.mean(cvar_counts))
        )
        hist_out["mean_safe_action_count"].append(
            float(np.mean(safe_counts))
        )
        hist_out["mean_risk_rejected_action_count"].append(
            float(np.mean(risk_rejected_counts))
        )
        hist_out["mean_risk_relaxed_action_count"].append(
            float(np.mean(risk_relaxed_counts))
        )
        hist_out["mean_urgency_removed_action_count"].append(
            float(np.mean(urgency_removed_counts))
        )
        hist_out["mean_urgency_score"].append(
            float(np.mean(urgency_scores))
        )
        hist_out["forced_min_risk_rate"].append(
            float(np.mean(forced_flags))
        )
        hist_out["shield_activation_rate"].append(
            float(np.mean(shield_flags))
        )
        hist_out["supported_action_fraction"].append(
            float(supported_fraction)
        )
        hist_out["effective_cvar_limit"].append(
            float(np.mean(limits))
        )
        hist_out["shift_error_ema"].append(float(shift.error_ema))
        hist_out["bootstrap_value"].append(float(bootstrap_value))
        hist_out["risk_replay_size"].append(int(len(replay)))
        hist_out["pending_unlabeled_risk_windows"].append(
            int(len(labeler))
        )
        hist_out["rollout_metrics"].append(metrics)

        if done and update_idx + 1 < int(rollout_updates):
            raise RuntimeError(
                "Continuing-task environment terminated before all "
                "requested rollout updates. Increase the configured "
                "training horizon."
            )

    final_support_counts = replay.action_counts(n_actions)
    policy = ForeSafePolicy(
        ac,
        risk_model,
        risk_cfg,
        safety_cfg,
        device,
        action_support_counts=final_support_counts,
    )
    policy.set_monitor_initial_state(
        shift.error_ema, shift.initialized
    )
    return {
        "policy": policy,
        "actor_critic": ac,
        "risk_model": risk_model,
        "history": hist_out,
        "replay_size": len(replay),
        "action_support_counts": final_support_counts,
        "total_steps": total_steps,
        "pending_unlabeled_risk_windows": len(labeler),
        "training_protocol": {
            "mode": "continuing_task",
            "environment_resets": 1,
            "rollout_updates": int(rollout_updates),
            "rollout_steps": int(rollout_steps),
            "total_training_steps": int(total_steps),
            "bootstrap_at_rollout_boundaries": True,
            "risk_windows_cross_rollout_boundaries": True,
            "trailing_incomplete_risk_windows_dropped": int(
                len(labeler)
            ),
            "predictor_warmup_steps": int(risk_cfg.warmup_steps),
            "hard_shield_warmup_steps": int(
                risk_cfg.shield_warmup_steps
            ),
            "min_supported_action_fraction": float(
                risk_cfg.min_supported_action_fraction
            ),
            "min_action_samples_for_filter": int(
                safety_cfg.min_action_samples_for_filter
            ),
            "min_operational_action_fraction": float(
                safety_cfg.min_operational_action_fraction
            ),
        },
    }
