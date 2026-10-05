from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from foresafe_config import RiskConfig

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    err = target.unsqueeze(-1) - pred
    return torch.maximum(quantiles * err, (quantiles - 1.0) * err).mean()

class QuantileRiskPredictor(nn.Module):
    """Temporal action-conditional predictor of finite-horizon reliability-cost quantiles."""
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int, quantiles: tuple[float, ...]):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.encoder = nn.GRU(self.obs_dim, hidden_dim, batch_first=True)
        emb_dim = max(16, hidden_dim // 4)
        self.action_emb = nn.Embedding(self.n_actions, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(self.quantiles)),
        )

    def _ordered(self, raw: torch.Tensor) -> torch.Tensor:
        # Sorting prevents quantile crossing while retaining differentiability
        # almost everywhere.
        return torch.sort(torch.sigmoid(raw), dim=-1).values

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(history)
        z = h[-1]
        a = self.action_emb(action.long())
        return self._ordered(self.head(torch.cat([z, a], dim=-1)))

    @torch.no_grad()
    def all_action_quantiles(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError("all_action_quantiles expects history shape [1,H,D].")
        _, h = self.encoder(history)
        z = h[-1].repeat(self.n_actions, 1)
        actions = torch.arange(self.n_actions, device=history.device)
        a = self.action_emb(actions)
        return self._ordered(self.head(torch.cat([z, a], dim=-1)))

@dataclass
class RiskSample:
    history: np.ndarray
    action: int
    target: float

class RiskReplay:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: deque[RiskSample] = deque()
        self._action_counts: dict[int, int] = {}

    def __len__(self):
        return len(self.data)

    def add_many(self, samples: Iterable[RiskSample]):
        for sample in samples:
            if len(self.data) >= self.capacity:
                old = self.data.popleft()
                old_action = int(old.action)
                self._action_counts[old_action] = max(
                    0, int(self._action_counts.get(old_action, 0)) - 1
                )
            self.data.append(sample)
            action = int(sample.action)
            self._action_counts[action] = int(self._action_counts.get(action, 0)) + 1

    def action_counts(self, n_actions: int) -> np.ndarray:
        out = np.zeros(int(n_actions), dtype=np.int64)
        for action, count in self._action_counts.items():
            if 0 <= int(action) < int(n_actions):
                out[int(action)] = int(count)
        return out

    def snapshot(self) -> list[RiskSample]:
        return list(self.data)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[RiskSample]:
        n = min(int(batch_size), len(self.data))
        idx = rng.choice(len(self.data), size=n, replace=False)
        seq = list(self.data)
        return [seq[int(i)] for i in idx]


class HorizonRiskLabeler:
    """Emit exact H-slot denominator-correct labels across rollout boundaries.

    Artificial PPO rollout cuts are not treated as prediction-horizon
    boundaries. A sample is emitted only after its complete H-slot future
    window has been observed.
    """

    def __init__(self, horizon: int):
        self.horizon = max(1, int(horizon))
        self.histories: deque[np.ndarray] = deque()
        self.actions: deque[int] = deque()
        self.predicted_medians: deque[float] = deque()
        self.failed: deque[float] = deque()
        self.scheduled: deque[float] = deque()

    def __len__(self) -> int:
        return len(self.actions)

    def push(
        self,
        history: np.ndarray,
        action: int,
        predicted_median: float,
        failed_scheduled: float,
        scheduled: float,
    ):
        self.histories.append(np.asarray(history, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.predicted_medians.append(float(predicted_median))
        self.failed.append(float(failed_scheduled))
        self.scheduled.append(float(scheduled))

        if len(self.actions) < self.horizon:
            return None

        failed_h = list(self.failed)[: self.horizon]
        scheduled_h = list(self.scheduled)[: self.horizon]
        den = float(np.sum(scheduled_h))
        num = float(np.sum(failed_h))
        target = num / den if den > 0.0 else 0.0

        sample = RiskSample(
            history=self.histories[0].copy(),
            action=int(self.actions[0]),
            target=float(np.clip(target, 0.0, 1.0)),
        )
        predicted = float(self.predicted_medians[0])

        self.histories.popleft()
        self.actions.popleft()
        self.predicted_medians.popleft()
        self.failed.popleft()
        self.scheduled.popleft()
        return sample, predicted, float(sample.target)

def finite_horizon_failure_ratio_targets(
    failed_counts: list[float],
    scheduled_counts: list[float],
    horizon: int,
) -> np.ndarray:
    """Denominator-correct future interruption target.

    For every decision time t, the H-slot label is

        Z_t^(H) = sum_{j=t}^{t+H-1} F_j / sum_{j=t}^{t+H-1} S_j,

    where F_j is the number of failed scheduled transmissions and S_j is
    the number of scheduled transmissions. If the complete window contains
    no scheduled transmission, the realized interruption ratio is defined as
    zero for training-label purposes; service preservation is handled
    separately by the urgency guard and the environment reward.

    This target is aligned with the simulator's published interruption metric
    instead of averaging slot-wise ratios, which would overweight slots with
    few transmissions and count idle/deferred slots as zero-risk observations.
    """
    f = np.asarray(failed_counts, dtype=np.float64)
    s = np.asarray(scheduled_counts, dtype=np.float64)
    if f.shape != s.shape:
        raise ValueError("failed_counts and scheduled_counts must have identical shape.")
    n = int(f.size)
    out = np.zeros(n, dtype=np.float32)
    h = max(1, int(horizon))
    for t in range(n):
        end = min(n, t + h)
        den = float(np.sum(s[t:end]))
        num = float(np.sum(f[t:end]))
        out[t] = float(num / den) if den > 0.0 else 0.0
    return np.clip(out, 0.0, 1.0)

def upper_cvar_from_quantiles(q_values: torch.Tensor, quantiles: tuple[float, ...], alpha: float) -> torch.Tensor:
    """Approximate upper-tail CVaR by integrating the learned quantile function.

    CVaR_alpha = (1/(1-alpha)) * integral_alpha^1 Q(u) du.

    The configured quantile grid must contain alpha. The final learned quantile
    is held constant from its level to u=1, a conservative and transparent
    endpoint approximation for the unresolved extreme tail.
    """
    levels = [float(q) for q in quantiles]
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("CVaR alpha must lie strictly between 0 and 1.")
    if all(abs(q - a) > 1e-8 for q in levels):
        raise ValueError("The quantile grid must contain cvar_alpha exactly.")
    idx = [i for i, q in enumerate(levels) if q >= a]
    tail_levels = [levels[i] for i in idx]
    tail_values = q_values[..., idx]
    if tail_levels[-1] < 1.0:
        tail_levels = tail_levels + [1.0]
        tail_values = torch.cat([tail_values, tail_values[..., -1:]], dim=-1)
    area = torch.zeros_like(tail_values[..., 0])
    for j in range(len(tail_levels) - 1):
        width = float(tail_levels[j + 1] - tail_levels[j])
        area = area + 0.5 * width * (tail_values[..., j] + tail_values[..., j + 1])
    return area / max(1.0 - a, 1e-8)

class ShiftMonitor:
    """Prediction-error monitor that may tighten but never loosen the nominal risk budget."""
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.error_ema = float(cfg.shift_reference_error)
        self.initialized = False

    def reset(self, error_ema: float | None = None, initialized: bool = False):
        self.error_ema = float(self.cfg.shift_reference_error if error_ema is None else error_ema)
        self.initialized = bool(initialized)

    def update(self, predicted: float, realized: float) -> float:
        err = abs(float(predicted) - float(realized))
        a = float(self.cfg.shift_ema_alpha)
        if not self.initialized:
            self.error_ema = err
            self.initialized = True
        else:
            self.error_ema = (1.0 - a) * self.error_ema + a * err
        return self.error_ema

    def effective_limit(self) -> float:
        ref = max(float(self.cfg.shift_reference_error), float(self.cfg.risk_eps))
        excess = max(0.0, self.error_ema - ref) / ref
        factor = 1.0 / (1.0 + float(self.cfg.shift_tightening_gain) * excess)
        return float(np.clip(
            float(self.cfg.base_cvar_limit) * factor,
            float(self.cfg.min_cvar_limit),
            float(self.cfg.base_cvar_limit),
        ))

def train_risk_predictor(model, replay, cfg, optimizer, device, rng) -> float:
    """Perform a fixed number of replay-SGD updates.

    A fixed update count keeps compute per PPO rollout bounded as replay grows.
    """
    if len(replay) < int(cfg.min_replay_size):
        return float("nan")
    model.train()
    losses = []
    qt = torch.tensor(cfg.quantiles, dtype=torch.float32, device=device)
    steps = max(1, int(cfg.predictor_gradient_steps_per_update))
    snapshot = replay.snapshot()
    n_available = len(snapshot)
    for _ in range(steps):
        n = min(int(cfg.predictor_batch_size), n_available)
        idx = rng.choice(n_available, size=n, replace=False)
        batch = [snapshot[int(i)] for i in idx]
        hist = torch.as_tensor(np.stack([s.history for s in batch]), dtype=torch.float32, device=device)
        act = torch.as_tensor([s.action for s in batch], dtype=torch.long, device=device)
        tgt = torch.as_tensor([s.target for s in batch], dtype=torch.float32, device=device)
        pred = model(hist, act)
        loss = pinball_loss(pred, tgt, qt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")
