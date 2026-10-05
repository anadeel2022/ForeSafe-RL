from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class PartialObservationConfig:
    history_len: int = 8
    observation_delay_slots: int = 2
    gaussian_noise_std: float = 0.02
    dropout_probability: float = 0.05
    hold_last_on_dropout: bool = True

@dataclass(frozen=True)
class RiskConfig:
    horizon: int = 12
    quantiles: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
    cvar_alpha: float = 0.80
    base_cvar_limit: float = 0.10
    min_cvar_limit: float = 0.04
    predictor_lr: float = 3e-4
    predictor_gradient_steps_per_update: int = 12
    predictor_batch_size: int = 256
    predictor_hidden_dim: int = 128
    warmup_steps: int = 1500
    shield_warmup_steps: int = 15000
    min_supported_action_fraction: float = 0.75
    min_replay_size: int = 512
    replay_capacity: int = 100000
    shift_ema_alpha: float = 0.05
    shift_reference_error: float = 0.08
    shift_tightening_gain: float = 1.5
    risk_eps: float = 1e-6

@dataclass(frozen=True)
class SafetyConfig:
    enable_cvar_filter: bool = True
    urgency_guard_threshold: float = 0.85
    max_defer_fraction_when_urgent: float = 0.34
    min_action_samples_for_filter: int = 25
    min_operational_action_fraction: float = 0.20
    min_safe_actions: int = 1

@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2e-4
    hidden_dim: int = 128
    gru_hidden_dim: int = 128
    train_epochs: int = 4
    minibatch_size: int = 128
    clip_ratio: float = 0.15
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    risk_penalty_coef: float = 1.0
    device: str = "cpu"

@dataclass(frozen=True)
class ExperimentConfig:
    train_episodes: int = 1200
    eval_episodes: int = 200
    seed: int = 11
    deterministic_eval: bool = True

def config_dict(*items) -> dict:
    out = {}
    for item in items:
        out[type(item).__name__] = asdict(item)
    return out
