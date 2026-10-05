from __future__ import annotations
import numpy as np
import torch

from foresafe_config import PartialObservationConfig, RiskConfig, SafetyConfig
from foresafe_observation import PartialObservationWrapper
from foresafe_risk import QuantileRiskPredictor, finite_horizon_failure_ratio_targets, upper_cvar_from_quantiles, ShiftMonitor, HorizonRiskLabeler, RiskReplay, RiskSample
from foresafe_ppo import _gae_bootstrap, _combined_mask

class DummyCfg:
    protection_repetitions = 2

class DummyEnv:
    def __init__(self):
        self.n_actions = 64
        self.n_channels = 3
        self.n_classes = 3
        self.steps_per_ep = 10
        self.t = 0
        self.cfg = DummyCfg()
    def reset(self, seed=None):
        self.t = 0
        return np.arange(12, dtype=np.float32) / 12.0, {}
    def step(self, action):
        self.t += 1
        done = self.t >= self.steps_per_ep
        obs = np.arange(12, dtype=np.float32) / 12.0 + 0.01 * self.t
        return obs, 1.0, done, {"scheduled": 3, "failed_scheduled": int(self.t % 4 == 0)}
    def decode_joint_action(self, action):
        modes = []
        x = int(action)
        for _ in range(3):
            modes.append(x % 4)
            x //= 4
        return tuple(modes)
    def episode_metrics(self):
        return {"dummy": 1.0}

def main():
    env = PartialObservationWrapper(
        DummyEnv(),
        PartialObservationConfig(history_len=8, observation_delay_slots=2,
                                 gaussian_noise_std=0.01, dropout_probability=0.0),
        seed=7,
    )
    hist, _ = env.reset(seed=7)
    assert hist.shape == (8, 12)
    assert env.decoded_defer_fraction(0) == 1.0
    mask_r2 = env.resource_feasible_action_mask()
    assert int(mask_r2.sum()) == 42
    env.env.cfg.protection_repetitions = 3
    mask_r3 = env.resource_feasible_action_mask()
    assert int(mask_r3.sum()) == 30
    env.env.cfg.protection_repetitions = 2
    assert not bool(mask_r2[31])  # modes=(3,3,1), RB use=5>K=3

    # Candidate-level urgency uses each candidate's own age/deadline ratio.
    # Final MobiSafe observation layout: 7 globals + 3 class-backlog features,
    # then three groups of K candidates with 8 features each.
    synthetic = np.zeros((8, 82), dtype=np.float32)
    synthetic[-1, 10 + 1] = 0.90
    assert abs(env.urgency_from_history(synthetic) - 0.90) < 1e-6

    # Evidence-aware CVaR masking must never collapse a 42-action native set
    # to a single action merely because the learned predictor is pessimistic.
    cvar_high = torch.full((64,), 0.20, dtype=torch.float32)
    safety = SafetyConfig(
        min_action_samples_for_filter=25,
        min_operational_action_fraction=0.20,
        urgency_guard_threshold=0.85,
    )
    support = np.full(64, 100, dtype=np.int64)
    low_urgency = np.zeros((8, 82), dtype=np.float32)
    (
        op_mask, native_t, forced, op_count, _legacy_removed, urgency_removed,
        supported_native_count, strict_supported_pass, risk_rejected,
        risk_relaxed, urgency_score,
    ) = _combined_mask(
        env, cvar_high, 0.10, True, safety, low_urgency,
        torch.device("cpu"), support_counts=support, return_details=True,
    )
    assert int(native_t.sum().item()) == 42
    assert supported_native_count == 42
    assert strict_supported_pass == 0
    assert risk_rejected == 42
    assert op_count == 9  # ceil(0.20 * 42)
    assert risk_relaxed == 9
    assert int(op_mask.sum().item()) == 9
    assert not forced
    assert urgency_removed == 0
    assert urgency_score == 0.0

    # Unsupported actions are not hard-rejected by unsupported counterfactual
    # risk estimates; the native MobiSafe gate remains the safety floor.
    no_support = np.zeros(64, dtype=np.int64)
    (
        op_mask2, _native2, _forced2, op_count2, _legacy_removed2, _urg2,
        supported2, strict2, rejected2, relaxed2, _urgscore2,
    ) = _combined_mask(
        env, cvar_high, 0.10, True, safety, low_urgency,
        torch.device("cpu"), support_counts=no_support, return_details=True,
    )
    assert supported2 == 0
    assert strict2 == 0
    assert rejected2 == 0
    assert relaxed2 == 0
    assert op_count2 == 42
    assert int(op_mask2.sum().item()) == 42

    # Once a candidate reaches 90% of its own deadline, the urgency guard must
    # remove actions that defer more than one of the three RBs.
    cvar_low = torch.full((64,), 0.05, dtype=torch.float32)
    (
        urgent_mask, _n3, _f3, _c3, _r3, urgency_removed3,
        _s3, _sp3, _rr3, _rl3, urgency_score3,
    ) = _combined_mask(
        env, cvar_low, 0.10, True, safety, synthetic,
        torch.device("cpu"), support_counts=support, return_details=True,
    )
    assert urgency_score3 >= 0.90 - 1e-6
    assert urgency_removed3 > 0
    assert 0 < int(urgent_mask.sum().item()) < 42

    model = QuantileRiskPredictor(12, 64, 32, (0.5, 0.8, 0.9, 0.95, 0.99))
    x = torch.as_tensor(hist[None, ...], dtype=torch.float32)
    q = model.all_action_quantiles(x)
    assert q.shape == (64, 5)
    assert torch.all((q >= 0) & (q <= 1))
    c = upper_cvar_from_quantiles(q, (0.5, 0.8, 0.9, 0.95, 0.99), 0.8)
    assert c.shape == (64,)
    # Constant quantile function must integrate to the same constant.
    qc = torch.full((3, 5), 0.2)
    cc = upper_cvar_from_quantiles(qc, (0.5, 0.8, 0.9, 0.95, 0.99), 0.8)
    assert torch.allclose(cc, torch.full((3,), 0.2), atol=1e-6)

    tgt = finite_horizon_failure_ratio_targets(
        failed_counts=[0, 1, 0, 1, 1],
        scheduled_counts=[1, 1, 1, 2, 2],
        horizon=3,
    )
    assert tgt.shape == (5,)
    assert np.all((tgt >= 0) & (tgt <= 1))
    # First window: 1 failed / 3 scheduled.
    assert abs(float(tgt[0]) - (1.0 / 3.0)) < 1e-6

    # Cross-rollout horizon labeling: first sample matures only after H outcomes.
    labeler = HorizonRiskLabeler(horizon=3)
    h0 = np.zeros((8, 12), dtype=np.float32)
    assert labeler.push(h0, 1, 0.10, 0, 1) is None
    assert labeler.push(h0 + 1, 2, 0.20, 1, 1) is None
    matured = labeler.push(h0 + 2, 3, 0.30, 0, 1)
    assert matured is not None
    sample, pred, realized = matured
    assert sample.action == 1
    assert abs(pred - 0.10) < 1e-12
    assert abs(realized - (1.0 / 3.0)) < 1e-6
    assert len(labeler) == 2

    replay = RiskReplay(capacity=3)
    replay.add_many([
        RiskSample(h0, 1, 0.1),
        RiskSample(h0, 1, 0.2),
        RiskSample(h0, 2, 0.3),
    ])
    counts = replay.action_counts(64)
    assert counts[1] == 2 and counts[2] == 1
    replay.add_many([RiskSample(h0, 3, 0.4)])
    counts = replay.action_counts(64)
    assert counts[1] == 1 and counts[2] == 1 and counts[3] == 1

    # Nonterminal rollout cuts must bootstrap from the next critic value.
    returns, adv = _gae_bootstrap(
        rewards=[1.0, 1.0],
        values=[0.5, 0.5],
        dones=[0.0, 0.0],
        gamma=1.0,
        lam=1.0,
        bootstrap_value=0.5,
    )
    assert np.allclose(returns, np.asarray([2.5, 1.5], dtype=np.float32), atol=1e-6)
    assert np.allclose(adv, np.asarray([2.0, 1.0], dtype=np.float32), atol=1e-6)

    cfg = RiskConfig(base_cvar_limit=0.10, min_cvar_limit=0.04)
    monitor = ShiftMonitor(cfg)
    nominal = monitor.effective_limit()
    for _ in range(20):
        monitor.update(0.0, 1.0)
    tightened = monitor.effective_limit()
    assert tightened <= nominal + 1e-12
    assert tightened >= cfg.min_cvar_limit - 1e-12

    print("ForeSafe-RL structural validation passed.")

if __name__ == "__main__":
    main()
