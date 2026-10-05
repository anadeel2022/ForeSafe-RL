from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np

from foresafe_config import PartialObservationConfig

@dataclass
class ObservationDiagnostics:
    dropouts: int = 0
    noisy_observations: int = 0
    delayed_observations: int = 0

class PartialObservationWrapper:
    """POMDP interface over the validated MobiSafe packet/PHY simulator.

    The underlying dynamics are not modified. The controller receives delayed,
    noisy, intermittently missing observations and a fixed-length history.
    """

    def __init__(self, env: Any, cfg: PartialObservationConfig, seed: int = 0):
        self.env = env
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.n_actions = int(env.n_actions)
        self.steps_per_ep = int(env.steps_per_ep)
        self.n_channels = int(getattr(env, "n_channels", 1))
        self._delay = max(0, int(cfg.observation_delay_slots))
        self._raw_buffer: deque[np.ndarray] = deque(maxlen=max(1, self._delay + 1))
        self._history: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.history_len)))
        self._last_delivered: np.ndarray | None = None
        self.base_obs_dim: int | None = None
        self.diagnostics = ObservationDiagnostics()

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def obs_dim(self) -> int:
        if self.base_obs_dim is None:
            raise RuntimeError("Call reset() before querying obs_dim.")
        return int(self.base_obs_dim)

    @property
    def history_len(self) -> int:
        return int(self.cfg.history_len)

    def _corrupt(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        if self.rng.random() < float(self.cfg.dropout_probability):
            self.diagnostics.dropouts += 1
            if bool(self.cfg.hold_last_on_dropout) and self._last_delivered is not None:
                return self._last_delivered.copy()
            return np.zeros_like(x)
        std = float(self.cfg.gaussian_noise_std)
        if std > 0.0:
            x += self.rng.normal(0.0, std, size=x.shape).astype(np.float32)
            self.diagnostics.noisy_observations += 1
        self._last_delivered = x.copy()
        return x

    def _deliver_partial(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float32).reshape(-1)
        self._raw_buffer.append(x.copy())
        if len(self._raw_buffer) <= self._delay:
            delayed = self._raw_buffer[0].copy()
        else:
            delayed = list(self._raw_buffer)[-(self._delay + 1)].copy()
            if self._delay > 0:
                self.diagnostics.delayed_observations += 1
        return self._corrupt(delayed)

    def _stack(self) -> np.ndarray:
        return np.stack(list(self._history), axis=0).astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed) + 100003)
        out = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        raw, ctx = out if isinstance(out, tuple) else (out, {})
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        self.base_obs_dim = int(raw.size)
        self._raw_buffer.clear()
        self._history.clear()
        self._last_delivered = None
        self.diagnostics = ObservationDiagnostics()
        partial = self._deliver_partial(raw)
        for _ in range(self.history_len):
            self._history.append(partial.copy())
        return self._stack(), ctx

    def step(self, action: int):
        raw, reward, done, info = self.env.step(int(action))
        partial = self._deliver_partial(raw)
        self._history.append(partial.copy())
        return self._stack(), float(reward), bool(done), dict(info or {})

    def resource_feasible_action_mask(self) -> np.ndarray:
        """Exact published RB-budget mask.

        Mode costs are:
          defer=0, grant=1, priority_grant=1,
          protected_grant=R,
        where R is the configured protection repetition count.
        A joint action is feasible iff total RB use <= K.

        For K=3 this yields 42 feasible actions at R=2 and 30 at R=3,
        matching the published MobiSafe computational-scaling audit.
        """
        cfg = getattr(self.env, "cfg", None)
        repetitions = int(getattr(cfg, "protection_repetitions", 2))
        repetitions = max(repetitions, 1)
        mask = np.zeros(self.n_actions, dtype=bool)
        for action in range(self.n_actions):
            modes = self.env.decode_joint_action(int(action))
            use = 0
            for mode in modes:
                mode = int(mode)
                if mode == 0:
                    continue
                use += repetitions if mode == 3 else 1
            mask[action] = use <= self.n_channels
        if not bool(mask.any()):
            raise RuntimeError("Resource-feasibility mask contains no valid action.")
        return mask

    def decoded_defer_fraction(self, action: int) -> float:
        modes = self.env.decode_joint_action(int(action))
        if not modes:
            return 0.0
        return float(sum(int(m) == 0 for m in modes) / len(modes))

    def urgency_from_history(self, history: np.ndarray) -> float:
        """Return the strongest observed candidate-level age/deadline ratio.

        The final MobiSafe observation exposes, for each candidate in each mode
        group, an age-to-own-deadline feature clipped to [0, 4]. This is the
        correct service-urgency signal for the ForeSafe deadline guard.

        The earlier V1.4 implementation divided the global max-age feature by
        five. That delayed urgency activation and was especially unsuitable for
        short-deadline safety traffic.
        """
        x = np.asarray(history, dtype=np.float32)
        current = x[-1] if x.ndim == 2 else x.reshape(-1)

        n_classes = int(getattr(self.env, "n_classes", 3))
        n_channels = int(getattr(self.env, "n_channels", self.n_channels))
        base = 7 + n_classes
        feature_width = 8
        group_width = n_channels * feature_width

        age_ratios: list[float] = []
        for group in range(3):  # normal, priority, protected candidate groups
            start = base + group * group_width
            for ch in range(n_channels):
                idx = start + ch * feature_width + 1
                if 0 <= idx < current.size:
                    value = float(current[idx])
                    if np.isfinite(value):
                        age_ratios.append(value)

        if age_ratios:
            return float(np.clip(max(age_ratios), 0.0, 4.0))

        # Backward-compatible fallback for reduced synthetic observations.
        if current.size >= 3:
            return float(np.clip(current[2], 0.0, 5.0))
        return 0.0
