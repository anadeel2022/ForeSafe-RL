"""Packet-level, resource-feasible multichannel V-IoT environment (v29.1).

A single RSU owns ``n_channels`` orthogonal 1-MHz resource blocks in every
1-ms slot. A joint action is a base-4 channel-mode vector:

0 = defer;
1 = chance-qualified dedicated grant;
2 = chance-qualified priority grant; and
3 = protected grant using an explicit set of orthogonal replicas.

V29.1 retains the 4^K raw policy outputs but applies a dynamic resource-feasibility
mask before action selection. Protected grants consume two or three explicit
resource blocks, and every executed allocation is audited against the K-block
budget. The safety critic and PID regulator use the common additive residual
(F_t - d S_t)/K.

Version 24 keeps v23's shared risk-qualified multi-channel PHY, but replaces
slot-opportunity service accounting with packet-conservation accounting. Every
packet is explicitly represented from arrival until on-time delivery, deadline
expiry, queue-overflow drop, or episode end. The primary system service metrics
are therefore on-time packet delivery and deadline-miss ratios, not repeated
per-slot backlog opportunities.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any
import math
import numpy as np

from viot_phy import (
    PHYConfig,
    SLOT_DURATION_S,
    capacity_bits,
    link_sinr,
    log_distance_path_loss_db,
    path_loss_db,
    packet_success_probability,
    protected_packet_success_probability,
    thermal_noise_w,
    tx_energy_j,
    doppler_ar1_correlation,
    update_correlated_complex_fading,
)
from viot_v28_constraints import (
    MODE_NAMES,
    DEFER,
    DEDICATED_GRANT,
    PRIORITY_GRANT,
    PROTECTED_GRANT,
    assert_allocation,
    is_resource_feasible,
    slot_constraint_residual,
    valid_action_mask as build_valid_action_mask,
)


@dataclass(frozen=True)
class TrafficClass:
    name: str
    arrival_rate_hz: float
    payload_bits: int
    deadline_ms: float
    priority_weight: float
    reward_weight: float
    tx_power_dbm: float
    energy_per_bit_j: float


DEFAULT_CLASSES = (
    TrafficClass("safety", 10.0, 4000, 20.0, 3.0, 1.50, 18.0, 1.0e-8),
    TrafficClass("telemetry", 5.0, 2500, 100.0, 1.7, 1.00, 12.0, 1.1e-8),
    TrafficClass("best_effort", 2.0, 1800, 250.0, 1.0, 0.70, 8.0, 1.4e-8),
)


@dataclass(frozen=True)
class Packet:
    packet_id: int
    device_id: int
    bits: int
    arrival_slot: int
    deadline_slot: int
    is_warm_start: bool
    in_measurement_cohort: bool = False


@dataclass(frozen=True)
class StressConfig:
    """Evaluation-only channel-model mismatch profile.

    The chance-admission gate always remains based on the nominal ``PHYConfig``.
    These fields change only the realised channel law and/or the CSI age used
    by the gate during stress evaluation.  A nominal training environment uses
    the default all-zero profile.
    """
    name: str = "nominal"
    actual_shadowing_std_db: float | None = None
    blockage_start_probability: float = 0.0
    blockage_mean_duration_slots: float = 0.0
    blockage_attenuation_db: float = 0.0
    external_interference_probability: float = 0.0
    external_interferer_power_dbm: float = 20.0
    external_interferer_distance_m: float = 120.0
    external_interferer_distance_jitter_m: float = 20.0
    stale_csi_slots: int = 0
    actual_speed_mean_mps: float | None = None
    actual_speed_std_mps: float | None = None

    # V28 standards-aligned correlated-channel profile. Shadowing is common
    # across replicas of the same vehicle and correlated over displacement;
    # small-scale fading is independent across resource blocks but correlated
    # over time using a Doppler-derived AR(1) coefficient.
    temporal_channel_correlation: bool = False
    temporal_shadowing_correlation: bool = False
    shadowing_decorrelation_distance_m: float = 25.0
    fading_correlation_override: float | None = None

    # V29.3 protected-replica dependence sensitivity. These parameters affect
    # realized protected transmissions only. ``replica_fading_correlation`` is
    # the pairwise complex-baseband correlation coefficient across protected
    # resource blocks. Common shadowing/interference options produce a
    # conservative same-vehicle/same-interferer dependence case.
    replica_fading_correlation: float = 0.0
    replica_common_shadowing: bool = False
    replica_common_interference: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EnvConfig:
    n_devices: int = 36
    n_channels: int = 3
    steps_per_episode: int = 150
    seed: int = 1
    class_mix: tuple[float, float, float] = (0.25, 0.45, 0.30)
    arrival_scale: float = 1.0
    road_length_m: float = 1000.0
    rsu_x_m: float = 500.0
    rsu_coverage_m: float = 350.0
    speed_mean_mps: float = 22.0
    speed_std_mps: float = 5.0
    warm_start_packets: int = 1
    max_queue_packets: int = 50

    # V29/V29.1 terminal-safe accounting. When measurement_cohort_slots > 0,
    # warm-start packets and packets arriving in slots [0, measurement_cohort_slots)
    # are tagged as the measurement cohort.
    measurement_cohort_slots: int = 0

    # V29.1 horizon-corrected training support. A positive cutoff suppresses
    # new stochastic arrivals at and after this slot while mobility, channel
    # evolution, service, reward, and reliability-cost accounting continue.
    # This creates an explicit drain phase without silently discarding packets.
    # Zero preserves the V28/V29 continuing-arrival behavior.
    arrival_cutoff_slots: int = 0

    # Shared chance-qualified admission mechanism.
    risk_success_threshold: float = 0.90
    protection_repetitions: int = 2
    priority_control_overhead_fraction: float = 0.10
    protected_control_overhead_fraction: float = 0.02
    strict_resource_feasibility: bool = True

    # V28 adaptive safety supervisor. These switches are disabled for legacy
    # baselines and enabled by the V28 stress runner only for MobiSafe
    # attribution variants. The nominal calibrated threshold remains the base
    # gate. The effective gate and protected replica count are adjusted online
    # from realised failures, while a service guard prevents reliability from
    # being obtained purely through starvation.
    adaptive_risk_gate: bool = False
    adaptive_protection: bool = False
    adaptive_protection_escalation: bool = False
    adaptive_risk_min_threshold: float = 0.86
    adaptive_risk_max_threshold: float = 0.985
    adaptive_risk_gain: float = 1.40
    adaptive_risk_service_relief_gain: float = 0.65
    adaptive_supervisor_alpha: float = 0.20
    adaptive_cost_limit: float = 0.05
    adaptive_service_target: float = 0.55
    adaptive_deadline_miss_target: float = 0.25
    adaptive_protection_repetitions_max: int = 3
    adaptive_protection_trigger_margin: float = 0.035
    adaptive_urgent_age_ratio: float = 0.70

    # Shared base reward. None of these contains the binary PHY failure cost.
    queue_delay_penalty: float = 0.12
    network_backlog_penalty: float = 0.05
    energy_reward_penalty: float = 0.02
    defer_backlog_penalty: float = 0.35
    defer_urgency_multiplier: float = 0.50
    battery_capacity_j: float = 30.0
    device_jitter_cv: float = 0.05
    # The nominal PHY is also the model used by the frozen risk gate.
    # ``stress`` modifies only realised evaluation conditions.
    stress: StressConfig = field(default_factory=StressConfig)
    phy: PHYConfig = field(default_factory=PHYConfig)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["phy"] = self.phy.to_dict()
        return payload


class VIoTEnv:
    mode_names = MODE_NAMES
    mode_to_id = {name: i for i, name in enumerate(mode_names)}
    n_modes = len(mode_names)

    def __init__(self, config: EnvConfig | None = None, **overrides: Any):
        if config is None:
            config = EnvConfig(**overrides)
        elif overrides:
            raw = config.to_dict()
            raw.update(overrides)
            if isinstance(raw.get("phy"), dict):
                raw["phy"] = PHYConfig(**raw["phy"])
            if isinstance(raw.get("stress"), dict):
                raw["stress"] = StressConfig(**raw["stress"])
            config = EnvConfig(**raw)
        if int(config.n_channels) < 1:
            raise ValueError("n_channels must be at least one")
        if int(config.measurement_cohort_slots) < 0:
            raise ValueError("measurement_cohort_slots must be non-negative")
        if int(config.measurement_cohort_slots) > int(config.steps_per_episode):
            raise ValueError("measurement_cohort_slots cannot exceed steps_per_episode")
        if int(config.arrival_cutoff_slots) < 0:
            raise ValueError("arrival_cutoff_slots must be non-negative")
        if int(config.arrival_cutoff_slots) > int(config.steps_per_episode):
            raise ValueError("arrival_cutoff_slots cannot exceed steps_per_episode")
        if int(config.protection_repetitions) < 2:
            raise ValueError("protection_repetitions must be at least two")
        if int(config.adaptive_protection_repetitions_max) < int(config.protection_repetitions):
            raise ValueError("adaptive_protection_repetitions_max must be >= protection_repetitions")
        if int(config.protection_repetitions) > int(config.n_channels):
            raise ValueError("protection_repetitions cannot exceed n_channels")
        if int(config.adaptive_protection_repetitions_max) > int(config.n_channels):
            raise ValueError("adaptive_protection_repetitions_max cannot exceed n_channels")
        self.cfg = config
        self.rng = np.random.default_rng(int(config.seed))
        self.n_devices = int(config.n_devices)
        self.n_channels = int(config.n_channels)
        self.steps_per_ep = int(config.steps_per_episode)
        self.slot_s = float(SLOT_DURATION_S)
        self.n_classes = len(DEFAULT_CLASSES)
        self.n_actions = int(self.n_modes ** self.n_channels)
        self.action_space_n = self.n_actions
        self.action_names = self.mode_names
        self._init_static_devices()
        self.reset()

    # ------------------------------------------------------------------
    # Action coding.
    # ------------------------------------------------------------------
    def encode_joint_action(self, modes: list[int] | tuple[int, ...]) -> int:
        if len(modes) != self.n_channels:
            raise ValueError(f"joint action needs {self.n_channels} channel modes")
        value, factor = 0, 1
        for mode in modes:
            mode = int(mode)
            if not 0 <= mode < self.n_modes:
                raise ValueError(f"invalid channel mode: {mode}")
            value += mode * factor
            factor *= self.n_modes
        return int(value)

    def decode_joint_action(self, action: int) -> tuple[int, ...]:
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid joint action {action}; expected 0..{self.n_actions - 1}")
        modes: list[int] = []
        x = action
        for _ in range(self.n_channels):
            modes.append(int(x % self.n_modes))
            x //= self.n_modes
        return tuple(modes)

    def valid_action_mask(self, protection_repetitions: int | None = None) -> np.ndarray:
        """Boolean mask over all 4^K raw actions for the current replica count."""
        repetitions = int(
            self._effective_protection_repetitions()
            if protection_repetitions is None else protection_repetitions
        )
        return build_valid_action_mask(self.n_channels, repetitions, self.n_modes)

    def is_action_resource_feasible(self, action: int) -> bool:
        modes = self.decode_joint_action(int(action))
        return is_resource_feasible(
            modes, self.n_channels, self._effective_protection_repetitions()
        )

    # ------------------------------------------------------------------
    # Devices and packet queues.
    # ------------------------------------------------------------------
    def _init_static_devices(self) -> None:
        mix = np.asarray(self.cfg.class_mix, dtype=float)
        mix = np.clip(mix, 1e-9, None)
        mix = mix / mix.sum()
        self.device_class = self.rng.choice(self.n_classes, size=self.n_devices, p=mix)
        jitter = np.clip(self.rng.normal(1.0, self.cfg.device_jitter_cv, size=self.n_devices), 0.7, 1.3)
        self.tx_power_dbm = np.array([DEFAULT_CLASSES[int(c)].tx_power_dbm for c in self.device_class], dtype=float)
        self.tx_power_dbm += self.rng.normal(0.0, 0.5, size=self.n_devices)
        self.payload_bits = np.maximum(
            1,
            np.rint(np.array([DEFAULT_CLASSES[int(c)].payload_bits for c in self.device_class], dtype=float) * jitter).astype(int),
        )
        self.arrival_hz = np.array([DEFAULT_CLASSES[int(c)].arrival_rate_hz for c in self.device_class], dtype=float)
        self.arrival_hz *= float(self.cfg.arrival_scale)
        self.deadline_ms = np.array([DEFAULT_CLASSES[int(c)].deadline_ms for c in self.device_class], dtype=float)
        self.deadline_slots = np.maximum(1, np.ceil(self.deadline_ms / (self.slot_s * 1e3)).astype(int))
        self.priority_w = np.array([DEFAULT_CLASSES[int(c)].priority_weight for c in self.device_class], dtype=float)
        self.reward_w = np.array([DEFAULT_CLASSES[int(c)].reward_weight for c in self.device_class], dtype=float)
        self.energy_per_bit_j = np.array([DEFAULT_CLASSES[int(c)].energy_per_bit_j for c in self.device_class], dtype=float)
        self.battery_capacity_j = np.full(self.n_devices, float(self.cfg.battery_capacity_j), dtype=float)

    def _actual_phy_config(self) -> PHYConfig:
        """Return realised PHY configuration without changing nominal gate law."""
        std = self.cfg.stress.actual_shadowing_std_db
        if std is None:
            return self.cfg.phy
        raw = self.cfg.phy.to_dict()
        raw["shadowing_std_db"] = float(std)
        return PHYConfig(**raw)

    def _actual_speed_parameters(self) -> tuple[float, float]:
        mean = self.cfg.stress.actual_speed_mean_mps
        std = self.cfg.stress.actual_speed_std_mps
        return (
            float(self.cfg.speed_mean_mps if mean is None else mean),
            float(self.cfg.speed_std_mps if std is None else std),
        )

    def _init_stress_state(self) -> None:
        self.actual_phy = self._actual_phy_config()
        self.blockage_remaining_slots = np.zeros(self.n_devices, dtype=int)
        self.blockage_slot_count_ep = 0
        self.blocked_replica_attempts_ep = 0
        self.external_interference_replica_attempts_ep = 0
        self.replica_attempts_ep = 0

        shadow_std = max(float(self.actual_phy.shadowing_std_db), 0.0)
        self.correlated_shadowing_db = self.rng.normal(0.0, shadow_std, size=self.n_devices)
        self.correlated_fading = (
            self.rng.normal(size=(self.n_devices, self.n_channels))
            + 1j * self.rng.normal(size=(self.n_devices, self.n_channels))
        ) / math.sqrt(2.0)
        self.fading_rho_sum_ep = 0.0
        self.fading_rho_count_ep = 0
        self.shadowing_rho_sum_ep = 0.0
        self.shadowing_rho_count_ep = 0

    def _advance_blockage_state(self) -> None:
        """Evolve a per-device burst-blockage Markov process once per slot."""
        st = self.cfg.stress
        if st.blockage_start_probability <= 0.0 or st.blockage_mean_duration_slots <= 0.0:
            self.blockage_remaining_slots.fill(0)
            return
        active = self.blockage_remaining_slots > 0
        self.blockage_remaining_slots[active] -= 1
        inactive = np.where(self.blockage_remaining_slots <= 0)[0]
        if inactive.size == 0:
            return
        starts = self.rng.random(inactive.size) < float(st.blockage_start_probability)
        if not np.any(starts):
            return
        selected = inactive[starts]
        p = min(max(1.0 / float(st.blockage_mean_duration_slots), 1e-6), 1.0)
        durations = self.rng.geometric(p, size=selected.size).astype(int)
        self.blockage_remaining_slots[selected] = np.maximum(durations, 1)

    def _is_blocked(self, idx: int) -> bool:
        return bool(self.blockage_remaining_slots[int(idx)] > 0)

    def _estimated_distance_m(self, idx: int) -> float:
        """Nominal/stale CSI distance used only by the frozen risk estimator."""
        stale = max(int(self.cfg.stress.stale_csi_slots), 0)
        if stale <= 0:
            return float(self._distance_m()[int(idx)])
        estimated_position = (
            float(self.positions_m[int(idx)])
            - float(self.speed_mps[int(idx)]) * stale * self.slot_s
        ) % float(self.cfg.road_length_m)
        return float(max(abs(estimated_position - float(self.cfg.rsu_x_m)), 1.0))

    def _sample_external_interferers(self) -> list[tuple[float, float]]:
        st = self.cfg.stress
        if st.external_interference_probability <= 0.0:
            return []
        if self.rng.random() >= float(st.external_interference_probability):
            return []
        distance = max(
            1.0,
            float(st.external_interferer_distance_m)
            + float(self.rng.normal(0.0, max(float(st.external_interferer_distance_jitter_m), 0.0))),
        )
        return [(float(st.external_interferer_power_dbm), float(distance))]

    def _uses_correlated_channel(self) -> bool:
        return bool(self.cfg.stress.temporal_channel_correlation)

    def _uses_correlated_shadowing(self) -> bool:
        return bool(
            self.cfg.stress.temporal_channel_correlation
            or self.cfg.stress.temporal_shadowing_correlation
        )

    def _fading_correlation(self, idx: int) -> float:
        override = self.cfg.stress.fading_correlation_override
        if override is not None:
            return float(np.clip(float(override), 0.0, 0.9999))
        return doppler_ar1_correlation(
            speed_mps=float(self.speed_mps[int(idx)]),
            carrier_frequency_hz=float(self.actual_phy.carrier_frequency_hz),
            slot_duration_s=self.slot_s,
        )

    def _advance_correlated_channel_state(self) -> None:
        if self._uses_correlated_channel():
            for idx in range(self.n_devices):
                rho = self._fading_correlation(idx)
                self.fading_rho_sum_ep += float(rho)
                self.fading_rho_count_ep += 1
                for channel in range(self.n_channels):
                    self.correlated_fading[idx, channel] = update_correlated_complex_fading(
                        self.rng, self.correlated_fading[idx, channel], rho
                    )

        if self._uses_correlated_shadowing():
            decorrelation = max(
                float(self.cfg.stress.shadowing_decorrelation_distance_m), 1e-6
            )
            sigma = max(float(self.actual_phy.shadowing_std_db), 0.0)
            displacement = np.abs(self.speed_mps) * self.slot_s
            rho = np.exp(-displacement / decorrelation)
            innovation = self.rng.normal(0.0, sigma, size=self.n_devices)
            self.correlated_shadowing_db = (
                rho * self.correlated_shadowing_db
                + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * innovation
            )
            self.shadowing_rho_sum_ep += float(np.sum(rho))
            self.shadowing_rho_count_ep += int(rho.size)

    def _realized_channel_state(self, idx: int, channel: int) -> tuple[float | None, float | None]:
        shadowing = (
            float(self.correlated_shadowing_db[int(idx)])
            if self._uses_correlated_shadowing() else None
        )
        fading = (
            float(abs(self.correlated_fading[int(idx), int(channel)]) ** 2)
            if self._uses_correlated_channel() else None
        )
        return shadowing, fading

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self._init_static_devices()
        self.t = 0
        self.done = False
        self.positions_m = self.rng.uniform(0.0, self.cfg.road_length_m, size=self.n_devices)
        speed_mean, speed_std = self._actual_speed_parameters()
        speeds = self.rng.normal(speed_mean, speed_std, size=self.n_devices)
        self.speed_mps = speeds * self.rng.choice(np.array([-1.0, 1.0]), size=self.n_devices)
        self._init_stress_state()
        self.queues: list[deque[Packet]] = [deque() for _ in range(self.n_devices)]
        self.battery_j = self.battery_capacity_j.copy()
        self.last_sinr_db = np.full(self.n_devices, np.nan, dtype=float)
        self.last_success = np.zeros(self.n_devices, dtype=float)
        self._packet_id = 0
        self._risk_cache: dict[tuple[int, int, float], float] = {}
        self._reset_counters()
        self._seed_warm_start_packets()
        # Arrival at slot 0 means each episode contains exactly ``steps``
        # stochastic arrival epochs, rather than a terminal extra epoch.
        self._generate_arrivals(slot=0)
        self._expire_deadlines()
        self._refresh_candidates()
        self._append_queue_trace()
        return self._make_obs(), self.decision_context()

    def _reset_counters(self) -> None:
        # PHY-attempt accounting.
        self.bits_tx_ep = 0.0
        self.energy_j_ep = 0.0
        self.scheduled_tx_ep = 0
        self.successful_tx_ep = 0
        self.failed_scheduled_tx_ep = 0
        self.channel_uses_ep = 0
        self.protected_tx_ep = 0
        self.protected_replica_uses_ep = 0
        self.escalated_replica_tx_ep = 0
        self.invalid_action_selection_count_ep = 0
        self.resource_budget_violation_count_ep = 0
        self.max_resource_blocks_used_ep = 0
        self.action_mask_feasible_count_sum_ep = 0
        self.action_mask_query_count_ep = 0
        self.priority_overhead_bits_ep = 0.0
        self.protected_overhead_bits_ep = 0.0
        self.deferred_risk_admissible_ep = 0
        self.backlogged_packet_opportunities_ep = 0
        self.coverage_packet_opportunities_ep = 0
        self.risk_admissible_packet_opportunities_ep = 0

        # Packet-conservation accounting.
        self.initial_packets_ep = 0
        self.arrival_packets_ep = 0
        self.offered_packets_ep = 0
        self.delivered_packets_ep = 0
        self.delivered_initial_packets_ep = 0
        self.delivered_arrival_packets_ep = 0
        self.deadline_missed_packets_ep = 0
        self.deadline_missed_initial_packets_ep = 0
        self.deadline_missed_arrival_packets_ep = 0
        self.overflow_dropped_packets_ep = 0
        self.overflow_dropped_initial_packets_ep = 0
        self.overflow_dropped_arrival_packets_ep = 0

        # V29.1 terminal-safe measurement-cohort accounting. These counters are
        # inert when measurement_cohort_slots == 0 and do not alter V28 metrics.
        self.cohort_offered_packets_ep = 0
        self.cohort_delivered_packets_ep = 0
        self.cohort_deadline_missed_packets_ep = 0
        self.cohort_overflow_dropped_packets_ep = 0
        self.cohort_scheduled_tx_ep = 0
        self.cohort_successful_tx_ep = 0
        self.cohort_failed_scheduled_tx_ep = 0
        self.cohort_delivered_bits_ep = 0.0
        self.cohort_delay_samples_ms_ep: list[float] = []
        self.cohort_offered_per_class_ep = np.zeros(self.n_classes, dtype=int)
        self.cohort_delivered_per_class_ep = np.zeros(self.n_classes, dtype=int)
        self.cohort_deadline_missed_per_class_ep = np.zeros(self.n_classes, dtype=int)
        self.cohort_overflow_dropped_per_class_ep = np.zeros(self.n_classes, dtype=int)

        self.delay_samples_ms_ep: list[float] = []

        # V28 adaptive-supervisor state. EMAs are updated once per slot from
        # realised attempts; they affect subsequent decisions only.
        self.supervisor_failure_ema = 0.0
        self.supervisor_service_ema = 1.0
        self.supervisor_miss_ema = 0.0
        self.supervisor_effective_threshold_sum = 0.0
        self.supervisor_effective_threshold_count = 0
        self.supervisor_effective_repetition_sum = 0.0
        self.supervisor_effective_repetition_count = 0
        self.adaptive_gate_activations_ep = 0
        self.adaptive_protection_activations_ep = 0
        self.escalated_protection_tx_ep = 0

        self.served_bits_per_class_ep = np.zeros(self.n_classes, dtype=float)
        self.served_bits_per_device_ep = np.zeros(self.n_devices, dtype=float)
        self.queue_packets_trace: list[float] = []
        self.episode_trace: list[dict] = []

    def _new_packet(self, idx: int, slot: int, is_warm_start: bool) -> Packet:
        cohort_slots = int(self.cfg.measurement_cohort_slots)
        in_cohort = bool(
            cohort_slots > 0
            and (bool(is_warm_start) or int(slot) < cohort_slots)
        )
        packet = Packet(
            packet_id=int(self._packet_id),
            device_id=int(idx),
            bits=int(self.payload_bits[idx]),
            arrival_slot=int(slot),
            deadline_slot=int(slot + int(self.deadline_slots[idx])),
            is_warm_start=bool(is_warm_start),
            in_measurement_cohort=in_cohort,
        )
        self._packet_id += 1
        return packet

    def _record_offer(self, packet: Packet) -> None:
        self.offered_packets_ep += 1
        if packet.is_warm_start:
            self.initial_packets_ep += 1
        else:
            self.arrival_packets_ep += 1
        if packet.in_measurement_cohort:
            self.cohort_offered_packets_ep += 1
            cls = int(self.device_class[int(packet.device_id)])
            self.cohort_offered_per_class_ep[cls] += 1

    def _record_overflow_drop(self, packet: Packet) -> None:
        self.overflow_dropped_packets_ep += 1
        if packet.is_warm_start:
            self.overflow_dropped_initial_packets_ep += 1
        else:
            self.overflow_dropped_arrival_packets_ep += 1
        if packet.in_measurement_cohort:
            self.cohort_overflow_dropped_packets_ep += 1
            cls = int(self.device_class[int(packet.device_id)])
            self.cohort_overflow_dropped_per_class_ep[cls] += 1

    def _enqueue_packet(self, packet: Packet) -> None:
        self._record_offer(packet)
        q = self.queues[packet.device_id]
        if len(q) >= int(self.cfg.max_queue_packets):
            self._record_overflow_drop(packet)
            return
        q.append(packet)

    def _seed_warm_start_packets(self) -> None:
        if int(self.cfg.warm_start_packets) <= 0:
            return
        warm = self.rng.poisson(lam=float(self.cfg.warm_start_packets), size=self.n_devices)
        warm = np.minimum(warm, int(self.cfg.max_queue_packets))
        for idx, count in enumerate(warm.tolist()):
            for _ in range(int(count)):
                self._enqueue_packet(self._new_packet(int(idx), slot=0, is_warm_start=True))

    def _generate_arrivals(self, slot: int) -> None:
        cutoff = int(self.cfg.arrival_cutoff_slots)
        if cutoff > 0 and int(slot) >= cutoff:
            return
        arrivals = self.rng.poisson(np.clip(self.arrival_hz * self.slot_s, 0.0, 20.0))
        arrivals = np.minimum(arrivals, int(self.cfg.max_queue_packets))
        for idx, count in enumerate(arrivals.tolist()):
            for _ in range(int(count)):
                self._enqueue_packet(self._new_packet(int(idx), slot=int(slot), is_warm_start=False))

    def _expire_deadlines(self) -> None:
        """Drop expired packets and retain packet-level origin accounting."""
        for idx, q in enumerate(self.queues):
            while q and int(q[0].deadline_slot) < int(self.t):
                packet = q.popleft()
                self.deadline_missed_packets_ep += 1
                if packet.is_warm_start:
                    self.deadline_missed_initial_packets_ep += 1
                else:
                    self.deadline_missed_arrival_packets_ep += 1
                if packet.in_measurement_cohort:
                    self.cohort_deadline_missed_packets_ep += 1
                    cls = int(self.device_class[int(packet.device_id)])
                    self.cohort_deadline_missed_per_class_ep[cls] += 1

    def _head_packet(self, idx: int) -> Packet | None:
        q = self.queues[int(idx)]
        return q[0] if q else None

    def _queue_packet_count(self) -> int:
        return int(sum(len(q) for q in self.queues))

    def _append_queue_trace(self) -> None:
        self.queue_packets_trace.append(float(self._queue_packet_count()))

    # ------------------------------------------------------------------
    # Chance-qualified candidate selection.
    # ------------------------------------------------------------------
    def _distance_m(self) -> np.ndarray:
        return np.maximum(np.abs(self.positions_m - self.cfg.rsu_x_m), 1.0)

    def _in_coverage(self) -> np.ndarray:
        return self._distance_m() <= float(self.cfg.rsu_coverage_m)

    def _base_eligible(self) -> np.ndarray:
        has_packet = np.asarray([len(q) > 0 for q in self.queues], dtype=bool)
        return np.where(has_packet & (self.battery_j > 0.0) & self._in_coverage())[0]

    def _packet_age_slots(self, packet: Packet) -> int:
        return max(0, int(self.t) - int(packet.arrival_slot))

    def _estimated_sinr_db(self, idx: int) -> float:
        distance = float(self._estimated_distance_m(int(idx)))
        pl_db = path_loss_db(distance, self.cfg.phy)
        rx_w = 10.0 ** ((float(self.tx_power_dbm[idx]) - pl_db - 30.0) / 10.0)
        return float(10.0 * np.log10(max(rx_w / max(thermal_noise_w(self.cfg.phy), 1e-18), 1e-18)))

    def _mode_capacity_factor(self, mode: int) -> float:
        if int(mode) == PRIORITY_GRANT:
            overhead = float(self.cfg.priority_control_overhead_fraction)
        elif int(mode) == PROTECTED_GRANT:
            overhead = float(self.cfg.protected_control_overhead_fraction)
        else:
            overhead = 0.0
        return max(1e-6, 1.0 - float(np.clip(overhead, 0.0, 0.95)))

    def _estimated_capacity_bits(self, idx: int, mode: int) -> float:
        sinr = 10.0 ** (self._estimated_sinr_db(int(idx)) / 10.0)
        return float(capacity_bits(sinr, self.cfg.phy, coding_capacity_factor=self._mode_capacity_factor(mode)))

    def _single_success_probability(self, idx: int, mode: int) -> float:
        packet = self._head_packet(int(idx))
        if packet is None:
            return 0.0
        estimated_distance = float(self._estimated_distance_m(int(idx)))
        rounded_distance = round(estimated_distance * 2.0) / 2.0
        key = (int(packet.packet_id), int(mode), rounded_distance)
        if key not in self._risk_cache:
            self._risk_cache[key] = packet_success_probability(
                tx_power_dbm=float(self.tx_power_dbm[idx]),
                distance_m=estimated_distance,
                packet_bits=float(packet.bits),
                cfg=self.cfg.phy,
                coding_capacity_factor=self._mode_capacity_factor(mode),
            )
        return float(self._risk_cache[key])

    def _current_service_shortfall(self) -> float:
        service_deficit = max(0.0, float(self.cfg.adaptive_service_target) - float(self.supervisor_service_ema))
        miss_excess = max(0.0, float(self.supervisor_miss_ema) - float(self.cfg.adaptive_deadline_miss_target))
        return float(service_deficit + miss_excess)

    def _effective_risk_threshold(self) -> float:
        base = float(self.cfg.risk_success_threshold)
        if not bool(self.cfg.adaptive_risk_gate):
            return base
        fail_excess = max(0.0, float(self.supervisor_failure_ema) - float(self.cfg.adaptive_cost_limit))
        service_shortfall = self._current_service_shortfall()
        threshold = base + float(self.cfg.adaptive_risk_gain) * fail_excess
        threshold -= float(self.cfg.adaptive_risk_service_relief_gain) * service_shortfall
        threshold = float(np.clip(
            threshold,
            float(self.cfg.adaptive_risk_min_threshold),
            float(self.cfg.adaptive_risk_max_threshold),
        ))
        return threshold

    def _adaptive_failure_active(self) -> bool:
        margin = float(self.cfg.adaptive_protection_trigger_margin)
        return bool(float(self.supervisor_failure_ema) > float(self.cfg.adaptive_cost_limit) + margin)

    def _effective_protection_repetitions(self) -> int:
        base = int(self.cfg.protection_repetitions)
        if not bool(self.cfg.adaptive_protection):
            return base
        if not self._adaptive_failure_active():
            return base
        # Avoid escalating when service has already collapsed.  This keeps the
        # adaptive shield from turning into a pure deferral mechanism.
        if self._current_service_shortfall() > 0.20:
            return base
        return int(min(self.n_channels, max(base, int(self.cfg.adaptive_protection_repetitions_max))))

    def _risk_success_probability(self, idx: int, mode: int) -> float:
        if int(mode) == 3:
            return protected_packet_success_probability(
                self._single_success_probability(int(idx), 1),
                int(self._effective_protection_repetitions()),
            )
        return self._single_success_probability(int(idx), int(mode))

    def _risk_eligible_indices(self, mode: int, available: set[int] | None = None) -> list[int]:
        if int(mode) == 0:
            return []
        required_blocks = int(self._effective_protection_repetitions()) if int(mode) == 3 else 1
        if required_blocks > self.n_channels:
            return []
        candidates = self._base_eligible().tolist()
        if available is not None:
            candidates = [int(i) for i in candidates if int(i) in available]
        threshold = float(self._effective_risk_threshold())
        return [int(i) for i in candidates if self._risk_success_probability(int(i), int(mode)) >= threshold]

    def _score(self, idx: int, mode: int, priority: bool) -> float:
        packet = self._head_packet(int(idx))
        if packet is None:
            return -math.inf
        probability = self._risk_success_probability(int(idx), int(mode))
        age_ms = float(self._packet_age_slots(packet) * self.slot_s * 1e3)
        age_ratio = age_ms / max(float(self.deadline_ms[idx]), 1e-6)
        packet_ratio = self._estimated_capacity_bits(int(idx), int(mode)) / max(float(packet.bits), 1.0)
        if priority:
            return float(
                2.2 * self.priority_w[idx] * (1.0 + age_ratio)
                + 0.50 * min(len(self.queues[idx]), 10)
                + 2.0 * probability
            )
        return float(3.0 * probability + 1.2 * min(packet_ratio, 2.0) + 0.03 * self._packet_age_slots(packet) + 0.1 * self.priority_w[idx])

    def _select_best(self, mode: int, available: set[int], priority: bool | None = None) -> int | None:
        candidates = self._risk_eligible_indices(int(mode), available)
        if not candidates:
            return None
        use_priority = bool(int(mode) == 2) if priority is None else bool(priority)
        return int(max(candidates, key=lambda idx: self._score(int(idx), int(mode), use_priority)))

    def _refresh_candidates(self) -> None:
        available = set(int(i) for i in self._base_eligible().tolist())
        self.normal_candidate = self._select_best(1, available, priority=False)
        self.priority_candidate = self._select_best(2, available, priority=True)
        self.protected_candidate = self._select_best(3, available, priority=True)

    def _candidate_dict(self, idx: int | None, mode: int) -> dict:
        if idx is None:
            return {
                "idx": -1, "has_backlog": 0, "in_coverage": 0,
                "estimated_sinr_db": -30.0, "estimated_capacity_bits": 0.0,
                "payload_bits": 0.0, "risk_success_probability": 0.0,
                "risk_qualified": 0, "urgency": 0.0, "packet_age_ms": 0.0,
            }
        packet = self._head_packet(int(idx))
        if packet is None:
            return self._candidate_dict(None, mode)
        age_ms = float(self._packet_age_slots(packet) * self.slot_s * 1e3)
        urgency = float(self.priority_w[idx] * (1.0 + age_ms / max(float(self.deadline_ms[idx]), 1e-6)))
        prob = self._risk_success_probability(int(idx), int(mode))
        return {
            "idx": int(idx), "has_backlog": 1, "in_coverage": int(self._in_coverage()[idx]),
            "estimated_sinr_db": float(self._estimated_sinr_db(int(idx))),
            "estimated_capacity_bits": float(self._estimated_capacity_bits(int(idx), int(mode)),),
            "payload_bits": float(packet.bits), "risk_success_probability": float(prob),
            "risk_qualified": int(prob >= float(self._effective_risk_threshold())),
            "urgency": urgency, "packet_age_ms": age_ms,
        }

    def _risk_admissible_packet_capacity(self) -> int:
        singles = set(self._risk_eligible_indices(1))
        protected = set(self._risk_eligible_indices(3))
        protected_only = protected.difference(singles)
        single_count = min(self.n_channels, len(singles))
        remaining = self.n_channels - single_count
        extra = min(len(protected_only), remaining // int(self._effective_protection_repetitions()))
        return int(single_count + extra)

    def _plan_joint_action(self, modes: tuple[int, ...]) -> tuple[list[dict], int]:
        """Map a feasible raw joint action to explicit, nonoverlapping RB sets.

        Each non-defer entry owns its primary resource block. Protected grants
        additionally reserve ``R_t-1`` blocks whose raw modes are defer. The
        same rule is used by the action mask, executor, and audit counters.
        """
        repetitions = int(self._effective_protection_repetitions())
        if not is_resource_feasible(modes, self.n_channels, repetitions):
            self.invalid_action_selection_count_ep += 1
            self.resource_budget_violation_count_ep += 1
            if bool(self.cfg.strict_resource_feasibility):
                raise ValueError(
                    f"resource-infeasible action {modes}: "
                    f"R={repetitions}, K={self.n_channels}"
                )
            return [], 0

        available_devices = set(int(i) for i in self._base_eligible().tolist())
        free_replica_blocks = [i for i, mode in enumerate(modes) if int(mode) == DEFER]
        records: list[dict] = []
        all_used_blocks: list[int] = []

        for primary_block, raw_mode in enumerate(modes):
            raw_mode = int(raw_mode)
            if raw_mode == DEFER:
                continue

            requested_mode = raw_mode
            mode = raw_mode
            idx = self._select_best(mode, available_devices)
            if idx is None:
                continue

            # Optional online escalation may use only deferred blocks that are
            # not required by explicit protected modes still waiting later in
            # the raw joint action. Without this reservation, an early ordinary
            # grant could consume the sole deferred replica block and make a
            # later explicit protected grant impossible even though the raw
            # action passed the resource-feasibility mask.
            if (
                bool(self.cfg.adaptive_protection_escalation)
                and mode in (DEDICATED_GRANT, PRIORITY_GRANT)
                and self._adaptive_failure_active()
            ):
                pkt = self._head_packet(int(idx))
                age_ratio = 0.0 if pkt is None else (
                    self._packet_age_slots(pkt) * self.slot_s * 1e3
                ) / max(float(self.deadline_ms[idx]), 1e-6)
                needed = max(repetitions - 1, 0)
                remaining_explicit_protected = sum(
                    int(future_mode) == PROTECTED_GRANT
                    for future_mode in modes[primary_block + 1 :]
                )
                reserved_for_explicit = remaining_explicit_protected * needed
                optional_capacity = max(
                    len(free_replica_blocks) - reserved_for_explicit,
                    0,
                )
                if (
                    age_ratio >= float(self.cfg.adaptive_urgent_age_ratio)
                    and optional_capacity >= needed
                ):
                    mode = PROTECTED_GRANT

            blocks = [int(primary_block)]
            if mode == PROTECTED_GRANT:
                needed = max(repetitions - 1, 0)
                if len(free_replica_blocks) < needed:
                    # Explicit protected actions are already masked; this branch
                    # can only be reached through an inconsistent external policy.
                    self.resource_budget_violation_count_ep += 1
                    if bool(self.cfg.strict_resource_feasibility):
                        raise AssertionError("protected grant lacks deferred replica blocks")
                    continue
                replica_blocks = tuple(free_replica_blocks[:needed])
                del free_replica_blocks[:needed]
                blocks.extend(replica_blocks)
            else:
                replica_blocks = tuple()

            records.append({
                "idx": int(idx),
                "mode": int(mode),
                "requested_mode": int(requested_mode),
                "primary_block": int(primary_block),
                "replica_blocks": tuple(int(x) for x in replica_blocks),
                "resource_blocks": tuple(int(x) for x in blocks),
                "channel_uses": int(len(blocks)),
            })
            available_devices.remove(int(idx))
            all_used_blocks.extend(blocks)

        try:
            assert_allocation(all_used_blocks, self.n_channels)
        except AssertionError:
            self.resource_budget_violation_count_ep += 1
            raise
        used = int(len(all_used_blocks))
        self.max_resource_blocks_used_ep = max(self.max_resource_blocks_used_ep, used)
        return records, used

    def _build_oracle_action(self, priority: bool = False) -> int:
        modes: list[int] = []
        available = set(int(i) for i in self._base_eligible().tolist())
        used = 0
        primary_mode = 2 if priority else 1
        while used < self.n_channels:
            idx = self._select_best(primary_mode, available, priority=priority)
            selected_mode = primary_mode
            if idx is None and priority:
                idx = self._select_best(1, available, priority=False)
                selected_mode = 1
            if idx is None:
                break
            modes.append(int(selected_mode))
            available.remove(int(idx))
            used += 1
        while used + int(self._effective_protection_repetitions()) <= self.n_channels:
            idx = self._select_best(3, available, priority=priority)
            if idx is None:
                break
            modes.append(3)
            available.remove(int(idx))
            used += int(self._effective_protection_repetitions())
        modes = modes[: self.n_channels] + [0] * max(0, self.n_channels - len(modes))
        return self.encode_joint_action(tuple(modes[: self.n_channels]))

    def _build_protected_maxweight_action(self) -> int:
        """Construct a deterministic protected-first reliability reference."""
        repetitions = int(self._effective_protection_repetitions())
        modes = [DEFER] * self.n_channels
        available = set(int(i) for i in self._base_eligible().tolist())
        protected_idx = self._select_best(PROTECTED_GRANT, available, priority=True)
        cursor = 0
        if protected_idx is not None and repetitions <= self.n_channels:
            modes[0] = PROTECTED_GRANT
            available.remove(int(protected_idx))
            cursor = repetitions
        while cursor < self.n_channels:
            idx = self._select_best(PRIORITY_GRANT, available, priority=True)
            if idx is None:
                break
            modes[cursor] = PRIORITY_GRANT
            available.remove(int(idx))
            cursor += 1
        action = self.encode_joint_action(tuple(modes))
        if not bool(self.valid_action_mask(repetitions)[action]):
            raise AssertionError("protected MaxWeight builder produced an infeasible action")
        return int(action)

    def decision_context(self) -> dict:
        repetitions = int(self._effective_protection_repetitions())
        mask = self.valid_action_mask(repetitions)
        feasible_count = int(np.sum(mask))
        self.action_mask_feasible_count_sum_ep += feasible_count
        self.action_mask_query_count_ep += 1
        return {
            "normal": self._candidate_dict(self.normal_candidate, DEDICATED_GRANT),
            "priority": self._candidate_dict(self.priority_candidate, PRIORITY_GRANT),
            "protected": self._candidate_dict(self.protected_candidate, PROTECTED_GRANT),
            "has_any_backlog": int(self._queue_packet_count() > 0),
            "coverage_available": int(len(self._base_eligible()) > 0),
            "risk_admissible_packet_capacity": int(self._risk_admissible_packet_capacity()),
            "risk_success_threshold": float(self.cfg.risk_success_threshold),
            "effective_risk_success_threshold": float(self._effective_risk_threshold()),
            "effective_protection_repetitions": repetitions,
            "action_mask": mask.copy(),
            "feasible_action_count": feasible_count,
            "n_actions": int(self.n_actions),
            "n_channels": int(self.n_channels),
            "supervisor_failure_ema": float(self.supervisor_failure_ema),
            "supervisor_service_ema": float(self.supervisor_service_ema),
            "supervisor_miss_ema": float(self.supervisor_miss_ema),
            "stress_profile": str(self.cfg.stress.name),
            "oracle_action": int(self._build_oracle_action(priority=False)),
            "maxweight_action": int(self._build_oracle_action(priority=True)),
            "protected_maxweight_action": int(self._build_protected_maxweight_action()),
        }

    def _correlated_replica_fading_powers(self, repetitions: int) -> list[float] | None:
        """Return correlated unit-mean Rayleigh powers for one protected grant.

        If rho=0 the caller retains the historical per-RB channel generation.
        For rho>0, h_i = sqrt(rho) z_0 + sqrt(1-rho) z_i with independent
        circular complex Gaussian z_0,z_i. Hence E[h_i h_j*]=rho and every
        marginal |h_i|^2 remains unit-mean exponential.
        """
        rho = float(np.clip(self.cfg.stress.replica_fading_correlation, 0.0, 0.9999))
        if repetitions <= 1 or rho <= 0.0:
            return None
        common = (self.rng.normal() + 1j * self.rng.normal()) / math.sqrt(2.0)
        out: list[float] = []
        for _ in range(int(repetitions)):
            independent = (self.rng.normal() + 1j * self.rng.normal()) / math.sqrt(2.0)
            h = math.sqrt(rho) * common + math.sqrt(max(1.0 - rho, 0.0)) * independent
            out.append(float(abs(h) ** 2))
        return out

    def _common_replica_shadowing_db(self) -> float | None:
        if not bool(self.cfg.stress.replica_common_shadowing):
            return None
        return float(self.rng.normal(0.0, max(float(self.actual_phy.shadowing_std_db), 0.0)))

    def _serve_record(self, record: dict) -> dict:
        idx = int(record["idx"])
        mode = int(record["mode"])
        resource_blocks = tuple(int(x) for x in record.get("resource_blocks", ()))
        assert_allocation(resource_blocks, self.n_channels)
        packet = self._head_packet(idx)
        if packet is None or not bool(self._in_coverage()[idx]) or self.battery_j[idx] <= 0.0:
            return {
                "delivered_bits": 0.0, "energy_j": 0.0, "success": 0,
                "sinr_db": np.nan, "replica_count": len(resource_blocks),
                "priority_overhead_bits": 0.0, "protected_overhead_bits": 0.0,
                "attempted_packet": packet,
            }

        repetitions = max(len(resource_blocks), 1)
        capacity_factor = self._mode_capacity_factor(mode)
        per_replica_energy = tx_energy_j(
            float(self.tx_power_dbm[idx]), float(packet.bits), float(self.energy_per_bit_j[idx])
        )
        total_energy = repetitions * per_replica_energy
        if total_energy > self.battery_j[idx]:
            return {
                "delivered_bits": 0.0, "energy_j": 0.0, "success": 0,
                "sinr_db": np.nan, "replica_count": repetitions,
                "priority_overhead_bits": 0.0, "protected_overhead_bits": 0.0,
                "attempted_packet": packet,
            }

        blocked = self._is_blocked(idx)
        desired_power = float(self.tx_power_dbm[idx]) - (
            float(self.cfg.stress.blockage_attenuation_db) if blocked else 0.0
        )
        realizations: list[dict] = []
        interference_hits = 0
        priority_overhead_bits = 0.0
        protected_overhead_bits = 0.0

        # Historical baseline: replica-specific shadowing, fading, and
        # interference draws (with common geometry/blockage). Reviewer
        # sensitivities can impose common shadowing/interference and correlated
        # small-scale fading across the protected RBs.
        correlated_replica_fading = (
            self._correlated_replica_fading_powers(repetitions)
            if mode == PROTECTED_GRANT else None
        )
        common_replica_shadowing = (
            self._common_replica_shadowing_db()
            if mode == PROTECTED_GRANT else None
        )
        common_replica_interferers = (
            self._sample_external_interferers()
            if mode == PROTECTED_GRANT and bool(self.cfg.stress.replica_common_interference)
            else None
        )

        for replica_index, channel in enumerate(resource_blocks):
            interferers = (
                common_replica_interferers
                if common_replica_interferers is not None
                else self._sample_external_interferers()
            )
            interference_hits += int(bool(interferers))
            shadowing_db, fading_power = self._realized_channel_state(idx, channel)
            if common_replica_shadowing is not None:
                shadowing_db = float(common_replica_shadowing)
            if correlated_replica_fading is not None:
                fading_power = float(correlated_replica_fading[replica_index])
            realization = link_sinr(
                self.rng,
                desired_tx_power_dbm=desired_power,
                desired_distance_m=float(self._distance_m()[idx]),
                cfg=self.actual_phy,
                interferers=interferers,
                coding_capacity_factor=capacity_factor,
                desired_shadowing_db=shadowing_db,
                desired_fading_power=fading_power,
            )
            gross_capacity = capacity_bits(
                float(realization["sinr_linear"]), self.actual_phy, coding_capacity_factor=1.0
            )
            overhead_bits = max(float(gross_capacity) - float(realization["capacity_bits"]), 0.0)
            if mode == PRIORITY_GRANT:
                priority_overhead_bits += overhead_bits
            elif mode == PROTECTED_GRANT:
                protected_overhead_bits += overhead_bits
            realizations.append(realization)

        self.replica_attempts_ep += repetitions
        self.blocked_replica_attempts_ep += int(blocked) * repetitions
        self.external_interference_replica_attempts_ep += int(interference_hits)
        self.priority_overhead_bits_ep += float(priority_overhead_bits)
        self.protected_overhead_bits_ep += float(protected_overhead_bits)

        success = bool(any(float(item["capacity_bits"]) >= float(packet.bits) for item in realizations))
        self.battery_j[idx] = max(0.0, self.battery_j[idx] - total_energy)
        completed_packet = self.queues[idx].popleft() if success else None
        return {
            "delivered_bits": float(packet.bits) if success else 0.0,
            "energy_j": float(total_energy),
            "success": int(success),
            "sinr_db": float(max(float(item["sinr_db"]) for item in realizations)),
            "replica_count": repetitions,
            "resource_blocks": resource_blocks,
            "completed_packet": completed_packet,
            "attempted_packet": packet,
            "predicted_success_probability": float(self._risk_success_probability(idx, mode)),
            "priority_overhead_bits": float(priority_overhead_bits),
            "protected_overhead_bits": float(protected_overhead_bits),
        }

    def _record_delivery(self, packet: Packet, idx: int, outcome: dict) -> None:
        self.delivered_packets_ep += 1
        if packet.is_warm_start:
            self.delivered_initial_packets_ep += 1
        else:
            self.delivered_arrival_packets_ep += 1
        self.bits_tx_ep += float(packet.bits)
        delay_ms = float(self._packet_age_slots(packet) * self.slot_s * 1e3)
        self.delay_samples_ms_ep.append(delay_ms)
        if packet.in_measurement_cohort:
            self.cohort_delivered_packets_ep += 1
            self.cohort_delivered_bits_ep += float(packet.bits)
            self.cohort_delay_samples_ms_ep.append(delay_ms)
            cls = int(self.device_class[int(packet.device_id)])
            self.cohort_delivered_per_class_ep[cls] += 1
        self.served_bits_per_class_ep[int(self.device_class[idx])] += float(packet.bits)
        self.served_bits_per_device_ep[idx] += float(packet.bits)

    def _advance_network(self) -> None:
        self.positions_m = (self.positions_m + self.speed_mps * self.slot_s) % self.cfg.road_length_m
        self._advance_correlated_channel_state()
        self._generate_arrivals(slot=int(self.t))
        self._expire_deadlines()
        self._refresh_candidates()
        self._append_queue_trace()

    def _network_queue_pressure(self) -> float:
        return float(np.clip(self._queue_packet_count() / max(float(self.n_devices * self.cfg.max_queue_packets), 1.0), 0.0, 1.0))

    def _base_reward(self, records: list[dict], outcomes: list[dict], risk_capacity: int) -> tuple[float, dict]:
        payload_reward = delay_penalty = energy_penalty = urgency_total = 0.0
        for record, outcome in zip(records, outcomes):
            idx = int(record["idx"])
            packet = outcome.get("completed_packet") or self._head_packet(idx)
            # On a failed attempt the packet is still in the queue; its age is
            # nevertheless well-defined and belongs in the shared delay term.
            if packet is None:
                continue
            age_ratio = (self._packet_age_slots(packet) * self.slot_s * 1e3) / max(float(self.deadline_ms[idx]), 1e-6)
            payload_reward += float(self.reward_w[idx]) * float(outcome["success"])
            delay_penalty += float(self.cfg.queue_delay_penalty) * min(float(age_ratio), 3.0)
            energy_ref = tx_energy_j(float(self.tx_power_dbm[idx]), float(packet.bits), float(self.energy_per_bit_j[idx]))
            energy_penalty += float(self.cfg.energy_reward_penalty) * min(float(outcome["energy_j"]) / max(energy_ref, 1e-12), 4.0)
            urgency_total += float(self.priority_w[idx] * (1.0 + age_ratio))
        deferred = max(int(risk_capacity) - len(records), 0)
        urgency_scale = 1.0 + float(self.cfg.defer_urgency_multiplier) * min(urgency_total / max(3.0 * max(len(records), 1), 1e-9), 2.0)
        defer_penalty = float(self.cfg.defer_backlog_penalty) * deferred * urgency_scale
        backlog_penalty = float(self.cfg.network_backlog_penalty) * self._network_queue_pressure()
        reward = payload_reward - delay_penalty - energy_penalty - defer_penalty - backlog_penalty
        return float(np.clip(reward, -12.0, 12.0)), {
            "payload_reward": float(payload_reward),
            "target_delay_penalty": float(delay_penalty),
            "energy_penalty": float(energy_penalty),
            "defer_backlog_penalty": float(defer_penalty),
            "network_backlog_penalty": float(backlog_penalty),
            "deferred_risk_admissible": int(deferred),
        }

    def _update_adaptive_supervisor(self, scheduled: int, failed: int, successful: int, risk_capacity: int) -> None:
        alpha = float(np.clip(self.cfg.adaptive_supervisor_alpha, 0.0, 1.0))
        if scheduled > 0:
            slot_failure = float(failed) / max(float(scheduled), 1.0)
            self.supervisor_failure_ema = float((1.0 - alpha) * self.supervisor_failure_ema + alpha * slot_failure)
        slot_service = float(successful) / max(float(risk_capacity), 1.0) if risk_capacity > 0 else 1.0
        self.supervisor_service_ema = float((1.0 - alpha) * self.supervisor_service_ema + alpha * slot_service)
        offered = max(float(self.offered_packets_ep), 1.0)
        miss = float(self.deadline_missed_packets_ep) / offered
        self.supervisor_miss_ema = float((1.0 - alpha) * self.supervisor_miss_ema + alpha * miss)
        eff_thr = self._effective_risk_threshold()
        eff_rep = self._effective_protection_repetitions()
        self.supervisor_effective_threshold_sum += float(eff_thr)
        self.supervisor_effective_threshold_count += 1
        self.supervisor_effective_repetition_sum += float(eff_rep)
        self.supervisor_effective_repetition_count += 1
        self.adaptive_gate_activations_ep += int(bool(self.cfg.adaptive_risk_gate) and abs(eff_thr - float(self.cfg.risk_success_threshold)) > 1e-9)
        self.adaptive_protection_activations_ep += int(bool(self.cfg.adaptive_protection) and eff_rep > int(self.cfg.protection_repetitions))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self._make_obs(), 0.0, True, {}
        self._expire_deadlines()
        self._refresh_candidates()
        self._advance_blockage_state()
        if np.any(self.blockage_remaining_slots > 0):
            self.blockage_slot_count_ep += 1
        modes = self.decode_joint_action(int(action))
        effective_threshold_before = float(self._effective_risk_threshold())
        effective_repetitions_before = int(self._effective_protection_repetitions())

        queued_packets = self._queue_packet_count()
        coverage_devices = int(len(self._base_eligible()))
        risk_capacity = int(self._risk_admissible_packet_capacity())
        self.backlogged_packet_opportunities_ep += min(self.n_channels, queued_packets)
        self.coverage_packet_opportunities_ep += min(self.n_channels, coverage_devices)
        self.risk_admissible_packet_opportunities_ep += risk_capacity

        mask = self.valid_action_mask(effective_repetitions_before)
        if not bool(mask[int(action)]):
            self.invalid_action_selection_count_ep += 1
            self.resource_budget_violation_count_ep += 1
            if bool(self.cfg.strict_resource_feasibility):
                raise ValueError(
                    f"policy selected masked action {action} with modes={modes}, "
                    f"R={effective_repetitions_before}"
                )
        records, channel_uses = self._plan_joint_action(modes)
        outcomes = [self._serve_record(rec) for rec in records]
        scheduled = len(records)
        successful = int(sum(int(outcome["success"]) for outcome in outcomes))
        failed = int(scheduled - successful)
        cohort_attempts = [
            outcome for outcome in outcomes
            if isinstance(outcome.get("attempted_packet"), Packet)
            and bool(outcome["attempted_packet"].in_measurement_cohort)
        ]
        cohort_scheduled = len(cohort_attempts)
        cohort_successful = int(sum(int(outcome["success"]) for outcome in cohort_attempts))
        self.cohort_scheduled_tx_ep += cohort_scheduled
        self.cohort_successful_tx_ep += cohort_successful
        self.cohort_failed_scheduled_tx_ep += int(cohort_scheduled - cohort_successful)
        self.scheduled_tx_ep += scheduled
        self.successful_tx_ep += successful
        self.failed_scheduled_tx_ep += failed
        self.channel_uses_ep += int(channel_uses)
        protected_now = int(sum(int(rec["mode"]) == PROTECTED_GRANT for rec in records))
        protected_replica_uses_now = int(sum(max(int(rec["channel_uses"]) - 1, 0) for rec in records if int(rec["mode"]) == PROTECTED_GRANT))
        escalated_now = int(sum(int(rec.get("requested_mode", rec["mode"])) != PROTECTED_GRANT and int(rec["mode"]) == PROTECTED_GRANT for rec in records))
        self.protected_tx_ep += protected_now
        self.protected_replica_uses_ep += protected_replica_uses_now
        self.escalated_protection_tx_ep += escalated_now
        self.escalated_replica_tx_ep += int(sum(int(rec["channel_uses"]) for rec in records if int(rec.get("requested_mode", rec["mode"])) != PROTECTED_GRANT and int(rec["mode"]) == PROTECTED_GRANT))
        self.deferred_risk_admissible_ep += max(risk_capacity - scheduled, 0)

        for record, outcome in zip(records, outcomes):
            idx = int(record["idx"])
            self.energy_j_ep += float(outcome["energy_j"])
            self.last_sinr_db[idx] = float(outcome.get("sinr_db", np.nan))
            self.last_success[idx] = float(outcome["success"])
            packet = outcome.get("completed_packet")
            if int(outcome["success"]) and isinstance(packet, Packet):
                self._record_delivery(packet, idx, outcome)

        self._update_adaptive_supervisor(scheduled, failed, successful, risk_capacity)
        reward, reward_terms = self._base_reward(records, outcomes, risk_capacity)
        trace = {
            "t": int(self.t), "joint_action": int(action),
            "joint_modes": ";".join(self.mode_names[m] for m in modes),
            "scheduled": int(scheduled), "successful_tx": int(successful),
            "failed_scheduled": int(failed), "channel_uses": int(channel_uses),
            "slot_resource_blocks_used": int(channel_uses),
            "n_channels": int(self.n_channels),
            "action_mask_feasible_count": int(np.sum(mask)),
            "invalid_action_selection_count": int(self.invalid_action_selection_count_ep),
            "resource_budget_violation_count": int(self.resource_budget_violation_count_ep),
            "protected_tx": int(protected_now),
            "protected_replica_uses": int(protected_replica_uses_now),
            "escalated_protection_tx": int(escalated_now),
            "constraint_numerator": float(failed - float(self.cfg.adaptive_cost_limit) * scheduled),
            "constraint_residual": float(slot_constraint_residual(
                failed, scheduled, float(self.cfg.adaptive_cost_limit), self.n_channels
            )),
            "effective_risk_success_threshold": float(effective_threshold_before),
            "effective_protection_repetitions": int(effective_repetitions_before),
            "supervisor_failure_ema": float(self.supervisor_failure_ema),
            "supervisor_service_ema": float(self.supervisor_service_ema),
            "supervisor_miss_ema": float(self.supervisor_miss_ema),
            "backlogged_packet_opportunities": int(min(self.n_channels, queued_packets)),
            "coverage_packet_opportunities": int(min(self.n_channels, coverage_devices)),
            "risk_admissible_packet_opportunities": int(risk_capacity),
            "queue_packets_before": int(queued_packets), "reward": float(reward),
            **reward_terms,
        }
        self.episode_trace.append(trace)

        self.t += 1
        self.done = self.t >= self.steps_per_ep
        if self.done:
            self._expire_deadlines()
            self._append_queue_trace()
        else:
            self._advance_network()
        info = dict(trace)
        info.update(self.episode_metrics() if self.done else {
            "scheduled_tx_ep": int(self.scheduled_tx_ep),
            "failed_scheduled_tx_ep": int(self.failed_scheduled_tx_ep),
            "risk_admissible_packet_opportunities_ep": int(self.risk_admissible_packet_opportunities_ep),
        })
        return self._make_obs(), float(reward), bool(self.done), info

    # ------------------------------------------------------------------
    # Observation and reporting.
    # ------------------------------------------------------------------
    def _feature_for_candidate(self, idx: int | None) -> list[float]:
        if idx is None:
            return [0.0] * 8
        packet = self._head_packet(int(idx))
        if packet is None:
            return [0.0] * 8
        payload = max(float(packet.bits), 1.0)
        return [
            float(np.clip(len(self.queues[idx]) / max(float(self.cfg.max_queue_packets), 1.0), 0.0, 1.0)),
            float(np.clip(self._packet_age_slots(packet) * self.slot_s * 1e3 / max(float(self.deadline_ms[idx]), 1.0), 0.0, 4.0)),
            float(np.tanh(self._estimated_sinr_db(int(idx)) / 20.0)),
            float(self._risk_success_probability(int(idx), 1)),
            float(self._risk_success_probability(int(idx), 3)),
            float(np.clip(self._estimated_capacity_bits(int(idx), 1) / payload, 0.0, 3.0) / 3.0),
            float(np.clip(self._estimated_distance_m(int(idx)) / max(float(self.cfg.rsu_coverage_m), 1.0), 0.0, 2.0)),
            float(self.device_class[idx] / max(self.n_classes - 1, 1)),
        ]

    def _top_candidates(self, mode: int, priority: bool) -> list[int | None]:
        available = set(int(i) for i in self._base_eligible().tolist())
        result: list[int | None] = []
        for _ in range(self.n_channels):
            idx = self._select_best(int(mode), available, priority=priority)
            result.append(idx)
            if idx is not None:
                available.remove(int(idx))
        return result

    def _make_obs(self) -> np.ndarray:
        queued_total = self._queue_packet_count()
        head_indices = [i for i, q in enumerate(self.queues) if q]
        ages_ms = np.asarray([self._packet_age_slots(self.queues[i][0]) * self.slot_s * 1e3 for i in head_indices], dtype=float)
        if ages_ms.size == 0:
            ages_ms = np.asarray([0.0], dtype=float)
        coverage_frac = float(np.mean(self._in_coverage()))
        battery_frac = float(np.mean(self.battery_j / np.maximum(self.battery_capacity_j, 1e-12)))
        class_backlog = [float(sum(len(self.queues[i]) for i in range(self.n_devices) if int(self.device_class[i]) == c) / max(float(self.n_devices), 1.0)) for c in range(self.n_classes)]
        recent_fail = float(self.failed_scheduled_tx_ep / max(self.scheduled_tx_ep, 1))
        risk_capacity_norm = float(self._risk_admissible_packet_capacity() / max(self.n_channels, 1))
        normal = [self._feature_for_candidate(idx) for idx in self._top_candidates(1, priority=False)]
        priority = [self._feature_for_candidate(idx) for idx in self._top_candidates(2, priority=True)]
        protected = [self._feature_for_candidate(idx) for idx in self._top_candidates(3, priority=True)]
        obs = np.asarray([
            float(np.clip(queued_total / max(float(self.n_devices * self.cfg.max_queue_packets), 1.0), 0.0, 1.0)),
            float(np.clip(np.mean(ages_ms) / max(float(np.max(self.deadline_ms)), 1.0), 0.0, 3.0)),
            float(np.clip(np.max(ages_ms) / max(float(np.min(self.deadline_ms)), 1.0), 0.0, 5.0)),
            coverage_frac, battery_frac, recent_fail, risk_capacity_norm,
            *class_backlog,
            *[value for group in normal for value in group],
            *[value for group in priority for value in group],
            *[value for group in protected for value in group],
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def episode_metrics(self) -> dict:
        self._expire_deadlines()
        duration_s = max(self.steps_per_ep * self.slot_s, 1e-12)
        scheduled = int(self.scheduled_tx_ep)
        successful = int(self.successful_tx_ep)
        failed = int(self.failed_scheduled_tx_ep)
        offered = int(self.offered_packets_ep)
        pending = int(self._queue_packet_count())
        backlog_opps = int(self.backlogged_packet_opportunities_ep)
        coverage_opps = int(self.coverage_packet_opportunities_ep)
        risk_opps = int(self.risk_admissible_packet_opportunities_ep)
        served = self.served_bits_per_class_ep
        fairness = float((served.sum() ** 2) / (self.n_classes * np.sum(served ** 2) + 1e-12)) if served.sum() > 0 else np.nan
        conservation_error = int(
            offered
            - self.delivered_packets_ep
            - self.deadline_missed_packets_ep
            - self.overflow_dropped_packets_ep
            - pending
        )
        cohort_pending = int(sum(
            1 for q in self.queues for packet in q if packet.in_measurement_cohort
        ))
        cohort_offered = int(self.cohort_offered_packets_ep)
        cohort_conservation_error = int(
            cohort_offered
            - self.cohort_delivered_packets_ep
            - self.cohort_deadline_missed_packets_ep
            - self.cohort_overflow_dropped_packets_ep
            - cohort_pending
        )
        cohort_measurement_s = max(int(self.cfg.measurement_cohort_slots) * self.slot_s, 1e-12)
        return {
            "throughput_mbps": float(self.bits_tx_ep / duration_s / 1e6),
            "avg_delay_ms": float(np.mean(self.delay_samples_ms_ep)) if self.delay_samples_ms_ep else np.nan,
            "interruption_probability": float(failed / scheduled) if scheduled > 0 else np.nan,
            "failed_scheduled_tx": int(failed),
            "scheduled_tx": int(scheduled), "successful_tx": int(successful),
            "n_channels": int(self.n_channels),
            "steps_per_episode": int(self.steps_per_ep),
            "constraint_limit": float(self.cfg.adaptive_cost_limit),
            "constraint_numerator": float(failed - float(self.cfg.adaptive_cost_limit) * scheduled),
            "episode_constraint_residual": float(
                (failed - float(self.cfg.adaptive_cost_limit) * scheduled)
                / max(self.n_channels * self.steps_per_ep, 1)
            ),
            "constraint_ratio_feasible": int(
                scheduled > 0 and failed <= float(self.cfg.adaptive_cost_limit) * scheduled + 1e-12
            ),
            "channel_uses": int(self.channel_uses_ep),
            "resource_blocks_used": int(self.channel_uses_ep),
            "max_resource_blocks_used": int(self.max_resource_blocks_used_ep),
            "resource_budget_violation_count": int(self.resource_budget_violation_count_ep),
            "invalid_action_selection_count": int(self.invalid_action_selection_count_ep),
            "mean_feasible_action_count": float(
                self.action_mask_feasible_count_sum_ep / max(self.action_mask_query_count_ep, 1)
            ),
            "channel_utilization": float(self.channel_uses_ep / max(self.n_channels * self.steps_per_ep, 1)),
            "protected_tx": int(self.protected_tx_ep),
            "protected_primary_tx": int(self.protected_tx_ep),
            "protected_replica_uses": int(self.protected_replica_uses_ep),
            "mean_replica_count": float(
                self.replica_attempts_ep / max(scheduled, 1)
            ),
            "escalated_protection_tx": int(self.escalated_protection_tx_ep),
            "escalated_replica_tx": int(self.escalated_replica_tx_ep),
            "priority_overhead_bits": float(self.priority_overhead_bits_ep),
            "protected_overhead_bits": float(self.protected_overhead_bits_ep),
            "adaptive_gate_activation_rate": float(self.adaptive_gate_activations_ep / max(self.steps_per_ep, 1)),
            "adaptive_protection_activation_rate": float(self.adaptive_protection_activations_ep / max(self.steps_per_ep, 1)),
            "mean_effective_risk_success_threshold": float(self.supervisor_effective_threshold_sum / max(self.supervisor_effective_threshold_count, 1)),
            "mean_effective_protection_repetitions": float(self.supervisor_effective_repetition_sum / max(self.supervisor_effective_repetition_count, 1)),
            "final_supervisor_failure_ema": float(self.supervisor_failure_ema),
            "final_supervisor_service_ema": float(self.supervisor_service_ema),
            "final_supervisor_miss_ema": float(self.supervisor_miss_ema),
            # Diagnostic, repeated-slot opportunity metrics.
            "backlogged_packet_opportunities": backlog_opps,
            "coverage_packet_opportunities": coverage_opps,
            "risk_admissible_packet_opportunities": risk_opps,
            "coverage_feasible_fraction": float(coverage_opps / backlog_opps) if backlog_opps > 0 else np.nan,
            "risk_admissible_fraction": float(risk_opps / coverage_opps) if coverage_opps > 0 else np.nan,
            "risk_admissible_service_rate": float(scheduled / risk_opps) if risk_opps > 0 else np.nan,
            "risk_admissible_success_rate": float(successful / risk_opps) if risk_opps > 0 else np.nan,
            "deferred_risk_admissible": int(self.deferred_risk_admissible_ep),
            "defer_probability": float(self.deferred_risk_admissible_ep / risk_opps) if risk_opps > 0 else np.nan,
            # V29.1 terminal-safe measurement-cohort metrics. Post-window
            # background traffic remains in the environment but is excluded
            # from these counters.
            "measurement_cohort_slots": int(self.cfg.measurement_cohort_slots),
            "arrival_cutoff_slots": int(self.cfg.arrival_cutoff_slots),
            "cohort_offered_packets": int(cohort_offered),
            "cohort_delivered_packets": int(self.cohort_delivered_packets_ep),
            "cohort_deadline_missed_packets": int(self.cohort_deadline_missed_packets_ep),
            "cohort_overflow_dropped_packets": int(self.cohort_overflow_dropped_packets_ep),
            "cohort_pending_packets_end": int(cohort_pending),
            "cohort_packet_conservation_error": int(cohort_conservation_error),
            "cohort_on_time_delivery_ratio": float(self.cohort_delivered_packets_ep / cohort_offered) if cohort_offered > 0 else np.nan,
            "cohort_packet_deadline_miss_ratio": float(self.cohort_deadline_missed_packets_ep / cohort_offered) if cohort_offered > 0 else np.nan,
            "cohort_overflow_drop_ratio": float(self.cohort_overflow_dropped_packets_ep / cohort_offered) if cohort_offered > 0 else np.nan,
            "cohort_pending_packet_ratio": float(cohort_pending / cohort_offered) if cohort_offered > 0 else np.nan,
            "cohort_interruption_probability": float(self.cohort_failed_scheduled_tx_ep / self.cohort_scheduled_tx_ep) if self.cohort_scheduled_tx_ep > 0 else np.nan,
            "cohort_scheduled_tx": int(self.cohort_scheduled_tx_ep),
            "cohort_failed_scheduled_tx": int(self.cohort_failed_scheduled_tx_ep),
            "cohort_successful_tx": int(self.cohort_successful_tx_ep),
            "cohort_avg_delay_ms": float(np.mean(self.cohort_delay_samples_ms_ep)) if self.cohort_delay_samples_ms_ep else np.nan,
            "cohort_delivered_traffic_rate_mbps": float(self.cohort_delivered_bits_ep / cohort_measurement_s / 1e6) if cohort_offered > 0 else np.nan,
            "cohort_safety_offered_packets": int(self.cohort_offered_per_class_ep[0]),
            "cohort_safety_delivered_packets": int(self.cohort_delivered_per_class_ep[0]),
            "cohort_safety_deadline_missed_packets": int(self.cohort_deadline_missed_per_class_ep[0]),
            "cohort_telemetry_offered_packets": int(self.cohort_offered_per_class_ep[1]),
            "cohort_telemetry_delivered_packets": int(self.cohort_delivered_per_class_ep[1]),
            "cohort_telemetry_deadline_missed_packets": int(self.cohort_deadline_missed_per_class_ep[1]),
            "cohort_best_effort_offered_packets": int(self.cohort_offered_per_class_ep[2]),
            "cohort_best_effort_delivered_packets": int(self.cohort_delivered_per_class_ep[2]),
            "cohort_best_effort_deadline_missed_packets": int(self.cohort_deadline_missed_per_class_ep[2]),

            # Primary packet-conservation metrics.
            "offered_packets": int(offered),
            "initial_packets": int(self.initial_packets_ep),
            "arrival_packets": int(self.arrival_packets_ep),
            "delivered_packets": int(self.delivered_packets_ep),
            "delivered_initial_packets": int(self.delivered_initial_packets_ep),
            "delivered_arrival_packets": int(self.delivered_arrival_packets_ep),
            "deadline_missed_packets": int(self.deadline_missed_packets_ep),
            "overflow_dropped_packets": int(self.overflow_dropped_packets_ep),
            "pending_packets_end": int(pending),
            "on_time_delivery_ratio": float(self.delivered_packets_ep / offered) if offered > 0 else np.nan,
            "arrival_delivery_ratio": float(self.delivered_arrival_packets_ep / self.arrival_packets_ep) if self.arrival_packets_ep > 0 else np.nan,
            "warm_start_delivery_ratio": float(self.delivered_initial_packets_ep / self.initial_packets_ep) if self.initial_packets_ep > 0 else np.nan,
            "packet_deadline_miss_ratio": float(self.deadline_missed_packets_ep / offered) if offered > 0 else np.nan,
            "queue_overflow_drop_ratio": float(self.overflow_dropped_packets_ep / offered) if offered > 0 else np.nan,
            "pending_packet_ratio": float(pending / offered) if offered > 0 else np.nan,
            "packet_conservation_error": int(conservation_error),
            "total_arrivals": int(self.arrival_packets_ep),
            "delivered_bits": float(self.bits_tx_ep), "energy_j": float(self.energy_j_ep),
            "energy_efficiency_bpj": float(self.bits_tx_ep / self.energy_j_ep) if self.energy_j_ep > 0 else np.nan,
            "jain_fairness_class": fairness,
            "coverage_fraction": float(np.mean(self._in_coverage())),
            "mean_queue_packets": float(np.mean(self.queue_packets_trace)) if self.queue_packets_trace else 0.0,
            "risk_success_threshold": float(self.cfg.risk_success_threshold),
            "effective_risk_success_threshold": float(self._effective_risk_threshold()),
            "effective_protection_repetitions": int(self._effective_protection_repetitions()),
            "supervisor_failure_ema": float(self.supervisor_failure_ema),
            "supervisor_service_ema": float(self.supervisor_service_ema),
            "supervisor_miss_ema": float(self.supervisor_miss_ema),
            # Evaluation-only stress telemetry. The risk gate remains nominal.
            "stress_profile": str(self.cfg.stress.name),
            "actual_shadowing_std_db": float(self.actual_phy.shadowing_std_db),
            "stale_csi_slots": int(self.cfg.stress.stale_csi_slots),
            "actual_speed_mean_mps": float(self.cfg.speed_mean_mps if self.cfg.stress.actual_speed_mean_mps is None else self.cfg.stress.actual_speed_mean_mps),
            "temporal_channel_correlation": int(self._uses_correlated_channel()),
            "temporal_shadowing_correlation": int(self._uses_correlated_shadowing()),
            "mean_fading_ar1_rho": float(self.fading_rho_sum_ep / max(self.fading_rho_count_ep, 1)),
            "mean_shadowing_ar1_rho": float(self.shadowing_rho_sum_ep / max(self.shadowing_rho_count_ep, 1)),
            "shadowing_decorrelation_distance_m": float(self.cfg.stress.shadowing_decorrelation_distance_m),
            "replica_fading_correlation": float(self.cfg.stress.replica_fading_correlation),
            "replica_common_shadowing": int(bool(self.cfg.stress.replica_common_shadowing)),
            "replica_common_interference": int(bool(self.cfg.stress.replica_common_interference)),
            "pathloss_model": str(self.cfg.phy.pathloss_model),
            "blockage_slot_fraction": float(self.blockage_slot_count_ep / max(self.steps_per_ep, 1)),
            "blocked_replica_fraction": float(self.blocked_replica_attempts_ep / max(self.replica_attempts_ep, 1)),
            "external_interference_replica_fraction": float(self.external_interference_replica_attempts_ep / max(self.replica_attempts_ep, 1)),
        }
