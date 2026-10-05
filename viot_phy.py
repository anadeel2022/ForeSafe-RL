"""Auditable PHY utilities for the V29.1 MobiSafe-PPO reviewer-response extension.

The base model uses log-distance path loss, log-normal shadowing, Rayleigh
power fading, thermal noise, and one orthogonal channel per RSU resource
block.  V29.1 retains the V28 correlated-channel support and additionally allows the environment to supply temporally
correlated shadowing and fading states while preserving the same packet-level
SINR and capacity calculation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
from functools import lru_cache
import math
import numpy as np

C_LIGHT_MPS = 299_792_458.0
SLOT_DURATION_S = 1e-3


@dataclass(frozen=True)
class PHYConfig:
    carrier_frequency_hz: float = 5.9e9
    channel_bandwidth_hz: float = 1.0e6
    noise_psd_dbm_hz: float = -174.0
    receiver_noise_figure_db: float = 7.0
    pathloss_exponent: float = 2.4
    reference_distance_m: float = 1.0
    shadowing_std_db: float = 4.0
    spectral_efficiency_cap_bps_hz: float = 6.0
    risk_quadrature_order: int = 16
    # V29.3 reviewer sensitivity. ``log_distance`` is the frozen manuscript
    # baseline. ``tr37_885_highway_los`` implements 3GPP TR 37.885
    # Table 6.2.1-1 Highway LOS base path loss for V2R/V2V links.
    pathloss_model: str = "log_distance"
    tx_antenna_height_m: float = 1.5
    rx_antenna_height_m: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


def dbm_to_w(dbm: float) -> float:
    return float(10.0 ** ((float(dbm) - 30.0) / 10.0))


def w_to_dbm(power_w: float) -> float:
    return float(10.0 * math.log10(max(float(power_w), 1e-18)) + 30.0)


def free_space_path_loss_db(distance_m: float, carrier_frequency_hz: float) -> float:
    d = max(float(distance_m), 1e-3)
    wavelength_m = C_LIGHT_MPS / float(carrier_frequency_hz)
    return float(20.0 * math.log10(4.0 * math.pi * d / wavelength_m))


def log_distance_path_loss_db(distance_m: float, cfg: PHYConfig) -> float:
    d = max(float(distance_m), float(cfg.reference_distance_m))
    pl0 = free_space_path_loss_db(float(cfg.reference_distance_m), float(cfg.carrier_frequency_hz))
    return float(pl0 + 10.0 * float(cfg.pathloss_exponent) * math.log10(d / float(cfg.reference_distance_m)))


def tr37_885_highway_los_path_loss_db(distance_m: float, cfg: PHYConfig) -> float:
    """3GPP TR 37.885 Table 6.2.1-1 Highway LOS base path loss.

    PL[dB] = 32.4 + 20 log10(d_3D[m]) + 20 log10(f_c[GHz]).
    TR 37.885 states that the V2V path-loss equation is reused for V2R.
    The same base equation is listed for NLOSv before the additional
    vehicle-blockage loss specified by the TR. V29.3 intentionally keeps the
    manuscript's mild-minus blockage process separate, so this sensitivity is
    a standards-based path-loss comparison rather than a complete TR 37.885
    channel implementation. Antenna heights convert horizontal separation to d_3D.
    """
    d2d = max(float(distance_m), 1.0)
    dh = float(cfg.rx_antenna_height_m) - float(cfg.tx_antenna_height_m)
    d3d = max(math.sqrt(d2d * d2d + dh * dh), 1.0)
    fc_ghz = max(float(cfg.carrier_frequency_hz) / 1e9, 1e-9)
    return float(32.4 + 20.0 * math.log10(d3d) + 20.0 * math.log10(fc_ghz))


def path_loss_db(distance_m: float, cfg: PHYConfig) -> float:
    model = str(getattr(cfg, "pathloss_model", "log_distance")).strip().lower()
    if model == "log_distance":
        return log_distance_path_loss_db(distance_m, cfg)
    if model in {"tr37_885_highway_los", "3gpp_tr37_885_highway_los"}:
        return tr37_885_highway_los_path_loss_db(distance_m, cfg)
    raise ValueError(f"Unsupported pathloss_model: {cfg.pathloss_model!r}")


def thermal_noise_w(cfg: PHYConfig, bandwidth_hz: float | None = None) -> float:
    bw = float(cfg.channel_bandwidth_hz if bandwidth_hz is None else bandwidth_hz)
    noise_dbm = float(cfg.noise_psd_dbm_hz + 10.0 * math.log10(max(bw, 1.0)) + cfg.receiver_noise_figure_db)
    return dbm_to_w(noise_dbm)


def mean_received_power_no_shadow_w(tx_power_dbm: float, distance_m: float, cfg: PHYConfig) -> float:
    """Mean received power before log-normal shadowing and Rayleigh fading."""
    return dbm_to_w(float(tx_power_dbm) - path_loss_db(distance_m, cfg))


def sample_rayleigh_power(rng: np.random.Generator) -> float:
    """Unit-mean power gain for a Rayleigh envelope."""
    return float(rng.exponential(scale=1.0))


def sample_received_power_w(
    rng: np.random.Generator,
    tx_power_dbm: float,
    distance_m: float,
    cfg: PHYConfig,
    *,
    shadowing_db: float | None = None,
    fading_power: float | None = None,
) -> tuple[float, float, float]:
    """Return received power with optional externally maintained channel states.

    ``shadowing_db`` and ``fading_power`` are used by the V28 correlated-channel
    profile.  When omitted, the original independent draws are retained.
    """
    shadow_db = float(
        rng.normal(0.0, float(cfg.shadowing_std_db))
        if shadowing_db is None else shadowing_db
    )
    fading = float(
        sample_rayleigh_power(rng)
        if fading_power is None else max(float(fading_power), 0.0)
    )
    base = mean_received_power_no_shadow_w(tx_power_dbm, distance_m, cfg)
    received = float(base * (10.0 ** (shadow_db / 10.0)) * fading)
    return received, shadow_db, fading


def doppler_ar1_correlation(
    speed_mps: float,
    carrier_frequency_hz: float,
    slot_duration_s: float = SLOT_DURATION_S,
) -> float:
    """Doppler-derived AR(1) coefficient for a correlated Rayleigh process.

    A coherence-time approximation ``T_c = 0.423/f_D`` is used and the
    one-slot correlation is ``exp(-Delta t/T_c)``.  The value is clipped below
    one for numerical stability.
    """
    fd = abs(float(speed_mps)) * float(carrier_frequency_hz) / C_LIGHT_MPS
    if fd <= 1e-12:
        return 0.9999
    coherence_time = 0.423 / fd
    rho = math.exp(-max(float(slot_duration_s), 0.0) / max(coherence_time, 1e-12))
    return float(np.clip(rho, 0.0, 0.9999))


def update_correlated_complex_fading(
    rng: np.random.Generator,
    previous: complex,
    rho: float,
) -> complex:
    """Advance a unit-power circular complex Gaussian AR(1) state."""
    r = float(np.clip(rho, 0.0, 0.9999))
    innovation = (rng.normal() + 1j * rng.normal()) / math.sqrt(2.0)
    return complex(r * previous + math.sqrt(max(1.0 - r * r, 0.0)) * innovation)


def required_sinr_linear(
    packet_bits: float,
    cfg: PHYConfig,
    bandwidth_fraction: float = 1.0,
    coding_capacity_factor: float = 1.0,
) -> float:
    """SINR required to finish one packet inside one slot.

    ``coding_capacity_factor`` represents control or coding overhead.  It
    scales usable capacity but does not alter the physical channel bandwidth.
    """
    bw = float(cfg.channel_bandwidth_hz) * float(np.clip(bandwidth_fraction, 1e-9, 1.0))
    usable = max(bw * SLOT_DURATION_S * float(np.clip(coding_capacity_factor, 1e-9, 1.0)), 1e-12)
    eta = max(float(packet_bits), 0.0) / usable
    # Guard against a numerical overflow while retaining a clear infeasible value.
    if eta > 60.0:
        return float("inf")
    return float(2.0 ** eta - 1.0)


def spectral_efficiency_bps_hz(sinr_linear: float, cfg: PHYConfig) -> float:
    value = math.log2(1.0 + max(float(sinr_linear), 0.0))
    return float(min(value, float(cfg.spectral_efficiency_cap_bps_hz)))


def capacity_bits(
    sinr_linear: float,
    cfg: PHYConfig,
    bandwidth_fraction: float = 1.0,
    coding_capacity_factor: float = 1.0,
) -> float:
    bw = float(cfg.channel_bandwidth_hz) * float(np.clip(bandwidth_fraction, 1e-9, 1.0))
    return float(
        spectral_efficiency_bps_hz(sinr_linear, cfg)
        * bw
        * SLOT_DURATION_S
        * float(np.clip(coding_capacity_factor, 1e-9, 1.0))
    )


@lru_cache(maxsize=16)
def _hermgauss(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Hermite nodes/weights for repeated per-slot risk queries."""
    return np.polynomial.hermite.hermgauss(int(order))


def packet_success_probability(
    *,
    tx_power_dbm: float,
    distance_m: float,
    packet_bits: float,
    cfg: PHYConfig,
    bandwidth_fraction: float = 1.0,
    coding_capacity_factor: float = 1.0,
    deterministic_interference_w: float = 0.0,
) -> float:
    """Predict one-shot packet success probability under the realised PHY law.

    Conditional on log-normal shadowing, the Rayleigh desired-channel power is
    exponential.  The conditional packet-success probability is therefore an
    exponential survival term.  Gauss-Hermite quadrature integrates this term
    over the configured zero-mean normal shadowing distribution.  Orthogonal
    RSU channels use ``deterministic_interference_w=0``.
    """
    theta = required_sinr_linear(
        packet_bits,
        cfg,
        bandwidth_fraction=bandwidth_fraction,
        coding_capacity_factor=coding_capacity_factor,
    )
    if not np.isfinite(theta):
        return 0.0
    base_rx = mean_received_power_no_shadow_w(tx_power_dbm, distance_m, cfg)
    if base_rx <= 0.0:
        return 0.0
    noise = thermal_noise_w(cfg, float(cfg.channel_bandwidth_hz) * float(np.clip(bandwidth_fraction, 1e-9, 1.0)))
    denom = max(float(noise) + max(float(deterministic_interference_w), 0.0), 1e-18)
    order = max(6, int(cfg.risk_quadrature_order))
    nodes, weights = _hermgauss(order)
    shadow_db = math.sqrt(2.0) * float(cfg.shadowing_std_db) * nodes
    rx = base_rx * np.power(10.0, shadow_db / 10.0)
    conditional = np.exp(-float(theta) * denom / np.maximum(rx, 1e-30))
    prob = float(np.sum(weights * conditional) / math.sqrt(math.pi))
    return float(np.clip(prob, 0.0, 1.0))


def protected_packet_success_probability_from_replicas(probabilities: Iterable[float]) -> float:
    """Success probability for replica-specific marginal probabilities.

    The analytical admission estimate uses conditional frequency-replica
    independence. The realized V28 channel can share large-scale shadowing
    across replicas while maintaining separate resource-block fading and
    interference states; realized packet success is evaluated directly from
    those resource-block outcomes in ``VIoTEnv``.
    """
    p = np.clip(np.asarray(list(probabilities), dtype=float), 0.0, 1.0)
    if p.size == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - p))


def protected_packet_success_probability(single_success_probability: float, repetitions: int = 2) -> float:
    """Convenience wrapper for equal marginal replica probabilities."""
    p = float(np.clip(single_success_probability, 0.0, 1.0))
    r = max(int(repetitions), 1)
    return protected_packet_success_probability_from_replicas([p] * r)


def link_sinr(
    rng: np.random.Generator,
    desired_tx_power_dbm: float,
    desired_distance_m: float,
    cfg: PHYConfig,
    interferers: Iterable[tuple[float, float]] = (),
    bandwidth_fraction: float = 1.0,
    coding_capacity_factor: float = 1.0,
    *,
    desired_shadowing_db: float | None = None,
    desired_fading_power: float | None = None,
) -> dict:
    """Sample one packet-link realization under the configured PHY model."""
    signal_w, desired_shadow_db, desired_fading = sample_received_power_w(
        rng,
        desired_tx_power_dbm,
        desired_distance_m,
        cfg,
        shadowing_db=desired_shadowing_db,
        fading_power=desired_fading_power,
    )
    interference_w = 0.0
    for tx_power_dbm, distance_m in interferers:
        p_w, _, _ = sample_received_power_w(rng, float(tx_power_dbm), float(distance_m), cfg)
        interference_w += p_w
    noise_w = thermal_noise_w(cfg, float(cfg.channel_bandwidth_hz) * float(np.clip(bandwidth_fraction, 1e-9, 1.0)))
    sinr_linear = float(signal_w / max(noise_w + interference_w, 1e-18))
    return {
        "sinr_linear": sinr_linear,
        "sinr_db": float(10.0 * math.log10(max(sinr_linear, 1e-18))),
        "signal_w": float(signal_w),
        "interference_w": float(interference_w),
        "noise_w": float(noise_w),
        "desired_shadowing_db": float(desired_shadow_db),
        "desired_fading_power": float(desired_fading),
        "capacity_bits": float(
            capacity_bits(
                sinr_linear,
                cfg,
                bandwidth_fraction=bandwidth_fraction,
                coding_capacity_factor=coding_capacity_factor,
            )
        ),
    }


def tx_energy_j(tx_power_dbm: float, packet_bits: float, energy_per_bit_j: float) -> float:
    tx_power_w = dbm_to_w(tx_power_dbm)
    circuit_energy = max(float(packet_bits), 0.0) * max(float(energy_per_bit_j), 0.0)
    return float(tx_power_w * SLOT_DURATION_S + circuit_energy)
