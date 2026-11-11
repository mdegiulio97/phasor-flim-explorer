"""Phasor transform for FLIM data.

The phasor of a decay I(t) sampled over one laser period is the pair

    g = sum_t I(t) cos(n * omega * t) / sum_t I(t)
    s = sum_t I(t) sin(n * omega * t) / sum_t I(t)

with omega = 2*pi/period and n the harmonic order. For a mono-exponential decay
the point (g, s) falls on the "universal circle" (semicircle centered at (0.5, 0)
with radius 0.5). A mixture of species falls on the linear combination of the
component phasors, weighted by the photon fractions; this property makes phasor
unmixing linear and *fit-free*.
"""

from __future__ import annotations

import numpy as np


def single_lifetime_phasor(tau: float, freq_mhz: float = 80.0, harmonic: int = 1) -> tuple[float, float]:
    """Analytical phasor position of a mono-exponential with lifetime `tau` (ns).

    g = 1 / (1 + (omega*tau)^2),  s = (omega*tau) / (1 + (omega*tau)^2)
    with omega = 2*pi*freq*harmonic.
    """
    omega = 2.0 * np.pi * (freq_mhz * 1e6) * harmonic  # rad/s
    tau_s = tau * 1e-9
    wt = omega * tau_s
    denom = 1.0 + wt**2
    return 1.0 / denom, wt / denom


def phasor_of_decay(
    decay: np.ndarray, t: np.ndarray, freq_mhz: float = 80.0, harmonic: int = 1
) -> tuple[float, float]:
    """Phasor (g, s) of a single 1-D decay sampled on the bins `t` (ns)."""
    decay = np.asarray(decay, dtype=float)
    period = t[-1] - t[0] + (t[1] - t[0])  # period covered by the bins
    omega = 2.0 * np.pi / period
    total = decay.sum()
    if total <= 0:
        return np.nan, np.nan
    g = np.sum(decay * np.cos(harmonic * omega * t)) / total
    s = np.sum(decay * np.sin(harmonic * omega * t)) / total
    return float(g), float(s)


def phasor_image(
    decay: np.ndarray, t: np.ndarray, freq_mhz: float = 80.0, harmonic: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Phasor maps (g, s) for a FLIM stack (H, W, T).

    Returns two (H, W) arrays. Zero-intensity pixels become NaN.
    """
    decay = np.asarray(decay, dtype=float)
    t = np.asarray(t, dtype=float)
    period = t[-1] - t[0] + (t[1] - t[0])
    omega = 2.0 * np.pi / period
    cos_k = np.cos(harmonic * omega * t)
    sin_k = np.sin(harmonic * omega * t)

    total = decay.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = decay @ cos_k / total
        s = decay @ sin_k / total
    g[total <= 0] = np.nan
    s[total <= 0] = np.nan
    return g, s


def universal_circle(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of the universal semicircle for drawing the phasor plot."""
    phi = np.linspace(0.0, np.pi, n)
    return 0.5 + 0.5 * np.cos(phi), 0.5 * np.sin(phi)
