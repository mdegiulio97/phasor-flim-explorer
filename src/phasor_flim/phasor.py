"""Trasformata fasoriale per dati FLIM.

Il phasor di un decadimento I(t) campionato su un periodo del laser è la coppia

    g = sum_t I(t) cos(n * omega * t) / sum_t I(t)
    s = sum_t I(t) sin(n * omega * t) / sum_t I(t)

con omega = 2*pi/period e n l'ordine armonico. Per un decadimento mono-esponenziale
il punto (g, s) cade sul "cerchio universale" (semicerchio di centro (0.5, 0) e
raggio 0.5). Una miscela di specie cade nella combinazione lineare dei phasor
delle componenti, pesata dalle frazioni di fotoni — proprietà che rende l'unmixing
fasoriale lineare e *fit-free*.
"""

from __future__ import annotations

import numpy as np


def single_lifetime_phasor(tau: float, freq_mhz: float = 80.0, harmonic: int = 1) -> tuple[float, float]:
    """Posizione fasoriale analitica di un mono-esponenziale di vita `tau` (ns).

    g = 1 / (1 + (omega*tau)^2),  s = (omega*tau) / (1 + (omega*tau)^2)
    con omega = 2*pi*freq*harmonic.
    """
    omega = 2.0 * np.pi * (freq_mhz * 1e6) * harmonic  # rad/s
    tau_s = tau * 1e-9
    wt = omega * tau_s
    denom = 1.0 + wt**2
    return 1.0 / denom, wt / denom


def phasor_of_decay(
    decay: np.ndarray, t: np.ndarray, freq_mhz: float = 80.0, harmonic: int = 1
) -> tuple[float, float]:
    """Phasor (g, s) di un singolo decadimento 1-D campionato sui bin `t` (ns)."""
    decay = np.asarray(decay, dtype=float)
    period = t[-1] - t[0] + (t[1] - t[0])  # periodo coperto dai bin
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
    """Mappe fasoriali (g, s) per uno stack FLIM (H, W, T).

    Ritorna due array (H, W). I pixel a intensità nulla diventano NaN.
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
    """Coordinate del semicerchio universale per il disegno del phasor plot."""
    phi = np.linspace(0.0, np.pi, n)
    return 0.5 + 0.5 * np.cos(phi), 0.5 * np.sin(phi)
