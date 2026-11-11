"""Generation of synthetic FLIM data with known ground truth.

Time-domain model: each pixel is a mixture of K fluorescent species, each with
lifetime tau_k. The intensity decay of the pixel is

    I(t) = A * sum_k  f_k * exp(-t / tau_k)

where f_k is the *photon fraction* of species k (sum_k f_k = 1) and A the total
amplitude (number of photons). Poisson noise is added, which is realistic for
photon counting in FLIM.

The spatial fraction maps are the ground truth used to evaluate phasor unmixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticFLIM:
    """Container for a synthetic FLIM dataset.

    Attributes
    ----------
    decay : np.ndarray, shape (H, W, T)
        Time-resolved stack (photon counts per time bin).
    t : np.ndarray, shape (T,)
        Time axis in nanoseconds.
    fractions : np.ndarray, shape (H, W, K)
        Ground-truth photon fraction maps (sum to 1 along K).
    lifetimes : np.ndarray, shape (K,)
        Species lifetimes in nanoseconds.
    freq_mhz : float
        Laser repetition frequency in MHz (defines the period).
    """

    decay: np.ndarray
    t: np.ndarray
    fractions: np.ndarray
    lifetimes: np.ndarray
    freq_mhz: float

    @property
    def period_ns(self) -> float:
        return 1e3 / self.freq_mhz

    @property
    def intensity(self) -> np.ndarray:
        """Integrated intensity image (H, W)."""
        return self.decay.sum(axis=-1)


def _smooth_blob(shape: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    """2D Gaussian normalized to [0, 1], used to build the fraction maps."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-(((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2.0 * sigma**2)))
    return g


def make_synthetic_flim(
    shape: tuple[int, int] = (64, 64),
    n_bins: int = 256,
    lifetimes: tuple[float, ...] = (0.6, 2.4, 4.0),
    freq_mhz: float = 80.0,
    total_photons: int = 2000,
    poisson: bool = True,
    seed: int = 0,
) -> SyntheticFLIM:
    """Create a synthetic K-species FLIM dataset with known fraction maps.

    Parameters
    ----------
    shape : (H, W) spatial size.
    n_bins : number of time bins within one laser period.
    lifetimes : species lifetimes (ns). K = len(lifetimes).
    freq_mhz : laser repetition frequency (MHz). 80 MHz -> period 12.5 ns.
    total_photons : mean photon count per pixel (scales the Poisson noise).
    poisson : if True, add Poisson noise (photon counting).
    seed : seed of the random generator.

    Returns
    -------
    SyntheticFLIM
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    lifetimes_arr = np.asarray(lifetimes, dtype=float)
    k = lifetimes_arr.size
    period = 1e3 / freq_mhz  # ns
    t = np.linspace(0.0, period, n_bins, endpoint=False)

    # Fraction maps: one blob per species + a uniform background, then normalized.
    blobs = []
    centers = [
        (h * 0.30, w * 0.30),
        (h * 0.30, w * 0.70),
        (h * 0.70, w * 0.50),
    ]
    for i in range(k):
        c = centers[i % len(centers)]
        blobs.append(_smooth_blob(shape, c, sigma=max(h, w) / 6.0) + 0.15)
    frac = np.stack(blobs, axis=-1)  # (H, W, K)
    frac /= frac.sum(axis=-1, keepdims=True)

    # Normalized decays per species: g_k(t) = exp(-t/tau_k), normalized to sum 1
    # over the bins -> the weights f_k become photon fractions.
    decays = np.exp(-t[None, :] / lifetimes_arr[:, None])  # (K, T)
    decays /= decays.sum(axis=1, keepdims=True)

    # Expected profile per pixel: weighted sum, scaled to the number of photons.
    expected = total_photons * np.einsum("hwk,kt->hwt", frac, decays)

    if poisson:
        decay = rng.poisson(expected).astype(float)
    else:
        decay = expected

    return SyntheticFLIM(
        decay=decay,
        t=t,
        fractions=frac,
        lifetimes=lifetimes_arr,
        freq_mhz=freq_mhz,
    )
