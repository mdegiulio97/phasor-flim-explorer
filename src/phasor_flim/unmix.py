"""Linear phasor unmixing of species fractions.

For K species with reference phasors P_k = (g_k, s_k), the phasor of a mixed pixel
is P = sum_k f_k P_k with the constraint sum_k f_k = 1. The fractions are obtained
by solving the linear system (underdetermined for K>3, exact for K<=3) by least
squares:

    | g_1 ... g_K |          | g |
    | s_1 ... s_K | * f  =   | s |
    |  1  ...  1  |          | 1 |

No nonlinear fitting: this is the key advantage of the phasor approach.
"""

from __future__ import annotations

import numpy as np

from phasor_flim.phasor import single_lifetime_phasor


def reference_phasors(
    lifetimes: np.ndarray, freq_mhz: float = 80.0, harmonic: int = 1
) -> np.ndarray:
    """Analytical phasors (K, 2) of the pure species given their lifetimes."""
    pts = [single_lifetime_phasor(float(tau), freq_mhz, harmonic) for tau in lifetimes]
    return np.asarray(pts, dtype=float)


def unmix_fractions(
    g: np.ndarray,
    s: np.ndarray,
    refs: np.ndarray,
    nonneg: bool = True,
) -> np.ndarray:
    """Estimate per-pixel photon fractions by linear phasor unmixing.

    Parameters
    ----------
    g, s : (H, W) arrays of the pixel phasor coordinates.
    refs : (K, 2) array of the reference phasors of the pure species.
    nonneg : if True, set negative fractions to zero and renormalize (sum to 1).

    Returns
    -------
    fractions : array (H, W, K)
    """
    h, w = g.shape
    k = refs.shape[0]

    a = np.vstack([refs[:, 0], refs[:, 1], np.ones(k)])  # (3, K)
    a_pinv = np.linalg.pinv(a)  # (K, 3)

    b = np.stack([g.ravel(), s.ravel(), np.ones(g.size)], axis=0)  # (3, N)
    f = (a_pinv @ b).T  # (N, K)

    if nonneg:
        f = np.clip(f, 0.0, None)
        ssum = f.sum(axis=1, keepdims=True)
        ssum[ssum == 0] = 1.0
        f = f / ssum

    fractions = f.reshape(h, w, k)
    fractions[np.isnan(g)] = np.nan
    return fractions
