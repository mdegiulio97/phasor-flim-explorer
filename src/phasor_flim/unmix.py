"""Unmixing fasoriale lineare delle frazioni di specie.

Per K specie con phasor di riferimento P_k = (g_k, s_k), il phasor di un pixel
miscela è P = sum_k f_k P_k con il vincolo sum_k f_k = 1. Le frazioni si
ricavano risolvendo il sistema lineare (sovradeterminato per K>3, esatto per
K<=3) ai minimi quadrati:

    | g_1 ... g_K |          | g |
    | s_1 ... s_K | * f  =   | s |
    |  1  ...  1  |          | 1 |

Nessun fitting non lineare: è il vantaggio chiave dell'approccio fasoriale.
"""

from __future__ import annotations

import numpy as np

from phasor_flim.phasor import single_lifetime_phasor


def reference_phasors(
    lifetimes: np.ndarray, freq_mhz: float = 80.0, harmonic: int = 1
) -> np.ndarray:
    """Phasor analitici (K, 2) delle specie pure date le loro vite medie."""
    pts = [single_lifetime_phasor(float(tau), freq_mhz, harmonic) for tau in lifetimes]
    return np.asarray(pts, dtype=float)


def unmix_fractions(
    g: np.ndarray,
    s: np.ndarray,
    refs: np.ndarray,
    nonneg: bool = True,
) -> np.ndarray:
    """Stima le frazioni di fotoni per pixel via unmixing fasoriale lineare.

    Parametri
    ---------
    g, s : array (H, W) delle coordinate fasoriali dei pixel.
    refs : array (K, 2) dei phasor di riferimento delle specie pure.
    nonneg : se True azzera le frazioni negative e rinormalizza (somma a 1).

    Ritorna
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
