"""Baseline: fraction estimation by fitting the decay (full curve).

With known lifetimes, the decay profile of each pixel is a non-negative linear
combination of the basis curves of the species:

    y(t) ≈ sum_k  c_k * g_k(t)          c_k >= 0

solved by non-negative least squares (NNLS). The photon fractions are the
normalized c_k. This is the "classical" curve-fitting method, used as a comparison
for fit-free phasor unmixing: both use the same prior knowledge (the lifetimes),
so the comparison isolates the estimation *method*, not the available information.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def fit_fractions(
    decay: np.ndarray,
    t: np.ndarray,
    lifetimes: np.ndarray,
    max_pixels: int | None = None,
) -> np.ndarray:
    """Estimate per-pixel fractions by NNLS on the full decay.

    Parameters
    ----------
    decay : (H, W, T) FLIM stack.
    t : (T,) time axis (ns).
    lifetimes : (K,) known lifetimes.
    max_pixels : if set, process only the first `max_pixels` pixels (debug/speed).

    Returns
    -------
    fractions : (H, W, K) with NaN on zero-intensity pixels.
    """
    decay = np.asarray(decay, dtype=float)
    t = np.asarray(t, dtype=float)
    lifetimes = np.asarray(lifetimes, dtype=float)
    h, w, _ = decay.shape
    k = lifetimes.size

    basis = np.exp(-t[:, None] / lifetimes[None, :])  # (T, K)
    basis /= basis.sum(axis=0, keepdims=True)

    flat = decay.reshape(-1, decay.shape[-1])  # (N, T)
    n = flat.shape[0]
    limit = n if max_pixels is None else min(max_pixels, n)

    out = np.full((n, k), np.nan)
    for i in range(limit):
        y = flat[i]
        if y.sum() <= 0:
            continue
        c, _ = nnls(basis, y)
        ssum = c.sum()
        out[i] = c / ssum if ssum > 0 else np.full(k, np.nan)

    return out.reshape(h, w, k)
