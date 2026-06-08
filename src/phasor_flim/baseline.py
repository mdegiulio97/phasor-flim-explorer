"""Baseline: stima delle frazioni via fitting del decadimento (curva intera).

Con le vite medie note, il profilo di decadimento di ogni pixel è una
combinazione lineare non negativa delle curve di base delle specie:

    y(t) ≈ sum_k  c_k * g_k(t)          c_k >= 0

risolta ai minimi quadrati non negativi (NNLS). Le frazioni di fotoni sono
c_k normalizzate. È il metodo "classico" basato sul fitting della curva, usato
come confronto rispetto all'unmixing fasoriale fit-free: entrambi sfruttano la
stessa conoscenza a priori (le vite medie), così il confronto isola il *metodo*
di stima, non l'informazione disponibile.
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
    """Stima le frazioni per pixel via NNLS sul decadimento completo.

    Parametri
    ---------
    decay : (H, W, T) stack FLIM.
    t : (T,) asse dei tempi (ns).
    lifetimes : (K,) vite medie note.
    max_pixels : se impostato, processa solo i primi `max_pixels` (debug/velocità).

    Ritorna
    -------
    fractions : (H, W, K) con NaN sui pixel a intensità nulla.
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
