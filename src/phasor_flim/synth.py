"""Generazione di dati FLIM sintetici con ground truth noto.

Modello time-domain: ogni pixel è una miscela di K specie fluorescenti, ognuna
con vita media (lifetime) tau_k. Il decadimento di intensità del pixel è

    I(t) = A * sum_k  f_k * exp(-t / tau_k)

dove f_k è la *frazione di fotoni* della specie k (sum_k f_k = 1) e A l'ampiezza
totale (numero di fotoni). Si aggiunge rumore Poissoniano, realistico per il
conteggio di fotoni in FLIM.

Le mappe di frazione spaziali costituiscono la ground truth per valutare
l'unmixing fasoriale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticFLIM:
    """Contenitore di un dataset FLIM sintetico.

    Attributi
    ---------
    decay : np.ndarray, shape (H, W, T)
        Stack risolto nel tempo (conteggi di fotoni per bin temporale).
    t : np.ndarray, shape (T,)
        Asse dei tempi in nanosecondi.
    fractions : np.ndarray, shape (H, W, K)
        Mappe di frazione di fotoni ground-truth (sommano a 1 lungo K).
    lifetimes : np.ndarray, shape (K,)
        Vite medie delle specie in nanosecondi.
    freq_mhz : float
        Frequenza di ripetizione del laser in MHz (definisce il periodo).
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
        """Immagine di intensità integrata (H, W)."""
        return self.decay.sum(axis=-1)


def _smooth_blob(shape: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    """Gaussiana 2D normalizzata a [0, 1] usata per comporre le mappe di frazione."""
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
    """Crea un dataset FLIM sintetico a K specie con mappe di frazione note.

    Parametri
    ---------
    shape : (H, W) dimensione spaziale.
    n_bins : numero di bin temporali entro un periodo del laser.
    lifetimes : vite medie (ns) delle specie. K = len(lifetimes).
    freq_mhz : frequenza di ripetizione del laser (MHz). 80 MHz -> periodo 12.5 ns.
    total_photons : conteggio medio di fotoni per pixel (scala il rumore Poisson).
    poisson : se True aggiunge rumore Poissoniano (conteggio fotoni).
    seed : seme del generatore casuale.

    Ritorna
    -------
    SyntheticFLIM
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    lifetimes_arr = np.asarray(lifetimes, dtype=float)
    k = lifetimes_arr.size
    period = 1e3 / freq_mhz  # ns
    t = np.linspace(0.0, period, n_bins, endpoint=False)

    # Mappe di frazione: una blob per specie + un fondo uniforme, poi normalizzate.
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

    # Decadimenti normalizzati per specie: g_k(t) = exp(-t/tau_k), normalizzati a
    # somma 1 sui bin -> i pesi f_k diventano frazioni di fotoni.
    decays = np.exp(-t[None, :] / lifetimes_arr[:, None])  # (K, T)
    decays /= decays.sum(axis=1, keepdims=True)

    # Profilo atteso per pixel: somma pesata, scalata al numero di fotoni.
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
