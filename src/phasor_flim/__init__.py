"""Phasor FLIM explorer.

Pipeline fit-free per dati FLIM (Fluorescence Lifetime Imaging Microscopy):
- generazione di dati sintetici con ground truth (`synth`)
- trasformata fasoriale (`phasor`)
- unmixing grafico delle frazioni di specie (`unmix`)
- baseline col fitting multi-esponenziale e metriche (`baseline`, `metrics`)
"""

from phasor_flim.phasor import phasor_image, phasor_of_decay, single_lifetime_phasor
from phasor_flim.synth import SyntheticFLIM, make_synthetic_flim

__all__ = [
    "SyntheticFLIM",
    "make_synthetic_flim",
    "phasor_image",
    "phasor_of_decay",
    "single_lifetime_phasor",
]
