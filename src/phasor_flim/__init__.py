"""Phasor FLIM explorer.

Fit-free pipeline for FLIM (Fluorescence Lifetime Imaging Microscopy) data:
- generation of synthetic data with ground truth (`synth`)
- phasor transform (`phasor`)
- graphical unmixing of species fractions (`unmix`)
- multi-exponential fitting baseline and metrics (`baseline`, `metrics`)
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
