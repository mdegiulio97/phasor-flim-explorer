"""Tests for phasor unmixing and for the linearity of the mixture."""

import numpy as np

from phasor_flim.phasor import phasor_image
from phasor_flim.synth import make_synthetic_flim
from phasor_flim.unmix import reference_phasors, unmix_fractions
from phasor_flim.metrics import fraction_mae


def test_mixture_phasor_is_linear():
    # The (discrete) phasor of a mixture = average of the (discrete) phasors of
    # the components, weighted by the photon fractions. This is the property that
    # makes unmixing linear. It is checked on the discrete phasors of the pure
    # components computed from the same sampled signal (not on the continuous
    # analytical formula, which differs because of the discretization).
    freq = 80.0
    period = 1e3 / freq
    t = np.linspace(0.0, period, 256, endpoint=False)
    f1 = 0.3

    g1 = np.exp(-t / 0.8)
    g1 /= g1.sum()
    g2 = np.exp(-t / 3.5)
    g2 /= g2.sum()

    p1 = np.array(phasor_image(g1.reshape(1, 1, -1), t, freq)).ravel()
    p2 = np.array(phasor_image(g2.reshape(1, 1, -1), t, freq)).ravel()
    mix = f1 * p1 + (1 - f1) * p2

    decay = (f1 * g1 + (1 - f1) * g2).reshape(1, 1, -1)
    g, s = phasor_image(decay, t, freq)
    assert abs(g[0, 0] - mix[0]) < 1e-9
    assert abs(s[0, 0] - mix[1]) < 1e-9


def test_unmix_recovers_fractions_noise_free():
    flim = make_synthetic_flim(shape=(48, 48), poisson=False, seed=1)
    g, s = phasor_image(flim.decay, flim.t, flim.freq_mhz)
    refs = reference_phasors(flim.lifetimes, flim.freq_mhz)
    est = unmix_fractions(g, s, refs)
    mae = fraction_mae(est, flim.fractions)
    assert mae < 0.05


def test_unmix_robust_under_poisson_noise():
    flim = make_synthetic_flim(shape=(48, 48), total_photons=1500, poisson=True, seed=2)
    g, s = phasor_image(flim.decay, flim.t, flim.freq_mhz)
    refs = reference_phasors(flim.lifetimes, flim.freq_mhz)
    est = unmix_fractions(g, s, refs)
    mae = fraction_mae(est, flim.fractions)
    # With realistic noise the error grows but stays bounded.
    assert mae < 0.15
