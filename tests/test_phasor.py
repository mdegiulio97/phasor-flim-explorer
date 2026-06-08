"""Test di correttezza della trasformata fasoriale."""

import numpy as np

from phasor_flim.phasor import (
    phasor_of_decay,
    single_lifetime_phasor,
)


def test_single_lifetime_lies_on_universal_circle():
    # Ogni mono-esponenziale deve cadere sul semicerchio centro (0.5, 0) raggio 0.5.
    for tau in [0.5, 1.0, 2.0, 4.0]:
        g, s = single_lifetime_phasor(tau, freq_mhz=80.0)
        r = np.hypot(g - 0.5, s - 0.0)
        assert abs(r - 0.5) < 1e-9


def test_longer_lifetime_has_larger_phase():
    # Vite più lunghe -> angolo di fase maggiore sul phasor plot.
    _, s_short = single_lifetime_phasor(0.5)
    _, s_long = single_lifetime_phasor(4.0)
    angle_short = np.arctan2(s_short, single_lifetime_phasor(0.5)[0])
    angle_long = np.arctan2(s_long, single_lifetime_phasor(4.0)[0])
    assert angle_long > angle_short


def test_discrete_phasor_near_universal_circle():
    # Il phasor discreto di un decadimento campionato deve stare vicino al cerchio.
    freq = 80.0
    period = 1e3 / freq
    t = np.linspace(0.0, period, 512, endpoint=False)
    for tau in [1.0, 2.0, 3.0]:
        decay = np.exp(-t / tau)
        g, s = phasor_of_decay(decay, t, freq_mhz=freq)
        r = np.hypot(g - 0.5, s)
        assert abs(r - 0.5) < 0.05


def test_zero_intensity_returns_nan():
    t = np.linspace(0, 12.5, 256, endpoint=False)
    g, s = phasor_of_decay(np.zeros_like(t), t)
    assert np.isnan(g) and np.isnan(s)
