import math
from dataclasses import replace

import numpy as np
import pytest

from opticalpass.link import (
    airmass,
    link_for_profile,
    pointing_loss,
    pointing_sensitivity,
    positive_time_s,
    received_power_w,
)


def test_received_power_hand_calculation(example_config):
    term = example_config.terminal
    # Zenith, 500 km: w = 30e-6 * 5e5 = 15 m; D^2/(2 w^2) = 0.16/450.
    p = received_power_w(500e3, math.pi / 2, term)
    expected = 1.0 * 0.5 * 0.8 * math.exp(-2 * (5 / 30) ** 2) * 0.16 / 450.0
    assert float(p) == pytest.approx(expected, rel=1e-12)


def test_inverse_square_with_range(example_config):
    term = replace(example_config.terminal, pointing_error_rad=0.0)
    p1 = received_power_w(500e3, math.pi / 2, term)
    p2 = received_power_w(1000e3, math.pi / 2, term)
    assert float(p1 / p2) == pytest.approx(4.0)


def test_airmass_and_pointing_loss():
    assert airmass(math.pi / 2) == pytest.approx(1.0)
    assert airmass(math.radians(30)) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        airmass(0.0)
    assert pointing_loss(0.0, 30e-6) == 1.0
    # At an error equal to the 1/e^2 half-angle the intensity is e^-2.
    assert pointing_loss(30e-6, 30e-6) == pytest.approx(math.exp(-2))


def test_positive_time_interpolates_crossings():
    t = np.array([0.0, 10.0, 20.0, 30.0])
    m = np.array([-1.0, 1.0, 1.0, -3.0])
    # Crossing at 5 s and at 22.5 s.
    assert positive_time_s(t, m) == pytest.approx(17.5)
    assert positive_time_s(t, -np.ones(4)) == 0.0


def test_volume_and_sensitivity_monotonic(example_config):
    term = example_config.terminal
    t = np.linspace(0, 300, 151)
    el = np.radians(20 + 60 * np.sin(np.pi * t / 300))
    rng = 500e3 / np.sin(el)
    res = link_for_profile(1, t, rng, el, term)
    assert res.data_volume_bits == pytest.approx(res.positive_margin_s * 1e9)
    rows = pointing_sensitivity([(1, t, rng, el)], term, [0, 5e-6, 15e-6, 25e-6], 1.0)
    vols = [r.total_volume_bits for r in rows]
    assert all(a >= b for a, b in zip(vols, vols[1:], strict=False))
    assert rows[-1].pointing_loss_db < rows[0].pointing_loss_db


def test_small_aperture_check(example_config):
    with pytest.raises(ValueError):
        received_power_w(1e3, math.pi / 2, example_config.terminal)
