import math

import numpy as np
import pytest

from opticalpass.constants import MU_EARTH_M3_S2, SECONDS_PER_DAY
from opticalpass.orbit import design_mean_elements, j2_raan_rate_rad_s, rv_to_raan_rad
from opticalpass.propagators import NumericalJ2Propagator, SGP4Propagator, build_propagator


@pytest.fixture(scope="module")
def elements(example_config):
    return design_mean_elements(example_config.orbit, example_config.epoch)


@pytest.fixture(scope="module")
def sgp4(elements):
    return SGP4Propagator(elements)


def test_sgp4_altitude_near_500km(sgp4):
    s = sgp4.propagate(np.arange(0, 86400, 30.0))
    alt = np.linalg.norm(s.r_m, axis=1) - 6378137.0
    # Near-circular orbit at a nominal 500 km; SGP4 periodic terms and Earth oblateness
    # in the radius definition give a few km of spread.
    assert 485e3 < alt.min() and alt.max() < 515e3


def test_two_body_conserves_energy_and_momentum(sgp4):
    T = 2 * 86400.0
    prop = NumericalJ2Propagator.from_propagator(sgp4, T, include_j2=False)
    s = prop.propagate(np.linspace(0, T, 2000))
    r = np.linalg.norm(s.r_m, axis=1)
    energy = 0.5 * np.sum(s.v_m_s**2, axis=1) - MU_EARTH_M3_S2 / r
    h = np.cross(s.r_m, s.v_m_s)
    # Tolerances: relative 1e-9 on energy and on each angular momentum component
    # magnitude, far looser than DOP853 at rtol 1e-11 should need.
    assert np.max(np.abs(energy / energy[0] - 1)) < 1e-9
    assert np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0]) < 1e-9


def test_two_body_period_matches_analytic(sgp4):
    s0 = sgp4.propagate(0.0)
    r0, v0 = s0.r_m[0], s0.v_m_s[0]
    energy = 0.5 * v0 @ v0 - MU_EARTH_M3_S2 / np.linalg.norm(r0)
    a = -MU_EARTH_M3_S2 / (2 * energy)
    period = 2 * math.pi * math.sqrt(a**3 / MU_EARTH_M3_S2)
    prop = NumericalJ2Propagator.from_propagator(sgp4, 1.01 * period, include_j2=False)
    s = prop.propagate(period)
    # After one analytic period the state returns to the start to within 1 mm.
    assert np.linalg.norm(s.r_m[0] - r0) < 1e-3


def test_j2_raan_drift_matches_secular_rate(sgp4, elements):
    T = 5 * 86400.0
    prop = NumericalJ2Propagator.from_propagator(sgp4, T, include_j2=True)
    t = np.arange(0, T, 60.0)
    s = prop.propagate(t)
    raan = np.unwrap(rv_to_raan_rad(s.r_m, s.v_m_s))
    rate_fit = np.polyfit(t, raan, 1)[0]
    rate_analytic = j2_raan_rate_rad_s(elements.semi_major_axis_m, elements.eccentricity,
                                       elements.inclination_rad)
    deg_day = math.degrees(rate_fit) * SECONDS_PER_DAY
    assert deg_day == pytest.approx(0.9856, rel=0.02)
    # The integrator starts from an osculating state, whose semi-major axis differs
    # from the mean value by short-period J2 terms, so allow 2 %.
    assert rate_fit == pytest.approx(rate_analytic, rel=0.02)


def test_out_of_span_raises(sgp4):
    prop = NumericalJ2Propagator.from_propagator(sgp4, 1000.0)
    with pytest.raises(ValueError):
        prop.propagate([0.0, 2000.0])


def test_bad_inputs(sgp4, elements):
    with pytest.raises(ValueError):
        sgp4.propagate([0.0, float("nan")])
    with pytest.raises(ValueError):
        build_propagator("kepler", elements, 100.0)
