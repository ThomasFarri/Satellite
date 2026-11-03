import math

import astropy.units as u
import numpy as np
import pytest

from opticalpass.frames import Station, teme_to_gcrs, times_from_epoch
from opticalpass.geometry import Tracker
from opticalpass.orbit import design_mean_elements
from opticalpass.passes import find_passes
from opticalpass.pointing import point_ahead, pointing_profile
from opticalpass.propagators import build_propagator


@pytest.fixture(scope="module")
def setup(example_config):
    c = example_config
    prop = build_propagator("sgp4", design_mean_elements(c.orbit, c.epoch), 2 * 86400.0)
    tr = Tracker(prop, Station.from_config(c.station))
    from dataclasses import replace
    cons = replace(c.constraints, night_only=False, min_sun_separation_rad=0.0)
    passes = find_passes(tr, cons, 0.0, 2 * 86400.0)
    # Use the highest pass: it exercises the largest rates.
    p = max(passes, key=lambda x: x.max_elevation_rad)
    return tr, p, pointing_profile(tr, p, step_s=1.0)


def test_point_ahead_is_order_50_microradians(setup):
    _, _, pr = setup
    assert 15e-6 < pr.point_ahead_rad.min() and pr.point_ahead_rad.max() < 60e-6
    # At closest approach nearly all of the ~7.6 km/s orbital speed is perpendicular.
    k = int(np.argmax(pr.el_rad))
    assert pr.point_ahead_rad[k] == pytest.approx(2 * 7.6e3 / 299792458.0, rel=0.05)


def test_point_ahead_direction_perpendicular_to_los(setup):
    _, _, pr = setup
    dots = np.sum(pr.point_ahead_dir_body * pr.los_body, axis=1)
    assert np.max(np.abs(dots)) < 1e-12
    assert np.allclose(np.linalg.norm(pr.point_ahead_dir_body, axis=1), 1.0)


def test_point_ahead_function_simple_case():
    paa, vp, d = point_ahead(np.array([[0, 0, 1.0]]), np.array([[3000.0, 0, 500.0]]))
    assert vp[0] == pytest.approx(3000.0)
    assert paa[0] == pytest.approx(2 * 3000.0 / 299792458.0)
    assert np.allclose(d[0], [1, 0, 0])


def test_v_perp_matches_astropy_gcrs(setup):
    """Independent route: satellite GCRS state and station GCRS velocity from astropy."""
    tr, _, pr = setup
    idx = np.linspace(0, len(pr.t_s) - 1, 9).astype(int)
    t = pr.t_s[idx]
    s = tr.propagator.propagate(t)
    r_sat, v_sat = teme_to_gcrs(tr.epoch, t, s.r_m, s.v_m_s)
    pos, vel = tr.station.earth_location.get_gcrs_posvel(times_from_epoch(tr.epoch, t))
    r_sta = pos.xyz.to_value(u.m).T
    v_sta = vel.xyz.to_value(u.m / u.s).T
    los = r_sta - r_sat
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    paa, _, _ = point_ahead(los, v_sta - v_sat)
    # 0.1 % covers the nominal-rotation-rate and polar-motion simplifications.
    assert np.allclose(pr.point_ahead_rad[idx], paa, rtol=1e-3)


def test_off_nadir_matches_spherical_geometry(setup):
    tr, _, pr = setup
    # sin(off-nadir) = (|r_station| / |r_sat|) cos(el) for a spherical Earth.
    g = tr.geometry(pr.t_s)
    ratio = np.linalg.norm(tr.station.ecef_m) / np.linalg.norm(g.r_itrs_m, axis=1)
    expected = np.arcsin(ratio * np.cos(pr.el_rad))
    # Oblateness (geodetic vs geocentric vertical) makes this approximate: 0.3 deg.
    assert np.max(np.abs(pr.off_nadir_rad - expected)) < math.radians(0.3)


def test_rates_match_numerical_derivative(setup):
    _, _, pr = setup
    el_num = np.gradient(pr.el_rad, pr.t_s)
    az_num = np.gradient(np.unwrap(pr.az_rad), pr.t_s)
    inner = slice(2, -2)
    assert np.allclose(pr.el_rate_rad_s[inner], el_num[inner], atol=math.radians(0.02))
    # Azimuth rate changes fast near zenith, so compare where it is moderate.
    mask = np.abs(pr.az_rate_rad_s) < math.radians(1.0)
    mask[:2] = mask[-2:] = False
    assert np.allclose(pr.az_rate_rad_s[mask], az_num[mask], atol=math.radians(0.01))
    rr_num = np.gradient(pr.range_m, pr.t_s)
    assert np.allclose(pr.range_rate_m_s[inner], rr_num[inner], atol=5.0)
