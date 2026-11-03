import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import ITRS, TEME, CartesianRepresentation, EarthLocation

from opticalpass.attitude import (
    dcm_to_quat,
    lvlh_dcm,
    off_nadir_angle_rad,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
    rotate,
    shortest_slew_quat,
    to_body,
)
from opticalpass.frames import (
    Station,
    astropy_altaz,
    ecef_to_enu,
    enu_to_aer,
    enu_to_aer_rates,
    geodetic_to_ecef,
    inertial_velocity_in_itrs_axes,
    teme_to_itrs,
    times_from_epoch,
    topocentric_aer,
)
from opticalpass.orbit import design_mean_elements
from opticalpass.propagators import SGP4Propagator

RNG = np.random.default_rng(42)


def random_quats(n):
    return quat_normalize(RNG.normal(size=(n, 4)))


# ------------------------------------------------------------------------ quaternions

def test_quaternion_identities():
    q = random_quats(50)
    ident = np.array([1.0, 0, 0, 0])
    assert np.allclose(quat_multiply(q, quat_conjugate(q)), ident, atol=1e-12)
    assert np.allclose(quat_multiply(ident, q), q)
    v = RNG.normal(size=(50, 3))
    # Rotation preserves length, and q then conj(q) returns the original vector.
    assert np.allclose(np.linalg.norm(rotate(q, v), axis=1), np.linalg.norm(v, axis=1))
    assert np.allclose(rotate(quat_conjugate(q), rotate(q, v)), v)
    # Composition: rotate by q2 then q1 equals rotate by q1*q2.
    q1, q2 = random_quats(50), random_quats(50)
    assert np.allclose(rotate(quat_multiply(q1, q2), v), rotate(q1, rotate(q2, v)))


def test_known_rotation_is_active():
    q = quat_from_axis_angle([0, 0, 1], math.pi / 2)
    # Active: +X turns into +Y under a +90 deg rotation about Z.
    assert np.allclose(rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-15)


def test_dcm_round_trips():
    q = random_quats(200)
    q = np.where(q[:, :1] < 0, -q, q)
    R = quat_to_dcm(q)
    v = RNG.normal(size=(200, 3))
    assert np.allclose(np.einsum("nij,nj->ni", R, v), rotate(q, v))
    assert np.allclose(dcm_to_quat(R), q, atol=1e-12)
    # 180 deg rotations exercise the non-trace branches of Shepperd's method.
    for axis in np.eye(3):
        qa = quat_from_axis_angle(axis, math.pi)
        assert np.allclose(quat_to_dcm(dcm_to_quat(quat_to_dcm(qa))), quat_to_dcm(qa))


def test_dcm_to_quat_rejects_non_rotation():
    with pytest.raises(ValueError):
        dcm_to_quat(np.diag([1.0, 1.0, -1.0]))


# ---------------------------------------------------------------------------- frames

def test_hand_computed_enu_case():
    # Station at 45 N, 90 E on the ellipsoid. East there is -X_ecef, Up is
    # (0, cos45, sin45). A target 1 km east and 1 km up is at az 90, el 45, 1414.21 m.
    st = Station("t", math.radians(45), math.radians(90), 0.0)
    up = np.array([0, math.cos(math.radians(45)), math.sin(math.radians(45))])
    east = np.array([-1.0, 0, 0])
    target = st.ecef_m + 1000 * up + 1000 * east
    az, el, rng = topocentric_aer(target, st)
    assert math.degrees(az) == pytest.approx(90.0, abs=1e-9)
    assert math.degrees(el) == pytest.approx(45.0, abs=1e-9)
    assert rng == pytest.approx(1000 * math.sqrt(2), abs=1e-6)


def test_geodetic_to_ecef_known_points():
    assert np.allclose(geodetic_to_ecef(0, 0, 0), [6378137.0, 0, 0])
    # Polar radius b = a (1 - f) = 6356752.314245 m.
    assert np.allclose(geodetic_to_ecef(math.pi / 2, 0, 0), [0, 0, 6356752.314245], atol=1e-5)
    # Matches astropy's independent WGS84 implementation at the station.
    loc = EarthLocation.from_geodetic(-16.51 * u.deg, 28.30 * u.deg, 2393 * u.m, "WGS84")
    ours = geodetic_to_ecef(math.radians(28.30), math.radians(-16.51), 2393.0)
    assert np.allclose(ours, [loc.x.value, loc.y.value, loc.z.value], atol=1e-6)


def test_enu_round_trip():
    st = Station("t", 0.3, -0.7, 100.0)
    v = RNG.normal(size=(20, 3))
    enu = ecef_to_enu(v, st)
    assert np.allclose(enu @ st.enu_matrix, v)


def test_teme_itrs_round_trip(example_config):
    sgp = SGP4Propagator(design_mean_elements(example_config.orbit, example_config.epoch))
    t = np.array([0.0, 1234.5, 86400.0])
    s = sgp.propagate(t)
    r_itrs, _ = teme_to_itrs(sgp.epoch, t, s.r_m)
    tt = times_from_epoch(sgp.epoch, t)
    back = ITRS(CartesianRepresentation(r_itrs.T * u.m), obstime=tt).transform_to(TEME(obstime=tt))
    assert np.allclose(back.cartesian.xyz.to_value(u.m).T, s.r_m, atol=1e-6)


def test_itrs_velocity_consistent_with_inertial_speed(example_config):
    sgp = SGP4Propagator(design_mean_elements(example_config.orbit, example_config.epoch))
    t = np.linspace(0, 6000, 7)
    s = sgp.propagate(t)
    r, v = teme_to_itrs(sgp.epoch, t, s.r_m, s.v_m_s)
    v_in = inertial_velocity_in_itrs_axes(r, v)
    # TEME is quasi-inertial, so |v_inertial| from ITRS + omega x r should match |v_TEME|.
    # 1 mm/s tolerance covers precession/nutation rates and polar motion.
    assert np.allclose(np.linalg.norm(v_in, axis=1), np.linalg.norm(s.v_m_s, axis=1), atol=1e-3)


@pytest.mark.parametrize("lat_deg", [0.0, 90.0])
def test_overhead_gives_zenith_and_zero_off_nadir(lat_deg):
    st = Station("t", math.radians(lat_deg), 0.2, 0.0)
    # At the equator and pole, geodetic up and the geocentric radial coincide.
    r_sat = st.ecef_m + 500e3 * st.enu_matrix[2]
    az, el, rng = topocentric_aer(r_sat, st)
    assert math.degrees(el) == pytest.approx(90.0, abs=1e-9)
    assert rng == pytest.approx(500e3)
    # Any velocity perpendicular to r defines a valid LVLH frame.
    v = np.cross(r_sat, [0.3, 1.0, 0.1])
    los_body = to_body(st.ecef_m - r_sat, r_sat, v)
    assert math.degrees(off_nadir_angle_rad(los_body)) == pytest.approx(0.0, abs=1e-6)


def test_overhead_mid_latitude_off_nadir_is_latitude_difference():
    # At 28.3 N, a satellite on the local vertical (geodetic) is offset from the
    # geocentric radial by the difference between geodetic and geocentric latitude
    # (scaled by the altitude geometry), so off-nadir is small but not zero.
    st = Station("t", math.radians(28.3), 0.0, 0.0)
    r_sat = st.ecef_m + 500e3 * st.enu_matrix[2]
    _, el, _ = topocentric_aer(r_sat, st)
    assert math.degrees(el) == pytest.approx(90.0, abs=1e-9)
    los_body = to_body(st.ecef_m - r_sat, r_sat, np.cross(r_sat, [0, 0, 1.0]))
    assert 0.0 < math.degrees(off_nadir_angle_rad(los_body)) < 0.2


def test_handwritten_chain_matches_astropy_altaz(example_config):
    sgp = SGP4Propagator(design_mean_elements(example_config.orbit, example_config.epoch))
    st = Station.from_config(example_config.station)
    t = np.arange(0, 86400, 60.0)
    s = sgp.propagate(t)
    r, _ = teme_to_itrs(sgp.epoch, t, s.r_m)
    az, el, rng = topocentric_aer(r, st)
    az2, el2, rng2 = astropy_altaz(sgp.epoch, t, r, st)
    vis = el > 0
    daz = np.angle(np.exp(1j * (az - az2)))
    # Same geometry, two implementations: agreement to 1e-9 rad and 1 mm.
    assert np.max(np.abs(el - el2)) < 1e-9
    assert np.max(np.abs(daz[vis])) < 1e-9
    assert np.max(np.abs(rng - rng2)) < 1e-3


def test_aer_rates_match_finite_difference():
    p0 = np.array([200e3, 400e3, 300e3])
    vel = np.array([-5e3, 3e3, 1e3])
    dt = 1e-3
    az_r, el_r, rr = enu_to_aer_rates(p0, vel)
    a1, e1, r1 = enu_to_aer(p0 - vel * dt)
    a2, e2, r2 = enu_to_aer(p0 + vel * dt)
    assert az_r == pytest.approx((a2 - a1) / (2 * dt), rel=1e-6)
    assert el_r == pytest.approx((e2 - e1) / (2 * dt), rel=1e-6)
    assert rr == pytest.approx((r2 - r1) / (2 * dt), rel=1e-6)


def test_lvlh_axes_and_slew():
    r = np.array([7000e3, 0, 0])
    v = np.array([0, 7.5e3, 0])
    M = lvlh_dcm(r, v)
    assert np.allclose(M[:, 2], [-1, 0, 0])   # Z to nadir
    assert np.allclose(M[:, 1], [0, 0, -1])   # Y opposite orbit normal (+Z here)
    assert np.allclose(M[:, 0], [0, 1, 0])    # X along velocity
    assert np.allclose(M.T @ M, np.eye(3))
    los = RNG.normal(size=(30, 3))
    q = shortest_slew_quat(los)
    unit = los / np.linalg.norm(los, axis=1, keepdims=True)
    assert np.allclose(rotate(q, np.tile([0, 0, 1.0], (30, 1))), unit)
    assert np.allclose(2 * np.arccos(np.clip(q[:, 0], -1, 1)), off_nadir_angle_rad(los))
    assert np.allclose(shortest_slew_quat([0, 0, 1.0]), [1, 0, 0, 0])
    assert np.allclose(rotate(shortest_slew_quat([0, 0, -1.0]), [0, 0, 1.0]), [0, 0, -1])
