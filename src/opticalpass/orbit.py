"""Orbit design: sun-synchronous inclination, RAAN for a given LTAN, mean elements."""

from __future__ import annotations

import math
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import TEME, get_sun
from astropy.time import Time

from opticalpass.config import OrbitConfig
from opticalpass.constants import (
    J2_EARTH,
    MU_EARTH_M3_S2,
    R_EARTH_EQ_M,
    SECONDS_PER_DAY,
    SUN_SYNC_RATE_RAD_S,
)


@dataclass(frozen=True)
class MeanElements:
    """Mean orbital elements in the TEME frame at ``epoch``. SI units and radians."""

    epoch: Time
    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    mean_anomaly_rad: float
    bstar_per_earth_radii: float = 0.0

    @property
    def mean_motion_rad_s(self) -> float:
        return math.sqrt(MU_EARTH_M3_S2 / self.semi_major_axis_m**3)

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.mean_motion_rad_s


def j2_raan_rate_rad_s(a_m: float, e: float, i_rad: float) -> float:
    """Secular RAAN drift from J2 [rad/s].

    dOmega/dt = -3/2 n J2 (Re/p)^2 cos(i), with p = a(1 - e^2).
    """
    _check_orbit(a_m, e)
    n = math.sqrt(MU_EARTH_M3_S2 / a_m**3)
    p = a_m * (1.0 - e**2)
    return -1.5 * n * J2_EARTH * (R_EARTH_EQ_M / p) ** 2 * math.cos(i_rad)


def sun_synchronous_inclination_rad(a_m: float, e: float = 0.0) -> float:
    """Inclination at which the J2 nodal drift equals the Sun's mean motion [rad].

    Solves j2_raan_rate(a, e, i) = 2 pi / tropical year for i.
    """
    _check_orbit(a_m, e)
    n = math.sqrt(MU_EARTH_M3_S2 / a_m**3)
    p = a_m * (1.0 - e**2)
    cos_i = -SUN_SYNC_RATE_RAD_S / (1.5 * n * J2_EARTH * (R_EARTH_EQ_M / p) ** 2)
    if not -1.0 <= cos_i <= 1.0:
        raise ValueError(
            f"No sun-synchronous inclination exists for a = {a_m:.0f} m, e = {e}: "
            f"required cos(i) = {cos_i:.3f}")
    return math.acos(cos_i)


def sun_right_ascension_teme_rad(epoch: Time) -> float:
    """Right ascension of the true (apparent) Sun in TEME at ``epoch`` [rad].

    RAAN for SGP4 is measured in TEME, so the Sun must be expressed in the same frame.
    """
    sun_gcrs = get_sun(epoch)
    sun_teme = sun_gcrs.transform_to(TEME(obstime=epoch))
    x, y = sun_teme.cartesian.x.to_value(u.m), sun_teme.cartesian.y.to_value(u.m)
    return math.atan2(y, x) % (2.0 * math.pi)


def raan_for_ltan_rad(epoch: Time, ltan_hours: float) -> float:
    """RAAN that puts the ascending node at local solar time ``ltan_hours`` [rad].

    Local solar time at the node is 12 h plus the hour angle of the node from the
    Sun, so RAAN = RA_sun + (LTAN - 12 h) * 15 deg/h. The true Sun is used, so this
    is apparent solar time. Using the mean Sun instead would shift the node by the
    equation of time (about 12 minutes on 1 March).
    """
    if not 0.0 <= ltan_hours <= 24.0:
        raise ValueError(f"ltan_hours must be in [0, 24], got {ltan_hours}")
    ra_sun = sun_right_ascension_teme_rad(epoch)
    return (ra_sun + (ltan_hours - 12.0) * math.pi / 12.0) % (2.0 * math.pi)


def design_mean_elements(orbit: OrbitConfig, epoch: Time) -> MeanElements:
    """Build the synthetic satellite's mean elements from the scenario."""
    a = R_EARTH_EQ_M + orbit.altitude_m
    return MeanElements(
        epoch=epoch,
        semi_major_axis_m=a,
        eccentricity=orbit.eccentricity,
        inclination_rad=sun_synchronous_inclination_rad(a, orbit.eccentricity),
        raan_rad=raan_for_ltan_rad(epoch, orbit.ltan_hours),
        arg_perigee_rad=orbit.arg_perigee_rad,
        mean_anomaly_rad=orbit.mean_anomaly_rad,
        bstar_per_earth_radii=orbit.bstar_per_earth_radii,
    )


def rv_to_raan_rad(r_m: np.ndarray, v_m_s: np.ndarray) -> np.ndarray:
    """Osculating RAAN from position and velocity arrays of shape (..., 3) [rad]."""
    h = np.cross(r_m, v_m_s)
    return np.arctan2(h[..., 0], -h[..., 1])


def deg_per_day(rate_rad_s: float) -> float:
    return math.degrees(rate_rad_s) * SECONDS_PER_DAY


def _check_orbit(a_m: float, e: float) -> None:
    if not (math.isfinite(a_m) and a_m > R_EARTH_EQ_M):
        raise ValueError(f"Semi-major axis must exceed the Earth radius, got {a_m} m")
    if not 0.0 <= e < 1.0:
        raise ValueError(f"Eccentricity must be in [0, 1), got {e}")
