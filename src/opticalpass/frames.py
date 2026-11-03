"""Reference frames: TEME to ITRS through astropy, and a handwritten
geodetic, ECEF, ENU and azimuth/elevation chain.

The handwritten chain is the one the planner uses. Astropy's AltAz is only used
to check it.

Near-Earth objects: astropy's ITRS documentation says that transforming a
geocentric ITRS position straight to AltAz applies the change in stellar
aberration between the geocentre and the observer. That is right for distant
sources and wrong for a satellite a few hundred km away. The documented approach,
used in ``astropy_altaz`` below, is to subtract the station's ITRS position by
hand and put the result in a topocentric ITRS frame (``location=station``). The
ITRS to AltAz step is then a pure rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    GCRS,
    ITRS,
    TEME,
    AltAz,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    get_sun,
)
from astropy.time import Time, TimeDelta
from numpy.typing import ArrayLike, NDArray

from opticalpass.config import StationConfig
from opticalpass.constants import OMEGA_EARTH_RAD_S, R_EARTH_EQ_M, WGS84_E2

OMEGA_EARTH_VEC = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])


def times_from_epoch(epoch: Time, t_s: ArrayLike) -> Time:
    """Absolute UTC times for seconds after ``epoch``."""
    return epoch + TimeDelta(np.asarray(t_s, dtype=float), format="sec")


# --------------------------------------------------------------------------- TEME/ITRS


def teme_to_itrs(epoch: Time, t_s: ArrayLike, r_teme_m: ArrayLike,
                 v_teme_m_s: ArrayLike | None = None
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Convert TEME states to ITRS (Earth-fixed) with astropy.

    Returns position [m] and, if a velocity is given, the Earth-fixed velocity
    [m/s], i.e. the velocity seen by an observer rotating with the Earth.
    """
    t = times_from_epoch(epoch, np.atleast_1d(t_s))
    r = np.atleast_2d(np.asarray(r_teme_m, dtype=float))
    if r.shape != (len(t), 3):
        raise ValueError(f"r_teme_m must have shape ({len(t)}, 3), got {r.shape}")
    rep = CartesianRepresentation(r.T * u.m)
    if v_teme_m_s is not None:
        v = np.atleast_2d(np.asarray(v_teme_m_s, dtype=float))
        if v.shape != r.shape:
            raise ValueError("v_teme_m_s must have the same shape as r_teme_m")
        rep = rep.with_differentials(CartesianDifferential(v.T * u.m / u.s))
    itrs = TEME(rep, obstime=t).transform_to(ITRS(obstime=t))
    r_out = itrs.cartesian.xyz.to_value(u.m).T
    if v_teme_m_s is None:
        return r_out, None
    v_out = itrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T
    return r_out, v_out


def teme_to_gcrs(epoch: Time, t_s: ArrayLike, r_teme_m: ArrayLike, v_teme_m_s: ArrayLike
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert TEME states to GCRS (inertial) with astropy. Used for cross-checks."""
    t = times_from_epoch(epoch, np.atleast_1d(t_s))
    r = np.atleast_2d(np.asarray(r_teme_m, dtype=float))
    v = np.atleast_2d(np.asarray(v_teme_m_s, dtype=float))
    rep = CartesianRepresentation(r.T * u.m).with_differentials(
        CartesianDifferential(v.T * u.m / u.s))
    g = TEME(rep, obstime=t).transform_to(GCRS(obstime=t))
    return (g.cartesian.xyz.to_value(u.m).T,
            g.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T)


def inertial_velocity_in_itrs_axes(r_itrs_m: ArrayLike, v_itrs_m_s: ArrayLike
                                   ) -> NDArray[np.float64]:
    """Inertial velocity expressed along ITRS axes: v_inertial = v_earth_fixed + omega x r.

    Uses the nominal rotation rate about the ITRS z axis. Polar motion and the
    small difference between ITRS z and the instantaneous spin axis are ignored
    (sub-microradian effects).
    """
    r = np.asarray(r_itrs_m, dtype=float)
    return np.asarray(v_itrs_m_s, dtype=float) + np.cross(OMEGA_EARTH_VEC, r)


# ------------------------------------------------------------------ handwritten chain


@dataclass(frozen=True)
class Station:
    name: str
    lat_rad: float
    lon_rad: float
    height_m: float

    @classmethod
    def from_config(cls, cfg: StationConfig) -> Station:
        return cls(cfg.name, cfg.lat_rad, cfg.lon_rad, cfg.height_m)

    @property
    def ecef_m(self) -> NDArray[np.float64]:
        return geodetic_to_ecef(self.lat_rad, self.lon_rad, self.height_m)

    @property
    def enu_matrix(self) -> NDArray[np.float64]:
        return ecef_to_enu_matrix(self.lat_rad, self.lon_rad)

    @property
    def earth_location(self) -> EarthLocation:
        return EarthLocation.from_geodetic(np.degrees(self.lon_rad) * u.deg,
                                           np.degrees(self.lat_rad) * u.deg,
                                           self.height_m * u.m, ellipsoid="WGS84")


def geodetic_to_ecef(lat_rad: float, lon_rad: float, h_m: float) -> NDArray[np.float64]:
    """WGS84 geodetic latitude, longitude, ellipsoidal height to ECEF [m].

    N is the prime vertical radius of curvature: the distance from the surface to
    the polar axis along the ellipsoid normal.
    """
    if not -np.pi / 2 - 1e-12 <= lat_rad <= np.pi / 2 + 1e-12:
        raise ValueError(f"Latitude must be within +/- pi/2 rad, got {lat_rad}")
    s, c = np.sin(lat_rad), np.cos(lat_rad)
    n = R_EARTH_EQ_M / np.sqrt(1.0 - WGS84_E2 * s * s)
    return np.array([
        (n + h_m) * c * np.cos(lon_rad),
        (n + h_m) * c * np.sin(lon_rad),
        (n * (1.0 - WGS84_E2) + h_m) * s,
    ])


def ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> NDArray[np.float64]:
    """Rotation matrix taking ECEF vector components to local East, North, Up.

    Rows are the East, North and Up unit vectors expressed in ECEF. Up is the
    ellipsoid normal (geodetic), not the direction from the Earth's centre.
    """
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    so, co = np.sin(lon_rad), np.cos(lon_rad)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def ecef_to_enu(vec_ecef: ArrayLike, station: Station) -> NDArray[np.float64]:
    """Rotate ECEF vector(s) of shape (..., 3) into the station ENU frame."""
    return np.asarray(vec_ecef, dtype=float) @ station.enu_matrix.T


def enu_to_aer(enu_m: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Azimuth [rad, 0 to 2 pi, clockwise from north], elevation [rad], range [m]."""
    enu = np.asarray(enu_m, dtype=float)
    e, n, up = enu[..., 0], enu[..., 1], enu[..., 2]
    horiz = np.hypot(e, n)
    rng = np.sqrt(horiz**2 + up**2)
    if np.any(rng == 0.0):
        raise ValueError("Zero range: target coincides with the station")
    az = np.mod(np.arctan2(e, n), 2 * np.pi)
    el = np.arctan2(up, horiz)
    return az, el, rng


def enu_to_aer_rates(enu_m: ArrayLike, enu_rate_m_s: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Analytic azimuth rate, elevation rate [rad/s] and range rate [m/s].

    From az = atan2(e, n) and el = atan2(u, h) with h = sqrt(e^2 + n^2):
    daz = (n de - e dn) / h^2, del = (h du - u dh) / (h^2 + u^2), dh = (e de + n dn) / h.
    Azimuth rate is singular at zenith (h = 0), which is the real tracking problem
    for an alt-azimuth mount, so it is returned as inf there rather than hidden.
    """
    p = np.asarray(enu_m, dtype=float)
    pd = np.asarray(enu_rate_m_s, dtype=float)
    e, n, up = p[..., 0], p[..., 1], p[..., 2]
    de, dn, du = pd[..., 0], pd[..., 1], pd[..., 2]
    h2 = e * e + n * n
    h = np.sqrt(h2)
    rng = np.sqrt(h2 + up * up)
    with np.errstate(divide="ignore", invalid="ignore"):
        az_rate = np.where(h2 > 0, (n * de - e * dn) / h2, np.inf)
        dh = np.where(h > 0, (e * de + n * dn) / h, 0.0)
        el_rate = (h * du - up * dh) / (rng * rng)
    range_rate = np.sum(p * pd, axis=-1) / rng
    return az_rate, el_rate, range_rate


def topocentric_aer(r_itrs_m: ArrayLike, station: Station) -> tuple[NDArray[np.float64], ...]:
    """Azimuth, elevation, range of ITRS position(s) from the station."""
    rho = np.asarray(r_itrs_m, dtype=float) - station.ecef_m
    return enu_to_aer(ecef_to_enu(rho, station))


# --------------------------------------------------------------- astropy cross-check


def astropy_altaz(epoch: Time, t_s: ArrayLike, r_itrs_m: ArrayLike, station: Station
                  ) -> tuple[NDArray[np.float64], ...]:
    """Azimuth, elevation, range from astropy, using a topocentric ITRS position.

    Refraction stays off (AltAz default pressure is zero).
    """
    t = times_from_epoch(epoch, np.atleast_1d(t_s))
    loc = station.earth_location
    r = np.atleast_2d(np.asarray(r_itrs_m, dtype=float))
    site = np.array([loc.x.to_value(u.m), loc.y.to_value(u.m), loc.z.to_value(u.m)])
    topo = ITRS(CartesianRepresentation((r - site).T * u.m), obstime=t, location=loc)
    aa = topo.transform_to(AltAz(obstime=t, location=loc))
    return (aa.az.to_value(u.rad), aa.alt.to_value(u.rad), aa.distance.to_value(u.m))


# ---------------------------------------------------------------------------- the Sun


def sun_enu_unit(epoch: Time, t_s: ArrayLike, station: Station) -> NDArray[np.float64]:
    """Unit vector to the Sun in the station ENU frame, shape (N, 3).

    The Sun is a distant source, so here the full astropy AltAz transform
    (including aberration) is correct. Refraction is off.
    """
    t = times_from_epoch(epoch, np.atleast_1d(t_s))
    aa = get_sun(t).transform_to(AltAz(obstime=t, location=station.earth_location))
    az, el = aa.az.to_value(u.rad), aa.alt.to_value(u.rad)
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1)
