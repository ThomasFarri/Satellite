"""Satellite-to-station geometry for any propagator.

``Tracker`` ties a propagator to a ground station and returns look angles,
Earth-fixed and inertial states, and the Sun direction, all vectorised over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from opticalpass.frames import (
    Station,
    ecef_to_enu,
    enu_to_aer,
    enu_to_aer_rates,
    inertial_velocity_in_itrs_axes,
    sun_enu_unit,
    teme_to_itrs,
)
from opticalpass.propagators import Propagator


@dataclass(frozen=True)
class Geometry:
    """Geometry samples. Angles in rad, distances in m, rates per second.

    r_itrs_m, v_itrs_m_s: Earth-fixed satellite state.
    v_inertial_itrs_m_s: inertial satellite velocity along ITRS axes.
    rho_itrs_m: station-to-satellite vector along ITRS axes.
    """

    t_s: NDArray[np.float64]
    az_rad: NDArray[np.float64]
    el_rad: NDArray[np.float64]
    range_m: NDArray[np.float64]
    az_rate_rad_s: NDArray[np.float64]
    el_rate_rad_s: NDArray[np.float64]
    range_rate_m_s: NDArray[np.float64]
    r_itrs_m: NDArray[np.float64]
    v_itrs_m_s: NDArray[np.float64]
    v_inertial_itrs_m_s: NDArray[np.float64]
    rho_itrs_m: NDArray[np.float64]


class Tracker:
    def __init__(self, propagator: Propagator, station: Station) -> None:
        self.propagator = propagator
        self.station = station
        self.epoch = propagator.epoch

    def itrs_position(self, t_s: ArrayLike) -> NDArray[np.float64]:
        s = self.propagator.propagate(t_s)
        r, _ = teme_to_itrs(self.epoch, s.t_s, s.r_m)
        return r

    def elevation(self, t_s: ArrayLike) -> NDArray[np.float64]:
        """Satellite elevation [rad] at the station. Cheapest call, used for pass finding."""
        rho = self.itrs_position(t_s) - self.station.ecef_m
        return enu_to_aer(ecef_to_enu(rho, self.station))[1]

    def geometry(self, t_s: ArrayLike) -> Geometry:
        s = self.propagator.propagate(t_s)
        r, v = teme_to_itrs(self.epoch, s.t_s, s.r_m, s.v_m_s)
        assert v is not None
        rho = r - self.station.ecef_m
        enu = ecef_to_enu(rho, self.station)
        enu_rate = ecef_to_enu(v, self.station)  # station is fixed in ITRS
        az, el, rng = enu_to_aer(enu)
        az_rate, el_rate, rr = enu_to_aer_rates(enu, enu_rate)
        return Geometry(
            t_s=s.t_s, az_rad=az, el_rad=el, range_m=rng, az_rate_rad_s=az_rate,
            el_rate_rad_s=el_rate, range_rate_m_s=rr, r_itrs_m=r, v_itrs_m_s=v,
            v_inertial_itrs_m_s=inertial_velocity_in_itrs_axes(r, v), rho_itrs_m=rho,
        )

    def los_enu_unit(self, t_s: ArrayLike) -> NDArray[np.float64]:
        rho = self.itrs_position(t_s) - self.station.ecef_m
        enu = ecef_to_enu(rho, self.station)
        return enu / np.linalg.norm(enu, axis=-1, keepdims=True)

    def sun_elevation(self, t_s: ArrayLike) -> NDArray[np.float64]:
        sun = sun_enu_unit(self.epoch, t_s, self.station)
        return np.arcsin(np.clip(sun[..., 2], -1.0, 1.0))

    def sun_separation(self, t_s: ArrayLike) -> NDArray[np.float64]:
        """Angle between the station-to-satellite line of sight and the Sun [rad]."""
        sun = sun_enu_unit(self.epoch, t_s, self.station)
        los = self.los_enu_unit(t_s)
        return np.arccos(np.clip(np.sum(sun * los, axis=-1), -1.0, 1.0))
