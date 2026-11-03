"""Pointing profile for each pass: ground telescope angles and rates, and the
spacecraft side (off-nadir angle, slew quaternion, point-ahead).

Point-ahead: light leaving the satellite takes R/c to reach the station, and in
that time the station moves sideways relative to the satellite. The satellite
also sees the station's uplink beacon where the station was R/c ago. The two
effects add, so the transmit beam must lead the received beacon direction by

    point-ahead angle = 2 * v_perp / c,

where v_perp is the relative inertial velocity component perpendicular to the
line of sight. Velocities must be inertial: the station's inertial velocity is
omega_earth x r_station (about 410 m/s at 28 deg latitude).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opticalpass.attitude import off_nadir_angle_rad, shortest_slew_quat, to_body
from opticalpass.constants import C_LIGHT_M_S
from opticalpass.frames import OMEGA_EARTH_VEC
from opticalpass.geometry import Tracker
from opticalpass.passes import Pass


@dataclass(frozen=True)
class PointingProfile:
    """Per-sample pointing quantities for one pass. SI units, radians.

    los_body: unit line of sight satellite to station in the body (LVLH) frame.
    point_ahead_dir_body: unit direction of the point-ahead offset in body axes.
    slew_quat: scalar-first quaternion turning body +Z onto los_body.
    """

    pass_number: int
    t_s: NDArray[np.float64]
    az_rad: NDArray[np.float64]
    el_rad: NDArray[np.float64]
    az_rate_rad_s: NDArray[np.float64]
    el_rate_rad_s: NDArray[np.float64]
    range_m: NDArray[np.float64]
    range_rate_m_s: NDArray[np.float64]
    off_nadir_rad: NDArray[np.float64]
    los_body: NDArray[np.float64]
    slew_quat: NDArray[np.float64]
    v_perp_m_s: NDArray[np.float64]
    point_ahead_rad: NDArray[np.float64]
    point_ahead_dir_body: NDArray[np.float64]

    @property
    def peak_az_rate_rad_s(self) -> float:
        return float(np.max(np.abs(self.az_rate_rad_s)))

    @property
    def peak_el_rate_rad_s(self) -> float:
        return float(np.max(np.abs(self.el_rate_rad_s)))

    @property
    def t_peak_az_rate_s(self) -> float:
        return float(self.t_s[int(np.argmax(np.abs(self.az_rate_rad_s)))])


def sample_times(start_s: float, end_s: float, step_s: float) -> NDArray[np.float64]:
    """Regular samples from start to end, always including both end points."""
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be after start_s ({start_s})")
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    n = max(2, int(math.ceil((end_s - start_s) / step_s)) + 1)
    return np.linspace(start_s, end_s, n)


def point_ahead(los_unit: NDArray[np.float64], v_rel_m_s: NDArray[np.float64]
                ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Point-ahead magnitude [rad], perpendicular speed [m/s] and unit direction.

    los_unit: satellite-to-station unit vector. v_rel_m_s: station inertial velocity
    minus satellite inertial velocity, same axes. The beam must lead towards
    where the station is going, i.e. along v_perp of the station relative to the
    satellite.
    """
    u = np.asarray(los_unit, dtype=float)
    v = np.asarray(v_rel_m_s, dtype=float)
    v_par = np.sum(v * u, axis=-1, keepdims=True) * u
    v_perp = v - v_par
    speed = np.linalg.norm(v_perp, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = np.where(speed[..., None] > 0, v_perp / speed[..., None], 0.0)
    return 2.0 * speed / C_LIGHT_M_S, speed, direction


def pointing_profile(tracker: Tracker, p: Pass, step_s: float = 2.0) -> PointingProfile:
    t = sample_times(p.start_s, p.end_s, step_s)
    g = tracker.geometry(t)
    los = -g.rho_itrs_m
    los_unit = los / np.linalg.norm(los, axis=-1, keepdims=True)
    los_body = to_body(los_unit, g.r_itrs_m, g.v_inertial_itrs_m_s)
    v_station = np.cross(OMEGA_EARTH_VEC, tracker.station.ecef_m)
    v_rel = v_station[None, :] - g.v_inertial_itrs_m_s
    paa, v_perp, pa_dir = point_ahead(los_unit, v_rel)
    pa_dir_body = to_body(pa_dir, g.r_itrs_m, g.v_inertial_itrs_m_s)
    return PointingProfile(
        pass_number=p.number, t_s=t, az_rad=g.az_rad, el_rad=g.el_rad,
        az_rate_rad_s=g.az_rate_rad_s, el_rate_rad_s=g.el_rate_rad_s,
        range_m=g.range_m, range_rate_m_s=g.range_rate_m_s,
        off_nadir_rad=off_nadir_angle_rad(los_body), los_body=los_body,
        slew_quat=shortest_slew_quat(los_body), v_perp_m_s=v_perp,
        point_ahead_rad=paa, point_ahead_dir_body=pa_dir_body,
    )
