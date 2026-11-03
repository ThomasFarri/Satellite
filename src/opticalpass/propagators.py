"""Orbit propagators behind one interface.

Both propagators return states in TEME (True Equator, Mean Equinox), the frame
SGP4 works in. Frame conversion to Earth-fixed coordinates happens in
``frames.py`` so the propagators stay frame-agnostic beyond that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from astropy.time import Time
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from sgp4.api import WGS84, Satrec

from opticalpass.constants import J2_EARTH, MU_EARTH_M3_S2, R_EARTH_EQ_M
from opticalpass.orbit import MeanElements

# SGP4 epoch counts days from 1949 December 31 00:00 UT (Julian date 2433281.5).
_SGP4_EPOCH_JD = 2433281.5


@dataclass(frozen=True)
class StateVectors:
    """Satellite states in TEME.

    t_s: seconds since the scenario epoch, shape (N,).
    r_m: positions [m], shape (N, 3).
    v_m_s: velocities [m/s], shape (N, 3).
    """

    t_s: NDArray[np.float64]
    r_m: NDArray[np.float64]
    v_m_s: NDArray[np.float64]


class Propagator(Protocol):
    epoch: Time
    name: str

    def propagate(self, t_s: ArrayLike) -> StateVectors:
        """States at ``t_s`` seconds after ``epoch``, in TEME."""
        ...


def _as_time_array(t_s: ArrayLike) -> NDArray[np.float64]:
    t = np.atleast_1d(np.asarray(t_s, dtype=float))
    if t.ndim != 1:
        raise ValueError(f"t_s must be a scalar or 1-D array, got shape {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("t_s contains non-finite values")
    return t


class SGP4Propagator:
    """SGP4 built directly from mean elements with ``sgp4init``.

    The WGS84 gravity constants are selected so SGP4 and the numerical propagator
    share the same Earth model as far as SGP4 allows.
    """

    name = "sgp4"

    def __init__(self, elements: MeanElements, satnum: int = 99999) -> None:
        self.epoch = elements.epoch
        self.elements = elements
        sat = Satrec()
        epoch_days = (elements.epoch.utc.jd1 - _SGP4_EPOCH_JD) + elements.epoch.utc.jd2
        # sgp4init wants mean motion in rad/min (Kozai convention) and angles in rad.
        no_kozai_rad_min = elements.mean_motion_rad_s * 60.0
        sat.sgp4init(
            WGS84, "i", satnum, epoch_days,
            elements.bstar_per_earth_radii, 0.0, 0.0,
            elements.eccentricity, elements.arg_perigee_rad, elements.inclination_rad,
            elements.mean_anomaly_rad, no_kozai_rad_min, elements.raan_rad,
        )
        if sat.error != 0:
            raise RuntimeError(f"sgp4init failed with error code {sat.error}")
        self.satrec = sat

    def propagate(self, t_s: ArrayLike) -> StateVectors:
        t = _as_time_array(t_s)
        jd = np.full_like(t, self.epoch.utc.jd1)
        fr = self.epoch.utc.jd2 + t / 86400.0
        err, r_km, v_km_s = self.satrec.sgp4_array(jd, fr)
        bad = np.flatnonzero(err)
        if bad.size:
            raise RuntimeError(
                f"SGP4 returned error code {int(err[bad[0]])} at t = {t[bad[0]]:.1f} s "
                f"({bad.size} bad samples)")
        return StateVectors(t, np.asarray(r_km) * 1e3, np.asarray(v_km_s) * 1e3)


def two_body_j2_accel(r_m: NDArray[np.float64], include_j2: bool = True) -> NDArray[np.float64]:
    """Acceleration from point-mass gravity plus J2 [m/s^2]. ``r_m`` shape (3,) or (N, 3).

    J2 is defined about the frame z axis, so the frame must have z along the
    Earth's rotation axis. TEME does (true pole of date).
    """
    r_m = np.asarray(r_m, dtype=float)
    x, y, z = r_m[..., 0], r_m[..., 1], r_m[..., 2]
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    if np.any(r < R_EARTH_EQ_M * 0.5):
        raise ValueError("Position inside the Earth; integration has diverged")
    k = -MU_EARTH_M3_S2 / (r2 * r)
    ax, ay, az = k * x, k * y, k * z
    if include_j2:
        f = 1.5 * J2_EARTH * (R_EARTH_EQ_M**2 / r2)
        zr2 = z * z / r2
        ax = ax * (1.0 + f * (1.0 - 5.0 * zr2))
        ay = ay * (1.0 + f * (1.0 - 5.0 * zr2))
        az = az * (1.0 + f * (3.0 - 5.0 * zr2))
    return np.stack([ax, ay, az], axis=-1)


class NumericalJ2Propagator:
    """Two-body plus J2 equations of motion integrated with DOP853.

    The integration runs once over [0, duration] with dense output, so any time in
    that span can be evaluated without re-integrating. The TEME axes at epoch are
    treated as inertial. TEME precesses and nutates slowly (well under an
    arcsecond over a week), which is negligible here.
    """

    name = "j2"

    def __init__(self, epoch: Time, r0_m: ArrayLike, v0_m_s: ArrayLike, duration_s: float,
                 include_j2: bool = True, rtol: float = 1e-11, atol: float = 1e-6) -> None:
        r0 = np.asarray(r0_m, dtype=float).reshape(3)
        v0 = np.asarray(v0_m_s, dtype=float).reshape(3)
        if not (math.isfinite(duration_s) and duration_s > 0):
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if np.linalg.norm(r0) <= R_EARTH_EQ_M:
            raise ValueError("Initial position is inside the Earth")
        self.epoch = epoch
        self.duration_s = float(duration_s)
        self.include_j2 = include_j2

        def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.concatenate([y[3:], two_body_j2_accel(y[:3], include_j2)])

        sol = solve_ivp(rhs, (0.0, self.duration_s), np.concatenate([r0, v0]),
                        method="DOP853", rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        self._sol = sol
        self.nfev = sol.nfev

    @classmethod
    def from_propagator(cls, other: Propagator, duration_s: float,
                        include_j2: bool = True, **kwargs: float) -> NumericalJ2Propagator:
        """Start from another propagator's state at epoch (e.g. SGP4)."""
        s = other.propagate(0.0)
        return cls(other.epoch, s.r_m[0], s.v_m_s[0], duration_s, include_j2, **kwargs)

    def propagate(self, t_s: ArrayLike) -> StateVectors:
        t = _as_time_array(t_s)
        # Allow a tiny overshoot so floating-point round-off at the end does not raise.
        eps = 1e-6
        if t.min() < -eps or t.max() > self.duration_s + eps:
            raise ValueError(
                f"Requested times [{t.min():.3f}, {t.max():.3f}] s are outside the "
                f"integrated span [0, {self.duration_s:.3f}] s")
        y = self._sol.sol(np.clip(t, 0.0, self.duration_s))
        return StateVectors(t, y[:3].T.copy(), y[3:].T.copy())


def ric_difference_m(ref: StateVectors, other: StateVectors) -> NDArray[np.float64]:
    """Position of ``other`` minus ``ref`` in ref's radial, in-track, cross-track axes [m]."""
    if ref.r_m.shape != other.r_m.shape:
        raise ValueError("State arrays must have the same shape")
    r_hat = ref.r_m / np.linalg.norm(ref.r_m, axis=1, keepdims=True)
    h = np.cross(ref.r_m, ref.v_m_s)
    c_hat = h / np.linalg.norm(h, axis=1, keepdims=True)
    i_hat = np.cross(c_hat, r_hat)
    d = other.r_m - ref.r_m
    return np.stack([np.sum(d * r_hat, 1), np.sum(d * i_hat, 1), np.sum(d * c_hat, 1)], axis=1)


def build_propagator(kind: str, elements: MeanElements, duration_s: float) -> Propagator:
    """Factory used by the CLI. ``kind`` is 'sgp4' or 'j2'."""
    sgp4 = SGP4Propagator(elements)
    if kind == "sgp4":
        return sgp4
    if kind == "j2":
        return NumericalJ2Propagator.from_propagator(sgp4, duration_s)
    raise ValueError(f"Unknown propagator {kind!r}; use 'sgp4' or 'j2'")
