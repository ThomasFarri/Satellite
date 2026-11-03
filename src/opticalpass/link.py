"""Simple Gaussian-beam optical downlink budget. All inputs are illustrative.

Received power for a Gaussian beam with 1/e^2 radius w at the receiver, much
larger than the aperture diameter D:

    on-axis intensity   I0 = 2 P / (pi w^2)
    collected power     P_rx = I0 * (pi D^2 / 4) = P D^2 / (2 w^2)
    with w = theta * R  (far field, theta = 1/e^2 half-angle divergence)

then multiplied by optics efficiency, atmospheric transmission
T_zenith ** airmass with airmass ~ 1 / sin(el), and the pointing loss
exp(-2 (e / theta)^2) for a pointing error e.

Not modelled: turbulence (scintillation, beam wander, fading), clouds,
background light, detector and coding details. The required received power is a
single config number that stands in for all of that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from opticalpass.config import TerminalConfig


def airmass(el_rad: ArrayLike) -> NDArray[np.float64]:
    """Plane-parallel airmass 1/sin(el). Within about 1 % of better models above 20 deg."""
    el = np.asarray(el_rad, dtype=float)
    if np.any(el <= 0):
        raise ValueError("Airmass undefined at or below the horizon")
    return 1.0 / np.sin(el)


def pointing_loss(error_rad: float, divergence_rad: float) -> float:
    """Fraction of on-axis intensity at an angular offset ``error_rad`` [-]."""
    if divergence_rad <= 0 or error_rad < 0:
        raise ValueError("Divergence must be positive and pointing error non-negative")
    return math.exp(-2.0 * (error_rad / divergence_rad) ** 2)


def received_power_w(range_m: ArrayLike, el_rad: ArrayLike, term: TerminalConfig,
                     pointing_error_rad: float | None = None) -> NDArray[np.float64]:
    rng = np.asarray(range_m, dtype=float)
    if np.any(rng <= 0):
        raise ValueError("Range must be positive")
    err = term.pointing_error_rad if pointing_error_rad is None else pointing_error_rad
    w = term.divergence_half_angle_rad * rng
    if np.any(w < 5 * term.rx_aperture_diameter_m):
        raise ValueError("Beam footprint is not much larger than the aperture; "
                         "the small-aperture approximation does not hold")
    geometric = term.rx_aperture_diameter_m**2 / (2.0 * w**2)
    atmos = term.zenith_transmission ** airmass(el_rad)
    return (term.tx_power_w * term.optics_efficiency * atmos
            * pointing_loss(err, term.divergence_half_angle_rad) * geometric)


def margin_db(p_rx_w: ArrayLike, required_w: float) -> NDArray[np.float64]:
    return 10.0 * np.log10(np.asarray(p_rx_w, dtype=float) / required_w)


def positive_time_s(t_s: ArrayLike, values: ArrayLike) -> float:
    """Total time where ``values`` > 0, linearly interpolating zero crossings."""
    t = np.asarray(t_s, dtype=float)
    m = np.asarray(values, dtype=float)
    if t.shape != m.shape or t.ndim != 1:
        raise ValueError("t_s and values must be 1-D arrays of equal length")
    total = 0.0
    for k in range(len(t) - 1):
        a, b, dt = m[k], m[k + 1], t[k + 1] - t[k]
        if a > 0 and b > 0:
            total += dt
        elif a > 0 or b > 0:
            frac = max(a, b) / abs(b - a)
            total += dt * frac
    return total


@dataclass(frozen=True)
class LinkResult:
    pass_number: int
    t_s: NDArray[np.float64]
    p_rx_w: NDArray[np.float64]
    margin_db: NDArray[np.float64]
    positive_margin_s: float
    data_volume_bits: float

    @property
    def min_margin_db(self) -> float:
        return float(np.min(self.margin_db))

    @property
    def max_margin_db(self) -> float:
        return float(np.max(self.margin_db))


def link_for_profile(pass_number: int, t_s: ArrayLike, range_m: ArrayLike, el_rad: ArrayLike,
                     term: TerminalConfig, pointing_error_rad: float | None = None
                     ) -> LinkResult:
    t = np.asarray(t_s, dtype=float)
    p = received_power_w(range_m, el_rad, term, pointing_error_rad)
    m = margin_db(p, term.required_rx_power_w)
    tpos = positive_time_s(t, m)
    return LinkResult(pass_number, t, p, m, tpos, tpos * term.data_rate_bps)


@dataclass(frozen=True)
class SensitivityRow:
    pointing_error_rad: float
    pointing_loss_db: float
    total_volume_bits: float
    volume_per_day_bits: float
    worst_min_margin_db: float
    positive_margin_s: float


def pointing_sensitivity(profiles: Sequence[tuple[int, ArrayLike, ArrayLike, ArrayLike]],
                         term: TerminalConfig, errors_rad: Sequence[float],
                         duration_days: float) -> list[SensitivityRow]:
    """Data volume and margin across pointing errors. profiles: (pass, t, range, el)."""
    if duration_days <= 0:
        raise ValueError("duration_days must be positive")
    rows = []
    for e in errors_rad:
        res = [link_for_profile(n, t, r, el, term, e) for n, t, r, el in profiles]
        vol = sum(x.data_volume_bits for x in res)
        rows.append(SensitivityRow(
            pointing_error_rad=e,
            pointing_loss_db=10 * math.log10(pointing_loss(e, term.divergence_half_angle_rad)),
            total_volume_bits=vol,
            volume_per_day_bits=vol / duration_days,
            worst_min_margin_db=min((x.min_margin_db for x in res), default=float("nan")),
            positive_margin_s=sum(x.positive_margin_s for x in res),
        ))
    return rows
