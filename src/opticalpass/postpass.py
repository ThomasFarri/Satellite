"""Post-pass analysis on SYNTHETIC telemetry.

The generator takes the true geometry of a pass and adds a fixed azimuth and
elevation bias, a clock (timing) offset and Gaussian noise, imitating what a
ground telescope's encoder log might show. It is synthetic: no real telemetry is
used anywhere in this project.

The analysis fits the three unknowns by least squares:

    az_meas(t) = az_pred(t + dt) + b_az + noise
    el_meas(t) = el_pred(t + dt) + b_el + noise

Bias and timing separate because a timing error shows up multiplied by the
angle rate, which changes over the pass (and changes sign in elevation at the
highest point), while a bias is constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from opticalpass.config import PostPassConfig
from opticalpass.geometry import Tracker
from opticalpass.passes import Pass
from opticalpass.pointing import sample_times

PREDICTION_PAD_S = 15.0
MAX_TIMING_OFFSET_S = 10.0


@dataclass(frozen=True)
class Telemetry:
    """SYNTHETIC measured angles. t_s is the station clock time since epoch."""

    t_s: NDArray[np.float64]
    az_rad: NDArray[np.float64]
    el_rad: NDArray[np.float64]
    injected: PostPassConfig
    synthetic: bool = True


@dataclass(frozen=True)
class Prediction:
    t_s: NDArray[np.float64]
    az_rad: NDArray[np.float64]
    el_rad: NDArray[np.float64]


def predict(tracker: Tracker, p: Pass, step_s: float = 0.5) -> Prediction:
    """Predicted angles over the pass plus a margin, so shifted times stay covered."""
    t = sample_times(p.start_s - PREDICTION_PAD_S, p.end_s + PREDICTION_PAD_S, step_s)
    g = tracker.geometry(t)
    return Prediction(t, g.az_rad, g.el_rad)


def generate_telemetry(tracker: Tracker, p: Pass, cfg: PostPassConfig) -> Telemetry:
    """SYNTHETIC telemetry for pass ``p`` with the errors set in ``cfg``."""
    if abs(cfg.timing_offset_s) > MAX_TIMING_OFFSET_S:
        raise ValueError(f"timing_offset_s must be within +/-{MAX_TIMING_OFFSET_S} s")
    rng = np.random.default_rng(cfg.seed)
    t = sample_times(p.start_s, p.end_s, cfg.sample_step_s)
    g = tracker.geometry(t + cfg.timing_offset_s)
    az = np.mod(g.az_rad + cfg.bias_az_rad + rng.normal(0, cfg.noise_rad, t.size), 2 * np.pi)
    el = g.el_rad + cfg.bias_el_rad + rng.normal(0, cfg.noise_rad, t.size)
    return Telemetry(t, az, el, cfg)


@dataclass(frozen=True)
class PostPassResult:
    bias_az_rad: float
    bias_el_rad: float
    timing_offset_s: float
    sigma_bias_az_rad: float
    sigma_bias_el_rad: float
    sigma_timing_s: float
    rms_residual_rad: float
    residual_az_rad: NDArray[np.float64]
    residual_el_rad: NDArray[np.float64]
    raw_residual_az_rad: NDArray[np.float64]
    raw_residual_el_rad: NDArray[np.float64]


def _wrap(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return (a + np.pi) % (2 * np.pi) - np.pi


def analyse(tel: Telemetry, pred: Prediction) -> PostPassResult:
    """Recover azimuth bias, elevation bias and timing offset from residuals."""
    if tel.t_s.size < 4:
        raise ValueError("Need at least 4 telemetry samples to fit 3 parameters")
    if pred.t_s.min() > tel.t_s.min() or pred.t_s.max() < tel.t_s.max():
        raise ValueError("Prediction does not cover the telemetry time span")
    az_spline = CubicSpline(pred.t_s, np.unwrap(pred.az_rad))
    el_spline = CubicSpline(pred.t_s, pred.el_rad)
    dt_lo = max(-MAX_TIMING_OFFSET_S, pred.t_s.min() - tel.t_s.min())
    dt_hi = min(MAX_TIMING_OFFSET_S, pred.t_s.max() - tel.t_s.max())

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        b_az, b_el, dt = x
        ra = _wrap(tel.az_rad - az_spline(tel.t_s + dt) - b_az)
        re = tel.el_rad - el_spline(tel.t_s + dt) - b_el
        return np.concatenate([ra, re])

    fit = least_squares(residuals, x0=np.zeros(3), bounds=([-0.1, -0.1, dt_lo],
                                                            [0.1, 0.1, dt_hi]),
                        x_scale=[1e-4, 1e-4, 0.1], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    if not fit.success:
        raise RuntimeError(f"Post-pass fit failed: {fit.message}")
    r = fit.fun
    n, k = r.size, 3
    s2 = float(r @ r) / (n - k)
    cov = np.linalg.inv(fit.jac.T @ fit.jac) * s2
    sig = np.sqrt(np.diag(cov))
    raw = residuals(np.zeros(3))
    m = tel.t_s.size
    return PostPassResult(
        bias_az_rad=float(fit.x[0]), bias_el_rad=float(fit.x[1]),
        timing_offset_s=float(fit.x[2]), sigma_bias_az_rad=float(sig[0]),
        sigma_bias_el_rad=float(sig[1]), sigma_timing_s=float(sig[2]),
        rms_residual_rad=math.sqrt(s2), residual_az_rad=r[:m], residual_el_rad=r[m:],
        raw_residual_az_rad=raw[:m], raw_residual_el_rad=raw[m:],
    )


def report_rows(res: PostPassResult, injected: PostPassConfig) -> list[tuple]:
    """Injected against recovered, in display units."""
    d = math.degrees
    return [
        ("bias_az_deg", d(injected.bias_az_rad), d(res.bias_az_rad), d(res.sigma_bias_az_rad),
         d(res.bias_az_rad - injected.bias_az_rad)),
        ("bias_el_deg", d(injected.bias_el_rad), d(res.bias_el_rad), d(res.sigma_bias_el_rad),
         d(res.bias_el_rad - injected.bias_el_rad)),
        ("timing_offset_s", injected.timing_offset_s, res.timing_offset_s, res.sigma_timing_s,
         res.timing_offset_s - injected.timing_offset_s),
    ]


REPORT_HEADER = ("parameter", "injected", "recovered", "one_sigma", "error")
