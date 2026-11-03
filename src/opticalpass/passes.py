"""Pass finding: rise and set times above a minimum elevation, then trimming or
splitting by Sun constraints.

Method: sample on a coarse grid, find sign changes, refine each with Brent's
method to better than 10 ms. A pass can peak just above the threshold between
two coarse samples and be missed by sign changes alone, so local maxima that
fall below the threshold are also checked with a bounded maximisation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar

from opticalpass.config import ConstraintConfig
from opticalpass.geometry import Tracker

VecFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

ROOT_XTOL_S = 1e-3  # 1 ms, ten times tighter than the 10 ms requirement


@dataclass(frozen=True)
class Interval:
    """Time interval [start_s, end_s] with a label saying what set each edge."""

    start_s: float
    end_s: float
    start_label: str
    end_label: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _scalar(f: VecFn) -> Callable[[float], float]:
    return lambda t: float(np.asarray(f(np.array([t])))[0])


def _refine(fs: Callable[[float], float], a: float, b: float, fa: float, fb: float) -> float:
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    return float(brentq(fs, a, b, xtol=ROOT_XTOL_S, rtol=4 * np.finfo(float).eps))


def find_intervals_above(f: VecFn, t0: float, t1: float, step_s: float,
                         near_miss_margin: float | None = None,
                         start_label: str = "rise", end_label: str = "set",
                         ) -> list[Interval]:
    """Intervals within [t0, t1] where f(t) >= 0.

    f must accept a 1-D array of times and return an array of the same length.
    near_miss_margin: if given, local maxima of the samples with
    -margin < f < 0 are refined to catch short excursions above zero between
    samples. Edges clipped by the window carry labels 'window_start' and
    'window_end'.
    """
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        raise ValueError(f"Need t1 > t0, got t0={t0}, t1={t1}")
    if not step_s > 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    n = max(2, int(math.ceil((t1 - t0) / step_s)) + 1)
    t = np.linspace(t0, t1, n)
    g = np.asarray(f(t), dtype=float)
    if g.shape != t.shape or not np.all(np.isfinite(g)):
        raise ValueError("f must return finite values with the same shape as its input")
    fs = _scalar(f)

    # Events: (time, +1 for entering, -1 for leaving).
    events: list[tuple[float, int]] = []
    inside = g >= 0
    for k in np.flatnonzero(inside[1:] != inside[:-1]):
        tk = _refine(fs, t[k], t[k + 1], g[k], g[k + 1])
        events.append((tk, +1 if inside[k + 1] else -1))

    if near_miss_margin is not None and n >= 3:
        mid = np.arange(1, n - 1)
        peaks = mid[(g[mid] >= g[mid - 1]) & (g[mid] >= g[mid + 1])
                    & (g[mid] < 0) & (g[mid] > -near_miss_margin)]
        for k in peaks:
            a, b = t[k - 1], t[k + 1]
            res = minimize_scalar(lambda x: -fs(x), bounds=(a, b), method="bounded",
                                  options={"xatol": ROOT_XTOL_S})
            tm, gm = float(res.x), -float(res.fun)
            if gm > 0:
                events.append((_refine(fs, a, tm, g[k - 1], gm), +1))
                events.append((_refine(fs, tm, b, gm, g[k + 1]), -1))

    events.sort()
    intervals: list[Interval] = []
    cur_start: float | None = t0 if inside[0] else None
    cur_label = "window_start"
    for tk, kind in events:
        if kind > 0 and cur_start is None:
            cur_start, cur_label = tk, start_label
        elif kind < 0 and cur_start is not None:
            if tk > cur_start:
                intervals.append(Interval(cur_start, tk, cur_label, end_label))
            cur_start = None
    if cur_start is not None and t1 > cur_start:
        intervals.append(Interval(cur_start, t1, cur_label, "window_end"))
    return intervals


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    """Intersection of two sorted, non-overlapping interval lists, keeping edge labels."""
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x.start_s >= y.start_s:
            s, sl = x.start_s, x.start_label
        else:
            s, sl = y.start_s, y.start_label
        if x.end_s <= y.end_s:
            e, el = x.end_s, x.end_label
        else:
            e, el = y.end_s, y.end_label
        if e > s:
            out.append(Interval(s, e, sl, el))
        if x.end_s < y.end_s:
            i += 1
        else:
            j += 1
    return out


@dataclass(frozen=True)
class Pass:
    number: int
    start_s: float
    end_s: float
    start_label: str
    end_label: str
    max_elevation_rad: float
    t_max_elevation_s: float
    min_range_m: float
    visibility: Interval = field(compare=False)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def trimmed_by(self) -> str:
        """Constraints that moved an edge away from the geometric rise or set."""
        names = []
        for lab in (self.start_label, self.end_label):
            if lab not in ("rise", "set") and lab not in names:
                names.append(lab)
        return ",".join(names)


CONSTRAINT_SUN_SEP = "sun_separation"
CONSTRAINT_NIGHT = "night"


def find_passes(tracker: Tracker, constraints: ConstraintConfig, t0: float, t1: float,
                coarse_step_s: float = 30.0, constraint_step_s: float = 5.0,
                ) -> list[Pass]:
    """Passes above the minimum elevation that also meet the Sun constraints.

    A pass whose constraint changes state mid-pass is trimmed, or split into two
    passes if the constraint fails only in the middle.
    """
    min_el = constraints.min_elevation_rad

    def g_el(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return tracker.elevation(t) - min_el

    def g_sun(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return tracker.sun_separation(t) - constraints.min_sun_separation_rad

    def g_night(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints.night_sun_elevation_rad - tracker.sun_elevation(t)

    vis = find_intervals_above(g_el, t0, t1, coarse_step_s,
                               near_miss_margin=math.radians(10.0))
    passes: list[Pass] = []
    for w in vis:
        # Re-label clipped window edges explicitly; find_intervals_above already does.
        pieces = [w]
        checks: list[tuple[VecFn, str]] = []
        if constraints.min_sun_separation_rad > 0:
            checks.append((g_sun, CONSTRAINT_SUN_SEP))
        if constraints.night_only:
            checks.append((g_night, CONSTRAINT_NIGHT))
        for fn, name in checks:
            ok = find_intervals_above(fn, w.start_s, w.end_s,
                                      min(constraint_step_s, max(w.duration_s / 2, 1e-3)),
                                      start_label=name, end_label=name)
            # Edges that coincide with the window keep the window's own labels.
            ok = [Interval(iv.start_s, iv.end_s,
                           w.start_label if iv.start_label == "window_start" else iv.start_label,
                           w.end_label if iv.end_label == "window_end" else iv.end_label)
                  for iv in ok]
            pieces = intersect(pieces, ok)
        for p in pieces:
            if p.duration_s <= ROOT_XTOL_S:
                continue
            t_max, el_max = _max_elevation(tracker, p.start_s, p.end_s)
            passes.append(Pass(
                number=len(passes) + 1, start_s=p.start_s, end_s=p.end_s,
                start_label=p.start_label, end_label=p.end_label,
                max_elevation_rad=el_max, t_max_elevation_s=t_max,
                min_range_m=_min_range(tracker, p.start_s, p.end_s, t_max),
                visibility=w,
            ))
    return passes


def _max_elevation(tracker: Tracker, a: float, b: float) -> tuple[float, float]:
    """Time and value of maximum elevation in [a, b], including the end points."""
    ts = np.linspace(a, b, max(3, int((b - a) / 5.0) + 1))
    els = tracker.elevation(ts)
    k = int(np.argmax(els))
    best_t, best_el = float(ts[k]), float(els[k])
    if 0 < k < len(ts) - 1:
        res = minimize_scalar(lambda x: -float(tracker.elevation(np.array([x]))[0]),
                              bounds=(ts[k - 1], ts[k + 1]), method="bounded",
                              options={"xatol": ROOT_XTOL_S})
        if -res.fun > best_el:
            best_t, best_el = float(res.x), float(-res.fun)
    return best_t, best_el


def _min_range(tracker: Tracker, a: float, b: float, t_hint: float) -> float:
    ts = np.unique(np.concatenate([np.linspace(a, b, max(3, int((b - a) / 2.0) + 1)),
                                   [t_hint]]))
    return float(np.min(tracker.geometry(ts).range_m))


PASS_TABLE_HEADER = ("pass", "start_utc", "end_utc", "duration_s", "max_elevation_deg",
                     "min_range_km", "trimmed_by")


def pass_table_rows(passes: Sequence[Pass], epoch) -> list[tuple]:
    """Rows for the pass table, converting to display units here."""
    from opticalpass.frames import times_from_epoch

    rows = []
    for p in passes:
        ts = times_from_epoch(epoch, [p.start_s, p.end_s])
        rows.append((p.number, ts[0].isot[:23], ts[1].isot[:23], round(p.duration_s, 2),
                     round(math.degrees(p.max_elevation_rad), 3),
                     round(p.min_range_m / 1e3, 3), p.trimmed_by or "-"))
    return rows
