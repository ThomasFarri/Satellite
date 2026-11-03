import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opticalpass.frames import Station
from opticalpass.geometry import Tracker
from opticalpass.orbit import design_mean_elements
from opticalpass.passes import Interval, find_intervals_above, find_passes, intersect
from opticalpass.propagators import build_propagator

TOL_S = 0.010  # requirement: rise and set refined to within 10 ms


def sine(t):
    return np.sin(2 * np.pi * np.asarray(t) / 100.0) - 0.5


def test_intervals_match_analytic_roots():
    iv = find_intervals_above(sine, 0.0, 300.0, 7.0)
    assert len(iv) == 3
    for k, x in enumerate(iv):
        assert x.start_s == pytest.approx(100 * k + 100 / 12, abs=TOL_S)
        assert x.end_s == pytest.approx(100 * k + 500 / 12, abs=TOL_S)
        assert (x.start_label, x.end_label) == ("rise", "set")


def test_pass_underway_at_start_and_in_progress_at_end():
    iv = find_intervals_above(sine, 20.0, 230.0, 7.0)
    assert iv[0].start_s == 20.0 and iv[0].start_label == "window_start"
    assert iv[-1].end_s == 230.0 and iv[-1].end_label == "window_end"


def test_no_passes():
    assert find_intervals_above(lambda t: -np.ones_like(t), 0.0, 1000.0, 30.0) == []


def test_near_miss_peak_between_samples_is_found():
    def g(t):
        return 0.02 - ((np.asarray(t) - 45.0) / 20.0) ** 2
    assert find_intervals_above(g, 0.0, 120.0, 30.0) == []
    iv = find_intervals_above(g, 0.0, 120.0, 30.0, near_miss_margin=1.0)
    half = 20.0 * math.sqrt(0.02)
    assert len(iv) == 1
    assert iv[0].start_s == pytest.approx(45 - half, abs=TOL_S)
    assert iv[0].end_s == pytest.approx(45 + half, abs=TOL_S)


def test_bad_window_raises():
    with pytest.raises(ValueError):
        find_intervals_above(sine, 10.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        find_intervals_above(lambda t: np.full_like(t, np.nan), 0.0, 5.0, 1.0)


def test_intersect_keeps_labels():
    a = [Interval(0, 10, "rise", "set")]
    b = [Interval(-5, 3, "x", "night"), Interval(6, 20, "night", "y")]
    out = intersect(a, b)
    assert [(i.start_s, i.end_s, i.start_label, i.end_label) for i in out] == [
        (0, 3, "rise", "night"), (6, 10, "night", "set")]


class StubTracker:
    """Synthetic geometry: one pass from t=100 to 300 s, peak 60 deg at 200 s,
    Sun separation dips below 30 deg between 180 and 220 s, always night."""

    def elevation(self, t):
        t = np.asarray(t, dtype=float)
        return math.radians(20) + math.radians(40) * (1 - ((t - 200) / 100) ** 2)

    def sun_separation(self, t):
        t = np.asarray(t, dtype=float)
        return math.radians(30) + math.radians(1) * (np.abs(t - 200) - 20)

    def sun_elevation(self, t):
        return np.full_like(np.asarray(t, dtype=float), math.radians(-30))

    def geometry(self, t):
        el = self.elevation(t)
        return SimpleNamespace(range_m=500e3 / np.sin(np.clip(el, 0.1, None)))


def test_constraint_splits_pass(example_config):
    cons = example_config.constraints
    passes = find_passes(StubTracker(), cons, 0.0, 400.0, coarse_step_s=30.0)
    assert len(passes) == 2
    p1, p2 = passes
    assert p1.start_s == pytest.approx(100, abs=TOL_S) and p1.start_label == "rise"
    assert p1.end_s == pytest.approx(180, abs=TOL_S) and p1.end_label == "sun_separation"
    assert p2.start_s == pytest.approx(220, abs=TOL_S) and p2.end_s == pytest.approx(300, abs=TOL_S)
    assert p1.trimmed_by == "sun_separation" and p2.trimmed_by == "sun_separation"


# ----------------------------------------------------------- real satellite geometry

@pytest.fixture(scope="module")
def tracker(example_config):
    prop = build_propagator("sgp4", design_mean_elements(example_config.orbit,
                                                          example_config.epoch), 2 * 86400.0)
    return Tracker(prop, Station.from_config(example_config.station))


@pytest.fixture(scope="module")
def geometric(example_config):
    return replace(example_config.constraints, night_only=False, min_sun_separation_rad=0.0)


def test_real_rise_set_hit_threshold(tracker, geometric):
    passes = find_passes(tracker, geometric, 0.0, 86400.0)
    assert len(passes) >= 2
    thr = geometric.min_elevation_rad
    for p in passes:
        el = tracker.elevation([p.start_s - TOL_S, p.start_s + TOL_S, p.end_s - TOL_S,
                                p.end_s + TOL_S])
        assert el[0] < thr < el[1] and el[2] > thr > el[3]
        assert p.max_elevation_rad > thr


def test_real_window_starting_and_ending_mid_pass(tracker, geometric):
    full = find_passes(tracker, geometric, 0.0, 86400.0)
    p = full[1]
    mid = 0.5 * (p.start_s + p.end_s)
    clipped = find_passes(tracker, geometric, mid, mid + 3600.0)
    assert clipped[0].start_s == mid and clipped[0].start_label == "window_start"
    assert clipped[0].end_s == pytest.approx(p.end_s, abs=TOL_S)
    clipped_end = find_passes(tracker, geometric, p.start_s - 600.0, mid)
    assert clipped_end[-1].end_label == "window_end"


def test_real_window_with_no_passes(tracker, geometric):
    full = find_passes(tracker, geometric, 0.0, 86400.0)
    gap_start = full[0].end_s + 60.0
    gap_end = full[1].start_s - 60.0
    assert find_passes(tracker, geometric, gap_start, gap_end) == []
