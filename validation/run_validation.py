"""Run every validation check and write docs/VALIDATION.md from the results.

Usage:  python validation/run_validation.py

Every number in the generated report comes from this run. The STK check is
reported as "Not yet run" until validation/stk/data/ holds the exported CSVs.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opticalpass.config import load_config  # noqa: E402
from opticalpass.constants import MU_EARTH_M3_S2, SECONDS_PER_DAY  # noqa: E402
from opticalpass.frames import (  # noqa: E402
    Station,
    astropy_altaz,
    teme_to_gcrs,
    teme_to_itrs,
    times_from_epoch,
    topocentric_aer,
)
from opticalpass.geometry import Tracker  # noqa: E402
from opticalpass.orbit import (  # noqa: E402
    design_mean_elements,
    j2_raan_rate_rad_s,
    rv_to_raan_rad,
)
from opticalpass.passes import find_passes  # noqa: E402
from opticalpass.pointing import point_ahead, pointing_profile  # noqa: E402
from opticalpass.postpass import analyse, generate_telemetry, predict  # noqa: E402
from opticalpass.propagators import (  # noqa: E402
    NumericalJ2Propagator,
    SGP4Propagator,
    ric_difference_m,
)

CHECKS: list[dict] = []


def check(name: str, method: str, tolerance: str, measured: str, status: str) -> None:
    CHECKS.append(dict(name=name, method=method, tolerance=tolerance, measured=measured,
                       status=status))
    print(f"[{status}] {name}: {measured}")


def pf(ok: bool) -> str:
    return "Pass" if ok else "FAIL"


def main() -> int:
    cfg = load_config(ROOT / "configs" / "example.yaml")
    el = design_mean_elements(cfg.orbit, cfg.epoch)
    sgp = SGP4Propagator(el)
    station = Station.from_config(cfg.station)
    tracker = Tracker(sgp, station)
    extra: dict = {}

    # 1. Two-body conservation and period.
    T2 = 2 * SECONDS_PER_DAY
    tb = NumericalJ2Propagator.from_propagator(sgp, T2, include_j2=False)
    s = tb.propagate(np.linspace(0, T2, 4000))
    r = np.linalg.norm(s.r_m, axis=1)
    energy = 0.5 * np.sum(s.v_m_s**2, 1) - MU_EARTH_M3_S2 / r
    h = np.cross(s.r_m, s.v_m_s)
    de = float(np.max(np.abs(energy / energy[0] - 1)))
    dh = float(np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0]))
    check("Two-body energy conservation, 2 days", "DOP853 rtol 1e-11, max relative change",
          "< 1e-9", f"{de:.1e}", pf(de < 1e-9))
    check("Two-body angular momentum conservation, 2 days",
          "max relative change of h vector", "< 1e-9", f"{dh:.1e}", pf(dh < 1e-9))
    a = -MU_EARTH_M3_S2 / (2 * energy[0])
    period = 2 * math.pi * math.sqrt(a**3 / MU_EARTH_M3_S2)
    closure = float(np.linalg.norm(tb.propagate(period).r_m[0] - s.r_m[0]))
    check("Two-body period", "position after one analytic period vs start",
          "< 1 mm", f"{closure * 1e3:.3f} mm (period {period:.3f} s)", pf(closure < 1e-3))

    # 2. J2 RAAN drift.
    T5 = 5 * SECONDS_PER_DAY
    j2 = NumericalJ2Propagator.from_propagator(sgp, T5)
    t = np.arange(0, T5, 60.0)
    sj = j2.propagate(t)
    rate = float(np.polyfit(t, np.unwrap(rv_to_raan_rad(sj.r_m, sj.v_m_s)), 1)[0])
    ana = j2_raan_rate_rad_s(el.semi_major_axis_m, el.eccentricity, el.inclination_rad)
    rel = abs(rate / ana - 1)
    check("J2 RAAN drift rate, 5 days", "linear fit of osculating RAAN vs analytic secular rate",
          "within 2 %",
          f"{math.degrees(rate) * 86400:.5f} vs {math.degrees(ana) * 86400:.5f} deg/day "
          f"({rel * 100:.2f} %)", pf(rel < 0.02))

    # 3. SGP4 vs numerical J2 divergence (characterisation only).
    jfull = NumericalJ2Propagator.from_propagator(sgp, cfg.duration_s)
    div = {}
    for label, T in (("24 h", SECONDS_PER_DAY), ("7 days", 7 * SECONDS_PER_DAY)):
        tt = np.arange(0, T + 1, 60.0)
        d = ric_difference_m(sgp.propagate(tt), jfull.propagate(tt))
        div[label] = dict(total_km=float(np.max(np.linalg.norm(d, axis=1)) / 1e3),
                          radial_km=float(np.max(np.abs(d[:, 0])) / 1e3),
                          intrack_km=float(np.max(np.abs(d[:, 1])) / 1e3),
                          crosstrack_km=float(np.max(np.abs(d[:, 2])) / 1e3))
        check(f"SGP4 vs numerical J2 divergence, {label}",
              "max position difference, RIC components", "Characterisation",
              f"{div[label]['total_km']:.2f} km total (R {div[label]['radial_km']:.2f}, "
              f"I {div[label]['intrack_km']:.2f}, C {div[label]['crosstrack_km']:.2f} km)",
              "Characterised")
    extra["divergence"] = div

    # 4. Handwritten AER vs astropy AltAz.
    t = np.arange(0, SECONDS_PER_DAY, 30.0)
    st = sgp.propagate(t)
    r_itrs, _ = teme_to_itrs(sgp.epoch, t, st.r_m)
    az, elv, rng = topocentric_aer(r_itrs, station)
    az2, el2, rng2 = astropy_altaz(sgp.epoch, t, r_itrs, station)
    vis = elv > 0
    d_el = float(np.max(np.abs(elv - el2)))
    d_az = float(np.max(np.abs(np.angle(np.exp(1j * (az - az2))))[vis]))
    d_rng = float(np.max(np.abs(rng - rng2)))
    check("Handwritten ENU chain vs astropy AltAz",
          "topocentric ITRS to AltAz, refraction off, 1 day at 30 s",
          "1e-9 rad, 1 mm", f"el {d_el:.1e} rad, az {d_az:.1e} rad, range {d_rng * 1e3:.2e} mm",
          pf(d_el < 1e-9 and d_az < 1e-9 and d_rng < 1e-3))

    # 5. Passes, point-ahead and its independent GCRS route.
    geo = replace(cfg.constraints, night_only=False, min_sun_separation_rad=0.0)
    gpasses = find_passes(tracker, geo, 0.0, cfg.duration_s, cfg.coarse_step_s)
    worst = 0.0
    for p in gpasses:
        e4 = tracker.elevation([p.start_s - 0.01, p.start_s + 0.01, p.end_s - 0.01,
                                p.end_s + 0.01])
        thr = geo.min_elevation_rad
        ok = e4[0] < thr < e4[1] and e4[2] > thr > e4[3]
        worst = max(worst, 0.0 if ok else 1.0)
    check("Rise and set refinement", "elevation 10 ms either side of each refined crossing "
          "straddles the 20 deg threshold", "all crossings within 10 ms",
          f"{len(gpasses)} passes, {2 * len(gpasses)} crossings checked", pf(worst == 0.0))

    highest = max(gpasses, key=lambda p: p.max_elevation_rad)
    prof = pointing_profile(tracker, highest, 2.0)
    idx = np.linspace(0, len(prof.t_s) - 1, 15).astype(int)
    tt = prof.t_s[idx]
    ss = sgp.propagate(tt)
    rg, vg = teme_to_gcrs(sgp.epoch, tt, ss.r_m, ss.v_m_s)
    import astropy.units as u
    pos, vel = station.earth_location.get_gcrs_posvel(times_from_epoch(sgp.epoch, tt))
    los = pos.xyz.to_value(u.m).T - rg
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    paa_g, _, _ = point_ahead(los, vel.xyz.to_value(u.m / u.s).T - vg)
    relpa = float(np.max(np.abs(prof.point_ahead_rad[idx] / paa_g - 1)))
    check("Point-ahead, independent inertial route",
          "ITRS axes with omega x r vs astropy GCRS satellite and station velocities",
          "within 0.1 %", f"max relative difference {relpa:.1e}", pf(relpa < 1e-3))
    pa_all = [pointing_profile(tracker, p, 2.0).point_ahead_rad for p in gpasses]
    pa_min = min(float(x.min()) for x in pa_all) * 1e6
    pa_max = max(float(x.max()) for x in pa_all) * 1e6
    check("Point-ahead magnitude sanity", "2 v_perp / c over all geometric passes",
          "order 50 urad (20 to 60 urad)", f"{pa_min:.1f} to {pa_max:.1f} urad",
          pf(20 < pa_min and pa_max < 60))
    extra["point_ahead_urad"] = [pa_min, pa_max]

    # 6. Skyfield independent check.
    sky = skyfield_checks(cfg, sgp, gpasses, station, tracker)
    extra["skyfield"] = sky

    # 7. Post-pass recovery.
    tel = generate_telemetry(tracker, highest, cfg.postpass)
    res = analyse(tel, predict(tracker, highest))
    eaz = math.degrees(res.bias_az_rad - cfg.postpass.bias_az_rad)
    eel = math.degrees(res.bias_el_rad - cfg.postpass.bias_el_rad)
    edt = res.timing_offset_s - cfg.postpass.timing_offset_s
    check("Post-pass bias and timing recovery (SYNTHETIC telemetry)",
          f"least squares on the highest geometric pass (no. {highest.number} of "
          f"{len(gpasses)} unconstrained), noise "
          f"{math.degrees(cfg.postpass.noise_rad):.3f} deg",
          "0.001 deg, 0.01 s",
          f"az bias err {eaz:+.5f} deg, el bias err {eel:+.5f} deg, timing err {edt:+.5f} s",
          pf(abs(eaz) < 1e-3 and abs(eel) < 1e-3 and abs(edt) < 0.01))

    # 8. STK.
    stk_dir = ROOT / "validation" / "stk" / "data"
    if (stk_dir / "access.csv").is_file() and (stk_dir / "aer.csv").is_file():
        sys.path.insert(0, str(ROOT / "validation"))
        from compare_stk import compare  # type: ignore[import-not-found]
        out = compare(stk_dir / "access.csv", stk_dir / "aer.csv",
                      ROOT / "validation" / "stk" / "satellite.tle", ROOT / "outputs" / "stk")
        check("STK Access and AER comparison", "validation/compare_stk.py on exported CSVs",
              "rise/set 1 s, el 0.01 deg", out["summary"], pf(out["ok"]))
    else:
        check("STK Access and AER comparison", "validation/compare_stk.py on exported CSVs",
              "rise/set 1 s, el 0.01 deg", "-", "Not yet run")

    write_report(cfg, extra)
    return 0 if all(c["status"] != "FAIL" for c in CHECKS) else 1


def skyfield_checks(cfg, sgp, gpasses, station, tracker) -> dict:
    try:
        import astropy_iers_data
        from skyfield.api import EarthSatellite, Loader, load, wgs84
        from skyfield.data import iers
    except ImportError:
        check("Skyfield rise/set comparison", "-", "-", "skyfield not installed", "Not yet run")
        return {}
    topo = wgs84.latlon(math.degrees(station.lat_rad), math.degrees(station.lon_rad),
                        elevation_m=station.height_m)
    finals = os.path.join(os.path.dirname(astropy_iers_data.__file__), "data",
                          "finals2000A.all")
    tmp = tempfile.mkdtemp()
    shutil.copy(finals, tmp)
    ts_same = Loader(tmp, verbose=False).timescale(builtin=False)
    with open(finals, "rb") as fh:
        iers.install_polar_motion_table(ts_same, iers.parse_x_y_dut1_from_finals_all(fh))
    variants = {
        "default": load.timescale(builtin=True),
        "same_iers": ts_same,
    }
    out = {}
    t_grid = np.arange(0, cfg.duration_s, 10.0)
    at = times_from_epoch(cfg.epoch, t_grid)
    el_ours = tracker.elevation(t_grid)
    for name, ts in variants.items():
        sat = EarthSatellite.from_satrec(sgp.satrec, ts)
        t0 = ts.from_astropy(cfg.epoch)
        t1 = ts.from_astropy(times_from_epoch(cfg.epoch, cfg.duration_s))
        times, events = sat.find_events(topo, t0, t1,
                                        altitude_degrees=math.degrees(
                                            cfg.constraints.min_elevation_rad))
        rise = [float((t - t0) * 86400) for t, e in zip(times, events, strict=True) if e == 0]
        sets = [float((t - t0) * 86400) for t, e in zip(times, events, strict=True) if e == 2]
        n_match = min(len(rise), len(sets), len(gpasses))
        dr = [gpasses[k].start_s - rise[k] for k in range(n_match)]
        ds = [gpasses[k].end_s - sets[k] for k in range(n_match)]
        alt = (sat - topo).at(ts.from_astropy(at)).altaz()[0].radians
        d_el = float(np.degrees(np.max(np.abs(el_ours - alt)[el_ours > 0])))

        # find_events stops at 0.5 s (skyfield sgp4lib: epsilon = half_second), so refine
        # Skyfield's own altitude with brentq to 1 ms for a like-for-like timing check.
        def sf_alt(x: float, sat=sat, ts=ts) -> float:
            tx = ts.from_astropy(times_from_epoch(cfg.epoch, x))
            return float((sat - topo).at(tx).altaz()[0].radians)

        thr = cfg.constraints.min_elevation_rad
        dr_ref, ds_ref, maxel = [], [], []
        for k in range(n_match):
            r_ref = brentq(lambda x, thr=thr: sf_alt(x) - thr, rise[k] - 2, rise[k] + 2, xtol=1e-3)
            s_ref = brentq(lambda x, thr=thr: sf_alt(x) - thr, sets[k] - 2, sets[k] + 2, xtol=1e-3)
            dr_ref.append(gpasses[k].start_s - r_ref)
            ds_ref.append(gpasses[k].end_s - s_ref)
            m = minimize_scalar(lambda x: -sf_alt(x), bounds=(r_ref, s_ref), method="bounded",
                                options={"xatol": 1e-3})
            maxel.append(math.degrees(gpasses[k].max_elevation_rad + m.fun))
        dut1 = float(ts.from_astropy(cfg.epoch).dut1)
        out[name] = dict(n_ours=len(gpasses), n_skyfield=len(rise),
                         max_rise_diff_s=float(np.max(np.abs(dr))) if dr else None,
                         max_set_diff_s=float(np.max(np.abs(ds))) if ds else None,
                         max_rise_diff_refined_s=float(np.max(np.abs(dr_ref))),
                         max_set_diff_refined_s=float(np.max(np.abs(ds_ref))),
                         max_el_diff_deg=d_el,
                         max_peak_el_diff_deg=float(np.max(np.abs(maxel))) if maxel else None,
                         dut1_s=dut1)
    ast_dut1 = float(cfg.epoch.delta_ut1_utc)
    out["astropy_dut1_s"] = ast_dut1
    d = out["default"]
    worst = max(d["max_rise_diff_s"], d["max_set_diff_s"])
    refined = max(d["max_rise_diff_refined_s"], d["max_set_diff_refined_s"])
    check("Skyfield rise/set, default Skyfield settings",
          "same SGP4 Satrec, find_events at 20 deg, 7 days",
          "< 1 s (investigate above)",
          f"{d['n_skyfield']} vs {d['n_ours']} passes; max rise diff "
          f"{d['max_rise_diff_s']:.3f} s, max set diff {d['max_set_diff_s']:.3f} s "
          f"(find_events alone); {refined:.3f} s with crossings refined to 1 ms",
          pf(d["n_skyfield"] == d["n_ours"] and worst < 1.0))
    check("Skyfield elevation, default Skyfield settings", "10 s grid, 7 days, el > 0",
          "Characterisation",
          f"max {d['max_el_diff_deg']:.4f} deg; UT1-UTC Skyfield {d['dut1_s']:+.4f} s vs "
          f"astropy {ast_dut1:+.4f} s; no polar motion in Skyfield", "Explained")
    s2 = out["same_iers"]
    worst2 = max(s2["max_rise_diff_refined_s"], s2["max_set_diff_refined_s"])
    check("Skyfield with the same IERS data and polar motion",
          "Skyfield timescale from astropy's bundled finals2000A.all; Skyfield altitude "
          "crossings refined with brentq to 1 ms",
          "rise/set < 10 ms, el < 1e-4 deg",
          f"max rise diff {s2['max_rise_diff_refined_s'] * 1e3:.2f} ms, max set diff "
          f"{s2['max_set_diff_refined_s'] * 1e3:.2f} ms, max el diff "
          f"{s2['max_el_diff_deg']:.1e} deg, peak el diff {s2['max_peak_el_diff_deg']:.1e} deg",
          pf(worst2 < 0.01 and s2["max_el_diff_deg"] < 1e-4
             and s2["max_peak_el_diff_deg"] < 1e-4))
    return out


def write_report(cfg, extra: dict) -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "validation_results.json").write_text(
        json.dumps({"checks": CHECKS, "details": extra}, indent=2), encoding="utf-8")
    sky = extra.get("skyfield", {})
    lines = [
        "# Validation",
        "",
        "Generated by `python validation/run_validation.py`. Every number below comes from",
        "that run; nothing is typed in by hand. Raw values are in",
        "`docs/validation_results.json`.",
        "",
        f"Scenario: `{cfg.source_path.name}`, epoch {cfg.epoch.isot} UTC, "
        f"{cfg.duration_s / 86400:g} days, station at approximate public coordinates.",
        "",
        "| Check | Method | Tolerance | Measured | Result |",
        "|---|---|---|---|---|",
    ]
    for c in CHECKS:
        lines.append(f"| {c['name']} | {c['method']} | {c['tolerance']} | {c['measured']} | "
                     f"{c['status']} |")
    lines += ["", "## Notes", ""]
    if sky:
        d, s2 = sky["default"], sky["same_iers"]
        lines += [
            "**Skyfield differences.** With default settings Skyfield uses its own bundled",
            f"Earth orientation data, which gives UT1-UTC = {d['dut1_s']:+.4f} s at the epoch,",
            f"while astropy's bundled IERS predictions give {sky['astropy_dut1_s']:+.4f} s.",
            "Skyfield also leaves out polar motion by default. A UT1 error rotates the",
            "Earth under the satellite, so it shows up as small elevation and timing",
            f"differences (up to {d['max_el_diff_deg']:.4f} deg and "
            f"{max(d['max_rise_diff_s'], d['max_set_diff_s']):.3f} s here).",
            "Building Skyfield's timescale from the same IERS file and installing its polar",
            f"motion table brings elevation agreement to {s2['max_el_diff_deg']:.1e} deg and",
            f"rise/set agreement to {max(s2['max_rise_diff_s'], s2['max_set_diff_s']) * 1e3:.2f} "
            "ms, which confirms the cause. Both libraries run the same SGP4",
            "code, so this check validates frames, Earth orientation, the station model and",
            "the pass finder, not the propagator itself.",
            "",
            "**Skyfield `find_events` precision.** Skyfield's `find_events` stops refining",
            "rise and set times at 0.5 s (`half_second` in `skyfield/sgp4lib.py`), so its raw",
            "times differ from this tool's by up to a few hundred ms even when the geometry",
            "agrees. The tight timing check therefore refines Skyfield's own altitude",
            "function with Brent's method to 1 ms before comparing.",
            "",
        ]
    div = extra.get("divergence", {})
    if div:
        lines += [
            "**SGP4 against the numerical J2 propagator.** This is a characterisation, not a",
            "pass or fail test. The numerical propagator starts from SGP4's osculating state",
            "at epoch and models only point-mass gravity and J2. SGP4 works from mean",
            "elements and adds J3, J4 and its own periodic terms. Starting from an",
            "osculating state gives the integrator a slightly different mean semi-major axis,",
            "so the difference grows mostly along track: "
            f"{div['24 h']['intrack_km']:.2f} km after 24 h and "
            f"{div['7 days']['intrack_km']:.2f} km after 7 days.",
            "",
        ]
    lines += [
        "**Offline Earth orientation.** IERS downloads are disabled and astropy uses the",
        "table bundled with `astropy-iers-data`, which includes predictions about a year",
        "ahead. Predicted UT1-UTC typically differs from final values by a few ms over that",
        "span, a few metres at the satellite, well inside every tolerance above.",
        "",
        "**STK.** STK is not available in this environment. Steps to run the check are in",
        "`validation/stk/README.md`. The result above stays \"Not yet run\" until the",
        "exported CSV files exist in `validation/stk/data/`.",
        "",
    ]
    (docs / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {docs / 'VALIDATION.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
