"""Compare STK Access and AER exports with this tool.

Usage:
    python validation/compare_stk.py --access validation/stk/data/access.csv \
        --aer validation/stk/data/aer.csv --out outputs/stk

Both sides use the same TLE (validation/stk/satellite.tle) and SGP4 with WGS72
constants, so differences come from frames, Earth orientation data, the
station model and root finding, not from the orbit.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astropy.time import Time  # noqa: E402

from opticalpass.config import load_config  # noqa: E402
from opticalpass.frames import Station  # noqa: E402
from opticalpass.geometry import Tracker  # noqa: E402
from opticalpass.passes import find_passes  # noqa: E402
from opticalpass.propagators import SGP4Propagator  # noqa: E402

TOL_RISE_SET_S = 1.0
TOL_EL_DEG = 0.01

_TIME_FORMATS = ("%d %b %Y %H:%M:%S.%f", "%d %b %Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                 "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")


def parse_stk_time(text: str) -> Time:
    """Parse STK UTCG ('1 Mar 2027 06:25:20.028') or ISO time strings as UTC."""
    s = text.strip().strip('"')
    for fmt in _TIME_FORMATS:
        try:
            return Time(datetime.strptime(s, fmt), scale="utc")
        except ValueError:
            continue
    raise ValueError(f"Unrecognised time format: {text!r}")


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.reader(lines))
    if not rows:
        raise ValueError(f"{path} has no data")
    return [h.strip() for h in rows[0]], rows[1:]


def _col(header: list[str], *keys: str) -> int:
    for i, h in enumerate(header):
        if all(k.lower() in h.lower() for k in keys):
            return i
    raise ValueError(f"No column containing {keys} in header {header}")


def read_access(path: Path) -> list[tuple[Time, Time]]:
    header, rows = _read_rows(path)
    i0, i1 = _col(header, "start", "time"), _col(header, "stop", "time")
    out = []
    for r in rows:
        try:
            out.append((parse_stk_time(r[i0]), parse_stk_time(r[i1])))
        except (ValueError, IndexError):
            continue  # STK appends statistics rows (Min, Max, Mean) that are not accesses
    if not out:
        raise ValueError(f"No access intervals parsed from {path}")
    return out


def read_aer(path: Path) -> tuple[Time, np.ndarray, np.ndarray, np.ndarray]:
    header, rows = _read_rows(path)
    it, ia = _col(header, "time"), _col(header, "azimuth")
    ie, ir = _col(header, "elevation"), _col(header, "range")
    t, az, el, rng = [], [], [], []
    for r in rows:
        try:
            tt = parse_stk_time(r[it])
            vals = float(r[ia]), float(r[ie]), float(r[ir])
        except (ValueError, IndexError):
            continue
        t.append(tt)
        az.append(vals[0])
        el.append(vals[1])
        rng.append(vals[2])
    if not t:
        raise ValueError(f"No AER samples parsed from {path}")
    return Time(t), np.array(az), np.array(el), np.array(rng)


def compare(access_csv: Path, aer_csv: Path, tle_path: Path, out_dir: Path,
            config_path: Path | None = None) -> dict:
    cfg = load_config(config_path or ROOT / "configs" / "example.yaml")
    lines = [ln for ln in tle_path.read_text().splitlines() if ln.startswith(("1 ", "2 "))]
    if len(lines) != 2:
        raise ValueError(f"{tle_path} must contain TLE lines 1 and 2")
    prop = SGP4Propagator.from_tle(lines[0], lines[1], cfg.epoch)
    tracker = Tracker(prop, Station.from_config(cfg.station))
    geo = replace(cfg.constraints, night_only=False, min_sun_separation_rad=0.0)

    access = read_access(access_csv)
    t_lo = min((a[0] - cfg.epoch).sec for a in access) - 600.0
    t_hi = max((a[1] - cfg.epoch).sec for a in access) + 600.0
    passes = find_passes(tracker, geo, max(t_lo, 0.0), t_hi, cfg.coarse_step_s)

    acc_rows, d_all = [], []
    for k, (a0, a1) in enumerate(access, start=1):
        s0, s1 = (a0 - cfg.epoch).sec, (a1 - cfg.epoch).sec
        if not passes:
            acc_rows.append((k, a0.isot, a1.isot, "", "", "no matching pass"))
            continue
        p = min(passes, key=lambda x: abs(x.start_s - s0))
        dr, ds = p.start_s - s0, p.end_s - s1
        clipped = p.start_label == "window_start" or p.end_label == "window_end"
        acc_rows.append((k, a0.isot, a1.isot, round(dr, 4), round(ds, 4),
                         "clipped by window" if clipped else ""))
        if not clipped:
            d_all += [dr, ds]

    t_aer, az_s, el_s, rng_s = read_aer(aer_csv)
    ts = (t_aer - cfg.epoch).sec
    g = tracker.geometry(ts)
    d_az = (np.degrees(g.az_rad) - az_s + 180.0) % 360.0 - 180.0
    d_el = np.degrees(g.el_rad) - el_s
    d_rng = g.range_m / 1e3 - rng_s

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stk_access_residuals.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("access", "stk_start_utc", "stk_stop_utc", "start_diff_s", "stop_diff_s",
                    "note"))
        w.writerows(acc_rows)
    with (out_dir / "stk_aer_residuals.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("time_utc", "d_az_deg", "d_el_deg", "d_range_km"))
        for k in range(ts.size):
            w.writerow((t_aer[k].isot, d_az[k], d_el[k], d_rng[k]))
    _plot(ts, d_az, d_el, d_rng, out_dir / "stk_residuals.png")

    max_t = float(np.max(np.abs(d_all))) if d_all else float("nan")
    max_el = float(np.max(np.abs(d_el)))
    ok = bool(d_all) and max_t < TOL_RISE_SET_S and max_el < TOL_EL_DEG
    summary = (f"{len(access)} accesses, max rise/set diff {max_t:.3f} s; "
               f"{ts.size} AER samples, max |d_el| {max_el:.4f} deg, "
               f"max |d_az| {float(np.max(np.abs(d_az))):.4f} deg, "
               f"max |d_range| {float(np.max(np.abs(d_rng))) * 1e3:.1f} m")
    return dict(ok=ok, summary=summary, max_rise_set_diff_s=max_t, max_el_diff_deg=max_el)


def _plot(ts, d_az, d_el, d_rng, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    hours = ts / 3600.0
    for ax, y, lab, mk in ((axes[0], d_az * 3600, "Azimuth [arcsec]", "o"),
                           (axes[1], d_el * 3600, "Elevation [arcsec]", "s"),
                           (axes[2], d_rng * 1e3, "Range [m]", "^")):
        ax.plot(hours, y, linestyle="none", marker=mk, markersize=2, color="0.0")
        ax.set_ylabel(lab)
        ax.grid(True, color="0.85")
    axes[0].set_title("This tool minus STK")
    axes[2].set_xlabel("Time from scenario epoch [h]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--access", type=Path, required=True)
    ap.add_argument("--aer", type=Path, required=True)
    ap.add_argument("--tle", type=Path, default=ROOT / "validation" / "stk" / "satellite.tle")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "stk")
    args = ap.parse_args()
    res = compare(args.access, args.aer, args.tle, args.out, args.config)
    print(("PASS: " if res["ok"] else "FAIL: ") + res["summary"])
    print(f"Residual tables and plot written to {args.out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
