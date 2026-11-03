"""Command line interface: ``python -m opticalpass plan --config ... --out ...``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from opticalpass.config import ConfigError, load_config
from opticalpass.passes import PASS_TABLE_HEADER, pass_table_rows
from opticalpass.planning import SESSION_HEADER, PlanResult, run_plan, session_rows


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def write_outputs(result: PlanResult, out: Path) -> list[Path]:
    from opticalpass import plots

    out.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    passes = [s.p for s in result.passes]
    pass_rows = pass_table_rows(passes, cfg.epoch)
    written = [_write_csv(out / "passes.csv", PASS_TABLE_HEADER, pass_rows)]
    written.append(_write_csv(out / "session_plan.csv", SESSION_HEADER, session_rows(result)))

    prof_rows = []
    for s in result.passes:
        pr, lk = s.profile, s.link
        for k in range(len(pr.t_s)):
            prof_rows.append((
                s.p.number, round(pr.t_s[k], 3), round(math.degrees(pr.az_rad[k]), 5),
                round(math.degrees(pr.el_rad[k]), 5),
                round(math.degrees(pr.az_rate_rad_s[k]), 5),
                round(math.degrees(pr.el_rate_rad_s[k]), 5),
                round(pr.range_m[k] / 1e3, 4), round(pr.range_rate_m_s[k], 3),
                round(math.degrees(pr.off_nadir_rad[k]), 4),
                round(pr.point_ahead_rad[k] * 1e6, 4),
                *[round(x, 6) for x in pr.point_ahead_dir_body[k]],
                *[round(x, 8) for x in pr.slew_quat[k]],
                round(lk.margin_db[k], 3),
            ))
    written.append(_write_csv(out / "pointing_profiles.csv", (
        "pass", "t_from_epoch_s", "az_deg", "el_deg", "az_rate_deg_s", "el_rate_deg_s",
        "range_km", "range_rate_m_s", "off_nadir_deg", "point_ahead_urad",
        "pa_dir_body_x", "pa_dir_body_y", "pa_dir_body_z",
        "slew_q_w", "slew_q_x", "slew_q_y", "slew_q_z", "link_margin_db"), prof_rows))

    written.append(_write_csv(out / "pointing_sensitivity.csv", (
        "pointing_error_urad", "pointing_loss_db", "total_volume_gbit", "volume_gbit_per_day",
        "worst_min_margin_db", "positive_margin_s"), [
        (round(r.pointing_error_rad * 1e6, 3), round(r.pointing_loss_db, 3),
         round(r.total_volume_bits / 1e9, 2), round(r.volume_per_day_bits / 1e9, 2),
         round(r.worst_min_margin_db, 2), round(r.positive_margin_s, 1))
        for r in result.sensitivity]))

    chrono = sorted(result.passes, key=lambda s: s.p.start_s)
    written += [
        plots.plot_el_az(chrono, cfg.epoch, out / "elevation_azimuth.png"),
        plots.plot_sky(chrono, cfg.epoch, out / "sky_plot.png", cfg.constraints.min_elevation_rad),
        plots.plot_point_ahead(chrono, cfg.epoch, out / "point_ahead.png"),
        plots.plot_link_margin(chrono, cfg.epoch, out / "link_margin.png"),
        plots.plot_sensitivity(result.sensitivity, out / "pointing_sensitivity.png"),
    ]
    summary = summary_dict(result)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(out / "summary.json")
    return written


def summary_dict(result: PlanResult) -> dict:
    el = result.elements
    profiles = [s.profile for s in result.passes]
    return {
        "scenario": result.config.name,
        "propagator": result.config.propagator,
        "epoch_utc": result.config.epoch.isot,
        "duration_days": result.config.duration_s / 86400.0,
        "inclination_deg": math.degrees(el.inclination_rad),
        "raan_deg": math.degrees(el.raan_rad),
        "n_passes": len(result.passes),
        "n_sessions": len(result.session_plan()),
        "total_volume_gbit": result.total_volume_bits / 1e9,
        "volume_per_day_gbit": result.volume_per_day_bits / 1e9,
        "point_ahead_urad_range": [min(float(p.point_ahead_rad.min()) for p in profiles) * 1e6,
                                   max(float(p.point_ahead_rad.max()) for p in profiles) * 1e6]
        if profiles else None,
        "peak_az_rate_deg_s": max((math.degrees(p.peak_az_rate_rad_s) for p in profiles),
                                  default=None),
        "max_off_nadir_deg": max((math.degrees(float(p.off_nadir_rad.max())) for p in profiles),
                                 default=None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opticalpass", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Find passes, compute pointing and link, write outputs")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--out", required=True, type=Path)
    plan.add_argument("--propagator", choices=["sgp4", "j2"], default=None,
                      help="Override the propagator set in the config")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if args.propagator:
        from dataclasses import replace
        cfg = replace(cfg, propagator=args.propagator)

    result = run_plan(cfg)
    files = write_outputs(result, args.out)
    s = summary_dict(result)
    print(f"Scenario {s['scenario']} ({s['propagator']}), {s['duration_days']:g} days from "
          f"{s['epoch_utc']}")
    print(f"Orbit: i = {s['inclination_deg']:.3f} deg, RAAN = {s['raan_deg']:.3f} deg")
    print(f"Passes meeting constraints: {s['n_passes']}, sessions selected: {s['n_sessions']}")
    print(f"Data volume: {s['total_volume_gbit']:.1f} Gbit total, "
          f"{s['volume_per_day_gbit']:.1f} Gbit/day (illustrative link values)")
    print(f"Wrote {len(files)} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
