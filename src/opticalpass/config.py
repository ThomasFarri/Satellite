"""Scenario configuration: load YAML, validate, convert to SI.

The YAML file uses degrees and human units because people edit it. Everything
past this module uses metres, seconds and radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from astropy.time import Time


class ConfigError(ValueError):
    """Raised when the scenario file is missing a value or a value is out of range."""


def _get(d: dict[str, Any], key: str, section: str) -> Any:
    if not isinstance(d, dict):
        raise ConfigError(f"Section '{section}' must be a mapping, got {type(d).__name__}")
    if key not in d:
        raise ConfigError(f"Missing required key '{section}.{key}'")
    return d[key]


def _num(d: dict[str, Any], key: str, section: str, lo: float | None = None,
         hi: float | None = None, default: float | None = None) -> float:
    if key not in d and default is not None:
        value = default
    else:
        value = _get(d, key, section)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{section}.{key}' must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"'{section}.{key}' must be finite, got {value}")
    if lo is not None and value < lo:
        raise ConfigError(f"'{section}.{key}' = {value} is below the minimum {lo}")
    if hi is not None and value > hi:
        raise ConfigError(f"'{section}.{key}' = {value} is above the maximum {hi}")
    return value


@dataclass(frozen=True)
class OrbitConfig:
    altitude_m: float
    eccentricity: float
    ltan_hours: float
    arg_perigee_rad: float
    mean_anomaly_rad: float
    bstar_per_earth_radii: float


@dataclass(frozen=True)
class StationConfig:
    name: str
    lat_rad: float
    lon_rad: float
    height_m: float


@dataclass(frozen=True)
class ConstraintConfig:
    min_elevation_rad: float
    min_sun_separation_rad: float
    night_only: bool
    night_sun_elevation_rad: float


@dataclass(frozen=True)
class TerminalConfig:
    """Illustrative optical terminal and link values. Not a real product."""

    tx_power_w: float
    divergence_half_angle_rad: float
    rx_aperture_diameter_m: float
    optics_efficiency: float
    zenith_transmission: float
    pointing_error_rad: float
    data_rate_bps: float
    required_rx_power_w: float


@dataclass(frozen=True)
class PostPassConfig:
    bias_az_rad: float
    bias_el_rad: float
    timing_offset_s: float
    noise_rad: float
    sample_step_s: float
    seed: int


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    epoch: Time
    duration_s: float
    coarse_step_s: float
    profile_step_s: float
    propagator: str
    orbit: OrbitConfig
    station: StationConfig
    constraints: ConstraintConfig
    terminal: TerminalConfig
    pointing_error_sweep_rad: tuple[float, ...]
    max_sessions: int
    postpass: PostPassConfig
    source_path: Path | None = field(default=None, compare=False)


_PROPAGATORS = ("sgp4", "j2")


def load_config(path: str | Path) -> ScenarioConfig:
    """Read and validate a scenario YAML file."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {p} must contain a YAML mapping at the top level")
    cfg = parse_config(raw)
    return ScenarioConfig(**{**cfg.__dict__, "source_path": p})


def parse_config(raw: dict[str, Any]) -> ScenarioConfig:
    """Validate a config mapping and convert it to SI units."""
    deg = math.radians

    sc = _get(raw, "scenario", "root")
    try:
        epoch = Time(str(_get(sc, "epoch_utc", "scenario")), scale="utc")
    except ValueError as exc:
        raise ConfigError(f"Cannot parse scenario.epoch_utc: {exc}") from exc
    duration_s = _num(sc, "duration_days", "scenario", lo=1e-3, hi=60.0) * 86400.0
    coarse_step_s = _num(sc, "coarse_step_s", "scenario", lo=1.0, hi=120.0, default=30.0)
    profile_step_s = _num(sc, "profile_step_s", "scenario", lo=0.1, hi=60.0, default=2.0)
    propagator = str(sc.get("propagator", "sgp4")).lower()
    if propagator not in _PROPAGATORS:
        raise ConfigError(f"scenario.propagator must be one of {_PROPAGATORS}, got {propagator!r}")

    ob = _get(raw, "orbit", "root")
    orbit = OrbitConfig(
        altitude_m=_num(ob, "altitude_km", "orbit", lo=160.0, hi=2000.0) * 1e3,
        eccentricity=_num(ob, "eccentricity", "orbit", lo=0.0, hi=0.05),
        ltan_hours=_num(ob, "ltan_hours", "orbit", lo=0.0, hi=24.0),
        arg_perigee_rad=deg(_num(ob, "arg_perigee_deg", "orbit", lo=0.0, hi=360.0, default=0.0)),
        mean_anomaly_rad=deg(_num(ob, "mean_anomaly_deg", "orbit", lo=0.0, hi=360.0, default=0.0)),
        bstar_per_earth_radii=_num(ob, "bstar", "orbit", lo=0.0, hi=1.0, default=0.0),
    )

    st = _get(raw, "station", "root")
    station = StationConfig(
        name=str(st.get("name", "station")),
        lat_rad=deg(_num(st, "lat_deg", "station", lo=-90.0, hi=90.0)),
        lon_rad=deg(_num(st, "lon_deg", "station", lo=-180.0, hi=360.0)),
        height_m=_num(st, "height_m", "station", lo=-500.0, hi=9000.0),
    )

    co = _get(raw, "constraints", "root")
    night_only = co.get("night_only", True)
    if not isinstance(night_only, bool):
        raise ConfigError(f"constraints.night_only must be true or false, got {night_only!r}")
    constraints = ConstraintConfig(
        min_elevation_rad=deg(_num(co, "min_elevation_deg", "constraints", lo=0.0, hi=89.0)),
        min_sun_separation_rad=deg(
            _num(co, "min_sun_separation_deg", "constraints", lo=0.0, hi=180.0)),
        night_only=night_only,
        night_sun_elevation_rad=deg(
            _num(co, "night_sun_elevation_deg", "constraints", lo=-18.0, hi=0.0, default=-6.0)),
    )

    te = _get(raw, "terminal", "root")
    terminal = TerminalConfig(
        tx_power_w=_num(te, "tx_power_w", "terminal", lo=1e-6, hi=1e3),
        divergence_half_angle_rad=_num(te, "divergence_half_angle_urad", "terminal",
                                       lo=0.1, hi=1e4) * 1e-6,
        rx_aperture_diameter_m=_num(te, "rx_aperture_diameter_m", "terminal", lo=1e-3, hi=20.0),
        optics_efficiency=_num(te, "optics_efficiency", "terminal", lo=1e-6, hi=1.0),
        zenith_transmission=_num(te, "zenith_transmission", "terminal", lo=1e-6, hi=1.0),
        pointing_error_rad=_num(te, "pointing_error_urad", "terminal", lo=0.0, hi=1e4) * 1e-6,
        data_rate_bps=_num(te, "data_rate_gbps", "terminal", lo=1e-6, hi=1e3) * 1e9,
        required_rx_power_w=_num(te, "required_rx_power_w", "terminal", lo=1e-15, hi=1e3),
    )

    sweep = te.get("pointing_error_sweep_urad", [1, 5, 10, 15, 20])
    if not isinstance(sweep, list) or not sweep:
        raise ConfigError("terminal.pointing_error_sweep_urad must be a non-empty list")
    sweep_rad = tuple(_num({"v": v}, "v", "terminal.pointing_error_sweep_urad", lo=0.0, hi=1e4)
                      * 1e-6 for v in sweep)

    pl = raw.get("planning", {}) or {}
    max_sessions = int(_num(pl, "max_sessions", "planning", lo=1, hi=1000, default=10))

    pp = raw.get("postpass", {}) or {}
    postpass = PostPassConfig(
        bias_az_rad=deg(_num(pp, "bias_az_deg", "postpass", lo=-5.0, hi=5.0, default=0.02)),
        bias_el_rad=deg(_num(pp, "bias_el_deg", "postpass", lo=-5.0, hi=5.0, default=-0.01)),
        timing_offset_s=_num(pp, "timing_offset_s", "postpass", lo=-10.0, hi=10.0, default=0.25),
        noise_rad=deg(_num(pp, "noise_deg", "postpass", lo=0.0, hi=1.0, default=0.002)),
        sample_step_s=_num(pp, "sample_step_s", "postpass", lo=0.01, hi=10.0, default=0.5),
        seed=int(_num(pp, "seed", "postpass", lo=0, hi=2**31, default=1)),
    )

    return ScenarioConfig(
        name=str(sc.get("name", "scenario")),
        epoch=epoch,
        duration_s=duration_s,
        coarse_step_s=coarse_step_s,
        profile_step_s=profile_step_s,
        propagator=propagator,
        orbit=orbit,
        station=station,
        constraints=constraints,
        terminal=terminal,
        pointing_error_sweep_rad=sweep_rad,
        max_sessions=max_sessions,
        postpass=postpass,
    )
