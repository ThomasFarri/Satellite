import copy
import math

import pytest
import yaml

from opticalpass.config import ConfigError, parse_config
from opticalpass.constants import SUN_SYNC_RATE_RAD_S
from opticalpass.orbit import (
    design_mean_elements,
    j2_raan_rate_rad_s,
    sun_right_ascension_teme_rad,
    sun_synchronous_inclination_rad,
)
from tests.conftest import ROOT


def _raw():
    with open(ROOT / "configs" / "example.yaml") as fh:
        return yaml.safe_load(fh)


def test_example_config_units(example_config):
    c = example_config
    assert c.duration_s == pytest.approx(7 * 86400)
    assert c.orbit.altitude_m == pytest.approx(500e3)
    assert c.constraints.min_elevation_rad == pytest.approx(math.radians(20))
    assert c.terminal.divergence_half_angle_rad == pytest.approx(30e-6)


@pytest.mark.parametrize("section,key,value", [
    ("constraints", "min_elevation_deg", 95),
    ("orbit", "altitude_km", -5),
    ("terminal", "tx_power_w", "one"),
    ("scenario", "propagator", "kepler"),
])
def test_bad_values_raise(section, key, value):
    raw = copy.deepcopy(_raw())
    raw[section][key] = value
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_missing_key_raises():
    raw = copy.deepcopy(_raw())
    del raw["station"]["lat_deg"]
    with pytest.raises(ConfigError, match="station.lat_deg"):
        parse_config(raw)


def test_sun_sync_inclination_matches_sun_rate():
    a = 6378137.0 + 500e3
    i = sun_synchronous_inclination_rad(a)
    assert math.degrees(i) == pytest.approx(97.40, abs=0.02)
    assert j2_raan_rate_rad_s(a, 0.0, i) == pytest.approx(SUN_SYNC_RATE_RAD_S, rel=1e-12)


def test_sun_sync_impossible_orbit():
    with pytest.raises(ValueError):
        sun_synchronous_inclination_rad(6378137.0 + 8000e3)


def test_raan_is_ltan_offset_from_sun(example_config):
    m = design_mean_elements(example_config.orbit, example_config.epoch)
    ra_sun = sun_right_ascension_teme_rad(example_config.epoch)
    diff = (m.raan_rad - ra_sun) % (2 * math.pi)
    assert math.degrees(diff) == pytest.approx(270.0, abs=1e-9)  # 06:00 LTAN = -90 deg
    # Sun RA on 1 March is about 22h47m, i.e. near 341.7 deg.
    assert math.degrees(ra_sun) == pytest.approx(341.7, abs=0.5)
