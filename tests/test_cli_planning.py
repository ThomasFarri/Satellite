import csv
import json

import pytest
import yaml

from opticalpass.cli import main
from opticalpass.planning import PassSummary, rank_passes, select_sessions
from tests.conftest import ROOT


def _short_config(tmp_path, overrides=None):
    with open(ROOT / "configs" / "example.yaml") as fh:
        raw = yaml.safe_load(fh)
    raw["scenario"]["duration_days"] = 1.0
    for (section, key), value in (overrides or {}).items():
        raw[section][key] = value
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_plan_command_writes_outputs(tmp_path):
    out = tmp_path / "out"
    assert main(["plan", "--config", str(_short_config(tmp_path)), "--out", str(out)]) == 0
    for name in ("passes.csv", "session_plan.csv", "pointing_profiles.csv",
                 "pointing_sensitivity.csv", "elevation_azimuth.png", "sky_plot.png",
                 "point_ahead.png", "link_margin.png", "summary.json"):
        assert (out / name).is_file(), name
    summary = json.loads((out / "summary.json").read_text())
    with open(out / "passes.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert summary["n_passes"] == len(rows) >= 1


def test_plan_with_no_passes(tmp_path):
    cfg = _short_config(tmp_path, {("constraints", "min_elevation_deg"): 89.0})
    out = tmp_path / "out"
    assert main(["plan", "--config", str(cfg), "--out", str(out)]) == 0
    with open(out / "passes.csv") as fh:
        assert len(list(csv.DictReader(fh))) == 0
    assert (out / "sky_plot.png").is_file()


def test_plan_bad_config_returns_error(tmp_path):
    cfg = _short_config(tmp_path, {("orbit", "altitude_km"): -1})
    assert main(["plan", "--config", str(cfg), "--out", str(tmp_path / "o")]) == 2


class _P:
    def __init__(self, n, start):
        self.number, self.start_s = n, start


class _L:
    def __init__(self, v):
        self.data_volume_bits = v


def test_rank_and_select():
    s = [PassSummary(_P(1, 0), None, _L(5)), PassSummary(_P(2, 10), None, _L(9)),
         PassSummary(_P(3, 20), None, _L(0)), PassSummary(_P(4, 30), None, _L(7))]
    assert [x.p.number for x in rank_passes(s)] == [2, 4, 1, 3]
    assert [x.p.number for x in select_sessions(s, 2)] == [2, 4]
    assert [x.p.number for x in select_sessions(s, 10)] == [1, 2, 4]
    with pytest.raises(ValueError):
        select_sessions(s, 0)
