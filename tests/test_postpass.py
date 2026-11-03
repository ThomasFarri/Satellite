import math
from dataclasses import replace

import pytest

from opticalpass.planning import run_plan
from opticalpass.postpass import analyse, generate_telemetry, predict


@pytest.fixture(scope="module")
def plan(example_config):
    cfg = replace(example_config, duration_s=86400.0)
    return run_plan(cfg)


@pytest.fixture(scope="module")
def high_pass(plan):
    return max(plan.passes, key=lambda s: s.p.max_elevation_rad).p


def test_recovers_injected_errors(plan, high_pass):
    cfg = plan.config.postpass
    tel = generate_telemetry(plan.tracker, high_pass, cfg)
    assert tel.synthetic
    res = analyse(tel, predict(plan.tracker, high_pass))
    # Stated tolerance: 0.001 deg on each bias and 0.01 s on timing, and within
    # 4 sigma of the fit's own uncertainty.
    assert abs(res.bias_az_rad - cfg.bias_az_rad) < math.radians(1e-3)
    assert abs(res.bias_el_rad - cfg.bias_el_rad) < math.radians(1e-3)
    assert abs(res.timing_offset_s - cfg.timing_offset_s) < 0.01
    assert abs(res.bias_az_rad - cfg.bias_az_rad) < 4 * res.sigma_bias_az_rad
    assert abs(res.timing_offset_s - cfg.timing_offset_s) < 4 * res.sigma_timing_s
    # Post-fit residual RMS should match the injected noise.
    assert res.rms_residual_rad == pytest.approx(cfg.noise_rad, rel=0.1)


@pytest.mark.parametrize("dt", [-2.0, 0.0, 3.5])
def test_noise_free_recovery_is_exact(plan, high_pass, dt):
    cfg = replace(plan.config.postpass, noise_rad=0.0, timing_offset_s=dt,
                  bias_az_rad=math.radians(-0.05), bias_el_rad=math.radians(0.03))
    res = analyse(generate_telemetry(plan.tracker, high_pass, cfg),
                  predict(plan.tracker, high_pass, step_s=0.25))
    assert res.timing_offset_s == pytest.approx(dt, abs=1e-3)
    assert res.bias_az_rad == pytest.approx(cfg.bias_az_rad, abs=math.radians(1e-5))
    assert res.bias_el_rad == pytest.approx(cfg.bias_el_rad, abs=math.radians(1e-5))


def test_too_few_samples(plan, high_pass):
    tel = generate_telemetry(plan.tracker, high_pass, plan.config.postpass)
    short = replace(tel, t_s=tel.t_s[:3], az_rad=tel.az_rad[:3], el_rad=tel.el_rad[:3])
    with pytest.raises(ValueError):
        analyse(short, predict(plan.tracker, high_pass))


def test_timing_limit(plan, high_pass):
    with pytest.raises(ValueError):
        generate_telemetry(plan.tracker, high_pass,
                           replace(plan.config.postpass, timing_offset_s=20.0))

