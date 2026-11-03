"""End-to-end planning run and pass ranking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from opticalpass.config import ScenarioConfig
from opticalpass.frames import Station, times_from_epoch
from opticalpass.geometry import Tracker
from opticalpass.link import LinkResult, SensitivityRow, link_for_profile, pointing_sensitivity
from opticalpass.orbit import MeanElements, design_mean_elements
from opticalpass.passes import Pass, find_passes
from opticalpass.pointing import PointingProfile, pointing_profile
from opticalpass.propagators import build_propagator


@dataclass(frozen=True)
class PassSummary:
    p: Pass
    profile: PointingProfile
    link: LinkResult

    @property
    def volume_bits(self) -> float:
        return self.link.data_volume_bits


@dataclass(frozen=True)
class PlanResult:
    config: ScenarioConfig
    elements: MeanElements
    tracker: Tracker
    passes: list[PassSummary]
    sensitivity: list[SensitivityRow]

    @property
    def total_volume_bits(self) -> float:
        return sum(s.volume_bits for s in self.passes)

    @property
    def volume_per_day_bits(self) -> float:
        return self.total_volume_bits / (self.config.duration_s / 86400.0)

    def ranked(self) -> list[PassSummary]:
        return rank_passes(self.passes)

    def session_plan(self) -> list[PassSummary]:
        return select_sessions(self.passes, self.config.max_sessions)


def rank_passes(passes: Sequence[PassSummary]) -> list[PassSummary]:
    """Highest deliverable data volume first; ties broken by earlier start."""
    return sorted(passes, key=lambda s: (-s.volume_bits, s.p.start_s))


def select_sessions(passes: Sequence[PassSummary], max_sessions: int) -> list[PassSummary]:
    """Top passes by volume, excluding passes with no deliverable data, in time order.

    One station can only track one satellite at a time, and passes of a single
    satellite never overlap, so no conflict resolution is needed here.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be at least 1")
    chosen = [s for s in rank_passes(passes) if s.volume_bits > 0][:max_sessions]
    return sorted(chosen, key=lambda s: s.p.start_s)


def run_plan(cfg: ScenarioConfig) -> PlanResult:
    elements = design_mean_elements(cfg.orbit, cfg.epoch)
    prop = build_propagator(cfg.propagator, elements, cfg.duration_s)
    tracker = Tracker(prop, Station.from_config(cfg.station))
    passes = find_passes(tracker, cfg.constraints, 0.0, cfg.duration_s, cfg.coarse_step_s)
    summaries = []
    for p in passes:
        prof = pointing_profile(tracker, p, cfg.profile_step_s)
        link = link_for_profile(p.number, prof.t_s, prof.range_m, prof.el_rad, cfg.terminal)
        summaries.append(PassSummary(p, prof, link))
    sens = pointing_sensitivity(
        [(s.p.number, s.profile.t_s, s.profile.range_m, s.profile.el_rad) for s in summaries],
        cfg.terminal, cfg.pointing_error_sweep_rad, cfg.duration_s / 86400.0)
    return PlanResult(cfg, elements, tracker, summaries, sens)


SESSION_HEADER = ("rank", "pass", "start_utc", "end_utc", "duration_s", "volume_gbit",
                  "positive_margin_s", "min_margin_db", "max_margin_db", "max_elevation_deg",
                  "max_off_nadir_deg", "peak_az_rate_deg_s", "peak_el_rate_deg_s",
                  "max_point_ahead_urad", "trimmed_by")


def session_rows(result: PlanResult) -> list[tuple]:
    ranks = {s.p.number: k + 1 for k, s in enumerate(result.ranked())}
    rows = []
    for s in result.session_plan():
        t = times_from_epoch(result.config.epoch, [s.p.start_s, s.p.end_s])
        pr = s.profile
        rows.append((
            ranks[s.p.number], s.p.number, t[0].isot[:23], t[1].isot[:23],
            round(s.p.duration_s, 2), round(s.volume_bits / 1e9, 2),
            round(s.link.positive_margin_s, 2), round(s.link.min_margin_db, 2),
            round(s.link.max_margin_db, 2), round(math.degrees(s.p.max_elevation_rad), 3),
            round(math.degrees(float(pr.off_nadir_rad.max())), 3),
            round(math.degrees(pr.peak_az_rate_rad_s), 4),
            round(math.degrees(pr.peak_el_rate_rad_s), 4),
            round(float(pr.point_ahead_rad.max()) * 1e6, 3), s.p.trimmed_by or "-",
        ))
    return rows
