"""Plots. Lines differ by style and marker as well as grey level, so every plot
stays readable when printed in greyscale."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from opticalpass.frames import times_from_epoch  # noqa: E402

_STYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v", "x", "+", "*"]
_GREYS = ["0.0", "0.35", "0.6"]


def _style_cycle():
    for k in itertools.count():
        yield dict(linestyle=_STYLES[k % 4], marker=_MARKERS[k % 8],
                   color=_GREYS[(k // 4) % 3], markevery=0.1, markersize=5, linewidth=1.4)


def _break_wraps(t: np.ndarray, az_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN where azimuth wraps through 0/360 so the plot does not draw a
    vertical line across the panel."""
    jumps = np.flatnonzero(np.abs(np.diff(az_deg)) > 180.0) + 1
    return (np.insert(t.astype(float), jumps, np.nan),
            np.insert(az_deg.astype(float), jumps, np.nan))


def _label(summary, epoch) -> str:
    t = times_from_epoch(epoch, summary.p.start_s)
    return f"Pass {summary.p.number} ({t.isot[5:16].replace('T', ' ')})"


def _empty(path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "No passes meet the constraints", ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_el_az(passes: Sequence, epoch, path: Path) -> Path:
    if not passes:
        return _empty(path, "Elevation and azimuth")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for s, st in zip(passes, _style_cycle(), strict=False):
        tm = (s.profile.t_s - s.profile.t_s[0]) / 60.0
        ax1.plot(tm, np.degrees(s.profile.el_rad), label=_label(s, epoch), **st)
        ax2.plot(*_break_wraps(tm, np.degrees(s.profile.az_rad)), **st)
    ax1.set_ylabel("Elevation [deg]")
    ax2.set_ylabel("Azimuth [deg, clockwise from N]")
    ax2.set_xlabel("Time from pass start [min]")
    ax1.set_title("Ground station elevation and azimuth per pass")
    ax1.legend(fontsize=7, ncol=2)
    for ax in (ax1, ax2):
        ax.grid(True, color="0.85")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sky(passes: Sequence, epoch, path: Path, min_el_rad: float) -> Path:
    if not passes:
        return _empty(path, "Sky plot")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for s, st in zip(passes, _style_cycle(), strict=False):
        ax.plot(s.profile.az_rad, 90 - np.degrees(s.profile.el_rad), label=_label(s, epoch), **st)
        ax.annotate("start", (s.profile.az_rad[0], 90 - math.degrees(s.profile.el_rad[0])),
                    fontsize=6)
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(theta, np.full_like(theta, 90 - math.degrees(min_el_rad)), color="0.5",
            linewidth=0.8, linestyle=(0, (1, 3)), label="Minimum elevation")
    ax.set_rlim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["90", "60", "30", "0 deg el"], fontsize=7)
    ax.set_title("Sky plot (radius = 90 - elevation, azimuth clockwise from N)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_point_ahead(passes: Sequence, epoch, path: Path) -> Path:
    if not passes:
        return _empty(path, "Point-ahead angle")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, st in zip(passes, _style_cycle(), strict=False):
        tm = (s.profile.t_s - s.profile.t_s[0]) / 60.0
        ax.plot(tm, s.profile.point_ahead_rad * 1e6, label=_label(s, epoch), **st)
    ax.set_xlabel("Time from pass start [min]")
    ax.set_ylabel("Point-ahead angle [urad]")
    ax.set_title("Point-ahead angle, 2 v_perp / c")
    ax.grid(True, color="0.85")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_link_margin(passes: Sequence, epoch, path: Path) -> Path:
    if not passes:
        return _empty(path, "Link margin")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, st in zip(passes, _style_cycle(), strict=False):
        tm = (s.link.t_s - s.link.t_s[0]) / 60.0
        ax.plot(tm, s.link.margin_db, label=_label(s, epoch), **st)
    ax.axhline(0.0, color="0.0", linewidth=1.0)
    ax.set_xlabel("Time from pass start [min]")
    ax.set_ylabel("Link margin [dB]")
    ax.set_title("Link margin (illustrative terminal values)")
    ax.grid(True, color="0.85")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sensitivity(rows: Sequence, path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    e = [r.pointing_error_rad * 1e6 for r in rows]
    ax1.plot(e, [r.volume_per_day_bits / 1e9 for r in rows], "k-o", label="Data volume per day")
    ax1.set_xlabel("Pointing error [urad]")
    ax1.set_ylabel("Data volume [Gbit/day]")
    ax2 = ax1.twinx()
    ax2.plot(e, [r.worst_min_margin_db for r in rows], color="0.45", linestyle="--",
             marker="s", label="Worst-case margin")
    ax2.set_ylabel("Worst-case margin across passes [dB]")
    ax2.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    ax1.set_title("Sensitivity to pointing error")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.grid(True, color="0.85")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_postpass(tel, res, path: Path) -> Path:
    """Residuals before and after removing the fitted bias and timing offset."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    tm = (tel.t_s - tel.t_s[0]) / 60.0
    for ax, raw, post, name in (
            (axes[0], res.raw_residual_az_rad, res.residual_az_rad, "Azimuth"),
            (axes[1], res.raw_residual_el_rad, res.residual_el_rad, "Elevation")):
        ax.plot(tm, np.degrees(raw) * 1e3, color="0.6", linestyle="-", linewidth=1.0,
                label="Measured minus predicted")
        ax.plot(tm, np.degrees(post) * 1e3, color="0.0", linestyle="none", marker=".",
                markersize=2, label="After fitted bias and timing removed")
        ax.set_ylabel(f"{name} residual [mdeg]")
        ax.grid(True, color="0.85")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Time from first sample [min]")
    axes[0].set_title("Post-pass residuals (SYNTHETIC telemetry)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
