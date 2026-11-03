"""Offline Earth orientation setup.

Astropy normally downloads fresh IERS Earth orientation parameters (UT1-UTC and
polar motion). Downloads make runs depend on the network and on the day they
happen, so we switch them off and use the table shipped in the
``astropy-iers-data`` package instead. That table includes IERS predictions about
a year ahead of its release date. The cost is that predicted UT1-UTC drifts from
the final values by up to a few milliseconds over a year, which moves a LEO
satellite's apparent position by a few metres at most. That is far below every
tolerance used in this project.
"""

from __future__ import annotations

from astropy.utils import iers

_CONFIGURED = False


def configure_offline_iers() -> None:
    """Disable IERS downloads. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    iers.conf.auto_download = False
    # Without downloads the bundled table ages; staleness is expected, not an error.
    iers.conf.auto_max_age = None
    _CONFIGURED = True
