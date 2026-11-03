"""opticalpass: plan LEO optical (laser) downlink passes over a ground station."""

from opticalpass._iers import configure_offline_iers

configure_offline_iers()

__version__ = "0.1.0"
