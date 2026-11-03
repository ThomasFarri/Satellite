"""Physical constants. SI units throughout.

Every value has its source next to it, so a reader can check it without
searching.
"""

from __future__ import annotations

import math

# Earth gravitational parameter GM [m^3/s^2].
# Source: NGA.STND.0036_1.0.0_WGS84 (2014), Table 3.1.
MU_EARTH_M3_S2: float = 3.986004418e14

# Second zonal harmonic J2 [-], unnormalised.
# Source: EGM2008 (Pavlis et al., JGR 2012), J2 = -sqrt(5) * C20_bar,
# C20_bar = -4.84165143790815e-4.
J2_EARTH: float = 1.08262668e-3

# Equatorial radius [m]. Source: NGA.STND.0036_1.0.0_WGS84, Table 3.1.
R_EARTH_EQ_M: float = 6378137.0

# Flattening [-]. Source: NGA.STND.0036_1.0.0_WGS84, Table 3.1.
WGS84_FLATTENING: float = 1.0 / 298.257223563

# First eccentricity squared, derived from the flattening above.
WGS84_E2: float = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)

# Nominal mean Earth rotation rate [rad/s].
# Source: NGA.STND.0036_1.0.0_WGS84, Table 3.1.
OMEGA_EARTH_RAD_S: float = 7.292115e-5

# Speed of light in vacuum [m/s], exact by definition.
# Source: BIPM, The International System of Units (SI Brochure), 9th ed., 2019.
C_LIGHT_M_S: float = 299792458.0

# Mean rate of the Sun in right ascension, i.e. one revolution per tropical year
# [rad/s]. This is the nodal rate a sun-synchronous orbit must match.
# Source: Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Sec. 11.4,
# tropical year 365.2421897 days.
TROPICAL_YEAR_S: float = 365.2421897 * 86400.0
SUN_SYNC_RATE_RAD_S: float = 2.0 * math.pi / TROPICAL_YEAR_S

SECONDS_PER_DAY: float = 86400.0
