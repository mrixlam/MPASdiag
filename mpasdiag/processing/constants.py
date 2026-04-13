#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Shared Constants and Messages

This module defines shared constants, configuration parameters, and standardized messages for use across all MPAS diagnostic processing modules. It serves as a centralized repository for values such as default file paths, variable names, geographic bounds, and common error messages to ensure consistency and maintainability throughout the MPASdiag codebase. By centralizing these constants, the module promotes code reuse and simplifies updates to key parameters without requiring changes in multiple locations. The constants defined here are intended to be used by various processing classes and functions to maintain uniformity in handling MPAS data and to provide clear, consistent messaging for users when issues arise during processing. This module is an essential component of the MPASdiag framework, enabling efficient management of shared values and improving the overall robustness of the diagnostic processing workflow. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

DIAG_GLOB = "diag*.nc"
MPASOUT_GLOB = "mpasout*.nc"

M2_PER_S2 = "m^2/s^2"
MM = "mm"
MM_PER_HR = "mm/hr"
INCHES_PER_HR = "in/hr"
MM_PER_DAY = "mm/day"
DBZ = "dBZ"
CELSIUS = "°C"
FAHRENHEIT = "°F"
KELVIN = "K"
M_PER_S = "m/s"
KNOT = "kt"
PA = "Pa"
HPA = "hPa"
MB = "mb"
PERCENT = "%"
METER = "m"
KILOMETER = "km"
KM_PER_HR = "km/h"
MILES_PER_HR = "mph"
FEET = "ft"
M2_PER_S = "m^2/s"
G_PER_KG = "g/kg"
KG_PER_KG = "kg/kg"
KG_PER_M2 = "kg/m^2"
KG_PER_M3 = "kg/m^3"
KG_PER_M_PER_S = "kg/(m s)"
MICRONS = "microns"
M3_PER_M3 = "m^3/m^3"
J_PER_KG = "J/kg"
PER_KG = "1/kg"
PER_S = "1/s"
W_PER_M2 = "W/m^2"
NOUNIT = ""

DATASET_NOT_LOADED_MSG = "Dataset not loaded. Call load_2d_data() or load_3d_data() first."
PERFORMANCE_MONITOR_MSG = "Performance monitor must be initialized"
DATASET_NOT_LOADED_3D_MSG = "Dataset not loaded. Call load_3d_data() first."
COORDS_FALLBACK_MSG = "Workers will extract coordinates individually"
PRELOAD_COORDS_MSG = "Pre-loading coordinates into cache..."
# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = 9.80665          # Gravitational acceleration (m/s²)
P0_REF_PA = 100000.0       # Reference pressure for potential temperature (Pa)
KAPPA = 0.2854             # Poisson constant R_d / c_p (dimensionless)
Rv_OVER_Rd = 1.608         # Ratio of gas constants: water vapour / dry air
EPSILON_RD_RV = 0.622      # Ratio of molar masses: Rd / Rv

# ---------------------------------------------------------------------------
# Dimension names
# ---------------------------------------------------------------------------
NVERT_LEVELS_DIM = 'nVertLevels'

# ---------------------------------------------------------------------------
# Diagnostic unit strings
# ---------------------------------------------------------------------------
WIND_SPEED_UNITS = 'm s^{-1}'

# ---------------------------------------------------------------------------
# Log / status message strings
# ---------------------------------------------------------------------------
COORDS_FALLBACK_MSG = "Workers will extract coordinates individually"
PRELOAD_COORDS_MSG  = "Pre-loading coordinates into cache..."

# ---------------------------------------------------------------------------
# CLI help strings
# ---------------------------------------------------------------------------
BATCH_MODE_HELP = 'Process all time steps in batch mode'

# ---------------------------------------------------------------------------
# Required variable lists for different diagnostic types
# ---------------------------------------------------------------------------
PRECIP_REQUIRED_VARS: dict = {
    'total':  ['rainc', 'rainnc'],
    'rainc':  ['rainc'],
    'rainnc': ['rainnc'],
}

CROSS_SECTION_AUX_VARS: list = [
    'pressure', 'pressure_p', 'pressure_base',
    'zgrid', 'height', 'fzp', 'surface_pressure',
]
