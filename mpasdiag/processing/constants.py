#!/usr/bin/env python3

"""
Shared constants for the mpasdiag.processing package.

Place commonly reused literal messages and unit strings here to avoid
duplication across modules.
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
KG_PER_M3 = "kg/m^3"
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