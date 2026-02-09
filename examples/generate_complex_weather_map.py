#!/usr/bin/env python3

"""
MPASdiag Example1: Complex Weather Map

This script demonstrates advanced composite plotting by overlaying multiple meteorological
variables at 850 hPa level:
- Specific humidity (shaded background)
- Wind vectors (barbs)
- Mean sea level pressure (contour lines)

Features showcased:
- Multi-variable composite plotting
- Automatic unit conversion (kg/kg→g/kg, Pa→hPa, m→gpm)
- Professional meteorological visualization
- Overlay techniques for different plot types
- Enhanced scientific notation and labeling

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union

# Load relevant MPASdiag modules 
from mpasdiag.processing import UnitConverter
from mpasdiag.processing import MPAS2DProcessor
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.visualization.surface import MPASSurfacePlotter

# Specify the path to sample data and grid file
dataDir = '../data/u120k/diag/'
gridPath = '../data/grids/x1.40962.init.nc'

# Create output directory if it doesn't exist
os.makedirs('./output', exist_ok=True)

# Load unstructured MPAS data
processor = MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# Initialize Surface and Wind Plotter
plotter = MPASSurfacePlotter(verbose=True, figsize=(14, 13), dpi=300)
wind_plotter = MPASWindPlotter(figsize=(14, 13), dpi=300)

# Define variable names for mean sea level pressure, specific humidity, and 10-m wind components
pres_var = 'mslp'
uwnd_var = 'u10'
vwnd_var = 'v10'
shum_var = 'q2'

# Extract 2D coordinates and variable data for the surface plot
surface_var = processor.dataset[pres_var].isel(Time=0)
lon, lat = processor.extract_2d_coordinates_for_variable(pres_var, surface_var)

# Extract variable data for the surface plot and overlays (flattening for contour and wind plotting)
pres = processor.dataset[pres_var].isel(Time=0).values.flatten()
uwnd = processor.dataset[uwnd_var].isel(Time=0).values.flatten()
vwnd = processor.dataset[vwnd_var].isel(Time=0).values.flatten()
shum = processor.dataset[shum_var].isel(Time=0).values.flatten()

# Define plot configuration
cfg = MPASConfig()

# Define map boundaries
cfg.lon_min = -130.0
cfg.lon_max = -50.0
cfg.lat_min = 10.0
cfg.lat_max = 60.0 

# -------------- Create base surface map with 2-m specific humidity --------------

# Main filled contour: 2-m specific humidity (interpolated)
fig, ax = plotter.create_surface_map(
    lon=lon,
    lat=lat,
    data=shum,
    var_name=shum_var,
    lon_min=cfg.lon_min,
    lon_max=cfg.lon_max,
    lat_min=cfg.lat_min,
    lat_max=cfg.lat_max,
    plot_type='contourf',
    grid_resolution=0.1,
    title='MPASdiag Advanced: 2-m specific humidity (filled contour), MSLP (contours), 10-m wind (barbs)'
)

print("\n" + "="*60)
print("Testing overlay with grid_resolution=0.1")
print("="*60)

# -------------- Overlay: MSLP contours at 0.1° resolution --------------

# Define overlay configuration for MSLP contours
mslp_config = {
    'data': pres,
    'var_name': pres_var,
    'plot_type': 'contour',
    'levels': [940, 945, 950, 955, 960, 965, 970, 975, 980, 985, 990, 995, 1000, 1005, 1010, 1015, 1020, 1025, 1030],
    'colors': 'red',
    'linewidth': 2.0,
    'grid_resolution': 0.1,
    'add_labels': True
}

# Add MSLP contours as an overlay on the existing surface map
plotter.add_surface_overlay(ax, lon, lat, mslp_config, 
                            lon_min=cfg.lon_min, lon_max=cfg.lon_max, 
                            lat_min=cfg.lat_min, lat_max=cfg.lat_max)

# -------------- Overlay: 10-m wind vectors at 0.1° resolution --------------

# Define overlay configuration for 10-m wind vectors
wind_config = {
    'u_data': uwnd,
    'v_data': vwnd,
    'plot_type': 'barbs',
    'subsample': '-1',
    'colors': 'black',
    'grid_resolution': 0.1
}

# Add 10-m wind vectors as an overlay on the existing surface map
wind_plotter.add_wind_overlay(ax, lon, lat, wind_config, 
                            lon_min=cfg.lon_min, lon_max=cfg.lon_max, 
                            lat_min=cfg.lat_min, lat_max=cfg.lat_max)

# Save the final plot with the filled contour, contour overlay, and wind vector overlay
plotter.save_plot('./output/example_moisture_transport_with_mslp', formats=['png'])
plotter.close_plot()

print("Plot saved to ./output/example_moisture_transport_with_mslp.png")
