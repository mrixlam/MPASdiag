#!/usr/bin/env python3
"""
MPASdiag Example6: Overlaying MSLP contours and 10-m wind vectors on a 2D surface map of 2-m temperature

This script demonstrates how to create a 2D surface map of 2-meter temperature using MPASdiag's `MPASSurfacePlotter` and then overlay it with mean sea level pressure (MSLP) contours and 10-meter wind vectors. The script uses real MPAS diagnostic data, applies the new overlay capabilities, and saves the resulting plot in high resolution. It showcases the enhanced features of MPASdiag v1.1.0, including automatic unit conversion, professional branding, enhanced scientific notation, and composite plotting capabilities.

It demonstrates the following key features:
- Automatic unit conversion for 2-m temperature (Kelvin to Celsius) and MSLP (Pa to hPa)
- Professional branding with timestamp and institution logos
- Enhanced scientific notation formatting for contour levels
- Composite plotting capabilities to overlay multiple variables on a single map

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

# Load relevant MPASdiag modules 
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.wind import MPASWindPlotter

import os

# Specify the path to sample data and grid file
dataDir = '../data/u120k/diag/'
gridPath = '../data/grids/x1.40962.init.nc'

# Load unstructured MPAS data
processor = MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# Initialize Surface Plotter
plotter = MPASSurfacePlotter(verbose=True, figsize=(14, 11), dpi=300)
wind_plotter = MPASWindPlotter(figsize=(14, 11), dpi=300)

# Directly extract 2-m temperature at time index 0 and flatten for plotting
t_var = 't2m'
pres_var = 'mslp'
uwnd_var = 'u10'
vwnd_var = 'v10'


surface_var = processor.dataset[t_var].isel(Time=0)
lon, lat = processor.extract_2d_coordinates_for_variable(t_var, surface_var)

temp = processor.dataset[t_var].isel(Time=0).values.flatten()
pres = processor.dataset[pres_var].isel(Time=0).values.flatten()
uwnd = processor.dataset[uwnd_var].isel(Time=0).values.flatten()
vwnd = processor.dataset[vwnd_var].isel(Time=0).values.flatten()

# Define plot configuration
cfg = MPASConfig()

# Define map boundaries
cfg.lon_min = -130.0
cfg.lon_max = -50.0
cfg.lat_min = 20.0
cfg.lat_max = 60.0 

# -------------- Create base surface map with 2-m temperature --------------

# Main filled contour: 2-m temperature (interpolated)
fig, ax = plotter.create_surface_map(
    lon=lon,
    lat=lat,
    data=temp,
    var_name=t_var,
    lon_min=cfg.lon_min,
    lon_max=cfg.lon_max,
    lat_min=cfg.lat_min,
    lat_max=cfg.lat_max,
    plot_type='contourf',
    grid_resolution=0.1,
    title='MPASdiag Advanced: 2-m temperature (filled contour), MSLP (contours), 10-m wind (arrows)'
)

print("\n" + "="*60)
print("Testing overlay with grid_resolution=0.1")
print("="*60)

# -------------- Overlay: MSLP contours at 0.1° resolution --------------

mslp_config = {
    'data': pres,
    'var_name': pres_var,
    'plot_type': 'contour',
    'levels': [995, 1000, 1005, 1010, 1015, 1020],
    'colors': 'red',
    'linewidth': 2.0,
    'grid_resolution': 0.1,
    'add_labels': True
}

plotter.add_surface_overlay(ax, lon, lat, mslp_config, 
                            lon_min=cfg.lon_min, lon_max=cfg.lon_max, 
                            lat_min=cfg.lat_min, lat_max=cfg.lat_max)

# -------------- Overlay: 10-m wind vectors at 0.1° resolution --------------

wind_config = {
    'u_data': uwnd,
    'v_data': vwnd,
    'plot_type': 'arrows',
    'subsample': '-1',
    'colors': 'black',
    'grid_resolution': 0.1
}

wind_plotter.add_wind_overlay(ax, lon, lat, wind_config, 
                            lon_min=cfg.lon_min, lon_max=cfg.lon_max, 
                            lat_min=cfg.lat_min, lat_max=cfg.lat_max)

os.makedirs('./output', exist_ok=True)
plotter.save_plot('./output/overlay_surface_and_wind', formats=['png'])
plotter.close_plot()

print("\n✓ Test completed! Check ./output/overlay_surface_and_wind.png")
