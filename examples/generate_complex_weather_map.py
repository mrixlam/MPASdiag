#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Example VII: Complex Weather Map

This example demonstrates how to create a complex weather map by overlaying multiple variables from MPAS 2D model output. The example extracts 2-meter specific humidity as the main surface variable and visualizes it as a filled contour map. It then overlays mean sea level pressure (MSLP) contours and 10-meter wind vectors on top of the surface map. The example uses data from the first time index of the dataset and focuses on the CONUS region. Note that the same plot can be generated with different variables, contour levels, colors, and wind vector styles by modifying the respective configuration dictionaries.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os

# Load relevant MPASdiag modules 
import mpasdiag as md

# Specify the path to sample data and grid file
dataDir = '../data/u240k/diag/'
gridPath = '../data/grids/x1.10242.static.nc'

# Create output directory if it doesn't exist
os.makedirs('./output', exist_ok=True)

# Load unstructured MPAS data
processor = md.MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# Define time index for surface variable extraction
tindex = 1 

# Initialize Surface and Wind Plotter
plotter = md.MPASSurfacePlotter(verbose=True, figsize=(14, 13), dpi=300)
wind_plotter = md.MPASWindPlotter(figsize=(14, 13), dpi=300)

# Define variable names for mean sea level pressure, specific humidity, and 10-m wind components
pres_var = 'mslp'
uwnd_var = 'u10'
vwnd_var = 'v10'
shum_var = 'q2'

# Extract 2D coordinates and variable data for the surface plot
surface_var = processor.dataset[pres_var].isel(Time=tindex)
lon, lat = processor.extract_2d_coordinates_for_variable(pres_var, surface_var)

# Extract variable data for the surface plot and overlays (flattening for contour and wind plotting)
pres = processor.dataset[pres_var].isel(Time=tindex).values.flatten()
uwnd = processor.dataset[uwnd_var].isel(Time=tindex).values.flatten()
vwnd = processor.dataset[vwnd_var].isel(Time=tindex).values.flatten()
shum = processor.dataset[shum_var].isel(Time=tindex).values.flatten()

# Define plot configuration
cfg = md.MPASConfig()

# Define map boundaries
cfg.lon_min = -130.0
cfg.lon_max = -50.0
cfg.lat_min = 10.0
cfg.lat_max = 60.0

# Cross-section transects to overlay on the map (label: {start, end, xoffset, yoffset, color})
TRANSECTS = {
    "A–B": {
        "start": (-120.0, 30.0), "start_label": "A", 
        "end": (-80.0,  50.0), "end_label": "B", 
        "xoffset": -1.0, "yoffset": 3.0, 
        "color": "red"},
    "C–D": {
        "start": (0.0,  0.0), "start_label": "C", 
        "end": ( 45.0,  30.0), "end_label": "D", 
        "xoffset": -1.0, "yoffset": 3.0, 
        "color": "royalblue"},
}

# Extract valid time from the dataset for the specified time index
valtime = processor.dataset['Time'][tindex].values
valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')

# Remapping configuration: controls how unstructured MPAS data is mapped to regular lat/lon grid for plotting.
cfg.remap_engine = 'kdtree'   # 'kdtree' (SciPy) or 'esmf' (ESMPy)
cfg.remap_method = 'nearest'  # 'nearest' | 'linear' for kdtree; 'conservative' | 'nearest_s2d' for esmf

# -------------- Create base surface map with 2-m specific humidity --------------

# Main filled contour: 2-m specific humidity (interpolated)
fig, ax = plotter.create_surface_map(
    lon=lon,
    lat=lat,
    data=shum,
    var_name=shum_var,
    bounds=md.GeographicBounds(cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max),
    style=md.SurfaceMapStyle(
        plot_type='contourf',
        grid_resolution=0.1,
        title=f'2-m specific humidity (filled contour), MSLP (contours), 10-m wind (barbs) | Valid Time: {valtime_str}',
    ),
    config=cfg
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
                            lat_min=cfg.lat_min, lat_max=cfg.lat_max, config=cfg)

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
                            lat_min=cfg.lat_min, lat_max=cfg.lat_max, config=cfg)

# Overlay cross-section transect lines
plotter.draw_transect_lines(ax, TRANSECTS)

# Save the final plot with the filled contour, contour overlay, and wind vector overlay
plotter.save_plot(f'./output/example_moisture_transport_with_mslp_valid_{valtime_str}', formats=['png'])
plotter.close_plot()

print(f"Plot saved to ./output/example_moisture_transport_with_mslp_valid_{valtime_str}.png")
