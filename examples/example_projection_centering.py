#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Example: Controlling the Map Projection Aesthetic (Centering & Projections)

This example demonstrates how to control the map projection aesthetic of a surface
map using the projection-centering fields on SurfaceMapStyle. The same fields
(central_longitude, central_latitude, proj_kwargs) are available on
PrecipitationRenderStyle and WindPlotStyle.

We plot global 2-meter temperature (t2m) three ways:
  1. Default PlateCarree centered on 0° longitude (Greenwich-centered).
  2. Pacific-centered PlateCarree using central_longitude=180, which places the
     dateline in the middle of the map -- ideal for fields that straddle ±180°.
  3. A Robinson global projection, with its center set via proj_kwargs.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: July 2026
Version: 1.0.0
"""
# Load relevant MPASdiag modules
import mpasdiag as md

# Specify the path to sample data and grid file
dataDir = '../data/u240k/diag'
gridPath = '../data/grids/x1.10242.static.nc'

# Load unstructured MPAS data
processor = md.MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# Define time index for surface variable extraction
tindex = 1

# Extract surface variable and coordinates
surface_var = processor.dataset['t2m'].isel(Time=tindex)
lon, lat = processor.extract_2d_coordinates_for_variable('t2m', surface_var)

# Extract valid time string for titles/filenames
valtime = processor.dataset['Time'][tindex].values
valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')

# Global map boundaries
bounds = md.GeographicBounds(-180.0, 180.0, -90.0, 90.0)

# Remapping configuration
cfg = md.MPASConfig()
cfg.remap_engine = 'kdtree'
cfg.remap_method = 'nearest'

# Default PlateCarree (central_longitude defaults to 0 -- unchanged output)
plotter = md.MPASSurfacePlotter(figsize=(12, 10), dpi=300)
fig, ax = plotter.create_surface_map(
    lon=lon, lat=lat, data=surface_var.values, var_name='t2m', bounds=bounds,
    style=md.SurfaceMapStyle(
        plot_type='contourf',
        title=f'2-meter Temperature | Greenwich-centered (default) | {valtime_str}',
    ),
    data_array=surface_var, config=cfg)
plotter.save_plot(f'./output/2m_temperature_default_{valtime_str}', formats=['png'])
plotter.close_plot()

# Pacific-centered PlateCarree (central_longitude=180)
plotter = md.MPASSurfacePlotter(figsize=(12, 10), dpi=300)
fig, ax = plotter.create_surface_map(
    lon=lon, lat=lat, data=surface_var.values, var_name='t2m', bounds=bounds,
    style=md.SurfaceMapStyle(
        plot_type='contourf',
        central_longitude=180.0,
        title=f'2-meter Temperature | Pacific-centered (central_longitude=180) | {valtime_str}',
    ),
    data_array=surface_var, config=cfg)
plotter.save_plot(f'./output/2m_temperature_pacific_{valtime_str}', formats=['png'])
plotter.close_plot()

# Robinson projection with center set via proj_kwargs (full passthrough)
plotter = md.MPASSurfacePlotter(figsize=(12, 10), dpi=300)
fig, ax = plotter.create_surface_map(
    lon=lon, lat=lat, data=surface_var.values, var_name='t2m', bounds=bounds,
    style=md.SurfaceMapStyle(
        plot_type='contourf',
        projection='Robinson',
        proj_kwargs={'central_longitude': 0.0},
        title=f'2-meter Temperature | Robinson projection | {valtime_str}',
    ),
    data_array=surface_var, config=cfg)
plotter.save_plot(f'./output/2m_temperature_robinson_{valtime_str}', formats=['png'])
plotter.close_plot()
