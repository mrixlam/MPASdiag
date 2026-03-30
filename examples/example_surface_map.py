#!/usr/bin/env python3

"""
MPASdiag Example I: Global Surface Map as Scatter Plot

This example demonstrates how to extract a surface variable from MPAS 2D model output and plot it as a surface map. We will extract 2-meter temperature (t2m) at a specified time index (0-based) of the dataset and visualize it using a scatter plot over a specified geographic region (e.g. GLOBAL). Note that the same variable can also be visualized as a filled contour plot by changing the plot_type argument in the create_surface_map method. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load relevant MPASdiag modules 
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter

# Specify the path to sample data and grid file
dataDir = '../data/u120k/diag'
gridPath = '../data/grids/x1.40962.static.nc'

# Load unstructured MPAS data
processor = MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# Define time index for surface variable extraction
tindex = 1 

# Initialize Surface Plotter
plotter = MPASSurfacePlotter(verbose=True)

# Extract surface variable at time index 0
surface_var = processor.dataset['t2m'].isel(Time=tindex)

# Extract coordinates
lon, lat = processor.extract_2d_coordinates_for_variable('t2m', surface_var)

# Define the surface plotter with desired figure size and resolution
plotter = MPASSurfacePlotter(figsize=(12, 10), dpi=300)

# Define plot configuration
cfg = MPASConfig()

# Define map boundaries
cfg.lon_min = -180.0
cfg.lon_max = 180.0
cfg.lat_min = -90.0
cfg.lat_max = 90.0

# Extract valid time from the dataset for the specified time index
valtime = processor.dataset['Time'][tindex].values
valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')

# Create scatter plot of 2-meter temperature
fig, ax = plotter.create_surface_map(
  lon=lon, lat=lat, data=surface_var.values, var_name='t2m', lon_min=cfg.lon_min, lon_max=cfg.lon_max, lat_min=cfg.lat_min, lat_max=cfg.lat_max,
  plot_type='scatter', title=f'2-meter Temperature | Plot Type: Scatter | Valid Time: {valtime_str}', data_array=surface_var)

# Save the generated plot in PNG format
plotter.save_plot(f'./output/2m_temperature_scatter_{valtime_str}', formats=['png'])

