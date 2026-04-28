#!/usr/bin/env python3

"""
MPASdiag Example III: Global Wind Vectors Represented as Barbs

This example demonstrates how to extract wind components from MPAS 2D model output and plot them as a global wind map. We will extract the 10-meter u and v wind components at a specified time index (0-based) of the dataset and visualize them using barbs over a specified geographic region (e.g. CONUS). Note that the wind vectors can also be visualized as arrows by changing the plot_type argument in the create_wind_plot method.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load relevant MPASdiag modules
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.visualization.wind import MPASWindPlotter

# Specify the path to sample data and grid file
dataDir = '../data/u240k/diag'
gridPath = '../data/grids/x1.10242.static.nc'

# Load unstructured MPAS data
processor = MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# Define time index for wind variable extraction 
tindex = 1 

# Extract wind components at time index 1
u_data = processor.get_2d_variable_data('u10', tindex)
v_data = processor.get_2d_variable_data('v10', tindex)

# Extract coordinates
lon, lat = processor.extract_2d_coordinates_for_variable('u10', u_data)

# Define the wind plotter with desired figure size and resolution
wind_plotter = MPASWindPlotter(figsize=(14, 11), dpi=300)

# Define plot configuration
cfg = MPASConfig()

# Define map boundaries
cfg.lon_min = -130.0
cfg.lon_max = -50.0
cfg.lat_min = 20.0
cfg.lat_max = 60.0

# Extract valid time from the dataset for the specified time index
valtime = processor.dataset['Time'][tindex].values
valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')

# Remapping configuration: controls how unstructured MPAS data is mapped to regular lat/lon grid for plotting.
cfg.remap_engine = 'kdtree'   # 'kdtree' (SciPy) or 'esmf' (ESMPy)
cfg.remap_method = 'nearest'  # 'nearest' | 'linear' for kdtree; 'conservative' | 'nearest_s2d' for esmf

# Generate wind plot over CONUS with vectors represented as barbs
fig, ax = wind_plotter.create_wind_plot(
    lon=lon, lat=lat, u_data=u_data.values, v_data=v_data.values,
    lon_min=cfg.lon_min, lon_max=cfg.lon_max, lat_min=cfg.lat_min, lat_max=cfg.lat_max, subsample=-1,
    plot_type='barbs', grid_resolution=0.1, regrid_method='linear',
    title=f'MPAS Wind Analysis | Vector Type: Barbs | Valid Time: {valtime_str}',
    config=cfg,
)

# Save figure as PNG file
wind_plotter.save_plot(f'./output/mpas_wind_plot_conus_{valtime_str}', formats=['png'])
wind_plotter.close_plot()
