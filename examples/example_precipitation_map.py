#!/usr/bin/env python3

"""
MPASdiag Example II: Global Precipitation Map as Filled Contour Plot

This example demonstrates how to compute total precipitation from MPAS 2D model output and plot it as a filled contour map. The example computes the 1-hour accumulated precipitation difference between time indices 0 and 1, which corresponds to the total precipitation over the first hour of the simulation. Note that the same variable can also be visualized as a scatter plot by changing the plot_type argument in the create_surface_map method. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load relevant MPASdiag modules 
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter

# Specify the path to sample data and grid file
dataDir = '../data/u120k/diag'
gridPath = '../data/grids/x1.40962.static.nc'

# Load unstructured MPAS data
processor = MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# Initialize Precipitation Diagnostics
diag = PrecipitationDiagnostics(verbose=True)

# Define time index for precipitation difference calculation
tindex = 1 

# Extract precipitation difference at time index 1
precip = diag.compute_precipitation_difference(
  processor.dataset, tindex, var_name='total', accum_period='a01h', 
  data_type=processor.data_type or 'xarray')
  
# Extract coordinates
lon, lat = processor.extract_2d_coordinates_for_variable('total', precip)

# Define the precipitation plotter with desired figure size and resolution
plotter = MPASPrecipitationPlotter(figsize=(12, 8), dpi=300)

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

# Create precipitation map for 1-hour accumulation over CONUS with filled contour plot
fig, ax = plotter.create_precipitation_map(
  lon, lat, precip.values, cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max, 
  plot_type='contourf', title=f'Total precipitation over CONUS | Plot Type: Filled Contour | Valid Time: {valtime_str}', 
  accum_period='a01h', data_array=precip, grid_resolution=0.1)

# Save the generated plot in PNG format
plotter.save_plot(f'./output/total_precipitation_valid_{valtime_str}', formats=['png'])
