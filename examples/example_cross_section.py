#!/usr/bin/env python3

"""
MPASdiag Example X: Vertical Cross-Section from Surface to 10 km

This example demonstrates how to create a vertical cross-section plot of a 3D variable (potential temperature in this case) from MPAS output data. The example uses sample MPAS output data and grid files to illustrate the process of loading 3D data, defining a great-circle transect between two geographic points, extracting the variable along the transect, and generating a vertical cross-section plot with height on the y-axis. The output is saved as a PNG file in the ./output directory.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import numpy as np

# Load relevant MPASdiag modules
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

# Specify the path to sample 3D output data and grid file
dataDir  = '../data/u240k/mpasout'
gridPath = '../data/grids/x1.10242.static.nc'

# 3D variable to plot in the cross-section
var_name = 'theta'

# Time index to extract (0-based; index 1 corresponds to the second time step)
tindex = 1

# Define model initialization time
init_time      = np.datetime64('2024-09-17T00', 'h')
init_time_str  = str(init_time).replace('-', '').replace('T', '')   
init_time_disp = str(init_time).replace('T', ' ') + ':00 UTC' 

# Load unstructured MPAS 3D data
processor = MPAS3DProcessor(grid_file=gridPath, verbose=True)
processor.load_3d_data(dataDir)

# Confirm the requested variable is available in the dataset
available_3d = processor.get_available_3d_variables()
if var_name not in processor.dataset.data_vars:
    raise KeyError(
        f"Variable '{var_name}' not found. "
        f"Available 3D variables: {available_3d}"
    )

# Define the cross-section transect
start_point = (-90.0, 29.0)
end_point   = (-79.0, 44.0)

# Extract valid time for this time index
valtime      = processor.dataset['Time'][tindex].values
valtime_np   = valtime.astype('datetime64[h]')
valtime_str  = str(valtime_np).replace('-', '').replace('T', '') 
valtime_disp = str(valtime_np).replace('T', ' ') + ':00 UTC'  

# Plot configuration
cfg = MPASConfig()

# Initialize the cross-section plotter
plotter = MPASVerticalCrossSectionPlotter(figsize=(14, 8), dpi=300)

# Create output directory if it does not already exist
os.makedirs('./output', exist_ok=True)

# Base output path (without extension; save_plot appends '.png')
output_base = f'./output/{var_name}_cross_section_{valtime_str}'

# Define custom levels
custom_levels = np.arange(15, 100, 10)

# Generate vertical cross-section from the surface to 10 km
fig, ax = plotter.create_vertical_cross_section(
    mpas_3d_processor=processor,
    var_name=var_name,
    start_point=start_point,
    end_point=end_point,
    time_index=tindex,
    vertical_coord='pressure',
    display_vertical='height',
    max_height=10.0,
    num_points=500,
    plot_type='contourf',
    extend='both',
    levels=custom_levels,
    colormap='YlGnBu',
    title=(
        f'Potential Temperature (\u03b8) | Vertical Cross-Section '
        f'(Surface \u2013 10 km) | {valtime_disp}'
    ),
)

# Annotate the top-right corner with init time and valid time
ax.text(
    0.99, 0.98,
    f'Init:  {init_time_disp}\nValid: {valtime_disp}',
    transform=ax.transAxes,
    fontsize=9,
    ha='right', va='top',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='white',
        alpha=0.75,
        edgecolor='gray',
        linewidth=0.8,
    ),
    zorder=10,
)

# Save the figure in PNG format
plotter.save_plot(output_base, formats=['png'])

print(f'\nVertical cross-section saved to {output_base}.png')
