#!/usr/bin/env python3

"""
MPASdiag Example IX: 3D Variable Horizontal Slice Map

This example demonstrates how to extract a horizontal slice of a 3D variable at a specified model level and time index, and visualize it using MPASdiag's plotting capabilities. The example uses sample MPAS output data and grid files to illustrate the process of loading 3D data, extracting the relevant slice, and generating a map of the variable at that level. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
# Load standard libraries
import os

# Load relevant MPASdiag modules
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter

# Specify the path to sample data and grid file
dataDir = '../data/u240k/mpasout'
gridPath = '../data/grids/x1.10242.static.nc'

# Variable name: any 3D field present in the mpasout files (e.g. 'qv', 'theta', etc.)
var_name = 'qv'

# Model level index (0-indexed; 0 = surface, nVertLevels-1 = model top)
level_index = 12

# Time index to extract (0-based)
tindex = 1

# Load unstructured MPAS 3D data
processor = MPAS3DProcessor(grid_file=gridPath, verbose=True)
processor.load_3d_data(dataDir)

# Verify the requested variable is available in the dataset
if var_name not in processor.dataset:
    available = [v for v in processor.dataset.data_vars if 'nVertLevels' in processor.dataset[v].dims]
    raise KeyError(
        f"Variable '{var_name}' not found. Available 3D variables: {available}"
    )

# Extract the 3D variable at the specified time index (shape: nCells × nVertLevels)
data_3d = processor.dataset[var_name].isel(Time=tindex)

# Extract 2D grid coordinates (lon/lat at cell centers) for the selected variable
lon, lat = processor.extract_2d_coordinates_for_variable(var_name, data_3d)

# Extract valid time string for the output file name
valtime     = processor.dataset['Time'][tindex].values
valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')

# -------------- Generate the horizontal slice map --------------------------

# plot_3d_variable_slice selects the given model level
plotter = MPASSurfacePlotter(verbose=True, figsize=(16, 10), dpi=300)

# Generate the plot for the specified variable and model level
fig, ax = plotter.plot_3d_variable_slice(
    data_array=data_3d,
    lon=lon,
    lat=lat,
    level=level_index,
    var_name=var_name,
    title=f"{var_name} at model level {level_index} | Valid Time: {valtime_str} UTC"
)

# -------------- Save the output --------------------------------------------

# Create output directory if it does not already exist
os.makedirs('./output', exist_ok=True)

output_path = f'./output/{var_name}_level{level_index:02d}_{valtime_str}'
fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
fig.clf()

print(f"\n{var_name} slice at model level {level_index} saved to {output_path}.png")
