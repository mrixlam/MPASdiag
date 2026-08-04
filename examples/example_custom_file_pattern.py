#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Example XI: Surface Map from a Custom Output Stream (wiso.*.nc)

This example demonstrates how to load and plot a 2D surface variable from a custom output stream of MPAS model data. Instead of the default 'diag*.nc' files, this example uses files matching the pattern 'wiso.*.nc', which contain  precipitation isotopic ratio data.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: July 2026
Version: 1.0.0
"""

# Load standard libraries
import os

import numpy as np

# Load relevant MPASdiag modules
import mpasdiag as md

# Specify the path to the model output and grid file
dataDir = "../data/isoMPAS/bench/gnip_annual"
gridPath = "../data/grids/x1.10242.static.nc"

# Glob pattern for the custom output stream (instead of the default 'diag*.nc')
file_pattern = "wiso.*.nc"

# Variable name: any 2D field present in the wiso files (e.g. 'd18O_precip', 'dD_precip', etc.).
var_name = "d18O_precip"

# Plot name: used for labeling the plot and output file name.
plot_name = "d18O"

# Sentinel value below which the data is considered invalid
undefined_below = -1.0e6

# Time index to extract (0-based)
tindex = 2


# -------------- Load the data -----------------------------------------------

# Create an MPAS2DProcessor instance with the custom file pattern
processor = md.MPAS2DProcessor(
    grid_file=gridPath, file_pattern=file_pattern, verbose=True
)
processor.load_2d_data(dataDir)

# Verify the requested variable is available in the dataset
if var_name not in processor.dataset:
    available = [
        v
        for v in processor.dataset.data_vars
        if "nCells" in processor.dataset[v].dims
        and "nVertLevels" not in processor.dataset[v].dims
    ]
    raise KeyError(
        f"Variable '{var_name}' not found. Available 2D variables: {available}"
    )

# Extract the 2D surface variable at the specified time index (shape: nCells)
surface_var = processor.dataset[var_name].isel(Time=tindex)

# Mask out invalid values below the defined threshold
surface_var = surface_var.where(surface_var > undefined_below)

# Extract grid coordinates (lon/lat at cell centers) for the selected variable
lon, lat = processor.extract_2d_coordinates_for_variable(var_name, surface_var)

# Extract valid time string for the plot title and output file name
valtime = processor.dataset["Time"][tindex].values
valtime_str = str(valtime.astype("datetime64[h]")).replace("-", "")

# -------------- Generate the surface map ------------------------------------

# Define the surface plotter with desired figure size and resolution
plotter = md.MPASSurfacePlotter(verbose=True, figsize=(16, 10), dpi=300)

# Define plot configuration
cfg = md.MPASConfig()

# Define map boundaries (global domain)
cfg.lon_min = -180.0
cfg.lon_max = 180.0
cfg.lat_min = -90.0
cfg.lat_max = 90.0

# Define remapping engine and method for scattered data interpolation
cfg.remap_engine = "kdtree"  # 'kdtree' (SciPy) or 'esmf' (ESMPy)
cfg.remap_method = "nearest"  # 'nearest' | 'linear' for kdtree

# Generate the surface map for the selected data variable
plotter.create_surface_map(
    lon=lon,
    lat=lat,
    data=surface_var.values,
    var_name=plot_name,
    bounds=md.GeographicBounds(cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max),
    style=md.SurfaceMapStyle(
        plot_type="scatter",
        colormap="Spectral_r",
        levels=list(np.arange(-30.0, 22.5, 2.5)),
        title=f"{var_name} from {file_pattern} | Valid Time: {valtime_str} UTC",
    ),
    data_array=surface_var,
    config=cfg,
)

# -------------- Save the output ---------------------------------------------

# Create output directory if it does not already exist
os.makedirs("./output", exist_ok=True)

output_path = f"./output/{var_name}_{valtime_str}"
plotter.save_plot(output_path, formats=["png"])
plotter.close_plot()

print(f"\n{var_name} surface map saved to {output_path}.png")
