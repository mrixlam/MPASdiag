#!/usr/bin/env python3

"""
MPASdiag Example VII: Thermodynamic Skew-T Log-P Diagram

This example demonstrates how to extract a vertical sounding profile from MPAS 3D model output at a specific geographic location and plot it as a Skew-T Log-P diagram. The example extracts the sounding profile at a location in Southeast Asia (near Singapore) at the first time index of the dataset. The Skew-T diagram includes temperature, dewpoint, and wind profiles, along with computed thermodynamic indices such as CAPE and CIN.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load common libraries
from pathlib import Path

# Load relevant MPASdiag modules 
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.visualization.skewt import MPASSkewTPlotter

# Specify the path to sample data and grid file
dataDir = '../data/u240k/mpasout'
gridPath = '../data/grids/x1.10242.static.nc'

# Specify output directory for plots
plotDir = Path("output/skewt")
plotDir.mkdir(parents=True, exist_ok=True)

# Define target location and time index for sounding extraction
lon = 103.212
lat = 3.772
time = 0  

# Load unstructured MPAS 3D data 
processor = MPAS3DProcessor(grid_file=gridPath, verbose=True)
processor.load_3d_data(dataDir)

# Initialize Sounding Diagnostics
diagnostics = SoundingDiagnostics(verbose=True)

# Extract sounding profile at the specified location and time 
profile = diagnostics.extract_sounding_profile(
    processor,
    lon=lon,
    lat=lat,
    time_index=time,
)

# Extract relevant variables from the profile for plotting 
pressure    = profile['pressure']      
temperature = profile['temperature'] 
dewpoint    = profile['dewpoint']  
u_wind      = profile.get('u_wind') 
v_wind      = profile.get('v_wind')   
station_lon = profile['station_lon']
station_lat = profile['station_lat']

# Compute thermodynamic indices for the sounding profile 
indices = diagnostics.compute_thermodynamic_indices(
    pressure, temperature, dewpoint,
    u_wind_kt=u_wind, v_wind_kt=v_wind,
    height_m=profile.get('height'),
)

# Define the Skew-T plotter with desired figure size and resolution
plotter = MPASSkewTPlotter(figsize=(9, 12), dpi=150, verbose=True)

# Format station coordinates for figure title
latstr = f"{abs(station_lat):.2f}{'N' if station_lat >= 0 else 'S'}"
lonstr = f"{abs(station_lon):.2f}{'E' if station_lon >= 0 else 'W'}"

# Extract valid time from the dataset for the specified time index
valtime = processor.dataset['Time'][time].values
valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')

# Define the title for the Skew-T plot using station coordinates and valid time
title = (
    f"Location: {lonstr} / {latstr}  |  Valid Time: {valtime_str}"
)

# Define the output path for the generated Skew-T plot
save_path = str(plotDir / f"mpas_skewt_{lonstr.replace('.', 'p')}_{latstr.replace('.', 'p')}_valid_{valtime_str}.png")

# Create the Skew-T diagram using the extracted profile and computed indices
fig, ax = plotter.create_skewt_diagram(
    pressure=pressure,
    temperature=temperature,
    dewpoint=dewpoint,
    u_wind=u_wind,
    v_wind=v_wind,
    title=title,
    indices=indices,
    show_parcel=True,
)

# Save the generated Skew-T diagram in PNG format
fig.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
