#!/usr/bin/env python3

"""
MPASdiag Example: Density Potential Temperature (Emanuel 1994)

This example script demonstrates how to compute and visualize density potential temperature (theta_rho) from MPAS model output using the ThermodynamicDiagnostics helper class. The script generates a 2x2 panel of filled contour maps of theta_rho at the lowest model level (near the surface) for a sequence of time steps, with consistent contour levels and a shared colorbar. Cross-section transects are overlaid on each map for reference. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Plotting libraries for the manually-assembled panel
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

# Load relevant MPASdiag modules
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.diagnostics.thermodynamic import ThermodynamicDiagnostics

# Path to MPAS grid file and 3D model-output directory
dataDir = '../data/u240k/mpasout'
gridPath = '../data/grids/x1.10242.static.nc'

# Time indices to use, one per subplot (2x2 = 4 panels). Edit per run.
TINDICES = [0, 1, 2, 3]

# Model initialization (forecast start) time, used to compute lead times.
initTime = '20240917T00'
init64 = np.datetime64(datetime.strptime(initTime, '%Y%m%dT%H'))

# Vertical level (model level index; 0 = surface)
LEVEL = 0

# Map boundaries (CONUS)
LON_MIN, LON_MAX = 91.0, 113.0
LAT_MIN, LAT_MAX = -9.60, 12.20

# Data coordinate reference system for the maps
data_crs = ccrs.PlateCarree()

# Cross-section transects to overlay on the map (label: {start, end, color})
TRANSECTS = {
    "A–B": {
        "start": (95.0, 1.50), "start_label": "A",
        "end": (110.0,  1.50), "end_label": "B",
        "xoffset": -0.30, "yoffset": -0.20,
        "color": "red"},
    # "C–D": {
    #     "start": (95.0, 1.50), "start_label": "C",
    #     "end": (110.0,  1.50), "end_label": "D",
    #     "xoffset": -0.30, "yoffset": 0.20,
    #     "color": "royalblue"},
}

# Output directory
OUTPUT_DIR = Path('./output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MPASdiag Example: 2x2 Panel of Density Potential Temperature")
print("=" * 70)

# Initialise the 3D processor and load MPAS output
processor = MPAS3DProcessor(grid_file=gridPath, verbose=True)
processor.load_3d_data(dataDir)

# Initialise the thermodynamic diagnostics helper
thermo = ThermodynamicDiagnostics(verbose=True)

# Access the underlying xarray dataset
ds = processor.dataset if hasattr(processor.dataset, 'data_vars') else processor.dataset.ds

# Single plotter instance, reused only for its gridding/transect helpers
plotter = MPASSurfacePlotter(verbose=True, dpi=300)

# Custom contour levels and shared colour normalization (identical for every panel)
custom_levels = [297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310]
norm = BoundaryNorm(custom_levels, ncolors=plt.get_cmap('viridis_r').N, clip=True)

# Create the 2x2 grid of cartopy GeoAxes
fig, axes = plt.subplots(
    2, 2, figsize=(18, 16), dpi=300,
    subplot_kw={'projection': data_crs},
)
axes_flat = axes.flat

# Coordinates are identical for every time, so extract them once
lon = lat = None

# Mappable kept from the last drawn panel for the shared colorbar
cs = None

# Loop over the time indices and corresponding axes to compute and plot theta_rho for each time step. 
for i, ax in enumerate(axes_flat):
    # Hide any unused trailing axes if fewer than 4 time indices are given
    if i >= len(TINDICES):
        ax.set_visible(False)
        continue
    
    # Current time index for this panel
    tindex = TINDICES[i]

    # Compute theta_rho directly from dataset using ThermodynamicDiagnostics
    theta_rho = thermo.compute_density_potential_temperature_from_dataset(
        dataset=ds,
        time_index=tindex,
        level=LEVEL,             # lowest model level (near surface)
        theta_var='theta',       # potential temperature
        qv_var='qv',             # water-vapour mixing ratio
        condensate_vars=['qc', 'qr', 'qi', 'qs', 'qg'],
    )

    # Extract coordinates for plotting (same mesh for all times)
    if lon is None:
        theta_at_level = thermo._extract_field(
            ds, 'theta',
            time_dim='Time' if 'Time' in ds.dims else 'time',
            time_index=tindex,
            level=LEVEL,
            data_type='xarray',
        )
        # Print a statistical summary of theta_rho vs theta for the first time
        thermo.analyze_density_potential_temperature(theta_rho, theta=theta_at_level)
        lon, lat = processor.extract_2d_coordinates_for_variable('theta', theta_at_level)

    # Flatten the 2D theta_rho array for interpolation
    theta_rho_flat = theta_rho.values.flatten()

    # Interpolate the irregular MPAS data onto a regular lat/lon grid for plotting.
    lon_mesh, lat_mesh, data_interp = plotter._interpolate_to_grid(
        lon, lat, theta_rho_flat,
        LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
        grid_resolution=0.01, dataset=ds,
    )

    # Filled contours (identical levels/norm across panels -> consistent colours)
    cs = ax.contourf(
        lon_mesh, lat_mesh, data_interp, levels=custom_levels,
        cmap='viridis_r', norm=norm, transform=data_crs, extend='both',
    )

    # Cartographic features and fixed extent (shared across panels)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=data_crs)

    # Gridlines: label latitudes only on the left column, longitudes only on the bottom row
    row, col = divmod(i, 2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (col == 0)
    gl.bottom_labels = (row == 1)

    # Per-panel title = valid time string plus forecast lead time from initTime
    valtime = processor.dataset['Time'][tindex].values
    valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')
    lead_minutes = int((valtime - init64) / np.timedelta64(1, 'm'))
    lead_h, lead_m = divmod(lead_minutes, 60)
    ax.set_title(f'{valtime_str} UTC (+{lead_h:02d}:{lead_m:02d})', fontsize=11)

    # Overlay cross-section transect lines on every panel
    plotter.draw_transect_lines(ax, TRANSECTS, data_crs=data_crs)

# Leave room at the bottom for a single shared horizontal colorbar
fig.subplots_adjust(wspace=0.05, hspace=0.12, bottom=0.12)

cax = fig.add_axes([0.20, 0.07, 0.60, 0.02])
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal',
                    ticks=custom_levels, extend='both')
cbar.set_label('Density Potential Temperature (K)')

# Figure title (uncomment if desired)
# fig.suptitle(f'Density Potential Temperature at Level {LEVEL}',
#              fontsize=16, fontweight='bold')

# Save and clean up
out_file = OUTPUT_DIR / f'density_potential_temperature_panel_{TINDICES[0]}_{TINDICES[-1]}.png'
fig.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nPlot saved to {out_file}")

