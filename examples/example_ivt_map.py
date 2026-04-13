#!/usr/bin/env python3

"""
MPASdiag Example VIII: Vertically Integrated Water Vapor Transport (IVT) Map

This example demonstrates how to calculate and visualize the vertically integrated water vapor transport (IVT) from MPAS 3D output data. The example uses sample MPAS output data and grid files to illustrate the process of loading 3D data, calculating IVT by integrating specific humidity and wind components over the vertical dimension, and generating a map of IVT at the surface level. The output is saved as a PNG file in the ./output directory. 

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
from mpasdiag.diagnostics.moisture_transport import MoistureTransportDiagnostics
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.wind import MPASWindPlotter

# Specify the path to sample data and grid file
dataDir = '../data/u240k/mpasout'
gridPath = '../data/grids/x1.10242.static.nc'

# Load unstructured MPAS 3D data
processor = MPAS3DProcessor(grid_file=gridPath, verbose=True)
processor.load_3d_data(dataDir)

# Define model initialization time
init_time      = np.datetime64('2024-09-17T00', 'h')
init_time_str  = str(init_time).replace('-', '').replace('T', '')  
init_time_disp = str(init_time).replace('T', ' ') + ':00 UTC'   

# Define time index for variable extraction
tindex = 0

# Extract specific humidity at each model level (kg/kg)
qv = processor.dataset['qv'].isel(Time=tindex)

# Construct total pressure from perturbation and base-state fields (Pa)
if 'pressure_p' in processor.dataset and 'pressure_base' in processor.dataset:
    pressure = (processor.dataset['pressure_p'].isel(Time=tindex) +
                processor.dataset['pressure_base'].isel(Time=tindex))
else:
    pressure = processor.dataset['pressure'].isel(Time=tindex)

# Extract 3D horizontal wind components at each model level (m/s)
u_3d = processor.dataset['uReconstructZonal'].isel(Time=tindex)
v_3d = processor.dataset['uReconstructMeridional'].isel(Time=tindex)

# Initialize moisture transport diagnostics and compute IWV, IVT_u, IVT_v, and IVT magnitude
diag   = MoistureTransportDiagnostics(verbose=True)
result = diag.analyze_moisture_transport(qv, u_3d, v_3d, pressure)

# Extract 1-D IVT DataArrays (nCells) for magnitude and components
ivt_da   = result['ivt']['data']  
ivt_u_da = result['ivt_u']['data'] 
ivt_v_da = result['ivt_v']['data'] 

# Extract 2-D longitude and latitude coordinates for the variable used in IVT calculation 
lon, lat = processor.extract_2d_coordinates_for_variable('qv', ivt_u_da)

# Now convert IVT data to 1-D numpy arrays for plotting
ivt_mag = ivt_da.values.flatten()
ivt_u   = ivt_u_da.values.flatten()
ivt_v   = ivt_v_da.values.flatten()

# Initialize Surface and Wind Plotters
plotter      = MPASSurfacePlotter(verbose=True, figsize=(12, 9), dpi=300)
wind_plotter = MPASWindPlotter(figsize=(12, 9), dpi=300)

# Define plot configuration and map boundaries (global)
cfg         = MPASConfig()
cfg.lon_min =  -180.0
cfg.lon_max =  180.0
cfg.lat_min =  -90.0
cfg.lat_max =   90.0

# Extract valid time from the dataset for the specified time index
valtime     = processor.dataset['Time'][tindex].values
valtime_np   = valtime.astype('datetime64[h]')
valtime_str = str(valtime_np).replace('-', '')
valtime_disp = str(valtime_np).replace('T', ' ') + ':00 UTC'  

# Define custom contour levels
custom_levels = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

# -------------- Create base surface map: IVT magnitude as shading --------------

fig, ax = plotter.create_surface_map(
    lon=lon,
    lat=lat,
    data=ivt_mag,
    var_name='ivt',
    lon_min=cfg.lon_min,
    lon_max=cfg.lon_max,
    lat_min=cfg.lat_min,
    lat_max=cfg.lat_max,
    plot_type='contourf',
    grid_resolution=0.1,
    levels=custom_levels,
    title=f'Vertically Integrated Water Vapor Transport (IVT) | Valid Time: {valtime_str}',
)

# -------------- Overlay: IVT vectors as arrows --------------

# Define overlay configuration for IVT direction vectors
ivt_vector_config = {
    'u_data':          ivt_u,
    'v_data':          ivt_v,
    'plot_type':       'arrows',
    'subsample':       '-1',
    'scale':           15000,
    'colors':          'black',
    'grid_resolution': 0.1,
}

# Add IVT vectors as arrows on top of the IVT magnitude shading
Q = wind_plotter.add_wind_overlay(
    ax, lon, lat, ivt_vector_config,
    lon_min=cfg.lon_min, lon_max=cfg.lon_max,
    lat_min=cfg.lat_min, lat_max=cfg.lat_max,
)

# Specify a reference IVT magnitude for the quiver key 
ivt_ref_magnitude = 250  # kg m⁻¹ s⁻¹

# Add a quiver key to the plot to indicate the reference IVT magnitude for the arrows
if Q is not None:
    qk = ax.quiverkey(
        Q, X=0.09, Y=0.975,
        U=ivt_ref_magnitude,
        label=f'{ivt_ref_magnitude} kg m$^{{-1}}$ s$^{{-1}}$',
        labelpos='S',
        coordinates='axes',
        fontproperties={'size': 15},
        color='black',
        labelcolor='black',
    )
    # Add a white box behind the label text so it remains legible over the shading
    qk.text.set_bbox(dict(
        facecolor='white', alpha=0.75,
        edgecolor='gray', boxstyle='round,pad=0.3',
    ))

# Annotate the top-right corner with init time and valid time
ax.text(
    0.985, 0.98,
    f'Init:  {init_time_disp}\nValid: {valtime_disp}',
    transform=ax.transAxes,
    fontsize=15,
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

# Create output directory if it does not already exist
os.makedirs('./output', exist_ok=True)

# Save the final IVT map with magnitude shading and vector overlay in PNG format
plotter.save_plot(f'./output/ivt_map_{valtime_str}', formats=['png'])
plotter.close_plot()

print(f"\nIVT map saved to ./output/ivt_map_{valtime_str}.png")
