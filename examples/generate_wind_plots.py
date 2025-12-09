#!/usr/bin/env python3

"""
Generate Wind Vector Plots from MPAS Output

This example script demonstrates the comprehensive wind visualization capabilities of the 
MPASWindPlotter class including wind barbs, wind arrows, streamlines, and complex multi-layer 
plots with wind speed backgrounds and overlaid wind vectors. The script shows four different 
visualization techniques for displaying MPAS atmospheric wind data with meteorological conventions 
and professional cartographic styling.

Examples:
    1. Wind Barbs Plot: Classical meteorological wind barbs showing speed and direction
    2. Wind Arrows (Quiver): Vector arrows showing wind magnitude and direction
    3. Streamlines: Flow visualization showing wind trajectories and patterns
    4. Complex Plot: Wind speed background with overlaid wind barbs for detailed analysis

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: December 2025
Version: 1.0.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import cast

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.visualization.surface import MPASSurfacePlotter


def example_1_wind_barbs(
    grid_file: str,
    data_file: str,
    output_dir: str
) -> None:
    """
    Create a wind barb plot showing classical meteorological wind representation with flags indicating speed ranges and direction. This example demonstrates the standard operational weather analysis visualization using wind barbs following WMO conventions where barbs show wind direction by pointing toward the source and speed through flag symbols (full barb = 10 knots, half barb = 5 knots, pennant = 50 knots). The plot includes coastlines, borders, ocean/land features, automatic wind statistics in the title, and geographic gridlines for spatial reference.

    Parameters:
        grid_file (str): Absolute path to MPAS grid NetCDF file containing mesh coordinates and connectivity.
        data_file (str): Absolute path to MPAS diagnostic file containing u10 and v10 wind component variables.
        output_dir (str): Directory path where the wind barb plot file will be saved with automatic naming based on timestamp.

    Returns:
        None: Creates and saves wind barb visualization to output_dir/mpas_wind_barbs_example.png.
    """
    print("="*70)
    print("EXAMPLE 1: Wind Barbs Plot")
    print("="*70)
    
    processor = MPAS2DProcessor(grid_file=grid_file)
    data_dir = os.path.dirname(data_file)
    processor.load_2d_data(data_dir)
    
    time_idx = 0
    u_data = processor.get_2d_variable_data('u10', time_idx)
    v_data = processor.get_2d_variable_data('v10', time_idx)
    lon, lat = processor.extract_2d_coordinates_for_variable('u10', u_data)
    
    lon_min, lon_max = -130, -60
    lat_min, lat_max = 20, 55
    
    plotter = MPASWindPlotter(figsize=(14, 11), dpi=150)
    
    time_str = MPASDateTimeUtils.get_time_info(processor.dataset, time_idx, var_context='wind', verbose=False)
    time_stamp = pd.to_datetime(processor.dataset.Time.values[time_idx]).to_pydatetime() if hasattr(processor.dataset, 'Time') else None
    
    fig, ax = plotter.create_wind_plot(
        lon=lon,
        lat=lat,
        u_data=u_data.values,
        v_data=v_data.values,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        wind_level="10-m",
        plot_type='barbs',
        subsample=-1,
        grid_resolution=0.1,
        regrid_method='linear',
        title="MPAS 10-m Wind Analysis - Wind Barbs",
        time_stamp=time_stamp,
        projection='PlateCarree'
    )
    
    output_path = os.path.join(output_dir, "mpas_wind_barbs_example")
    plotter.add_timestamp_and_branding()
    plotter.save_plot(output_path, formats=['png'])
    plotter.close_plot()
    
    print(f"\n✓ Wind barbs plot saved to: {output_path}.png")
    print("  - Uses meteorological wind barb convention")
    print("  - Data regridded to 0.1° resolution for smooth visualization")
    print("  - Automatic intelligent subsampling for optimal density")
    print("  - Flags indicate wind speed (full barb = ~5 m/s)")
    print("  - Barbs point toward wind source direction")


def example_2_wind_arrows(
    grid_file: str,
    data_file: str,
    output_dir: str
) -> None:
    """
    Create a wind arrow (quiver) plot showing vector representations with arrow length proportional to wind speed. This example demonstrates alternative wind visualization using matplotlib's quiver function where arrows point in the wind direction and arrow length indicates wind magnitude. The plot uses a custom scale factor to optimize arrow sizes for readability, includes subsampling for performance on high-resolution meshes, and shows wind statistics in the title. This visualization style is useful for highlighting flow patterns and speed variations across spatial domains.

    Parameters:
        grid_file (str): Absolute path to MPAS grid NetCDF file containing mesh coordinates and connectivity.
        data_file (str): Absolute path to MPAS diagnostic file containing u10 and v10 wind component variables.
        output_dir (str): Directory path where the wind arrow plot file will be saved with descriptive naming.

    Returns:
        None: Creates and saves wind arrow visualization to output_dir/mpas_wind_arrows_example.png.
    """
    print("="*70)
    print("EXAMPLE 2: Wind Arrows (Quiver) Plot")
    print("="*70)
    
    processor = MPAS2DProcessor(grid_file=grid_file)
    data_dir = os.path.dirname(data_file)
    processor.load_2d_data(data_dir)
    
    time_idx = 0
    u_data = processor.get_2d_variable_data('u10', time_idx)
    v_data = processor.get_2d_variable_data('v10', time_idx)
    lon, lat = processor.extract_2d_coordinates_for_variable('u10', u_data)
    
    lon_min, lon_max = -130, -60
    lat_min, lat_max = 20, 55
    
    plotter = MPASWindPlotter(figsize=(14, 11), dpi=150)
    
    time_str = MPASDateTimeUtils.get_time_info(processor.dataset, time_idx, var_context='wind', verbose=False)
    time_stamp = pd.to_datetime(processor.dataset.Time.values[time_idx]).to_pydatetime() if hasattr(processor.dataset, 'Time') else None
    
    fig, ax = plotter.create_wind_plot(
        lon=lon,
        lat=lat,
        u_data=u_data.values,
        v_data=v_data.values,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        wind_level="10-m",
        plot_type='arrows',
        subsample=-1,
        scale=300,
        grid_resolution=0.1,
        regrid_method='linear',
        title="MPAS 10-m Wind Analysis - Vector Arrows",
        time_stamp=time_stamp,
        projection='PlateCarree'
    )
    
    output_path = os.path.join(output_dir, "mpas_wind_arrows_example")
    plotter.add_timestamp_and_branding()
    plotter.save_plot(output_path, formats=['png'])
    plotter.close_plot()
    
    print(f"\n✓ Wind arrows plot saved to: {output_path}.png")
    print("  - Arrow length proportional to wind speed")
    print("  - Data regridded to 0.1° resolution for smooth visualization")
    print("  - Automatic intelligent subsampling for optimal density")
    print("  - Arrows point in wind direction")
    print("  - Custom scale=300 for optimal arrow sizing")


def example_3_wind_streamlines(
    grid_file: str,
    data_file: str,
    output_dir: str
) -> None:
    """
    Create a streamline plot showing continuous wind flow trajectories using matplotlib streamplot on a regular grid. This example demonstrates advanced wind visualization by regridding irregular MPAS mesh data onto a regular latitude-longitude grid using linear interpolation, then computing streamlines that follow the wind flow field. Streamlines are curves tangent to the wind velocity vector at every point, providing intuitive visualization of atmospheric circulation patterns, convergence/divergence zones, and flow topology. The plot includes density control for streamline spacing and shows how wind patterns evolve across the spatial domain.

    Parameters:
        grid_file (str): Absolute path to MPAS grid NetCDF file containing mesh coordinates and connectivity.
        data_file (str): Absolute path to MPAS diagnostic file containing u10 and v10 wind component variables.
        output_dir (str): Directory path where the streamline plot file will be saved with descriptive naming.

    Returns:
        None: Creates and saves wind streamline visualization to output_dir/mpas_wind_streamlines_example.png.
    """
    print("="*70)
    print("EXAMPLE 3: Wind Streamlines Plot")
    print("="*70)
    
    processor = MPAS2DProcessor(grid_file=grid_file)
    data_dir = os.path.dirname(data_file)
    processor.load_2d_data(data_dir)
    
    time_idx = 0
    u_data = processor.get_2d_variable_data('u10', time_idx)
    v_data = processor.get_2d_variable_data('v10', time_idx)
    lon, lat = processor.extract_2d_coordinates_for_variable('u10', u_data)
    
    lon_min, lon_max = -130, -60
    lat_min, lat_max = 20, 55
    
    plotter = MPASWindPlotter(figsize=(14, 11), dpi=150)
    
    time_str = MPASDateTimeUtils.get_time_info(processor.dataset, time_idx, var_context='wind', verbose=False)
    time_stamp = pd.to_datetime(processor.dataset.Time.values[time_idx]).to_pydatetime() if hasattr(processor.dataset, 'Time') else None
    
    print("Regridding wind data for streamlines (this may take a moment)...")
    grid_resolution = 0.5
    
    lon_2d, lat_2d, u_2d, v_2d = plotter._regrid_wind_components(
        lon=lon,
        lat=lat,
        u_data=u_data.values,
        v_data=v_data.values,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        grid_resolution=grid_resolution,
        regrid_method='linear'
    )
    
    lon_1d = lon_2d[0, :]
    lat_1d = lat_2d[:, 0]
    
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.geoaxes import GeoAxes
    
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = cast(GeoAxes, plt.axes(projection=ccrs.PlateCarree()))
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    wind_speed = np.sqrt(u_2d**2 + v_2d**2)
    
    strm = ax.streamplot(
        lon_1d, lat_1d, u_2d, v_2d,
        transform=ccrs.PlateCarree(),
        color=wind_speed,
        cmap='viridis',
        linewidth=1.5,
        density=2,
        arrowsize=1.5,
        arrowstyle='->',
        minlength=0.1
    )
    
    cbar = plt.colorbar(strm.lines, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=40)
    
    wind_speed_metadata = MPASFileMetadata.get_variable_metadata('wind_speed')
    wind_speed_units = wind_speed_metadata.get('units', 'm s^{-1}')
    wind_speed_long_name = wind_speed_metadata.get('long_name', 'Wind Speed')
    cbar.set_label(f'{wind_speed_long_name} [{wind_speed_units}]', fontsize=12, fontweight='bold', labelpad=10)
    
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    max_speed = np.max(wind_speed)
    mean_speed = np.mean(wind_speed)
    
    wind_speed_metadata = MPASFileMetadata.get_variable_metadata('wind_speed')
    speed_unit = wind_speed_metadata.get('units', 'm s^{-1}')
    
    title = f"MPAS 10-m Wind Analysis - Streamlines\n"
    if time_stamp:
        title += f"{time_stamp.strftime('%Y-%m-%d %H:%M UTC')} - "
    title += f"Max: {max_speed:.1f} {speed_unit}, Mean: {mean_speed:.1f} {speed_unit}"
    
    ax.set_title(title, fontsize=12, pad=20)
    
    plotter.fig = fig
    plotter.ax = ax
    
    output_path = os.path.join(output_dir, "mpas_wind_streamlines_example")
    plotter.add_timestamp_and_branding()
    plotter.save_plot(output_path, formats=['png'])
    plotter.close_plot()
    
    print(f"\n✓ Wind streamlines plot saved to: {output_path}.png")
    print("  - Streamlines follow wind flow trajectories")
    print("  - Color indicates wind speed magnitude")
    print("  - Regridded to regular lat-lon grid for smooth visualization")


def example_4_complex_wind_plot(
    grid_file: str,
    data_file: str,
    output_dir: str
) -> None:
    """
    Create a complex multi-layer wind plot combining shaded wind speed background with overlaid wind barbs for detailed meteorological analysis. This example demonstrates advanced visualization by integrating MPASSurfacePlotter for the wind speed contour field with MPASWindPlotter for the overlaid wind direction vectors. The result shows both magnitude (through color shading) and direction (through barbs) simultaneously, providing comprehensive wind field representation similar to operational weather analysis products. The plot includes custom colormaps, contour levels, automatic unit conversion, and professional cartographic features making it suitable for publication and operational decision-making.

    Parameters:
        grid_file (str): Absolute path to MPAS grid NetCDF file containing mesh coordinates and connectivity.
        data_file (str): Absolute path to MPAS diagnostic file containing u10 and v10 wind component variables.
        output_dir (str): Directory path where the complex wind plot file will be saved with descriptive naming.

    Returns:
        None: Creates and saves complex wind visualization to output_dir/mpas_wind_complex_example.png.
    """
    print("="*70)
    print("EXAMPLE 4: Complex Wind Plot (Wind Speed + Wind Barbs)")
    print("="*70)
    
    processor = MPAS2DProcessor(grid_file=grid_file)
    data_dir = os.path.dirname(data_file)
    processor.load_2d_data(data_dir)
    
    time_idx = 0
    u_data = processor.get_2d_variable_data('u10', time_idx)
    v_data = processor.get_2d_variable_data('v10', time_idx)
    lon, lat = processor.extract_2d_coordinates_for_variable('u10', u_data)
    
    wind_speed = np.sqrt(u_data.values**2 + v_data.values**2)
    
    lon_min, lon_max = -130, -60
    lat_min, lat_max = 20, 55
    
    surface_plotter = MPASSurfacePlotter(figsize=(14, 11), dpi=150)
    
    time_str = MPASDateTimeUtils.get_time_info(processor.dataset, time_idx, var_context='wind', verbose=False)
    time_stamp = pd.to_datetime(processor.dataset.Time.values[time_idx]).to_pydatetime() if hasattr(processor.dataset, 'Time') else None
    
    print("Creating wind speed background...")
    fig, ax = surface_plotter.create_surface_map(
        lon=lon,
        lat=lat,
        data=wind_speed,
        var_name='wind_speed',
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        title="MPAS 10-m Wind Analysis - Wind Speed with Direction Barbs",
        plot_type='contourf',
        grid_resolution=100,
        colormap='YlOrRd',
        time_stamp=time_stamp,
        projection='PlateCarree'
    )
    
    print("Adding wind barb overlay...")
    wind_plotter = MPASWindPlotter()
    
    wind_config = {
        'u_data': u_data.values,
        'v_data': v_data.values,
        'plot_type': 'barbs',
        'subsample': -1,
        'color': 'black',
        'scale': None,
        'grid_resolution': 0.1,
        'regrid_method': 'linear'
    }
    
    wind_plotter.add_wind_overlay(
        ax=ax,
        lon=lon,
        lat=lat,
        wind_config=wind_config,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max
    )
    
    wind_speed_max = np.max(wind_speed)
    wind_speed_mean = np.mean(wind_speed)
    
    wind_speed_metadata = MPASFileMetadata.get_variable_metadata('wind_speed')
    speed_unit = wind_speed_metadata.get('units', 'm s^{-1}')
    
    current_title = ax.get_title()
    new_title = current_title + f"\nMax: {wind_speed_max:.1f} {speed_unit}, Mean: {wind_speed_mean:.1f} {speed_unit}"
    ax.set_title(new_title, fontsize=12, pad=20)
    
    surface_plotter.fig = fig
    surface_plotter.ax = ax
    
    output_path = os.path.join(output_dir, "mpas_wind_complex_example")
    surface_plotter.add_timestamp_and_branding()
    surface_plotter.save_plot(output_path, formats=['png'])
    surface_plotter.close_plot()
    
    print(f"\n✓ Complex wind plot saved to: {output_path}.png")
    print("  - Background shows wind speed magnitude (colored contours)")
    print("  - Wind speed regridded to regular grid for smooth contours")
    print("  - Overlaid wind barbs regridded to 0.1° resolution")
    print("  - Automatic intelligent subsampling for optimal barb density")
    print("  - Combines MPASSurfacePlotter + MPASWindPlotter")
    print("  - Publication-quality multi-layer visualization")


def main():
    """
    Main execution function orchestrating all four wind visualization examples with error handling and user guidance. This function sets up the example environment by defining input MPAS file path and output directory, creates the output directory if needed, executes all four example functions in sequence (wind barbs, wind arrows, streamlines, complex plot), provides comprehensive progress reporting with section separators, handles errors gracefully with descriptive messages, and summarizes the complete example run with output locations and visualization descriptions.

    Returns:
        None: Executes all wind plotting examples and saves visualizations to the output directory.
    """
    print("\n" + "="*70)
    print("MPAS WIND PLOTTER - COMPREHENSIVE EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates four different wind visualization techniques:")
    print("  1. Wind Barbs - Meteorological convention with flags")
    print("  2. Wind Arrows - Vector representation with quiver")
    print("  3. Wind Streamlines - Flow trajectories and patterns")
    print("  4. Complex Plot - Wind speed background + wind barbs overlay")
    print("\n" + "="*70)
    
    grid_file = "../data/grids/x1.40962.init.nc"
    data_file = "../data/u120k/diag/diag.2024-09-17_02.00.00.nc"
    output_dir = "testPlot/wind"
    
    if not os.path.exists(grid_file):
        print(f"\n❌ ERROR: Grid file not found: {grid_file}")
        print("\nPlease update the 'grid_file' path in the script to point to your MPAS grid file.")
        return
    
    if not os.path.exists(data_file):
        print(f"\n❌ ERROR: Data file not found: {data_file}")
        print("\nPlease update the 'data_file' path in the script to point to your MPAS diagnostic file.")
        print("The file should contain 'u10' and 'v10' variables (10-meter winds).")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Grid file: {grid_file}")
    print(f"✓ Data file: {data_file}")
    print(f"✓ Output directory: {output_dir}")
    
    try:
        example_1_wind_barbs(grid_file, data_file, output_dir)
        
        example_2_wind_arrows(grid_file, data_file, output_dir)
        
        example_3_wind_streamlines(grid_file, data_file, output_dir)
        
        example_4_complex_wind_plot(grid_file, data_file, output_dir)
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nGenerated files in: {output_dir}/")
        print("  1. mpas_wind_barbs_example.png - Classic meteorological wind barbs")
        print("  2. mpas_wind_arrows_example.png - Vector arrows showing wind flow")
        print("  3. mpas_wind_streamlines_example.png - Continuous flow trajectories")
        print("  4. mpas_wind_complex_example.png - Multi-layer wind analysis")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease check:")
        print("  - Grid file and data file paths are correct")
        print("  - Data file contains 'u10' and 'v10' variables")
        print("  - Data file has valid Time dimension")
        print("  - Output directory is writable")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
