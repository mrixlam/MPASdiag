#!/usr/bin/env python3

"""
MPASdiag Example1: Complex Weather Map

This script demonstrates advanced composite plotting by overlaying multiple meteorological
variables at 850 hPa level:
- Specific humidity (shaded background)
- Wind vectors (barbs)
- Geopotential height (contour lines)

This type of composite plot is commonly used in operational meteorology for
analyzing moisture transport, wind patterns, and synoptic features.

Features showcased:
- Multi-variable composite plotting
- Automatic unit conversion (kg/kg→g/kg, Pa→hPa, m→gpm)
- Professional meteorological visualization
- Overlay techniques for different plot types
- Enhanced scientific notation and labeling

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union

from mpasdiag.processing import UnitConverter
from mpasdiag.processing import MPAS2DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter

def setup_warnings() -> None:
    """
    Configure warning filters to suppress common third-party library messages for cleaner output.
    This function applies warning filters to silence known cartopy and shapely deprecation warnings
    that do not affect functionality. By suppressing these routine warnings, the console output
    becomes more readable and focuses on relevant diagnostic messages from the analysis workflow.
    
    Returns:
        None
    """
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')

def create_complex_weather_map(
    processor: MPAS2DProcessor,
    visualizer: MPASSurfacePlotter,
    time_index: Union[int, slice],
    extent: Tuple[float, float, float, float],
    output_dir: str,
) -> str:
    """
    Generate a publication-quality composite meteorological map at the 850 hPa pressure level.
    This function creates an advanced multi-variable visualization by combining specific humidity
    shading, geopotential height contours, and horizontal wind barbs into a single comprehensive
    plot. The function automatically handles unit conversions and applies professional styling with
    legends and annotations. It raises an error if required variables are missing.
    
    Parameters:
        processor (MPAS2DProcessor): Processor instance for reading MPAS diagnostic fields and extracting 2D grids.
        visualizer (MPASVisualizer): Visualization helper providing styling utilities and branding elements.
        time_index (int or slice): Index or slice into the processor time dimension for selecting analysis time.
        extent (tuple of float): Geographic domain as (lon_min, lon_max, lat_min, lat_max) in degrees.
        output_dir (str): Directory path to save the generated PNG file.
    
    Returns:
        str: Absolute path to the saved composite plot PNG file.
    """
    if isinstance(time_index, slice):
        time_index = 0 
    
    lon_min, lon_max, lat_min, lat_max = extent
    time_str = processor.get_time_info(time_index)
    
    variables_850 = {
        'humidity': ['relhum_850hPa', 'humidity_850hPa', 'q_850hPa', 'qv_850hPa', 'specific_humidity_850hPa'],
        'u_wind': ['uzonal_850hPa', 'u_850hPa', 'u850', 'uwind_850hPa'],
        'v_wind': ['umeridional_850hPa', 'v_850hPa', 'v850', 'vwind_850hPa'],
        'geopotential': ['height_500hPa', 'geopotential_500hPa', 'z_500hPa', 'gph_500hPa']
    }
    
    available_vars = processor.get_available_variables()    
    found_vars = {}

    for var_type, var_list in variables_850.items():
        for var_name in var_list:
            if var_name in available_vars:
                found_vars[var_type] = var_name
                print(f"✓ Found {var_type}: {var_name}")
                break
        else:
            print(f"⚠️  No {var_type} variable found at 850 hPa level")
    
    required_vars = ['humidity', 'u_wind', 'v_wind', 'geopotential']
    missing_vars = [var for var in required_vars if var not in found_vars]
    
    if missing_vars:
        raise RuntimeError(f"Missing required variables for 850 hPa composite: {missing_vars}")
    
    humidity_var = found_vars['humidity']
    humidity_data = processor.get_2d_variable_data(humidity_var, time_index)
    humidity_lon, humidity_lat = processor.extract_2d_coordinates_for_variable(humidity_var, humidity_data)
    
    u_var = found_vars['u_wind']
    v_var = found_vars['v_wind']
    geopotential_var = found_vars['geopotential']

    u_data = processor.get_2d_variable_data(u_var, time_index)
    v_data = processor.get_2d_variable_data(v_var, time_index)
    wind_lon, wind_lat = processor.extract_2d_coordinates_for_variable(u_var, u_data)
    
    geopotential_data = processor.get_2d_variable_data(geopotential_var, time_index)
    gph_lon, gph_lat = processor.extract_2d_coordinates_for_variable(geopotential_var, geopotential_data)
    
    converted_humidity, _ = UnitConverter.convert_data_for_display(humidity_data, humidity_var, humidity_data)
    converted_geopotential, _ = UnitConverter.convert_data_for_display(geopotential_data, geopotential_var, geopotential_data)
    
    if isinstance(converted_humidity, (int, float)):
        converted_humidity = np.array([converted_humidity])

    if isinstance(converted_geopotential, (int, float)):
        converted_geopotential = np.array([converted_geopotential])
    
    humidity_values = getattr(converted_humidity, 'values', converted_humidity)
    geopotential_values = getattr(converted_geopotential, 'values', converted_geopotential)
    
    fig, ax = visualizer.create_surface_map(
        humidity_lon, humidity_lat, humidity_values,
        humidity_var, lon_min, lon_max, lat_min, lat_max,
        plot_type='contourf',
        colormap='BuGn'
    )

    try:
        gph_min = np.nanmin(geopotential_values)
        gph_max = np.nanmax(geopotential_values)

        gph_levels = list(np.arange(np.floor(gph_min / 60) * 60, np.ceil(gph_max / 60) * 60 + 1, 60))

        surface_cfg = {
            'data': geopotential_values,
            'var_name': geopotential_var,
            'plot_type': 'contour',
            'levels': gph_levels,
            'colors': 'black',
            'linewidth': 1.2,
            'alpha': 0.9,
            'add_labels': True,
        }

        visualizer._add_surface_overlay(gph_lon, gph_lat, surface_cfg)
    except Exception:
        ax.scatter(gph_lon, gph_lat, c=geopotential_values, s=2, cmap='gray', alpha=0.3)
    
    max_barbs = 1200
    n_points = len(wind_lon)
    n_select = min(n_points, max_barbs)

    if n_select <= 0:
        n_select = 0

    if n_select > 0:
        indices = np.linspace(0, n_points - 1, num=n_select, dtype=int)
        wind_lon_sub = wind_lon[indices]
        wind_lat_sub = wind_lat[indices]
        u_sub = u_data.values[indices]
        v_sub = v_data.values[indices]

        u_kts = UnitConverter.convert_units(u_sub, 'm/s', 'kts')
        v_kts = UnitConverter.convert_units(v_sub, 'm/s', 'kts')

        barbs = ax.barbs(
            wind_lon_sub, wind_lat_sub, u_kts, v_kts,
            length=4,
            barbcolor='darkred',
            flagcolor='darkred',
            linewidth=0.6,
            alpha=0.8,
            pivot='middle'
        )
    else:
        barbs = None
    
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    title = f"Weather Map | Valid: {time_str} | Humidity (shaded) Height (contour) Wind (barbs)"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    visualizer.fig = fig
    visualizer.add_timestamp_and_branding()
    
    output_path = os.path.join(output_dir, f"mpasdiag_sample_plot_complex_weather_map_{time_str}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    
    plt.close()
    return output_path


def main() -> Union[str, None]:
    """
    Execute the complete 850 hPa composite analysis workflow from configuration to visualization.
    This driver function orchestrates the entire composite plotting process including environment
    setup, file path configuration, data loading, and visualization generation. The function
        raises an error if required MPAS diagnostic files are unavailable, providing an informative message.
    
    Returns:
        Union[str, None]: Path to the generated composite plot image file, or None if creation fails.
    """
    setup_warnings()

    print("=" * 80)
    print("MPASdiag Example1: Complex Weather Map")
    print("=" * 80)

    grid_file = "../data/grids/x1.2621442.init.nc"
    data_dir = "../data/u15k/"
    output_dir = "testPlot/"

    extent = (-180.0, 180.0, -90.0, 90.0)

    print("\nConfiguration:")
    print(f"  grid_file: {grid_file}")
    print(f"  data_dir:  {data_dir}")
    print(f"  output_dir:{output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    from mpasdiag.processing.constants import DIAG_GLOB
    diag_files = glob.glob(os.path.join(data_dir, DIAG_GLOB)) if os.path.exists(data_dir) else []

    if not os.path.exists(grid_file) or not diag_files:
        print("Data files not found. Update `grid_file` and `data_dir` to point to MPAS files.")
        return None

    processor = MPAS2DProcessor(grid_file, verbose=True)
    processor.load_2d_data(data_dir, use_pure_xarray=False)
    visualizer = MPASSurfacePlotter()

    time_index = 1

    composite_plot_path = create_complex_weather_map(
        processor, visualizer, time_index, extent, output_dir
    )

    return composite_plot_path

if __name__ == "__main__":
    main()