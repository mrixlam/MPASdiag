#!/usr/bin/env python3
"""
MPAS Surface Variable Analysis Example

This example demonstrates how to create surface variable plots from MPAS
model output using the mpas-analysis toolkit. It covers temperature, pressure,
humidity, and wind speed variables with different visualization techniques.

Requirements:
- MPAS grid file (e.g., grid.nc)
- MPAS diagnostic files containing surface variables
- mpas-analysis package installed

Author: Rubaiat Islam
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

from mpas_analysis.data_processing import MPASDataProcessor
from mpas_analysis.visualization import MPASVisualizer
from mpas_analysis.utils import MPASConfig, MPASLogger

def example_temperature_analysis():
    """
    Example 1: 2-meter temperature analysis with scatter and contour plots.

    Parameters:
        None

    Returns:
        None
    """
    print("=== Example 1: 2-Meter Temperature Analysis ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/surface/",
        variable="t2m",  
        time_index=12,  
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0
    )
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    temp_data = processor.get_variable_data(config.variable, config.time_index)
    lon, lat = processor.extract_spatial_coordinates()
    
    if hasattr(temp_data, 'units') and 'K' in temp_data.units:
        temp_data = temp_data - 273.15
        temp_units = "°C"
    else:
        temp_units = getattr(temp_data, 'units', '°C')
    
    plot_types = [
        ("scatter", "Scatter Plot"),
        ("contour", "Contour Plot")
    ]
    
    for plot_type, plot_name in plot_types:
        visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
        
        if plot_type == "scatter":
            fig, ax = visualizer.create_simple_scatter_plot(
                lon, lat, temp_data.values,
                config.lon_min, config.lon_max,
                config.lat_min, config.lat_max,
                title=f"MPAS 2m Temperature - {plot_name}",
                colormap="RdYlBu_r",
                var_units=temp_units
            )
        else:
            fig, ax = visualizer.create_contour_plot(
                lon, lat, temp_data.values,
                config.lon_min, config.lon_max,
                config.lat_min, config.lat_max,
                title=f"MPAS 2m Temperature - {plot_name}",
                colormap="RdYlBu_r",
                var_units=temp_units
            )
        
        output_file = Path(config.output_dir) / f"temperature_{plot_type}"
        visualizer.save_plot(output_file, formats=['png'])
        
        print(f"✅ Temperature {plot_name} saved to: {output_file}.png")


def example_pressure_analysis():
    """
    Example 2: Surface pressure analysis with custom contour levels.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 2: Surface Pressure Analysis ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/surface/",
        variable="surface_pressure",
        time_index=0,
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    pressure_data = processor.get_variable_data(config.variable, config.time_index)
    lon, lat = processor.extract_spatial_coordinates()
    
    if hasattr(pressure_data, 'units') and 'Pa' in pressure_data.units:
        pressure_data = pressure_data / 100.0  
        pressure_units = "hPa"
    else:
        pressure_units = getattr(pressure_data, 'units', 'hPa')
    
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
    
    pressure_min = float(pressure_data.min())
    pressure_max = float(pressure_data.max())
    pressure_levels = np.arange(
        int(pressure_min/2)*2, 
        int(pressure_max/2)*2 + 4, 
        2
    )  
    
    fig, ax = visualizer.create_contour_plot(
        lon, lat, pressure_data.values,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
        title="MPAS Surface Pressure Analysis",
        colormap="viridis",
        levels=pressure_levels.tolist(),
        var_units=pressure_units
    )
    
    output_file = Path(config.output_dir) / "surface_pressure"
    visualizer.save_plot(output_file, formats=['png', 'pdf'])
    
    print(f"✅ Surface pressure analysis saved to: {output_file}")


def example_humidity_analysis():
    """
    Example 3: 2-meter specific humidity with relative humidity calculation.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 3: Humidity Analysis ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/surface/",
        time_index=18, 
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    humidity_vars = ['q2', 'q2m', 'specific_humidity_2m', 'qv2m']
    humidity_data = None
    
    for var in humidity_vars:
        try:
            humidity_data = processor.get_variable_data(var, config.time_index)
            humidity_var = var
            break
        except ValueError:
            continue
    
    if humidity_data is None:
        print("⚠️ No humidity variable found. Available variables:")
        print(processor.get_available_variables()[:10])
        return
    
    lon, lat = processor.extract_spatial_coordinates()
    
    if hasattr(humidity_data, 'units') and 'kg/kg' in humidity_data.units:
        humidity_data = humidity_data * 1000.0  
        humidity_units = "g/kg"
    else:
        humidity_units = getattr(humidity_data, 'units', 'g/kg')
    
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
    
    fig, ax = visualizer.create_simple_scatter_plot(
        lon, lat, humidity_data.values,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
        title=f"MPAS 2m Specific Humidity ({humidity_var})",
        colormap="BuGn",
        var_units=humidity_units
    )
    
    output_file = Path(config.output_dir) / "humidity_analysis"
    visualizer.save_plot(output_file, formats=['png'])
    
    print(f"✅ Humidity analysis saved to: {output_file}.png")


def example_wind_speed_analysis():
    """
    Example 4: 10-meter wind speed calculation and visualization.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 4: Wind Speed Analysis ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/surface/",
        time_index=6, 
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    try:
        u_data, v_data = processor.get_wind_components('u10', 'v10', config.time_index)
        
        wind_speed = np.sqrt(u_data.values**2 + v_data.values**2)
        
        lon, lat = processor.extract_spatial_coordinates()
        
        visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
        
        wind_levels = [0, 2, 4, 6, 8, 10, 12, 15, 18, 20, 25]
        
        fig, ax = visualizer.create_contour_plot(
            lon, lat, wind_speed,
            config.lon_min, config.lon_max,
            config.lat_min, config.lat_max,
            title="MPAS 10m Wind Speed",
            colormap="plasma",
            levels=wind_levels,
            var_units="m/s"
        )
        
        output_file = Path(config.output_dir) / "wind_speed_analysis"
        visualizer.save_plot(output_file, formats=['png', 'pdf'])
        
        print(f"✅ Wind speed analysis saved to: {output_file}")
        
        print(f"📊 Wind Speed Statistics:")
        print(f"   - Mean: {np.mean(wind_speed):.2f} m/s")
        print(f"   - Max:  {np.max(wind_speed):.2f} m/s")
        print(f"   - Min:  {np.min(wind_speed):.2f} m/s")
        
    except ValueError as e:
        print(f"⚠️ Could not extract wind components: {e}")
        print("Available variables:", processor.get_available_variables()[:10])


def example_multi_variable_comparison():
    """
    Example 5: Compare multiple surface variables in a grid layout.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 5: Multi-Variable Comparison ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/surface/",
        time_index=12,
        lat_min=-5.0, lat_max=10.0,
        lon_min=95.0, lon_max=110.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    variables = [
        ("t2m", "2m Temperature", "RdYlBu_r", "°C", -273.15),
        ("surface_pressure", "Surface Pressure", "viridis", "hPa", 0.01),
        ("q2", "2m Humidity", "BuGn", "g/kg", 1000),
    ]
    
    lon, lat = processor.extract_spatial_coordinates()
    
    for var_name, title, colormap, units, conversion in variables:
        try:
            var_data = processor.get_variable_data(var_name, config.time_index)
            
            if conversion != 0:
                if conversion == -273.15:  
                    var_data = var_data + conversion
                else:  
                    var_data = var_data * conversion
            
            visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
            
            fig, ax = visualizer.create_simple_scatter_plot(
                lon, lat, var_data.values,
                config.lon_min, config.lon_max,
                config.lat_min, config.lat_max,
                title=f"MPAS {title}",
                colormap=colormap,
                var_units=units
            )
            
            output_file = Path(config.output_dir) / f"comparison_{var_name}"
            visualizer.save_plot(output_file, formats=['png'])
            
            print(f"✅ {title} comparison saved to: {output_file}.png")
            
        except ValueError as e:
            print(f"⚠️ Could not process {var_name}: {e}")


def main():
    """
    Run all surface variable analysis examples.

    Parameters:
        None

    Returns:
        int: Exit status code (0 for success).
    """
    print("🌡️  MPAS Surface Variable Analysis Examples")
    print("==========================================")
    
    Path("output/surface").mkdir(parents=True, exist_ok=True)
    
    try:
        example_temperature_analysis()
        example_pressure_analysis()
        example_humidity_analysis()
        example_wind_speed_analysis()
        example_multi_variable_comparison()
        
        print("\n🎉 All surface variable examples completed successfully!")
        print("📁 Check the 'output/surface/' directory for generated plots.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required data files not found: {e}")
        print("💡 Make sure you have:")
        print("   - MPAS grid file at: data/grid.nc")
        print("   - MPAS diagnostic files at: data/diagnostics/diag*.nc")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())