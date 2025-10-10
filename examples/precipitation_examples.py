#!/usr/bin/env python3
"""
MPAS Precipitation Analysis Example

This example demonstrates how to create precipitation analysis plots from MPAS
model output using the mpas-analysis toolkit. It shows various precipitation
variables, accumulation periods, and visualization options.

Requirements:
- MPAS grid file (e.g., grid.nc)
- MPAS diagnostic files containing precipitation data
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

def example_basic_precipitation_plot():
    """
    Example 1: Basic precipitation plot with default settings.

    Parameters:
        None

    Returns:
        None
    """
    print("=== Example 1: Basic Precipitation Plot ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/",
        variable="rainnc",  
        time_index=0,
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
    
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    precip_data = processor.compute_precipitation_difference(
        time_index=config.time_index,
        var_name=config.variable
    )
    
    lon, lat = processor.extract_spatial_coordinates()
    
    fig, ax = visualizer.create_precipitation_map(
        lon, lat, precip_data.values,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
        title="MPAS Non-Convective Precipitation (1-hour)",
        accum_period="a01h"
    )
    
    output_file = Path(config.output_dir) / "basic_precipitation_plot"
    visualizer.save_plot(output_file, formats=['png', 'pdf'])
    
    print(f"✅ Basic precipitation plot saved to: {output_file}")


def example_multi_accumulation_precipitation():
    """
    Example 2: Multiple accumulation periods comparison.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 2: Multiple Accumulation Periods ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/",
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    accumulations = [
        ("a01h", "1-Hour Accumulation", 0),
        ("a03h", "3-Hour Accumulation", 2),
        ("a06h", "6-Hour Accumulation", 5),
        ("a24h", "24-Hour Accumulation", 23)
    ]
    
    for accum_code, title_suffix, time_idx in accumulations:
        visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
        
        precip_data = processor.compute_precipitation_difference(
            time_index=time_idx,
            var_name="rainnc"
        )
        
        lon, lat = processor.extract_spatial_coordinates()
        
        fig, ax = visualizer.create_precipitation_map(
            lon, lat, precip_data.values,
            config.lon_min, config.lon_max,
            config.lat_min, config.lat_max,
            title=f"MPAS Precipitation - {title_suffix}",
            accum_period=accum_code
        )
        
        output_file = Path(config.output_dir) / f"precipitation_{accum_code}"
        visualizer.save_plot(output_file, formats=['png'])
        
        print(f"✅ {title_suffix} plot saved to: {output_file}.png")


def example_precipitation_comparison():
    """
    Example 3: Compare convective vs non-convective precipitation.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 3: Convective vs Non-Convective Comparison ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/",
        time_index=12,  
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    precip_types = [
        ("rainc", "Convective Precipitation", "Reds"),
        ("rainnc", "Non-Convective Precipitation", "Blues"),
        ("total", "Total Precipitation", "viridis")
    ]
    
    for var_name, title, colormap in precip_types:
        visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
        
        precip_data = processor.compute_precipitation_difference(
            time_index=config.time_index,
            var_name=var_name
        )
        
        lon, lat = processor.extract_spatial_coordinates()
        
        fig, ax = visualizer.create_precipitation_map(
            lon, lat, precip_data.values,
            config.lon_min, config.lon_max,
            config.lat_min, config.lat_max,
            title=f"MPAS {title} (Hour 12)",
            colormap=colormap,
            accum_period="a01h"
        )
        
        output_file = Path(config.output_dir) / f"precipitation_{var_name}_comparison"
        visualizer.save_plot(output_file, formats=['png', 'pdf'])
        
        print(f"✅ {title} comparison saved to: {output_file}")


def example_high_resolution_precipitation():
    """
    Example 4: High-resolution precipitation analysis with custom settings.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 4: High-Resolution Analysis ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/",
        variable="total",  
        time_index=18,
        lat_min=0.0, lat_max=8.0,
        lon_min=100.0, lon_max=108.0
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)  
    
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    precip_data = processor.compute_precipitation_difference(
        time_index=config.time_index,
        var_name=config.variable
    )
    
    lon, lat = processor.extract_spatial_coordinates()
    
    filtered_precip = processor.filter_by_spatial_extent(
        precip_data, config.lon_min, config.lon_max, 
        config.lat_min, config.lat_max
    )
    
    custom_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    fig, ax = visualizer.create_precipitation_map(
        lon, lat, filtered_precip.values,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
        title="High-Resolution MPAS Total Precipitation Analysis",
        levels=custom_levels,
        colormap="plasma",
        accum_period="a01h"
    )
    
    output_file = Path(config.output_dir) / "high_resolution_precipitation"
    visualizer.save_plot(output_file, formats=['png', 'pdf', 'svg'])
    
    print(f"✅ High-resolution precipitation analysis saved to: {output_file}")


def main():
    """
    Run all precipitation analysis examples.

    Parameters:
        None

    Returns:
        int: Exit status code (0 for success).
    """
    print("🌧️  MPAS Precipitation Analysis Examples")
    print("=========================================")
    
    Path("output").mkdir(exist_ok=True)
    
    try:
        example_basic_precipitation_plot()
        example_multi_accumulation_precipitation()
        example_precipitation_comparison()
        example_high_resolution_precipitation()
        
        print("\n🎉 All precipitation examples completed successfully!")
        print("📁 Check the 'output/' directory for generated plots.")
        
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