#!/usr/bin/env python3
"""
MPAS Wind Vector Analysis Example

This example demonstrates how to create wind vector plots from MPAS
model output using the mpas-analysis toolkit. It covers wind barbs,
arrow vectors, different atmospheric levels, and advanced visualization techniques.

Requirements:
- MPAS grid file (e.g., grid.nc)
- MPAS diagnostic files containing wind data (u10, v10, u850, v850, etc.)
- mpas-analysis package installed

Author: Rubaiat Islam
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from mpas_analysis.data_processing import MPASDataProcessor
from mpas_analysis.visualization import MPASVisualizer
from mpas_analysis.utils import MPASConfig, MPASLogger

def example_surface_wind_barbs():
    """
    Example 1: Surface wind analysis with meteorological wind barbs.

    Parameters:
        None

    Returns:
        None
    """
    print("=== Example 1: Surface Wind Barbs ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/wind/",
        time_index=12, 
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0,
        u_variable="u10",
        v_variable="v10",
        wind_level="surface",
        wind_plot_type="barbs",
        subsample_factor=5
    )
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
    
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    u_data, v_data = processor.get_wind_components(
        config.u_variable, config.v_variable, config.time_index
    )
    
    lon, lat = processor.extract_spatial_coordinates()
    
    fig, ax = visualizer.create_wind_plot(
        lon, lat, u_data.values, v_data.values,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
        wind_level=config.wind_level,
        plot_type=config.wind_plot_type,
        subsample=config.subsample_factor,
        title="MPAS Surface Wind Analysis - Wind Barbs"
    )
    
    output_file = Path(config.output_dir) / "surface_wind_barbs"
    visualizer.save_plot(output_file, formats=['png', 'pdf'])
    
    print(f"✅ Surface wind barbs saved to: {output_file}")


def example_upper_level_wind_arrows():
    """
    Example 2: Upper-level wind analysis with arrow vectors.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 2: 850mb Wind Arrows ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/wind/",
        time_index=6,  
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0,
        u_variable="u850",
        v_variable="v850",
        wind_level="850mb",
        wind_plot_type="arrows",
        subsample_factor=3,  
        wind_scale=800.0 
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
    
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    try:
        u_data, v_data = processor.get_wind_components(
            config.u_variable, config.v_variable, config.time_index
        )
        
        lon, lat = processor.extract_spatial_coordinates()
        
        fig, ax = visualizer.create_wind_plot(
            lon, lat, u_data.values, v_data.values,
            config.lon_min, config.lon_max,
            config.lat_min, config.lat_max,
            wind_level=config.wind_level,
            plot_type=config.wind_plot_type,
            subsample=config.subsample_factor,
            scale=config.wind_scale,
            title="MPAS 850mb Wind Analysis - Vector Arrows"
        )
        
        output_file = Path(config.output_dir) / "850mb_wind_arrows"
        visualizer.save_plot(output_file, formats=['png', 'pdf'])
        
        print(f"✅ 850mb wind arrows saved to: {output_file}")
        
    except ValueError as e:
        print(f"⚠️ Could not find 850mb wind data: {e}")
        print("💡 Try using surface winds (u10, v10) instead")


def example_wind_with_background():
    """
    Example 3: Wind vectors with background wind speed.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 3: Wind Vectors with Speed Background ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/wind/",
        time_index=18,  
        lat_min=-8.0, lat_max=12.0,
        lon_min=95.0, lon_max=108.0,
        u_variable="u10",
        v_variable="v10",
        wind_level="surface",
        wind_plot_type="barbs",
        subsample_factor=4,
        show_background=True,
        background_colormap="plasma"
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
    
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    u_data, v_data = processor.get_wind_components(
        config.u_variable, config.v_variable, config.time_index
    )
    
    lon, lat = processor.extract_spatial_coordinates()
    
    fig, ax = visualizer.create_wind_plot(
        lon, lat, u_data.values, v_data.values,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
        wind_level=config.wind_level,
        plot_type=config.wind_plot_type,
        subsample=config.subsample_factor,
        show_background=config.show_background,
        bg_colormap=config.background_colormap,
        title="MPAS Surface Winds with Speed Background"
    )
    
    output_file = Path(config.output_dir) / "wind_with_background"
    visualizer.save_plot(output_file, formats=['png', 'pdf'])
    
    print(f"✅ Wind with background saved to: {output_file}")


def example_high_resolution_wind():
    """
    Example 4: High-resolution wind analysis for detailed regions.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 4: High-Resolution Regional Wind Analysis ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/wind/",
        time_index=15,  
        lat_min=2.0, lat_max=6.0,
        lon_min=101.0, lon_max=105.0,
        u_variable="u10",
        v_variable="v10",
        wind_level="surface",
        wind_plot_type="barbs",
        subsample_factor=2,  
        show_background=True,
        background_colormap="viridis"
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)  
    
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    u_data, v_data = processor.get_wind_components(
        config.u_variable, config.v_variable, config.time_index
    )
    
    lon, lat = processor.extract_spatial_coordinates()
    
    filtered_u = processor.filter_by_spatial_extent(
        u_data, config.lon_min, config.lon_max, 
        config.lat_min, config.lat_max
    )
    filtered_v = processor.filter_by_spatial_extent(
        v_data, config.lon_min, config.lon_max, 
        config.lat_min, config.lat_max
    )
    
    fig, ax = visualizer.create_wind_plot(
        lon, lat, filtered_u.values, filtered_v.values,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
        wind_level=config.wind_level,
        plot_type=config.wind_plot_type,
        subsample=config.subsample_factor,
        show_background=config.show_background,
        bg_colormap=config.background_colormap,
        title="High-Resolution MPAS Surface Wind Analysis"
    )
    
    output_file = Path(config.output_dir) / "high_resolution_wind"
    visualizer.save_plot(output_file, formats=['png', 'pdf', 'svg'])
    
    print(f"✅ High-resolution wind analysis saved to: {output_file}")


def example_multi_level_wind_comparison():
    """
    Example 5: Compare winds at multiple atmospheric levels.

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 5: Multi-Level Wind Comparison ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/wind/",
        time_index=9,  
        lat_min=-10.0, lat_max=15.0,
        lon_min=91.0, lon_max=113.0,
        wind_plot_type="barbs",
        subsample_factor=6
    )
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    wind_levels = [
        ("u10", "v10", "Surface (10m)", "surface"),
        ("u850", "v850", "850mb Level", "850mb"),
        ("u500", "v500", "500mb Level", "500mb"),
        ("u200", "v200", "200mb Level", "200mb")
    ]
    
    lon, lat = processor.extract_spatial_coordinates()
    
    for u_var, v_var, title, level_name in wind_levels:
        try:
            visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
            
            u_data, v_data = processor.get_wind_components(
                u_var, v_var, config.time_index
            )
            
            fig, ax = visualizer.create_wind_plot(
                lon, lat, u_data.values, v_data.values,
                config.lon_min, config.lon_max,
                config.lat_min, config.lat_max,
                wind_level=level_name,
                plot_type=config.wind_plot_type,
                subsample=config.subsample_factor,
                title=f"MPAS {title} Wind Analysis"
            )
            
            output_file = Path(config.output_dir) / f"wind_comparison_{level_name}"
            visualizer.save_plot(output_file, formats=['png'])
            
            print(f"✅ {title} wind analysis saved to: {output_file}.png")
            
        except ValueError as e:
            print(f"⚠️ Could not process {level_name} winds: {e}")


def example_time_series_wind():
    """
    Example 6: Wind evolution over time (multiple time steps).

    Parameters:
        None

    Returns:
        None
    """
    print("\n=== Example 6: Wind Evolution Time Series ===")
    
    config = MPASConfig(
        grid_file="data/grid.nc",
        data_dir="data/diagnostics/",
        output_dir="output/wind/time_series/",
        lat_min=-5.0, lat_max=10.0,
        lon_min=100.0, lon_max=107.0,
        u_variable="u10",
        v_variable="v10",
        wind_level="surface",
        wind_plot_type="barbs",
        subsample_factor=4,
        show_background=True,
        background_colormap="plasma"
    )
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    processor = MPASDataProcessor(config.grid_file, verbose=True)
    processor.load_data(config.data_dir, pattern="diag*.nc")
    
    start_time, end_time = processor.get_time_range()
    time_steps = list(range(0, min(24, len(processor.dataset.Time)), 3))  
    
    lon, lat = processor.extract_spatial_coordinates()
    
    print(f"Creating wind plots for {len(time_steps)} time steps...")
    
    for i, time_idx in enumerate(time_steps):
        try:
            visualizer = MPASVisualizer(figsize=(10, 12), dpi=300)
            
            u_data, v_data = processor.get_wind_components(
                config.u_variable, config.v_variable, time_idx
            )
            
            time_stamp = processor.dataset.Time[time_idx].values
            if hasattr(time_stamp, 'item'):
                time_stamp = time_stamp.item()
            
            time_str = f"Time Step {time_idx:02d}"
            if isinstance(time_stamp, (np.datetime64, datetime)):
                try:
                    time_str = pd.to_datetime(time_stamp).strftime('%Y-%m-%d %H:%M UTC')
                except:
                    pass
            
            fig, ax = visualizer.create_wind_plot(
                lon, lat, u_data.values, v_data.values,
                config.lon_min, config.lon_max,
                config.lat_min, config.lat_max,
                wind_level=config.wind_level,
                plot_type=config.wind_plot_type,
                subsample=config.subsample_factor,
                show_background=config.show_background,
                bg_colormap=config.background_colormap,
                title=f"MPAS Surface Wind Evolution\\n{time_str}"
            )
            
            output_file = Path(config.output_dir) / f"wind_evolution_{time_idx:02d}"
            visualizer.save_plot(output_file, formats=['png'])
            
            print(f"✅ Time step {time_idx:02d} saved to: {output_file}.png")
            
        except Exception as e:
            print(f"⚠️ Could not process time step {time_idx}: {e}")
    
    print(f"🎬 Wind evolution series completed! Check {config.output_dir}")


def main():
    """
    Run all wind vector analysis examples.

    Parameters:
        None

    Returns:
        int: Exit status code (0 for success).
    """
    print("💨 MPAS Wind Vector Analysis Examples")
    print("====================================")
    
    Path("output/wind").mkdir(parents=True, exist_ok=True)
    
    try:
        example_surface_wind_barbs()
        example_upper_level_wind_arrows()
        example_wind_with_background()
        example_high_resolution_wind()
        example_multi_level_wind_comparison()
        example_time_series_wind()
        
        print("\n🎉 All wind vector examples completed successfully!")
        print("📁 Check the 'output/wind/' directory for generated plots.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required data files not found: {e}")
        print("💡 Make sure you have:")
        print("   - MPAS grid file at: data/grid.nc")
        print("   - MPAS diagnostic files at: data/diagnostics/diag*.nc")
        print("   - Wind variables (u10, v10, etc.) in the diagnostic files")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())