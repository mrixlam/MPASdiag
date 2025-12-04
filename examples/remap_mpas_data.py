#!/usr/bin/env python3

"""
Example: Remapping MPAS Data to Regular Lat-Lon Grid using xESMF

This example demonstrates how to use the MPASRemapper class to convert MPAS
unstructured mesh data to regular latitude-longitude grids using various
interpolation methods including conservative remapping.

Requirements:
    - xesmf (install via: conda install -c conda-forge xesmf)
    - MPAS grid file with coordinates
    - MPAS output data file

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from typing import Tuple, Dict
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.figure
from mpasdiag.processing import MPAS2DProcessor
from mpasdiag.processing.remapping import MPASRemapper, remap_mpas_to_latlon


def example_basic_remapping() -> xr.DataArray:
    """
    Demonstrate basic MPAS data remapping to regular latitude-longitude grid using the convenience function with optimized memory-efficient processing. This example shows the simplest workflow for converting unstructured MPAS mesh data to a regular grid using the high-level remap_mpas_to_latlon function with default nearest-neighbor interpolation. The function loads 2D temperature data from MPAS diagnostic files, extracts spatial coordinates, and performs grid transformation to a 1-degree global regular grid. Data is automatically chunked internally during loading to avoid memory issues with large datasets, and lazy loading defers computation until explicitly needed. This approach is ideal for quick data exploration and visualization workflows requiring minimal setup.

    Parameters:
        None

    Returns:
        xr.DataArray: Remapped temperature data on regular latitude-longitude grid with dimensions [lat, lon] and coordinate arrays in degrees.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Remapping (Memory Optimized)")
    print("="*60)
    
    grid_file = "data/grids/x1.40962.init.nc"
    data_dir = "data/u120k/diag"
    
    processor = MPAS2DProcessor(grid_file, verbose=True)
    processor.load_2d_data(data_dir)
    
    var_name = 't2m' 
    time_index = 0
    
    var_data = processor.get_2d_variable_data(var_name, time_index)
    lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)
    
    remapped = remap_mpas_to_latlon(
        data=var_data,  
        lon=lon,
        lat=lat,
        lon_min=-180,
        lon_max=180,
        lat_min=-90,
        lat_max=90,
        resolution=1.0
    )
    
    print(f"Original MPAS data shape: {var_data.shape}")
    print(f"Remapped data shape: {remapped.shape}")
    print(f"Data range: [{float(remapped.min().values):.2f}, {float(remapped.max().values):.2f}]")
    
    return remapped


def example_conservative_remapping() -> Tuple[xr.DataArray, MPASRemapper]:
    """
    Demonstrate conservative remapping workflow for flux variables ensuring total quantity preservation during grid transformation with memory-optimized processing. This example shows the proper methodology for remapping precipitation and other integrated quantities using conservative_normed interpolation which preserves spatial totals across grid transformations. The function initializes an MPASRemapper instance with weight caching for reuse, prepares source grid coordinates from MPAS cells, creates a target regular grid at 1-degree resolution, and builds the regridder with conservative weights. Conservative methods are essential for variables like precipitation, radiation fluxes, and mass fields where spatial integration must be preserved. The example prints conservation diagnostics showing the ratio of remapped to original totals, which should be near 1.0 for proper conservative remapping.

    Parameters:
        None

    Returns:
        Tuple[xr.DataArray, MPASRemapper]: Two-element tuple containing (remapped_precipitation_data, configured_remapper_instance) where the data array has [lat, lon] dimensions and the remapper can be reused for additional variables.
    """
    print("\n" + "="*60)
    print("Example 2: Conservative Remapping (Memory Optimized)")
    print("="*60)
    
    grid_file = "data/grids/x1.40962.init.nc"
    data_dir = "data/u120k/diag"
    
    processor = MPAS2DProcessor(grid_file, verbose=True)
    processor.load_2d_data(data_dir)
    
    var_name = 'rainnc'  
    time_index = 0
    
    var_data = processor.get_2d_variable_data(var_name, time_index)
    lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)
    
    remapper = MPASRemapper(
        method='conservative_normed',  
        weights_dir='./weights',  
        reuse_weights=True
    )
    
    remapper.prepare_source_grid(lon, lat)
    
    remapper.create_target_grid(
        lon_min=-180,
        lon_max=180,
        lat_min=-90,
        lat_max=90,
        dlon=1.0, 
        dlat=1.0 
    )
    
    remapper.build_regridder()
    
    remapped = remapper.remap(var_data)
    
    print(f"Total before remapping: {float(var_data.sum().values):.2f}")
    print(f"Total after remapping: {float(remapped.sum().values):.2f}")
    print(f"Conservation ratio: {float(remapped.sum().values) / float(var_data.sum().values):.6f}")
    
    return remapped, remapper


def example_batch_remapping() -> Tuple[Dict[str, xr.DataArray], MPASRemapper]:
    """
    Demonstrate efficient batch remapping of multiple meteorological variables using cached regridder weights to avoid redundant computation. This example shows the optimal workflow for processing multiple variables from the same MPAS mesh by building the regridder once and reusing it for all variables, dramatically improving performance compared to separate remapping operations. The function loads four surface variables (temperature, moisture, u-wind, v-wind), initializes a bilinear remapper with weight persistence, and processes each variable sequentially while explicitly managing memory through deletion of intermediate arrays. Processing variables one at a time minimizes peak memory usage compared to loading all data simultaneously. This approach is essential for operational workflows requiring consistent grid transformations across multiple fields with minimal computational overhead.

    Parameters:
        None

    Returns:
        Tuple[Dict[str, xr.DataArray], MPASRemapper]: Two-element tuple containing (remapped_variables_dict, configured_remapper) where the dictionary maps variable names to remapped DataArrays and the remapper instance can be reused for additional processing.
    """
    print("\n" + "="*60)
    print("Example 3: Batch Remapping Multiple Variables (Memory Optimized)")
    print("="*60)
    
    grid_file = "data/grids/x1.40962.init.nc"
    data_file = "data/u120k/diag/diag.2024-09-17_01.00.00.nc"
    
    processor = MPAS2DProcessor(grid_file, verbose=True)
    processor.load_2d_data(data_file)
    
    variables = ['t2m', 'q2', 'u10', 'v10']
    time_index = 0
    
    remapper = MPASRemapper(method='bilinear', weights_dir='./weights', reuse_weights=True)
    
    var_data = processor.get_2d_variable_data(variables[0], time_index)
    lon, lat = processor.extract_2d_coordinates_for_variable(variables[0], var_data)
    
    remapper.prepare_source_grid(lon, lat)
    remapper.create_target_grid(dlon=1.0, dlat=1.0) 
    remapper.build_regridder()
    
    remapped_data = {}

    for var_name in variables:
        print(f"Remapping {var_name}...")
        var_data = processor.get_2d_variable_data(var_name, time_index)
        remapped = remapper.remap(var_data)
        remapped_data[var_name] = remapped.compute()
        del var_data  
    
    print(f"Remapped {len(variables)} variables")
    
    return remapped_data, remapper


def example_time_series_remapping() -> xr.DataArray:
    """
    Demonstrate time series remapping workflow processing multiple temporal snapshots using cached regridder weights for computational efficiency. This example shows how to process temporal evolution of MPAS variables by building the interpolation weights once and applying them repeatedly across time dimensions, avoiding expensive weight recomputation for each timestep. The function loads temperature data across available time steps, initializes a patch remapper with higher-order interpolation, and iterates through timesteps while accumulating remapped results. The remapped timesteps are concatenated into a unified time series DataArray suitable for temporal analysis and animation workflows. This approach is critical for climate model output analysis where hundreds or thousands of timesteps require processing with consistent spatial transformation.

    Parameters:
        None

    Returns:
        xr.DataArray: Concatenated time series of remapped temperature data with dimensions [time, lat, lon] containing up to 10 timesteps.
    """
    print("\n" + "="*60)
    print("Example 4: Time Series Remapping")
    print("="*60)
    
    grid_file = "path/to/grid.nc"
    data_file = "path/to/diag.nc"
    
    processor = MPAS2DProcessor(grid_file)
    processor.load_2d_data(data_file)
    
    var_name = 't2m'
    
    time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
    n_times = processor.dataset.sizes[time_dim]
    
    remapper = MPASRemapper(method='patch', weights_dir='./weights')
    
    var_data = processor.get_2d_variable_data(var_name, 0)
    lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)
    
    remapper.prepare_source_grid(lon, lat)
    remapper.create_target_grid(dlon=1.0, dlat=1.0)
    remapper.build_regridder()
    
    remapped_times = []

    for t in range(min(n_times, 10)):  
        var_data = processor.get_2d_variable_data(var_name, t)
        remapped = remapper.remap(var_data)
        remapped_times.append(remapped)
        
        if (t + 1) % 5 == 0:
            print(f"Processed {t + 1} time steps")
    
    remapped_ts = xr.concat(remapped_times, dim='time')
    print(f"Created time series with shape: {remapped_ts.shape}")
    
    return remapped_ts


def example_different_methods_comparison() -> Dict[str, xr.DataArray]:
    """
    Demonstrate systematic comparison of four interpolation methods showing trade-offs between accuracy, smoothness, and conservation properties. This example applies bilinear (smooth gradients), conservative (flux preservation), patch (high-order accuracy), and nearest-neighbor (value preservation) methods to identical source data over a regional domain. The function initializes separate remappers for each method using consistent target grid specifications, performs the remapping, and computes statistical diagnostics including range, mean, and standard deviation for quality assessment. Each method has distinct characteristics: bilinear provides smooth fields suitable for visualization, conservative preserves spatial integrals for flux variables, patch offers higher-order accuracy at computational cost, and nearest-neighbor maintains exact source values without interpolation. Understanding these differences is essential for selecting appropriate methods based on variable physics and analysis requirements.

    Parameters:
        None

    Returns:
        Dict[str, xr.DataArray]: Dictionary mapping interpolation method names to their corresponding remapped DataArrays for statistical comparison and visualization.
    """
    print("\n" + "="*60)
    print("Example 5: Method Comparison")
    print("="*60)
    
    grid_file = "data/grids/x1.40962.init.nc"
    data_file = "data/u120k/diag/diag.2024-09-17_01.00.00.nc"
    
    processor = MPAS2DProcessor(grid_file)
    processor.load_2d_data(data_file)
    
    var_name = 't2m'
    time_index = 0
    
    var_data = processor.get_2d_variable_data(var_name, time_index)
    lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)
    
    methods = ['bilinear', 'conservative', 'patch', 'nearest_s2d']
    results = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        
        remapper = MPASRemapper(method=method, weights_dir='./weights')
        remapper.prepare_source_grid(lon, lat)
        remapper.create_target_grid(lon_min=-120, lon_max=-80, 
                                    lat_min=30, lat_max=50, dlon=1.0, dlat=1.0)
        remapper.build_regridder()
        
        remapped = remapper.remap(var_data)
        results[method] = remapped
        
        print(f"  Data range: [{remapped.min().values:.2f}, {remapped.max().values:.2f}]")
        print(f"  Mean: {remapped.mean().values:.2f}")
        print(f"  Std: {remapped.std().values:.2f}")
    
    return results


def example_memory_estimation() -> None:
    """
    Demonstrate memory requirement estimation for various MPAS mesh resolutions and target grid configurations to enable planning of large-scale remapping operations. This example computes approximate memory usage for five standard MPAS global mesh resolutions (15km through 240km) combined with four target grid spacings (0.25° through 2.0°) using conservative remapping which requires the most memory. The function utilizes MPASRemapper's static memory estimation method that accounts for sparse weight matrix storage and temporary array allocations during regridding operations. Memory estimates help users determine appropriate computational resources, identify potential out-of-memory scenarios, and select optimal target resolutions for available hardware. The printed table provides quick reference for operational planning of climate model postprocessing workflows.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "="*60)
    print("Example 6: Memory Estimation")
    print("="*60)
    
    meshes = {
        '15km': 2621442,
        '30km': 655362,
        '60km': 163842,
        '120km': 40962,
        '240km': 10242
    }
    
    target_res = [0.25, 0.5, 1.0, 2.0]
    
    print("\nEstimated memory usage (GB) for different configurations:")
    print(f"{'Mesh':<10} | " + " ".join([f"{res}°" for res in target_res]))
    print("-" * 60)
    
    for mesh_name, n_source in meshes.items():
        estimates = []
        for res in target_res:
            n_lon = int(360 / res)
            n_lat = int(180 / res)
            n_target = n_lon * n_lat
            
            mem = MPASRemapper.estimate_memory_usage(n_source, n_target, 'conservative')
            estimates.append(f"{mem:.2f}")
        
        print(f"{mesh_name:<10} | " + " | ".join([f"{e:>6}" for e in estimates]))
    
    print("\nNote: These are rough estimates. Actual memory may vary.")


def plot_comparison(
    original_data: xr.DataArray,
    remapped_data: xr.DataArray,
    title: str = "MPAS Remapping"
) -> matplotlib.figure.Figure:
    """
    Create side-by-side comparison visualization of original MPAS unstructured data and remapped regular grid data for quality assessment. This helper function generates a two-panel matplotlib figure displaying the source MPAS data as scatter points colored by variable value on the left, and the remapped regular grid data as filled contours on the right panel. The visualization enables visual verification of remapping quality, identification of interpolation artifacts, and assessment of spatial pattern preservation during grid transformation. Both panels share colormap and color scaling for direct comparison, include geographic coordinate labels, and display individual colorbars. This plotting utility is essential for validating remapping results before using transformed data in downstream analysis workflows.

    Parameters:
        original_data (xr.DataArray): Source MPAS data on unstructured mesh with lon/lat coordinates and data values.
        remapped_data (xr.DataArray): Remapped data on regular latitude-longitude grid with coordinate arrays.
        title (str): Overall figure title displayed above both panels (default: "MPAS Remapping").

    Returns:
        plt.Figure: Matplotlib figure object containing the two-panel comparison plot with colorbars and labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    scatter = ax1.scatter(original_data.lon, original_data.lat, 
                         c=original_data.values, s=1, cmap='RdBu_r')
    ax1.set_title('Original MPAS Data')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax1)
    
    ax2 = axes[1]
    im = ax2.contourf(remapped_data.lon, remapped_data.lat, 
                     remapped_data.values, levels=20, cmap='RdBu_r')
    ax2.set_title('Remapped to Regular Grid')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax2)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


if __name__ == '__main__':
    print("MPAS Remapping Examples (Memory Optimized)")
    print("=" * 60)
    print("\nUsing data from: data/grids/x1.40962.init.nc and data/u120k/diag/")
    print("\nAvailable examples:")
    print("  1. Basic remapping with convenience function (lazy loading)")
    print("  2. Conservative remapping for flux variables (chunked)")
    print("  3. Batch remapping multiple variables (memory efficient)")
    print("  4. Time series remapping")
    print("  5. Compare different interpolation methods")
    print("  6. Memory estimation for large datasets")
    print("\nMemory Optimization Tips:")
    print("  - Using chunking to load data in smaller pieces")
    print("  - Processing one time step at a time")
    print("  - Using coarser target grid resolutions")
    print("  - Reusing weights to avoid redundant computation")
    print("  - Explicit memory cleanup after processing")
    
    try:
        example_memory_estimation()
    except Exception as e:
        print(f"Error in memory estimation: {e}")
    
    print("\n" + "="*60)
    print("Running memory-optimized basic remapping example...")
    print("="*60)
    
    try:
        remapped = example_basic_remapping()
        print("\n✓ Basic remapping completed successfully!")
        print("\nTo run other examples, uncomment them below.")
        # remapped, remapper = example_conservative_remapping()
        # remapped_data, remapper = example_batch_remapping()
        # remapped_ts = example_time_series_remapping()
        # results = example_different_methods_comparison()
    except Exception as e:
        print(f"\n✗ Error during remapping: {e}")
        import traceback
        traceback.print_exc()
