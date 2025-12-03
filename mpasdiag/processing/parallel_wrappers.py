#!/usr/bin/env python3

"""
Parallel Processing Wrappers for MPAS Visualization

This module provides parallel-enabled wrapper classes and functions for MPAS batch visualization workflows allowing seamless integration of MPI-based parallelization without modifying core visualization classes. It implements specialized parallel wrappers (ParallelPrecipitationProcessor, ParallelSurfaceProcessor, ParallelCrossSectionProcessor) that coordinate distributed processing of time series plots across multiple MPI ranks, handle data caching to minimize redundant I/O operations, manage matplotlib backend configuration for fork-safe execution on macOS systems, and provide automatic load balancing with dynamic work distribution. The parallel wrappers maintain the same interface as their serial counterparts, enabling drop-in replacement in existing workflows while dramatically improving performance for multi-timestep batch processing through spatial or temporal domain decomposition. Core capabilities include MPI rank coordination for distributing time steps across workers, shared data caching for grid and coordinate information, comprehensive error handling with graceful degradation to serial mode, progress monitoring and performance statistics, and compatibility with both interactive and batch processing environments suitable for operational weather analysis and climate model diagnostics.

Classes:
    ParallelPrecipitationProcessor: Parallel wrapper for batch precipitation map generation with MPI coordination.
    ParallelSurfaceProcessor: Parallel wrapper for batch surface variable visualization with distributed processing.
    ParallelCrossSectionProcessor: Parallel wrapper for batch vertical cross-section plots with work distribution.
    
Functions:
    auto_batch_processor: Automatically selects and configures appropriate parallel or serial processor based on system capabilities.
    setup_parallel_environment: Configures matplotlib backend and MPI environment for parallel execution.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import matplotlib
matplotlib.use('Agg')  

import os
import time
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
import pandas as pd

try:
    from .data_cache import MPASDataCache, get_global_cache
except ImportError:
    from mpasdiag.processing.data_cache import MPASDataCache, get_global_cache

try:
    from .parallel import MPASParallelManager
    from .processors_2d import MPAS2DProcessor
    from .processors_3d import MPAS3DProcessor
except ImportError:
    from mpasdiag.processing.parallel import MPASParallelManager
    from mpasdiag.processing.processors_2d import MPAS2DProcessor
    from mpasdiag.processing.processors_3d import MPAS3DProcessor

try:
    from ..visualization.precipitation import MPASPrecipitationPlotter
    from ..visualization.surface import MPASSurfacePlotter
    from ..visualization.wind import MPASWindPlotter
    from ..visualization.cross_section import MPASVerticalCrossSectionPlotter
    from ..diagnostics.precipitation import PrecipitationDiagnostics
except ImportError:
    from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
    from mpasdiag.visualization.surface import MPASSurfacePlotter
    from mpasdiag.visualization.wind import MPASWindPlotter
    from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
    from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics


def _precipitation_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute precipitation processing and plotting for a single timestep with comprehensive timing metrics. This worker function serves as a picklable entry point for parallel processing of precipitation data across multiple timesteps using a shared data cache. It computes precipitation differences using diagnostic modules, generates plots with the precipitation plotter, and saves outputs in specified formats. The function measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds, variable settings, and plotting options.

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args
    processor = kwargs['processor']
    cache = kwargs.get('cache', None)
    output_dir = kwargs['output_dir']
    lon_min = kwargs['lon_min']
    lon_max = kwargs['lon_max']
    lat_min = kwargs['lat_min']
    lat_max = kwargs['lat_max']
    var_name = kwargs['var_name']
    accum_period = kwargs['accum_period']
    file_prefix = kwargs['file_prefix']
    formats = kwargs['formats']
    custom_title_template = kwargs.get('custom_title_template')
    colormap = kwargs.get('colormap')
    levels = kwargs.get('levels')
    
    start_time = time.time()
    timings = {}
    cache_hits = {'coordinates': False, 'data': False}
    
    data_start = time.time()
    
    if cache is not None:
        try:
            lon, lat = cache.get_coordinates(var_name)
            cache_hits['coordinates'] = True
        except KeyError:
            lon, lat = processor.extract_2d_coordinates_for_variable(var_name)
            try:
                cache.load_coordinates_from_dataset(processor.dataset, var_name)
            except:
                pass
    else:
        lon, lat = processor.extract_2d_coordinates_for_variable(var_name)
    
    precip_diag = PrecipitationDiagnostics(verbose=False)
    precip_data = precip_diag.compute_precipitation_difference(
        processor.dataset, 
        time_idx, 
        var_name, 
        accum_period,
        data_type=processor.data_type or 'UXarray'
    )
    
    time_end = None
    if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
        time_end = pd.Timestamp(processor.dataset.Time.values[time_idx]).to_pydatetime()
        time_str = time_end.strftime('%Y%m%dT%H')
    else:
        time_str = f"t{time_idx:03d}"
    
    timings['data_processing'] = time.time() - data_start
    
    plotter = MPASPrecipitationPlotter(figsize=(10, 14))
    
    if custom_title_template:
        title = custom_title_template.format(
            var_name=var_name.upper(),
            time_str=time_str,
            accum_period=accum_period
        )
    else:
        title = f"MPAS Precipitation | VarType: {var_name.upper()} | Valid Time: {time_str}"
    
    plot_start = time.time()
    fig, ax = plotter.create_precipitation_map(
        lon, lat, precip_data.values,
        lon_min, lon_max, lat_min, lat_max,
        title=title,
        accum_period=accum_period,
        colormap=colormap,
        levels=levels,
        data_array=precip_data,
        time_end=time_end,
        var_name=var_name
    )
    timings['plotting'] = time.time() - plot_start
    
    save_start = time.time()
    output_path = os.path.join(
        output_dir,
        f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_point"
    )
    
    plotter.save_plot(output_path, formats=formats)
    plotter.close_plot()
    
    output_files = [f"{output_path}.{fmt}" for fmt in formats]
    timings['saving'] = time.time() - save_start
    
    timings['total'] = time.time() - start_time
    
    result = {
        'files': output_files,
        'timings': timings,
        'time_str': time_str,
        'cache_hits': cache_hits
    }
    
    if cache is not None:
        result['cache_info'] = cache.get_cache_info()
    
    return result
    

def _surface_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute surface variable processing and plotting for a single timestep with performance tracking. This worker function processes 2D surface fields such as temperature, pressure, or moisture variables across MPAS unstructured grids using cached coordinates. It retrieves the surface variable data, generates contour or filled contour plots using the surface plotter, and saves outputs in requested formats. The function measures timing for data extraction, plotting operations, and file writing while tracking cache hit statistics for analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple with (time_idx, kwargs) where time_idx is the timestep index and kwargs contains processor instance, cache object, spatial bounds, variable name, plot type, and formatting options.

    Returns:
        Dict[str, Any]: Result dictionary with 'files' (list of str paths), 'timings' (dict with phase durations), 'cache_hits' (dict), and 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args
    processor = kwargs['processor']
    cache = kwargs.get('cache', None)
    output_dir = kwargs['output_dir']
    lon_min = kwargs['lon_min']
    lon_max = kwargs['lon_max']
    lat_min = kwargs['lat_min']
    lat_max = kwargs['lat_max']
    var_name = kwargs['var_name']
    plot_type = kwargs['plot_type']
    file_prefix = kwargs['file_prefix']
    formats = kwargs['formats']
    custom_title = kwargs.get('custom_title')
    colormap = kwargs.get('colormap')
    levels = kwargs.get('levels')
    
    start_time = time.time()
    timings = {}
    cache_hits = {'coordinates': False, 'data': False}
    
    data_start = time.time()
    
    if cache is not None:
        try:
            lon, lat = cache.get_coordinates(var_name)
            cache_hits['coordinates'] = True
        except KeyError:
            lon, lat = processor.extract_spatial_coordinates()
            try:
                cache.load_coordinates_from_dataset(processor.dataset, var_name)
            except:
                pass 
    else:
        lon, lat = processor.extract_spatial_coordinates()
    
    var_data = processor.dataset[var_name].isel(Time=time_idx)
    
    plotter = MPASSurfacePlotter(figsize=(10, 14))
    timings['data_processing'] = time.time() - data_start
    
    plot_start = time.time()
    
    fig, ax = plotter.create_surface_map(
        lon=lon,
        lat=lat,
        data=var_data.values,
        var_name=var_name,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        title=custom_title,
        plot_type=plot_type,
        colormap=colormap,
        levels=levels,
        time_stamp=pd.Timestamp(str(processor.dataset['Time'].values[time_idx]))
    )
    timings['plotting'] = time.time() - plot_start
    
    save_start = time.time()
    
    time_str = str(processor.dataset['Time'].values[time_idx])
    safe_time_str = time_str.replace(':', '').replace('-', '').replace(' ', 'T')[:13]
    output_path = os.path.join(
        output_dir,
        f"{file_prefix}_{var_name}_{plot_type}_valid_{safe_time_str}"
    )
    
    plotter.save_plot(output_path, formats=formats)
    plotter.close_plot()
    
    output_files = [f"{output_path}.{fmt}" for fmt in formats]
    
    timings['saving'] = time.time() - save_start
    timings['total'] = time.time() - start_time
    
    result = {
        'files': output_files,
        'timings': timings,
        'time_str': time_str,
        'cache_hits': cache_hits
    }
    
    if cache is not None:
        result['cache_info'] = cache.get_cache_info()
    
    return result

def _cross_section_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute vertical cross-section extraction and visualization for a single timestep with timing analysis. This worker function computes vertical slices through 3D atmospheric fields along specified great circle paths between start and end coordinates. It processes variables on height, pressure, or model levels using the 3D processor and generates cross-section plots with the vertical plotter. The function tracks execution time for data extraction, interpolation, plotting, and I/O operations to provide performance insights.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the timestep index and kwargs includes processor instance, cross-section endpoints, variable name, vertical coordinate type, and visualization settings.

    Returns:
        Dict[str, Any]: Result dictionary with 'files' (list of output paths), 'timings' (dict mapping phases to durations), and 'time_str' (str timestamp).
    """
    time_idx, kwargs = args
    processor_3d = kwargs['processor']  
    output_dir = kwargs['output_dir']
    start_lat = kwargs['start_lat']
    start_lon = kwargs['start_lon']
    end_lat = kwargs['end_lat']
    end_lon = kwargs['end_lon']
    var_name = kwargs['var_name']
    file_prefix = kwargs['file_prefix']
    formats = kwargs['formats']
    custom_title = kwargs.get('custom_title')
    colormap = kwargs.get('colormap')
    levels = kwargs.get('levels')
    vertical_coord = kwargs.get('vertical_coord', 'pressure')
    num_points = kwargs.get('num_points', 100)
    
    start_time = time.time()
    timings = {}
    
    data_start = time.time()
    
    plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 14))
    timings['data_processing'] = time.time() - data_start
    
    plot_start = time.time()
    
    time_str = str(processor_3d.dataset['Time'].values[time_idx])
    safe_time_str = time_str.replace(':', '').replace('-', '').replace(' ', 'T')[:13]
    save_path = os.path.join(output_dir, f"{file_prefix}_{safe_time_str}.png")
    
    fig, ax = plotter.create_vertical_cross_section(
        mpas_3d_processor=processor_3d,
        var_name=var_name,
        start_point=(start_lon, start_lat),
        end_point=(end_lon, end_lat),
        time_index=time_idx,
        vertical_coord=vertical_coord,
        num_points=num_points,
        colormap=colormap,
        levels=levels,
        save_path=save_path
    )
    timings['plotting'] = time.time() - plot_start
    
    save_start = time.time()
    output_files = [save_path]
    
    for fmt in formats:
        if fmt != 'png':
            output_file = save_path.replace('.png', f'.{fmt}')
            save_kwargs = {'dpi': 100, 'bbox_inches': 'tight'}
            if fmt.lower() == 'png':
                save_kwargs['pil_kwargs'] = {'compress_level': 1}
            fig.savefig(output_file, **save_kwargs)
            output_files.append(output_file)
    
    timings['saving'] = time.time() - save_start
    timings['total'] = time.time() - start_time
    
    return {
        'files': output_files,
        'timings': timings,
        'time_str': time_str
    }


class ParallelPrecipitationProcessor:
    """
    Parallel wrapper for precipitation batch processing using MPI-based distributed computing. This class wraps MPASPrecipitationPlotter batch methods to enable efficient parallel processing of multiple timesteps across available MPI processes. It provides static methods that distribute work, collect results, and aggregate outputs from parallel execution. The wrapper handles task distribution, load balancing, and result collection transparently while preserving the same API as serial batch processing methods.
    """
    
    @staticmethod
    def create_batch_precipitation_maps_parallel(
        processor: MPAS2DProcessor,
        output_dir: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        var_name: str = 'rainnc',
        accum_period: str = 'a01h',
        file_prefix: str = 'mpas_precipitation_map',
        formats: List[str] = ['png'],
        custom_title_template: Optional[str] = None,
        colormap: Optional[str] = None,
        levels: Optional[List[float]] = None,
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "dynamic"
    ) -> Optional[List[str]]:
        """
        Generate precipitation maps in parallel across multiple timesteps using MPI-based distributed processing. This static method distributes timesteps across available MPI processes using the specified load balancing strategy and creates individual precipitation maps for each timestep. It initializes a shared data cache to avoid redundant coordinate extraction, creates precipitation accumulation visualizations, and collects results on the master process. The method supports customizable output formats, color scales, and accumulation periods while automatically handling task distribution and result aggregation.

        Parameters:
            processor (MPAS2DProcessor): Initialized MPAS2DProcessor instance with loaded precipitation data and grid information.
            output_dir (str): Directory path where precipitation map files will be saved.
            lon_min (float): Minimum longitude for map spatial extent in degrees.
            lon_max (float): Maximum longitude for map spatial extent in degrees.
            lat_min (float): Minimum latitude for map spatial extent in degrees.
            lat_max (float): Maximum latitude for map spatial extent in degrees.
            var_name (str): Name of precipitation variable in dataset (default: 'rainnc').
            accum_period (str): Accumulation period identifier like 'a01h' or 'a24h' (default: 'a01h').
            file_prefix (str): Prefix string for output filenames (default: 'mpas_precipitation_map').
            formats (List[str]): List of output image formats such as ['png', 'pdf'] (default: ['png']).
            custom_title_template (Optional[str]): Custom template string for plot titles (default: None).
            colormap (Optional[str]): Matplotlib colormap name for precipitation visualization (default: None).
            levels (Optional[List[float]]): Specific contour levels for precipitation coloring (default: None).
            time_indices (Optional[List[int]]): Specific timestep indices to process, None processes all valid indices (default: None).
            n_processes (Optional[int]): Number of MPI processes to use, None uses all available (default: None).
            load_balance_strategy (str): Strategy for task distribution - 'static', 'dynamic', 'block', or 'cyclic' (default: 'dynamic').

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        accum_hours = int(accum_period[1:3])
        hours_per_file = 1
        min_time_idx = accum_hours // hours_per_file
        
        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
        total_times = processor.dataset.sizes[time_dim]
        
        if time_indices is None:
            time_indices = list(range(min_time_idx, total_times))
        else:
            time_indices = [idx for idx in time_indices if idx >= min_time_idx]
        
        if not time_indices:
            print(f"Warning: No valid time indices for accumulation period {accum_period}")
            return []
        
        cache = MPASDataCache(max_variables=5)
        
        print("Pre-loading coordinates into cache...")

        try:
            cache.load_coordinates_from_dataset(processor.dataset, var_name)
            print(f"Coordinates cached for variable: {var_name}")
        except Exception as e:
            print(f"Warning: Could not pre-load coordinates into cache: {e}")
            print("Workers will extract coordinates individually")
        
        worker_kwargs = {
            'processor': processor,
            'cache': cache,
            'output_dir': output_dir,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'var_name': var_name,
            'accum_period': accum_period,
            'file_prefix': file_prefix,
            'formats': formats,
            'custom_title_template': custom_title_template,
            'colormap': colormap,
            'levels': levels
        }
        
        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes
        )
        manager.set_error_policy('collect')
        
        os.makedirs(output_dir, exist_ok=True)
        
        if manager.is_master:
            print(f"\nCreating precipitation maps for {len(time_indices)} time steps in parallel...")
            print(f"Using accumulation period: {accum_period} ({accum_hours} hours)")
            print(f"Output directory: {output_dir}")
        
        tasks = [(idx, worker_kwargs) for idx in time_indices]
        
        results = manager.parallel_map(
            _precipitation_worker,
            tasks
        )
        
        if manager.is_master and results is not None:
            created_files = []
            successful = 0
            failed = 0
            
            all_timings = {
                'data_processing': [],
                'plotting': [],
                'saving': [],
                'total': []
            }
            
            for result in results:
                if result.success:
                    created_files.extend(result.result['files'])
                    successful += 1
                    
                    timings = result.result['timings']
                    for key in all_timings:
                        all_timings[key].append(timings[key])
                else:
                    failed += 1
                    print(f"Failed time index {time_indices[result.task_id]}: {result.error}")
            
            timing_stats = {}

            for key, values in all_timings.items():
                if values:
                    timing_stats[key] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'total': sum(values)
                    }
            
            print(f"\n{'='*70}")
            print("PRECIPITATION BATCH PROCESSING RESULTS")
            print(f"{'='*70}")
            print("Status:")
            print(f"  Successful: {successful}/{len(time_indices)}")
            print(f"  Failed: {failed}/{len(time_indices)}")
            print(f"  Created files: {len(created_files)} in {output_dir}")
            
            if timing_stats:
                print("\nTiming Breakdown (per time step):")
                print("  Data Processing:")
                print(f"    Min:  {timing_stats['data_processing']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['data_processing']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['data_processing']['mean']:6.3f}s")
                print("  Plotting:")
                print(f"    Min:  {timing_stats['plotting']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['plotting']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['plotting']['mean']:6.3f}s")
                print("  Saving:")
                print(f"    Min:  {timing_stats['saving']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['saving']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['saving']['mean']:6.3f}s")
                print("  Total per step:")
                print(f"    Min:  {timing_stats['total']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['total']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['total']['mean']:6.3f}s")
                
                stats = manager.get_statistics()

                if stats:
                    print("\nOverall Parallel Execution:")
                    print(f"  Wall time: {stats.total_time:.2f}s")
                    print(f"  Speedup potential: {timing_stats['total']['total']:.2f}s / {stats.total_time:.2f}s = {timing_stats['total']['total']/stats.total_time:.2f}x")
                    print(f"  Load imbalance: {100*stats.load_imbalance:.1f}%")
            
            print(f"{'='*70}\n")
            
            return created_files
        
        return None


class ParallelSurfaceProcessor:
    """
    Parallel wrapper for surface variable batch processing using MPI-based distributed computing. This class wraps MPASSurfacePlotter batch methods to enable efficient parallel processing of 2D surface fields across multiple timesteps and MPI processes. It provides static methods that distribute work, collect results, and aggregate outputs from parallel execution. The wrapper handles task distribution, load balancing, and result collection transparently while preserving the same API as serial batch processing methods.
    """
    
    @staticmethod
    def create_batch_surface_maps_parallel(
        processor: MPAS2DProcessor,
        output_dir: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        var_name: str = 't2m',
        plot_type: str = 'scatter',
        file_prefix: str = 'mpas_surface',
        formats: List[str] = ['png'],
        grid_resolution: Optional[int] = None,
        grid_resolution_deg: Optional[float] = None,
        clim_min: Optional[float] = None,
        clim_max: Optional[float] = None,
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "dynamic"
    ) -> Optional[List[str]]:
        """
        Generate surface variable maps in parallel across multiple timesteps using distributed MPI processing. This static method distributes timesteps across MPI processes using the specified load balancing strategy and creates scatter or contour plots for each timestep. It initializes a shared data cache to avoid redundant coordinate extraction, creates 2D surface field visualizations for variables like temperature and pressure, and collects results on the master process. The method supports customizable gridding resolutions, color limits, and multiple output formats while handling task distribution automatically.

        Parameters:
            processor (MPAS2DProcessor): Initialized MPAS2DProcessor instance with loaded surface data and grid information.
            output_dir (str): Directory path where surface map files will be saved.
            lon_min (float): Minimum longitude for map spatial extent in degrees.
            lon_max (float): Maximum longitude for map spatial extent in degrees.
            lat_min (float): Minimum latitude for map spatial extent in degrees.
            lat_max (float): Maximum latitude for map spatial extent in degrees.
            var_name (str): Name of surface variable in dataset like 't2m' or 'mslp' (default: 't2m').
            plot_type (str): Visualization type - 'scatter' for point plots or 'contour' for filled contours (default: 'scatter').
            file_prefix (str): Prefix string for output filenames (default: 'mpas_surface').
            formats (List[str]): List of output image formats such as ['png', 'pdf'] (default: ['png']).
            grid_resolution (Optional[int]): Grid resolution in number of points for interpolation (default: None).
            grid_resolution_deg (Optional[float]): Grid resolution in degrees for interpolation grid (default: None).
            clim_min (Optional[float]): Minimum value for color scale limits (default: None for automatic).
            clim_max (Optional[float]): Maximum value for color scale limits (default: None for automatic).
            time_indices (Optional[List[int]]): Specific timestep indices to process, None processes all (default: None).
            n_processes (Optional[int]): Number of MPI processes to use, None uses all available (default: None).
            load_balance_strategy (str): Strategy for task distribution - 'static', 'dynamic', 'block', or 'cyclic' (default: 'dynamic').

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
        total_times = processor.dataset.sizes[time_dim]
        
        if time_indices is None:
            time_indices = list(range(total_times))
        
        cache = MPASDataCache(max_variables=5)
        
        print("Pre-loading coordinates into cache...")

        try:
            cache.load_coordinates_from_dataset(processor.dataset, var_name)
            print(f"Coordinates cached for variable: {var_name}")
        except Exception as e:
            print(f"Warning: Could not pre-load coordinates into cache: {e}")
            print("Workers will extract coordinates individually")
        
        worker_kwargs = {
            'processor': processor,
            'cache': cache,
            'output_dir': output_dir,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'var_name': var_name,
            'plot_type': plot_type,
            'file_prefix': file_prefix,
            'formats': formats,
            'custom_title': None,
            'colormap': None,
            'levels': None
        }
        
        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]
        
        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes
        )
        manager.set_error_policy('collect')
        
        os.makedirs(output_dir, exist_ok=True)
        
        if manager.is_master:
            print(f"\nCreating surface maps for {len(time_indices)} time steps in parallel...")
            print(f"Variable: {var_name}, Plot type: {plot_type}")
            print(f"Output directory: {output_dir}")
        
        results = manager.parallel_map(_surface_worker, worker_args)
        
        if manager.is_master and results is not None:
            created_files = []
            successful = 0
            failed = 0
            
            all_timings = {
                'data_processing': [],
                'plotting': [],
                'saving': [],
                'total': []
            }
            
            for result in results:
                if result.success:
                    created_files.extend(result.result['files'])
                    successful += 1                    
                    timings = result.result['timings']
                    for key in all_timings:
                        all_timings[key].append(timings[key])
                else:
                    failed += 1
                    print(f"Failed time index {time_indices[result.task_id]}: {result.error}")
            
            timing_stats = {}

            for key, values in all_timings.items():
                if values:
                    timing_stats[key] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'total': sum(values)
                    }
            
            print(f"\n{'='*70}")
            print("SURFACE BATCH PROCESSING RESULTS")
            print(f"{'='*70}")
            print(f"Variable: {var_name}, Plot type: {plot_type}")
            print("\nStatus:")
            print(f"  Successful: {successful}/{len(time_indices)}")
            print(f"  Failed: {failed}/{len(time_indices)}")
            print(f"  Created files: {len(created_files)} in {output_dir}")
            
            if timing_stats:
                print("\nTiming Breakdown (per time step):")
                print("  Data Processing:")
                print(f"    Min:  {timing_stats['data_processing']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['data_processing']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['data_processing']['mean']:6.3f}s")
                print("  Plotting:")
                print(f"    Min:  {timing_stats['plotting']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['plotting']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['plotting']['mean']:6.3f}s")
                print("  Saving:")
                print(f"    Min:  {timing_stats['saving']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['saving']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['saving']['mean']:6.3f}s")
                print("  Total per step:")
                print(f"    Min:  {timing_stats['total']['min']:6.3f}s")
                print(f"    Max:  {timing_stats['total']['max']:6.3f}s")
                print(f"    Mean: {timing_stats['total']['mean']:6.3f}s")
                
                stats = manager.get_statistics()

                if stats:
                    print("\nOverall Parallel Execution:")
                    print(f"  Wall time: {stats.total_time:.2f}s")
                    print(f"  Speedup potential: {timing_stats['total']['total']:.2f}s / {stats.total_time:.2f}s = {timing_stats['total']['total']/stats.total_time:.2f}x")
                    print(f"  Load imbalance: {100*stats.load_imbalance:.1f}%")
            
            print(f"{'='*70}\n")
            
            return created_files
        
        return None


class ParallelCrossSectionProcessor:
    """
    Parallel wrapper for vertical cross-section batch processing using MPI-based distributed computing. This class wraps MPASVerticalCrossSectionPlotter batch methods to enable efficient parallel processing of 3D atmospheric slices across multiple timesteps and MPI processes. It provides static methods that distribute work, collect results, and aggregate outputs from parallel execution. The wrapper handles task distribution, load balancing, and result collection transparently while preserving the same API as serial batch processing methods.
    """
    
    @staticmethod
    def create_batch_cross_section_plots_parallel(
        mpas_3d_processor: MPAS3DProcessor,
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        output_dir: str,
        vertical_coord: str = 'height_agl',
        num_points: int = 100,
        levels: Optional[np.ndarray] = None,
        colormap: str = 'viridis',
        extend: str = 'both',
        plot_type: str = 'contourf',
        max_height: Optional[float] = None,
        file_prefix: str = 'mpas_cross_section',
        formats: List[str] = ['png'],
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "dynamic"
    ) -> Optional[List[str]]:
        """
        Generate vertical cross-section plots in parallel across multiple timesteps using distributed MPI processing. This static method distributes timesteps across MPI processes using the specified load balancing strategy and creates vertical cross-sections through the atmosphere for each timestep. It processes 3D atmospheric variables along specified great circle paths, generates visualizations with customizable vertical coordinates and color schemes, and collects results on the master process. The method supports multiple vertical coordinate systems, customizable contour levels and colormaps, and various output formats while handling task distribution automatically.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Initialized MPAS3DProcessor instance with loaded 3D atmospheric data and vertical structure.
            var_name (str): Name of the 3D atmospheric variable to plot in cross-section like 'theta' or 'w'.
            start_point (Tuple[float, float]): Starting coordinates as (longitude, latitude) for cross-section path in degrees.
            end_point (Tuple[float, float]): Ending coordinates as (longitude, latitude) for cross-section path in degrees.
            output_dir (str): Directory path where cross-section plot files will be saved.
            vertical_coord (str): Vertical coordinate system - 'height_agl', 'pressure', or 'model_levels' (default: 'height_agl').
            num_points (int): Number of interpolation points along the horizontal cross-section path (default: 100).
            levels (Optional[np.ndarray]): Specific contour levels for variable visualization (default: None for automatic).
            colormap (str): Matplotlib colormap name for cross-section coloring (default: 'viridis').
            extend (str): Colorbar extension mode - 'neither', 'both', 'min', or 'max' (default: 'both').
            plot_type (str): Visualization type - 'contourf' for filled contours or 'contour' for line contours (default: 'contourf').
            max_height (Optional[float]): Maximum height in kilometers for vertical axis limit (default: None for automatic).
            file_prefix (str): Prefix string for output filenames (default: 'mpas_cross_section').
            formats (List[str]): List of output image formats such as ['png', 'pdf'] (default: ['png']).
            time_indices (Optional[List[int]]): Specific timestep indices to process, None processes all (default: None).
            n_processes (Optional[int]): Number of MPI processes to use, None uses all available (default: None).
            load_balance_strategy (str): Strategy for task distribution - 'static', 'dynamic', 'block', or 'cyclic' (default: 'dynamic').

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        time_dim = 'Time' if 'Time' in mpas_3d_processor.dataset.sizes else 'time'
        total_times = mpas_3d_processor.dataset.sizes[time_dim]
        
        if time_indices is None:
            time_indices = list(range(total_times))
        
        worker_kwargs = {
            'processor': mpas_3d_processor,
            'output_dir': output_dir,
            'start_lat': start_point[1],
            'start_lon': start_point[0],
            'end_lat': end_point[1],
            'end_lon': end_point[0],
            'var_name': var_name,
            'file_prefix': file_prefix,
            'formats': formats,
            'custom_title': None,
            'colormap': colormap,
            'levels': levels,
            'vertical_coord': vertical_coord,
            'num_points': num_points
        }
        
        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]
        
        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes
        )
        manager.set_error_policy('collect')
        
        os.makedirs(output_dir, exist_ok=True)
        
        if manager.is_master:
            print(f"\nCreating vertical cross-section plots for {len(time_indices)} time steps in parallel...")
            print(f"Variable: {var_name}")
            print(f"Cross-section from ({start_point[0]:.2f}, {start_point[1]:.2f}) to ({end_point[0]:.2f}, {end_point[1]:.2f})")
            print(f"Vertical coordinate: {vertical_coord}")
            if max_height:
                print(f"Maximum height: {max_height} km")
            print(f"Output directory: {output_dir}")
        
        results = manager.parallel_map(_cross_section_worker, worker_args)
        
        if manager.is_master and results is not None:
            created_files = []
            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)
            
            for result in results:
                if result.success:
                    created_files.extend(result.result)
                else:
                    print(f"Failed time index {time_indices[result.task_id]}: {result.error}")
            
            print("\nBatch processing completed:")
            print(f"  Successful: {successful}/{len(time_indices)}")
            print(f"  Failed: {failed}/{len(time_indices)}")
            print(f"  Created {len(created_files)} files in: {output_dir}")
            
            return created_files
        
        return None


def auto_batch_processor(use_parallel: Optional[bool] = None, **kwargs) -> bool:
    """
    Automatically select between parallel and serial batch processing modes based on runtime environment. This convenience function detects MPI availability and the number of available processes to determine the optimal execution strategy. When use_parallel is None, it attempts to import mpi4py and checks if multiple processes are available via COMM_WORLD. If MPI is not available or only a single process is running, it defaults to serial mode. Users can override auto-detection by explicitly passing True or False. Returns the selected mode as a boolean for downstream processing decisions.

    Parameters:
        use_parallel (Optional[bool]): Explicit mode selection - True for parallel, False for serial, None for auto-detection based on MPI environment.
        **kwargs (dict): Additional arguments to pass to batch processors (currently unused but reserved for future extensibility).

    Returns:
        bool: True if parallel processing should be used, False for serial execution.
    """
    if use_parallel is None:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            use_parallel = comm.Get_size() > 1
        except ImportError:
            use_parallel = False
    
    return use_parallel
