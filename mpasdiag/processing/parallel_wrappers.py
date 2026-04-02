#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Parallel Processing Wrappers

This module provides worker functions and parallel processing management for executing MPAS diagnostic computations and visualizations in parallel across multiple time steps. It includes worker functions for precipitation maps, surface variable maps, wind vector plots, and vertical cross-sections, each designed to be picklable for use with multiprocessing or MPI-based parallel execution. The module also contains a helper function to aggregate results from parallel workers and generate performance reports summarizing the outcomes of the parallel processing operations. 
    
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
    This worker function executes precipitation diagnostic processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of precipitation diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, computes precipitation differences, generates visualizations with the MPASPrecipitationPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis. 

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple with (time_idx, kwargs) where time_idx is the timestep index and kwargs contains processor instance, cache object, spatial bounds, variable name, accumulation period, plot type, and formatting options. 

    Returns:
        Dict[str, Any]: Result dictionary with 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics). 
    """
    try:
        time_idx, kwargs = args
        
        if 'grid_file' in kwargs and 'data_dir' in kwargs:
            from mpasdiag.processing.processors_2d import MPAS2DProcessor
            processor = MPAS2DProcessor(kwargs['grid_file'], verbose=False)
            processor = processor.load_2d_data(kwargs['data_dir'])
            cache = None  # Each rank uses its own local coordinates
        else:
            processor = kwargs['processor']
            cache = kwargs.get('cache', None)

        output_dir = kwargs['output_dir']
        lon_min = kwargs['lon_min']
        lon_max = kwargs['lon_max']
        lat_min = kwargs['lat_min']
        lat_max = kwargs['lat_max']
        var_name = kwargs['var_name']
        accum_period = kwargs['accum_period']
        plot_type = kwargs.get('plot_type', 'scatter')
        grid_resolution = kwargs.get('grid_resolution', None)
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
            title = f"MPAS Precipitation | PlotType: {plot_type.upper()} | VarType: {var_name.upper()} | Valid Time: {time_str}"
        
        plot_start = time.time()

        fig, ax = plotter.create_precipitation_map(
            lon, lat, precip_data.values,
            lon_min, lon_max, lat_min, lat_max,
            title=title,
            accum_period=accum_period,
            plot_type=plot_type,
            grid_resolution=grid_resolution,
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
            f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_ptype_{plot_type}"
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
    
    except Exception as e:
        import traceback
        time_idx = args[0] if args else 'unknown'
        error_msg = f"Error processing time index {time_idx}: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"\n{'='*60}\nWORKER ERROR\n{'='*60}")
        print(error_msg)
        print(error_trace)
        print('='*60)
        return {
            'error': error_msg,
            'traceback': error_trace,
            'time_idx': time_idx,
            'files': [],
            'timings': {},
            'time_str': f't{time_idx:03d}' if isinstance(time_idx, int) else 'unknown',
            'cache_hits': {}
        }
    

def _surface_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    This worker function executes surface variable processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of surface diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves variable data, generates visualizations with the MPASSurfacePlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis. 

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds, variable name, plot type, and formatting options. 

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics). 
    """
    time_idx, kwargs = args
    
    if 'grid_file' in kwargs and 'data_dir' in kwargs:
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        processor = MPAS2DProcessor(kwargs['grid_file'], verbose=False)
        processor = processor.load_2d_data(kwargs['data_dir'])
        cache = None
    else:
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


def _wind_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    This worker function executes wind vector processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of wind diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves 2D variable data for the specified U and V wind components, generates visualizations with the MPASWindPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis. 

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds, variable names for U and V components, plot type, subsampling factor, scaling options, and formatting settings. 

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics). 
    """
    time_idx, kwargs = args
    
    if 'grid_file' in kwargs and 'data_dir' in kwargs:
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        processor = MPAS2DProcessor(kwargs['grid_file'], verbose=False)
        processor = processor.load_2d_data(kwargs['data_dir'])
        cache = None
    else:
        processor = kwargs['processor']
        cache = kwargs.get('cache', None)

    output_dir = kwargs['output_dir']
    lon_min = kwargs['lon_min']
    lon_max = kwargs['lon_max']
    lat_min = kwargs['lat_min']
    lat_max = kwargs['lat_max']
    u_variable = kwargs['u_variable']
    v_variable = kwargs['v_variable']
    plot_type = kwargs.get('plot_type', 'barbs')
    subsample = kwargs.get('subsample', 1)
    scale = kwargs.get('scale', None)
    show_background = kwargs.get('show_background', False)
    grid_resolution = kwargs.get('grid_resolution', None)
    regrid_method = kwargs.get('regrid_method', 'linear')
    file_prefix = kwargs['file_prefix']
    formats = kwargs['formats']
    
    start_time = time.time()
    timings = {}
    cache_hits = {'coordinates': False, 'data': False}
    
    data_start = time.time()
    
    u_data = processor.get_2d_variable_data(u_variable, time_idx)
    v_data = processor.get_2d_variable_data(v_variable, time_idx)
    
    if cache is not None:
        try:
            lon, lat = cache.get_coordinates(u_variable)
            cache_hits['coordinates'] = True
        except KeyError:
            lon, lat = processor.extract_2d_coordinates_for_variable(u_variable, u_data)
            try:
                cache.load_coordinates_from_dataset(processor.dataset, u_variable)
            except:
                pass
    else:
        lon, lat = processor.extract_2d_coordinates_for_variable(u_variable, u_data)
    
    time_end = None

    if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
        time_end = pd.Timestamp(processor.dataset.Time.values[time_idx]).to_pydatetime()
        time_str = time_end.strftime('%Y%m%dT%H')
    else:
        time_str = f"t{time_idx:03d}"
    
    timings['data_processing'] = time.time() - data_start
    
    plotter = MPASWindPlotter(figsize=(12, 10))    
    plot_start = time.time()

    fig, ax = plotter.create_wind_plot(
        lon, lat, u_data.values, v_data.values,
        lon_min, lon_max, lat_min, lat_max,
        plot_type=plot_type,
        subsample=subsample,
        scale=scale,
        show_background=show_background,
        time_stamp=None,
        grid_resolution=grid_resolution,
        regrid_method=regrid_method
    )

    timings['plotting'] = time.time() - plot_start
    
    save_start = time.time()
    base_name = f"mpas_wind_{u_variable}_{v_variable}_{plot_type}_valid_{time_str}"
    output_path = os.path.join(output_dir, base_name)
    
    plotter.add_timestamp_and_branding()
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
    This worker function executes vertical cross-section processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of vertical cross-section diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves 3D variable data, generates visualizations with the MPASVerticalCrossSectionPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis. 

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds for start and end points, variable name, vertical coordinate type, number of points along the cross-section, plot type, and formatting options. 

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics). 
    """
    time_idx, kwargs = args
    
    if 'grid_file' in kwargs and 'data_dir' in kwargs:
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        processor_3d = MPAS3DProcessor(kwargs['grid_file'], verbose=False)
        processor_3d = processor_3d.load_3d_data(kwargs['data_dir'])
    else:
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
    save_path = os.path.join(output_dir, f"{file_prefix}_{var_name}_vcrd_{vertical_coord}_valid_{safe_time_str}.png")
    
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


def _process_parallel_results(results: List[Any], 
                              time_indices: List[int], 
                              output_dir: str, 
                              manager: MPASParallelManager, 
                              processing_type: str, 
                              var_info: Optional[str] = None) -> List[str]:
    """
    This helper function processes the results returned by parallel worker functions, aggregates timing metrics, counts successes and failures, and generates a comprehensive report summarizing the outcomes of the parallel processing operation. It collects successfully generated file paths, computes timing statistics for data processing, plotting, and saving phases, and retrieves overall parallel execution statistics from the manager to provide insights into performance and efficiency. The function prints a detailed report to the console with status counts, timing breakdowns, and potential speedup information based on the collected results. 

    Parameters:
        results (List[Any]): List of results returned by parallel worker functions, where each result is expected to be a dictionary containing 'files', 'timings', 'time_str', and optionally 'cache_hits' and 'cache_info'.
        time_indices (List[int]): List of time indices that were processed in parallel, used for reporting purposes.
        output_dir (str): Directory path where output files were saved, used for reporting the location of generated files.
        manager (MPASParallelManager): Instance of the parallel manager used to execute the tasks, from which overall execution statistics can be retrieved.
        processing_type (str): String indicating the type of processing performed (e.g., "Precipitation Maps", "Surface Maps", "Wind Plots", "Cross-Sections") for report header.
        var_info (Optional[str]): Optional string with variable information to include in the report header for additional context. 

    Returns:
        List[str]: List of successfully created file paths aggregated from the results. 
    """
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
    print(f"{processing_type} BATCH PROCESSING RESULTS")
    print(f"{'='*70}")

    if var_info:
        print(var_info)

    print("Status:" if not var_info else "\nStatus:")
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


class ParallelPrecipitationProcessor:
    """ This class provides static methods for parallel processing of precipitation diagnostics and visualizations using MPI-based distributed processing. """
    
    @staticmethod
    def create_batch_precipitation_maps_parallel(processor: MPAS2DProcessor,
                                                 output_dir: str, 
                                                 lon_min: float, 
                                                 lon_max: float, 
                                                 lat_min: float, 
                                                 lat_max: float, 
                                                 var_name: str = 'rainnc', 
                                                 accum_period: str = 'a01h', 
                                                 plot_type: str = 'scatter', 
                                                 grid_resolution: Optional[float] = None, 
                                                 file_prefix: str = 'mpas_precipitation_map', 
                                                 formats: List[str] = ['png'], 
                                                 custom_title_template: Optional[str] = None, 
                                                 colormap: Optional[str] = None, 
                                                 levels: Optional[List[float]] = None, 
                                                 time_indices: Optional[List[int]] = None, 
                                                 n_processes: Optional[int] = None, 
                                                 load_balance_strategy: str = "dynamic") -> Optional[List[str]]:
        """
        This method creates precipitation maps in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics and cache performance for insights into efficiency. 

        Parameters:
            processor (MPAS2DProcessor): Initialized MPAS2DProcessor instance with loaded data and grid information.
            output_dir (str): Directory path where precipitation map files will be saved.
            lon_min (float): Minimum longitude for map spatial extent in degrees.
            lon_max (float): Maximum longitude for map spatial extent in degrees.
            lat_min (float): Minimum latitude for map spatial extent in degrees.
            lat_max (float): Maximum latitude for map spatial extent in degrees.
            var_name (str): Name of the precipitation variable to process (e.g., 'rainnc').
            accum_period (str): Accumulation period string (e.g., 'a01h' for 1 hour, 'a03h' for 3 hours).
            plot_type (str): Type of plot to create ('scatter', 'contourf', etc.).
            grid_resolution (Optional[float]): Desired grid resolution for plotting, if regridding is needed.
            file_prefix (str): Prefix for output file names.
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            custom_title_template (Optional[str]): Custom title template string with placeholders {var_name}, {time_str}, {accum_period}.
            colormap (Optional[str]): Colormap name to use for precipitation visualization.
            levels (Optional[List[float]]): List of contour levels to use for plotting.
            time_indices (Optional[List[int]]): List of time indices to process. If None, all valid time indices will be processed based on accumulation period.
            n_processes (Optional[int]): Number of parallel processes to use. If None, it will default to the number of available CPU cores.
            load_balance_strategy (str): Strategy for load balancing tasks across workers ('dynamic', 'static', etc.).

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
        
        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes
        )
        manager.set_error_policy('collect')
        
        is_mpi_mode = manager.backend == 'mpi'
        
        if is_mpi_mode:
            if not hasattr(processor, 'data_dir'):
                raise AttributeError(
                    "MPI mode requires processor to have 'data_dir' attribute. "
                    "Please update mpasdiag/processing/processors_2d.py to store data_dir in load_2d_data() method, "
                    "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
                )
            
            worker_kwargs = {
                'grid_file': processor.grid_file,
                'data_dir': processor.data_dir,
                'output_dir': output_dir,
                'lon_min': lon_min,
                'lon_max': lon_max,
                'lat_min': lat_min,
                'lat_max': lat_max,
                'var_name': var_name,
                'accum_period': accum_period,
                'plot_type': plot_type,
                'grid_resolution': grid_resolution,
                'file_prefix': file_prefix,
                'formats': formats,
                'custom_title_template': custom_title_template,
                'colormap': colormap,
                'levels': levels
            }
        else:
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
                'plot_type': plot_type,
                'grid_resolution': grid_resolution,
                'file_prefix': file_prefix,
                'formats': formats,
                'custom_title_template': custom_title_template,
                'colormap': colormap,
                'levels': levels
            }
        
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
            return _process_parallel_results(
                results, time_indices, output_dir, manager, "PRECIPITATION"
            )
        
        return None


class ParallelSurfaceProcessor:
    """ This class provides static methods for parallel processing of surface variable maps and visualizations using MPI-based distributed processing. """
    
    @staticmethod
    def create_batch_surface_maps_parallel(processor: MPAS2DProcessor, 
                                           output_dir: str, 
                                           lon_min: float, 
                                           lon_max: float, 
                                           lat_min: float, 
                                           lat_max: float, 
                                           var_name: str = 't2m', 
                                           plot_type: str = 'scatter', 
                                           file_prefix: str = 'mpas_surface', 
                                           formats: List[str] = ['png'], 
                                           grid_resolution: Optional[float] = None, 
                                           clim_min: Optional[float] = None, 
                                           clim_max: Optional[float] = None, 
                                           time_indices: Optional[List[int]] = None, 
                                           n_processes: Optional[int] = None, 
                                           load_balance_strategy: str = "dynamic") -> Optional[List[str]]:
        """
        This method creates surface variable maps in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics and cache performance for insights into efficiency. 

        Parameters:
            processor (MPAS2DProcessor): Initialized MPAS2DProcessor instance with loaded data and grid information.
            output_dir (str): Directory path where surface map files will be saved.
            lon_min (float): Minimum longitude for map spatial extent in degrees.
            lon_max (float): Maximum longitude for map spatial extent in degrees.
            lat_min (float): Minimum latitude for map spatial extent in degrees.
            lat_max (float): Maximum latitude for map spatial extent in degrees.
            var_name (str): Name of the surface variable to process (e.g., 't2m').
            plot_type (str): Type of plot to create ('scatter', 'contourf', etc.).
            file_prefix (str): Prefix for output file names.
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            grid_resolution (Optional[float]): Desired grid resolution for plotting, if regridding is needed.
            clim_min (Optional[float]): Minimum value for color scale limits, if applicable.
            clim_max (Optional[float]): Maximum value for color scale limits, if applicable.
            time_indices (Optional[List[int]]): List of time indices to process. If None, all time indices in the dataset will be processed.
            n_processes (Optional[int]): Number of parallel processes to use. If None, it will default to the number of available CPU cores.
            load_balance_strategy (str): Strategy for load balancing tasks across workers ('dynamic', 'static', etc.). 

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
        total_times = processor.dataset.sizes[time_dim]
        
        if time_indices is None:
            time_indices = list(range(total_times))
        
        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes
        )
        manager.set_error_policy('collect')
        
        is_mpi_mode = manager.backend == 'mpi'
        
        if is_mpi_mode:
            if not hasattr(processor, 'data_dir'):
                raise AttributeError(
                    "MPI mode requires processor to have 'data_dir' attribute. "
                    "Please update mpasdiag/processing/processors_2d.py on your HPC system, "
                    "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
                )
            
            worker_kwargs = {
                'grid_file': processor.grid_file,
                'data_dir': processor.data_dir,
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
        else:
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
        
        os.makedirs(output_dir, exist_ok=True)
        
        if manager.is_master:
            print(f"\nCreating surface maps for {len(time_indices)} time steps in parallel...")
            print(f"Variable: {var_name}, Plot type: {plot_type}")
            print(f"Output directory: {output_dir}")
        
        results = manager.parallel_map(_surface_worker, worker_args)
        
        if manager.is_master and results is not None:
            var_info = f"Variable: {var_name}, Plot type: {plot_type}"
            return _process_parallel_results(
                results, time_indices, output_dir, manager, "SURFACE", var_info
            )
        
        return None


class ParallelWindProcessor:
    """ This class provides static methods for parallel processing of wind vector plots and visualizations using MPI-based distributed processing. """
    
    @staticmethod
    def create_batch_wind_plots_parallel(processor: MPAS2DProcessor, 
                                         output_dir: str, 
                                         lon_min: float, 
                                         lon_max: float, 
                                         lat_min: float, 
                                         lat_max: float, 
                                         u_variable: str = 'u', 
                                         v_variable: str = 'v', 
                                         plot_type: str = 'barbs', 
                                         formats: Optional[List[str]] = None, 
                                         subsample: int = 1, 
                                         scale: Optional[float] = None, 
                                         show_background: bool = False, 
                                         grid_resolution: Optional[float] = None, 
                                         regrid_method: str = 'linear', 
                                         time_indices: Optional[List[int]] = None, 
                                         n_processes: Optional[int] = None, 
                                         load_balance_strategy: str = "dynamic") -> Optional[List[str]]:
        """
        This method creates wind vector plots in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics and cache performance for insights into efficiency. 

        Parameters:
            processor (MPAS2DProcessor): Initialized MPAS2DProcessor instance with loaded wind data and grid information.
            output_dir (str): Directory path where wind plot files will be saved.   
            lon_min (float): Minimum longitude for plot spatial extent in degrees.
            lon_max (float): Maximum longitude for plot spatial extent in degrees.
            lat_min (float): Minimum latitude for plot spatial extent in degrees.
            lat_max (float): Maximum latitude for plot spatial extent in degrees.
            u_variable (str): Name of u-component wind variable in dataset (default: 'u').
            v_variable (str): Name of v-component wind variable in dataset (default: 'v').
            plot_type (str): Visualization type - 'barbs' for wind barbs or 'quiver' for arrows (default: 'barbs').
            formats (Optional[List[str]]): List of output image formats such as ['png', 'pdf'] (default: None for ['png']).
            subsample (int): Subsampling factor for wind vectors to reduce plot density (default: 1 for no subsampling).
            scale (Optional[float]): Scaling factor for wind vector lengths, None for automatic scaling (default: None).
            show_background (bool): Whether to include a background color field representing wind speed (default: False).
            grid_resolution (Optional[float]): Grid spacing in degrees for interpolation of background field (default: None for adaptive).
            regrid_method (str): Interpolation method for background field - 'nearest', 'linear', or 'cubic' (default: 'linear').
            time_indices (Optional[List[int]]): Specific timestep indices to process, None processes all (default: None).
            n_processes (Optional[int]): Number of MPI processes to use, None uses all available (default: None).
            load_balance_strategy (str): Strategy for task distribution - 'static', 'dynamic', 'block', or 'cyclic' (default: 'dynamic').

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        if formats is None:
            formats = ['png']
        
        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
        total_times = processor.dataset.sizes[time_dim]
        
        if time_indices is None:
            time_indices = list(range(total_times))
        
        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes
        )
        manager.set_error_policy('collect')
        
        is_mpi_mode = manager.backend == 'mpi'
        
        if is_mpi_mode:
            if not hasattr(processor, 'data_dir'):
                raise AttributeError(
                    "MPI mode requires processor to have 'data_dir' attribute. "
                    "Please update mpasdiag/processing/processors_2d.py on your HPC system, "
                    "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
                )
            
            worker_kwargs = {
                'grid_file': processor.grid_file,
                'data_dir': processor.data_dir,
                'output_dir': output_dir,
                'lon_min': lon_min,
                'lon_max': lon_max,
                'lat_min': lat_min,
                'lat_max': lat_max,
                'u_variable': u_variable,
                'v_variable': v_variable,
                'plot_type': plot_type,
                'subsample': subsample,
                'scale': scale,
                'show_background': show_background,
                'grid_resolution': grid_resolution,
                'regrid_method': regrid_method,
                'file_prefix': 'mpas_wind',
                'formats': formats
            }
        else:
            cache = MPASDataCache(max_variables=5)
            
            print("Pre-loading coordinates into cache...")
            try:
                cache.load_coordinates_from_dataset(processor.dataset, u_variable)
                print(f"Coordinates cached for variable: {u_variable}")
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
                'u_variable': u_variable,
                'v_variable': v_variable,
                'plot_type': plot_type,
                'subsample': subsample,
                'scale': scale,
                'show_background': show_background,
                'grid_resolution': grid_resolution,
                'regrid_method': regrid_method,
                'file_prefix': 'mpas_wind',
                'formats': formats
            }
        
        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]
        
        os.makedirs(output_dir, exist_ok=True)
        
        if manager.is_master:
            print(f"\nCreating wind vector plots for {len(time_indices)} time steps in parallel...")
            print(f"U variable: {u_variable}, V variable: {v_variable}")
            print(f"Plot type: {plot_type}")
            if show_background:
                print("Background wind speed field: enabled")
            if grid_resolution:
                print(f"Grid resolution: {grid_resolution}° (regrid method: {regrid_method})")
            print(f"Output directory: {output_dir}")
        
        results = manager.parallel_map(_wind_worker, worker_args)
        
        if manager.is_master and results is not None:
            var_info = f"U: {u_variable}, V: {v_variable}, Plot type: {plot_type}"
            return _process_parallel_results(
                results, time_indices, output_dir, manager, "WIND", var_info
            )
        
        return None


class ParallelCrossSectionProcessor:
    """ This class provides static methods for parallel processing of vertical cross-section plots and visualizations using MPI-based distributed processing. """
    
    @staticmethod
    def create_batch_cross_section_plots_parallel(mpas_3d_processor: MPAS3DProcessor, 
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
                                                  load_balance_strategy: str = "dynamic") -> Optional[List[str]]:
        """
        This method creates vertical cross-section plots in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics for insights into efficiency. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Initialized MPAS3DProcessor instance with loaded 3D data and grid information.
            var_name (str): Name of the variable to plot in the cross-section (e.g., 'temperature').
            start_point (Tuple[float, float]): Starting point of the cross-section line as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section line as (longitude, latitude) in degrees.
            output_dir (str): Directory path where cross-section plot files will be saved.
            vertical_coord (str): Vertical coordinate to use for the cross-section ('height_agl', 'pressure', etc.).
            num_points (int): Number of points along the cross-section line to sample for plotting.
            levels (Optional[np.ndarray]): Array of contour levels to use for plotting, if applicable.
            colormap (str): Colormap name to use for the variable visualization.
            extend (str): Extend option for contour plots - 'neither', 'both', 'min', or 'max'.
            plot_type (str): Type of plot to create ('contourf', 'pcolormesh', etc.).
            max_height (Optional[float]): Maximum height in kilometers to include in the cross-section, if applicable.
            file_prefix (str): Prefix for output file names.
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            time_indices (Optional[List[int]]): List of time indices to process. If None, all time indices in the dataset will be processed.
            n_processes (Optional[int]): Number of parallel processes to use. If None, it will default to the number of available CPU cores.
            load_balance_strategy (str): Strategy for load balancing tasks across workers ('dynamic', 'static', etc.).

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        time_dim = 'Time' if 'Time' in mpas_3d_processor.dataset.sizes else 'time'
        total_times = mpas_3d_processor.dataset.sizes[time_dim]
        
        if time_indices is None:
            time_indices = list(range(total_times))
        
        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes
        )
        manager.set_error_policy('collect')
        
        is_mpi_mode = manager.backend == 'mpi'
        
        if is_mpi_mode:
            if not hasattr(mpas_3d_processor, 'data_dir'):
                raise AttributeError(
                    "MPI mode requires processor to have 'data_dir' attribute. "
                    "Please update mpasdiag/processing/processors_3d.py on your HPC system, "
                    "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
                )
            
            worker_kwargs = {
                'grid_file': mpas_3d_processor.grid_file,
                'data_dir': mpas_3d_processor.data_dir,
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
        else:
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


def auto_batch_processor(use_parallel: Optional[bool] = None, 
                         **kwargs: Any) -> bool:
    """
    This function automatically determines whether to use parallel processing based on the presence of an MPI environment and the specified `use_parallel` flag. It checks for the availability of the `mpi4py` library and the number of MPI processes to decide if parallel execution is feasible. The function provides a flexible interface for selecting processing mode, allowing users to explicitly override the decision or rely on automatic detection. 

    Parameters:
        use_parallel (Optional[bool]): Optional boolean flag to explicitly select processing mode. If None, the function will attempt to auto-detect MPI environment and decide based on the number of processes. If True, it will force parallel processing; if False, it will force serial execution. 

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
