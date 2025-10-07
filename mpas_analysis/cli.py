#!/usr/bin/env python3

"""
Command Line Interface for MPAS Analysis Package

This module provides command-line interfaces for the MPAS analysis package,
including main analysis tool, batch processing, and validation utilities.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Last Modified: 2025-10-06
"""

import sys
import os
from typing import Optional

import pandas as pd

from .utils import (
    ArgumentParser, 
    MPASConfig, 
    MPASLogger, 
    PerformanceMonitor,
    load_config_file,
    validate_input_files,
    print_system_info
)
from .data_processing import MPASDataProcessor
from .visualization import MPASVisualizer, create_batch_precipitation_maps, create_batch_surface_maps, create_batch_wind_plots


def main() -> int:
    """
    Entry point for the mpas-precip-plot command.

    Orchestrates argument parsing, configuration loading, input validation and
    dispatches the main data processing and visualization pipeline.

    Parameters:
        None (reads arguments from the command line)

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        parser = ArgumentParser.create_parser()
        args = parser.parse_args()
        
        if hasattr(args, 'config') and args.config:
            config = load_config_file(args.config)
            cli_config = ArgumentParser.parse_args_to_config(args)
            for key, value in cli_config.to_dict().items():
                if value is not None:
                    setattr(config, key, value)
        else:
            config = ArgumentParser.parse_args_to_config(args)
        
        log_level = 20  
        if config.quiet:
            log_level = 40  
        elif config.verbose:
            log_level = 10  
        
        logger = MPASLogger("mpas-analyze", level=log_level, 
                           log_file=getattr(args, 'log_file', None),
                           verbose=not config.quiet)
        
        if config.verbose and not config.quiet:
            print_system_info()
            print(f"\nConfiguration:")
            for key, value in config.to_dict().items():
                print(f"  {key}: {value}")
            print()
        
        if not validate_input_files(config):
            logger.error("Input validation failed")
            return 1
        
        perf_monitor = PerformanceMonitor()
        
        with perf_monitor.timer("Total processing"):
            success = process_mpas_data(config, logger, perf_monitor)
        
        if config.verbose:
            perf_monitor.print_summary()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if config.verbose if 'config' in locals() else True:
            import traceback
            traceback.print_exc()
        return 1


def process_mpas_data(config: MPASConfig, logger: MPASLogger, 
                     perf_monitor: PerformanceMonitor) -> bool:
    """
    Process MPAS data according to configuration and generate visualizations.

    Parameters:
        config (MPASConfig): Configuration object containing input paths and plotting options.
        logger (MPASLogger): Logger instance for progress and error reporting.
        perf_monitor (PerformanceMonitor): Performance monitor used to time operations.

    Returns:
        bool: True when processing completed successfully, False on error.
    """
    try:
        with perf_monitor.timer("Data loading"):
            processor = MPASDataProcessor(config.grid_file, verbose=config.verbose)
            dataset, data_type = processor.load_data(
                config.data_dir, 
                use_pure_xarray=config.use_pure_xarray
            )
        
        logger.info(f"Data loaded successfully using {data_type}")
        
        visualizer = MPASVisualizer(
            figsize=(config.figure_width, config.figure_height),
            dpi=config.dpi
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        if config.batch_mode:
            with perf_monitor.timer("Batch visualization"):
                created_files = create_batch_precipitation_maps(
                    processor, visualizer, config.output_dir,
                    config.lon_min, config.lon_max,
                    config.lat_min, config.lat_max,
                    var_name=config.variable,
                    accum_period=config.accumulation_period,
                    formats=config.output_formats
                )
            
            logger.info(f"Created {len(created_files)} output files")
            
        else:
            time_index = getattr(config, 'time_index', 0)
            
            with perf_monitor.timer("Single step processing"):
                lon, lat = processor.extract_spatial_coordinates()
                
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_index:
                    time_end = pd.to_datetime(processor.dataset.Time.values[time_index])
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_end = None
                    time_str = f"t{time_index:03d}"
                
                precip_data = processor.compute_precipitation_difference(
                    time_index, config.variable
                )
                
                title = f"MPAS Precipitation | VarType: {config.variable.upper()} | Valid Time: {time_str}"
                
                fig, ax = visualizer.create_precipitation_map(
                    lon, lat, precip_data.values,
                    config.lon_min, config.lon_max,
                    config.lat_min, config.lat_max,
                    title=title,
                    accum_period=config.accumulation_period,
                    time_end=time_end
                )
                
                output_path = os.path.join(
                    config.output_dir,
                    f"mpas_precipitation_map_vartype_{config.variable}_acctype_{config.accumulation_period}_valid_{time_str}_point"
                )
                
                visualizer.save_plot(output_path, formats=config.output_formats)
                visualizer.close_plot()
        
        logger.info("Processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def batch_main() -> int:
    """
    Entry point for the mpas-batch-process command.

    Forces batch mode on the argument parser and invokes the main processing
    function to run the full pipeline over all detected time steps.

    Parameters:
        None (reads arguments from the command line)

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        parser = ArgumentParser.create_parser()
        parser.description = "MPAS Batch Processing Tool"
        args = parser.parse_args()
        
        args.batch_all = True
        
        return main()
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        return 1


def validate_main() -> int:
    """
    Entry point for the mpas-validate command.

    Performs basic file and dataset validation to confirm files, variables
    and coordinate information are usable by the processing pipeline.

    Parameters:
        None (reads arguments from the command line)

    Returns:
        int: Exit code (0 for success, non-zero for validation errors)
    """
    try:
        parser = ArgumentParser.create_parser()
        parser.description = "MPAS Data Validation Tool"
        args = parser.parse_args()
        
        if hasattr(args, 'config') and args.config:
            config = load_config_file(args.config)
        else:
            config = ArgumentParser.parse_args_to_config(args)
        
        print("=== MPAS Data Validation ===")
        print(f"Grid file: {config.grid_file}")
        print(f"Data directory: {config.data_dir}")
        print(f"Output directory: {config.output_dir}")
        print()
        
        if not validate_input_files(config):
            print("❌ Input validation failed")
            return 1
        
        print("✅ Basic file validation passed")
        
        try:
            processor = MPASDataProcessor(config.grid_file, verbose=True)
            dataset, data_type = processor.load_data(config.data_dir, use_pure_xarray=True)
            print(f"✅ Data loading successful ({data_type})")
            
            variables = processor.get_available_variables()
            time_range = processor.get_time_range()
            
            print(f"✅ Available variables: {len(variables)}")
            print(f"   - {', '.join(variables[:10])}")
            if len(variables) > 10:
                print(f"   - ... and {len(variables) - 10} more")
            
            print(f"✅ Time range: {time_range[0]} to {time_range[1]}")
            
            try:
                lon, lat = processor.extract_spatial_coordinates()
                print(f"✅ Spatial coordinates: {len(lon)} points")
                print(f"   - Longitude range: {lon.min():.2f} to {lon.max():.2f}")
                print(f"   - Latitude range: {lat.min():.2f} to {lat.max():.2f}")
            except Exception as e:
                print(f"⚠️  Spatial coordinate extraction: {e}")
            
            if config.variable in variables:
                try:
                    precip_data = processor.compute_precipitation_difference(0, config.variable)
                    print(f"✅ Precipitation computation successful for {config.variable}")
                    print(f"   - Data range: {precip_data.min().values:.2f} to {precip_data.max().values:.2f} mm")
                except Exception as e:
                    print(f"⚠️  Precipitation computation: {e}")
            else:
                print(f"⚠️  Variable '{config.variable}' not found in dataset")
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return 1
        
        print("\n=== Validation Summary ===")
        print("✅ All validation checks passed")
        print("Ready for data processing and visualization")
        
        return 0
        
    except Exception as e:
        print(f"Validation error: {e}")
        return 1


def surface_plot_main() -> int:
    """
    Entry point for the mpas-surface-plot command.

    Parses surface-specific CLI arguments, loads dataset and either creates a
    single surface plot for a requested time index or runs batch processing
    to produce maps for all available times.

    Parameters:
        None (reads arguments from the command line)

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        parser = ArgumentParser.create_surface_parser()
        args = parser.parse_args()
        
        config = ArgumentParser.parse_surface_args_to_config(args)
        
        logger = MPASLogger(level='INFO' if config.verbose else 'WARNING')
        perf_monitor = PerformanceMonitor()
        
        logger.info("=== MPAS Surface Variable Plotting ===")
        logger.info(f"Grid file: {config.grid_file}")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Variable: {config.variable}")
        logger.info(f"Plot type: {config.plot_type}")
        logger.info(f"Time index: {config.time_index}")
        logger.info(f"Map extent: [{config.lon_min}, {config.lon_max}] × [{config.lat_min}, {config.lat_max}]")
        logger.info("")
        
        with perf_monitor.timer("Data loading"):
            processor = MPASDataProcessor(config.grid_file, verbose=config.verbose)
            dataset, data_type = processor.load_data(config.data_dir, use_pure_xarray=True)
            logger.info(f"Data loaded successfully ({data_type})")
        
        available_vars = processor.get_available_variables()
        if config.variable not in available_vars:
            logger.error(f"Variable '{config.variable}' not found in dataset.")
            logger.info(f"Available variables: {', '.join(available_vars[:20])}")
            if len(available_vars) > 20:
                logger.info(f"... and {len(available_vars) - 20} more")
            return 1
        
        with perf_monitor.timer("Coordinate extraction"):
            lon, lat = processor.extract_spatial_coordinates()
            logger.info(f"Extracted coordinates for {len(lon)} points")
        
        with perf_monitor.timer("Variable data extraction"):
            var_data = processor.get_variable_data(config.variable, config.time_index)
            logger.info(f"Extracted {config.variable} data: {var_data.min():.3f} to {var_data.max():.3f}")
        
        time_stamp = None
        try:
            time_range = processor.get_time_range()
            if len(time_range) > config.time_index:
                time_stamp = pd.to_datetime(time_range[config.time_index])
        except Exception as e:
            logger.warning(f"Could not extract time information: {e}")
        
        with perf_monitor.timer("Visualization"):
            visualizer = MPASVisualizer(figsize=config.figure_size, dpi=config.dpi)
            
            data_array = processor.dataset[config.variable].isel(Time=config.time_index) if hasattr(processor.dataset, config.variable) else None
            
            if config.batch_mode:
                created_files = create_batch_surface_maps(processor, visualizer, config.output_dir,
                                                          config.lon_min, config.lon_max,
                                                          config.lat_min, config.lat_max,
                                                          var_name=config.variable,
                                                          plot_type=config.plot_type,
                                                          formats=config.output_formats,
                                                          grid_resolution=getattr(config, 'grid_resolution', None),
                                                          grid_resolution_deg=getattr(config, 'grid_resolution_deg', None),
                                                          clim_min=getattr(config, 'clim_min', None),
                                                          clim_max=getattr(config, 'clim_max', None))
                logger.info(f"Created {len(created_files)} output files")
                return 0

            fig, ax = visualizer.create_surface_map(
                lon, lat, var_data.values,
                config.variable,
                config.lon_min, config.lon_max,
                config.lat_min, config.lat_max,
                title=config.title,
                plot_type=config.plot_type,
                colormap=config.colormap if config.colormap != 'default' else None,
                clim_min=config.clim_min,
                clim_max=config.clim_max,
                grid_resolution=getattr(config, 'grid_resolution', None),
                grid_resolution_deg=getattr(config, 'grid_resolution_deg', None),
                time_stamp=time_stamp,
                data_array=data_array
            )
        
        with perf_monitor.timer("Output"):
            if config.output:
                output_path = config.output
            else:
                time_str = time_stamp.strftime('%Y%m%dT%H') if time_stamp else f't{config.time_index:03d}'
                output_path = os.path.join(
                    config.output_dir,
                    f"mpas_surface_{config.variable}_{config.plot_type}_{time_str}"
                )
            
            visualizer.save_plot(output_path, formats=config.output_formats)
            visualizer.close_plot()
            
            logger.info(f"Surface plot saved: {output_path}")
        
        if config.verbose:
            perf_monitor.print_summary()
        
        logger.info("Surface plotting completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Surface plotting failed: {e}")
        if config.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

def wind_plot_main() -> int:
    """
    Entry point for the mpas-wind-plot command.

    Parses wind-specific CLI arguments, loads dataset and either creates a
    single wind plot for the requested time index or runs batch processing
    to produce wind plots for all available times.

    Parameters:
        None (reads arguments from the command line)

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        parser = ArgumentParser.create_wind_parser()
        args = parser.parse_args()
        
        config = ArgumentParser.parse_wind_args_to_config(args)
        
        logger = MPASLogger(level='INFO' if config.verbose else 'WARNING')
        perf_monitor = PerformanceMonitor()
        
        logger.info("=== MPAS Wind Vector Plotting ===")
        logger.info(f"Grid file: {config.grid_file}")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"U-component: {config.u_variable}")
        logger.info(f"V-component: {config.v_variable}")
        logger.info(f"Wind level: {config.wind_level}")
        logger.info(f"Plot type: {config.wind_plot_type}")
        logger.info(f"Time index: {config.time_index}")
        logger.info(f"Map extent: [{config.lon_min}, {config.lon_max}] × [{config.lat_min}, {config.lat_max}]")
        logger.info("")
        
        with perf_monitor.timer("Data loading"):
            processor = MPASDataProcessor(config.grid_file, verbose=config.verbose)
            dataset, data_type = processor.load_data(config.data_dir, use_pure_xarray=True)
            logger.info(f"Data loaded successfully ({data_type})")
        
        available_vars = processor.get_available_variables()
        missing_vars = []

        if config.u_variable not in available_vars:
            missing_vars.append(config.u_variable)

        if config.v_variable not in available_vars:
            missing_vars.append(config.v_variable)
        
        if missing_vars:
            logger.error(f"Wind variables {missing_vars} not found in dataset.")
            logger.info(f"Available variables: {', '.join(available_vars[:20])}")
            if len(available_vars) > 20:
                logger.info(f"... and {len(available_vars) - 20} more")
            return 1
        
        with perf_monitor.timer("Coordinate extraction"):
            lon, lat = processor.extract_spatial_coordinates()
            logger.info(f"Extracted coordinates for {len(lon)} points")
        
        with perf_monitor.timer("Wind data extraction"):
            u_data, v_data = processor.get_wind_components(config.u_variable, config.v_variable, config.time_index)
            logger.info(f"Extracted wind components: U={u_data.min():.2f} to {u_data.max():.2f}, V={v_data.min():.2f} to {v_data.max():.2f}")
        
        time_stamp = None
        try:
            time_range = processor.get_time_range()
            if len(time_range) > config.time_index:
                time_stamp = pd.to_datetime(time_range[config.time_index])
        except Exception as e:
            logger.warning(f"Could not extract time information: {e}")
        
        with perf_monitor.timer("Visualization"):
            visualizer = MPASVisualizer(figsize=config.figure_size, dpi=config.dpi)
            
            lon_min = config.lon_min if config.lon_min is not None else float(lon.min())
            lon_max = config.lon_max if config.lon_max is not None else float(lon.max())
            lat_min = config.lat_min if config.lat_min is not None else float(lat.min())
            lat_max = config.lat_max if config.lat_max is not None else float(lat.max())
            
            plot_title = config.title if config.title else None

            if config.batch_mode:
                created_files = create_batch_wind_plots(processor, visualizer, config.output_dir,
                                                       lon_min, lon_max, lat_min, lat_max,
                                                       u_variable=config.u_variable,
                                                       v_variable=config.v_variable,
                                                       plot_type=config.wind_plot_type,
                                                       formats=config.output_formats,
                                                       subsample=config.subsample_factor,
                                                       scale=config.wind_scale,
                                                       show_background=config.show_background,
                                                       background_colormap=config.background_colormap)
                logger.info(f"Created {len(created_files)} wind output files")
                return 0

            fig, ax = visualizer.create_wind_plot(
                lon, lat, u_data.values, v_data.values,
                lon_min, lon_max, lat_min, lat_max,
                wind_level=config.wind_level,
                plot_type=config.wind_plot_type,
                title=plot_title,
                subsample=config.subsample_factor,
                scale=config.wind_scale,
                show_background=config.show_background,
                bg_colormap=config.background_colormap
                , time_stamp=time_stamp
            )
        
        with perf_monitor.timer("Output"):
            if config.output:
                output_path = config.output
            else:
                time_str = time_stamp.strftime('%Y%m%dT%H') if time_stamp else f't{config.time_index:03d}'
                output_path = os.path.join(
                    config.output_dir,
                    f"mpas_wind_{config.wind_level}_{config.wind_plot_type}_{time_str}"
                )
            
            visualizer.save_plot(output_path, formats=config.output_formats)
            visualizer.close_plot()
            
            logger.info(f"Wind plot saved: {output_path}")
        
        if config.verbose:
            perf_monitor.print_summary()
        
        logger.info("Wind plotting completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Wind plotting failed: {e}")
        if config.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

