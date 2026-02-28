#!/usr/bin/env python3

"""
Unified Command Line Interface for MPAS Analysis

This module provides a comprehensive command-line interface for MPAS atmospheric model diagnostics supporting all visualization types (precipitation, surface, wind, vertical cross-sections, and complex multi-variable overlays) with both serial and batch processing capabilities. It implements a modular argparse-based CLI with subcommands for each analysis type, specialized argument mapping and validation, centralized configuration management via YAML files with command-line overrides, and flexible processing modes including single time step analysis, batch processing over all time steps or specified ranges, and experimental parallel processing support. The unified CLI integrates seamlessly with MPASdiag visualization classes (MPASPrecipitationPlotter, MPASSurfacePlotter, MPASVerticalCrossSectionPlotter, MPASWindPlotter) and processing utilities (MPAS2DProcessor, MPAS3DProcessor), providing performance monitoring, comprehensive error handling, detailed logging with multiple verbosity levels, and extensive customization options. Core capabilities include automatic file discovery with glob patterns, geographic extent specification, colormap and level customization, output format control, and extensible architecture for adding new analysis types with minimal code changes.

Classes:
    MPASUnifiedCLI: Main class implementing the unified command-line interface

Commands:
    precipitation: Precipitation accumulation map generation with period-specific colormaps
    surface: Surface variable scalar field visualization with scatter/contour rendering
    wind: Wind vector plots with barbs or arrows and optional background fields
    cross: 3D vertical atmospheric cross-sections along great-circle paths
    overlay: Complex multi-variable overlay analyses with temporal comparisons
    
Functions:
    create_cli_parser: Creates the main argument parser with subcommands for all analysis types
    parse_and_validate_args: Parses command-line arguments and validates against configuration requirements
    map_args_to_plotter_params: Maps CLI arguments to plotter-specific parameter dictionaries
    execute_analysis: Executes the requested analysis workflow with performance monitoring
    main: Entry point function orchestrating CLI parsing, validation, and analysis execution
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import sys
import os
import argparse
import textwrap
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from .utils_config import MPASConfig
from .utils_logger import MPASLogger
from .utils_monitor import PerformanceMonitor
from .utils_validator import DataValidator
from .processors_2d import MPAS2DProcessor
from .processors_3d import MPAS3DProcessor
from .parallel_wrappers import (
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor,
    auto_batch_processor
)
from .constants import DIAG_GLOB, MPASOUT_GLOB, PERFORMANCE_MONITOR_MSG

try:
    from ..visualization.precipitation import MPASPrecipitationPlotter
    from ..visualization.surface import MPASSurfacePlotter
    from ..visualization.cross_section import MPASVerticalCrossSectionPlotter
    from ..visualization.base_visualizer import MPASVisualizer
    from ..visualization.wind import MPASWindPlotter
    from ..diagnostics.precipitation import PrecipitationDiagnostics
except ImportError:
    try:
        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
        from mpasdiag.visualization.base_visualizer import MPASVisualizer
        from mpasdiag.visualization.wind import MPASWindPlotter
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
    except ImportError:
        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
        from mpasdiag.visualization.base_visualizer import MPASVisualizer
        from mpasdiag.visualization.wind import MPASWindPlotter
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics


class MPASUnifiedCLI:
    """
    Unified command-line interface for all MPAS visualization types.
    
    This class provides a comprehensive CLI that supports:
    - Precipitation analysis (serial/batch)
    - Surface variable plotting (serial/batch)
    - Wind vector plotting (serial/batch)
    - 3D vertical cross-sections (serial/batch)
    - Complex overlay analyses (serial/batch)
    
    Features:
    - Automatic plot type detection
    - Flexible configuration management
    - Performance monitoring
    - Error handling and validation
    - Multiple output formats
    - Extensible architecture for new plot types
    """
    
    PLOT_TYPES = {
        'precipitation': 'Precipitation accumulation maps',
        'surface': 'Surface variable scalar plots',
        'wind': 'Wind vector plots (barbs/arrows)',
        'cross': '3D vertical cross-section plots',
        'overlay': 'Complex multi-variable overlay plots'
    }
    
    def __init__(self) -> None:
        """
        Initialize the unified CLI instance with placeholder components enabling lazy configuration and fast startup performance. This constructor creates the CLI object without immediately allocating heavy components like loggers, performance monitors, or configuration objects to minimize initialization overhead. Placeholders are set for the logger, performance monitor, and configuration object attributes which will be properly instantiated when parse_args_to_config is invoked with parsed arguments. This deferred initialization design pattern enables the CLI to be instantiated quickly for help display or argument validation while deferring expensive setup until analysis execution begins. The instance maintains mutable state across the complete argument parsing, configuration validation, and analysis execution pipeline.

        Parameters:
            None

        Returns:
            None
        """
        self.logger = None
        self.perf_monitor = None
        self.config = None
    
    def create_main_parser(self) -> argparse.ArgumentParser:
        """
        Construct the main argument parser with comprehensive subcommands supporting all MPAS atmospheric visualization types and analysis workflows. This method initializes the top-level ArgumentParser with program metadata including version information, description text, and detailed usage examples formatted with RawDescriptionHelpFormatter for multi-line epilogs. It creates a subparser system for organizing the five primary analysis types (precipitation, surface, wind, cross-section, overlay) each with specialized argument groups. The method delegates to specialized helper methods (_add_precipitation_parser, _add_surface_parser, etc.) to configure analysis-specific arguments while maintaining consistent interface patterns. The resulting parser provides a unified command-line interface with global options (config file, verbosity, logging) and analysis-specific arguments with proper defaults and validation constraints.

        Parameters:
            None

        Returns:
            argparse.ArgumentParser: Fully configured main parser object with all registered subcommands, argument groups, defaults, and help text for the five analysis types.
        """
        parser = argparse.ArgumentParser(
            prog='mpasdiag',
            description='Unified MPAS Data Analysis and Visualization Tool',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Precipitation analysis (single time step)
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc --time-index 12
              
              # Precipitation batch processing
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc --batch-all
              
              # Surface temperature plot
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable t2m --plot-type contour
              
              # Wind vectors with background
              mpasdiag wind --grid-file grid.nc --data-dir ./data --u-variable u10 --v-variable v10 --show-background
              
              # Vertical cross-section
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable theta --start-lon -105 --start-lat 40 --end-lon -95 --end-lat 35
              
              # Complex overlay (precipitation + wind)
              mpasdiag overlay --grid-file grid.nc --data-dir ./data --overlay-type precip_wind --variable rainnc --u-variable u10 --v-variable v10
              
              # Use configuration file
              mpasdiag --config analysis_config.yaml
            """)
        )
        
        parser.add_argument('--config', type=str, help='Configuration file path (YAML format)')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output messages')
        parser.add_argument('--log-file', type=str, help='Log file path')
        parser.add_argument('--version', action='version', version='MPASdiag 1.0.0')
        
        subparsers = parser.add_subparsers(
            dest='analysis_command',
            title='Analysis Types',
            description='Choose the type of analysis to perform',
            help='Available analysis types'
        )
        
        self._add_precipitation_parser(subparsers)
        self._add_surface_parser(subparsers)
        self._add_wind_parser(subparsers)
        self._add_cross_parser(subparsers)
        self._add_overlay_parser(subparsers)
        
        return parser
    
    def _add_common_arguments(
        self,
        parser: argparse.ArgumentParser,
        required_grid: bool = True
    ) -> None:
        """
        Add common command-line arguments shared across multiple analysis type subparsers ensuring consistent user interface patterns. This method defines and registers five argument groups covering essential analysis parameters: input/output paths, spatial extent specification, temporal selection, output formatting options, and processing controls. The arguments include grid file path, data directory location, output directory, geographic bounding box (lat/lon min/max), time index or range selection, batch processing modes, figure dimensions, DPI settings, output format specifications, colormap selection, color limits, and parallel processing configuration. Making the grid file requirement optionally configurable allows flexibility for analysis types that may not require explicit grid files. This centralized argument definition eliminates code duplication across subcommands and ensures uniform parameter naming conventions.

        Parameters:
            parser (argparse.ArgumentParser): Argument parser or subparser object to which the common argument groups will be added through add_argument calls.
            required_grid (bool): Flag controlling whether the grid file argument is marked as required or optional in the parser configuration (default: True).

        Returns:
            None
        """
        io_group = parser.add_argument_group('Input/Output')
        io_group.add_argument('--grid-file', type=str, required=required_grid,
                             help='Path to MPAS grid file')
        io_group.add_argument('--data-dir', type=str, required=True,
                             help='Directory containing MPAS diagnostic files')
        io_group.add_argument('--output-dir', type=str, default='./output',
                             help='Output directory for plots and results')
        
        spatial_group = parser.add_argument_group('Spatial Extent')
        spatial_group.add_argument('--lat-min', type=float, default=-9.60,
                                  help='Minimum latitude (default: -9.60)')
        spatial_group.add_argument('--lat-max', type=float, default=12.20,
                                  help='Maximum latitude (default: 12.20)')
        spatial_group.add_argument('--lon-min', type=float, default=91.00,
                                  help='Minimum longitude (default: 91.00)')
        spatial_group.add_argument('--lon-max', type=float, default=113.00,
                                  help='Maximum longitude (default: 113.00)')
        
        time_group = parser.add_argument_group('Time Selection')
        time_group.add_argument('--time-index', type=int, default=0,
                               help='Time index to process (default: 0)')
        time_group.add_argument('--batch-all', action='store_true',
                               help='Process all time steps in batch mode')
        time_group.add_argument('--time-range', type=int, nargs=2, metavar=('START', 'END'),
                               help='Process time range [start, end] (inclusive)')
        
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--output', '-o', type=str,
                                 help='Output filename (without extension)')
        output_group.add_argument('--title', type=str,
                                 help='Custom plot title')
        output_group.add_argument('--dpi', type=int, default=100,
                                 help='Output resolution (DPI) - default: 100, use 300+ for publication quality')
        output_group.add_argument('--figure-size', type=float, nargs=2,
                                 default=[12.0, 10.0], metavar=('WIDTH', 'HEIGHT'),
                                 help='Figure size in inches (default: 12.0 10.0)')
        output_group.add_argument('--formats', type=str, nargs='+',
                                 default=['png'], choices=['png', 'pdf', 'svg', 'eps', 'jpg'],
                                 help='Output formats (default: png)')
        
        color_group = parser.add_argument_group('Color Settings')
        color_group.add_argument('--colormap', type=str, default='default',
                               help='Colormap name (default: auto-selected)')
        color_group.add_argument('--clim-min', type=float,
                               help='Minimum color limit')
        color_group.add_argument('--clim-max', type=float,
                               help='Maximum color limit')
        
        proc_group = parser.add_argument_group('Processing Options')
        proc_group.add_argument('--parallel', action='store_true',
                               help='Enable parallel processing (experimental)')
        proc_group.add_argument('--workers', type=int, default=None,
                               help='Number of parallel workers (default: auto-detect based on CPU cores)')
        proc_group.add_argument('--chunk-size', type=int, default=100000,
                               help='Data chunk size for processing (default: 100000)')
    
    def _add_precipitation_parser(
        self,
        subparsers: argparse._SubParsersAction
    ) -> None:
        """
        Create and register the precipitation subparser enabling accumulation analysis workflows for convective and stratiform rainfall. This method constructs a new subparser with command aliases ('precip', 'rain') to handle precipitation-specific arguments including variable selection between convective (rainc), non-convective (rainnc), and total precipitation. The subparser configures accumulation period options (1h, 3h, 6h, 12h, 24h), plot types (scatter for direct cell values or contourf for interpolated smooth fields), grid resolution controls for interpolation, threshold highlighting, and unit conversions. It delegates to _add_common_arguments for shared parameters then adds a precipitation-specific argument group. The method includes comprehensive usage examples in the help epilog demonstrating single time step and batch processing workflows.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the precipitation parser is registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            'precipitation',
            aliases=['precip', 'rain'],
            help='Precipitation accumulation analysis',
            description='Generate precipitation accumulation maps',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Single time step
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc
              
              # Batch processing
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc --batch-all
              
              # Custom accumulation period
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc --accumulation a24h
            """)
        )
        
        self._add_common_arguments(parser)
        
        precip_group = parser.add_argument_group('Precipitation Options')
        precip_group.add_argument('--variable', '--var', type=str, default='rainnc',
                                choices=['rainc', 'rainnc', 'total'],
                                help='Precipitation variable (default: rainnc)')
        precip_group.add_argument('--accumulation', '--accum', type=str, default='a01h',
                                choices=['a01h', 'a03h', 'a06h', 'a12h', 'a24h'],
                                help='Accumulation period (default: a01h)')
        precip_group.add_argument('--plot-type', type=str, default='scatter',
                                choices=['scatter', 'contourf'],
                                help='Plot type - scatter for direct cell display or contourf for interpolated smooth fields (default: scatter)')
        precip_group.add_argument('--grid-resolution', type=float,
                                help='Grid resolution in degrees for contourf interpolation (e.g., 0.5 for 0.5° grid, uses adaptive if not specified)')
        precip_group.add_argument('--threshold', type=float,
                                help='Precipitation threshold for highlighting')
        precip_group.add_argument('--units', type=str, choices=['mm', 'in', 'cm'],
                                default='mm', help='Output units (default: mm)')
    
    def _add_surface_parser(
        self,
        subparsers: argparse._SubParsersAction
    ) -> None:
        """
        Create and register the surface subparser enabling 2D meteorological scalar field visualization workflows for variables like temperature, pressure, and moisture. This method constructs a new subparser with command aliases ('surf', '2d') to handle surface-specific arguments including required variable name selection, plot type specification among four rendering styles (scatter for direct point display, contour for line contours, contourf for smooth filled regions, pcolormesh for grid cell coloring), and gridding options for interpolation control. The subparser configures grid resolution in both point count and degree spacing formats, interpolation method selection (linear, cubic, nearest), and contour level specification. It delegates to _add_common_arguments for shared parameters then adds a surface-specific argument group. The method provides comprehensive usage examples demonstrating temperature scatter plots, pressure contour plots, and custom grid resolution configurations.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the surface parser is registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            'surface',
            aliases=['surf', '2d'],
            help='Surface variable analysis',
            description='Generate surface variable scalar plots',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Temperature scatter plot
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable t2m
              
              # Pressure contour plot
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable mslp --plot-type contour
              
              # Custom grid resolution
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable t2m --plot-type contour --grid-resolution 0.1
            """)
        )
        
        self._add_common_arguments(parser)
        
        surf_group = parser.add_argument_group('Surface Variable Options')
        surf_group.add_argument('--variable', '--var', type=str, required=True,
                              help='Surface variable name (e.g., t2m, mslp, q2, u10, v10)')
        surf_group.add_argument('--plot-type', type=str, default='scatter',
                              choices=['scatter', 'contour', 'contourf', 'pcolormesh'],
                              help='Plot type (default: scatter)')
        surf_group.add_argument('--grid-resolution', type=float,
                              help='Grid resolution in degrees (e.g., 0.1 for 0.1° grid)')
        surf_group.add_argument('--interpolation', type=str, default='linear',
                              choices=['linear', 'cubic', 'nearest'],
                              help='Interpolation method for contour plots (default: linear)')
        surf_group.add_argument('--contour-levels', type=int, default=15,
                              help='Number of contour levels (default: 15)')
    
    def _add_wind_parser(
        self,
        subparsers: argparse._SubParsersAction
    ) -> None:
        """
        Create and register the wind subparser enabling vector field visualization workflows for horizontal wind components at various atmospheric levels. This method constructs a new subparser with command aliases ('vector', 'winds') to handle wind-specific arguments including u and v component variable names for zonal and meridional winds, vertical level descriptions for labeling, and vector plot type selection among three representation styles (barbs for meteorological convention, arrows for directional flow, streamlines for continuous trajectories). The subparser configures subsampling factors to control vector density, scaling factors for arrow and barb sizes, optional background wind speed field overlays with customizable colormaps, vector color and transparency controls, and regridding options for interpolation to regular grids. It delegates to _add_common_arguments for shared parameters then adds a comprehensive wind-specific argument group. The method includes detailed usage examples for surface wind barbs, 850mb wind arrows with background fields, and custom subsampling configurations.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the wind parser is registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            'wind',
            aliases=['vector', 'winds'],
            help='Wind vector analysis',
            description='Generate wind vector plots with barbs or arrows',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Surface wind barbs
              mpasdiag wind --grid-file grid.nc --data-dir ./data --u-variable u10 --v-variable v10
              
              # 850mb wind arrows with background
              mpasdiag wind --grid-file grid.nc --data-dir ./data --u-variable u850 --v-variable v850 --wind-plot-type arrows --show-background
              
              # Custom subsampling and scaling
              mpasdiag wind --grid-file grid.nc --data-dir ./data --u-variable u10 --v-variable v10 --subsample 3 --wind-scale 50
            """)
        )
        
        self._add_common_arguments(parser)
        
        wind_group = parser.add_argument_group('Wind Vector Options')
        wind_group.add_argument('--u-variable', type=str, default='u10',
                              help='U-component wind variable (default: u10)')
        wind_group.add_argument('--v-variable', type=str, default='v10',
                              help='V-component wind variable (default: v10)')
        wind_group.add_argument('--wind-level', type=str, default='surface',
                              help='Wind level description for labeling (default: surface)')
        wind_group.add_argument('--wind-plot-type', type=str, default='barbs',
                              choices=['barbs', 'arrows', 'streamlines'],
                              help='Wind vector representation (default: barbs)')
        wind_group.add_argument('--subsample', type=int, default=0,
                              help='Subsample factor (plot every Nth point, 0=auto)')
        wind_group.add_argument('--wind-scale', type=float,
                              help='Scale factor for wind vectors (auto if not specified)')
        wind_group.add_argument('--show-background', action='store_true',
                              help='Show background wind speed as filled contours')
        wind_group.add_argument('--background-colormap', type=str, default='viridis',
                              help='Colormap for background wind speed (default: viridis)')
        wind_group.add_argument('--vector-color', type=str, default='black',
                              help='Color for wind vectors (default: black)')
        wind_group.add_argument('--vector-alpha', type=float, default=0.8,
                              help='Transparency for wind vectors (default: 0.8)')
        wind_group.add_argument('--grid-resolution', type=float,
                              help='Grid resolution in degrees for regridding wind components (e.g., 0.5 for 0.5° grid, no regridding if not specified)')
        wind_group.add_argument('--regrid-method', type=str, default='linear',
                              choices=['linear', 'nearest'],
                              help='Interpolation method for regridding - linear for smooth fields or nearest for preserving values (default: linear)')
    
    def _add_cross_parser(
        self,
        subparsers: argparse._SubParsersAction
    ) -> None:
        """
        Create and register the cross-section subparser enabling 3D vertical atmospheric slice visualization workflows along specified great circle paths. This method constructs a new subparser with command aliases ('xsec', '3d', 'vertical') to handle cross-section-specific arguments including required start and end point coordinates in longitude and latitude, 3D atmospheric variable selection, and vertical coordinate system specification among three types (pressure for isobaric levels, height for geometric altitude, model_levels for native vertical coordinates). The subparser configures interpolation resolution along the horizontal path with the num_points parameter, maximum vertical extent controls, plot style selection (contourf for smooth color fields, contour for line contours, pcolormesh for grid cells), and colorbar extension settings. It requires grid file specification and delegates to _add_common_arguments for shared parameters. The method provides comprehensive usage examples demonstrating temperature and wind cross-sections with different vertical coordinate systems.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the cross-section parser is registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            'cross',
            aliases=['xsec', '3d', 'vertical'],
            help='3D vertical cross-section analysis',
            description='Generate vertical cross-section plots along specified paths',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Temperature cross-section
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable theta --start-lon -105 --start-lat 40 --end-lon -95 --end-lat 35
              
              # Wind cross-section with pressure coordinates
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable uReconstructZonal --start-lon -110 --start-lat 35 --end-lon -80 --end-lat 45 --vertical-coord pressure
            """)
        )
        
        self._add_common_arguments(parser, required_grid=True)
        
        xsec_group = parser.add_argument_group('Cross-Section Options')
        xsec_group.add_argument('--variable', type=str, required=True,
                              help='3D atmospheric variable name (e.g., theta, uReconstructZonal)')
        xsec_group.add_argument('--start-lon', type=float, required=True,
                              help='Starting longitude in degrees')
        xsec_group.add_argument('--start-lat', type=float, required=True,
                              help='Starting latitude in degrees')
        xsec_group.add_argument('--end-lon', type=float, required=True,
                              help='Ending longitude in degrees')
        xsec_group.add_argument('--end-lat', type=float, required=True,
                              help='Ending latitude in degrees')
        xsec_group.add_argument('--vertical-coord', type=str, default='pressure',
                              choices=['pressure', 'model_levels', 'height'],
                              help='Vertical coordinate system (default: pressure)')
        xsec_group.add_argument('--num-points', type=int, default=100,
                              help='Number of interpolation points along path (default: 100)')
        xsec_group.add_argument('--max-height', type=float,
                              help='Maximum height in km for vertical axis (auto if not specified)')
        xsec_group.add_argument('--plot-style', type=str, default='contourf',
                              choices=['contourf', 'contour', 'pcolormesh'],
                              help='Cross-section plot style (default: contourf)')
        xsec_group.add_argument('--extend', type=str, default='both',
                              choices=['both', 'min', 'max', 'neither'],
                              help='Colorbar extension (default: both)')
    
    def _add_overlay_parser(
        self,
        subparsers: argparse._SubParsersAction
    ) -> None:
        """
        Create and register the overlay subparser enabling complex multi-variable composite visualization workflows combining multiple atmospheric fields. This method constructs a new subparser with command aliases ('complex', 'multi', 'composite') to handle overlay-specific arguments including required overlay type selection from five predefined combinations (precip_wind for precipitation with wind vectors, temp_pressure for temperature with pressure contours, wind_temp for winds with temperature background, multi_level for arbitrary multi-variable displays, custom for user-defined composites). The subparser configures primary and secondary variable specifications, comma-separated variable lists for multi-variable overlays, u/v component wind variables for wind overlays, pressure variables for isobaric analysis, separate colormaps for each overlay layer, transparency controls for blending, and optional contour line overlays for secondary variables. It delegates to _add_common_arguments for shared parameters then adds comprehensive overlay-specific argument groups. The method provides detailed usage examples for precipitation-wind, temperature-pressure, and multi-level composite visualizations.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the overlay parser is registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            'overlay',
            aliases=['complex', 'multi', 'composite'],
            help='Complex multi-variable overlay analysis',
            description='Generate complex overlay plots combining multiple variables',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Precipitation + wind overlay
              mpas-analyze overlay --grid-file grid.nc --data-dir ./data --overlay-type precip_wind --variable rainnc --u-variable u10 --v-variable v10
              
              # Temperature + pressure overlay
              mpas-analyze overlay --grid-file grid.nc --data-dir ./data --overlay-type temp_pressure --variable t2m --pressure-variable mslp
              
              # Multi-level analysis
              mpas-analyze overlay --grid-file grid.nc --data-dir ./data --overlay-type multi_level --variables t2m,mslp,u10,v10
            """)
        )
        
        self._add_common_arguments(parser)
        
        overlay_group = parser.add_argument_group('Overlay Options')
        overlay_group.add_argument('--overlay-type', type=str, required=True,
                                 choices=['precip_wind', 'temp_pressure', 'wind_temp', 'multi_level', 'custom'],
                                 help='Type of overlay analysis to perform')
        overlay_group.add_argument('--primary-variable', '--variable', type=str,
                                 help='Primary variable for the overlay')
        overlay_group.add_argument('--secondary-variable', type=str,
                                 help='Secondary variable for the overlay')
        overlay_group.add_argument('--variables', type=str,
                                 help='Comma-separated list of variables for multi-variable overlays')
        overlay_group.add_argument('--u-variable', type=str, default='u10',
                                 help='U-component wind variable (for wind overlays)')
        overlay_group.add_argument('--v-variable', type=str, default='v10',
                                 help='V-component wind variable (for wind overlays)')
        overlay_group.add_argument('--pressure-variable', type=str, default='mslp',
                                 help='Pressure variable (for pressure overlays)')
        overlay_group.add_argument('--primary-colormap', type=str,
                                 help='Colormap for primary variable')
        overlay_group.add_argument('--secondary-colormap', type=str,
                                 help='Colormap for secondary variable')
        overlay_group.add_argument('--transparency', type=float, default=0.7,
                                 help='Transparency for overlay elements (default: 0.7)')
        overlay_group.add_argument('--contour-overlay', action='store_true',
                                 help='Add contour lines for secondary variable')
    
    def parse_args_to_config(
        self,
        args: argparse.Namespace
    ) -> MPASConfig:
        """
        Transform parsed command-line arguments into a comprehensive MPASConfig object suitable for analysis pipeline execution and validation. This method performs systematic mapping of argparse namespace attributes to configuration dictionary keys through two transformation phases: common parameter mapping applying to all analysis types, followed by analysis-specific mapping based on the command subparser invoked. The common mapping handles shared parameters including file paths, spatial bounds, temporal selection, output formatting, and processing options. Analysis-specific mapping delegates to specialized helper methods (_map_precipitation_args, _map_surface_args, _map_wind_args, _map_cross_args, _map_overlay_args) based on the analysis_command attribute. The method automatically enables batch mode when time range specifications are provided and handles figure size tuple conversion from list format. Returns a fully-populated MPASConfig instance ready for validation and analysis execution.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments from argparse containing all user-specified options, defaults, and the selected analysis command with its subcommand-specific parameters.

        Returns:
            MPASConfig: Fully configured configuration object with all analysis settings populated from command-line arguments and ready for validation pipeline execution.
        """
        config_dict = {}
        
        common_mapping = {
            'grid_file': 'grid_file',
            'data_dir': 'data_dir',
            'output_dir': 'output_dir',
            'lat_min': 'lat_min',
            'lat_max': 'lat_max',
            'lon_min': 'lon_min',
            'lon_max': 'lon_max',
            'time_index': 'time_index',
            'batch_all': 'batch_mode',
            'colormap': 'colormap',
            'dpi': 'dpi',
            'formats': 'output_formats',
            'title': 'title',
            'output': 'output',
            'clim_min': 'clim_min',
            'clim_max': 'clim_max',
            'parallel': 'parallel',
            'workers': 'workers',  
            'chunk_size': 'chunk_size'
        }
        
        for arg_name, config_attr in common_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
        
        if hasattr(args, 'figure_size') and args.figure_size:
            config_dict['figure_size'] = tuple(args.figure_size)
        
        if hasattr(args, 'time_range') and args.time_range:
            config_dict['time_start'] = args.time_range[0]
            config_dict['time_end'] = args.time_range[1]
            config_dict['batch_mode'] = True
        
        if hasattr(args, 'analysis_command') and args.analysis_command:
            config_dict['analysis_type'] = args.analysis_command
            
            if args.analysis_command in ['precipitation', 'precip', 'rain']:
                self._map_precipitation_args(args, config_dict)
            elif args.analysis_command in ['surface', 'surf', '2d']:
                self._map_surface_args(args, config_dict)
            elif args.analysis_command in ['wind', 'vector', 'winds']:
                self._map_wind_args(args, config_dict)
            elif args.analysis_command in ['cross', 'xsec', '3d', 'vertical']:
                self._map_cross_args(args, config_dict)
            elif args.analysis_command in ['overlay', 'complex', 'multi', 'composite']:
                self._map_overlay_args(args, config_dict)
        
        return MPASConfig(**config_dict)
    
    def _map_precipitation_args(
        self,
        args: argparse.Namespace,
        config_dict: Dict[str, Any]
    ) -> None:
        """
        Map precipitation-specific command-line arguments into the configuration dictionary enabling accumulation analysis parameter configuration. This method extracts precipitation-related parameters from the argparse namespace including variable name selection (rainc, rainnc, total), accumulation period specification (a01h through a24h), plot type choice between scatter and contourf rendering, grid resolution for interpolation operations, precipitation threshold values for highlighting significant accumulation, and output unit preferences (mm, cm, in). The mapping translates user-facing CLI argument names to internal configuration attribute names through a systematic dictionary lookup while gracefully handling optional parameters by checking for attribute existence before assignment. This specialized mapping ensures precipitation-specific options flow correctly from command-line inputs to internal configuration structures used by PrecipitationDiagnostics and MPASPrecipitationPlotter classes.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing precipitation-specific options extracted from the precipitation subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with precipitation analysis parameters through in-place modification.

        Returns:
            None
        """
        precip_mapping = {
            'variable': 'variable',
            'accumulation': 'accumulation_period',
            'plot_type': 'plot_type',
            'grid_resolution': 'grid_resolution',
            'threshold': 'precip_threshold',
            'units': 'precip_units'
        }
        
        for arg_name, config_attr in precip_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_surface_args(
        self,
        args: argparse.Namespace,
        config_dict: Dict[str, Any]
    ) -> None:
        """
        Map surface-specific command-line arguments into the configuration dictionary enabling 2D scalar field visualization parameter configuration. This method extracts surface analysis parameters from the argparse namespace including required variable name selection for meteorological fields (t2m, mslp, q2, etc.), plot type specification among four rendering options (scatter, contour, contourf, pcolormesh), grid resolution controls specified either as integer point counts per axis or as floating-point degree spacing for interpolation grids, interpolation method selection from three algorithms (linear for smooth gradients, cubic for high-quality smoothing, nearest for preserving exact values), and contour level count specification. The mapping systematically translates CLI argument names to internal configuration attribute names while handling optional parameters through existence checks before assignment. This specialized mapping ensures surface plotting options correctly propagate from command-line inputs to internal configuration structures used by MPAS2DProcessor and MPASSurfacePlotter classes.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing surface-specific options extracted from the surface subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with surface visualization parameters through in-place modification.

        Returns:
            None
        """
        surface_mapping = {
            'variable': 'variable',
            'plot_type': 'plot_type',
            'grid_resolution': 'grid_resolution',
            'interpolation': 'interpolation_method',
            'contour_levels': 'contour_levels'
        }
        
        for arg_name, config_attr in surface_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_wind_args(
        self,
        args: argparse.Namespace,
        config_dict: Dict[str, Any]
    ) -> None:
        """
        Map wind-specific command-line arguments into the configuration dictionary enabling vector field visualization parameter configuration. This method extracts wind analysis parameters from the argparse namespace including u and v component variable names for zonal and meridional wind components, vertical level description string for plot labeling (surface, 850mb, etc.), wind plot type selection from three vector representation styles (barbs for meteorological standard, arrows for directional flow, streamlines for continuous trajectories), subsampling factor to control vector display density, wind scale factor for arrow and barb sizing, background field display toggle for overlaying wind speed magnitude as filled contours, background colormap specification, vector color and alpha transparency controls for styling, grid resolution for optional regridding to regular latitude-longitude grids, and regridding interpolation method (linear or nearest). The mapping systematically converts CLI argument names to internal configuration keys while gracefully handling optional parameters. This specialized mapping ensures wind visualization options correctly flow from command-line inputs to MPASWindPlotter configurations.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing wind-specific options extracted from the wind subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with wind visualization parameters through in-place modification.

        Returns:
            None
        """
        wind_mapping = {
            'u_variable': 'u_variable',
            'v_variable': 'v_variable',
            'wind_level': 'wind_level',
            'wind_plot_type': 'wind_plot_type',
            'subsample': 'subsample_factor',
            'wind_scale': 'wind_scale',
            'show_background': 'show_background',
            'background_colormap': 'background_colormap',
            'vector_color': 'vector_color',
            'vector_alpha': 'vector_alpha',
            'grid_resolution': 'grid_resolution',
            'regrid_method': 'regrid_method'
        }
        
        for arg_name, config_attr in wind_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_cross_args(
        self,
        args: argparse.Namespace,
        config_dict: Dict[str, Any]
    ) -> None:
        """
        Map cross-section-specific command-line arguments into the configuration dictionary enabling vertical atmospheric slice parameter configuration. This method extracts 3D cross-section parameters from the argparse namespace including 3D atmospheric variable name selection, required start point coordinates (start_lon, start_lat) defining the cross-section path origin, required end point coordinates (end_lon, end_lat) defining the path terminus, vertical coordinate system specification from three options (pressure for isobaric levels, height for geometric altitude, model_levels for native sigma or hybrid coordinates), number of interpolation points controlling horizontal resolution along the great circle path, maximum vertical extent in kilometers for plot axis limits, plot style selection from three rendering types (contourf for smooth color fields, contour for line contours, pcolormesh for grid cell display), and colorbar extension settings controlling whether colorbar extends beyond data limits. The mapping systematically translates user-facing CLI argument names to internal configuration keys while handling all required and optional parameters. This specialized mapping ensures cross-section visualization options correctly propagate to MPAS3DProcessor and MPASVerticalCrossSectionPlotter classes.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing cross-section-specific options extracted from the cross-section subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with cross-section analysis parameters through in-place modification.

        Returns:
            None
        """
        xsec_mapping = {
            'variable': 'variable',
            'start_lon': 'start_lon',
            'start_lat': 'start_lat',
            'end_lon': 'end_lon',
            'end_lat': 'end_lat',
            'vertical_coord': 'vertical_coord',
            'num_points': 'num_points',
            'max_height': 'max_height',
            'plot_style': 'plot_style',
            'extend': 'extend'
        }
        
        for arg_name, config_attr in xsec_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_overlay_args(
        self,
        args: argparse.Namespace,
        config_dict: Dict[str, Any]
    ) -> None:
        """
        Map overlay-specific command-line arguments into the configuration dictionary enabling multi-variable composite visualization parameter configuration. This method extracts overlay parameters from the argparse namespace including required overlay type specification from five predefined combinations, primary variable name for the base layer, secondary variable name for overlay layers, comma-separated variable list string for multi-variable displays which gets parsed into a list, u and v component wind variable names for wind overlay components, pressure variable specification for isobaric contour overlays, separate colormap selections for primary and secondary layers enabling independent styling, transparency alpha value for blending overlay elements, and contour overlay boolean flag enabling line contour overlays atop filled fields. The mapping systematically converts CLI argument names to internal configuration keys while handling the special case of splitting comma-separated variable lists into arrays. This specialized mapping ensures complex overlay visualization options correctly propagate from command-line inputs to composite plotting routines combining multiple plotter classes.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing overlay-specific options extracted from the overlay subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with overlay analysis parameters through in-place modification.

        Returns:
            None
        """
        overlay_mapping = {
            'overlay_type': 'overlay_type',
            'primary_variable': 'primary_variable',
            'secondary_variable': 'secondary_variable',
            'variables': 'variables',
            'u_variable': 'u_variable',
            'v_variable': 'v_variable',
            'pressure_variable': 'pressure_variable',
            'primary_colormap': 'primary_colormap',
            'secondary_colormap': 'secondary_colormap',
            'transparency': 'transparency',
            'contour_overlay': 'contour_overlay'
        }
        
        for arg_name, config_attr in overlay_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
        
        if hasattr(args, 'variables') and args.variables:
            config_dict['variables'] = [v.strip() for v in args.variables.split(',')]
    
    def setup_logging(
        self,
        config: MPASConfig,
        log_file: Optional[str] = None
    ) -> MPASLogger:
        """
        Initialize and configure the logging system with appropriate verbosity levels determined by configuration flags for diagnostic output control. This method creates an MPASLogger instance with log level mappings that respect user preferences: quiet mode maps to ERROR level showing only critical failures, normal mode maps to INFO level displaying progress and status messages, and verbose mode maps to DEBUG level providing detailed diagnostic information including timing breakdowns and internal state. The logger supports dual output to both console streams and optional file persistence when a log file path is provided. The configured logger instance is stored as an instance attribute (self.logger) for subsequent access throughout the CLI execution pipeline. Returns the configured logger enabling immediate use for validation, processing, and analysis logging.

        Parameters:
            config (MPASConfig): Configuration object containing verbosity preferences through quiet and verbose boolean flags.
            log_file (Optional[str]): Optional absolute or relative path to persistent log file for long-term output retention, None enables console-only logging (default: None).

        Returns:
            MPASLogger: Fully configured logger instance ready for CLI operation logging with appropriate verbosity level and output destinations.
        """
        log_level = logging.INFO
        if hasattr(config, 'quiet') and config.quiet:
            log_level = logging.ERROR
        elif hasattr(config, 'verbose') and config.verbose:
            log_level = logging.DEBUG
        
        verbose = not (hasattr(config, 'quiet') and config.quiet)
        
        self.logger = MPASLogger(
            name="mpas-unified-cli",
            level=log_level,
            log_file=log_file,
            verbose=verbose
        )
        
        return self.logger
    
    def _validate_file_path(
        self,
        file_path: Optional[str],
        file_type: str,
        errors: List[str]
    ) -> None:
        """
        This method checks if the provided file path is not None and points to an existing file on the filesystem. If the file path is missing or the file does not exist, it appends descriptive error messages to the provided errors list indicating the specific issue with the file path. This validation step is crucial for ensuring that required input files such as grid files are correctly specified before proceeding with analysis execution.

        Parameters:
            file_path (Optional[str]): The file path to validate, which may be None if not specified by the user.
            file_type (str): A descriptive string indicating the type of file being validated (e.g., "Grid file") for use in error messages.
            errors (List[str]): A list to which error messages will be appended if validation checks fail, allowing for accumulation of multiple validation issues to be reported together.

        Returns:
            None
        
        """
        if not file_path:
            errors.append(f"{file_type} not specified")
        elif not Path(file_path).exists():
            errors.append(f"{file_type} not found: {file_path}")
    
    def _validate_directory_path(
        self,
        dir_path: Optional[str],
        dir_type: str,
        errors: List[str]
    ) -> bool:
        """
        Validate that a directory path is specified, exists, and is a directory. If the directory path is missing, does not exist, or is not a directory, it appends descriptive error messages to the provided errors list indicating the specific issue with the directory path. This validation step is essential for ensuring that required input data directories are correctly specified and accessible before attempting to discover data files for analysis. The method returns a boolean indicating whether the directory path passed all validation checks, allowing calling code to conditionally proceed with file discovery only if the directory is valid.

        Parameters:
            dir_path (Optional[str]): The directory path to validate, which may be None if not specified by the user.
            dir_type (str): A descriptive string indicating the type of directory being validated (e.g., "Data directory") for use in error messages.
            errors (List[str]): A list to which error messages will be appended if validation checks fail, allowing for accumulation of multiple validation issues to be reported together.

        Returns:
            bool: True if directory is valid, False otherwise.
        """
        if not dir_path:
            errors.append(f"{dir_type} not specified")
            return False
        elif not Path(dir_path).exists():
            errors.append(f"{dir_type} not found: {dir_path}")
            return False
        elif not Path(dir_path).is_dir():
            errors.append(f"{dir_type} path is not a directory: {dir_path}")
            return False
        return True
    
    def _find_data_files(
        self,
        data_path: Path
    ) -> List[Path]:
        """
        The method first checks for files matching the DIAG_GLOB pattern directly within the specified data_path. If no files are found, it checks for a 'diag' subdirectory and looks for DIAG_GLOB files there. If still no files are found, it repeats this process using the MPASOUT_GLOB pattern, first checking the base directory and then an 'mpasout' subdirectory. If no files are found in these specific locations, it performs a recursive search through all subdirectories for both patterns. This systematic approach ensures that common MPAS diagnostic file naming conventions and directory structures are accounted for when attempting to locate data files for analysis.

        Parameters:
            data_path (Path): The base directory path to search for MPAS data files.

        Returns:
            List[Path]: List of discovered data files.
        """
        data_files = list(data_path.glob(DIAG_GLOB))
        if data_files:
            return data_files
        
        if (data_path / 'diag').exists():
            data_files = list((data_path / 'diag').glob(DIAG_GLOB))
            if data_files:
                return data_files
        
        data_files = list(data_path.glob(MPASOUT_GLOB))
        if data_files:
            return data_files
        
        if (data_path / 'mpasout').exists():
            data_files = list((data_path / 'mpasout').glob(MPASOUT_GLOB))
            if data_files:
                return data_files
        
        return list(data_path.rglob(DIAG_GLOB)) + list(data_path.rglob(MPASOUT_GLOB))
    
    def _validate_coordinate_range(
        self,
        config: MPASConfig,
        min_attr: str,
        max_attr: str,
        coord_type: str,
        errors: List[str]
    ) -> None:
        """
        This method checks if both the minimum and maximum attributes for a given coordinate type (latitude or longitude) are present in the configuration. If both attributes are found, it retrieves their values and checks if the minimum value is less than the maximum value. If the minimum value is greater than or equal to the maximum value, it appends an error message to the provided errors list indicating that the specified coordinate range is invalid. This validation step is crucial for ensuring that spatial bounds specified for analysis are logically consistent and will not lead to errors during data processing or visualization stages.
        
        Parameters:
            config (MPASConfig): The configuration object containing the attributes to validate.
            min_attr (str): The name of the attribute representing the minimum value of the coordinate range (e.g., 'lat_min' or 'lon_min').
            max_attr (str): The name of the attribute representing the maximum value of the coordinate range (e.g., 'lat_max' or 'lon_max').
            coord_type (str): A descriptive string indicating the type of coordinate being validated (e.g., "latitude" or "longitude") for use in error messages.
            errors (List[str]): A list to which error messages will be appended if validation checks fail, allowing for accumulation of multiple validation issues to be reported together.

        Returns:
            None
        """
        if hasattr(config, min_attr) and hasattr(config, max_attr):
            min_val = getattr(config, min_attr)
            max_val = getattr(config, max_attr)
            if min_val >= max_val:
                errors.append(f"Invalid {coord_type} range: {min_attr} >= {max_attr}")
    
    def _validate_cross_section_params(
        self,
        config: MPASConfig,
        errors: List[str]
    ) -> None:
        """
        If the analysis type is identified as a cross-section workflow (cross, xsec, 3d, vertical), this method checks for the presence of required attributes defining the start and end coordinates of the cross-section path (start_lon, start_lat, end_lon, end_lat). If any of these attributes are missing or None, it appends descriptive error messages to the provided errors list indicating that the specific parameter is required for cross-section analysis. This validation step ensures that all necessary spatial parameters for defining a vertical slice through the atmosphere are specified before attempting to execute cross-section processing and visualization routines.

        Parameters:
            config (MPASConfig): The configuration object containing the attributes to validate.
            errors (List[str]): A list to which error messages will be appended if validation checks fail, allowing for accumulation of multiple validation issues to be reported together.

        Returns:            
            None
        """
        if hasattr(config, 'analysis_type'):
            if config.analysis_type in ['cross', 'xsec', '3d', 'vertical']:
                required_attrs = ['start_lon', 'start_lat', 'end_lon', 'end_lat']
                for attr in required_attrs:
                    if not hasattr(config, attr) or getattr(config, attr) is None:
                        errors.append(f"Cross-section analysis requires --{attr.replace('_', '-')}")
    
    def _report_validation_errors(
        self,
        errors: List[str]
    ) -> None:
        """
        Report validation errors through logger or console. If a logger is configured, it logs an error header followed by each individual error message. If no logger is available, it prints the error messages to the console. This method provides a centralized way to communicate all detected configuration issues to the user in a clear and organized manner, facilitating troubleshooting and correction of invalid configurations before analysis execution.

        Parameters:
            errors (List[str]): A list of error messages to report, typically accumulated during the validation process when checks fail.

        Returns:
            None        
        """
        error_header = "Configuration validation failed:"
        if self.logger:
            self.logger.error(error_header)
            for error in errors:
                self.logger.error(f"  - {error}")
        else:
            print(error_header)
            for error in errors:
                print(f"  - {error}")
    
    def validate_config(
        self,
        config: MPASConfig
    ) -> bool:
        """
        Perform comprehensive validation of configuration settings ensuring all required parameters are present and input files exist before analysis execution. This method systematically checks configuration completeness through multiple validation stages: verifying grid file specification and filesystem existence, confirming data directory specification with directory existence checks, discovering MPAS diagnostic files using standard glob patterns (diag and mpasout variants), validating spatial extent bounds for logical consistency (min < max), and performing analysis-specific validation such as requiring cross-section endpoint coordinates for vertical slice workflows. Validation errors are accumulated in a list and reported through the logger with detailed descriptive messages highlighting missing files, invalid parameter ranges, or incomplete specifications. The method uses DataValidator utility for systematic configuration checking following established validation patterns. Failed validation halts execution preventing wasted processing time on invalid configurations.

        Parameters:
            config (MPASConfig): Configuration object to validate containing all analysis settings, file paths, spatial bounds, and analysis-type-specific parameters.

        Returns:
            bool: True if all validation checks pass and analysis can safely proceed, False if any validation errors are detected requiring user correction.
        """
        validator = DataValidator()
        errors = []
        
        self._validate_file_path(config.grid_file, "Grid file", errors)
        
        is_valid_dir = self._validate_directory_path(config.data_dir, "Data directory", errors)
        
        if is_valid_dir:
            data_path = Path(config.data_dir)
            data_files = self._find_data_files(data_path)
            if not data_files:
                errors.append(f"No MPAS data files found in: {config.data_dir}")
        
        self._validate_coordinate_range(config, 'lat_min', 'lat_max', 'latitude', errors)
        self._validate_coordinate_range(config, 'lon_min', 'lon_max', 'longitude', errors)
        
        self._validate_cross_section_params(config, errors)
        
        if errors:
            self._report_validation_errors(errors)
            return False
        
        return True
    
    def _check_analysis_type_specified(
        self,
        config: MPASConfig
    ) -> bool:
        """
        If the analysis_type attribute is missing or None, it logs an error message indicating that no analysis type was specified and returns False. If the analysis type is present, it returns True allowing execution to proceed. This check ensures that the main analysis dispatcher has a valid analysis type to route execution to the appropriate specialized method.
        
        Parameters:
            config (MPASConfig): The configuration object containing the analysis_type attribute to check.

        Returns:
            bool: True if analysis type is specified, False otherwise.
        """
        if not hasattr(config, 'analysis_type') or not config.analysis_type:
            self._log_error("No analysis type specified")
            return False
        return True
    
    def _dispatch_analysis(
        self,
        analysis_type: Optional[str],
        config: MPASConfig
    ) -> Optional[bool]:
        """
        The method defines a mapping of analysis type aliases to their corresponding specialized analysis methods. It iterates through the mapping and checks if the provided analysis_type matches any of the defined aliases. If a match is found, it calls the corresponding analysis method with the configuration object and returns its result. If no match is found after checking all aliases, it returns None indicating that the analysis type is unknown. This dispatcher centralizes the routing logic for different analysis workflows based on user-specified analysis types in the configuration.

        Parameters:
            analysis_type (Optional[str]): The type of analysis to perform.
            config (MPASConfig): The configuration object for the analysis.

        Returns:
            Optional[bool]: Analysis result if type is recognized, None if unknown.
        """
        if not analysis_type:
            return None
        
        analysis_map = {
            ('precipitation', 'precip', 'rain'): self._run_precipitation_analysis,
            ('surface', 'surf', '2d'): self._run_surface_analysis,
            ('wind', 'vector', 'winds'): self._run_wind_analysis,
            ('cross', 'xsec', '3d', 'vertical'): self._run_cross_analysis,
            ('overlay', 'complex', 'multi', 'composite'): self._run_overlay_analysis,
        }
        
        for type_aliases, analysis_method in analysis_map.items():
            if analysis_type in type_aliases:
                return analysis_method(config)
        
        return None
    
    def _log_error(
        self,
        message: str,
        include_traceback: bool = False
    ) -> None:
        """
        If a logger is configured, it logs the error message and optionally includes the full traceback if include_traceback is True. If no logger is available, it prints the error message to the console. This method provides a centralized way to report errors encountered during analysis execution with optional detailed diagnostic information when verbose mode is enabled.
        
        Parameters:
            message (str): Error message to log.
            include_traceback (bool): Whether to include full traceback.
        
        Returns:
            None
        """
        if self.logger:
            self.logger.error(message)
            if include_traceback:
                import traceback
                self.logger.error(traceback.format_exc())
    
    def _print_performance_summary(
        self,
        config: MPASConfig
    ) -> None:
        """
        If a logger is configured and the configuration has a verbose attribute set to True, it calls the print_summary method of the PerformanceMonitor instance to display detailed timing statistics for each stage of the analysis pipeline. This provides users with insights into where time is being spent during execution, which can be valuable for performance tuning and understanding the computational cost of different analysis steps.

        Parameters:
            config (MPASConfig): Configuration object containing verbosity preferences.

        Returns:
            None
        """
        if self.logger and hasattr(config, 'verbose') and config.verbose:
            if self.perf_monitor is not None:
                self.perf_monitor.print_summary()
    
    def run_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        Execute the configured MPAS analysis workflow with comprehensive error handling, performance monitoring, and graceful failure recovery mechanisms. This main dispatcher method determines the analysis type from configuration, initializes a PerformanceMonitor instance for timing instrumentation, and routes execution to the appropriate specialized analysis method based on the analysis_type attribute (precipitation, surface, wind, cross-section, or overlay). It wraps the entire analysis execution in a performance timer context manager to track total pipeline runtime including data loading, processing, visualization, and file I/O operations. The method implements robust error handling through three exception classes: KeyboardInterrupt for graceful user cancellation allowing clean shutdown, specific exception types for known failure modes, and generic Exception catch-all for unexpected errors with optional verbose traceback logging. After successful completion in verbose mode, the method prints detailed performance statistics showing timing breakdowns for each pipeline stage.

        Parameters:
            config (MPASConfig): Configuration object specifying analysis type, input file locations, visualization parameters, output settings, and processing options.

        Returns:
            bool: True if the analysis pipeline executed successfully through completion, False if errors occurred during execution or user interrupted the process.
        """
        try:
            self.perf_monitor = PerformanceMonitor()
            
            if not self._check_analysis_type_specified(config):
                return False
            
            analysis_type = config.analysis_type
            
            with self.perf_monitor.timer("Total analysis"):
                success = self._dispatch_analysis(analysis_type, config)
                
                if success is None:
                    self._log_error(f"Unknown analysis type: {analysis_type}")
                    return False
            
            self._print_performance_summary(config)
            
            return success
            
        except KeyboardInterrupt:
            self._log_error("Analysis interrupted by user")
            return False
        except Exception as e:
            include_trace = hasattr(config, 'verbose') and config.verbose
            self._log_error(f"Analysis failed: {e}", include_traceback=include_trace)
            return False
    
    def _run_precipitation_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        Execute precipitation accumulation analysis workflow orchestrating data loading, diagnostic computation, and map generation for rainfall visualization. This method manages the complete precipitation visualization pipeline by initializing MPAS2DProcessor to load diagnostic files containing precipitation variables, creating MPASPrecipitationPlotter with specified figure dimensions and DPI settings, and generating either single-timestep accumulation maps or batch-mode time series depending on configuration. It supports both serial execution for single-core processing and parallel MPI-based execution when enabled through configuration flags, automatically selecting the appropriate execution strategy using ParallelPrecipitationProcessor for distributed workloads. The method creates output directories as needed, extracts spatial coordinates for the specified variable, computes precipitation differences using PrecipitationDiagnostics with configurable accumulation periods, applies spatial bounds and interpolation settings, and saves visualizations in requested formats. Successfully generated plot counts are logged for user feedback.

        Parameters:
            config (MPASConfig): Configuration object containing all required precipitation parameters including grid file path, data directory, variable name (rainc/rainnc), accumulation period (a01h through a24h), plot type, spatial bounds, and output settings.

        Returns:
            bool: True if precipitation analysis workflow completes successfully without errors, False if processing or plotting failures occur during execution.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG
        with self.perf_monitor.timer("Precipitation analysis"):
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_2d_data(config.data_dir)
            dataset = processor.dataset
            
            plotter = MPASPrecipitationPlotter(
                figsize=config.figure_size,
                dpi=config.dpi
            )
            
            os.makedirs(config.output_dir, exist_ok=True)
            
            if config.batch_mode:
                use_parallel = getattr(config, 'parallel', False)
                
                if use_parallel:
                    if self.logger:
                        self.logger.info("Using parallel processing for batch precipitation analysis")
                    created_files = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                        processor, config.output_dir,
                        config.lon_min, config.lon_max,
                        config.lat_min, config.lat_max,
                        var_name=config.variable,
                        accum_period=config.accumulation_period,
                        plot_type=getattr(config, 'plot_type', 'scatter'),
                        grid_resolution=getattr(config, 'grid_resolution', None),
                        formats=config.output_formats or ['png'],
                        n_processes=config.workers
                    )
                else:
                    created_files = plotter.create_batch_precipitation_maps(
                        processor, config.output_dir,
                        config.lon_min, config.lon_max,
                        config.lat_min, config.lat_max,
                        var_name=config.variable,
                        accum_period=config.accumulation_period,
                        plot_type=getattr(config, 'plot_type', 'scatter'),
                        grid_resolution=getattr(config, 'grid_resolution', None),
                        formats=config.output_formats or ['png']
                    )
                
                if self.logger and created_files:
                    self.logger.info(f"Created {len(created_files)} precipitation maps")
            else:
                lon, lat = processor.extract_2d_coordinates_for_variable(config.variable)
                
                precip_diag = PrecipitationDiagnostics(verbose=config.verbose)
                precip_data = precip_diag.compute_precipitation_difference(
                    dataset, config.time_index, config.variable, config.accumulation_period, 
                    data_type=getattr(processor, 'data_type', 'UXarray')
                )
                
                fig, ax = plotter.create_precipitation_map(
                    lon, lat, precip_data.values,
                    config.lon_min, config.lon_max,
                    config.lat_min, config.lat_max,
                    title=config.title or f"Precipitation: {config.variable}",
                    accum_period=config.accumulation_period,
                    plot_type=getattr(config, 'plot_type', 'scatter'),
                    grid_resolution=getattr(config, 'grid_resolution', None)
                )
                
                output_path = config.output or os.path.join(
                    config.output_dir,
                    f"mpas_precipitation_{config.variable}_{config.time_index:03d}"
                )
                
                plotter.save_plot(output_path, formats=config.output_formats or ['png'])
                plotter.close_plot()
                
                if self.logger:
                    self.logger.info(f"Precipitation plot saved: {output_path}")
        
        return True
    
    def _run_surface_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        Execute surface variable visualization workflow for 2D meteorological scalar fields orchestrating data retrieval, interpolation, and contour generation. This method manages the complete surface analysis pipeline by loading 2D diagnostic data through MPAS2DProcessor initialization, creating MPASSurfacePlotter with specified figure dimensions and resolution settings, and generating either single-timestep field visualizations or batch-mode time series depending on configuration flags. It supports both serial execution for single-core workflows and parallel MPI-based execution when enabled, automatically selecting ParallelSurfaceProcessor for distributed batch processing across timesteps. The method handles variables like temperature (t2m), pressure (mslp), and moisture (q2) on MPAS unstructured grids, creates output directories as needed, extracts spatial coordinates and variable data, applies spatial bounds and gridding options for interpolation to regular grids, and saves visualizations in requested output formats. Generated surface map counts are logged for user confirmation.

        Parameters:
            config (MPASConfig): Configuration object with surface analysis parameters including grid file path, data directory location, variable name selection, plot type (scatter/contour/contourf/pcolormesh), gridding resolution specifications, spatial bounds, and output preferences.

        Returns:
            bool: True if surface analysis workflow completes successfully through all processing stages, False if data loading, processing, or plotting errors occur.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG
        with self.perf_monitor.timer("Surface analysis"):
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_2d_data(config.data_dir)
            dataset = processor.dataset
            
            plotter = MPASSurfacePlotter(
                figsize=config.figure_size,
                dpi=config.dpi
            )
            
            os.makedirs(config.output_dir, exist_ok=True)
            
            if config.batch_mode:
                use_parallel = getattr(config, 'parallel', False)
                
                if use_parallel:
                    if self.logger:
                        self.logger.info("Using parallel processing for batch surface analysis")
                    created_files = ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                        processor, config.output_dir,
                        config.lon_min, config.lon_max,
                        config.lat_min, config.lat_max,
                        var_name=config.variable,
                        plot_type=config.plot_type,
                        formats=config.output_formats or ['png'],
                        grid_resolution=getattr(config, 'grid_resolution', None),
                        n_processes=config.workers
                    )
                else:
                    created_files = plotter.create_batch_surface_maps(
                        processor, config.output_dir,
                        config.lon_min, config.lon_max,
                        config.lat_min, config.lat_max,
                        var_name=config.variable,
                        plot_type=config.plot_type,
                        formats=config.output_formats or ['png'],
                        grid_resolution=getattr(config, 'grid_resolution', None)
                    )
                
                if self.logger and created_files:
                    self.logger.info(f"Created {len(created_files)} surface maps")
            else:
                var_data = processor.get_2d_variable_data(config.variable, config.time_index)
                lon, lat = processor.extract_2d_coordinates_for_variable(config.variable, var_data)
                
                fig, ax = plotter.create_surface_map(
                    lon, lat, var_data.values,
                    config.variable,
                    config.lon_min, config.lon_max,
                    config.lat_min, config.lat_max,
                    title=config.title,
                    plot_type=config.plot_type,
                    colormap=config.colormap if config.colormap != 'default' else None,
                    clim_min=config.clim_min,
                    clim_max=config.clim_max,
                    data_array=var_data  
                )
                
                output_path = config.output or os.path.join(
                    config.output_dir,
                    f"mpas_surface_{config.variable}_{config.plot_type}_{config.time_index:03d}"
                )
                
                plotter.save_plot(output_path, formats=config.output_formats or ['png'])
                plotter.close_plot()
                
                if self.logger:
                    self.logger.info(f"Surface plot saved: {output_path}")
        
        return True
    
    def _run_wind_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        Execute wind vector analysis workflow generating barb, arrow, or streamline visualizations from horizontal wind components on MPAS unstructured grids. This method orchestrates the wind visualization pipeline by loading 2D wind component data (u and v) through MPAS2DProcessor initialization, creating MPASWindPlotter with specified figure dimensions and resolution settings, and producing either single-timestep vector plots or batch-mode time series depending on configuration. It handles wind field visualization at specified atmospheric levels (surface, 850mb, 500mb, etc.) with configurable subsampling to control vector density for clarity, scaling factors to adjust arrow and barb sizes for optimal visibility, and optional background field overlays showing wind speed magnitude as filled contours. The method supports multiple vector representation types (barbs for meteorological convention, arrows for directional flow, streamlines for continuous trajectories), manages output directory creation, extracts spatial coordinates and wind component data, applies optional regridding to regular grids with configurable interpolation methods, and saves visualizations in requested formats.

        Parameters:
            config (MPASConfig): Configuration object containing wind analysis settings including grid file path, data directory, u/v variable names, wind level description, plot type (barbs/arrows/streamlines), subsampling factor, vector scale, background field toggles, regridding options, and output preferences.

        Returns:
            bool: True if wind visualization workflow completes successfully through all processing stages, False if data loading, vector computation, or plotting errors occur.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG
        with self.perf_monitor.timer("Wind analysis"):
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_2d_data(config.data_dir)
            dataset = processor.dataset
            
            plotter = MPASWindPlotter(
                figsize=config.figure_size,
                dpi=config.dpi
            )
            
            os.makedirs(config.output_dir, exist_ok=True)
            
            if config.batch_mode:
                use_parallel = getattr(config, 'parallel', False)
                
                if use_parallel:
                    if self.logger:
                        self.logger.info("Using parallel processing for batch wind analysis")
                    created_files = ParallelWindProcessor.create_batch_wind_plots_parallel(
                        processor, config.output_dir,
                        config.lon_min, config.lon_max,
                        config.lat_min, config.lat_max,
                        u_variable=config.u_variable,
                        v_variable=config.v_variable,
                        plot_type=config.wind_plot_type,
                        formats=config.output_formats or ['png'],
                        subsample=config.subsample_factor,
                        scale=config.wind_scale,
                        show_background=config.show_background,
                        grid_resolution=getattr(config, 'grid_resolution', None),
                        regrid_method=getattr(config, 'regrid_method', 'linear'),
                        n_processes=config.workers
                    )
                else:
                    created_files = plotter.create_batch_wind_plots(  
                        processor, config.output_dir,
                        config.lon_min, config.lon_max,
                        config.lat_min, config.lat_max,
                        u_variable=config.u_variable,
                        v_variable=config.v_variable,
                        plot_type=config.wind_plot_type,
                        formats=config.output_formats,
                        subsample=config.subsample_factor,
                        scale=config.wind_scale,
                        show_background=config.show_background,
                        grid_resolution=getattr(config, 'grid_resolution', None),
                        regrid_method=getattr(config, 'regrid_method', 'linear')
                    )
                
                if self.logger and created_files:
                    self.logger.info(f"Created {len(created_files)} wind plots")
            else:
                u_data = processor.get_2d_variable_data(config.u_variable, config.time_index)
                v_data = processor.get_2d_variable_data(config.v_variable, config.time_index)
                lon, lat = processor.extract_2d_coordinates_for_variable(config.u_variable, u_data)
                
                fig, ax = plotter.create_wind_plot(
                    lon, lat, u_data.values, v_data.values,
                    config.lon_min, config.lon_max,
                    config.lat_min, config.lat_max,
                    wind_level=config.wind_level,
                    plot_type=config.wind_plot_type,
                    title=config.title,
                    subsample=config.subsample_factor,
                    scale=config.wind_scale,
                    show_background=config.show_background,
                    grid_resolution=getattr(config, 'grid_resolution', None),
                    regrid_method=getattr(config, 'regrid_method', 'linear')
                )
                
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > config.time_index:
                    time_end = pd.Timestamp(processor.dataset.Time.values[config.time_index]).to_pydatetime()
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_str = f"t{config.time_index:03d}"
                
                output_path = config.output or os.path.join(
                    config.output_dir,
                    f"mpas_wind_{config.u_variable}_{config.v_variable}_{config.wind_plot_type}_valid_{time_str}"
                )
                
                plotter.save_plot(output_path, formats=config.output_formats or ['png'])
                plotter.close_plot()
                
                if self.logger:
                    self.logger.info(f"Wind plot saved: {output_path}")
        
        return True
    
    def _validate_cross_section_coordinates(
        self,
        config: MPASConfig
    ) -> None:
        """
        This method asserts that both start and end longitude/latitude pairs are provided on the configuration object. It is intended as an early guard to prevent attempting interpolation or cross-section generation with incomplete inputs. Callers should handle AssertionError to provide user-friendly diagnostics.

        Parameters:
            config (MPASConfig): Configuration object expected to contain
                `start_lon`, `start_lat`, `end_lon`, and `end_lat` attributes.

        Returns:
            None: Raises AssertionError if any of the required coordinates are missing.
        """
        assert config.start_lon is not None and config.start_lat is not None, \
            "Cross-section start coordinates must be specified"
        assert config.end_lon is not None and config.end_lat is not None, \
            "Cross-section end coordinates must be specified"
    
    def _extract_cross_section_params(
        self,
        config: MPASConfig
    ) -> Dict[str, Any]:
        """
        The returned dictionary centralizes parameter names used by cross-section plotting routines (start/end points, variable name, vertical coordinate, and interpolation settings) so subsequent functions can remain generic.

        Parameters:
            config (MPASConfig): Configuration object containing cross-section
                settings such as `variable`, `start_lon`, `start_lat`, `end_lon`,
                `end_lat`, `vertical_coord`, `num_points`, and `max_height`.

        Returns:
            Dict[str, Any]: Dictionary with keys `var_name`, `start_point`,
                `end_point`, `vertical_coord`, `num_points`, and `max_height`.
        """
        return {
            'var_name': config.variable,
            'start_point': (config.start_lon, config.start_lat),
            'end_point': (config.end_lon, config.end_lat),
            'vertical_coord': config.vertical_coord,
            'num_points': config.num_points,
            'max_height': config.max_height,
        }
    
    def _run_batch_cross_sections(
        self,
        processor: 'MPAS3DProcessor',
        plotter: 'MPASVerticalCrossSectionPlotter',
        config: MPASConfig,
        params: Dict[str, Any]
    ) -> Optional[List[str]]:
        """
        This helper decides between parallel and serial creation of cross-section plots based on the configuration and logs the chosen mode. It delegates the actual plotting to either the `ParallelCrossSectionProcessor` or the provided `plotter` instance depending on the execution mode.

        Parameters:
            processor (MPAS3DProcessor): Processor that provides 3D data access.
            plotter (MPASVerticalCrossSectionPlotter): Plotter used for serial plotting.
            config (MPASConfig): Configuration object controlling batch behavior
                (e.g., `parallel`, `output_dir`, `workers`, `output_formats`).
            params (Dict[str, Any]): Cross-section parameters produced by
                `_extract_cross_section_params`.

        Returns:
            Optional[List[str]]: List of created file paths, or None when no files
                were created or an error prevented generation.
        """
        use_parallel = getattr(config, 'parallel', False)
        
        if use_parallel:
            if self.logger:
                self.logger.info("Running parallel batch cross-section analysis...")
            return ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                mpas_3d_processor=processor,
                output_dir=config.output_dir,
                formats=config.output_formats or ['png'],
                n_processes=config.workers,
                **params
            )
        else:
            if self.logger:
                self.logger.info("Running serial batch cross-section analysis...")
            return plotter.create_batch_cross_section_plots(
                mpas_3d_processor=processor,
                output_dir=config.output_dir,
                formats=config.output_formats or ['png'],
                **params
            )
    
    def _run_single_cross_section(
        self,
        processor: 'MPAS3DProcessor',
        plotter: 'MPASVerticalCrossSectionPlotter',
        config: MPASConfig,
        params: Dict[str, Any]
    ) -> None:
        """
        The function constructs a title, invokes the plotter to create the figure, saves the output using the configured formats and path, and closes the plotter to release resources. It is intended for single-timestep use and does not perform batch iteration.

        Parameters:
            processor (MPAS3DProcessor): Processor providing 3D atmospheric data.
            plotter (MPASVerticalCrossSectionPlotter): Plotter instance to render the figure.
            config (MPASConfig): Configuration with `time_index`, `title`, `output_dir`,
                and `output_formats` controlling the output file naming and format.
            params (Dict[str, Any]): Cross-section parameters produced by
                `_extract_cross_section_params`.

        Returns:
            None
        """
        fig, ax = plotter.create_vertical_cross_section(
            mpas_3d_processor=processor,
            time_index=config.time_index,
            title=config.title or f"Vertical Cross-Section: {config.variable}",
            colormap=config.colormap if config.colormap != 'default' else None,
            **params
        )
        
        output_path = config.output or os.path.join(
            config.output_dir,
            f"mpas_cross_section_{config.variable}_{config.time_index:03d}"
        )
        
        plotter.save_plot(output_path, formats=config.output_formats or ['png'])
        plotter.close_plot()
        
        if self.logger:
            self.logger.info(f"Cross-section plot saved: {output_path}")
    
    def _log_created_files(
        self,
        created_files: Optional[List[str]],
        file_type: str = "plots"
    ) -> None:
        """
        This small utility centralizes how created file counts are reported so callers do not need to check the logger or list truthiness each time. It is tolerant of a `None` value for `created_files`.

        Parameters:
            created_files (Optional[List[str]]): List of paths created, or None.
            file_type (str): Human-readable label for the type of files created.

        Returns:
            None
        """
        if self.logger and created_files:
            self.logger.info(f"Created {len(created_files)} {file_type}")
    
    def _run_cross_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        This method loads 3D data via `MPAS3DProcessor`, prepares a `MPASVerticalCrossSectionPlotter`, validates inputs, and then either runs a batch or single cross-section creation depending on `config.batch_mode`. It supports interpolation along the requested great-circle path and handles output directory creation and file saving.

        Parameters:
            config (MPASConfig): Configuration object providing grid file, data
                directory, variable selection, start/end coordinates, vertical
                coordinate, interpolation resolution, output directory, and
                batch/single mode selection.

        Returns:
            bool: True when the workflow completes successfully, False on error.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG
        with self.perf_monitor.timer("Cross-section analysis"):
            processor = MPAS3DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_3d_data(config.data_dir)
            
            plotter = MPASVerticalCrossSectionPlotter(
                figsize=config.figure_size,
                dpi=config.dpi
            )
            
            os.makedirs(config.output_dir, exist_ok=True)
            
            self._validate_cross_section_coordinates(config)
            params = self._extract_cross_section_params(config)
            
            if config.batch_mode:
                created_files = self._run_batch_cross_sections(processor, plotter, config, params)
                self._log_created_files(created_files, "cross-section plots")
            else:
                self._run_single_cross_section(processor, plotter, config, params)
        
        return True
    
    def _run_overlay_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        The function loads 2D diagnostic data via `MPAS2DProcessor` and selects appropriate plotter components according to `config.overlay_type`. Current implementations provide placeholder flows for common combinations (e.g., precipitation + wind, temperature + pressure) and perform output saving.

        Parameters:
            config (MPASConfig): Configuration containing overlay type and
                settings such as variable lists, colormaps, transparency, and
                output preferences.

        Returns:
            bool: True on successful completion (placeholder behavior included),
                False if initialization or configuration validation fails.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG
        with self.perf_monitor.timer("Overlay analysis"):
            if self.logger:
                self.logger.info(f"Running {config.overlay_type} overlay analysis...")
                self.logger.warning("Overlay analysis is not fully implemented yet")
            
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_2d_data(config.data_dir)
            dataset = processor.dataset
            
            os.makedirs(config.output_dir, exist_ok=True)
            
            if config.overlay_type == 'precip_wind':
                precip_plotter = MPASPrecipitationPlotter(figsize=config.figure_size, dpi=config.dpi)
                wind_plotter = MPASWindPlotter(figsize=config.figure_size, dpi=config.dpi)
                
                if self.logger:
                    self.logger.info("Creating precipitation + wind overlay")
            
            elif config.overlay_type == 'temp_pressure':
                surface_plotter = MPASSurfacePlotter(figsize=config.figure_size, dpi=config.dpi)
                if self.logger:
                    self.logger.info("Creating temperature + pressure overlay")
            
            if self.logger:
                self.logger.info("Overlay analysis completed (placeholder implementation)")
        
        return True
    
    def _parse_args_with_fallback(
        self,
        parser: argparse.ArgumentParser
    ) -> argparse.Namespace:
        """
        Some Python versions or environments may not support `parse_intermixed_args`. This helper attempts the intermixed parse first and falls back to a manual reordering strategy that separates global options from subcommands.

        Parameters:
            parser (argparse.ArgumentParser): The argument parser to use for parsing.

        Returns:
            argparse.Namespace: The parsed arguments namespace.
        """
        try:
            return parser.parse_intermixed_args()
        except Exception:
            return self._parse_args_with_reordering(parser)
    
    def _parse_args_with_reordering(
        self,
        parser: argparse.ArgumentParser
    ) -> argparse.Namespace:
        """
        This function is a robust fallback when users provide global flags after subcommands; it extracts global options and recombines the argv list so the parser can interpret the inputs correctly.

        Parameters:
            parser (argparse.ArgumentParser): The parser used to parse the
                reordered argument list.

        Returns:
            argparse.Namespace: Parsed arguments after reordering.
        """
        argv = sys.argv[1:]
        globals_with_value = {'--config', '--log-file'}
        globals_no_value = {'--verbose', '-v', '--quiet', '-q', '--version'}
        
        front = []
        rest = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in globals_no_value:
                front.append(a)
                i += 1
            elif any(a.startswith(g + "=") for g in globals_with_value):
                front.append(a)
                i += 1
            elif a in globals_with_value:
                val = argv[i + 1] if i + 1 < len(argv) else ''
                front.extend([a, val])
                i += 2
            else:
                rest.append(a)
                i += 1
        
        try:
            return parser.parse_args(front + rest)
        except Exception:
            return parser.parse_args()
    
    def _load_and_merge_config(
        self,
        args: argparse.Namespace
    ) -> MPASConfig:
        """
        When the `--config` option is used, this helper loads the YAML file into an `MPASConfig` instance and then overlays any non-None CLI arguments so that command-line settings take precedence. If no config file is provided, the CLI-derived configuration is returned directly.

        Parameters:
            args (argparse.Namespace): Parsed CLI arguments which may include
                a `config` attribute pointing to a YAML file.

        Returns:
            MPASConfig: The final merged configuration instance used by the CLI.
        """
        if hasattr(args, 'config') and args.config:
            config = MPASConfig.load_from_file(args.config)
            cli_config = self.parse_args_to_config(args)
            for key, value in cli_config.to_dict().items():
                if value is not None:
                    setattr(config, key, value)
            return config
        else:
            return self.parse_args_to_config(args)
    
    def _print_verbose_output(
        self,
        config: MPASConfig
    ) -> None:
        """
        This convenience method calls `_print_system_info` and `_print_config_summary` when the configuration requests verbose logging, providing helpful diagnostics for debugging or reproducibility. It is a no-op when verbose mode is not set.

        Parameters:
            config (MPASConfig): Configuration object potentially containing
                a `verbose` boolean attribute.

        Returns:
            None
        """
        if hasattr(config, 'verbose') and config.verbose:
            self._print_system_info()
            self._print_config_summary()
    
    def _handle_main_exception(
        self,
        error: Exception
    ) -> int:
        """
        The helper prints the exception traceback to the configured logger when available; otherwise it falls back to printing to stdout. The function returns a non-zero exit code suitable for use as a process exit value.

        Parameters:
            error (Exception): The caught exception instance to report.

        Returns:
            int: Exit code (1) indicating an error condition.
        """
        import traceback
        if self.logger:
            self.logger.error(f"Unexpected error: {error}")
            self.logger.error(traceback.format_exc())
        else:
            print(f"Error: {error}")
            traceback.print_exc()
        return 1
    
    def main(self) -> int:
        """
        This method builds the argument parser, handles intermixed argument parsing, merges CLI options with an optional configuration file, configures logging, validates the final configuration, and then invokes the requested analysis workflow. It includes robust exception handling to return appropriate Unix-style exit codes for success, error, and user interruption.

        Parameters:
            None

        Returns:
            int: Exit code where 0 indicates success, 1 indicates failure, and
                130 indicates user interruption (KeyboardInterrupt).
        """
        try:
            parser = self.create_main_parser()
            args = self._parse_args_with_fallback(parser)
            
            self.config = self._load_and_merge_config(args)
            
            log_file = getattr(args, 'log_file', None)
            self.setup_logging(self.config, log_file)
            
            if not self.validate_config(self.config):
                return 1
            
            self._print_verbose_output(self.config)
            
            success = self.run_analysis(self.config)
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
            return 130
        except Exception as e:
            return self._handle_main_exception(e)
    
    def _print_system_info(self) -> None:
        """
        The method reports Python version, platform identifier, current working directory, and—when available—free memory as reported by `psutil`. Output is written to the configured logger and formatted for readability.

        Parameters:
            None

        Returns:
            None
        """
        if self.logger:
            self.logger.info("=== System Information ===")
            self.logger.info(f"Python version: {sys.version}")
            self.logger.info(f"Platform: {sys.platform}")
            self.logger.info(f"Current working directory: {os.getcwd()}")
            try:
                import psutil
                memory_gb = psutil.virtual_memory().available / (1024**3)
                self.logger.info(f"Available memory: {memory_gb:.1f} GB")
            except ImportError:
                pass
            self.logger.info("=" * 30)
    
    def _print_config_summary(self) -> None:
        """
        Only non-None configuration values are included so the summary focuses on actively set options. The output is sent to the configured logger and is suitable for inclusion in verbose diagnostic logs. If the configuration object does not have a `to_dict` method, this function will fail gracefully without printing a summary. 

        Parameters:
            None

        Returns:
            None
        """
        if self.logger and self.config:
            self.logger.info("\n=== Configuration Summary ===")
            for key, value in self.config.to_dict().items():
                if value is not None:
                    self.logger.info(f"  {key}: {value}")
            self.logger.info("=" * 30)


def main() -> int:
    """
    Module-level entry point providing convenient programmatic access to the unified CLI functionality from command line interfaces or Python scripts. This function instantiates the MPASUnifiedCLI class and delegates complete execution flow to its main method, encapsulating argument parsing, configuration file loading, settings validation, logging initialization, and analysis execution in a single convenient call. It serves as the primary interface registered as the 'mpasdiag' console script entry point through setuptools configuration and returns standard Unix exit codes for integration with shell scripts, batch processing systems, and continuous integration pipelines. The function can be invoked directly from Python scripts for programmatic analysis execution or called indirectly through the command-line mpasdiag command installed by setuptools console_scripts.

    Parameters:
        None

    Returns:
        int: Unix exit status code where 0 indicates successful analysis completion with all outputs generated, and non-zero values indicate various failure conditions (1 for errors, 130 for interruption).
    """
    cli = MPASUnifiedCLI()
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())