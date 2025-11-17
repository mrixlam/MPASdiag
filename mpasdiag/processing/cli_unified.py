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
    ParallelCrossSectionProcessor,
    auto_batch_processor
)

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
        Initialize the unified CLI instance with placeholder components for lazy configuration. This constructor creates the CLI object without immediately initializing heavy components like loggers or configuration objects to enable fast startup. Placeholders are set for the logger, performance monitor, and configuration object which will be properly initialized when parse_args_to_config is called. This design pattern allows the CLI to be instantiated quickly and configured flexibly based on runtime parameters. The instance maintains state across the argument parsing, validation, and execution pipeline.

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
        Construct the main argument parser with subcommands for all MPAS visualization types. This method initializes the top-level ArgumentParser with program metadata and creates subparsers for each analysis type. It configures the parser with usage examples and calls specialized methods to add argument groups for each subcommand. The resulting parser provides a unified command-line interface with consistent argument handling across all visualization workflows.

        Parameters:
            None

        Returns:
            argparse.ArgumentParser: Configured main parser with all subcommands and argument groups registered for precipitation, surface, wind, cross-section, and overlay analysis types.
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
    
    def _add_common_arguments(self, parser: argparse.ArgumentParser, required_grid: bool = True) -> None:
        """
        Add common command-line arguments shared across multiple analysis type subparsers for consistent interface. This method defines and adds argument groups for file paths, spatial domains, time selection, output configuration, and processing options. The arguments include grid file, data directory, geographic extent, time index specification, batch processing flags, and performance tuning parameters. Making the grid file requirement optional allows flexibility for different analysis workflows. This centralized argument definition ensures consistency across subcommands and reduces code duplication.

        Parameters:
            parser (argparse.ArgumentParser): Argument parser or subparser object to which common arguments will be added.
            required_grid (bool): Flag indicating whether grid file argument should be required or optional (default: True).

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
    
    def _add_precipitation_parser(self, subparsers) -> None:
        """
        Create and register the precipitation subparser for accumulation analysis workflows. This method adds a new subparser to handle precipitation-specific command-line arguments including variable selection, accumulation periods, and unit specifications. The subparser includes common arguments plus precipitation-specific options for analyzing convective and non-convective rainfall. It provides aliases for convenient command invocation and includes usage examples in the help epilog.

        Parameters:
            subparsers: The parent subparsers object returned by ArgumentParser.add_subparsers() to which the precipitation parser is added.

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
        precip_group.add_argument('--threshold', type=float,
                                help='Precipitation threshold for highlighting')
        precip_group.add_argument('--units', type=str, choices=['mm', 'in', 'cm'],
                                default='mm', help='Output units (default: mm)')
    
    def _add_surface_parser(self, subparsers) -> None:
        """
        Create and register the surface subparser for 2D meteorological field visualization workflows. This method adds a new subparser to handle surface-specific command-line arguments including variable selection, plot types, and gridding options. The subparser supports multiple visualization styles such as scatter plots, contour lines, filled contours, and pcolormesh for variables like temperature, pressure, and moisture. It provides command aliases and includes usage examples in the help documentation.

        Parameters:
            subparsers: The parent subparsers object returned by ArgumentParser.add_subparsers() to which the surface parser is added.

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
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable t2m --plot-type contour --grid-resolution-deg 0.1
            """)
        )
        
        self._add_common_arguments(parser)
        
        surf_group = parser.add_argument_group('Surface Variable Options')
        surf_group.add_argument('--variable', '--var', type=str, required=True,
                              help='Surface variable name (e.g., t2m, mslp, q2, u10, v10)')
        surf_group.add_argument('--plot-type', type=str, default='scatter',
                              choices=['scatter', 'contour', 'filled_contour', 'pcolormesh'],
                              help='Plot type (default: scatter)')
        surf_group.add_argument('--grid-resolution', type=int,
                              help='Grid resolution (number of points per axis) for interpolation')
        surf_group.add_argument('--grid-resolution-deg', type=float,
                              help='Grid resolution in degrees (e.g., 0.1 for 0.1° grid)')
        surf_group.add_argument('--interpolation', type=str, default='linear',
                              choices=['linear', 'cubic', 'nearest'],
                              help='Interpolation method for contour plots (default: linear)')
        surf_group.add_argument('--contour-levels', type=int, default=15,
                              help='Number of contour levels (default: 15)')
    
    def _add_wind_parser(self, subparsers) -> None:
        """
        Create and register the wind subparser for vector field visualization workflows. This method adds a new subparser to handle wind-specific command-line arguments including u/v component variables, plot types, and vector styling options. The subparser supports multiple visualization styles such as barbs, arrows, and streamlines with configurable subsampling and scaling. It provides options to render background wind speed fields and includes usage examples for different wind levels.

        Parameters:
            subparsers: The parent subparsers object returned by ArgumentParser.add_subparsers() to which the wind parser is added.

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
    
    def _add_cross_parser(self, subparsers) -> None:
        """
        Create and register the cross-section subparser for 3D vertical slice visualization workflows. This method adds a new subparser to handle cross-section-specific arguments including start/end coordinates, vertical coordinate systems, and interpolation settings. The subparser supports pressure, height, and model-level vertical coordinates with configurable path resolution and maximum height limits. It provides command aliases and includes usage examples for atmospheric cross-sections.

        Parameters:
            subparsers: The parent subparsers object returned by ArgumentParser.add_subparsers() to which the cross-section parser is added.

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
        xsec_group.add_argument('--plot-style', type=str, default='filled_contour',
                              choices=['filled_contour', 'contour', 'pcolormesh'],
                              help='Cross-section plot style (default: filled_contour)')
        xsec_group.add_argument('--extend', type=str, default='both',
                              choices=['both', 'min', 'max', 'neither'],
                              help='Colorbar extension (default: both)')
    
    def _add_overlay_parser(self, subparsers) -> None:
        """
        Create and register the overlay subparser for multi-variable composite visualization workflows. This method adds a new subparser to handle overlay-specific arguments including overlay type selection, multiple variable specifications, and transparency controls. The subparser supports various overlay combinations such as precipitation with wind, temperature with pressure, and custom multi-variable composites. It provides command aliases and includes usage examples for complex overlay analyses.

        Parameters:
            subparsers: The parent subparsers object returned by ArgumentParser.add_subparsers() to which the overlay parser is added.

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
    
    def parse_args_to_config(self, args: argparse.Namespace) -> MPASConfig:
        """
        Transform parsed command-line arguments into a comprehensive MPASConfig object for analysis execution. This method maps argument namespace attributes to configuration dictionary keys through both common and analysis-specific transformations. It processes shared parameters like grid files and spatial bounds first, then delegates to specialized mapping methods based on the analysis command type. The method enables automatic batch mode when time ranges are specified and returns a fully-initialized MPASConfig object ready for validation.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments from argparse containing all user inputs and command options.

        Returns:
            MPASConfig: Configuration object populated with validated settings for the requested analysis type and ready for pipeline execution.
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
    
    def _map_precipitation_args(self, args: argparse.Namespace, config_dict: Dict[str, Any]) -> None:
        """
        Map precipitation-specific command-line arguments into the configuration dictionary for accumulation analysis workflows. This method extracts precipitation-related parameters from the argparse namespace including variable names, accumulation periods, thresholds, and units. The mapping handles optional parameters gracefully by checking for attribute existence before assignment. This specialized mapping ensures that precipitation-specific options are properly translated from CLI argument names to internal configuration keys.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing precipitation-specific options from argparse.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with precipitation parameters in-place.

        Returns:
            None
        """
        precip_mapping = {
            'variable': 'variable',
            'accumulation': 'accumulation_period',
            'threshold': 'precip_threshold',
            'units': 'precip_units'
        }
        
        for arg_name, config_attr in precip_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_surface_args(self, args: argparse.Namespace, config_dict: Dict[str, Any]) -> None:
        """
        Map surface-specific command-line arguments into the configuration dictionary for 2D field plotting workflows. This method extracts surface analysis parameters from the argparse namespace including variable names, plot types, grid resolution settings, interpolation methods, and contour levels. The mapping translates CLI argument names to internal configuration attribute names while handling optional parameters through existence checks. This specialized mapping ensures surface plotting options are properly configured from command-line inputs.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing surface-specific options from argparse.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with surface plotting parameters in-place.

        Returns:
            None
        """
        surface_mapping = {
            'variable': 'variable',
            'plot_type': 'plot_type',
            'grid_resolution': 'grid_resolution',
            'grid_resolution_deg': 'grid_resolution_deg',
            'interpolation': 'interpolation_method',
            'contour_levels': 'contour_levels'
        }
        
        for arg_name, config_attr in surface_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_wind_args(self, args: argparse.Namespace, config_dict: Dict[str, Any]) -> None:
        """
        Map wind-specific command-line arguments into the configuration dictionary for vector field plotting workflows. This method extracts wind analysis parameters from the argparse namespace including u/v component variable names, vertical level selection, plot type specification, subsampling factors, and vector styling parameters. The mapping handles both barb and arrow plot types with appropriate visualization settings. This specialized mapping ensures wind plotting options are properly translated from CLI inputs to internal configuration attributes.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing wind-specific options from argparse.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with wind plotting parameters in-place.

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
            'vector_alpha': 'vector_alpha'
        }
        
        for arg_name, config_attr in wind_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_cross_args(self, args: argparse.Namespace, config_dict: Dict[str, Any]) -> None:
        """
        Map cross-section-specific command-line arguments into the configuration dictionary for vertical slice workflows. This method extracts 3D cross-section parameters from the argparse namespace including variable name, start/end coordinates, vertical coordinate system, and interpolation settings. The mapping handles path definition, resolution control, and vertical extent specification. This specialized mapping ensures cross-section plotting options are properly translated from CLI inputs to internal configuration attributes.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing cross-section-specific options from argparse.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with cross-section parameters in-place.

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
    
    def _map_overlay_args(self, args: argparse.Namespace, config_dict: Dict[str, Any]) -> None:
        """
        Map overlay-specific command-line arguments into the configuration dictionary for multi-variable composite workflows. This method extracts overlay parameters from the argparse namespace including overlay type, primary and secondary variables, and transparency controls. The mapping handles comma-separated variable lists by splitting them into arrays and populates configuration keys used by the processing and visualization routines. This specialized mapping ensures overlay plotting options are properly translated from CLI inputs to internal configuration attributes.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing overlay-specific options from argparse.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with overlay parameters in-place.

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
    
    def setup_logging(self, config: MPASConfig, log_file: Optional[str] = None) -> MPASLogger:
        """
        Initialize and configure the logging system based on configuration verbosity settings. This method creates an MPASLogger instance with appropriate log levels determined by the quiet and verbose flags in the configuration object. It maps quiet mode to ERROR level, normal mode to INFO level, and verbose mode to DEBUG level for detailed diagnostic information. The logger can optionally write to a file in addition to console output. Returns the configured logger instance which is stored as an instance attribute for subsequent use.

        Parameters:
            config (MPASConfig): Configuration object containing verbosity and logging preferences including quiet and verbose flags.
            log_file (Optional[str]): Optional path to log file for persistent output, None for console-only logging (default: None).

        Returns:
            MPASLogger: Configured and ready-to-use logger instance for CLI operations with appropriate verbosity level set.
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
    
    def validate_config(self, config: MPASConfig) -> bool:
        """
        Perform comprehensive validation of configuration settings and verify input file existence. This method checks all required parameters for the requested analysis type, validates file paths exist and are accessible, and ensures spatial extent bounds are logical. It uses the DataValidator utility to systematically check configuration completeness and performs analysis-specific validation such as requiring cross-section endpoints for vertical slice analysis. Validation errors are collected and reported through the logger or console with detailed descriptions.

        Parameters:
            config (MPASConfig): Configuration object to validate containing all analysis settings and file paths.

        Returns:
            bool: True if validation succeeds and analysis can proceed, False if any validation errors are detected.
        """
        validator = DataValidator()
        errors = []
        
        if not config.grid_file:
            errors.append("Grid file not specified")
        elif not Path(config.grid_file).exists():
            errors.append(f"Grid file not found: {config.grid_file}")
        
        if not config.data_dir:
            errors.append("Data directory not specified")
        elif not Path(config.data_dir).exists():
            errors.append(f"Data directory not found: {config.data_dir}")
        elif not Path(config.data_dir).is_dir():
            errors.append(f"Data path is not a directory: {config.data_dir}")
        
        if config.data_dir and Path(config.data_dir).exists():
            data_path = Path(config.data_dir)
            data_files = list(data_path.glob("diag*.nc"))
            if not data_files and (data_path / 'diag').exists():
                data_files = list((data_path / 'diag').glob("diag*.nc"))
            if not data_files:
                data_files = list(data_path.glob("mpasout*.nc"))
            if not data_files and (data_path / 'mpasout').exists():
                data_files = list((data_path / 'mpasout').glob("mpasout*.nc"))
            if not data_files:
                data_files = list(data_path.rglob("diag*.nc")) + list(data_path.rglob("mpasout*.nc"))
            if not data_files:
                errors.append(f"No MPAS data files found in: {config.data_dir}")
        
        if hasattr(config, 'lat_min') and hasattr(config, 'lat_max'):
            if config.lat_min >= config.lat_max:
                errors.append("Invalid latitude range: lat_min >= lat_max")
        
        if hasattr(config, 'lon_min') and hasattr(config, 'lon_max'):
            if config.lon_min >= config.lon_max:
                errors.append("Invalid longitude range: lon_min >= lon_max")
        
        if hasattr(config, 'analysis_type'):
            if config.analysis_type in ['cross', 'xsec', '3d', 'vertical']:
                required_attrs = ['start_lon', 'start_lat', 'end_lon', 'end_lat']
                for attr in required_attrs:
                    if not hasattr(config, attr) or getattr(config, attr) is None:
                        errors.append(f"Cross-section analysis requires --{attr.replace('_', '-')}")
        
        if errors:
            if self.logger:
                self.logger.error("Configuration validation failed:")
                for error in errors:
                    self.logger.error(f"  - {error}")
            else:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
            return False
        
        return True
    
    def run_analysis(self, config: MPASConfig) -> bool:
        """
        Execute the configured MPAS analysis workflow with comprehensive error handling and performance monitoring. This main dispatcher method determines the analysis type from configuration, initializes performance monitoring, and routes execution to the appropriate specialized analysis method. It wraps the analysis execution in a performance timer to track total runtime, handles keyboard interrupts gracefully to allow user cancellation, and catches all exceptions with detailed error reporting. The method supports five analysis types: precipitation, surface, wind, cross-section, and overlay.

        Parameters:
            config (MPASConfig): Configuration object specifying analysis type, input files, and all visualization parameters.

        Returns:
            bool: True if the analysis pipeline executed successfully, False if errors occurred or execution was interrupted.
        """
        try:
            self.perf_monitor = PerformanceMonitor()
            
            if not hasattr(config, 'analysis_type') or not config.analysis_type:
                if self.logger:
                    self.logger.error("No analysis type specified")
                return False
            
            analysis_type = config.analysis_type
            
            with self.perf_monitor.timer("Total analysis"):
                if analysis_type in ['precipitation', 'precip', 'rain']:
                    success = self._run_precipitation_analysis(config)
                elif analysis_type in ['surface', 'surf', '2d']:
                    success = self._run_surface_analysis(config)
                elif analysis_type in ['wind', 'vector', 'winds']:
                    success = self._run_wind_analysis(config)
                elif analysis_type in ['cross', 'xsec', '3d', 'vertical']:
                    success = self._run_cross_analysis(config)
                elif analysis_type in ['overlay', 'complex', 'multi', 'composite']:
                    success = self._run_overlay_analysis(config)
                else:
                    if self.logger:
                        self.logger.error(f"Unknown analysis type: {analysis_type}")
                    return False
            
            if self.logger and hasattr(config, 'verbose') and config.verbose:
                self.perf_monitor.print_summary()
            
            return success
            
        except KeyboardInterrupt:
            if self.logger:
                self.logger.warning("Analysis interrupted by user")
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Analysis failed: {e}")
                if hasattr(config, 'verbose') and config.verbose:
                    import traceback
                    self.logger.error(traceback.format_exc())
            return False
    
    def _run_precipitation_analysis(self, config: MPASConfig) -> bool:
        """
        Execute precipitation accumulation analysis workflow including data loading, processing, and map generation. This method orchestrates the complete precipitation visualization pipeline by initializing the 2D processor to load diagnostic data, creating a precipitation plotter with specified figure settings, and generating either single-timestep or batch-mode accumulation maps. It supports both serial and parallel batch processing modes when enabled, automatically selecting the appropriate execution strategy. The method handles output directory creation, applies spatial bounds and accumulation periods, and reports the number of successfully generated plots.

        Parameters:
            config (MPASConfig): Configuration object containing all required precipitation parameters including grid file, data directory, variable name, accumulation period, and output settings.

        Returns:
            bool: True if precipitation analysis completes successfully, False if errors occur during processing or plotting.
        """
        assert self.perf_monitor is not None, "Performance monitor must be initialized"
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
                    accum_period=config.accumulation_period
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
    
    def _run_surface_analysis(self, config: MPASConfig) -> bool:
        """
        Execute surface variable visualization workflow for 2D meteorological fields including contour and filled contour plots. This method manages the complete surface analysis pipeline by loading 2D diagnostic data, initializing the surface plotter with figure specifications, and generating either single-timestep or batch-mode visualizations. It supports both serial and parallel batch processing when enabled, handling variables like temperature, pressure, and moisture on MPAS unstructured grids. The method creates output directories, applies spatial bounds and gridding options, and logs the number of successfully generated surface maps.

        Parameters:
            config (MPASConfig): Configuration object with surface analysis parameters including grid file, data directory, variable name, plot type, gridding resolution, and output preferences.

        Returns:
            bool: True if surface analysis workflow completes successfully, False if processing or plotting errors occur.
        """
        assert self.perf_monitor is not None, "Performance monitor must be initialized"
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
                        grid_resolution_deg=getattr(config, 'grid_resolution_deg', None),
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
                        grid_resolution=getattr(config, 'grid_resolution', None),
                        grid_resolution_deg=getattr(config, 'grid_resolution_deg', None)
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
                    clim_max=config.clim_max
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
    
    def _run_wind_analysis(self, config: MPASConfig) -> bool:
        """
        Execute wind vector analysis workflow generating barb or arrow plots from u/v wind components on MPAS grids. This method orchestrates the wind visualization pipeline by loading 2D wind component data, initializing the wind plotter with figure settings, and producing either single-timestep or batch-mode vector plots. It handles wind field visualization at specified levels with configurable subsampling to control vector density, scaling factors for arrow/barb sizes, and optional background field overlays. The method supports multiple plot types including barbs, arrows, and streamlines while managing output directory creation.

        Parameters:
            config (MPASConfig): Configuration object containing wind analysis settings including u/v variable names, wind level, plot type, subsampling factor, vector scale, and output preferences.

        Returns:
            bool: True if wind visualization completes successfully, False if data loading or plotting errors occur.
        """
        assert self.perf_monitor is not None, "Performance monitor must be initialized"
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
                    show_background=config.show_background
                )
                if self.logger:
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
                    show_background=config.show_background
                )
                
                output_path = config.output or os.path.join(
                    config.output_dir,
                    f"mpas_wind_{config.wind_level}_{config.wind_plot_type}_{config.time_index:03d}"
                )
                
                plotter.save_plot(output_path, formats=config.output_formats or ['png'])
                plotter.close_plot()
                
                if self.logger:
                    self.logger.info(f"Wind plot saved: {output_path}")
        
        return True
    
    def _run_cross_analysis(self, config: MPASConfig) -> bool:
        """
        Execute vertical cross-section analysis workflow extracting and visualizing atmospheric slices along great circle paths. This method manages the 3D cross-section pipeline by loading atmospheric data with vertical structure, initializing the cross-section plotter with figure specifications, and generating vertical slices through specified start and end coordinates. It handles interpolation along the cross-section path, supports multiple vertical coordinate systems including pressure, height, and model levels, and produces either single-timestep or batch-mode cross-section visualizations. The method validates endpoint coordinates and creates output directories.

        Parameters:
            config (MPASConfig): Configuration object with cross-section parameters including start/end lat/lon coordinates, variable name, vertical coordinate type, number of interpolation points, and output settings.

        Returns:
            bool: True if cross-section analysis completes successfully, False if data loading, interpolation, or plotting errors occur.
        """
        assert self.perf_monitor is not None, "Performance monitor must be initialized"
        with self.perf_monitor.timer("Cross-section analysis"):
            processor = MPAS3DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_3d_data(config.data_dir)
            dataset = processor.dataset
            
            plotter = MPASVerticalCrossSectionPlotter(
                figsize=config.figure_size,
                dpi=config.dpi
            )
            
            os.makedirs(config.output_dir, exist_ok=True)
            
            assert config.start_lon is not None and config.start_lat is not None, \
                "Cross-section start coordinates must be specified"
            assert config.end_lon is not None and config.end_lat is not None, \
                "Cross-section end coordinates must be specified"
            
            if config.batch_mode:
                use_parallel = getattr(config, 'parallel', False)
                
                if use_parallel:
                    if self.logger:
                        self.logger.info("Running parallel batch cross-section analysis...")
                    created_files = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                        mpas_3d_processor=processor,
                        var_name=config.variable,
                        start_point=(config.start_lon, config.start_lat),
                        end_point=(config.end_lon, config.end_lat),
                        output_dir=config.output_dir,
                        vertical_coord=config.vertical_coord,
                        num_points=config.num_points,
                        max_height=config.max_height,
                        formats=config.output_formats or ['png'],
                        n_processes=config.workers
                    )
                else:
                    if self.logger:
                        self.logger.info("Running serial batch cross-section analysis...")
                    created_files = plotter.create_batch_cross_section_plots(
                        mpas_3d_processor=processor,
                        var_name=config.variable,
                        start_point=(config.start_lon, config.start_lat),
                        end_point=(config.end_lon, config.end_lat),
                        output_dir=config.output_dir,
                        vertical_coord=config.vertical_coord,
                        num_points=config.num_points,
                        max_height=config.max_height,
                        formats=config.output_formats or ['png']
                    )
                
                if self.logger and created_files:
                    self.logger.info(f"Created {len(created_files)} cross-section plots")
            
            else:
                fig, ax = plotter.create_vertical_cross_section(
                    mpas_3d_processor=processor,
                    var_name=config.variable,
                    start_point=(config.start_lon, config.start_lat),
                    end_point=(config.end_lon, config.end_lat),
                    time_index=config.time_index,
                    vertical_coord=config.vertical_coord,
                    num_points=config.num_points,
                    max_height=config.max_height,
                    title=config.title or f"Vertical Cross-Section: {config.variable}",
                    colormap=config.colormap if config.colormap != 'default' else None
                )
                
                output_path = config.output or os.path.join(
                    config.output_dir,
                    f"mpas_cross_section_{config.variable}_{config.time_index:03d}"
                )
                
                plotter.save_plot(output_path, formats=config.output_formats or ['png'])
                plotter.close_plot()
                
                if self.logger:
                    self.logger.info(f"Cross-section plot saved: {output_path}")
        
        return True
    
    def _run_overlay_analysis(self, config: MPASConfig) -> bool:
        """
        Execute overlay analysis combining multiple variables and plotters for composite visualizations. This method loads data and selects the appropriate plotting routines based on the overlay type specified in configuration. It currently contains placeholder implementations for common overlay types such as precipitation with wind and temperature with pressure. The method initializes appropriate plotters for each overlay component and combines them according to the specified overlay strategy.

        Parameters:
            config (MPASConfig): Configuration object with overlay-specific options including overlay type, variables list, colormaps, transparency settings, and output preferences.

        Returns:
            bool: True when the overlay routine completes even if placeholder implementation, False on error.
        """
        assert self.perf_monitor is not None, "Performance monitor must be initialized"
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
    
    def main(self) -> int:
        """
        Main entry point for the unified CLI providing comprehensive argument parsing and analysis execution. This method creates the argument parser, processes command-line arguments or configuration files, sets up logging, validates configuration, and executes the requested analysis. It handles configuration file loading with CLI argument overrides, prints system information in verbose mode, and manages keyboard interrupts and exceptions gracefully. Returns standard Unix exit codes for integration with shell scripts and automation workflows.

        Parameters:
            None

        Returns:
            int: Exit code where 0 indicates successful completion, 1 for errors, and 130 for user interruption.
        """
        try:
            parser = self.create_main_parser()
            try:
                args = parser.parse_intermixed_args()
            except Exception:
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
                    args = parser.parse_args(front + rest)
                except Exception:
                    args = parser.parse_args()
            
            if hasattr(args, 'config') and args.config:
                self.config = MPASConfig.load_from_file(args.config)
                cli_config = self.parse_args_to_config(args)
                for key, value in cli_config.to_dict().items():
                    if value is not None:
                        setattr(self.config, key, value)
            else:
                self.config = self.parse_args_to_config(args)
            
            log_file = getattr(args, 'log_file', None)
            self.setup_logging(self.config, log_file)
            
            if not self.validate_config(self.config):
                return 1
            
            if hasattr(self.config, 'verbose') and self.config.verbose:
                self._print_system_info()
                self._print_config_summary()
            
            success = self.run_analysis(self.config)
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
            return 130
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error: {e}")
            else:
                print(f"Error: {e}")
            return 1
    
    def _print_system_info(self) -> None:
        """
        Display system information including Python version, platform, and available memory to the log output. This utility method provides diagnostic information useful for troubleshooting and performance analysis by printing Python interpreter details, operating system platform, current working directory, and available RAM. It attempts to use psutil for memory information and gracefully handles its absence. The output is formatted with section headers for readability in log files.

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
        Display a formatted summary of the current configuration settings to the log output. This utility method iterates through all configuration parameters stored in the config object, filtering out None values and printing non-empty settings in a readable format with indentation. It wraps the output in section headers with separator lines for visual clarity. The summary provides a comprehensive view of all active analysis settings including file paths, spatial bounds, variable names, and output options.

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
    Module-level entry point providing convenient access to the unified CLI functionality from command line or scripts. This function instantiates the MPASUnifiedCLI class and delegates execution to its main method, handling argument parsing, configuration, validation, and analysis execution in a single call. It serves as the primary interface for the mpasdiag command-line tool and returns standard Unix exit codes. The function can be called directly from Python scripts or through setuptools console_scripts entry points.

    Parameters:
        None

    Returns:
        int: Exit status code where 0 indicates successful analysis completion and non-zero values indicate various failure conditions.
    """
    cli = MPASUnifiedCLI()
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())