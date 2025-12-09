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
        Create and register the surface subparser enabling 2D meteorological scalar field visualization workflows for variables like temperature, pressure, and moisture. This method constructs a new subparser with command aliases ('surf', '2d') to handle surface-specific arguments including required variable name selection, plot type specification among four rendering styles (scatter for direct point display, contour for line contours, filled_contour for smooth filled regions, pcolormesh for grid cell coloring), and gridding options for interpolation control. The subparser configures grid resolution in both point count and degree spacing formats, interpolation method selection (linear, cubic, nearest), and contour level specification. It delegates to _add_common_arguments for shared parameters then adds a surface-specific argument group. The method provides comprehensive usage examples demonstrating temperature scatter plots, pressure contour plots, and custom grid resolution configurations.

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
        Create and register the cross-section subparser enabling 3D vertical atmospheric slice visualization workflows along specified great circle paths. This method constructs a new subparser with command aliases ('xsec', '3d', 'vertical') to handle cross-section-specific arguments including required start and end point coordinates in longitude and latitude, 3D atmospheric variable selection, and vertical coordinate system specification among three types (pressure for isobaric levels, height for geometric altitude, model_levels for native vertical coordinates). The subparser configures interpolation resolution along the horizontal path with the num_points parameter, maximum vertical extent controls, plot style selection (filled_contour for smooth color fields, contour for line contours, pcolormesh for grid cells), and colorbar extension settings. It requires grid file specification and delegates to _add_common_arguments for shared parameters. The method provides comprehensive usage examples demonstrating temperature and wind cross-sections with different vertical coordinate systems.

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
        xsec_group.add_argument('--plot-style', type=str, default='filled_contour',
                              choices=['filled_contour', 'contour', 'pcolormesh'],
                              help='Cross-section plot style (default: filled_contour)')
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
        Map surface-specific command-line arguments into the configuration dictionary enabling 2D scalar field visualization parameter configuration. This method extracts surface analysis parameters from the argparse namespace including required variable name selection for meteorological fields (t2m, mslp, q2, etc.), plot type specification among four rendering options (scatter, contour, filled_contour, pcolormesh), grid resolution controls specified either as integer point counts per axis or as floating-point degree spacing for interpolation grids, interpolation method selection from three algorithms (linear for smooth gradients, cubic for high-quality smoothing, nearest for preserving exact values), and contour level count specification. The mapping systematically translates CLI argument names to internal configuration attribute names while handling optional parameters through existence checks before assignment. This specialized mapping ensures surface plotting options correctly propagate from command-line inputs to internal configuration structures used by MPAS2DProcessor and MPASSurfacePlotter classes.

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
            'grid_resolution_deg': 'grid_resolution_deg',
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
        Map cross-section-specific command-line arguments into the configuration dictionary enabling vertical atmospheric slice parameter configuration. This method extracts 3D cross-section parameters from the argparse namespace including 3D atmospheric variable name selection, required start point coordinates (start_lon, start_lat) defining the cross-section path origin, required end point coordinates (end_lon, end_lat) defining the path terminus, vertical coordinate system specification from three options (pressure for isobaric levels, height for geometric altitude, model_levels for native sigma or hybrid coordinates), number of interpolation points controlling horizontal resolution along the great circle path, maximum vertical extent in kilometers for plot axis limits, plot style selection from three rendering types (filled_contour for smooth color fields, contour for line contours, pcolormesh for grid cell display), and colorbar extension settings controlling whether colorbar extends beyond data limits. The mapping systematically translates user-facing CLI argument names to internal configuration keys while handling all required and optional parameters. This specialized mapping ensures cross-section visualization options correctly propagate to MPAS3DProcessor and MPASVerticalCrossSectionPlotter classes.

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
            data_files = list(data_path.glob(DIAG_GLOB))
            if not data_files and (data_path / 'diag').exists():
                data_files = list((data_path / 'diag').glob(DIAG_GLOB))
            if not data_files:
                data_files = list(data_path.glob(MPASOUT_GLOB))
            if not data_files and (data_path / 'mpasout').exists():
                data_files = list((data_path / 'mpasout').glob(MPASOUT_GLOB))
            if not data_files:
                data_files = list(data_path.rglob(DIAG_GLOB)) + list(data_path.rglob(MPASOUT_GLOB))
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
            config (MPASConfig): Configuration object with surface analysis parameters including grid file path, data directory location, variable name selection, plot type (scatter/contour/filled_contour/pcolormesh), gridding resolution specifications, spatial bounds, and output preferences.

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
                        self.logger.warning("Parallel processing for wind analysis is not yet implemented. Falling back to serial processing.")
                    # Explicitly disable parallel flag to prevent multiprocessing issues
                    config.parallel = False
                
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
                    show_background=config.show_background,
                    grid_resolution=getattr(config, 'grid_resolution', None),
                    regrid_method=getattr(config, 'regrid_method', 'linear')
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
    
    def _run_cross_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        Execute vertical cross-section analysis workflow extracting and visualizing atmospheric slices through 3D model output along specified great circle paths. This method manages the 3D cross-section pipeline by loading atmospheric data with complete vertical structure through MPAS3DProcessor initialization, creating MPASVerticalCrossSectionPlotter with specified figure dimensions and resolution settings, and generating vertical slices through user-specified start and end coordinates defining the cross-section path. It handles interpolation along the great circle path between endpoints, supports three vertical coordinate systems (pressure for isobaric levels, height for geometric altitude AGL, model_levels for native sigma or hybrid coordinates), and produces either single-timestep cross-sections or batch-mode time series depending on configuration. The method validates that required endpoint coordinates are specified with assertion checks, supports both serial execution and parallel MPI-based execution when enabled through ParallelCrossSectionProcessor, creates output directories as needed, and saves visualizations in requested formats with appropriate naming conventions.

        Parameters:
            config (MPASConfig): Configuration object with cross-section parameters including grid file path, data directory, variable name, start/end lat/lon coordinates, vertical coordinate type, number of interpolation points along path, maximum vertical extent, plot style, and output settings.

        Returns:
            bool: True if cross-section analysis workflow completes successfully through interpolation and visualization, False if data loading, coordinate validation, interpolation, or plotting errors occur.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG
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
    
    def _run_overlay_analysis(
        self,
        config: MPASConfig
    ) -> bool:
        """
        Execute overlay analysis workflow combining multiple variables and plotters for composite visualizations enabling complex multi-field atmospheric displays. This method loads 2D diagnostic data through MPAS2DProcessor initialization and selects appropriate plotting routines based on the overlay type specified in configuration (precip_wind for precipitation with wind vectors, temp_pressure for temperature with pressure contours, wind_temp for winds with temperature background, multi_level for arbitrary multi-variable displays, custom for user-defined composites). The method currently contains placeholder implementations for common overlay type combinations demonstrating the intended architecture, creates output directories as needed, and initializes appropriate plotter instances (MPASPrecipitationPlotter, MPASWindPlotter, MPASSurfacePlotter) for each overlay component. Full implementation will combine multiple visualization layers according to specified overlay strategies with transparency blending and optional contour line overlays.

        Parameters:
            config (MPASConfig): Configuration object with overlay-specific options including overlay type selection, primary and secondary variable specifications, variables list for multi-variable displays, colormaps for independent layer styling, transparency settings, contour overlay flags, and output preferences.

        Returns:
            bool: True when the overlay routine completes execution even with placeholder implementation, False on initialization or configuration errors.
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
    
    def main(self) -> int:
        """
        Main entry point for the unified CLI orchestrating comprehensive argument parsing, configuration management, validation, and analysis execution with robust error handling. This method creates the argument parser using create_main_parser, processes command-line arguments through sophisticated parsing that handles both intermixed global and subcommand arguments, loads optional YAML configuration files with CLI argument override capability, sets up logging with appropriate verbosity levels, validates complete configuration through validate_config, and executes the requested analysis via run_analysis. It implements intelligent argument reordering to handle global options appearing after subcommands, merges configuration file settings with command-line overrides giving precedence to CLI arguments, prints comprehensive system information and configuration summaries in verbose mode for debugging, and manages three exception types: standard exceptions for error reporting, KeyboardInterrupt for graceful user cancellation, and generic Exception catch-all for unexpected failures. Returns standard Unix exit codes enabling integration with shell scripts, automation workflows, and CI/CD pipelines.

        Parameters:
            None

        Returns:
            int: Unix exit code where 0 indicates successful analysis completion, 1 indicates errors or validation failures, and 130 indicates user interruption via keyboard interrupt.
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
            import traceback
            if self.logger:
                self.logger.error(f"Unexpected error: {e}")
                self.logger.error(traceback.format_exc())
            else:
                print(f"Error: {e}")
                traceback.print_exc()
            return 1
    
    def _print_system_info(self) -> None:
        """
        Display comprehensive system information including Python interpreter version, operating system platform, and available memory resources to log output for diagnostic purposes. This utility method provides essential diagnostic information useful for troubleshooting analysis failures, performance investigation, and reproducibility documentation by printing Python version string with full version details, operating system platform identifier (darwin, linux, win32), current working directory path for relative file resolution context, and available system RAM in gigabytes through optional psutil integration. The method attempts to import psutil for memory reporting and gracefully handles its absence by skipping memory information when the package is unavailable. Output is formatted with section headers and separator lines for visual clarity in verbose log files enabling quick identification of system-specific issues.

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
        Display a formatted comprehensive summary of all active configuration settings to log output enabling verification of analysis parameters before execution. This utility method iterates through all configuration parameters stored in the config object's internal dictionary representation obtained through to_dict(), filters out None-valued parameters to show only actively-set options, and prints each non-empty setting in a readable indented format with clear key-value pairs. It wraps the parameter listing in visually distinct section headers with equal-sign separator lines for clear delineation in log output. The summary provides a complete view of all active analysis settings including file paths, spatial extent bounds, variable selections, visualization options, and processing flags enabling users to verify configuration correctness before long-running analyses begin.

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