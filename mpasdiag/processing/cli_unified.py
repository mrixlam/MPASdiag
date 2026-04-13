#!/usr/bin/env python3

"""
Unified Command Line Interface for MPAS Analysis

This module implements a unified command-line interface (CLI) for the MPAS diagnostic processing and visualization tool. It provides a single entry point for users to access various types of analyses, including precipitation accumulation, surface variable plotting, wind vector visualization, vertical cross-section generation, and complex multi-variable overlays. The CLI is designed to be flexible and user-friendly, allowing users to specify input parameters, output settings, and processing options through command-line arguments or configuration files. It also incorporates performance monitoring and logging capabilities to facilitate efficient execution and debugging of diagnostic workflows. By centralizing the CLI functionality in this module, it promotes maintainability and extensibility of the MPAS diagnostic tool as new analysis types or features can be added in a structured manner while maintaining a consistent user interface. 
    
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
from typing import Dict, List, Any, Optional
from pathlib import Path

from .utils_config import MPASConfig
from .utils_logger import MPASLogger
from .utils_monitor import PerformanceMonitor
from .processors_2d import MPAS2DProcessor
from .processors_3d import MPAS3DProcessor

from .parallel_wrappers import (
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor
)

from .constants import DIAG_GLOB, MPASOUT_GLOB, PERFORMANCE_MONITOR_MSG
from .utils_datetime import MPASDateTimeUtils

try:
    from ..visualization.precipitation import MPASPrecipitationPlotter
    from ..visualization.surface import MPASSurfacePlotter
    from ..visualization.cross_section import MPASVerticalCrossSectionPlotter
    from ..visualization.wind import MPASWindPlotter
    from ..diagnostics.precipitation import PrecipitationDiagnostics
    from ..diagnostics.sounding import SoundingDiagnostics
    from ..visualization.skewt import MPASSkewTPlotter
except ImportError:
    try:
        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
        from mpasdiag.visualization.wind import MPASWindPlotter
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        from mpasdiag.diagnostics.sounding import SoundingDiagnostics
        from mpasdiag.visualization.skewt import MPASSkewTPlotter
    except ImportError:
        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
        from mpasdiag.visualization.wind import MPASWindPlotter
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        from mpasdiag.diagnostics.sounding import SoundingDiagnostics
        from mpasdiag.visualization.skewt import MPASSkewTPlotter


class MPASUnifiedCLI:
    """ Unified command-line interface for all MPAS visualization types. """
    
    PLOT_TYPES = {
        'precipitation': 'Precipitation accumulation maps',
        'surface': 'Surface variable scalar plots',
        'wind': 'Wind vector plots (barbs/arrows)',
        'cross': '3D vertical cross-section plots',
        'overlay': 'Complex multi-variable overlay plots'
    }
    
    def __init__(self: "MPASUnifiedCLI") -> None:
        """
        This method initializes the MPASUnifiedCLI class, setting up instance variables for logging, performance monitoring, and configuration management. It prepares the CLI for execution by initializing these components to None, which will later be configured based on user input and command-line arguments. The logger will be used to record messages and events during the execution of the CLI, while the performance monitor will track the execution time of various processing steps. The configuration variable will hold the parsed configuration settings from command-line arguments or configuration files, allowing for flexible and customizable execution of different analysis types. By initializing these components in the constructor, it ensures that they are available throughout the lifecycle of the CLI instance for consistent logging, performance tracking, and configuration management across all analysis types. 

        Parameters:
            None

        Returns:
            None
        """
        self.logger = None
        self.perf_monitor = None
        self.config = None
    
    def create_main_parser(self: "MPASUnifiedCLI") -> argparse.ArgumentParser:
        """
        This method creates and configures the main argument parser for the MPASUnifiedCLI, which serves as the entry point for users to access various types of analyses. The main parser is set up with a program name, description, and an epilog that provides usage examples for different analysis types. It includes global arguments for configuration file input, verbosity control, logging options, and version information. The method also creates subparsers for each analysis type (precipitation, surface variables, wind vectors, vertical cross-sections, and complex overlays) and registers them with the main parser. Each subparser is configured with its own specific arguments relevant to the type of analysis it performs, while also incorporating common arguments through a shared method. By structuring the CLI in this way, it allows users to easily navigate and execute different analyses while maintaining a consistent interface and providing helpful guidance through detailed help text and examples. 

        Parameters:
            None

        Returns:
            argparse.ArgumentParser: The configured main argument parser for the MPASUnifiedCLI. 
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
        self._add_sounding_parser(subparsers)
        self._add_overlay_parser(subparsers)
        
        return parser
    
    def _add_common_arguments(self: "MPASUnifiedCLI", 
                              parser: argparse.ArgumentParser,
                              required_grid: bool = True) -> None:
        """
        This method adds common command-line arguments to the provided parser object for MPAS diagnostic analyses. These common arguments include options for specifying input and output paths, spatial extent of the analysis, time selection criteria, output formatting and resolution, color settings for visualizations, and processing options such as parallel execution and chunk size. By centralizing these common arguments in a single method, it promotes consistency across different analysis types and reduces redundancy in argument definitions. The method takes a parser object as input, which can be either the main parser or any of the subparsers for specific analysis types, and adds the relevant arguments to it. The required_grid parameter allows for flexibility in marking the --grid-file argument as required or optional based on the needs of specific analyses. This approach ensures that all analyses have access to a consistent set of common parameters while still allowing for customization based on the specific requirements of each analysis type. 

        Parameters:
            parser (argparse.ArgumentParser): The argument parser object to which the common arguments will be added. This can be the main parser or any of the subparsers for specific analysis types.
            required_grid (bool): A flag indicating whether the --grid-file argument should be marked as required. Default is True, meaning that the grid file is required for most analyses, but it can be set to False for analyses that do not require a grid file. 

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
    
    def _add_precipitation_parser(self: "MPASUnifiedCLI", 
                                  subparsers: argparse._SubParsersAction) -> None:
        """
        This method creates and registers the precipitation subparser for the MPASUnifiedCLI, enabling users to generate precipitation accumulation maps from MPAS diagnostic data. The subparser is configured with command aliases ('precip', 'rain') for user convenience and includes specific arguments for selecting the precipitation variable, accumulation period, plot type (scatter or contourf), grid resolution for interpolation, thresholding options, and output units. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing precipitation analyses effectively, allowing for flexible visualization of precipitation accumulation over various time periods and with different plotting styles. By registering this subparser under the main CLI, it allows users to easily access precipitation analysis functionality while maintaining a consistent interface across different types of analyses. 

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the precipitation parser will be registered. 

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
    
    def _add_surface_parser(self: "MPASUnifiedCLI", 
                            subparsers: argparse._SubParsersAction) -> None:
        """
        This method creates and registers the surface variable subparser for the MPASUnifiedCLI, enabling users to generate surface variable scalar plots from MPAS diagnostic data. The subparser is configured with command aliases ('surf', '2d') for user convenience and includes specific arguments for selecting the surface variable to plot, choosing the plot type (scatter, contour, contourf, pcolormesh), specifying grid resolution for interpolation, controlling interpolation method for contour plots, and setting the number of contour levels. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing surface variable analyses effectively, allowing for flexible visualization of various surface variables such as temperature, pressure, humidity, or wind components with different plotting styles. By registering this subparser under the main CLI, it allows users to easily access surface variable analysis functionality while maintaining a consistent interface across different types of analyses. 

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the surface parser will be registered. 

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
    
    def _add_wind_parser(self: "MPASUnifiedCLI", 
                         subparsers: argparse._SubParsersAction) -> None:
        """
        This method creates and registers the wind vector subparser for the MPASUnifiedCLI, enabling users to generate wind vector plots from MPAS diagnostic data. The subparser is configured with command aliases ('vector', 'winds') for user convenience and includes specific arguments for selecting the U and V wind component variables, wind level description for labeling, wind vector representation (barbs, arrows, streamlines), subsampling factor for vector density control, scale factor for wind vectors, options to show background wind speed as filled contours, colormap selection for background speed, color and transparency settings for vectors, and grid resolution and interpolation method for regridding wind components if needed. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing wind vector analyses effectively, allowing for flexible visualization of surface or upper-level winds with various customization options. By registering this subparser under the main CLI, it allows users to easily access wind vector analysis functionality while maintaining a consistent interface across different types of analyses. 

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the wind parser will be registered. 

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
    
    def _add_cross_parser(self: "MPASUnifiedCLI", 
                          subparsers: argparse._SubParsersAction) -> None:
        """
        This method creates and registers the vertical cross-section subparser for the MPASUnifiedCLI, enabling users to generate 3D vertical cross-section plots from MPAS diagnostic data. The subparser is configured with command aliases ('cross', 'xsection') for user convenience and includes specific arguments for selecting the variable to plot, defining the start and end points of the cross-section in terms of longitude and latitude, choosing the plot type (contour, contourf, pcolormesh), specifying grid resolution for interpolation, controlling interpolation method for contour plots, and setting the number of contour levels. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing vertical cross-section analyses effectively, allowing for flexible visualization of atmospheric variables along a defined transect with different plotting styles. By registering this subparser under the main CLI, it allows users to easily access vertical cross-section analysis functionality while maintaining a consistent interface across different types of analyses. 

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the vertical cross-section parser will be registered. 

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
                              choices=['pressure', 'modlev', 'height'],
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
    
    def _add_sounding_parser(self: "MPASUnifiedCLI", 
                             subparsers: argparse._SubParsersAction) -> None:
        """
        This method creates and registers the sounding subparser for the MPASUnifiedCLI, enabling users to generate Skew-T Log-P diagrams from MPAS 3D atmospheric data at specified locations. The subparser is configured with command aliases ('skewt', 'profile') for user convenience and includes specific arguments for selecting the longitude and latitude of the sounding location, options to compute and display thermodynamic indices (CAPE, CIN, LCL, LFC, EL), and an option to plot the lifted parcel profile if MetPy is available. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing sounding analyses effectively, allowing for flexible visualization of atmospheric profiles at specific locations with various customization options. By registering this subparser under the main CLI, it allows users to easily access sounding analysis functionality while maintaining a consistent interface across different types of analyses. 

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the sounding parser will be registered. 

        Returns:
            None
        """
        parser = subparsers.add_parser(
            'sounding',
            aliases=['skewt', 'profile'],
            help='Skew-T Log-P sounding diagram',
            description='Generate Skew-T Log-P diagrams from MPAS 3D atmospheric data at a specified location',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Sounding at Denver
              mpasdiag sounding --grid-file grid.nc --data-dir ./data --lon -105.0 --lat 39.7
              
              # With thermodynamic indices and parcel trace
              mpasdiag sounding --grid-file grid.nc --data-dir ./data --lon -105.0 --lat 39.7 --show-indices --show-parcel
              
              # Batch all time steps
              mpasdiag sounding --grid-file grid.nc --data-dir ./data --lon -97.5 --lat 35.2 --batch-all --show-indices
            """)
        )
        
        self._add_common_arguments(parser, required_grid=True)
        
        sounding_group = parser.add_argument_group('Sounding Options')
        sounding_group.add_argument('--lon', type=float, required=True,
                                   help='Sounding location longitude in degrees')
        sounding_group.add_argument('--lat', type=float, required=True,
                                   help='Sounding location latitude in degrees')
        sounding_group.add_argument('--show-indices', action='store_true',
                                   help='Compute and display thermodynamic indices (CAPE, CIN, LCL, LFC, EL)')
        sounding_group.add_argument('--show-parcel', action='store_true',
                                   help='Plot lifted parcel profile (requires MetPy)')

    def _add_overlay_parser(self: "MPASUnifiedCLI", 
                            subparsers: argparse._SubParsersAction) -> None:
        """
        This method creates and registers the complex overlay subparser for the MPASUnifiedCLI, enabling users to generate multi-variable overlay plots that combine different types of data (e.g., precipitation with wind vectors, temperature with pressure contours) from MPAS diagnostic files. The subparser is configured with command aliases ('complex', 'multi', 'composite') for user convenience and includes specific arguments for selecting the type of overlay analysis to perform, specifying primary and secondary variables for the overlay, defining custom variable lists for multi-variable overlays, configuring wind vector options for overlays that include wind data, and customizing colormap and transparency settings for the overlay elements. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing complex overlay analyses effectively, allowing for flexible visualization of multiple atmospheric variables in a single plot with various customization options. By registering this subparser under the main CLI, it allows users to easily access complex overlay analysis functionality while maintaining a consistent interface across different types of analyses. 

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the overlay parser will be registered. 

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
    
    def parse_args_to_config(self: "MPASUnifiedCLI", 
                             args: argparse.Namespace) -> MPASConfig:
        """
        This method takes the parsed command-line arguments from the argparse namespace and translates them into a structured MPASConfig object that encapsulates all the necessary settings for executing the selected analysis command. It first initializes an empty configuration dictionary and then populates it with common parameters that are applicable across all analysis types, such as input/output paths, spatial extent, time selection, output formatting, color settings, and processing options. The method then checks which specific analysis command was invoked (e.g., precipitation, surface, wind, cross-section, sounding, overlay) and calls dedicated mapping methods to extract and translate analysis-specific parameters from the argparse namespace into the configuration dictionary. Each mapping method handles the unique parameters relevant to its respective analysis type while ensuring that only user-specified options are included in the final configuration. Finally, the method constructs and returns an MPASConfig object using the populated configuration dictionary, which can then be used by the processing classes to execute the desired analysis with the specified settings. 

        Parameters:
            args (argparse.Namespace): The namespace object containing the parsed command-line arguments from the argparse parser. This object includes all the options and parameters specified by the user when invoking the CLI, organized as attributes corresponding to each argument defined in the parser and subparsers. 

        Returns:
            MPASConfig: A structured configuration object that encapsulates all the settings and parameters needed to execute the selected analysis command based on the user's input from the command line. This object is constructed from the configuration dictionary that has been populated with both common and analysis-specific parameters extracted from the argparse namespace. 
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
            self._dispatch_analysis_args(args, config_dict)
        
        return MPASConfig(**config_dict)

    def _dispatch_analysis_args(self: "MPASUnifiedCLI",
                                args: argparse.Namespace,
                                config_dict: Dict[str, Any]) -> None:
        """
        This method serves as a dispatcher that routes the parsed command-line arguments to the appropriate mapping function based on the selected analysis command. It checks the value of the 'analysis_command' attribute in the argparse namespace to determine which specific analysis type was invoked (e.g., precipitation, surface, wind, cross-section, sounding, overlay) and then calls the corresponding mapping method to extract and translate analysis-specific parameters from the argparse namespace into the configuration dictionary. Each mapping method is responsible for handling the unique parameters relevant to its respective analysis type while ensuring that only user-specified options are included in the final configuration. This dispatcher method allows for a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework, enabling flexible configuration of various analyses based on user input.

        Parameters:
            args (argparse.Namespace): The namespace object containing the parsed command-line arguments from the argparse parser, which includes an attribute 'analysis_command' that indicates the specific analysis type selected by the user.
            config_dict (Dict[str, Any]): The mutable configuration dictionary that is being populated with parameters extracted from the argparse namespace. This dictionary will be modified in-place by the mapping methods to include analysis-specific parameters based on the selected analysis command.

        Returns:
            None
        """
        cmd = args.analysis_command
        if cmd in ['precipitation', 'precip', 'rain']:
            self._map_precipitation_args(args, config_dict)
        elif cmd in ['surface', 'surf', '2d']:
            self._map_surface_args(args, config_dict)
        elif cmd in ['wind', 'vector', 'winds']:
            self._map_wind_args(args, config_dict)
        elif cmd in ['cross', 'xsec', '3d', 'vertical']:
            self._map_cross_args(args, config_dict)
        elif cmd in ['sounding', 'skewt', 'profile']:
            self._map_sounding_args(args, config_dict)
        elif cmd in ['overlay', 'complex', 'multi', 'composite']:
            self._map_overlay_args(args, config_dict)

    def _map_precipitation_args(self: "MPASUnifiedCLI", 
                                args: argparse.Namespace, 
                                config_dict: Dict[str, Any]) -> None:
        """
        This method maps precipitation-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of precipitation accumulation analysis parameters. It extracts the selected precipitation variable, accumulation period, plot type (scatter or contourf), grid resolution for interpolation, thresholding options, and output units from the parsed arguments and translates them into corresponding configuration keys used by the MPASPrecipitationProcessor and MPASPrecipitationPlotter classes. The mapping handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of precipitation analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework. 

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
    
    def _map_surface_args(self: "MPASUnifiedCLI", 
                          args: argparse.Namespace, 
                          config_dict: Dict[str, Any]) -> None:
        """
        This method maps surface variable-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of surface variable analysis parameters. It extracts the selected surface variable to plot, plot type (scatter, contour, contourf, pcolormesh), grid resolution for interpolation, interpolation method for contour plots, and number of contour levels from the parsed arguments and translates them into corresponding configuration keys used by the MPASSurfaceProcessor and MPASSurfacePlotter classes. The mapping handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of surface variable analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework. 

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing surface variable-specific options extracted from the surface subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with surface variable analysis parameters through in-place modification.

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
    
    def _map_wind_args(self: "MPASUnifiedCLI", 
                       args: argparse.Namespace, 
                       config_dict: Dict[str, Any]) -> None:
        """
        This method maps wind-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of wind vector analysis parameters. It extracts the U and V wind component variable names, wind level description for labeling, wind vector representation (barbs, arrows, streamlines), subsampling factor for controlling vector density, scale factor for wind vectors, options to show background wind speed as filled contours along with colormap selection for the background, color and transparency settings for the vectors themselves, and grid resolution and interpolation method for regridding wind components if needed. The mapping translates these user-specified options into corresponding configuration keys used by the MPASWindProcessor and MPASWindPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of wind vector analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework. 

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing wind-specific options extracted from the wind subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with wind vector analysis parameters through in-place modification. 

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
    
    def _map_cross_args(self: "MPASUnifiedCLI", 
                        args: argparse.Namespace, 
                        config_dict: Dict[str, Any]) -> None:
        """
        This method maps vertical cross-section-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of 3D vertical cross-section analysis parameters. It extracts the selected variable to plot along the cross-section, the start and end points of the cross-section defined by longitude and latitude coordinates, the vertical coordinate system to use (pressure, model levels, or height), the number of interpolation points along the path, maximum height for the vertical axis, plot style (contourf, contour, pcolormesh), and colorbar extension settings. The mapping translates these user-specified options into corresponding configuration keys used by the MPASCrossSectionProcessor and MPASCrossSectionPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of vertical cross-section analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework. 

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing vertical cross-section-specific options extracted from the cross-section subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with vertical cross-section analysis parameters through in-place modification.

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
    
    def _map_sounding_args(self: "MPASUnifiedCLI", 
                           args: argparse.Namespace, 
                           config_dict: Dict[str, Any]) -> None:
        """
        This method maps sounding-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of Skew-T Log-P diagram analysis parameters. It extracts the longitude and latitude of the sounding location, options to compute and display thermodynamic indices (CAPE, CIN, LCL, LFC, EL), and an option to plot the lifted parcel profile if MetPy is available. The mapping translates these user-specified options into corresponding configuration keys used by the MPASSoundingProcessor and MPASSoundingPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of sounding analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework. 

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing sounding-specific options extracted from the sounding subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with sounding analysis parameters through in-place modification. 

        Returns:
            None
        """
        sounding_mapping = {
            'lon': 'sounding_lon',
            'lat': 'sounding_lat',
            'show_indices': 'show_indices',
            'show_parcel': 'show_parcel',
        }
        
        for arg_name, config_attr in sounding_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
    
    def _map_overlay_args(self: "MPASUnifiedCLI", 
                          args: argparse.Namespace, 
                          config_dict: Dict[str, Any]) -> None:
        """
        This method maps overlay-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of complex multi-variable overlay analysis parameters. It extracts the type of overlay analysis to perform, primary and secondary variables for the overlay, custom variable lists for multi-variable overlays, wind vector options for overlays that include wind data, and colormap and transparency settings for the overlay elements. The mapping translates these user-specified options into corresponding configuration keys used by the MPASOverlayProcessor and MPASOverlayPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of complex overlay analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework. 

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
    
    def setup_logging(self: "MPASUnifiedCLI", 
                      config: MPASConfig, 
                      log_file: Optional[str] = None) -> MPASLogger:
        """
        This method sets up the logging configuration for the MPASUnifiedCLI based on the user's preferences for logging verbosity and an optional log file path. It determines the appropriate logging level (e.g., ERROR, INFO, DEBUG) based on the presence of 'quiet' and 'verbose' flags in the configuration object. If the 'quiet' flag is set, it configures logging to only capture ERROR messages; if the 'verbose' flag is set, it configures logging to capture DEBUG messages; otherwise, it defaults to INFO level. The method also checks if a log file path is provided and configures the MPASLogger to write log messages to that file in addition to the console output. By setting up logging in this way, it allows for flexible control over the amount of information logged during execution and provides an option for persistent logging to a file for later review or debugging purposes. The configured MPASLogger instance is then returned for use throughout the CLI execution. 

        Parameters:
            config (MPASConfig): The configuration object containing user preferences for logging verbosity (e.g., 'quiet', 'verbose') that will be used to determine the appropriate logging level for the MPASLogger. 
            log_file (Optional[str]): An optional file path to which log messages should be written. If provided, the MPASLogger will be configured to write logs to this file in addition to the console output. If None, logging will only occur on the console. 

        Returns:
            MPASLogger: A configured instance of the MPASLogger class that is set up with the appropriate logging level and file handler based on the user's preferences specified in the configuration object and the optional log file path. This logger can then be used throughout the CLI execution to log messages at various levels (ERROR, INFO, DEBUG) as needed. 
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
    
    def _validate_file_path(self: "MPASUnifiedCLI", 
                            file_path: Optional[str], 
                            file_type: str, 
                            errors: List[str]) -> None:
        """
        This method validates that a file path is specified and exists on the filesystem. If the file path is missing (i.e., None or empty), it appends an error message to the provided errors list indicating that the specific type of file was not specified. If a file path is provided but does not exist on the filesystem, it appends an error message indicating that the specific type of file was not found at the given path. This validation step is crucial for ensuring that required input files, such as the grid file, are correctly specified and available before attempting to read them during analysis execution. By accumulating error messages in a list, this method allows for comprehensive reporting of multiple validation issues to the user in a single output. 

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
    
    def _validate_directory_path(self: "MPASUnifiedCLI", 
                                 dir_path: Optional[str], 
                                 dir_type: str, 
                                 errors: List[str]) -> bool:
        """
        This method validates that a directory path is specified, exists on the filesystem, and is indeed a directory. If the directory path is missing (i.e., None or empty), it appends an error message to the provided errors list indicating that the specific type of directory was not specified. If a directory path is provided but does not exist on the filesystem, it appends an error message indicating that the specific type of directory was not found at the given path. If a path is provided but exists as a file rather than a directory, it appends an error message indicating that the path is not a directory. This validation step is crucial for ensuring that required input/output directories are correctly specified and available before attempting to read from or write to them during analysis execution. By accumulating error messages in a list, this method allows for comprehensive reporting of multiple validation issues to the user in a single output. The method returns True if the directory path passes all validation checks (i.e., it is specified, exists, and is a directory) and False if any check fails. 

        Parameters:
            dir_path (Optional[str]): The directory path to validate, which may be None if not specified by the user.
            dir_type (str): A descriptive string indicating the type of directory being validated (e.g., "Data directory", "Output directory") for use in error messages.
            errors (List[str]): A list to which error messages will be appended if validation checks fail, allowing for accumulation of multiple validation issues to be reported together. 

        Returns:
            bool: True if the directory path is valid (specified, exists, and is a directory), False if any validation check fails. 
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
    
    def _find_data_files(self: "MPASUnifiedCLI", 
                         data_path: Path) -> List[Path]:
        """
        This method attempts to discover MPAS data files within the specified data directory by checking for common naming patterns and directory structures used in MPAS diagnostic output. It first looks for files matching the DIAG_GLOB pattern directly within the provided data_path. If no files are found, it checks if there is a 'diag' subdirectory and looks for DIAG_GLOB files there. If still no files are found, it looks for files matching the MPASOUT_GLOB pattern directly within the data_path, and if not found, it checks for an 'mpasout' subdirectory. If no files are found in these common locations, it performs a recursive search through all subdirectories of data_path for files matching either DIAG_GLOB or MPASOUT_GLOB patterns. The method returns a list of Path objects representing the discovered MPAS data files that match the expected naming patterns within the specified data directory and its subdirectories. This file discovery process is essential for ensuring that the analysis has access to the necessary input data files based on user-specified directory paths, even if the files are organized in different ways. 

        Parameters:
            data_path (Path): A Path object representing the directory path where MPAS data files are expected to be located. The method will search this directory and its common subdirectories for files matching the expected naming patterns for MPAS diagnostic output. 

        Returns:
            List[Path]: A list of Path objects representing the MPAS data files that were discovered within the specified data directory and its subdirectories based on the defined glob patterns. If no files are found, an empty list will be returned. 
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
    
    def _validate_coordinate_range(self: "MPASUnifiedCLI",
                                   config: MPASConfig,
                                   min_attr: str,
                                   max_attr: str,
                                   coord_type: str,
                                   errors: List[str]) -> None:
        """
        This method validates that the specified minimum and maximum values for a coordinate range (e.g., latitude or longitude) in the configuration object are logically consistent, ensuring that the minimum value is less than the maximum value. It first checks if both the minimum and maximum attributes are present in the configuration object using hasattr. If both attributes exist, it retrieves their values using getattr and compares them. If the minimum value is greater than or equal to the maximum value, it appends an error message to the provided errors list indicating that the specified coordinate range is invalid. This validation step is crucial for ensuring that spatial bounds specified by the user make sense and will not lead to errors during data processing or plotting due to invalid coordinate ranges. By accumulating error messages in a list, this method allows for comprehensive reporting of multiple validation issues to the user in a single output. 
        
        Parameters:
            config (MPASConfig): The configuration object containing the attributes for the coordinate range to validate, which should include the minimum and maximum values for the specified coordinate type (e.g., lat_min, lat_max for latitude).
            min_attr (str): The name of the attribute in the configuration object that represents the minimum value of the coordinate range (e.g., 'lat_min').
            max_attr (str): The name of the attribute in the configuration object that represents the maximum value of the coordinate range (e.g., 'lat_max').
            coord_type (str): A descriptive string indicating the type of coordinate being validated (e.g., "latitude", "longitude") for use in error messages.
            errors (List[str]): A list to which error messages will be appended if validation checks fail, allowing for accumulation of multiple validation issues to be reported together. 

        Returns:
            None
        """
        if hasattr(config, min_attr) and hasattr(config, max_attr):
            min_val = getattr(config, min_attr)
            max_val = getattr(config, max_attr)
            if min_val >= max_val:
                errors.append(f"Invalid {coord_type} range: {min_attr} >= {max_attr}")
    
    def _validate_cross_section_params(self: "MPASUnifiedCLI",
                                       config: MPASConfig,
                                       errors: List[str]) -> None:
        """
        This method performs additional validation checks specific to cross-section analysis by verifying that if the analysis type is identified as a cross-section type (e.g., 'cross', 'xsec', '3d', 'vertical'), then the required parameters for defining the cross-section path (start and end longitude and latitude) are present in the configuration object. It first checks if the 'analysis_type' attribute exists in the configuration object and if it matches any of the known cross-section analysis types. If it does, it defines a list of required attributes for cross-section analysis (start_lon, start_lat, end_lon, end_lat) and iterates through this list to check if each attribute is present and not None in the configuration object. If any required attribute is missing or None, it appends an error message to the provided errors list indicating that the specific parameter is required for cross-section analysis. This validation step ensures that users have provided all necessary information to define a valid cross-section path before attempting to execute the analysis, preventing runtime errors due to missing parameters and providing clear feedback on what needs to be specified for successful execution. By accumulating error messages in a list, this method allows for comprehensive reporting of multiple validation issues related to cross-section parameters in a single output. 

        Parameters:
            config (MPASConfig): The configuration object containing the attributes for the analysis, which should include the 'analysis_type' attribute to determine if cross-section-specific validation is needed, as well as the required parameters for cross-section analysis if applicable. 
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
    
    def _report_validation_errors(self: "MPASUnifiedCLI", 
                                  errors: List[str]) -> None:
        """
        This method reports the accumulated configuration validation errors to the user by logging them through the configured logger if available, or printing them to the console if no logger is set up. It first defines a header message indicating that configuration validation has failed, and then iterates through the list of error messages, logging or printing each one with a consistent format. This method centralizes the reporting of validation errors, ensuring that users receive clear and organized feedback on what issues were detected during the validation process and what needs to be corrected in their configuration settings before they can successfully execute the analysis. By providing detailed error messages, it helps users understand exactly what parameters are missing or invalid, facilitating easier troubleshooting and correction of their configuration. 

        Parameters:
            errors (List[str]): A list of error messages that were accumulated during the configuration validation process, describing the specific issues that were detected with the provided configuration settings. 

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
    
    def validate_config(self: "MPASUnifiedCLI", 
                        config: MPASConfig) -> bool:
        """
        This method performs comprehensive validation of the configuration object to ensure that all required parameters are specified, file paths exist, and spatial bounds are logically consistent before attempting to execute the analysis. It uses helper methods to validate individual aspects of the configuration, such as checking for the existence of required files and directories, validating coordinate ranges for latitude and longitude, and performing additional checks specific to certain analysis types (e.g., cross-section parameters). If any validation checks fail, it accumulates error messages in a list and reports them to the user through the logger or console output. The method returns True if all validation checks pass successfully, allowing execution to proceed, or False if any validation errors are detected that require user correction before execution can continue. This validation step is crucial for preventing runtime errors during analysis execution by ensuring that the configuration is complete and valid before processing begins. 

        Parameters:
            config (MPASConfig): The configuration object containing all parameters for the analysis, which will be validated for completeness, correctness, and logical consistency before execution. 

        Returns:
            bool: True if the configuration passes all validation checks and is considered valid for execution, False if any validation errors are detected that require user correction before execution can continue.
        """
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
    
    def _check_analysis_type_specified(self: "MPASUnifiedCLI", 
                                       config: MPASConfig) -> bool:
        """
        This method checks whether the 'analysis_type' attribute is specified in the configuration object, which is essential for determining which analysis workflow to execute. It verifies that the 'analysis_type' attribute exists and is not empty or None. If the analysis type is not specified, it logs an error message indicating that no analysis type was specified and returns False, signaling that execution cannot proceed without this critical piece of information. If the analysis type is specified, it returns True, allowing the execution flow to continue to the dispatching of the appropriate analysis method based on the specified type. This check ensures that users have provided the necessary information to identify which analysis they want to perform before any processing begins. 
        
        Parameters:
            config (MPASConfig): The configuration object containing the parameters for the analysis, which should include the 'analysis_type' attribute that specifies which type of analysis to execute. 

        Returns:
            bool: True if the 'analysis_type' attribute is specified and valid in the configuration object, False if it is missing or empty, indicating that execution cannot proceed without this information. 
        """
        if not hasattr(config, 'analysis_type') or not config.analysis_type:
            self._log_error("No analysis type specified")
            return False
        return True
    
    def _dispatch_analysis(self: "MPASUnifiedCLI", 
                           analysis_type: Optional[str], 
                           config: MPASConfig) -> Optional[bool]:
        """
        This method dispatches the execution to the appropriate analysis method based on the specified analysis type in the configuration. It defines a mapping of known analysis types and their aliases to corresponding methods that implement the specific analysis workflows (e.g., precipitation, surface variables, wind vectors, cross-sections, soundings, overlays). The method iterates through this mapping to find a match for the provided analysis type and calls the corresponding method with the configuration object if a match is found. If no known analysis type matches the provided input, it returns None, indicating that the analysis type is unknown and no method was executed. This dispatching mechanism allows for flexible execution of different analysis workflows based on user input while maintaining a clean and organized structure for handling multiple types of analyses within the unified CLI framework. 

        Parameters:
            analysis_type (Optional[str]): The type of analysis to execute, which may be None if not specified. This string is used to determine which analysis method to call based on predefined mappings of analysis types and their aliases.
            config (MPASConfig): The configuration object containing all parameters for the analysis, which will be passed to the specific analysis method that is executed based on the analysis type. 

        Returns:
            Optional[bool]: The result of the executed analysis method (True for success, False for failure) if a known analysis type is matched and executed, or None if the analysis type is unknown and no method was executed. 
        """
        if not analysis_type:
            return None
        
        analysis_map = {
            ('precipitation', 'precip', 'rain'): self._run_precipitation_analysis,
            ('surface', 'surf', '2d'): self._run_surface_analysis,
            ('wind', 'vector', 'winds'): self._run_wind_analysis,
            ('cross', 'xsec', '3d', 'vertical'): self._run_cross_analysis,
            ('sounding', 'skewt', 'profile'): self._run_sounding_analysis,
            ('overlay', 'complex', 'multi', 'composite'): self._run_overlay_analysis,
        }
        
        for type_aliases, analysis_method in analysis_map.items():
            if analysis_type in type_aliases:
                return analysis_method(config)
        
        return None
    
    def _log_error(self: "MPASUnifiedCLI", 
                   message: str, 
                   include_traceback: bool = False) -> None:
        """
        This method logs an error message using the configured logger if available, or prints it to the console if no logger is set up. It also has an option to include the full traceback of the current exception in the log output for detailed debugging information if the 'include_traceback' flag is set to True. This method centralizes error logging and reporting, ensuring that users receive consistent and informative feedback on any issues that occur during execution, along with optional detailed traceback information when verbose output is enabled. 
        
        Parameters:
            message (str): The error message to log or print, describing the specific issue that occurred during execution.
            include_traceback (bool): A flag indicating whether to include the full traceback of the current exception in the log output for detailed debugging information. If True, the traceback will be included; if False, only the error message will be logged or printed. 
        
        Returns:
            None
        """
        if self.logger:
            self.logger.error(message)
            if include_traceback:
                import traceback
                self.logger.error(traceback.format_exc())
    
    def _print_performance_summary(self: "MPASUnifiedCLI", 
                                   config: MPASConfig) -> None:
        """
        This method prints a performance summary of the analysis execution if the logger is configured and the 'verbose' flag is set in the configuration object. It checks if the logger is available and if the configuration has a 'verbose' attribute that is True. If both conditions are met, it calls the 'print_summary' method of the PerformanceMonitor instance to output a summary of the execution times for various stages of the analysis. This allows users to see detailed performance metrics when they have enabled verbose output, providing insights into how long different parts of the analysis took to execute. If either condition is not met (i.e., no logger or verbose mode not enabled), this method will not print anything, ensuring that performance information is only displayed when requested by the user. 

        Parameters:
            config (MPASConfig): The configuration object containing user preferences for logging verbosity (e.g., 'verbose') that will be used to determine whether to print the performance summary. 

        Returns:
            None
        """
        if self.logger and hasattr(config, 'verbose') and config.verbose:
            if self.perf_monitor is not None:
                self.perf_monitor.print_summary()
    
    def run_analysis(self: "MPASUnifiedCLI", 
                     config: MPASConfig) -> bool:
        """
        This method serves as the main entry point for executing the analysis based on the provided configuration. It first initializes a PerformanceMonitor to track execution times, then checks if the analysis type is specified in the configuration. If not, it logs an error and returns False. If the analysis type is specified, it dispatches the execution to the appropriate analysis method based on the analysis type. If the analysis type is unknown, it logs an error and returns False. After executing the analysis, it prints a performance summary if verbose mode is enabled. The method returns True if the analysis completes successfully without errors, or False if any issues occur during execution or if the analysis type is unknown. It also includes exception handling to catch KeyboardInterrupt for user-initiated interruptions and general exceptions for any other errors that may occur during execution, logging appropriate error messages in each case.   

        Parameters:
            config (MPASConfig): The configuration object containing all parameters for the analysis, including the specified analysis type, file paths, spatial bounds, output preferences, and any other settings needed to execute the analysis. 

        Returns:
            bool: True if the analysis completes successfully without errors, False if any issues occur during execution (e.g., unknown analysis type, exceptions) or if the analysis type is not specified. 
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
    
    def _run_precipitation_batch(self: "MPASUnifiedCLI",
                                 processor: "MPAS2DProcessor",
                                 plotter: "MPASPrecipitationPlotter",
                                 config: MPASConfig,) -> None:
        """
        This method executes the batch precipitation analysis workflow for MPAS data, handling the process of generating multiple precipitation maps across a range of time steps or spatial bounds based on user-specified parameters. It first checks if parallel processing is enabled in the configuration and, if so, it utilizes the ParallelPrecipitationProcessor to create batch precipitation maps in parallel, logging the use of parallel processing. If parallel processing is not enabled, it calls the create_batch_precipitation_maps method of the plotter to generate the maps sequentially. The method passes all necessary parameters to these functions, including spatial bounds, variable name, accumulation period, plot type, grid resolution, and output formats. After the maps are created, it logs the number of generated maps for user confirmation. This method is designed to be called when batch processing is enabled in the configuration, allowing for efficient generation of multiple precipitation maps based on the specified criteria.

        Parameters:
            processor (MPAS2DProcessor): An instance of the MPAS2DProcessor class that has been initialized with the grid and loaded with the 2D diagnostic data, which will be used to extract coordinates and compute precipitation differences for the specified variable and time range in batch mode.
            plotter (MPASPrecipitationPlotter): An instance of the MPASPrecipitationPlotter class that will be used to create and save the precipitation map visualizations based on the computed precipitation differences and user-specified parameters for plotting (e.g., spatial bounds, plot type, grid resolution) in batch mode.
            config (MPASConfig): The configuration object containing all parameters for the batch precipitation analysis, including the variable name, spatial bounds, accumulation period, plot type, grid resolution, output preferences, and any other settings needed to execute the batch analysis across multiple time steps or spatial criteria.

        Returns:
            None
        """
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

    def _run_precipitation_single(self: "MPASUnifiedCLI",
                                  processor: "MPAS2DProcessor",
                                  plotter: "MPASPrecipitationPlotter",
                                  dataset: Any, 
                                  config: MPASConfig,) -> None:
        """
        This method executes the precipitation analysis workflow for a single timestep of MPAS data, handling the process of extracting coordinates, computing precipitation differences, and generating a visualization based on user-specified parameters. It uses the provided MPAS2DProcessor to extract longitude and latitude coordinates for the specified variable, computes the precipitation difference using the PrecipitationDiagnostics class, and creates a precipitation map with appropriate titles and formatting. The method ensures that output directories are created as needed and saves the visualization in the requested formats. It also logs the output path of the saved plot for user confirmation. This method is designed to be called when batch processing is not enabled, allowing for focused analysis on a single timestep of data.

        Parameters:
            processor (MPAS2DProcessor): An instance of the MPAS2DProcessor class that has been initialized with the grid and loaded with the 2D diagnostic data, which will be used to extract coordinates and compute precipitation differences for the specified variable and time index.
            plotter (MPASPrecipitationPlotter): An instance of the MPASPrecipitationPlotter class that will be used to create and save the precipitation map visualization based on the computed precipitation differences and user-specified parameters for plotting (e.g., spatial bounds, plot type, grid resolution).
            dataset (Any): The dataset object containing the loaded MPAS data, which will be used to extract time information for the plot title and to access the variable data needed for precipitation difference calculations.
            config (MPASConfig): The configuration object containing all parameters for the precipitation analysis, including the variable name, time index, spatial bounds, accumulation period, plot type, grid resolution, output preferences, and any other settings needed to execute the analysis for a single timestep.

        Returns:
            None
        """
        lon, lat = processor.extract_2d_coordinates_for_variable(config.variable)

        precip_diag = PrecipitationDiagnostics(verbose=config.verbose)
        precip_data = precip_diag.compute_precipitation_difference(
            dataset, config.time_index, config.variable, config.accumulation_period,
            data_type=getattr(processor, 'data_type', 'UXarray')
        )

        time_str = MPASDateTimeUtils.get_time_info(dataset, config.time_index, verbose=False)

        plotter.create_precipitation_map(
            lon, lat, precip_data.values,
            config.lon_min, config.lon_max,
            config.lat_min, config.lat_max,
            title=config.title or f"MPAS Precipitation | Variable: {config.variable} | Valid: {time_str}",
            accum_period=config.accumulation_period,
            plot_type=getattr(config, 'plot_type', 'scatter'),
            grid_resolution=getattr(config, 'grid_resolution', None)
        )

        output_path = config.output or os.path.join(
            config.output_dir,
            f"mpas_precipitation_{config.variable}_{time_str}"
        )

        plotter.save_plot(output_path, formats=config.output_formats or ['png'])
        plotter.close_plot()

        if self.logger:
            self.logger.info(f"Precipitation plot saved: {output_path}")

    def _run_precipitation_analysis(self: "MPASUnifiedCLI", 
                                    config: MPASConfig) -> bool:
        """
        This method executes the precipitation analysis workflow for MPAS data, handling the complete process of loading 2D diagnostic data, computing precipitation differences, and generating visualizations based on user-specified parameters. It initializes an MPAS2DProcessor to load the grid and data, creates an MPASPrecipitationPlotter for visualization, and supports both single-timestep and batch-mode processing. In batch mode, it can utilize ParallelPrecipitationProcessor for efficient generation of multiple precipitation maps. For single-timestep analysis, it extracts longitude and latitude coordinates, computes the precipitation difference using PrecipitationDiagnostics, and creates a precipitation map with appropriate titles and formatting. The method ensures that output directories are created as needed and saves visualizations in the requested formats. It also logs the number of generated maps for user confirmation. The method returns True if the analysis completes successfully without errors, or False if any issues occur during processing or plotting. 

        Parameters:
            config (MPASConfig): Configuration object containing parameters for precipitation analysis, including grid file path, data directory, variable name, spatial bounds, accumulation period, plot type, gridding options, output preferences, and flags for batch processing and parallel execution. 

        Returns:
            bool: True if precipitation analysis completes successfully, False if any errors occur during data loading, processing, or plotting. 
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
                self._run_precipitation_batch(processor, plotter, config)
            else:
                self._run_precipitation_single(processor, plotter, dataset, config)

        return True
    
    def _run_surface_batch(self: "MPASUnifiedCLI",
                           processor: "MPAS2DProcessor",
                           plotter: "MPASSurfacePlotter",
                           config: MPASConfig,) -> None:
        """
        This method executes the batch surface variable analysis workflow for MPAS data, handling the process of generating multiple surface maps across a range of time steps or spatial bounds based on user-specified parameters. It first checks if parallel processing is enabled in the configuration and, if so, it utilizes the ParallelSurfaceProcessor to create batch surface maps in parallel, logging the use of parallel processing. If parallel processing is not enabled, it calls the create_batch_surface_maps method of the plotter to generate the maps sequentially. The method passes all necessary parameters to these functions, including spatial bounds, variable name, plot type, grid resolution, and output formats. After the maps are created, it logs the number of generated maps for user confirmation. This method is designed to be called when batch processing is enabled in the configuration, allowing for efficient generation of multiple surface maps based on the specified criteria.

        Parameters:
            processor (MPAS2DProcessor): An instance of the MPAS2DProcessor class that has been initialized with the grid and loaded with the 2D diagnostic data, which will be used to extract coordinates and variable data for the specified variable and time range in batch mode.
            plotter (MPASSurfacePlotter): An instance of the MPASSurfacePlotter class that will be used to create and save the surface map visualizations based on the extracted variable data and user-specified parameters for plotting (e.g., spatial bounds, plot type, grid resolution) in batch mode.
            config (MPASConfig): The configuration object containing all parameters for the batch surface analysis, including the variable name, spatial bounds, plot type, grid resolution, output preferences, and any other settings needed to execute the batch analysis across multiple time steps or spatial criteria.

        Returns:
            None
        """
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

    def _run_surface_single(self: "MPASUnifiedCLI",
                            processor: "MPAS2DProcessor",
                            plotter: "MPASSurfacePlotter",
                            dataset: Any,
                            config: MPASConfig,) -> None:
        """
        This method executes the surface variable analysis workflow for a single timestep of MPAS data, handling the process of extracting coordinates, retrieving variable data, and generating a visualization based on user-specified parameters. It uses the provided MPAS2DProcessor to extract longitude and latitude coordinates for the specified variable, retrieves the 2D variable data for the specified time index, and creates a surface map with appropriate titles and formatting. The method ensures that output directories are created as needed and saves the visualization in the requested formats. It also logs the output path of the saved plot for user confirmation. This method is designed to be called when batch processing is not enabled, allowing for focused analysis on a single timestep of data.

        Parameters:
            processor (MPAS2DProcessor): An instance of the MPAS2DProcessor class that has been initialized with the grid and loaded with the 2D diagnostic data, which will be used to extract coordinates and variable data for the specified variable and time index.
            plotter (MPASSurfacePlotter): An instance of the MPASSurfacePlotter class that will be used to create and save the surface map visualization based on the extracted variable data and user-specified parameters for plotting (e.g., spatial bounds, plot type, grid resolution).
            dataset (Any): The dataset object containing the loaded MPAS data, which will be used to extract time information for the plot title and to access the variable data needed for creating the surface map.
            config (MPASConfig): The configuration object containing all parameters for the surface analysis, including the variable name, time index, spatial bounds, plot type, grid resolution, output preferences, and any other settings needed to execute the analysis for a single timestep.

        Returns:
            None
        """
        var_data = processor.get_2d_variable_data(config.variable, config.time_index)
        lon, lat = processor.extract_2d_coordinates_for_variable(config.variable, var_data)

        time_str = MPASDateTimeUtils.get_time_info(dataset, config.time_index, verbose=False)

        plotter.create_surface_map(
            lon, lat, var_data.values,
            config.variable,
            config.lon_min, config.lon_max,
            config.lat_min, config.lat_max,
            title=config.title or f"MPAS Surface | Variable: {config.variable} | Valid: {time_str}",
            plot_type=config.plot_type,
            colormap=config.colormap if config.colormap != 'default' else None,
            clim_min=config.clim_min,
            clim_max=config.clim_max,
            data_array=var_data
        )

        output_path = config.output or os.path.join(
            config.output_dir,
            f"mpas_surface_{config.variable}_{config.plot_type}_{time_str}"
        )

        plotter.save_plot(output_path, formats=config.output_formats or ['png'])
        plotter.close_plot()

        if self.logger:
            self.logger.info(f"Surface plot saved: {output_path}")

    def _run_surface_analysis(self: "MPASUnifiedCLI", 
                              config: MPASConfig) -> bool:
        """
        This method executes the surface variable analysis workflow for MPAS data, managing the entire process of loading 2D diagnostic data, extracting the specified variable, and generating visualizations based on user-specified parameters. It initializes an MPAS2DProcessor to load the grid and data, creates an MPASSurfacePlotter for visualization, and supports both single-timestep and batch-mode processing. In batch mode, it can utilize ParallelSurfaceProcessor for efficient generation of multiple surface maps. For single-timestep analysis, it extracts longitude and latitude coordinates, retrieves the specified variable data, and creates a surface map with appropriate titles and formatting. The method ensures that output directories are created as needed and saves visualizations in the requested formats. It also logs the number of generated maps for user confirmation. The method returns True if the analysis completes successfully without errors, or False if any issues occur during processing or plotting. 

        Parameters:
            config (MPASConfig): Configuration object containing parameters for surface analysis, including grid file path, data directory, variable name, spatial bounds, plot type, gridding options, output preferences, and flags for batch processing and parallel execution. 

        Returns:
            bool: True if surface analysis completes successfully, False if any errors occur during data loading, processing, or plotting. 
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
                self._run_surface_batch(processor, plotter, config)
            else:
                self._run_surface_single(processor, plotter, dataset, config)

        return True
    
    def _run_wind_batch(self: "MPASUnifiedCLI",
                        processor: "MPAS2DProcessor",
                        plotter: "MPASWindPlotter",
                        config: MPASConfig,) -> None:
        """
        This method executes the batch wind vector analysis workflow for MPAS data, handling the process of generating multiple wind plots across a range of time steps or spatial bounds based on user-specified parameters. It first checks if parallel processing is enabled in the configuration and, if so, it utilizes the ParallelWindProcessor to create batch wind plots in parallel, logging the use of parallel processing. If parallel processing is not enabled, it calls the create_batch_wind_plots method of the plotter to generate the plots sequentially. The method passes all necessary parameters to these functions, including spatial bounds, variable names for u and v components, plot type, gridding options, and output formats. After the plots are created, it logs the number of generated plots for user confirmation. This method is designed to be called when batch processing is enabled in the configuration, allowing for efficient generation of multiple wind plots based on the specified criteria.

        Parameters:
            processor (MPAS2DProcessor): An instance of the MPAS2DProcessor class that has been initialized with the grid and loaded with the 2D diagnostic data, which will be used to extract coordinates and variable data for the specified u and v wind components across multiple time steps or spatial criteria in batch mode.
            plotter (MPASWindPlotter): An instance of the MPASWindPlotter class that will be used to create and save the wind plot visualizations based on the extracted variable data for the u and v wind components and user-specified parameters for plotting (e.g., spatial bounds, plot type, gridding options) in batch mode.
            config (MPASConfig): The configuration object containing all parameters for the batch wind analysis, including the variable names for the u and v wind components, spatial bounds, plot type, gridding options, output preferences, and any other settings needed to execute the batch analysis across multiple time steps or spatial criteria.

        Returns:
            None
        """
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

    def _run_wind_single(self: "MPASUnifiedCLI",
                         processor: "MPAS2DProcessor",
                         plotter: "MPASWindPlotter",
                         config: MPASConfig,) -> None:
        """
        This method executes the wind vector analysis workflow for a single timestep of MPAS data, handling the process of extracting coordinates, retrieving u and v wind component data, and generating a visualization based on user-specified parameters. It uses the provided MPAS2DProcessor to extract longitude and latitude coordinates for the specified u variable, retrieves the 2D variable data for both the u and v wind components for the specified time index, and creates a wind plot with appropriate titles and formatting. The method ensures that output directories are created as needed and saves the visualization in the requested formats. It also logs the output path of the saved plot for user confirmation. This method is designed to be called when batch processing is not enabled, allowing for focused analysis on a single timestep of data.

        Parameters:
            processor (MPAS2DProcessor): An instance of the MPAS2DProcessor class that has been initialized with the grid and loaded with the 2D diagnostic data, which will be used to extract coordinates and variable data for the specified u and v wind components and time index.
            plotter (MPASWindPlotter): An instance of the MPASWindPlotter class that will be used to create and save the wind plot visualization based on the extracted variable data for the u and v wind components and user-specified parameters for plotting (e.g., spatial bounds, plot type, gridding options).
            config (MPASConfig): The configuration object containing all parameters for the wind analysis, including the variable names for the u and v wind components, time index, spatial bounds, plot type, gridding options, output preferences, and any other settings needed to execute the analysis for a single timestep.

        Returns:
            None
        """
        u_data = processor.get_2d_variable_data(config.u_variable, config.time_index)
        v_data = processor.get_2d_variable_data(config.v_variable, config.time_index)
        lon, lat = processor.extract_2d_coordinates_for_variable(config.u_variable, u_data)

        plotter.create_wind_plot(
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

        time_str = MPASDateTimeUtils.get_time_info(processor.dataset, config.time_index, verbose=False)

        output_path = config.output or os.path.join(
            config.output_dir,
            f"mpas_wind_{config.u_variable}_{config.v_variable}_{config.wind_plot_type}_valid_{time_str}"
        )

        plotter.save_plot(output_path, formats=config.output_formats or ['png'])
        plotter.close_plot()

        if self.logger:
            self.logger.info(f"Wind plot saved: {output_path}")

    def _run_wind_analysis(self: "MPASUnifiedCLI", 
                           config: MPASConfig) -> bool:
        """
        This method executes the wind vector analysis workflow for MPAS data, handling the complete process of loading 2D diagnostic data, extracting the specified u and v wind components, and generating visualizations based on user-specified parameters. It initializes an MPAS2DProcessor to load the grid and data, creates an MPASWindPlotter for visualization, and supports both single-timestep and batch-mode processing. In batch mode, it can utilize ParallelWindProcessor for efficient generation of multiple wind plots. For single-timestep analysis, it extracts longitude and latitude coordinates, retrieves the u and v wind component data, and creates a wind plot with appropriate titles and formatting. The method ensures that output directories are created as needed and saves visualizations in the requested formats. It also logs the number of generated plots for user confirmation. The method returns True if the analysis completes successfully without errors, or False if any issues occur during processing or plotting. 

        Parameters:
            config (MPASConfig): Configuration object containing parameters for wind analysis, including grid file path, data directory, u and v variable names, spatial bounds, plot type, gridding options, output preferences, and flags for batch processing and parallel execution. 

        Returns:
            bool: True if wind analysis completes successfully, False if any errors occur during data loading, processing, or plotting. 
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        with self.perf_monitor.timer("Wind analysis"):
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_2d_data(config.data_dir)

            plotter = MPASWindPlotter(
                figsize=config.figure_size,
                dpi=config.dpi
            )

            os.makedirs(config.output_dir, exist_ok=True)

            if config.batch_mode:
                self._run_wind_batch(processor, plotter, config)
            else:
                self._run_wind_single(processor, plotter, config)

        return True
    
    def _validate_cross_section_coordinates(self: "MPASUnifiedCLI", 
                                            config: MPASConfig) -> None:
        """
        This method validates that the necessary coordinates for defining a vertical cross-section are specified in the configuration object when the analysis type is set to a cross-section analysis. It checks that both the start and end longitude and latitude coordinates are provided and not None. If any of these required coordinates are missing, it raises an assertion error with a message indicating which specific coordinate is missing. This validation ensures that users have provided all necessary information to define the start and end points of the vertical cross-section before attempting to execute the cross-section analysis, preventing runtime errors during processing due to missing parameters. 

        Parameters:
            config (MPASConfig): The configuration object containing parameters for the analysis, which should include the start and end longitude and latitude coordinates for defining the vertical cross-section when the analysis type is set to a cross-section analysis. 

        Returns:
            None
        """
        assert config.start_lon is not None and config.start_lat is not None, \
            "Cross-section start coordinates must be specified"

        assert config.end_lon is not None and config.end_lat is not None, \
            "Cross-section end coordinates must be specified"
    
    def _extract_cross_section_params(self: "MPASUnifiedCLI", 
                                      config: MPASConfig) -> Dict[str, Any]:
        """
        This method extracts the necessary parameters for performing a vertical cross-section analysis from the provided configuration object and organizes them into a dictionary format that can be easily used in the plotting functions. It retrieves the variable name, start and end coordinates, vertical coordinate type, number of points along the cross-section, and maximum height for the plot from the configuration object. The method assumes that these parameters have already been validated to ensure that they are present and valid for cross-section analysis. By centralizing the extraction of these parameters into a single method, it promotes cleaner code and easier maintenance, allowing the main analysis workflow to focus on processing and plotting logic while relying on this helper method to provide the necessary parameters in a consistent format. 

        Parameters:
            config (MPASConfig): The configuration object containing all parameters for the cross-section analysis, including variable name, start and end coordinates, vertical coordinate type, number of points, and maximum height. These parameters should be validated before calling this method to ensure they are present and valid for cross-section analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted parameters for cross-section analysis, organized in a format that can be easily used in the plotting functions. The dictionary includes keys such as 'var_name', 'start_point', 'end_point', 'vertical_coord', 'num_points', and 'max_height' with corresponding values extracted from the configuration object. 
        """
        return {
            'var_name': config.variable,
            'start_point': (config.start_lon, config.start_lat),
            'end_point': (config.end_lon, config.end_lat),
            'vertical_coord': config.vertical_coord,
            'num_points': config.num_points,
            'max_height': config.max_height,
        }
    
    def _run_batch_cross_sections(self: "MPASUnifiedCLI",
                                  processor: 'MPAS3DProcessor', 
                                  plotter: 'MPASVerticalCrossSectionPlotter', 
                                  config: MPASConfig, 
                                  params: Dict[str, Any]) -> Optional[List[str]]:
        """
        This method executes the workflow for creating multiple vertical cross-section plots in batch mode based on the provided configuration and parameters. It checks if parallel processing is enabled in the configuration and, if so, it calls the `ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel` method to generate the cross-section plots using multiple processes for improved performance. If parallel processing is not enabled, it calls the `plotter.create_batch_cross_section_plots` method to generate the plots sequentially. The method returns a list of file paths for the created cross-section plots if generation was successful, or None if no files were created or if an error occurred during processing. This method allows for efficient generation of multiple cross-section visualizations based on user specifications while providing flexibility in execution mode (parallel vs. serial) depending on user preferences and system capabilities.   

        Parameters:
            processor (MPAS3DProcessor): An instance of the MPAS3DProcessor class that has loaded the necessary 3D atmospheric data for analysis.
            plotter (MPASVerticalCrossSectionPlotter): An instance of the MPASVerticalCrossSectionPlotter class used for creating vertical cross-section visualizations.
            config (MPASConfig): The configuration object containing all necessary parameters for batch cross-section analysis, including output directory, output formats, batch mode flag, and any other relevant settings for generating multiple cross-section plots.
            params (Dict[str, Any]): A dictionary of parameters extracted from the configuration that are needed for creating the cross-section plots, such as variable name, start and end coordinates, vertical coordinate type, number of points, and maximum height. 

        Returns:
            Optional[List[str]]: A list of file paths for the created cross-section plots if generation was successful, or None if no files were created or if an error occurred during processing. 
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
    
    def _run_single_cross_section(self: "MPASUnifiedCLI", 
                                  processor: 'MPAS3DProcessor', 
                                  plotter: 'MPASVerticalCrossSectionPlotter', 
                                  config: MPASConfig, 
                                  params: Dict[str, Any]) -> None:
        """
        This method executes the workflow for creating a single vertical cross-section plot based on the provided configuration and parameters. It retrieves the time information for the specified time index from the processor's dataset to include in the plot title. It then calls the `plotter.create_vertical_cross_section` method to generate the cross-section plot using the provided processor, time index, title, colormap, and other parameters extracted from the configuration. After creating the plot, it constructs the output file path based on the configuration and saves the plot in the requested formats. Finally, it logs a message indicating that the cross-section plot has been saved. This method handles all steps necessary to create and save a single vertical cross-section visualization based on user specifications. 

        Parameters:
            processor (MPAS3DProcessor): An instance of the MPAS3DProcessor class that has loaded the necessary 3D atmospheric data for analysis.
            plotter (MPASVerticalCrossSectionPlotter): An instance of the MPASVerticalCrossSectionPlotter class used for creating vertical cross-section visualizations.
            config (MPASConfig): The configuration object containing all necessary parameters for single cross-section analysis, including time index, title, colormap, output directory, output formats, and any other relevant settings for generating the cross-section plot.
            params (Dict[str, Any]): A dictionary of parameters extracted from the configuration that are needed for creating the cross-section plot, such as variable name, start and end coordinates, vertical coordinate type, number of points, and maximum height.

        Returns:
            None
        """
        time_str = MPASDateTimeUtils.get_time_info(processor.dataset, config.time_index, verbose=False)
        
        _, _ = plotter.create_vertical_cross_section(
            mpas_3d_processor=processor,
            time_index=config.time_index,
            title=config.title or f"MPAS Vertical Cross-Section | Variable: {config.variable} | Valid: {time_str}",
            colormap=config.colormap if config.colormap != 'default' else None,
            **params
        )
        
        output_path = config.output or os.path.join(
            config.output_dir,
            f"mpas_cross_section_{config.variable}_{time_str}"
        )
        
        plotter.save_plot(output_path, formats=config.output_formats or ['png'])
        plotter.close_plot()
        
        if self.logger:
            self.logger.info(f"Cross-section plot saved: {output_path}")
    
    def _log_created_files(self: "MPASUnifiedCLI", 
                           created_files: Optional[List[str]], 
                           file_type: str = "plots") -> None:
        """
        This method logs a message indicating the number of files created during the analysis process, providing context about the type of files generated. It checks if the logger is configured and if the list of created files is not None or empty. If both conditions are met, it logs an informational message that includes the count of created files and the specified file type (e.g., "Created 5 cross-section plots"). This method helps provide feedback to users about the results of the analysis, confirming how many visualizations or output files were successfully generated based on their configuration. If there are no created files or if the logger is not available, this method will not log anything. 

        Parameters:
            created_files (Optional[List[str]]): A list of file paths for the created files during the analysis process. This can be None or empty if no files were created or if an error occurred during processing.
            file_type (str): A string describing the type of files created (e.g., "cross-section plots", "Skew-T diagrams") to provide context in the log message. Default is "plots".

        Returns:
            None
        """
        if self.logger and created_files:
            self.logger.info(f"Created {len(created_files)} {file_type}")
    
    def _run_cross_analysis(self: "MPASUnifiedCLI", 
                            config: MPASConfig) -> bool:
        """
        This method executes the vertical cross-section analysis workflow for MPAS data, managing the entire process of loading 3D atmospheric data, validating cross-section coordinates, extracting necessary parameters, and generating visualizations based on user-specified configuration. It initializes an MPAS3DProcessor to load the grid and 3D data, creates an MPASVerticalCrossSectionPlotter for visualization, and ensures that output directories are created as needed. The method supports both single-timestep and batch-mode processing for cross-section analysis. In batch mode, it generates multiple cross-section plots across specified time indices using either parallel or serial processing based on user preferences. In single-timestep mode, it creates a single cross-section plot for the specified time index. The method also handles logging of created files and returns True if the analysis completes successfully without errors, or False if any issues occur during processing or plotting. 

        Parameters:
            config (MPASConfig): The configuration object containing all necessary parameters for cross-section analysis, including grid file path, data directory, variable name, start and end coordinates, vertical coordinate type, number of points, maximum height, output directory, output formats, batch mode flag, and any other relevant settings for generating cross-section visualizations. 

        Returns:
            bool: True if cross-section analysis completes successfully, False if any errors occur during data loading, processing, or plotting. 
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
    
    def _compute_sounding_indices(self: "MPASUnifiedCLI",
                                  show_indices: bool,
                                  sounding_diag: "SoundingDiagnostics",
                                  profile: Dict[str, Any],) -> Optional[Dict[str, Any]]:
        """
        This method computes thermodynamic indices for a sounding profile if the user has requested to show indices in the Skew-T diagram. It checks the `show_indices` flag, and if it is False, it returns None, indicating that no indices should be computed or displayed. If `show_indices` is True, it calls the `compute_thermodynamic_indices` method of the `SoundingDiagnostics` instance, passing in the necessary profile data such as pressure, temperature, dewpoint, and optionally wind components and height if they are available in the profile. The computed indices are returned as a dictionary that can be used in the plotting function to display relevant thermodynamic information on the Skew-T diagram. This method centralizes the logic for determining whether to compute indices and how to compute them based on the provided sounding profile data.

        Parameters:
            show_indices (bool): A flag indicating whether the user has requested to compute and display thermodynamic indices on the Skew-T diagram. If False, this method will return None and no indices will be computed or displayed.
            sounding_diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class that provides the method for computing thermodynamic indices based on the sounding profile data.
            profile (Dict[str, Any]): A dictionary containing the sounding profile data, including pressure, temperature, dewpoint, and optionally wind components and height. This data will be used to compute the thermodynamic indices if `show_indices` is True.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the computed thermodynamic indices if `show_indices` is True, or None if `show_indices` is False and no indices should be computed or displayed. The dictionary of indices can include values such as CAPE, CIN, Lifted Index, K-Index, etc., depending on the implementation of the `compute_thermodynamic_indices` method in the SoundingDiagnostics class.
        """
        if not show_indices:
            return None
        return sounding_diag.compute_thermodynamic_indices(
            profile['pressure'], profile['temperature'], profile['dewpoint'],
            u_wind_kt=profile.get('u_wind'),
            v_wind_kt=profile.get('v_wind'),
            height_m=profile.get('height'),
        )

    def _build_skewt_tags(self: "MPASUnifiedCLI",
                          profile: Dict[str, Any],) -> tuple:
        """
        This method constructs longitude and latitude tags for the Skew-T diagram title based on the station longitude and latitude from the sounding profile. It formats the longitude and latitude values to two decimal places and appends the appropriate directional indicators (E/W for longitude and N/S for latitude) based on the sign of the coordinates. The resulting tags are returned as a tuple of strings that can be used in the plot title to indicate the location of the sounding profile being visualized.

        Parameters:
            profile (Dict[str, Any]): A dictionary containing the sounding profile data, including station longitude and latitude.

        Returns:
            tuple: A tuple containing the longitude and latitude tags as strings.
        """
        stn_lon = profile['station_lon']
        stn_lat = profile['station_lat']
        lon_tag = f"{abs(stn_lon):.2f}{'W' if stn_lon < 0 else 'E'}"
        lat_tag = f"{abs(stn_lat):.2f}{'S' if stn_lat < 0 else 'N'}"
        return lon_tag, lat_tag

    def _run_sounding_batch(self: "MPASUnifiedCLI",
                            processor: "MPAS3DProcessor",
                            plotter: "MPASSkewTPlotter",
                            sounding_diag: "SoundingDiagnostics",
                            config: MPASConfig,
                            sounding_lon: float,
                            sounding_lat: float,
                            show_indices: bool,
                            show_parcel: bool,) -> None:
        """
        This method handles the batch processing of sounding profiles for MPAS data, generating Skew-T diagrams across a range of time indices for a specified sounding location. It determines the appropriate time dimension in the dataset, calculates the number of time steps available, and sets the start and end time indices based on the configuration. It then iterates over the specified time range, extracting the sounding profile for each time index at the given longitude and latitude. For each profile, it computes thermodynamic indices if requested, builds longitude and latitude tags for the plot title, and creates a Skew-T diagram using the plotter. The generated plots are saved with filenames that include the location and valid time information. Finally, it logs the created files for user confirmation. This method allows users to efficiently generate multiple Skew-T diagrams for a specific location across different time steps in their MPAS dataset.

        Parameters:
            processor (MPAS3DProcessor): An instance of the MPAS3DProcessor class that has loaded the necessary 3D atmospheric data for analysis.
            plotter (MPASSkewTPlotter): An instance of the MPASSkewTPlotter class used for creating Skew-T diagram visualizations.
            sounding_diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class that provides methods for extracting sounding profiles and computing thermodynamic indices.
            config (MPASConfig): The configuration object containing all necessary parameters for sounding analysis, including time range settings, output directory, figure size, DPI, verbosity, and any other relevant settings for generating Skew-T diagrams in batch mode.
            sounding_lon (float): The longitude of the sounding location for which to extract profiles and generate Skew-T diagrams.
            sounding_lat (float): The latitude of the sounding location for which to extract profiles and generate Skew-T diagrams.
            show_indices (bool): A flag indicating whether to compute and display thermodynamic indices on the Skew-T diagrams.
            show_parcel (bool): A flag indicating whether to show the parcel profile on the Skew-T diagrams.

        Returns:
            None
        """
        time_dim = 'Time' if 'Time' in processor.dataset.dims else 'time'
        n_times = processor.dataset.sizes.get(time_dim, 1)
        time_start = config.time_start if config.time_start is not None else 0
        time_end = config.time_end if config.time_end is not None else (n_times - 1)
        created_files = []

        for t_idx in range(time_start, min(time_end + 1, n_times)):
            profile = sounding_diag.extract_sounding_profile(
                processor, sounding_lon, sounding_lat, time_index=t_idx,
            )
            indices = self._compute_sounding_indices(show_indices, sounding_diag, profile)
            time_str = MPASDateTimeUtils.get_time_info(processor.dataset, t_idx, verbose=False)
            lon_tag, lat_tag = self._build_skewt_tags(profile)

            title = f"MPAS Skew-T | {lon_tag}, {lat_tag} | Valid: {time_str}"
            save_path = os.path.join(
                config.output_dir,
                f"mpas_skewt_{lon_tag.replace('.', 'p')}_{lat_tag.replace('.', 'p')}_valid_{time_str}"
            )

            plotter.create_skewt_diagram(
                pressure=profile['pressure'],
                temperature=profile['temperature'],
                dewpoint=profile['dewpoint'],
                u_wind=profile['u_wind'],
                v_wind=profile['v_wind'],
                title=config.title or title,
                indices=indices,
                show_parcel=show_parcel,
                save_path=save_path,
            )
            plotter.close_plot()
            created_files.append(save_path)

        self._log_created_files(created_files, "Skew-T diagrams")

    def _run_sounding_single(self: "MPASUnifiedCLI",
                             processor: "MPAS3DProcessor",
                             plotter: "MPASSkewTPlotter",
                             sounding_diag: "SoundingDiagnostics",
                             config: MPASConfig,
                             sounding_lon: float,
                             sounding_lat: float,
                             show_indices: bool,                             
                             show_parcel: bool,) -> None:
        """
        This method handles the processing of a single sounding profile for MPAS data, generating a Skew-T diagram for a specified sounding location and time index. It extracts the sounding profile at the given longitude and latitude for the specified time index, computes thermodynamic indices if requested, builds longitude and latitude tags for the plot title, and creates a Skew-T diagram using the plotter. The generated plot is saved with a filename that includes the location and valid time information. Finally, it logs the created file for user confirmation. This method allows users to generate a focused Skew-T diagram for a specific location and time step in their MPAS dataset without needing to process multiple time steps in batch mode.

        Parameters:
            processor (MPAS3DProcessor): An instance of the MPAS3DProcessor class that has loaded the necessary 3D atmospheric data for analysis.
            plotter (MPASSkewTPlotter): An instance of the MPASSkewTPlotter class used for creating Skew-T diagram visualizations.
            sounding_diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class that provides methods for extracting sounding profiles and computing thermodynamic indices.
            config (MPASConfig): The configuration object containing all necessary parameters for sounding analysis, including time index, output directory, figure size, DPI, verbosity, and any other relevant settings for generating a Skew-T diagram for a single time step.
            sounding_lon (float): The longitude of the sounding location for which to extract the profile and generate the Skew-T diagram.
            sounding_lat (float): The latitude of the sounding location for which to extract the profile and generate the Skew-T diagram.
            show_indices (bool): A flag indicating whether to compute and display thermodynamic indices on the Skew-T diagram.
            show_parcel (bool): A flag indicating whether to show the parcel profile on the Skew-T diagram.

        Returns:
            None
        """
        profile = sounding_diag.extract_sounding_profile(
            processor, sounding_lon, sounding_lat,
            time_index=config.time_index,
        )
        indices = self._compute_sounding_indices(show_indices, sounding_diag, profile)
        time_str = MPASDateTimeUtils.get_time_info(
            processor.dataset, config.time_index, verbose=False
        )
        lon_tag, lat_tag = self._build_skewt_tags(profile)

        title = f"MPAS Skew-T | {lon_tag}, {lat_tag} | Valid: {time_str}"
        output_name = getattr(config, 'output', None) or (
            f"mpas_skewt_{lon_tag.replace('.', 'p')}_{lat_tag.replace('.', 'p')}_valid_{time_str}"
        )
        save_path = os.path.join(config.output_dir, f"{output_name}")

        plotter.create_skewt_diagram(
            pressure=profile['pressure'],
            temperature=profile['temperature'],
            dewpoint=profile['dewpoint'],
            u_wind=profile['u_wind'],
            v_wind=profile['v_wind'],
            title=config.title or title,
            indices=indices,
            show_parcel=show_parcel,
            save_path=save_path,
        )

    def _run_sounding_analysis(self: "MPASUnifiedCLI", 
                               config: MPASConfig) -> bool:
        """
        This method executes the sounding analysis workflow for MPAS data, managing the complete process of loading 3D atmospheric data, extracting sounding profiles at specified locations and times, computing thermodynamic indices if requested, and generating Skew-T diagrams based on user-specified configuration. It initializes an MPAS3DProcessor to load the grid and 3D data, creates a SoundingDiagnostics instance for extracting profiles and computing indices, and an MPASSkewTPlotter for visualization. The method supports both single-timestep and batch-mode processing for sounding analysis. In batch mode, it generates multiple Skew-T diagrams across specified time indices for the given sounding location. In single-timestep mode, it creates a single Skew-T diagram for the specified time index. The method also handles logging of created files and returns True if the analysis completes successfully without errors, or False if any issues occur during processing or plotting. 

        Parameters:
            config (MPASConfig): The configuration object containing all necessary parameters for sounding analysis, including grid file path, data directory, output directory, figure size, DPI, verbosity, sounding location (longitude and latitude), time index or batch mode settings, flags for showing indices and parcel information, and any other relevant settings for generating Skew-T diagrams. 

        Returns:
            bool: True if sounding analysis completes successfully, False if any errors occur during data loading, processing, or plotting.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        with self.perf_monitor.timer("Sounding analysis"):
            processor = MPAS3DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_3d_data(config.data_dir)

            sounding_diag = SoundingDiagnostics(verbose=config.verbose)

            show_indices = getattr(config, 'show_indices', False)
            figsize = config.figure_size
            if show_indices:
                figsize = (figsize[0], figsize[1] + 4)

            plotter = MPASSkewTPlotter(
                figsize=figsize,
                dpi=config.dpi,
                verbose=config.verbose,
            )

            os.makedirs(config.output_dir, exist_ok=True)

            sounding_lon = getattr(config, 'sounding_lon', 0.0)
            sounding_lat = getattr(config, 'sounding_lat', 0.0)
            show_parcel = getattr(config, 'show_parcel', False)

            if config.batch_mode:
                self._run_sounding_batch(
                    processor, plotter, sounding_diag, config,
                    sounding_lon, sounding_lat, show_indices, show_parcel
                )
            else:
                self._run_sounding_single(
                    processor, plotter, sounding_diag, config,
                    sounding_lon, sounding_lat, show_indices, show_parcel
                )

        return True
    
    def _run_overlay_analysis(self: "MPASUnifiedCLI", 
                              config: MPASConfig) -> bool:
        """
        This method executes the overlay analysis workflow for MPAS data, managing the process of loading 2D diagnostic data and generating overlay visualizations based on user-specified configuration. It initializes an MPAS2DProcessor to load the grid and data, and then creates appropriate plotter instances based on the specified overlay type (e.g., precipitation + wind, temperature + pressure). The method supports different types of overlays as indicated by the `overlay_type` attribute in the configuration. It ensures that output directories are created as needed and logs messages about the progress of the analysis. Currently, this method contains placeholder implementation for the actual overlay plotting logic, and it returns True to indicate successful completion of the workflow. Future implementations will include detailed processing and plotting steps to create meaningful overlay visualizations based on the loaded MPAS data. 

        Parameters:
            config (MPASConfig): The configuration object containing all necessary parameters for overlay analysis, including grid file path, data directory, overlay type, output directory, figure size, DPI, verbosity, and any other relevant settings for generating overlay visualizations. 

        Returns:
            bool: True if overlay analysis completes successfully, False if any errors occur during data loading, processing, or plotting. 
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG
        
        with self.perf_monitor.timer("Overlay analysis"):
            if self.logger:
                self.logger.info(f"Running {config.overlay_type} overlay analysis...")
                self.logger.warning("Overlay analysis is not fully implemented yet")
            
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = processor.load_2d_data(config.data_dir)
            
            os.makedirs(config.output_dir, exist_ok=True)
            
            if config.overlay_type == 'precip_wind':
                if self.logger:
                    self.logger.info("Creating precipitation + wind overlay")
            
            elif config.overlay_type == 'temp_pressure':
                if self.logger:
                    self.logger.info("Creating temperature + pressure overlay")
            
            if self.logger:
                self.logger.info("Overlay analysis completed (placeholder implementation)")
        
        return True
    
    def _parse_args_with_fallback(self: "MPASUnifiedCLI", 
                                  parser: argparse.ArgumentParser) -> argparse.Namespace:
        """
        This method attempts to parse command-line arguments using the `parse_intermixed_args` method, which allows for more flexible ordering of global options and subcommands. If parsing with `parse_intermixed_args` fails due to an exception (which can occur in certain Python versions or with specific argument configurations), it falls back to a custom parsing strategy implemented in the `_parse_args_with_reordering` method. This fallback method manually reorders the command-line arguments to ensure that global options are processed correctly even when intermixed with subcommands, improving compatibility across different user input styles and Python versions. By implementing this two-tiered parsing approach, the CLI can provide a more robust and user-friendly experience when handling complex command-line inputs. 

        Parameters:
            parser (argparse.ArgumentParser): The argument parser instance configured with the expected command-line arguments and subcommands for the CLI. 

        Returns:
            argparse.Namespace: The parsed command-line arguments after attempting to use `parse_intermixed_args` and falling back to the custom reordering strategy if necessary. 
        """
        try:
            return parser.parse_intermixed_args()
        except Exception:
            return self._parse_args_with_reordering(parser)
    
    def _parse_args_with_reordering(self: "MPASUnifiedCLI", 
                                    parser: argparse.ArgumentParser) -> argparse.Namespace:
        """
        This method implements a custom argument parsing strategy that manually reorders command-line arguments to ensure that global options are processed correctly even when intermixed with subcommands. It identifies global options that can either have values (e.g., `--config`, `--log-file`) or be flags without values (e.g., `--verbose`, `--quiet`), and separates them from the rest of the arguments. The method then attempts to parse the reordered arguments using the provided parser. If parsing fails, it falls back to parsing the original arguments without reordering. This approach allows for greater flexibility in how users can input command-line arguments while maintaining compatibility across different Python versions and argument configurations. 

        Parameters:
            parser (argparse.ArgumentParser): The argument parser instance configured with the expected command-line arguments and subcommands for the CLI. 

        Returns:
            argparse.Namespace: The parsed command-line arguments after reordering global options and attempting to parse them with the provided parser. If parsing fails, it returns the result of parsing the original arguments without reordering. 
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
    
    def _load_and_merge_config(self: "MPASUnifiedCLI", 
                               args: argparse.Namespace) -> MPASConfig:
        """
        This method loads the configuration from a YAML file if the `--config` option is provided in the command-line arguments, and then merges any additional CLI options that are specified directly in the command line to override the corresponding settings in the configuration file. It first checks if the `config` attribute is present and not None in the parsed arguments. If a configuration file is specified, it loads the configuration using the `MPASConfig.load_from_file` method. It then calls the `parse_args_to_config` method to convert any additional CLI options into a configuration object. The method iterates through the attributes of the CLI configuration and updates the loaded configuration with any values that are not None, allowing CLI options to take precedence over configuration file settings. Finally, it returns the merged configuration instance that will be used for running the analysis. If no configuration file is specified, it simply parses the CLI arguments into a configuration object and returns it. 

        Parameters:
            args (argparse.Namespace): The parsed command-line arguments that may include a `config` attribute specifying the path to a YAML configuration file, as well as other CLI options that can override settings in the configuration file. 

        Returns:
            MPASConfig: The merged configuration instance that combines settings from the YAML configuration file (if provided) and any additional CLI options specified in the command line, with CLI options taking precedence over file settings. If no configuration file is provided, this will be a configuration object created solely from the CLI arguments. 
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
    
    def _print_verbose_output(self: "MPASUnifiedCLI", 
                              config: MPASConfig) -> None:
        """
        This method prints detailed information about the system and the active configuration settings to the logger if the `verbose` flag is set in the configuration. It calls helper methods to print system information (such as Python version, platform details, and available memory) and a summary of the configuration settings that will be used for the analysis. This verbose output can help users understand the environment in which the analysis is being run and verify that their configuration is correct before the analysis starts. If the `verbose` flag is not set or if a logger is not configured, this method does nothing, allowing for cleaner output during normal operation. 

        Parameters:
            config (MPASConfig): The configuration object containing all settings for the analysis, including a `verbose` flag that indicates whether detailed information should be printed to the logger. 

        Returns:
            None
        """
        if hasattr(config, 'verbose') and config.verbose:
            self._print_system_info()
            self._print_config_summary()
    
    def _handle_main_exception(self: "MPASUnifiedCLI", 
                               error: Exception) -> int:
        """
        This method handles any unexpected exceptions that occur during the execution of the main workflow in the `main` method. It logs the error message and stack trace using the configured logger if available, or prints them to standard output if no logger is configured. The method then returns an exit code of 1 to indicate that an error occurred during processing. This centralized exception handling ensures that users receive informative feedback about what went wrong while also allowing the CLI to exit gracefully with an appropriate status code. 

        Parameters:
            error (Exception): The exception object representing the unexpected error that occurred during the execution of the main workflow. This can be any type of exception that was not anticipated or handled by specific error handling logic within the analysis methods. 

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
        This method serves as the main entry point for the MPAS diagnostics CLI, orchestrating the entire workflow of parsing command-line arguments, loading and merging configuration settings, setting up logging, validating the configuration, printing verbose output if requested, and running the appropriate analysis based on the specified configuration. It uses a try-except block to catch any unexpected exceptions that occur during execution, allowing for graceful error handling and user feedback. The method returns an exit code where 0 indicates successful completion of the analysis with all outputs generated, 1 indicates an error occurred during processing, and 130 indicates that the analysis was interrupted by the user (e.g., via Ctrl+C). This structured approach ensures that users receive clear feedback about the outcome of their analysis while also providing robust error handling for unforeseen issues. 

        Parameters:
            None

        Returns:
            int: Exit code where 0 indicates success, 1 indicates an error, and 130 indicates interruption by the user.
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
    
    def _print_system_info(self: "MPASUnifiedCLI") -> None:
        """
        This method prints detailed information about the system environment to the logger, including the Python version, platform details, current working directory, and available memory (if the `psutil` library is installed). This information can be useful for debugging and understanding the context in which the analysis is being run. The output is formatted with a section header for readability. If a logger is not configured, this method does nothing, allowing for cleaner output during normal operation. 

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
    
    def _print_config_summary(self: "MPASUnifiedCLI") -> None:
        """
        This method prints a summary of the active configuration settings to the logger, providing users with a clear overview of the parameters that will be used for the analysis. It iterates through the attributes of the configuration object and logs any settings that are not None, allowing users to verify that their configuration is correct before the analysis starts. The output is formatted with a section header for readability. If a logger is not configured or if the configuration object is None, this method does nothing, allowing for cleaner output during normal operation. 

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
    This function serves as the main entry point for the MPAS diagnostics CLI when executed as a script. It creates an instance of the `MPASUnifiedCLI` class and calls its `main` method to execute the workflow of parsing command-line arguments, loading configuration, setting up logging, validating the configuration, and running the analysis. The function returns the exit code produced by the `main` method of the `MPASUnifiedCLI` instance, which indicates whether the analysis completed successfully (0), encountered an error (1), or was interrupted by the user (130). This structure allows for clean separation of concerns and provides a clear entry point for users executing the CLI from the command line. 

    Parameters:
        None

    Returns:
        int: Unix exit status code where 0 indicates successful analysis completion with all outputs generated, and non-zero values indicate various failure conditions (1 for errors, 130 for interruption).
    """
    cli = MPASUnifiedCLI()
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())