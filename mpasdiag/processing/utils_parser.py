#!/usr/bin/env python3

"""
MPAS Command-Line Argument Parser Utilities

This module provides comprehensive command-line argument parsing functionality for MPAS diagnostic workflows through factory methods that generate specialized ArgumentParser instances. It includes pre-configured parsers for general CLI operations, surface variable plotting, wind vector visualization, and vertical cross-section analysis, each with organized argument groups and helpful usage examples. The module features converter methods that transform parsed argparse.Namespace objects into MPASConfig data structures, enabling seamless integration between command-line interfaces and internal configuration management. This centralized parser factory approach ensures consistent CLI design across all MPASdiag tools while simplifying maintenance of argument definitions. The parsers support both direct command-line usage and configuration file input, providing flexible specification of analysis parameters for meteorological diagnostics.

Classes:
    ArgumentParser: Factory class providing static methods for creating specialized command-line parsers for different MPAS diagnostic types.

Functions:
    convert_namespace_to_config: Converts argparse.Namespace objects to MPASConfig instances for internal use.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import argparse
import textwrap
from typing import Any

from .utils_config import MPASConfig


class ArgumentParser:
    """
    Command-line argument parser factory utilities for MPAS analysis workflows with pre-configured parsers for different diagnostic types. This class provides static factory methods that produce argparse.ArgumentParser instances specialized for general CLI operations, surface variable plotting, wind vector visualization, and vertical cross-section analysis. Each factory method returns a fully configured parser with organized argument groups, sensible defaults, and helpful usage examples tailored to specific MPAS diagnostic workflows. Helper converter methods transform parsed argparse.Namespace objects into the project's MPASConfig data structure, enabling seamless integration between command-line interfaces and internal configuration management. This centralized parser factory approach ensures consistent command-line interface design across all MPASdiag tools and simplifies maintenance of argument definitions.
    """
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """
        Create a general-purpose command-line argument parser for MPAS data analysis and visualization workflows with comprehensive option groups. This factory method constructs an argparse.ArgumentParser instance pre-configured with argument groups for input/output paths, spatial extent bounds, variable selection, visualization settings, processing options, and output control flags. The parser supports both direct command-line usage and configuration file input, enabling flexible specification of analysis parameters for precipitation, surface, wind, and cross-section diagnostic plots. The returned parser includes helpful examples in the epilog, sensible defaults for common use cases, and organized argument groups for improved help message readability in the MPASdiag command-line interface.

        Returns:
            argparse.ArgumentParser: Fully configured ArgumentParser instance with groups for IO, spatial extent, variables, visualization, processing, and output control ready for parsing sys.argv or custom argument lists.
        """
        parser = argparse.ArgumentParser(
            description="MPAS Data Analysis and Visualization Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic precipitation analysis
  mpasdiag --grid-file grid.nc --data-dir ./data --output-dir ./output
  
  # Custom spatial extent
  mpasdiag --grid-file grid.nc --data-dir ./data --lat-min -10 --lat-max 15
  
  # Batch processing
  mpasdiag --grid-file grid.nc --data-dir ./data --batch-all
  
  # Use configuration file
  mpasdiag --config config.yaml
            """
        )
        
        io_group = parser.add_argument_group('Input/Output')
        io_group.add_argument('--grid-file', type=str, required=False,
                             help='Path to MPAS grid file')
        io_group.add_argument('--data-dir', type=str, required=False,
                             help='Directory containing diagnostic files')
        io_group.add_argument('--output-dir', type=str, default='./output',
                             help='Output directory for plots and results')
        io_group.add_argument('--config', type=str,
                             help='Configuration file path (YAML format)')
        
        spatial_group = parser.add_argument_group('Spatial Extent')
        spatial_group.add_argument('--lat-min', type=float, default=-9.60,
                                  help='Minimum latitude')
        spatial_group.add_argument('--lat-max', type=float, default=12.20,
                                  help='Maximum latitude')
        spatial_group.add_argument('--lon-min', type=float, default=91.00,
                                  help='Minimum longitude')
        spatial_group.add_argument('--lon-max', type=float, default=113.00,
                                  help='Maximum longitude')
        
        var_group = parser.add_argument_group('Variables')
        var_group.add_argument('--var', '--variable', type=str, default='rainnc',
                              choices=['rainc', 'rainnc', 'total'],
                              help='Precipitation variable to analyze')
        var_group.add_argument('--accum', '--accumulation', type=str, default='a01h',
                              help='Accumulation period (e.g., a01h, a24h)')
        
        viz_group = parser.add_argument_group('Visualization')
        viz_group.add_argument('--colormap', type=str, default='default',
                              help='Colormap for plots')
        viz_group.add_argument('--dpi', type=int, default=100,
                              help='Output resolution (DPI) - default: 100, use 300+ for publication quality')
        viz_group.add_argument('--figure-size', type=float, nargs=2, 
                              default=[10.0, 12.0], metavar=('WIDTH', 'HEIGHT'),
                              help='Figure size in inches')
        viz_group.add_argument('--formats', type=str, nargs='+', 
                              default=['png'], choices=['png', 'pdf', 'svg', 'eps'],
                              help='Output formats')
        viz_group.add_argument('--clim-min', type=float,
                              help='Minimum color limit')
        viz_group.add_argument('--clim-max', type=float,
                              help='Maximum color limit')
        
        proc_group = parser.add_argument_group('Processing')
        proc_group.add_argument('--data-type', type=str, default='uxarray',
                               choices=['uxarray', 'xarray'],
                               help='Data processing backend')
        proc_group.add_argument('--batch-all', action='store_true',
                               help='Process all time steps in batch mode')
        proc_group.add_argument('--time-index', type=int,
                               help='Specific time index to process')
        proc_group.add_argument('--parallel', action='store_true',
                               help='Enable parallel processing')
        
        output_group = parser.add_argument_group('Output Control')
        output_group.add_argument('--verbose', '-v', action='store_true', default=True,
                                 help='Enable verbose output')
        output_group.add_argument('--quiet', '-q', action='store_true',
                                 help='Suppress output messages')
        output_group.add_argument('--log-file', type=str,
                                 help='Log file path')
        
        return parser
    
    @staticmethod
    def parse_args_to_config(args: argparse.Namespace) -> MPASConfig:
        """
        Convert parsed command-line arguments from argparse.Namespace to the MPASConfig data structure used throughout MPASdiag workflows. This method maps recognized argument names from the general-purpose parser to MPASConfig attribute names, extracting values from the Namespace object and constructing a configuration dictionary that can initialize an MPASConfig instance. The conversion handles type preservation, default value assignment, and ensures all required configuration parameters are properly transferred from command-line input to the internal configuration representation. This transformation enables consistent configuration handling whether inputs come from command-line arguments, configuration files, or programmatic API calls.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments returned from create_parser().parse_args() containing user-specified options.

        Returns:
            MPASConfig: Configuration object populated with values from command-line arguments, ready for use in MPAS analysis workflows.
        """
        config_dict = {}
        
        arg_mapping = {
            'grid_file': 'grid_file',
            'data_dir': 'data_dir',
            'output_dir': 'output_dir',
            'lat_min': 'lat_min',
            'lat_max': 'lat_max',
            'lon_min': 'lon_min',
            'lon_max': 'lon_max',
            'var': 'variable',
            'accum': 'accumulation_period',
            'time_index': 'time_index',
            'colormap': 'colormap',
            'dpi': 'dpi',
            'formats': 'output_formats',
            'batch_all': 'batch_mode',
            'verbose': 'verbose',
            'quiet': 'quiet',
            'parallel': 'parallel',
        }
        
        for arg_name, config_attr in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)
        
        if hasattr(args, 'figure_size') and args.figure_size:
            config_dict['figure_width'] = args.figure_size[0]
            config_dict['figure_height'] = args.figure_size[1]
        
        if hasattr(args, 'data_type') and args.data_type == 'xarray':
            config_dict['use_pure_xarray'] = True
        
        return MPASConfig(**config_dict)
    
    @staticmethod
    def create_surface_parser() -> argparse.ArgumentParser:
        """
        Create a specialized command-line argument parser for MPAS surface variable visualization with options tailored to 2D field plotting. This factory method constructs an ArgumentParser configured specifically for surface diagnostic analysis including 2-meter temperature, sea-level pressure, humidity, and 10-meter winds, with support for both scatter and contour plot types. The parser includes required arguments for grid and data files, variable selection with time indexing, plot type and styling options, spatial extent configuration, color limit controls, and output format specifications. The returned parser provides comprehensive examples in the epilog demonstrating common surface plotting workflows and includes organized argument groups for improved command-line help readability in surface-specific MPASdiag operations.

        Returns:
            argparse.ArgumentParser: Configured ArgumentParser instance specialized for surface variable plotting with groups for required inputs, variable selection, plot settings, spatial extent, color controls, output options, and processing flags.
        """
        parser = argparse.ArgumentParser(
            description="MPAS Surface Variable Plotting Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Temperature scatter plot
  mpasdiag surface --grid-file grid.nc --data-dir ./data --variable t2m --plot-type scatter
  
  # Pressure contour plot with custom extent
  mpasdiag surface --grid-file grid.nc --data-dir ./data --variable surface_pressure --plot-type contour --lat-min -10 --lat-max 15
  
  # Wind speed with custom colormap
  mpasdiag surface --grid-file grid.nc --data-dir ./data --variable wspd10 --colormap plasma --time-index 12
  
  # Sea level pressure for specific time
  mpasdiag surface --grid-file grid.nc --data-dir ./data --variable mslp --plot-type contour --time-index 24 --output mslp_analysis
            """
        )
        
        required_group = parser.add_argument_group('Required')
        required_group.add_argument('--grid-file', type=str, required=True,
                                   help='MPAS grid file')
        required_group.add_argument('--data-dir', type=str, required=True,
                                   help='Directory containing MPAS diagnostic files')
        
        var_group = parser.add_argument_group('Variable')
        var_group.add_argument('--variable', '--var', type=str, required=True,
                              help='Variable name to plot (e.g., t2m, surface_pressure, q2, u10, etc.)')
        var_group.add_argument('--time-index', type=int, default=0,
                              help='Time index to plot (default: 0)')
        
        plot_group = parser.add_argument_group('Plot Settings')
        plot_group.add_argument('--plot-type', type=str, default='scatter',
                               choices=['scatter', 'contour'],
                               help='Plot type: scatter (points) or contour (interpolated)')
        plot_group.add_argument('--colormap', type=str, default='default',
                               help='Colormap name (default: auto-selected based on variable)')
        plot_group.add_argument('--title', type=str,
                               help='Custom plot title (default: auto-generated)')
        
        spatial_group = parser.add_argument_group('Spatial Extent')
        spatial_group.add_argument('--lat-min', type=float, default=-9.60,
                                  help='Minimum latitude (default: -9.60)')
        spatial_group.add_argument('--lat-max', type=float, default=12.20,
                                  help='Maximum latitude (default: 12.20)')
        spatial_group.add_argument('--lon-min', type=float, default=91.00,
                                  help='Minimum longitude (default: 91.00)')
        spatial_group.add_argument('--lon-max', type=float, default=113.00,
                                  help='Maximum longitude (default: 113.00)')
        
        color_group = parser.add_argument_group('Color Settings')
        color_group.add_argument('--clim-min', type=float,
                                help='Minimum color limit')
        color_group.add_argument('--clim-max', type=float,
                                help='Maximum color limit')
        
        output_group = parser.add_argument_group('Output')
        output_group.add_argument('--output', '-o', type=str,
                                 help='Output filename (without extension)')
        output_group.add_argument('--output-dir', type=str, default='.',
                                 help='Output directory (default: current directory)')
        output_group.add_argument('--dpi', type=int, default=100,
                                 help='Output resolution (DPI) - default: 100, use 300+ for publication quality')
        output_group.add_argument('--figure-size', type=float, nargs=2,
                                 default=[10.0, 12.0], metavar=('WIDTH', 'HEIGHT'),
                                 help='Figure size in inches (default: 10.0 12.0)')
        output_group.add_argument('--formats', type=str, nargs='+',
                                 default=['png'], choices=['png', 'pdf', 'svg', 'eps'],
                                 help='Output formats (default: png)')
        
        proc_group = parser.add_argument_group('Processing')
        proc_group.add_argument('--verbose', '-v', action='store_true',
                                help='Enable verbose output')
        proc_group.add_argument('--quiet', '-q', action='store_true',
                                help='Suppress output messages')
        proc_group.add_argument('--batch-all', action='store_true',
                                help='Process all time steps in batch mode')
        proc_group.add_argument('--grid-resolution', type=int,
                                help='Grid resolution (number of points per axis) for contour interpolation. If not set, an adaptive heuristic is used (default: adaptive)')
        proc_group.add_argument('--grid-resolution-deg', type=float,
                                help='Grid resolution in degrees (e.g., 0.1 for 0.1° × 0.1° grid). If set, takes precedence over --grid-resolution')

        return parser
    
    @staticmethod
    def parse_surface_args_to_config(args: argparse.Namespace) -> MPASConfig:
        """
        Convert parsed surface plotting command-line arguments from argparse.Namespace to MPASConfig for surface diagnostic workflows. This method performs argument name mapping from the surface-specific parser to MPASConfig attributes, extracting surface plotting options including variable name, plot type, time index, spatial extent, color limits, grid resolution, and output settings. The conversion ensures all surface-specific parameters like scatter vs contour plot type, variable-specific colormaps, and adaptive grid resolution settings are properly transferred to the configuration object. This specialized conversion handles surface plotting defaults and validation requirements distinct from precipitation or wind analysis configurations.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments returned from create_surface_parser().parse_args() with surface plotting options.

        Returns:
            MPASConfig: Configuration object populated with surface plotting parameters ready for use in MPAS surface diagnostic visualization workflows.
        """
        config_dict = {}
        
        arg_mapping = {
            'grid_file': 'grid_file',
            'data_dir': 'data_dir', 
            'output_dir': 'output_dir',
            'variable': 'variable',
            'time_index': 'time_index',
            'plot_type': 'plot_type',
            'colormap': 'colormap',
            'title': 'title',
            'lat_min': 'lat_min',
            'lat_max': 'lat_max',
            'lon_min': 'lon_min',
            'lon_max': 'lon_max',
            'clim_min': 'clim_min',
            'clim_max': 'clim_max',
            'grid_resolution': 'grid_resolution',
            'grid_resolution_deg': 'grid_resolution_deg',
            'output': 'output',
            'dpi': 'dpi',
            'formats': 'output_formats',
            'verbose': 'verbose',
            'quiet': 'quiet',
        }
        
        for arg_name, config_attr in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

        if hasattr(args, 'batch_all') and args.batch_all:
            config_dict['batch_mode'] = True
        
        if hasattr(args, 'figure_size') and args.figure_size:
            config_dict['figure_size'] = tuple(args.figure_size)
        
        if 'plot_type' not in config_dict:
            config_dict['plot_type'] = 'scatter'

        if 'time_index' not in config_dict:
            config_dict['time_index'] = 0
        
        return MPASConfig(**config_dict)

    @staticmethod
    def create_wind_parser() -> argparse.ArgumentParser:
        """
        Create a specialized command-line argument parser for MPAS wind vector visualization with options for barbs and arrows representation. This factory method constructs an ArgumentParser configured specifically for wind field analysis including 10-meter surface winds and upper-air wind components, with support for both wind barb and arrow vector plot types, optional background wind speed shading, and vector subsampling controls. The parser includes arguments for U and V wind component variable specification, wind level descriptors, vector styling parameters, optional background contour fill, time indexing, spatial extent configuration, and comprehensive output format options. The returned parser provides practical examples demonstrating wind plotting workflows and organized argument groups for wind-specific visualization in MPASdiag operations.

        Returns:
            argparse.ArgumentParser: Configured ArgumentParser instance specialized for wind vector plotting with options for component variables, plot type (barbs/arrows), subsampling, background shading, spatial extent, and output settings.
        """
        parser = argparse.ArgumentParser(
            description="Generate MPAS wind vector plots with barbs or arrows",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Surface wind with barbs
              mpasdiag wind --grid-file /path/to/grid.nc --data-dir /path/to/data --u-variable u10 --v-variable v10

              # 850mb wind with arrows
              mpasdiag wind --grid-file /path/to/grid.nc --data-dir /path/to/data --u-variable u850 --v-variable v850 --wind-plot-type arrows

              # Custom extent and subsampling
              mpasdiag wind --grid-file /path/to/grid.nc --data-dir /path/to/data --u-variable u10 --v-variable v10 --extent -105 -95 35 45 --subsample 3
            """)
        )
        
        parser.add_argument("--grid_file", help="Path to MPAS grid file (.nc)")
        parser.add_argument("--data_dir", help="Path to directory containing MPAS data files")
        
        parser.add_argument("--u-variable", default="u10", 
                          help="U-component wind variable name (default: u10)")
        parser.add_argument("--v-variable", default="v10",
                          help="V-component wind variable name (default: v10)")
        parser.add_argument("--wind-level", default="surface",
                          help="Wind level description for labeling (default: surface)")
        
        parser.add_argument("--wind-plot-type", choices=["barbs", "arrows"], default="barbs",
                  help="Wind vector representation type (default: barbs)")
        parser.add_argument("--subsample", type=int, default=0, dest="subsample_factor",
                  help="Subsample factor for wind vectors (plot every Nth point, default: 0 => auto)")
        parser.add_argument("--wind-scale", type=float, default=None,
                          help="Scale factor for wind vectors (auto-determined if not specified)")
        
        parser.add_argument("--show-background", action="store_true",
                          help="Show background wind speed as filled contours")
        parser.add_argument("--background-colormap", default="viridis",
                          help="Colormap for background wind speed (default: viridis)")
        
        parser.add_argument("--time-index", type=int, default=0,
                          help="Time index to plot (0-based, default: 0)")
        
        parser.add_argument("--extent", nargs=4, type=float, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                          help="Map extent [lon_min lon_max lat_min lat_max] (default: auto from data)")
        
        parser.add_argument("--output", "-o", help="Output file path (without extension)")
        parser.add_argument("--output-dir", default=".", help="Output directory (default: current directory)")
        parser.add_argument("--output-formats", nargs="+", default=["png"], 
                          choices=["png", "pdf", "svg", "jpg"],
                          help="Output format(s) (default: png)")
        
        parser.add_argument("--title", help="Custom plot title")
        parser.add_argument("--figure-size", nargs=2, type=float, default=[10, 13], metavar=("WIDTH", "HEIGHT"),
                          help="Figure size in inches (default: 10 13)")
        parser.add_argument("--dpi", type=int, default=100, 
                          help="Output DPI - default: 100, use 300+ for publication quality")
        
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
        proc_group = parser.add_argument_group('Processing')
        proc_group.add_argument('--batch-all', action='store_true',
                               help='Process all time steps in batch mode')

        return parser

    @staticmethod
    def create_crosssection_parser() -> argparse.ArgumentParser:
        """
        Create a specialized command-line argument parser for MPAS 3D vertical cross-section plotting along user-defined transects through the atmosphere. This factory method constructs an ArgumentParser configured for extracting and visualizing vertical slices of 3D atmospheric variables (temperature, winds, moisture, reflectivity) along great circle paths between specified start and end coordinates. The parser includes required arguments for grid file, data directory, and 3D variable name, with options for cross-section endpoints, vertical coordinate selection (pressure, height, model levels), number of interpolation points, colormap and contour level controls, time indexing, and output format specifications. The returned parser provides detailed examples of common cross-section workflows and organized argument groups for 3D diagnostic visualization in MPASdiag.

        Returns:
            argparse.ArgumentParser: Configured ArgumentParser instance specialized for vertical cross-section plotting with options for transect definition, 3D variable selection, vertical coordinate system, interpolation resolution, and visualization styling.
        """
        parser = argparse.ArgumentParser(
            description="Generate MPAS 3D vertical cross-section plots",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Temperature cross-section from Denver to Kansas City
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable theta --start-lon -105.0 --start-lat 39.7 --end-lon -94.6 --end-lat 39.1

              # Wind cross-section with pressure coordinates
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable uReconstructZonal --start-lon -110 --start-lat 35 --end-lon -80 --end-lat 45 --vertical-coord pressure

              # Custom cross-section with model levels
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable theta --start-lon -100 --start-lat 30 --end-lon -90 --end-lat 50 --vertical-coord model_levels --colormap plasma
            """)
        )
        
        parser.add_argument("--grid-file", required=True,
                          help="Path to MPAS grid/static file (.nc)")
        parser.add_argument("--data-dir", required=True,
                          help="Path to directory containing MPAS 3D output files (mpasout*.nc)")
        parser.add_argument("--variable", required=True,
                          help="3D atmospheric variable name (e.g., 'theta', 'uReconstructZonal', 'temperature')")
        
        path_group = parser.add_argument_group('Cross-section Path')
        path_group.add_argument("--start-lon", type=float, required=True,
                              help="Starting longitude in degrees")
        path_group.add_argument("--start-lat", type=float, required=True,
                              help="Starting latitude in degrees")
        path_group.add_argument("--end-lon", type=float, required=True,
                              help="Ending longitude in degrees")
        path_group.add_argument("--end-lat", type=float, required=True,
                              help="Ending latitude in degrees")
        
        xsec_group = parser.add_argument_group('Cross-section Parameters')
        xsec_group.add_argument("--time-index", type=int, default=0,
                              help="Time index to extract (default: 0)")
        xsec_group.add_argument("--vertical-coord", choices=["pressure", "model_levels", "height"], 
                              default="pressure",
                              help="Vertical coordinate system (default: pressure)")
        xsec_group.add_argument("--num-points", type=int, default=100,
                              help="Number of interpolation points along cross-section (default: 100)")
        xsec_group.add_argument("--max-height", type=float,
                              help="Maximum height in km for the vertical axis (default: auto)")
        
        viz_group = parser.add_argument_group('Visualization Options')
        viz_group.add_argument("--plot-type", choices=["filled_contour", "contour", "pcolormesh"], 
                             default="filled_contour",
                             help="Plot type (default: filled_contour)")
        viz_group.add_argument("--colormap", default="viridis",
                             help="Matplotlib colormap name (default: viridis)")
        viz_group.add_argument("--levels", type=float, nargs='+',
                             help="Custom contour levels (space-separated)")
        viz_group.add_argument("--extend", choices=["both", "min", "max", "neither"], 
                             default="both",
                             help="Colorbar extension (default: both)")
        
        parser.add_argument("--output", "-o", help="Output file path (without extension)")
        parser.add_argument("--output-dir", default="./output",
                          help="Output directory (default: ./output)")
        parser.add_argument("--output-formats", nargs='+', default=['png'],
                          choices=['png', 'pdf', 'svg', 'jpg', 'tiff'],
                          help="Output formats (default: png)")
        parser.add_argument("--title", help="Custom plot title")
        
        fig_group = parser.add_argument_group('Figure Options')
        fig_group.add_argument("--figure-size", nargs=2, type=float, default=[14, 8], 
                             metavar=("WIDTH", "HEIGHT"),
                             help="Figure size in inches (default: 14 8)")
        fig_group.add_argument("--dpi", type=int, default=100, 
                             help="Output DPI - default: 100, use 300+ for publication quality")
        
        parser.add_argument("--verbose", "-v", action="store_true", 
                          help="Enable verbose output")
        
        return parser

    @staticmethod
    def parse_wind_args_to_config(args: argparse.Namespace) -> MPASConfig:
        """
        Convert parsed wind plotting command-line arguments from argparse.Namespace to MPASConfig for wind vector visualization workflows. This method performs argument name mapping from the wind-specific parser to MPASConfig attributes, extracting wind plotting options including U and V component variable names, wind level descriptors, vector plot type (barbs or arrows), subsampling factor, optional wind scale, background wind speed display settings, time index, spatial extent, and output formatting parameters. The conversion handles default spatial extent assignment when not explicitly provided and ensures all wind-specific visualization parameters are properly transferred to the configuration object for use in wind diagnostic plotting.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments returned from create_wind_parser().parse_args() with wind plotting options.

        Returns:
            MPASConfig: Configuration object populated with wind plotting parameters ready for use in MPAS wind vector visualization workflows.
        """
        if args.extent:
            lon_min, lon_max, lat_min, lat_max = args.extent
        else:
            lon_min = -180.0
            lon_max = 180.0
            lat_min = -90.0
            lat_max = 90.0
        
        config = MPASConfig(
            grid_file=args.grid_file,
            data_dir=args.data_dir,
            u_variable=args.u_variable,
            v_variable=args.v_variable,
            wind_level=args.wind_level,
            wind_plot_type=args.wind_plot_type,
            subsample_factor=args.subsample_factor,
            wind_scale=args.wind_scale,
            show_background=args.show_background,
            background_colormap=args.background_colormap,
            time_index=args.time_index,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            output=args.output,
            output_dir=args.output_dir,
            output_formats=args.output_formats,
            title=args.title,
            figure_size=tuple(args.figure_size),
            dpi=args.dpi,
            verbose=args.verbose
        )

        if hasattr(args, 'batch_all') and args.batch_all:
            config.batch_mode = True
        
        return config

    @staticmethod
    def parse_crosssection_args_to_config(args: argparse.Namespace) -> MPASConfig:
        """
        Convert parsed vertical cross-section plotting command-line arguments from argparse.Namespace to MPASConfig for 3D transect visualization workflows. This method performs argument name mapping from the cross-section-specific parser to MPASConfig attributes, extracting 3D plotting options including variable name, transect start/end coordinates, vertical coordinate system choice, number of interpolation points, maximum height limits, plot type, colormap selection, custom contour levels, and colorbar extension settings. The conversion handles optional contour level list parsing and ensures all cross-section-specific parameters including transect geometry, vertical coordinate preferences, and visualization styling are properly transferred to the configuration object for use in 3D diagnostic plotting operations.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments returned from create_crosssection_parser().parse_args() with vertical cross-section plotting options.

        Returns:
            MPASConfig: Configuration object populated with cross-section plotting parameters ready for use in MPAS 3D vertical transect visualization workflows.
        """
        levels = None

        if hasattr(args, 'levels') and args.levels:
            levels = list(args.levels)
        
        config = MPASConfig(
            grid_file=args.grid_file,
            data_dir=args.data_dir,
            variable=args.variable,
            time_index=args.time_index,
            output=args.output,
            output_dir=args.output_dir,
            output_formats=args.output_formats,
            title=args.title,
            figure_size=tuple(args.figure_size),
            dpi=args.dpi,
            verbose=args.verbose
        )
        
        config.start_lon = args.start_lon
        config.start_lat = args.start_lat
        config.end_lon = args.end_lon
        config.end_lat = args.end_lat
        config.vertical_coord = args.vertical_coord
        config.num_points = args.num_points
        config.max_height = getattr(args, 'max_height', None)
        config.plot_type = args.plot_type
        config.colormap = args.colormap
        config.levels = levels
        config.extend = args.extend
        
        return config