#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

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
import io
import argparse
import contextlib
import dataclasses
import inspect
import re
import textwrap
import logging
from dataclasses import replace
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from .utils_config import MPASConfig
from .utils_logger import (
    MPASLogger,
    get_logger,
    is_root_process,
    mpi_launcher_detected,
)
from .utils_monitor import PerformanceMonitor
from .processors_2d import MPAS2DProcessor
from .processors_3d import MPAS3DProcessor
from .parallel import start_worker_server

from .parallel_wrappers import (
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor,
    BatchRenderOptions,
    CrossSectionBatchStyle,
    cross_section_file_stem,
    default_precipitation_title,
    RemapConfig,
    SurfaceBatchStyle,
    WindBatchStyle,
)

from . import parallel_wrappers as _pw
from .base import MissingVariablesError
from .constants import (
    CROSS_SECTION_AUX_VARS,
    DIAG_GLOB,
    MPASOUT_GLOB,
    PERFORMANCE_MONITOR_MSG,
    PRECIP_REQUIRED_VARS,
)
from .utils_path import safe_resolve_within, sanitize_filename_component
from .utils_datetime import MPASDateTimeUtils
from .utils_geog import GeographicBounds

try:
    from ..visualization.precipitation import (
        MPASPrecipitationPlotter,
        PrecipitationRenderStyle,
        PrecipitationMapStyle,
    )
    from ..visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
    from ..visualization.cross_section import (
        MPASVerticalCrossSectionPlotter,
        CrossSectionStyle,
    )
    from ..visualization.wind import MPASWindPlotter, WindPlotStyle
    from ..diagnostics.precipitation import (
        PrecipitationDiagnostics,
        InsufficientLookbackError,
    )
    from ..diagnostics.sounding import SoundingDiagnostics
    from ..visualization.skewt import MPASSkewTPlotter
except ImportError:
    try:
        from mpasdiag.visualization.precipitation import (
            MPASPrecipitationPlotter,
            PrecipitationRenderStyle,
            PrecipitationMapStyle,
        )
        from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
        from mpasdiag.visualization.cross_section import (
            MPASVerticalCrossSectionPlotter,
            CrossSectionStyle,
        )
        from mpasdiag.visualization.wind import MPASWindPlotter, WindPlotStyle
        from mpasdiag.diagnostics.precipitation import (
            PrecipitationDiagnostics,
            InsufficientLookbackError,
        )
        from mpasdiag.diagnostics.sounding import SoundingDiagnostics
        from mpasdiag.visualization.skewt import MPASSkewTPlotter
    except ImportError:
        from mpasdiag.visualization.precipitation import (
            MPASPrecipitationPlotter,
            PrecipitationRenderStyle,
            PrecipitationMapStyle,
        )
        from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
        from mpasdiag.visualization.cross_section import (
            MPASVerticalCrossSectionPlotter,
            CrossSectionStyle,
        )
        from mpasdiag.visualization.wind import MPASWindPlotter, WindPlotStyle
        from mpasdiag.diagnostics.precipitation import (
            PrecipitationDiagnostics,
            InsufficientLookbackError,
        )
        from mpasdiag.diagnostics.sounding import SoundingDiagnostics
        from mpasdiag.visualization.skewt import MPASSkewTPlotter


DEFAULT_OUTPUT_DIR = "./output"


def _accepted_kwargs(target: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    This helper filters keyword arguments down to the ones a dataclass or callable actually declares, so optional settings (for example a style option introduced by another module) are forwarded only when the target supports them and None-valued options never reach it.

    Parameters:
        target (Any): A dataclass type or a callable whose accepted parameters are inspected.
        **kwargs (Any): Candidate keyword arguments.

    Returns:
        Dict[str, Any]: The subset of kwargs the target accepts and whose value is not None.
    """
    if dataclasses.is_dataclass(target):
        names = {f.name for f in dataclasses.fields(target)}
    else:
        names = set(inspect.signature(target).parameters)
    return {k: v for k, v in kwargs.items() if v is not None and k in names}


def _sanitize_output_name(name: str) -> str:
    """
    This helper turns a user-supplied --output name into a safe, readable file name: path separators, whitespace and characters that are illegal or awkward in file names ('?', '*', ':', quotes, '<', '>', '|') become underscores, and the shared sanitize_filename_component then removes control characters, dot runs, and leading or trailing separators, so the user's intent stays readable (e.g. 'a/b c?.x' becomes 'a_b_c.x').

    Parameters:
        name (str): The raw --output value.

    Returns:
        str: The sanitized file name without directory components.
    """
    cleaned = re.sub(r"[\s?*:\"'<>|]+", "_", str(name))
    cleaned = re.sub(r"_+\.", ".", cleaned)
    return sanitize_filename_component(cleaned)


def _resolve_against_cwd(value: str, base: Path) -> str:
    """
    This helper resolves a user-supplied path against the current working directory (relative paths are relative to where the command is run, as documented) and then confines the result within the trusted base directory.

    Parameters:
        value (str): The user-supplied path, relative or absolute.
        base (Path): The trusted base directory the resolved path must stay within.

    Returns:
        str: The resolved absolute path.

    Raises:
        ValueError: If the resolved path lies outside base.
    """
    candidate = (Path.cwd() / Path(value).expanduser()).resolve()
    return str(safe_resolve_within(candidate, base))


class MPASUnifiedCLI:
    """Unified command-line interface for all MPAS visualization types."""

    PLOT_TYPES = {
        "precipitation": "Precipitation accumulation maps",
        "surface": "Surface variable scalar plots",
        "wind": "Wind vector plots (barbs/arrows)",
        "cross": "3D vertical cross-section plots",
        "overlay": "Complex multi-variable overlay plots",
    }

    def __init__(self: "MPASUnifiedCLI") -> None:
        """
        This method initializes the MPASUnifiedCLI class, setting up instance variables for logging, performance monitoring, and configuration management. It prepares the CLI for execution by initializing these components to None, which will later be configured based on user input and command-line arguments. The logger will be used to record messages and events during the execution of the CLI, while the performance monitor will track the execution time of various processing steps. The configuration variable will hold the parsed configuration settings from command-line arguments or configuration files, allowing for flexible and customizable execution of different analysis types. By initializing these components in the constructor, it ensures that they are available throughout the lifecycle of the CLI instance for consistent logging, performance tracking, and configuration management across all analysis types.

        Parameters:
            None

        Returns:
            None
        """
        self.logger: Optional[logging.Logger] = None
        self.perf_monitor: Optional[PerformanceMonitor] = None
        self._config_notes: List[str] = []
        self.config: Optional[MPASConfig] = None
        self._subparsers_by_name: Dict[str, argparse.ArgumentParser] = {}
        self._required_args: Dict[str, List[Tuple[str, str]]] = {}

    def create_main_parser(self: "MPASUnifiedCLI") -> argparse.ArgumentParser:
        """
        This method creates and configures the main argument parser for the MPASUnifiedCLI, which serves as the entry point for users to access various types of analyses. The main parser is set up with a program name, description, and an epilog that provides usage examples for different analysis types. It includes global arguments for configuration file input, verbosity control, logging options, and version information. The method also creates subparsers for each analysis type (precipitation, surface variables, wind vectors, vertical cross-sections, and complex overlays) and registers them with the main parser. Each subparser is configured with its own specific arguments relevant to the type of analysis it performs, while also incorporating common arguments through a shared method. By structuring the CLI in this way, it allows users to easily navigate and execute different analyses while maintaining a consistent interface and providing helpful guidance through detailed help text and examples.

        Parameters:
            None

        Returns:
            argparse.ArgumentParser: The configured main argument parser for the MPASUnifiedCLI.
        """
        parser = argparse.ArgumentParser(
            prog="mpasdiag",
            description="Unified MPAS Data Analysis and Visualization Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
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
            """
            ),
        )

        parser.add_argument(
            "--config", type=str, help="Configuration file path (YAML format)"
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable verbose output (shortcut for --log-level DEBUG)",
        )
        parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Suppress output messages (shortcut for --log-level ERROR)",
        )
        parser.add_argument("--log-file", type=str, help="Log file path")
        parser.add_argument(
            "--base-dir",
            type=str,
            default=None,
            help="Trusted base directory that all input/output paths must resolve "
            "within (default: current working directory). Set this to work with "
            "grid/data/output on an explicit out-of-tree location (e.g. /scratch).",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=None,
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging severity threshold (overrides --verbose/--quiet)",
        )
        parser.add_argument("--version", action="version", version="MPASdiag 1.0.0")

        subparsers = parser.add_subparsers(
            dest="analysis_command",
            title="Analysis Types",
            description="Choose the type of analysis to perform",
            help="Available analysis types",
        )

        self._add_precipitation_parser(subparsers)
        self._add_surface_parser(subparsers)
        self._add_wind_parser(subparsers)
        self._add_cross_parser(subparsers)
        self._add_sounding_parser(subparsers)
        self._add_overlay_parser(subparsers)
        self._relax_subparsers(subparsers)

        return parser

    def _relax_subparsers(
        self: "MPASUnifiedCLI", subparsers: argparse._SubParsersAction
    ) -> None:
        """
        This method prepares every analysis subparser for the precedence rule explicit command-line flags > YAML configuration > defaults. It records which arguments each subparser declared as required (so the requirement can be checked after the YAML file has been merged, letting --grid-file, --data-dir, --variable and similar values come from the configuration file), then clears the required flag and sets every option default to argparse.SUPPRESS. With suppressed defaults the parsed namespace contains only the flags the user actually typed, so an argparse default can never silently overwrite a value read from YAML; the real defaults live in the MPASConfig dataclass and are quoted in the help text.

        Parameters:
            subparsers (argparse._SubParsersAction): The subparsers collection whose parsers (including aliases) are relaxed.

        Returns:
            None
        """
        for name, sub in subparsers.choices.items():
            self._subparsers_by_name[name] = sub
            required: List[Tuple[str, str]] = []
            for action in sub._actions:
                if not action.option_strings or isinstance(
                    action, (argparse._HelpAction, argparse._VersionAction)
                ):
                    continue
                if action.required:
                    required.append((action.dest, action.option_strings[0]))
                    action.required = False
                action.default = argparse.SUPPRESS
            self._required_args[name] = required

    def _add_common_arguments(
        self: "MPASUnifiedCLI",
        parser: argparse.ArgumentParser,
        required_grid: bool = True,
    ) -> None:
        """
        This method adds common command-line arguments to the provided parser object for MPAS diagnostic analyses. These common arguments include options for specifying input and output paths, spatial extent of the analysis, time selection criteria, output formatting and resolution, color settings for visualizations, and processing options such as parallel execution and chunk size. By centralizing these common arguments in a single method, it promotes consistency across different analysis types and reduces redundancy in argument definitions. The method takes a parser object as input, which can be either the main parser or any of the subparsers for specific analysis types, and adds the relevant arguments to it. The required_grid parameter allows for flexibility in marking the --grid-file argument as required or optional based on the needs of specific analyses. This approach ensures that all analyses have access to a consistent set of common parameters while still allowing for customization based on the specific requirements of each analysis type.

        Parameters:
            parser (argparse.ArgumentParser): The argument parser object to which the common arguments will be added. This can be the main parser or any of the subparsers for specific analysis types.
            required_grid (bool): A flag indicating whether the --grid-file argument should be marked as required. Default is True, meaning that the grid file is required for most analyses, but it can be set to False for analyses that do not require a grid file.

        Returns:
            None
        """
        io_group = parser.add_argument_group("Input/Output")
        io_group.add_argument(
            "--grid-file",
            type=str,
            required=required_grid,
            help="Path to MPAS grid file",
        )
        io_group.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="Directory containing MPAS diagnostic files",
        )
        io_group.add_argument(
            "--output-dir",
            type=str,
            default="./output",
            help="Output directory for plots and results",
        )

        spatial_group = parser.add_argument_group("Spatial Extent")
        spatial_group.add_argument(
            "--lat-min",
            type=float,
            default=None,
            help="Minimum latitude (default: data extent)",
        )
        spatial_group.add_argument(
            "--lat-max",
            type=float,
            default=None,
            help="Maximum latitude (default: data extent)",
        )
        spatial_group.add_argument(
            "--lon-min",
            type=float,
            default=None,
            help="Minimum longitude, -360..360 (default: data extent)",
        )
        spatial_group.add_argument(
            "--lon-max",
            type=float,
            default=None,
            help="Maximum longitude, -360..360; may exceed 180 for dateline-crossing "
            "boxes, span <= 360 (default: data extent)",
        )

        time_group = parser.add_argument_group("Time Selection")
        time_group.add_argument(
            "--time-index",
            type=int,
            default=0,
            help="Time index to process (default: 0)",
        )
        time_group.add_argument(
            "--batch-all",
            action=argparse.BooleanOptionalAction,
            help="Process all time steps in batch mode (use --no-batch-all to override a "
            "YAML batch_mode: true). Exit status is non-zero if any step failed.",
        )
        time_group.add_argument(
            "--time-range",
            type=int,
            nargs=2,
            metavar=("START", "END"),
            help="Process time indices START..END inclusive (implies batch mode); "
            "indices must be non-negative and within the data",
        )

        output_group = parser.add_argument_group("Output Options")
        output_group.add_argument(
            "--output", "-o", type=str, help="Output filename (without extension)"
        )
        output_group.add_argument("--title", type=str, help="Custom plot title")
        output_group.add_argument(
            "--dpi",
            type=int,
            default=100,
            help="Output resolution (DPI) - default: 100, use 300+ for publication quality",
        )
        output_group.add_argument(
            "--figure-size",
            type=float,
            nargs=2,
            default=[12.0, 10.0],
            metavar=("WIDTH", "HEIGHT"),
            help="Figure size in inches (default: 12.0 10.0)",
        )
        output_group.add_argument(
            "--formats",
            type=str,
            nargs="+",
            default=["png"],
            choices=["png", "pdf", "svg", "eps", "jpg"],
            help="Output formats (default: png)",
        )

        color_group = parser.add_argument_group("Color Settings")
        color_group.add_argument(
            "--colormap",
            type=str,
            default="default",
            help="Colormap name (default: auto-selected)",
        )
        color_group.add_argument(
            "--clim-min",
            type=float,
            help="Minimum color limit, in the plotted (display) units, e.g. degC for t2m or hPa for mslp; no unit conversion is applied",
        )
        color_group.add_argument(
            "--clim-max",
            type=float,
            help="Maximum color limit, in the plotted (display) units, e.g. degC for t2m or hPa for mslp; no unit conversion is applied",
        )

        color_group.add_argument(
            "--levels",
            type=float,
            nargs="+",
            metavar="LEVEL",
            help="Explicit contour/color levels in the plotted (display) units, e.g. "
            "--levels 0 5 10 20 40 (overrides the automatic levels)",
        )

        proj_group = parser.add_argument_group("Map Projection")
        proj_group.add_argument(
            "--projection",
            type=str,
            help="Cartopy projection name for map plots, e.g. PlateCarree, Robinson, "
            "Orthographic (default: PlateCarree)",
        )
        proj_group.add_argument(
            "--central-longitude",
            type=float,
            help="Central longitude of the map projection in degrees",
        )
        proj_group.add_argument(
            "--central-latitude",
            type=float,
            help="Central latitude of the map projection in degrees",
        )

        proc_group = parser.add_argument_group("Processing Options")
        proc_group.add_argument(
            "--parallel",
            action=argparse.BooleanOptionalAction,
            help="Enable parallel processing (experimental; --no-parallel overrides YAML)",
        )
        proc_group.add_argument(
            "--use-pure-xarray",
            action=argparse.BooleanOptionalAction,
            help="Load the data with plain xarray instead of UXarray "
            "(--no-use-pure-xarray overrides YAML use_pure_xarray: true)",
        )
        proc_group.add_argument(
            "--workers",
            type=int,
            default=None,
            help="Number of parallel workers (default: auto-detect based on CPU cores)",
        )
        proc_group.add_argument(
            "--chunk-size",
            type=int,
            default=None,
            help="Number of cells per data chunk along nCells when loading (default: "
            "automatic, one chunk per time record); smaller values bound peak memory "
            "on very large meshes",
        )

        log_group = parser.add_argument_group("Logging Options")
        log_group.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable verbose output (shortcut for --log-level DEBUG)",
        )
        log_group.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Suppress output messages (shortcut for --log-level ERROR)",
        )
        log_group.add_argument(
            "--log-file", type=str, default=argparse.SUPPRESS, help="Log file path"
        )
        log_group.add_argument(
            "--log-level",
            type=str,
            default=argparse.SUPPRESS,
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging severity threshold (overrides --verbose/--quiet)",
        )

        remap_group = parser.add_argument_group("Remapping Options")
        remap_group.add_argument(
            "--remap-engine",
            type=str,
            default="kdtree",
            choices=["kdtree", "esmf"],
            help=(
                "Remapping engine: kdtree (scipy KDTree, default) or "
                "esmf (ESMPy — requires conda install -c conda-forge esmpy)"
            ),
        )
        remap_group.add_argument(
            "--remap-method",
            type=str,
            default="nearest",
            help=(
                "Remapping method. For --remap-engine kdtree: nearest (default), linear. "
                "For --remap-engine esmf: bilinear (default; selected automatically when "
                "no method is given), conservative, conservative_normed, patch, "
                "nearest_s2d, nearest_d2s."
            ),
        )

    def _add_precipitation_parser(
        self: "MPASUnifiedCLI", subparsers: argparse._SubParsersAction
    ) -> None:
        """
        This method creates and registers the precipitation subparser for the MPASUnifiedCLI, enabling users to generate precipitation accumulation maps from MPAS diagnostic data. The subparser is configured with command aliases ('precip', 'rain') for user convenience and includes specific arguments for selecting the precipitation variable, accumulation period, plot type (scatter or contourf), grid resolution for interpolation, thresholding options, and output units. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing precipitation analyses effectively, allowing for flexible visualization of precipitation accumulation over various time periods and with different plotting styles. By registering this subparser under the main CLI, it allows users to easily access precipitation analysis functionality while maintaining a consistent interface across different types of analyses.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the precipitation parser will be registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            "precipitation",
            aliases=["precip", "rain"],
            help="Precipitation accumulation analysis",
            description="Generate precipitation accumulation maps",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
            Examples:
              # Single time step
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc

              # Batch processing
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc --batch-all

              # Custom accumulation period
              mpasdiag precipitation --grid-file grid.nc --data-dir ./data --variable rainnc --accumulation a24h
            """
            ),
        )

        self._add_common_arguments(parser)

        precip_group = parser.add_argument_group("Precipitation Options")
        precip_group.add_argument(
            "--variable",
            "--var",
            type=str,
            default="rainnc",
            choices=["rainc", "rainnc", "total"],
            help="Precipitation variable (default: rainnc)",
        )
        precip_group.add_argument(
            "--accumulation",
            "--accum",
            type=str,
            default="a01h",
            choices=["a01h", "a03h", "a06h", "a12h", "a24h"],
            help="Accumulation period (default: a01h)",
        )
        precip_group.add_argument(
            "--plot-type",
            type=str,
            default="scatter",
            choices=["scatter", "contourf"],
            help="Plot type - scatter for direct cell display or contourf for interpolated smooth fields (default: scatter)",
        )
        precip_group.add_argument(
            "--grid-resolution",
            type=float,
            help="Grid resolution in degrees for contourf interpolation (e.g., 0.5 for 0.5° grid, uses adaptive if not specified)",
        )
        precip_group.add_argument(
            "--threshold", type=float, help="Precipitation threshold for highlighting"
        )
        precip_group.add_argument(
            "--units",
            type=str,
            choices=["mm", "in", "cm"],
            default="mm",
            help="Output units (default: mm)",
        )

    def _add_surface_parser(
        self: "MPASUnifiedCLI", subparsers: argparse._SubParsersAction
    ) -> None:
        """
        This method creates and registers the surface variable subparser for the MPASUnifiedCLI, enabling users to generate surface variable scalar plots from MPAS diagnostic data. The subparser is configured with command aliases ('surf', '2d') for user convenience and includes specific arguments for selecting the surface variable to plot, choosing the plot type (scatter, contour, contourf, pcolormesh), specifying grid resolution for interpolation, controlling interpolation method for contour plots, and setting the number of contour levels. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing surface variable analyses effectively, allowing for flexible visualization of various surface variables such as temperature, pressure, humidity, or wind components with different plotting styles. By registering this subparser under the main CLI, it allows users to easily access surface variable analysis functionality while maintaining a consistent interface across different types of analyses.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the surface parser will be registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            "surface",
            aliases=["surf", "2d"],
            help="Surface variable analysis",
            description="Generate surface variable scalar plots",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
            Examples:
              # Temperature scatter plot
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable t2m

              # Pressure contour plot
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable mslp --plot-type contour

              # Custom grid resolution
              mpasdiag surface --grid-file grid.nc --data-dir ./data --variable t2m --plot-type contour --grid-resolution 0.1
            """
            ),
        )

        self._add_common_arguments(parser)

        surf_group = parser.add_argument_group("Surface Variable Options")
        surf_group.add_argument(
            "--variable",
            "--var",
            type=str,
            required=True,
            help="Surface variable name (e.g., t2m, mslp, q2, u10, v10)",
        )
        surf_group.add_argument(
            "--plot-type",
            type=str,
            default="scatter",
            choices=["scatter", "contour", "contourf", "pcolormesh"],
            help="Plot type (default: scatter)",
        )
        surf_group.add_argument(
            "--grid-resolution",
            type=float,
            help="Grid resolution in degrees (e.g., 0.1 for 0.1° grid)",
        )
        surf_group.add_argument(
            "--interpolation",
            type=str,
            default=None,
            choices=["nearest", "linear", "cubic"],
            help="Interpolation method used to grid the cells for contour, contourf "
            "and pcolormesh plots (default: the plotter's built-in choice)",
        )
        surf_group.add_argument(
            "--contour-levels",
            type=int,
            default=None,
            help="Number of automatically generated contour levels, used when no "
            "explicit levels are given (default: automatic)",
        )

    def _add_wind_parser(
        self: "MPASUnifiedCLI", subparsers: argparse._SubParsersAction
    ) -> None:
        """
        This method creates and registers the wind vector subparser for the MPASUnifiedCLI, enabling users to generate wind vector plots from MPAS diagnostic data. The subparser is configured with command aliases ('vector', 'winds') for user convenience and includes specific arguments for selecting the U and V wind component variables, wind level description for labeling, wind vector representation (barbs, arrows, streamlines), subsampling factor for vector density control, scale factor for wind vectors, options to show background wind speed as filled contours, colormap selection for background speed, color and transparency settings for vectors, and grid resolution and interpolation method for regridding wind components if needed. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing wind vector analyses effectively, allowing for flexible visualization of surface or upper-level winds with various customization options. By registering this subparser under the main CLI, it allows users to easily access wind vector analysis functionality while maintaining a consistent interface across different types of analyses.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the wind parser will be registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            "wind",
            aliases=["vector", "winds"],
            help="Wind vector analysis",
            description="Generate wind vector plots with barbs or arrows",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
            Examples:
              # Surface wind barbs
              mpasdiag wind --grid-file grid.nc --data-dir ./data --u-variable u10 --v-variable v10

              # 850mb wind arrows with background
              mpasdiag wind --grid-file grid.nc --data-dir ./data --u-variable u850 --v-variable v850 --wind-plot-type arrows --show-background

              # Custom subsampling and scaling
              mpasdiag wind --grid-file grid.nc --data-dir ./data --u-variable u10 --v-variable v10 --subsample 3 --wind-scale 50
            """
            ),
        )

        self._add_common_arguments(parser)

        wind_group = parser.add_argument_group("Wind Vector Options")
        wind_group.add_argument(
            "--u-variable",
            type=str,
            default="u10",
            help="U-component wind variable (default: u10)",
        )
        wind_group.add_argument(
            "--v-variable",
            type=str,
            default="v10",
            help="V-component wind variable (default: v10)",
        )
        wind_group.add_argument(
            "--wind-level",
            type=str,
            default="surface",
            help="Vertical level for 3D wind components (e.g. uReconstructZonal): "
            "'surface' (lowest level), an integer model level index (e.g. 5), a "
            "pressure in Pa (e.g. 85000) or in hPa (e.g. 850hPa); ignored for 2D "
            "variables such as u10/v10 except as a title label (default: surface)",
        )
        wind_group.add_argument(
            "--wind-plot-type",
            type=str,
            default="barbs",
            choices=["barbs", "arrows", "streamlines"],
            help="Wind vector representation (default: barbs)",
        )
        wind_group.add_argument(
            "--subsample",
            type=int,
            default=0,
            help="Subsample factor (plot every Nth point; 0 or -1 = automatic, default: automatic)",
        )
        wind_group.add_argument(
            "--wind-scale",
            type=float,
            help="Scale factor for wind vectors (auto if not specified)",
        )
        wind_group.add_argument(
            "--show-background",
            action=argparse.BooleanOptionalAction,
            help="Show background wind speed as filled contours",
        )
        wind_group.add_argument(
            "--background-colormap",
            type=str,
            default="viridis",
            help="Colormap for background wind speed (default: viridis)",
        )
        wind_group.add_argument(
            "--vector-color",
            type=str,
            default="black",
            help="Color of the wind barbs and arrows (default: black); streamlines keep the speed colormap",
        )
        wind_group.add_argument(
            "--vector-alpha",
            type=float,
            default=1.0,
            help="Opacity of the wind vectors, 0 (transparent) to 1 (opaque), applied to barbs, arrows and streamlines (default: 1.0)",
        )
        wind_group.add_argument(
            "--grid-resolution",
            type=float,
            help="Grid resolution in degrees for regridding wind components (e.g., 0.5 for 0.5° grid, no regridding if not specified)",
        )
        wind_group.add_argument(
            "--regrid-method",
            type=str,
            default="linear",
            choices=["linear", "nearest"],
            help="Interpolation method for regridding - linear for smooth fields or nearest for preserving values (default: linear)",
        )

    def _add_cross_parser(
        self: "MPASUnifiedCLI", subparsers: argparse._SubParsersAction
    ) -> None:
        """
        This method creates and registers the vertical cross-section subparser for the MPASUnifiedCLI, enabling users to generate 3D vertical cross-section plots from MPAS diagnostic data. The subparser is configured with command aliases ('cross', 'xsection') for user convenience and includes specific arguments for selecting the variable to plot, defining the start and end points of the cross-section in terms of longitude and latitude, choosing the plot type (contour, contourf, pcolormesh), specifying grid resolution for interpolation, controlling interpolation method for contour plots, and setting the number of contour levels. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing vertical cross-section analyses effectively, allowing for flexible visualization of atmospheric variables along a defined transect with different plotting styles. By registering this subparser under the main CLI, it allows users to easily access vertical cross-section analysis functionality while maintaining a consistent interface across different types of analyses.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the vertical cross-section parser will be registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            "cross",
            aliases=["xsec", "3d", "vertical"],
            help="3D vertical cross-section analysis",
            description="Generate vertical cross-section plots along specified paths",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
            Examples:
              # Temperature cross-section
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable theta --start-lon -105 --start-lat 40 --end-lon -95 --end-lat 35

              # Wind cross-section with pressure coordinates
              mpasdiag cross --grid-file grid.nc --data-dir ./data --variable uReconstructZonal --start-lon -110 --start-lat 35 --end-lon -80 --end-lat 45 --vertical-coord pressure
            """
            ),
        )

        self._add_common_arguments(parser, required_grid=True)

        xsec_group = parser.add_argument_group("Cross-Section Options")
        xsec_group.add_argument(
            "--variable",
            type=str,
            required=True,
            help="3D atmospheric variable name (e.g., theta, uReconstructZonal)",
        )
        xsec_group.add_argument(
            "--start-lon",
            type=float,
            required=True,
            help="Starting longitude in degrees",
        )
        xsec_group.add_argument(
            "--start-lat",
            type=float,
            required=True,
            help="Starting latitude in degrees",
        )
        xsec_group.add_argument(
            "--end-lon", type=float, required=True, help="Ending longitude in degrees"
        )
        xsec_group.add_argument(
            "--end-lat", type=float, required=True, help="Ending latitude in degrees"
        )
        xsec_group.add_argument(
            "--vertical-coord",
            type=str,
            default="pressure",
            choices=["pressure", "modlev", "height"],
            help="Vertical coordinate system (default: pressure)",
        )
        xsec_group.add_argument(
            "--num-points",
            type=int,
            default=100,
            help="Number of interpolation points along path (default: 100)",
        )
        xsec_group.add_argument(
            "--max-height",
            type=float,
            help="Maximum height in km for vertical axis (auto if not specified)",
        )
        xsec_group.add_argument(
            "--plot-style",
            type=str,
            default="contourf",
            choices=["contourf", "contour", "pcolormesh"],
            help="Cross-section plot style (default: contourf)",
        )
        xsec_group.add_argument(
            "--extend",
            type=str,
            default="both",
            choices=["both", "min", "max", "neither"],
            help="Colorbar extension (default: both)",
        )

    def _add_sounding_parser(
        self: "MPASUnifiedCLI", subparsers: argparse._SubParsersAction
    ) -> None:
        """
        This method creates and registers the sounding subparser for the MPASUnifiedCLI, enabling users to generate Skew-T Log-P diagrams from MPAS 3D atmospheric data at specified locations. The subparser is configured with command aliases ('skewt', 'profile') for user convenience and includes specific arguments for selecting the longitude and latitude of the sounding location, options to compute and display thermodynamic indices (CAPE, CIN, LCL, LFC, EL), and an option to plot the lifted parcel profile if MetPy is available. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing sounding analyses effectively, allowing for flexible visualization of atmospheric profiles at specific locations with various customization options. By registering this subparser under the main CLI, it allows users to easily access sounding analysis functionality while maintaining a consistent interface across different types of analyses.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the sounding parser will be registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            "sounding",
            aliases=["skewt", "profile"],
            help="Skew-T Log-P sounding diagram",
            description="Generate Skew-T Log-P diagrams from MPAS 3D atmospheric data at a specified location",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
            Examples:
              # Sounding at Denver
              mpasdiag sounding --grid-file grid.nc --data-dir ./data --lon -105.0 --lat 39.7

              # With thermodynamic indices and parcel trace
              mpasdiag sounding --grid-file grid.nc --data-dir ./data --lon -105.0 --lat 39.7 --show-indices --show-parcel

              # Batch all time steps
              mpasdiag sounding --grid-file grid.nc --data-dir ./data --lon -97.5 --lat 35.2 --batch-all --show-indices
            """
            ),
        )

        self._add_common_arguments(parser, required_grid=True)

        sounding_group = parser.add_argument_group("Sounding Options")
        sounding_group.add_argument(
            "--lon",
            type=float,
            required=True,
            help="Sounding location longitude in degrees",
        )
        sounding_group.add_argument(
            "--lat",
            type=float,
            required=True,
            help="Sounding location latitude in degrees",
        )
        sounding_group.add_argument(
            "--max-distance-km",
            type=float,
            help="Largest allowed distance in km between the requested point and the "
            "nearest mesh cell; a point farther away (for example outside a regional "
            "domain) is an error (default: 3 x the local cell spacing)",
        )
        sounding_group.add_argument(
            "--show-indices",
            action=argparse.BooleanOptionalAction,
            help="Compute and display thermodynamic indices (CAPE, CIN, LCL, LFC, EL)",
        )
        sounding_group.add_argument(
            "--show-parcel",
            action=argparse.BooleanOptionalAction,
            help="Plot lifted parcel profile (requires MetPy)",
        )

    def _add_overlay_parser(
        self: "MPASUnifiedCLI", subparsers: argparse._SubParsersAction
    ) -> None:
        """
        This method creates and registers the complex overlay subparser for the MPASUnifiedCLI, enabling users to generate multi-variable overlay plots that combine different types of data (e.g., precipitation with wind vectors, temperature with pressure contours) from MPAS diagnostic files. The subparser is configured with command aliases ('complex', 'multi', 'composite') for user convenience and includes specific arguments for selecting the type of overlay analysis to perform, specifying primary and secondary variables for the overlay, defining custom variable lists for multi-variable overlays, configuring wind vector options for overlays that include wind data, and customizing colormap and transparency settings for the overlay elements. It also incorporates common arguments for input/output settings, spatial extent, time selection, and processing options through a shared method. The method provides detailed help text and usage examples to guide users in executing complex overlay analyses effectively, allowing for flexible visualization of multiple atmospheric variables in a single plot with various customization options. By registering this subparser under the main CLI, it allows users to easily access complex overlay analysis functionality while maintaining a consistent interface across different types of analyses.

        Parameters:
            subparsers (argparse._SubParsersAction): The parent subparsers collection object returned by ArgumentParser.add_subparsers() to which the overlay parser will be registered.

        Returns:
            None
        """
        parser = subparsers.add_parser(
            "overlay",
            aliases=["complex", "multi", "composite"],
            help="Complex multi-variable overlay analysis",
            description="Generate complex overlay plots combining multiple variables",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
            Examples:
              # Precipitation + wind overlay
              mpas-analyze overlay --grid-file grid.nc --data-dir ./data --overlay-type precip_wind --variable rainnc --u-variable u10 --v-variable v10

              # Temperature + pressure overlay
              mpas-analyze overlay --grid-file grid.nc --data-dir ./data --overlay-type temp_pressure --variable t2m --pressure-variable mslp

              # Multi-level analysis
              mpas-analyze overlay --grid-file grid.nc --data-dir ./data --overlay-type multi_level --variables t2m,mslp,u10,v10
            """
            ),
        )

        self._add_common_arguments(parser)

        overlay_group = parser.add_argument_group("Overlay Options")
        overlay_group.add_argument(
            "--overlay-type",
            type=str,
            required=True,
            choices=[
                "precip_wind",
                "temp_pressure",
                "wind_temp",
                "multi_level",
                "custom",
            ],
            help="Type of overlay analysis to perform",
        )
        overlay_group.add_argument(
            "--primary-variable",
            "--variable",
            type=str,
            help="Primary variable for the overlay",
        )
        overlay_group.add_argument(
            "--secondary-variable", type=str, help="Secondary variable for the overlay"
        )
        overlay_group.add_argument(
            "--variables",
            type=str,
            help="Comma-separated list of variables for multi-variable overlays",
        )
        overlay_group.add_argument(
            "--u-variable",
            type=str,
            default="u10",
            help="U-component wind variable (for wind overlays)",
        )
        overlay_group.add_argument(
            "--v-variable",
            type=str,
            default="v10",
            help="V-component wind variable (for wind overlays)",
        )
        overlay_group.add_argument(
            "--pressure-variable",
            type=str,
            default="mslp",
            help="Pressure variable (for pressure overlays)",
        )
        overlay_group.add_argument(
            "--primary-colormap", type=str, help="Colormap for primary variable"
        )
        overlay_group.add_argument(
            "--secondary-colormap", type=str, help="Colormap for secondary variable"
        )
        overlay_group.add_argument(
            "--transparency",
            type=float,
            default=0.7,
            help="Transparency for overlay elements (default: 0.7)",
        )
        overlay_group.add_argument(
            "--contour-overlay",
            action=argparse.BooleanOptionalAction,
            help="Add contour lines for secondary variable",
        )

    def _args_to_dict(
        self: "MPASUnifiedCLI", args: argparse.Namespace
    ) -> Dict[str, Any]:
        """
        This method translates the parsed command-line namespace into a configuration dictionary that contains only the options the user actually passed. Because the parsers suppress argparse defaults, every attribute present on the namespace is an explicit flag, which is what allows the caller to layer these values on top of a YAML configuration (explicit flags > YAML > defaults). The method maps common arguments, normalizes figure size to a tuple, converts --verbose and --quiet into the verbose/quiet configuration flags (each one clearing the other, and passing both is rejected with a ValueError just like the YAML check), treats --time-range as a batch request over the inclusive range, and delegates to the analysis-specific mapping methods for the selected subcommand.

        Parameters:
            args (argparse.Namespace): The namespace object containing the parsed command-line arguments from the argparse parser.

        Returns:
            Dict[str, Any]: Configuration keys and values for exactly the options supplied on the command line.
        """
        config_dict: Dict[str, Any] = {}

        common_mapping = {
            "grid_file": "grid_file",
            "data_dir": "data_dir",
            "output_dir": "output_dir",
            "base_dir": "base_dir",
            "lat_min": "lat_min",
            "lat_max": "lat_max",
            "lon_min": "lon_min",
            "lon_max": "lon_max",
            "time_index": "time_index",
            "batch_all": "batch_mode",
            "colormap": "colormap",
            "dpi": "dpi",
            "formats": "output_formats",
            "title": "title",
            "output": "output",
            "clim_min": "clim_min",
            "clim_max": "clim_max",
            "levels": "levels",
            "use_pure_xarray": "use_pure_xarray",
            "parallel": "parallel",
            "workers": "workers",
            "chunk_size": "chunk_size",
            "remap_engine": "remap_engine",
            "remap_method": "remap_method",
            "log_level": "log_level",
            "log_file": "log_file",
            "projection": "projection",
            "central_longitude": "central_longitude",
            "central_latitude": "central_latitude",
        }

        for arg_name, config_attr in common_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

        if getattr(args, "figure_size", None):
            config_dict["figure_size"] = tuple(args.figure_size)

        if getattr(args, "verbose", False) and getattr(args, "quiet", False):
            raise ValueError("--verbose and --quiet are mutually exclusive")
        if getattr(args, "verbose", False):
            config_dict["verbose"] = True
            config_dict["quiet"] = False
        if getattr(args, "quiet", False):
            config_dict["quiet"] = True
            config_dict["verbose"] = False
        if (getattr(args, "verbose", False) or getattr(args, "quiet", False)) and (
            "log_level" not in config_dict
        ):
            # An explicit --verbose/--quiet beats a YAML log_level (which would
            # otherwise take precedence in setup_logging); --log-level still wins
            config_dict["log_level"] = None

        if getattr(args, "time_range", None):
            config_dict["time_start"] = args.time_range[0]
            config_dict["time_end"] = args.time_range[1]
            config_dict["batch_mode"] = True

        if hasattr(args, "analysis_command") and args.analysis_command:
            config_dict["analysis_type"] = args.analysis_command
            self._dispatch_analysis_args(args, config_dict)

        return config_dict

    def parse_args_to_config(
        self: "MPASUnifiedCLI", args: argparse.Namespace
    ) -> MPASConfig:
        """
        This method takes the parsed command-line arguments from the argparse namespace and translates them into a structured MPASConfig object that encapsulates all the settings needed to execute the selected analysis command. It builds a dictionary containing only the options the user explicitly supplied (see _args_to_dict), applies the output directory default, and constructs the MPASConfig so that every unspecified option takes its dataclass default, including an unspecified spatial extent that is later resolved to the full extent of the data. The constructor validates the configuration and raises ValueError with a descriptive message when a value is invalid.

        Parameters:
            args (argparse.Namespace): The namespace object containing the parsed command-line arguments from the argparse parser and subparsers.

        Returns:
            MPASConfig: A validated configuration object built from the command-line options and the dataclass defaults.
        """
        config_dict = self._args_to_dict(args)
        config_dict.setdefault("output_dir", DEFAULT_OUTPUT_DIR)
        return MPASConfig(**config_dict)

    def _dispatch_analysis_args(
        self: "MPASUnifiedCLI", args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> None:
        """
        This method serves as a dispatcher that routes the parsed command-line arguments to the appropriate mapping function based on the selected analysis command. It checks the value of the 'analysis_command' attribute in the argparse namespace to determine which specific analysis type was invoked (e.g., precipitation, surface, wind, cross-section, sounding, overlay) and then calls the corresponding mapping method to extract and translate analysis-specific parameters from the argparse namespace into the configuration dictionary. Each mapping method is responsible for handling the unique parameters relevant to its respective analysis type while ensuring that only user-specified options are included in the final configuration. This dispatcher method allows for a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework, enabling flexible configuration of various analyses based on user input.

        Parameters:
            args (argparse.Namespace): The namespace object containing the parsed command-line arguments from the argparse parser, which includes an attribute 'analysis_command' that indicates the specific analysis type selected by the user.
            config_dict (Dict[str, Any]): The mutable configuration dictionary that is being populated with parameters extracted from the argparse namespace. This dictionary will be modified in-place by the mapping methods to include analysis-specific parameters based on the selected analysis command.

        Returns:
            None
        """
        analysis_cmd = args.analysis_command
        if analysis_cmd in ["precipitation", "precip", "rain"]:
            self._map_precipitation_args(args, config_dict)
        elif analysis_cmd in ["surface", "surf", "2d"]:
            self._map_surface_args(args, config_dict)
        elif analysis_cmd in ["wind", "vector", "winds"]:
            self._map_wind_args(args, config_dict)
        elif analysis_cmd in ["cross", "xsec", "3d", "vertical"]:
            self._map_cross_args(args, config_dict)
        elif analysis_cmd in ["sounding", "skewt", "profile"]:
            self._map_sounding_args(args, config_dict)
        elif analysis_cmd in ["overlay", "complex", "multi", "composite"]:
            self._map_overlay_args(args, config_dict)

    def _map_precipitation_args(
        self: "MPASUnifiedCLI", args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> None:
        """
        This method maps precipitation-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of precipitation accumulation analysis parameters. It extracts the selected precipitation variable, accumulation period, plot type (scatter or contourf), grid resolution for interpolation, thresholding options, and output units from the parsed arguments and translates them into corresponding configuration keys used by the MPASPrecipitationProcessor and MPASPrecipitationPlotter classes. The mapping handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of precipitation analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing precipitation-specific options extracted from the precipitation subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with precipitation analysis parameters through in-place modification.

        Returns:
            None
        """
        precip_mapping = {
            "variable": "variable",
            "accumulation": "accumulation_period",
            "plot_type": "plot_type",
            "grid_resolution": "grid_resolution",
            "threshold": "precip_threshold",
            "units": "precip_units",
        }

        for arg_name, config_attr in precip_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

    def _map_surface_args(
        self: "MPASUnifiedCLI", args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> None:
        """
        This method maps surface variable-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of surface variable analysis parameters. It extracts the selected surface variable to plot, plot type (scatter, contour, contourf, pcolormesh), grid resolution for interpolation, interpolation method for contour plots, and number of contour levels from the parsed arguments and translates them into corresponding configuration keys used by the MPASSurfaceProcessor and MPASSurfacePlotter classes. The mapping handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of surface variable analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing surface variable-specific options extracted from the surface subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with surface variable analysis parameters through in-place modification.

        Returns:
            None
        """
        surface_mapping = {
            "variable": "variable",
            "plot_type": "plot_type",
            "grid_resolution": "grid_resolution",
            "interpolation": "interpolation_method",
            "contour_levels": "contour_levels",
        }

        for arg_name, config_attr in surface_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

    def _map_wind_args(
        self: "MPASUnifiedCLI", args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> None:
        """
        This method maps wind-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of wind vector analysis parameters. It extracts the U and V wind component variable names, wind level description for labeling, wind vector representation (barbs, arrows, streamlines), subsampling factor for controlling vector density, scale factor for wind vectors, options to show background wind speed as filled contours along with colormap selection for the background, color and transparency settings for the vectors themselves, and grid resolution and interpolation method for regridding wind components if needed. The mapping translates these user-specified options into corresponding configuration keys used by the MPASWindProcessor and MPASWindPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of wind vector analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing wind-specific options extracted from the wind subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with wind vector analysis parameters through in-place modification.

        Returns:
            None
        """
        wind_mapping = {
            "u_variable": "u_variable",
            "v_variable": "v_variable",
            "wind_level": "wind_level",
            "wind_plot_type": "wind_plot_type",
            "subsample": "subsample_factor",
            "wind_scale": "wind_scale",
            "show_background": "show_background",
            "background_colormap": "background_colormap",
            "vector_color": "vector_color",
            "vector_alpha": "vector_alpha",
            "grid_resolution": "grid_resolution",
            "regrid_method": "regrid_method",
        }

        for arg_name, config_attr in wind_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

        if config_dict.get("subsample_factor") == 0:
            config_dict["subsample_factor"] = -1

    def _map_cross_args(
        self: "MPASUnifiedCLI", args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> None:
        """
        This method maps vertical cross-section-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of 3D vertical cross-section analysis parameters. It extracts the selected variable to plot along the cross-section, the start and end points of the cross-section defined by longitude and latitude coordinates, the vertical coordinate system to use (pressure, model levels, or height), the number of interpolation points along the path, maximum height for the vertical axis, plot style (contourf, contour, pcolormesh), and colorbar extension settings. The mapping translates these user-specified options into corresponding configuration keys used by the MPASCrossSectionProcessor and MPASCrossSectionPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of vertical cross-section analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing vertical cross-section-specific options extracted from the cross-section subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with vertical cross-section analysis parameters through in-place modification.

        Returns:
            None
        """
        xsec_mapping = {
            "variable": "variable",
            "start_lon": "start_lon",
            "start_lat": "start_lat",
            "end_lon": "end_lon",
            "end_lat": "end_lat",
            "vertical_coord": "vertical_coord",
            "num_points": "num_points",
            "max_height": "max_height",
            "plot_style": "plot_style",
            "extend": "extend",
        }

        for arg_name, config_attr in xsec_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

    def _map_sounding_args(
        self: "MPASUnifiedCLI", args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> None:
        """
        This method maps sounding-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of Skew-T Log-P diagram analysis parameters. It extracts the longitude and latitude of the sounding location, options to compute and display thermodynamic indices (CAPE, CIN, LCL, LFC, EL), and an option to plot the lifted parcel profile if MetPy is available. The mapping translates these user-specified options into corresponding configuration keys used by the MPASSoundingProcessor and MPASSoundingPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of sounding analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing sounding-specific options extracted from the sounding subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with sounding analysis parameters through in-place modification.

        Returns:
            None
        """
        sounding_mapping = {
            "lon": "sounding_lon",
            "lat": "sounding_lat",
            "max_distance_km": "sounding_max_distance_km",
            "show_indices": "show_indices",
            "show_parcel": "show_parcel",
        }

        for arg_name, config_attr in sounding_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

    def _map_overlay_args(
        self: "MPASUnifiedCLI", args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> None:
        """
        This method maps overlay-specific command-line arguments from the argparse namespace into the configuration dictionary, enabling the configuration of complex multi-variable overlay analysis parameters. It extracts the type of overlay analysis to perform, primary and secondary variables for the overlay, custom variable lists for multi-variable overlays, wind vector options for overlays that include wind data, and colormap and transparency settings for the overlay elements. The mapping translates these user-specified options into corresponding configuration keys used by the MPASOverlayProcessor and MPASOverlayPlotter classes. The method handles optional parameters gracefully by checking for their existence before assignment, ensuring that only user-specified options are included in the final configuration. This specialized mapping allows for flexible configuration of complex overlay analysis parameters based on user input while maintaining a clean separation between common argument handling and analysis-specific parameter mapping within the unified CLI framework.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments containing overlay-specific options extracted from the overlay subparser.
            config_dict (Dict[str, Any]): Mutable configuration dictionary to be populated with overlay analysis parameters through in-place modification.

        Returns:
            None
        """
        overlay_mapping = {
            "overlay_type": "overlay_type",
            "primary_variable": "primary_variable",
            "secondary_variable": "secondary_variable",
            "variables": "variables",
            "u_variable": "u_variable",
            "v_variable": "v_variable",
            "pressure_variable": "pressure_variable",
            "primary_colormap": "primary_colormap",
            "secondary_colormap": "secondary_colormap",
            "transparency": "transparency",
            "contour_overlay": "contour_overlay",
        }

        for arg_name, config_attr in overlay_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_attr] = getattr(args, arg_name)

        if hasattr(args, "variables") and args.variables:
            config_dict["variables"] = [v.strip() for v in args.variables.split(",")]

    def setup_logging(
        self: "MPASUnifiedCLI", config: "MPASConfig", log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        This method sets up the logging configuration for the MPASUnifiedCLI based on the provided configuration object and an optional log file path. It determines the effective log file to use by checking if a log_file argument is provided; if not, it looks for a 'log_file' attribute in the configuration object. The method then determines the logging level based on the 'log_level' attribute in the configuration, or falls back to setting it to ERROR if 'quiet' mode is enabled, DEBUG if 'verbose' mode is enabled, or INFO by default. The console handler is always attached (quiet mode only raises the threshold to ERROR), so errors are still reported under --quiet. Finally, it initializes the MPASLogger with the determined logging level and log file, and retrieves a logger instance for the current module to be used for logging messages throughout the CLI execution. This setup allows for flexible logging configurations based on user preferences specified through command-line arguments or configuration settings.

        Parameters:
            config (MPASConfig): The configuration object containing settings that may influence logging behavior, such as log level and log file path.
            log_file (Optional[str]): An optional argument specifying the log file path to use for logging output. If not provided, the method will look for a 'log_file' attribute in the configuration object.

        Returns:
            logging.Logger: A logger instance configured according to the specified log level, log file, and verbosity settings, ready to be used for logging messages throughout the CLI execution.
        """
        effective_log_file = (
            log_file if log_file is not None else getattr(config, "log_file", None)
        )

        if effective_log_file:
            effective_log_file = _resolve_against_cwd(
                effective_log_file, config.resolved_base_dir()
            )

        if config.log_level:
            log_level = getattr(logging, config.log_level)
        elif getattr(config, "quiet", False):
            log_level = logging.ERROR
        elif getattr(config, "verbose", False):
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        MPASLogger(
            name="mpasdiag",
            level=log_level,
            log_file=effective_log_file,
            verbose=True,
        )

        self.logger = get_logger(__name__)
        return self.logger

    def _confine_config_paths(
        self: "MPASUnifiedCLI", config: "MPASConfig", errors: List[str]
    ) -> None:
        """
        This method ensures that all file and directory paths specified in the configuration object are confined within a trusted base directory, preventing potential security risks associated with path traversal. It iterates over key path attributes in the configuration (grid_file, data_dir, output_dir) and attempts to resolve each path relative to the base directory. If a path resolves outside the base directory, it appends an error message to the provided errors list indicating that the specific path is not allowed. Additionally, if an output filename is specified, it sanitizes the filename to remove any potentially unsafe characters or sequences. This confinement step is crucial for maintaining security and integrity when processing user-specified paths in the MPASUnifiedCLI.

        Parameters:
            config (MPASConfig): Configuration whose path fields are confined in place.
            errors (List[str]): Accumulator for human-readable validation errors.

        Returns:
            None
        """
        base = config.resolved_base_dir()

        for attr, label in (
            ("grid_file", "Grid file"),
            ("data_dir", "Data directory"),
            ("output_dir", "Output directory"),
        ):
            value = getattr(config, attr, None)
            if not value:
                continue
            try:
                if attr == "output_dir" and value == DEFAULT_OUTPUT_DIR:
                    # The built-in default follows an explicit --base-dir
                    setattr(config, attr, str(safe_resolve_within(value, base)))
                    continue
                setattr(config, attr, _resolve_against_cwd(value, base))
            except ValueError:
                errors.append(
                    f"{label} resolves outside the trusted base directory "
                    f"'{base}': {value}. Pass --base-dir to permit this location."
                )

        if config.output:
            config.output = _sanitize_output_name(config.output)

    def _safe_output_path(
        self: "MPASUnifiedCLI", config: "MPASConfig", output_path: str
    ) -> str:
        """
        This method safely resolves the specified output path relative to the trusted output directory defined in the configuration object, ensuring that the resulting absolute path does not escape the confines of the designated output directory. It uses the safe_resolve_within utility function to perform this resolution, which checks for potential path traversal attempts and raises an error if the resolved path would be outside the allowed output directory. This safety measure is important for preventing unauthorized file writes to unintended locations on the filesystem, maintaining security and integrity when generating output files during analysis execution.

        Parameters:
            config (MPASConfig): Configuration providing the confined output_dir.
            output_path (str): The output path (without extension) to confine.

        Returns:
            str: The confined absolute output path.
        """
        return str(safe_resolve_within(output_path, config.output_dir))

    def _validate_file_path(
        self: "MPASUnifiedCLI",
        file_path: Optional[str],
        file_type: str,
        errors: List[str],
    ) -> None:
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

    def _validate_directory_path(
        self: "MPASUnifiedCLI",
        dir_path: Optional[str],
        dir_type: str,
        errors: List[str],
    ) -> bool:
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

    def _find_data_files(self: "MPASUnifiedCLI", data_path: Path) -> List[Path]:
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

        if (data_path / "diag").exists():
            data_files = list((data_path / "diag").glob(DIAG_GLOB))
            if data_files:
                return data_files

        data_files = list(data_path.glob(MPASOUT_GLOB))
        if data_files:
            return data_files

        if (data_path / "mpasout").exists():
            data_files = list((data_path / "mpasout").glob(MPASOUT_GLOB))
            if data_files:
                return data_files

        return list(data_path.rglob(DIAG_GLOB)) + list(data_path.rglob(MPASOUT_GLOB))

    def _validate_coordinate_range(
        self: "MPASUnifiedCLI",
        config: "MPASConfig",
        min_attr: str,
        max_attr: str,
        coord_type: str,
        errors: List[str],
    ) -> None:
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
            if min_val is not None and max_val is not None and min_val >= max_val:
                errors.append(f"Invalid {coord_type} range: {min_attr} >= {max_attr}")

    def _validate_cross_section_params(
        self: "MPASUnifiedCLI", config: "MPASConfig", errors: List[str]
    ) -> None:
        """
        This method performs additional validation checks specific to cross-section analysis by verifying that if the analysis type is identified as a cross-section type (e.g., 'cross', 'xsec', '3d', 'vertical'), then the required parameters for defining the cross-section path (start and end longitude and latitude) are present in the configuration object. It first checks if the 'analysis_type' attribute exists in the configuration object and if it matches any of the known cross-section analysis types. If it does, it defines a list of required attributes for cross-section analysis (start_lon, start_lat, end_lon, end_lat) and iterates through this list to check if each attribute is present and not None in the configuration object. If any required attribute is missing or None, it appends an error message to the provided errors list indicating that the specific parameter is required for cross-section analysis. This validation step ensures that users have provided all necessary information to define a valid cross-section path before attempting to execute the analysis, preventing runtime errors due to missing parameters and providing clear feedback on what needs to be specified for successful execution. By accumulating error messages in a list, this method allows for comprehensive reporting of multiple validation issues related to cross-section parameters in a single output.

        Parameters:
            config (MPASConfig): The configuration object containing the attributes for the analysis, which should include the 'analysis_type' attribute to determine if cross-section-specific validation is needed, as well as the required parameters for cross-section analysis if applicable.
            errors (List[str]): A list to which error messages will be appended if validation checks fail, allowing for accumulation of multiple validation issues to be reported together.

        Returns:
            None
        """
        if hasattr(config, "analysis_type"):
            if config.analysis_type in ["cross", "xsec", "3d", "vertical"]:
                required_attrs = ["start_lon", "start_lat", "end_lon", "end_lat"]
                for attr in required_attrs:
                    if not hasattr(config, attr) or getattr(config, attr) is None:
                        errors.append(
                            f"Cross-section analysis requires --{attr.replace('_', '-')}"
                        )
                for attr in ("start_lat", "end_lat"):
                    value = getattr(config, attr, None)
                    if value is not None and not -90.0 <= value <= 90.0:
                        errors.append(
                            f"--{attr.replace('_', '-')} {value} is outside [-90, 90]"
                        )

    def _validate_sounding_params(
        self: "MPASUnifiedCLI", config: "MPASConfig", errors: List[str]
    ) -> None:
        """
        This method verifies that a sounding analysis has the station location it needs, appending an error for each of --lon and --lat (sounding_lon and sounding_lat in YAML) that is missing. Without the check a missing location would silently default to 0, 0 and plot the wrong profile.

        Parameters:
            config (MPASConfig): The configuration to check.
            errors (List[str]): Accumulator for human-readable validation errors.

        Returns:
            None
        """
        if config.analysis_type in ["sounding", "skewt", "profile"]:
            for attr, flag in (("sounding_lon", "--lon"), ("sounding_lat", "--lat")):
                if getattr(config, attr, None) is None:
                    errors.append(f"Sounding analysis requires {flag}")

    def _report_validation_errors(self: "MPASUnifiedCLI", errors: List[str]) -> None:
        """
        This method reports the accumulated configuration validation errors to the user by logging them through the configured logger if available, or printing them to the console if no logger is set up. It first defines a header message indicating that configuration validation has failed, and then iterates through the list of error messages, logging or printing each one with a consistent format. This method centralizes the reporting of validation errors, ensuring that users receive clear and organized feedback on what issues were detected during the validation process and what needs to be corrected in their configuration settings before they can successfully execute the analysis. By providing detailed error messages, it helps users understand exactly what parameters are missing or invalid, facilitating easier troubleshooting and correction of their configuration.

        Parameters:
            errors (List[str]): A list of error messages that were accumulated during the configuration validation process, describing the specific issues that were detected with the provided configuration settings.

        Returns:
            None
        """
        error_header = "Configuration validation failed:"

        emit = self.logger if self.logger else get_logger(__name__)
        emit.error(error_header)
        for error in errors:
            emit.error("  - %s", error)

    def validate_config(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method performs comprehensive validation of the configuration object to ensure that all required parameters are specified, file paths exist, and spatial bounds are logically consistent before attempting to execute the analysis. It uses helper methods to validate individual aspects of the configuration, such as checking for the existence of required files and directories, validating coordinate ranges for latitude and longitude, and performing additional checks specific to certain analysis types (e.g., cross-section parameters). If any validation checks fail, it accumulates error messages in a list and reports them to the user through the logger or console output. The method returns True if all validation checks pass successfully, allowing execution to proceed, or False if any validation errors are detected that require user correction before execution can continue. This validation step is crucial for preventing runtime errors during analysis execution by ensuring that the configuration is complete and valid before processing begins.

        Parameters:
            config ('MPASConfig'): The configuration object containing all parameters for the analysis, which will be validated for completeness, correctness, and logical consistency before execution.

        Returns:
            bool: True if the configuration passes all validation checks and is considered valid for execution, False if any validation errors are detected that require user correction before execution can continue.
        """
        errors: List[str] = []

        self._confine_config_paths(config, errors)

        try:
            config.revalidate()
        except ValueError as exc:
            errors.append(str(exc))

        self._validate_file_path(config.grid_file, "Grid file", errors)

        is_valid_dir = self._validate_directory_path(
            config.data_dir, "Data directory", errors
        )

        if is_valid_dir:
            data_path = Path(config.data_dir)
            data_files = self._find_data_files(data_path)
            if not data_files:
                errors.append(f"No MPAS data files found in: {config.data_dir}")

        self._validate_cross_section_params(config, errors)
        self._validate_sounding_params(config, errors)

        if errors:
            self._report_validation_errors(errors)
            return False

        return True

    def _check_analysis_type_specified(
        self: "MPASUnifiedCLI", config: "MPASConfig"
    ) -> bool:
        """
        This method checks whether the 'analysis_type' attribute is specified in the configuration object, which is essential for determining which analysis workflow to execute. It verifies that the 'analysis_type' attribute exists and is not empty or None. If the analysis type is not specified, it logs an error message indicating that no analysis type was specified and returns False, signaling that execution cannot proceed without this critical piece of information. If the analysis type is specified, it returns True, allowing the execution flow to continue to the dispatching of the appropriate analysis method based on the specified type. This check ensures that users have provided the necessary information to identify which analysis they want to perform before any processing begins.

        Parameters:
            config ('MPASConfig'): The configuration object containing the parameters for the analysis, which should include the 'analysis_type' attribute that specifies which type of analysis to execute.

        Returns:
            bool: True if the 'analysis_type' attribute is specified and valid in the configuration object, False if it is missing or empty, indicating that execution cannot proceed without this information.
        """
        if not hasattr(config, "analysis_type") or not config.analysis_type:
            self._log_error("No analysis type specified")
            return False
        return True

    def _dispatch_analysis(
        self: "MPASUnifiedCLI", analysis_type: Optional[str], config: "MPASConfig"
    ) -> Optional[bool]:
        """
        This method dispatches the execution to the appropriate analysis method based on the specified analysis type in the configuration. It defines a mapping of known analysis types and their aliases to corresponding methods that implement the specific analysis workflows (e.g., precipitation, surface variables, wind vectors, cross-sections, soundings, overlays). The method iterates through this mapping to find a match for the provided analysis type and calls the corresponding method with the configuration object if a match is found. If no known analysis type matches the provided input, it returns None, indicating that the analysis type is unknown and no method was executed. This dispatching mechanism allows for flexible execution of different analysis workflows based on user input while maintaining a clean and organized structure for handling multiple types of analyses within the unified CLI framework.

        Parameters:
            analysis_type (Optional[str]): The type of analysis to execute, which may be None if not specified. This string is used to determine which analysis method to call based on predefined mappings of analysis types and their aliases.
            config ('MPASConfig'): The configuration object containing all parameters for the analysis, which will be passed to the specific analysis method that is executed based on the analysis type.

        Returns:
            Optional[bool]: The result of the executed analysis method (True for success, False for failure) if a known analysis type is matched and executed, or None if the analysis type is unknown and no method was executed.
        """
        if not analysis_type:
            return None

        analysis_map = {
            ("precipitation", "precip", "rain"): self._run_precipitation_analysis,
            ("surface", "surf", "2d"): self._run_surface_analysis,
            ("wind", "vector", "winds"): self._run_wind_analysis,
            ("cross", "xsec", "3d", "vertical"): self._run_cross_analysis,
            ("sounding", "skewt", "profile"): self._run_sounding_analysis,
            ("overlay", "complex", "multi", "composite"): self._run_overlay_analysis,
        }

        for analysis_type_aliases, analysis_method in analysis_map.items():
            if analysis_type in analysis_type_aliases:
                return analysis_method(config)

        return None

    def _log_error(
        self: "MPASUnifiedCLI", message: str, include_traceback: bool = False
    ) -> None:
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

    def _info(self: "MPASUnifiedCLI", message: str) -> None:
        """
        This helper logs an informational message through the configured logger when one is available and silently does nothing otherwise. It lets the batch helpers report skipped time steps, time-range trimming, and per-batch summaries without having to check whether logging has been initialized in every call site, which keeps the analysis methods short and free of repeated conditionals.

        Parameters:
            message (str): The informational message to log.

        Returns:
            None
        """
        if self.logger:
            self.logger.info(message)

    @staticmethod
    def _n_times(dataset: Any) -> int:
        """
        This helper returns the number of time steps in a dataset by looking up the 'Time' or 'time' dimension, falling back to a single time step when the dataset has no time dimension. It is the single source of truth used to validate time indices and time ranges before any plotting work starts.

        Parameters:
            dataset (Any): The loaded xarray (or uxarray) dataset.

        Returns:
            int: The number of time steps available in the dataset.
        """
        time_dim = "Time" if "Time" in dataset.sizes else "time"
        return int(dataset.sizes.get(time_dim, 1))

    def _check_time_index(self: "MPASUnifiedCLI", dataset: Any, time_index: int) -> int:
        """
        This method verifies that a requested time index exists in the dataset and returns it as a non-negative index, raising an IndexError with a clear message when it does not exist. A negative index is accepted as a Python-style index counting from the end (-1 is the last time step) and is normalized against the actual dataset size here, which is the first point where that size is known. An index outside the data is an error rather than something to clamp, because silently substituting another time step would produce a plot whose content and file name do not match what the user asked for. The error text states the requested index, the number of time steps, and the valid index range so the user can correct the command immediately, and a dataset with an empty time dimension is reported as having no time steps.

        Parameters:
            dataset (Any): The loaded dataset whose time dimension is checked.
            time_index (int): The requested time index; negative values count from the end.

        Returns:
            int: The validated, non-negative time index.
        """
        n_times = self._n_times(dataset)
        if n_times == 0:
            raise IndexError("the data have no time steps")
        if time_index < -n_times or time_index >= n_times:
            raise IndexError(
                f"time index {time_index} is out of range: the data have "
                f"{n_times} time step(s) (valid indices 0 to {n_times - 1}, "
                f"or -{n_times} to -1)"
            )
        return int(time_index + n_times if time_index < 0 else time_index)

    def _check_lookback(
        self: "MPASUnifiedCLI",
        config: "MPASConfig",
        time_index: int,
        dataset: Any = None,
    ) -> None:
        """
        This method verifies that a precipitation accumulation ending at the requested time index can actually be formed, raising InsufficientLookbackError when it cannot. When the dataset is supplied the decision uses its real Time coordinate by the same rule as PrecipitationDiagnostics: the output exactly N hours before the valid time must exist in the data, so a 3-hourly dataset supports a03h from its second step onward while a01h on 3-hourly output is reported as impossible, instead of assuming that one index equals one hour. Without a dataset (or when the dataset has no usable Time coordinate) hourly output is assumed and the first usable index equals the accumulation length. The error message names the requested index, the accumulation period, and why the window cannot be built.

        Parameters:
            config (MPASConfig): Configuration providing the accumulation period.
            time_index (int): The requested (non-negative) valid-time index.
            dataset (Any): The loaded dataset whose Time coordinate defines the output interval (default: None, hourly output assumed).

        Returns:
            None
        """
        hours = int(config.accumulation_period[1:3])
        if dataset is None:
            if time_index < hours:
                raise InsufficientLookbackError(
                    f"time index {time_index} cannot provide the "
                    f"{config.accumulation_period} accumulation: it needs {hours} "
                    f"earlier time step(s), so the first usable time index is {hours}"
                )
            return
        time_dim = "Time" if "Time" in dataset.sizes else "time"
        PrecipitationDiagnostics(verbose=False)._resolve_previous_index(
            dataset, time_dim, time_index, hours, config.accumulation_period
        )

    def _first_lookback_index(
        self: "MPASUnifiedCLI", dataset: Any, config: "MPASConfig"
    ) -> Tuple[int, Optional[str]]:
        """
        This method finds the earliest time index at which the configured precipitation accumulation can be formed, using the dataset's actual Time coordinate rather than assuming hourly output. Each index is tested with the same rule as the diagnostics (the accumulation start must exist in the data) and the first one that passes is returned. When no index can provide the accumulation, for example a01h on 3-hourly output, the number of time steps is returned together with the reason reported for the last index so the caller can raise a clear error.

        Parameters:
            dataset (Any): The loaded dataset providing the Time coordinate.
            config (MPASConfig): Configuration providing the accumulation period.

        Returns:
            Tuple[int, Optional[str]]: The first usable time index (equal to the number of time steps when none is usable) and the failure reason, which is None when an index was found.
        """
        n_times = self._n_times(dataset)
        reason: Optional[str] = None
        for idx in range(n_times):
            try:
                self._check_lookback(config, idx, dataset)
                return idx, None
            except InsufficientLookbackError as exc:
                reason = str(exc)
        return n_times, reason

    @staticmethod
    def _valid_datetime(dataset: Any, time_index: int) -> Optional[Any]:
        """
        This helper returns the valid time of a time step as a datetime taken from the dataset's Time coordinate, or None when the dataset has no usable Time coordinate. It lets the precipitation plot annotate the accumulation window with the time of the step actually being drawn instead of letting the plotter fall back to the configured time index.

        Parameters:
            dataset (Any): The loaded dataset providing the Time coordinate.
            time_index (int): The non-negative time index.

        Returns:
            Optional[Any]: The valid time as a python datetime, or None when unavailable.
        """
        try:
            stamp = pd.Timestamp(dataset["Time"].values[time_index])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        return None if pd.isna(stamp) else stamp.to_pydatetime()

    def _resolve_time_indices(
        self: "MPASUnifiedCLI",
        dataset: Any,
        config: "MPASConfig",
        first_index: int = 0,
    ) -> List[int]:
        """
        This method determines the time indices a batch run should process. The inclusive range comes from --time-range (time_start and time_end in the configuration) and defaults to the whole dataset when unspecified. A range that lies outside the data raises an IndexError naming the range and the number of available time steps. Indices below first_index, the earliest index with enough lookback for the requested precipitation accumulation, are dropped with an informational message so a batch starts at the first index that can produce a valid result instead of failing on the earlier ones.

        Parameters:
            dataset (Any): The loaded dataset whose time dimension bounds the range.
            config (MPASConfig): Configuration providing the optional time_start and time_end.
            first_index (int): Earliest usable time index (default: 0).

        Returns:
            List[int]: The ordered list of time indices to process, possibly empty.
        """
        n_times = self._n_times(dataset)
        start = config.time_start if config.time_start is not None else 0
        end = config.time_end if config.time_end is not None else n_times - 1

        if start >= n_times or end >= n_times:
            raise IndexError(
                f"time range [{start}, {end}] is out of range: the data have "
                f"{n_times} time step(s) (valid indices 0 to {n_times - 1})"
            )

        indices = list(range(start, end + 1))
        usable = [idx for idx in indices if idx >= first_index]

        if len(usable) < len(indices):
            self._info(
                f"Skipping time indices {indices[0]}..{first_index - 1}: "
                f"{config.accumulation_period} accumulation needs {first_index} "
                f"earlier time step(s); starting at index {first_index}"
            )

        return usable

    @staticmethod
    def _longitude_arc(lon_arr: Any) -> Tuple[float, float, bool]:
        """
        This helper computes the minimal longitude arc that covers a set of longitudes using the largest-gap method: the longitudes are wrapped to [0, 360), sorted, and the largest empty gap between neighbours (including the gap across the 0/360 seam) is taken as the part of the circle that is not covered. The arc runs from the longitude just after that gap, eastward, to the longitude just before it, so a regional mesh that straddles the Greenwich meridian or the dateline is described by its true small span instead of a whole-world extent. The start is expressed in [-180, 180) and the end is start plus the span, so the end may exceed 180 for a dateline-crossing arc, which is the convention the plotters accept (lon_min < lon_max, span at most 360). An arc covering 350 degrees or more is reported as global.

        Parameters:
            lon_arr (Any): Finite longitudes in degrees in any frame.

        Returns:
            Tuple[float, float, bool]: The arc start, the arc end (start plus span), and whether the data cover the whole globe.
        """
        values = np.asarray(lon_arr, dtype=float)
        wrapped = np.unique(np.mod(values, 360.0))
        if wrapped.size == 0:
            return -180.0, 180.0, True
        if wrapped.size == 1:
            # -180 and 180 are the same meridian, but data spanning that whole range is a global grid
            if float(np.ptp(values)) >= 350.0:
                return -180.0, 180.0, True
            start = float(wrapped[0]) - (360.0 if wrapped[0] >= 180.0 else 0.0)
            return start, start, False
        gaps = np.diff(np.concatenate([wrapped, [wrapped[0] + 360.0]]))
        k = int(np.argmax(gaps))
        span = float(360.0 - gaps[k])
        start = float(wrapped[(k + 1) % wrapped.size])
        if start >= 180.0:
            start -= 360.0
        return start, start + span, span >= 350.0

    def _resolve_longitudes(
        self: "MPASUnifiedCLI", config: "MPASConfig", lon_arr: Any
    ) -> Tuple[float, float]:
        """
        This method resolves the longitude bounds of the map extent. When both bounds are configured they are returned unchanged. A missing bound is filled from the data using the minimal covering arc (see _longitude_arc), so a regional mesh crossing the 0/360 seam or the dateline receives its small true extent. For a global mesh with no bounds the result is -180 to 180; when only one bound is given on a global mesh the other is the end of the frame that bound belongs to (180 or 360 for a minimum, -180, 0 or -360 for a maximum) limited so the span never exceeds 360 degrees, which keeps lon_max greater than lon_min. For a regional mesh with one bound the missing one is the arc end (or start) moved by whole turns into the window the given bound defines, rounded outward to whole degrees.

        Parameters:
            config (MPASConfig): Configuration providing any user-specified longitude bounds.
            lon_arr (Any): Finite longitudes of the data in degrees.

        Returns:
            Tuple[float, float]: The resolved (lon_min, lon_max) in degrees with lon_min < lon_max and a span of at most 360.
        """
        lon_min, lon_max = config.lon_min, config.lon_max
        if lon_min is not None and lon_max is not None:
            return float(lon_min), float(lon_max)

        arc_start, arc_end, is_global = self._longitude_arc(lon_arr)

        if lon_min is None and lon_max is None:
            if is_global:
                return -180.0, 180.0
            low = float(np.floor(arc_start))
            return low, max(float(np.ceil(arc_end)), low + 1.0)

        if lon_min is not None:
            low = float(lon_min)
            if is_global:
                high = min(180.0 if low < 180.0 else 360.0, low + 360.0)
            else:
                high = float(
                    np.ceil(arc_end + 360.0 * np.floor((low - arc_end) / 360.0 + 1.0))
                )
            return low, high

        assert lon_max is not None
        high = float(lon_max)
        if is_global:
            frame_min = -360.0 if high <= -180.0 else (-180.0 if high <= 180.0 else 0.0)
            low = max(frame_min, high - 360.0)
        else:
            low = float(
                np.floor(
                    arc_start + 360.0 * (np.ceil((high - arc_start) / 360.0) - 1.0)
                )
            )
        return low, high

    def _resolve_extent(
        self: "MPASUnifiedCLI", config: "MPASConfig", lon: Any, lat: Any
    ) -> GeographicBounds:
        """
        This method turns the configured spatial extent into concrete map bounds. Any bound the user did not supply (neither on the command line nor in YAML) is taken from the extent of the data: a global mesh yields the full globe (-180 to 180 and -90 to 90) and a regional mesh yields its own latitude range rounded outward to whole degrees and the minimal longitude arc covering the data (largest-gap method, see _resolve_longitudes), so meshes that cross the Greenwich meridian or the dateline get their true small extent rather than a whole-world one. When only one longitude bound is given, the other is filled from the data in the same frame so that lon_max is always greater than lon_min and the span never exceeds 360 degrees. The resolved bounds are validated with the same rules as the configuration and a ValueError naming the values is raised when they are inconsistent.

        Parameters:
            config (MPASConfig): Configuration providing any user-specified bounds.
            lon (Any): Longitudes of the data points in degrees.
            lat (Any): Latitudes of the data points in degrees.

        Returns:
            GeographicBounds: The resolved (lon_min, lon_max, lat_min, lat_max) bounds in degrees.
        """
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        lon_arr = lon_arr[np.isfinite(lon_arr)]
        lat_arr = lat_arr[np.isfinite(lat_arr)]

        lon_min, lon_max = self._resolve_longitudes(config, lon_arr)
        lat_min, lat_max = config.lat_min, config.lat_max

        if lat_min is None or lat_max is None:
            global_lat = lat_arr.size == 0 or float(np.ptp(lat_arr)) >= 178.0
            if lat_min is None:
                lat_min = (
                    -90.0 if global_lat else max(-90.0, float(np.floor(lat_arr.min())))
                )
            if lat_max is None:
                lat_max = (
                    90.0 if global_lat else min(90.0, float(np.ceil(lat_arr.max())))
                )

        resolved = replace(
            config,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
        )
        assert resolved.lon_min is not None and resolved.lon_max is not None
        assert resolved.lat_min is not None and resolved.lat_max is not None
        return GeographicBounds(
            resolved.lon_min, resolved.lon_max, resolved.lat_min, resolved.lat_max
        )

    @staticmethod
    def _load_data(
        loader: Any, config: "MPASConfig", variables: Optional[List[str]] = None
    ) -> Any:
        """
        This helper calls a processor's load method (load_2d_data or load_3d_data) with the configured data directory and, only when the user supplied --chunk-size (or chunk_size in YAML), the chunk size along nCells, so --chunk-size controls how the data are chunked while the default loading behavior is unchanged when the option is absent. When the analysis names the variables it reads, only those (plus the time bookkeeping the loader always keeps) are loaded, which is the same selection the batch workers use; MPAS output files hold over a hundred variables, so this saves seconds per run and per MPI rank. If any of the named variables (other than the optional vertical-coordinate helpers) is not in the files, everything is loaded instead, so the analysis reports the missing variable together with the full list of available ones.

        Parameters:
            loader (Any): The bound load_2d_data or load_3d_data method of a processor.
            config (MPASConfig): Configuration providing the data directory and the optional chunk size.
            variables (Optional[List[str]]): Variables the analysis reads, or None to load every variable (default: None).

        Returns:
            Any: The processor returned by the loader, with its dataset loaded.
        """
        kwargs: Dict[str, Any] = {}
        if config.chunk_size is not None:
            kwargs["chunk_size"] = config.chunk_size
        if variables and all(variables):
            try:
                processor = loader(config.data_dir, variables=list(variables), **kwargs)
            except MissingVariablesError:
                processor = None
            dataset = getattr(processor, "dataset", None)
            if dataset is not None and all(
                name in dataset
                for name in variables
                if name not in CROSS_SECTION_AUX_VARS
            ):
                return processor
        return loader(config.data_dir, **kwargs)

    @staticmethod
    def _cmap(config: "MPASConfig") -> Optional[str]:
        """
        This helper converts the configured colormap name into the value plot styles expect, mapping the sentinel 'default' (automatic selection) to None.

        Parameters:
            config (MPASConfig): Configuration providing the colormap name.

        Returns:
            Optional[str]: The colormap name, or None to let the plotter choose.
        """
        return config.colormap if config.colormap != "default" else None

    @staticmethod
    def _output_name_kwargs(batch_method: Any, config: "MPASConfig") -> Dict[str, Any]:
        """
        This helper forwards the user's --output name to a parallel batch method as output_name (so batch files are named '<output>_<time>' exactly as in the serial path), when the method accepts it and a name was given.

        Parameters:
            batch_method (Any): The parallel create_batch_*_parallel callable.
            config (MPASConfig): Configuration providing the (already sanitized) output name.

        Returns:
            Dict[str, Any]: {'output_name': name} or an empty dict.
        """
        return _accepted_kwargs(batch_method, output_name=config.output)

    @staticmethod
    def _projection_kwargs(config: "MPASConfig") -> Dict[str, Any]:
        """
        This helper collects the map projection settings from the configuration as the keyword arguments accepted by the SurfaceMapStyle, PrecipitationRenderStyle, and WindPlotStyle dataclasses, so every serial plot honors --projection, --central-longitude, --central-latitude and a YAML proj_kwargs mapping.

        Parameters:
            config (MPASConfig): Configuration providing the projection settings.

        Returns:
            Dict[str, Any]: Projection keyword arguments for a plot style.
        """
        return {
            "projection": config.projection,
            "central_longitude": config.central_longitude,
            "central_latitude": config.central_latitude,
            "proj_kwargs": config.proj_kwargs,
        }

    def _render_options(
        self: "MPASUnifiedCLI",
        config: "MPASConfig",
        wind_level: Optional[Tuple[Optional[Union[str, int, float]], str]] = None,
    ) -> BatchRenderOptions:
        """
        This method builds the shared rendering options forwarded to the parallel batch workers from the configuration: color limits, figure size and resolution, the map projection controls, the custom title, the wind level (a readable label plus, for 3D wind components, the raw level selection), background colormap, the cross-section maximum height, the vector style, and the data chunk size. Passing these through is what makes a parallel batch render identically to the serial path.

        Parameters:
            config (MPASConfig): Configuration providing the rendering settings.
            wind_level (Optional[Tuple[Optional[Union[str, int, float]], str]]): The resolved (selection, label) pair from _wind_level_info for wind plots, or None to use the configured level string as the label with no 3D selection.

        Returns:
            BatchRenderOptions: The options object handed to the parallel batch processors.
        """
        selection, label = (
            wind_level if wind_level is not None else (None, config.wind_level)
        )
        return BatchRenderOptions(
            clim_min=config.clim_min,
            clim_max=config.clim_max,
            figure_size=config.figure_size,
            dpi=config.dpi,
            title=config.title,
            wind_level=label,
            wind_level_selection=selection,
            background_colormap=config.background_colormap,
            max_height=config.max_height,
            vector_color=config.vector_color,
            vector_alpha=config.vector_alpha,
            chunk_size=config.chunk_size,
            **self._projection_kwargs(config),
        )

    @staticmethod
    def _parse_wind_level(
        text: Any,
    ) -> Tuple[Union[str, int, float], str]:
        """
        This helper interprets the --wind-level value used to pick a horizontal slice of 3D wind components. 'surface' (the lowest model level) and 'top' are passed through; a value ending in 'hPa' (for example '850hPa' or '850 hPa') is a pressure converted to Pa; a plain integer below 2000 is a model level index; an integer of 2000 or more, or any number with a decimal point, is a pressure in Pa (for example 85000). The returned selection is what MPAS3DProcessor.get_3d_variable_data expects (a string, an int model level, or a float pressure in Pa) and the label is a readable description for the plot title. Anything else raises a ValueError listing the accepted forms.

        Parameters:
            text (Any): The configured wind level value.

        Returns:
            Tuple[Union[str, int, float], str]: The level selection for the processor and a readable label for titles.
        """
        raw = str(text).strip()
        lowered = raw.lower().replace(" ", "")
        accepted = (
            "use 'surface', 'top', an integer model level (e.g. 5), a pressure in Pa "
            "(e.g. 85000) or a pressure in hPa (e.g. '850hPa')"
        )
        if lowered in ("surface", "top"):
            return lowered, lowered
        try:
            if lowered.isdigit() and int(lowered) < 2000:
                return int(lowered), f"model level {int(lowered)}"
            scale = 100.0 if lowered.endswith("hpa") else 1.0
            number = float(lowered.removesuffix("hpa").removesuffix("pa"))
            if not number > 0 or not np.isfinite(number):
                raise ValueError
            return number * scale, f"{number * scale / 100.0:g} hPa"
        except ValueError:
            raise ValueError(f"Invalid --wind-level '{raw}': {accepted}") from None

    def _wind_level_info(
        self: "MPASUnifiedCLI", processor: Any, config: "MPASConfig"
    ) -> Tuple[Optional[Union[str, int, float]], str]:
        """
        This method decides how the wind components are read. When the processor is a 3D processor (chosen because the u/v variables have a vertical dimension) the configured --wind-level is parsed into a level selection for MPAS3DProcessor.get_3d_variable_data together with a readable label, and an invalid value raises ValueError. For 2D wind variables such as u10/v10 there is nothing to select, so no selection is returned and the configured level string is used unchanged as the plot label.

        Parameters:
            processor (Any): The processor used for the wind plots (2D or 3D).
            config (MPASConfig): Configuration providing the wind level string.

        Returns:
            Tuple[Optional[Union[str, int, float]], str]: The 3D level selection (None for 2D variables) and the label for the plot title.
        """
        if isinstance(processor, MPAS3DProcessor):
            return self._parse_wind_level(config.wind_level)
        return None, config.wind_level

    def _run_batch_cleanup(self: "MPASUnifiedCLI", cleanup: Any) -> None:
        """
        This method invokes the optional cleanup callable of a serial batch after a failed step, ignoring any exception it raises so the cleanup can never abort the remaining steps.

        Parameters:
            cleanup (Any): Optional callable that releases resources (typically closing the open figure), or None.

        Returns:
            None
        """
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass

    def _run_batch_loop(
        self: "MPASUnifiedCLI",
        label: str,
        time_indices: List[int],
        step: Any,
        cleanup: Any = None,
    ) -> bool:
        """
        This method runs a serial batch by calling a per-time-step function for every index, isolating failures so one bad step does not abort the rest. A step that raises InsufficientLookbackError is skipped with an informational message and is not counted as a failure; any other exception is logged with its time index and counted as a failure, after the optional cleanup callable (typically closing the open figure) is invoked. When the loop ends it logs a summary of succeeded, failed, and skipped steps. The batch is reported as successful only when no step failed and at least one step produced output, which is what drives the non-zero exit status when every step fails or when there was nothing to process.

        Parameters:
            label (str): Description of the product being created, used in messages (e.g. "surface maps").
            time_indices (List[int]): The time indices to process.
            step (Any): Callable taking a time index and producing the output for it.
            cleanup (Any): Optional callable invoked after a failed step to release resources.

        Returns:
            bool: True if at least one step succeeded and none failed, False otherwise.
        """
        if not time_indices:
            self._log_error(
                f"No time steps to process for {label}: the selected time range "
                f"and accumulation lookback leave nothing to do"
            )
            return False

        succeeded = failed = skipped = 0

        for time_idx in time_indices:
            try:
                step(time_idx)
                succeeded += 1
            except InsufficientLookbackError as exc:
                skipped += 1
                self._info(f"Skipping time index {time_idx}: {exc}")
            except Exception as exc:
                failed += 1
                self._log_error(f"Failed {label} for time index {time_idx}: {exc}")
                self._run_batch_cleanup(cleanup)

        if is_root_process():
            print(
                f"Batch {label}: {succeeded} succeeded, {failed} failed, "
                f"{skipped} skipped (of {len(time_indices)} time steps)"
            )

        if succeeded == 0 and failed == 0:
            self._log_error(f"No {label} were created: every time step was skipped")
            return False

        return failed == 0

    def _finish_parallel(
        self: "MPASUnifiedCLI", created_files: Optional[List[str]], label: str
    ) -> bool:
        """
        This method converts the result of a parallel batch call into a success flag. The parallel processors return the list of created files, annotated with per-step outcome counts when they come from the package's own wrappers. The batch succeeds only when no step failed and at least one file was produced; a None result (a non-root MPI rank, which holds no aggregated results) is treated as success because the root rank reports the outcome. A summary line is printed and failures are logged as errors so the exit status is non-zero whenever any step failed.

        Parameters:
            created_files (Optional[List[str]]): Files created by the parallel batch, or None on non-root MPI ranks.
            label (str): Description of the product being created, used in messages.

        Returns:
            bool: True if the batch fully succeeded, False if any step failed or nothing was created.
        """
        if created_files is None:
            return True

        failed = int(getattr(created_files, "failed", 0))
        successful = int(getattr(created_files, "successful", len(created_files)))

        print(f"Created {len(created_files)} {label}")

        if failed:
            self._log_error(f"{failed} time step(s) failed while creating {label}")
            return False
        if successful == 0:
            self._log_error(f"No {label} were created")
            return False
        return True

    def _output_path(
        self: "MPASUnifiedCLI",
        config: "MPASConfig",
        default_name: str,
        time_str: str,
    ) -> str:
        """
        This method chooses and confines the output path (without extension) for one plot. A user-supplied --output name is used as given for a single plot, and in batch mode the time tag is appended so successive time steps do not overwrite one another; without --output the descriptive default name is placed in the output directory. The result is always resolved within the output directory.

        Parameters:
            config (MPASConfig): Configuration providing the output name, output directory, and batch flag.
            default_name (str): Descriptive file name used when no --output was given.
            time_str (str): Time tag appended to a user-supplied name in batch mode.

        Returns:
            str: The confined absolute output path without extension.
        """
        if config.output:
            name = f"{config.output}_{time_str}" if config.batch_mode else config.output
        else:
            name = os.path.join(config.output_dir, default_name)
        return self._safe_output_path(config, name)

    def _create_skewt(
        self: "MPASUnifiedCLI",
        processor: "MPAS3DProcessor",
        plotter: "MPASSkewTPlotter",
        sounding_diag: "SoundingDiagnostics",
        config: "MPASConfig",
        sounding_lon: float,
        sounding_lat: float,
        show_indices: bool,
        show_parcel: bool,
        time_index: int,
    ) -> str:
        """
        This method extracts the sounding profile for one time index, optionally computes the thermodynamic indices, draws the Skew-T diagram, and saves it in every configured output format. The time index is validated first so an out-of-range index raises IndexError. The file name is derived from the station coordinates and valid time unless the user supplied --output, in which case the time tag is appended in batch mode.

        Parameters:
            processor ('MPAS3DProcessor'): The 3D processor with the loaded dataset.
            plotter ('MPASSkewTPlotter'): The Skew-T plotter.
            sounding_diag ('SoundingDiagnostics'): Diagnostics used to extract the profile and compute indices.
            config ('MPASConfig'): The configuration with the output settings and title.
            sounding_lon (float): Sounding longitude in degrees.
            sounding_lat (float): Sounding latitude in degrees.
            show_indices (bool): Whether to compute and show thermodynamic indices.
            show_parcel (bool): Whether to plot the lifted parcel profile.
            time_index (int): The time index to plot.

        Returns:
            str: The output path (without extension) of the saved diagram.
        """
        time_index = self._check_time_index(processor.dataset, time_index)

        profile = sounding_diag.extract_sounding_profile(
            processor,
            sounding_lon,
            sounding_lat,
            time_index=time_index,
            max_distance_km=getattr(config, "sounding_max_distance_km", None),
        )
        indices = self._compute_sounding_indices(show_indices, sounding_diag, profile)
        time_str = MPASDateTimeUtils.get_time_info(
            processor.dataset, time_index, verbose=False
        )
        lon_tag, lat_tag = self._build_skewt_tags(profile)

        title = f"MPAS Skew-T | {lon_tag}, {lat_tag} | Valid: {time_str}"
        save_path = self._output_path(
            config,
            f"mpas_skewt_{lon_tag.replace('.', 'p')}_{lat_tag.replace('.', 'p')}"
            f"{'_indices' if show_indices else ''}{'_parcel' if show_parcel else ''}"
            f"_valid_{time_str}",
            time_str,
        )

        plotter.create_skewt_diagram(
            pressure=profile["pressure"],
            temperature=profile["temperature"],
            dewpoint=profile["dewpoint"],
            u_wind=profile["u_wind"],
            v_wind=profile["v_wind"],
            title=config.title or title,
            indices=indices,
            show_parcel=show_parcel,
            save_path=None,
            location=profile,
        )
        plotter.save_plot(save_path, formats=config.output_formats or ["png"])
        plotter.close_plot()
        return save_path

    def _check_required_arguments(
        self: "MPASUnifiedCLI", merged: Dict[str, Any]
    ) -> None:
        """
        This method enforces the arguments each subcommand requires after the YAML configuration has been merged with the command-line flags, so that a value supplied by either source satisfies the requirement (for example --grid-file, --data-dir, --variable, the cross-section end points, the sounding location, or the overlay type). When something is missing it calls the subparser's error method, which prints the usage and the standard 'the following arguments are required' message and exits with status 2, exactly like argparse would for a missing required flag.

        Parameters:
            merged (Dict[str, Any]): The merged YAML and command-line configuration values.

        Returns:
            None
        """
        analysis_type = merged.get("analysis_type")
        required = self._required_args.get(str(analysis_type), [])
        config_names = {"lon": "sounding_lon", "lat": "sounding_lat"}

        missing = [
            flag
            for dest, flag in required
            if merged.get(config_names.get(dest, dest)) in (None, "")
        ]

        if missing:
            self._subparsers_by_name[str(analysis_type)].error(
                "the following arguments are required (on the command line or in "
                "the YAML configuration): " + ", ".join(missing)
            )

    def _report_early_error(
        self: "MPASUnifiedCLI", args: argparse.Namespace, message: str
    ) -> None:
        """
        This method reports an error that occurs while the configuration is still being built (an unreadable YAML file, an invalid value, conflicting flags). Logging is initialized first from the flags that were typed (--log-level, --quiet, --log-file, --base-dir) so the message goes through the normal logger and honors the requested level instead of surfacing as an 'Unexpected error', then the message is reported in the standard validation-failure format.

        Parameters:
            args (argparse.Namespace): The parsed command-line arguments.
            message (str): The description of the problem.

        Returns:
            None
        """
        try:
            bootstrap = MPASConfig(
                log_level=getattr(args, "log_level", None),
                quiet=bool(getattr(args, "quiet", False))
                and not getattr(args, "verbose", False),
                verbose=bool(getattr(args, "verbose", False)),
                base_dir=getattr(args, "base_dir", None),
            )
            self.setup_logging(bootstrap, getattr(args, "log_file", None))
        except Exception:
            pass
        self._report_validation_errors([message])

    def _print_performance_summary(
        self: "MPASUnifiedCLI", config: "MPASConfig"
    ) -> None:
        """
        This method prints a performance summary of the analysis execution when the logger is configured, verbose output was requested, and quiet mode is not active. It calls the print_summary method of the PerformanceMonitor to output the elapsed time of each stage; the total counts only top-level timers so nested stages are never double-counted. When --quiet is given, or verbose output was not requested, nothing is printed.

        Parameters:
            config ('MPASConfig'): The configuration object containing the verbose and quiet flags that decide whether the summary is printed.

        Returns:
            None
        """
        if (
            self.logger
            and getattr(config, "verbose", False)
            and not getattr(config, "quiet", False)
        ):
            if self.perf_monitor is not None:
                self.perf_monitor.print_summary()

    def run_analysis(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method serves as the main entry point for executing the analysis based on the provided configuration. It initializes a PerformanceMonitor, checks that an analysis type is specified, and dispatches to the matching analysis method. Under --quiet, standard output is suppressed for the duration of the run while errors still reach the logger. The method returns True only when the analysis completed without any failed step: batch analyses propagate their per-step outcome so an all-failed or partially failed batch yields False (and a non-zero exit status). Invalid time selections (IndexError) and precipitation accumulations that reach before the first time step (InsufficientLookbackError) are reported with a clear message, KeyboardInterrupt is reported as a user interruption, and any other exception is logged as an analysis failure.

        Parameters:
            config ('MPASConfig'): The configuration object containing all parameters for the analysis, including the analysis type, file paths, spatial bounds, output preferences, and processing options.

        Returns:
            bool: True if the analysis completed and no step failed, False if any issue occurred or the analysis type is missing or unknown.
        """
        succeeded = False

        try:
            # The monitor stays silent while timing; the outcome-aware messages are printed afterwards
            self.perf_monitor = PerformanceMonitor(quiet=True)

            if not self._check_analysis_type_specified(config):
                return False

            analysis_type = config.analysis_type

            silence: Any = (
                contextlib.redirect_stdout(io.StringIO())
                if config.quiet
                else contextlib.nullcontext()
            )

            with silence:
                with self.perf_monitor.timer("Total analysis"):
                    success = self._dispatch_analysis(analysis_type, config)

                    if success is None:
                        self._log_error(f"Unknown analysis type: {analysis_type}")
                        return False

                succeeded = bool(success)
                if is_root_process():
                    self._print_performance_summary(config)

            return succeeded

        except KeyboardInterrupt:
            self._log_error("Analysis interrupted by user")
            return False
        except InsufficientLookbackError as e:
            self._log_error(f"Cannot compute the precipitation accumulation: {e}")
            return False
        except IndexError as e:
            self._log_error(f"Invalid time selection: {e}")
            return False
        except Exception as e:
            include_traceback = bool(getattr(config, "verbose", False))
            self._log_error(
                f"Analysis failed: {e}", include_traceback=include_traceback
            )
            return False
        finally:
            self._print_timing_messages(config, succeeded)

    def _print_timing_messages(
        self: "MPASUnifiedCLI", config: "MPASConfig", succeeded: bool
    ) -> None:
        """
        This method prints the per-operation timing lines recorded by the performance monitor, once the outcome of the analysis is known. A successful run prints "<operation> completed in X seconds" for every timed operation, whereas a failed run (an exception, an unknown analysis type, or a batch that reported failed steps) prints "<operation> failed after X seconds" so a failing command never claims it completed. Nothing is printed under --quiet.

        Parameters:
            config ('MPASConfig'): The configuration providing the quiet flag.
            succeeded (bool): True when the analysis completed without any failed step.

        Returns:
            None
        """
        if (
            self.perf_monitor is None
            or getattr(config, "quiet", False)
            or not is_root_process()
        ):
            return

        verb = "completed in" if succeeded else "failed after"

        for operation, duration in self.perf_monitor.durations.items():
            print(f"{operation} {verb} {duration.total_seconds():.2f} seconds")

    def _run_precipitation_batch(
        self: "MPASUnifiedCLI",
        processor: "MPAS2DProcessor",
        plotter: "MPASPrecipitationPlotter",
        config: "MPASConfig",
    ) -> bool:
        """
        This method executes the batch precipitation workflow over the selected time steps. The time steps come from --time-range (inclusive, defaulting to the whole dataset) and start at the first index with enough lookback for the accumulation period, with the skipped leading steps reported as information rather than failures. With parallel processing enabled it hands the indices, bounds, and every plot option (colormap, color limits, levels, title, figure size, dpi, projection controls) to ParallelPrecipitationProcessor; otherwise it loops over the steps serially so the same options are honored. Per-step failures are counted rather than aborting the batch.

        Parameters:
            processor ('MPAS2DProcessor'): The 2D processor with the loaded dataset and coordinate access.
            plotter ('MPASPrecipitationPlotter'): The plotter used for serial rendering.
            config ('MPASConfig'): The configuration with the variable, bounds, accumulation period, plot options, and time selection.

        Returns:
            bool: True if at least one map was created and no step failed, False otherwise.
        """
        assert processor.dataset is not None
        first_index, lookback_reason = self._first_lookback_index(
            processor.dataset, config
        )
        if first_index >= self._n_times(processor.dataset) and lookback_reason:
            raise InsufficientLookbackError(lookback_reason)
        time_indices = self._resolve_time_indices(
            processor.dataset, config, first_index
        )
        lon, lat = processor.extract_2d_coordinates_for_variable(config.variable)
        bounds = self._resolve_extent(config, lon, lat)

        if getattr(config, "parallel", False):
            self._info("Using parallel processing for batch precipitation analysis")
            if not time_indices:
                return self._run_batch_loop("precipitation maps", [], lambda i: None)
            title = (
                config.title.replace("{", "{{").replace("}", "}}")
                if config.title
                else None
            )
            created_files = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                processor,
                config.output_dir,
                bounds,
                var_name=config.variable,
                accum_period=config.accumulation_period,
                plot_type=getattr(config, "plot_type", "scatter"),
                grid_resolution=getattr(config, "grid_resolution", None),
                formats=config.output_formats or ["png"],
                time_indices=time_indices,
                n_processes=config.workers,
                style=PrecipitationMapStyle(
                    custom_title_template=title,
                    colormap=self._cmap(config),
                    levels=config.levels,
                    display_units=config.precip_units,
                    threshold=config.precip_threshold,
                ),
                remap_config=RemapConfig(
                    remap_engine=config.remap_engine,
                    remap_method=config.remap_method,
                ),
                render=self._render_options(config),
                **self._output_name_kwargs(
                    ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel,
                    config,
                ),
            )
            return self._finish_parallel(created_files, "precipitation maps")

        return self._run_batch_loop(
            "precipitation maps",
            time_indices,
            lambda idx: self._run_precipitation_single(
                processor, plotter, processor.dataset, config, idx, bounds
            ),
            plotter.close_plot,
        )

    def _run_precipitation_single(
        self: "MPASUnifiedCLI",
        processor: "MPAS2DProcessor",
        plotter: "MPASPrecipitationPlotter",
        dataset: Any,
        config: "MPASConfig",
        time_index: Optional[int] = None,
        bounds: Optional[GeographicBounds] = None,
    ) -> str:
        """
        This method creates and saves one precipitation map. The time index defaults to config.time_index and is validated against the data (an out-of-range index raises IndexError, a negative index counts from the end, and an index for which the dataset's Time coordinate has no output at the accumulation start raises InsufficientLookbackError) so the plot and file name always correspond to the real valid time. The accumulation is computed with PrecipitationDiagnostics, the spatial bounds default to the extent of the data when none were configured, and every plot option (colormap, levels, color limits, title, projection controls, grid resolution, display units, highlight threshold) is applied. The real valid time and the accumulation DataArray (with its valid_time and accumulation_start_time metadata) are handed to the plotter so the 'Accumulation:' annotation shows the true window for every step of a serial batch, not the window of config.time_index. In batch mode a user-supplied --output name receives the time tag so steps do not overwrite each other.

        Parameters:
            processor ('MPAS2DProcessor'): The processor providing coordinates and data type.
            plotter ('MPASPrecipitationPlotter'): The plotter that renders and saves the map.
            dataset (Any): The loaded dataset used for the accumulation.
            config ('MPASConfig'): The configuration with the variable, accumulation, and plot options.
            time_index (Optional[int]): The valid-time index to plot (default: config.time_index).
            bounds (Optional[GeographicBounds]): Pre-resolved map bounds (default: resolved from the data).

        Returns:
            str: The output path (without extension) of the saved plot.
        """
        time_index = config.time_index if time_index is None else time_index
        time_index = self._check_time_index(dataset, time_index)
        self._check_lookback(config, time_index, dataset)

        lon, lat = processor.extract_2d_coordinates_for_variable(config.variable)
        bounds = bounds or self._resolve_extent(config, lon, lat)

        precip_diag = PrecipitationDiagnostics(verbose=config.verbose)

        precip_data = precip_diag.compute_precipitation_difference(
            dataset,
            time_index,
            config.variable,
            config.accumulation_period,
            data_type=getattr(processor, "data_type", "UXarray"),
        )

        time_str = MPASDateTimeUtils.get_time_info(dataset, time_index, verbose=False)
        plot_type = getattr(config, "plot_type", "scatter")

        plotter.create_precipitation_map(
            lon,
            lat,
            precip_data.values,
            bounds,
            accum_period=config.accumulation_period,
            time_end=self._valid_datetime(dataset, time_index),
            data_array=precip_data,
            var_name=config.variable,
            dataset=dataset,
            config=config,
            style=PrecipitationRenderStyle(
                title=config.title
                or default_precipitation_title(config.variable, time_str),
                plot_type=plot_type,
                colormap=self._cmap(config),
                levels=config.levels,
                clim_min=config.clim_min,
                clim_max=config.clim_max,
                grid_resolution=getattr(config, "grid_resolution", None),
                display_units=config.precip_units,
                threshold=config.precip_threshold,
                **self._projection_kwargs(config),
            ),
        )

        output_path = self._output_path(
            config,
            f"mpas_precipitation_map_vartype_{sanitize_filename_component(config.variable)}"
            f"_acctype_{sanitize_filename_component(config.accumulation_period)}"
            f"_valid_{time_str}_ptype_{sanitize_filename_component(plot_type)}",
            time_str,
        )

        plotter.save_plot(output_path, formats=config.output_formats or ["png"])
        plotter.close_plot()

        print(f"Precipitation plot saved: {output_path}")
        return output_path

    def _run_precipitation_analysis(
        self: "MPASUnifiedCLI", config: "MPASConfig"
    ) -> bool:
        """
        This method orchestrates the precipitation analysis. It loads the grid and diagnostic data, creates the plotter with the configured figure size and dpi, ensures the output directory exists, and runs either the batch workflow or a single time step. The return value reflects the outcome: a batch returns False when any step failed (or none succeeded), and a single step that raises is reported by run_analysis, so the command exits non-zero on any failure.

        Parameters:
            config ('MPASConfig'): The configuration with the paths, variable, accumulation period, time selection, and plot options.

        Returns:
            bool: True if the analysis completed without failed steps, False otherwise.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        with self.perf_monitor.timer("Precipitation analysis"):
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = self._load_data(
                processor.load_2d_data,
                config,
                PRECIP_REQUIRED_VARS.get(config.variable, [config.variable]),
            )
            dataset = processor.dataset

            plotter = MPASPrecipitationPlotter(
                figsize=config.figure_size, dpi=config.dpi
            )

            os.makedirs(config.output_dir, exist_ok=True)

            if config.batch_mode:
                return (
                    self._run_precipitation_batch(processor, plotter, config)
                    is not False
                )

            self._run_precipitation_single(processor, plotter, dataset, config)

        return True

    def _run_surface_batch(
        self: "MPASUnifiedCLI",
        processor: "MPAS2DProcessor",
        plotter: "MPASSurfacePlotter",
        config: "MPASConfig",
    ) -> bool:
        """
        This method executes the batch surface-variable workflow over the selected time steps (--time-range inclusive, defaulting to the whole dataset). With parallel processing it passes the indices, resolved bounds, and every plot option (plot type, colormap, color limits, levels, title, grid resolution, figure size, dpi, projection controls) to ParallelSurfaceProcessor; otherwise it loops serially over the steps with the same options. Failures are counted per step instead of aborting the batch.

        Parameters:
            processor ('MPAS2DProcessor'): The 2D processor with the loaded dataset.
            plotter ('MPASSurfacePlotter'): The plotter used for serial rendering.
            config ('MPASConfig'): The configuration with the variable, bounds, plot options, and time selection.

        Returns:
            bool: True if at least one map was created and no step failed, False otherwise.
        """
        assert processor.dataset is not None
        time_indices = self._resolve_time_indices(processor.dataset, config)
        lon, lat = processor.extract_2d_coordinates_for_variable(config.variable)
        bounds = self._resolve_extent(config, lon, lat)

        if getattr(config, "parallel", False):
            self._info("Using parallel processing for batch surface analysis")
            if not time_indices:
                return self._run_batch_loop("surface maps", [], lambda i: None)
            created_files = ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                processor,
                config.output_dir,
                bounds,
                var_name=config.variable,
                formats=config.output_formats or ["png"],
                grid_resolution=getattr(config, "grid_resolution", None),
                time_indices=time_indices,
                n_processes=config.workers,
                style=SurfaceBatchStyle(
                    plot_type=config.plot_type,
                    clim_min=config.clim_min,
                    clim_max=config.clim_max,
                    colormap=self._cmap(config),
                    levels=config.levels,
                    num_levels=config.contour_levels,
                    **_accepted_kwargs(
                        SurfaceBatchStyle, interpolation=config.interpolation_method
                    ),
                ),
                remap_config=RemapConfig(
                    remap_engine=config.remap_engine,
                    remap_method=config.remap_method,
                ),
                render=self._render_options(config),
                **self._output_name_kwargs(
                    ParallelSurfaceProcessor.create_batch_surface_maps_parallel,
                    config,
                ),
            )
            return self._finish_parallel(created_files, "surface maps")

        return self._run_batch_loop(
            "surface maps",
            time_indices,
            lambda idx: self._run_surface_single(
                processor, plotter, processor.dataset, config, idx, bounds
            ),
            plotter.close_plot,
        )

    def _run_surface_single(
        self: "MPASUnifiedCLI",
        processor: "MPAS2DProcessor",
        plotter: "MPASSurfacePlotter",
        dataset: Any,
        config: "MPASConfig",
        time_index: Optional[int] = None,
        bounds: Optional[GeographicBounds] = None,
    ) -> str:
        """
        This method creates and saves one surface-variable map. The time index defaults to config.time_index and is validated (an out-of-range index raises IndexError instead of being clamped), the bounds default to the extent of the data when none were configured, and the plot honors the configured plot type, colormap, levels, color limits, grid resolution, title, and projection controls. In batch mode a user-supplied --output name receives the time tag so steps do not overwrite each other.

        Parameters:
            processor ('MPAS2DProcessor'): The processor providing the variable data and coordinates.
            plotter ('MPASSurfacePlotter'): The plotter that renders and saves the map.
            dataset (Any): The loaded dataset used for the time label.
            config ('MPASConfig'): The configuration with the variable and plot options.
            time_index (Optional[int]): The time index to plot (default: config.time_index).
            bounds (Optional[GeographicBounds]): Pre-resolved map bounds (default: resolved from the data).

        Returns:
            str: The output path (without extension) of the saved plot.
        """
        time_index = config.time_index if time_index is None else time_index
        time_index = self._check_time_index(dataset, time_index)

        var_data = processor.get_2d_variable_data(config.variable, time_index)
        lon, lat = processor.extract_2d_coordinates_for_variable(
            config.variable, var_data
        )
        bounds = bounds or self._resolve_extent(config, lon, lat)

        time_str = MPASDateTimeUtils.get_time_info(dataset, time_index, verbose=False)

        plotter.create_surface_map(
            lon,
            lat,
            var_data.values,
            config.variable,
            bounds,
            style=SurfaceMapStyle(
                title=config.title
                or f"MPAS Surface | Variable: {config.variable} | Valid: {time_str}",
                plot_type=config.plot_type,
                colormap=self._cmap(config),
                levels=config.levels,
                num_levels=config.contour_levels,
                clim_min=config.clim_min,
                clim_max=config.clim_max,
                grid_resolution=getattr(config, "grid_resolution", None),
                **_accepted_kwargs(
                    SurfaceMapStyle, interpolation=config.interpolation_method
                ),
                **self._projection_kwargs(config),
            ),
            data_array=var_data,
            dataset=processor.dataset,
            config=config,
        )

        output_path = self._output_path(
            config,
            f"mpas_surface_{sanitize_filename_component(config.variable)}"
            f"_{sanitize_filename_component(config.plot_type)}_valid_{time_str}",
            time_str,
        )

        plotter.save_plot(output_path, formats=config.output_formats or ["png"])
        plotter.close_plot()

        print(f"Surface plot saved: {output_path}")
        return output_path

    def _run_surface_analysis(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method orchestrates the surface-variable analysis: it loads the grid and data, creates the plotter with the configured figure size and dpi, ensures the output directory exists, and runs the batch workflow or a single time step. A batch returns False when any step failed (for example an unknown variable) or when nothing was created, so the command exits non-zero; a failing single step raises and is reported by run_analysis.

        Parameters:
            config ('MPASConfig'): The configuration with the paths, variable, time selection, and plot options.

        Returns:
            bool: True if the analysis completed without failed steps, False otherwise.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        with self.perf_monitor.timer("Surface analysis"):
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = self._load_data(
                processor.load_2d_data, config, [config.variable]
            )
            dataset = processor.dataset

            plotter = MPASSurfacePlotter(figsize=config.figure_size, dpi=config.dpi)

            os.makedirs(config.output_dir, exist_ok=True)

            if config.batch_mode:
                return self._run_surface_batch(processor, plotter, config) is not False

            self._run_surface_single(processor, plotter, dataset, config)

        return True

    def _run_wind_batch(
        self: "MPASUnifiedCLI",
        processor: Any,
        plotter: "MPASWindPlotter",
        config: "MPASConfig",
    ) -> bool:
        """
        This method executes the batch wind-vector workflow over the selected time steps (--time-range inclusive, defaulting to the whole dataset). With parallel processing it passes the indices, resolved bounds, the wind plot style (type, subsampling, scale, background, regridding), the wind level, background colormap, title, figure size, dpi, and projection controls to ParallelWindProcessor; otherwise it loops serially with the same options. Failures are counted per step instead of aborting the batch.

        Parameters:
            processor (Any): The 2D processor with the loaded dataset.
            plotter ('MPASWindPlotter'): The plotter used for serial rendering.
            config ('MPASConfig'): The configuration with the wind variables, bounds, plot options, and time selection.

        Returns:
            bool: True if at least one plot was created and no step failed, False otherwise.
        """
        assert processor.dataset is not None
        time_indices = self._resolve_time_indices(processor.dataset, config)
        lon, lat = processor.extract_2d_coordinates_for_variable(config.u_variable)
        bounds = self._resolve_extent(config, lon, lat)

        if getattr(config, "parallel", False):
            self._info("Using parallel processing for batch wind analysis")
            if not time_indices:
                return self._run_batch_loop("wind plots", [], lambda i: None)
            created_files = ParallelWindProcessor.create_batch_wind_plots_parallel(
                processor,
                config.output_dir,
                bounds,
                u_variable=config.u_variable,
                v_variable=config.v_variable,
                formats=config.output_formats or ["png"],
                time_indices=time_indices,
                n_processes=config.workers,
                style=WindBatchStyle(
                    plot_type=config.wind_plot_type,
                    subsample=config.subsample_factor,
                    scale=config.wind_scale,
                    show_background=config.show_background,
                    grid_resolution=getattr(config, "grid_resolution", None),
                    regrid_method=getattr(config, "regrid_method", "linear"),
                ),
                remap_config=RemapConfig(
                    remap_engine=config.remap_engine,
                    remap_method=config.remap_method,
                ),
                render=self._render_options(
                    config, self._wind_level_info(processor, config)
                ),
                **self._output_name_kwargs(
                    ParallelWindProcessor.create_batch_wind_plots_parallel, config
                ),
            )
            return self._finish_parallel(created_files, "wind plots")

        return self._run_batch_loop(
            "wind plots",
            time_indices,
            lambda idx: self._run_wind_single(processor, plotter, config, idx, bounds),
            plotter.close_plot,
        )

    def _run_wind_single(
        self: "MPASUnifiedCLI",
        processor: Any,
        plotter: "MPASWindPlotter",
        config: "MPASConfig",
        time_index: Optional[int] = None,
        bounds: Optional[GeographicBounds] = None,
    ) -> str:
        """
        This method creates and saves one wind-vector plot. The time index defaults to config.time_index and is validated (an out-of-range index raises IndexError instead of being clamped), the bounds default to the extent of the data when none were configured, and the plot honors the wind level, plot type, subsampling (-1 for automatic), scale, background field and its colormap, grid resolution, regrid method, title, and projection controls. In batch mode a user-supplied --output name receives the time tag so steps do not overwrite each other.

        Parameters:
            processor (Any): The processor providing the wind components and coordinates.
            plotter ('MPASWindPlotter'): The plotter that renders and saves the plot.
            config ('MPASConfig'): The configuration with the wind variables and plot options.
            time_index (Optional[int]): The time index to plot (default: config.time_index).
            bounds (Optional[GeographicBounds]): Pre-resolved map bounds (default: resolved from the data).

        Returns:
            str: The output path (without extension) of the saved plot.
        """
        time_index = config.time_index if time_index is None else time_index
        time_index = self._check_time_index(processor.dataset, time_index)

        level_selection, level_label = self._wind_level_info(processor, config)
        if level_selection is None:
            u_data = processor.get_2d_variable_data(config.u_variable, time_index)
            v_data = processor.get_2d_variable_data(config.v_variable, time_index)
        else:
            u_data = processor.get_3d_variable_data(
                config.u_variable, level_selection, time_index
            )
            v_data = processor.get_3d_variable_data(
                config.v_variable, level_selection, time_index
            )
        lon, lat = processor.extract_2d_coordinates_for_variable(
            config.u_variable, u_data
        )
        bounds = bounds or self._resolve_extent(config, lon, lat)

        plotter.create_wind_plot(
            lon,
            lat,
            u_data.values,
            v_data.values,
            bounds,
            style=WindPlotStyle(
                wind_level=level_label,
                plot_type=config.wind_plot_type,
                title=config.title,
                subsample=config.subsample_factor,
                scale=config.wind_scale,
                show_background=config.show_background,
                bg_colormap=config.background_colormap,
                time_stamp=MPASWindPlotter._valid_time_at(
                    processor.dataset, time_index
                ),
                vector_color=config.vector_color,
                vector_alpha=config.vector_alpha,
                **self._projection_kwargs(config),
            ),
            grid_resolution=getattr(config, "grid_resolution", None),
            regrid_method=getattr(config, "regrid_method", "linear"),
            dataset=processor.dataset,
            config=config,
        )

        time_str = MPASDateTimeUtils.get_time_info(
            processor.dataset, time_index, verbose=False
        )

        output_path = self._output_path(
            config,
            _pw.wind_file_stem(
                config.u_variable,
                config.v_variable,
                config.wind_plot_type,
                time_str,
                level_label if level_selection is not None else None,
            ),
            time_str,
        )

        plotter.save_plot(output_path, formats=config.output_formats or ["png"])
        plotter.close_plot()

        print(f"Wind plot saved: {output_path}")
        return output_path

    def _select_wind_processor(
        self: "MPASUnifiedCLI", processor: "MPAS2DProcessor", config: "MPASConfig"
    ) -> Any:
        """
        This method chooses the processor used to read the wind components. When both the u and v variables have a vertical dimension (for example uReconstructZonal and uReconstructMeridional from MPAS output files), the horizontal slice at --wind-level has to be taken from the 3D variable, so an MPAS3DProcessor is loaded and returned and the level string is validated immediately (an invalid --wind-level raises ValueError before any plotting starts). When both are 2D (for example u10 and v10) the original 2D processor is returned unchanged. Mixing one 2D and one 3D variable is an error, because the two components would not describe the same field.

        Parameters:
            processor ('MPAS2DProcessor'): The loaded 2D processor, used to inspect the variable dimensions.
            config ('MPASConfig'): Configuration providing the wind variables, data paths, and wind level.

        Returns:
            Any: The 2D processor for 2D wind variables, or a loaded MPAS3DProcessor for 3D wind variables.
        """
        dataset = processor.dataset
        if dataset is None:
            return processor

        def _is_3d(name: str) -> bool:
            """
            This helper reports whether a dataset variable has a vertical dimension (a dimension whose name contains 'VertLevels'), returning False for missing variables so the later variable lookup reports the usual error.

            Parameters:
                name (str): The variable name.

            Returns:
                bool: True when the variable has a vertical dimension.
            """
            return name in dataset and any(
                "VertLevels" in str(dim) for dim in dataset[name].dims
            )

        u_3d, v_3d = _is_3d(config.u_variable), _is_3d(config.v_variable)
        if u_3d != v_3d:
            raise ValueError(
                f"The wind components must both be 2D or both be 3D: "
                f"{config.u_variable} is {'3D' if u_3d else '2D'} and "
                f"{config.v_variable} is {'3D' if v_3d else '2D'}"
            )
        if not u_3d:
            return processor

        self._parse_wind_level(config.wind_level)
        processor_3d = MPAS3DProcessor(config.grid_file, verbose=config.verbose)
        return self._load_data(
            processor_3d.load_3d_data,
            config,
            [config.u_variable, config.v_variable] + CROSS_SECTION_AUX_VARS,
        )

    def _run_wind_analysis(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method orchestrates the wind-vector analysis: it loads the grid and data, creates the plotter with the configured figure size and dpi, ensures the output directory exists, and runs the batch workflow or a single time step. A batch returns False when any step failed or nothing was created, so the command exits non-zero; a failing single step raises and is reported by run_analysis.

        Parameters:
            config ('MPASConfig'): The configuration with the paths, wind variables, time selection, and plot options.

        Returns:
            bool: True if the analysis completed without failed steps, False otherwise.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        with self.perf_monitor.timer("Wind analysis"):
            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = self._load_data(
                processor.load_2d_data, config, [config.u_variable, config.v_variable]
            )
            processor = self._select_wind_processor(processor, config)

            plotter = MPASWindPlotter(figsize=config.figure_size, dpi=config.dpi)

            os.makedirs(config.output_dir, exist_ok=True)

            if config.batch_mode:
                return self._run_wind_batch(processor, plotter, config) is not False

            self._run_wind_single(processor, plotter, config)

        return True

    def _validate_cross_section_coordinates(
        self: "MPASUnifiedCLI", config: "MPASConfig"
    ) -> None:
        """
        This method validates that the necessary coordinates for defining a vertical cross-section are specified in the configuration object when the analysis type is set to a cross-section analysis. It checks that both the start and end longitude and latitude coordinates are provided and not None. If any of these required coordinates are missing, it raises an assertion error with a message indicating which specific coordinate is missing. This validation ensures that users have provided all necessary information to define the start and end points of the vertical cross-section before attempting to execute the cross-section analysis, preventing runtime errors during processing due to missing parameters.

        Parameters:
            config ('MPASConfig'): The configuration object containing parameters for the analysis, which should include the start and end longitude and latitude coordinates for defining the vertical cross-section when the analysis type is set to a cross-section analysis.

        Returns:
            None
        """
        assert (
            config.start_lon is not None and config.start_lat is not None
        ), "Cross-section start coordinates must be specified"

        assert (
            config.end_lon is not None and config.end_lat is not None
        ), "Cross-section end coordinates must be specified"

    def _extract_cross_section_params(
        self: "MPASUnifiedCLI", config: "MPASConfig"
    ) -> Dict[str, Any]:
        """
        This method extracts the necessary parameters for performing a vertical cross-section analysis from the provided configuration object and organizes them into a dictionary format that can be easily used in the plotting functions. It retrieves the variable name, start and end coordinates, vertical coordinate type, number of points along the cross-section, and maximum height for the plot from the configuration object. The method assumes that these parameters have already been validated to ensure that they are present and valid for cross-section analysis. By centralizing the extraction of these parameters into a single method, it promotes cleaner code and easier maintenance, allowing the main analysis workflow to focus on processing and plotting logic while relying on this helper method to provide the necessary parameters in a consistent format.

        Parameters:
            config ('MPASConfig'): The configuration object containing all parameters for the cross-section analysis, including variable name, start and end coordinates, vertical coordinate type, number of points, and maximum height. These parameters should be validated before calling this method to ensure they are present and valid for cross-section analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted parameters for cross-section analysis, organized in a format that can be easily used in the plotting functions. The dictionary includes keys such as 'var_name', 'start_point', 'end_point', 'vertical_coord', 'num_points', and 'max_height' with corresponding values extracted from the configuration object.
        """
        return {
            "var_name": config.variable,
            "start_point": (config.start_lon, config.start_lat),
            "end_point": (config.end_lon, config.end_lat),
            "vertical_coord": config.vertical_coord,
            "num_points": config.num_points,
            "max_height": config.max_height,
        }

    def _run_batch_cross_sections(
        self: "MPASUnifiedCLI",
        processor: "MPAS3DProcessor",
        plotter: "MPASVerticalCrossSectionPlotter",
        config: "MPASConfig",
        params: Dict[str, Any],
    ) -> bool:
        """
        This method executes the batch vertical cross-section workflow over the selected time steps (--time-range inclusive, defaulting to the whole dataset). With parallel processing it passes the indices together with the colormap, levels, colorbar extension, plot style, maximum height, title, figure size, and dpi to ParallelCrossSectionProcessor; otherwise it loops serially over the steps with the same options. Failures are counted per step instead of aborting the batch.

        Parameters:
            processor ('MPAS3DProcessor'): The 3D processor with the loaded dataset.
            plotter ('MPASVerticalCrossSectionPlotter'): The plotter used for serial rendering.
            config ('MPASConfig'): The configuration with the transect, vertical coordinate, plot options, and time selection.
            params (Dict[str, Any]): Cross-section parameters (variable, start and end points, vertical coordinate, number of points, maximum height).

        Returns:
            bool: True if at least one plot was created and no step failed, False otherwise.
        """
        assert processor.dataset is not None
        time_indices = self._resolve_time_indices(processor.dataset, config)

        if not time_indices:
            return self._run_batch_loop("cross-section plots", [], None)

        if getattr(config, "parallel", False):
            self._info("Running parallel batch cross-section analysis...")
            result = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                mpas_3d_processor=processor,
                output_dir=config.output_dir,
                formats=config.output_formats or ["png"],
                time_indices=time_indices,
                n_processes=config.workers,
                style=CrossSectionBatchStyle(
                    levels=(
                        np.asarray(config.levels, dtype=float)
                        if config.levels
                        else None
                    ),
                    colormap=self._cmap(config),
                    extend=config.extend,
                    plot_type=config.plot_style,
                ),
                render=self._render_options(config),
                **self._output_name_kwargs(
                    ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel,
                    config,
                ),
                **params,
            )
            return self._finish_parallel(result, "cross-section plots")

        self._info("Running serial batch cross-section analysis...")
        return self._run_batch_loop(
            "cross-section plots",
            time_indices,
            lambda idx: self._run_single_cross_section(
                processor, plotter, config, params, idx
            ),
            plotter.close_plot,
        )

    def _run_single_cross_section(
        self: "MPASUnifiedCLI",
        processor: "MPAS3DProcessor",
        plotter: "MPASVerticalCrossSectionPlotter",
        config: "MPASConfig",
        params: Dict[str, Any],
        time_index: Optional[int] = None,
    ) -> str:
        """
        This method creates and saves one vertical cross-section plot. The time index defaults to config.time_index and is validated (an out-of-range index raises IndexError instead of being clamped), and the plot honors the colormap, levels, colorbar extension, plot style, title, and the transect parameters. The default file name carries a '_maxh_<value>km' suffix when --max-height is given (identical to the parallel path), so runs with different maximum heights never overwrite each other. In batch mode a user-supplied --output name receives the time tag so steps do not overwrite each other.

        Parameters:
            processor ('MPAS3DProcessor'): The 3D processor providing the data.
            plotter ('MPASVerticalCrossSectionPlotter'): The plotter that renders and saves the plot.
            config ('MPASConfig'): The configuration with the plot options.
            params (Dict[str, Any]): Cross-section parameters (variable, start and end points, vertical coordinate, number of points, maximum height).
            time_index (Optional[int]): The time index to plot (default: config.time_index).

        Returns:
            str: The output path (without extension) of the saved plot.
        """
        time_index = config.time_index if time_index is None else time_index
        time_index = self._check_time_index(processor.dataset, time_index)

        time_str = MPASDateTimeUtils.get_time_info(
            processor.dataset, time_index, verbose=False
        )

        _, _ = plotter.create_vertical_cross_section(
            mpas_3d_processor=processor,
            time_index=time_index,
            title=config.title
            or f"MPAS Vertical Cross-Section | Variable: {config.variable} | Valid: {time_str}",
            style=CrossSectionStyle(
                colormap=self._cmap(config),
                levels=(
                    np.asarray(config.levels, dtype=float) if config.levels else None
                ),
                extend=config.extend,
                plot_type=config.plot_style,
                **_accepted_kwargs(
                    CrossSectionStyle,
                    clim_min=getattr(config, "clim_min", None),
                    clim_max=getattr(config, "clim_max", None),
                ),
            ),
            **params,
        )

        output_path = self._output_path(
            config,
            cross_section_file_stem(
                config.variable,
                config.vertical_coord,
                time_str,
                params.get("max_height"),
                **_accepted_kwargs(
                    cross_section_file_stem,
                    start_point=params.get("start_point"),
                    end_point=params.get("end_point"),
                ),
            ),
            time_str,
        )

        plotter.save_plot(output_path, formats=config.output_formats or ["png"])
        plotter.close_plot()

        print(f"Cross-section plot saved: {output_path}")
        return output_path

    def _log_created_files(
        self: "MPASUnifiedCLI",
        created_files: Optional[List[str]],
        file_type: str = "plots",
    ) -> None:
        """
        This method logs a message indicating the number of files created during the analysis process, providing context about the type of files generated. It checks if the logger is configured and if the list of created files is not None or empty. If both conditions are met, it logs an informational message that includes the count of created files and the specified file type (e.g., "Created 5 cross-section plots"). This method helps provide feedback to users about the results of the analysis, confirming how many visualizations or output files were successfully generated based on their configuration. If there are no created files or if the logger is not available, this method will not log anything.

        Parameters:
            created_files (Optional[List[str]]): A list of file paths for the created files during the analysis process. This can be None or empty if no files were created or if an error occurred during processing.
            file_type (str): A string describing the type of files created (e.g., "cross-section plots", "Skew-T diagrams") to provide context in the log message. Default is "plots".

        Returns:
            None
        """
        if created_files and is_root_process():
            print(f"Created {len(created_files)} {file_type}")

    def _run_cross_analysis(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method orchestrates the vertical cross-section analysis: it loads the grid and 3D data, creates the plotter with the configured figure size and dpi, validates the transect coordinates, and runs the batch workflow or a single time step. A batch returns False when any step failed or nothing was created, so the command exits non-zero; a failing single step raises and is reported by run_analysis.

        Parameters:
            config ('MPASConfig'): The configuration with the paths, variable, transect, time selection, and plot options.

        Returns:
            bool: True if the analysis completed without failed steps, False otherwise.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        with self.perf_monitor.timer("Cross-section analysis"):
            processor = MPAS3DProcessor(config.grid_file, verbose=config.verbose)
            processor = self._load_data(
                processor.load_3d_data,
                config,
                [config.variable] + CROSS_SECTION_AUX_VARS,
            )

            plotter = MPASVerticalCrossSectionPlotter(
                figsize=config.figure_size, dpi=config.dpi
            )

            os.makedirs(config.output_dir, exist_ok=True)

            self._validate_cross_section_coordinates(config)
            params = self._extract_cross_section_params(config)

            if config.batch_mode:
                return (
                    self._run_batch_cross_sections(processor, plotter, config, params)
                    is not False
                )

            self._run_single_cross_section(processor, plotter, config, params)

        return True

    def _compute_sounding_indices(
        self: "MPASUnifiedCLI",
        show_indices: bool,
        sounding_diag: "SoundingDiagnostics",
        profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        This method computes thermodynamic indices for a sounding profile if the user has requested to show indices in the Skew-T diagram. It checks the `show_indices` flag, and if it is False, it returns None, indicating that no indices should be computed or displayed. If `show_indices` is True, it calls the `compute_thermodynamic_indices` method of the `SoundingDiagnostics` instance, passing in the necessary profile data such as pressure, temperature, dewpoint, and optionally wind components and height if they are available in the profile. The computed indices are returned as a dictionary that can be used in the plotting function to display relevant thermodynamic information on the Skew-T diagram. This method centralizes the logic for determining whether to compute indices and how to compute them based on the provided sounding profile data.

        Parameters:
            show_indices (bool): A flag indicating whether the user has requested to compute and display thermodynamic indices on the Skew-T diagram. If False, this method will return None and no indices will be computed or displayed.
            sounding_diag ('SoundingDiagnostics'): An instance of the SoundingDiagnostics class that provides the method for computing thermodynamic indices based on the sounding profile data.
            profile (Dict[str, Any]): A dictionary containing the sounding profile data, including pressure, temperature, dewpoint, and optionally wind components and height. This data will be used to compute the thermodynamic indices if `show_indices` is True.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the computed thermodynamic indices if `show_indices` is True, or None if `show_indices` is False and no indices should be computed or displayed. The dictionary of indices can include values such as CAPE, CIN, Lifted Index, K-Index, etc., depending on the implementation of the `compute_thermodynamic_indices` method in the SoundingDiagnostics class.
        """
        if not show_indices:
            return None
        return sounding_diag.compute_thermodynamic_indices(
            profile["pressure"],
            profile["temperature"],
            profile["dewpoint"],
            u_wind_kt=profile.get("u_wind"),
            v_wind_kt=profile.get("v_wind"),
            height_m=profile.get("height"),
            surface_height_m=profile.get("surface_height"),
        )

    def _build_skewt_tags(
        self: "MPASUnifiedCLI",
        profile: Dict[str, Any],
    ) -> tuple:
        """
        This method constructs longitude and latitude tags for the Skew-T diagram title based on the station longitude and latitude from the sounding profile. It formats the longitude and latitude values to two decimal places and appends the appropriate directional indicators (E/W for longitude and N/S for latitude) based on the sign of the coordinates. The resulting tags are returned as a tuple of strings that can be used in the plot title to indicate the location of the sounding profile being visualized.

        Parameters:
            profile (Dict[str, Any]): A dictionary containing the sounding profile data, including station longitude and latitude.

        Returns:
            tuple: A tuple containing the longitude and latitude tags as strings.
        """
        stn_lon = profile["station_lon"]
        stn_lat = profile["station_lat"]
        lon_tag = f"{abs(stn_lon):.2f}{'W' if stn_lon < 0 else 'E'}"
        lat_tag = f"{abs(stn_lat):.2f}{'S' if stn_lat < 0 else 'N'}"
        return lon_tag, lat_tag

    def _run_sounding_batch(
        self: "MPASUnifiedCLI",
        processor: "MPAS3DProcessor",
        plotter: "MPASSkewTPlotter",
        sounding_diag: "SoundingDiagnostics",
        config: "MPASConfig",
        sounding_lon: float,
        sounding_lat: float,
        show_indices: bool,
        show_parcel: bool,
    ) -> bool:
        """
        This method creates Skew-T diagrams for the selected time steps (--time-range inclusive, defaulting to the whole dataset; a range outside the data raises IndexError). Each step is isolated so that a failing profile extraction or plot is logged with its time index and counted as a failure while the remaining steps still run. Diagrams are written in every requested output format (--formats), and a user-supplied --output name is used as a prefix with the time tag appended. The batch succeeds only when no step failed and at least one diagram was created.

        Parameters:
            processor ('MPAS3DProcessor'): The 3D processor with the loaded dataset.
            plotter ('MPASSkewTPlotter'): The Skew-T plotter.
            sounding_diag ('SoundingDiagnostics'): Diagnostics used to extract profiles and compute indices.
            config ('MPASConfig'): The configuration with the output settings, title, and time selection.
            sounding_lon (float): Sounding longitude in degrees.
            sounding_lat (float): Sounding latitude in degrees.
            show_indices (bool): Whether to compute and show thermodynamic indices.
            show_parcel (bool): Whether to plot the lifted parcel profile.

        Returns:
            bool: True if at least one diagram was created and no step failed, False otherwise.
        """
        assert processor.dataset is not None
        time_indices = self._resolve_time_indices(processor.dataset, config)

        return self._run_batch_loop(
            "Skew-T diagrams",
            time_indices,
            lambda idx: self._create_skewt(
                processor,
                plotter,
                sounding_diag,
                config,
                sounding_lon,
                sounding_lat,
                show_indices,
                show_parcel,
                idx,
            ),
            plotter.close_plot,
        )

    def _run_sounding_single(
        self: "MPASUnifiedCLI",
        processor: "MPAS3DProcessor",
        plotter: "MPASSkewTPlotter",
        sounding_diag: "SoundingDiagnostics",
        config: "MPASConfig",
        sounding_lon: float,
        sounding_lat: float,
        show_indices: bool,
        show_parcel: bool,
    ) -> None:
        """
        This method creates a single Skew-T diagram for config.time_index. An out-of-range time index raises IndexError rather than being clamped, and the diagram is saved in every requested output format under the user-supplied --output name or a descriptive default name.

        Parameters:
            processor ('MPAS3DProcessor'): The 3D processor with the loaded dataset.
            plotter ('MPASSkewTPlotter'): The Skew-T plotter.
            sounding_diag ('SoundingDiagnostics'): Diagnostics used to extract the profile and compute indices.
            config ('MPASConfig'): The configuration with the output settings, title, and time index.
            sounding_lon (float): Sounding longitude in degrees.
            sounding_lat (float): Sounding latitude in degrees.
            show_indices (bool): Whether to compute and show thermodynamic indices.
            show_parcel (bool): Whether to plot the lifted parcel profile.

        Returns:
            None
        """
        self._create_skewt(
            processor,
            plotter,
            sounding_diag,
            config,
            sounding_lon,
            sounding_lat,
            show_indices,
            show_parcel,
            config.time_index,
        )

    def _run_sounding_analysis(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method orchestrates the Skew-T sounding analysis: it loads the 3D data, creates the diagnostics and plotter (taller figure when indices are shown), ensures the output directory exists, and runs the batch workflow or a single time step. A batch returns False when any step failed or nothing was created, so the command exits non-zero; a failing single step raises and is reported by run_analysis.

        Parameters:
            config ('MPASConfig'): The configuration with the paths, sounding location, time selection, and output options.

        Returns:
            bool: True if the analysis completed without failed steps, False otherwise.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        with self.perf_monitor.timer("Sounding analysis"):
            processor = MPAS3DProcessor(config.grid_file, verbose=config.verbose)
            processor = self._load_data(processor.load_3d_data, config)

            sounding_diag = SoundingDiagnostics(verbose=config.verbose)

            show_indices = getattr(config, "show_indices", False)
            adjusted_figsize = config.figure_size

            if show_indices:
                adjusted_figsize = (adjusted_figsize[0], adjusted_figsize[1] + 4)

            plotter = MPASSkewTPlotter(
                figsize=adjusted_figsize,
                dpi=config.dpi,
                verbose=config.verbose,
            )

            os.makedirs(config.output_dir, exist_ok=True)

            sounding_lon = getattr(config, "sounding_lon", 0.0)
            sounding_lat = getattr(config, "sounding_lat", 0.0)
            show_parcel = getattr(config, "show_parcel", False)

            args = (
                processor,
                plotter,
                sounding_diag,
                config,
                sounding_lon,
                sounding_lat,
                show_indices,
                show_parcel,
            )

            if config.batch_mode:
                return self._run_sounding_batch(*args) is not False

            self._run_sounding_single(*args)

        return True

    _OVERLAY_LINE_COLORS = ("black", "red", "blue", "green", "purple", "darkorange")

    def _multi_level_overlay_plan(
        self: "MPASUnifiedCLI", config: "MPASConfig"
    ) -> Tuple[str, List[Tuple[str, bool]], bool]:
        """
        This method builds the multi_level part of the overlay plan from --variables: the first variable is the base field, the configured u/v variables become wind vectors when both are listed, and every other variable becomes a contour layer.

        Parameters:
            config (MPASConfig): Configuration providing the variables and the u/v variable names.

        Returns:
            Tuple[str, List[Tuple[str, bool]], bool]: The base variable, the list of (variable, filled) layers, and whether wind vectors are drawn.

        Raises:
            ValueError: If fewer than two variables are given or only one of the wind components is listed.
        """
        names = list(config.variables or [])
        if len(names) < 2:
            raise ValueError(
                "multi_level overlays need --variables with at least two "
                "comma-separated variables, e.g. --variables t2m,mslp,u10,v10"
            )
        base = names[0]
        uv = (config.u_variable, config.v_variable)
        rest = names[1:]
        present = [name for name in uv if name in rest]
        if len(present) == 1:
            raise ValueError(
                f"--variables contains {present[0]} but not its wind partner: "
                f"list both {uv[0]} and {uv[1]} to draw wind vectors"
            )
        wind = len(present) == 2
        layers = [(name, False) for name in rest if name not in uv]
        return base, layers, wind

    def _overlay_spec(self: "MPASUnifiedCLI", config: "MPASConfig") -> Dict[str, Any]:
        """
        This method turns the overlay options into a rendering plan. Every overlay has a filled base field (precipitation for precip_wind, otherwise a scalar surface variable), zero or more scalar layers drawn over it, and optionally 10-m style wind vectors. precip_wind draws the accumulated precipitation with wind vectors; temp_pressure draws a scalar (default t2m) with the pressure variable (default mslp) as contour lines; wind_temp draws a scalar (default t2m) with wind vectors; multi_level takes --variables, uses the first as the base field, treats the configured u/v variables as wind vectors and every other variable as a contour layer; custom draws --primary-variable with --secondary-variable (a transparent filled layer, plus contour lines when --contour-overlay is given). A --secondary-variable given for precip_wind, temp_pressure or wind_temp is added as an extra contour layer.

        Parameters:
            config (MPASConfig): Configuration providing the overlay type and variables.

        Returns:
            Dict[str, Any]: The plan with the keys 'base', 'precip', 'layers' (list of (variable, filled) pairs), 'wind', and 'names' (the variable names used in file names).

        Raises:
            ValueError: If the overlay type or the combination of variables is invalid.
        """
        kind = config.overlay_type
        primary, secondary = config.primary_variable, config.secondary_variable
        layers: List[Tuple[str, bool]] = []
        wind = False

        if kind == "precip_wind":
            base, wind = primary or "rainnc", True
            if secondary:
                layers.append((secondary, False))
        elif kind == "temp_pressure":
            base = primary or "t2m"
            layers.append((secondary or config.pressure_variable, False))
        elif kind == "wind_temp":
            base, wind = primary or "t2m", True
            if secondary:
                layers.append((secondary, False))
        elif kind == "multi_level":
            base, layers, wind = self._multi_level_overlay_plan(config)
        elif kind == "custom":
            if not primary or not secondary:
                raise ValueError(
                    "custom overlays need both --primary-variable and "
                    "--secondary-variable"
                )
            base = primary
            layers.append((secondary, True))
        else:
            raise ValueError(f"Unknown overlay type: {kind}")

        names = [base] + [name for name, _ in layers]
        if wind:
            names += [config.u_variable, config.v_variable]
        return {
            "base": base,
            "precip": kind == "precip_wind",
            "layers": layers,
            "wind": wind,
            "names": names,
        }

    def _overlay_base_map(
        self: "MPASUnifiedCLI",
        processor: "MPAS2DProcessor",
        config: "MPASConfig",
        spec: Dict[str, Any],
        time_index: int,
        bounds: GeographicBounds,
        time_str: str,
        title: str,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Any]:
        """
        This method draws the filled base field of an overlay (accumulated precipitation with the precipitation plotter, or a scalar surface variable with the surface plotter) and returns the plotter, the cell coordinates, and the axes the overlays are added to.

        Parameters:
            processor (MPAS2DProcessor): The loaded 2D processor.
            config (MPASConfig): Configuration providing the style options.
            spec (Dict[str, Any]): The plan from _overlay_spec.
            time_index (int): The time index to draw.
            bounds (GeographicBounds): The map extent.
            time_str (str): The valid-time tag for the default title.
            title (str): The plot title.

        Returns:
            Tuple[Any, np.ndarray, np.ndarray, Any]: The plotter, the longitude and latitude arrays, and the map axes.
        """
        dataset = processor.dataset
        colormap = config.primary_colormap or self._cmap(config)
        base = spec["base"]

        if spec["precip"]:
            self._check_lookback(config, time_index, dataset)
            precip = PrecipitationDiagnostics(verbose=config.verbose)
            data_array = precip.compute_precipitation_difference(
                dataset,
                time_index,
                base,
                config.accumulation_period,
                data_type=getattr(processor, "data_type", "UXarray"),
            )
            lon, lat = processor.extract_2d_coordinates_for_variable(base)
            plotter: Any = MPASPrecipitationPlotter(
                figsize=config.figure_size, dpi=config.dpi
            )
            plotter.create_precipitation_map(
                lon,
                lat,
                data_array.values,
                bounds,
                accum_period=config.accumulation_period,
                time_end=self._valid_datetime(dataset, time_index),
                data_array=data_array,
                var_name=base,
                dataset=dataset,
                config=config,
                style=PrecipitationRenderStyle(
                    title=title,
                    plot_type="contourf",
                    colormap=colormap,
                    levels=config.levels,
                    clim_min=config.clim_min,
                    clim_max=config.clim_max,
                    grid_resolution=config.grid_resolution,
                    display_units=config.precip_units,
                    threshold=config.precip_threshold,
                    **self._projection_kwargs(config),
                ),
            )
            return plotter, np.asarray(lon), np.asarray(lat), plotter.ax

        var_data = processor.get_2d_variable_data(base, time_index)
        lon, lat = processor.extract_2d_coordinates_for_variable(base, var_data)
        plotter = MPASSurfacePlotter(figsize=config.figure_size, dpi=config.dpi)
        plotter.create_surface_map(
            lon,
            lat,
            var_data.values,
            base,
            bounds,
            style=SurfaceMapStyle(
                title=title,
                plot_type="contourf",
                colormap=colormap,
                levels=config.levels,
                num_levels=config.contour_levels,
                clim_min=config.clim_min,
                clim_max=config.clim_max,
                grid_resolution=config.grid_resolution,
                **_accepted_kwargs(
                    SurfaceMapStyle, interpolation=config.interpolation_method
                ),
                **self._projection_kwargs(config),
            ),
            data_array=var_data,
            dataset=dataset,
            config=config,
        )
        return plotter, np.asarray(lon), np.asarray(lat), plotter.ax

    def _overlay_layers(
        self: "MPASUnifiedCLI",
        processor: "MPAS2DProcessor",
        config: "MPASConfig",
        spec: Dict[str, Any],
        time_index: int,
        bounds: GeographicBounds,
        axes: Any,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> None:
        """
        This method draws the scalar contour layers and the wind vectors of an overlay on the base map. Scalar layers use MPASSurfacePlotter.add_surface_overlay (black, red, blue... contour lines with labels, or a transparent filled layer with --secondary-colormap and --transparency for the custom overlay; --contour-overlay adds the lines on top of a filled layer) and wind vectors use MPASWindPlotter.add_wind_overlay with the configured wind plot type, subsampling and color.

        Parameters:
            processor (MPAS2DProcessor): The loaded 2D processor.
            config (MPASConfig): Configuration providing the style options.
            spec (Dict[str, Any]): The plan from _overlay_spec.
            time_index (int): The time index to draw.
            bounds (GeographicBounds): The map extent used to filter the overlay data.
            axes (Any): The map axes of the base plot.
            lon (np.ndarray): Cell longitudes in degrees.
            lat (np.ndarray): Cell latitudes in degrees.

        Returns:
            None
        """
        extent: Dict[str, Any] = {
            "config": config,
            "lon_min": bounds.lon_min,
            "lon_max": bounds.lon_max,
            "lat_min": bounds.lat_min,
            "lat_max": bounds.lat_max,
        }
        helper = MPASSurfacePlotter(figsize=config.figure_size, dpi=config.dpi)
        line_index = 0

        for name, filled in spec["layers"]:
            layer = processor.get_2d_variable_data(name, time_index)
            values = np.asarray(layer.values).flatten()
            common = {
                "data": values,
                "var_name": name,
                "grid_resolution": config.grid_resolution,
            }
            # Pass the file units so the layer is drawn in display units (e.g. q2 in g/kg)
            if getattr(layer, "attrs", {}).get("units"):
                common["original_units"] = layer.attrs["units"]
            if filled:
                helper.add_surface_overlay(
                    axes,
                    lon,
                    lat,
                    {
                        **common,
                        "plot_type": "contourf",
                        "colormap": config.secondary_colormap or "Blues",
                        "alpha": config.transparency,
                    },
                    **extent,
                )
            if not filled or config.contour_overlay:
                color = self._OVERLAY_LINE_COLORS[
                    line_index % len(self._OVERLAY_LINE_COLORS)
                ]
                if config.secondary_colormap and not filled:
                    color = matplotlib.colormaps[config.secondary_colormap](0.85)
                helper.add_surface_overlay(
                    axes,
                    lon,
                    lat,
                    {
                        **common,
                        "plot_type": "contour",
                        "colors": color,
                        "linewidth": 1.2,
                        "add_labels": True,
                    },
                    **extent,
                )
                line_index += 1

        if spec["wind"]:
            u_values = np.asarray(
                processor.get_2d_variable_data(config.u_variable, time_index).values
            ).flatten()
            v_values = np.asarray(
                processor.get_2d_variable_data(config.v_variable, time_index).values
            ).flatten()
            MPASWindPlotter(
                figsize=config.figure_size, dpi=config.dpi
            ).add_wind_overlay(
                axes,
                lon,
                lat,
                {
                    "u_data": u_values,
                    "v_data": v_values,
                    "plot_type": config.wind_plot_type,
                    "subsample": config.subsample_factor,
                    "scale": config.wind_scale,
                    "colors": config.vector_color,
                    "grid_resolution": config.grid_resolution,
                },
                **extent,
            )

    def _run_overlay_single(
        self: "MPASUnifiedCLI",
        processor: "MPAS2DProcessor",
        config: "MPASConfig",
        spec: Dict[str, Any],
        time_index: Optional[int] = None,
        bounds: Optional[GeographicBounds] = None,
    ) -> str:
        """
        This method creates and saves one overlay plot: the filled base field, the scalar contour layers, and the wind vectors, saved as 'mpas_overlay_<type>_<variables>_valid_<time>' (or the --output name) in every requested format.

        Parameters:
            processor (MPAS2DProcessor): The loaded 2D processor.
            config (MPASConfig): Configuration providing the overlay options.
            spec (Dict[str, Any]): The plan from _overlay_spec.
            time_index (Optional[int]): The time index to plot (default: config.time_index).
            bounds (Optional[GeographicBounds]): Pre-resolved map bounds (default: resolved from the data).

        Returns:
            str: The output path (without extension) of the saved plot.
        """
        dataset = processor.dataset
        time_index = config.time_index if time_index is None else time_index
        time_index = self._check_time_index(dataset, time_index)
        time_str = MPASDateTimeUtils.get_time_info(dataset, time_index, verbose=False)

        if bounds is None:
            lon0, lat0 = processor.extract_2d_coordinates_for_variable(spec["base"])
            bounds = self._resolve_extent(config, lon0, lat0)

        label = " + ".join(spec["names"])
        title = config.title or f"MPAS Overlay | {label} | Valid: {time_str}"
        plotter, lon, lat, axes = self._overlay_base_map(
            processor, config, spec, time_index, bounds, time_str, title
        )
        try:
            self._overlay_layers(
                processor, config, spec, time_index, bounds, axes, lon, lat
            )
            output_path = self._output_path(
                config,
                f"mpas_overlay_{sanitize_filename_component(str(config.overlay_type))}_"
                + "_".join(sanitize_filename_component(n) for n in spec["names"])
                + f"_valid_{time_str}",
                time_str,
            )
            plotter.save_plot(output_path, formats=config.output_formats or ["png"])
        finally:
            plotter.close_plot()

        print(f"Overlay plot saved: {output_path}")
        return output_path

    def _run_overlay_analysis(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method orchestrates the overlay analysis for every overlay type (precip_wind, temp_pressure, wind_temp, multi_level, custom). It validates the overlay options, loads the grid and data, resolves the extent and time steps (a single --time-index or a batch from --batch-all/--time-range, applying the accumulation look-back for precipitation overlays), and renders each time step with the public surface, precipitation, and wind plotter APIs. Overlays are always rendered serially; --parallel produces an informational message and a serial run. Errors are reported with a clear message and a False return so the process exits with status 1.

        Parameters:
            config (MPASConfig): The configuration with the paths, overlay type and variables, extent, time selection, and style options.

        Returns:
            bool: True if every overlay was created, False if the options are invalid or any step failed.
        """
        assert self.perf_monitor is not None, PERFORMANCE_MONITOR_MSG

        try:
            spec = self._overlay_spec(config)
        except ValueError as exc:
            self._log_error(f"Invalid overlay options: {exc}")
            return False

        with self.perf_monitor.timer("Overlay analysis"):
            self._info(f"Running {config.overlay_type} overlay analysis...")
            if getattr(config, "parallel", False):
                self._info(
                    "Parallel processing is not available for overlay plots; "
                    "running serially"
                )

            processor = MPAS2DProcessor(config.grid_file, verbose=config.verbose)
            processor = self._load_data(processor.load_2d_data, config)
            dataset = processor.dataset
            os.makedirs(config.output_dir, exist_ok=True)

            lon, lat = processor.extract_2d_coordinates_for_variable(spec["base"])
            bounds = self._resolve_extent(config, lon, lat)

            if not config.batch_mode:
                self._run_overlay_single(processor, config, spec, None, bounds)
                return True

            first_index = 0
            if spec["precip"]:
                first_index, reason = self._first_lookback_index(dataset, config)
                if first_index >= self._n_times(dataset) and reason:
                    raise InsufficientLookbackError(reason)
            time_indices = self._resolve_time_indices(dataset, config, first_index)

            return self._run_batch_loop(
                "overlay plots",
                time_indices,
                lambda idx: self._run_overlay_single(
                    processor, config, spec, idx, bounds
                ),
            )

    def _parse_args_with_fallback(
        self: "MPASUnifiedCLI", parser: argparse.ArgumentParser
    ) -> argparse.Namespace:
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

    def _parse_args_with_reordering(
        self: "MPASUnifiedCLI", parser: argparse.ArgumentParser
    ) -> argparse.Namespace:
        """
        This method implements a custom argument parsing strategy that manually reorders command-line arguments to ensure that global options are processed correctly even when intermixed with subcommands. It identifies global options that can either have values (e.g., `--config`, `--log-file`) or be flags without values (e.g., `--verbose`, `--quiet`), and separates them from the rest of the arguments. The method then attempts to parse the reordered arguments using the provided parser. If parsing fails, it falls back to parsing the original arguments without reordering. This approach allows for greater flexibility in how users can input command-line arguments while maintaining compatibility across different Python versions and argument configurations.

        Parameters:
            parser (argparse.ArgumentParser): The argument parser instance configured with the expected command-line arguments and subcommands for the CLI.

        Returns:
            argparse.Namespace: The parsed command-line arguments after reordering global options and attempting to parse them with the provided parser. If parsing fails, it returns the result of parsing the original arguments without reordering.
        """
        cli_args = sys.argv[1:]
        globals_with_value = {"--config", "--log-file", "--base-dir"}
        globals_no_value = {"--verbose", "-v", "--quiet", "-q", "--version"}

        global_args = []
        subcommand_args = []
        i = 0

        while i < len(cli_args):
            arg = cli_args[i]
            if arg in globals_no_value or any(
                arg.startswith(g + "=") for g in globals_with_value
            ):
                global_args.append(arg)
                i += 1
            elif arg in globals_with_value:
                arg_value = cli_args[i + 1] if i + 1 < len(cli_args) else ""
                global_args.extend([arg, arg_value])
                i += 2
            else:
                subcommand_args.append(arg)
                i += 1

        try:
            return parser.parse_args(global_args + subcommand_args)
        except Exception:
            return parser.parse_args()

    def _load_and_merge_config(
        self: "MPASUnifiedCLI", args: argparse.Namespace
    ) -> "MPASConfig":
        """
        This method builds the final configuration with the precedence explicit command-line flags > YAML configuration > defaults. When --config is given the YAML file is read (confined to --base-dir, or the working directory when --base-dir is not given) into a plain dictionary and only the flags the user actually typed are layered on top, so argparse defaults can never overwrite file values, including booleans such as batch_mode and parallel and values such as figure_size and output_formats. Required arguments are checked on the merged result so the YAML can supply them, a time range or YAML time_start/time_end implies batch mode unless an explicit --no-batch-all is given (the YAML range is then ignored with a log message), the output directory defaults to ./output, and the MPASConfig constructor validates the result, raising ValueError with a descriptive message for invalid values.

        Parameters:
            args (argparse.Namespace): The parsed command-line arguments.

        Returns:
            MPASConfig: The merged and validated configuration.
        """
        merged: Dict[str, Any] = {}

        if getattr(args, "config", None):
            merged.update(
                MPASConfig.load_dict_from_file(
                    str((Path.cwd() / Path(args.config).expanduser())),
                    base_dir=getattr(args, "base_dir", None),
                )
            )

        merged.update(self._args_to_dict(args))

        has_time_range = (
            merged.get("time_start") is not None or merged.get("time_end") is not None
        )

        if (
            has_time_range
            and getattr(args, "batch_all", None) is False
            and not getattr(args, "time_range", None)
        ):
            # An explicit --no-batch-all wins over a YAML time range: run the single step at time_index
            self._config_notes.append(
                "--no-batch-all given: ignoring the YAML time range "
                f"(time_start={merged.get('time_start')}, "
                f"time_end={merged.get('time_end')}) and processing a single time step"
            )
            merged.pop("time_start", None)
            merged.pop("time_end", None)
            merged["batch_mode"] = False
        elif has_time_range:
            merged["batch_mode"] = True

        if merged.get("verbose") and merged.get("quiet"):
            raise ValueError("--verbose and --quiet are mutually exclusive")

        self._check_required_arguments(merged)

        if not merged.get("output_dir"):
            merged["output_dir"] = DEFAULT_OUTPUT_DIR

        return MPASConfig.from_dict(merged)

    def _print_verbose_output(self: "MPASUnifiedCLI", config: "MPASConfig") -> None:
        """
        This method prints system information and a configuration summary when verbose output is enabled in the configuration and quiet mode is not active. It is a no-op otherwise, so --quiet (and the default non-verbose mode) never prints the System Information and Configuration Summary blocks.

        Parameters:
            config ('MPASConfig'): The configuration object containing the verbose and quiet flags.

        Returns:
            None
        """
        if getattr(config, "verbose", False) and not getattr(config, "quiet", False):
            self._print_system_info()
            self._print_config_summary()

    def _handle_main_exception(self: "MPASUnifiedCLI", error: Exception) -> int:
        """
        This method handles any unexpected exceptions that occur during the execution of the main workflow in the `main` method. It logs the error message and stack trace using the configured logger if available, or prints them to standard output if no logger is configured. The method then returns an exit code of 1 to indicate that an error occurred during processing. This centralized exception handling ensures that users receive informative feedback about what went wrong while also allowing the CLI to exit gracefully with an appropriate status code.

        Parameters:
            error (Exception): The exception object representing the unexpected error that occurred during the execution of the main workflow. This can be any type of exception that was not anticipated or handled by specific error handling logic within the analysis methods.

        Returns:
            int: Exit code (1) indicating an error condition.
        """
        import traceback

        emit = self.logger if self.logger else get_logger(__name__)
        emit.error("Unexpected error: %s", error)
        emit.debug("Full traceback:\n%s", traceback.format_exc())
        return 1

    def main(self: "MPASUnifiedCLI") -> int:
        """
        This method is the main execution flow of the CLI. It parses the arguments (reordering global flags if needed), builds the configuration with the precedence explicit flags > YAML > defaults, sets up logging, validates the configuration, prints verbose output when requested, and runs the analysis. Configuration errors (an unreadable or invalid YAML file, invalid values, conflicting flags) are printed as clear validation messages through the logger and return 1. The return value is the process exit status: 0 only when the analysis succeeded with no failed steps, 1 on validation errors or any failure, and 130 on keyboard interrupt; entry points pass it to sys.exit.

        Parameters:
            None

        Returns:
            int: The exit status (0 success, 1 failure, 130 interrupted).
        """
        try:
            parser = self.create_main_parser()
            args = self._parse_args_with_fallback(parser)

            try:
                self.config = self._load_and_merge_config(args)
            except (ValueError, OSError, yaml.YAMLError) as exc:
                self._report_early_error(args, str(exc))
                return 1

            log_file = getattr(args, "log_file", None)
            self.setup_logging(self.config, log_file)

            for note in self._config_notes:
                self._info(note)
            self._config_notes.clear()

            if not self.validate_config(self.config):
                return 1

            if not self._prepare_execution(self.config):
                return 0

            self._print_verbose_output(self.config)

            success = self.run_analysis(self.config)

            return 0 if success else 1

        except KeyboardInterrupt:
            emit = self.logger if self.logger else get_logger(__name__)
            emit.warning("Analysis interrupted by user")
            return 130
        except Exception as e:
            return self._handle_main_exception(e)

    def _prepare_execution(self: "MPASUnifiedCLI", config: "MPASConfig") -> bool:
        """
        This method decides how this process takes part in the run before any data are loaded. Under an MPI launcher, a run that does not divide its work across ranks is executed by rank 0 alone: the other ranks are told to exit (they would only repeat rank 0's work and race to write the same files) and rank 0 warns once. Without a launcher, a parallel batch runs in a worker pool, so on macOS the pool's forkserver is started here, letting it import the plotting stack while this process loads the data.

        Parameters:
            config (MPASConfig): The validated configuration.

        Returns:
            bool: False if this process should exit without running the analysis, True otherwise.
        """
        shares_work = self._mpi_ranks_share_work(config)
        if not mpi_launcher_detected():
            if shares_work:
                start_worker_server()
            return True
        if shares_work:
            return True
        if not is_root_process():
            return False
        (self.logger or get_logger(__name__)).warning(
            "Started under an MPI launcher, but this run does not distribute work across ranks "
            "(add --parallel to a --time-range batch of precipitation, surface, wind or cross "
            "analysis to use them); running on rank 0 only"
        )
        return True

    @staticmethod
    def _mpi_ranks_share_work(config: "MPASConfig") -> bool:
        """
        This helper reports whether the configured run divides its work across MPI ranks. Only parallel batch runs of the precipitation, surface, wind and cross-section analyses do; every other run (single time steps, serial batches, soundings and overlays) is executed in full by each process that starts it, so under an MPI launcher the extra ranks would only repeat rank 0's work and race to write the same output files.

        Parameters:
            config (MPASConfig): The validated configuration.

        Returns:
            bool: True if the run distributes its time steps across MPI ranks.
        """
        distributed_types = (
            "precipitation",
            "precip",
            "rain",
            "surface",
            "surf",
            "2d",
            "wind",
            "vector",
            "winds",
            "cross",
            "xsec",
            "3d",
            "vertical",
        )
        return bool(
            getattr(config, "parallel", False)
            and getattr(config, "batch_mode", False)
            and getattr(config, "analysis_type", None) in distributed_types
        )

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
