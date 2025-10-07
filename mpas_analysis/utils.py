#!/usr/bin/env python3

"""
MPAS Utilities Module

This module provides utility functions and helper classes for MPAS data analysis,
including file handling, configuration management, logging, and common operations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Last Modified: 2025-10-06

Features:
    - Configuration file handling
    - Logging utilities
    - File and directory operations
    - Data validation helpers
    - Command-line argument parsing
    - Performance monitoring
    - Error handling utilities
"""

import os
import sys
import json
import yaml
import logging
import argparse
import warnings
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np
import pandas as pd


@dataclass
class MPASConfig:
    """
    Configuration class for MPAS analysis parameters.
    
    Centralized configuration management for all MPAS analysis operations including
    data processing, visualization, and output generation. Supports all plot types
    (precipitation, surface variables, wind vectors) with comprehensive parameter sets.
    
    Attributes:
        Data Parameters:
            grid_file (str): Path to MPAS grid file
            data_dir (str): Directory containing MPAS diagnostic files
            output_dir (str): Output directory for plots and results
            
        Spatial Parameters:
            lat_min, lat_max (float): Latitude bounds for analysis extent
            lon_min, lon_max (float): Longitude bounds for analysis extent
            
        Processing Parameters:
            variable (str): Primary variable to analyze
            time_index (int): Time step index for processing
            data_type (str): Data processing backend ('uxarray' or 'xarray')
            
        Visualization Parameters:
            colormap (str): Colormap name for plots
            figure_size (Tuple[float, float]): Figure dimensions in inches
            dpi (int): Output resolution
            output_formats (List[str]): List of output formats
            
        Wind-Specific Parameters:
            u_variable, v_variable (str): Wind component variable names
            wind_level (str): Atmospheric level description
            wind_plot_type (str): Vector representation ('barbs' or 'arrows')
            subsample_factor (int): Vector density subsampling
            wind_scale (float): Vector scaling factor
            show_background (bool): Enable background wind speed display
            
        Performance Parameters:
            verbose (bool): Enable detailed logging
            parallel (bool): Enable parallel processing
            batch_all (bool): Process all available time steps
    """
    grid_file: str = ""
    data_dir: str = ""
    output_dir: str = ""
    
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    
    variable: str = "rainnc"
    accumulation_period: str = "a01h"
    
    plot_type: str = "scatter"
    time_index: int = 0
    title: Optional[str] = None
    clim_min: Optional[float] = None
    clim_max: Optional[float] = None
    output: Optional[str] = None
    figure_size: Tuple[float, float] = (12.0, 10.0)
    grid_resolution: Optional[int] = None
    grid_resolution_deg: Optional[float] = None

    u_variable: str = "u10"
    v_variable: str = "v10"
    wind_level: str = "surface"
    wind_plot_type: str = "barbs"
    subsample_factor: int = 0
    wind_scale: Optional[float] = None
    show_background: bool = False
    background_colormap: str = "viridis"
    
    colormap: str = "default"
    figure_width: float = 10.0
    figure_height: float = 9.0
    dpi: int = 300
    output_formats: List[str] = None
    
    use_pure_xarray: bool = False
    batch_mode: bool = False
    verbose: bool = True
    quiet: bool = False
    
    chunk_size: int = 100000
    parallel: bool = False
    
    def __post_init__(self):
        """
        Post-initialization processing.

        Parameters:
            None

        Returns:
            None
        """
        if self.output_formats is None:
            self.output_formats = ["png"]
        
        if not self._validate_spatial_extent():
            raise ValueError("Invalid spatial extent parameters")
    
    def _validate_spatial_extent(self) -> bool:
        """
        Validate spatial extent parameters.

        Parameters:
            None

        Returns:
            bool: True if spatial extent parameters are valid.
        """
        if any(x is None for x in [self.lon_min, self.lon_max, self.lat_min, self.lat_max]):
            return True
            
        return (
            -180.0 <= self.lon_min <= 180.0 and
            -180.0 <= self.lon_max <= 180.0 and
            -90.0 <= self.lat_min <= 90.0 and
            -90.0 <= self.lat_max <= 90.0 and
            self.lon_max > self.lon_min and
            self.lat_max > self.lat_min
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Parameters:
            None

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MPASConfig':
        """
        Create configuration from dictionary.

        Parameters:
            config_dict (Dict[str, Any]): Configuration dictionary.

        Returns:
            MPASConfig: Constructed configuration object.
        """
        if 'figure_size' in config_dict and isinstance(config_dict['figure_size'], list):
            config_dict['figure_size'] = tuple(config_dict['figure_size'])
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to YAML file.

        Parameters:
            filepath (str): Path to output YAML file.

        Returns:
            None
        """
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MPASConfig':
        """
        Load configuration from YAML file.

        Parameters:
            filepath (str): Path to YAML configuration file.

        Returns:
            MPASConfig: Loaded configuration object.
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)


class MPASLogger:
    """Logging utility for MPAS analysis."""
    
    def __init__(self, name: str = "mpas_analysis", level: int = logging.INFO,
                 log_file: Optional[str] = None, verbose: bool = True):
        """
        Initialize logger.

        Parameters:
            name (str): Logger name.
            level (int): Logging level.
            log_file (Optional[str]): Log file path.
            verbose (bool): Enable console output.

        Returns:
            None
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """
        Log info message.

        Parameters:
            message (str): Message to log.

        Returns:
            None
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        Log warning message.

        Parameters:
            message (str): Message to log.

        Returns:
            None
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        Log error message.

        Parameters:
            message (str): Message to log.

        Returns:
            None
        """
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """
        Log debug message.

        Parameters:
            message (str): Message to log.

        Returns:
            None
        """
        self.logger.debug(message)


class FileManager:
    """File and directory management utilities."""
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Ensure directory exists, create if necessary.

        Parameters:
            directory (str): Directory path to ensure.

        Returns:
            None
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def find_files(directory: str, pattern: str = "*.nc", 
                  recursive: bool = False) -> List[str]:
        """
        Find files matching pattern.

        Parameters:
            directory (str): Directory to search.
            pattern (str): File pattern.
            recursive (bool): Search recursively.

        Returns:
            List[str]: List of matching file paths.
        """
        path = Path(directory)
        
        if recursive:
            return sorted([str(p) for p in path.rglob(pattern)])
        else:
            return sorted([str(p) for p in path.glob(pattern)])
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """
        Get file information.

        Parameters:
            filepath (str): File path.

        Returns:
            Dict[str, Any]: File information.
        """
        path = Path(filepath)
        
        if not path.exists():
            return {"exists": False}
        
        stat = path.stat()
        
        return {
            "exists": True,
            "size": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "created": datetime.fromtimestamp(stat.st_ctime),
        }
    
    @staticmethod
    def cleanup_files(directory: str, pattern: str = "*.tmp",
                     older_than_days: int = 7) -> int:
        """
        Clean up temporary files older than specified days.

        Parameters:
            directory (str): Directory to clean.
            pattern (str): File pattern.
            older_than_days (int): Age threshold in days.

        Returns:
            int: Number of files deleted.
        """
        path = Path(directory)
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0
        
        for filepath in path.glob(pattern):
            if datetime.fromtimestamp(filepath.stat().st_mtime) < cutoff_time:
                filepath.unlink()
                deleted_count += 1
        
        return deleted_count


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_coordinates(lon: np.ndarray, lat: np.ndarray) -> bool:
        """
        Validate coordinate arrays.

        Parameters:
            lon (np.ndarray): Longitude array.
            lat (np.ndarray): Latitude array.

        Returns:
            bool: True if valid.
        """
        if len(lon) != len(lat):
            return False
        
        if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
            return False
        
        if not (-180 <= np.min(lon) and np.max(lon) <= 180):
            return False
        
        if not (-90 <= np.min(lat) and np.max(lat) <= 90):
            return False
        
        return True
    
    @staticmethod
    def validate_data_array(data: np.ndarray, 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate data array and return statistics.

        Parameters:
            data (np.ndarray): Data array.
            min_val (Optional[float]): Expected minimum value.
            max_val (Optional[float]): Expected maximum value.

        Returns:
            Dict[str, Any]: Validation results and statistics.
        """
        results = {
            "valid": True,
            "issues": [],
            "stats": {}
        }
        
        finite_mask = np.isfinite(data)
        finite_count = np.sum(finite_mask)
        total_count = len(data.flatten())
        
        results["stats"]["total_points"] = total_count
        results["stats"]["finite_points"] = finite_count
        results["stats"]["finite_percentage"] = (finite_count / total_count) * 100
        
        if finite_count == 0:
            results["valid"] = False
            results["issues"].append("No finite values found")
            return results
        
        finite_data = data[finite_mask]
        
        results["stats"]["min"] = float(np.min(finite_data))
        results["stats"]["max"] = float(np.max(finite_data))
        results["stats"]["mean"] = float(np.mean(finite_data))
        results["stats"]["std"] = float(np.std(finite_data))
        results["stats"]["median"] = float(np.median(finite_data))
        
        if min_val is not None and results["stats"]["min"] < min_val:
            results["issues"].append(f"Minimum value {results['stats']['min']:.2f} below expected {min_val}")
        
        if max_val is not None and results["stats"]["max"] > max_val:
            results["issues"].append(f"Maximum value {results['stats']['max']:.2f} above expected {max_val}")
        
        if results["stats"]["min"] == results["stats"]["max"]:
            results["issues"].append("All values are identical")
        
        if results["stats"]["std"] == 0:
            results["issues"].append("Zero standard deviation")
        
        if results["issues"]:
            results["valid"] = False
        
        return results


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self):
        """
        Initialize performance monitor.

        Parameters:
            None

        Returns:
            None
        """
        self.start_times = {}
        self.durations = {}
    
    @contextmanager
    def timer(self, operation_name: str):
        """
        Context manager for timing operations.

        Parameters:
            operation_name (str): Name of the timed operation.

        Returns:
            contextmanager: Context manager that times the operation.
        """
        start_time = datetime.now()
        self.start_times[operation_name] = start_time
        
        try:
            yield
        finally:
            end_time = datetime.now()
            duration = end_time - start_time
            self.durations[operation_name] = duration
            
            print(f"{operation_name} completed in {duration.total_seconds():.2f} seconds")
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get timing summary.

        Parameters:
            None

        Returns:
            Dict[str, float]: Mapping of operation names to durations (s).
        """
        return {name: duration.total_seconds() 
                for name, duration in self.durations.items()}
    
    def print_summary(self) -> None:
        """
        Print timing summary.

        Parameters:
            None

        Returns:
            None
        """
        print("\n=== Performance Summary ===")
        for name, duration in self.durations.items():
            print(f"{name}: {duration.total_seconds():.2f} seconds")
        
        if self.durations:
            total_time = sum(d.total_seconds() for d in self.durations.values())
            print(f"Total time: {total_time:.2f} seconds")


class ArgumentParser:
    """Command-line argument parser for MPAS analysis."""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """
        Create argument parser with all MPAS analysis options.

        Parameters:
            None

        Returns:
            argparse.ArgumentParser: Configured argument parser.
        """
        parser = argparse.ArgumentParser(
            description="MPAS Data Analysis and Visualization Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic precipitation analysis
  mpas-analyze --grid-file grid.nc --data-dir ./data --output-dir ./output
  
  # Custom spatial extent
  mpas-analyze --grid-file grid.nc --data-dir ./data --lat-min -10 --lat-max 15
  
  # Batch processing
  mpas-analyze --grid-file grid.nc --data-dir ./data --batch-all
  
  # Use configuration file
  mpas-analyze --config config.yaml
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
        viz_group.add_argument('--dpi', type=int, default=300,
                              help='Output resolution (DPI)')
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
        Convert parsed arguments to MPASConfig object.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments.

        Returns:
            MPASConfig: Configuration object built from arguments.
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
        Create argument parser for surface variable plotting.

        Parameters:
            None

        Returns:
            argparse.ArgumentParser: Configured parser for surface plotting.
        """
        parser = argparse.ArgumentParser(
            description="MPAS Surface Variable Plotting Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Temperature scatter plot
  mpas-surface-plot --grid-file grid.nc --data-dir ./data --variable t2m --plot-type scatter
  
  # Pressure contour plot with custom extent
  mpas-surface-plot --grid-file grid.nc --data-dir ./data --variable surface_pressure --plot-type contour --lat-min -10 --lat-max 15
  
  # Wind speed with custom colormap
  mpas-surface-plot --grid-file grid.nc --data-dir ./data --variable wspd10 --colormap plasma --time-index 12
  
  # Sea level pressure for specific time
  mpas-surface-plot --grid-file grid.nc --data-dir ./data --variable mslp --plot-type contour --time-index 24 --output mslp_analysis
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
        output_group.add_argument('--dpi', type=int, default=300,
                                 help='Output resolution (DPI) (default: 300)')
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
        Convert surface plot arguments to MPASConfig object.

        Parameters:
            args (argparse.Namespace): Parsed surface plotting arguments.

        Returns:
            MPASConfig: Configuration object for surface plotting.
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
        Create argument parser for wind plotting.

        Parameters:
            None

        Returns:
            argparse.ArgumentParser: Configured parser for wind plotting.
        """
        parser = argparse.ArgumentParser(
            description="Generate MPAS wind vector plots with barbs or arrows",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              # Surface wind with barbs
              mpas-wind-plot --grid_file /path/to/grid.nc --data_dir /path/to/data --u-variable u10 --v-variable v10

              # 850mb wind with arrows
              mpas-wind-plot --grid_file /path/to/grid.nc --data_dir /path/to/data --u-variable u850 --v-variable v850 --wind-plot-type arrows

              # Custom extent and subsampling
              mpas-wind-plot --grid_file /path/to/grid.nc --data_dir /path/to/data --u-variable u10 --v-variable v10 --extent -105 -95 35 45 --subsample 3
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
        parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
        
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
        proc_group = parser.add_argument_group('Processing')
        proc_group.add_argument('--batch-all', action='store_true',
                               help='Process all time steps in batch mode')

        return parser

    @staticmethod
    def parse_wind_args_to_config(args: argparse.Namespace) -> MPASConfig:
        """
        Parse wind plotting command line arguments to MPASConfig.

        Parameters:
            args (argparse.Namespace): Parsed wind plotting arguments.

        Returns:
            MPASConfig: Configuration object for wind plotting.
        """
        if args.extent:
            lon_min, lon_max, lat_min, lat_max = args.extent
        else:
            lon_min = lon_max = lat_min = lat_max = None
        
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


def setup_warnings() -> None:
    """Configure warning filters for MPAS analysis."""
    warnings.filterwarnings('ignore', message='The specified chunks separate the stored chunks.*')
    warnings.filterwarnings('ignore', message='invalid value encountered in create_collection')
    warnings.filterwarnings('ignore', message='.*Shapely.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*chunks.*degrade performance.*')


def print_system_info() -> None:
    """Print system and environment information."""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Available memory: {get_available_memory():.1f} GB")
    print("=" * 30)


def get_available_memory() -> float:
    """
    Get available system memory in GB.
    
    Returns:
        float: Available memory in GB
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        return 0.0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Parameters:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def create_output_filename(base_name: str, time_str: str, var_name: str,
                          accum_period: str, extension: str = 'png') -> str:
    """
    Create standardized output filename.
    
    Parameters:
        base_name (str): Base filename
        time_str (str): Time string
        var_name (str): Variable name
        accum_period (str): Accumulation period
        extension (str): File extension
        
    Returns:
        str: Formatted filename
    """
    return f"{base_name}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_point.{extension}"


def load_config_file(config_file: str) -> MPASConfig:
    """
    Load configuration from file with error handling.
    
    Parameters:
        config_file (str): Configuration file path
        
    Returns:
        MPASConfig: Loaded configuration
    """
    try:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        return MPASConfig.load_from_file(config_file)
    
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print("Using default configuration")
        return MPASConfig()


def validate_input_files(config: MPASConfig) -> bool:
    """
    Validate input files exist and are accessible.
    
    Parameters:
        config (MPASConfig): Configuration object
        
    Returns:
        bool: True if all files are valid
    """
    errors = []
    
    if not config.grid_file:
        errors.append("Grid file not specified")
    elif not os.path.exists(config.grid_file):
        errors.append(f"Grid file not found: {config.grid_file}")
    
    if not config.data_dir:
        errors.append("Data directory not specified")
    elif not os.path.exists(config.data_dir):
        errors.append(f"Data directory not found: {config.data_dir}")
    elif not os.path.isdir(config.data_dir):
        errors.append(f"Data path is not a directory: {config.data_dir}")
    
    if config.data_dir and os.path.exists(config.data_dir):
        data_files = FileManager.find_files(config.data_dir, "diag*.nc")
        if not data_files:
            errors.append(f"No diagnostic files found in: {config.data_dir}")
    
    if errors:
        print("Input validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def get_accumulation_hours(accum_period: str) -> int:
    """
    Get accumulation hours from accumulation period string.
    
    Parameters:
        accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h')
        
    Returns:
        int: Number of hours for the accumulation period
    """
    accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
    return accum_hours_map.get(accum_period, 1)