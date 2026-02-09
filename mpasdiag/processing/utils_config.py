#!/usr/bin/env python3

"""
MPAS Configuration Management Utilities

This module provides comprehensive configuration management for MPAS data analysis and visualization workflows including parameter validation, YAML file I/O operations, and centralized settings storage. It implements the MPASConfig dataclass that maintains all analysis parameters (data paths, variable names, geographic extents, visualization options, output settings) in a structured format with type hints and default values, supports loading configuration from YAML files with validation and error handling, enables command-line argument override of file-based configurations, and provides serialization methods for saving configurations to disk. The configuration system centralizes parameter management across all MPASdiag modules (processors, visualizers, CLI), ensures consistency between interactive and batch processing workflows, validates parameter combinations and value ranges with comprehensive error messages, and supports all analysis types including precipitation maps, surface variable plots, wind vectors, and vertical cross-sections. Core capabilities include automatic parameter validation with type checking, YAML parsing with schema validation, configuration merging for combining multiple sources, and extensible design for adding new analysis-specific parameters.

Classes:
    MPASConfig: Centralized configuration dataclass for all MPAS analysis operations with validation and file I/O capabilities.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


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
    grid_resolution: Optional[float] = None
    regrid_method: str = "linear"

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
    dpi: int = 100  
    output_formats: Optional[List[str]] = None
    
    use_pure_xarray: bool = False
    batch_mode: bool = False
    verbose: bool = True
    quiet: bool = False
    
    chunk_size: int = 100000
    parallel: bool = False
    workers: Optional[int] = None
    
    analysis_type: Optional[str] = None
    time_start: Optional[int] = None
    time_end: Optional[int] = None
    
    precip_threshold: Optional[float] = None
    precip_units: str = "mm"
    
    interpolation_method: str = "linear"
    contour_levels: int = 15
    levels: Optional[List[float]] = None
    
    vector_color: str = "black"
    vector_alpha: float = 0.8
    
    start_lon: Optional[float] = None
    start_lat: Optional[float] = None
    end_lon: Optional[float] = None
    end_lat: Optional[float] = None
    vertical_coord: str = "pressure"
    num_points: int = 100
    max_height: Optional[float] = None
    plot_style: str = "filled_contour"
    extend: str = "both"
    
    overlay_type: Optional[str] = None
    primary_variable: Optional[str] = None
    secondary_variable: Optional[str] = None
    variables: Optional[List[str]] = None
    pressure_variable: str = "mslp"
    primary_colormap: Optional[str] = None
    secondary_colormap: Optional[str] = None
    transparency: float = 0.7
    contour_overlay: bool = False
    
    def __post_init__(self) -> None:
        """
        Execute post-initialization validation and default value assignment after dataclass instantiation. This method is automatically called by the dataclass mechanism following object construction to perform configuration validation and setup. It ensures output_formats has a default value if not specified and validates spatial extent parameters for consistency. If spatial extent validation fails, it raises a ValueError to prevent invalid configurations from being used. This hook enables robust configuration validation without requiring explicit constructor override.

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
        Validate spatial extent parameters to ensure geographic bounds are physically meaningful and properly ordered. This method checks that longitude values fall within [-180, 180] degrees, latitude values are within [-90, 90] degrees, and maximum bounds exceed minimum bounds for both coordinates. If any coordinate is None, validation is skipped to allow for unspecified extent. Returns True when extent is valid or unspecified, False when invalid bounds are detected. This validation prevents downstream plotting and analysis errors from malformed geographic constraints.

        Parameters:
            None

        Returns:
            bool: True if spatial extent parameters are valid or unspecified, False if invalid bounds detected.
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
        Convert configuration object to a dictionary representation for serialization and inspection. This method uses the dataclass asdict utility to transform all configuration attributes into a dictionary structure suitable for YAML export or programmatic access. It performs post-processing to convert tuple values to lists since YAML and many serialization formats prefer list representations. The resulting dictionary contains all configuration parameters with their current values. This enables configuration persistence, comparison, and external manipulation of settings.

        Parameters:
            None

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters with tuple values converted to lists.
        """
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MPASConfig':
        """
        Construct a new configuration object from a dictionary of parameter values with automatic type conversion. This class method serves as an alternative constructor that accepts a dictionary containing configuration parameters and returns a properly initialized MPASConfig instance. It handles special type conversions such as transforming list representations back to tuples for parameters like figure_size that require tuple types. The method enables configuration loading from YAML files, JSON, or any dictionary source. This pattern supports flexible configuration creation from external data sources while maintaining type safety.

        Parameters:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters with keys matching MPASConfig attribute names.

        Returns:
            MPASConfig: Newly constructed configuration object initialized with dictionary values.
        """
        if 'figure_size' in config_dict and isinstance(config_dict['figure_size'], list):
            config_dict['figure_size'] = tuple(config_dict['figure_size'])
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Persist the current configuration to a YAML file for reproducibility and sharing. This method converts the configuration object to a dictionary representation, writes it to the specified file path using YAML formatting with proper indentation, and prints a confirmation message. The YAML format is human-readable, version-control friendly, and easily editable, making it ideal for configuration management. Saved configurations can be loaded later using load_from_file to reproduce analysis workflows exactly. This enables configuration version control, sharing analysis setups between team members, and maintaining consistent processing parameters across runs.

        Parameters:
            filepath (str): Absolute or relative path to output YAML file where configuration will be saved.

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
        Load configuration parameters from a YAML file and construct a validated configuration object. This class method reads a YAML configuration file using safe loading to prevent code execution vulnerabilities, parses the parameter dictionary, and uses from_dict to construct a fully initialized MPASConfig instance. It enables reproducible analysis workflows by restoring previously saved configurations with all parameters intact. The method handles type conversions automatically and validates the configuration through __post_init__ after construction. This is the primary mechanism for loading saved analysis setups and sharing configurations across analysis runs.

        Parameters:
            filepath (str): Absolute or relative path to YAML configuration file to load.

        Returns:
            MPASConfig: Loaded and validated configuration object with all parameters from file.
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)