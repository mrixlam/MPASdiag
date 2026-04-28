#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Configuration Management Utilities

This module defines the MPASConfig dataclass, which encapsulates all configuration parameters for MPAS analysis and plotting. It provides methods for validating spatial extent parameters, converting the configuration to and from dictionary format, and saving/loading configurations to YAML files. This structured approach to configuration management ensures that all parameters are organized, validated, and easily accessible throughout the processing workflow. 

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
    """ Configuration class for MPAS analysis parameters. """

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

    remap_engine: str = "kdtree"
    remap_method: str = "nearest"

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
    plot_style: str = "contourf"
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
    
    sounding_lon: Optional[float] = None
    sounding_lat: Optional[float] = None
    show_indices: bool = False
    show_parcel: bool = False
    
    def __post_init__(self: 'MPASConfig') -> None:
        """
        This method is automatically called after the dataclass is initialized. It performs validation and sets default values for certain parameters if they are not provided. Specifically, it checks if the output_formats parameter is None and sets it to a default list containing "png". It also calls the _validate_spatial_extent method to ensure that the spatial extent parameters (latitude and longitude bounds) are valid. If the spatial extent parameters are invalid, it raises a ValueError with an appropriate message. This post-initialization step ensures that the configuration object is in a consistent and valid state before it is used in any processing or plotting operations. 

        Parameters:
            None

        Returns:
            None
        """
        if self.output_formats is None:
            self.output_formats = ["png"]

        if not self._validate_spatial_extent():
            raise ValueError("Invalid spatial extent parameters")

        self._validate_remap_settings()
    
    def _validate_spatial_extent(self: 'MPASConfig') -> bool:
        """
        This method validates the spatial extent parameters (latitude and longitude bounds) of the configuration. It checks if any of the spatial extent parameters (lon_min, lon_max, lat_min, lat_max) are None, in which case it considers the spatial extent to be valid (as it may be unspecified). If all parameters are specified, it ensures that the longitude values are between -180 and 180 degrees, the latitude values are between -90 and 90 degrees, and that the maximum bounds are greater than the minimum bounds. This validation step is crucial for ensuring that any spatial subsetting or plotting operations based on these parameters will function correctly without encountering errors due to invalid geographic bounds. 

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
    
    def _validate_remap_settings(self: 'MPASConfig') -> None:
        """
        This method validates the remapping settings specified in the configuration. It checks if the remap_engine parameter is set to a known value (either "kdtree" or "esmf") and raises a ValueError if it is not. If the remap_engine is "esmf" and the remap_method is set to "nearest", it automatically changes the remap_method to "nearest_s2d" since "nearest" is not a valid method for ESMF. Then, it checks if the remap_method is valid for the specified remap_engine by comparing it against a predefined set of allowed methods for each engine. If the remap_method is not valid for the chosen remap_engine, it raises a ValueError with an appropriate message indicating the allowed methods. This validation ensures that the remapping settings are consistent and compatible with the underlying remapping libraries, preventing errors during the remapping process.

        Parameters:
            None

        Returns:
            None
        """
        _KDTREE_METHODS = {"nearest", "linear"}

        try:
            from .remapping import MPASRemapper
            _ESMF_METHODS = set(MPASRemapper._METHOD_MAP.keys())
        except Exception:
            _ESMF_METHODS = {
                "bilinear", "conservative", "conservative_normed",
                "patch", "nearest_s2d", "nearest_d2s",
            }
        _KNOWN_ENGINES = {"kdtree", "esmf"}

        if self.remap_engine not in _KNOWN_ENGINES:
            raise ValueError(
                f"Invalid remap_engine '{self.remap_engine}'. "
                f"Must be one of: {sorted(_KNOWN_ENGINES)}"
            )

        if self.remap_engine == "esmf" and self.remap_method == "nearest":
            self.remap_method = "bilinear"

        allowed = _KDTREE_METHODS if self.remap_engine == "kdtree" else _ESMF_METHODS

        if self.remap_method not in allowed:
            raise ValueError(
                f"Invalid remap_method '{self.remap_method}' for "
                f"remap_engine='{self.remap_engine}'. "
                f"Allowed methods: {sorted(allowed)}"
            )

    def to_dict(self: 'MPASConfig') -> Dict[str, Any]:
        """
        This method converts the MPASConfig dataclass instance into a dictionary format. It uses the asdict function from the dataclasses module to create a dictionary representation of the dataclass. Additionally, it iterates through the dictionary items and checks if any values are tuples (such as figure_size) and converts them to lists, since YAML serialization does not support tuples. This method ensures that all configuration parameters are in a format suitable for saving to a YAML file or for other uses where a dictionary representation is needed. 

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
    def from_dict(cls: 'MPASConfig', 
                  config_dict: Dict[str, Any]) -> 'MPASConfig':
        """
        This class method creates an instance of MPASConfig from a given dictionary of configuration parameters. It takes a dictionary as input, which should contain keys that match the attribute names of the MPASConfig dataclass. The method checks if the 'figure_size' key is present in the dictionary and if its value is a list; if so, it converts it to a tuple since the MPASConfig class expects figure_size to be a tuple. Finally, it uses the unpacking operator (**) to pass the dictionary items as keyword arguments to the MPASConfig constructor, which will initialize the instance with the provided configuration values. This method allows for easy creation of configuration objects from dictionaries, such as those loaded from YAML files or other sources.  

        Parameters: 
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters with keys matching MPASConfig attributes. 

        Returns:
            MPASConfig: An instance of MPASConfig initialized with the provided configuration parameters. 
        """
        if 'figure_size' in config_dict and isinstance(config_dict['figure_size'], list):
            config_dict['figure_size'] = tuple(config_dict['figure_size'])
        return cls(**config_dict)
    
    def save_to_file(self: 'MPASConfig', 
                     filepath: str) -> None:
        """
        This method saves the current configuration parameters of the MPASConfig instance to a YAML file at the specified filepath. It first converts the configuration to a dictionary format using the to_dict method, which also handles any necessary conversions (such as tuples to lists). Then, it opens the specified file in write mode and uses the yaml.dump function to write the configuration dictionary to the file in a human-readable format with proper indentation. After successfully saving the configuration, it prints a confirmation message indicating where the configuration has been saved. This method allows users to easily export their current configuration settings for later use or sharing with others. 

        Parameters:
            filepath (str): Absolute or relative path to YAML file where configuration should be saved. 

        Returns:
            None 
        """
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_from_file(cls: 'MPASConfig', 
                       filepath: str) -> 'MPASConfig':
        """
        This class method loads configuration parameters from a specified YAML file and creates an instance of MPASConfig with those parameters. It opens the YAML file in read mode and uses the yaml.safe_load function to parse the contents of the file into a dictionary. Then, it calls the from_dict class method to convert the loaded dictionary into an MPASConfig instance, which will be initialized with the parameters from the file. This method allows users to easily load previously saved configurations or configurations shared by others, ensuring that all parameters are correctly set up for use in analysis and plotting operations. 

        Parameters:
            filepath (str): Absolute or relative path to YAML file containing configuration parameters. 

        Returns:
            MPASConfig: An instance of MPASConfig initialized with parameters loaded from the specified YAML file. 
        """
        with open(filepath, 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
        
        return cls.from_dict(config_dict)