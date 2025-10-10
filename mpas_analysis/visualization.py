#!/usr/bin/env python3

"""
MPAS Visualization Module

This module provides comprehensive visualization functionality for MPAS model output data,
including precipitation maps, scatter plots, and cartographic presentations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Last Modified: 2025-10-10

Architecture:
    The module follows a hierarchical class structure for better maintainability:
    
    - UnitConverter: Utility class for meteorological unit conversions
    - MPASFileMetadata: Centralized metadata management for MPAS variables
    - MPASVisualizer: Base class with common functionality (coordinate formatting,
      map projections, file I/O, plot validation, etc.)
    - MPASPrecipitationPlotter: Specialized for precipitation visualizations
    - MPASSurfacePlotter: Specialized for surface variable plotting (scatter/contour)
    - MPASWindPlotter: Specialized for wind vector plots (barbs/arrows)

Features:
    - Professional cartographic precipitation maps
    - Scatter plot visualization for unstructured data
    - Contour plots with interpolation to regular grids
    - Wind vector plotting with automatic subsampling
    - Customizable colormaps and color levels
    - High-quality output with multiple format support
    - Flexible map extents and projections
    - Comprehensive plot annotation and metadata

Usage Examples:
    # Using specialized plotters (clean architecture)
    precip_plotter = MPASPrecipitationPlotter(figsize=(12, 8))
    fig, ax = precip_plotter.create_precipitation_map(lon, lat, precip_data, ...)
    
    surface_plotter = MPASSurfacePlotter()
    fig, ax = surface_plotter.create_surface_map(lon, lat, temp_data, 't2m', ...)
    
    wind_plotter = MPASWindPlotter()
    fig, ax = wind_plotter.create_wind_plot(lon, lat, u_data, v_data, ...)

Unit Conversion & Metadata Pipeline:
    The visualization functions (create_surface_map, create_precipitation_map, create_wind_plot)
    handle unit conversions internally through the metadata system. Users should pass raw
    MPAS data directly to these functions without manual conversion.
    
    Conversion Flow:
    1. Raw MPAS data → visualization function
    2. Function gets metadata via MPASFileMetadata.get_2d_variable_metadata()
    3. Metadata includes display units and appropriate ranges
    4. UnitConverter.convert_data_for_display() handles automatic conversion
    5. Data is plotted with correct units in labels and ranges
    
    Manual Conversion (Advanced Users):
    If manual conversion is needed, use UnitConverter.convert_data_for_display() BEFORE calling
    visualization functions, but ensure no double conversion occurs.
    
    Metadata Management:
    MPASFileMetadata provides centralized access to variable attributes including:
    - Units (original and display), long names, colormaps
    - Contour levels, spatial dimensions
    - Support for 90+ MPAS atmospheric and ocean variables
    - Placeholder framework for future 3D variable support
"""

import os
import re
import warnings
from datetime import datetime
from typing import Tuple, Optional, List, Any, Union

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .utils import get_accumulation_hours
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.geoaxes import GeoAxes

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False})

warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')


class UnitConverter:
    """
    Unit conversion utility class for meteorological and atmospheric data.
    
    This class provides comprehensive unit conversion functionality following
    meteorological conventions, including temperature, pressure, precipitation,
    wind speed, and other atmospheric variables.
    """
    
    @staticmethod
    def convert_units(data: Union[np.ndarray, xr.DataArray, float], 
                      from_unit: str, 
                      to_unit: str) -> Union[np.ndarray, xr.DataArray, float]:
        """
        Convert data from one unit to another following meteorological conventions.
        
        Parameters:
            data: Input data (array, DataArray, or scalar)
            from_unit (str): Source unit
            to_unit (str): Target unit
            
        Returns:
            Converted data in the same format as input
            
        Supported conversions:
            Temperature: K ↔ °C, K ↔ °F, °C ↔ °F
            Pressure: Pa ↔ hPa, Pa ↔ mb, hPa ↔ mb
            Mixing ratio: kg/kg ↔ g/kg
            Precipitation: mm/hr ↔ mm/day, mm/hr ↔ in/hr
            Wind speed: m/s ↔ kt, m/s ↔ mph, m/s ↔ km/h
            Distance: m ↔ km, m ↔ ft
        """
        from_unit = UnitConverter._normalize_unit_string(from_unit)
        to_unit = UnitConverter._normalize_unit_string(to_unit)
        
        if from_unit == to_unit:
            return data
        
        conversion_map = {
        ('K', '°C'): lambda x: x - 273.15,
        ('°C', 'K'): lambda x: x + 273.15,
        ('K', '°F'): lambda x: (x - 273.15) * 9/5 + 32,
        ('°F', 'K'): lambda x: (x - 32) * 5/9 + 273.15,
        ('°C', '°F'): lambda x: x * 9/5 + 32,
        ('°F', '°C'): lambda x: (x - 32) * 5/9,
        
        ('Pa', 'hPa'): lambda x: x / 100.0,
        ('hPa', 'Pa'): lambda x: x * 100.0,
        ('Pa', 'mb'): lambda x: x / 100.0,
        ('mb', 'Pa'): lambda x: x * 100.0,
        ('hPa', 'mb'): lambda x: x,  
        ('mb', 'hPa'): lambda x: x,  
        
        ('kg/kg', 'g/kg'): lambda x: x * 1000.0,
        ('g/kg', 'kg/kg'): lambda x: x / 1000.0,
        
        ('mm/hr', 'mm/day'): lambda x: x * 24.0,
        ('mm/day', 'mm/hr'): lambda x: x / 24.0,
        ('mm/hr', 'in/hr'): lambda x: x / 25.4,
        ('in/hr', 'mm/hr'): lambda x: x * 25.4,
        
        ('m/s', 'kt'): lambda x: x * 1.94384,
        ('kt', 'm/s'): lambda x: x / 1.94384,
        ('m/s', 'mph'): lambda x: x * 2.23694,
        ('mph', 'm/s'): lambda x: x / 2.23694,
        ('m/s', 'km/h'): lambda x: x * 3.6,
        ('km/h', 'm/s'): lambda x: x / 3.6,
        
        ('m', 'km'): lambda x: x / 1000.0,
        ('km', 'm'): lambda x: x * 1000.0,
        ('m', 'ft'): lambda x: x * 3.28084,
            ('ft', 'm'): lambda x: x / 3.28084,
        }
        
        conversion_key = (from_unit, to_unit)
        if conversion_key not in conversion_map:
            raise ValueError(f"Conversion from '{from_unit}' to '{to_unit}' is not supported.\n"
                            f"Supported conversions: {list(conversion_map.keys())}")
        
        converter = conversion_map[conversion_key]
        return converter(data)

    @staticmethod
    def _normalize_unit_string(unit: str) -> str:
        """Normalize unit strings to handle common variations."""
        unit = unit.strip()
        
        unit_map = {
        'kelvin': 'K', 'k': 'K',
        'celsius': '°C', 'degc': '°C', 'deg_c': '°C', 'c': '°C',
        'fahrenheit': '°F', 'degf': '°F', 'deg_f': '°F', 'f': '°F',
        
        'pascal': 'Pa', 'pa': 'Pa',
        'hectopascal': 'hPa', 'hpa': 'hPa',
        'millibar': 'mb', 'mbar': 'mb',
        
        'kg kg-1': 'kg/kg', 'kg_kg-1': 'kg/kg', 'kg kg^{-1}': 'kg/kg',
        'g kg-1': 'g/kg', 'g_kg-1': 'g/kg', 'g kg^{-1}': 'g/kg',
        
        'mm hr-1': 'mm/hr', 'mm_hr-1': 'mm/hr',
        'mm day-1': 'mm/day', 'mm_day-1': 'mm/day',
        'in hr-1': 'in/hr', 'in_hr-1': 'in/hr',
        
        'knots': 'kt', 'knot': 'kt', 'kts': 'kt',
        'miles per hour': 'mph', 'mi/hr': 'mph',
        'kilometers per hour': 'km/h', 'km hr-1': 'km/h', 'km_hr-1': 'km/h',
        
        'meter': 'm', 'meters': 'm',
        'kilometer': 'km', 'kilometers': 'km',
            'foot': 'ft', 'feet': 'ft',
        }
        
        return unit_map.get(unit.lower(), unit)

    @staticmethod
    def _format_colorbar_label(text: str) -> str:
        """
        Format colorbar label text by capitalizing words while preserving unit formatting.
        
        Parameters:
            text (str): Label text, potentially containing units in parentheses
            
        Returns:
            str: Formatted text with capitalized words but preserved unit formatting
            
        Examples:
            'specific humidity (g/kg)' -> 'Specific Humidity (g/kg)'
            'wind speed (m/s)' -> 'Wind Speed (m/s)'
            'temperature (°C)' -> 'Temperature (°C)'
        """
        import re
        
        unit_pattern = r'(\([^)]+\))$'
        unit_match = re.search(unit_pattern, text)
        
        if unit_match:
            main_text = text[:unit_match.start()].strip()
            units = unit_match.group(1)
            
            formatted_main = main_text.title()
            return f"{formatted_main} {units}"
        else:
            return text.title()

    @staticmethod
    def get_display_units(variable_name: str, current_unit: str) -> str:
        """
        Get the preferred display unit for a given variable.
        
        Parameters:
            variable_name (str): Variable name (e.g., 't2m', 'rainnc', 'u10')
            current_unit (str): Current unit of the variable
            
        Returns:
            str: Preferred display unit following meteorological conventions
        """
        current_unit = UnitConverter._normalize_unit_string(current_unit)
    
        
        display_unit_preferences = {
            't2m': '°C', 'temperature': '°C', 'temp': '°C',
            'tsk': '°C', 'sst': '°C', 'meanT': '°C', 
            'dewpoint': '°C', 'dewpt': '°C', 'dpt': '°C',
            
            'pressure': 'hPa', 'slp': 'hPa', 'psfc': 'hPa',
            'mslp': 'hPa', 'pmsl': 'hPa',
            
            'rainnc': 'mm/hr', 'rainc': 'mm/hr', 'rain': 'mm/hr',
            'precipitation': 'mm/hr', 'precip': 'mm/hr',
            
            'u10': 'm/s', 'v10': 'm/s', 'wspd10': 'm/s',
            'wind_speed': 'm/s', 'wind': 'm/s',
            
            'qv': 'g/kg', 'humidity': 'g/kg', 'mixing_ratio': 'g/kg',
            'q2': 'g/kg', 'qv2m': 'g/kg',
        }
        
        for var_pattern, preferred_unit in display_unit_preferences.items():
            if var_pattern.lower() in variable_name.lower():
                if current_unit != preferred_unit:
                    try:
                        UnitConverter.convert_units(1.0, current_unit, preferred_unit)
                        return preferred_unit
                    except ValueError:
                        pass
                return preferred_unit
        
        return current_unit

    @staticmethod
    def convert_data_for_display(data: Union[np.ndarray, xr.DataArray], 
                                var_name: str, 
                                data_array: Optional[xr.DataArray] = None) -> Tuple[Union[np.ndarray, xr.DataArray], dict]:
        """
        Convert data values to preferred display units and return converted data with updated metadata.
        
        Parameters:
            data: Input data to convert
            var_name (str): Variable name
            data_array (Optional[xr.DataArray]): Original data array for extracting units
            
        Returns:
            Tuple containing:
            - Converted data in display units
            - Updated metadata dictionary with display units and levels
        """
        metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        
        original_unit = metadata.get('original_units', metadata['units'])
        display_unit = metadata['units']
        
        if original_unit != display_unit:
            try:
                converted_data = UnitConverter.convert_units(data, original_unit, display_unit)
                return converted_data, metadata
            except ValueError:
                pass
        
        return data, metadata


class MPASFileMetadata:
    """
    Metadata management class for MPAS variable attributes and visualization parameters.
    
    This class provides centralized metadata lookup for MPAS variables including
    units, long names, colormaps, and contour levels. It handles both 2D surface
    variables and placeholder functionality for future 3D atmospheric variables.
    """
    
    @staticmethod
    def get_2d_variable_metadata(var_name: str, data_array: Optional[xr.DataArray] = None) -> dict:
        """
        Get metadata for MPAS 2D surface variables including units, names, and colormaps.
        
        This function handles 2D surface variables only. For future 3D variable support,
        a separate get_3d_variable_metadata() function will be implemented.

        Parameters:
            var_name (str): 2D surface variable name.
            data_array (Optional[xr.DataArray]): Data array containing variable attributes.

        Returns:
            dict: Variable metadata including 'units', 'long_name', 'colormap', 'levels', and 'spatial_dim'.
            
        Note:
            All variables returned are 2D surface fields. 3D atmospheric variables will be
            handled by a separate function when 3D support is implemented.
        """
        standard_metadata = {
            'olrtoa': {'units': 'W/m^2', 'long_name': 'Outgoing Longwave Radiation at TOA', 'colormap': 'inferno'},
            
            'rainc': {'units': 'mm', 'long_name': 'Convective Precipitation', 'colormap': 'Blues'},  
            'rainnc': {'units': 'mm', 'long_name': 'Non-Convective Precipitation', 'colormap': 'Blues'},  
            'precipw': {'units': 'mm', 'long_name': 'Precipitable Water', 'colormap': 'Blues'}, 

            'refl10cm_max': {'units': 'dBZ', 'long_name': 'Maximum 10cm Reflectivity', 'colormap': 'pyart_NWSRef'},
            'refl10cm_1km': {'units': 'dBZ', 'long_name': '10cm Reflectivity at 1km AGL', 'colormap': 'pyart_NWSRef'},
            'refl10cm_1km_max': {'units': 'dBZ', 'long_name': 'Maximum 10cm Reflectivity at 1km AGL', 'colormap': 'pyart_NWSRef'},
            
            'u10': {'units': 'm/s', 'long_name': '10-meter U-wind', 'colormap': 'RdBu_r'},
            'v10': {'units': 'm/s', 'long_name': '10-meter V-wind', 'colormap': 'RdBu_r'}, 
            'q2': {'units': 'g/kg', 'long_name': '2-meter Specific Humidity', 'colormap': 'Blues'}, 
            't2m': {'units': 'K', 'long_name': '2-meter Temperature', 'colormap': 'RdYlBu_r'}, 
            'th2m': {'units': 'K', 'long_name': '2-meter Potential Temperature', 'colormap': 'RdYlBu_r'}, 
            'mslp': {'units': 'Pa', 'long_name': 'Mean Sea Level Pressure', 'colormap': 'viridis'},
            
            'relhum_50hPa': {'units': '%', 'long_name': 'Relative Humidity at 50 hPa', 'colormap': 'Blues'},
            'relhum_100hPa': {'units': '%', 'long_name': 'Relative Humidity at 100 hPa', 'colormap': 'Blues'},
            'relhum_200hPa': {'units': '%', 'long_name': 'Relative Humidity at 200 hPa', 'colormap': 'Blues'},
            'relhum_250hPa': {'units': '%', 'long_name': 'Relative Humidity at 250 hPa', 'colormap': 'Blues'},
            'relhum_500hPa': {'units': '%', 'long_name': 'Relative Humidity at 500 hPa', 'colormap': 'Blues'},
            'relhum_700hPa': {'units': '%', 'long_name': 'Relative Humidity at 700 hPa', 'colormap': 'Blues'},
            'relhum_850hPa': {'units': '%', 'long_name': 'Relative Humidity at 850 hPa', 'colormap': 'Blues'},
            'relhum_925hPa': {'units': '%', 'long_name': 'Relative Humidity at 925 hPa', 'colormap': 'Blues'},
            
            'dewpoint_50hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 50 hPa', 'colormap': 'Blues'},
            'dewpoint_100hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 100 hPa', 'colormap': 'Blues'},
            'dewpoint_200hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 200 hPa', 'colormap': 'Blues'},
            'dewpoint_250hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 250 hPa', 'colormap': 'Blues'},
            'dewpoint_500hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 500 hPa', 'colormap': 'Blues'},
            'dewpoint_700hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 700 hPa', 'colormap': 'Blues'},
            'dewpoint_850hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 850 hPa', 'colormap': 'Blues'},  
            'dewpoint_925hPa': {'units': 'K', 'long_name': 'Dewpoint Temperature at 925 hPa', 'colormap': 'Blues'},  
            
            'temperature_50hPa': {'units': 'K', 'long_name': 'Temperature at 50 hPa', 'colormap': 'RdYlBu_r'},
            'temperature_100hPa': {'units': 'K', 'long_name': 'Temperature at 100 hPa', 'colormap': 'RdYlBu_r'},
            'temperature_200hPa': {'units': 'K', 'long_name': 'Temperature at 200 hPa', 'colormap': 'RdYlBu_r'},
            'temperature_250hPa': {'units': 'K', 'long_name': 'Temperature at 250 hPa', 'colormap': 'RdYlBu_r'},
            'temperature_500hPa': {'units': 'K', 'long_name': 'Temperature at 500 hPa', 'colormap': 'RdYlBu_r'},
            'temperature_700hPa': {'units': 'K', 'long_name': 'Temperature at 700 hPa', 'colormap': 'RdYlBu_r'},
            'temperature_850hPa': {'units': 'K', 'long_name': 'Temperature at 850 hPa', 'colormap': 'RdYlBu_r'},  
            'temperature_925hPa': {'units': 'K', 'long_name': 'Temperature at 925 hPa', 'colormap': 'RdYlBu_r'}, 
            
            'height_50hPa': {'units': 'm', 'long_name': 'Geopotential Height at 50 hPa', 'colormap': 'viridis'},
            'height_100hPa': {'units': 'm', 'long_name': 'Geopotential Height at 100 hPa', 'colormap': 'viridis'},
            'height_200hPa': {'units': 'm', 'long_name': 'Geopotential Height at 200 hPa', 'colormap': 'viridis'},
            'height_250hPa': {'units': 'm', 'long_name': 'Geopotential Height at 250 hPa', 'colormap': 'viridis'},
            'height_500hPa': {'units': 'm', 'long_name': 'Geopotential Height at 500 hPa', 'colormap': 'viridis'},
            'height_700hPa': {'units': 'm', 'long_name': 'Geopotential Height at 700 hPa', 'colormap': 'viridis'},
            'height_850hPa': {'units': 'm', 'long_name': 'Geopotential Height at 850 hPa', 'colormap': 'viridis'},
            'height_925hPa': {'units': 'm', 'long_name': 'Geopotential Height at 925 hPa', 'colormap': 'viridis'},

            'uzonal_50hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 50 hPa', 'colormap': 'RdBu_r'},
            'uzonal_100hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 100 hPa', 'colormap': 'RdBu_r'},
            'uzonal_200hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 200 hPa', 'colormap': 'RdBu_r'},
            'uzonal_250hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 250 hPa', 'colormap': 'RdBu_r'},
            'uzonal_500hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 500 hPa', 'colormap': 'RdBu_r'},
            'uzonal_700hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 700 hPa', 'colormap': 'RdBu_r'},
            'uzonal_850hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 850 hPa', 'colormap': 'RdBu_r'},
            'uzonal_925hPa': {'units': 'm/s', 'long_name': 'Zonal Wind at 925 hPa', 'colormap': 'RdBu_r'},

            'umeridional_50hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 50 hPa', 'colormap': 'RdBu_r'},
            'umeridional_100hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 100 hPa', 'colormap': 'RdBu_r'},
            'umeridional_200hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 200 hPa', 'colormap': 'RdBu_r'},
            'umeridional_250hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 250 hPa', 'colormap': 'RdBu_r'},
            'umeridional_500hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 500 hPa', 'colormap': 'RdBu_r'},
            'umeridional_700hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 700 hPa', 'colormap': 'RdBu_r'},
            'umeridional_850hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 850 hPa', 'colormap': 'RdBu_r'},
            'umeridional_925hPa': {'units': 'm/s', 'long_name': 'Meridional Wind at 925 hPa', 'colormap': 'RdBu_r'},

            'w_50hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 50 hPa', 'colormap': 'RdBu_r'},
            'w_100hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 100 hPa', 'colormap': 'RdBu_r'},
            'w_200hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 200 hPa', 'colormap': 'RdBu_r'},
            'w_250hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 250 hPa', 'colormap': 'RdBu_r'},
            'w_500hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 500 hPa', 'colormap': 'RdBu_r'},
            'w_700hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 700 hPa', 'colormap': 'RdBu_r'},
            'w_850hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 850 hPa', 'colormap': 'RdBu_r'},
            'w_925hPa': {'units': 'm/s', 'long_name': 'Vertical Velocity at 925 hPa', 'colormap': 'RdBu_r'},

            'vorticity_50hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 50 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},
            'vorticity_100hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 100 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},
            'vorticity_200hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 200 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},
            'vorticity_250hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 250 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},
            'vorticity_500hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 500 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},
            'vorticity_700hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 700 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},
            'vorticity_850hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 850 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},
            'vorticity_925hPa': {'units': '1/s', 'long_name': 'Relative Vorticity at 925 hPa', 'colormap': 'RdBu_r', 'spatial_dim': 'nVertices'},

            'meanT_500_300': {'units': 'K', 'long_name': 'Mean Temperature 500-300 hPa', 'colormap': 'RdYlBu_r'},
            'cape': {'units': 'J/kg', 'long_name': 'Convective Available Potential Energy', 'colormap': 'plasma'},
            'cin': {'units': 'J/kg', 'long_name': 'Convective Inhibition', 'colormap': 'viridis_r'},
            'lcl': {'units': 'm', 'long_name': 'Lifting Condensation Level', 'colormap': 'viridis'},
            'lfc': {'units': 'm', 'long_name': 'Level of Free Convection', 'colormap': 'viridis'},
            'srh_0_1km': {'units': 'm^2/s^2', 'long_name': 'Storm Relative Helicity 0-1km', 'colormap': 'plasma'},
            'srh_0_3km': {'units': 'm^2/s^2', 'long_name': 'Storm Relative Helicity 0-3km', 'colormap': 'plasma'},

            'uzonal_surface': {'units': 'm/s', 'long_name': 'Surface Zonal Wind', 'colormap': 'RdBu_r'},
            'uzonal_1km': {'units': 'm/s', 'long_name': 'Zonal Wind at 1km AGL', 'colormap': 'RdBu_r'},
            'uzonal_6km': {'units': 'm/s', 'long_name': 'Zonal Wind at 6km AGL', 'colormap': 'RdBu_r'},
            'umeridional_surface': {'units': 'm/s', 'long_name': 'Surface Meridional Wind', 'colormap': 'RdBu_r'},
            'umeridional_1km': {'units': 'm/s', 'long_name': 'Meridional Wind at 1km AGL', 'colormap': 'RdBu_r'},
            'umeridional_6km': {'units': 'm/s', 'long_name': 'Meridional Wind at 6km AGL', 'colormap': 'RdBu_r'},
            'temperature_surface': {'units': 'K', 'long_name': 'Surface Temperature', 'colormap': 'RdYlBu_r'},
            'dewpoint_surface': {'units': 'K', 'long_name': 'Surface Dewpoint Temperature', 'colormap': 'Blues'},

            'updraft_helicity_max': {'units': 'm^2/s^2', 'long_name': 'Maximum Updraft Helicity', 'colormap': 'plasma'},
            'w_velocity_max': {'units': 'm/s', 'long_name': 'Maximum Vertical Velocity', 'colormap': 'plasma'},
            'wind_speed_level1_max': {'units': 'm/s', 'long_name': 'Maximum Wind Speed at Level 1', 'colormap': 'plasma'},

            't_oml': {'units': 'K', 'long_name': 'Ocean Mixed Layer Temperature', 'colormap': 'RdYlBu_r'},
            't_oml_initial': {'units': 'K', 'long_name': 'Initial Ocean Mixed Layer Temperature', 'colormap': 'RdYlBu_r'},
            't_oml_200m_initial': {'units': 'K', 'long_name': 'Initial Ocean Temperature at 200m', 'colormap': 'RdYlBu_r'},
            'h_oml': {'units': 'm', 'long_name': 'Ocean Mixed Layer Depth', 'colormap': 'viridis'},
            'h_oml_initial': {'units': 'm', 'long_name': 'Initial Ocean Mixed Layer Depth', 'colormap': 'viridis'},
            'hu_oml': {'units': 'm^2/s', 'long_name': 'Ocean Mixed Layer U-momentum', 'colormap': 'RdBu_r'},
            'hv_oml': {'units': 'm^2/s', 'long_name': 'Ocean Mixed Layer V-momentum', 'colormap': 'RdBu_r'},

            'cldfrac_low_UPP': {'units': '', 'long_name': 'Low Cloud Fraction', 'colormap': 'gray'},
            'cldfrac_mid_UPP': {'units': '', 'long_name': 'Mid Cloud Fraction', 'colormap': 'gray'},
            'cldfrac_high_UPP': {'units': '', 'long_name': 'High Cloud Fraction', 'colormap': 'gray'},
            'cldfrac_tot_UPP': {'units': '', 'long_name': 'Total Cloud Fraction', 'colormap': 'gray'},
        }
        
        metadata = standard_metadata.get(var_name.lower(), {
            'units': '',
            'long_name': var_name,
            'colormap': 'viridis',
            'levels': list(range(0, 21)),
            'spatial_dim': 'nCells'  
        })
        
        if 'spatial_dim' not in metadata:
            metadata['spatial_dim'] = 'nCells'
        
        if data_array is not None:
            if hasattr(data_array, 'attrs'):
                if 'units' in data_array.attrs:
                    metadata['units'] = data_array.attrs['units']
                if 'long_name' in data_array.attrs:
                    metadata['long_name'] = data_array.attrs['long_name']
                elif 'standard_name' in data_array.attrs:
                    metadata['long_name'] = data_array.attrs['standard_name']
                    
            try:
                data_values = data_array.values
                if hasattr(data_values, 'flatten'):
                    flat_data = data_values.flatten()
                    finite_data = flat_data[np.isfinite(flat_data)]
                    if len(finite_data) > 0:
                        data_min = float(np.percentile(finite_data, 5))
                        data_max = float(np.percentile(finite_data, 95))
                        if data_max > data_min:
                            n_levels = 15
                            level_step = (data_max - data_min) / n_levels
                            metadata['levels'] = [data_min + i * level_step for i in range(n_levels + 1)]
            except Exception:
                pass
        
        original_unit = metadata['units']
        display_unit = UnitConverter.get_display_units(var_name, original_unit)
        
        if original_unit != display_unit:
            try:
                if 'levels' in metadata and metadata['levels']:
                    converted_levels = [UnitConverter.convert_units(level, original_unit, display_unit) 
                                      for level in metadata['levels']]
                    metadata['levels'] = converted_levels
                
                metadata['units'] = display_unit
                metadata['original_units'] = original_unit  
                
                if display_unit in ['°C', '°F']:
                    metadata['long_name'] = metadata['long_name'].replace(' Temperature', f' Temperature ({display_unit})')
                elif display_unit in ['hPa', 'mb']:
                    metadata['long_name'] = metadata['long_name'].replace(' Pressure', f' Pressure ({display_unit})')
                elif display_unit == 'g/kg':
                    metadata['long_name'] = metadata['long_name'].replace(' Humidity', f' Humidity ({display_unit})')
                elif display_unit == 'mm/hr':
                    metadata['long_name'] = metadata['long_name'].replace(' Precipitation', f' Precipitation ({display_unit})')
                    
            except ValueError:
                pass
        
        return metadata

    @staticmethod
    def get_3d_variable_metadata(var_name: str, level: Optional[Union[str, float]] = None, 
                               data_array: Optional[xr.DataArray] = None) -> dict:
        """
        PLACEHOLDER: Get metadata for MPAS 3D atmospheric variables.
        
        This function will handle 3D atmospheric variables that vary with height/pressure.
        
        Parameters:
            var_name (str): 3D atmospheric variable name (e.g., 'temperature', 'humidity').
            level (Optional[Union[str, float]]): Vertical level specification (pressure, height, model level).
            data_array (Optional[xr.DataArray]): Data array containing variable attributes.
            
        Returns:
            dict: Variable metadata including 'units', 'long_name', 'colormap', 'levels', 
                  'spatial_dim', and 'vertical_dim'.
                  
        Note:
            This is a placeholder function. Implementation needed for 3D variable support.
        """
        raise NotImplementedError(
            "3D variable support not yet implemented. "
            "This function is a placeholder for future development."
        )

    @staticmethod
    def get_3d_colormap_and_levels(var_name: str, level: Optional[Union[str, float]] = None,
                                  data_array: Optional[xr.DataArray] = None) -> Tuple[str, List[float]]:
        """
        PLACEHOLDER: Get appropriate colormap and levels for 3D atmospheric variables.
        
        Parameters:
            var_name (str): 3D atmospheric variable name.
            level (Optional[Union[str, float]]): Vertical level specification.
            data_array (Optional[xr.DataArray]): Data array for automatic level detection.
            
        Returns:
            Tuple[str, List[float]]: Colormap name and contour levels for 3D variables.
            
        Note:
            This is a placeholder function. Implementation needed for 3D variable support.
        """
        raise NotImplementedError(
            "3D variable support not yet implemented. "
            "This function is a placeholder for future development."
        )

    @staticmethod
    def plot_3d_variable_slice(data_array: xr.DataArray, lon: np.ndarray, lat: np.ndarray,
                              level: Union[str, float], var_name: str,
                              colormap: str = 'viridis', levels: Optional[List[float]] = None,
                              plot_type: str = 'contour', **kwargs) -> plt.Figure:
        """
        PLACEHOLDER: Create visualization for 3D atmospheric variable at specific level.
        
        This function will handle plotting of 3D atmospheric variables at specified
        vertical levels (pressure, height, or model levels).
        
        Parameters:
            data_array (xr.DataArray): 3D variable data array.
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            level (Union[str, float]): Vertical level to plot.
            var_name (str): Variable name for labeling.
            colormap (str): Colormap name.
            levels (Optional[List[float]]): Contour levels.
            plot_type (str): Plot type ('contour', 'contourf', 'scatter').
            **kwargs: Additional plotting parameters.
            
        Returns:
            plt.Figure: Matplotlib figure object.
            
        Note:
            This is a placeholder function. Implementation needed for 3D variable support.
        """
        raise NotImplementedError(
            "3D variable support not yet implemented. "
            "This function is a placeholder for future development."
        )


class MPASVisualizer:
    """
    Base class for MPAS model output visualization.
    
    This class provides common functionality and utilities for creating publication-quality
    maps and plots of MPAS unstructured mesh data with professional cartographic presentation.
    
    Specialized plotting classes inherit from this base class to provide specific
    visualization capabilities for different variable types.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 10), dpi: int = 300):
        """
        Initialize the MPAS visualizer.

        Parameters:
            figsize (Tuple[float, float]): Figure size in inches (width, height).
            dpi (int): Resolution for output images.

        Returns:
            None
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
    
    def add_timestamp_and_branding(self) -> None:
        """
        Add timestamp and MPASdiag branding to the bottom left of the current figure.

        Parameters:
            None

        Returns:
            None
        """
        if self.fig is not None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            self.fig.text(0.02, 0.09, f'Generated with MPASdiag on: {timestamp}', 
                         fontsize=8, alpha=0.7, transform=self.fig.transFigure)
    
    def format_latitude(self, value: float, _) -> str:
        """
        Format latitude value with N/S direction.

        Parameters:
            value (float): Latitude value in degrees.
            _ : Unused parameter (required by FuncFormatter signature).

        Returns:
            str: Formatted latitude string with N/S suffix (e.g., "45.0°N").
        """
        direction = 'N' if value >= 0 else 'S'
        return f"{abs(value):.1f}°{direction}"

    def format_longitude(self, value: float, _) -> str:
        """
        Format longitude value with E/W direction.

        Parameters:
            value (float): Longitude value in degrees.
            _ : Unused parameter (required by FuncFormatter signature).

        Returns:
            str: Formatted longitude string with E/W suffix (e.g., "120.0°W").
        """
        direction = 'E' if value >= 0 else 'W'
        return f"{abs(value):.1f}°{direction}"
    
    def calculate_adaptive_marker_size(self, map_extent: Tuple[float, float, float, float], 
                                     num_points: int, fig_size: Tuple[float, float] = (12, 10)) -> float:
        """
        Calculate adaptive marker size based on plot extent, data density, and figure size.
        
        Parameters:
            map_extent (Tuple[float, float, float, float]): Map bounds (lon_min, lon_max, lat_min, lat_max)
            num_points (int): Number of data points to plot
            fig_size (Tuple[float, float]): Figure size in inches (width, height)
            
        Returns:
            float: Appropriate marker size for scatter plot
            
        Notes:
            - Calculates marker size based on map extent area and point density.
            - Scales with figure size to maintain consistent visual appearance.
            - Provides reasonable bounds to avoid extremely small or large markers.
        """
        if map_extent is None:
            return 5.0  
        
        lon_min, lon_max, lat_min, lat_max = map_extent
        map_area = (lon_max - lon_min) * (lat_max - lat_min)
        
        if map_area > 0 and num_points > 0:
            point_density = num_points / map_area
        else:
            return 5.0 
        
        if point_density < 1:
            base_size = 8.0   
        elif point_density < 10:
            base_size = 3.0  
        elif point_density < 50:
            base_size = 1.5 
        elif point_density < 150:
            base_size = 0.8   
        elif point_density < 500:
            base_size = 0.4   
        else:
            base_size = 0.25 
        
        if map_area < 50: 
            area_scale = max(8.0, 400.0 / map_area) 
        elif map_area < 500:  
            area_scale = max(3.0, 150.0 / map_area)  
        elif map_area < 5000: 
            area_scale = max(1.5, 50.0 / map_area) 
        else:  
            area_scale = max(0.5, 10.0 / map_area)  
        
        fig_scale = min(1.2, max(0.8, (fig_size[0] * fig_size[1]) / (10 * 12)))
        
        marker_size = base_size * area_scale * fig_scale    
        marker_size = max(0.1, min(20.0, marker_size))
        
        print(f"Adaptive marker sizing: area={map_area:.1f}°², density={point_density:.2f} pts/°², base={base_size:.2f}, scales=({area_scale:.2f}×{fig_scale:.2f}), final_size={marker_size:.2f}")
        
        return marker_size

    def _format_ticks_dynamic(self, ticks: List[float]) -> List[str]:
        """
        Choose a sensible numeric format for tick labels based on value magnitude and type.

        Parameters:
            ticks (list of float): Tick values to format.

        Returns:
            list of str: Formatted tick strings following heuristics for precision.

        Rules:
            - If values are very small (< 1e-3) or very large (>= 1e4), use scientific notation
            - If all values are close to integers, use 0 decimal places
            - Magnitude-based formatting for non-integers:
                * 100-999: 0 decimal places  
                * 10-99: 1 decimal place
                * 1-9: 2 decimal places
                * 0.1-0.99: 2 decimal places
                * 0.01-0.099: 2 decimal places
                * 0.001-0.0099: 3 decimal places
                * Other cases: 2 decimal places
        """
        if not ticks:
            return []

        t = np.array(ticks)        
        non_zero_t = t[t != 0]  

        if len(non_zero_t) > 0:
            max_abs = np.max(np.abs(non_zero_t))
            min_abs = np.min(np.abs(non_zero_t))
            
            if max_abs >= 1e4 or min_abs < 1e-3:
                if max_abs >= 1e4:
                    return [f'{x:.1e}' for x in ticks]
                else:
                    return [f'{x:.1e}' for x in ticks]
        
        if len(t) > 1:
            spacings = np.abs(np.diff(np.sort(t)))
            median_spacing = float(np.median(spacings[spacings > 0])) if np.any(spacings > 0) else 0.0
        else:
            median_spacing = 0.0

        if np.allclose(t, np.round(t), atol=1e-6):
            fmt = '{:.0f}'
        elif len(non_zero_t) > 0:
            typical_magnitude = float(np.median(np.abs(non_zero_t)))
            if typical_magnitude >= 100:
                fmt = '{:.0f}'
            elif 10 <= typical_magnitude < 100:
                fmt = '{:.1f}'
            elif 1 <= typical_magnitude < 10:
                fmt = '{:.2f}'
            elif 0.1 <= typical_magnitude < 1:
                fmt = '{:.2f}'  
            elif 0.01 <= typical_magnitude < 0.1:
                fmt = '{:.2f}'  
            elif typical_magnitude <= 0.01:
                fmt = '{:.3f}'
            else:
                fmt = '{:.2f}'
        else:
            fmt = '{:.0f}'

        formatted_labels = [fmt.format(x) for x in ticks]
        
        if len(set(formatted_labels)) < len(formatted_labels):
            if fmt == '{:.0f}':
                fmt = '{:.1f}'
            elif fmt == '{:.1f}':
                fmt = '{:.2f}'
            elif fmt == '{:.2f}':
                fmt = '{:.3f}'
            formatted_labels = [fmt.format(x) for x in ticks]
            
            if len(set(formatted_labels)) < len(formatted_labels):
                formatted_labels = [f'{x:g}' for x in ticks]
        
        return formatted_labels
    
    def setup_map_projection(self, lon_min: float, lon_max: float, 
                           lat_min: float, lat_max: float,
                           projection: str = 'PlateCarree') -> Tuple[ccrs.Projection, ccrs.PlateCarree]:
        """
        Set up map projection and data coordinate system.

        Parameters:
            lon_min (float): Minimum longitude bound.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Minimum latitude bound.
            lat_max (float): Maximum latitude bound.
            projection (str): Map projection name.

        Returns:
            Tuple[ccrs.Projection, ccrs.PlateCarree]: Map projection and data CRS.
        """
        central_lon = (lon_min + lon_max) / 2
        central_lat = (lat_min + lat_max) / 2
        
        if projection.lower() == 'platecarree':
            map_proj = ccrs.PlateCarree(central_longitude=central_lon)
        elif projection.lower() == 'mercator':
            map_proj = ccrs.Mercator(central_longitude=central_lon)
        elif projection.lower() == 'lambertconformal':
            map_proj = ccrs.LambertConformal(central_longitude=central_lon, central_latitude=central_lat)
        else:
            map_proj = ccrs.PlateCarree(central_longitude=central_lon)
        
        data_crs = ccrs.PlateCarree()
        
        return map_proj, data_crs
    
    def save_plot(self, 
                  output_path: str, 
                  formats: List[str] = ['png'],
                  bbox_inches: str = 'tight',
                  pad_inches: float = 0.1) -> None:
        """
        Save the current plot to file(s).

        Parameters:
            output_path (str): Base output path (without extension).
            formats (List[str]): List of output formats.
            bbox_inches (str): Bounding box mode.
            pad_inches (float): Padding around the figure.

        Returns:
            None
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            full_path = f"{output_path}.{fmt}"
            self.fig.savefig(full_path, dpi=self.dpi, bbox_inches=bbox_inches, 
                           pad_inches=pad_inches, format=fmt)
            print(f"Saved plot: {full_path}")
    
    def close_plot(self) -> None:
        """Close the current figure to free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def create_time_series_plot(self,
                              times: List[datetime],
                              values: List[float],
                              title: str = "Time Series",
                              ylabel: str = "Value",
                              xlabel: str = "Time") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a time series plot.

        Parameters:
            times (list of datetime): Time values.
            values (list of float): Data values.
            title (str): Plot title.
            ylabel (str): Y-axis label.
            xlabel (str): X-axis label.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        self.ax.plot(times, values, linewidth=2, marker='o', markersize=4)
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.autofmt_xdate()
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def create_histogram(self,
                        data: np.ndarray,
                        bins: Union[int, np.ndarray] = 50,
                        title: str = "Data Distribution",
                        xlabel: str = "Value",
                        ylabel: str = "Frequency",
                        log_scale: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a histogram of data values.

        Parameters:
            data (np.ndarray): Data values.
            bins (int or np.ndarray): Number of bins or bin edges.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            log_scale (bool): Use logarithmic scale for y-axis.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) > 0:
            n, bins, patches = self.ax.hist(valid_data, bins=bins, alpha=0.7, 
                                          edgecolor='black', linewidth=0.5)
            
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            self.ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.2f}')
            self.ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1, 
                          label=f'Mean ± Std: {mean_val-std_val:.2f} to {mean_val+std_val:.2f}')
            self.ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1)
            
            self.ax.legend()
        
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        if log_scale:
            self.ax.set_yscale('log')
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax

    @staticmethod
    def validate_plot_parameters(lon_min: float, lon_max: float, 
                               lat_min: float, lat_max: float) -> bool:
        """
        Validate plotting parameters.

        Parameters:
            lon_min (float): Minimum longitude bound.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Minimum latitude bound.
            lat_max (float): Maximum latitude bound.

        Returns:
            bool: True if parameters are valid.
        """
        return (
            -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0
            and -90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0
            and lon_max > lon_min and lat_max > lat_min
        )


class MPASPrecipitationPlotter(MPASVisualizer):
    """
    Specialized class for creating precipitation visualizations from MPAS model output.
    
    This class inherits common functionality from MPASVisualizer and provides
    precipitation-specific methods for creating professional cartographic maps
    with appropriate color schemes and accumulation period handling.
    """
    
    def create_precip_colormap(self, accum: str = "a24h") -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        Create a discrete colormap and contour levels for precipitation plotting.

        Parameters:
            accum (str): Accumulation period (e.g., 'a24h', 'a01h') to determine appropriate levels.

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: Colormap and contour levels.
        """
        colors = [
            "white", "lightskyblue", "dodgerblue", "seagreen", "lawngreen",
            "darkorange", "red", "maroon", "darkviolet", "fuchsia", "violet",
        ]

        accum_key = (accum or "a24h").lower().strip()
        hours = None

        try:
            m = re.search(r"(\d+)", accum_key)
            if m:
                hours = int(m.group(1))
        except Exception:
            hours = None

        if hours is None or hours >= 12:
            levels = [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]
        else:
            levels = [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75]

        cmap = mcolors.ListedColormap(colors)
        return cmap, levels
    
    def create_precipitation_map(self, 
                               lon: np.ndarray, 
                               lat: np.ndarray, 
                               precip_data: np.ndarray,
                               lon_min: float, 
                               lon_max: float, 
                               lat_min: float, 
                               lat_max: float,
                               title: str = "MPAS Precipitation",
                               accum_period: str = "a01h",
                               colormap: Optional[str] = None,
                               levels: Optional[List[float]] = None,
                               clim_min: Optional[float] = None,
                               clim_max: Optional[float] = None,
                               projection: str = 'PlateCarree',
                               time_end: Optional[datetime] = None,
                               time_start: Optional[datetime] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a precipitation map with professional cartographic presentation.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            precip_data (np.ndarray): Precipitation data values.
            lon_min (float): Minimum longitude bound for map extent.
            lon_max (float): Maximum longitude bound for map extent.
            lat_min (float): Minimum latitude bound for map extent.
            lat_max (float): Maximum latitude bound for map extent.
            title (str): Plot title.
            accum_period (str): Accumulation period for colormap selection.
            colormap (Optional[str]): Custom colormap name.
            levels (Optional[List[float]]): Custom contour levels.
            clim_min (Optional[float]): Minimum color limit.
            clim_max (Optional[float]): Maximum color limit.
            projection (str): Map projection.
            time_end (Optional[datetime]): End time for accumulation period.
            time_start (Optional[datetime]): Start time for accumulation period.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)
        
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
        
        if colormap and levels:
            cmap = plt.get_cmap(colormap)
            color_levels = levels
        elif colormap:
            cmap = plt.get_cmap(colormap)
            cmap_obj, color_levels = self.create_precip_colormap(accum_period)
        else:
            cmap, color_levels = self.create_precip_colormap(accum_period)
        
        if clim_min is not None and clim_max is not None:
            color_levels = [level for level in color_levels if clim_min <= level <= clim_max]
            if clim_min not in color_levels:
                color_levels.insert(0, clim_min)
            if clim_max not in color_levels:
                color_levels.append(clim_max)
        
        color_levels_sorted = sorted(set([v for v in color_levels if np.isfinite(v)]))
        last_bound = max(color_levels_sorted) + 1
        bounds = [0] + color_levels_sorted + [last_bound]
        norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        
        precip_data_flat = np.asarray(precip_data).flatten()
        lon_flat = np.asarray(lon).flatten()
        lat_flat = np.asarray(lat).flatten()
        
        min_length = min(len(precip_data_flat), len(lon_flat), len(lat_flat))
        precip_data_flat = precip_data_flat[:min_length]
        lon_flat = lon_flat[:min_length]
        lat_flat = lat_flat[:min_length]
        
        valid_mask = (np.isfinite(precip_data_flat) & 
                     (precip_data_flat >= 0) & 
                     (precip_data_flat < 1e5) &
                     (lon_flat >= lon_min) & (lon_flat <= lon_max) &
                     (lat_flat >= lat_min) & (lat_flat <= lat_max))
        
        if np.any(valid_mask):
            lon_valid = lon_flat[valid_mask]
            lat_valid = lat_flat[valid_mask]
            precip_valid = precip_data_flat[valid_mask]
            
            map_extent = (lon_min, lon_max, lat_min, lat_max)
            fig_size = (self.figsize[0], self.figsize[1])
            marker_size = self.calculate_adaptive_marker_size(map_extent, len(precip_valid), fig_size)
            
            map_area = (lon_max - lon_min) * (lat_max - lat_min)
            point_density = len(precip_valid) / map_area if map_area > 0 else 0
            
            if point_density > 1000:
                alpha_val = 0.9   
            elif point_density > 100:
                alpha_val = 0.9   
            else:
                alpha_val = 0.9  
            
            sort_indices = np.argsort(precip_valid)
            lon_sorted = lon_valid[sort_indices]
            lat_sorted = lat_valid[sort_indices]
            precip_sorted = precip_valid[sort_indices]
            
            scatter = self.ax.scatter(lon_sorted, lat_sorted, c=precip_sorted, 
                                   cmap=cmap, norm=norm, s=marker_size, alpha=alpha_val, 
                                   transform=data_crs, edgecolors='none')
            
            cbar = self.fig.colorbar(scatter, ax=self.ax, orientation='horizontal', extend='both',
                                   pad=0.06, shrink=0.8, aspect=30)
            cbar.set_label('Precipitation (mm)', fontsize=12, fontweight='bold', labelpad=-50)
            cbar.ax.tick_params(labelsize=8)  
            
            if len(color_levels_sorted) <= 15:
                cbar.set_ticks(color_levels_sorted)
                cbar.set_ticklabels(self._format_ticks_dynamic(color_levels_sorted))
        
        gl = self.ax.gridlines(crs=data_crs, draw_labels=True, 
                             linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if time_end is not None:
            if time_start is None:
                n_hours = get_accumulation_hours(accum_period)
                time_start = time_end - pd.Timedelta(hours=n_hours)
            
            start_utc = time_start.strftime('%Y-%m-%d %H:%M UTC')
            end_utc = time_end.strftime('%Y-%m-%d %H:%M UTC')
            n_hours = int((time_end - time_start).total_seconds() / 3600)
            
            txt = f"Accumulation: {start_utc} to {end_utc} ({n_hours} h)"
            self.ax.text(0.01, 0.02, txt, transform=self.ax.transAxes, fontsize=9,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        elif accum_period:
            accum_hours_map = {'a01h': '1-h', 'a03h': '3-h', 'a06h': '6-h', 'a12h': '12-h', 'a24h': '24-h'}
            accum_display = accum_hours_map.get(accum_period, accum_period)
            txt = f"Accumulation: {accum_display}"
            self.ax.text(0.01, 0.02, txt, transform=self.ax.transAxes, fontsize=9,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def create_batch_precipitation_maps(self, processor, output_dir: str,
                                      lon_min: float, lon_max: float,
                                      lat_min: float, lat_max: float,
                                      var_name: str = 'rainnc',
                                      accum_period: str = 'a01h',
                                      file_prefix: str = 'mpas_precipitation_map',
                                      formats: List[str] = ['png']) -> List[str]:
        """
        Create precipitation maps for all time steps in batch mode.

        Parameters:
            processor: MPASDataProcessor instance with loaded data.
            output_dir (str): Output directory for plots.
            lon_min (float): Minimum longitude bound.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Minimum latitude bound.
            lat_max (float): Maximum latitude bound.
            var_name (str): Precipitation variable name.
            accum_period (str): Accumulation period.
            file_prefix (str): Prefix for output filenames.
            formats (List[str]): Output file formats.

        Returns:
            List[str]: List of created file paths.
        """
        if processor.dataset is None:
            raise ValueError("No data loaded in processor")
        
        lon, lat = processor.extract_spatial_coordinates()
        
        time_dim = 'Time' if 'Time' in processor.dataset.dims else 'time'
        total_times = processor.dataset.sizes[time_dim]
        
        from .utils import get_accumulation_hours
        accum_hours = get_accumulation_hours(accum_period)
        min_time_idx = accum_hours  
        
        if min_time_idx >= total_times:
            print(f"\nWarning: Accumulation period {accum_period} ({accum_hours} hours) requires at least {min_time_idx + 1} time steps.")
            print(f"Dataset only has {total_times} time steps. No plots will be generated.")
            return []
        
        actual_time_steps = total_times - min_time_idx
        created_files = []
        
        print(f"\nCreating precipitation maps for {actual_time_steps} time steps (skipping first {min_time_idx} due to {accum_period} accumulation)...")
        
        for time_idx in range(min_time_idx, total_times):
            try:
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                    time_end = pd.to_datetime(processor.dataset.Time.values[time_idx])
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_end = None
                    time_str = f"t{time_idx:03d}"
                
                precip_data = processor.compute_precipitation_difference(time_idx, var_name, accum_period)
                
                title = f"MPAS Precipitation | VarType: {var_name.upper()} | Valid Time: {time_str}"
                fig, ax = self.create_precipitation_map(
                    lon, lat, precip_data.values,
                    lon_min, lon_max, lat_min, lat_max,
                    title=title,
                    accum_period=accum_period,
                    time_end=time_end
                )
                
                output_path = os.path.join(output_dir, f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_point")
                self.save_plot(output_path, formats=formats)
                
                for fmt in formats:
                    created_files.append(f"{output_path}.{fmt}")
                
                self.close_plot()
                
                if (time_idx - min_time_idx + 1) % 10 == 0:
                    print(f"Completed {time_idx - min_time_idx + 1}/{actual_time_steps} maps (time index {time_idx})...")
                    
            except Exception as e:
                print(f"Error creating map for time index {time_idx}: {e}")
                continue
        
        print(f"\nBatch processing completed. Created {len(created_files)} files.")
        return created_files


class MPASSurfacePlotter(MPASVisualizer):
    """
    Specialized class for creating surface variable visualizations from MPAS model output.
    
    This class inherits common functionality from MPASVisualizer and provides
    surface-specific methods for creating scatter and contour plots with
    automatic unit conversion and appropriate color schemes.
    """
    
    def create_surface_map(self,
                         lon: np.ndarray,
                         lat: np.ndarray,
                         data: np.ndarray,
                         var_name: str,
                         lon_min: float,
                         lon_max: float,
                         lat_min: float,
                         lat_max: float,
                         title: Optional[str] = None,
                         plot_type: str = 'scatter',
                         colormap: Optional[str] = None,
                         levels: Optional[List[float]] = None,
                         clim_min: Optional[float] = None,
                         clim_max: Optional[float] = None,
                         projection: str = 'PlateCarree',
                         time_stamp: Optional[datetime] = None,
                         data_array: Optional[xr.DataArray] = None,
                         grid_resolution: Optional[int] = None,
                         grid_resolution_deg: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a 2D surface map for any MPAS forecast variable.

        Supports scatter and contour plots (with interpolation to regular grids)
        for visualizing MPAS surface variables like temperature, humidity, pressure,
        and wind speed.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            data (np.ndarray): Variable data values.
            var_name (str): Variable name for metadata lookup.
            lon_min (float): Minimum longitude bound for map extent.
            lon_max (float): Maximum longitude bound for map extent.
            lat_min (float): Minimum latitude bound for map extent.
            lat_max (float): Maximum latitude bound for map extent.
            title (Optional[str]): Custom plot title (auto-generated if None).
            plot_type (str): 'scatter' or 'contour'.
            colormap (Optional[str]): Custom colormap name.
            levels (Optional[List[float]]): Custom contour levels.
            clim_min (Optional[float]): Minimum color limit.
            clim_max (Optional[float]): Maximum color limit.
            projection (str): Map projection.
            time_stamp (Optional[datetime]): Time stamp for title.
            data_array (Optional[xr.DataArray]): Original data array for metadata extraction.
            grid_resolution (Optional[int]): Grid resolution (points per axis) for interpolation.
            grid_resolution_deg (Optional[float]): Grid resolution in degrees for interpolation.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.

        Raises:
            ValueError: If plot_type is not 'scatter' or 'contour'.
        """
        if plot_type not in ['scatter', 'contour']:
            raise ValueError(f"plot_type must be 'scatter' or 'contour', got '{plot_type}'")
        
        var_metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        
        original_unit = var_metadata.get('original_units', var_metadata['units'])
        display_unit = var_metadata['units']
        
        if original_unit != display_unit:
            try:
                data = UnitConverter.convert_units(data, original_unit, display_unit)
                print(f"Converted {var_name} from {original_unit} to {display_unit}")
            except ValueError as e:
                print(f"Warning: Could not convert {var_name} from {original_unit} to {display_unit}: {e}")
        
        if colormap is None:
            colormap = var_metadata['colormap']
        if levels is None:
            levels = var_metadata.get('levels', None)
        
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)
        
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
        
        if clim_min is not None and clim_max is not None:
            levels = [level for level in levels if clim_min <= level <= clim_max]
            if clim_min not in levels:
                levels.insert(0, clim_min)
            if clim_max not in levels:
                levels.append(clim_max)

        try:
            cmap_obj = plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
        except Exception:
            cmap_obj = plt.get_cmap('viridis')

        norm = None
        if clim_min is not None or clim_max is not None:
            vmin = clim_min if clim_min is not None else float(np.nanmin(data))
            vmax = clim_max if clim_max is not None else float(np.nanmax(data))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            try:
                color_levels_sorted = sorted(set([v for v in levels if np.isfinite(v)]))
                if color_levels_sorted:
                    last_bound = max(color_levels_sorted) + 1
                    bounds = [min(color_levels_sorted)] + color_levels_sorted + [last_bound]
                    norm = BoundaryNorm(bounds, ncolors=cmap_obj.N, clip=True)
            except Exception:
                norm = None
        
        valid_mask = (np.isfinite(data) & 
                     np.isfinite(lon) & np.isfinite(lat) &
                     (lon >= lon_min) & (lon <= lon_max) &
                     (lat >= lat_min) & (lat <= lat_max))
        
        if not np.any(valid_mask):
            print(f"Warning: No valid data points found for {var_name}")
            return self.fig, self.ax
            
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        data_valid = data[valid_mask]
        
        print(f"Plotting {len(data_valid):,} data points for {var_name}")
        print(f"Data range: {data_valid.min():.3f} to {data_valid.max():.3f} {var_metadata['units']}")
        
        if plot_type == 'scatter':
            map_extent = (lon_min, lon_max, lat_min, lat_max)
            fig_size = (self.figsize[0], self.figsize[1])
            marker_size = self.calculate_adaptive_marker_size(map_extent, len(data_valid), fig_size)
            
            map_area = (lon_max - lon_min) * (lat_max - lat_min)
            point_density = len(data_valid) / map_area if map_area > 0 else 0
            
            if point_density > 1000:
                alpha_val = 0.8
            elif point_density > 100:
                alpha_val = 0.9
            else:
                alpha_val = 0.9
            
            sort_indices = np.argsort(data_valid)
            lon_sorted = lon_valid[sort_indices]
            lat_sorted = lat_valid[sort_indices]
            data_sorted = data_valid[sort_indices]
            
            scatter = self.ax.scatter(lon_sorted, lat_sorted, c=data_sorted,
                                   cmap=cmap_obj, norm=norm, s=marker_size, alpha=alpha_val,
                                   transform=data_crs, edgecolors='none')
            
            cbar = self.fig.colorbar(scatter, ax=self.ax, orientation='horizontal', extend='both',
                                   pad=0.06, shrink=0.8, aspect=30)
            
            try:
                ticks = cbar.get_ticks().tolist()
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
                cbar.ax.tick_params(labelsize=8)  
            except Exception:
                pass
            
        elif plot_type == 'contour':            
            try:
                from scipy.interpolate import griddata

                if grid_resolution_deg is not None:
                    step = float(grid_resolution_deg)
                    if step <= 0:
                        raise ValueError("grid_resolution_deg must be > 0")

                    lon_coords = np.arange(lon_min, lon_max + 1e-12, step)
                    lat_coords = np.arange(lat_min, lat_max + 1e-12, step)

                    nx = len(lon_coords)
                    ny = len(lat_coords)

                    max_points = 1200
                    if nx > max_points or ny > max_points:
                        lon_coords = np.linspace(lon_min, lon_max, min(nx, max_points))
                        lat_coords = np.linspace(lat_min, lat_max, min(ny, max_points))
                        print(f"Requested degree step {step}° produces >{max_points} points per axis; clipping to {len(lat_coords)}x{len(lon_coords)} grid")

                    lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
                    print(f"Interpolating {len(data_valid)} points to {lon_mesh.shape[0]}x{lon_mesh.shape[1]} grid (~{step}° resolution)...")

                else:
                    if grid_resolution is None:
                        adaptive = int(np.sqrt(len(data_valid)) / 9)
                        grid_resolution = int(np.clip(adaptive, 50, 200))
                    else:
                        grid_resolution = int(grid_resolution)

                    grid_resolution = max(40, min(400, grid_resolution))

                    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
                    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
                    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

                    print(f"Interpolating {len(data_valid)} points to {grid_resolution}x{grid_resolution} grid...")

                data_interp = griddata(
                    (lon_valid, lat_valid), data_valid,
                    (lon_mesh, lat_mesh), method='linear', fill_value=np.nan
                )
                
                if levels is not None:
                    contourf = self.ax.contourf(lon_mesh, lat_mesh, data_interp, levels=levels,
                                             cmap=cmap_obj, norm=norm, transform=data_crs, extend='both')
                    
                    contour_lines = self.ax.contour(lon_mesh, lat_mesh, data_interp, levels=levels,
                                                  colors='black', linewidths=0.5, alpha=0.3,
                                                  transform=data_crs)
                else:
                    contourf = self.ax.contourf(lon_mesh, lat_mesh, data_interp,
                                             cmap=cmap_obj, norm=norm, transform=data_crs, extend='both')
                    
                    contour_lines = self.ax.contour(lon_mesh, lat_mesh, data_interp,
                                                  colors='black', linewidths=0.5, alpha=0.3,
                                                  transform=data_crs)
                
                cbar = self.fig.colorbar(contourf, ax=self.ax, orientation='horizontal', extend='both',
                                       pad=0.06, shrink=0.8, aspect=30)
                try:
                    if levels is not None and len(levels) > 0:
                        tick_levels = list(levels)
                        cbar.set_ticks(tick_levels)
                        cbar.set_ticklabels(self._format_ticks_dynamic(tick_levels))
                    else:
                        ticks = cbar.get_ticks().tolist()
                        cbar.set_ticks(ticks)
                        cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
                    cbar.ax.tick_params(labelsize=8)  
                except Exception:
                    pass
                
            except ImportError:
                print("Warning: scipy not available for contour plotting. Falling back to scatter plot.")
                return self.create_surface_map(lon, lat, data, var_name, lon_min, lon_max, 
                                             lat_min, lat_max, title, 'scatter', colormap, 
                                             levels, clim_min, clim_max, projection, time_stamp, data_array)
        
        units_str = f" ({var_metadata['units']})" if var_metadata['units'] else ""
        label_text = UnitConverter._format_colorbar_label(f"{var_metadata['long_name']}{units_str}")
        cbar.set_label(label_text, fontsize=12, fontweight='bold', labelpad=-50)
        cbar.ax.tick_params(labelsize=8)  
        
        if plot_type == 'scatter' and levels is not None and len(levels) > 0:
            tick_levels = list(levels)
            cbar.set_ticks(tick_levels)
            cbar.set_ticklabels(self._format_ticks_dynamic(tick_levels))
        
        gl = self.ax.gridlines(crs=data_crs, draw_labels=True,
                             linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        time_in_title = False

        if title is None:
            title = f"MPAS {var_metadata['long_name']}"
            if time_stamp:
                time_str = time_stamp.strftime('%Y%m%dT%H')
                title += f" | Valid Time: {time_str}"
                time_in_title = True
            title += f" | Plot Type: {plot_type.title()}"
        else:
            if time_stamp:
                time_str = time_stamp.strftime('%Y%m%dT%H')
                time_in_title = (time_str in title or 'Valid Time:' in title or 'Valid:' in title)
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if time_stamp and not time_in_title:
            time_str = time_stamp.strftime('%Y%m%dT%H')
            self.ax.text(0.02, 0.98, f'Valid: {time_str}', 
                        transform=self.ax.transAxes, 
                        fontsize=12, fontweight='bold',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def create_simple_scatter_plot(self,
                                 lon: np.ndarray,
                                 lat: np.ndarray, 
                                 data: np.ndarray,
                                 title: str = "MPAS Data",
                                 colorbar_label: str = "Value",
                                 colormap: str = 'viridis',
                                 point_size: float = 1.0) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a simple scatter plot of MPAS data.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            data (np.ndarray): Data values.
            title (str): Plot title.
            colorbar_label (str): Colorbar label.
            colormap (str): Colormap name.
            point_size (float): Size of scatter points.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        valid_mask = np.isfinite(data) & np.isfinite(lon) & np.isfinite(lat)
        
        if np.any(valid_mask):
            scatter = self.ax.scatter(lon[valid_mask], lat[valid_mask], 
                                   c=data[valid_mask], cmap=colormap, 
                                   s=point_size, alpha=0.7, edgecolors='none')
            
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            try:
                ticks = cbar.get_ticks().tolist()
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
                cbar.ax.tick_params(labelsize=8)  
            except Exception:
                pass
            cbar.set_label(UnitConverter._format_colorbar_label(colorbar_label), fontsize=12)
        
        self.ax.set_xlabel('Longitude', fontsize=12)
        self.ax.set_ylabel('Latitude', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax

    def create_batch_surface_maps(self, processor, output_dir: str,
                                 lon_min: float, lon_max: float,
                                 lat_min: float, lat_max: float,
                                 var_name: str = 't2m',
                                 plot_type: str = 'scatter',
                                 file_prefix: str = 'mpas_surface',
                                 formats: List[str] = ['png'],
                                 grid_resolution: Optional[int] = None,
                                 grid_resolution_deg: Optional[float] = None,
                                 clim_min: Optional[float] = None,
                                 clim_max: Optional[float] = None) -> List[str]:
        """
        Create surface variable maps for all time steps in batch mode.

        Parameters:
            processor: MPASDataProcessor instance with loaded data.
            output_dir (str): Output directory for plots.
            lon_min (float): Minimum longitude bound.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Minimum latitude bound.
            lat_max (float): Maximum latitude bound.
            var_name (str): Variable name to plot.
            plot_type (str): 'scatter' or 'contour'.
            file_prefix (str): Prefix for output filenames.
            formats (List[str]): Output formats.
            grid_resolution (Optional[int]): Grid resolution for interpolation.
            grid_resolution_deg (Optional[float]): Grid resolution in degrees.
            clim_min (Optional[float]): Minimum color limit.
            clim_max (Optional[float]): Maximum color limit.

        Returns:
            List[str]: List of created file paths.
        """
        if processor.dataset is None:
            raise ValueError("No data loaded in processor")

        time_dim = 'Time' if 'Time' in processor.dataset.dims else 'time'
        total_times = processor.dataset.sizes[time_dim]

        created_files = []
        print(f"\nCreating surface maps for {total_times} time steps...")

        for time_idx in range(total_times):
            try:
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                    time_end = pd.to_datetime(processor.dataset.Time.values[time_idx])
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_end = None
                    time_str = f"t{time_idx:03d}"

                var_data = processor.get_variable_data(var_name, time_idx)
                
                lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)

                title = f"MPAS {var_name.upper()} | Valid: {time_str} | {plot_type.title()}"

                fig, ax = self.create_surface_map(
                    lon, lat, var_data.values, var_name,
                    lon_min, lon_max, lat_min, lat_max,
                    title=title,
                    plot_type=plot_type,
                    time_stamp=time_end,
                    data_array=var_data,
                    grid_resolution=grid_resolution,
                    grid_resolution_deg=grid_resolution_deg,
                    clim_min=clim_min,
                    clim_max=clim_max
                )

                output_path = os.path.join(output_dir, f"{file_prefix}_{var_name}_{plot_type}_{time_str}")
                self.save_plot(output_path, formats=formats)

                for fmt in formats:
                    created_files.append(f"{output_path}.{fmt}")

                self.close_plot()

                if (time_idx + 1) % 10 == 0:
                    print(f"Completed {time_idx + 1}/{total_times} surface maps...")

            except Exception as e:
                print(f"Error creating surface map for time index {time_idx}: {e}")
                continue

        print(f"\nBatch processing completed. Created {len(created_files)} files.")
        return created_files

    def get_surface_colormap_and_levels(self, var_name: str, data_array: Optional[xr.DataArray] = None) -> Tuple[str, List[float]]:
        """
        Get appropriate colormap and levels for 2D surface variables.

        Parameters:
            var_name (str): 2D surface variable name.
            data_array (Optional[xr.DataArray]): Data array for automatic level detection.

        Returns:
            Tuple[str, List[float]]: Colormap name and contour levels.
        """
        metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        return metadata['colormap'], metadata['levels']


class MPASWindPlotter(MPASVisualizer):
    """
    Specialized class for creating wind vector visualizations from MPAS model output.
    
    This class inherits common functionality from MPASVisualizer and provides
    wind-specific methods for creating barb and arrow plots with automatic
    subsampling and background wind speed visualization.
    """
    
    def _choose_wind_subsample(self, lon: np.ndarray, lat: np.ndarray, plot_type: str = 'barbs', max_vectors: Optional[int] = None) -> int:
        """
        Choose an automatic subsample factor for wind plotting to limit plotted vectors.

        Parameters:
            lon (np.ndarray): Input longitude array (1D or 2D).
            lat (np.ndarray): Input latitude array (1D or 2D).
            plot_type (str): Plot type, e.g., 'barbs' or 'arrows'.
            max_vectors (Optional[int]): Maximum number of wind vectors to plot.

        Returns:
            int: Subsample step (>=1). A value of 1 means plot all vectors.
        """
        if max_vectors is None:
            if (plot_type or '').lower() == 'barbs':
                max_vectors = 30  
            else:
                max_vectors = 50  

        try:
            n_points = int(np.size(lon))
        except Exception:
            n_points = int(np.size(lat)) if lat is not None else 0

        if n_points <= 0:
            return 1

        if n_points <= max_vectors:
            return 1

        step = int(np.ceil(np.sqrt(n_points / max_vectors)))
        step = max(1, step)

        if n_points > max_vectors * 25:
            step = int(step * 2)

        print(f"Auto wind subsample: n_points={n_points}, max_vectors={max_vectors}, step={step}")
        return step
    
    def create_wind_plot(self, lon: np.ndarray, lat: np.ndarray, 
                        u_data: np.ndarray, v_data: np.ndarray,
                        lon_min: float, lon_max: float, 
                        lat_min: float, lat_max: float,
                        wind_level: str = "surface",
                        plot_type: str = "barbs",
                        subsample: int = 0,
                        scale: Optional[float] = None,
                        show_background: bool = False,
                        bg_colormap: str = "viridis",
                        title: Optional[str] = None,
                        time_stamp: Optional[object] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a wind vector plot using MPAS data.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            u_data (np.ndarray): U-component wind data.
            v_data (np.ndarray): V-component wind data.
            lon_min (float): Minimum longitude for extent.
            lon_max (float): Maximum longitude for extent.
            lat_min (float): Minimum latitude for extent.
            lat_max (float): Maximum latitude for extent.
            wind_level (str): Wind level description for labeling (default: 'surface').
            plot_type (str): 'barbs' or 'arrows' (default: 'barbs').
            subsample (int): Subsample factor for wind vectors (<=0 means auto).
            scale (Optional[float]): Scale factor for wind vectors (auto if None).
            show_background (bool): Show background wind speed contours.
            bg_colormap (str): Colormap for background.
            title (Optional[str]): Custom plot title.
            time_stamp (Optional[object]): Time stamp used for title formatting.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        import pandas as pd
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        wind_speed = np.sqrt(u_data**2 + v_data**2)

        is_gridded = all(arr is not None and getattr(arr, 'ndim', 1) == 2 for arr in (lon, lat, u_data, v_data))

        if show_background:
            if is_gridded:
                levels = np.linspace(0, np.percentile(wind_speed, 95), 15)
                cs = ax.contourf(lon, lat, wind_speed, levels=levels,
                                 cmap=bg_colormap, alpha=0.6, transform=ccrs.PlateCarree())

                cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
            else:
                lon_flat = np.ravel(lon)
                lat_flat = np.ravel(lat)
                wind_flat = np.ravel(wind_speed)

                sc = ax.scatter(lon_flat, lat_flat, c=wind_flat, cmap=bg_colormap,
                                alpha=0.6, s=12, transform=ccrs.PlateCarree(), edgecolors='none')
                cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)

            try:
                ticks = cbar.get_ticks().tolist()
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
                cbar.ax.tick_params(labelsize=8)  
            except Exception:
                pass
            cbar.set_label(UnitConverter._format_colorbar_label('wind speed (m/s)'), fontsize=10)
        
        if subsample <= 0:
            subsample = self._choose_wind_subsample(lon, lat, plot_type=plot_type)
        subsample = max(1, int(subsample))

        if is_gridded:
            if subsample > 1:
                lon_sub = lon[::subsample, ::subsample]
                lat_sub = lat[::subsample, ::subsample]
                u_sub = u_data[::subsample, ::subsample]
                v_sub = v_data[::subsample, ::subsample]
            else:
                lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
        else:
            lon_flat = np.ravel(lon)
            lat_flat = np.ravel(lat)
            u_flat = np.ravel(u_data)
            v_flat = np.ravel(v_data)

            if subsample > 1:
                lon_sub = lon_flat[::subsample]
                lat_sub = lat_flat[::subsample]
                u_sub = u_flat[::subsample]
                v_sub = v_flat[::subsample]
            else:
                lon_sub, lat_sub, u_sub, v_sub = lon_flat, lat_flat, u_flat, v_flat
        
        if plot_type.lower() == "barbs":
            ax.barbs(lon_sub, lat_sub, u_sub, v_sub, 
                    length=6, barbcolor='black', flagcolor='red',
                    linewidth=0.8, transform=ccrs.PlateCarree())
        elif plot_type.lower() == "arrows":
            if scale is None:
                scale = float(np.percentile(wind_speed, 90)) * 15
                
            ax.quiver(lon_sub, lat_sub, u_sub, v_sub,
                     scale=scale, scale_units='xy', angles='xy',
                     color='black', width=0.003, alpha=0.8,
                     transform=ccrs.PlateCarree())
        
        if title is None:
            if time_stamp is not None:
                try:
                    ts = pd.to_datetime(time_stamp)
                    time_str = ts.strftime('%Y%m%dT%H')
                    title = f"MPAS {wind_level.title()} Wind Vectors | Valid: {time_str}"
                except Exception:
                    title = f"MPAS {wind_level.title()} Wind Vectors | Valid: {time_stamp}"
            else:
                title = f"MPAS {wind_level.title()} Wind Vectors"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        wind_stats = f"""Wind Statistics:
    Max: {wind_speed.max():.1f} m/s
    Mean: {wind_speed.mean():.1f} m/s
    Min: {wind_speed.min():.1f} m/s"""
        
        ax.text(0.02, 0.98, wind_stats, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
        
        if plot_type.lower() == "barbs":
            legend_text = """Wind Barbs:
    Half barb = 2.5 m/s (5 kt)
    Full barb = 5 m/s (10 kt)  
    Flag = 25 m/s (50 kt)"""
        else:
            legend_text = f"Wind Arrows\nScale: {scale:.0f} units"
        
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.fig = fig
        self.ax = ax
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        return fig, ax

    def create_batch_wind_plots(self, processor, output_dir: str,
                               lon_min: float, lon_max: float,
                               lat_min: float, lat_max: float,
                               u_variable: str = 'u10',
                               v_variable: str = 'v10',
                               plot_type: str = 'barbs',
                               file_prefix: str = 'mpas_wind',
                               formats: List[str] = ['png'],
                               subsample: int = 0,
                               scale: Optional[float] = None,
                               show_background: bool = False,
                               background_colormap: str = 'viridis') -> List[str]:
        """
        Create wind vector plots for all time steps in batch mode.

        Parameters:
            processor: MPASDataProcessor instance with loaded data.
            output_dir (str): Output directory for plots.
            lon_min (float): Minimum longitude bound.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Minimum latitude bound.
            lat_max (float): Maximum latitude bound.
            u_variable (str): U-component variable name.
            v_variable (str): V-component variable name.
            plot_type (str): 'barbs' or 'arrows'.
            file_prefix (str): Prefix for output filenames.
            formats (List[str]): Output formats.
            subsample (int): Subsample factor for wind vectors (<=0 means auto).
            scale (Optional[float]): Scale for quiver arrows.
            show_background (bool): Whether to show wind speed background.
            background_colormap (str): Colormap for background wind speed.

        Returns:
            List[str]: List of created file paths.
        """
        if processor.dataset is None:
            raise ValueError("No data loaded in processor")

        lon, lat = processor.extract_spatial_coordinates()

        time_dim = 'Time' if 'Time' in processor.dataset.dims else 'time'
        total_times = processor.dataset.sizes[time_dim]

        created_files = []
        print(f"\nCreating wind plots for {total_times} time steps...")

        for time_idx in range(total_times):
            try:
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                    time_end = pd.to_datetime(processor.dataset.Time.values[time_idx])
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_end = None
                    time_str = f"t{time_idx:03d}"

                u_data, v_data = processor.get_wind_components(u_variable, v_variable, time_idx)

                title = f"MPAS {plot_type.title()} Wind | Valid: {time_str}"

                fig, ax = self.create_wind_plot(
                    lon, lat, u_data.values, v_data.values,
                    lon_min, lon_max, lat_min, lat_max,
                    wind_level='surface',
                    plot_type=plot_type,
                    title=title,
                    subsample=subsample,
                    scale=scale,
                    show_background=show_background,
                    bg_colormap=background_colormap,
                    time_stamp=time_end
                )

                output_path = os.path.join(output_dir, f"{file_prefix}_{plot_type}_{time_str}")
                self.save_plot(output_path, formats=formats)

                for fmt in formats:
                    created_files.append(f"{output_path}.{fmt}")

                self.close_plot()

                if (time_idx + 1) % 10 == 0:
                    print(f"Completed {time_idx + 1}/{total_times} plots...")

            except Exception as e:
                print(f"Error creating wind plot for time index {time_idx}: {e}")
                continue

        print(f"\nBatch processing completed. Created {len(created_files)} files.")
        return created_files


# Backward compatibility: Export functions at module level for direct import
get_2d_variable_metadata = MPASFileMetadata.get_2d_variable_metadata
get_3d_variable_metadata = MPASFileMetadata.get_3d_variable_metadata
convert_units = UnitConverter.convert_units
convert_data_for_display = UnitConverter.convert_data_for_display
get_display_units = UnitConverter.get_display_units
_normalize_unit_string = UnitConverter._normalize_unit_string
_format_colorbar_label = UnitConverter._format_colorbar_label
validate_plot_parameters = MPASVisualizer.validate_plot_parameters






