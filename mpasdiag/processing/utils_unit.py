#!/usr/bin/env python3

"""
MPAS Unit Conversion Utilities

This module provides comprehensive unit conversion functionality for meteorological and atmospheric data following standard meteorological conventions used in MPAS model diagnostics. It supports conversion between common atmospheric units including temperature scales (Kelvin, Celsius, Fahrenheit), pressure units (Pascal, hectopascal, millibar), precipitation rates, wind speeds, mixing ratios, and distances. The conversion methods handle numpy arrays, xarray DataArrays, and scalar values while preserving input data structure and type, with automatic unit string normalization to accommodate various notation styles. The UnitConverter class provides static methods for bidirectional conversions, preferred display unit selection for human-readable output, and specialized workflows for preparing data for visualization. This utility module ensures consistent and accurate unit handling across all MPASdiag processing and visualization workflows.

Classes:
    UnitConverter: Static utility class providing comprehensive meteorological unit conversion methods with support for multiple data types and unit notation variants.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, Dict, Any

from .constants import (
    MM_PER_HR, MM_PER_DAY, DBZ, KELVIN, M_PER_S, PA, PERCENT,
    METER, MILES_PER_HR, KG_PER_KG, KG_PER_M3, MICRONS, PER_KG,
    M3_PER_M3, G_PER_KG, KM_PER_HR, NOUNIT, HPA, MB, KNOT, FEET, 
    CELSIUS, FAHRENHEIT, KILOMETER, INCHES_PER_HR
)


class UnitConverter:
    """
    Unit conversion utility class for meteorological and atmospheric data with comprehensive conversion support following meteorological conventions. This class provides static methods for converting between common atmospheric units including temperature scales (Kelvin, Celsius, Fahrenheit), pressure units (Pascal, hectopascal, millibar), precipitation rates, wind speeds, mixing ratios, and distances. The conversion functionality supports numpy arrays, xarray DataArrays, and scalar values while preserving input data structure and type. Methods include automatic unit string normalization to handle various notation styles, preferred display unit selection for human-readable output, and specialized conversion workflows for preparing data for visualization with appropriate units and metadata.
    """
    
    @staticmethod
    def convert_units(data: Union[np.ndarray, xr.DataArray, float], 
                      from_unit: str, 
                      to_unit: str) -> Union[np.ndarray, xr.DataArray, float]:
        """
        Convert numerical data between common meteorological units with automatic handling of multiple data types and unit notation variants. This method accepts data as numpy arrays, xarray DataArrays, or scalar floats and returns the converted result preserving the original data structure and type when possible. The conversion logic supports temperature scales (K, °C, °F), pressure units (Pa, hPa, mb), mixing ratios (kg/kg, g/kg), precipitation rates (mm/hr, mm/day, in/hr), wind speeds (m/s, kt, mph, km/h), and distance units (m, km, ft) with bidirectional conversion mappings. Unit strings are automatically normalized to canonical forms to handle common notation variations like 'Kelvin' → 'K' or 'degC' → '°C', enabling flexible user input while maintaining consistent internal representations.

        Parameters:
            data (Union[np.ndarray, xr.DataArray, float]): Numeric data to convert, can be scalar, numpy array, or xarray DataArray with values to transform.
            from_unit (str): Source unit string using canonical or common synonym notation (e.g., 'K', 'kelvin', 'Pa', 'hPa').
            to_unit (str): Target unit string for conversion output using canonical or synonym notation.

        Returns:
            Union[np.ndarray, xr.DataArray, float]: Converted data in the same structure and type as input (scalars remain scalars, arrays remain arrays, DataArrays remain DataArrays).

        Raises:
            ValueError: If the requested conversion pair (from_unit, to_unit) is not supported in the internal conversion map.
        """
        from_unit = UnitConverter._normalize_unit_string(from_unit)
        to_unit = UnitConverter._normalize_unit_string(to_unit)
        
        if from_unit == to_unit:
            return data
        
        conversion_map = {
        (KELVIN, CELSIUS): lambda x: x - 273.15,
        (CELSIUS, KELVIN): lambda x: x + 273.15,
        (KELVIN, FAHRENHEIT): lambda x: (x - 273.15) * 9/5 + 32,
        (FAHRENHEIT, KELVIN): lambda x: (x - 32) * 5/9 + 273.15,
        (CELSIUS, FAHRENHEIT): lambda x: x * 9/5 + 32,
        (FAHRENHEIT, CELSIUS): lambda x: (x - 32) * 5/9,
        
        (PA, HPA): lambda x: x / 100.0,
        (HPA, PA): lambda x: x * 100.0,
        (PA, MB): lambda x: x / 100.0,
        (MB, PA): lambda x: x * 100.0,
        (HPA, MB): lambda x: x,  
        (MB, HPA): lambda x: x,  
        
        (KG_PER_KG, G_PER_KG): lambda x: x * 1000.0,
        (G_PER_KG, KG_PER_KG): lambda x: x / 1000.0,
        
        (MM_PER_HR, MM_PER_DAY): lambda x: x * 24.0,
        (MM_PER_DAY, MM_PER_HR): lambda x: x / 24.0,
        (MM_PER_HR, INCHES_PER_HR): lambda x: x / 25.4,
        (INCHES_PER_HR, MM_PER_HR): lambda x: x * 25.4,
        
        (M_PER_S, KNOT): lambda x: x * 1.94384,
        (KNOT, M_PER_S): lambda x: x / 1.94384,
        (M_PER_S, MILES_PER_HR): lambda x: x * 2.23694,
        (MILES_PER_HR, M_PER_S): lambda x: x / 2.23694,
        (M_PER_S, KM_PER_HR): lambda x: x * 3.6,
        (KM_PER_HR, M_PER_S): lambda x: x / 3.6,
        
        (METER, KILOMETER): lambda x: x / 1000.0,
        (KILOMETER, METER): lambda x: x * 1000.0,
        (METER, FEET): lambda x: x * 3.28084,
        (FEET, METER): lambda x: x / 3.28084,
        }
        
        conversion_key = (from_unit, to_unit)

        if conversion_key not in conversion_map:
            raise ValueError(f"Conversion from '{from_unit}' to '{to_unit}' is not supported.\n"
                            f"Supported conversions: {list(conversion_map.keys())}")
        
        converter = conversion_map[conversion_key]
        return converter(data)
    
    @staticmethod
    def convert_data_for_display(data: xr.DataArray, var_name: str, metadata_source: xr.DataArray) -> Tuple[Union[np.ndarray, xr.DataArray, float], Dict[str, Any]]:
        """
        Prepare data array and metadata for human-readable visualization by converting to preferred display units based on variable type. This method consults MPAS metadata registry to determine optimal display units for specific variables (e.g., converting Kelvin to Celsius for temperature variables, Pascal to hectopascal for pressure), automatically performing the conversion when possible and preserving original units when conversion fails or is unavailable. The method extracts unit information from the input data's attributes or the metadata source, attempts conversion to the preferred display unit using convert_units(), and returns both the converted data and an updated metadata dictionary containing both current and original unit information. This workflow is essential for creating properly labeled and intuitively scaled plots where temperature appears in Celsius rather than Kelvin and pressure in hPa rather than Pa for scientific audiences.

        Parameters:
            data (xr.DataArray): Original data array to convert, expected to have optional 'units' attribute in data.attrs for unit detection.
            var_name (str): MPAS variable name used to lookup metadata and display unit preferences from registry (e.g., 't2m', 'mslp', 'rainnc').
            metadata_source (xr.DataArray): Data array or dataset variable used to extract fallback attributes like units or long_name when not present in data.

        Returns:
            Tuple[Union[np.ndarray, xr.DataArray, float], Dict[str, Any]]: Two-element tuple containing (converted_data, metadata_dict) where converted_data has values in preferred display units and metadata_dict includes 'units' (current), 'original_units', and other variable properties.
        """
        from ..visualization import MPASFileMetadata
        
        metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, metadata_source)
        original_units = data.attrs.get('units', metadata.get('units', ''))
        display_units = original_units

        if var_name in ['t2m', 'temperature', 'temp'] and original_units == KELVIN:
            display_units = CELSIUS
        elif var_name in ['mslp', 'pressure'] and original_units == PA:
            display_units = HPA 
            
        converted_data = data

        if original_units != display_units:
            try:
                converted_data = UnitConverter.convert_units(data, original_units, display_units)
            except ValueError:
                display_units = original_units
        
        result_metadata = metadata.copy()
        result_metadata['units'] = display_units
        result_metadata['original_units'] = original_units
        
        return converted_data, result_metadata

    @staticmethod
    def _normalize_unit_string(unit: str) -> str:
        """
        Normalize unit strings to canonical short forms used consistently throughout MPASdiag for reliable unit matching and conversion operations. This helper method maps a variety of verbose or alternative unit representations (e.g., 'kelvin', 'degC', 'mm hr-1', 'hectopascal') to standardized short forms ('K', '°C', 'mm/hr', 'hPa') used internally by the UnitConverter class. The normalization supports case-insensitive matching and handles common notation variants including spelled-out names, abbreviated forms, and various separator styles for compound units. If no canonical mapping is found in the internal dictionary, the method conservatively returns the original string unchanged to avoid breaking conversions for units not in the predefined map.

        Parameters:
            unit (str): Unit string to normalize, can use verbose names, abbreviations, or various notation styles (e.g., 'Kelvin', 'degC', 'mm/hr', 'kg kg-1').

        Returns:
            str: Canonicalized unit string in standard MPASdiag notation (e.g., 'K', '°C', 'mm/hr', 'kg/kg'), or original string if no mapping exists.
        """
        unit = unit.strip()
        
        unit_map = {
        'kelvin': KELVIN, 'k': KELVIN,
        'celsius': CELSIUS, 'degc': CELSIUS, 'deg_c': CELSIUS, 'c': CELSIUS,
        'fahrenheit': FAHRENHEIT, 'degf': FAHRENHEIT, 'deg_f': FAHRENHEIT, 'f': FAHRENHEIT,
        
        'pascal': PA, 'pa': PA,
        'hectopascal': HPA, 'hpa': HPA,
        'millibar': MB, 'mbar': MB,
        
        'kg kg-1': KG_PER_KG, 'kg_kg-1': KG_PER_KG, 'kg kg^{-1}': KG_PER_KG,
        'g kg-1': G_PER_KG, 'g_kg-1': G_PER_KG, 'g kg^{-1}': G_PER_KG,
        
        'mm hr-1': MM_PER_HR, 'mm_hr-1': MM_PER_HR,
        'mm day-1': MM_PER_DAY, 'mm_day-1': MM_PER_DAY,
        
        'percent': PERCENT, 'pct': PERCENT,
        'in hr-1': INCHES_PER_HR, 'in_hr-1': INCHES_PER_HR,
        
        'knots': KNOT, 'knot': KNOT, 'kts': KNOT,
        'miles per hour': MILES_PER_HR, 'mi/hr': MILES_PER_HR,
        'kilometers per hour': KM_PER_HR, 'km hr-1': KM_PER_HR, 'km_hr-1': KM_PER_HR,
        
        'meter': METER, 'meters': METER,
        'kilometer': KILOMETER, 'kilometers': KILOMETER,
        'foot': FEET, 'feet': FEET,
        }
        
        return unit_map.get(unit.lower(), unit)

    @staticmethod
    def get_display_units(variable_name: str, current_unit: str) -> str:
        """
        Determine the preferred human-friendly display unit for MPAS variables based on meteorological conventions and visualization best practices. This method looks up the variable name in an internal preference dictionary mapping common MPAS variables to their ideal display units (e.g., temperature in °C, pressure in hPa, precipitation in mm/hr, wind speed in m/s), then verifies that a valid conversion exists from the current unit to the preferred unit before returning it. The method performs case-insensitive matching and supports both exact variable name matches and substring pattern matching to handle variable naming variants. If no preference is defined or the conversion is not supported, the method returns the original current_unit unchanged to ensure safe fallback behavior in all cases.

        Parameters:
            variable_name (str): MPAS variable name to look up display unit preference (e.g., 't2m', 'rainnc', 'u10', 'mslp').
            current_unit (str): Current unit string for the variable, will be normalized before preference lookup and conversion validation.

        Returns:
            str: Preferred display unit for visualization (e.g., '°C' for temperatures, 'hPa' for pressure), or original current_unit if no preference exists or conversion is unavailable.
        """
        current_unit = UnitConverter._normalize_unit_string(current_unit)
            
        display_unit_preferences = {
            't2m': CELSIUS, 'temperature': CELSIUS, 'temp': CELSIUS, 'theta': CELSIUS,
            'tsk': CELSIUS, 'sst': CELSIUS, 'meanT': CELSIUS, 'theta_base': CELSIUS, 'tslb': CELSIUS,
            'dewpoint': CELSIUS, 'dewpt': CELSIUS, 'dpt': CELSIUS,
            
            'pressure': HPA, 'slp': HPA, 'psfc': HPA, 'pressure_p': HPA,
            'mslp': HPA, 'pmsl': HPA, 'pressure_base': HPA,
            
            'rainnc': MM_PER_HR, 'rainc': MM_PER_HR, 'rain': MM_PER_HR,
            'precipitation': MM_PER_HR, 'precip': MM_PER_HR,
            
            'u10': M_PER_S, 'v10': M_PER_S, 'wspd10': M_PER_S, 'u': M_PER_S, 'w': M_PER_S,
            'wind_speed': M_PER_S, 'wind': M_PER_S, 'uReconstructZonal': M_PER_S, 'uReconstructMeridional': M_PER_S,
            
            'qv': G_PER_KG, 'qc': G_PER_KG, 'qr': G_PER_KG, 'qi': G_PER_KG, 'qs': G_PER_KG, 'qg': G_PER_KG,
            'humidity': G_PER_KG, 'mixing_ratio': G_PER_KG, 'q2': G_PER_KG, 'qv2m': G_PER_KG,
            'relhum': PERCENT,
            
            'rho': KG_PER_M3, 'rho_base': KG_PER_M3,
            
            'refl10cm': DBZ,
            
            'ni': PER_KG, 're_cloud': MICRONS, 're_ice': MICRONS, 're_snow': MICRONS,
            'cldfrac': NOUNIT,
            
            'sh2o': M3_PER_M3, 'smois': M3_PER_M3,
        }
        
        var_name_lower = variable_name.lower()

        for var_pattern, preferred_unit in display_unit_preferences.items():
            if var_pattern.lower() == var_name_lower:
                if current_unit != preferred_unit:
                    try:
                        UnitConverter.convert_units(1.0, current_unit, preferred_unit)
                        return preferred_unit
                    except ValueError:
                        pass
                return preferred_unit
        
        for var_pattern, preferred_unit in display_unit_preferences.items():
            if len(var_pattern) > 1 and var_pattern.lower() in var_name_lower:
                if current_unit != preferred_unit:
                    try:
                        UnitConverter.convert_units(1.0, current_unit, preferred_unit)
                        return preferred_unit
                    except ValueError:
                        pass
                return preferred_unit
        
        return current_unit

    @staticmethod
    def _format_colorbar_label(label: str) -> str:
        """
        Clean and standardize colorbar label text by replacing verbose unit tokens with proper symbols for professional plot appearance. This helper method applies string substitutions to convert various informal or verbose unit representations commonly found in MPAS metadata (e.g., 'deg_C', 'degrees C', 'degC') to standardized mathematical symbols ('°C') suitable for publication-quality figures. The method handles multiple notation variants for degree Celsius to ensure consistent labeling regardless of the input metadata format. This text cleaning step is typically applied to colorbar labels, axis labels, and plot titles before rendering to improve readability and maintain consistent typography across all MPASdiag visualizations.

        Parameters:
            label (str): Input label text that may contain verbose or informal unit tokens like 'deg_C', 'degrees C', 'degrees_C', or 'degC'.

        Returns:
            str: Cleaned label string with standardized unit symbols such as '°C' replacing all verbose temperature unit representations.
        """
        label = label.replace('deg_C', CELSIUS)
        label = label.replace('deg C', CELSIUS)
        label = label.replace('degrees C', CELSIUS)
        label = label.replace('degrees_C', CELSIUS)
        label = label.replace('degC', CELSIUS)
        
        return label