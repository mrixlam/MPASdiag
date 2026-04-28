#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Unit Conversion Utilities

This module provides a comprehensive UnitConverter class for handling unit conversions of meteorological and atmospheric variables commonly found in MPAS model output. It supports a wide range of conversions for temperature, pressure, wind speed, humidity, precipitation rates, and more, following standard meteorological conventions. The class includes methods for converting data between units, determining preferred display units based on variable names, and normalizing unit strings to a canonical form. This utility ensures accurate and consistent unit handling across all MPASdiag processing and visualization workflows. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, Dict, Any, Optional

from .constants import (
    MM_PER_HR, MM_PER_DAY, DBZ, KELVIN, M_PER_S, PA, PERCENT,
    METER, MILES_PER_HR, KG_PER_KG, KG_PER_M2, KG_PER_M3, MICRONS, PER_KG,
    M3_PER_M3, G_PER_KG, KM_PER_HR, NOUNIT, HPA, MB, KNOT, FEET, 
    CELSIUS, FAHRENHEIT, KILOMETER, INCHES_PER_HR, MM
)


class UnitConverter:
    """ Unit conversion utility class for meteorological and atmospheric data with comprehensive conversion support following meteorological conventions. """
    
    @staticmethod
    def convert_units(data: Union[np.ndarray, xr.DataArray, float], 
                      from_unit: str, 
                      to_unit: str) -> Union[np.ndarray, xr.DataArray, float]:
        """
        This method converts numeric data from one unit to another based on a predefined set of supported conversions for meteorological variables. It first normalizes the input unit strings to a canonical form using the _normalize_unit_string helper method, then looks up the appropriate conversion function from an internal conversion map. The method supports conversions for temperature (K, °C, °F), pressure (Pa, hPa, mb), humidity (kg/kg, g/kg), precipitation rates (mm/hr, mm/day, in/hr), wind speed (m/s, knots, mph, km/hr), and more. If the requested conversion is not supported, it raises a ValueError with a clear message indicating the unsupported conversion and listing available options. The method returns the converted data in the same structure and type as the input (scalars remain scalars, arrays remain arrays, DataArrays remain DataArrays) to ensure seamless integration with existing MPASdiag workflows. 

        Parameters:
            data (Union[np.ndarray, xr.DataArray, float]): Numeric data to convert, can be a scalar, NumPy array, or xarray DataArray.
            from_unit (str): Original unit of the data, will be normalized before conversion lookup (e.g., 'Kelvin', 'degC', 'mm/hr').
            to_unit (str): Desired unit to convert to, will be normalized before conversion lookup (e.g., 'K', '°C', 'mm/hr'). 

        Returns:
            Union[np.ndarray, xr.DataArray, float]: Converted data in the same structure and type as the input, with values transformed to the desired unit. 
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
        
        (KG_PER_M2, MM): lambda x: x,  # 1 kg/m² = 1 mm of water
        (MM, KG_PER_M2): lambda x: x,  # 1 mm of water = 1 kg/m²
        
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
    def convert_data_for_display(data: xr.DataArray, 
                                 var_name: str, 
                                 metadata_source: xr.DataArray) -> Tuple[Union[np.ndarray, xr.DataArray, float], Dict[str, Any]]:
        """
        This method converts an xarray DataArray to preferred display units based on the variable name and its original units, following meteorological conventions for visualization. It first retrieves metadata for the variable using the MPASFileMetadata class, then determines the original units from the data attributes or metadata. Based on the variable name and original units, it looks up the preferred display unit (e.g., '°C' for temperature, 'hPa' for pressure) using the get_display_units method. If the original units differ from the preferred display units, it attempts to convert the data using the convert_units method. The method returns a tuple containing the converted data (in preferred display units) and a metadata dictionary that includes the current display units, original units, and other relevant variable properties extracted from the metadata source. This utility ensures that MPAS variables are presented in a consistent and intuitive manner for analysis and visualization while maintaining accurate unit handling. 

        Parameters: 
            data (xr.DataArray): Input data array to convert, must have 'units' attribute or corresponding metadata entry.
            var_name (str): Name of the variable to determine display unit preference (e.g., 't2m', 'mslp').
            metadata_source (xr.DataArray): DataArray containing metadata for the variable, used to extract original units and properties.

        Returns:
            Tuple[Union[np.ndarray, xr.DataArray, float], Dict[str, Any]]: A tuple containing the converted data in preferred display units and a metadata dictionary with 'units', 'original_units', and other relevant properties for the variable. 
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
        This helper method normalizes unit strings to a canonical form recognized by the UnitConverter class, allowing for flexible input formats while ensuring consistent unit handling. It uses a predefined mapping of common unit synonyms and variations (e.g., 'Kelvin', 'kelvin', 'K' all map to 'K') to convert various input representations into a standardized format. This normalization step enables users to input units in different styles or with common abbreviations without worrying about exact string matches, while still ensuring that the internal conversion logic operates on a consistent set of unit identifiers. If the input unit string does not match any known variations, it returns the original string, allowing for graceful handling of unrecognized units. 

        Parameters:
            unit (str): Input unit string to normalize, can be in various formats or synonyms (e.g., 'Kelvin', 'degC', 'mm/hr'). 

        Returns:
            str: Normalized unit string in a canonical form recognized by the UnitConverter (e.g., 'K', '°C', 'mm/hr'), or original string if no normalization rule applies. 
        """
        unit = unit.strip()
        
        unit_map = {
        'kelvin': KELVIN, 'k': KELVIN, KELVIN.lower(): KELVIN,
        'celsius': CELSIUS, 'degc': CELSIUS, 'deg_c': CELSIUS, 'c': CELSIUS, CELSIUS.lower(): CELSIUS,
        'fahrenheit': FAHRENHEIT, 'degf': FAHRENHEIT, 'deg_f': FAHRENHEIT, 'f': FAHRENHEIT, FAHRENHEIT.lower(): FAHRENHEIT,
        
        'pascal': PA, 'pa': PA, PA.lower(): PA,
        'hectopascal': HPA, 'hpa': HPA, HPA.lower(): HPA,
        'millibar': MB, 'mbar': MB, MB.lower(): MB,
        
        'kg kg-1': KG_PER_KG, 'kg_kg-1': KG_PER_KG, 'kg kg^{-1}': KG_PER_KG, KG_PER_KG.lower(): KG_PER_KG,
        'g kg-1': G_PER_KG, 'g_kg-1': G_PER_KG, 'g kg^{-1}': G_PER_KG, G_PER_KG.lower(): G_PER_KG,
        
        'kg m-2': KG_PER_M2, 'kg_m-2': KG_PER_M2, 'kg m^{-2}': KG_PER_M2, KG_PER_M2.lower(): KG_PER_M2,
        
        'mm hr-1': MM_PER_HR, 'mm_hr-1': MM_PER_HR, MM_PER_HR.lower(): MM_PER_HR,
        'mm day-1': MM_PER_DAY, 'mm_day-1': MM_PER_DAY, MM_PER_DAY.lower(): MM_PER_DAY,
        
        'percent': PERCENT, 'pct': PERCENT, PERCENT.lower(): PERCENT,
        'in hr-1': INCHES_PER_HR, 'in_hr-1': INCHES_PER_HR, INCHES_PER_HR.lower(): INCHES_PER_HR,
        
        'knots': KNOT, 'knot': KNOT, 'kts': KNOT, KNOT.lower(): KNOT,
        'miles per hour': MILES_PER_HR, 'mi/hr': MILES_PER_HR, MILES_PER_HR.lower(): MILES_PER_HR,
        'kilometers per hour': KM_PER_HR, 'km hr-1': KM_PER_HR, 'km_hr-1': KM_PER_HR, KM_PER_HR.lower(): KM_PER_HR,
        'm s-1': M_PER_S, 'm_s-1': M_PER_S, 'm s^{-1}': M_PER_S, M_PER_S.lower(): M_PER_S,
        
        'meter': METER, 'meters': METER, METER.lower(): METER,
        'kilometer': KILOMETER, 'kilometers': KILOMETER, KILOMETER.lower(): KILOMETER,
        'foot': FEET, 'feet': FEET, FEET.lower(): FEET,
        }
        
        return unit_map.get(unit.lower(), unit)

    @staticmethod
    def _preferred_unit_for_match(current_unit: str, 
                                  preferred_unit: str) -> Optional[str]:
        """
        This helper method checks if the current unit can be converted to the preferred unit and returns the preferred unit if conversion is possible, otherwise returns None. It first checks if the current unit is already the preferred unit, in which case it returns the preferred unit directly. If not, it attempts to perform a test conversion using the convert_units method with a dummy value (e.g., 1.0) to verify if the conversion between the current unit and preferred unit is supported. If the conversion is successful, it returns the preferred unit; if a ValueError is raised indicating that the conversion is not supported, it returns None. This method is used within get_display_units to determine whether to use the preferred display unit or fall back to the original unit based on conversion compatibility.

        Parameters:
            current_unit (str): The current unit of the variable, used to check if conversion to the preferred unit is possible (e.g., 'K', 'Pa').
            preferred_unit (str): The preferred display unit for the variable, used to check if conversion from the current unit is possible (e.g., '°C', 'hPa').

        Returns:
            Optional[str]: The preferred unit if conversion from the current unit is possible, or None if conversion is not supported.
        """
        if current_unit == preferred_unit:
            return preferred_unit
        try:
            UnitConverter.convert_units(1.0, current_unit, preferred_unit)
            return preferred_unit
        except ValueError:
            return None

    @staticmethod
    def get_display_units(variable_name: str,
                          current_unit: str) -> str:
        """
        This method determines the preferred display unit for a given MPAS variable based on its name and current unit, following meteorological conventions for visualization. It uses a predefined mapping of variable name patterns to preferred display units (e.g., 't2m' maps to '°C', 'mslp' maps to 'hPa') to identify the appropriate unit for visualization. The method first normalizes the current unit string using the _normalize_unit_string helper method, then checks if the variable name matches any of the patterns in the mapping. If a match is found and the current unit differs from the preferred display unit, it attempts to convert between the units using the convert_units method to ensure compatibility. If conversion is successful, it returns the preferred display unit; otherwise, it falls back to returning the original current unit. This utility ensures that MPAS variables are displayed in intuitive and standardized units for analysis and visualization while maintaining accurate unit handling. 

        Parameters:
            variable_name (str): Name of the variable to determine display unit preference (e.g., 't2m', 'mslp').
            current_unit (str): Current unit of the variable, will be normalized before checking for display unit preference (e.g., 'K', 'Pa'). 

        Returns:
            str: Preferred display unit for the variable based on its name and current unit, or the original current unit if no preferred display unit is applicable or if conversion is not possible. 
        """
        current_unit = UnitConverter._normalize_unit_string(current_unit)
            
        display_unit_preferences = {
            't2m': CELSIUS, 'temperature': CELSIUS, 'temp': CELSIUS, 'theta': CELSIUS,
            'tsk': CELSIUS, 'sst': CELSIUS, 'meanT': CELSIUS, 'theta_base': CELSIUS, 'tslb': CELSIUS,
            'dewpoint': CELSIUS, 'dewpt': CELSIUS, 'dpt': CELSIUS,
            
            'pressure': HPA, 'slp': HPA, 'psfc': HPA, 'pressure_p': HPA,
            'mslp': HPA, 'pmsl': HPA, 'pressure_base': HPA,
            
            'precipw': MM,            
            'precipitation': MM_PER_HR, 'precip': MM_PER_HR, 'precip_rate': MM_PER_HR,
            
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
                candidate_unit = UnitConverter._preferred_unit_for_match(current_unit, preferred_unit)
                return candidate_unit if candidate_unit is not None else preferred_unit

        for var_pattern, preferred_unit in display_unit_preferences.items():
            if len(var_pattern) > 2 and var_pattern.lower() in var_name_lower:
                candidate_unit = UnitConverter._preferred_unit_for_match(current_unit, preferred_unit)
                return candidate_unit if candidate_unit is not None else preferred_unit

        return current_unit

    @staticmethod
    def _format_colorbar_label(label: str) -> str:
        """
        This helper method formats colorbar label strings by replacing verbose or informal unit tokens with standardized symbols for improved readability in visualizations. It specifically targets common temperature unit representations such as 'deg_C', 'degrees C', 'degrees_C', and 'degC', replacing them with the standard '°C' symbol. This ensures that colorbar labels are concise and visually consistent, enhancing the clarity of plots and figures generated by MPASdiag. The method can be easily extended in the future to handle additional unit formatting rules as needed. 

        Parameters:
            label (str): Original colorbar label string that may contain verbose unit representations (e.g., 'Temperature (deg_C)'). 

        Returns:
            str: Formatted colorbar label string with standardized unit symbols (e.g., 'Temperature (°C)'). 
        """
        label = label.replace('deg_C', CELSIUS)
        label = label.replace('deg C', CELSIUS)
        label = label.replace('degrees C', CELSIUS)
        label = label.replace('degrees_C', CELSIUS)
        label = label.replace('degC', CELSIUS)
        
        return label