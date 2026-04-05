#!/usr/bin/env python3

"""
MPASdiag Core Visualization Module: Presentation Styling Utilities

This module provides comprehensive styling utilities for visualizing MPAS atmospheric diagnostics, including variable-specific colormaps, contour levels, and plot appearance parameters. It includes intelligent defaults based on variable names and data characteristics, as well as dynamic formatting of axis ticks and adaptive marker sizing for scatter plots. The module is designed to enhance the visual clarity and consistency of MPAS diagnostic plots while providing flexibility for customization based on specific variables and data ranges. It also includes functionality for adding standardized branding and metadata annotations to figures, and for saving plots in multiple formats with optimized settings for performance and quality balance. 
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Load necessary libraries for data handling, plotting, and styling
import os
import re
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.contour import QuadContourSet
from typing import Optional, Union, Tuple, List, Dict, Any, Literal, cast
from matplotlib.collections import PathCollection, QuadMesh, LineCollection

from ..processing.utils_metadata import MPASFileMetadata


class MPASVisualizationStyle:
    """ Comprehensive visualization styling utility class providing variable-specific colormaps, contour levels, and plot appearance parameters for MPAS atmospheric diagnostics. """
    
    @staticmethod
    def create_precip_colormap(accum: Optional[str] = None) -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        This method creates a discrete colormap and corresponding contour levels for precipitation variables based on the specified accumulation period. It defines a set of visually distinct colors suitable for representing different precipitation intensities, and then determines appropriate contour levels based on the accumulation period identifier (e.g., 'a24h' for 24-hour accumulation, 'a01h' for 1-hour accumulation). The method uses regular expressions to extract the number of hours from the accumulation string and applies heuristic rules to select contour levels that are commonly used in meteorological analyses for different accumulation periods. The resulting colormap and levels can be directly used in plotting functions to visualize precipitation diagnostics from MPAS output with clear differentiation between intensity ranges. 

        Parameters:
            accum (str): Accumulation period identifier (e.g., 'a24h', 'a01h', 'daily', 'hourly') used to determine contour levels for precipitation plotting (default: "a24h"). 

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: A tuple containing a ListedColormap object with predefined colors for precipitation intensity and a list of contour levels corresponding to the specified accumulation period. 
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

    @staticmethod
    def _hours_to_accum_period(hours: int) -> str:
        """
        This internal method converts a number of hours into a standardized accumulation period key string (e.g., 'a01h' for 1 hour, 'a24h' for 24 hours) that can be used to determine appropriate contour levels for precipitation plotting. It uses conditional logic to map specific hour values to their corresponding accumulation period identifiers, and also includes fallback rules to assign accumulation periods for non-standard hour values based on common meteorological conventions. This allows for flexible handling of various accumulation periods that may be indicated in variable names or metadata when determining styling parameters for precipitation diagnostics.

        Parameters:
            hours (int): The number of hours representing the accumulation period (e.g., 1, 3, 6, 12, 24).

        Returns:
            str: A standardized accumulation period key string (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h') corresponding to the input number of hours.
        """
        if hours == 1:  return 'a01h'
        if hours == 3:  return 'a03h'
        if hours == 6:  return 'a06h'
        if hours == 12: return 'a12h'
        if hours == 24: return 'a24h'
        if hours < 3:   return 'a01h'
        if hours < 6:   return 'a03h'
        if hours < 12:  return 'a06h'
        if hours < 24:  return 'a12h'
        return 'a24h'

    @staticmethod
    def _resolve_accum_period(var_lower: str) -> str:
        """
        This internal method resolves the accumulation period for precipitation variables by analyzing the lower-cased variable name string. It uses regular expressions to search for patterns indicating the number of hours (e.g., '1h', '3h', '6h', '12h', '24h') and maps them to standardized accumulation period keys using the _hours_to_accum_period method. Additionally, it checks for keywords like 'daily' and 'hourly' to assign appropriate accumulation periods when explicit hour values are not present in the variable name. This allows for robust detection of accumulation periods from variable names, which is essential for applying correct styling parameters for precipitation diagnostics in MPAS visualizations. 

        Parameters:
            var_lower (str): Lower-cased variable name.

        Returns:
            str: Accumulation period key (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h').
        """
        accum_match = re.search(r'(?<!\d)(\d{1,3})h', var_lower)

        if accum_match:
            return MPASVisualizationStyle._hours_to_accum_period(int(accum_match.group(1)))

        if 'daily' in var_lower:
            return 'a24h'

        if 'hourly' in var_lower:
            return 'a01h'

        return 'a01h'

    @staticmethod
    def get_variable_style(var_name: str,
                           data_array: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        This method retrieves styling parameters for a given MPAS variable name, including colormap, contour levels, and other visual attributes. It uses a predefined mapping of variable names (case-insensitive) to styling configurations, which can include specific colormap choices and level settings based on common meteorological conventions. For precipitation variables, it detects accumulation periods from the variable name and applies the create_precip_colormap method to obtain specialized colormaps and levels. If a data array is provided, the method can also attempt to generate contour levels automatically based on the data's statistical distribution (e.g., using percentiles) if no predefined levels are specified for that variable. The returned dictionary contains all necessary styling parameters that can be directly used in plotting functions to ensure consistent and informative visualizations of MPAS diagnostic variables. 

        Parameters:
            var_name (str): Variable name for which to retrieve styling parameters (case-insensitive).
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing the variable's data, used for automatic level generation if needed.

        Returns:
            Dict[str, Any]: A dictionary containing styling parameters such as 'colormap', 'levels', 'norm_type', 'alpha', and 'interpolation' that can be used for plotting the specified variable. 
        """
        metadata = MPASFileMetadata.get_variable_metadata(var_name, data_array)
        
        style_configs = {
            'olrtoa': {'colormap': 'inferno', 'extend': 'both'},
            
            'rainc': {'use_precip_colormap': True, 'extend': 'both'},  
            'rainnc': {'use_precip_colormap': True, 'extend': 'both'},
            'precip': {'use_precip_colormap': True, 'extend': 'both'},
            'total_precip': {'use_precip_colormap': True, 'extend': 'both'},  
            'precipw': {'colormap': 'Blues', 'extend': 'max'}, 

            'refl10cm_max': {'colormap': 'pyart_NWSRef', 'extend': 'both'},
            'refl10cm_1km': {'colormap': 'pyart_NWSRef', 'extend': 'both'},
            'refl10cm_1km_max': {'colormap': 'pyart_NWSRef', 'extend': 'both'},
            
            't2m': {'colormap': 'RdYlBu_r', 'extend': 'both'}, 
            'th2m': {'colormap': 'RdYlBu_r', 'extend': 'both'}, 
            'u10': {'colormap': 'RdBu_r', 'extend': 'both'},
            'v10': {'colormap': 'RdBu_r', 'extend': 'both'}, 
            'q2': {'colormap': 'Blues', 'extend': 'max'}, 
            'mslp': {'colormap': 'viridis', 'extend': 'both'},
            
            'relhum_50hPa': {'colormap': 'Blues', 'extend': 'max'},
            'relhum_100hPa': {'colormap': 'Blues', 'extend': 'max'},
            'relhum_200hPa': {'colormap': 'Blues', 'extend': 'max'},
            'relhum_250hPa': {'colormap': 'Blues', 'extend': 'max'},
            'relhum_500hPa': {'colormap': 'Blues', 'extend': 'max'},
            'relhum_700hPa': {'colormap': 'Blues', 'extend': 'max'},
            'relhum_850hPa': {'colormap': 'Blues', 'extend': 'max'},
            'relhum_925hPa': {'colormap': 'Blues', 'extend': 'max'},
            
            'temperature_50hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'},
            'temperature_100hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'},
            'temperature_200hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'},
            'temperature_250hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'},
            'temperature_500hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'},
            'temperature_700hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'},
            'temperature_850hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'},  
            'temperature_925hPa': {'colormap': 'RdYlBu_r', 'extend': 'both'}, 
            
            'height_50hPa': {'colormap': 'viridis', 'extend': 'both'},
            'height_100hPa': {'colormap': 'viridis', 'extend': 'both'},
            'height_200hPa': {'colormap': 'viridis', 'extend': 'both'},
            'height_250hPa': {'colormap': 'viridis', 'extend': 'both'},
            'height_500hPa': {'colormap': 'viridis', 'extend': 'both'},
            'height_700hPa': {'colormap': 'viridis', 'extend': 'both'},
            'height_850hPa': {'colormap': 'viridis', 'extend': 'both'},
            'height_925hPa': {'colormap': 'viridis', 'extend': 'both'},

            'uzonal_50hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'uzonal_100hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'uzonal_200hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'uzonal_250hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'uzonal_500hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'uzonal_700hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'uzonal_850hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'uzonal_925hPa': {'colormap': 'RdBu_r', 'extend': 'both'},

            'umeridional_50hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'umeridional_100hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'umeridional_200hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'umeridional_250hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'umeridional_500hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'umeridional_700hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'umeridional_850hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'umeridional_925hPa': {'colormap': 'RdBu_r', 'extend': 'both'},

            'vorticity_50hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'vorticity_100hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'vorticity_200hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'vorticity_250hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'vorticity_500hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'vorticity_700hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'vorticity_850hPa': {'colormap': 'RdBu_r', 'extend': 'both'},
            'vorticity_925hPa': {'colormap': 'RdBu_r', 'extend': 'both'},

            'cape': {'colormap': 'plasma', 'extend': 'max'},
            'cin': {'colormap': 'viridis_r', 'extend': 'both'},
            'lcl': {'colormap': 'viridis', 'extend': 'both'},
            'lfc': {'colormap': 'viridis', 'extend': 'both'},
            'srh_0_1km': {'colormap': 'plasma', 'extend': 'both'},
            'srh_0_3km': {'colormap': 'plasma', 'extend': 'both'},

            'updraft_helicity_max': {'colormap': 'plasma', 'extend': 'max'},
            'w_velocity_max': {'colormap': 'plasma', 'extend': 'both'},
            'wind_speed_level1_max': {'colormap': 'plasma', 'extend': 'max'},

            't_oml': {'colormap': 'RdYlBu_r', 'extend': 'both'},
            'h_oml': {'colormap': 'viridis', 'extend': 'both'},
            'hu_oml': {'colormap': 'RdBu_r', 'extend': 'both'},
            'hv_oml': {'colormap': 'RdBu_r', 'extend': 'both'},

            'cldfrac_low_UPP': {'colormap': 'gray', 'extend': 'max'},
            'cldfrac_mid_UPP': {'colormap': 'gray', 'extend': 'max'},
            'cldfrac_high_UPP': {'colormap': 'gray', 'extend': 'max'},
            'cldfrac_tot_UPP': {'colormap': 'gray', 'extend': 'max'},
        }
        
        style = style_configs.get(var_name.lower(), {
            'colormap': 'viridis',
            'extend': 'both'
        })
        
        var_lower = var_name.lower()
        is_precipitation = (style.get('use_precip_colormap') or 
                           any(precip_key in var_lower for precip_key in ['precip', 'rain', 'snow', 'qpf']))
        
        if is_precipitation:
            accum_period = MPASVisualizationStyle._resolve_accum_period(var_lower)
            cmap, levels = MPASVisualizationStyle.create_precip_colormap(accum_period)
            style['colormap'] = cmap
            style['levels'] = levels
            style.pop('use_precip_colormap', None)
        
        if data_array is not None and 'levels' not in style and not is_precipitation:
            try:
                levels = MPASVisualizationStyle._generate_levels_from_data(data_array, var_name)
                if levels is not None:
                    style['levels'] = levels
            except Exception:
                pass
        
        if 'levels' not in style:
            style['levels'] = list(range(0, 21))
        
        style.update({
            'norm_type': 'linear',  # Could be 'linear', 'log', 'symlog', etc.
            'alpha': 1.0,
            'interpolation': 'nearest'
        })
        
        return style

    @staticmethod
    def _generate_levels_from_data(data_array: xr.DataArray, 
                                   var_name: str) -> Optional[List[float]]:
        """
        This internal method generates contour levels automatically from the provided data array based on its statistical distribution. It computes the 5th and 95th percentiles of the finite data values to determine a reasonable range for contour levels, and then applies heuristic rules to create a list of contour levels that are appropriately spaced for the variable type (e.g., finer spacing for temperature variables, wider spacing for pressure or wind speed). The method includes safeguards to handle cases where the data may be invalid (e.g., all values are NaN or infinite) or where the computed range is too narrow to generate meaningful levels. The resulting list of contour levels can be used in plotting functions to visualize the variable with appropriate level spacing based on the actual data characteristics. 

        Parameters:
            data_array (xr.DataArray): The xarray DataArray containing the variable's data from which to generate contour levels.
            var_name (str): The variable name, used for applying variable-specific heuristics in level generation (case-insensitive). 
            
        Returns:
            Optional[List[float]]: A list of contour levels generated from the data statistics, or None if levels could not be generated due to invalid data or insufficient range. 
        """
        try:
            data_values = data_array.values
            if hasattr(data_values, 'flatten'):
                flat_data = data_values.flatten()
                finite_data = flat_data[np.isfinite(flat_data)]
                
                if len(finite_data) == 0:
                    return None
                
                data_min = float(np.percentile(finite_data, 5))
                data_max = float(np.percentile(finite_data, 95))
                
                if data_max <= data_min:
                    return None
                
                if any(x in var_name.lower() for x in ['temp', 't2m']):
                    n_levels = 15
                    level_step = (data_max - data_min) / n_levels
                    if level_step < 1:
                        step = 0.5
                    elif level_step < 2:
                        step = 1
                    elif level_step < 5:
                        step = 2
                    else:
                        step = 5
                    
                    start = np.floor(data_min / step) * step
                    end = np.ceil(data_max / step) * step
                    levels = np.arange(start, end + step, step).tolist()
                else:
                    n_levels = 15
                    level_step = (data_max - data_min) / n_levels
                    levels = [data_min + i * level_step for i in range(n_levels + 1)]
                
                return levels
                
        except Exception:
            return None

    @staticmethod
    def _levels_for_temperature(data_min: float, 
                                data_max: float, 
                                data_range: float) -> Tuple[str, List[float]]:
        """
        This internal method returns a colormap and contour levels specifically tailored for temperature-like variables. It uses a diverging colormap ('RdYlBu_r') that is commonly used for temperature data to enhance visual contrast between warm and cool values. The method determines the contour levels based on the data range, applying finer spacing for smaller ranges (e.g., 1 degree intervals for ranges less than 20 degrees) and coarser spacing for larger ranges (e.g., 5 degree intervals for ranges greater than 40 degrees). The levels are generated as a list of floats that are evenly spaced within the data range, ensuring that they are appropriate for visualizing the temperature variable effectively. This approach allows for clear differentiation of temperature values in MPAS diagnostic plots while maintaining consistency with meteorological visualization conventions.

        Parameters:
            data_min (float): The minimum data value for the temperature variable, used to determine the starting point for contour levels.
            data_max (float): The maximum data value for the temperature variable, used to determine the ending point for contour levels.
            data_range (float): The range of the data values (data_max - data_min), used to determine the spacing of contour levels.

        Returns:
            Tuple[str, List[float]]: A tuple containing the colormap name ('RdYlBu_r') and a list of contour levels appropriate for the temperature variable based on the data characteristics.
        """
        colormap = 'RdYlBu_r'
        if data_range > 40:
            step = 5
        elif data_range > 20:
            step = 2
        else:
            step = 1
        levels = [float(x) for x in range(int(data_min // step) * step,
                                           int(data_max // step) * step + step * 2, step)]
        levels = [lv for lv in levels if data_min <= lv <= data_max]
        return colormap, levels

    @staticmethod
    def _levels_for_precipitation(var_lower: str, 
                                  data_max: float) -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        This internal method returns a colormap and contour levels specifically tailored for precipitation-like variables based on the variable name and maximum data value. It detects the accumulation period from the variable name (e.g., 'a01h' for 1-hour accumulation, 'a24h' for 24-hour accumulation) using pattern matching, and then applies the create_precip_colormap method to obtain a specialized colormap and corresponding contour levels that are commonly used for visualizing precipitation intensity. The method also adjusts the contour levels based on the maximum data value to ensure that the levels are appropriate for the range of precipitation values being visualized, allowing for clear differentiation of precipitation intensities in MPAS diagnostic plots while adhering to meteorological visualization standards.

        Parameters:
            var_lower (str): The lowercase name of the precipitation variable, used to determine the accumulation period.
            data_max (float): The maximum data value for the precipitation variable, used to adjust the upper limit of the levels.

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: A tuple containing the colormap and a list of contour levels appropriate for the precipitation variable based on the data characteristics.
        """
        if any(p in var_lower for p in ['01h', '1h', 'hourly']):
            accum_period = 'a01h'
        elif any(p in var_lower for p in ['03h', '3h']):
            accum_period = 'a03h'
        elif any(p in var_lower for p in ['06h', '6h']):
            accum_period = 'a06h'
        elif '12h' in var_lower:
            accum_period = 'a12h'
        elif any(p in var_lower for p in ['24h', 'daily']):
            accum_period = 'a24h'
        else:
            accum_period = 'a24h'

        cmap, levels = MPASVisualizationStyle.create_precip_colormap(accum_period)

        if data_max > 0:
            levels = [lv for lv in levels if lv <= data_max * 1.2]
        return cmap, levels

    @staticmethod
    def _levels_for_pressure(var_lower: str, 
                             data_min: float, 
                             data_max: float, 
                             data_range: float) -> Tuple[str, List[float]]:
        """
        This internal method returns a colormap and contour levels specifically tailored for pressure-like variables based on the variable name and data characteristics. It uses a diverging colormap ('RdBu_r') that is commonly used for pressure data to enhance visual contrast between high and low pressure values. The method determines the contour levels based on the data range, applying finer spacing (e.g., 5 hPa intervals) for smaller ranges (e.g., less than 100 hPa) and coarser spacing (e.g., 500 hPa intervals) for larger ranges (e.g., greater than 1000 hPa). The levels are generated as a list of floats that are evenly spaced within the data range, ensuring that they are appropriate for visualizing the pressure variable effectively. This approach allows for clear differentiation of pressure values in MPAS diagnostic plots while maintaining consistency with meteorological visualization conventions.

        Parameters:
            var_lower (str): The lowercase name of the pressure variable, used to identify if it is a pressure variable.
            data_min (float): The minimum data value for the pressure variable, used to determine the starting point for contour levels.
            data_max (float): The maximum data value for the pressure variable, used to determine the ending point for contour levels.
            data_range (float): The range of the data values (data_max - data_min), used to determine the spacing of contour levels.

        Returns:
            Tuple[str, List[float]]: A tuple containing the colormap name ('RdBu_r') and a list of contour levels appropriate for the pressure variable based on the data characteristics.
        """
        colormap = 'RdBu_r'

        if 'slp' in var_lower or 'mslp' in var_lower:
            step = 500 if data_range > 100 else 5
        else:
            step = max(100, int(data_range / 20)) if data_range > 1000 else max(10, int(data_range / 15))

        levels = [float(x) for x in range(int(data_min // step) * step,
                                           int(data_max // step) * step + step * 2, step)]

        levels = [lv for lv in levels if data_min <= lv <= data_max]
        return colormap, levels

    @staticmethod
    def _levels_for_wind(data_max: float) -> Tuple[str, List[float]]:
        """
        This internal method returns a colormap and contour levels specifically tailored for wind speed variables based on the maximum data value. It uses a perceptually uniform colormap ('plasma') that is suitable for representing wind speed data, providing clear visual differentiation across a range of values. The method determines the contour levels based on the maximum data value, applying finer spacing (e.g., 0.5 m/s intervals) for lower maximum values (e.g., less than 3 m/s) and coarser spacing (e.g., 10 m/s intervals) for higher maximum values (e.g., greater than 30 m/s). The levels are generated as a list of floats that are appropriate for visualizing the wind speed variable effectively, allowing for clear differentiation of wind speed values in MPAS diagnostic plots while adhering to meteorological visualization standards.

        Parameters:
            data_max (float): The maximum data value for the wind speed variable, used to determine the appropriate contour levels.

        Returns:
            Tuple[str, List[float]]: A tuple containing the colormap name ('plasma') and a list of contour levels appropriate for the wind speed variable based on the data characteristics.
        """
        colormap = 'plasma'
        if data_max < 3:
            raw = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        elif data_max < 15:
            raw = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        elif data_max < 30:
            raw = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        else:
            raw = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        levels = [lv for lv in raw if lv <= data_max * 1.1]
        return colormap, levels

    @staticmethod
    def _levels_for_geopotential(data_min: float, 
                                 data_max: float, 
                                 data_range: float) -> Tuple[str, List[float]]:
        """
        This internal method returns a colormap and contour levels specifically tailored for geopotential height variables based on the data characteristics. It uses a colormap ('terrain') that is suitable for representing height data, providing clear visual differentiation across a range of values. The method determines the contour levels based on the data range, applying finer spacing (e.g., 20 m intervals) for smaller ranges (e.g., less than 200 m) and coarser spacing (e.g., 60 m intervals) for larger ranges (e.g., greater than 1000 m). The levels are generated as a list of floats that are evenly spaced within the data range, ensuring that they are appropriate for visualizing the geopotential height variable effectively. This approach allows for clear differentiation of height values in MPAS diagnostic plots while maintaining consistency with meteorological visualization conventions. 

        Parameters:
            data_min (float): The minimum data value for the geopotential/height variable, used to determine the starting point for contour levels.
            data_max (float): The maximum data value for the geopotential/height variable, used to determine the ending point for contour levels.
            data_range (float): The range of the data values (data_max - data_min), used to determine the spacing of contour levels.

        Returns:
            Tuple[str, List[float]]: A tuple containing the colormap name ('terrain') and a list of contour levels appropriate for the geopotential/height variable based on the data characteristics.
        """
        colormap = 'terrain'
        if data_range > 1000:
            step = 60 if data_range > 2000 else 30
        else:
            step = 20 if data_range > 200 else 10
        levels = [float(x) for x in range(int(data_min // step) * step,
                                           int(data_max // step) * step + step * 2, step)]
        levels = [lv for lv in levels if data_min <= lv <= data_max]
        return colormap, levels

    @staticmethod
    def _levels_for_humidity(var_lower: str, 
                             data_min: float, 
                             data_max: float) -> Tuple[str, List[float]]:
        """
        This internal method returns a colormap and contour levels specifically tailored for humidity-related variables based on the variable name and data characteristics. It uses a colormap ('BuGn') that is suitable for representing humidity data, providing clear visual differentiation across a range of values. The method determines the contour levels based on the maximum data value, applying finer spacing (e.g., 0.1 intervals) for relative humidity variables that typically range from 0 to 100%, and wider spacing (e.g., logarithmic intervals) for specific humidity or mixing ratio variables that can have a wider range of values. The levels are generated as a list of floats that are appropriate for visualizing the humidity variable effectively, allowing for clear differentiation of humidity values in MPAS diagnostic plots while adhering to meteorological visualization standards.

        Parameters:
            var_lower (str): The lowercase name of the humidity variable, used to identify if it is a relative humidity variable or a specific humidity/mixing ratio variable.
            data_min (float): The minimum data value for the humidity variable, used to determine the starting point for contour levels.
            data_max (float): The maximum data value for the humidity variable, used to determine the ending point for contour levels.

        Returns:
            Tuple[str, List[float]]: A tuple containing the colormap name ('BuGn') and a list of contour levels appropriate for the humidity variable based on the data characteristics. 
        """
        colormap = 'BuGn'
        if data_max <= 1.1:
            if 'rh' in var_lower or 'humidity' in var_lower:
                raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            else:
                raw = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        else:
            raw = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        levels = [lv for lv in raw if data_min <= lv <= data_max]
        return colormap, levels

    @staticmethod
    def _levels_default(data_min: float, 
                        data_max: float, 
                        data_range: float) -> Tuple[str, List[float]]:
        """
        This internal method returns a default colormap and contour levels for variables that do not match specific patterns for temperature, precipitation, pressure, wind speed, geopotential height, or humidity. It uses a general colormap ('viridis') that is suitable for a wide range of variable types, providing clear visual differentiation across a range of values. The method determines the contour levels based on the data range, applying a simple heuristic to create evenly spaced levels within the data range. If the data range is small, it generates more closely spaced levels; if the data range is large, it generates fewer levels with wider spacing. The levels are generated as a list of floats that are appropriate for visualizing the variable effectively, allowing for clear differentiation of values in MPAS diagnostic plots while maintaining a consistent visual style for variables without specific styling rules.

        Parameters:
            data_min (float): The minimum data value, used to determine the starting point for contour levels.
            data_max (float): The maximum data value, used to determine the ending point for contour levels.
            data_range (float): The range of the data, used to calculate the step size for evenly-spaced levels.

        Returns:
            Tuple[str, List[float]]: A tuple containing the colormap name and a list of contour levels appropriate for the data range.

        """
        if data_min < 0 and data_max > 0:
            colormap = 'RdBu_r'
        elif data_min >= 0:
            colormap = 'viridis'
        else:
            colormap = 'plasma'
        n_levels = 10
        step = data_range / n_levels
        levels = [data_min + i * step for i in range(n_levels + 1)]
        return colormap, levels

    @staticmethod
    def get_variable_specific_settings(var_name: str,
                                       data: np.ndarray) -> Tuple[Union[str, mcolors.ListedColormap], Optional[List[float]]]:
        """
        This method determines the appropriate colormap and contour levels for a given variable name and its associated data array. It uses pattern matching on the variable name to identify common meteorological variable types (e.g., temperature, precipitation, pressure, wind speed, geopotential height, humidity) and applies specific rules for selecting colormaps and generating contour levels based on the data characteristics. For example, it may use a diverging colormap for temperature variables and a specialized precipitation colormap for precipitation variables, while also adjusting the contour levels based on the data range and distribution. If the variable name does not match any specific patterns, it falls back to a default colormap and generates levels based on the data range. The method returns a tuple containing the selected colormap (either as a string name or a ListedColormap object) and a list of contour levels that are appropriate for visualizing the variable effectively in MPAS diagnostic plots.

        Parameters:
            var_name (str): The variable name, used for determining specific styling rules based on common meteorological variable patterns (case-insensitive).
            data (np.ndarray): The data array for the variable, used to compute data characteristics such as range and percentiles for level generation.
            
        Returns:
            Tuple[Union[str, mcolors.ListedColormap], Optional[List[float]]]: A tuple containing the selected colormap (either as a string name or a ListedColormap object) and a list of contour levels appropriate for the variable and its data characteristics. The levels may be None if they could not be generated due to invalid data or insufficient range. 
        """
        var_lower = var_name.lower().strip()

        if isinstance(data, xr.DataArray):
            data_values = data.values.flatten()
        else:
            data_values = np.asarray(data).flatten()

        valid_data = data_values[np.isfinite(data_values)]
        
        if len(valid_data) == 0:
            return 'viridis', None

        data_min, data_max = float(np.min(valid_data)), float(np.max(valid_data))
        data_range = data_max - data_min

        if any(k in var_lower for k in ['temp', 't2m', 'temperature', 'sst']):
            return MPASVisualizationStyle._levels_for_temperature(data_min, data_max, data_range)

        if any(k in var_lower for k in ['precip', 'rain', 'snow', 'qpf']):
            return MPASVisualizationStyle._levels_for_precipitation(var_lower, data_max)

        if any(k in var_lower for k in ['pressure', 'slp', 'mslp', 'pres']):
            return MPASVisualizationStyle._levels_for_pressure(var_lower, data_min, data_max, data_range)

        if any(k in var_lower for k in ['wind', 'wspd', 'speed']):
            return MPASVisualizationStyle._levels_for_wind(data_max)

        if any(k in var_lower for k in ['geopotential', 'height', 'hgt', 'z']):
            return MPASVisualizationStyle._levels_for_geopotential(data_min, data_max, data_range)

        if any(k in var_lower for k in ['humidity', 'rh', 'q', 'mixing']):
            return MPASVisualizationStyle._levels_for_humidity(var_lower, data_min, data_max)

        return MPASVisualizationStyle._levels_default(data_min, data_max, data_range)

    @staticmethod
    def setup_map_projection(lon_min: float,
                             lon_max: float,
                             lat_min: float,
                             lat_max: float,
                             projection: str = 'PlateCarree') -> Tuple[ccrs.Projection, ccrs.PlateCarree]:
        """
        This method sets up the map projection for plotting MPAS diagnostic data based on the specified longitude and latitude bounds and the desired projection type. It calculates the central longitude and latitude from the provided bounds to use as reference points for certain projections (e.g., Mercator, Lambert Conformal). The method supports multiple projection types, including 'PlateCarree', 'Mercator', and 'LambertConformal', and defaults to 'PlateCarree' if an unrecognized projection name is provided. It returns a tuple containing the map projection object to be used for plotting and the data coordinate system (which is typically PlateCarree for MPAS data). This setup allows for flexible visualization of MPAS diagnostics on different map projections while ensuring that the data is correctly transformed to match the chosen projection. 

        Parameters:
            lon_min (float): Minimum longitude of the plot area.
            lon_max (float): Maximum longitude of the plot area.
            lat_min (float): Minimum latitude of the plot area.
            lat_max (float): Maximum latitude of the plot area.
            projection (str): Desired map projection type (e.g., 'PlateCarree', 'Mercator', 'LambertConformal'). Defaults to 'PlateCarree'. 

        Returns:
            Tuple[ccrs.Projection, ccrs.PlateCarree]: A tuple containing the map projection object to be used for plotting and the data coordinate system (PlateCarree). The map projection is configured based on the specified bounds and projection type, while the data coordinate system is set to PlateCarree for compatibility with MPAS data. 
        """
        central_lon = (lon_min + lon_max) / 2
        central_lat = (lat_min + lat_max) / 2
        
        if projection.lower() == 'platecarree':
            map_proj = ccrs.PlateCarree()
        elif projection.lower() == 'mercator':
            map_proj = ccrs.Mercator(central_longitude=central_lon)
        elif projection.lower() == 'lambertconformal':
            map_proj = ccrs.LambertConformal(central_longitude=central_lon, central_latitude=central_lat)
        else:
            map_proj = ccrs.PlateCarree()
        
        data_crs = ccrs.PlateCarree()
        
        return map_proj, data_crs
    
    @staticmethod
    def add_timestamp_and_branding(fig: Figure) -> None:
        """
        This method adds a timestamp and branding annotation to the provided matplotlib Figure object. It generates a timestamp string in UTC format and attempts to retrieve the version of MPASdiag being used for plot generation. The method then constructs a branding text string that includes both the MPASdiag version and the timestamp. To ensure that the branding does not interfere with the main plot area, it creates or reuses a dedicated footer Axes at the bottom of the figure where the branding text is placed. The footer is designed to be non-intrusive, with a smaller font size and reduced opacity, allowing it to provide useful metadata about the plot generation without detracting from the visual presentation of the diagnostic data. If any issues arise while creating the footer, the method falls back to placing the branding text directly on the figure with a lower opacity to maintain visibility while minimizing interference with the plot content. 

        Parameters:
            fig (Figure): The matplotlib Figure object to which the timestamp and branding annotation will be added. The method modifies the figure in-place by adding text annotation to a dedicated footer area at the bottom of the figure. 

        Returns:
            None: This method does not return any value; it modifies the provided Figure object in-place by adding the timestamp and branding annotation. The annotation includes the MPASdiag version and the timestamp of plot generation, and is placed in a way that minimizes interference with the main plot area. 
        """
        if fig is None:
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        try:
            from .. import __version__
            version_str = f"v{__version__}"
        except ImportError:
            version_str = "v1.1.2"

        branding_text = f'Generated with MPASdiag {version_str} on: {timestamp}'

        # Create or reuse a footer Axes so branding is placed in its own reserved area
        try:
            # Use a slightly larger footer to improve readability
            footer_ax = MPASVisualizationStyle._create_footer_axes(fig, height=0.7, pad=0.02)
            print("MPASVisualizationStyle: using footer axes for branding", flush=True)
            footer_ax.text(0.001, -0.2, branding_text, va='center', ha='left', fontsize=9, alpha=0.85, transform=footer_ax.transAxes)
        except Exception as e:
            # Fallback to simple fig.text if anything goes wrong
            print(f"MPASVisualizationStyle: footer failed ({e}), falling back to fig.text", flush=True)
            fig.text(0.02, 0.02, branding_text, fontsize=8, alpha=0.7, transform=fig.transFigure)

    @staticmethod
    def _create_footer_axes(fig: Figure, 
                            height: float = 0.07, 
                            pad: float = 0.02) -> Axes:
        """
        This internal method creates a dedicated footer Axes at the bottom of the provided matplotlib Figure object for placing annotations such as timestamps and branding. It first checks if a footer Axes already exists (identified by a specific gid) and reuses it if found, allowing for multiple calls to add_timestamp_and_branding without creating duplicate footers. If no existing footer is found, it computes the position for the new footer, taking into account any horizontal colorbars that may be present near the bottom of the figure to ensure the footer is placed above them with a sensible gap. The method then creates a new Axes with the specified height and padding, sets its gid for future identification, and turns off the axis to create a clean area for text annotation. This approach ensures that branding and timestamp information can be added in a consistent and non-intrusive manner across different plots generated by MPASdiag. 

        Parameters:
            fig (Figure): The matplotlib Figure object to which the footer Axes will be added. The method modifies the figure in-place by adding a new Axes at the bottom of the figure for branding and timestamp annotations.
            height (float): The height of the footer Axes as a fraction of the figure height (default: 0.07). This determines how much vertical space is reserved for the footer.
            pad (float): The padding space as a fraction of the figure height to place between the bottom of the main plot area and the top of the footer Axes (default: 0.02). This helps ensure that the footer does not overlap with any plot elements or colorbars near the bottom of the figure. 

        Returns:
            Axes: The created or reused footer Axes object where branding and timestamp annotations can be placed. This Axes is configured to be non-intrusive, with the axis turned off, allowing for clear placement of text without interfering with the main plot area. 
        """
        # Reuse existing footer if present (identified by gid)
        for ax in fig.axes:
            try:
                if getattr(ax, 'get_gid', lambda: None)() == 'mpasdiag_footer':
                    print('MPASVisualizationStyle: reusing existing footer axes', flush=True)
                    return ax
            except Exception:
                continue

        # Compute footer position: full-width with small left/right margins
        left = 0.02
        width = 0.96

        # Detect horizontal colorbars near the bottom and place footer above them
        colorbar_tops: List[float] = []
        try:
            for ax in fig.axes:
                pos = ax.get_position()
                # Heuristic for horizontal colorbar: wide, short height, located low in figure
                if pos.width > 0.35 and pos.height < 0.12 and pos.y0 < 0.35:
                    colorbar_tops.append(pos.y1)
        except Exception:
            colorbar_tops = []

        gap = 0.03

        if colorbar_tops:
            # Place footer just above the highest colorbar top, with sensible cap
            bottom = min(max(colorbar_tops) + gap, 0.18)
            print(f"MPASVisualizationStyle: detected colorbar tops={colorbar_tops}, placing footer bottom={bottom}", flush=True)
        else:
            bottom = pad + 0.05
            print(f"MPASVisualizationStyle: no colorbar detected, using pad bottom={bottom}", flush=True)

        footer_ax = fig.add_axes((left, bottom, width, height), frameon=False, zorder=105)
        footer_ax.set_gid('mpasdiag_footer')
        footer_ax.set_axis_off()
        print(f"MPASVisualizationStyle: created footer axes at {(left, bottom, width, height)}", flush=True)
        return footer_ax
    
    @staticmethod
    def save_plot(fig: Figure, 
                  output_path: str, 
                  formats: List[str] = ['png'],
                  bbox_inches: str = 'tight',
                  pad_inches: float = 0.1,
                  dpi: int = 100) -> None:
        """
        This method saves the provided matplotlib Figure object to disk in one or more specified formats with configurable options for bounding box, padding, and resolution. It first checks if the figure is valid (not None) and raises a ValueError if there is no figure to save. The method then ensures that the output directory exists, creating it if necessary. For each specified format in the formats list, it constructs the full output file path by appending the appropriate extension to the base output path and saves the figure using fig.savefig with the provided parameters. If the format is 'png', it applies additional PIL-specific parameters to optimize file size while maintaining quality. After saving each file, it prints a confirmation message indicating where the plot was saved. This method provides a flexible and robust way to save MPAS diagnostic plots in various formats with consistent styling and layout. 

        Parameters:
            fig (Figure): The matplotlib Figure object to be saved. This should be a valid figure containing the plot to be saved.
            output_path (str): The base file path (without extension) where the plot will be saved. The method will append the appropriate extension for each format specified in the formats list.
            formats (List[str]): A list of string format identifiers indicating the desired output formats (e.g., ['png', 'pdf', 'svg']). Defaults to ['png'] if not provided.
            bbox_inches (str): The bounding box option for saving the figure, typically 'tight' to minimize whitespace around the plot. Default is 'tight'.
            pad_inches (float): The amount of padding in inches to add around the figure when bbox_inches is set to 'tight'. Default is 0.1 inches.
            dpi (int): The resolution in dots per inch for raster formats like PNG. Higher values result in better quality but larger file sizes. Default is 100 dpi.

        Returns:
            None: This method does not return any value; it saves the figure to disk in the specified formats and prints confirmation messages for each saved file. If the figure is invalid (None), it raises a ValueError to indicate that there is no figure to save. 
        """
        if fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            full_path = f"{output_path}.{fmt}"
            save_kwargs = {'dpi': dpi, 'bbox_inches': bbox_inches, 'pad_inches': pad_inches, 'format': fmt}
            if fmt.lower() == 'png':
                save_kwargs['pil_kwargs'] = {'compress_level': 1}
            fig.savefig(full_path, **save_kwargs)
            print(f"Saved plot: {full_path}")

    @staticmethod
    def get_3d_variable_style(var_name: str, 
                              level: Optional[Union[str, float]] = None, 
                              data_array: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        This method is a placeholder for future implementation of styling parameters specific to 3D atmospheric variables in MPAS diagnostics. It is designed to determine appropriate colormaps, contour levels, and other styling options based on the variable name, vertical level specification, and optionally the data array itself for automatic level-specific styling detection. The method is intended to support various types of 3D variables such as temperature, wind components, humidity, and vorticity at different pressure levels, model levels, or height levels. However, as of the current version, this method raises a NotImplementedError to indicate that the functionality for 3D variable styling has not yet been developed and is planned for future enhancement of the MPASVisualizationStyle class. 

        Parameters:
            var_name (str): The name of the 3D variable for which styling parameters are to be determined. This could include variables like 'temperature', 'u_wind', 'v_wind', 'humidity', etc.
            level (Union[str, float], optional): The vertical level specification for the variable, which could be a pressure level (e.g., '500hPa'), a model level (e.g., 'model_level_10'), or a height level (e.g., 'z_1000m'). This parameter is optional and may be used in future implementations to provide level-specific styling.
            data_array (xr.DataArray, optional): The xarray DataArray containing the variable's data, which can be used for automatic detection of appropriate contour levels and styling based on the data distribution. This parameter is optional and may be utilized in future implementations to enhance the styling logic for 3D variables.
            
        Returns:
            Dict[str, Any]: A dictionary containing styling parameters such as 'colormap', 'levels', 'norm_type', etc., specific to the 3D variable and its vertical level. This is intended for future implementation and currently raises NotImplementedError. 
        """
        raise NotImplementedError(
            "3D variable support not yet implemented. "
            "This function is a placeholder for future development."
        )

    @staticmethod
    def _configure_horizontal_colorbar(cbar: Colorbar, 
                                       label: Optional[str],
                                       label_pos: str, 
                                       tick_labelsize: int,
                                       labelpad: float) -> None:
        """
        This internal method applies label and tick positioning for a horizontal colorbar based on the specified label position ('top' or 'bottom') and tick label size. It attempts to set the label position and tick positions according to the provided parameters, with error handling to ensure that if any issues arise (e.g., due to unsupported configurations), it falls back gracefully without crashing. The method also sets the colorbar label with the specified font size and padding, again with error handling to accommodate different versions of matplotlib or potential issues with label setting. This approach ensures that horizontal colorbars in MPAS diagnostic plots are styled consistently while maintaining robustness against potential configuration issues.

        Parameters:
            cbar (Colorbar): The Colorbar object to be configured.
            label (str, optional): The text label for the colorbar. If None, no label will be set.
            label_pos (str): The position of the colorbar label, either 'top' or 'bottom' for horizontal colorbars.
            tick_labelsize (int): The font size for the colorbar tick labels, used to determine the label font size as well.
            labelpad (float): The padding between the colorbar and its label in points.

        Returns:
            None: This method does not return any value; it modifies the provided Colorbar object in-place by setting the label and tick positions according to the specified parameters, with error handling to ensure robustness against potential issues.
        """
        try:
            cbar.ax.xaxis.set_label_position(cast(Literal['top', 'bottom'], label_pos))
            ticks_pos = 'bottom' if label_pos == 'top' else 'top'
            cbar.ax.xaxis.set_ticks_position(ticks_pos)
        except Exception:
            pass
        if label is not None:
            try:
                cbar.set_label(label, fontsize=max(10, tick_labelsize), labelpad=labelpad)
            except Exception:
                cbar.ax.set_xlabel(label, fontsize=max(10, tick_labelsize), labelpad=labelpad)

    @staticmethod
    def _configure_vertical_colorbar(cbar: Colorbar, 
                                     label: Optional[str],
                                     label_pos: str, 
                                     tick_labelsize: int,
                                     labelpad: float) -> None:
        """
        This internal method applies label and tick positioning for a vertical colorbar based on the specified label position ('left' or 'right') and tick label size. It attempts to set the label position and tick positions according to the provided parameters, with error handling to ensure that if any issues arise (e.g., due to unsupported configurations), it falls back gracefully without crashing. The method also sets the colorbar label with the specified font size and padding, again with error handling to accommodate different versions of matplotlib or potential issues with label setting. This approach ensures that vertical colorbars in MPAS diagnostic plots are styled consistently while maintaining robustness against potential configuration issues. 

        Parameters:
            cbar (Colorbar): The Colorbar object to be configured.
            label (str, optional): The text label for the colorbar. If None, no label will be set.
            label_pos (str): The position of the colorbar label, either 'left' or 'right' for vertical colorbars.
            tick_labelsize (int): The font size for the colorbar tick labels, used to determine the label font size as well.
            labelpad (float): The padding between the colorbar and its label in points.

        Returns:
            None: This method does not return any value; it modifies the provided Colorbar object in-place by setting the label and tick positions according to the specified parameters, with error handling to ensure robustness against potential issues.
        """
        try:
            cbar.ax.yaxis.set_label_position(cast(Literal['left', 'right'], label_pos))
            ticks_pos = 'left' if label_pos == 'right' else 'right'
            cbar.ax.yaxis.set_ticks_position(ticks_pos)
        except Exception:
            pass
        if label is not None:
            try:
                cbar.set_label(label, fontsize=max(8, tick_labelsize), labelpad=labelpad)
            except Exception:
                cbar.ax.set_ylabel(label, fontsize=max(8, tick_labelsize), labelpad=labelpad)

    @staticmethod
    def _apply_colorbar_formatter(cbar: Colorbar, 
                                  fmt: Any) -> None:
        """
        This internal method applies a custom formatter to the colorbar tick labels based on the provided format specification. It attempts to set the colorbar's formatter using matplotlib's FormatStrFormatter if the provided format is a string, or directly if it is already a formatter object. The method includes error handling to ensure that if any issues arise while setting the formatter (e.g., due to incompatible types or matplotlib versions), it fails gracefully without crashing, allowing the colorbar to retain its default tick formatting. This approach ensures that users can customize the appearance of colorbar tick labels in MPAS diagnostic plots while maintaining robustness against potential configuration issues.

        Parameters:
            cbar (Colorbar): The Colorbar object to which the formatter will be applied.
            fmt (str or Any): The format specification for the colorbar tick labels. This can be a string format (e.g., '%d' for integers) or a custom formatter object compatible with matplotlib's colorbar formatting.

        Returns:
            None: This method does not return any value; it modifies the provided Colorbar object in-place by setting the tick label formatter according to the specified format, with error handling to ensure robustness against potential issues. If the formatter cannot be applied, the colorbar will retain its default tick formatting without crashing the program.
        """
        import matplotlib.ticker as mticker
        try:
            cbar.formatter = mticker.FormatStrFormatter(fmt) if isinstance(fmt, str) else fmt
            cbar.update_ticks()
        except Exception:
            pass

    @staticmethod
    def add_colorbar(fig: Figure,
                     ax: Optional[Axes],
                     mappable: Union[ScalarMappable, QuadContourSet, PathCollection, QuadMesh, LineCollection, AxesImage, Any],
                     label: Optional[str] = None,
                     orientation: str = 'horizontal',
                     fraction: float = 0.04,
                     pad: float = 0.06,
                     shrink: float = 0.8,
                     fmt: Union[str, None] = '%d',
                     labelpad: float = -50.0,
                     extend: Optional[str] = None,
                     label_pos: Union[Literal['top', 'bottom'], Literal['left', 'right']] = 'top',
                     tick_labelsize: int = 8) -> Optional[Colorbar]:
        """
        This method adds a colorbar to the provided matplotlib Figure and Axes, associated with the given mappable object (e.g., QuadMesh, ContourSet). It allows for extensive customization of the colorbar's appearance and formatting, including orientation (horizontal or vertical), size adjustments (fraction, pad, shrink), label configuration (text, font size, padding, position), and tick label formatting. The method first checks if the necessary parameters (figure, axes, mappable) are provided and returns early if any are missing. It then creates the colorbar using fig.colorbar with the specified parameters and applies the label and tick formatting based on the orientation. The method also includes error handling to ensure that if any issues arise while setting labels or formatting, it falls back to default behavior without crashing. Finally, it returns the created Colorbar object for further manipulation if needed. 

        Parameters:
            fig (Figure): The matplotlib Figure object to which the colorbar will be added.
            ax (Axes, optional): The Axes object to which the colorbar will be associated. This is typically the Axes containing the plot for which the colorbar is relevant. If None, the method will not add a colorbar and will return None.
            mappable (ScalarMappable, QuadContourSet, PathCollection, QuadMesh, LineCollection, AxesImage, Any): The mappable object (e.g., QuadMesh, ContourSet) for which the colorbar will be created. This is typically the result of a plotting function that returns a mappable object with an associated colormap and normalization.
            label (str, optional): The text label for the colorbar. If None, no label will be added.
            orientation (str): The orientation of the colorbar, either 'horizontal' or 'vertical'. Default is 'horizontal'.
            fraction (float): The fraction of the original axes to use for the colorbar. Default is 0.04.
            pad (float): The fraction of the original axes to use as padding between the plot and the colorbar. Default is 0.06.
            shrink (float): The fraction by which to shrink the colorbar. Default is 0.8.
            fmt (str or None): The format string for the colorbar tick labels (e.g., '%d' for integers). If None, default formatting will be used. Default is '%d'.
            labelpad (float): The padding between the colorbar and its label in points. Default is 6.0.
            label_pos (str): The position of the colorbar label. For horizontal colorbars, valid options are 'top' or 'bottom'. For vertical colorbars, valid options are 'left' or 'right'. Default is 'top' for horizontal and 'left' for vertical.
            extend (str, optional): The extend parameter for the colorbar, which can be 'neither', 'both', 'min', or 'max'. This controls whether the colorbar has extensions at the ends to indicate out-of-range values. Default is None.
            tick_labelsize (int): The font size for the colorbar tick labels. Default is 8.

        Returns:
            Optional[Colorbar]: The created Colorbar object if the colorbar was successfully added, or None if the necessary parameters were not provided. This allows for further customization of the colorbar after creation if needed. 
        """
        if fig is None or ax is None or mappable is None:
            return

        cbar = fig.colorbar(mappable, ax=ax, orientation=orientation,
                            fraction=fraction, pad=pad, shrink=shrink, extend=extend)

        # Set label and its position based on orientation
        if orientation.lower().startswith('h'):
            MPASVisualizationStyle._configure_horizontal_colorbar(
                cbar, label, label_pos, tick_labelsize, labelpad)
        else:
            MPASVisualizationStyle._configure_vertical_colorbar(
                cbar, label, label_pos, tick_labelsize, labelpad)

        cbar.ax.tick_params(labelsize=tick_labelsize)

        if fmt is not None:
            MPASVisualizationStyle._apply_colorbar_formatter(cbar, fmt)

        return cbar

    @staticmethod
    def build_colorbar_label(var_metadata: Optional[Dict[str, Any]] = None,
                             default_long_name: Optional[str] = None,
                             default_units: Optional[str] = None) -> Optional[str]:
        """
        This method constructs a colorbar label string based on variable metadata, with a fallback to default long name and units if metadata is not provided. It checks for the presence of 'long_name' and 'units' in the provided metadata dictionary, and if not found, it uses the default values if available. The method then formats the label in the form of "Long Name [units]" if both long name and units are present, or just "Long Name" or "[units]" if only one of them is available. If neither long name nor units can be determined, it returns None, indicating that no label can be constructed. This method provides a standardized way to generate informative colorbar labels for MPAS diagnostic plots based on available metadata and sensible defaults. 

        Parameters:
            var_metadata (Dict[str, Any], optional): A dictionary containing variable metadata, which may include keys like 'long_name' or 'longName' for the descriptive name of the variable, and 'units' or 'unit' for the measurement units. This metadata is typically extracted from the xarray DataArray attributes associated with the variable.
            default_long_name (str, optional): A default long name to use if the metadata does not contain a long name. This can be provided as a fallback to ensure that the colorbar label has a descriptive name even if metadata is incomplete.
            default_units (str, optional): A default units string to use if the metadata does not contain units. This can be provided as a fallback to ensure that the colorbar label includes units information even if metadata is incomplete. 

        Returns:
            Optional[str]: A formatted colorbar label string in the form of "Long Name [units]", "Long Name", or "[units]" based on the available metadata and defaults. If neither long name nor units can be determined, returns None to indicate that no label can be constructed. This allows for flexible labeling of colorbars in MPAS diagnostic plots while providing informative descriptions when possible. 
        """
        long_name = None
        units = None
        if isinstance(var_metadata, dict):
            long_name = var_metadata.get('long_name') or var_metadata.get('longName')
            units = var_metadata.get('units') or var_metadata.get('unit')

        if long_name is None and default_long_name is not None:
            long_name = default_long_name
        if units is None and default_units is not None:
            units = default_units

        if long_name:
            if units and f'[{units}]' in long_name:
                return long_name
            return f"{long_name} [{units}]" if units else long_name

        if units:
            return f"[{units}]"

        return None
    
    @staticmethod
    def format_latitude(value: float, _) -> str:
        """
        This method formats latitude coordinate values for axis tick labels with cardinal direction (N/S) suffix following geographic conventions. This method takes a latitude value in degrees, determines the appropriate cardinal direction (N for non-negative, S for negative values), computes the absolute value to remove sign, and returns a formatted string with one decimal place followed by degree symbol and direction letter (e.g., "45.0°N", "30.5°S"). The second parameter is required by matplotlib FuncFormatter signature but unused in this implementation, allowing the method to be passed directly to FuncFormatter for y-axis tick labeling on cartographic plots. 

        Parameters:
            value (float): Latitude coordinate value in degrees [-90, 90]. 
            _ (Any): Unused parameter required by matplotlib.ticker.FuncFormatter callback signature.

        Returns:
            str: Formatted latitude string with one decimal place and N/S cardinal direction (e.g., "45.0°N").
        """
        direction = 'N' if value >= 0 else 'S'
        return f"{abs(value):.1f}°{direction}"

    @staticmethod
    def format_longitude(value: float, _) -> str:
        """
        This method formats longitude coordinate values for axis tick labels with cardinal direction (E/W) suffix following geographic conventions. It takes a longitude value in degrees, determines the appropriate cardinal direction (E for non-negative, W for negative values), computes the absolute value to remove sign, and returns a formatted string with one decimal place followed by degree symbol and direction letter (e.g., "120.0°W", "75.5°E"). The second parameter is required by matplotlib FuncFormatter signature but unused in this implementation, allowing the method to be passed directly to FuncFormatter for x-axis tick labeling on cartographic plots. 

        Parameters:
            value (float): Longitude coordinate value in degrees [-180, 180].
            _ (Any): Unused parameter required by matplotlib.ticker.FuncFormatter callback signature. 

        Returns:
            str: Formatted longitude string with one decimal place and E/W cardinal direction (e.g., "120.0°W"). 
        """
        direction = 'E' if value >= 0 else 'W'
        return f"{abs(value):.1f}°{direction}"
    
    @staticmethod
    def calculate_adaptive_marker_size(map_extent: Tuple[float, float, float, float], 
                                     num_points: int, 
                                     fig_size: Tuple[float, float] = (12, 10)) -> float:
        """
        This method calculates an adaptive marker size for scatter plots based on the geographic area of the map extent and the density of data points. It takes into account the longitude and latitude bounds to compute the map area, and then determines the point density by dividing the number of points by the map area. Based on predefined density thresholds, it assigns a base marker size that decreases as density increases to prevent overcrowding. The method also applies scaling factors based on the map area and figure size to further adjust the marker size for optimal visibility. Finally, it clamps the computed marker size to a reasonable range to ensure that markers are neither too small nor too large, providing a balanced visual representation of data points across different map extents and densities in MPAS diagnostic plots. 

        Parameters:
            map_extent (Tuple[float, float, float, float]): A tuple containing the longitude and latitude bounds of the map in the form (lon_min, lon_max, lat_min, lat_max).
            num_points (int): The total number of data points to be plotted in the scatter plot.
            fig_size (Tuple[float, float]): The size of the figure in inches (width, height) used for scaling the marker size. Default is (12, 10). 
            
        Returns:
            float: The calculated marker size to be used in scatter plots, adjusted based on map extent, point density, and figure size. This size is clamped to a reasonable range to ensure visibility across different plotting scenarios. 
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

    @staticmethod
    def _choose_tick_fmt(t: np.ndarray, 
                         non_zero_t: np.ndarray) -> str:
        """
        This internal method determines the appropriate format string for axis tick labels based on the range and distribution of tick values. It first checks if all ticks are close to integers, in which case it returns a format string with no decimal places. If there are non-zero ticks, it calculates the typical magnitude using the median of the absolute values of the non-zero ticks and selects a format string with an appropriate number of decimal places based on predefined thresholds (e.g., 1 decimal for values between 10 and 100, 2 decimals for values between 0.01 and 10, etc.). If there are no non-zero ticks, it defaults to integer formatting. This method helps ensure that tick labels are formatted in a way that is both informative and visually clean across a wide range of data values in MPAS diagnostic plots.

        Parameters:
            t (np.ndarray): The array of tick values.
            non_zero_t (np.ndarray): The array of non-zero tick values.

        Returns:
            str: The format string for tick labels.
        """
        if np.allclose(t, np.round(t), atol=1e-6):
            return '{:.0f}'

        if len(non_zero_t) == 0:
            return '{:.0f}'

        typical_magnitude = float(np.median(np.abs(non_zero_t)))

        if typical_magnitude >= 100:  return '{:.0f}'
        if typical_magnitude >= 10:   return '{:.1f}'
        if typical_magnitude >= 0.01: return '{:.2f}'

        return '{:.3f}'

    @staticmethod
    def _resolve_duplicate_labels(fmt: str, 
                                  ticks: List[float]) -> List[str]:
        """
        This method resolves duplicate tick labels by increasing the decimal precision of the format string. It defines a mapping of format strings to their next higher precision version (e.g., '{:.0f}' to '{:.1f}', '{:.1f}' to '{:.2f}', etc.) and applies this upgrade to the provided format string. It then formats the tick values with the upgraded format and checks for duplicates. If duplicates are still present, it falls back to a general format that preserves significant digits (using 'g' formatting) to ensure unique labels. This method is used internally by the dynamic tick formatting logic to ensure that axis tick labels are informative and distinct, even when the initial formatting results in duplicates due to rounding.

        Parameters:
            fmt (str): The initial format string used for tick labels (e.g., '{:.0f}', '{:.1f}', etc.) that may have resulted in duplicate labels.
            ticks (List[float]): The list of tick values that are being formatted. This is used to check for duplicates after applying the upgraded format.

        Returns:
            List[str]: A list of formatted tick label strings with increased precision to resolve duplicates. If duplicates persist even after upgrading the format, it returns a list of labels formatted with 'g' to preserve significant digits and ensure uniqueness. This method helps maintain clarity in axis labeling by ensuring that each tick has a distinct label, even when the tick values are close together.
        """
        _next: Dict[str, str] = {'{:.0f}': '{:.1f}', '{:.1f}': '{:.2f}', '{:.2f}': '{:.3f}'}
        upgraded = _next.get(fmt, fmt)
        labels = [upgraded.format(x) for x in ticks]
        if len(set(labels)) < len(labels):
            labels = [f'{x:g}' for x in ticks]
        return labels

    @staticmethod
    def format_ticks_dynamic(ticks: List[float]) -> List[str]:
        """
        This method formats axis tick labels dynamically based on the range and distribution of tick values. It first checks for non-zero ticks to determine the maximum and minimum absolute values, which helps decide if scientific notation is needed for very large or small numbers. If the ticks are close to integers, it formats them with no decimal places. For other cases, it assesses the typical magnitude of the ticks to choose an appropriate number of decimal places (e.g., 1 decimal for values between 10 and 100, 2 decimals for values between 1 and 10, etc.). The method also checks for duplicate formatted labels and increases precision if necessary to ensure unique labels. If duplicates persist, it falls back to a general format that preserves significant digits. This adaptive formatting ensures that tick labels are both informative and visually clean across a wide range of data values in MPAS diagnostic plots. 

        Parameters:
            ticks (List[float]): A list of tick values to be formatted for axis labels. These values can vary widely in magnitude and may include integers, decimals, or very large/small numbers. 

        Returns:
            List[str]: A list of formatted tick label strings corresponding to the input tick values, formatted dynamically based on their range and distribution to ensure clarity and uniqueness in axis labeling. 
        """
        if not ticks:
            return []

        t = np.array(ticks)
        non_zero_t = t[t != 0]

        if len(non_zero_t) > 0:
            max_abs = np.max(np.abs(non_zero_t))
            min_abs = np.min(np.abs(non_zero_t))
            if max_abs >= 1e4 or min_abs < 1e-3:
                return [f'{x:.1e}' for x in ticks]

        if len(t) > 1:
            spacings = np.abs(np.diff(np.sort(t)))
            median_spacing = float(np.median(spacings[spacings > 0])) if np.any(spacings > 0) else 0.0
        else:
            median_spacing = 0.0

        fmt = MPASVisualizationStyle._choose_tick_fmt(t, non_zero_t)
        formatted_labels = [fmt.format(x) for x in ticks]

        if len(set(formatted_labels)) < len(formatted_labels):
            formatted_labels = MPASVisualizationStyle._resolve_duplicate_labels(fmt, ticks)

        return formatted_labels