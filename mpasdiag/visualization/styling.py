#!/usr/bin/env python3

"""
MPAS Visualization Styling and Colormaps

This module provides comprehensive visualization styling utilities for MPAS atmospheric variables including variable-specific colormaps, contour levels, and plot appearance parameters ensuring consistent professional presentation across all diagnostic plots. It implements the MPASVisualizationStyle class as a central repository for styling logic including precipitation-specific discrete colormaps with meteorological contour levels, automatic colormap and level selection based on variable names and data characteristics, coordinate axis formatters for latitude/longitude labels, map projection configuration, and publication-quality plot enhancement functions. The styling system supports both 2D surface fields and 3D atmospheric variables with level-specific configurations for temperature, winds, moisture, reflectivity, and other model diagnostics, maintains backward compatibility with original mpas_analysis visualization schemes, and provides flexible customization options. Core capabilities include intelligent colormap selection using pattern matching on variable names, data-driven contour level generation, adaptive marker sizing, and standardized file saving with optimized compression for high-throughput diagnostic workflows.

Classes:
    MPASVisualizationStyle: Comprehensive styling utility class providing variable-specific visualization parameters for MPAS diagnostics.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import cartopy.crs as ccrs
import os
import re
from datetime import datetime
from typing import Optional, Union, Tuple, List, Dict, Any

from ..processing.utils_metadata import MPASFileMetadata
from ..processing.utils_unit import UnitConverter


class MPASVisualizationStyle:
    """
    Comprehensive visualization styling utility class providing variable-specific colormaps, contour levels, and plot appearance parameters for MPAS atmospheric diagnostics. This class serves as a central repository for all visualization styling logic including precipitation-specific discrete colormaps with meteorological contour levels, automatic colormap and level selection based on variable names and data characteristics, coordinate axis formatters for latitude/longitude labels, map projection configuration utilities, and publication-quality plot enhancement functions (timestamping, branding, file saving). All methods are static utilities designed for use across multiple plotter classes (MPASVisualizer, MPASPrecipitationPlotter, MPASVerticalCrossSectionPlotter) to ensure consistent styling conventions, maintain backward compatibility with original mpas_analysis visualization schemes, and provide flexible customization options for specialized applications. The styling system supports both 2D surface fields and 3D atmospheric variables with level-specific configurations for temperature, winds, moisture, reflectivity, and other model output diagnostics.
    """
    
    @staticmethod
    def create_precip_colormap(accum: str = "a24h") -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        Creates a discrete precipitation colormap and associated contour levels optimized for meteorological precipitation visualization with accumulation-period-specific configurations. This method implements the exact precipitation color scheme logic from the original mpas_analysis visualization module using a fixed 11-color palette ranging from white (trace amounts) through blues and greens to reds, violets, and magenta (extreme precipitation), ensuring consistent precipitation display across all MPASdiag plots for backward compatibility. The accumulation period string is parsed to extract time duration (e.g., '24' from 'a24h'), which determines contour level spacing: periods ≥12 hours use levels [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150] mm suitable for daily totals, while shorter periods use finer levels [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75] mm appropriate for hourly rates. This method returns a matplotlib ListedColormap object and corresponding level array for use with BoundaryNorm in precipitation scatter plots and filled contours.

        Parameters:
            accum (str): Accumulation period identifier string (e.g., 'a24h', 'a01h', 'a06h') for contour level selection (default: "a24h").

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: Two-element tuple containing (discrete_colormap, contour_levels_list) for precipitation plotting.
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
    def get_variable_style(var_name: str, data_array: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        Retrieves comprehensive visualization styling parameters for MPAS atmospheric variables including colormap names, contour levels, colorbar extension modes, and normalization types. This method maintains a central style configuration dictionary mapping variable names to meteorologically-appropriate colormaps (e.g., RdYlBu_r for temperature, Blues for moisture, plasma for CAPE, pyart_NWSRef for reflectivity), handles precipitation variables specially by detecting accumulation periods and invoking create_precip_colormap, automatically generates data-driven contour levels when explicit levels are not configured, and applies sensible fallback defaults (viridis colormap, linear normalization) for unrecognized variables. The method supports both surface diagnostics (t2m, mslp, precip) and isobaric level fields (temperature_500hPa, uzonal_850hPa) with consistent styling conventions, ensuring publication-quality visualizations across the entire MPAS diagnostic suite.

        Parameters:
            var_name (str): MPAS variable name for style configuration lookup (case-insensitive).
            data_array (Optional[xr.DataArray]): Optional data array for automatic contour level generation from data statistics (default: None).

        Returns:
            Dict[str, Any]: Dictionary containing styling parameters with keys 'colormap', 'levels', 'extend', 'norm_type', 'alpha', 'interpolation'.
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
            accum_match = re.search(r'(\d+)h', var_lower)
            if accum_match:
                hours = int(accum_match.group(1))
                if hours == 1:
                    accum_period = 'a01h'
                elif hours == 3:
                    accum_period = 'a03h'
                elif hours == 6:
                    accum_period = 'a06h'
                elif hours == 12:
                    accum_period = 'a12h'
                elif hours == 24:
                    accum_period = 'a24h'
                else:
                    if hours < 3:
                        accum_period = 'a01h'
                    elif hours < 6:
                        accum_period = 'a03h'
                    elif hours < 12:
                        accum_period = 'a06h'
                    elif hours < 24:
                        accum_period = 'a12h'
                    else:
                        accum_period = 'a24h'
            elif 'daily' in var_lower:
                accum_period = 'a24h'
            elif 'hourly' in var_lower:
                accum_period = 'a01h'
            else:
                accum_period = 'a01h'
            
            cmap, levels = MPASVisualizationStyle.create_precip_colormap(accum_period)

            style['colormap'] = cmap
            style['levels'] = levels

            if 'use_precip_colormap' in style:
                del style['use_precip_colormap']
        
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
    def _generate_levels_from_data(data_array: xr.DataArray, var_name: str) -> Optional[List[float]]:
        """
        Generates appropriate contour levels automatically from data value distribution using percentile-based range detection and variable-specific level spacing strategies. This internal method flattens the data array, filters out non-finite values (NaN, inf), computes 5th and 95th percentiles to determine robust data range excluding outliers, and applies intelligent level generation based on variable type: temperature variables use rounded intervals (0.5°, 1°, 2°, or 5° spacing), while other variables use linear spacing with 15 levels spanning the data range. The method handles edge cases (empty data, zero range, all-NaN arrays) by returning None to trigger fallback level selection, ensuring robust automatic level generation for diverse MPAS output variables.

        Parameters:
            data_array (xr.DataArray): Data array for statistical analysis and level computation.
            var_name (str): Variable name providing context for variable-specific level spacing rules.
            
        Returns:
            Optional[List[float]]: Generated contour level list or None if level generation fails or data is invalid.
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
    def get_variable_specific_settings(var_name: str, data: np.ndarray) -> Tuple[Union[str, mcolors.ListedColormap], Optional[List[float]]]:
        """
        Generates variable-specific colormap and contour level recommendations based on meteorological conventions and data characteristics using comprehensive pattern matching and adaptive level spacing. This method implements intelligent defaults for common atmospheric variables by analyzing the variable name for keywords (temperature, precipitation, pressure, wind, geopotential, humidity) and computing data statistics to determine appropriate level ranges and spacings, similar to create_precip_colormap but extended to the full suite of meteorological diagnostics. Temperature fields use RdYlBu_r colormap with spacing adapted to temperature range (1° to 5° intervals), precipitation uses period-specific discrete colormaps from create_precip_colormap, pressure uses RdBu_r with scale-appropriate intervals (500 Pa for large-scale pressure or 5 hPa for sea-level pressure), wind uses plasma colormap with physically-meaningful speed thresholds, geopotential height uses terrain colormap with elevation-appropriate intervals, and humidity uses BuGn colormap with fractional or percentage levels. For unrecognized variables, the method applies symmetric RdBu_r colormap for bipolar data or viridis/plasma for unipolar data with linearly-spaced levels.

        Parameters:
            var_name (str): Variable name for pattern matching and styling selection (case-insensitive).
            data (np.ndarray): Data array or xarray.DataArray for statistics-based level computation.
            
        Returns:
            Tuple[Union[str, mcolors.ListedColormap], Optional[List[float]]]: Two-element tuple (colormap_name_or_object, contour_levels_list or None).
        """
        var_lower = var_name.lower().strip()
        
        if isinstance(data, xr.DataArray):
            data_values = data.values.flatten()
        else:
            data_values = np.asarray(data).flatten()
            
        valid_data = data_values[np.isfinite(data_values)]
        if len(valid_data) == 0:
            return 'viridis', None
            
        data_min, data_max = np.min(valid_data), np.max(valid_data)
        data_range = data_max - data_min
        
        if any(temp_key in var_lower for temp_key in ['temp', 't2m', 'temperature', 'sst']):
            colormap = 'RdYlBu_r'
            
            if data_range > 40:  
                step = 5
                levels = [float(x) for x in range(int(data_min//step)*step, int(data_max//step)*step + step*2, step)]
            elif data_range > 20:  
                step = 2
                levels = [float(x) for x in range(int(data_min//step)*step, int(data_max//step)*step + step*2, step)]
            else:  
                step = 1
                levels = [float(x) for x in range(int(data_min//step)*step, int(data_max//step)*step + step*2, step)]
            
            levels = [level for level in levels if data_min <= level <= data_max]
            
            return colormap, levels
            
        elif any(precip_key in var_lower for precip_key in ['precip', 'rain', 'snow', 'qpf']):
            if any(period in var_lower for period in ['01h', '1h', 'hourly']):
                accum_period = 'a01h'
            elif any(period in var_lower for period in ['03h', '3h']):
                accum_period = 'a03h'
            elif any(period in var_lower for period in ['06h', '6h']):
                accum_period = 'a06h'
            elif any(period in var_lower for period in ['12h']):
                accum_period = 'a12h'
            elif any(period in var_lower for period in ['24h', 'daily']):
                accum_period = 'a24h'
            else:
                accum_period = 'a24h'
            
            cmap, levels = MPASVisualizationStyle.create_precip_colormap(accum_period)
            
            if data_max > 0:
                levels = [level for level in levels if level <= data_max * 1.2]
            
            return cmap, levels
            
        elif any(press_key in var_lower for press_key in ['pressure', 'slp', 'mslp', 'pres']):
            colormap = 'RdBu_r'
            
            if 'slp' in var_lower or 'mslp' in var_lower:
                if data_range > 100:  
                    step = 500  
                else:  
                    step = 5
            else:
                if data_range > 1000:
                    step = max(100, int(data_range / 20))
                else:
                    step = max(10, int(data_range / 15))
            
            levels = [float(x) for x in range(int(data_min//step)*step, int(data_max//step)*step + step*2, step)]
            levels = [level for level in levels if data_min <= level <= data_max]
            
            return colormap, levels
            
        elif any(wind_key in var_lower for wind_key in ['wind', 'wspd', 'speed']):
            colormap = 'plasma'
            
            if data_max < 3: 
                levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
            elif data_max < 15:  
                levels = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
            elif data_max < 30: 
                levels = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
            else:  
                levels = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
            
            levels = [level for level in levels if level <= data_max * 1.1]
            
            return colormap, levels
            
        elif any(geop_key in var_lower for geop_key in ['geopotential', 'height', 'hgt', 'z']):
            colormap = 'terrain'
            
            height_range = data_max - data_min
            if height_range > 1000:  
                step = 60 if height_range > 2000 else 30
                levels = [float(x) for x in range(int(data_min//step)*step, int(data_max//step)*step + step*2, step)]
            else: 
                step = 20 if height_range > 200 else 10
                levels = [float(x) for x in range(int(data_min//step)*step, int(data_max//step)*step + step*2, step)]
            
            levels = [level for level in levels if data_min <= level <= data_max]
            
            return colormap, levels
            
        elif any(humid_key in var_lower for humid_key in ['humidity', 'rh', 'q', 'mixing']):
            colormap = 'BuGn'
            
            if data_max <= 1.1:  
                if 'rh' in var_lower or 'humidity' in var_lower:
                    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                else:  
                    levels = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            else:  
                levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            
            levels = [level for level in levels if data_min <= level <= data_max]
            
            return colormap, levels
            
        else:
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
    def setup_map_projection(lon_min: float, lon_max: float, 
                           lat_min: float, lat_max: float,
                           projection: str = 'PlateCarree') -> Tuple[ccrs.Projection, ccrs.PlateCarree]:
        """
        Configures Cartopy map projection and data coordinate reference system for geographic plotting with automatic central point calculation. This method computes the central longitude and latitude from the provided map extent bounds, instantiates the requested Cartopy projection object (PlateCarree for global/regional cylindrical equidistant, Mercator for conformal mapping, LambertConformal for mid-latitude conic projection) with appropriate central longitude/latitude parameters, and returns both the map projection for plot axes and PlateCarree data CRS for data coordinate transformation. The PlateCarree data CRS ensures MPAS lon/lat cell coordinates are correctly projected onto any map projection, enabling flexible visualization of MPAS unstructured mesh data on various map projections without data transformation errors.

        Parameters:
            lon_min (float): Western boundary of map extent in degrees.
            lon_max (float): Eastern boundary of map extent in degrees.
            lat_min (float): Southern boundary of map extent in degrees.
            lat_max (float): Northern boundary of map extent in degrees.
            projection (str): Projection name string ('PlateCarree', 'Mercator', 'LambertConformal') (default: 'PlateCarree').

        Returns:
            Tuple[ccrs.Projection, ccrs.PlateCarree]: Two-element tuple containing (map_projection_object, data_coordinate_system).
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
    
    @staticmethod
    def add_timestamp_and_branding(fig: Figure) -> None:
        """
        Adds timestamp and MPASdiag version branding annotation to the bottom-left corner of figure for provenance tracking and professional presentation. This method generates a current UTC timestamp string, attempts to import MPASdiag package version number from __version__ module attribute with fallback to default version string if import fails, and adds a small semi-transparent text annotation to the figure using figure-relative coordinates (0.02, 0.02) positioned just above the figure bottom margin. The branding text includes both version information and generation timestamp, providing essential provenance metadata for plots used in publications, presentations, and archival datasets.

        Parameters:
            fig (Figure): Matplotlib Figure object to annotate with timestamp and branding.

        Returns:
            None: Modifies the figure in-place by adding text annotation.
        """
        if fig is not None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            
            try:
                from .. import __version__
                version_str = f"v{__version__}"
            except ImportError:
                version_str = "v1.1.2"  
            
            fig.text(0.02, 0.02, f'Generated with MPASdiag {version_str} on: {timestamp}', 
                         fontsize=8, alpha=0.7, transform=fig.transFigure)
    
    @staticmethod
    def save_plot(fig: Figure, 
                  output_path: str, 
                  formats: List[str] = ['png'],
                  bbox_inches: str = 'tight',
                  pad_inches: float = 0.1,
                  dpi: int = 100) -> None:
        """
        Saves matplotlib figure to file(s) in multiple output formats with optimized compression settings for performance and quality balance. This method creates output directory if necessary, iterates through requested formats to save separate files with format-specific extensions, applies fast PNG compression (level 1 instead of default 6-9) for 5-10x faster file I/O with only 10-20% larger file sizes, uses tight bounding box mode to minimize whitespace around plots, and prints confirmation messages with full file paths. The method supports all matplotlib-savefig formats (png, pdf, svg, eps, jpg) and is designed for high-throughput diagnostic plotting where I/O performance matters more than maximum compression ratio.

        Parameters:
            fig (Figure): Matplotlib Figure object to save to disk.
            output_path (str): Base output file path without extension (e.g., '/path/to/plot').
            formats (List[str]): Output format list (extensions like 'png', 'pdf', 'svg') (default: ['png']).
            bbox_inches (str): Bounding box mode for tight layout ('tight' or standard) (default: 'tight').
            pad_inches (float): Padding space in inches around figure when using tight bbox (default: 0.1).
            dpi (int): Output resolution in dots-per-inch for raster formats (default: 300).

        Returns:
            None: Writes file(s) to disk and prints save confirmation messages.

        Raises:
            ValueError: If figure is None (no figure to save).
        """
        if fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_path):
            os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            full_path = f"{output_path}.{fmt}"
            save_kwargs = {'dpi': dpi, 'bbox_inches': bbox_inches, 'pad_inches': pad_inches, 'format': fmt}
            if fmt.lower() == 'png':
                save_kwargs['pil_kwargs'] = {'compress_level': 1}
            fig.savefig(full_path, **save_kwargs)
            print(f"Saved plot: {full_path}")

    @staticmethod
    def get_3d_variable_style(var_name: str, level: Optional[Union[str, float]] = None,
                             data_array: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        Placeholder method for retrieving styling parameters specific to 3D atmospheric variables at specified vertical levels for future 3D visualization capabilities. This method is reserved for future implementation of level-specific styling configurations where colormap and contour level choices depend on both variable type and vertical level (e.g., different temperature colormap ranges for upper troposphere vs lower troposphere, or different wind speed levels for jet stream levels vs surface layers). The method signature accepts variable name, vertical level specification (pressure in hPa, model level index, or height string), and optional data array for statistics-based styling, following the same pattern as get_variable_style but extended to the vertical dimension. Currently raises NotImplementedError to indicate the feature is planned but not yet available in this release.

        Parameters:
            var_name (str): 3D atmospheric variable name (e.g., 'temperature', 'uzonal', 'relhum').
            level (Optional[Union[str, float]]): Vertical level specification (pressure level in hPa, model level index, or height string) (default: None).
            data_array (Optional[xr.DataArray]): Optional data array for automatic level-specific styling detection (default: None).
            
        Returns:
            Dict[str, Any]: Dictionary containing styling parameters with keys 'colormap', 'levels', 'extend' (future implementation).
            
        Raises:
            NotImplementedError: Always raised as this feature is not yet implemented in current version.
        """
        raise NotImplementedError(
            "3D variable support not yet implemented. "
            "This function is a placeholder for future development."
        )
    
    @staticmethod
    def format_latitude(value: float, _) -> str:
        """
        Formats latitude coordinate values for axis tick labels with cardinal direction (N/S) suffix following geographic conventions. This method takes a latitude value in degrees, determines the appropriate cardinal direction (N for non-negative, S for negative values), computes the absolute value to remove sign, and returns a formatted string with one decimal place followed by degree symbol and direction letter (e.g., "45.0°N", "33.5°S"). The second parameter is required by matplotlib FuncFormatter signature but unused in this implementation, allowing the method to be passed directly to FuncFormatter for y-axis tick labeling.

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
        Formats longitude coordinate values for axis tick labels with cardinal direction (E/W) suffix following geographic conventions. This method takes a longitude value in degrees, determines the appropriate cardinal direction (E for non-negative, W for negative values), computes the absolute value to remove sign, and returns a formatted string with one decimal place followed by degree symbol and direction letter (e.g., "120.0°W", "75.5°E"). The second parameter is required by matplotlib FuncFormatter signature but unused in this implementation, allowing the method to be passed directly to FuncFormatter for x-axis tick labeling on cartographic plots.

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
                                     num_points: int, fig_size: Tuple[float, float] = (12, 10)) -> float:
        """
        Calculates adaptive marker size for MPAS unstructured mesh scatter plots based on map extent, data point density, and figure dimensions to optimize visual clarity. This method computes map area from extent bounds, calculates point density (points per square degree), and applies multi-stage scaling using density-based size selection (8.0 for sparse data to 0.25 for dense data), area-based adjustment factors (larger markers for small regional domains, smaller for large continental/global domains), and figure-size normalization to maintain consistent appearance across different plot sizes. The final marker size is clamped between 0.1 and 20.0 to prevent pathological cases, and diagnostic information is printed showing area, density, scaling factors, and final size for transparency and debugging.

        Parameters:
            map_extent (Tuple[float, float, float, float]): Geographic bounds as (lon_min, lon_max, lat_min, lat_max) in degrees.
            num_points (int): Total number of data points to plot on the map.
            fig_size (Tuple[float, float]): Figure dimensions in inches as (width, height) (default: (12, 10)).
            
        Returns:
            float: Computed marker size value for matplotlib scatter plot (clamped to [0.1, 20.0] range).
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
    def format_ticks_dynamic(ticks: List[float]) -> List[str]:
        """
        Generates formatted tick label strings with appropriate precision based on value magnitude and distribution using intelligent heuristics. This method analyzes the tick value array to determine optimal formatting: scientific notation (1.2e+03) for very large (≥1e4) or very small (<1e-3) values, integer formatting for values close to whole numbers, and decimal place selection based on typical magnitude (0 decimals for 100-999, 1 decimal for 10-99, 2 decimals for 1-9 and 0.1-0.99, 3 decimals for 0.001-0.0099). The method includes duplicate detection and automatic precision increase to ensure all tick labels are distinguishable, preventing overlapping labels on axes with small value ranges.

        Parameters:
            ticks (List[float]): List of tick values from matplotlib axis for formatting.

        Returns:
            List[str]: List of formatted tick label strings with magnitude-appropriate precision.
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