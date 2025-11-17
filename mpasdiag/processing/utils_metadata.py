#!/usr/bin/env python3

"""
MPAS Variable Metadata Management

This module provides comprehensive metadata management for MPAS atmospheric model variables including physical units, descriptive names, dimensional information, and optional visualization properties. It implements the MPASFileMetadata class as a centralized registry maintaining attribute information for MPAS model output variables including surface diagnostics (2-meter temperature, sea-level pressure, precipitation), upper-air fields (temperature, humidity, winds at pressure levels), microphysics variables (mixing ratios, reflectivity, cloud fractions), and derived quantities (CAPE, helicity, vorticity). The metadata system ensures consistent variable identification and unit handling across all MPASdiag processing and visualization modules, provides lookup methods that return standardized units and long names with fallback to xarray dataset attributes when available, supports integration with UnitConverter for automatic conversion between model output and display units, and optionally includes visualization-specific properties like colormaps and contour levels when needed for plotting workflows. Core capabilities include extensible variable registry covering comprehensive MPAS diagnostic suites, robust fallback handling for unrecognized variables, integration with xarray metadata conventions, and centralized management ensuring consistency across the entire MPASdiag toolkit.

Classes:
    MPASFileMetadata: Centralized metadata management class providing variable attribute lookup and registry services for MPAS model output.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Optional, Dict, Any, Union, List, Tuple

from mpasdiag.processing.utils_unit import UnitConverter


class MPASFileMetadata:
    """
    Metadata management class for MPAS variable attributes and data processing parameters with centralized registry of variable properties. This class provides lookup functionality for MPAS model output variables including physical units, descriptive long names, spatial dimension information, and optional visualization properties like colormaps and contour levels. The metadata registry covers surface diagnostics (temperature, pressure, precipitation), upper-air variables (temperature, humidity, winds at pressure levels), microphysics fields (mixing ratios, reflectivity), and derived quantities (CAPE, helicity, vorticity). The class is focused on data content attributes rather than purely visualization-specific properties, though it can optionally include plotting metadata when requested. This centralized approach ensures consistency across all analysis and visualization operations in the MPASdiag toolkit.
    """
    
    @staticmethod
    def get_variable_metadata(var_name: str, data_array: Optional[xr.DataArray] = None, 
                            include_visualization: bool = False) -> Dict[str, Any]:
        """
        Retrieve comprehensive metadata for MPAS variables including units, descriptive names, and optional visualization properties. This method looks up variable information from a centralized registry of MPAS model output variables, returning standardized units, human-readable long names, and spatial dimension information. When a data array is provided, it extracts additional attributes directly from the xarray object. The method supports optional inclusion of visualization-specific metadata like colormaps and contour levels for plotting workflows. This centralized metadata management ensures consistency across all analysis and visualization operations in the MPASdiag toolkit.

        Parameters:
            var_name (str): MPAS variable name to look up metadata for (e.g., 't2m', 'rainnc', 'mslp').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing variable attributes to extract (default: None).
            include_visualization (bool): If True, include visualization properties like colormap and contour levels in returned metadata (default: False).

        Returns:
            dict: Metadata dictionary with keys 'units', 'long_name', 'spatial_dim', and optionally 'colormap' and 'levels' if include_visualization is True.
        """
        standard_metadata = {
            'olrtoa': {'units': 'W/m^2', 'long_name': 'Outgoing Longwave Radiation at TOA'},
            
            'rainc': {'units': 'mm', 'long_name': 'Convective Precipitation'},  
            'rainnc': {'units': 'mm', 'long_name': 'Non-Convective Precipitation'},  
            'precipw': {'units': 'mm', 'long_name': 'Precipitable Water'}, 

            'refl10cm_max': {'units': 'dBZ', 'long_name': 'Maximum 10cm Reflectivity'},
            'refl10cm_1km': {'units': 'dBZ', 'long_name': '10cm Reflectivity at 1km AGL'},
            'refl10cm_1km_max': {'units': 'dBZ', 'long_name': 'Maximum 10cm Reflectivity at 1km AGL'},
            
            't2m': {'units': 'K', 'long_name': '2-meter Temperature'}, 
            'th2m': {'units': 'K', 'long_name': '2-meter Potential Temperature'}, 
            'u10': {'units': 'm/s', 'long_name': '10-meter U-wind'},
            'v10': {'units': 'm/s', 'long_name': '10-meter V-wind'}, 
            'q2': {'units': 'g/kg', 'long_name': '2-meter Specific Humidity'}, 
            'mslp': {'units': 'Pa', 'long_name': 'Mean Sea Level Pressure'},
            
            'relhum_50hpa': {'units': '%', 'long_name': 'Relative Humidity at 50 hPa'},
            'relhum_100hpa': {'units': '%', 'long_name': 'Relative Humidity at 100 hPa'},
            'relhum_200hpa': {'units': '%', 'long_name': 'Relative Humidity at 200 hPa'},
            'relhum_250hpa': {'units': '%', 'long_name': 'Relative Humidity at 250 hPa'},
            'relhum_500hpa': {'units': '%', 'long_name': 'Relative Humidity at 500 hPa'},
            'relhum_700hpa': {'units': '%', 'long_name': 'Relative Humidity at 700 hPa'},
            'relhum_850hpa': {'units': '%', 'long_name': 'Relative Humidity at 850 hPa'},
            'relhum_925hpa': {'units': '%', 'long_name': 'Relative Humidity at 925 hPa'},
            
            'temperature_50hpa': {'units': 'K', 'long_name': 'Temperature at 50 hPa'},
            'temperature_100hpa': {'units': 'K', 'long_name': 'Temperature at 100 hPa'},
            'temperature_200hpa': {'units': 'K', 'long_name': 'Temperature at 200 hPa'},
            'temperature_250hpa': {'units': 'K', 'long_name': 'Temperature at 250 hPa'},
            'temperature_500hpa': {'units': 'K', 'long_name': 'Temperature at 500 hPa'},
            'temperature_700hpa': {'units': 'K', 'long_name': 'Temperature at 700 hPa'},
            'temperature_850hpa': {'units': 'K', 'long_name': 'Temperature at 850 hPa'},  
            'temperature_925hpa': {'units': 'K', 'long_name': 'Temperature at 925 hPa'}, 
            
            'height_50hpa': {'units': 'm', 'long_name': 'Geopotential Height at 50 hPa'},
            'height_100hpa': {'units': 'm', 'long_name': 'Geopotential Height at 100 hPa'},
            'height_200hpa': {'units': 'm', 'long_name': 'Geopotential Height at 200 hPa'},
            'height_250hpa': {'units': 'm', 'long_name': 'Geopotential Height at 250 hPa'},
            'height_500hpa': {'units': 'm', 'long_name': 'Geopotential Height at 500 hPa'},
            'height_700hpa': {'units': 'm', 'long_name': 'Geopotential Height at 700 hPa'},
            'height_850hpa': {'units': 'm', 'long_name': 'Geopotential Height at 850 hPa'},
            'height_925hpa': {'units': 'm', 'long_name': 'Geopotential Height at 925 hPa'},

            'uzonal_50hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 50 hPa'},
            'uzonal_100hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 100 hPa'},
            'uzonal_200hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 200 hPa'},
            'uzonal_250hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 250 hPa'},
            'uzonal_500hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 500 hPa'},
            'uzonal_700hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 700 hPa'},
            'uzonal_850hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 850 hPa'},
            'uzonal_925hpa': {'units': 'm/s', 'long_name': 'Zonal Wind at 925 hPa'},

            'umeridional_50hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 50 hPa'},
            'umeridional_100hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 100 hPa'},
            'umeridional_200hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 200 hPa'},
            'umeridional_250hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 250 hPa'},
            'umeridional_500hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 500 hPa'},
            'umeridional_700hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 700 hPa'},
            'umeridional_850hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 850 hPa'},
            'umeridional_925hpa': {'units': 'm/s', 'long_name': 'Meridional Wind at 925 hPa'},

            'vorticity_50hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 50 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_100hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 100 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_200hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 200 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_250hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 250 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_500hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 500 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_700hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 700 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_850hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 850 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_925hpa': {'units': '1/s', 'long_name': 'Relative Vorticity at 925 hPa', 'spatial_dim': 'nVertices'},

            'cape': {'units': 'J/kg', 'long_name': 'Convective Available Potential Energy'},
            'cin': {'units': 'J/kg', 'long_name': 'Convective Inhibition'},
            'lcl': {'units': 'm', 'long_name': 'Lifting Condensation Level'},
            'lfc': {'units': 'm', 'long_name': 'Level of Free Convection'},
            'srh_0_1km': {'units': 'm^2/s^2', 'long_name': 'Storm Relative Helicity 0-1km'},
            'srh_0_3km': {'units': 'm^2/s^2', 'long_name': 'Storm Relative Helicity 0-3km'},

            'updraft_helicity_max': {'units': 'm^2/s^2', 'long_name': 'Maximum Updraft Helicity'},
            'w_velocity_max': {'units': 'm/s', 'long_name': 'Maximum Vertical Velocity'},
            'wind_speed_level1_max': {'units': 'm/s', 'long_name': 'Maximum Wind Speed at Level 1'},

            't_oml': {'units': 'K', 'long_name': 'Ocean Mixed Layer Temperature'},
            'h_oml': {'units': 'm', 'long_name': 'Ocean Mixed Layer Depth'},
            'hu_oml': {'units': 'm^2/s', 'long_name': 'Ocean Mixed Layer U-momentum'},
            'hv_oml': {'units': 'm^2/s', 'long_name': 'Ocean Mixed Layer V-momentum'},

            'cldfrac_low_upp': {'units': '', 'long_name': 'Low Cloud Fraction'},
            'cldfrac_mid_upp': {'units': '', 'long_name': 'Mid Cloud Fraction'},
            'cldfrac_high_upp': {'units': '', 'long_name': 'High Cloud Fraction'},
            'cldfrac_tot_upp': {'units': '', 'long_name': 'Total Cloud Fraction'},

            'refl10cm': {'units': 'dBZ', 'long_name': '10cm Reflectivity', 'spatial_dim': 'nCells'},
            'qv': {'units': 'kg kg^{-1}', 'long_name': 'Water Vapor Mixing Ratio', 'spatial_dim': 'nCells'},
            'qc': {'units': 'kg kg^{-1}', 'long_name': 'Cloud Water Mixing Ratio', 'spatial_dim': 'nCells'},
            'qr': {'units': 'kg kg^{-1}', 'long_name': 'Rain Water Mixing Ratio', 'spatial_dim': 'nCells'},
            'qi': {'units': 'kg kg^{-1}', 'long_name': 'Ice Mixing Ratio', 'spatial_dim': 'nCells'},
            'qs': {'units': 'kg kg^{-1}', 'long_name': 'Snow Mixing Ratio', 'spatial_dim': 'nCells'},
            'qg': {'units': 'kg kg^{-1}', 'long_name': 'Graupel Mixing Ratio', 'spatial_dim': 'nCells'},
            'ni': {'units': '1/kg', 'long_name': 'Ice Number Concentration', 'spatial_dim': 'nCells'},
            'u': {'units': 'm/s', 'long_name': 'Horizontal Velocity (U-component)', 'spatial_dim': 'nEdges'},
            'w': {'units': 'm/s', 'long_name': 'Vertical Velocity', 'spatial_dim': 'nVertices'},
            'rho': {'units': 'kg/m^3', 'long_name': 'Density', 'spatial_dim': 'nCells'},
            'pressure_p': {'units': 'Pa', 'long_name': 'Pressure Perturbation', 'spatial_dim': 'nCells'},
            'pressure_base': {'units': 'Pa', 'long_name': 'Base State Pressure', 'spatial_dim': 'nCells'},
            'theta': {'units': 'K', 'long_name': 'Potential Temperature', 'spatial_dim': 'nCells'},
            'relhum': {'units': '%', 'long_name': 'Relative Humidity', 'spatial_dim': 'nCells'},
            'rho_base': {'units': 'kg/m^3', 'long_name': 'Base State Density', 'spatial_dim': 'nCells'},
            'theta_base': {'units': 'K', 'long_name': 'Base State Potential Temperature', 'spatial_dim': 'nCells'},
            'uReconstructZonal': {'units': 'm/s', 'long_name': 'Reconstructed Zonal Wind', 'spatial_dim': 'nCells'},
            'uReconstructMeridional': {'units': 'm/s', 'long_name': 'Reconstructed Meridional Wind', 'spatial_dim': 'nCells'},
            'cldfrac': {'units': '', 'long_name': 'Cloud Fraction', 'spatial_dim': 'nCells'},
            're_cloud': {'units': 'microns', 'long_name': 'Cloud Droplet Effective Radius', 'spatial_dim': 'nCells'},
            're_ice': {'units': 'microns', 'long_name': 'Ice Crystal Effective Radius', 'spatial_dim': 'nCells'},
            're_snow': {'units': 'microns', 'long_name': 'Snow Effective Radius', 'spatial_dim': 'nCells'},
            'sh2o': {'units': 'm^3/m^3', 'long_name': 'Soil Moisture', 'spatial_dim': 'nCells'},
            'smois': {'units': 'm^3/m^3', 'long_name': 'Soil Moisture Content', 'spatial_dim': 'nCells'},
            'tslb': {'units': 'K', 'long_name': 'Soil Temperature', 'spatial_dim': 'nCells'},
        }
        
        visualization_metadata = {
            't2m': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(16, 32, 2)),
                'spatial_dims': 2
            },
            'temperature': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-40, 45, 5)),
                'spatial_dims': 2
            },
            'surface_temperature': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-40, 45, 5)),
                'spatial_dims': 2
            },
            'surface_pressure': {
                'colormap': 'viridis',
                'levels': list(range(960, 1050, 4)),
                'spatial_dims': 2
            },
            'mslp': {
                'colormap': 'viridis',
                'levels': list(range(960, 1050, 4)),
                'spatial_dims': 2
            },
            'pressure': {
                'colormap': 'viridis',
                'levels': list(range(960, 1050, 4)),
                'spatial_dims': 2
            },
            'rainnc': {
                'colormap': 'Blues',
                'levels': [0, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50],
                'spatial_dims': 2
            },
            'rainc': {
                'colormap': 'Blues',
                'levels': [0, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50],
                'spatial_dims': 2
            },
            'precipitation': {
                'colormap': 'Blues',
                'levels': [0, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50],
                'spatial_dims': 2
            },
            'u10': {
                'colormap': 'RdBu_r',
                'levels': list(range(-20, 25, 2)),
                'spatial_dims': 2
            },
            'v10': {
                'colormap': 'RdBu_r',
                'levels': list(range(-20, 25, 2)),
                'spatial_dims': 2
            },
            'wspd10': {
                'colormap': 'plasma',
                'levels': list(range(0, 25, 2)),
                'spatial_dims': 2
            },
            'wind_speed': {
                'colormap': 'plasma',
                'levels': list(range(0, 25, 2)),
                'spatial_dims': 2
            },
            'q2': {
                'colormap': 'BuGn',
                'levels': list(range(14, 22, 1)),
                'spatial_dims': 2
            },
            'rh2': {
                'colormap': 'BuGn',
                'levels': list(range(0, 105, 5)),
                'spatial_dims': 2
            },
            'th2m': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-40, 45, 5)),
                'spatial_dims': 2
            },
            
            'temperature_50hpa': {
                'colormap': 'RdYlBu_r', 
                'levels': list(range(-80, -40, 2)),
                'spatial_dims': 2
            },
            'temperature_100hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-80, -40, 2)), 
                'spatial_dims': 2
            },
            'temperature_200hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-70, -30, 2)),
                'spatial_dims': 2
            },
            'temperature_250hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-60, -20, 2)),
                'spatial_dims': 2
            },
            'temperature_500hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-40, 10, 2)),
                'spatial_dims': 2
            },
            'temperature_700hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-20, 20, 2)),
                'spatial_dims': 2
            },
            'temperature_850hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-10, 30, 2)),
                'spatial_dims': 2
            },
            'temperature_925hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-5, 35, 2)),
                'spatial_dims': 2
            },
            
            'relhum_50hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(0, 105, 5)),
                'spatial_dims': 2
            },
            'relhum_100hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(0, 105, 5)),
                'spatial_dims': 2
            },
            'relhum_200hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(100, 132, 2)),
                'spatial_dims': 2
            },
            'relhum_250hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(100, 132, 2)),
                'spatial_dims': 2
            },
            'relhum_500hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(70, 112, 2)),
                'spatial_dims': 2
            },
            'relhum_700hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(60, 102, 2)),
                'spatial_dims': 2
            },
            'relhum_850hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(60, 102, 2)),
                'spatial_dims': 2
            },
            'relhum_925hpa': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(60, 102, 2)),
                'spatial_dims': 2
            },
            
            'precipw': {
                'colormap': 'Blues',
                'levels': [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80],
                'spatial_dims': 2
            },
            
            'refl10cm_max': {
                'colormap': 'pyart_NWSRef',
                'levels': list(range(-10, 75, 5)),
                'spatial_dims': 2
            },
            'refl10cm_1km': {
                'colormap': 'pyart_NWSRef',
                'levels': list(range(-10, 75, 5)),
                'spatial_dims': 2
            },
            'refl10cm_1km_max': {
                'colormap': 'pyart_NWSRef',
                'levels': list(range(-10, 75, 5)),
                'spatial_dims': 2
            },
            
            'height_50hpa': {
                'colormap': 'viridis',
                'levels': list(range(19000, 21500, 100)),
                'spatial_dims': 2
            },
            'height_100hpa': {
                'colormap': 'viridis',
                'levels': list(range(15500, 17000, 50)),
                'spatial_dims': 2
            },
            'height_200hpa': {
                'colormap': 'viridis',
                'levels': list(range(11000, 13000, 100)),
                'spatial_dims': 2
            },
            'height_250hpa': {
                'colormap': 'viridis',
                'levels': list(range(9500, 11500, 100)),
                'spatial_dims': 2
            },
            'height_500hpa': {
                'colormap': 'viridis',
                'levels': list(range(5000, 6000, 40)),
                'spatial_dims': 2
            },
            'height_700hpa': {
                'colormap': 'viridis',
                'levels': list(range(2800, 3400, 30)),
                'spatial_dims': 2
            },
            'height_850hpa': {
                'colormap': 'viridis',
                'levels': list(range(1200, 1800, 30)),
                'spatial_dims': 2
            },
            'height_925hpa': {
                'colormap': 'viridis',
                'levels': list(range(400, 1000, 30)),
                'spatial_dims': 2
            },
            
            'uzonal_50hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-60, 65, 5)),
                'spatial_dims': 2
            },
            'uzonal_100hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-60, 65, 5)),
                'spatial_dims': 2
            },
            'uzonal_200hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-50, 55, 5)),
                'spatial_dims': 2
            },
            'uzonal_250hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-50, 55, 5)),
                'spatial_dims': 2
            },
            'uzonal_500hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-40, 45, 5)),
                'spatial_dims': 2
            },
            'uzonal_700hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-30, 35, 5)),
                'spatial_dims': 2
            },
            'uzonal_850hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-25, 30, 2)),
                'spatial_dims': 2
            },
            'uzonal_925hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-25, 30, 2)),
                'spatial_dims': 2
            },
            
            'umeridional_50hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-30, 35, 2)),
                'spatial_dims': 2
            },
            'umeridional_100hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-30, 35, 2)),
                'spatial_dims': 2
            },
            'umeridional_200hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-25, 30, 2)),
                'spatial_dims': 2
            },
            'umeridional_250hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-25, 30, 2)),
                'spatial_dims': 2
            },
            'umeridional_500hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-20, 25, 2)),
                'spatial_dims': 2
            },
            'umeridional_700hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-15, 20, 2)),
                'spatial_dims': 2
            },
            'umeridional_850hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-15, 20, 2)),
                'spatial_dims': 2
            },
            'umeridional_925hpa': {
                'colormap': 'RdBu_r',
                'levels': list(range(-15, 20, 2)),
                'spatial_dims': 2
            },
            
            'vorticity_50hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            'vorticity_100hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            'vorticity_200hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            'vorticity_250hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            'vorticity_500hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            'vorticity_700hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            'vorticity_850hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            'vorticity_925hpa': {
                'colormap': 'RdBu_r',
                'levels': [-2e-4, -1.5e-4, -1e-4, -5e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4],
                'spatial_dims': 2
            },
            
            'cape': {
                'colormap': 'plasma',
                'levels': [0, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
                'spatial_dims': 2
            },
            'cin': {
                'colormap': 'viridis_r',
                'levels': [0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500],
                'spatial_dims': 2
            },
            'lcl': {
                'colormap': 'viridis',
                'levels': list(range(0, 3500, 250)),
                'spatial_dims': 2
            },
            'lfc': {
                'colormap': 'viridis',
                'levels': list(range(0, 5000, 250)),
                'spatial_dims': 2
            },
            'srh_0_1km': {
                'colormap': 'RdYlBu_r',
                'levels': [-200, -150, -100, -75, -50, -25, 0, 25, 50, 75, 100, 150, 200, 300, 400],
                'spatial_dims': 2
            },
            'srh_0_3km': {
                'colormap': 'RdYlBu_r',
                'levels': [-400, -300, -200, -150, -100, -50, 0, 50, 100, 150, 200, 300, 400, 500, 600],
                'spatial_dims': 2
            },
            'updraft_helicity_max': {
                'colormap': 'plasma',
                'levels': [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000],
                'spatial_dims': 2
            },
            'w_velocity_max': {
                'colormap': 'plasma',
                'levels': [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100],
                'spatial_dims': 2
            },
            'wind_speed_level1_max': {
                'colormap': 'plasma',
                'levels': list(range(0, 60, 5)),
                'spatial_dims': 2
            },
            
            't_oml': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-2, 32, 2)),
                'spatial_dims': 2
            },
            'h_oml': {
                'colormap': 'viridis',
                'levels': [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500],
                'spatial_dims': 2
            },
            'hu_oml': {
                'colormap': 'RdBu_r',
                'levels': list(range(-50, 55, 5)),
                'spatial_dims': 2
            },
            'hv_oml': {
                'colormap': 'RdBu_r',
                'levels': list(range(-50, 55, 5)),
                'spatial_dims': 2
            },
            
            'cldfrac_low_upp': {
                'colormap': 'gray',
                'levels': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'spatial_dims': 2
            },
            'cldfrac_mid_upp': {
                'colormap': 'gray',
                'levels': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'spatial_dims': 2
            },
            'cldfrac_high_upp': {
                'colormap': 'gray',
                'levels': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'spatial_dims': 2
            },
            'cldfrac_tot_upp': {
                'colormap': 'gray',
                'levels': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'spatial_dims': 2
            },
            
            'olrtoa': {
                'colormap': 'viridis',
                'levels': list(range(120, 320, 10)),
                'spatial_dims': 2
            },
            
            'refl10cm': {
                'colormap': 'pyart_NWSRef',
                'levels': list(range(-10, 75, 5)),
                'spatial_dims': 3
            },
            'qv': {
                'colormap': 'BuGn',
                'levels': [0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 25, 30],
                'spatial_dims': 3
            },
            'qc': {
                'colormap': 'Blues',
                'levels': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20],
                'spatial_dims': 3
            },
            'qr': {
                'colormap': 'plasma',
                'levels': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20],
                'spatial_dims': 3
            },
            'qi': {
                'colormap': 'viridis',
                'levels': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20],
                'spatial_dims': 3
            },
            'qs': {
                'colormap': 'Greys',
                'levels': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20],
                'spatial_dims': 3
            },
            'qg': {
                'colormap': 'inferno',
                'levels': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20],
                'spatial_dims': 3
            },
            'ni': {
                'colormap': 'plasma',
                'levels': [1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11],
                'spatial_dims': 3
            },
            'u': {
                'colormap': 'RdBu_r',
                'levels': list(range(-50, 55, 5)),
                'spatial_dims': 3
            },
            'w': {
                'colormap': 'RdBu_r',
                'levels': [-20, -15, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 15, 20],
                'spatial_dims': 3
            },
            'rho': {
                'colormap': 'viridis',
                'levels': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                'spatial_dims': 3
            },
            'pressure_p': {
                'colormap': 'RdBu_r',
                'levels': [-50, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 50],
                'spatial_dims': 3
            },
            'pressure_base': {
                'colormap': 'viridis',
                'levels': [100, 200, 300, 400, 500, 600, 700, 800, 850, 900, 950, 1000],
                'spatial_dims': 3
            },
            'theta': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-40, 45, 5)),
                'spatial_dims': 3
            },
            'relhum': {
                'colormap': 'BuGn',
                'levels': list(range(0, 105, 5)),
                'spatial_dims': 3
            },
            'rho_base': {
                'colormap': 'viridis',
                'levels': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                'spatial_dims': 3
            },
            'theta_base': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-40, 45, 5)),
                'spatial_dims': 3
            },
            'uReconstructZonal': {
                'colormap': 'RdBu_r',
                'levels': list(range(-50, 55, 5)),
                'spatial_dims': 3
            },
            'uReconstructMeridional': {
                'colormap': 'RdBu_r',
                'levels': list(range(-30, 35, 2)),
                'spatial_dims': 3
            },
            'cldfrac': {
                'colormap': 'gray',
                'levels': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'spatial_dims': 3
            },
            're_cloud': {
                'colormap': 'Blues',
                'levels': [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40],
                'spatial_dims': 3
            },
            're_ice': {
                'colormap': 'Purples',
                'levels': [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200],
                'spatial_dims': 3
            },
            're_snow': {
                'colormap': 'Greys',
                'levels': [50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000],
                'spatial_dims': 3
            },
            'sh2o': {
                'colormap': 'Blues',
                'levels': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                'spatial_dims': 3
            },
            'smois': {
                'colormap': 'BuGn',
                'levels': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                'spatial_dims': 3
            },
            'tslb': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-40, 45, 5)),
                'spatial_dims': 3
            }
        }
        
        metadata: Dict[str, Any] = standard_metadata.get(var_name.lower(), {
            'units': '',
            'long_name': var_name,
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
        
        if include_visualization:
            viz_meta = visualization_metadata.get(var_name.lower(), {})
            if viz_meta:
                metadata.update(viz_meta)
            else:
                metadata.update({
                    'colormap': 'viridis',
                    'levels': None,
                    'spatial_dims': 2
                })
        
        return metadata

    @staticmethod
    def get_2d_variable_metadata(var_name: str, data_array: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        Retrieve comprehensive metadata for 2D MPAS surface variables including visualization properties optimized for plotting workflows. This method wraps `get_variable_metadata()` with `include_visualization=True` to return units, long names, colormaps, and contour levels specifically tailored for 2D surface field visualization. It automatically converts units to display-friendly formats using the UnitConverter utility and provides compatibility with visualization modules that require plotting-ready metadata. This method serves as the primary metadata interface for surface diagnostic plots including temperature, pressure, precipitation, and wind speed maps.

        Parameters:
            var_name (str): Name of the 2D MPAS variable to retrieve metadata for (e.g., 't2m', 'mslp', 'rainnc').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing variable attributes for extraction (default: None).

        Returns:
            dict: Metadata dictionary with 'units', 'long_name', 'colormap', 'levels', 'spatial_dims', 'original_units', and 'display_units' keys optimized for 2D visualization.
        """
        from mpasdiag.processing.utils_unit import UnitConverter
        metadata = MPASFileMetadata.get_variable_metadata(var_name, data_array, include_visualization=True)
        
        if 'units' in metadata:
            original_units = metadata['units']
            display_units = UnitConverter.get_display_units(var_name, original_units)
            metadata['original_units'] = original_units
            metadata['units'] = display_units
            
            if display_units != original_units and 'long_name' in metadata:
                long_name = metadata['long_name']
                if f'({original_units})' in long_name:
                    metadata['long_name'] = long_name.replace(f'({original_units})', f'({display_units})')
                else:
                    metadata['long_name'] = f"{long_name} ({display_units})"
        
        return metadata
    
    @staticmethod
    def get_3d_variable_metadata(var_name: str, data_array: Optional[xr.DataArray] = None, level: Optional[Any] = None) -> Dict[str, Any]:
        """
        Retrieve comprehensive metadata for 3D MPAS atmospheric variables including visualization properties for volumetric field analysis. This method is designed to handle variables with vertical levels (e.g., temperature, humidity, wind components at pressure levels or model levels) and would return units, long names, colormaps, and contour levels suitable for cross-section or vertical profile visualizations. The level parameter allows specification of which vertical level to extract metadata for, enabling level-specific colormap and contour adjustments for optimal representation of atmospheric structure. Currently this functionality is not implemented and the method raises a NotImplementedError, indicating future expansion for 3D diagnostic capabilities in the MPASdiag toolkit.

        Parameters:
            var_name (str): Name of the 3D MPAS variable (e.g., 'theta', 'qv', 'w', 'pressure_p').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray with variable attributes and vertical dimension (default: None).
            level (Optional[Any]): Specific vertical level for metadata retrieval, can be pressure level, height, or model level index (default: None).

        Returns:
            dict: Metadata dictionary with 'units', 'long_name', 'colormap', 'levels', 'spatial_dims' keys for 3D visualization.

        Raises:
            NotImplementedError: 3D variable support is not yet implemented in this release.
        """
        raise NotImplementedError("3D variable support not yet implemented")
    
    @staticmethod
    def get_3d_colormap_and_levels(var_name: str, data_array: Optional[xr.DataArray] = None, level: Optional[Any] = None) -> Tuple[str, Optional[List[float]]]:
        """
        Extract colormap name and contour level specifications for 3D MPAS atmospheric variables optimized for cross-section and vertical profile visualizations. This method retrieves plotting-specific metadata including matplotlib colormap names (e.g., 'RdYlBu_r', 'viridis') and discrete contour level arrays tailored to the dynamic range of 3D atmospheric fields at specified vertical levels. The returned values are designed for direct use in matplotlib plotting functions to ensure consistent and scientifically appropriate color representations across different 3D diagnostic plots. Currently this functionality is not implemented and the method raises a NotImplementedError, reserving this interface for future 3D visualization expansion in MPASdiag.

        Parameters:
            var_name (str): Name of the 3D MPAS variable to retrieve colormap and levels for (e.g., 'theta', 'relhum', 'w').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray with variable data for dynamic level calculation (default: None).
            level (Optional[Any]): Specific vertical level to optimize colormap/levels for, can be pressure level or model level (default: None).

        Returns:
            tuple: Two-element tuple containing (colormap_name: str, contour_levels: Optional[List[float]]) where colormap_name is a valid matplotlib colormap identifier and contour_levels is a list of discrete values for contouring.

        Raises:
            NotImplementedError: 3D variable colormap/level support is not yet implemented in this release.
        """
        raise NotImplementedError("3D variable support not yet implemented")
    
    @staticmethod
    def plot_3d_variable_slice(data_array: xr.DataArray, lon: np.ndarray, lat: np.ndarray, 
                              level: Union[int, float], var_name: str) -> Any:
        """
        Generate a visualization of a 3D MPAS variable at a specified vertical level extracted from volumetric atmospheric data. This method creates 2D plots of atmospheric fields by slicing 3D data arrays at user-specified pressure levels, height levels, or model vertical indices, enabling visualization of horizontal structure at constant vertical levels. The plotting function handles coordinate transformation from MPAS unstructured mesh to regular grids suitable for contouring, applies appropriate colormaps and contour levels based on variable type, and produces publication-quality diagnostic plots of atmospheric state. Currently this functionality is not implemented and the method raises a NotImplementedError, indicating future development for comprehensive 3D cross-section and level-slice visualization capabilities.

        Parameters:
            data_array (xr.DataArray): 3D xarray DataArray containing MPAS variable data with spatial and vertical dimensions.
            lon (np.ndarray): Longitude coordinate array corresponding to horizontal MPAS mesh cells.
            lat (np.ndarray): Latitude coordinate array corresponding to horizontal MPAS mesh cells.
            level (Union[int, float]): Vertical level to extract and plot, can be pressure value (hPa), height (m), or model level index.
            var_name (str): Name of the MPAS variable for metadata lookup and plot labeling (e.g., 'theta', 'qv', 'w').

        Returns:
            Any: Matplotlib figure or axes object containing the plotted 3D variable slice at specified level.

        Raises:
            NotImplementedError: 3D variable slice plotting is not yet implemented in this release.
        """
        raise NotImplementedError("3D variable support not yet implemented")
    
    @staticmethod
    def get_available_variables() -> List[str]:
        """
        Return the complete list of MPAS variable names supported by the metadata registry for validation and introspection. This method provides the canonical set of variable identifiers that have predefined metadata including units, long names, and visualization properties registered in the MPASFileMetadata class. The returned list includes both 2D surface variables (temperature, pressure, precipitation, winds) and 3D atmospheric variables (mixing ratios, reflectivity, vertical motion), spanning diagnostic, prognostic, and derived MPAS model output fields. This function is particularly useful for input validation in command-line interfaces, building user interface selection menus, auto-discovery of available diagnostic capabilities, and programmatic iteration over supported variables in batch processing workflows.

        Returns:
            list: Complete list of supported MPAS variable name strings including surface diagnostics, upper-air variables, microphysics fields, and derived quantities.
        """
        return [
            't2m', 'th2m', 'temperature', 'surface_temperature',
            'surface_pressure', 'mslp', 'pressure',
            'u10', 'v10', 'wspd10', 'wind_speed', 'q2', 'rh2',
            
            'rainnc', 'rainc', 'precipitation', 'precipw',
            
            'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max',
            
            'temperature_50hPa', 'temperature_100hPa', 'temperature_200hPa', 'temperature_250hPa',
            'temperature_500hPa', 'temperature_700hPa', 'temperature_850hPa', 'temperature_925hPa',
            
            'relhum_50hPa', 'relhum_100hPa', 'relhum_200hPa', 'relhum_250hPa',
            'relhum_500hPa', 'relhum_700hPa', 'relhum_850hPa', 'relhum_925hPa',
            
            'height_50hPa', 'height_100hPa', 'height_200hPa', 'height_250hPa',
            'height_500hPa', 'height_700hPa', 'height_850hPa', 'height_925hPa',
            
            'uzonal_50hPa', 'uzonal_100hPa', 'uzonal_200hPa', 'uzonal_250hPa',
            'uzonal_500hPa', 'uzonal_700hPa', 'uzonal_850hPa', 'uzonal_925hPa',
            'umeridional_50hPa', 'umeridional_100hPa', 'umeridional_200hPa', 'umeridional_250hPa',
            'umeridional_500hPa', 'umeridional_700hPa', 'umeridional_850hPa', 'umeridional_925hPa',
            
            'vorticity_50hPa', 'vorticity_100hPa', 'vorticity_200hPa', 'vorticity_250hPa',
            'vorticity_500hPa', 'vorticity_700hPa', 'vorticity_850hPa', 'vorticity_925hPa',
            
            'cape', 'cin', 'lcl', 'lfc', 'srh_0_1km', 'srh_0_3km',
            'updraft_helicity_max', 'w_velocity_max', 'wind_speed_level1_max',
            
            't_oml', 'h_oml', 'hu_oml', 'hv_oml',
            
            'cldfrac_low_UPP', 'cldfrac_mid_UPP', 'cldfrac_high_UPP', 'cldfrac_tot_UPP',
            
            'olrtoa',
            
            'refl10cm', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'ni', 'u', 'w', 'rho',
            'pressure_p', 'pressure_base', 'theta', 'relhum', 'rho_base', 'theta_base',
            'uReconstructZonal', 'uReconstructMeridional', 'cldfrac', 're_cloud', 're_ice', 're_snow',
            'sh2o', 'smois', 'tslb'
        ]