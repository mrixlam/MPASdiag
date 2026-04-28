#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Variable Metadata Management

This module defines the MPASFileMetadata class, which provides a centralized registry of metadata for MPAS variables, including units, long names, and visualization properties. The class includes methods to retrieve standardized metadata for given variable names, with optional inclusion of colormap and contour level information optimized for 2D surface variable visualization. The module is designed to ensure consistency in variable metadata across the MPASdiag processing pipeline and to facilitate integration with visualization components that require plotting-ready metadata. Future expansions will include support for 3D variable metadata and visualization properties. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import xarray as xr
from typing import Optional, Dict, Any, List, Tuple

from mpasdiag.processing.utils_unit import UnitConverter

from .constants import (
    M2_PER_S2, MM, DBZ, KELVIN, M_PER_S, PA, PERCENT,
    METER, M2_PER_S, KG_PER_KG, KG_PER_M2, KG_PER_M3, KG_PER_M_PER_S,
    MICRONS, PER_KG, M3_PER_M3, J_PER_KG, PER_S, W_PER_M2, NOUNIT
)


class MPASFileMetadata:
    """ Metadata management class for MPAS variable attributes and data processing parameters with centralized registry of variable properties. """
    
    _3D_NOT_IMPLEMENTED = "3D variable support not yet implemented"

    @staticmethod
    def get_variable_metadata(var_name: str, 
                              data_array: Optional[xr.DataArray] = None, 
                              include_visualization: bool = False) -> Dict[str, Any]:
        """
        This method retrieves standardized metadata for a given MPAS variable name, including units, long names, and optionally visualization properties such as colormap and contour levels. The metadata is stored in a centralized registry within the method, allowing for consistent retrieval of variable attributes across the MPASdiag processing pipeline. The include_visualization flag enables the addition of plotting-specific metadata for 2D surface variables, facilitating integration with visualization components that require ready-to-use metadata for generating diagnostic plots. The method is designed to be extensible, allowing for future additions of new variables and their associated metadata as the MPASdiag toolkit evolves. 

        Parameters:
            var_name (str): Name of the MPAS variable to retrieve metadata for (e.g., 't2m', 'mslp', 'relhum_500hpa').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing the variable data, used for context-aware metadata retrieval if needed.
            include_visualization (bool): Flag indicating whether to include visualization-specific metadata such as colormap and contour levels for 2D surface variables. 

        Returns:
            dict: A dictionary containing standardized metadata for the specified variable, including keys such as 'units', 'long_name', and optionally 'colormap' and 'levels' if include_visualization is True. The metadata values are designed to be directly usable in data processing and visualization workflows within MPASdiag. 
        """
        standard_metadata = {
            'olrtoa': {'units': W_PER_M2, 'long_name': 'Outgoing Longwave Radiation at TOA'},
            
            'rainc': {'units': MM, 'long_name': 'Convective Precipitation'},
            'rainnc': {'units': MM, 'long_name': 'Non-Convective Precipitation'},
            'precipw': {'units': MM, 'long_name': 'Precipitable Water'},

            'iwv': {'units': KG_PER_M2, 'long_name': 'Integrated Water Vapor'},
            'ivt': {'units': KG_PER_M_PER_S, 'long_name': 'Integrated Vapor Transport'},
            'ivt_u': {'units': KG_PER_M_PER_S, 'long_name': 'Eastward Integrated Vapor Transport'},
            'ivt_v': {'units': KG_PER_M_PER_S, 'long_name': 'Northward Integrated Vapor Transport'},

            'refl10cm_max': {'units': DBZ, 'long_name': 'Maximum 10cm Reflectivity'},
            'refl10cm_1km': {'units': DBZ, 'long_name': '10cm Reflectivity at 1km AGL'},
            'refl10cm_1km_max': {'units': DBZ, 'long_name': 'Maximum 10cm Reflectivity at 1km AGL'},
            
            't2m': {'units': KELVIN, 'long_name': '2-meter Temperature'}, 
            'th2m': {'units': KELVIN, 'long_name': '2-meter Potential Temperature'}, 
            'u10': {'units': M_PER_S, 'long_name': '10-meter U-wind'},
            'v10': {'units': M_PER_S, 'long_name': '10-meter V-wind'}, 
            'q2': {'units': KG_PER_KG, 'long_name': '2-meter Specific Humidity'}, 
            'mslp': {'units': PA, 'long_name': 'Mean Sea Level Pressure'},
            
            'relhum_50hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 50 hPa'},
            'relhum_100hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 100 hPa'},
            'relhum_200hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 200 hPa'},
            'relhum_250hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 250 hPa'},
            'relhum_500hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 500 hPa'},
            'relhum_700hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 700 hPa'},
            'relhum_850hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 850 hPa'},
            'relhum_925hpa': {'units': PERCENT, 'long_name': 'Relative Humidity at 925 hPa'},
            
            'temperature_50hpa': {'units': KELVIN, 'long_name': 'Temperature at 50 hPa'},
            'temperature_100hpa': {'units': KELVIN, 'long_name': 'Temperature at 100 hPa'},
            'temperature_200hpa': {'units': KELVIN, 'long_name': 'Temperature at 200 hPa'},
            'temperature_250hpa': {'units': KELVIN, 'long_name': 'Temperature at 250 hPa'},
            'temperature_500hpa': {'units': KELVIN, 'long_name': 'Temperature at 500 hPa'},
            'temperature_700hpa': {'units': KELVIN, 'long_name': 'Temperature at 700 hPa'},
            'temperature_850hpa': {'units': KELVIN, 'long_name': 'Temperature at 850 hPa'},  
            'temperature_925hpa': {'units': KELVIN, 'long_name': 'Temperature at 925 hPa'}, 
            
            'height_50hpa': {'units': METER, 'long_name': 'Geopotential Height at 50 hPa'},
            'height_100hpa': {'units': METER, 'long_name': 'Geopotential Height at 100 hPa'},
            'height_200hpa': {'units': METER, 'long_name': 'Geopotential Height at 200 hPa'},
            'height_250hpa': {'units': METER, 'long_name': 'Geopotential Height at 250 hPa'},
            'height_500hpa': {'units': METER, 'long_name': 'Geopotential Height at 500 hPa'},
            'height_700hpa': {'units': METER, 'long_name': 'Geopotential Height at 700 hPa'},
            'height_850hpa': {'units': METER, 'long_name': 'Geopotential Height at 850 hPa'},
            'height_925hpa': {'units': METER, 'long_name': 'Geopotential Height at 925 hPa'},

            'uzonal_50hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 50 hPa'},
            'uzonal_100hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 100 hPa'},
            'uzonal_200hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 200 hPa'},
            'uzonal_250hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 250 hPa'},
            'uzonal_500hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 500 hPa'},
            'uzonal_700hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 700 hPa'},
            'uzonal_850hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 850 hPa'},
            'uzonal_925hpa': {'units': M_PER_S, 'long_name': 'Zonal Wind at 925 hPa'},

            'umeridional_50hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 50 hPa'},
            'umeridional_100hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 100 hPa'},
            'umeridional_200hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 200 hPa'},
            'umeridional_250hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 250 hPa'},
            'umeridional_500hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 500 hPa'},
            'umeridional_700hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 700 hPa'},
            'umeridional_850hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 850 hPa'},
            'umeridional_925hpa': {'units': M_PER_S, 'long_name': 'Meridional Wind at 925 hPa'},

            'vorticity_50hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 50 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_100hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 100 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_200hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 200 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_250hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 250 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_500hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 500 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_700hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 700 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_850hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 850 hPa', 'spatial_dim': 'nVertices'},
            'vorticity_925hpa': {'units': PER_S, 'long_name': 'Relative Vorticity at 925 hPa', 'spatial_dim': 'nVertices'},

            'cape': {'units': J_PER_KG, 'long_name': 'Convective Available Potential Energy'},
            'cin': {'units': J_PER_KG, 'long_name': 'Convective Inhibition'},
            'lcl': {'units': METER, 'long_name': 'Lifting Condensation Level'},
            'lfc': {'units': METER, 'long_name': 'Level of Free Convection'},
            'srh_0_1km': {'units': M2_PER_S2, 'long_name': 'Storm Relative Helicity 0-1km'},
            'srh_0_3km': {'units': M2_PER_S2, 'long_name': 'Storm Relative Helicity 0-3km'},

            'updraft_helicity_max': {'units': M2_PER_S2, 'long_name': 'Maximum Updraft Helicity'},
            'w_velocity_max': {'units': M_PER_S, 'long_name': 'Maximum Vertical Velocity'},
            'wind_speed_level1_max': {'units': M_PER_S, 'long_name': 'Maximum Wind Speed at Level 1'},
            'wind_speed': {'units': M_PER_S, 'long_name': 'Wind Speed'},
            'wspd10': {'units': M_PER_S, 'long_name': '10-m Wind Speed'},

            't_oml': {'units': KELVIN, 'long_name': 'Ocean Mixed Layer Temperature'},
            'h_oml': {'units': METER, 'long_name': 'Ocean Mixed Layer Depth'},
            'hu_oml': {'units': M2_PER_S, 'long_name': 'Ocean Mixed Layer U-momentum'},
            'hv_oml': {'units': M2_PER_S, 'long_name': 'Ocean Mixed Layer V-momentum'},
            
            't_oml_initial': {'units': KELVIN, 'long_name': 'Initial Ocean Mixed Layer Temperature'},
            'h_oml_initial': {'units': METER, 'long_name': 'Initial Ocean Mixed Layer Depth'},
            't_oml_200m_initial': {'units': KELVIN, 'long_name': 'Initial Ocean Mixed Layer Temperature (200m)'},

            'cldfrac_low_upp': {'units': NOUNIT, 'long_name': 'Low Cloud Fraction'},
            'cldfrac_mid_upp': {'units': NOUNIT, 'long_name': 'Mid Cloud Fraction'},
            'cldfrac_high_upp': {'units': NOUNIT, 'long_name': 'High Cloud Fraction'},
            'cldfrac_tot_upp': {'units': NOUNIT, 'long_name': 'Total Cloud Fraction'},

            'refl10cm': {'units': DBZ, 'long_name': '10cm Reflectivity', 'spatial_dim': 'nCells'},
            'qv': {'units': KG_PER_KG, 'long_name': 'Water Vapor Mixing Ratio', 'spatial_dim': 'nCells'},
            'qc': {'units': KG_PER_KG, 'long_name': 'Cloud Water Mixing Ratio', 'spatial_dim': 'nCells'},
            'qr': {'units': KG_PER_KG, 'long_name': 'Rain Water Mixing Ratio', 'spatial_dim': 'nCells'},
            'qi': {'units': KG_PER_KG, 'long_name': 'Ice Mixing Ratio', 'spatial_dim': 'nCells'},
            'qs': {'units': KG_PER_KG, 'long_name': 'Snow Mixing Ratio', 'spatial_dim': 'nCells'},
            'qg': {'units': KG_PER_KG, 'long_name': 'Graupel Mixing Ratio', 'spatial_dim': 'nCells'},
            'ni': {'units': PER_KG, 'long_name': 'Ice Number Concentration', 'spatial_dim': 'nCells'},
            'u': {'units': M_PER_S, 'long_name': 'Horizontal Velocity (U-component)', 'spatial_dim': 'nEdges'},
            'w': {'units': M_PER_S, 'long_name': 'Vertical Velocity', 'spatial_dim': 'nVertices'},
            'rho': {'units': KG_PER_M3, 'long_name': 'Density', 'spatial_dim': 'nCells'},
            'pressure_p': {'units': PA, 'long_name': 'Pressure Perturbation', 'spatial_dim': 'nCells'},
            'pressure_base': {'units': PA, 'long_name': 'Base State Pressure', 'spatial_dim': 'nCells'},
            'theta': {'units': KELVIN, 'long_name': 'Potential Temperature', 'spatial_dim': 'nCells'},
            'relhum': {'units': PERCENT, 'long_name': 'Relative Humidity', 'spatial_dim': 'nCells'},
            'rho_base': {'units': KG_PER_M3, 'long_name': 'Base State Density', 'spatial_dim': 'nCells'},
            'theta_base': {'units': KELVIN, 'long_name': 'Base State Potential Temperature', 'spatial_dim': 'nCells'},
            'uReconstructZonal': {'units': M_PER_S, 'long_name': 'Reconstructed Zonal Wind', 'spatial_dim': 'nCells'},
            'uReconstructMeridional': {'units': M_PER_S, 'long_name': 'Reconstructed Meridional Wind', 'spatial_dim': 'nCells'},
            'cldfrac': {'units': NOUNIT, 'long_name': 'Cloud Fraction', 'spatial_dim': 'nCells'},
            're_cloud': {'units': MICRONS, 'long_name': 'Cloud Droplet Effective Radius', 'spatial_dim': 'nCells'},
            're_ice': {'units': MICRONS, 'long_name': 'Ice Crystal Effective Radius', 'spatial_dim': 'nCells'},
            're_snow': {'units': MICRONS, 'long_name': 'Snow Effective Radius', 'spatial_dim': 'nCells'},
            'sh2o': {'units': M3_PER_M3, 'long_name': 'Soil Moisture', 'spatial_dim': 'nCells'},
            'smois': {'units': M3_PER_M3, 'long_name': 'Soil Moisture Content', 'spatial_dim': 'nCells'},
            'tslb': {'units': KELVIN, 'long_name': 'Soil Temperature', 'spatial_dim': 'nCells'},
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
            'iwv': {
                'colormap': 'GnBu',
                'levels': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70],
                'spatial_dims': 2
            },
            'ivt': {
                'colormap': 'YlGnBu',
                'levels': [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000],
                'spatial_dims': 2
            },
            'ivt_u': {
                'colormap': 'RdBu_r',
                'levels': list(range(-600, 700, 100)),
                'spatial_dims': 2
            },
            'ivt_v': {
                'colormap': 'RdBu_r',
                'levels': list(range(-600, 700, 100)),
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
            
            't_oml_initial': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-2, 32, 2)),
                'spatial_dims': 2
            },
            'h_oml_initial': {
                'colormap': 'viridis',
                'levels': [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500],
                'spatial_dims': 2
            },
            't_oml_200m_initial': {
                'colormap': 'RdYlBu_r',
                'levels': list(range(-2, 32, 2)),
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
            visualization_entry = visualization_metadata.get(var_name.lower(), {})
            if visualization_entry:
                metadata.update(visualization_entry)
            else:
                metadata.update({
                    'colormap': 'viridis',
                    'levels': None,
                    'spatial_dims': 2
                })
        
        return metadata

    @staticmethod
    def get_2d_variable_metadata(var_name: str, 
                                 data_array: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        This method is designed to retrieve comprehensive metadata for 2D MPAS atmospheric variables, including visualization properties optimized for horizontal field analysis. It returns units, long names, colormaps, and contour levels suitable for visualizations of surface or near-surface atmospheric fields (e.g., temperature at 2m, mean sea level pressure, accumulated precipitation) on the MPAS unstructured mesh. The method integrates standard variable metadata with predefined visualization settings to facilitate consistent and scientifically appropriate plotting of 2D diagnostic fields across the MPASdiag toolkit. 

        Parameters:
            var_name (str): Name of the 2D MPAS variable (e.g., 't2m', 'mslp', 'precipw').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray with variable attributes for metadata extraction (default: None). 

        Returns:
            dict: Metadata dictionary with 'units', 'long_name', 'colormap', 'levels', 'spatial_dims' keys for 2D visualization. 
        """
        from mpasdiag.processing.utils_unit import UnitConverter
        metadata = MPASFileMetadata.get_variable_metadata(var_name, data_array, include_visualization=True)
        
        if 'units' in metadata:
            original_units = metadata['units']
            display_units = UnitConverter.get_display_units(var_name, original_units)
            metadata['original_units'] = original_units
            metadata['units'] = display_units
        
        return metadata
    
    @staticmethod
    def get_3d_variable_metadata(var_name: str, 
                                 data_array: Optional[xr.DataArray] = None, 
                                 level: Optional[Any] = None) -> Dict[str, Any]:
        """
        This method is intended to retrieve comprehensive metadata for 3D MPAS atmospheric variables, including visualization properties optimized for volumetric field analysis at specified vertical levels. It returns units, long names, colormaps, and contour levels suitable for visualizations of upper-air atmospheric fields (e.g., temperature, humidity, vertical velocity) on the MPAS unstructured mesh. The method is designed to handle coordinate transformation from MPAS unstructured mesh to regular grids suitable for contouring, and to apply variable-specific visualization settings based on metadata. Currently this functionality is not implemented and the method raises a NotImplementedError, reserving this interface for future development of comprehensive 3D variable visualization capabilities in MPASdiag. 

        Parameters:
            var_name (str): Name of the 3D MPAS variable (e.g., 'theta', 'qv', 'w').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray with variable attributes for metadata extraction (default: None).
            level (Optional[Any]): Specific vertical level to optimize metadata for, can be pressure level or model level (default: None). 

        Returns:
            dict: Metadata dictionary with 'units', 'long_name', 'colormap', 'levels', 'spatial_dims' keys for 3D visualization.
        """
        metadata = MPASFileMetadata.get_variable_metadata(var_name, data_array, include_visualization=True)

        if 'units' in metadata:
            original_units = metadata['units']
            display_units = UnitConverter.get_display_units(var_name, original_units)
            metadata['original_units'] = original_units
            metadata['units'] = display_units

        if level is not None:
            metadata['level'] = level

        return metadata

    @staticmethod
    def get_3d_colormap_and_levels(var_name: str,
                                   data_array: Optional[xr.DataArray] = None, 
                                   level: Optional[Any] = None) -> Tuple[str, Optional[List[float]]]:
        """
        This method is designed to retrieve appropriate colormap names and contour levels for visualizing 3D MPAS atmospheric variables at specified vertical levels. It takes into account the variable name, optional data attributes, and the vertical level of interest to determine scientifically meaningful visualization settings for upper-air fields (e.g., temperature, humidity, vertical velocity) on the MPAS unstructured mesh. The method is intended to handle coordinate transformation from MPAS unstructured mesh to regular grids suitable for contouring, and to apply variable-specific visualization settings based on metadata. Currently this functionality is not implemented and the method raises a NotImplementedError, reserving this interface for future development of comprehensive 3D variable visualization capabilities in MPASdiag. 

        Parameters:
            var_name (str): Name of the 3D MPAS variable (e.g., 'theta', 'qv', 'w').
            data_array (Optional[xr.DataArray]): Optional xarray DataArray with variable attributes for metadata extraction (default: None).
            level (Optional[Any]): Specific vertical level to optimize colormap and levels for, can be pressure level or model level (default: None).

        Returns:
            Tuple[str, Optional[List[float]]]: Tuple containing colormap name and list of contour levels for 3D variable visualization.
        """
        metadata = MPASFileMetadata.get_variable_metadata(var_name, data_array, include_visualization=True)
        colormap = metadata.get('colormap', 'viridis')
        levels = metadata.get('levels', None)
        return colormap, levels
    
    @staticmethod
    def get_available_variables() -> List[str]:
        """
        This method returns a comprehensive list of supported MPAS variable name strings that can be used for metadata retrieval and visualization within the MPASdiag toolkit. The list includes commonly used surface diagnostics (e.g., temperature at 2m, mean sea level pressure), upper-air variables at standard pressure levels (e.g., temperature, relative humidity, wind components), microphysics fields (e.g., cloud water content, rain water content), and derived quantities (e.g., CAPE, CIN, storm-relative helicity). This method serves as a reference for users to understand which variable names are recognized by the metadata retrieval functions and to ensure consistent usage of variable names across the MPASdiag toolkit. 

        Parameters:
            None

        Returns:
            List[str]: List of supported MPAS variable name strings for metadata retrieval and visualization. 
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