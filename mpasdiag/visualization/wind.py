#!/usr/bin/env python3

"""
MPAS Wind Vector Visualization

This module provides comprehensive wind plotting capabilities for MPAS atmospheric model output including wind barbs, arrows, and overlay functionality on surface maps for both 2D and 3D wind data. It implements the MPASWindPlotter class that creates professional cartographic wind visualizations using meteorological conventions (wind barbs showing speed and direction with flags) or vector representations (arrows), supports automatic extraction of 2D wind fields from 3D datasets at specified pressure levels or model layers, computes wind speed and direction from u/v components, and performs data subsampling for performance optimization on high-resolution meshes. The plotter provides flexible overlay configurations for superimposing wind vectors on existing surface variable maps (temperature, pressure, moisture), handles geographic map setup with coastlines and gridlines, includes automatic wind statistics in titles, and integrates seamlessly with MPASSurfacePlotter for multi-variable diagnostic plots. Core capabilities include configurable vector scaling and density, background field rendering, coordinate system handling, and publication-quality output suitable for operational weather analysis and atmospheric dynamics studies.

Classes:
    MPASWindPlotter: Specialized class for creating wind plots and wind overlays for MPAS model output with meteorological conventions.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Optional, Dict, Any, Tuple, Union, List
from datetime import datetime
import xarray as xr
import os

from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.processing.utils_unit import UnitConverter


class MPASWindPlotter(MPASVisualizer):
    """
    Specialized plotter for creating professional wind vector visualizations and wind overlay capabilities for MPAS atmospheric model output including both 2D surface winds and 3D winds at pressure levels. This class extends MPASVisualizer to provide comprehensive wind plotting functionality supporting wind barbs (meteorological convention showing speed and direction with flags), wind arrows (vector representation), and flexible overlay configurations for superimposing wind vectors on existing surface variable maps (temperature, pressure, moisture plots). The plotter handles automatic extraction of 2D wind fields from 3D datasets at specified pressure levels or model layers, computes wind speed and direction from u/v components, performs data subsampling for performance optimization on high-resolution meshes, and provides geographic map setup with coastlines, borders, and gridlines. Wind plots include automatic statistics (mean, maximum speed) in titles, configurable vector scaling and density, and seamless integration with MPASSurfacePlotter for creating multi-variable diagnostic plots.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 10), dpi: int = 100):
        """
        Initializes the wind plotter with figure dimensions and resolution settings for creating
        cartographic wind visualizations. This constructor inherits from MPASVisualizer to establish
        the base plotting framework and configures default figure size and DPI for high-quality
        publication-ready wind barb and arrow plots with proper map projections.

        Parameters:
            figsize (Tuple[int, int]): Figure dimensions in inches as (width, height) (default: (12, 10)).
            dpi (int): Figure resolution in dots per inch for output quality (default: 300).

        Returns:
            None: Initializes instance attributes through parent class constructor.
        """
        super().__init__(figsize=figsize, dpi=dpi)
        
    def create_wind_plot(self,
                        lon: np.ndarray,
                        lat: np.ndarray,
                        u_data: np.ndarray,
                        v_data: np.ndarray,
                        lon_min: float,
                        lon_max: float,
                        lat_min: float,
                        lat_max: float,
                        wind_level: str = "surface",
                        plot_type: str = 'barbs',
                        subsample: int = 1,
                        scale: Optional[float] = None,
                        show_background: bool = False,
                        bg_colormap: str = "viridis",
                        title: Optional[str] = None,
                        time_stamp: Optional[datetime] = None,
                        projection: str = 'PlateCarree',
                        level_info: Optional[str] = None) -> Tuple[Figure, Axes]:
        """
        Creates a cartographic wind plot displaying wind vectors as barbs or arrows with geographic
        features and automatic statistics. This method generates a complete wind visualization with
        coastlines, borders, land/ocean features, regional map elements, and gridlines, supports
        data subsampling for performance optimization, filters invalid values, and computes wind
        speed statistics for the title. It returns a fully configured figure and axes for display or saving.

        Parameters:
            lon (np.ndarray): 1D longitude array in degrees for vector positions.
            lat (np.ndarray): 1D latitude array in degrees for vector positions.
            u_data (np.ndarray): U-component (eastward) wind values in m/s.
            v_data (np.ndarray): V-component (northward) wind values in m/s.
            lon_min (float): Western longitude bound in degrees for map extent.
            lon_max (float): Eastern longitude bound in degrees for map extent.
            lat_min (float): Southern latitude bound in degrees for map extent.
            lat_max (float): Northern latitude bound in degrees for map extent.
            wind_level (str): Descriptive level string for labeling (default: "surface").
            plot_type (str): Vector rendering style 'barbs' or 'arrows' (default: 'barbs').
            subsample (int): Subsampling factor for reducing vector density (default: 1).
            scale (Optional[float]): Arrow length scaling for quiver plots (default: 200 for arrows).
            show_background (bool): Whether to show wind speed background color (default: False).
            bg_colormap (str): Colormap for background if enabled (default: "viridis").
            title (Optional[str]): Custom plot title (default: auto-generated with statistics).
            time_stamp (Optional[datetime]): Timestamp for title annotation (default: None).
            projection (str): Cartopy projection name for map axes (default: 'PlateCarree').
            level_info (Optional[str]): Additional level descriptor for title (default: None).

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and GeoAxes containing the wind plot.
        """
        proj = getattr(ccrs, projection)()
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi,
                                        subplot_kw={'projection': proj})
        
        assert isinstance(self.ax, GeoAxes), "Axes must be a GeoAxes instance"
        
        is_global_lon = (lon_max - lon_min) >= 359.0
        is_global_lat = (lat_max - lat_min) >= 179.0
        
        if is_global_lon and is_global_lat:
            adjusted_lon_min = max(lon_min, -179.99)
            adjusted_lon_max = min(lon_max, 179.99)
            adjusted_lat_min = max(lat_min, -89.99)
            adjusted_lat_max = min(lat_max, 89.99)
            self.ax.set_extent([adjusted_lon_min, adjusted_lon_max, adjusted_lat_min, adjusted_lat_max], crs=ccrs.PlateCarree())
            print(f"Using global extent (adjusted to avoid dateline): [{adjusted_lon_min}, {adjusted_lon_max}, {adjusted_lat_min}, {adjusted_lat_max}]")
        else:
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
        self.ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        
        if subsample > 1:
            indices = np.arange(0, len(lon), subsample)
            lon_sub = lon[indices]
            lat_sub = lat[indices]
            u_sub = u_data[indices]
            v_sub = v_data[indices]
        else:
            lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
        
        valid_mask = ~(np.isnan(u_sub) | np.isnan(v_sub))
        lon_valid = lon_sub[valid_mask]
        lat_valid = lat_sub[valid_mask]
        u_valid = u_sub[valid_mask]
        v_valid = v_sub[valid_mask]
        
        if len(lon_valid) == 0:
            print("Warning: No valid wind data found")
            return self.fig, self.ax
        
        color = 'black'
        
        if plot_type == 'barbs':
            self.ax.barbs(lon_valid, lat_valid, u_valid, v_valid,
                         transform=ccrs.PlateCarree(), color=color, length=6)
        elif plot_type == 'arrows':
            if scale is None:
                scale = 200  
            self.ax.quiver(lon_valid, lat_valid, u_valid, v_valid,
                          transform=ccrs.PlateCarree(), color=color, scale=scale)
        else:
            raise ValueError(f"plot_type must be 'barbs' or 'arrows', got '{plot_type}'")
        
        gl = self.ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
        
        if title is None:
            wind_speed = np.sqrt(u_valid**2 + v_valid**2)
            max_speed = np.max(wind_speed)
            mean_speed = np.mean(wind_speed)
            
            title_parts = ["MPAS Wind Analysis"]
            if level_info:
                title_parts.append(f"({level_info})")
            if time_stamp:
                time_str = time_stamp.strftime('%Y-%m-%d %H:%M UTC')
                title_parts.append(f"- {time_str}")
            
            title = " ".join(title_parts)
            title += f"\nMax: {max_speed:.1f} m/s, Mean: {mean_speed:.1f} m/s"
        
        self.ax.set_title(title, fontsize=12, pad=20)
        
        print(f"Plotted {len(lon_valid)} wind vectors")
        print(f"Wind speed range: {np.min(np.sqrt(u_valid**2 + v_valid**2)):.1f} to {np.max(np.sqrt(u_valid**2 + v_valid**2)):.1f} m/s")
        
        return self.fig, self.ax
    
    def add_wind_overlay(self,
                        ax: Axes,
                        lon: np.ndarray,
                        lat: np.ndarray,
                        wind_config: Dict[str, Any]) -> None:
        """
        Adds wind vectors as an overlay onto an existing map axes using a configuration dictionary
        for flexible styling and data selection. This method extracts U/V wind components from the
        configuration, handles 3D data by selecting a specific vertical level, applies optional
        subsampling for performance, filters invalid values, and renders wind barbs or arrows using
        the specified plot type and styling parameters without creating a new figure.

        Parameters:
            ax (Axes): Existing map axes (typically GeoAxes) to receive the wind overlay.
            lon (np.ndarray): 1D longitude array in degrees for wind vector positions.
            lat (np.ndarray): 1D latitude array in degrees for wind vector positions.
            wind_config (Dict[str, Any]): Configuration dictionary with required keys 'u_data' (ndarray), 'v_data' (ndarray) and optional keys 'plot_type' ('barbs'/'arrows'), 'subsample' (int), 'color' (str), 'scale' (float), 'level_index' (int).

        Returns:
            None: Draws wind overlay directly onto provided axes without returning objects.
        """
        u_data = wind_config['u_data']
        v_data = wind_config['v_data']
        plot_type = wind_config.get('plot_type', 'barbs')
        subsample = wind_config.get('subsample', 1)
        color = wind_config.get('color', 'black')
        scale = wind_config.get('scale', None)
        level_index = wind_config.get('level_index', None)
        
        if u_data.ndim > 1:
            if level_index is not None:
                u_data = u_data[:, level_index]
                v_data = v_data[:, level_index]
            else:
                u_data = u_data[:, -1]
                v_data = v_data[:, -1]
        
        if subsample > 1:
            indices = np.arange(0, len(lon), subsample)
            lon_sub = lon[indices]
            lat_sub = lat[indices]
            u_sub = u_data[indices]
            v_sub = v_data[indices]
        else:
            lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
        
        valid_mask = ~(np.isnan(u_sub) | np.isnan(v_sub))
        lon_valid = lon_sub[valid_mask]
        lat_valid = lat_sub[valid_mask]
        u_valid = u_sub[valid_mask]
        v_valid = v_sub[valid_mask]
        
        if len(lon_valid) == 0:
            print("Warning: No valid wind data for overlay")
            return
        
        if plot_type == 'barbs':
            ax.barbs(lon_valid, lat_valid, u_valid, v_valid,
                    transform=ccrs.PlateCarree(), color=color, length=6)
        elif plot_type == 'arrows':
            if scale is None:
                scale = 200
            ax.quiver(lon_valid, lat_valid, u_valid, v_valid,
                     transform=ccrs.PlateCarree(), color=color, scale=scale)
        else:
            raise ValueError(f"Unsupported wind plot_type: {plot_type}")
        
        print(f"Added {len(lon_valid)} wind vectors as overlay")

    def create_batch_wind_plots(self,
                                processor,
                                output_dir: str,
                                lon_min: float,
                                lon_max: float,
                                lat_min: float,
                                lat_max: float,
                                u_variable: str = 'u',
                                v_variable: str = 'v',
                                plot_type: str = 'barbs',
                                formats: Optional[List[str]] = None,
                                subsample: int = 1,
                                scale: Optional[float] = None,
                                show_background: bool = False) -> List[str]:
        """
        Create a batch of wind plots from a processor's loaded dataset and save them to disk.

        Parameters mirror the CLI usage and the processor API. Returns a list of created file paths
        (base paths without extensions).
        """
        if formats is None:
            formats = ['png']

        if not hasattr(processor, 'dataset') or processor.dataset is None:
            raise ValueError("Processor has no loaded dataset. Call load_2d_data() first.")

        dataset = processor.dataset
        # Determine time dimension and length using existing utility
        _, _, time_size = MPASDateTimeUtils.validate_time_parameters(dataset, 0, False)

        created_files: List[str] = []

        for time_idx in range(time_size):
            # Extract u/v and coordinates for this time step
            u_data = processor.get_2d_variable_data(u_variable, time_idx)
            v_data = processor.get_2d_variable_data(v_variable, time_idx)
            lon, lat = processor.extract_2d_coordinates_for_variable(u_variable, u_data)

            # Create the wind plot
            fig, ax = self.create_wind_plot(
                lon, lat, u_data.values, v_data.values,
                lon_min, lon_max, lat_min, lat_max,
                plot_type=plot_type,
                subsample=subsample,
                scale=scale,
                show_background=show_background,
                time_stamp=None,
            )

            # Add branding/timestamp and save
            try:
                time_str = MPASDateTimeUtils.get_time_info(dataset, time_idx, var_context='wind', verbose=False)
            except Exception:
                time_str = f"time_{time_idx}"

            base_name = f"mpas_wind_{plot_type}_{time_str}_{time_idx:03d}"
            output_path = os.path.join(output_dir, base_name)

            self.add_timestamp_and_branding()
            self.save_plot(output_path, formats=formats)
            self.close_plot()

            created_files.append(output_path)

        return created_files
    
    def extract_2d_from_3d_wind(self,
                               u_data_3d: np.ndarray,
                               v_data_3d: np.ndarray,
                               level_index: Optional[int] = None,
                               level_value: Optional[float] = None,
                               pressure_levels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts 2D horizontal wind components from 3D wind data at a specified vertical level using
        either direct indexing or pressure-based level matching. This utility method supports three
        selection modes: explicit level index, pressure value matching (finding nearest level), or
        default top-level extraction when no parameters are provided. It returns horizontal slices
        of U and V wind components suitable for 2D plotting and overlay operations.

        Parameters:
            u_data_3d (np.ndarray): 3D U-component wind array with shape (cells, levels) in m/s.
            v_data_3d (np.ndarray): 3D V-component wind array with shape (cells, levels) in m/s.
            level_index (Optional[int]): Direct vertical level index for extraction (default: None).
            level_value (Optional[float]): Target pressure level value in Pa or hPa for nearest match (default: None).
            pressure_levels (Optional[np.ndarray]): Pressure level values for level_value matching (default: None).

        Returns:
            Tuple[np.ndarray, np.ndarray]: 2D U and V wind component arrays at the selected level.
        """
        if level_index is not None:
            return u_data_3d[:, level_index], v_data_3d[:, level_index]
        
        if level_value is not None and pressure_levels is not None:
            level_idx = np.argmin(np.abs(pressure_levels - level_value))
            return u_data_3d[:, level_idx], v_data_3d[:, level_idx]
        
        return u_data_3d[:, -1], v_data_3d[:, -1]
    
    def compute_wind_speed_and_direction(self,
                                       u_data: np.ndarray,
                                       v_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes wind speed magnitude and meteorological direction from U and V wind components using
        standard vector mathematics and meteorological conventions. This utility method calculates wind
        speed as the vector magnitude (sqrt(u² + v²)) and converts mathematical angles to meteorological
        direction (0-360° where 0 is north, 90 is east) using the transformation (270 - arctan2(v, u)) % 360.
        It returns arrays matching the input shapes for element-wise wind analysis.

        Parameters:
            u_data (np.ndarray): U-component (eastward) wind array in m/s.
            v_data (np.ndarray): V-component (northward) wind array in m/s.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Wind speed in m/s and meteorological direction in degrees (0-360, 0=north).
        """
        wind_speed = np.sqrt(u_data**2 + v_data**2)
        wind_direction = np.arctan2(v_data, u_data) * 180 / np.pi
        wind_direction = (270 - wind_direction) % 360
        
        return wind_speed, wind_direction