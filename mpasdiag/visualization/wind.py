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
from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking


class MPASWindPlotter(MPASVisualizer):
    """
    Specialized plotter for creating professional wind vector visualizations and wind overlay capabilities for MPAS atmospheric model output including both 2D surface winds and 3D winds at pressure levels. This class extends MPASVisualizer to provide comprehensive wind plotting functionality supporting wind barbs (meteorological convention showing speed and direction with flags), wind arrows (vector representation), and flexible overlay configurations for superimposing wind vectors on existing surface variable maps (temperature, pressure, moisture plots). The plotter handles automatic extraction of 2D wind fields from 3D datasets at specified pressure levels or model layers, computes wind speed and direction from u/v components, performs data subsampling for performance optimization on high-resolution meshes, and provides geographic map setup with coastlines, borders, and gridlines. Wind plots include automatic statistics (mean, maximum speed) in titles, configurable vector scaling and density, and seamless integration with MPASSurfacePlotter for creating multi-variable diagnostic plots.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 10), dpi: int = 100) -> None:
        """
        Initialize the wind plotter with customized figure dimensions and resolution for professional cartographic wind visualizations. This constructor inherits from the MPASVisualizer base class to establish the foundational plotting framework with matplotlib and cartopy integration. The method configures default figure size optimized for publication-quality wind barb and arrow plots with proper map projections. These settings provide appropriate canvas dimensions for displaying wind vectors with geographic features including coastlines, borders, and gridlines. The DPI parameter ensures high-resolution output suitable for both screen display and print publication.

        Parameters:
            figsize (Tuple[float, float]): Figure dimensions in inches as (width, height) tuple for the plot canvas (default: (12, 10) for landscape orientation).
            dpi (int): Figure resolution in dots per inch determining output image quality and detail level (default: 100 for screen display, use 300+ for publication).

        Returns:
            None: This constructor initializes instance attributes through parent class inheritance and does not return a value.
        """
        super().__init__(figsize=figsize, dpi=dpi)
    
    def calculate_optimal_subsample(
        self,
        num_points: int,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: str = 'barbs',
        target_density: Optional[int] = None
    ) -> int:
        """
        Calculate optimal subsampling factor to prevent visual clutter in wind vector plots based on data density and map extent. This intelligent method determines the appropriate subsample value by analyzing the number of data points, geographic extent, figure dimensions, and desired vector density. The calculation considers both the spatial density of MPAS cells and the available screen/paper space to ensure vectors are well-spaced and readable. For wind barbs, the target density is more conservative (fewer vectors) compared to arrows due to barbs' more complex visual footprint. The method uses a heuristic approach balancing between showing sufficient detail and avoiding overlap, with adjustable target density for user customization.

        Parameters:
            num_points (int): Total number of data points in the wind dataset before any subsampling is applied.
            lon_min (float): Western longitude boundary in degrees defining the map extent.
            lon_max (float): Eastern longitude boundary in degrees defining the map extent.
            lat_min (float): Southern latitude boundary in degrees defining the map extent.
            lat_max (float): Northern latitude boundary in degrees defining the map extent.
            figsize (Optional[Tuple[float, float]]): Figure dimensions as (width, height) in inches, uses instance figsize if None (default: None).
            plot_type (str): Vector type being plotted - 'barbs' for wind barbs or 'arrows' for quiver arrows (default: 'barbs').
            target_density (Optional[int]): Desired number of vectors per figure dimension, overrides automatic calculation if provided (default: None for automatic).

        Returns:
            int: Optimal subsample factor where 1 means no subsampling, 2 means every other point, higher values for sparser sampling.
        """
        if figsize is None:
            figsize = self.figsize
        
        map_lon_range = lon_max - lon_min
        map_lat_range = lat_max - lat_min
        map_area = map_lon_range * map_lat_range
        
        fig_width, fig_height = figsize
        fig_area = fig_width * fig_height
        
        if target_density is None:
            if plot_type == 'barbs':
                target_vectors_per_inch = 3
            else:
                target_vectors_per_inch = 4
        else:
            target_vectors_per_inch = target_density
        
        target_total_vectors = int(fig_area * target_vectors_per_inch)
        
        if num_points <= target_total_vectors:
            return 1
        
        subsample = int(np.sqrt(num_points / target_total_vectors))
        
        subsample = max(1, min(subsample, 50))
        
        return subsample
        
    def convert_to_numpy(self, arr: Union[np.ndarray, xr.DataArray, Any]) -> np.ndarray:
        """
        Convert various array-like objects including xarray DataArray and dask arrays to standard NumPy ndarray format. This utility method handles the heterogeneous data types commonly encountered when processing MPAS model output by providing a unified conversion pathway. The function prioritizes lazy evaluation by calling compute() on dask arrays before conversion to avoid creating inefficient boolean dask indexers during subsequent operations. For xarray DataArray objects, it extracts the underlying values attribute to obtain the raw NumPy array. For all other array-like objects, the method attempts conversion using np.asarray as a robust fallback mechanism.

        Parameters:
            arr (Union[np.ndarray, xr.DataArray, Any]): Input array-like object which may be a NumPy array, xarray DataArray, dask array, or any object supporting array conversion protocols.

        Returns:
            np.ndarray: Standard NumPy ndarray containing the array data with all lazy computations resolved and metadata stripped.
        """
        try:
            if isinstance(arr, xr.DataArray):
                arr = arr.values
        except Exception:
            pass

        if hasattr(arr, 'compute') and not isinstance(arr, np.ndarray):
            try:
                arr = arr.compute()
            except Exception:
                pass

        return np.asarray(arr)
    
    def _prepare_wind_data(
        self,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
        subsample: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare wind component data for visualization by converting to NumPy arrays, applying spatial subsampling, and filtering invalid values. This internal helper method standardizes data preparation workflows to eliminate code duplication between create_wind_plot and add_wind_overlay methods. The function handles array type conversion from xarray or dask formats, reduces data density through systematic subsampling for performance optimization, and removes NaN values that would cause rendering errors. Supports both 1D arrays (irregular MPAS mesh) and 2D arrays (regridded data). For 2D arrays, preserves the grid structure which is essential for proper quiver rendering. For 1D arrays, applies stride-based subsampling. Invalid data filtering ensures only finite wind vectors are passed to matplotlib rendering functions.

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array in degrees east, may be 1D or 2D NumPy array or xarray DataArray requiring conversion.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinate array in degrees north, may be 1D or 2D NumPy array or xarray DataArray requiring conversion.
            u_data (Union[np.ndarray, xr.DataArray]): U-component (eastward) wind data in m/s, may be 1D or 2D NumPy array or xarray DataArray requiring conversion.
            v_data (Union[np.ndarray, xr.DataArray]): V-component (northward) wind data in m/s, may be 1D or 2D NumPy array or xarray DataArray requiring conversion.
            subsample (int): Spatial subsampling stride factor where 1 means no subsampling, 2 means every other point, etc. (default: 1 for full resolution).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four-element tuple containing (longitude, latitude, u_component, v_component). For 1D input, returns 1D arrays with NaN values removed and subsampling applied. For 2D input, returns 2D arrays with subsampling applied along both axes and NaN values preserved.
        """
        lon = self.convert_to_numpy(lon)
        lat = self.convert_to_numpy(lat)
        u_data = self.convert_to_numpy(u_data)
        v_data = self.convert_to_numpy(v_data)
        
        is_2d = lon.ndim == 2
        
        if is_2d:
            if subsample > 1:
                lon_sub = lon[::subsample, ::subsample]
                lat_sub = lat[::subsample, ::subsample]
                u_sub = u_data[::subsample, ::subsample]
                v_sub = v_data[::subsample, ::subsample]
            else:
                lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
            
            return lon_sub, lat_sub, u_sub, v_sub
        else:
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
            
            return lon_valid, lat_valid, u_valid, v_valid
    
    def _render_wind_vectors(
        self,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        u_data: np.ndarray,
        v_data: np.ndarray,
        plot_type: str = 'barbs',
        color: str = 'black',
        scale: Optional[float] = None
    ) -> None:
        """
        Render wind vectors as meteorological barbs, directional arrows, or streamlines onto an existing cartographic axes. This internal helper method provides a unified interface for wind vector visualization, eliminating code duplication between create_wind_plot and add_wind_overlay methods. The function supports three rendering styles: wind barbs following meteorological conventions with flags indicating speed, arrow vectors showing magnitude and direction through length and orientation, and streamlines showing continuous flow trajectories colored by wind speed. Supports both 1D scattered data (irregular MPAS mesh) for barbs and arrows, and 2D gridded data (regridded fields) for all three types, with streamlines requiring 2D gridded data. Vector rendering utilizes cartopy's PlateCarree coordinate transformation to ensure proper geographic positioning regardless of the axes projection. The method validates plot_type parameters and raises descriptive errors for unsupported visualization modes.

        Parameters:
            ax (Axes): Matplotlib axes object (typically GeoAxes) serving as the rendering target for wind vectors.
            lon (np.ndarray): Longitude coordinate array (1D for scattered data, 2D meshgrid for gridded data) in degrees east specifying vector base positions.
            lat (np.ndarray): Latitude coordinate array (1D for scattered data, 2D meshgrid for gridded data) in degrees north specifying vector base positions.
            u_data (np.ndarray): U-component (eastward) wind array (1D or 2D matching lon/lat dimensions) in m/s determining horizontal vector components.
            v_data (np.ndarray): V-component (northward) wind array (1D or 2D matching lon/lat dimensions) in m/s determining vertical vector components.
            plot_type (str): Vector rendering style selector - 'barbs' for meteorological wind barbs, 'arrows' for quiver arrows, or 'streamlines' for flow trajectories (default: 'barbs').
            color (str): Matplotlib color specification for vector rendering, accepts named colors, hex codes, or RGB tuples (default: 'black'). Not used for streamlines.
            scale (Optional[float]): Scaling factor for arrow length in quiver plots where larger values produce shorter arrows, unused for barbs and streamlines (default: None which sets scale=200 for arrows).

        Returns:
            None: This method draws wind vectors directly onto the provided axes and does not return objects.
        """
        if plot_type == 'barbs':
            ax.barbs(lon, lat, u_data, v_data,
                    transform=ccrs.PlateCarree(), color=color, length=6)
        elif plot_type == 'arrows':
            if scale is None:
                scale = 200
            ax.quiver(lon, lat, u_data, v_data,
                     transform=ccrs.PlateCarree(), color=color, scale=scale)
        elif plot_type == 'streamlines':
            if lon.ndim == 1:
                raise ValueError("Streamlines require gridded data. Use grid_resolution parameter to enable regridding.")
            
            lon_1d = lon[0, :] if lon.ndim == 2 else lon
            lat_1d = lat[:, 0] if lat.ndim == 2 else lat
            
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            
            strm = ax.streamplot(
                lon_1d, lat_1d, u_data, v_data,
                transform=ccrs.PlateCarree(),
                color=wind_speed,
                cmap='viridis',
                linewidth=1.5,
                density=2,
                arrowsize=1.5,
                arrowstyle='->',
                minlength=0.1
            )
            
            from matplotlib import pyplot as plt
            cbar = plt.colorbar(strm.lines, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8, aspect=40)
            cbar.set_label('Wind Speed [m s$^{-1}$]', fontsize=12, fontweight='bold', labelpad=10)
        else:
            raise ValueError(f"plot_type must be 'barbs', 'arrows', or 'streamlines', got '{plot_type}'")
    
    def _regrid_wind_components(
        self,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
        dataset: xr.Dataset,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        grid_resolution: float,
        regrid_method: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Regrid irregular MPAS wind components onto a regular latitude-longitude grid using spatial interpolation. This internal helper method transforms unstructured mesh data into structured grid format, eliminating code duplication between create_wind_plot and add_wind_overlay methods. The function processes U and V components independently through the remap_mpas_to_latlon_with_masking utility, supporting both linear interpolation for smooth fields and nearest-neighbor for value preservation. Linear interpolation uses scipy's griddata with triangulation to create continuous wind fields suitable for contour visualization, while nearest-neighbor uses KDTree for computational efficiency. The method converts input data to xarray DataArray format when needed, performs remapping with automatic coordinate extraction from dataset, creates 2D meshgrid coordinates, and flattens all arrays for vector plotting compatibility.

        Parameters:
            lon (np.ndarray): Original 1D longitude coordinate array in degrees east from the irregular MPAS mesh.
            lat (np.ndarray): Original 1D latitude coordinate array in degrees north from the irregular MPAS mesh.
            u_data (np.ndarray): U-component (eastward) wind data array in m/s on the irregular MPAS mesh.
            v_data (np.ndarray): V-component (northward) wind data array in m/s on the irregular MPAS mesh.
            dataset (xr.Dataset): MPAS dataset containing coordinate information for automatic extraction.
            lon_min (float): Western longitude boundary in degrees east for the target regular grid extent.
            lon_max (float): Eastern longitude boundary in degrees east for the target regular grid extent.
            lat_min (float): Southern latitude boundary in degrees north for the target regular grid extent.
            lat_max (float): Northern latitude boundary in degrees north for the target regular grid extent.
            grid_resolution (float): Spacing between grid points in degrees for both longitude and latitude dimensions of the target regular grid.
            regrid_method (str): Spatial interpolation algorithm selector, either 'linear' for smooth continuous fields using triangulation or 'nearest' for value-preserving nearest-neighbor (default: 'linear').

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four-element tuple containing (longitude, latitude, u_component, v_component) as flattened 1D NumPy arrays on the regular grid suitable for matplotlib vector plotting functions.
        """
        print(f"Regridding wind components to {grid_resolution}° grid using {regrid_method} interpolation...")
        
        u_xr = xr.DataArray(u_data, dims=['nCells']) if not isinstance(u_data, xr.DataArray) else u_data
        v_xr = xr.DataArray(v_data, dims=['nCells']) if not isinstance(v_data, xr.DataArray) else v_data
        
        u_regridded = remap_mpas_to_latlon_with_masking(
            data=u_xr,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=grid_resolution,
            method=regrid_method,
            apply_mask=True,
            lon_convention='auto'
        )
        
        v_regridded = remap_mpas_to_latlon_with_masking(
            data=v_xr,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=grid_resolution,
            method=regrid_method,
            apply_mask=True,
            lon_convention='auto'
        )
        
        lon_grid = u_regridded.lon.values
        lat_grid = u_regridded.lat.values
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        u_2d = u_regridded.values
        v_2d = v_regridded.values
        
        print(f"Regridded to {u_regridded.shape[0]}x{u_regridded.shape[1]} grid")
        
        return lon_2d, lat_2d, u_2d, v_2d
    
    def create_wind_plot(
        self,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
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
        level_info: Optional[str] = None,
        grid_resolution: Optional[float] = None,
        regrid_method: str = 'linear',
        dataset: Optional[xr.Dataset] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a cartographic wind plot displaying wind vectors as barbs, arrows, or streamlines with geographic features and automatic statistics. This method generates a complete wind visualization with coastlines, borders, land/ocean features, regional map elements, and gridlines, supports data subsampling for performance optimization, filters invalid values, and computes wind speed statistics for the title. Optionally regrids wind components to a regular lat-lon grid using linear interpolation for smooth fields. When subsample is set to -1, the method automatically calculates optimal subsampling to prevent visual clutter. Streamlines require regridded data and will automatically enable regridding if not already specified. It returns a fully configured figure and axes for display or saving.

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
            plot_type (str): Vector rendering style - 'barbs', 'arrows', or 'streamlines' (default: 'barbs').
            subsample (int): Subsampling factor for reducing vector density, use -1 for automatic calculation (default: 1). Not used for streamlines.
            scale (Optional[float]): Arrow length scaling for quiver plots (default: 200 for arrows).
            show_background (bool): Whether to show wind speed background color (default: False).
            bg_colormap (str): Colormap for background if enabled (default: "viridis").
            title (Optional[str]): Custom plot title (default: auto-generated with statistics).
            time_stamp (Optional[datetime]): Timestamp for title annotation (default: None).
            projection (str): Cartopy projection name for map axes (default: 'PlateCarree').
            level_info (Optional[str]): Additional level descriptor for title (default: None).
            grid_resolution (Optional[float]): Grid resolution in degrees for regridding wind components (default: None for no regridding).
            regrid_method (str): Interpolation method for regridding - 'linear' for smooth fields or 'nearest' for preserving values (default: 'linear').
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinate information, auto-created from lon/lat if not provided (default: None).

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and GeoAxes containing the wind plot.
        """
        proj = getattr(ccrs, projection)()
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi,
                                        subplot_kw={'projection': proj})
        
        assert isinstance(self.ax, GeoAxes), "Axes must be a GeoAxes instance"
        
        if plot_type == 'streamlines' and grid_resolution is None:
            grid_resolution = 0.5  
            print(f"Streamlines require gridded data. Auto-enabling regridding with resolution: {grid_resolution}°")
        
        if grid_resolution is not None:
            if dataset is None:
                lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
                lat_arr = lat if isinstance(lat, np.ndarray) else lat.values
                dataset = xr.Dataset({
                    'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
                    'latCell': xr.DataArray(lat_arr, dims=['nCells'])
                })
            
            lon, lat, u_data, v_data = self._regrid_wind_components(
                lon, lat, u_data, v_data, dataset,
                lon_min, lon_max, lat_min, lat_max,
                grid_resolution, regrid_method
            )
        
        lon_converted = self.convert_to_numpy(lon)
        u_converted = self.convert_to_numpy(u_data)
        
        if lon_converted.ndim == 2:
            num_points = np.sum(np.isfinite(lon_converted) & np.isfinite(u_converted))
        else:
            num_points = len(lon_converted[np.isfinite(lon_converted) & np.isfinite(u_converted)])
        
        if subsample == -1:
            subsample = self.calculate_optimal_subsample(
                num_points=num_points,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                figsize=self.figsize,
                plot_type=plot_type
            )
            print(f"Auto-calculated subsample factor: {subsample} (from {num_points} points)")
        
        lon_valid, lat_valid, u_valid, v_valid = self._prepare_wind_data(
            lon, lat, u_data, v_data, subsample
        )
        
        if len(lon_valid) == 0:
            print("Warning: No valid wind data found")
            return self.fig, self.ax
        
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
        
        color = 'black'
        self._render_wind_vectors(self.ax, lon_valid, lat_valid, u_valid, v_valid,
                                 plot_type, color, scale)
        
        gl = self.ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
        
        if title is None:
            wind_speed = np.sqrt(u_valid**2 + v_valid**2)
            
            if wind_speed.ndim == 2:
                wind_speed_valid = wind_speed[np.isfinite(wind_speed)]
                max_speed = np.max(wind_speed_valid)
                mean_speed = np.mean(wind_speed_valid)
            else:
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
        
        if lon_valid.ndim == 2:
            num_vectors = lon_valid.shape[0] * lon_valid.shape[1]
            wind_speed = np.sqrt(u_valid**2 + v_valid**2)
            wind_speed_valid = wind_speed[np.isfinite(wind_speed)]
            print(f"Plotted {num_vectors} wind vectors on {lon_valid.shape[0]}x{lon_valid.shape[1]} grid")
            if len(wind_speed_valid) > 0:
                print(f"Wind speed range: {np.min(wind_speed_valid):.1f} to {np.max(wind_speed_valid):.1f} m/s")
        else:
            print(f"Plotted {len(lon_valid)} wind vectors")
            print(f"Wind speed range: {np.min(np.sqrt(u_valid**2 + v_valid**2)):.1f} to {np.max(np.sqrt(u_valid**2 + v_valid**2)):.1f} m/s")
        
        return self.fig, self.ax
    
    def add_wind_overlay(
        self,
        ax: Axes,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        wind_config: Dict[str, Any],
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> None:
        """
        Adds wind vectors as an overlay onto an existing map axes using a configuration dictionary for flexible styling and data selection. This method extracts U/V wind components from the configuration, handles 3D data by selecting a specific vertical level, optionally regrids wind components to a regular lat-lon grid using linear interpolation, applies subsampling for performance (with automatic calculation if subsample=-1), filters invalid values, and renders wind barbs or arrows using the specified plot type and styling parameters without creating a new figure.

        Parameters:
            ax (Axes): Existing map axes (typically GeoAxes) to receive the wind overlay.
            lon (np.ndarray): 1D longitude array in degrees for wind vector positions.
            lat (np.ndarray): 1D latitude array in degrees for wind vector positions.
            wind_config (Dict[str, Any]): Configuration dictionary with required keys 'u_data' (ndarray), 'v_data' (ndarray) and optional keys 'plot_type' ('barbs'/'arrows'), 'subsample' (int, use -1 for automatic), 'color' (str), 'scale' (float), 'level_index' (int), 'grid_resolution' (float), 'regrid_method' (str), 'figsize' (tuple).
            lon_min (Optional[float]): Western longitude bound for regridding (required if grid_resolution specified).
            lon_max (Optional[float]): Eastern longitude bound for regridding (required if grid_resolution specified).
            lat_min (Optional[float]): Southern latitude bound for regridding (required if grid_resolution specified).
            lat_max (Optional[float]): Northern latitude bound for regridding (required if grid_resolution specified).
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinate information, auto-created from lon/lat if not provided (default: None).

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
        grid_resolution = wind_config.get('grid_resolution', None)
        regrid_method = wind_config.get('regrid_method', 'linear')
        figsize = wind_config.get('figsize', self.figsize)
        
        u_data = self.convert_to_numpy(u_data)
        v_data = self.convert_to_numpy(v_data)

        if getattr(u_data, 'ndim', 1) > 1:
            if level_index is not None:
                u_data = u_data[:, level_index]
                v_data = v_data[:, level_index]
            else:
                u_data = u_data[:, -1]
                v_data = v_data[:, -1]
        
        if grid_resolution is not None:
            if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
                raise ValueError("lon_min, lon_max, lat_min, lat_max must be provided when grid_resolution is specified")
            
            if dataset is None:
                lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
                lat_arr = lat if isinstance(lat, np.ndarray) else lat.values
                dataset = xr.Dataset({
                    'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
                    'latCell': xr.DataArray(lat_arr, dims=['nCells'])
                })
            
            lon, lat, u_data, v_data = self._regrid_wind_components(
                lon, lat, u_data, v_data, dataset,
                lon_min, lon_max, lat_min, lat_max,
                grid_resolution, regrid_method
            )
        
        if subsample == -1:
            if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
                raise ValueError("lon_min, lon_max, lat_min, lat_max must be provided when subsample=-1 for automatic calculation")
            
            lon_converted = self.convert_to_numpy(lon)
            u_converted = self.convert_to_numpy(u_data)
            
            if lon_converted.ndim == 2:
                num_points = np.sum(np.isfinite(lon_converted) & np.isfinite(u_converted))
            else:
                num_points = len(lon_converted[np.isfinite(lon_converted) & np.isfinite(u_converted)])
            
            subsample = self.calculate_optimal_subsample(
                num_points=num_points,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                figsize=figsize,
                plot_type=plot_type
            )
            print(f"Auto-calculated overlay subsample factor: {subsample} (from {num_points} points)")
        
        lon_valid, lat_valid, u_valid, v_valid = self._prepare_wind_data(
            lon, lat, u_data, v_data, subsample
        )
        
        if lon_valid.ndim == 2:
            num_vectors = np.sum(np.isfinite(lon_valid))
            if num_vectors == 0:
                print("Warning: No valid wind data for overlay")
                return
            print(f"Added {num_vectors} wind vectors as overlay")
        else:
            if len(lon_valid) == 0:
                print("Warning: No valid wind data for overlay")
                return
            print(f"Added {len(lon_valid)} wind vectors as overlay")
        
        self._render_wind_vectors(ax, lon_valid, lat_valid, u_valid, v_valid,
                                 plot_type, color, scale)

    def create_batch_wind_plots(
        self,
        processor: Any,
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
        show_background: bool = False,
        grid_resolution: Optional[float] = None,
        regrid_method: str = 'linear'
    ) -> List[str]:
        """
        Generate a complete batch of wind vector plots from an MPAS processor's loaded dataset and save them to disk with configurable formatting. This high-level method orchestrates the entire batch plotting workflow including data extraction, optional regridding, visualization creation, and file output for all available time steps in the dataset. The function iterates through each time index, extracts U and V wind components, creates individual wind plots with consistent styling and map features, and saves them in user-specified formats with descriptive filenames. Optional regridding transforms irregular MPAS mesh data onto regular latitude-longitude grids for smoother vector field visualization. The method integrates with MPAS2DProcessor for data access and supports both wind barbs and arrows with customizable density through subsampling.

        Parameters:
            processor (Any): MPAS2DProcessor instance with pre-loaded wind dataset containing u and v variables across time dimension.
            output_dir (str): Absolute or relative path to output directory where plot files will be saved with auto-generated filenames.
            lon_min (float): Western longitude boundary in degrees east defining the map extent for all plots in the batch.
            lon_max (float): Eastern longitude boundary in degrees east defining the map extent for all plots in the batch.
            lat_min (float): Southern latitude boundary in degrees north defining the map extent for all plots in the batch.
            lat_max (float): Northern latitude boundary in degrees north defining the map extent for all plots in the batch.
            u_variable (str): NetCDF variable name for U-component (eastward) wind in the dataset (default: 'u' for standard MPAS convention).
            v_variable (str): NetCDF variable name for V-component (northward) wind in the dataset (default: 'v' for standard MPAS convention).
            plot_type (str): Vector visualization style selector, either 'barbs' for meteorological wind barbs or 'arrows' for quiver arrows (default: 'barbs').
            formats (Optional[List[str]]): List of output file format extensions for saving plots such as 'png', 'pdf', 'jpg' (default: ['png'] for raster output).
            subsample (int): Spatial subsampling stride factor to reduce vector density where 1 means full resolution and higher values skip points (default: 1).
            scale (Optional[float]): Arrow length scaling factor for quiver plots where larger values produce shorter arrows (default: None which auto-scales or uses 200).
            show_background (bool): Flag to enable wind speed magnitude as colored background field under vectors (default: False for vectors only).
            grid_resolution (Optional[float]): Target grid spacing in degrees for regridding MPAS data to regular lat-lon grid (default: None disables regridding).
            regrid_method (str): Spatial interpolation algorithm for regridding, either 'linear' for smooth fields or 'nearest' for value preservation (default: 'linear').

        Returns:
            List[str]: Ordered list of base file paths (without format extensions) for all successfully created wind plot files corresponding to each time step.
        """
        if formats is None:
            formats = ['png']

        if not hasattr(processor, 'dataset') or processor.dataset is None:
            raise ValueError("Processor has no loaded dataset. Call load_2d_data() first.")

        dataset = processor.dataset
        _, _, time_size = MPASDateTimeUtils.validate_time_parameters(dataset, 0, False)

        created_files: List[str] = []

        for time_idx in range(time_size):
            u_data = processor.get_2d_variable_data(u_variable, time_idx)
            v_data = processor.get_2d_variable_data(v_variable, time_idx)
            lon, lat = processor.extract_2d_coordinates_for_variable(u_variable, u_data)

            fig, ax = self.create_wind_plot(
                lon, lat, u_data.values, v_data.values,
                lon_min, lon_max, lat_min, lat_max,
                plot_type=plot_type,
                subsample=subsample,
                scale=scale,
                show_background=show_background,
                time_stamp=None,
                grid_resolution=grid_resolution,
                regrid_method=regrid_method
            )

            try:
                time_str = MPASDateTimeUtils.get_time_info(dataset, time_idx, var_context='wind', verbose=False)
            except Exception:
                time_str = f"time_{time_idx}"

            base_name = f"mpas_wind_{u_variable}_{v_variable}_{plot_type}_valid_{time_str}"
            output_path = os.path.join(output_dir, base_name)

            self.add_timestamp_and_branding()
            self.save_plot(output_path, formats=formats)
            self.close_plot()

            created_files.append(output_path)

        return created_files
    
    def extract_2d_from_3d_wind(
        self,
        u_data_3d: Union[np.ndarray, xr.DataArray],
        v_data_3d: Union[np.ndarray, xr.DataArray],
        level_index: Optional[int] = None,
        level_value: Optional[float] = None,
        pressure_levels: Optional[np.ndarray] = None
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Extracts 2D horizontal wind components from 3D wind data at a specified vertical level using either direct indexing or pressure-based level matching. This utility method supports three selection modes: explicit level index, pressure value matching (finding nearest level), or default top-level extraction when no parameters are provided. It returns horizontal slices of U and V wind components suitable for 2D plotting and overlay operations.

        Parameters:
            u_data_3d (Union[np.ndarray, xr.DataArray]): 3D U-component wind array with shape (cells, levels) in m/s, may be NumPy or xarray.
            v_data_3d (Union[np.ndarray, xr.DataArray]): 3D V-component wind array with shape (cells, levels) in m/s, may be NumPy or xarray.
            level_index (Optional[int]): Direct vertical level index for extraction where 0 is bottom and higher values go upward (default: None).
            level_value (Optional[float]): Target pressure level value in Pa or hPa for nearest match requiring pressure_levels array (default: None).
            pressure_levels (Optional[np.ndarray]): 1D array of pressure level values corresponding to the vertical dimension for level_value matching (default: None).

        Returns:
            Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]: Two-element tuple containing 2D U and V wind component arrays at the selected level, preserving input type.
        """
        if level_index is not None:
            return u_data_3d[:, level_index], v_data_3d[:, level_index]
        
        if level_value is not None and pressure_levels is not None:
            level_idx = np.argmin(np.abs(pressure_levels - level_value))
            return u_data_3d[:, level_idx], v_data_3d[:, level_idx]
        
        return u_data_3d[:, -1], v_data_3d[:, -1]
    
    def compute_wind_speed_and_direction(
        self,
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes wind speed magnitude and meteorological direction from U and V wind components using standard vector mathematics and meteorological conventions. This utility method calculates wind speed as the vector magnitude (sqrt(u² + v²)) and converts mathematical angles to meteorological direction (0-360° where 0 is north, 90 is east) using the transformation (270 - arctan2(v, u)) % 360. It returns arrays matching the input shapes for element-wise wind analysis.

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