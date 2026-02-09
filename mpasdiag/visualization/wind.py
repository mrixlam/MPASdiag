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
# Import necessary libraries and modules for data handling, plotting, and MPAS-specific processing
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cartopy.feature as cfeature
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
from typing import Optional, Dict, Any, Tuple, Union, List

# Import MPASdiag modules for configuration, data processing, remapping, and visualization
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.visualization.base_visualizer import MPASVisualizer
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
        # Use instance figsize if not provided for calculation
        if figsize is None:
            figsize = self.figsize
        
        # Calculate map extent in degrees to understand spatial density of data points
        map_lon_range = lon_max - lon_min
        map_lat_range = lat_max - lat_min

        # Calculate map area in square degrees and figure area in square inches
        map_area = map_lon_range * map_lat_range
        
        # Calculate figure area in square inches for density estimation
        fig_width, fig_height = figsize
        fig_area = fig_width * fig_height
        
        # Determine target vector density based on plot type if not explicitly provided
        if target_density is None:
            if plot_type == 'barbs':
                target_vectors_per_inch = 3
            else:
                target_vectors_per_inch = 4
        else:
            # Allow user to specify custom target density for more control over subsampling
            target_vectors_per_inch = target_density
        
        # Calculate total target vectors based on figure area and desired density, then determine subsample factor
        target_total_vectors = int(fig_area * target_vectors_per_inch)
        
        # If the number of points is already less than or equal to the target, no subsampling is needed
        if num_points <= target_total_vectors:
            return 1
        
        # Calculate subsample factor as the square root of the ratio of total points to target vectors
        subsample = int(np.sqrt(num_points / target_total_vectors))
        
        # Enforce a maximum subsample factor to avoid excessive thinning of data
        subsample = max(1, min(subsample, 50))
        
        # Return the calculated subsample factor for use in data preparation before plotting
        return subsample
    
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
        # Convert input data to NumPy arrays if they are xarray DataArrays or dask arrays
        lon = self.convert_to_numpy(lon)
        lat = self.convert_to_numpy(lat)

        # Convert u and v data to NumPy arrays if they are xarray DataArrays or dask arrays
        u_data = self.convert_to_numpy(u_data)
        v_data = self.convert_to_numpy(v_data)
        
        # Determine if data is 2D (regridded) or 1D (unstructured mesh)
        is_2d = lon.ndim == 2
        
        # Apply subsampling to reduce data density for performance optimization 
        if is_2d:
            if subsample > 1:
                # For 2D data, apply subsampling along both dimensions while preserving grid structure for quiver rendering
                lon_sub = lon[::subsample, ::subsample]
                lat_sub = lat[::subsample, ::subsample]
                u_sub = u_data[::subsample, ::subsample]
                v_sub = v_data[::subsample, ::subsample]
            else:
                # No subsampling, return original 2D arrays
                lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
            # For 2D data, we typically want to preserve the grid structure even if there are NaN values
            return lon_sub, lat_sub, u_sub, v_sub
        else:
            if subsample > 1:
                # For 1D data, apply stride-based subsampling to reduce the number of points while maintaining representative coverage of the spatial domain
                indices = np.arange(0, len(lon), subsample)
                lon_sub = lon[indices]
                lat_sub = lat[indices]
                u_sub = u_data[indices]
                v_sub = v_data[indices]
            else:
                # No subsampling, return original 1D arrays
                lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
            
            # Filter out any points where either u or v component is NaN 
            valid_mask = ~(np.isnan(u_sub) | np.isnan(v_sub))
            lon_valid = lon_sub[valid_mask]
            lat_valid = lat_sub[valid_mask]
            u_valid = u_sub[valid_mask]
            v_valid = v_sub[valid_mask]
            
            # Return the prepared wind data arrays ready for plotting, with subsampling applied and invalid values removed
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
        # Render wind vectors based on the specified plot type using cartopy's PlateCarree transformation for geographic positioning
        if plot_type == 'barbs':
            # Use barbs for meteorological convention showing wind speed and direction with flags indicating speed thresholds.
            ax.barbs(lon, lat, u_data, v_data,
                    transform=ccrs.PlateCarree(), color=color, length=6)
        elif plot_type == 'arrows':
            if scale is None:
                scale = 200
            # Use quiver for simple arrow representation of wind vectors where length and orientation indicate magnitude and direction. 
            ax.quiver(lon, lat, u_data, v_data,
                     transform=ccrs.PlateCarree(), color=color, scale=scale)
        elif plot_type == 'streamlines':
            # Streamlines require 2D gridded data to show continuous flow trajectories colored by wind speed. Validate that input data is 2D and raise error if not.
            if lon.ndim == 1:
                raise ValueError("Streamlines require gridded data. Use grid_resolution parameter to enable regridding.")
            
            # Identify 1D longitude and latitude arrays from 2D meshgrid for streamplot function
            lon_1d = lon[0, :] if lon.ndim == 2 else lon
            lat_1d = lat[:, 0] if lat.ndim == 2 else lat
            
            # Calculate wind speed for coloring streamlines by magnitude
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            
            # Render streamlines with color mapping based on wind speed, using a colormap suitable for meteorological data
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
            
            # Add a colorbar for streamlines to indicate wind speed values corresponding to streamline colors
            from matplotlib import pyplot as plt
            cbar = plt.colorbar(strm.lines, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8, aspect=40)
            
            # Set colorbar label with appropriate units and styling for clarity in interpreting wind speed from streamline colors
            cbar.set_label('Wind Speed [m s$^{-1}$]', fontsize=12, fontweight='bold', labelpad=10)
        else:
            # Raise an error for unsupported plot types to ensure users are aware of valid options and prevent silent failures
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
        
        # Convert input data to xarray DataArrays if they are not already, as the remapping utility expects xarray inputs for coordinate handling and masking functionality
        u_xr = xr.DataArray(u_data, dims=['nCells']) if not isinstance(u_data, xr.DataArray) else u_data
        v_xr = xr.DataArray(v_data, dims=['nCells']) if not isinstance(v_data, xr.DataArray) else v_data
        
        # Regrid U component using the remap_mpas_to_latlon_with_masking utility which handles interpolation and masking of invalid points 
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
        
        # Regrid V component using the same parameters to ensure consistent grid and masking between U and V fields
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
        
        # Extract 1D longitude and latitude arrays from the regridded DataArrays for use in vector plotting functions 
        lon_grid = u_regridded.lon.values
        lat_grid = u_regridded.lat.values

        # Create 2D meshgrid coordinates from the 1D longitude and latitude arrays for proper vector plotting with quiver or streamplot 
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        # Flatten the regridded U and V components to 1D arrays for compatibility with matplotlib vector plotting functions 
        u_2d = u_regridded.values
        v_2d = v_regridded.values
        
        print(f"Regridded to {u_regridded.shape[0]}x{u_regridded.shape[1]} grid")
        
        # Return the regridded longitude, latitude, and wind component arrays ready for plotting on a regular grid with proper masking of invalid points
        return lon_2d, lat_2d, u_2d, v_2d
    
    def _setup_wind_plot_figure(self, projection: str) -> Tuple[Figure, GeoAxes]:
        """
        This internal helper method standardizes the figure and axes setup process for both create_wind_plot and add_wind_overlay methods, eliminating code duplication. The function determines the appropriate cartopy projection class based on the provided projection name, creates a matplotlib figure with the instance's configured figsize and dpi, and initializes a GeoAxes with the specified projection for geographic plotting. The method includes an assertion to validate that the created axes is indeed a GeoAxes instance, ensuring compatibility with cartopy's geographic plotting capabilities. This setup provides a consistent and professional canvas for rendering wind vectors with proper map projections, coastlines, borders, and gridlines.

        Parameters:
            projection (str): Cartopy projection name (e.g., 'PlateCarree', 'Mercator').

        Returns:
            Tuple[Figure, GeoAxes]: Matplotlib figure and GeoAxes instance for wind plotting.
        """
        # Determine the cartopy projection class based on the provided projection name and create an instance for use in GeoAxes
        proj = getattr(ccrs, projection)()

        # Create a matplotlib figure and GeoAxes with the specified projection
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi,
                              subplot_kw={'projection': proj})
        
        # Validate that the created axes is indeed a GeoAxes instance to ensure compatibility 
        assert isinstance(ax, GeoAxes), "Axes must be a GeoAxes instance"

        # Return the figure and axes for use in wind plotting
        return fig, ax
    
    def _handle_streamline_regridding(
        self,
        plot_type: str,
        grid_resolution: Optional[float]
    ) -> Optional[float]:
        """
        This internal helper function checks if the plot type is 'streamlines' and if no grid resolution is provided, it assigns a default resolution (e.g., 0.1°) to ensure that the necessary gridded data is available for streamline rendering. This approach eliminates code duplication between create_wind_plot and add_wind_overlay methods by centralizing the logic for handling streamline-specific requirements. The method returns the (possibly updated) grid resolution for use in the plotting workflow, ensuring that users can create streamline plots without needing to manually specify grid resolution when it is required for proper visualization.

        Parameters:
            plot_type (str): Vector plot type ('barbs', 'arrows', or 'streamlines').
            grid_resolution (Optional[float]): Current grid resolution setting.

        Returns:
            Optional[float]: Grid resolution (original or auto-set to 0.5° for streamlines).
        """
        # If the plot type is streamlines and no grid resolution is provided, set a default resolution 
        if plot_type == 'streamlines' and grid_resolution is None:
            grid_resolution = 0.1
            print(f"Streamlines require gridded data. Auto-enabling regridding with resolution: {grid_resolution}°")
        # Return the (possibly updated) grid resolution for use in the plotting workflow
        return grid_resolution
    
    def _ensure_dataset_exists(
        self,
        dataset: Optional[xr.Dataset],
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray]
    ) -> xr.Dataset:
        """
        This internal helper method checks if a dataset is already provided and returns it directly for use in remapping. If no dataset is provided, it creates a minimal xarray Dataset containing the necessary longitude and latitude variables (lonCell and latCell) extracted from the provided coordinate arrays. This approach eliminates code duplication between create_wind_plot and add_wind_overlay methods by centralizing the logic for ensuring that a compatible dataset exists for remapping operations, allowing both methods to seamlessly perform regridding when needed without requiring redundant dataset construction code.

        Parameters:
            dataset (Optional[xr.Dataset]): Existing dataset or None.
            lon (np.ndarray or xarray.DataArray): Longitude coordinates.
            lat (np.ndarray or xarray.DataArray): Latitude coordinates.

        Returns:
            xarray.Dataset: Original dataset or newly created minimal Dataset.
        """
        # If a dataset is already provided, return it directly for use in remapping. Otherwise, create a minimal Dataset 
        if dataset is not None:
            return dataset
        
        # Convert lon and lat to NumPy arrays if they are xarray DataArrays 
        lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
        lat_arr = lat if isinstance(lat, np.ndarray) else lat.values

        # Create and return a minimal xarray Dataset with lonCell and latCell variables required for remapping utilities
        return xr.Dataset({
            'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
            'latCell': xr.DataArray(lat_arr, dims=['nCells'])
        })
    
    def _calculate_valid_point_count(
        self,
        lon: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray]
    ) -> int:
        """
        This internal helper method calculates the number of valid (finite) data points by checking both longitude and U-component wind data arrays. It supports both 1D and 2D input arrays, converting them to NumPy arrays if necessary for consistent processing. The method returns the count of valid points, which is useful for determining appropriate subsampling rates or data quality assessments in wind vector visualizations.

        Parameters:
            lon (np.ndarray or xarray.DataArray): Longitude coordinate array.
            u_data (np.ndarray or xarray.DataArray): U-component wind data array.

        Returns:
            int: Count of valid finite data points.
        """
        # Convert input data to NumPy arrays if they are xarray DataArrays or dask arrays for consistent processing
        lon_converted = self.convert_to_numpy(lon)
        u_converted = self.convert_to_numpy(u_data)
        
        # Count valid points by checking for finite values in both longitude and U component arrays
        if lon_converted.ndim == 2:
            # Count valid points by checking for finite values in both longitude and U component arrays
            return int(np.sum(np.isfinite(lon_converted) & np.isfinite(u_converted)))
        else:
            # For 1D arrays, count valid points by checking finite values in the subsampled arrays.
            return int(len(lon_converted[np.isfinite(lon_converted) & np.isfinite(u_converted)]))
    
    def _setup_map_extent(
        self,
        ax: GeoAxes,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float
    ) -> None:
        """
        This internal helper method centralizes the logic for configuring map extents in both create_wind_plot and add_wind_overlay methods, eliminating code duplication. The function checks if the provided longitude and latitude boundaries indicate global coverage (spanning nearly 360° longitude and 180° latitude) and applies slight adjustments to the boundaries to prevent rendering issues with cartopy projections at the dateline. For global coverage, it sets the extent using adjusted boundaries to ensure proper rendering without artifacts. For regional plots, it sets the extent directly using the provided boundaries. This approach ensures that both global and regional wind plots are displayed correctly with appropriate geographic context.

        Parameters:
            ax (GeoAxes): GeoAxes instance for extent configuration.
            lon_min (float): Western longitude boundary.
            lon_max (float): Eastern longitude boundary.
            lat_min (float): Southern latitude boundary.
            lat_max (float): Northern latitude boundary.

        Returns:
            None: Modifies axes extent in place.
        """
        # Determine if the provided longitude and latitude boundaries indicate global coverage
        is_global_lon = (lon_max - lon_min) >= 359.0
        is_global_lat = (lat_max - lat_min) >= 179.0
        
        if is_global_lon and is_global_lat:
            # For global coverage, adjust boundaries slightly to avoid dateline artifacts in cartopy projections.
            adjusted_lon_min = max(lon_min, -179.99)
            adjusted_lon_max = min(lon_max, 179.99)
            adjusted_lat_min = max(lat_min, -89.99)
            adjusted_lat_max = min(lat_max, 89.99)

            # Set the map extent using the adjusted boundaries to ensure proper rendering without dateline artifacts for global coverage
            ax.set_extent([adjusted_lon_min, adjusted_lon_max, adjusted_lat_min, adjusted_lat_max], 
                         crs=ccrs.PlateCarree())
            
            # Print the adjusted extent for debugging purposes to confirm that global coverage is being handled correctly
            print(f"Using global extent (adjusted): [{adjusted_lon_min}, {adjusted_lon_max}, {adjusted_lat_min}, {adjusted_lat_max}]")
        else:
            # Set the map extent directly using the provided longitude and latitude boundaries for regional plots
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    def _add_map_features(
        self,
        ax: GeoAxes,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float
    ) -> None:
        """
        This internal helper method centralizes the logic for adding cartographic features to the map in both create_wind_plot and add_wind_overlay methods, eliminating code duplication. The function uses cartopy's built-in features to enhance the visual context of the map, including coastlines for geographic boundaries, borders for political boundaries, ocean and land shading for visual distinction of geographic areas. Additionally, it calls a method to add regional features such as lakes, rivers, and urban areas within the specified extent for enhanced detail. This approach ensures that all wind plots have a consistent and professional cartographic appearance with appropriate geographic context.

        Parameters:
            ax (GeoAxes): GeoAxes instance for feature addition.
            lon_min (float): Western longitude boundary for regional features.
            lon_max (float): Eastern longitude boundary for regional features.
            lat_min (float): Southern latitude boundary for regional features.
            lat_max (float): Northern latitude boundary for regional features.

        Returns:
            None: Adds features to axes in place.
        """
        # Add standard cartographic features such as coastlines, borders, ocean, and land shading for enhanced map aesthetics and geographic context.
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)

        # Add regional features such as lakes, rivers, and urban areas for enhanced map detail within the specified extent
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
    
    def _setup_gridlines(self, ax: GeoAxes) -> None:
        """
        This internal helper method centralizes the logic for configuring gridlines in both create_wind_plot and add_wind_overlay methods, eliminating code duplication. The function uses cartopy's gridlines functionality to add latitude and longitude lines with labels for improved readability and geographic reference. It disables top and right labels to avoid clutter and sets label styles for consistency with the overall map design. This approach ensures that all wind plots have clear and informative gridlines that enhance the interpretability of the geographic data being visualized.

        Parameters:
            ax (GeoAxes): GeoAxes instance for gridline configuration.

        Returns:
            None: Adds gridlines to axes in place.
        """
        # Add gridlines with labels on the left and bottom edges of the map for improved readability and geographic reference, following standard cartographic conventions.
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Disable top and right labels to avoid clutter and maintain a clean map appearance
        gl.top_labels = False
        gl.right_labels = False

        # Set label styles for longitude and latitude labels to ensure readability and consistency with the overall map design.
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    def _generate_wind_title(
        self,
        u_valid: np.ndarray,
        v_valid: np.ndarray,
        custom_title: Optional[str],
        level_info: Optional[str],
        time_stamp: Optional[datetime]
    ) -> str:
        """
        This internal helper method centralizes the logic for generating informative titles for wind plots in both create_wind_plot and add_wind_overlay methods, eliminating code duplication. If a custom title is provided, it uses that directly without calculating statistics. Otherwise, it calculates wind speed from the U and V components to compute maximum and mean wind speeds, which are included in the title for added context. The method also incorporates optional level information and formatted timestamp into the title to provide a comprehensive description of the wind data being visualized.

        Parameters:
            u_valid (np.ndarray): Valid U-component wind data.
            v_valid (np.ndarray): Valid V-component wind data.
            custom_title (Optional[str]): User-provided custom title.
            level_info (Optional[str]): Vertical level descriptor.
            time_stamp (Optional[datetime]): Time for title annotation.

        Returns:
            str: Complete formatted title string with statistics.
        """
        # If a custom title is provided, use it directly without calculating statistics
        if custom_title is not None:
            return custom_title
        
        # Calculate wind speed from U and V components to compute statistics for the title.
        wind_speed = np.sqrt(u_valid**2 + v_valid**2)
        
        if wind_speed.ndim == 2:
            # For 2D arrays, calculate statistics while ignoring NaN values to avoid skewing results due to invalid points in the regridded data.
            wind_speed_valid = wind_speed[np.isfinite(wind_speed)]
            max_speed = np.max(wind_speed_valid)
            mean_speed = np.mean(wind_speed_valid)
        else:
            # For 1D arrays, calculate statistics directly as all values should be valid after filtering.
            max_speed = np.max(wind_speed)
            mean_speed = np.mean(wind_speed)
        
        # Construct the title string with the base title
        title_parts = ["MPAS Wind Analysis"]

        # Append level information to the title if provided to indicate the vertical level of the wind data being visualized
        if level_info:
            title_parts.append(f"({level_info})")

        # Append formatted timestamp to the title if provided to indicate the time of the wind data being visualized.
        if time_stamp:
            time_str = time_stamp.strftime('%Y-%m-%d %H:%M UTC')
            title_parts.append(f"- {time_str}")
        
        # Combine the title parts into a single string and append wind speed statistics for maximum and mean wind speed
        title = " ".join(title_parts)
        title += f"\nMax: {max_speed:.1f} m/s, Mean: {mean_speed:.1f} m/s"

        # Return the complete title string for use in the plot annotation
        return title
    
    def _print_wind_diagnostics(
        self,
        lon_valid: np.ndarray,
        u_valid: np.ndarray,
        v_valid: np.ndarray
    ) -> None:
        """
        This internal helper method centralizes the logic for printing diagnostic information about the wind data being visualized in both create_wind_plot and add_wind_overlay methods, eliminating code duplication. The function calculates wind speed from the U and V components to provide insight into the range of wind speeds being plotted. For 2D arrays, it prints the number of vectors based on grid dimensions and the range of valid wind speeds while ignoring NaN values. For 1D arrays, it prints the number of valid vectors directly from the length of the valid longitude array and the range of wind speeds. This diagnostic information helps users understand the characteristics of the wind data being visualized and can assist in troubleshooting or interpreting the plot.

        Parameters:
            lon_valid (np.ndarray): Valid longitude array (1D or 2D).
            u_valid (np.ndarray): Valid U-component wind array.
            v_valid (np.ndarray): Valid V-component wind array.

        Returns:
            None: Prints diagnostic messages to console.
        """
        # Calculate wind speed from U and V components to provide diagnostic information about the plotted wind vectors
        wind_speed = np.sqrt(u_valid**2 + v_valid**2)
        
        if lon_valid.ndim == 2:
            # For 2D arrays, calculate the number of vectors based on the grid dimensions and print the range of wind speeds while ignoring NaN values
            num_vectors = lon_valid.shape[0] * lon_valid.shape[1]
            wind_speed_valid = wind_speed[np.isfinite(wind_speed)]
            print(f"Plotted {num_vectors} wind vectors on {lon_valid.shape[0]}x{lon_valid.shape[1]} grid")

            # Print the range of wind speeds for the valid points in the 2D array to provide insight into the wind conditions being visualized
            if len(wind_speed_valid) > 0:
                print(f"Wind speed range: {np.min(wind_speed_valid):.1f} to {np.max(wind_speed_valid):.1f} m/s")
        else:
            # For 1D arrays, calculate the number of valid vectors directly from the length of the valid longitude array 
            print(f"Plotted {len(lon_valid)} wind vectors")
            print(f"Wind speed range: {np.min(wind_speed):.1f} to {np.max(wind_speed):.1f} m/s")
    
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
        # Setup figure and axes with projection
        self.fig, self.ax = self._setup_wind_plot_figure(projection)
        
        # Auto-enable regridding for streamlines if needed
        grid_resolution = self._handle_streamline_regridding(plot_type, grid_resolution)
        
        # Regrid wind components to a regular lat-lon grid if grid_resolution is specified, ensuring that the dataset is available for coordinate extraction and masking 
        if grid_resolution is not None:
            dataset = self._ensure_dataset_exists(dataset, lon, lat)
            lon, lat, u_data, v_data = self._regrid_wind_components(
                lon, lat, u_data, v_data, dataset,
                lon_min, lon_max, lat_min, lat_max,
                grid_resolution, regrid_method
            )
        
        # If subsample is set to -1, automatically calculate an optimal subsampling factor based on the number of valid points and the map extent 
        if subsample == -1:
            num_points = self._calculate_valid_point_count(lon, u_data)
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
        
        # Prepare wind data by applying subsampling and filtering out invalid values, returning only the valid points for plotting.
        lon_valid, lat_valid, u_valid, v_valid = self._prepare_wind_data(
            lon, lat, u_data, v_data, subsample
        )
        
        # Check if there are valid points to plot after preparation and print a warning if not
        if len(lon_valid) == 0:
            print("Warning: No valid wind data found")
            return self.fig, self.ax
        
        # Set map extent and add features before plotting vectors to ensure proper layering of elements on the map
        self._setup_map_extent(self.ax, lon_min, lon_max, lat_min, lat_max)
        self._add_map_features(self.ax, lon_min, lon_max, lat_min, lat_max)
        
        # Set color for barbs and arrows; streamlines will use a colormap based on wind speed
        color = 'black'

        # For streamlines, color is determined by wind speed and colormap, so we only set color for barbs and arrows.
        self._render_wind_vectors(self.ax, lon_valid, lat_valid, u_valid, v_valid,
                                 plot_type, color, scale)
        
        # Setup gridlines after features to ensure they are on top for better visibility
        self._setup_gridlines(self.ax)
        
        # Generate and set title with wind statistics or use custom title if provided
        title = self._generate_wind_title(u_valid, v_valid, title, level_info, time_stamp)
        self.ax.set_title(title, fontsize=12, pad=20)
        
        # Print diagnostics about the plotted wind vectors for debugging and verification
        self._print_wind_diagnostics(lon_valid, u_valid, v_valid)
        
        # Return the figure and axes for display or saving
        return self.fig, self.ax
    
    def _extract_wind_config(self, wind_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method eliminates code duplication between create_wind_plot and add_wind_overlay by standardizing the process of extracting necessary parameters from the provided configuration dictionary. It handles both required parameters (u_data and v_data) and optional parameters with defaults (plot_type, subsample, color, scale, level_index, grid_resolution, regrid_method, figsize, original_units). The method also includes backward compatibility for color specification by checking for both 'color' and 'colors' keys. By consolidating this logic into a single method, we ensure consistency in how wind overlay configurations are processed across different parts of the codebase.

        Parameters:
            wind_config (Dict[str, Any]): Configuration dictionary containing wind overlay settings.

        Returns:
            Dict[str, Any]: Normalized configuration with all parameters extracted and defaults applied.
        """
        # Extract subsample parameter with default value of 1, allowing for automatic calculation when set to -1. 
        subsample_raw = wind_config.get('subsample', 1)

        # Attempt to convert subsample to an integer, defaulting to 1 if conversion fails or if the value is invalid (e.g., negative or zero).
        try:
            subsample = int(subsample_raw)
        except Exception:
            subsample = 1

        # Extract color with backward compatibility for 'colors' key, defaulting to 'black' if not specified
        color = wind_config.get('color', None)

        # If 'color' is not specified, check for 'colors' key for backward compatibility, and default to 'black' if neither is provided.
        if color is None:
            color = wind_config.get('colors', 'black')

        # Return a dictionary with all extracted and validated configuration parameters for use in the overlay plotting workflow
        return {
            'u_data': wind_config['u_data'],
            'v_data': wind_config['v_data'],
            'plot_type': wind_config.get('plot_type', 'barbs'),
            'subsample': subsample,
            'color': color,
            'scale': wind_config.get('scale', None),
            'level_index': wind_config.get('level_index', None),
            'grid_resolution': wind_config.get('grid_resolution', None),
            'regrid_method': wind_config.get('regrid_method', 'linear'),
            'figsize': wind_config.get('figsize', self.figsize),
            'original_units': wind_config.get('original_units', None)
        }
    
    def _convert_wind_units(
        self,
        u_data: np.ndarray,
        v_data: np.ndarray,
        original_units: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        If original units are not provided, this method attempts to auto-detect based on typical wind speed magnitudes and prints a warning if the data may not be in m/s. If original units are specified and differ from the display units (m/s), it attempts to convert using the UnitConverter utility, printing a warning if conversion fails. This method ensures that wind data is in consistent units for plotting while providing flexibility for different input unit conventions.

        Parameters:
            u_data (np.ndarray): U-component wind array in original units.
            v_data (np.ndarray): V-component wind array in original units.
            original_units (Optional[str]): Original units string (e.g., 'knots', 'km/h').

        Returns:
            Tuple[np.ndarray, np.ndarray]: Converted (u_data, v_data) arrays in m/s.
        """
        # If original units are not specified, attempt to auto-detect based on typical wind speed magnitudes 
        if original_units is None:
            u_mean = np.nanmean(np.abs(u_data))
            if u_mean > 100:
                print("Warning: Wind data may not be in m/s. Consider specifying 'original_units' in config.")
            return u_data, v_data
        
        # Define the display units for wind speed, which is typically meters per second (m/s) for meteorological data visualization.
        display_units = 'm/s'

        # If the original units are already the same as display units, return the data unchanged 
        if original_units == display_units:
            return u_data, v_data
        
        try:
            # Attempt to convert the U and V components from the original units to the display units using the UnitConverter utility
            u_converted = UnitConverter.convert_units(u_data, original_units, display_units)
            v_converted = UnitConverter.convert_units(v_data, original_units, display_units)

            # Convert the converted data to NumPy arrays if they are not already, ensuring that the returned data is in a consistent format 
            u_data = self.convert_to_numpy(u_converted)
            v_data = self.convert_to_numpy(v_converted)
            print(f"Converted overlay wind from {original_units} to {display_units}")
        except ValueError as e:
            # If conversion fails, print a warning and return the original data without modification 
            print(f"Warning: Could not convert overlay wind from {original_units} to {display_units}: {e}")
        
        # Return the (possibly converted) U and V component arrays for use in the overlay plotting workflow
        return u_data, v_data
    
    def _extract_2d_wind_slice(
        self,
        u_data: np.ndarray,
        v_data: np.ndarray,
        level_index: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method ensures that the wind data is in the correct 2D format for plotting regardless of the original dimensionality, eliminating code duplication between create_wind_plot and add_wind_overlay when dealing with multi-level wind data. If the input data is already 2D (i.e., has 2 or fewer dimensions), it returns the data directly without slicing. For 3D data, it extracts the specified vertical level index if provided, or defaults to using the top level (last index) of the 3D arrays for plotting. This approach allows both methods to seamlessly handle wind data with varying dimensionality while ensuring that the correct horizontal slice is used for visualization.

        Parameters:
            u_data (np.ndarray): U-component wind array (may be 2D or 3D).
            v_data (np.ndarray): V-component wind array (may be 2D or 3D).
            level_index (Optional[int]): Vertical level index for 3D data extraction.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 2D (u_data, v_data) arrays as horizontal slices.
        """
        # If the wind data is already 2D (i.e., has 2 or fewer dimensions), return it directly without slicing.
        if getattr(u_data, 'ndim', 1) <= 1:
            return u_data, v_data
        
        # For 3D data, extract the specified level index if provided, otherwise default to the top level (last index)
        if level_index is not None:
            return u_data[:, level_index], v_data[:, level_index]
        
        # If no level index is provided, default to using the top level (last index) of the 3D arrays for plotting
        return u_data[:, -1], v_data[:, -1]
    
    def _create_wind_dataset(
        self,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        dataset: Optional[xr.Dataset]
    ) -> xr.Dataset:
        """
        This method eliminates code duplication between create_wind_plot and add_wind_overlay by centralizing the logic for ensuring that a compatible dataset exists for remapping operations. If a dataset is already provided, it returns it directly for use in remapping. If no dataset is provided, it creates a minimal xarray Dataset containing the necessary longitude and latitude variables (lonCell and latCell) extracted from the provided coordinate arrays. This approach allows both methods to seamlessly perform regridding when needed without requiring redundant dataset construction code, ensuring that the remapping utilities have the necessary coordinate information regardless of how the method is called.

        Parameters:
            lon (np.ndarray or xarray.DataArray): Longitude coordinates.
            lat (np.ndarray or xarray.DataArray): Latitude coordinates.
            dataset (Optional[xarray.Dataset]): Existing dataset to return unchanged.

        Returns:
            xarray.Dataset: Dataset with lonCell and latCell variables.
        """
        # If a dataset is already provided, return it directly for use in remapping
        if dataset is not None:
            return dataset
        
        # Convert lon and lat to NumPy arrays if they are xarray DataArrays to ensure compatibility with the remapping utilities 
        lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
        lat_arr = lat if isinstance(lat, np.ndarray) else lat.values

        # Create and return a minimal xarray Dataset with lonCell and latCell variables required for remapping utilities
        return xr.Dataset({
            'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
            'latCell': xr.DataArray(lat_arr, dims=['nCells'])
        })
    
    def _calculate_auto_subsample(
        self,
        lon: Union[np.ndarray, xr.DataArray],
        u_data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        figsize: Tuple[float, float],
        plot_type: str
    ) -> int:
        """
        This method supports both 1D and 2D input arrays, converting them to NumPy arrays if necessary for consistent processing. For 2D arrays, it counts valid points by checking for finite values in both longitude and U-component arrays. For 1D arrays, it counts valid points by checking finite values in the subsampled arrays. The resulting count of valid points is then passed to the existing calculate_optimal_subsample method along with map extent and figure size to determine the appropriate subsampling factor for wind vector visualization. The calculated subsample factor is logged along with the number of valid points for debugging and verification purposes.

        Parameters:
            lon (np.ndarray or xarray.DataArray): Longitude coordinates for point counting.
            u_data (np.ndarray): U-component wind for validity checking.
            lon_min (float): Western longitude bound.
            lon_max (float): Eastern longitude bound.
            lat_min (float): Southern latitude bound.
            lat_max (float): Northern latitude bound.
            figsize (Tuple[float, float]): Figure dimensions for density calculation.
            plot_type (str): Plot type ('barbs' or 'arrows') affecting target density.

        Returns:
            int: Optimal subsample factor for wind overlay.
        """
        # Convert input data to NumPy arrays if they are xarray DataArrays
        lon_converted = self.convert_to_numpy(lon)
        u_converted = self.convert_to_numpy(u_data)
        
        if lon_converted.ndim == 2:
            # For 2D arrays, count valid points by checking finite values in both longitude and U-component arrays.
            num_points = np.sum(np.isfinite(lon_converted) & np.isfinite(u_converted))
        else:
            # For 1D arrays, count valid points by checking finite values in the subsampled arrays.
            num_points = len(lon_converted[np.isfinite(lon_converted) & np.isfinite(u_converted)])
        
        # Delegate to the existing calculate_optimal_subsample method to determine the appropriate subsampling factor
        subsample = self.calculate_optimal_subsample(
            num_points=num_points,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            figsize=figsize,
            plot_type=plot_type
        )

        # Log the calculated subsample factor along with the number of valid points 
        print(f"Auto-calculated overlay subsample factor: {subsample} (from {num_points} points)")

        # Return the calculated subsample factor 
        return subsample
    
    def _validate_and_log_wind_overlay(
        self,
        lon_valid: np.ndarray,
        lat_valid: np.ndarray
    ) -> bool:
        """
        For 2D arrays, it counts valid points by checking for finite values in the longitude array and prints the count of valid vectors added as an overlay. If no valid points are found, it prints a warning and returns False to indicate that the overlay cannot be plotted. For 1D arrays, it checks if the length of the valid longitude array is greater than zero to determine if there are valid points to plot, printing the count of valid vectors or a warning accordingly. If valid points are found, it returns True to indicate that the overlay can be plotted successfully.

        Parameters:
            lon_valid (np.ndarray): Validated longitude array (1D or 2D).
            lat_valid (np.ndarray): Validated latitude array (1D or 2D).

        Returns:
            bool: True if valid data exists, False otherwise.
        """
        if lon_valid.ndim == 2:
            # For 2D arrays, count valid points by checking for finite values in the longitude array
            num_vectors = np.sum(np.isfinite(lon_valid))

            # If no valid points are found, print a warning and return False to indicate that the overlay cannot be plotted.
            if num_vectors == 0:
                print("Warning: No valid wind data for overlay")
                return False
            print(f"Added {num_vectors} wind vectors as overlay")
        else:
            # For 1D arrays, count valid points by checking for finite values in the longitude array
            if len(lon_valid) == 0:
                print("Warning: No valid wind data for overlay")
                return False
            print(f"Added {len(lon_valid)} wind vectors as overlay")
        # If valid points are found, return True to indicate that the overlay can be plotted successfully.
        return True
    
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
        This streamlined method extracts configuration, converts units, handles 3D data, optionally regrids, calculates automatic subsampling, and validates data before rendering. Each step is delegated to focused helper methods for clarity and maintainability. It supports both 1D and 2D input arrays, converting them to NumPy arrays if necessary for consistent processing. The method ensures that the wind overlay is added to the existing map axes with appropriate styling and performance optimizations while providing informative diagnostics about the plotted data. By centralizing the logic for adding wind overlays in this method, we can easily maintain and extend the functionality without duplicating code across different parts of the codebase.

        Parameters:
            ax (Axes): Existing map axes (typically GeoAxes) to receive the wind overlay.
            lon (np.ndarray or xarray.DataArray): 1D longitude array in degrees for wind vector positions.
            lat (np.ndarray or xarray.DataArray): 1D latitude array in degrees for wind vector positions.
            wind_config (Dict[str, Any]): Configuration dictionary with required keys 'u_data', 'v_data' and optional styling/processing keys (see module documentation for complete list).
            lon_min (Optional[float]): Western longitude bound for regridding (required if grid_resolution specified).
            lon_max (Optional[float]): Eastern longitude bound for regridding (required if grid_resolution specified).
            lat_min (Optional[float]): Southern latitude bound for regridding (required if grid_resolution specified).
            lat_max (Optional[float]): Northern latitude bound for regridding (required if grid_resolution specified).
            dataset (Optional[xarray.Dataset]): MPAS dataset with coordinate information, auto-created from lon/lat if not provided.

        Returns:
            None: Draws wind overlay directly onto provided axes without returning objects.
        """
        # Extract and validate configuration parameters from the provided wind_config dictionary
        config = self._extract_wind_config(wind_config)
        
        # Convert input data to NumPy arrays if they are xarray DataArrays for consistent processing in subsequent steps
        u_data = self.convert_to_numpy(config['u_data'])
        v_data = self.convert_to_numpy(config['v_data'])

        # Convert wind component units from original units to display units (m/s) using the _convert_wind_units helper method
        u_data, v_data = self._convert_wind_units(u_data, v_data, config['original_units'])
        
        # Extract 2D slice from multi-dimensional wind arrays if necessary, using the provided level index or defaulting to the top level for 3D data
        u_data, v_data = self._extract_2d_wind_slice(u_data, v_data, config['level_index'])
        
        # Regrid wind components if requested
        if config['grid_resolution'] is not None:
            # Validate that longitude and latitude boundaries are provided for regridding to ensure that the remapping utilities have the necessary information 
            if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
                raise ValueError("lon_min, lon_max, lat_min, lat_max must be provided when grid_resolution is specified")
            
            # Ensure that a dataset is available for regridding, creating one from lon/lat if not provided
            dataset = self._create_wind_dataset(lon, lat, dataset)

            # Regrid the wind components to a regular lat-lon grid using the specified resolution and interpolation method
            lon, lat, u_data, v_data = self._regrid_wind_components(
                lon, lat, u_data, v_data, dataset,
                lon_min, lon_max, lat_min, lat_max,
                config['grid_resolution'], config['regrid_method']
            )
        
        # Extract subsample factor from config, allowing for automatic calculation when set to -1. 
        subsample = config['subsample']

        # If subsample is set to -1, automatically calculate an optimal subsampling factor based on the number of valid points and the map extent 
        if subsample == -1:
            # Validate that longitude and latitude boundaries are provided for subsample calculation
            if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
                raise ValueError("lon_min, lon_max, lat_min, lat_max must be provided when subsample=-1")
            
            # Calculate the optimal subsample factor based on the number of valid points and the map extent to prevent visual clutter in the overlay.
            subsample = self._calculate_auto_subsample(
                lon, u_data, lon_min, lon_max, lat_min, lat_max,
                config['figsize'], config['plot_type']
            )
        
        # Prepare wind data by applying subsampling and filtering out invalid values, returning only the valid points for plotting.
        lon_valid, lat_valid, u_valid, v_valid = self._prepare_wind_data(
            lon, lat, u_data, v_data, subsample
        )
        
        # Validate that there are valid points to plot and log the count before rendering the overlay. 
        if not self._validate_and_log_wind_overlay(lon_valid, lat_valid):
            return
        
        # Render wind vectors on the provided axes using the specified plot type and styling from the configuration
        self._render_wind_vectors(
            ax, lon_valid, lat_valid, u_valid, v_valid,
            config['plot_type'], config['color'], config['scale']
        )

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
        # Set default formats to ['png'] if not provided to ensure that there is at least one output format for saving the plots.
        if formats is None:
            formats = ['png']

        # Validate that the processor has a loaded dataset before attempting to access data for plotting
        if not hasattr(processor, 'dataset') or processor.dataset is None:
            raise ValueError("Processor has no loaded dataset. Call load_2d_data() first.")

        # Access the dataset from the processor to validate variable existence and time dimension for iteration.
        dataset = processor.dataset

        # Validate that the specified U and V variables exist in the dataset and that the time dimension is properly defined for iteration.
        _, _, time_size = MPASDateTimeUtils.validate_time_parameters(dataset, 0, False)

        # Initialize an empty list to keep track of the base file paths for all created plot files
        created_files: List[str] = []

        for time_idx in range(time_size):
            # Extract 2D U and V wind component data for the current time index using the processor's data access methods
            u_data = processor.get_2d_variable_data(u_variable, time_idx)
            v_data = processor.get_2d_variable_data(v_variable, time_idx)

            # Extract longitude and latitude coordinates for the current time index using the processor's utility method
            lon, lat = processor.extract_2d_coordinates_for_variable(u_variable, u_data)

            # Create the wind plot for the current time index using the extracted data and specified parameters
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

            # Attempt to extract time information from the dataset for the current time index to include in the plot title and filename
            try:
                time_str = MPASDateTimeUtils.get_time_info(dataset, time_idx, var_context='wind', verbose=False)
            except Exception:
                time_str = f"time_{time_idx}"

            # Generate a descriptive base filename for the plot using variable names, plot type, and time information 
            base_name = f"mpas_wind_{u_variable}_{v_variable}_{plot_type}_valid_{time_str}"
            output_path = os.path.join(output_dir, base_name)

            # Add timestamp and branding to the plot before saving to ensure consistent annotation across all generated files.
            self.add_timestamp_and_branding()

            # Save the plot in all specified formats and close the figure to free memory before the next iteration.
            self.save_plot(output_path, formats=formats)
            self.close_plot()

            # Append the base output path (without format extension) to the list of created files
            created_files.append(output_path)

        # After processing all time steps, return the list of created file paths for user reference and potential further processing.
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
        # If the input wind data is already 2D (i.e., has 2 or fewer dimensions), return it directly without slicing.
        if level_index is not None:
            return u_data_3d[:, level_index], v_data_3d[:, level_index]
        
        # If a specific pressure level value is provided along with the corresponding pressure levels array, find the index of the nearest pressure level 
        if level_value is not None and pressure_levels is not None:
            level_idx = np.argmin(np.abs(pressure_levels - level_value))
            return u_data_3d[:, level_idx], v_data_3d[:, level_idx]
        
        # Return the top level (last index) of the 3D arrays for plotting if no specific level selection parameters are provided, as a default behavior for multi-level data.
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
        # Calculate wind speed as the vector magnitude of the U and V components using the Pythagorean theorem.
        wind_speed = np.sqrt(u_data**2 + v_data**2)

        # Calculate wind direction in degrees using arctan2, converting from mathematical angles to meteorological convention 
        wind_direction = np.arctan2(v_data, u_data) * 180 / np.pi

        # Convert mathematical angles to meteorological direction where 0° is north, 90° is east, etc.
        wind_direction = (270 - wind_direction) % 360
        
        # Return the computed wind speed and direction arrays 
        return wind_speed, wind_direction