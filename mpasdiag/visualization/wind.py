#!/usr/bin/env python3

"""
MPASdiag Core Visualization Module: Wind Vector Plotting and Overlay

This module defines the `MPASWindPlotter` class, a specialized visualizer for rendering wind vector fields from MPAS model output on cartographic maps using matplotlib and cartopy. The plotter supports various options for customizing the appearance of wind vectors, such as scaling, color coding, and density. It includes automated unit conversion from model output to display units (m/s), flexible map projections via Cartopy, and geographic feature overlays (coastlines, borders, terrain). Visualization outputs include publication-quality single-panel wind vector maps, multi-panel comparison plots for model-observation evaluation, and batch processing capabilities for creating time series of wind analyses with consistent styling and automatic file naming. The class extends the base `MPASVisualizer` to leverage common plotting infrastructure while implementing the specific logic and styling for wind vector visualization, ensuring backward compatibility with the original mpas_analysis module while utilizing modern MPASdiag architecture and best practices for data handling and visualization in Python. 
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Import necessary libraries
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
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.visualization.styling import MPASVisualizationStyle


class MPASWindPlotter(MPASVisualizer):
    """ Specialized class for creating wind vector visualizations from MPAS model output with cartographic presentation. """
    
    def __init__(self: "MPASWindPlotter", 
                 figsize: Tuple[float, float] = (12, 10), 
                 dpi: int = 100) -> None:
        """
        This constructor initializes the MPASWindPlotter instance with specified figure dimensions and resolution. It calls the parent class constructor to set up the plotting environment and styling defaults. The figsize parameter allows users to customize the size of the plot canvas in inches, while the dpi parameter controls the resolution of the output image, affecting the quality and detail level of the visualization. By providing these parameters at initialization, users can easily configure the visual appearance of their wind vector plots according to their needs for publication or presentation. 

        Parameters:
            figsize (Tuple[float, float]): Figure dimensions in inches as (width, height) tuple (default: (12, 10)).
            dpi (int): Dots per inch for the output figure resolution, affecting quality and detail level (default: 100).

        Returns:
            None: Initializes the MPASWindPlotter instance with specified figure size and resolution.
        """
        super().__init__(figsize=figsize, dpi=dpi)
    
    def calculate_optimal_subsample(self: "MPASWindPlotter", 
                                    num_points: int,
                                    lon_min: float,
                                    lon_max: float,
                                    lat_min: float,
                                    lat_max: float,
                                    figsize: Optional[Tuple[float, float]] = None,
                                    plot_type: str = 'barbs',
                                    target_density: Optional[int] = None) -> int:
        """
        This internal helper method calculates an optimal subsampling factor for wind vector plotting based on the total number of data points, the geographic extent of the plot, and the desired density of vectors in the visualization. It takes into account the figure size and plot type to determine appropriate defaults for target vector density, allowing for more control over subsampling through an optional parameter. The method computes the map area in square degrees and the figure area in square inches to estimate how many vectors can be plotted without overcrowding the visualization. If the number of points exceeds the target, it calculates a subsample factor as the square root of the ratio of total points to target vectors, ensuring that the resulting plot remains clear and informative while maintaining performance. The method also enforces a maximum subsample factor to prevent excessive thinning of data. 

        Parameters:
            num_points (int): Total number of data points in the original dataset.
            lon_min (float): Minimum longitude of the plot extent in degrees east.
            lon_max (float): Maximum longitude of the plot extent in degrees east.
            lat_min (float): Minimum latitude of the plot extent in degrees north.
            lat_max (float): Maximum latitude of the plot extent in degrees north.
            figsize (Optional[Tuple[float, float]]): Figure dimensions in inches as (width, height) tuple for density estimation (default: None, uses instance figsize).
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine default target density (default: 'barbs').
            target_density (Optional[int]): Desired target vector density in vectors per inch for more control over subsampling (default: None, uses plot type defaults).

        Returns:
            int: Calculated subsample factor to reduce data density for plotting.  
        """
        # Use instance figsize if not provided for calculation
        if figsize is None:
            figsize = self.figsize
        
        # Calculate map extent in degrees to understand spatial density of data points
        map_lon_range = lon_max - lon_min
        map_lat_range = lat_max - lat_min

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
    
    def _prepare_wind_data(self: "MPASWindPlotter",
                           lon: Union[np.ndarray, xr.DataArray],
                           lat: Union[np.ndarray, xr.DataArray],
                           u_data: Union[np.ndarray, xr.DataArray],
                           v_data: Union[np.ndarray, xr.DataArray],
                           subsample: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal helper method prepares the wind data for plotting by converting input data to NumPy arrays if they are xarray DataArrays or dask arrays, applying subsampling to reduce data density for performance optimization, and filtering out any points where either U or V component is NaN to ensure that only valid vectors are plotted. The method handles both 1D arrays (for unstructured mesh data) and 2D arrays (for regridded data), applying subsampling appropriately while preserving the grid structure for 2D data. For 1D data, it applies stride-based subsampling and then filters out invalid points, while for 2D data, it applies subsampling along both dimensions and returns the resulting arrays without filtering to preserve the grid structure for quiver rendering. The prepared longitude, latitude, U component, and V component arrays are returned ready for use in the plotting functions. 

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinate array for vector positions in degrees north.
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s.
            v_data (Union[np.ndarray, xr.DataArray]): V-component wind data array in m/s.
            subsample (int): Subsampling factor to reduce data density for plotting (default: 1, no subsampling).       

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Prepared longitude, latitude, U component, and V component arrays ready for plotting.
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
    
    def _render_wind_vectors(self: "MPASWindPlotter",
                             ax: Axes, 
                             lon: np.ndarray,
                             lat: np.ndarray,
                             u_data: np.ndarray,
                             v_data: np.ndarray,
                             plot_type: str = 'barbs',
                             color: str = 'black',
                             scale: Optional[float] = None) -> None:
        """
        This internal helper method renders the wind vectors on the provided Matplotlib axes based on the specified plot type (barbs, arrows, or streamlines). It uses cartopy's PlateCarree transformation for geographic positioning of the vectors. For 'barbs', it renders meteorological barbs that indicate wind speed and direction with flags representing speed thresholds. For 'arrows', it uses quiver to render simple arrows where length and orientation indicate magnitude and direction, with an optional scale factor for arrow length. For 'streamlines', it requires 2D gridded data to show continuous flow trajectories colored by wind speed, and it validates that the input data is 2D before rendering. The method also includes error handling for unsupported plot types to ensure users are aware of valid options. By centralizing the rendering logic in this method, it allows for consistent styling and rendering of wind vectors across different types of visualizations while leveraging cartopy's capabilities for geographic plotting. 

        Parameters:
            ax (Axes): Matplotlib axes on which to render the wind vectors.
            lon (np.ndarray): Longitude array for vector positions in degrees east.
            lat (np.ndarray): Latitude array for vector positions in degrees north.
            u_data (np.ndarray): U-component wind data array in m/s.
            v_data (np.ndarray): V-component wind data array in m/s.
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine rendering method (default: 'barbs').
            color (str): Color for the wind vectors (default: 'black').
            scale (Optional[float]): Scale factor for arrow length when plot_type is 'arrows' (default: None, uses automatic scaling).

        Returns:
            None: Renders wind vectors on the provided axes based on the specified plot type and styling options.
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
            cbar = MPASVisualizationStyle.add_colorbar(
                plt.gcf(), ax, strm.lines,
                label='Wind Speed [m s$^{-1}$]', orientation='horizontal',
                fraction=0.03, pad=0.05, shrink=0.8, fmt=None, labelpad=10, label_pos='top', tick_labelsize=10
            )
        else:
            # Raise an error for unsupported plot types to ensure users are aware of valid options and prevent silent failures
            raise ValueError(f"plot_type must be 'barbs', 'arrows', or 'streamlines', got '{plot_type}'")
    
    def _regrid_wind_components(self: "MPASWindPlotter",
                                lon: Union[np.ndarray, xr.DataArray],
                                lat: Union[np.ndarray, xr.DataArray],
                                u_data: Union[np.ndarray, xr.DataArray],
                                v_data: Union[np.ndarray, xr.DataArray],
                                dataset: Optional[xr.Dataset],
                                lon_min: float,
                                lon_max: float,
                                lat_min: float,
                                lat_max: float,
                                grid_resolution: float,
                                regrid_method: str = 'linear') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal helper method performs regridding of the wind components from the original MPAS mesh to a regular latitude-longitude grid based on the specified geographic extent and grid resolution. It uses the remap_mpas_to_latlon_with_masking function to interpolate the U and V component data onto a regular grid defined by the provided longitude and latitude boundaries and desired grid resolution. The method converts input data to NumPy arrays if they are xarray DataArrays or dask arrays for consistent processing. It applies the specified interpolation method (e.g., 'linear', 'nearest', 'cubic') during regridding to ensure that the resulting gridded data is suitable for rendering with streamlines or quiver plots. The regridded longitude, latitude, U component, and V component arrays are returned as 2D arrays on the regular grid, ready for use in plotting functions that require gridded data. 

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Original longitude coordinate array for the MPAS mesh in degrees east.
            lat (Union[np.ndarray, xr.DataArray]): Original latitude coordinate array for the MPAS mesh in degrees north.
            u_data (Union[np.ndarray, xr.DataArray]): Original U-component wind data array on the MPAS mesh in m/s.
            v_data (Union[np.ndarray, xr.DataArray]): Original V-component wind data array on the MPAS mesh in m/s.
            dataset (Optional[xr.Dataset]): Original xarray Dataset containing the MPAS data for metadata reference during regridding (default: None).
            lon_min (float): Minimum longitude of the regridded grid extent in degrees east.
            lon_max (float): Maximum longitude of the regridded grid extent in degrees east.
            lat_min (float): Minimum latitude of the regridded grid extent in degrees north.
            lat_max (float): Maximum latitude of the regridded grid extent in degrees north.
            grid_resolution (float): Desired grid resolution for the regular latitude-longitude grid in degrees (e.g., 0.1 for 0.1° grid).
            regrid_method (str): Interpolation method to use during regridding (e.g., 'linear', 'nearest', 'cubic') (default: 'linear').

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Regridded longitude, latitude, U component, and V component arrays on the regular grid ready for plotting.
        """
        print(f"Regridding wind components to {grid_resolution}° grid using {regrid_method} interpolation...")

        lon_np = self.convert_to_numpy(lon)
        lat_np = self.convert_to_numpy(lat)
        u_np = self.convert_to_numpy(u_data)
        v_np = self.convert_to_numpy(v_data)

        lon_2d, lat_2d, u_2d = self._interpolate_to_grid(
            lon_np, lat_np, u_np,
            lon_min, lon_max, lat_min, lat_max,
            grid_resolution=grid_resolution, dataset=dataset,
            method=regrid_method,
        )
        _, _, v_2d = self._interpolate_to_grid(
            lon_np, lat_np, v_np,
            lon_min, lon_max, lat_min, lat_max,
            grid_resolution=grid_resolution, dataset=dataset,
            method=regrid_method,
        )

        return lon_2d, lat_2d, u_2d, v_2d
    
    def _setup_wind_plot_figure(self: "MPASWindPlotter", 
                                projection: str) -> Tuple[Figure, GeoAxes]:
        """
        This internal helper method sets up the Matplotlib figure and GeoAxes for wind plotting based on the specified Cartopy projection. It determines the appropriate Cartopy projection class based on the provided projection name and creates an instance for use in the GeoAxes. The method then creates a Matplotlib figure and GeoAxes with the specified projection, ensuring that the axes is a GeoAxes instance for compatibility with cartographic plotting. By centralizing the figure and axes setup in this method, it allows for consistent configuration of the plotting environment across different types of wind visualizations while leveraging Cartopy's capabilities for geographic projections. 

        Parameters:
            projection (str): Name of the Cartopy projection to use for the GeoAxes (e.g., 'PlateCarree', 'Mercator', 'LambertConformal').

        Returns:
            Tuple[Figure, GeoAxes]: Matplotlib Figure and GeoAxes instances configured with the specified Cartopy projection for wind plotting.
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
    
    def _handle_streamline_regridding(self: "MPASWindPlotter",
                                      plot_type: str,
                                      grid_resolution: Optional[float]) -> Optional[float]:
        """
        This internal helper method checks if the plot type is 'streamlines' and if no grid resolution is provided, it sets a default grid resolution (e.g., 0.1°) for regridding the wind data to ensure that streamlines can be rendered properly. Streamlines require gridded data to show continuous flow trajectories, so if the user has selected 'streamlines' as the plot type but has not provided a grid resolution, this method will automatically enable regridding with a default resolution suitable for visualizing streamlines. The method returns the (possibly updated) grid resolution for use in the plotting workflow, allowing downstream methods to know whether regridding is needed and what resolution to use for creating the regular latitude-longitude grid. 

        Parameters:
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine if regridding is required for streamlines.
            grid_resolution (Optional[float]): Original grid resolution provided by the user, which may be None if not specified.

        Returns:
            Optional[float]: Updated grid resolution for regridding if plot type is 'streamlines' and no resolution was provided, otherwise returns the original grid resolution.
        """
        # If the plot type is streamlines and no grid resolution is provided, set a default resolution 
        if plot_type == 'streamlines' and grid_resolution is None:
            grid_resolution = 0.1
            print(f"Streamlines require gridded data. Auto-enabling regridding with resolution: {grid_resolution}°")
        # Return the (possibly updated) grid resolution for use in the plotting workflow
        return grid_resolution
    
    def _calculate_valid_point_count(self: "MPASWindPlotter",
                                     lon: Union[np.ndarray, xr.DataArray],
                                     u_data: Union[np.ndarray, xr.DataArray]) -> int:
        """
        This internal helper method calculates the count of valid finite data points in the longitude and U component arrays, which is useful for diagnostics and understanding the density of valid wind vectors being plotted. It converts input data to NumPy arrays if they are xarray DataArrays or dask arrays for consistent processing. The method checks for finite values in both the longitude and U component arrays to ensure that only valid points are counted, as NaN values would indicate missing or invalid data that should not be included in the plot. For 2D arrays, it counts valid points by checking for finite values across the entire grid, while for 1D arrays, it counts valid points directly from the subsampled arrays. The resulting count of valid finite data points is returned as an integer for use in diagnostics or logging. 

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east.
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s.

        Returns:
            int: Count of valid finite data points in the longitude and U component arrays.
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
    
    def _setup_map_extent(self: "MPASWindPlotter",
                          ax: GeoAxes,
                          lon_min: float,
                          lon_max: float,
                          lat_min: float,
                          lat_max: float) -> None:
        """
        This internal helper method sets up the map extent for the GeoAxes based on the provided longitude and latitude boundaries. It determines if the specified boundaries indicate global coverage (e.g., longitude range of 360° and latitude range of 180°) and adjusts the extent accordingly to ensure proper rendering without dateline artifacts in cartopy projections. For global coverage, it slightly adjusts the longitude and latitude boundaries to avoid issues with rendering across the dateline, while for regional plots, it sets the extent directly using the provided boundaries. By centralizing the logic for setting map extent in this method, it ensures consistent handling of geographic extents across different types of wind visualizations while leveraging cartopy's capabilities for geographic plotting. 

        Parameters:
            ax (GeoAxes): GeoAxes instance on which to set the map extent.
            lon_min (float): Minimum longitude boundary for the map extent in degrees east.
            lon_max (float): Maximum longitude boundary for the map extent in degrees east.
            lat_min (float): Minimum latitude boundary for the map extent in degrees north.
            lat_max (float): Maximum latitude boundary for the map extent in degrees north.

        Returns:
            None: Sets the map extent on the provided GeoAxes based on the specified longitude and latitude boundaries, with adjustments for global coverage if necessary.
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
    
    def _add_map_features(self: "MPASWindPlotter",
                          ax: GeoAxes,
                          lon_min: float,
                          lon_max: float,
                          lat_min: float,
                          lat_max: float) -> None:
        """
        This internal helper method adds standard cartographic features such as coastlines, borders, ocean shading, and land shading to the GeoAxes for enhanced map aesthetics and geographic context. It uses cartopy's built-in features to add these elements with specified styling options (e.g., linewidth, color, alpha) to improve the visual appeal and readability of the wind vector plots. Additionally, it calls another method to add regional features such as lakes, rivers, and urban areas within the specified geographic extent for further detail in the map presentation. By centralizing the logic for adding map features in this method, it ensures consistent styling and inclusion of geographic context across different types of wind visualizations while leveraging cartopy's capabilities for feature rendering. 

        Parameters:
            ax (GeoAxes): GeoAxes instance on which to add cartographic features.
            lon_min (float): Minimum longitude boundary for determining regional features in degrees east.
            lon_max (float): Maximum longitude boundary for determining regional features in degrees east.
            lat_min (float): Minimum latitude boundary for determining regional features in degrees north.
            lat_max (float): Maximum latitude boundary for determining regional features in degrees north.

        Returns:
            None: Adds standard cartographic features and regional features to the provided GeoAxes for enhanced map aesthetics and geographic context.
        """
        # Add standard cartographic features such as coastlines, borders, ocean, and land shading for enhanced map aesthetics and geographic context.
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)

        # Add regional features such as lakes, rivers, and urban areas for enhanced map detail within the specified extent
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
    
    def _generate_wind_title(self: "MPASWindPlotter",
                             u_valid: np.ndarray,
                             v_valid: np.ndarray,
                             custom_title: Optional[str],
                             level_info: Optional[str],
                             time_stamp: Optional[datetime]) -> str:
        """
        This internal helper method generates a title string for the wind plot based on the provided U and V component data, custom title, level information, and timestamp. If a custom title is provided, it uses that directly without calculating statistics. Otherwise, it calculates wind speed from the U and V components to compute statistics such as maximum and mean wind speed for inclusion in the title. For 2D arrays, it calculates statistics while ignoring NaN values to avoid skewing results due to invalid points in the regridded data, while for 1D arrays it calculates statistics directly as all values should be valid after filtering. The method constructs the title string with a base title, appends level information if provided, and formats the timestamp if available. Finally, it combines all parts into a complete title string that includes wind speed statistics for maximum and mean wind speed, which can be used for annotation in the plot to provide context about the conditions being visualized.

        Parameters:
            u_valid (np.ndarray): Valid U-component wind array used for calculating wind speed statistics.
            v_valid (np.ndarray): Valid V-component wind array used for calculating wind speed statistics.
            custom_title (Optional[str]): Custom title string provided by the user, which takes precedence over calculated statistics if provided.
            level_info (Optional[str]): String containing level information (e.g., "Surface", "850 hPa") to include in the title for context about the vertical level of the wind data.
            time_stamp (Optional[datetime]): Datetime object representing the timestamp of the wind data to include in the title for temporal context.

        Returns:
            str: Generated title string for the wind plot, including custom title if provided or calculated statistics with level and time information if not.
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
    
    def _print_wind_diagnostics(self: "MPASWindPlotter",
                                lon_valid: np.ndarray,
                                u_valid: np.ndarray,
                                v_valid: np.ndarray) -> None:
        """
        This internal helper method prints diagnostic messages to the console about the plotted wind vectors, including the number of vectors plotted and the range of wind speeds. It calculates wind speed from the U and V components to provide insight into the conditions being visualized. For 2D arrays, it calculates the number of vectors based on the grid dimensions and prints the range of wind speeds while ignoring NaN values to avoid skewing the statistics due to invalid points in the regridded data. For 1D arrays, it calculates the number of valid vectors directly from the length of the valid longitude array and prints the range of wind speeds for all valid points. This diagnostic information can be useful for debugging and understanding the density and characteristics of the wind data being visualized in the plot. 

        Parameters:
            lon_valid (np.ndarray): Valid longitude array for the plotted wind vectors in degrees east.
            u_valid (np.ndarray): Valid U-component wind array used for calculating wind speed diagnostics.
            v_valid (np.ndarray): Valid V-component wind array used for calculating wind speed diagnostics.

        Returns:
            None: Prints diagnostic messages about the plotted wind vectors, including the number of vectors and the range of wind speeds, to the console for debugging purposes.
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
    
    def create_wind_plot(self: "MPASWindPlotter",
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
                         dataset: Optional[xr.Dataset] = None) -> Tuple[Figure, Axes]:
        """
        This method creates a wind plot based on the provided longitude, latitude, U and V component data, and various plotting options. It sets up the figure and axes with the specified Cartopy projection, handles regridding for streamlines if necessary, prepares the wind data by applying subsampling and filtering out invalid values, sets the map extent and adds features before plotting vectors to ensure proper layering of elements on the map, renders the wind vectors based on the specified plot type (barbs, arrows, or streamlines), generates a title for the plot based on wind speed statistics or a custom title if provided, and prints diagnostic information about the plotted wind vectors. The method returns the Matplotlib figure and GeoAxes containing the wind plot for further customization or saving. 

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinate array for vector positions in degrees north.
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s.
            v_data (Union[np.ndarray, xr.DataArray]): V-component wind data array in m/s.
            lon_min (float): Minimum longitude boundary for the map extent in degrees east.
            lon_max (float): Maximum longitude boundary for the map extent in degrees east.
            lat_min (float): Minimum latitude boundary for the map extent in degrees north.
            lat_max (float): Maximum latitude boundary for the map extent in degrees north.
            wind_level (str): String indicating the vertical level of the wind data (e.g., "surface", "850 hPa") for title annotation (default: "surface").
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine rendering method (default: 'barbs').
            subsample (int): Subsampling factor to reduce the number of plotted vectors for performance optimization; if set to -1, it will be automatically calculated based on the number of valid points and map extent (default: 1).
            scale (Optional[float]): Scale factor for arrow length when plot_type is 'arrows' (default: None, uses automatic scaling).
            show_background (bool): Whether to show a background colormap based on wind speed when plot_type is 'streamlines' (default: False).
            bg_colormap (str): Colormap to use for background shading when show_background is True and plot_type is 'streamlines' (default: "viridis").
            title (Optional[str]): Custom title string for the plot; if None, a title will be generated based on wind speed statistics and level information (default: None).
            time_stamp (Optional[datetime]): Datetime object representing the timestamp of the wind data to include in the title for temporal context (default: None).
            projection (str): Name of the Cartopy projection to use for the GeoAxes (e.g., 'PlateCarree', 'Mercator', 'LambertConformal') (default: 'PlateCarree').
            level_info (Optional[str]): String containing level information to include in the title for context about the vertical level of the wind data being visualized (default: None).

        Returns:
            Tuple[Figure, Axes]: Matplotlib Figure and GeoAxes instances containing the wind plot for further customization or saving.
        """
        # Setup figure and axes with projection
        self.fig, self.ax = self._setup_wind_plot_figure(projection)
        
        # Auto-enable regridding for streamlines if needed
        grid_resolution = self._handle_streamline_regridding(plot_type, grid_resolution)
        
        # Regrid wind components to a regular lat-lon grid if grid_resolution is specified, ensuring that the dataset is available for coordinate extraction and masking 
        if grid_resolution is not None:
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
        self._add_gridlines(ccrs.PlateCarree())
        
        # Generate and set title with wind statistics or use custom title if provided
        title = self._generate_wind_title(u_valid, v_valid, title, level_info, time_stamp)
        self.ax.set_title(title, fontsize=12, pad=20)
        
        # Print diagnostics about the plotted wind vectors for debugging and verification
        self._print_wind_diagnostics(lon_valid, u_valid, v_valid)
        
        # Return the figure and axes for display or saving
        return self.fig, self.ax
    
    def _extract_wind_config(self: "MPASWindPlotter", 
                             wind_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        This internal helper method extracts and normalizes the wind overlay configuration parameters from the provided configuration dictionary. It handles backward compatibility for color specification by checking for both 'color' and 'colors' keys, and it ensures that the subsample parameter is properly converted to an integer with a default value of 1 if conversion fails or if the value is invalid. The method returns a normalized configuration dictionary containing all necessary parameters for creating a wind overlay, including U and V component data, plot type, subsampling factor, color, scale, level index, grid resolution, regridding method, figure size, and original units for potential unit conversion. By centralizing the extraction and normalization of configuration parameters in this method, it allows for consistent handling of wind overlay settings across different parts of the codebase while providing flexibility for users to specify their desired options in a single configuration dictionary. 

        Parameters:
            wind_config (Dict[str, Any]): Dictionary containing the wind overlay configuration parameters, which may include keys such as 'u_data', 'v_data', 'plot_type', 'subsample', 'color' or 'colors', 'scale', 'level_index', 'grid_resolution', 'regrid_method', 'figsize', and 'original_units'. 

        Returns:
            Dict[str, Any]: Normalized dictionary containing the extracted and validated wind overlay configuration parameters for use in the overlay plotting workflow.
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
    
    def _convert_wind_units(self: "MPASWindPlotter",
                            u_data: np.ndarray,
                            v_data: np.ndarray,
                            original_units: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal helper method converts the U and V component wind data from the original units specified in the configuration to the display units used for plotting (typically meters per second, m/s). If the original units are not specified, it attempts to auto-detect based on typical wind speed magnitudes and prints a warning if the data may not be in m/s. If the original units are already the same as the display units, it returns the data unchanged. Otherwise, it uses a UnitConverter utility to perform the conversion and handles any exceptions that may arise during conversion by printing a warning and returning the original data without modification. This method ensures that the wind data is in consistent units for accurate representation in the plot while providing flexibility for users to specify their original data units. 

        Parameters:
            u_data (np.ndarray): U-component wind array to be converted.
            v_data (np.ndarray): V-component wind array to be converted.
            original_units (Optional[str]): String specifying the original units of the wind data (e.g., 'm/s', 'km/h', 'knots'); if None, it will attempt to auto-detect based on typical magnitudes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Converted U and V component wind arrays in display units (m/s) for use in the overlay plotting workflow.
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
    
    def _extract_2d_wind_slice(self: "MPASWindPlotter",
                               u_data: np.ndarray,
                               v_data: np.ndarray,
                               level_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal helper method extracts a 2D horizontal slice of the U and V component wind data from potentially 3D arrays based on the specified vertical level index. If the input wind data is already 2D (i.e., has 2 or fewer dimensions), it returns the data directly without slicing. For 3D data, it checks if a level index is provided; if so, it extracts the corresponding horizontal slice at that level index. If no level index is provided, it defaults to using the top level (last index) of the 3D arrays for plotting. This method ensures that the wind data is in the correct 2D format for plotting while providing flexibility for users to specify which vertical level to visualize when working with 3D wind data. 

        Parameters:
            u_data (np.ndarray): U-component wind data array, which may be 2D or 3D.
            v_data (np.ndarray): V-component wind data array, which may be 2D or 3D.
            level_index (Optional[int]): Integer index specifying which vertical level to extract for 3D data; if None, defaults to the top level (last index).

        Returns:
            Tuple[np.ndarray, np.ndarray]: 2D horizontal slices of the U and V component wind data for use in the overlay plotting workflow.
        """
        # If the wind data is already 2D (i.e., has 2 or fewer dimensions), return it directly without slicing.
        if getattr(u_data, 'ndim', 1) <= 1:
            return u_data, v_data
        
        # For 3D data, extract the specified level index if provided, otherwise default to the top level (last index)
        if level_index is not None:
            return u_data[:, level_index], v_data[:, level_index]
        
        # If no level index is provided, default to using the top level (last index) of the 3D arrays for plotting
        return u_data[:, -1], v_data[:, -1]
    
    def _calculate_auto_subsample(self: "MPASWindPlotter",
                                  lon: Union[np.ndarray, xr.DataArray],
                                  u_data: np.ndarray,
                                  lon_min: float,
                                  lon_max: float,
                                  lat_min: float,
                                  lat_max: float,
                                  figsize: Tuple[float, float],
                                  plot_type: str) -> int:
        """
        This internal helper method calculates an optimal subsampling factor for the wind overlay based on the number of valid points in the longitude and U component arrays, as well as the map extent and figure size. It first converts the input longitude and U component data to NumPy arrays if they are xarray DataArrays for consistent processing. Then, it counts the number of valid points by checking for finite values in both the longitude and U component arrays, which is important for determining how many vectors would be plotted without subsampling. For 2D arrays, it counts valid points across the entire grid, while for 1D arrays it counts valid points directly from the subsampled arrays. Finally, it delegates to the existing calculate_optimal_subsample method to determine the appropriate subsampling factor based on the count of valid points and the map extent, and it logs the calculated subsample factor along with the number of valid points for diagnostics. This method allows for automatic optimization of vector density in the plot while ensuring that performance is maintained when dealing with large datasets. 

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east, which may be 1D or 2D.
            u_data (np.ndarray): U-component wind data array in m/s, which may be 2D or 3D.
            lon_min (float): Minimum longitude boundary for the map extent in degrees east.
            lon_max (float): Maximum longitude boundary for the map extent in degrees east.
            lat_min (float): Minimum latitude boundary for the map extent in degrees north.
            lat_max (float): Maximum latitude boundary for the map extent in degrees north.
            figsize (Tuple[float, float]): Figure size in inches (width, height) for calculating subsampling based on the map area.
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine subsampling strategy based on the expected density of vectors for each plot type.

        Returns:
            int: Calculated optimal subsampling factor for the wind overlay based on the number of valid points and map extent, which can be used to reduce the number of plotted vectors for performance optimization.
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
    
    def _validate_and_log_wind_overlay(self: "MPASWindPlotter",
                                       lon_valid: np.ndarray,
                                       lat_valid: np.ndarray) -> bool:
        """
        This internal helper method validates the presence of valid wind data points for the overlay and logs the number of vectors that will be plotted. It checks the dimensions of the valid longitude array to determine how to count valid points, which is important for providing accurate diagnostics about the density of the wind vectors being plotted. For 2D arrays, it counts valid points by checking for finite values in the longitude array across the entire grid, while for 1D arrays it counts valid points directly from the length of the valid longitude array. If no valid points are found, it prints a warning message and returns False to indicate that the overlay cannot be plotted. If valid points are found, it logs the number of wind vectors that will be added as an overlay and returns True to indicate that the overlay can be plotted successfully. This method ensures that users are informed about the presence or absence of valid data for the wind overlay and provides insight into how many vectors will be visualized in the plot. 

        Parameters:
            lon_valid (np.ndarray): Valid longitude array for the plotted wind vectors in degrees east, which may be 1D or 2D.
            lat_valid (np.ndarray): Valid latitude array for the plotted wind vectors in degrees north, which may be 1D or 2D.

        Returns:
            bool: True if valid wind data points are present for the overlay and it can be plotted successfully; False if no valid points are found, indicating that the overlay cannot be plotted.
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
    
    def add_wind_overlay(self: "MPASWindPlotter",
                         ax: Axes,
                         lon: Union[np.ndarray, xr.DataArray],
                         lat: Union[np.ndarray, xr.DataArray],
                         wind_config: Dict[str, Any],
                         lon_min: Optional[float] = None,
                         lon_max: Optional[float] = None,
                         lat_min: Optional[float] = None,
                         lat_max: Optional[float] = None,
                         dataset: Optional[xr.Dataset] = None) -> None:
        """
        This method adds a wind vector overlay to an existing map axes (typically GeoAxes) based on the provided longitude, latitude, and wind configuration. It extracts and validates the necessary parameters from the wind configuration dictionary, converts the input data to NumPy arrays if they are xarray DataArrays for consistent processing, converts wind component units from original units to display units (m/s) if necessary, extracts a 2D slice from multi-dimensional wind arrays if needed, regrids the wind components to a regular lat-lon grid if requested, calculates an optimal subsampling factor if subsample is set to -1, prepares the wind data by applying subsampling and filtering out invalid values, validates that there are valid points to plot and logs the count before rendering the overlay, and finally renders the wind vectors on the provided axes using the specified plot type and styling from the configuration. This method allows for flexible addition of wind overlays to existing maps while ensuring that the data is properly processed and visualized according to user specifications. 

        Parameters:
            ax (Axes): Matplotlib Axes (typically GeoAxes) on which to add the wind vector overlay.
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east, which may be 1D or 2D.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinate array for vector positions in degrees north, which may be 1D or 2D.
            wind_config (Dict[str, Any]): Dictionary containing the wind overlay configuration parameters, which may include keys such as 'u_data', 'v_data', 'plot_type', 'subsample', 'color' or 'colors', 'scale', 'level_index', 'grid_resolution', 'regrid_method', 'figsize', and 'original_units'.
            lon_min (Optional[float]): Minimum longitude boundary for the map extent in degrees east, required if grid_resolution is specified for regridding (default: None).
            lon_max (Optional[float]): Maximum longitude boundary for the map extent in degrees east, required if grid_resolution is specified for regridding (default: None).
            lat_min (Optional[float]): Minimum latitude boundary for the map extent in degrees north, required if grid_resolution is specified for regridding (default: None).
            lat_max (Optional[float]): Maximum latitude boundary for the map extent in degrees north, required if grid_resolution is specified for regridding (default: None).
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinate information, auto-created from lon/lat if not provided. Required for regridding to ensure that the remapping utilities have access to the necessary coordinate information (default: None).

        Returns:
            None: This method modifies the provided axes in place by adding the wind vector overlay based on the specified configuration and does not return any value.
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

    def create_batch_wind_plots(self: "MPASWindPlotter",
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
                                regrid_method: str = 'linear') -> List[str]:
        """
        This method creates a series of wind plots for each time step in the MPAS dataset using the specified U and V component variable names. It iterates through each time step, extracts the relevant wind data and coordinates, creates a wind plot using the create_wind_plot method with the specified parameters, generates a descriptive filename based on the variable names, plot type, and time information, adds timestamps and branding to the plot, saves the plot in all specified formats, and keeps track of the created file paths. The method returns a list of base file paths (without format extensions) for all successfully created wind plot files corresponding to each time step. This allows users to generate a comprehensive set of wind visualizations across multiple time steps while maintaining consistent styling and organization of output files. 

        Parameters:
            processor (Any): An instance of the MPASDataProcessor class that has loaded the dataset containing the U and V component variables for plotting.
            output_dir (str): Directory path where the generated wind plot files will be saved.
            lon_min (float): Minimum longitude boundary for the map extent in degrees east.
            lon_max (float): Maximum longitude boundary for the map extent in degrees east.
            lat_min (float): Minimum latitude boundary for the map extent in degrees north.
            lat_max (float): Maximum latitude boundary for the map extent in degrees north.
            u_variable (str): Name of the U-component variable in the dataset to be used for plotting (default: 'u').   
            v_variable (str): Name of the V-component variable in the dataset to be used for plotting (default: 'v').
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine rendering method (default: 'barbs').
            formats (Optional[List[str]]): List of file formats (e.g., ['png', 'pdf']) to save the plots in; if None, defaults to ['png'] (default: None).
            subsample (int): Subsampling factor to reduce the number of plotted vectors for performance optimization; if set to -1, it will be automatically calculated based on the number of valid points and map extent (default: 1).
            scale (Optional[float]): Scale factor for arrow length when plot_type is 'arrows' (default: None, uses automatic scaling).
            show_background (bool): Whether to show a background colormap based on wind speed when plot_type is 'streamlines' (default: False).
            grid_resolution (Optional[float]): Grid resolution in degrees for regridding the wind data to a regular lat-lon grid; if None, no regridding is performed (default: None).
            regrid_method (str): Interpolation method to use for regridding the wind data when grid_resolution is specified (e.g., 'nearest', 'linear', 'cubic') (default: 'linear').

        Returns:
            List[str]: A list of base file paths (without format extensions) for all successfully created wind plot files corresponding to each time step, which can be used for reference or further processing.
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
            _, _ = self.create_wind_plot(
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
    
    def extract_2d_from_3d_wind(self: "MPASWindPlotter", 
                                u_data_3d: Union[np.ndarray, xr.DataArray],
                                v_data_3d: Union[np.ndarray, xr.DataArray],
                                level_index: Optional[int] = None,
                                level_value: Optional[float] = None,
                                pressure_levels: Optional[np.ndarray] = None) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        This method extracts a 2D horizontal slice of the U and V component wind data from potentially 3D arrays based on the specified vertical level index or pressure level value. If the input wind data is already 2D (i.e., has 2 or fewer dimensions), it returns the data directly without slicing. For 3D data, it checks if a level index is provided; if so, it extracts the corresponding horizontal slice at that level index. If no level index is provided but a specific pressure level value is given along with the corresponding pressure levels array, it finds the index of the nearest pressure level to the specified value and extracts that slice. If neither level selection parameter is provided, it defaults to using the top level (last index) of the 3D arrays for plotting. This method ensures that the wind data is in the correct 2D format for plotting while providing flexibility for users to specify which vertical level to visualize when working with 3D wind data, whether by index or by pressure level. 

        Parameters:
            u_data_3d (Union[np.ndarray, xr.DataArray]): U-component wind data array, which may be 2D or 3D.
            v_data_3d (Union[np.ndarray, xr.DataArray]): V-component wind data array, which may be 2D or 3D.
            level_index (Optional[int]): Integer index specifying which vertical level to extract for 3D data; if None, defaults to the top level (last index).
            level_value (Optional[float]): Specific pressure level value to extract for 3D data; if provided, it will find the nearest pressure level index based on the pressure_levels array.
            pressure_levels (Optional[np.ndarray]): Array of pressure levels corresponding to the vertical dimension of the wind data, required if level_value is provided to find the nearest pressure level index.

        Returns:
            Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]: 2D horizontal slices of the U and V component wind data for use in the overlay plotting workflow, returned as either NumPy arrays or xarray DataArrays depending on the input type.
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
    
    def compute_wind_speed_and_direction(self: "MPASWindPlotter",
                                         u_data: Union[np.ndarray, xr.DataArray],
                                         v_data: Union[np.ndarray, xr.DataArray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method computes the wind speed and meteorological direction from the U and V component wind data. Wind speed is calculated as the vector magnitude of the U and V components using the Pythagorean theorem, while wind direction is calculated in degrees using the arctan2 function to determine the angle of the wind vector relative to the north direction. The method converts mathematical angles to meteorological convention where 0° corresponds to north, 90° to east, 180° to south, and 270° to west. The resulting wind speed and direction arrays are returned for use in visualization or further analysis. This method allows users to derive meaningful meteorological information from the raw U and V component data, enabling more informative visualizations of wind patterns. 

        Parameters:
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s, which may be 2D or 3D.
            v_data (Union[np.ndarray, xr.DataArray]): V-component wind data array in m/s, which may be 2D or 3D. 

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the computed wind speed array and wind direction array in degrees, both returned as NumPy arrays for consistent processing in visualization workflows.
        """
        # Calculate wind speed as the vector magnitude of the U and V components using the Pythagorean theorem.
        wind_speed = np.sqrt(u_data**2 + v_data**2)

        # Calculate wind direction in degrees using arctan2, converting from mathematical angles to meteorological convention 
        wind_direction = np.arctan2(v_data, u_data) * 180 / np.pi

        # Convert mathematical angles to meteorological direction where 0° is north, 90° is east, etc.
        wind_direction = (270 - wind_direction) % 360
        
        # Return the computed wind speed and direction arrays 
        return wind_speed, wind_direction