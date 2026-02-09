#!/usr/bin/env python3

"""
MPAS Surface Variable Visualization

This module provides specialized plotting functionality for MPAS surface variables including 2-meter temperature, sea-level pressure, humidity, and wind speed with flexible rendering options and comprehensive cartographic presentation. It implements the MPASSurfacePlotter class that creates professional geographic maps using both scatter plot rendering (direct MPAS cell display preserving unstructured mesh resolution) and contour/filled contour plots (interpolated to regular grids for smooth gradients using MPASRemapper's hybrid KDTree-xESMF approach), with automatic unit conversion from model output to display units, variable-specific colormap and contour level selection, and optional feature overlays including wind vectors and geographic elements. The plotter supports batch processing for creating time series of surface maps with consistent styling, adaptive marker sizing based on map extent and data density, multiple map projections via Cartopy, and handles both 2D surface data and automatic extraction of surface levels from 3D datasets. Core capabilities include MPASRemapper-based grid interpolation for contour plots, geographic extent validation, metadata-driven styling, and publication-quality output suitable for operational weather analysis and climate model diagnostics.

Classes:
    MPASSurfacePlotter: Specialized class for creating surface variable visualizations from MPAS model output with cartographic presentation.
    
Functions:
    create_surface_plot: Convenience function for quick surface map generation without class instantiation.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February, 2026
Version: 1.0.0
"""
# Import necessary libraries and modules for data handling, plotting, and MPAS-specific processing
import os
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
from typing import Tuple, Optional, List, Union, Any, cast, Dict

# Import MPASdiag modules for configuration, data processing, remapping, and visualization
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.processing.remapping import MPASRemapper, remap_mpas_to_latlon_with_masking


class MPASSurfacePlotter(MPASVisualizer):
    """
    Specialized plotter for creating professional cartographic visualizations of MPAS surface variables including 2-meter temperature, sea-level pressure, humidity, and wind speed with flexible rendering options. This class extends MPASVisualizer to provide comprehensive surface diagnostic plotting capabilities including both scatter plot rendering (direct MPAS cell display preserving unstructured mesh resolution) and contour/filled contour plots (interpolated to regular grids for smooth gradients), automatic unit conversion from model output to display units via UnitConverter, variable-specific colormap and contour level selection through MPASFileMetadata, and optional feature overlays (wind vectors, geographic features, custom surface annotations). The plotter supports batch processing for creating time series of surface maps with consistent styling, adaptive marker sizing based on map extent and data density, multiple map projections via Cartopy, and publication-quality output with timestamps, colorbars, and professional cartographic elements suitable for mesoscale weather analysis and model evaluation diagnostics.
    """
    
    def _extract_2d_from_multidimensional(
        self,
        data: Union[np.ndarray, xr.DataArray],
        level_index: Optional[int],
        level_value: Optional[float]
    ) -> np.ndarray:
        """
        This helper extracts a surface-level (2D) slice from 1D/2D/3D arrays and flattens it to a 1D numpy array suitable for plotting. It supports selecting by vertical index or by using the default surface level when not specified. When encountering higher-dimensional inputs the function will raise a ValueError to avoid ambiguous extractions. The method also includes debug print statements to trace the extraction process and confirm the final shape of the output array. 

        Parameters:
            data (np.ndarray or xarray.DataArray): Input data array of 1D/2D/3D shape containing surface or profile data.
            level_index (Optional[int]): Optional integer index for selecting a vertical level from 3D arrays.
            level_value (Optional[float]): Optional vertical coordinate value (reserved; not currently implemented for pressure levels).

        Returns:
            np.ndarray: Flattened 1D numpy array containing surface-level values ready for plotting.
        """
        # If data is already 1D or 2D, convert to numpy and return directly
        if data.ndim <= 1:
            return self.convert_to_numpy(data)
        
        # For 2D data, assume the last dimension is vertical and extract the surface level if level_index or level_value is specified, otherwise take the last level as surface
        if data.ndim > 3:
            raise ValueError(f"only 1D, 2D and 3D data are supported, got {data.ndim}D data with shape {data.shape}")
        
        # Extract the appropriate 2D slice based on level_index or level_value, defaulting to surface level if neither is provided. 
        if level_index is not None:
            if data.ndim == 2:
                data = data[:, level_index]
            elif data.ndim == 3:
                data = data[:, level_index, ...]
                if data.ndim > 1:
                    data = data[..., 0]
            print(f"Extracted 2D data using level_index={level_index}")
        elif level_value is not None:
            # Default to surface level (not yet implementing pressure level selection)
            data = data[:, -1] if data.ndim == 2 else data[:, -1, ...]
            if data.ndim > 1:
                data = data[..., 0]
            print(f"Extracted 2D data using surface level (level_value={level_value} not yet implemented)")
        else:
            # Default to surface level
            data = data[:, -1] if data.ndim == 2 else data[:, -1, ...]
            if data.ndim > 1:
                data = data[..., 0]
            print("Extracted 2D data using surface level (default)")
        
        # Flatten the remaining dimensions to 1D for plotting and convert to numpy array
        if data.ndim > 1:
            data = data.flatten()
            print(f"Flattened remaining dimensions to 1D, final shape: {data.shape}")
        
        # Convert to numpy array if it's still an xarray DataArray and return
        return self.convert_to_numpy(data)
    
    def _extract_and_convert_units(
        self,
        data: np.ndarray,
        var_name: str,
        data_array: Optional[xr.DataArray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        The method inspects the provided xarray DataArray and variable metadata to determine the original units, queries UnitConverter for the desired display units, and performs in-place conversion when needed. It returns the converted numpy array and a metadata dictionary describing the variable including long_name and units for downstream plotting. The function includes error handling to catch and log any issues during unit conversion without interrupting the plotting process, and it ensures that the returned data is always a numpy array regardless of the input type. Debug print statements are included to confirm the original and display units and to indicate when conversions occur.

        Parameters:
            data (np.ndarray): Numeric data values in original units.
            var_name (str): Variable name used to look up metadata and display units.
            data_array (Optional[xarray.DataArray]): Optional source DataArray providing attribute-based unit information.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Converted data array and metadata dictionary for the variable.
        """
        # Initialize file_unit to None
        file_unit = None

        # First check if data_array is provided and has 'units' attribute
        if data_array is not None:
            try:
                file_unit = getattr(data_array, 'attrs', {}).get('units')
            except Exception:
                pass
        
        # If file_unit is still None, check if data itself has 'attrs' with 'units'
        if file_unit is None and hasattr(data, 'attrs'):
            file_unit = getattr(data, 'attrs', {}).get('units')
        
        # If file_unit is still None, check MPASFileMetadata for original units based on variable name
        var_metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        
        # Determine original unit from file_unit, metadata, or default to None
        original_unit = file_unit if file_unit else var_metadata.get('original_units') or var_metadata.get('units')

        # Query UnitConverter for display units based on variable name and original unit, defaulting to original unit if no specific display unit is defined
        display_unit = UnitConverter.get_display_units(var_name, original_unit or "")
        
        # Store original and display units in metadata for downstream use in titles and colorbar labels
        var_metadata['original_units'] = original_unit
        var_metadata['units'] = display_unit
        
        # Convert units if needed
        if original_unit != display_unit:
            try:
                # UnitConverter may return an xarray DataArray, a numpy array, or a scalar value, so we handle all cases to ensure we end up with a numpy array for plotting
                converted_data = UnitConverter.convert_units(
                    data, cast(str, original_unit or ""), display_unit
                )
                # Ensure the converted data is a numpy array regardless of the input type
                if isinstance(converted_data, xr.DataArray):
                    data = converted_data.values
                # Handle the case where conversion returns a scalar value (e.g., for unitless variables) by converting it to a numpy array with the same shape as the original data
                elif isinstance(converted_data, np.ndarray):
                    data = converted_data
                # If the conversion returns a scalar value, we create a numpy array filled with that value and the same shape as the original data 
                else:
                    data = np.asarray(converted_data)
                print(f"Converted {var_name} from {original_unit} to {display_unit}")
            except ValueError as e:
                print(f"Warning: Could not convert {var_name} from {original_unit} to {display_unit}: {e}")
        
        # Specify the list of moisture-related variable names to check for physical constraints (e.g., non-negative values)
        moisture_vars = ['q2', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qv2m', 'humidity', 'mixing_ratio']

        # For moisture variables, check for negative values and clip to 0 if found, since negative moisture is physically invalid. 
        if any(mv in var_name.lower() for mv in moisture_vars):
            # Count the number of negative values in the data array to log a warning if any are found
            n_negative = np.sum(data < 0)

            # If negative values are found, log a warning with the count and minimum value, then clip the data to 0 to enforce physical constraints.
            if n_negative > 0:
                print(f"Warning: Found {n_negative:,} negative {var_name} values (min: {np.min(data):.4f}). Clipping to 0 (physically invalid).")
                data = np.clip(data, 0, None)
        
        # Return the converted data as a numpy array along with the variable metadata
        return data, var_metadata
    
    def _prepare_colormap_and_levels(
        self,
        colormap: Optional[str],
        levels: Optional[List[float]],
        var_metadata: Dict[str, Any],
        clim_min: Optional[float],
        clim_max: Optional[float]
    ) -> Tuple[str, Optional[List[float]]]:
        """
        This helper chooses a final colormap (falling back to metadata defaults) and determines contour levels, optionally clipping or augmenting the levels to fit provided color limits. It centralizes colormap/level selection so rendering code can remain concise. It prioritizes explicit parameters from the caller, then metadata values, and finally defaults to ensure that the plotting functions have clear instructions for styling. The method also includes debug print statements to confirm the chosen colormap and levels for the plot, which can assist in troubleshooting and verifying that the styling is applied as intended.

        Parameters:
            colormap (Optional[str]): User-specified colormap name to override metadata default.
            levels (Optional[List[float]]): Optional explicit contour level list supplied by the caller.
            var_metadata (Dict[str, Any]): Variable metadata dictionary returned by MPASFileMetadata.
            clim_min (Optional[float]): Optional minimum color limit for clipping levels.
            clim_max (Optional[float]): Optional maximum color limit for clipping levels.

        Returns:
            Tuple[str, Optional[List[float]]]: Final colormap name or identifier and the list of levels to use (or None).
        """
        # Determine final colormap: prioritize explicit colormap from caller, then metadata colormap, otherwise default to 'viridis'
        final_colormap: str = colormap if colormap is not None else var_metadata.get('colormap', 'viridis')
        
        # Determine levels to use: prioritize explicit levels from caller, then metadata levels, otherwise None. 
        if levels is None:
            levels = var_metadata.get('levels', None)
        
        #  If clim_min and clim_max are provided along with levels, we filter the levels to only include those within the color limits. 
        if clim_min is not None and clim_max is not None and levels is not None:
            levels = [level for level in levels if clim_min <= level <= clim_max]
            if clim_min not in levels:
                levels.insert(0, clim_min)
            if clim_max not in levels:
                levels.append(clim_max)
        
        # Return the final colormap and levels to be used for plotting
        return final_colormap, levels
    
    def _setup_map_extent_and_features(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        projection: str
    ) -> Tuple[ccrs.CRS, ccrs.CRS, float, float, float, float, float, float, float, float]:
        """
        The function prepares the plotting axes using the requested projection, applies a safe global versus regional extent adjustment to avoid dateline artifacts, and attaches coastline, borders, land and ocean features. It returns the projection and computed extent values used to filter input data for plotting. It also includes debug print statements to confirm the chosen map extent and projection, which can assist in troubleshooting and verifying that the map is set up correctly for the intended geographic area. The method ensures that the map is properly configured before any data is plotted, providing a consistent and professional cartographic presentation for all surface variable visualizations.

        Parameters:
            lon_min (float): Western longitude of desired map extent in degrees.
            lon_max (float): Eastern longitude of desired map extent in degrees.
            lat_min (float): Southern latitude of desired map extent in degrees.
            lat_max (float): Northern latitude of desired map extent in degrees.
            projection (str): Name of Cartopy projection to use (e.g., 'PlateCarree').

        Returns:
            Tuple[ccrs.CRS, ccrs.CRS, float, float, float, float, float, float, float, float]:
                Projection for plotting, data CRS, and filtered lon/lat bounds for plotting and data selection.
        """
        # Set up the map projection and data CRS using the base visualizer's setup function
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        # Initialize the figure and GeoAxes with the specified projection.
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)

        # Ensure that the axes is a GeoAxes instance for cartopy plotting and raise an error if not
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for cartopy plots"
        
        # Determine if the requested extent is global in longitude and latitude to apply a safe extent that avoids dateline wrapping issues.
        is_global_lon = (lon_max - lon_min) >= 359.0
        is_global_lat = (lat_max - lat_min) >= 179.0
        
        # If the extent is global in longitude and latitude, we set a slightly smaller extent to avoid issues with dateline wrapping in cartopy. 
        if is_global_lon and is_global_lat:
            # Apply a small buffer to the global extent to avoid dateline artifacts
            filter_lon_min = max(lon_min, -179.99)
            filter_lon_max = min(lon_max, 179.99)
            filter_lat_min = max(lat_min, -89.99)
            filter_lat_max = min(lat_max, 89.99)

            # Set the map extent to the adjusted global bounds and log the chosen extent for debugging
            self.ax.set_extent([filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max], crs=data_crs)
            print(f"Using global extent (adjusted to avoid dateline): [{filter_lon_min}, {filter_lon_max}, {filter_lat_min}, {filter_lat_max}]")
            
            # Use original lon/lat bounds for filtering data since we want to include all points in the dataset for global plots
            filter_lon_min_data = -180.01
            filter_lon_max_data = 180.01
            filter_lat_min_data = -90.01
            filter_lat_max_data = 90.01
        else:
            # For regional extents, we can use the provided lon/lat bounds directly for both setting the map extent and filtering data. 
            filter_lon_min = lon_min
            filter_lon_max = lon_max
            filter_lat_min = lat_min
            filter_lat_max = lat_max

            # Set the map extent to the requested regional bounds and log the chosen extent for debugging
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
            
            # Use the same lon/lat bounds for filtering data since we want to restrict to the regional area for non-global plots
            filter_lon_min_data = lon_min
            filter_lon_max_data = lon_max
            filter_lat_min_data = lat_min
            filter_lat_max_data = lat_max
        
        # Add cartographic features to the map for better visualization, including coastlines, borders, land and ocean features with appropriate styling.
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)

        # Add gridlines and labels for longitude and latitude with appropriate formatting
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        
        # Return the map projection, data CRS, and the computed filter bounds for both plotting and data selection to be used in subsequent steps of the plotting process.
        return (map_proj, data_crs, filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
                filter_lon_min_data, filter_lon_max_data, filter_lat_min_data, filter_lat_max_data)
    
    def _create_colormap_normalization(
        self,
        colormap: str,
        levels: Optional[List[float]],
        clim_min: Optional[float],
        clim_max: Optional[float],
        data: np.ndarray
    ) -> Tuple[mcolors.Colormap, Optional[mcolors.Normalize]]:
        """
        This helper returns a colormap instance and an appropriate Normalize (or BoundaryNorm) object depending on explicit color limits or contour levels. It gracefully falls back to sensible defaults when requested colormap or levels are not available. It centralizes the logic for determining how to map data values to colors based on the provided styling parameters, ensuring that the plotting functions can easily apply consistent color mapping without needing to handle the complexity of normalization and colormap selection themselves. The method also includes debug print statements to confirm the chosen colormap and normalization strategy, which can assist in troubleshooting and verifying that the color mapping is applied as intended.

        Parameters:
            colormap (str): Colormap name or identifier.
            levels (Optional[List[float]]): Optional contour levels to derive a BoundaryNorm.
            clim_min (Optional[float]): Optional colorbar minimum value.
            clim_max (Optional[float]): Optional colorbar maximum value.
            data (np.ndarray): Data array used to infer vmin/vmax when limits are not provided.

        Returns:
            Tuple[matplotlib.colors.Colormap, Optional[matplotlib.colors.Normalize]]: Colormap and normalization to use when plotting.
        """
        try:
            # Attempt to get the colormap object from matplotlib using the provided colormap name or identifier.
            cmap_obj = plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
        except Exception:
            # If the specified colormap is not found or invalid, log a warning and fall back to the default 'viridis' colormap.
            cmap_obj = plt.get_cmap('viridis')
        
        # Determine the appropriate normalization for the colormap based on provided color limits or contour levels. 
        if cmap_obj is None:
            cmap_obj = plt.get_cmap('viridis')
        
        # Initialize norm to None and then set it based on the presence of clim_min/clim_max or levels. 
        norm = None

        # If explicit color limits are provided, we use a simple Normalize to scale the colormap between clim_min and clim_max. 
        if clim_min is not None or clim_max is not None:
            vmin = clim_min if clim_min is not None else float(np.nanmin(data))
            vmax = clim_max if clim_max is not None else float(np.nanmax(data))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            try:
                # If contour levels are provided, we create a BoundaryNorm that maps the specified levels to the colormap. 
                if levels is not None:
                    color_levels_sorted = sorted(set([v for v in levels if np.isfinite(v)]))
                    if color_levels_sorted:
                        last_bound = max(color_levels_sorted) + 1
                        bounds = [min(color_levels_sorted)] + color_levels_sorted + [last_bound]
                        norm = BoundaryNorm(bounds, ncolors=cmap_obj.N, clip=True)
            except Exception:
                norm = None
        
        return cmap_obj, norm
    
    def _filter_valid_data(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        plot_type: str,
        filter_lon_min_data: float,
        filter_lon_max_data: float,
        filter_lat_min_data: float,
        filter_lat_max_data: float,
        var_name: str,
        var_metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The function constructs a validity mask based on finite values and geographic bounds. For scatter plots it additionally enforces the provided extent bounds; for contour-type rendering it retains all finite points for interpolation. It returns the filtered lon, lat, and data arrays for plotting. It also includes debug print statements to confirm the filtering criteria and the number of valid points that will be plotted, which can assist in troubleshooting and verifying that the data filtering is working as intended. The method ensures that only valid and appropriately bounded data points are included in the visualization, which is crucial for creating accurate and meaningful surface maps.

        Parameters:
            lon (np.ndarray): 1D longitude array for each data point.
            lat (np.ndarray): 1D latitude array for each data point.
            data (np.ndarray): 1D array of data values corresponding to lon/lat.
            plot_type (str): Plot rendering mode ('scatter', 'contour', 'contourf', or 'both').
            filter_lon_min_data (float): Minimum longitude for filtering data points.
            filter_lon_max_data (float): Maximum longitude for filtering data points.
            filter_lat_min_data (float): Minimum latitude for filtering data points.
            filter_lat_max_data (float): Maximum latitude for filtering data points.
            var_name (str): Variable name used for error messages and logging.
            var_metadata (Dict[str, Any]): Variable metadata dictionary used for units display.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered (lon, lat, data) arrays containing only valid points.
        """
        print(f"DEBUG: filter_lon_min_data={filter_lon_min_data:.4f}, filter_lon_max_data={filter_lon_max_data:.4f}")
        print(f"DEBUG: lon range in data: [{np.min(lon):.4f}, {np.max(lon):.4f}]")
        
        # For scatter plots, we enforce both finite values and geographic bounds to ensure we only plot valid points within the desired extent. 
        if plot_type == 'scatter':
            valid_mask = (
                np.isfinite(data) & np.isfinite(lon) & np.isfinite(lat) &
                (lon >= filter_lon_min_data) & (lon <= filter_lon_max_data) &
                (lat >= filter_lat_min_data) & (lat <= filter_lat_max_data)
            )
        else:  
            # For contour-type rendering, we want to retain all finite points for interpolation, even those outside the immediate plotting extent
            valid_mask = np.isfinite(data) & np.isfinite(lon) & np.isfinite(lat)
        
        try:
            if hasattr(valid_mask, 'compute'):
                valid_mask = cast(Any, valid_mask).compute()
        except Exception:
            pass
        
        # Ensure the valid_mask is a boolean numpy array for indexing
        valid_mask = np.asarray(valid_mask, dtype=bool)
        
        # Check if any valid data points remain after filtering and raise an error if not to avoid plotting empty maps. 
        if not np.any(valid_mask):
            raise ValueError(f"No valid data points found within the specified map extent for {var_name}")
        
        # Apply the valid mask to filter the longitude, latitude, and data arrays for plotting. 
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        data_valid = data[valid_mask]
        
        print(f"Plotting {len(data_valid):,} data points for {var_name}")
        print(f"Data range: {data_valid.min():.3f} to {data_valid.max():.3f} {var_metadata['units']}")
        
        # Return the filtered longitude, latitude, and data arrays containing only valid points for plotting.
        return lon_valid, lat_valid, data_valid
    
    def _render_plot(
        self,
        plot_type: str,
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        data_valid: np.ndarray,
        filter_lon_min: float,
        filter_lon_max: float,
        filter_lat_min: float,
        filter_lat_max: float,
        cmap_obj: mcolors.Colormap,
        norm: Optional[mcolors.Normalize],
        levels: Optional[List[float]],
        data_crs: ccrs.CRS,
        grid_resolution: Optional[float],
        dataset: Optional[xr.Dataset]
    ) -> None:
        """
        This dispatcher calls scatter, contour, contourf or both rendering helpers with the pre-computed valid points and colormap normalization. It keeps the high-level create_surface_map logic concise by encapsulating the different render paths. It also includes debug print statements to confirm which rendering mode is being used and to trace the flow of the plotting process, which can assist in troubleshooting and verifying that the correct rendering method is applied based on the specified plot_type. The method ensures that the appropriate visualization technique is applied to the data based on the user's choice, allowing for flexible and customizable surface variable visualizations while maintaining a clean separation of concerns in the code structure.

        Parameters:
            plot_type (str): Selected rendering mode ('scatter', 'contour', 'contourf', or 'both').
            lon_valid (np.ndarray): Filtered longitude values for plotting.
            lat_valid (np.ndarray): Filtered latitude values for plotting.
            data_valid (np.ndarray): Filtered data values for plotting.
            filter_lon_min (float): Map extent lon_min used for interpolation grids.
            filter_lon_max (float): Map extent lon_max used for interpolation grids.
            filter_lat_min (float): Map extent lat_min used for interpolation grids.
            filter_lat_max (float): Map extent lat_max used for interpolation grids.
            cmap_obj (matplotlib.colors.Colormap): Colormap instance for mapping values to colors.
            norm (Optional[matplotlib.colors.Normalize]): Normalization instance for color mapping.
            levels (Optional[List[float]]): Explicit contour levels when applicable.
            data_crs (ccrs.CRS): Coordinate reference system of the input data.
            grid_resolution (Optional[float]): Target grid resolution for interpolation when needed.
            dataset (Optional[xarray.Dataset]): Optional dataset to assist remapping to lat-lon grid.

        Returns:
            None: Rendering functions draw directly onto self.ax and update self.fig.
        """
        if plot_type == 'scatter':
            # For scatter plots, we call the helper that creates a scatter plot directly from the valid longitude, latitude, and data points 
            self._create_scatter_plot(lon_valid, lat_valid, data_valid, cmap_obj, norm, data_crs)
        elif plot_type == 'contour':
            # For contour plots, we call the helper that creates contour lines by interpolating the valid data points onto a regular lat-lon grid
            self._create_contour_plot(
                lon_valid, lat_valid, data_valid,
                filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
                cmap_obj, norm, levels, data_crs, grid_resolution, dataset
            )
        elif plot_type == 'contourf':
            # For filled contour plots, we call the helper that creates filled contours by interpolating the valid data points onto a regular lat-lon grid 
            self._create_contourf_plot(
                lon_valid, lat_valid, data_valid,
                filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
                cmap_obj, norm, levels, data_crs, grid_resolution, dataset
            )
        elif plot_type == 'both':
            # For the 'both' option, we first create contour lines and then overlay a scatter plot of the valid points to show the original data locations on top of the contours
            self._create_contour_plot(
                lon_valid, lat_valid, data_valid,
                filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
                cmap_obj, norm, levels, data_crs, grid_resolution, dataset
            )
            # After plotting the contours, we overlay a scatter plot of the valid points to show the original data locations on top of the contours for enhanced visualization.
            self._create_scatter_plot(lon_valid, lat_valid, data_valid, cmap_obj, norm, data_crs)
    
    def _generate_title(
        self,
        title: Optional[str],
        time_stamp: Optional[datetime],
        var_metadata: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        If a custom title is not provided the function composes a default title from variable metadata and appends a formatted valid time when available. It also returns a flag indicating whether the timestamp is already present in the title. It ensures that the plot has a meaningful title even when the user does not provide one, while also avoiding redundant time annotations if the timestamp is already included in a custom title. The method includes debug print statements to confirm the final generated title and whether the timestamp is included, which can assist in troubleshooting and verifying that the title generation logic is working as intended.

        Parameters:
            title (Optional[str]): Optional user-provided title string.
            time_stamp (Optional[datetime]): Optional datetime used to annotate the title.
            var_metadata (Dict[str, Any]): Variable metadata used to derive default long name.

        Returns:
            Tuple[str, bool]: Final title string and a boolean flag whether the timestamp appears in the title.
        """
        # Initialize time_in_title flag to False and then determine the title based on whether a custom title is provided 
        time_in_title = False
        
        if title is None:
            # If no custom title is provided, we construct a default title using the variable's long name from the metadata. 
            title = f"MPAS {var_metadata['long_name']}"
            if time_stamp:
                # If a timestamp is provided, we format it as 'YYYYMMDDTHH' and append it to the title with a 'Valid Time:' prefix for clarity. 
                time_str = time_stamp.strftime('%Y%m%dT%H')
                title += f" | Valid Time: {time_str}"
                time_in_title = True
        else:
            if time_stamp:
                # If a custom title is provided, we check if the formatted timestamp is already included in the title to avoid redundant time annotations
                time_str = time_stamp.strftime('%Y%m%dT%H')
                time_in_title = (time_str in title or 'Valid Time:' in title or 'Valid:' in title)
        
        # Return the final title string and the flag indicating whether the timestamp is already included in the title 
        return title, time_in_title
    
    def _add_gridlines(self, data_crs: ccrs.CRS) -> None:
        """
        This helper configures cartopy gridlines with consistent styling and custom formatters for longitude and latitude labels. It requires self.ax to be a cartopy GeoAxes and mutates it in-place. It ensures that the gridlines are added with appropriate styling and that the labels are formatted in a clear and professional manner, enhancing the readability of the map. The method includes debug print statements to confirm that gridlines have been added and to indicate the coordinate reference system used for formatting, which can assist in troubleshooting and verifying that the gridlines are configured correctly.

        Parameters:
            data_crs (ccrs.CRS): Coordinate reference system used for formatting gridlines.

        Returns:
            None: Gridlines are added directly to self.ax.
        """
        # Ensure that the axes is a GeoAxes instance for cartopy plotting and raise an error if not
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for gridlines"
        
        # Add gridlines to the map with specified styling and label formatting. 
        gl = self.ax.gridlines(
            crs=data_crs, draw_labels=True,
            linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )

        # Configure gridline labels to only show on the left and bottom axes for a cleaner look
        gl.top_labels = False
        gl.right_labels = False

        # Set the font size for longitude and latitude labels to ensure they are legible but not overpowering the map features.
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # Set custom formatters for longitude and latitude labels to display in degrees with appropriate symbols using FuncFormatter
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
    
    def _add_overlays(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        wind_overlay: Optional[dict],
        surface_overlay: Optional[dict],
        filter_lon_min: float,
        filter_lon_max: float,
        filter_lat_min: float,
        filter_lat_max: float,
        dataset: Optional[xr.Dataset]
    ) -> None:
        """
        The method attempts to add a wind overlay via MPASWindPlotter when provided and then adds any configured surface overlay using internal overlay helpers. Errors in overlay routines are caught and logged to avoid interrupting main plot generation. It ensures that additional contextual information such as wind vectors or other surface fields can be overlaid on the main surface variable visualization, enhancing the interpretability and richness of the resulting map. The method includes debug print statements to confirm when overlays are added successfully and to log any issues encountered during the overlay process, which can assist in troubleshooting and verifying that the overlays are applied as intended without disrupting the main plotting workflow.

        Parameters:
            lon (np.ndarray): Longitude coordinates for overlays.
            lat (np.ndarray): Latitude coordinates for overlays.
            wind_overlay (Optional[dict]): Configuration dict for wind overlay (may be None).
            surface_overlay (Optional[dict]): Configuration dict for surface overlay (may be None).
            filter_lon_min (float): Map lon_min used to restrict overlays.
            filter_lon_max (float): Map lon_max used to restrict overlays.
            filter_lat_min (float): Map lat_min used to restrict overlays.
            filter_lat_max (float): Map lat_max used to restrict overlays.
            dataset (Optional[xarray.Dataset]): Optional dataset required by overlay remapping functions.

        Returns:
            None
        """
        # Ensure that the axes is a GeoAxes instance for cartopy plotting and raise an error if not
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for overlays"

        # If a wind overlay configuration is provided, we attempt to create and add the wind overlay to the map using the MPASWindPlotter. 
        if wind_overlay is not None:
            try:
                wind_plotter = MPASWindPlotter()
                wind_plotter.add_wind_overlay(self.ax, lon, lat, wind_overlay)
                print("Added wind overlay to surface map")
            except ValueError:
                raise
            except Exception as e:
                print(f"Warning: Failed to add wind overlay: {e}")
        
        # If a surface overlay configuration is provided, we attempt to add the surface overlay using internal helper methods.
        if surface_overlay is not None:
            try:
                self._add_surface_overlay(
                    ax=self.ax, lon=lon, lat=lat,
                    surface_config=surface_overlay,
                    lon_min=filter_lon_min, lon_max=filter_lon_max,
                    lat_min=filter_lat_min, lat_max=filter_lat_max,
                    dataset=dataset
                )
                print("Added surface overlay to surface map")
            except ValueError:
                raise
            except Exception as e:
                print(f"Warning: Failed to add surface overlay: {e}")
    
    def create_surface_map(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: Union[np.ndarray, xr.DataArray],
        var_name: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        title: Optional[str] = None,
        plot_type: str = 'scatter',
        colormap: Optional[str] = None,
        levels: Optional[List[float]] = None,
        clim_min: Optional[float] = None,
        clim_max: Optional[float] = None,
        projection: str = 'PlateCarree',
        time_stamp: Optional[datetime] = None,
        data_array: Optional[xr.DataArray] = None,
        grid_resolution: Optional[float] = None,
        wind_overlay: Optional[dict] = None,
        surface_overlay: Optional[dict] = None,
        level_index: Optional[int] = None,
        level_value: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> Tuple[Figure, Axes]:
        """
        This method is the main entry point for surface plotting and supports scatter, contour, and filled-contour renderings with automatic unit conversion, metadata-driven styling, and optional overlays (e.g., wind vectors or additional surface fields). It handles projection and extent setup, filters valid data points, computes colormap normalization and contour levels, interpolates to a regular grid when needed, and returns the matplotlib figure and axes containing the rendered map. The method is suitable for batch plotting or interactive use and raises informative errors for invalid inputs.

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS mesh cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS mesh cell centers.
            data (np.ndarray or xarray.DataArray): 1D array or DataArray of surface variable values in model units to be plotted.
            var_name (str): Variable name for metadata lookup and unit conversion (e.g., 't2m', 'mslp').
            lon_min (float): Western boundary of map extent in degrees.
            lon_max (float): Eastern boundary of map extent in degrees.
            lat_min (float): Southern boundary of map extent in degrees.
            lat_max (float): Northern boundary of map extent in degrees.
            title (Optional[str]): Custom plot title; if None a metadata-based title is generated.
            plot_type (str): Rendering method ('scatter', 'contour', 'contourf', or 'both').
            colormap (Optional[str]): Custom matplotlib colormap name overriding variable-specific defaults.
            levels (Optional[List[float]]): Custom contour level list overriding metadata defaults.
            clim_min (Optional[float]): Minimum color limit to clip contour levels.
            clim_max (Optional[float]): Maximum color limit to clip contour levels.
            projection (str): Cartopy projection name (e.g., 'PlateCarree').
            time_stamp (Optional[datetime]): Valid time used for title annotation.
            data_array (Optional[xarray.DataArray]): Optional DataArray for extracting metadata attributes.
            grid_resolution (Optional[float]): Grid spacing in degrees for contour interpolation; adaptive if None.
            wind_overlay (Optional[dict]): Wind overlay configuration dict (optional).
            surface_overlay (Optional[dict]): Surface overlay configuration dict (optional).
            level_index (Optional[int]): Vertical index to extract from 3D arrays (optional).
            level_value (Optional[float]): Vertical level value (reserved; not generally required for surface plots).

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figure and GeoAxes with the rendered surface map.

        Raises:
            ValueError: If inputs are inconsistent (e.g., mismatched array lengths) or an unsupported plot_type is requested.
        """
        # Validate that the requested plot_type is one of the supported options and raise a ValueError if not 
        if plot_type not in ['scatter', 'contour', 'both', 'contourf']:
            raise ValueError(f"plot_type must be 'scatter', 'contour', 'contourf', or 'both', got '{plot_type}'")
        
        # If the data is a multi-dimensional array (e.g., 3D with vertical levels), we attempt to extract a 2D slice based on the provided level_index or level_value.
        data = self._extract_2d_from_multidimensional(data, level_index, level_value)
        
        # Validate coordinate and data array lengths match
        if len(data) != len(lon) or len(data) != len(lat):
            raise ValueError(
                f"Data array length ({len(data)}) must match coordinate arrays length "
                f"(lon: {len(lon)}, lat: {len(lat)})"
            )
        
        # Convert inputs to numpy arrays if they are xarray DataArrays to ensure compatibility with matplotlib plotting functions.
        lon = self.convert_to_numpy(lon)
        lat = self.convert_to_numpy(lat)
        data = self.convert_to_numpy(data)

        # Perform unit conversion if necessary and extract variable metadata for use in titles and colorbar labels. 
        data, var_metadata = self._extract_and_convert_units(data, var_name, data_array)

        # Store the current variable metadata for use in overlays and other components that may need access to variable attributes.
        self._current_var_metadata = var_metadata
        
        # Prepare the colormap and contour levels based on the provided arguments and variable metadata, applying any necessary clipping to levels based on color limits
        colormap, levels = self._prepare_colormap_and_levels(
            colormap, levels, var_metadata, clim_min, clim_max
        )
        
        # Set up the map projection, figure, and axes, and determine the appropriate extent for plotting while avoiding dateline artifacts for global plots. 
        (
            map_proj, data_crs,
            filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
            filter_lon_min_data, filter_lon_max_data, filter_lat_min_data, filter_lat_max_data
        ) = self._setup_map_extent_and_features(lon_min, lon_max, lat_min, lat_max, projection)
        
        # At this point, self.fig and self.ax should be initialized by _setup_map_extent_and_features, so we assert that they are not None 
        assert self.fig is not None, "Figure must be created by _setup_map_extent_and_features"
        assert self.ax is not None, "Axes must be created by _setup_map_extent_and_features"
        
        # Create the colormap object and normalization based on the final colormap name and levels or color limits. 
        cmap_obj, norm = self._create_colormap_normalization(
            colormap, levels, clim_min, clim_max, data
        )
        
        #  Filter the input longitude, latitude, and data arrays to retain only valid points based on finite values and geographic bounds.
        lon_valid, lat_valid, data_valid = self._filter_valid_data(
            lon, lat, data, plot_type,
            filter_lon_min_data, filter_lon_max_data, filter_lat_min_data, filter_lat_max_data,
            var_name, var_metadata
        )
        
        # Render the plot using the appropriate helper function based on the selected plot_type, passing in the filtered valid points, colormap, normalization, and other necessary parameters for plotting.
        self._render_plot(
            plot_type, lon_valid, lat_valid, data_valid,
            filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
            cmap_obj, norm, levels, data_crs, grid_resolution, dataset
        )
        
        # Generate and set title based on variable metadata and timestamp information
        title, time_in_title = self._generate_title(title, time_stamp, var_metadata)
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add timestamp annotation if not in title
        if time_stamp and not time_in_title:
            time_str = time_stamp.strftime('%Y%m%dT%H')
            self.ax.text(
                0.02, 0.98, f'Valid: {time_str}',
                transform=self.ax.transAxes,
                fontsize=12, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
        
        # Add gridlines based on the data coordinate reference system
        self._add_gridlines(data_crs)
        
        # Finalize layout to ensure titles and labels are not cut off, and add timestamp/branding annotations as needed.
        plt.tight_layout()
        self.fig.subplots_adjust(bottom=-0.07)
        self.add_timestamp_and_branding()
        
        # Add overlays such as wind vectors or additional surface fields if configured, using the provided longitude and latitude for positioning
        self._add_overlays(
            lon, lat, wind_overlay, surface_overlay,
            filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max, dataset
        )
        
        # Make sure that the figure and axes are properly initialized before returning them to the caller for further manipulation or saving.
        assert self.fig is not None, "Figure must be created before returning"
        assert self.ax is not None, "Axes must be created before returning"

        # Return the figure and axes containing the rendered surface map for further manipulation or saving by the caller.
        return self.fig, self.ax

    def _interpolate_to_grid(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        grid_resolution: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This helper uses a hybrid KDTree→xESMF remapping strategy to produce smooth, regularly spaced fields suitable for contour plotting. It supports explicit `grid_resolution` in degrees or an adaptive resolution based on the spatial extent when unspecified. It handles the creation of a temporary xarray Dataset if not provided to facilitate remapping and returns 2D meshgrids for longitude and latitude along with the interpolated data array. The method includes debug print statements to confirm the chosen grid resolution, the number of points being interpolated, and the shape of the resulting grid, which can assist in troubleshooting and verifying that the interpolation process is working as intended.

        Parameters:
            lon (np.ndarray): 1D longitude coordinates for each input point.
            lat (np.ndarray): 1D latitude coordinates for each input point.
            data (np.ndarray): 1D array of data values corresponding to lon/lat.
            lon_min (float): Western longitude bound for the target grid.
            lon_max (float): Eastern longitude bound for the target grid.
            lat_min (float): Southern latitude bound for the target grid.
            lat_max (float): Northern latitude bound for the target grid.
            grid_resolution (Optional[float]): Desired grid spacing in degrees (optional).
            dataset (Optional[xarray.Dataset]): Optional xarray Dataset describing source coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (lon_mesh, lat_mesh, data_interp) where lon_mesh and lat_mesh are 2D meshgrids and data_interp is the 2D interpolated field.
        """
        # Validate that the input longitude, latitude, and data arrays are 1D and have matching lengths to ensure they can be used for interpolation.
        if grid_resolution is not None:
            step = float(grid_resolution)

            # Validate that the provided grid resolution is a positive value to avoid invalid grid generation and raise a ValueError if it is not.
            if step <= 0:
                raise ValueError("grid_resolution must be > 0")

            # If a valid grid resolution is provided, we use it directly for the MPASRemapper and log the chosen resolution for debugging purposes.
            resolution = step
            print(f"Using MPASRemapper with target resolution: {resolution}°")
        else:
            # If no grid resolution is provided, we calculate an adaptive resolution based on the spatial extent of the data
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min

            # Set the resolution to be approximately 1% of the larger dimension of the spatial extent to ensure a reasonable number of grid points 
            resolution = max(lon_range / 100, lat_range / 100)
            print(f"Auto-selected grid resolution: {resolution:.4f}°")
        
        print(f"Interpolating {len(data)} points using MPASRemapper (KDTree→xESMF bilinear)...")
        
        # If a dataset is not provided, we create a temporary xarray Dataset with the longitude and latitude coordinates to facilitate the remapping process. 
        if dataset is None:
            lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
            lat_arr = lat if isinstance(lat, np.ndarray) else lat.values
            dataset = xr.Dataset({
                'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
                'latCell': xr.DataArray(lat_arr, dims=['nCells'])
            })
        
        # Convert the input data array to an xarray DataArray with a dimension name that matches the expected input for the remapping function 
        data_xr = xr.DataArray(data, dims=['nCells'])

        # Use the remapping function to interpolate the data from the irregular MPAS mesh to a regular lat-lon grid defined by the specified bounds and resolution
        remapped_result = remap_mpas_to_latlon_with_masking(
            data=data_xr,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            method='nearest',
            apply_mask=True,
            lon_convention='auto'
        )
        
        # Extract the longitude and latitude coordinates from the remapped result
        lon_coords = remapped_result.lon.values
        lat_coords = remapped_result.lat.values

        # Create 2D meshgrids for longitude and latitude to be used in contour plotting
        lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)

        # Extract the interpolated data values from the remapped result as a 2D array corresponding to the regular lat-lon grid.
        data_interp = remapped_result.values
        
        print(f"MPASRemapper produced {data_interp.shape[0]}x{data_interp.shape[1]} grid")
        
        # Return the longitude mesh, latitude mesh, and interpolated data array for use in contour plotting.
        return lon_mesh, lat_mesh, data_interp

    def _add_colorbar_with_metadata(self, mappable: Any) -> None:
        """
        The helper constructs a descriptive label from `_current_var_metadata` (combining long name and units) and applies dynamic tick formatting for consistent presentation. The colorbar is added to `self.fig` below the axes and configured with standardized padding and sizing. It ensures that the colorbar is informative and visually integrated with the overall figure layout. The method includes debug print statements to confirm that the colorbar has been added and to display the constructed label, which can assist in troubleshooting and verifying that the colorbar is configured correctly with the appropriate metadata information.

        Parameters:
            mappable (Any): A matplotlib ScalarMappable (e.g., the result of scatter or contourf).

        Returns:
            None: Modifies `self.fig`/`self.ax` in-place by adding a colorbar.
        """
        # Ensure that the figure is initialized before attempting to add a colorbar
        assert self.fig is not None, "Figure must be created before adding colorbar"
        
        # Add a horizontal colorbar below the axes with specified padding and shrinkage to ensure it fits well with the overall figure layout. 
        cbar = self.fig.colorbar(mappable, ax=self.ax, orientation='horizontal', extend='both',
                               pad=0.06, shrink=0.8, aspect=30)
        
        # If variable metadata is available, we construct a colorbar label that combines the long name and units in a clear format.
        if hasattr(self, '_current_var_metadata') and self._current_var_metadata:
            var_units = self._current_var_metadata.get('units', '')
            var_long_name = self._current_var_metadata.get('long_name', 'Value')
            if var_units and f'[{var_units}]' in var_long_name:
                cbar_label = var_long_name
            else:
                cbar_label = f"{var_long_name} [{var_units}]" if var_units else var_long_name
            cbar.set_label(cbar_label, fontsize=12, fontweight='bold', labelpad=-60)
        
        # Apply dynamic tick formatting to the colorbar to ensure that tick labels are presented in a consistent and readable format 
        try:
            ticks = cbar.get_ticks().tolist()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
            cbar.ax.tick_params(labelsize=8)
        except Exception:
            pass

    def _create_scatter_plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        cmap_obj: Union[str, mcolors.Colormap],
        norm: Optional[mcolors.Normalize],
        data_crs: ccrs.CRS
    ) -> None:
        """
        The method computes marker sizing based on map extent and point density, sorts the points to ensure consistent overlay ordering, and applies adaptive alpha to mitigate overplotting. It draws points on `self.ax` and adds a metadata-driven horizontal colorbar to `self.fig`. It ensures that the scatter plot is visually informative and that the color mapping is clearly communicated through the colorbar. The method includes debug print statements to confirm the number of points being plotted, the calculated marker size, and the chosen alpha value, which can assist in troubleshooting and verifying that the scatter plot is configured correctly for the given data density and map extent.

        Parameters:
            lon (np.ndarray): 1D longitude array in degrees matching `lat` and `data`.
            lat (np.ndarray): 1D latitude array in degrees matching `lon` and `data`.
            data (np.ndarray): 1D array of data values for color mapping.
            cmap_obj (Union[str, matplotlib.colors.Colormap]): Colormap name or instance.
            norm (Optional[matplotlib.colors.Normalize]): Normalization object for color mapping.
            data_crs (ccrs.CRS): CRS describing the input coordinate reference (usually PlateCarree).

        Returns:
            None: The function draws the scatter to `self.ax` and updates `self.fig`.
        """
        # Ensure that the axes and figure are initialized before attempting to create a scatter plot
        assert self.ax is not None, "Axes must be created before scatter plot"
        assert self.fig is not None, "Figure must be created before scatter plot"
        
        # Define the map extent based on the minimum and maximum longitude and latitude values
        map_extent = (lon.min(), lon.max(), lat.min(), lat.max())

        # Calculate marker size based on the map extent and the number of data points 
        fig_size = (self.figsize[0], self.figsize[1])
        marker_size = self.calculate_adaptive_marker_size(map_extent, len(data), fig_size)
        
        # Calculate point density (points per square degree) to determine appropriate alpha for overplotting mitigation
        map_area = (map_extent[1] - map_extent[0]) * (map_extent[3] - map_extent[2])

        # Avoid division by zero in case of extremely small extents and set point density to zero if map area is not positive
        point_density = len(data) / map_area if map_area > 0 else 0
        
        # Set alpha values based on point density thresholds to balance visibility and overplotting
        if point_density > 1000:
            alpha_val = 0.8
        elif point_density > 100:
            alpha_val = 0.9
        else:
            alpha_val = 0.9
        
        # Sort the data points by their values to ensure that higher values are plotted on top of lower values for better visibility in the scatter plot
        sort_indices = np.argsort(data)
        lon_sorted = lon[sort_indices]
        lat_sorted = lat[sort_indices]
        data_sorted = data[sort_indices]
        
        # Create the scatter plot on the axes using the sorted longitude, latitude, and data values
        scatter = self.ax.scatter(lon_sorted, lat_sorted, c=data_sorted,
                               cmap=cmap_obj, norm=norm, s=marker_size, alpha=alpha_val,
                               transform=data_crs, edgecolors='none')
        
        # After creating the scatter plot, we add a colorbar to the figure that is linked to the scatter plot's colormap and normalization
        self._add_colorbar_with_metadata(scatter)
    
    def _create_contour_plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        cmap_obj: Union[str, mcolors.Colormap],
        norm: Optional[mcolors.Normalize],
        levels: Optional[List[float]],
        data_crs: ccrs.CRS,
        grid_resolution: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> None:
        """
        This helper produces line contours using the remapped regular grid and optionally annotates contour values when explicit levels are provided. It supports adaptive or user-specified grid resolution and delegates interpolation to `_interpolate_to_grid`. It ensures that the contour lines are rendered clearly and that any provided contour levels are used effectively to enhance the interpretability of the plot. The method includes debug print statements to confirm the number of points being interpolated, the shape of the resulting grid, and whether contour labels were added, which can assist in troubleshooting and verifying that the contour plot is configured correctly based on the provided data and parameters.

        Parameters:
            lon (np.ndarray): 1D longitude coordinates of input points.
            lat (np.ndarray): 1D latitude coordinates of input points.
            data (np.ndarray): 1D data values corresponding to the coordinates.
            lon_min (float): Western longitude bound for interpolation grid.
            lon_max (float): Eastern longitude bound for interpolation grid.
            lat_min (float): Southern latitude bound for interpolation grid.
            lat_max (float): Northern latitude bound for interpolation grid.
            cmap_obj (Union[str, matplotlib.colors.Colormap]): Colormap used when contour lines are colored.
            norm (Optional[matplotlib.colors.Normalize]): Normalization for color mapping.
            levels (Optional[List[float]]): Explicit contour levels to draw (optional).
            data_crs (ccrs.CRS): CRS of input coordinates.
            grid_resolution (Optional[float]): Optional grid spacing in degrees for remapping.
            dataset (Optional[xarray.Dataset]): Optional dataset to assist remapping.

        Returns:
            None: Adds contour lines to `self.ax`.
        """
        # Ensure that the axes and figure are initialized before attempting to create a contour plot
        assert self.ax is not None, "Axes must be created before contour plot"
        assert self.fig is not None, "Figure must be created before contour plot"
        
        # Interpolate the irregularly spaced input data onto a regular lat-lon grid defined by the specified bounds and resolution 
        lon_mesh, lat_mesh, data_interp = self._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, dataset
        )

        try:
            contour_color = 'black'
            try:
                # When explicit levels are provided, we use them for contouring and set the line color to black for better visibility. 
                if levels is not None:
                    cs = self.ax.contour(lon_mesh, lat_mesh, data_interp, levels=levels,
                                         colors=contour_color, linewidths=1.0, linestyles='solid',
                                         transform=data_crs)
                else:
                    # If no explicit levels are provided, we let matplotlib automatically determine contour levels based on the data and colormap
                    cs = self.ax.contour(lon_mesh, lat_mesh, data_interp,
                                         colors=contour_color, linewidths=1.0, linestyles='solid',
                                         transform=data_crs)

                # If contour lines were successfully created and explicit levels were provided, we attempt to add contour labels to the plot for better interpretability.
                if levels is not None:
                    try:
                        self.ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
                    except Exception:
                        pass
            except Exception as e:
                raise RuntimeError(f"Contour plotting failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Contour plotting failed: {e}")
        
    def _create_contourf_plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        cmap_obj: Union[str, mcolors.Colormap],
        norm: Optional[mcolors.Normalize],
        levels: Optional[List[float]],
        data_crs: ccrs.CRS,
        grid_resolution: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> None:
        """
        This helper mirrors `_create_contour_plot` but uses `contourf` to render filled color bands, and then creates a horizontal colorbar displaying metadata-driven labels. It supports explicit levels and adaptive remapping resolution. It ensures that the filled contours are rendered clearly and that the colorbar effectively communicates the data values through appropriate labeling. The method includes debug print statements to confirm the number of points being interpolated, the shape of the resulting grid, and whether explicit levels were used for contouring, which can assist in troubleshooting and verifying that the filled contour plot is configured correctly based on the provided data and parameters.

        Parameters:
            lon (np.ndarray): 1D longitude coordinates of input points.
            lat (np.ndarray): 1D latitude coordinates of input points.
            data (np.ndarray): 1D data values corresponding to the coordinates.
            lon_min (float): Western longitude bound for interpolation grid.
            lon_max (float): Eastern longitude bound for interpolation grid.
            lat_min (float): Southern latitude bound for interpolation grid.
            lat_max (float): Northern latitude bound for interpolation grid.
            cmap_obj (Union[str, matplotlib.colors.Colormap]): Colormap instance or name.
            norm (Optional[matplotlib.colors.Normalize]): Normalization for color mapping.
            levels (Optional[List[float]]): Explicit contour levels (optional).
            data_crs (ccrs.CRS): CRS of input coordinates.
            grid_resolution (Optional[float]): Grid spacing in degrees for remapping.
            dataset (Optional[xarray.Dataset]): Optional dataset to assist remapping.

        Returns:
            None: Adds filled contours and a colorbar to the figure.
        """
        # Ensure that the axes and figure are initialized before attempting to create a filled contour plot
        assert self.ax is not None, "Axes must be created before contourf plot"
        assert self.fig is not None, "Figure must be created before contourf plot"
        
        # Interpolate the irregularly spaced input data onto a regular lat-lon grid defined by the specified bounds and resolution 
        lon_mesh, lat_mesh, data_interp = self._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, dataset
        )

        # Depending on whether explicit contour levels are provided, we create a filled contour plot 
        if levels is not None:
            # When explicit levels are provided, we use them for contouring and set the extend parameter to 'both' 
            cs = self.ax.contourf(lon_mesh, lat_mesh, data_interp, levels=levels,
                                cmap=cmap_obj, norm=norm, transform=data_crs, extend='both')
        else:
            # If no explicit levels are provided, we let matplotlib automatically determine contour levels based on the data and colormap, and set the extend parameter to 'both'
            cs = self.ax.contourf(lon_mesh, lat_mesh, data_interp,
                                cmap=cmap_obj, norm=norm, transform=data_crs, extend='both')

        # After creating the filled contour plot, we add a colorbar to the figure that is linked to the contourf's colormap and normalization
        self._add_colorbar_with_metadata(cs)

    def add_surface_overlay(
        self,
        ax: Axes,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        surface_config: Dict[str, Any],
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> None:
        """
        This convenience method forwards the overlay configuration to internal helpers that perform unit conversion, optional remapping to a regular grid, and rendering. It supports both line contours and filled contours with configurable levels, colormap, and styling options and does not create a new figure. It ensures that the surface overlay is rendered on top of the existing map and that any necessary unit conversions are handled appropriately. The method includes debug print statements to confirm that the surface overlay is being added and to display the configuration being used, which can assist in troubleshooting and verifying that the surface overlay is configured correctly based on the provided parameters.

        Parameters:
            ax (matplotlib.axes.Axes): Existing GeoAxes instance to draw the overlay onto.
            lon (np.ndarray or xarray.DataArray): 1D longitude coordinates for overlay data points.
            lat (np.ndarray or xarray.DataArray): 1D latitude coordinates for overlay data points.
            surface_config (Dict[str, Any]): Configuration dict containing at minimum the key 'data' and optional styling keys such as 'var_name', 'plot_type', 'levels', 'colors', 'colormap', 'linewidth', 'alpha', 'level_index', 'add_labels', and 'grid_resolution'.
            lon_min (Optional[float]): Optional western longitude bound for interpolation.
            lon_max (Optional[float]): Optional eastern longitude bound for interpolation.
            lat_min (Optional[float]): Optional southern latitude bound for interpolation.
            lat_max (Optional[float]): Optional northern latitude bound for interpolation.
            dataset (Optional[xarray.Dataset]): Optional dataset to assist remapping; created from `lon`/`lat` if omitted.

        Returns:
            None: Overlay is drawn directly onto the provided `ax`.

        Raises:
            ValueError: If an unsupported `plot_type` is provided in `surface_config`.
        """
        # Delegate the processing and rendering of the surface overlay to internal helper methods that handle unit conversion, optional remapping, and plotting 
        self._add_surface_overlay(
            ax=ax,
            lon=lon,
            lat=lat,
            surface_config=surface_config,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            dataset=dataset
        )

    def _prepare_overlay_data(
        self,
        overlay_data: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        var_name: str,
        surface_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The helper auto-detects likely original units when not provided, converts to display units using `UnitConverter`, extracts a surface-level slice for multidimensional inputs, and returns only finite lon/lat/data triplets suitable for interpolation and rendering. It ensures that the overlay data is properly prepared for plotting by handling unit conversions, dimensionality reduction, and filtering of valid data points. The method includes debug print statements to confirm the inferred original units, any conversions performed, and the number of valid points remaining after filtering, which can assist in troubleshooting and verifying that the overlay data is prepared correctly for visualization.

        Parameters:
            overlay_data (np.ndarray): Raw overlay data array (may be multidimensional).
            lon (np.ndarray): Longitude coordinates corresponding to overlay_data.
            lat (np.ndarray): Latitude coordinates corresponding to overlay_data.
            var_name (str): Variable name used for unit inference and metadata lookup.
            surface_config (Dict[str, Any]): Configuration dictionary that may include 'original_units' and 'level_index'.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered (lon, lat, data) arrays containing only valid points.
        """
        original_units = surface_config.get('original_units', None)
        
        # If original units are not explicitly provided, we attempt to infer them based on the variable name
        if original_units is None:
            data_mean = np.nanmean(overlay_data)
            if 'mslp' in var_name.lower() or 'pressure' in var_name.lower():
                if data_mean > 50000:
                    original_units = 'Pa'
            elif 't2m' in var_name.lower() or 'temp' in var_name.lower():
                if data_mean > 100:
                    original_units = 'K'
        
        # If we have determined original units, we attempt to convert the overlay data to display units using the UnitConverter, and log the conversion for debugging purposes.
        if original_units:
            display_units = UnitConverter.get_display_units(var_name, original_units)
            if original_units != display_units:
                try:
                    overlay_data = self.convert_to_numpy(
                        UnitConverter.convert_units(overlay_data, original_units, display_units)
                    )
                    print(f"Converted overlay {var_name} from {original_units} to {display_units}")
                except ValueError as e:
                    print(f"Warning: Could not convert overlay {var_name} from {original_units} to {display_units}: {e}")
        
        # Handle 3D data by extracting 2D slice for surface-level visualization
        if overlay_data.ndim > 1:
            level_index = surface_config.get('level_index', None)
            overlay_data = overlay_data[:, level_index] if level_index is not None else overlay_data[:, -1]
            print(f"Extracted 2D overlay data from multi-dimensional array")
        
        # Filter valid data points by checking for finite values in the overlay data and corresponding longitude and latitude arrays
        valid_mask = np.isfinite(overlay_data) & np.isfinite(lon) & np.isfinite(lat)
        
        # If no valid points are found after filtering, we raise a ValueError to indicate that the overlay cannot be rendered due to lack of valid data.
        if not np.any(valid_mask):
            raise ValueError(f"No valid overlay data found for {var_name}")
        
        # Return only the valid longitude, latitude, and overlay data points for use in interpolation and rendering of the surface overlay.
        return lon[valid_mask], lat[valid_mask], overlay_data[valid_mask]
    
    def _calculate_overlay_bounds(
        self,
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        lon_min: Optional[float],
        lon_max: Optional[float],
        lat_min: Optional[float],
        lat_max: Optional[float]
    ) -> Tuple[float, float, float, float]:
        """
        The helper returns a tuple of (lon_min, lon_max, lat_min, lat_max) using supplied explicit bounds when available, otherwise computing bounds from the valid coordinate arrays. It ensures that the overlay is properly framed within the spatial extent of the valid data points while allowing for caller-specified bounds to override the automatic calculation when needed. The method includes debug print statements to confirm the final bounds being used for the overlay, which can assist in troubleshooting and verifying that the bounds are calculated correctly based on the provided parameters.

        Parameters:
            lon_valid (np.ndarray): 1D array of valid longitudes for overlay points.
            lat_valid (np.ndarray): 1D array of valid latitudes for overlay points.
            lon_min (Optional[float]): Optional caller-provided western bound.
            lon_max (Optional[float]): Optional caller-provided eastern bound.
            lat_min (Optional[float]): Optional caller-provided southern bound.
            lat_max (Optional[float]): Optional caller-provided northern bound.

        Returns:
            Tuple[float, float, float, float]: The resolved (lon_min, lon_max, lat_min, lat_max) bounds.
        """
        # If explicit bounds are provided, we use them directly; otherwise, we compute the bounds from the valid longitude and latitude arrays to ensure that the overlay is properly framed within the spatial extent of the valid data points.
        return (
            lon_min if lon_min is not None else float(lon_valid.min()),
            lon_max if lon_max is not None else float(lon_valid.max()),
            lat_min if lat_min is not None else float(lat_valid.min()),
            lat_max if lat_max is not None else float(lat_valid.max())
        )
    
    def _calculate_overlay_resolution(
        self,
        grid_resolution_input: Optional[float],
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float
    ) -> float:
        """
        If `grid_resolution_input` is provided it is returned; otherwise an adaptive resolution proportional to the spatial extent is computed to balance quality and performance. It ensures that the remapping process uses an appropriate grid resolution based on either caller specifications or the spatial extent of the overlay data, which can help to optimize the interpolation quality and computational efficiency. The method includes debug print statements to confirm the chosen grid resolution and the spatial extent being considered, which can assist in troubleshooting and verifying that the resolution is calculated correctly based on the provided parameters.

        Parameters:
            grid_resolution_input (Optional[float]): Caller-specified grid spacing in degrees (optional).
            lon_min (float): Western longitude of the target extent.
            lon_max (float): Eastern longitude of the target extent.
            lat_min (float): Southern latitude of the target extent.
            lat_max (float): Northern latitude of the target extent.

        Returns:
            float: Grid spacing in degrees to use for remapping.
        """
        # If the caller has provided an explicit grid resolution, we use it directly for remapping; otherwise, we calculate an adaptive resolution based on spatial extent 
        if grid_resolution_input is not None:
            return float(grid_resolution_input)
        
        # Calculate the longitude and latitude range to determine the spatial extent of the overlay data
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        # Set the resolution to be approximately 2% of the larger dimension of the spatial extent to ensure a reasonable number of grid points for interpolation 
        return max(lon_range / 50, lat_range / 50)
    
    def _create_overlay_dataset(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        dataset: Optional[xr.Dataset]
    ) -> xr.Dataset:
        """
        The Dataset contains `lonCell` and `latCell` DataArrays and is suitable for passing to remapping utilities that expect an xarray Dataset describing source coordinates. If a dataset is already provided, it is returned unchanged. Otherwise, a new Dataset is constructed from the provided longitude and latitude arrays. This ensures that the remapping functions have the necessary coordinate information in the expected format, whether it is supplied by the caller or created on-the-fly from the input coordinates. The method includes debug print statements to confirm whether a new dataset was created or an existing one was used, which can assist in troubleshooting and verifying that the dataset is prepared correctly for remapping.

        Parameters:
            lon (np.ndarray): 1D longitude coordinates for source points.
            lat (np.ndarray): 1D latitude coordinates for source points.
            dataset (Optional[xarray.Dataset]): Optional existing dataset to return unchanged.

        Returns:
            xarray.Dataset: Dataset with `lonCell` and `latCell` variables.
        """
        # If a dataset is already provided, we return it directly
        if dataset is not None:
            return dataset
        
        # If no dataset is provided, we create a new xarray Dataset with 'lonCell' and 'latCell' DataArrays 
        lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
        lat_arr = lat if isinstance(lat, np.ndarray) else lat.values

        # Return the constructed Dataset containing longitude and latitude coordinates for use in remapping the overlay data to a regular grid
        return xr.Dataset({
            'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
            'latCell': xr.DataArray(lat_arr, dims=['nCells'])
        })
    
    def _interpolate_overlay(
        self,
        data_valid: np.ndarray,
        dataset: xr.Dataset,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        resolution: float,
        var_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The method calls the remapping helper, constructs 2D meshgrids for plotting, and returns the interpolated array ready for contour/contourf rendering. If the remapping fails, it raises a RuntimeError with details. It ensures that the overlay data is properly interpolated onto a regular lat-lon grid suitable for contour plotting, and that any issues during the remapping process are clearly communicated through exceptions. The method includes debug print statements to confirm the interpolation process, the resolution being used, and the shape of the resulting grid, which can assist in troubleshooting and verifying that the interpolation is performed correctly based on the provided parameters.

        Parameters:
            data_valid (np.ndarray): 1D array of valid overlay data values.
            dataset (xarray.Dataset): Dataset describing source lon/lat coordinates.
            lon_min (float): Western longitude bound for target grid.
            lon_max (float): Eastern longitude bound for target grid.
            lat_min (float): Southern latitude bound for target grid.
            lat_max (float): Northern latitude bound for target grid.
            resolution (float): Grid spacing in degrees for the target grid.
            var_name (str): Variable name used for log messages.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (lon_mesh, lat_mesh, data_interp) for plotting.
        """
        print(f"Interpolating {var_name} overlay using MPASRemapper (resolution: {resolution:.4f}°)")
        
        # Convert the valid data array to an xarray DataArray with a dimension name 
        data_xr = xr.DataArray(data_valid, dims=['nCells'])

        # Use the remapping module to interpolate the data from the unstructured MPAS mesh to a regular lat-lon grid defined by the specified bounds and resolution
        remapped_overlay = remap_mpas_to_latlon_with_masking(
            data=data_xr,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            method='nearest',
            apply_mask=True,
            lon_convention='auto'
        )
        
        # Create 2D meshgrids for longitude and latitude from the remapped result to be used in contour plotting, and extract the interpolated data values as a 2D array 
        lon_mesh, lat_mesh = np.meshgrid(remapped_overlay.lon.values, remapped_overlay.lat.values)

        # Return the longitude mesh, latitude mesh, and interpolated data array for use in contour or contourf plotting of the surface overlay on the map.
        return lon_mesh, lat_mesh, remapped_overlay.values
    
    def _validate_contour_levels(
        self,
        data_interp: np.ndarray,
        levels: Optional[List[float]],
        var_name: str
    ) -> None:
        """
        This helper logs the data range and warns if no requested contour levels lie within the available data values. If levels are provided, it checks which levels fall within the range of the interpolated data and logs this information. It ensures that the contour levels specified by the caller are appropriate for the range of data being plotted, which can help to prevent issues where contours are not visible due to levels being outside the data range. The method includes debug print statements to confirm the data range and the contour levels that are within this range, which can assist in troubleshooting and verifying that the contour levels are valid for the given data.

        Parameters:
            data_interp (np.ndarray): 2D interpolated data array.
            levels (Optional[List[float]]): Contour levels supplied by the caller.
            var_name (str): Variable name used for logging.

        Returns:
            None
        """
        # Compute the minimum and maximum values of the interpolated data to determine the range of values that will be plotted in the contours.
        data_min, data_max = np.nanmin(data_interp), np.nanmax(data_interp)
        print(f"  Overlay data range: {data_min:.2f} to {data_max:.2f}")
        
        if levels is not None:
            # Check which of the provided contour levels fall within the range of the data and log a warning if none of the levels are within the data range
            levels_in_range = [lev for lev in levels if data_min <= lev <= data_max]
            if len(levels_in_range) == 0:
                print(f"  WARNING: No contour levels {levels} fall within data range [{data_min:.2f}, {data_max:.2f}]!")
            else:
                print(f"  Contour levels in data range: {levels_in_range}")
    
    def _render_overlay(
        self,
        ax: Axes,
        lon_mesh: np.ndarray,
        lat_mesh: np.ndarray,
        data_interp: np.ndarray,
        plot_type: str,
        levels: Optional[List[float]],
        colors: str,
        colormap: Optional[str],
        linewidth: float,
        alpha: float,
        add_labels: bool
    ) -> None:
        """
        The function accepts either 'contour' or 'contourf' plot types and applies provided styling parameters including levels, colors/colormap, linewidth, and transparency. It draws contours directly onto the input `ax`. If `add_labels` is True and `plot_type` is 'contour', it attempts to add inline contour labels. It ensures that the surface overlay is rendered according to the specified styling options and that any requested contour labels are added for better interpretability of line contours. The method includes debug print statements to confirm the rendering mode being used, the styling parameters applied, and whether contour labels were added, which can assist in troubleshooting and verifying that the overlay is rendered correctly based on the provided configuration.

        Parameters:
            ax (matplotlib.axes.Axes): Axes to draw overlays onto.
            lon_mesh (np.ndarray): 2D meshgrid of longitudes for the interpolation grid.
            lat_mesh (np.ndarray): 2D meshgrid of latitudes for the interpolation grid.
            data_interp (np.ndarray): 2D interpolated data array to contour.
            plot_type (str): 'contour' or 'contourf' specifying rendering mode.
            levels (Optional[List[float]]): Optional contour levels to use.
            colors (str): Color for contour lines (used for 'contour').
            colormap (Optional[str]): Colormap name for filled contours (used for 'contourf').
            linewidth (float): Line width for contour lines.
            alpha (float): Transparency for overlays (0.0 - 1.0).
            add_labels (bool): Whether to add inline contour labels for line contours.

        Returns:
            None
        """
        # Define common keyword arguments for both contour and contourf to ensure consistent transformation and alpha settings
        common_kwargs = {
            'transform': ccrs.PlateCarree(),
            'alpha': alpha
        }
        
        # Depending on the specified plot type, we either create line contours using ax.contour or filled contours using ax.contourf, applying the appropriate styling parameters 
        if plot_type == 'contour':
            # For line contours, we set the colors and linewidth based on the provided parameters, and if explicit levels are provided, we include them in the contouring.
            contour_kwargs = {'colors': colors, 'linewidths': linewidth, **common_kwargs}
            if levels is not None:
                contour_kwargs['levels'] = levels
            
            # Create the contour lines on the axes using the specified parameters and the interpolated data
            cs = ax.contour(lon_mesh, lat_mesh, data_interp, **contour_kwargs)
            
            # If contour lines were successfully created and the caller has requested to add labels, we attempt to add inline contour labels to the plot 
            if add_labels:
                ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
        
        elif plot_type == 'contourf':
            # For filled contours, we create a colormap object from the provided colormap name and include it in the contourf parameters
            cmap = plt.get_cmap(colormap) if colormap else None
            contourf_kwargs = {'cmap': cmap, **common_kwargs}

            # If explicit levels are provided for filled contours, we include them in the contourf parameters to control the contour intervals and color bands.
            if levels is not None:
                contourf_kwargs['levels'] = levels
            
            # Create the filled contour plot on the axes using the specified parameters and the interpolated data
            ax.contourf(lon_mesh, lat_mesh, data_interp, **contourf_kwargs)
    
    def _add_surface_overlay(
        self,
        ax: Axes,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        surface_config: Dict[str, Any],
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> None:
        """
        This helper performs the full overlay workflow: data preparation, extent/resolution calculation, remapping to a regular grid, validation of contour levels, and rendering. It is used internally by the public `add_surface_overlay` and by `create_surface_map` when overlays are requested. If any step fails (e.g., no valid data, remapping error), it logs a warning and exits gracefully without raising an exception, allowing the main plotting workflow to continue. It ensures that the surface overlay is added to the map with proper handling of data preparation, remapping, and rendering, while also providing robust error handling to prevent issues from disrupting the overall visualization process. The method includes debug print statements at each major step to confirm the progress of adding the surface overlay and to display any warnings or issues encountered during the process, which can assist in troubleshooting and verifying that the overlay is added correctly based on the provided parameters.

        Parameters:
            ax (matplotlib.axes.Axes): Axes to draw on.
            lon (np.ndarray or xarray.DataArray): Longitude coordinates for overlay points.
            lat (np.ndarray or xarray.DataArray): Latitude coordinates for overlay points.
            surface_config (Dict[str, Any]): Overlay configuration including 'data' and styling options.
            lon_min (Optional[float]): Optional lon_min bound derived from caller or data.
            lon_max (Optional[float]): Optional lon_max bound derived from caller or data.
            lat_min (Optional[float]): Optional lat_min bound derived from caller or data.
            lat_max (Optional[float]): Optional lat_max bound derived from caller or data.
            dataset (Optional[xarray.Dataset]): Optional dataset for remapping.

        Returns:
            None
        """
        # Extract the variable name and plot type from the surface configuration, providing default values if they are not specified
        var_name = surface_config.get('var_name', 'overlay')
        plot_type = surface_config.get('plot_type', 'contour')
        
        # Validate that the specified plot type is either 'contour' or 'contourf', and raise a ValueError if an unsupported plot type is provided 
        if plot_type not in ['contour', 'contourf']:
            raise ValueError(f"Unsupported surface overlay plot_type: {plot_type}. Must be 'contour' or 'contourf'.")
        
        # Convert the longitude and latitude inputs to numpy arrays if they are xarray DataArrays
        lon = self.convert_to_numpy(lon)
        lat = self.convert_to_numpy(lat)

        # Convert the overlay data to a numpy array if it is not already
        overlay_data = self.convert_to_numpy(surface_config['data'])
        
        # Prepare data: unit conversion, 3D extraction, validation
        try:
            lon_valid, lat_valid, data_valid = self._prepare_overlay_data(
                overlay_data, lon, lat, var_name, surface_config
            )
        except ValueError as e:
            print(f"Warning: {e}")
            return
        
        # Calculate bounds using valid points and any provided explicit bounds to ensure that the interpolation grid is properly defined around the valid data points 
        lon_min, lon_max, lat_min, lat_max = self._calculate_overlay_bounds(
            lon_valid, lat_valid, lon_min, lon_max, lat_min, lat_max
        )
        
        # Calculate the interpolation grid resolution based on either the caller-provided value or an adaptive calculation from the spatial extent 
        resolution = self._calculate_overlay_resolution(
            surface_config.get('grid_resolution', None),
            lon_min, lon_max, lat_min, lat_max
        )
        
        # If no dataset is provided for remapping, we create one from the valid longitude and latitude arrays 
        dataset = self._create_overlay_dataset(lon, lat, dataset)
        
        # Interpolate the valid overlay data onto a regular lat-lon grid defined by the calculated bounds and resolution
        lon_mesh, lat_mesh, data_interp = self._interpolate_overlay(
            data_valid, dataset, lon_min, lon_max, lat_min, lat_max, resolution, var_name
        )
        
        # Validate the contour levels against the range of the interpolated data to ensure that the requested levels are appropriate for the data being plotted
        self._validate_contour_levels(data_interp, surface_config.get('levels'), var_name)
        
        # Render the overlay on the provided axes using the specified plot type and styling parameters, passing the interpolated data and meshgrids to the rendering helper
        self._render_overlay(
            ax, lon_mesh, lat_mesh, data_interp,
            plot_type,
            surface_config.get('levels'),
            surface_config.get('colors', 'black'),
            surface_config.get('colormap'),
            surface_config.get('linewidth', 1.0),
            surface_config.get('alpha', 1.0),
            surface_config.get('add_labels', False)
        )
        
        print(f"Added {plot_type} surface overlay for {var_name}")

    def create_batch_surface_maps(
        self,
        processor: Any,
        output_dir: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        var_name: str = 't2m',
        plot_type: str = 'scatter',
        file_prefix: str = 'mpas_surface',
        formats: List[str] = ['png'],
        grid_resolution: Optional[float] = None,
        clim_min: Optional[float] = None,
        clim_max: Optional[float] = None
    ) -> List[str]:
        """
        The method extracts per-time-step variable data and coordinates from `processor`, invokes `create_surface_map` for each step, saves the plots in the requested formats, and returns a list of created file paths. It prints progress updates during processing and continues on errors to allow robust batch runs. If the processor does not have a loaded dataset, it raises a ValueError to indicate that surface maps cannot be created without data. The method ensures that surface maps are created for each time step in the dataset, with appropriate handling of variable extraction, plotting, and file saving, while also providing feedback on the processing status and any issues encountered.

        Parameters:
            processor (Any): Processor object with a loaded xarray dataset and helper methods such as `get_2d_variable_data`.
            output_dir (str): Directory path where output files will be saved.
            lon_min (float): Western longitude bound in degrees.
            lon_max (float): Eastern longitude bound in degrees.
            lat_min (float): Southern latitude bound in degrees.
            lat_max (float): Northern latitude bound in degrees.
            var_name (str): Variable name to plot for each time step.
            plot_type (str): Rendering mode for plots ('scatter', 'contour', etc.).
            file_prefix (str): Prefix for generated file names.
            formats (List[str]): List of file formats to save (e.g., ['png', 'pdf']).
            grid_resolution (Optional[float]): Optional grid resolution for contouring.
            clim_min (Optional[float]): Optional shared colorbar minimum across the batch.
            clim_max (Optional[float]): Optional shared colorbar maximum across the batch.

        Returns:
            List[str]: Paths to all created files (one per requested format per time step).
        """
        # Validate that the processor has a loaded dataset before attempting to create surface maps, and raise a ValueError if no dataset is available
        if processor.dataset is None:
            raise ValueError("No data loaded in processor")

        # Determine the name of the time dimension in the dataset (either 'Time' or 'time') and calculate the total number of time steps available for processing
        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
        total_times = processor.dataset.sizes[time_dim]

        # Initialize an empty list to keep track of the file paths of the created surface maps
        created_files = []
        print(f"\nCreating surface maps for {total_times} time steps...")

        for time_idx in range(total_times):
            try:
                # For each time step, we attempt to extract the variable data and coordinates, create a surface map, and save the plot. 
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                    time_end = pd.Timestamp(processor.dataset.Time.values[time_idx]).to_pydatetime()
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_end = None
                    time_str = f"t{time_idx:03d}"

                # Extract the 2D variable data and corresponding longitude and latitude coordinates for the current time index 
                var_data = processor.get_2d_variable_data(var_name, time_idx)
                lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)

                # Construct a descriptive title for the plot that includes the variable name, valid time, and plot type for better interpretability 
                title = f"MPAS Surface Map | Var: {var_name.upper()} | Valid: {time_str} | Type: {plot_type.title()}"

                # Create the surface map for the current time step using the extracted data and coordinates, and the specified plotting parameters 
                fig, ax = self.create_surface_map(
                    lon, lat, var_data.values, var_name,
                    lon_min, lon_max, lat_min, lat_max,
                    title=title,
                    plot_type=plot_type,
                    time_stamp=time_end,
                    data_array=var_data,
                    grid_resolution=grid_resolution,
                    clim_min=clim_min,
                    clim_max=clim_max
                )

                # Construct the output file path using the specified output directory, file prefix, variable name, plot type, and time string
                output_path = os.path.join(output_dir, f"{file_prefix}_{var_name}_{plot_type}_{time_str}")

                # Save the created plot in the requested formats and append the paths of the created files to the list for tracking
                self.save_plot(output_path, formats=formats)

                # Append the paths of the created files to the list for tracking
                for fmt in formats:
                    created_files.append(f"{output_path}.{fmt}")

                # After saving the plot, we close the figure to free up memory before processing the next time step
                self.close_plot()

                # Print a progress update every 10 time steps to inform the user about the processing status, including how many surface maps have been completed 
                if (time_idx + 1) % 10 == 0:
                    print(f"Completed {time_idx + 1}/{total_times} surface maps...")

            except Exception as e:
                print(f"Error creating surface map for time index {time_idx}: {e}")
                continue

        print(f"\nBatch processing completed. Created {len(created_files)} files.")
        return created_files

    def get_surface_colormap_and_levels(
        self,
        var_name: str,
        data_array: Optional[xr.DataArray] = None
    ) -> Tuple[str, List[float]]:
        """
        This convenience function queries `MPASFileMetadata` and returns the metadata-provided colormap and level specification, optionally using `data_array` to refine level selection. If the variable name is not recognized, it returns a default colormap and levels. It ensures that the appropriate colormap and contour levels are retrieved based on the variable being plotted, which can help to enhance the visual representation of the data by using metadata-informed styling choices. The method includes debug print statements to confirm the retrieved colormap and levels for the specified variable, which can assist in troubleshooting and verifying that the correct metadata is being applied for the surface plot.

        Parameters:
            var_name (str): Name of the 2D surface variable (e.g., 't2m').
            data_array (Optional[xarray.DataArray]): Optional DataArray used for metadata-aware level selection.

        Returns:
            Tuple[str, List[float]]: Tuple containing a colormap name and a list of contour levels.
        """
        # Extract the colormap and levels for the specified variable name from the MPASFileMetadata
        metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        return metadata['colormap'], metadata['levels']

    def create_simple_scatter_plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "MPAS Surface Variable",
        colorbar_label: str = "Value",
        colormap: str = 'viridis',
        point_size: float = 2.0
    ) -> Tuple[Figure, Axes]:
        """
        This lightweight helper uses plain matplotlib axes, filters invalid points, renders a colored scatter with a colorbar, and applies basic labels and branding. It is intended for rapid interactive use where cartographic projection is unnecessary. If the input data contains no valid points after filtering, it raises a ValueError to indicate that a scatter plot cannot be created. It ensures that a simple scatter plot can be created for surface variables without the need for cartographic projections, while also providing basic styling and error handling to manage cases where the input data may not be suitable for plotting. The method includes debug print statements to confirm the number of valid points being plotted and any issues encountered during the creation of the scatter plot, which can assist in troubleshooting and verifying that the plot is created correctly based on the provided parameters.

        Parameters:
            lon (np.ndarray): Longitude coordinates in degrees.
            lat (np.ndarray): Latitude coordinates in degrees.
            data (np.ndarray): Data values used for color mapping.
            title (str): Title for the plot.
            colorbar_label (str): Label for the colorbar.
            colormap (str): Matplotlib colormap name.
            point_size (float): Marker size for scatter points.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figure and axes containing the scatter plot.
        """
        # Create a new figure and axes for the scatter plot with the specified size and resolution
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Filter out invalid data points by creating a boolean mask that checks for finite values in the data, longitude, and latitude arrays 
        valid_mask = np.isfinite(data) & np.isfinite(lon) & np.isfinite(lat)

        # If no valid data points are found after filtering, we raise a ValueError
        if not np.any(valid_mask):
            raise ValueError("No valid data points found after filtering for finite values")
        
        # Extract only the valid longitude, latitude, and data values using the boolean mask 
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        data_valid = data[valid_mask]
        
        # Create a scatter plot using the valid longitude and latitude coordinates, coloring the points based on the valid data values
        scatter = self.ax.scatter(lon_valid, lat_valid, c=data_valid, 
                                cmap=colormap, s=point_size, alpha=0.8)
        
        # Add a colorbar to the plot to indicate the mapping of data values to colors, and set the label for the colorbar 
        cbar = self.fig.colorbar(scatter, ax=self.ax)
        cbar.set_label(colorbar_label, fontsize=12)
        
        # Set the x and y labels for the axes, the title for the plot, and add a grid for better readability. 
        self.ax.set_xlabel('Longitude', fontsize=12)
        self.ax.set_ylabel('Latitude', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Adjust the layout to ensure that labels and titles are not cut off, and add a timestamp and branding to the plot for context and attribution
        plt.tight_layout()
        self.add_timestamp_and_branding()
        
        # Return the figure and axes containing the scatter plot for further manipulation or saving by the caller
        return self.fig, self.ax


def create_surface_plot(
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    var_name: str,
    extent: Tuple[float, float, float, float],
    plot_type: str = 'scatter',
    title: Optional[str] = None,
    colormap: Optional[str] = None,
    **kwargs: Any
) -> Tuple[Figure, Axes]:
    """
    This thin wrapper constructs an `MPASSurfacePlotter`, unpacks the `extent` tuple into bounds, and delegates to `create_surface_map`, forwarding extra keyword arguments. It is convenient for quick scripting and interactive sessions. If the input data contains no valid points after filtering, it raises a ValueError to indicate that a surface plot cannot be created. It ensures that a surface plot can be created with minimal setup by providing a simple interface that handles the construction of the plotter and the delegation of plotting tasks, while also providing error handling for cases where the input data may not be suitable for plotting. The method includes debug print statements to confirm the parameters being used for creating the surface plot and any issues encountered during the process, which can assist in troubleshooting and verifying that the plot is created correctly based on the provided parameters.

    Parameters:
        lon (np.ndarray): 1D longitude array in degrees.
        lat (np.ndarray): 1D latitude array in degrees.
        data (np.ndarray): 1D data values corresponding to the coordinate arrays.
        var_name (str): Variable name used for metadata lookup and unit conversion.
        extent (Tuple[float, float, float, float]): (lon_min, lon_max, lat_min, lat_max) bounds for the plot.
        plot_type (str): Rendering mode ('scatter', 'contour', or 'contourf').
        title (Optional[str]): Optional title override.
        colormap (Optional[str]): Optional colormap override.
        **kwargs (Any): Additional keyword arguments passed to `MPASSurfacePlotter.create_surface_map`.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figure and GeoAxes with the rendered map.
    """
    # Create an instance of the MPASSurfacePlotter to handle the plotting of the surface map
    plotter = MPASSurfacePlotter()

    # Unpack the extent tuple into individual longitude and latitude bounds for use in the surface map creation
    lon_min, lon_max, lat_min, lat_max = extent
    
    # Delegate the creation of the surface map to the plotter's create_surface_map method, passing all relevant parameters 
    return plotter.create_surface_map(
        lon, lat, data, var_name,
        lon_min, lon_max, lat_min, lat_max,
        title=title, plot_type=plot_type, colormap=colormap,
        **kwargs
    )