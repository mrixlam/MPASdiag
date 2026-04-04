#!/usr/bin/env python3

"""
MPASdiag Core Visualization Module: Surface Variable Plotting and Overlays

This module provides the MPASSurfacePlotter class, which specializes in creating professional cartographic visualizations of MPAS 2D/3D surface variables. It includes functionality for extracting surface-level data from multidimensional arrays, handling unit conversions based on metadata and variable names, determining appropriate colormaps and contour levels, setting up map projections and features, filtering valid data points for plotting, and rendering scatter, contour, filled contour, or combined plots with proper styling. The class is designed to be flexible and robust, allowing users to create high-quality visualizations of MPAS surface variables while ensuring that the underlying data is processed correctly and displayed with meaningful units and color mappings. 
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February, 2026
Version: 1.0.0
"""
# Import necessary libraries 
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
from .styling import MPASVisualizationStyle


class MPASSurfacePlotter(MPASVisualizer):
    """ Specialized plotter for creating professional cartographic visualizations of MPAS 2D/3D surface variables. """

    _TIME_FMT = '%Y%m%dT%H'

    @staticmethod
    def _apply_level_index_slice(data: Union[np.ndarray, xr.DataArray], 
                                 level_index: int) -> Union[np.ndarray, xr.DataArray]:
        """
        This helper function applies slicing to the input data based on the specified level index. It handles both 2D and 3D data arrays, extracting the appropriate slice corresponding to the given level index. For 2D data, it assumes that the vertical dimension is the second dimension and slices accordingly. For 3D data, it slices along the vertical dimension while preserving any additional horizontal dimensions, and if there are more than 2 dimensions after slicing, it further reduces it to a 2D array by taking the first horizontal slice. If the data has fewer than 2 dimensions, it returns the data as is since there is no vertical dimension to slice. This function ensures that we can extract a consistent 2D surface-level slice from input data that may have varying numbers of dimensions, which is essential for creating surface plots. Debug print statements can be included to confirm the slicing process and the resulting shape of the extracted data, which can assist in troubleshooting and verifying that the correct slice of data is being used for visualization.

        Parameters:
            data (Union[np.ndarray, xr.DataArray]): Input data array that may be 2D or 3D.
            level_index (int): Index of the vertical level to extract (e.g., 0 for surface).

        Returns:
            Union[np.ndarray, xr.DataArray]: Extracted 2D slice of data corresponding to the specified level index.
        """
        if data.ndim == 2:
            return data[:, level_index]
        elif data.ndim == 3:
            sliced = data[:, level_index, ...]
            if sliced.ndim > 1:
                sliced = sliced[..., 0]
            return sliced
        return data

    def _extract_2d_from_multidimensional(self: "MPASSurfacePlotter",
                                          data: Union[np.ndarray, xr.DataArray],
                                          level_index: Optional[int],
                                          level_value: Optional[float]) -> np.ndarray:
        """
        This helper function extracts a 2D surface-level slice from input data that may be 1D, 2D, or 3D. It handles various cases based on the number of dimensions and the presence of optional level_index or level_value parameters. If the data is already 1D or 2D, it converts it to a numpy array and returns it directly. For 3D data, it checks if level_index or level_value is specified to select the appropriate vertical level; if neither is provided, it defaults to taking the last level as the surface. After extracting the relevant slice, it flattens any remaining dimensions to ensure a 1D array of surface values ready for plotting. The function also includes error handling for unsupported dimensions and debug print statements to confirm the extraction process, which can assist in troubleshooting and verifying that the correct slice of data is being used for visualization. 

        Parameters:
            data (Union[np.ndarray, xr.DataArray]): Input data array that may be 1D, 2D, or 3D.
            level_index (Optional[int]): Optional index of the vertical level to extract (e.g., 0 for surface).
            level_value (Optional[float]): Optional value of the vertical coordinate to extract (not yet implemented).

        Returns:
            np.ndarray: 1D array of surface-level data values ready for plotting. 
        """
        # If data is already 1D or 2D, convert to numpy and return directly
        if data.ndim <= 1:
            return self.convert_to_numpy(data)
        
        # For 2D data, assume the last dimension is vertical and extract the surface level if level_index or level_value is specified, otherwise take the last level as surface
        if data.ndim > 3:
            raise ValueError(f"only 1D, 2D and 3D data are supported, got {data.ndim}D data with shape {data.shape}")
        
        # Extract the appropriate 2D slice based on level_index or level_value, defaulting to surface level if neither is provided. 
        if level_index is not None:
            data = self._apply_level_index_slice(data, level_index)
            print(f"Extracted 2D data using level_index={level_index}")
        else:
            # Default to surface level (level_value selection not yet implemented)
            data = data[:, -1] if data.ndim == 2 else data[:, -1, ...]
            if data.ndim > 1:
                data = data[..., 0]
            if level_value is not None:
                print(f"Extracted 2D data using surface level (level_value={level_value} not yet implemented)")
            else:
                print("Extracted 2D data using surface level (default)")
        
        # Flatten the remaining dimensions to 1D for plotting and convert to numpy array
        if data.ndim > 1:
            data = data.flatten()
            print(f"Flattened remaining dimensions to 1D, final shape: {data.shape}")
        
        # Convert to numpy array if it's still an xarray DataArray and return
        return self.convert_to_numpy(data)

    @staticmethod
    def _coerce_converted_data(converted: Any) -> np.ndarray:
        """
        This helper function takes the output from a unit conversion operation, which may be an xarray DataArray, a numpy array, or a scalar value, and coerces it into a numpy array format suitable for plotting. It checks the type of the converted data and extracts the underlying values if it's an xarray DataArray, returns it directly if it's already a numpy array, or converts it to a numpy array if it's a scalar or another type. This ensures that regardless of the return type from the unit conversion process, we end up with a consistent numpy array format for subsequent plotting steps. Debug print statements can be included to confirm the type of the converted data and the resulting shape of the numpy array, which can assist in troubleshooting and verifying that the unit conversion is producing data in the expected format for visualization.

        Parameters:
            converted (Any): The output from a unit conversion operation, which may be an xarray DataArray, a numpy array, or a scalar value.

        Returns:
            np.ndarray: The converted data coerced into a numpy array format suitable for plotting.
        """
        if isinstance(converted, xr.DataArray):
            return converted.values
        
        if isinstance(converted, np.ndarray):
            return converted

        return np.asarray(converted)

    def _extract_and_convert_units(self: "MPASSurfacePlotter",
                                   data: np.ndarray,
                                   var_name: str,
                                   data_array: Optional[xr.DataArray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        This helper function handles the extraction of original units from the input data, associated DataArray attributes, or variable metadata, and performs unit conversion to display units based on the variable name and original units. It first attempts to determine the original units by checking the provided DataArray's attributes, then the data's attributes, and finally the MPASFileMetadata for the variable. Once the original units are identified, it queries the UnitConverter for appropriate display units based on the variable name and original units. If a conversion is needed, it attempts to convert the data to the display units while handling different return types from the conversion (e.g., xarray DataArray, numpy array, or scalar). Additionally, for moisture-related variables, it checks for negative values in the data and clips them to 0 if found since negative moisture is physically invalid. The function returns the converted data as a numpy array along with a metadata dictionary that includes both original and display units for downstream use in titles and colorbar labels. Debug print statements are included to confirm unit extraction and conversion processes, which can assist in troubleshooting and verifying that the correct units are being applied to the plot. 

        Parameters:
            data (np.ndarray): Input data array to be converted.
            var_name (str): Variable name used to determine display units and for logging.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray that may contain unit attributes.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Converted data array and variable metadata dictionary containing original and display units.
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
                data = self._coerce_converted_data(converted_data)
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
    
    def _prepare_colormap_and_levels(self: "MPASSurfacePlotter",
                                     colormap: Optional[str],
                                     levels: Optional[List[float]],
                                     var_metadata: Dict[str, Any],
                                     clim_min: Optional[float],
                                     clim_max: Optional[float]) -> Tuple[str, Optional[List[float]]]:
        """
        This helper function determines the final colormap and contour levels to use for plotting based on a combination of user-specified parameters and variable metadata. It prioritizes explicit inputs from the caller (colormap and levels) over metadata defaults, while also allowing for optional color limits (clim_min and clim_max) to filter the levels if both are provided. The function ensures that the final colormap is valid and falls back to a default if necessary, and it handles the logic for determining which levels to use based on the presence of explicit inputs versus metadata. The resulting colormap and levels are returned for use in the plotting functions. Debug print statements can be included to confirm the chosen colormap and levels, which can assist in troubleshooting and verifying that the correct styling parameters are being applied to the plot. 

        Parameters:
            colormap (Optional[str]): Optional colormap name or identifier provided by the caller.
            levels (Optional[List[float]]): Optional list of contour levels provided by the caller.
            var_metadata (Dict[str, Any]): Variable metadata dictionary that may contain default colormap and levels.
            clim_min (Optional[float]): Optional colorbar minimum value for filtering levels.
            clim_max (Optional[float]): Optional colorbar maximum value for filtering levels.

        Returns:
            Tuple[str, Optional[List[float]]]: Final colormap name and list of contour levels to use for plotting.
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
    
    def _setup_map_extent_and_features(self: "MPASSurfacePlotter",
                                       lon_min: float,
                                       lon_max: float,
                                       lat_min: float,
                                       lat_max: float,
                                       projection: str) -> Tuple[ccrs.CRS, ccrs.CRS, float, float, float, float, float, float, float, float]:
        """
        This helper function sets up the map projection, data coordinate reference system (CRS), and map features for the surface plot. It uses the base visualizer's setup function to determine the appropriate map projection and data CRS based on the requested geographic bounds and projection type. The function initializes the figure and GeoAxes with the specified projection, ensuring that the axes is a GeoAxes instance for cartopy plotting. It then determines if the requested extent is global in longitude and latitude to apply a safe extent that avoids dateline wrapping issues in cartopy. For global extents, it applies a small buffer to the bounds to prevent artifacts at the dateline, while for regional extents it uses the provided bounds directly. The function adds cartographic features such as coastlines, borders, land, and ocean with appropriate styling for better visualization. Finally, it returns the map projection, data CRS, and the computed filter bounds for both plotting and data selection to be used in subsequent steps of the plotting process. Debug print statements can be included to confirm the chosen extent and features, which can assist in troubleshooting and verifying that the map is being set up correctly based on the inputs provided. 

        Parameters:
            lon_min (float): Minimum longitude for the map extent.
            lon_max (float): Maximum longitude for the map extent.
            lat_min (float): Minimum latitude for the map extent.
            lat_max (float): Maximum latitude for the map extent.
            projection (str): Map projection type (e.g., 'PlateCarree', 'Mercator', etc.) to use for the plot.

        Returns:
            Tuple[ccrs.CRS, ccrs.CRS, float, float, float, float, float, float, float, float]: Map projection, data CRS, and filter bounds for plotting and data selection. 
        """
        # Set up the map projection and data CRS using the base visualizer's setup function
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)

        # Mercator cannot project latitudes at ±90° (the math diverges to ±∞). To avoid issues with cartopy when using a Mercator projection, we clamp the latitude bounds to a safe range just below ±90° to ensure that the CRS transformation does not encounter singularities at the poles. 
        if projection.lower() == 'mercator':
            _LAT_MERCATOR_MAX = 85.051
            lat_min = max(lat_min, -_LAT_MERCATOR_MAX)
            lat_max = min(lat_max,  _LAT_MERCATOR_MAX)

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

    def _build_boundary_norm(self: "MPASSurfacePlotter", 
                             levels: Optional[List[float]], 
                             cmap_obj: mcolors.Colormap) -> Optional[mcolors.Normalize]:
        """
        This helper function constructs a BoundaryNorm instance for the colormap based on the provided contour levels. It first checks if valid levels are provided (i.e., a non-empty list of finite values) and then creates a list of bounds that includes the minimum level, all the sorted levels, and an additional upper bound that is one unit above the maximum level to ensure proper mapping of colors. The BoundaryNorm is then created using these bounds and the number of colors in the colormap, with clipping enabled to handle values outside the specified levels. If no valid levels are provided, the function returns None to indicate that default normalization should be used for continuous colormaps. Debug print statements can be included to confirm the construction of the BoundaryNorm and the bounds used, which can assist in troubleshooting and verifying that the correct normalization is being applied to the plot based on the provided levels.

        Parameters:
            levels (Optional[List[float]]): Optional list of contour levels that may influence the construction of the BoundaryNorm.
            cmap_obj (mcolors.Colormap): Colormap object for which the BoundaryNorm is being constructed, used to determine the number of colors.

        Returns:
            Optional[mcolors.Normalize]: A BoundaryNorm instance if valid levels are provided, otherwise None to indicate default normalization should be used.
        """
        try:
            if levels is not None:
                color_levels_sorted = sorted(set([v for v in levels if np.isfinite(v)]))
                if color_levels_sorted:
                    last_bound = max(color_levels_sorted) + 1
                    bounds = [min(color_levels_sorted)] + color_levels_sorted + [last_bound]
                    return BoundaryNorm(bounds, ncolors=cmap_obj.N, clip=True)
        except Exception:
            pass
        return None

    def _create_colormap_normalization(self: "MPASSurfacePlotter",
                                       colormap: str,
                                       levels: Optional[List[float]],
                                       clim_min: Optional[float],
                                       clim_max: Optional[float],
                                       data: np.ndarray) -> Tuple[mcolors.Colormap, Optional[mcolors.Normalize]]:
        """
        This helper function creates a matplotlib colormap object and an optional normalization instance based on the provided colormap name, contour levels, color limits, and data range. It first attempts to get the colormap object from matplotlib using the provided colormap name or identifier, and falls back to a default if the specified colormap is not found or invalid. Then it determines the appropriate normalization for the colormap based on the presence of explicit color limits (clim_min and clim_max) or contour levels. If explicit color limits are provided, it uses a simple Normalize to scale the colormap between those limits. If contour levels are provided, it creates a BoundaryNorm that maps the specified levels to the colormap. The function includes error handling to ensure that a valid colormap is always returned and debug print statements to confirm the chosen colormap and normalization parameters, which can assist in troubleshooting and verifying that the correct styling is being applied to the plot. 

        Parameters:
            colormap (str): Colormap name or identifier to use for plotting.
            levels (Optional[List[float]]): Optional list of contour levels that may influence normalization.
            clim_min (Optional[float]): Optional minimum color limit for normalization.
            clim_max (Optional[float]): Optional maximum color limit for normalization.
            data (np.ndarray): Data array used to determine the range for normalization if color limits are not provided.

        Returns:
            Tuple[mcolors.Colormap, Optional[mcolors.Normalize]]: Colormap object and optional normalization instance for plotting.
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
            norm = self._build_boundary_norm(levels, cmap_obj)
        
        return cmap_obj, norm
    
    def _filter_valid_data(self: "MPASSurfacePlotter",
                           lon: np.ndarray,
                           lat: np.ndarray,
                           data: np.ndarray,
                           plot_type: str,
                           filter_lon_min_data: float,
                           filter_lon_max_data: float,
                           filter_lat_min_data: float,
                           filter_lat_max_data: float,
                           var_name: str,
                           var_metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This helper function filters the longitude, latitude, and data arrays to retain only valid points for plotting based on the specified plot type and geographic bounds. For scatter plots, it enforces that all longitude, latitude, and data values are finite and fall within the specified geographic bounds to ensure that only valid points are plotted. For contour-type rendering (contour or contourf), it retains all finite points regardless of their location relative to the plotting extent since contouring can interpolate values outside the immediate bounds. The function applies the appropriate filtering logic based on the plot type, checks for any remaining valid points after filtering to avoid plotting empty maps, and returns the filtered longitude, latitude, and data arrays containing only valid points for plotting. Debug print statements are included to confirm the number of valid points being plotted and their data range, which can assist in troubleshooting and verifying that the correct subset of data is being used for visualization. 

        Parameters:
            lon (np.ndarray): Array of longitude values corresponding to the data points.
            lat (np.ndarray): Array of latitude values corresponding to the data points.
            data (np.ndarray): Array of data values to be plotted.
            plot_type (str): Type of plot being created ('scatter', 'contour', 'contourf', or 'both') which influences the filtering logic.
            filter_lon_min_data (float): Minimum longitude for filtering data points based on the map extent.
            filter_lon_max_data (float): Maximum longitude for filtering data points based on the map extent.
            filter_lat_min_data (float): Minimum latitude for filtering data points based on the map extent.
            filter_lat_max_data (float): Maximum latitude for filtering data points based on the map extent.
            var_name (str): Variable name used for logging and error messages.
            var_metadata (Dict[str, Any]): Variable metadata used for logging units and other information.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered longitude, latitude, and data arrays containing only valid points for plotting.
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
    
    def _render_plot(self: "MPASSurfacePlotter",
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
                     dataset: Optional[xr.Dataset]) -> None:
        """
        This helper function renders the plot based on the specified plot type ('scatter', 'contour', 'contourf', or 'both') using the filtered valid longitude, latitude, and data points. It calls the appropriate internal helper functions to create scatter plots directly from the valid points or to create contour/filled contour plots by interpolating the valid data onto a regular lat-lon grid. For the 'both' option, it first creates contour lines and then overlays a scatter plot of the valid points to show the original data locations on top of the contours for enhanced visualization. The function ensures that all rendering is done directly onto the GeoAxes instance (self.ax) and that the figure is updated accordingly. Debug print statements can be included to confirm which rendering mode is being used and to verify that the plotting functions are being called with the correct parameters, which can assist in troubleshooting and verifying that the plot is being rendered as intended based on the inputs provided. 

        Parameters:
            plot_type (str): Type of plot to render ('scatter', 'contour', 'contourf', or 'both').
            lon_valid (np.ndarray): Array of valid longitude values for plotting.
            lat_valid (np.ndarray): Array of valid latitude values for plotting.
            data_valid (np.ndarray): Array of valid data values for plotting.
            filter_lon_min (float): Minimum longitude for the plotting extent.
            filter_lon_max (float): Maximum longitude for the plotting extent.
            filter_lat_min (float): Minimum latitude for the plotting extent.
            filter_lat_max (float): Maximum latitude for the plotting extent.
            cmap_obj (mcolors.Colormap): Colormap object to use for plotting.
            norm (Optional[mcolors.Normalize]): Optional normalization instance for colormap scaling.
            levels (Optional[List[float]]): Optional list of contour levels for contour plotting.
            data_crs (ccrs.CRS): Coordinate reference system of the data for proper plotting.
            grid_resolution (Optional[float]): Optional grid resolution for contour interpolation.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset that may be needed for contour interpolation.

        Returns:
            None: This function renders the plot directly onto the GeoAxes instance and does not return any value.
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
    
    def _generate_title(self: "MPASSurfacePlotter",
                        title: Optional[str],
                        time_stamp: Optional[datetime],
                        var_metadata: Dict[str, Any]) -> Tuple[str, bool]:
        """
        This helper function generates the title for the plot based on an optional user-provided title, an optional timestamp, and variable metadata. If a custom title is not provided, it constructs a default title using the variable's long name from the metadata. If a timestamp is provided and no custom title is given, it appends a formatted timestamp to the default title with a 'Valid Time:' prefix for clarity. If a custom title is provided along with a timestamp, it checks if the formatted timestamp or common time indicators are already included in the title to avoid redundant time annotations. The function returns the final title string to be used for the plot and a boolean flag indicating whether the timestamp is already included in the title, which can be useful for determining whether to add additional time annotations elsewhere in the plot (e.g., in subtitles or captions). Debug print statements can be included to confirm the generated title and whether the timestamp is included, which can assist in troubleshooting and verifying that the title is being created as intended based on the inputs provided. 

        Parameters:
            title (Optional[str]): Optional custom title provided by the user.
            time_stamp (Optional[datetime]): Optional timestamp to include in the title if not already present.
            var_metadata (Dict[str, Any]): Variable metadata that may contain the long name for constructing a default title.

        Returns:
            Tuple[str, bool]: Final title string and a boolean flag indicating whether the timestamp is already included in the title.
        """
        # Initialize time_in_title flag to False and then determine the title based on whether a custom title is provided 
        time_in_title = False
        
        if title is None:
            # If no custom title is provided, we construct a default title using the variable's long name from the metadata. 
            title = f"MPAS {var_metadata['long_name']}"
            if time_stamp:
                # If a timestamp is provided, we format it as 'YYYYMMDDTHH' and append it to the title with a 'Valid Time:' prefix for clarity. 
                time_str = time_stamp.strftime(self._TIME_FMT)
                title += f" | Valid Time: {time_str}"
                time_in_title = True
        else:
            if time_stamp:
                # If a custom title is provided, we check if the formatted timestamp is already included in the title to avoid redundant time annotations
                time_str = time_stamp.strftime(self._TIME_FMT)
                time_in_title = (time_str in title or 'Valid Time:' in title or 'Valid:' in title)
        
        # Return the final title string and the flag indicating whether the timestamp is already included in the title 
        return title, time_in_title
    
    def _add_overlays(self: "MPASSurfacePlotter",
                      lon: np.ndarray,
                      lat: np.ndarray,
                      wind_overlay: Optional[dict],
                      surface_overlay: Optional[dict],
                      filter_lon_min: float,
                      filter_lon_max: float,
                      filter_lat_min: float,
                      filter_lat_max: float,
                      dataset: Optional[xr.Dataset]) -> None:
        """
        This helper function adds optional overlays to the surface map based on the provided configurations for wind and surface overlays. If a wind overlay configuration is provided, it attempts to create and add the wind overlay to the map using the MPASWindPlotter class, handling any exceptions that may arise during this process and logging appropriate warnings. If a surface overlay configuration is provided, it attempts to add the surface overlay using internal helper methods, again with error handling to catch and log any issues that occur. The function ensures that the axes is a GeoAxes instance for cartopy plotting before attempting to add any overlays, and it uses the provided longitude and latitude coordinates along with the filter bounds to restrict the overlays to the desired geographic area. The dataset parameter is passed to the overlay functions as needed for remapping or interpolation of overlay data. Debug print statements can be included to confirm when overlays are successfully added or if any warnings occur during the process, which can assist in troubleshooting and verifying that the overlays are being applied correctly based on the inputs provided. 

        Parameters:
            lon (np.ndarray): Array of longitude values corresponding to the data points.
            lat (np.ndarray): Array of latitude values corresponding to the data points.
            wind_overlay (Optional[dict]): Optional configuration dictionary for adding a wind overlay to the map.
            surface_overlay (Optional[dict]): Optional configuration dictionary for adding an additional surface overlay to the map.
            filter_lon_min (float): Minimum longitude for filtering overlay data based on the map extent.
            filter_lon_max (float): Maximum longitude for filtering overlay data based on the map extent.
            filter_lat_min (float): Minimum latitude for filtering overlay data based on the map extent.
            filter_lat_max (float): Maximum latitude for filtering overlay data based on the map extent.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset that may be needed for remapping or interpolation of overlay data.

        Returns:
            None: This function adds overlays directly onto the GeoAxes instance and does not return any value.
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
    
    def create_surface_map(self: "MPASSurfacePlotter",
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
                           dataset: Optional[xr.Dataset] = None) -> Tuple[Figure, Axes]:
        """
        This function creates a surface map visualization for MPAS mesh cell center data based on the provided longitude, latitude, and data arrays, along with various configuration parameters for styling, projection, and overlays. It validates the input parameters, performs necessary unit conversions using variable metadata, prepares the colormap and contour levels, sets up the map projection and features, filters the data for valid points based on the plot type and geographic bounds, renders the plot using the appropriate method (scatter, contour, contourf, or both), generates a title based on metadata and timestamp information, adds gridlines and optional overlays such as wind vectors or additional surface fields, and finalizes the layout with timestamp and branding annotations. The function returns the figure and axes objects containing the rendered surface map for further manipulation or saving by the caller. Debug print statements can be included throughout the function to confirm key steps such as data filtering results, chosen colormap and levels, generated title, and successful addition of overlays, which can assist in troubleshooting and verifying that each component of the plotting process is functioning as intended based on the inputs provided. 

        Parameters:
            lon (np.ndarray): Array of longitude values corresponding to the data points.
            lat (np.ndarray): Array of latitude values corresponding to the data points.
            data (Union[np.ndarray, xr.DataArray]): Array of data values to be plotted, which can be a numpy array or an xarray DataArray.
            var_name (str): Variable name used for labeling and metadata extraction.
            lon_min (float): Minimum longitude for the map extent.
            lon_max (float): Maximum longitude for the map extent.
            lat_min (float): Minimum latitude for the map extent.
            lat_max (float): Maximum latitude for the map extent.
            title (Optional[str]): Optional custom title for the plot. If not provided, a default title will be generated based on variable metadata.
            plot_type (str): Type of plot to create ('scatter', 'contour', 'contourf', or 'both').
            colormap (Optional[str]): Optional colormap name or identifier to use for plotting. If not provided, a default colormap will be used.
            levels (Optional[List[float]]): Optional list of contour levels to use for contour plotting. If not provided, levels will be determined automatically.
            clim_min (Optional[float]): Optional minimum color limit for normalization. If not provided, it will be determined from the data.
            clim_max (Optional[float]): Optional maximum color limit for normalization. If not provided, it will be determined from the data.
            projection (str): Map projection type to use for the plot (e.g., 'PlateCarree', 'Mercator', etc.).
            time_stamp (Optional[datetime]): Optional timestamp to include in the title or annotations.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing the data, used for metadata extraction and unit conversion.
            grid_resolution (Optional[float]): Optional grid resolution for contour interpolation if needed.
            wind_overlay (Optional[dict]): Optional configuration for adding a wind overlay to the map.
            surface_overlay (Optional[dict]): Optional configuration for adding an additional surface overlay to the map.
            level_index (Optional[int]): Optional index for selecting a vertical level from a multi-dimensional data array.
            level_value (Optional[float]): Optional value for selecting a vertical level from a multi-dimensional data array based on coordinate values.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset that may be needed for remapping or interpolation of data for plotting.            

        Returns:
            Tuple[Figure, Axes]: The figure and axes objects containing the rendered surface map for further manipulation or saving by the caller.
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
            time_str = time_stamp.strftime(self._TIME_FMT)
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

    def _interpolate_to_grid(self: "MPASSurfacePlotter",
                             lon: np.ndarray,
                             lat: np.ndarray,
                             data: np.ndarray,
                             lon_min: float,
                             lon_max: float,
                             lat_min: float,
                             lat_max: float,
                             grid_resolution: Optional[float] = None,
                             dataset: Optional[xr.Dataset] = None,
                             method: str = 'nearest',
                             resolution_bounds: Optional[Tuple[float, float]] = None,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This helper function performs interpolation of the original longitude, latitude, and data points onto a regular lat-lon grid defined by the specified geographic bounds and grid resolution. It uses the specified interpolation method (e.g., 'nearest', 'linear', 'cubic') to interpolate the data values onto the new grid. The function can optionally take an xarray Dataset to assist with interpolation, such as providing coordinate information or metadata, but it can also operate solely based on the provided lon/lat/data arrays. The resolution_bounds parameter can be used to specify minimum and maximum allowable grid resolutions for adaptive interpolation if a specific grid resolution is not provided. The function returns the interpolated longitude, latitude, and data arrays corresponding to the regular grid for use in contour plotting or other visualizations that require gridded data. 

        Parameters:
            lon (np.ndarray): Original longitude values corresponding to the data points.
            lat (np.ndarray): Original latitude values corresponding to the data points.
            data (np.ndarray): Original data values corresponding to the longitude and latitude points.
            lon_min (float): Minimum longitude for defining the grid extent.
            lon_max (float): Maximum longitude for defining the grid extent.
            lat_min (float): Minimum latitude for defining the grid extent.
            lat_max (float): Maximum latitude for defining the grid extent.
            grid_resolution (Optional[float]): Optional grid resolution for the regular lat-lon grid. If not provided, it may be determined based on the input data or resolution_bounds.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset that may assist with interpolation, such as providing coordinate information or metadata.
            method (str): Interpolation method to use ('nearest', 'linear', 'cubic').
            resolution_bounds (Optional[Tuple[float, float]]): Minimum and maximum allowable grid resolutions for adaptive interpolation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Interpolated longitude, latitude, and data arrays corresponding to the regular grid.
        """
        return super()._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, dataset, method=method,
            resolution_bounds=resolution_bounds,
        )


    def add_surface_overlay(self: "MPASSurfacePlotter",
                            ax: Axes,
                            lon: Union[np.ndarray, xr.DataArray],
                            lat: Union[np.ndarray, xr.DataArray],
                            surface_config: Dict[str, Any],
                            lon_min: Optional[float] = None,
                            lon_max: Optional[float] = None,
                            lat_min: Optional[float] = None,
                            lat_max: Optional[float] = None,
                            dataset: Optional[xr.Dataset] = None) -> None:
        """
        This function adds an additional surface overlay to the existing GeoAxes instance based on the provided longitude, latitude, and surface configuration. It serves as a public interface for adding surface overlays, which internally calls a helper function that performs the actual processing and rendering of the overlay. The surface_config dictionary is expected to contain at least the key 'data' which holds the overlay data array, and it may also include optional styling keys such as 'var_name', 'plot_type', 'levels', 'colors', 'colormap', 'linewidth', 'alpha', 'level_index', 'add_labels', and 'grid_resolution' to customize the appearance of the overlay. The function allows for optional geographic bounds (lon_min, lon_max, lat_min, lat_max) to restrict the overlay to a specific area if desired. If a dataset is provided, it can be used to assist with remapping or interpolation of the overlay data as needed. The function does not return any value as it directly modifies the provided GeoAxes instance by adding the surface overlay on top of the existing plot. Debug print statements can be included to confirm when the overlay is being added and to log any relevant information about the overlay configuration or data being used, which can assist in troubleshooting and verifying that the overlay is being applied correctly based on the inputs provided. 

        Parameters:
            ax (Axes): The GeoAxes instance to which the surface overlay will be added.
            lon (Union[np.ndarray, xr.DataArray]): Longitude values corresponding to the overlay data points, which can be a numpy array or an xarray DataArray.
            lat (Union[np.ndarray, xr.DataArray]): Latitude values corresponding to the overlay data points, which can be a numpy array or an xarray DataArray.
            surface_config (Dict[str, Any]): Configuration dictionary for the surface overlay, expected to contain at least the key 'data' for the overlay data array and optional styling keys.
            lon_min (Optional[float]): Optional minimum longitude for restricting the overlay to a specific area.
            lon_max (Optional[float]): Optional maximum longitude for restricting the overlay to a specific area.
            lat_min (Optional[float]): Optional minimum latitude for restricting the overlay to a specific area.
            lat_max (Optional[float]): Optional maximum latitude for restricting the overlay to a specific area.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset that may assist with remapping or interpolation of the overlay data as needed.

        Returns:
            None: This function modifies the provided GeoAxes instance by adding the surface overlay and does not return any value.
        """
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

    def _infer_overlay_units(self: "MPASSurfacePlotter",
                             var_name: str,
                             overlay_data: np.ndarray) -> Optional[str]:
        """
        This helper function attempts to infer the original units of the overlay data based on the variable name and mean value of the data if explicit units are not provided in the surface_config. It uses common conventions in variable naming (e.g., 'mslp' or 'pressure' for pressure fields, 't2m' or 'temp' for temperature fields) along with typical value ranges to make an educated guess about the units. For example, if the variable name suggests it is a pressure field and the mean value is greater than 50,000, it may infer that the units are Pascals (Pa). If the variable name suggests it is a temperature field and the mean value is greater than 100, it may infer that the units are Kelvin (K). If it cannot confidently infer the units based on these heuristics, it returns None, indicating that unit conversion may not be possible without explicit information. Debug print statements can be included to log the inferred units or the reasoning behind why certain units were inferred based on the variable name and data values, which can assist in troubleshooting and verifying that unit inference is working as intended based on common conventions.

        Parameters:
            var_name (str): Variable name used for inferring units based on naming conventions.
            overlay_data (np.ndarray): The overlay data array for which units are being inferred, used to calculate mean values for heuristic inference.

        Returns:
            Optional[str]: The inferred original units of the overlay data (e.g., 'Pa', 'K') or None if units cannot be confidently inferred.
        """
        data_mean = np.nanmean(overlay_data)

        if 'mslp' in var_name.lower() or 'pressure' in var_name.lower():
            if data_mean > 50000:
                return 'Pa'
            
        elif 't2m' in var_name.lower() or 'temp' in var_name.lower():
            if data_mean > 100:
                return 'K'
            
        return None

    def _apply_overlay_unit_conversion(self: "MPASSurfacePlotter",
                                       overlay_data: np.ndarray,
                                       var_name: str,
                                       original_units: str) -> np.ndarray:
        """
        This helper function applies unit conversion to the overlay data if the original units can be determined (either through explicit configuration or inference) and if they differ from the display units determined by the UnitConverter. It uses the UnitConverter to convert the overlay data from its original units to the display units for consistency with the main plot. If the conversion is successful, it logs the conversion for debugging purposes. If a ValueError occurs during conversion (e.g., due to incompatible units), it catches the exception and logs a warning without interrupting the plotting process, allowing the overlay to be plotted with its original units if conversion fails. The function returns the overlay data array, which may have been converted to display units if conversion was successful, or left unchanged if conversion was not possible. Debug print statements can be included to confirm when unit conversion is applied and to log any warnings if conversion fails, which can assist in troubleshooting and verifying that unit conversion is being handled correctly based on the inputs provided.

        Parameters:
            overlay_data (np.ndarray): The overlay data array to be converted.
            var_name (str): Variable name used for determining display units.
            original_units (str): The original units of the overlay data.

        Returns:
            np.ndarray: The overlay data array converted to display units, if conversion was possible.
        """
        display_units = UnitConverter.get_display_units(var_name, original_units)
        if original_units != display_units:
            try:
                overlay_data = self.convert_to_numpy(
                    UnitConverter.convert_units(overlay_data, original_units, display_units)
                )
                print(f"Converted overlay {var_name} from {original_units} to {display_units}")
            except ValueError as e:
                print(f"Warning: Could not convert overlay {var_name} from {original_units} to {display_units}: {e}")
        return overlay_data

    def _prepare_overlay_data(self: "MPASSurfacePlotter",
                              overlay_data: np.ndarray,
                              lon: np.ndarray,
                              lat: np.ndarray,
                              var_name: str,
                              surface_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This helper function prepares the overlay data for plotting by performing unit inference and conversion, handling multi-dimensional data by extracting a 2D slice if necessary, and filtering valid data points based on finite values in the overlay data and corresponding longitude and latitude arrays. It first attempts to infer the original units of the overlay data based on the variable name and mean value if explicit units are not provided in the surface_config. If original units are determined, it uses the UnitConverter to convert the overlay data to display units for consistency with the main plot. If the overlay data is multi-dimensional (e.g., 3D with vertical levels), it extracts a 2D slice based on a specified level index or defaults to the last level. Finally, it creates a valid mask to filter out any points where the overlay data or corresponding longitude/latitude values are not finite, ensuring that only valid points are returned for interpolation and rendering of the surface overlay. If no valid points are found after filtering, it raises a ValueError to indicate that the overlay cannot be rendered due to lack of valid data. The function returns the filtered longitude, latitude, and overlay data arrays containing only valid points for use in subsequent plotting steps. Debug print statements can be included to confirm unit conversions, handling of multi-dimensional data, and results of filtering for valid points, which can assist in troubleshooting and verifying that the overlay data is being prepared correctly based on the inputs provided. 

        Parameters:
            overlay_data (np.ndarray): The original overlay data array that may require unit conversion and filtering.
            lon (np.ndarray): Longitude values corresponding to the overlay data points.
            lat (np.ndarray): Latitude values corresponding to the overlay data points.
            var_name (str): Variable name used for unit inference and conversion.
            surface_config (Dict[str, Any]): Configuration dictionary for the surface overlay, which may contain information about original units and level index for multi-dimensional data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered longitude, latitude, and overlay data arrays containing only valid points for plotting.
        """
        original_units = surface_config.get('original_units', None)

        # If original units are not explicitly provided in the surface_config, we attempt to infer them
        if original_units is None:
            original_units = self._infer_overlay_units(var_name, overlay_data)

        # If original units are determined, we apply unit conversion to the overlay data
        if original_units:
            overlay_data = self._apply_overlay_unit_conversion(overlay_data, var_name, original_units)
        
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
    
    def _calculate_overlay_bounds(self: "MPASSurfacePlotter",
                                  lon_valid: np.ndarray,
                                  lat_valid: np.ndarray,
                                  lon_min: Optional[float],
                                  lon_max: Optional[float],
                                  lat_min: Optional[float],
                                  lat_max: Optional[float]) -> Tuple[float, float, float, float]:
        """
        This helper function calculates the geographic bounds for the surface overlay based on the valid longitude and latitude points. If explicit bounds are provided by the caller, it uses those directly. Otherwise, it computes the bounds from the valid longitude and latitude arrays to ensure that the overlay is properly framed within the spatial extent of the valid data points. This approach allows for flexibility in defining the overlay extent while also providing a sensible default based on the actual data being plotted. The function returns a tuple containing the resolved (lon_min, lon_max, lat_min, lat_max) bounds that can be used for subsequent interpolation and plotting of the surface overlay. Debug print statements can be included to confirm the calculated bounds for the overlay, which can assist in troubleshooting and verifying that the bounds are being determined correctly based on the input parameters. 

        Parameters:
            lon_valid (np.ndarray): Array of valid longitude values corresponding to the overlay data points.
            lat_valid (np.ndarray): Array of valid latitude values corresponding to the overlay data points.
            lon_min (Optional[float]): Optional minimum longitude for the overlay bounds. If not provided, it will be calculated from the valid longitude array.
            lon_max (Optional[float]): Optional maximum longitude for the overlay bounds. If not provided, it will be calculated from the valid longitude array.
            lat_min (Optional[float]): Optional minimum latitude for the overlay bounds. If not provided, it will be calculated from the valid latitude array.
            lat_max (Optional[float]): Optional maximum latitude for the overlay bounds. If not provided, it will be calculated from the valid latitude array.

        Returns:
            Tuple[float, float, float, float]: Resolved (lon_min, lon_max, lat_min, lat_max) bounds for the surface overlay. 
        """
        # If explicit bounds are provided, we use them directly; otherwise, we compute the bounds from the valid longitude and latitude arrays to ensure that the overlay is properly framed within the spatial extent of the valid data points.
        return (
            lon_min if lon_min is not None else float(lon_valid.min()),
            lon_max if lon_max is not None else float(lon_valid.max()),
            lat_min if lat_min is not None else float(lat_valid.min()),
            lat_max if lat_max is not None else float(lat_valid.max())
        )
    
    def _calculate_overlay_resolution(self: "MPASSurfacePlotter",
                                      grid_resolution_input: Optional[float],
                                      lon_min: float,
                                      lon_max: float,
                                      lat_min: float,
                                      lat_max: float) -> float:
        """
        This helper function calculates the grid resolution to use for remapping the surface overlay data onto a regular lat-lon grid. If the caller has provided an explicit grid resolution, it uses that directly for remapping. Otherwise, it calculates an adaptive resolution based on the spatial extent of the overlay data defined by the longitude and latitude bounds. The resolution is set to be approximately 2% of the larger dimension of the spatial extent to ensure a reasonable number of grid points for interpolation without being too coarse or too fine. This approach allows for flexibility in defining the grid resolution while also providing a sensible default based on the actual geographic area covered by the overlay data. The function returns the determined grid spacing in degrees that can be used for remapping the overlay data to a regular lat-lon grid. Debug print statements can be included to confirm the calculated resolution for the overlay, which can assist in troubleshooting and verifying that the resolution is being determined correctly based on the input parameters and spatial extent. 

        Parameters:
            grid_resolution_input (Optional[float]): Optional explicit grid resolution provided by the caller. If not provided, it will be calculated based on the spatial extent of the overlay data.
            lon_min (float): Minimum longitude bound for the overlay data, used to calculate spatial extent if grid_resolution_input is not provided.
            lon_max (float): Maximum longitude bound for the overlay data, used to calculate spatial extent if grid_resolution_input is not provided.
            lat_min (float): Minimum latitude bound for the overlay data, used to calculate spatial extent if grid_resolution_input is not provided.
            lat_max (float): Maximum latitude bound for the overlay data, used to calculate spatial extent if grid_resolution_input is not provided.

        Returns:
            float: Grid resolution in degrees to use for remapping the surface overlay data to a regular lat-lon grid.
        """
        # If the caller has provided an explicit grid resolution, we use it directly for remapping; otherwise, we calculate an adaptive resolution based on spatial extent 
        if grid_resolution_input is not None:
            return float(grid_resolution_input)
        
        # Calculate the longitude and latitude range to determine the spatial extent of the overlay data
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        # Set the resolution to be approximately 2% of the larger dimension of the spatial extent to ensure a reasonable number of grid points for interpolation 
        return max(lon_range / 50, lat_range / 50)
    
    def _create_overlay_dataset(self: "MPASSurfacePlotter",
                                lon: np.ndarray,
                                lat: np.ndarray,
                                dataset: Optional[xr.Dataset]) -> xr.Dataset:
        """
        This helper function creates an xarray Dataset containing the longitude and latitude coordinates for the source points of the overlay data. If a dataset is already provided by the caller, it returns that dataset directly without modification. If no dataset is provided, it constructs a new xarray Dataset with 'lonCell' and 'latCell' DataArrays based on the provided longitude and latitude arrays. This dataset can then be used for remapping or interpolation of the overlay data to a regular lat-lon grid as needed for plotting. The function ensures that the longitude and latitude arrays are properly converted to numpy arrays if they are provided as xarray DataArrays, and it assigns appropriate dimension names for use in subsequent processing steps. The resulting dataset provides a structured way to represent the source coordinates for the overlay data, which can be essential for accurate remapping and visualization on the map. Debug print statements can be included to confirm whether a new dataset was created or an existing one was used, which can assist in troubleshooting and verifying that the dataset is being prepared correctly based on the input parameters. 

        Parameters:
            lon (np.ndarray): Longitude values corresponding to the overlay data points.
            lat (np.ndarray): Latitude values corresponding to the overlay data points.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset that may already contain the necessary coordinate information for remapping. If provided, it will be returned directly. If not provided, a new Dataset will be created using the provided longitude and latitude arrays. 

        Returns:
            xr.Dataset: An xarray Dataset containing 'lonCell' and 'latCell' DataArrays for the source coordinates of the overlay data, either from the provided dataset or newly created from the longitude and latitude arrays.
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
    
    def _interpolate_overlay(self: "MPASSurfacePlotter",
                             data_valid: np.ndarray,
                             dataset: xr.Dataset,
                             lon_min: float,
                             lon_max: float,
                             lat_min: float,
                             lat_max: float,
                             resolution: float,
                             var_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This helper function performs the interpolation of the valid overlay data points onto a regular lat-lon grid defined by the specified geographic bounds and resolution. It uses the remapping module to interpolate the data from the unstructured MPAS mesh to a regular lat-lon grid, applying masking to ensure that only valid data points are included in the interpolation. The function creates 2D meshgrids for longitude and latitude from the remapped result to be used in contour plotting, and it extracts the interpolated data values as a 2D array for use in rendering the surface overlay on the map. The function returns the longitude mesh, latitude mesh, and interpolated data array corresponding to the regular grid for use in contour or contourf plotting of the surface overlay on the map. Debug print statements can be included to confirm that interpolation is being performed and to log the resolution being used, which can assist in troubleshooting and verifying that the interpolation is being executed correctly based on the input parameters. 

        Parameters:
            data_valid (np.ndarray): Array of valid data values corresponding to the overlay points that will be interpolated.
            dataset (xr.Dataset): xarray Dataset containing the source coordinates for the overlay data, used for remapping.
            lon_min (float): Minimum longitude for defining the grid extent for interpolation.
            lon_max (float): Maximum longitude for defining the grid extent for interpolation.
            lat_min (float): Minimum latitude for defining the grid extent for interpolation.
            lat_max (float): Maximum latitude for defining the grid extent for interpolation.
            resolution (float): Grid spacing in degrees for the target grid.
            var_name (str): Variable name used for log messages.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Longitude mesh, latitude mesh, and interpolated data array corresponding to the regular grid for contour plotting. 
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
    
    def _validate_contour_levels(self: "MPASSurfacePlotter",
                                 data_interp: np.ndarray,
                                 levels: Optional[List[float]],
                                 var_name: str) -> None:
        """
        This helper function validates the provided contour levels against the range of the interpolated data to ensure that the levels are appropriate for the data being plotted. It computes the minimum and maximum values of the interpolated data to determine the range of values that will be plotted in the contours. If explicit contour levels are provided, it checks which of those levels fall within the range of the data and logs a warning if none of the levels are within the data range, which could indicate that the contours may not be visible or meaningful on the plot. If some levels are within the data range, it logs which levels are valid for plotting. This validation step helps to ensure that the contour visualization will be effective and that users are aware if their specified levels may not be suitable for the data being visualized. Debug print statements can be included to confirm the data range and which contour levels are valid, which can assist in troubleshooting and verifying that the contour levels are being validated correctly based on the input parameters. 

        Parameters:
            data_interp (np.ndarray): The 2D array of interpolated data values that will be contoured, used to determine the range of values for validation.
            levels (Optional[List[float]]): The list of contour levels provided by the user for validation against the data range.
            var_name (str): Variable name used for log messages.

        Returns:
            None: This function performs validation and logging but does not return any value. It raises a warning if contour levels are not within the data range.
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
    
    def _render_overlay(self: "MPASSurfacePlotter",
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
                        add_labels: bool) -> None:
        """
        This helper function renders the surface overlay on the provided GeoAxes instance using either contour lines or filled contours based on the specified plot type. It applies the appropriate styling parameters such as colors, colormap, line width, and transparency to ensure that the overlay is visually distinct and informative when plotted on top of the existing map features. For line contours, it uses ax.contour to create contour lines with the specified colors and linewidth, and it optionally adds inline contour labels if requested. For filled contours, it uses ax.contourf to create a filled contour plot with the specified colormap. The function ensures that the contours are plotted using the correct coordinate reference system transformation for geographic data. This rendering step is crucial for visualizing the surface overlay effectively on the map, and it allows for customization of the appearance of the contours to enhance readability and interpretability. Debug print statements can be included to confirm which plot type is being rendered and to log any relevant styling information, which can assist in troubleshooting and verifying that the overlay is being rendered correctly based on the input parameters. 

        Parameters:
            ax (Axes): The GeoAxes instance on which to render the surface overlay.
            lon_mesh (np.ndarray): 2D array of longitude values corresponding to the regular grid for contour plotting.
            lat_mesh (np.ndarray): 2D array of latitude values corresponding to the regular grid for contour plotting.
            data_interp (np.ndarray): 2D array of interpolated data values corresponding to the regular grid for contour plotting.
            plot_type (str): The type of contour plot to create, either 'contour' for line contours or 'contourf' for filled contours.
            levels (Optional[List[float]]): The list of contour levels to use for plotting. If None, it will be determined automatically by matplotlib.
            colors (str): The color specification for contour lines when plot_type is 'contour'.
            colormap (Optional[str]): The name of the colormap to use for filled contours when plot_type is 'contourf'.
            linewidth (float): The line width to use for contour lines when plot_type is 'contour'.
            alpha (float): The transparency level to apply to the contours, where 0 is fully transparent and 1 is fully opaque.
            add_labels (bool): Whether to add inline contour labels for line contours when plot_type is 'contour'.

        Returns:
            None: This function renders the surface overlay on the provided GeoAxes instance and does not return any value. It modifies the axes in place by adding the contour or filled contour plot based on the specified parameters.
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
    
    def _add_surface_overlay(self: "MPASSurfacePlotter",
                             ax: Axes,
                             lon: Union[np.ndarray, xr.DataArray],
                             lat: Union[np.ndarray, xr.DataArray],
                             surface_config: Dict[str, Any],
                             lon_min: Optional[float] = None,
                             lon_max: Optional[float] = None,
                             lat_min: Optional[float] = None,
                             lat_max: Optional[float] = None,
                             dataset: Optional[xr.Dataset] = None) -> None:
        """
        This helper function performs the actual processing and rendering of the surface overlay based on the provided longitude, latitude, and surface configuration. It extracts necessary information from the surface_config such as variable name, plot type, and styling options, and it validates the plot type to ensure it is supported. The function then prepares the overlay data by performing unit inference and conversion, handling multi-dimensional data if necessary, and filtering valid data points. It calculates the geographic bounds for the overlay based on valid points and any provided explicit bounds, and it determines the grid resolution for interpolation. If no dataset is provided for remapping, it creates one from the valid longitude and latitude arrays. The function then performs interpolation of the valid overlay data onto a regular lat-lon grid using the remapping module, and it validates the contour levels against the range of the interpolated data. Finally, it renders the overlay on the provided axes using either contour lines or filled contours based on the specified plot type and styling parameters. Debug print statements can be included throughout this process to confirm each step of the preparation, interpolation, validation, and rendering of the surface overlay, which can assist in troubleshooting and verifying that each part of the process is being executed correctly based on the inputs provided. 

        Parameters:
            ax (Axes): The GeoAxes instance to which the surface overlay will be added.
            lon (Union[np.ndarray, xr.DataArray]): Longitude values corresponding to the overlay data points, which can be a numpy array or an xarray DataArray.
            lat (Union[np.ndarray, xr.DataArray]): Latitude values corresponding to the overlay data points, which can be a numpy array or an xarray DataArray.
            surface_config (Dict[str, Any]): Configuration dictionary for the surface overlay, expected to contain at least the key 'data' for the overlay data array and optional styling keys.
            lon_min (Optional[float]): Optional minimum longitude for restricting the overlay to a specific area.
            lon_max (Optional[float]): Optional maximum longitude for restricting the overlay to a specific area.
            lat_min (Optional[float]): Optional minimum latitude for restricting the overlay to a specific area.
            lat_max (Optional[float]): Optional maximum latitude for restricting the overlay to a specific area.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset that may assist with remapping or interpolation of the overlay data as needed.

        Returns:
            None: This function modifies the provided GeoAxes instance by adding the surface overlay and does not return any value. It performs all necessary processing steps to prepare, interpolate, validate, and render the overlay based on the input parameters and configuration.
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

    def create_batch_surface_maps(self: "MPASSurfacePlotter",
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
                                  clim_max: Optional[float] = None) -> List[str]:
        """
        This function performs batch processing to create surface maps for each time step in the loaded dataset using the specified variable name and plot type. It iterates through all available time steps, extracts the necessary data for the specified variable, and creates a surface map for each time step using the `create_surface_map` method. The generated plots are saved to the specified output directory with file names that include the variable name, plot type, and time information for easy identification. The function handles errors gracefully by catching exceptions during the creation of each surface map and logging them without interrupting the entire batch process. It also provides progress updates every 10 time steps to inform the user about the processing status. At the end of the batch processing, it returns a list of file paths for all created surface map files across all requested formats. Debug print statements can be included throughout this process to confirm the progress of batch processing, any errors encountered, and the details of each created surface map, which can assist in troubleshooting and verifying that the batch processing is being executed correctly based on the input parameters. 

        Parameters:
            processor (Any): The MPAS data processor instance that has a loaded dataset from which to extract variable data for plotting.
            output_dir (str): The directory where the generated surface map files will be saved.
            lon_min (float): Minimum longitude for the surface map extent.
            lon_max (float): Maximum longitude for the surface map extent.
            lat_min (float): Minimum latitude for the surface map extent.
            lat_max (float): Maximum latitude for the surface map extent.
            var_name (str): Name of the 2D surface variable to plot (e.g., 't2m').
            plot_type (str): Type of plot to create for the surface map, either 'scatter', 'contour', or 'contourf'.
            file_prefix (str): Prefix to use for the output file names of the generated surface maps.
            formats (List[str]): List of file formats to save the plots in (e.g., ['png', 'pdf']).
            grid_resolution (Optional[float]): Optional grid resolution in degrees for remapping the data to a regular lat-lon grid. If not provided, an adaptive resolution will be calculated based on the spatial extent of the data.
            clim_min (Optional[float]): Optional minimum value for color limits in the plot. If not provided, it will be determined automatically based on the data range.
            clim_max (Optional[float]): Optional maximum value for color limits in the plot. If not provided, it will be determined automatically based on the data range. 

        Returns:
            List[str]: A list of file paths for all created surface map files across all requested formats.
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
                    time_str = time_end.strftime(self._TIME_FMT)
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

    def get_surface_colormap_and_levels(self: "MPASSurfacePlotter",
                                        var_name: str,
                                        data_array: Optional[xr.DataArray] = None) -> Tuple[str, List[float]]:
        """
        This helper function retrieves the appropriate colormap and contour levels for a given 2D surface variable name by looking up metadata from the MPASFileMetadata class. It uses the variable name and optionally the data array to access metadata that specifies the recommended colormap and contour levels for that variable, which can be used to ensure consistent and meaningful visualizations across different variables. The function returns a tuple containing the colormap name and a list of contour levels that can be applied when creating surface maps for the specified variable. Debug print statements can be included to confirm the retrieved colormap and levels for the variable, which can assist in troubleshooting and verifying that the correct styling information is being obtained based on the input parameters. 

        Parameters:
            var_name (str): The name of the 2D surface variable for which to retrieve the colormap and contour levels. This variable name is used to look up metadata that specifies the recommended styling for that variable.
            data_array (Optional[xr.DataArray]): An optional xarray DataArray containing the variable data, which may be used in some cases to determine appropriate levels based on the data range. However, in this implementation, it is primarily used for metadata lookup and is not required for retrieving the colormap and levels. 

        Returns:
            Tuple[str, List[float]]: A tuple containing the colormap name (as a string) and a list of contour levels (as floats) that are recommended for visualizing the specified variable. This information can be used when creating surface maps to ensure that the visual styling is appropriate for the variable being plotted.
        """
        # Extract the colormap and levels for the specified variable name from the MPASFileMetadata
        metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        return metadata['colormap'], metadata['levels']

    def create_simple_scatter_plot(self: "MPASSurfacePlotter",
                                   lon: np.ndarray,
                                   lat: np.ndarray,
                                   data: np.ndarray,
                                   title: str = "MPAS Surface Variable",
                                   colorbar_label: str = "Value",
                                   colormap: str = 'viridis',
                                   point_size: float = 2.0) -> Tuple[Figure, Axes]:
        """
        This function creates a simple scatter plot of the provided longitude, latitude, and data values without performing any interpolation or remapping. It is designed for cases where the user wants to visualize the raw data points directly on a map without applying contouring or gridding. The function takes in the coordinate arrays, data values, and styling parameters such as title, colorbar label, colormap, and point size to create a scatter plot that shows the distribution of the data points in geographic space. It filters out any invalid data points (e.g., NaN or infinite values) to ensure that only valid points are plotted. The resulting figure and axes containing the scatter plot are returned for further manipulation or saving by the caller. Debug print statements can be included to confirm the number of valid data points being plotted and the parameters being used for the scatter plot, which can assist in troubleshooting and verifying that the plot is being created correctly based on the input parameters. 

        Parameters:
            lon (np.ndarray): 1D array of longitude values corresponding to the data points to be plotted.
            lat (np.ndarray): 1D array of latitude values corresponding to the data points to be plotted.
            data (np.ndarray): 1D array of data values corresponding to the longitude and latitude coordinates, which will be used for coloring the scatter points.
            title (str): The title for the scatter plot, which will be displayed at the top of the figure.
            colorbar_label (str): The label for the colorbar that indicates what the colors represent in terms of data values.
            colormap (str): The name of the colormap to use for coloring the scatter points based on their data values.
            point_size (float): The size of the scatter points in the plot, where larger values will result in bigger points. 

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and axes objects for the created scatter plot, which can be further manipulated or saved by the caller. 
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
        
        # Add a colorbar using the centralized styling helper
        cbar = MPASVisualizationStyle.add_colorbar(
            self.fig, self.ax, scatter,
            label=colorbar_label, orientation='horizontal', fraction=0.03,
            pad=0.04, shrink=0.8, fmt=None, labelpad=4, label_pos='top', tick_labelsize=10
        )

        # Specify the tick label size for the colorbar 
        if cbar is not None:
            try:
                # Set the tick label size for the colorbar to ensure that the labels are legible and appropriately sized for the plot
                cbar.ax.tick_params(labelsize=10)
            except Exception:
                # If any issues occur while setting the tick label size, catch the exception and pass without raising an error
                pass
        
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


def create_surface_plot(lon: np.ndarray,
                        lat: np.ndarray,
                        data: np.ndarray,
                        var_name: str,
                        extent: Tuple[float, float, float, float],
                        plot_type: str = 'scatter',
                        title: Optional[str] = None,
                        colormap: Optional[str] = None,
                        **kwargs: Any) -> Tuple[Figure, Axes]:
    """
    This function serves as a high-level interface for creating a surface plot of a specified variable using the provided longitude, latitude, and data values. It takes in the coordinate arrays, data values, variable name, plot type, and optional styling parameters to create a surface map that visualizes the spatial distribution of the variable across the specified geographic extent. The function creates an instance of the MPASSurfacePlotter class to handle the plotting logic and delegates the creation of the surface map to the plotter's `create_surface_map` method, passing all relevant parameters including any additional keyword arguments for customization. The resulting figure and axes containing the rendered surface map are returned for further manipulation or saving by the caller. Debug print statements can be included to confirm the parameters being used for creating the surface plot and to indicate when the plot creation process is being initiated, which can assist in troubleshooting and verifying that the function is being called correctly based on the input parameters. 

    Parameters:
        lon (np.ndarray): 1D array of longitude values corresponding to the data points to be plotted.
        lat (np.ndarray): 1D array of latitude values corresponding to the data points to be plotted.
        data (np.ndarray): 1D array of data values corresponding to the longitude and latitude coordinates, which will be used for coloring the plot.
        var_name (str): The name of the variable being plotted, which can be used for labeling and metadata lookup.
        extent (Tuple[float, float, float, float]): A tuple specifying the geographic extent of the plot in the format (lon_min, lon_max, lat_min, lat_max).
        plot_type (str): The type of plot to create for the surface map, either 'scatter', 'contour', or 'contourf'.
        title (Optional[str]): An optional title for the plot that will be displayed at the top of the figure. If not provided, a default title may be generated based on the variable name and plot type.
        colormap (Optional[str]): An optional name of the colormap to use for coloring the plot based on data values. If not provided, a default colormap may be used based on the variable metadata.
        **kwargs: Additional keyword arguments that can be passed to customize various aspects of the plot creation process, such as styling options for contours or scatter points.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and axes objects for the created surface plot, which can be further manipulated or saved by the caller.
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