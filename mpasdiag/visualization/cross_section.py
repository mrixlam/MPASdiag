#!/usr/bin/env python3

"""
MPASdiag Core Visualization Module: Vertical Cross-Section Plotting and Analysis

This module provides the MPASVerticalCrossSectionPlotter class, which specializes in creating vertical cross-section visualizations of 3D MPAS atmospheric data along user-defined transects. It includes methods for generating cross-section data through spatial interpolation, handling various vertical coordinate systems, applying unit conversions and variable-specific styling, and rendering professional contour plots with appropriate axis formatting. The class is designed to be flexible and robust, supporting multiple plot types and customization options while ensuring informative and visually effective representations of atmospheric variables in a vertical cross-section context. It also includes internal helper methods for generating great-circle paths, extracting height data, converting vertical coordinates, configuring pressure axes, and retrieving time information for plot annotations. 
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Load necessary libraries 
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from math import radians, degrees, sin, cos, atan2, sqrt, asin
from scipy.spatial import KDTree 
from typing import Tuple, Optional, List, Dict, Any, Union, cast
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .base_visualizer import MPASVisualizer
from .styling import MPASVisualizationStyle
from ..processing.processors_3d import MPAS3DProcessor
from ..processing.utils_unit import UnitConverter
from ..processing.utils_metadata import MPASFileMetadata


class MPASVerticalCrossSectionPlotter(MPASVisualizer):
    """ Specialized plotter for creating vertical cross-section visualizations of 3D MPAS atmospheric data along user-defined transects through the atmosphere. """
    
    def __init__(self: 'MPASVerticalCrossSectionPlotter', 
                 figsize: Tuple[float, float] = (10, 12), 
                 dpi: int = 100) -> None:
        """
        This constructor initializes the MPASVerticalCrossSectionPlotter instance with specified figure dimensions and resolution. It calls the parent class constructor to set up the base visualization environment, allowing for consistent styling and branding across different plot types. The figsize parameter determines the size of the output figure in inches, while the dpi parameter controls the resolution of the saved figure for high-quality output. This setup ensures that the vertical cross-section plots are rendered with appropriate dimensions and clarity for analysis and presentation purposes. 

        Parameters:
            figsize (Tuple[float, float]): Figure size in inches (width, height) for the cross-section plot (default: (10, 12)).
            dpi (int): Dots per inch resolution for saved figure output (default: 100).

        Returns:
            None
        """
        super().__init__(figsize, dpi)
        
    def create_vertical_cross_section(self: 'MPASVerticalCrossSectionPlotter', 
                                      mpas_3d_processor: Any, 
                                      var_name: str, 
                                      start_point: Tuple[float, float], 
                                      end_point: Tuple[float, float], 
                                      time_index: int = 0, 
                                      vertical_coord: str = 'pressure', 
                                      display_vertical: Optional[str] = None, 
                                      num_points: int = 100, 
                                      levels: Optional[np.ndarray] = None, 
                                      colormap: Optional[Union[str, mcolors.Colormap]] = None, 
                                      extend: str = 'both', 
                                      plot_type: str = 'filled_contour', 
                                      save_path: Optional[str] = None, 
                                      title: Optional[str] = None, 
                                      max_height: Optional[float] = None, 
                                      **kwargs: Any) -> Tuple[Figure, Axes]:
        """
        This method creates a vertical cross-section plot for a specified 3D atmospheric variable along a user-defined transect between two geographic points. It first validates the input processor and variable, then generates the cross-section data by interpolating the 3D variable along the great-circle path defined by the start and end points. The method handles unit conversions, applies variable-specific styling, and renders the plot using the specified plot type (filled contour, contour, or pcolormesh). It formats the axes based on the vertical coordinate system, adds titles and colorbars, and includes options for saving the figure. The method returns the matplotlib figure and axes objects containing the generated cross-section plot for further customization or display. 

        Parameters:
            mpas_3d_processor (Any): MPAS3DProcessor instance with loaded dataset and processing capabilities.
            var_name (str): Name of the 3D variable to plot in the cross-section.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            time_index (int): Time index for selecting temporal data from the dataset (default: 0).
            vertical_coord (str): Type of vertical coordinate to use ('pressure', 'height', 'model_levels') for data extraction (default: 'pressure').
            display_vertical (Optional[str]): Desired vertical coordinate type for display ('pressure', 'height', 'model_levels'); if None, uses vertical_coord type (default: None).
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting (default: 100).
            levels (Optional[np.ndarray]): Array of contour levels to use for plotting; if None, levels are automatically determined based on data range and variable type (default: None).
            colormap (Optional[Union[str, mcolors.Colormap]]): Colormap to use for plotting; if None, a default colormap is selected based on variable type and styling rules (default: None).
            extend (str): Contour extend option for out-of-range values ('neither', 'both', 'min', 'max') when using filled contours (default: 'both').
            plot_type (str): Type of plot to create ('filled_contour', 'contour', 'pcolormesh') with different rendering styles for the cross-section visualization (default: 'filled_contour').
            save_path (Optional[str]): File path to save the generated figure; if None, the figure is not saved to disk (default: None).
            title (Optional[str]): Title for the plot; if None, a default title is generated based on variable name and time information (default: None).
            max_height (Optional[float]): Maximum height in kilometers to display on the vertical axis; if None, full height range is shown based on data and vertical coordinate system (default: None).
            **kwargs: Additional keyword arguments passed to the underlying matplotlib plotting functions for further customization. 

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects containing the generated vertical cross-section plot. 
        """
        if not isinstance(mpas_3d_processor, MPAS3DProcessor):
            raise ValueError("mpas_3d_processor must be an instance of MPAS3DProcessor")
            
        if mpas_3d_processor.dataset is None:
            raise ValueError("MPAS3DProcessor must have loaded data. Call load_3d_data() first.")
            
        if var_name not in mpas_3d_processor.dataset.data_vars:
            available_vars = list(mpas_3d_processor.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars[:10]}...")
            
        var_dims = mpas_3d_processor.dataset[var_name].sizes

        if 'nVertLevels' not in var_dims and 'nVertLevelsP1' not in var_dims:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")
            
        print(f"Creating vertical cross-section for {var_name}")
        print(f"Cross-section from ({start_point[0]:.2f}, {start_point[1]:.2f}) to ({end_point[0]:.2f}, {end_point[1]:.2f})")
        
        cross_section_data = self._generate_cross_section_data(
            mpas_3d_processor, var_name, start_point, end_point, 
            time_index, vertical_coord, num_points
        )
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)
        
        distances = cross_section_data['distances']
        longitudes = cross_section_data['longitudes']
        vertical_coords = cross_section_data['vertical_coords']
        vertical_coord_type = cross_section_data.get('vertical_coord_type', vertical_coord)
        data_values = cross_section_data['data_values']
        
        try:
            metadata = MPASFileMetadata.get_variable_metadata(var_name)
            original_units = metadata.get('units', '')
            display_units = UnitConverter.get_display_units(var_name, original_units)
            
            if original_units != display_units and original_units:
                print(f"Converting {var_name} from {original_units} to {display_units}")
                data_values = UnitConverter.convert_units(data_values, original_units, display_units)
                metadata['units'] = display_units
                print(f"Data range after conversion: {np.nanmin(data_values):.4f} to {np.nanmax(data_values):.4f} {display_units}")
            else:
                print(f"No unit conversion needed for {var_name} (units: {original_units})")
        except Exception as e:
            print(f"Warning: Unit conversion failed for {var_name}: {e}")
            metadata = {'units': '', 'long_name': var_name}
        
        # Specify the list of moisture-related variable names to check for physical constraints (e.g., non-negative values)
        moisture_vars = ['q2', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qv2m', 'humidity', 'mixing_ratio']

        # For moisture variables, check for negative values and clip to 0 if found, since negative moisture is physically invalid. 
        if any(mv in var_name.lower() for mv in moisture_vars):
            # Count the number of negative values in the data array to log a warning if any are found
            n_negative = np.sum(data_values < 0)

            # If negative values are found, log a warning with the count and minimum value, then clip the data to 0 to enforce physical constraints.
            if n_negative > 0:
                print(f"Warning: Found {n_negative:,} negative {var_name} values (min: {np.nanmin(data_values):.4f}). Clipping to 0 (physically invalid).")
                data_values = np.clip(data_values, 0, None)
        
        # Specify the desired vertical coordinate for display based on user input or automatic behavior. 
        desired_display = display_vertical if display_vertical is not None else vertical_coord_type

        if desired_display == 'height':
            vertical_display, vertical_coord_display = self._convert_vertical_to_height(
                vertical_coords, vertical_coord_type, mpas_3d_processor, time_index
            )
        elif desired_display == 'pressure':
            try:
                v = vertical_coords.astype(float).copy()
            except Exception:
                v = np.asarray(vertical_coords, dtype=float)

            if not np.all(np.isfinite(v)) or np.nanmin(v) <= 0:
                print("Warning: vertical coordinates contain non-positive or non-finite values; cannot display as pressure. Falling back to model levels.")
                vertical_display = np.arange(len(vertical_coords), dtype=float)
                vertical_coord_display = 'model_levels'
            else:
                is_pa = np.nanmax(v) > 10000
                if is_pa:
                    vertical_display = v / 100.0
                else:
                    vertical_display = v
                vertical_coord_display = 'pressure_hPa'
        elif desired_display == 'model_levels':
            vertical_display = vertical_coords
            vertical_coord_display = 'model_levels'
        else:
            vertical_display, vertical_coord_display = self._convert_vertical_to_height(
                vertical_coords, vertical_coord_type, mpas_3d_processor, time_index
            )
        
        if vertical_coord_display == 'height_km' and max_height is not None:
            try:
                mask = np.asarray(vertical_display) <= float(max_height)
                if np.any(mask):
                    vertical_display = np.asarray(vertical_display)[mask]
                    data_values = np.asarray(data_values)[mask, :]
                else:
                    print("Warning: No vertical levels are below the requested max_height; showing full range")
            except Exception:
                pass

        X, Y = np.meshgrid(longitudes, vertical_display)
        
        if colormap is None or levels is None:
            try:
                dummy_data = xr.DataArray(data_values, dims=['level', 'distance'], name=var_name)
                style = MPASVisualizationStyle.get_variable_style(var_name, dummy_data)
                
                if colormap is None:
                    colormap = style.get('colormap', 'viridis')
                if levels is None:
                    levels = style.get('levels', self._get_default_levels(data_values, var_name))
            except Exception:
                if colormap is None:
                    colormap = 'viridis'
                if levels is None:
                    levels = self._get_default_levels(data_values, var_name)
        
        if levels is None:
            levels = self._get_default_levels(data_values, var_name)
            
        if plot_type == 'filled_contour':
            cs = self.ax.contourf(X, Y, data_values, levels=levels, cmap=colormap, extend=extend, **kwargs)
            cs_lines = self.ax.contour(X, Y, data_values, levels=levels, colors='black', linewidths=0.5, alpha=0.6)
            self.ax.clabel(cs_lines, inline=True, fontsize=8, fmt='%.1f')
        elif plot_type == 'contour':
            cs = self.ax.contour(X, Y, data_values, levels=levels, cmap=colormap, **kwargs)
            self.ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        elif plot_type == 'pcolormesh':
            cs = self.ax.pcolormesh(X, Y, data_values, cmap=colormap, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'filled_contour', 'contour', or 'pcolormesh'")
            
        if plot_type in ['filled_contour', 'pcolormesh']:
            cbar = MPASVisualizationStyle.add_colorbar(
                self.fig, self.ax, cs,
                label=f"{metadata.get('long_name', var_name)} [{metadata.get('units', '')}]" if metadata.get('units','') else metadata.get('long_name', var_name),
                orientation='vertical', fraction=0.03, pad=0.05, shrink=0.8, fmt=None, labelpad=4, label_pos='right', tick_labelsize=10
            )
            
        self._format_cross_section_axes(longitudes, vertical_display, vertical_coord_display, 
                                      start_point, end_point, max_height)
        
        if title is None:
            try:
                time_str = self._get_time_string(mpas_3d_processor, time_index)
                title = f"Vertical Cross-Section: {var_name} | Valid Time: {time_str}"
            except Exception:
                title = f"Vertical Cross-Section: {var_name}"

        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)        
        self.ax.grid(True, alpha=0.3, linestyle='--')        
        plt.tight_layout()

        self.fig.subplots_adjust(bottom=0.09)
        self.add_timestamp_and_branding()
        
        if save_path:
            save_kwargs = {'dpi': self.dpi, 'bbox_inches': 'tight'}
            if save_path.lower().endswith('.png'):
                save_kwargs['pil_kwargs'] = {'compress_level': 1}
            self.fig.savefig(save_path, **save_kwargs)
            print(f"Vertical cross-section saved to: {save_path}")
            
        return self.fig, self.ax
        
    def _generate_cross_section_data(self: 'MPASVerticalCrossSectionPlotter', 
                                     mpas_3d_processor: MPAS3DProcessor, 
                                     var_name: str, 
                                     start_point: Tuple[float, float], 
                                     end_point: Tuple[float, float], 
                                     time_index: int, 
                                     vertical_coord: str, 
                                     num_points: int) -> Dict[str, Any]:
        """
        This internal method generates the cross-section data by interpolating the specified 3D variable along a great-circle path defined by the start and end points. It retrieves the vertical levels for the variable, extracts the corresponding data slices for each level, and performs spatial interpolation to obtain data values along the cross-section path. The method handles different vertical coordinate types, checks for valid data, and returns a dictionary containing the distances along the path, vertical coordinates, interpolated data values, and other relevant information for plotting. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance providing dataset access and processing capabilities.
            var_name (str): Name of the 3D variable to extract and interpolate for the cross-section.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            time_index (int): Time index for selecting temporal data from the dataset.
            vertical_coord (str): Type of vertical coordinate to use for data extraction ('pressure', 'height', 'model_levels').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.

        Returns:
            Dict[str, Any]: Dictionary containing 'distances' (array of distances along the path), 'vertical_coords' (array of vertical coordinate values), 'data_values' (2D array of interpolated data values along the cross-section), 'path_lons' (array of longitude coordinates along the path), 'path_lats' (array of latitude coordinates along the path), and 'vertical_coord_type' (string indicating the type of vertical coordinate used). 
        """
        path_lons, path_lats, distances = self._generate_great_circle_path(
            start_point, end_point, num_points
        )
        
        try:
            if vertical_coord == 'pressure':
                vertical_levels = mpas_3d_processor.get_vertical_levels(var_name, return_pressure=True, time_index=time_index)
            elif vertical_coord == 'model_levels':
                vertical_levels = mpas_3d_processor.get_vertical_levels(var_name, return_pressure=False, time_index=time_index)
            else:
                vertical_levels = mpas_3d_processor.get_vertical_levels(var_name, return_pressure=False, time_index=time_index)
                vertical_coord = 'model_levels'

            vertical_levels = np.array(vertical_levels)

            if np.issubdtype(vertical_levels.dtype, np.integer):
                vertical_coord = 'model_levels'
                if self.fig is not None and self.verbose:
                    print("Note: vertical levels appear to be integer indices; switching vertical_coord to 'model_levels'")
        except Exception as e:
            print(f"Warning: Could not get vertical levels, using indices: {e}")
            if 'nVertLevels' in mpas_3d_processor.dataset[var_name].sizes:
                n_levels = mpas_3d_processor.dataset.sizes['nVertLevels']
            elif 'nVertLevelsP1' in mpas_3d_processor.dataset[var_name].sizes:
                n_levels = mpas_3d_processor.dataset.sizes['nVertLevelsP1']
            else:
                n_levels = 10 
            vertical_levels = np.arange(n_levels)
            vertical_coord = 'model_levels'
            
        cross_section_data = np.full((len(vertical_levels), num_points), np.nan)
        
        try:
            var_da = mpas_3d_processor.dataset[var_name]
            lon_coords, lat_coords = mpas_3d_processor.extract_2d_coordinates_for_variable(var_name, var_da)
        except Exception:
            lon_coords = mpas_3d_processor.dataset['lonCell'].values
            lat_coords = mpas_3d_processor.dataset['latCell'].values
            
        if np.max(np.abs(lon_coords)) <= np.pi:
            lon_coords = np.degrees(lon_coords)
            lat_coords = np.degrees(lat_coords)
        
        print(f"Grid domain: lon [{np.min(lon_coords):.2f}, {np.max(lon_coords):.2f}], lat [{np.min(lat_coords):.2f}, {np.max(lat_coords):.2f}]")
        print(f"Cross-section path: ({path_lons[0]:.2f}, {path_lats[0]:.2f}) to ({path_lons[-1]:.2f}, {path_lats[-1]:.2f})")
        
        path_in_lon = (path_lons[0] >= np.min(lon_coords) and path_lons[-1] <= np.max(lon_coords))
        path_in_lat = (min(path_lats[0], path_lats[-1]) >= np.min(lat_coords) and max(path_lats[0], path_lats[-1]) <= np.max(lat_coords))

        if not (path_in_lon and path_in_lat):
            print("WARNING: Cross-section path extends outside grid domain!")
            print(f"  Longitude OK: {path_in_lon}, Latitude OK: {path_in_lat}")
        
        print(f"Interpolating {var_name} data along cross-section...")
        
        # Convert UxDataset to plain xr.Dataset for reliable isel operations.
        # UxDataset subclasses xr.Dataset but its UxDataArray.isel() silently
        # fails to reduce dimensions. xr.Dataset() strips the wrapper.
        ds = mpas_3d_processor.dataset

        if type(ds) is not xr.Dataset and isinstance(ds, xr.Dataset):
            ds = xr.Dataset(ds)
        
        var_da = ds[var_name]
        
        # Determine dimension names for time and vertical axes
        time_dim = 'Time' if 'Time' in var_da.sizes else ('time' if 'time' in var_da.sizes else None)
        
        if 'nVertLevels' in var_da.sizes:
            vert_dim = 'nVertLevels'
        elif 'nVertLevelsP1' in var_da.sizes:
            vert_dim = 'nVertLevelsP1'
        else:
            vert_dim = None
        
        for level_idx, level in enumerate(vertical_levels):
            try:
                # Build simultaneous isel dict to reduce all dimensions at once
                isel_dict = {}
                if time_dim is not None:
                    isel_dict[time_dim] = time_index
                if vert_dim is not None:
                    isel_dict[vert_dim] = level_idx
                else:
                    continue
                
                level_data = var_da.isel(isel_dict)
                
                if hasattr(level_data, 'compute'):
                    level_data = level_data.compute()
                
                if hasattr(level_data, 'values'):
                    data_values = level_data.values
                else:
                    data_values = np.asarray(level_data)
                
                # Validate that extraction produced 1D data matching coordinate size
                if data_values.ndim != 1 or data_values.shape[0] != lon_coords.shape[0]:
                    raise ValueError(
                        f"Level extraction produced shape {data_values.shape}, "
                        f"expected ({lon_coords.shape[0]},)"
                    )
                
                interpolated_values = self._interpolate_along_path(
                    lon_coords, lat_coords, data_values,
                    path_lons, path_lats
                )

                cross_section_data[level_idx, :] = interpolated_values
                
            except Exception as e:
                print(f"Warning: Could not extract data for level {level}: {e}")
                continue
        
        valid_data = ~np.isnan(cross_section_data)

        if np.any(valid_data):
            data_min, data_max = np.min(cross_section_data[valid_data]), np.max(cross_section_data[valid_data])
            print(f"Final cross-section data: {data_min:.3f} to {data_max:.3f} ({np.sum(valid_data)}/{cross_section_data.size} valid points)")
        else:
            print("WARNING: Final cross-section data contains NO valid values!")
                
        return {
            'distances': distances,
            'vertical_coords': np.array(vertical_levels),
            'data_values': cross_section_data,
            'path_lons': path_lons,
            'path_lats': path_lats,
            'longitudes': path_lons,  
            'vertical_coord_type': vertical_coord
        }
        
    def _generate_great_circle_path(self: 'MPASVerticalCrossSectionPlotter', 
                                    start_point: Tuple[float, float], 
                                    end_point: Tuple[float, float], 
                                    num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal method generates a great-circle path between two geographic points specified by their longitude and latitude coordinates. It calculates the intermediate points along the great-circle route, providing arrays of longitude, latitude, and distance along the path. The method uses spherical trigonometry to compute the path and handles edge cases such as very short distances. The output can be used for spatial interpolation of data along the cross-section path in the vertical cross-section plotting process. 

        Parameters:
            start_point (Tuple[float, float]): Starting point of the path as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the path as (longitude, latitude) in degrees.
            num_points (int): Number of interpolation points along the path for smooth plotting. 

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of longitude coordinates, latitude coordinates, and distances along the path (in kilometers) for each interpolation point. 
        """
        lon1, lat1 = radians(start_point[0]), radians(start_point[1])
        lon2, lat2 = radians(end_point[0]), radians(end_point[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        total_distance = 2 * asin(sqrt(a)) * 6371.0  
        
        fractions = np.linspace(0, 1, num_points)
        lons = np.zeros(num_points)
        lats = np.zeros(num_points)
        distances = np.zeros(num_points)
        
        if total_distance < 1e-6:  
            lons.fill(start_point[0])
            lats.fill(start_point[1])
            distances.fill(0)
            return lons, lats, distances
        
        for i, f in enumerate(fractions):
            if f == 0:
                lons[i] = start_point[0]
                lats[i] = start_point[1]
                distances[i] = 0
            elif f == 1:
                lons[i] = end_point[0]
                lats[i] = end_point[1]
                distances[i] = total_distance
            else:
                angular_distance = total_distance / 6371.0  
                A = sin((1-f) * angular_distance) / sin(angular_distance)
                B = sin(f * angular_distance) / sin(angular_distance)
                
                x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
                y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
                z = A * sin(lat1) + B * sin(lat2)
                
                lats[i] = degrees(atan2(z, sqrt(x**2 + y**2)))
                lons[i] = degrees(atan2(y, x))
                distances[i] = f * total_distance
            
        return lons, lats, distances
        
    def _interpolate_along_path(self: 'MPASVerticalCrossSectionPlotter',
                                 grid_lons: np.ndarray, 
                                 grid_lats: np.ndarray, 
                                 grid_data: Union[np.ndarray, xr.DataArray, Any], 
                                 path_lons: np.ndarray, 
                                 path_lats: np.ndarray) -> np.ndarray:
        """
        This internal method performs spatial interpolation of grid data values along a specified path defined by longitude and latitude coordinates. It uses a KDTree for efficient nearest-neighbor interpolation, handling cases where the grid data may contain NaN values by masking them out during the interpolation process. The method converts geographic coordinates to Cartesian coordinates for distance calculations, queries the KDTree to find the nearest grid points for each path point, and returns an array of interpolated data values along the path, with NaN values where no valid data is found. This interpolation is crucial for generating accurate cross-section data along the user-defined transect in the vertical cross-section plotting process. 

        Parameters:
            grid_lons (np.ndarray): 1D array of longitude coordinates for the grid points.
            grid_lats (np.ndarray): 1D array of latitude coordinates for the grid points.
            grid_data (Union[np.ndarray, xr.DataArray, Any]): 1D array of data values corresponding to the grid points, which may contain NaN values.
            path_lons (np.ndarray): 1D array of longitude coordinates for the path points along the cross-section.
            path_lats (np.ndarray): 1D array of latitude coordinates for the path points along the cross-section. 

        Returns:
            np.ndarray: 1D array of interpolated data values along the path, with NaN values where no valid data is found. 
        """
        if isinstance(grid_data, xr.DataArray):
            grid_data = grid_data.values
        elif not isinstance(grid_data, np.ndarray):
            grid_data = np.asarray(grid_data)
        
        grid_lons_flat = grid_lons.ravel()
        grid_lats_flat = grid_lats.ravel()
        grid_data_flat = grid_data.ravel()
        
        valid_mask = ~np.isnan(grid_data_flat)
        grid_lons_valid = grid_lons_flat[valid_mask]
        grid_lats_valid = grid_lats_flat[valid_mask]
        grid_data_valid = grid_data_flat[valid_mask]
        
        if len(grid_data_valid) == 0:
            return np.full(len(path_lons), np.nan)
            
        grid_points = np.column_stack([
            np.cos(np.radians(grid_lats_valid)) * np.cos(np.radians(grid_lons_valid)),
            np.cos(np.radians(grid_lats_valid)) * np.sin(np.radians(grid_lons_valid)),
            np.sin(np.radians(grid_lats_valid))
        ])
        
        path_points = np.column_stack([
            np.cos(np.radians(path_lats)) * np.cos(np.radians(path_lons)),
            np.cos(np.radians(path_lats)) * np.sin(np.radians(path_lons)),
            np.sin(np.radians(path_lats))
        ])
        
        tree = KDTree(grid_points)
        distances, indices = tree.query(path_points)
        
        return grid_data_valid[indices]
        
    def _get_default_levels(self: 'MPASVerticalCrossSectionPlotter', 
                            data_values: Union[np.ndarray, xr.DataArray, Any], 
                            var_name: str) -> np.ndarray:
        """
        This internal method determines default contour levels for plotting based on the data range and variable type. It handles different variable types (e.g., temperature, pressure, wind) by applying specific rules for level spacing and range. The method first checks for valid data values, computes the minimum and maximum, and then applies variable-specific logic to generate an array of contour levels that are optimized for the variable's typical range and variability. This ensures that the resulting cross-section plot has meaningful contours that enhance the visualization of the atmospheric variable along the transect. 

        Parameters:
            data_values (Union[np.ndarray, xr.DataArray, Any]): Array of data values for the variable being plotted, which may contain NaN values.
            var_name (str): Name of the variable being plotted, used to determine appropriate level spacing and range based on typical variable characteristics.

        Returns:
            np.ndarray: Array of contour levels determined based on the data range and variable type, optimized for visualization in the cross-section plot. 
        """
        if isinstance(data_values, xr.DataArray):
            data_values = data_values.values
        elif not isinstance(data_values, np.ndarray):
            data_values = np.asarray(data_values)
        
        valid_data = data_values[~np.isnan(data_values)]
        
        if len(valid_data) == 0:
            return np.linspace(0, 1, 11)
            
        data_min, data_max = valid_data.min(), valid_data.max()
        data_range = data_max - data_min
        
        if data_range == 0:
            return np.array([data_min])
            
        var_lower = var_name.lower()

        if 'temperature' in var_lower or 'temp' in var_lower:
            if data_range > 50: 
                levels = np.arange(data_min, data_max + 5, 5)
            else: 
                levels = np.arange(data_min, data_max + 2, 2)
        elif 'pressure' in var_lower:
            if data_min > 0:
                levels = np.logspace(np.log10(data_min), np.log10(data_max), 15)
            else:
                levels = np.linspace(data_min, data_max, 15)
        elif 'wind' in var_lower or var_name.startswith('u') or var_name.startswith('v'):
            max_abs = max(abs(data_min), abs(data_max))
            if data_min < 0 and data_max > 0:
                levels = np.linspace(-max_abs, max_abs, 21)
            else:
                levels = np.linspace(data_min, data_max, 15)
        else:
            levels = np.linspace(data_min, data_max, 15)
            
        return levels
    
    def _extract_height_from_dataset(self: 'MPASVerticalCrossSectionPlotter', 
                                     mpas_3d_processor: MPAS3DProcessor, 
                                     time_index: int, 
                                     vertical_coords: np.ndarray, 
                                     var_name: str) -> Optional[np.ndarray]:
        """
        This internal method attempts to extract geometric height data from the MPAS dataset for the specified variable name (e.g., 'zgrid' or 'height') at the given time index. It checks if the variable exists in the dataset, retrieves the height data for the first cell (assuming it's representative of the vertical structure), and processes it to match the expected length based on the vertical coordinates. If the height data has one more level than the vertical coordinates, it computes mid-level heights by averaging adjacent levels. If it matches the length of vertical coordinates, it returns it directly. If neither condition is met, it attempts to interpolate the height data to match the vertical coordinates. The method includes error handling to return None if extraction fails or if the variable is unavailable, allowing for fallback methods to determine height when necessary. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance providing access to the dataset for height variable extraction.
            time_index (int): Time index for selecting the appropriate temporal slice of the height variable.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the cross-section, which determines the expected length of the height data.
            var_name (str): Name of the height variable to extract from the dataset (e.g., 'zgrid' or 'height'). 

        Returns:
            Optional[np.ndarray]: Array of geometric height values in meters corresponding to the vertical coordinates, or None if extraction fails or variable is unavailable. 
        """
        try:
            if var_name not in mpas_3d_processor.dataset.data_vars:
                return None
                
            height_data = mpas_3d_processor.dataset[var_name].isel(Time=time_index, nCells=0).values
            height_data = np.asarray(height_data, dtype=float)
            
            if len(height_data) == len(vertical_coords) + 1:
                mid_heights = 0.5 * (height_data[:-1] + height_data[1:])
                return mid_heights
            elif len(height_data) == len(vertical_coords):
                return height_data
            else:
                try:
                    from scipy.interpolate import interp1d
                    xp = np.linspace(0, 1, len(height_data))
                    fp = height_data
                    f = interp1d(xp, fp, bounds_error=False, fill_value=cast(Any, 'extrapolate'))
                    xq = np.linspace(0, 1, len(vertical_coords))
                    return f(xq)
                except Exception:
                    return None
        except Exception:
            return None
    
    def _convert_vertical_to_height(self: 'MPASVerticalCrossSectionPlotter', 
                                    vertical_coords: np.ndarray, 
                                    vertical_coord_type: str, 
                                    mpas_3d_processor: MPAS3DProcessor, 
                                    time_index: int) -> Tuple[np.ndarray, str]:
        """
        This internal method converts vertical coordinate values to geometric height in kilometers based on the specified vertical coordinate type. If the input coordinate type is 'height', it simply converts from meters to kilometers. If the input type is 'pressure', it first attempts to extract geometric height from the dataset using known variable names ('zgrid' or 'height'). If successful, it converts to kilometers and returns. If extraction fails, it applies a standard atmosphere approximation to convert pressure to height, while handling potential issues with non-positive or non-finite pressure values. For 'model_levels', it also tries to extract geometric height if available, otherwise it returns the original model level indices. The method ensures that the returned vertical coordinates are in a consistent format for plotting and includes error handling for various edge cases. 

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values to convert.
            vertical_coord_type (str): Type of vertical coordinate ('height', 'pressure', 'model_levels') indicating how to interpret the input coordinates.
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance for accessing the dataset to extract height information if needed.
            time_index (int): Time index for selecting the appropriate temporal slice when extracting height from the dataset. 

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the converted vertical coordinates in kilometers and a string indicating the type of vertical coordinate used for display (e.g., 'height_km', 'pressure_hPa', or 'model_levels'). 
        """
        if vertical_coord_type == 'height':
            return vertical_coords / 1000.0, 'height_km'
        elif vertical_coord_type == 'pressure':
            try:
                height_m = self._extract_height_from_dataset(mpas_3d_processor, time_index, vertical_coords, 'zgrid')

                if height_m is not None:
                    return height_m / 1000.0, 'height_km'
                
                height_m = self._extract_height_from_dataset(mpas_3d_processor, time_index, vertical_coords, 'height')

                if height_m is not None:
                    return height_m / 1000.0, 'height_km'
            except Exception:
                pass

            # Approximate height from pressure using standard atmosphere
            # h = -H * ln(P/P0) where H ≈ 8.4 km, P0 = 101325 Pa
            try:
                pressure_pa = vertical_coords.astype(float).copy()

                if np.nanmax(pressure_pa) < 10000:  # Likely in hPa or indices
                    pressure_pa = pressure_pa * 100.0

                min_positive = 1.0
                if np.any(pressure_pa <= 0) or np.any(~np.isfinite(pressure_pa)):
                    if self.verbose:
                        print("Warning: pressure levels contained non-positive or non-finite values; clipping to minimum positive value to avoid log(0)")
                    pressure_pa = np.where(np.isfinite(pressure_pa), pressure_pa, min_positive)
                    pressure_pa = np.clip(pressure_pa, min_positive, None)

                # Standard atmosphere approximation
                H = 8.4  # Scale height in km
                P0 = 101325.0  # Sea level pressure in Pa
                height_km = -H * np.log(pressure_pa / P0)

                # Clip negative heights to 0
                height_km = np.maximum(height_km, 0.0)

                return height_km, 'height_km'
            except Exception:
                return vertical_coords / 100.0, 'pressure_hPa'  # Convert Pa to hPa
        else:  # model_levels
            # For model levels, try to get geometric height if available
            try:
                height_m = self._extract_height_from_dataset(mpas_3d_processor, time_index, vertical_coords, 'zgrid')

                if height_m is not None:
                    return height_m / 1000.0, 'height_km'
                
                height_m = self._extract_height_from_dataset(mpas_3d_processor, time_index, vertical_coords, 'height')

                if height_m is not None:
                    return height_m / 1000.0, 'height_km'
                
                return vertical_coords, 'model_levels'
            except Exception:
                return vertical_coords, 'model_levels'
    
    def _setup_pressure_axis(self: 'MPASVerticalCrossSectionPlotter', 
                             vertical_coords: np.ndarray, 
                             use_standard_ticks: bool = True) -> None:
        """
        This internal method configures the y-axis of the vertical cross-section plot for pressure coordinates by setting a logarithmic scale and applying meteorological standard pressure levels as major tick marks. It first checks if the minimum pressure coordinate value is positive to ensure that a logarithmic scale is appropriate. If so, it sets the y-axis to a logarithmic scale and applies standard pressure levels (e.g., 1000 hPa, 850 hPa, etc.) as major ticks if the flag is enabled. The method includes error handling to gracefully fall back to a linear scale if the pressure values are not suitable for logarithmic scaling, providing informative warnings when necessary. This setup enhances the readability of pressure-based cross-sections by using a scale and tick marks that are familiar to meteorologists. 

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values representing pressure levels, used to determine appropriate scaling and tick placement for the y-axis.
            use_standard_ticks (bool): Flag indicating whether to apply standard meteorological pressure levels as major ticks on the y-axis when using a logarithmic scale (default: True). 

        Returns:
            None: Modifies self.ax y-axis properties directly without returning a value. 
        """
        assert self.ax is not None, "Axes must be created before setup"
        
        try:
            vmin = np.nanmin(vertical_coords)
            if vmin > 0:
                self.ax.set_yscale('log')
                if use_standard_ticks:
                    try:
                        from matplotlib.ticker import FixedLocator, FuncFormatter
                        standard_ticks = [1000, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 1]
                        data_min = np.nanmin(vertical_coords)
                        data_max = np.nanmax(vertical_coords)
                        tick_vals = [t for t in standard_ticks if (t >= data_min and t <= data_max)]
                        if len(tick_vals) >= 2:
                            self.ax.yaxis.set_major_locator(FixedLocator(tick_vals))
                            def _fmt(x, pos):
                                if x >= 1:
                                    return f"{int(x):d}"
                                return f"{x:.2f}"
                            self.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
                    except Exception:
                        pass
            else:
                print("Warning: detected non-positive pressure coordinate values; using linear y-scale for pressure display")
        except Exception:
            print("Warning: could not determine pressure coordinate min; using linear y-scale")
        
    def _format_cross_section_axes(self: 'MPASVerticalCrossSectionPlotter', 
                                   longitudes: np.ndarray, 
                                   vertical_coords: np.ndarray, 
                                   vertical_coord_type: str, 
                                   start_point: Tuple[float, float], 
                                   end_point: Tuple[float, float], 
                                   max_height: Optional[float] = None) -> None:
        """
        This internal method formats the axes of the vertical cross-section plot by setting appropriate labels, limits, and tick formatting based on the longitude range and vertical coordinate type. It configures the x-axis to display longitude values with degree symbols and sets limits based on the provided longitude array. For the y-axis, it applies different formatting depending on whether the vertical coordinate is height (in km), pressure (in hPa or Pa), or model levels. It also adds a text box to indicate the start and end points of the cross-section path. The method includes error handling to ensure that axis formatting does not fail even if certain styling features are unavailable, providing a robust setup for clear visualization of the cross-section data. 

        Parameters:
            longitudes (np.ndarray): Array of longitude values along the cross-section path, used to set x-axis limits and formatting.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the y-axis, with formatting determined by the vertical_coord_type.
            vertical_coord_type (str): Type of vertical coordinate ('height_km', 'pressure_hPa', 'pressure', 'height', 'model_levels') that dictates y-axis labeling and limits.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees, used for annotating the plot.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees, used for annotating the plot.
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data. 

        Returns:
            None: Modifies self.ax properties directly without returning a value. 
        """
        assert self.ax is not None, "Axes must be created before formatting"
        
        self.ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
        self.ax.set_xlim(longitudes.min(), longitudes.max())
        
        try:
            from matplotlib.ticker import FuncFormatter
            lon_formatter = FuncFormatter(MPASVisualizationStyle.format_longitude)
            self.ax.xaxis.set_major_formatter(lon_formatter)
        except Exception:
            pass  # Fall back to default formatting if styling fails
        
        if vertical_coord_type == 'height_km':
            self.ax.set_ylabel('Height [km]', fontsize=12)
            y_max = max_height if max_height is not None else vertical_coords.max()
            self.ax.set_ylim(0, y_max)  # Start from bottom (0 km)
        elif vertical_coord_type == 'pressure_hPa':
            self.ax.set_ylabel('Pressure [hPa]', fontsize=12)
            if max_height is not None:
                # Convert max_height to pressure using standard atmosphere
                # P = P0 * exp(-h/H) where H ≈ 8.4 km
                P0 = 1013.25  # Sea level pressure in hPa
                H = 8.4  # Scale height in km
                min_pressure = P0 * np.exp(-max_height / H)
                valid_coords = vertical_coords[vertical_coords >= min_pressure]
                if len(valid_coords) > 0:
                    self.ax.set_ylim(valid_coords.max(), min_pressure)
                else:
                    self.ax.set_ylim(vertical_coords.max(), vertical_coords.min())
            else:
                self.ax.set_ylim(vertical_coords.max(), vertical_coords.min()) 
            self._setup_pressure_axis(vertical_coords, use_standard_ticks=True)
        elif vertical_coord_type == 'pressure':
            self.ax.set_ylabel('Pressure [Pa]', fontsize=12)
            if max_height is not None:
                P0 = 101325  # Sea level pressure in Pa
                H = 8.4  # Scale height in km
                min_pressure = P0 * np.exp(-max_height / H)
                valid_coords = vertical_coords[vertical_coords >= min_pressure]
                if len(valid_coords) > 0:
                    self.ax.set_ylim(valid_coords.max(), min_pressure)
                else:
                    self.ax.set_ylim(vertical_coords.max(), vertical_coords.min())
            else:
                self.ax.set_ylim(vertical_coords.max(), vertical_coords.min()) 
            self._setup_pressure_axis(vertical_coords, use_standard_ticks=False)
        elif vertical_coord_type == 'height':
            self.ax.set_ylabel('Height [m]', fontsize=12)
            y_max = max_height * 1000 if max_height is not None else vertical_coords.max() 
            self.ax.set_ylim(vertical_coords.min(), y_max)
        else: 
            self.ax.set_ylabel('Model Level', fontsize=12)
            try:
                self.ax.set_ylim(vertical_coords.min(), vertical_coords.max())
            except Exception:
                self.ax.set_ylim(vertical_coords.max(), vertical_coords.min())
            
        path_info = f"From ({start_point[0]:.1f}°, {start_point[1]:.1f}°) to ({end_point[0]:.1f}°, {end_point[1]:.1f}°)"
        self.ax.text(0.02, 0.98, path_info, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _get_time_string(self: 'MPASVerticalCrossSectionPlotter', 
                         mpas_3d_processor: MPAS3DProcessor, 
                         time_index: int) -> str:
        """
        This internal method retrieves a formatted time string for the given time index from the MPAS3DProcessor instance. It first checks if the processor has a method to get time information directly; if so, it uses that method. If not, it attempts to access a 'Time' variable in the dataset and convert it to a datetime object for formatting. If both methods fail, it falls back to returning a simple string with the time index. This approach ensures that the plot can display meaningful time information when available while maintaining robustness in cases where time data is not accessible or formatted differently. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance that may contain methods or variables for retrieving time information.
            time_index (int): Time index for which to retrieve the formatted time string. 

        Returns:
            str: Formatted time string for display in the plot title or annotations, or a fallback string with the time index if specific time information is not available. 
        """
        try:
            if hasattr(mpas_3d_processor, 'get_time_info'):
                return mpas_3d_processor.get_time_info(time_index)
            elif hasattr(mpas_3d_processor.dataset, 'Time') and len(mpas_3d_processor.dataset.Time) > time_index:
                time_value = pd.to_datetime(mpas_3d_processor.dataset.Time.values[time_index])
                return time_value.strftime('Valid: %Y-%m-%d %H:%M UTC')
            else:
                return f"Time Index: {time_index}"
        except Exception:
            return f"Time Index: {time_index}"
    
    def create_batch_cross_section_plots(self: 'MPASVerticalCrossSectionPlotter', 
                                         mpas_3d_processor: MPAS3DProcessor, 
                                         output_dir: str, 
                                         var_name: str, 
                                         start_point: Tuple[float, float], 
                                         end_point: Tuple[float, float], 
                                         vertical_coord: str = 'pressure', 
                                         num_points: int = 100, 
                                         levels: Optional[np.ndarray] = None, 
                                         colormap: Optional[Union[str, mcolors.Colormap]] = None, 
                                         extend: str = 'both', 
                                         plot_type: str = 'filled_contour', 
                                         max_height: Optional[float] = None, 
                                         file_prefix: str = 'mpas_crosssection', 
                                         formats: List[str] = ['png']) -> List[str]:
        """
        This method creates vertical cross-section plots for a specified 3D atmospheric variable across all available time steps in the MPAS dataset. It generates a plot for each time step, saving them to the specified output directory with filenames that include the variable name, vertical coordinate type, and valid time information. The method handles various vertical coordinate systems (pressure, height, model levels) and allows for customization of contour levels, colormap, plot type, and maximum height. It includes error handling to ensure that issues with data loading or variable availability are reported clearly, and it provides progress updates during batch processing. The resulting list of created file paths is returned upon completion. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): An instance of MPAS3DProcessor with loaded 3D data for plotting.
            output_dir (str): Directory where the generated plots will be saved.
            var_name (str): Name of the variable to plot from the dataset.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            vertical_coord (str): Type of vertical coordinate to use ('pressure', 'height', 'model_levels').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.
            levels (Optional[np.ndarray]): Optional array of contour levels; if None, levels are determined automatically.
            colormap (Optional[Union[str, mcolors.Colormap]]): Colormap to use for filled contour plots; can be a string name or a Matplotlib colormap object.
            extend (str): Direction to extend the colormap for out-of-range values ('neither', 'both', 'min', 'max').
            plot_type (str): Type of plot to create ('filled_contour' or 'contour_lines').
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data.
            file_prefix (str): Prefix for the output filenames; additional information will be appended to this prefix.
            formats (List[str]): List of file formats to save the plots in (e.g., ['png', 'pdf']). 

        Returns:
            List[str]: A list of file paths for the created cross-section plots in the specified formats.  
        """
        if not isinstance(mpas_3d_processor, MPAS3DProcessor):
            raise ValueError("mpas_3d_processor must be an instance of MPAS3DProcessor")
            
        if mpas_3d_processor.dataset is None:
            raise ValueError("MPAS3DProcessor must have loaded data. Call load_3d_data() first.")
            
        if var_name not in mpas_3d_processor.dataset.data_vars:
            available_vars = list(mpas_3d_processor.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars[:10]}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        time_dim = 'Time' if 'Time' in mpas_3d_processor.dataset.sizes else 'time'
        total_times = mpas_3d_processor.dataset.sizes[time_dim]
        
        created_files = []
        print(f"\nCreating vertical cross-section plots for {total_times} time steps...")
        print(f"Variable: {var_name}")
        print(f"Cross-section from ({start_point[0]:.2f}, {start_point[1]:.2f}) to ({end_point[0]:.2f}, {end_point[1]:.2f})")
        print(f"Vertical coordinate: {vertical_coord}")
        if max_height:
            print(f"Maximum height: {max_height} km")
        print()
        
        for time_idx in range(total_times):
            try:
                if hasattr(mpas_3d_processor.dataset, 'Time') and len(mpas_3d_processor.dataset.Time) > time_idx:
                    time_value = pd.to_datetime(mpas_3d_processor.dataset.Time.values[time_idx])
                    time_str = time_value.strftime('%Y%m%dT%H')
                else:
                    time_str = f"t{time_idx:03d}"
                
                path_str = f"({start_point[0]:.1f}°, {start_point[1]:.1f}°) to ({end_point[0]:.1f}°, {end_point[1]:.1f}°)"
                title = f"Vertical Cross-Section: {var_name} | Valid Time: {time_str}\nPath: {path_str}"
                
                fig, ax = self.create_vertical_cross_section(
                    mpas_3d_processor=mpas_3d_processor,
                    var_name=var_name,
                    start_point=start_point,
                    end_point=end_point,
                    time_index=time_idx,
                    vertical_coord=vertical_coord,
                    num_points=num_points,
                    levels=levels,
                    colormap=colormap,
                    extend=extend,
                    plot_type=plot_type,
                    max_height=max_height,
                    title=title
                )
                
                height_suffix = f"_maxh{int(max_height)}km" if max_height else ""
                output_path = os.path.join(
                    output_dir, 
                    f"{file_prefix}_vcrd_{vertical_coord}_valid_{time_str}{height_suffix}"
                )
                
                self.save_plot(output_path, formats=formats)
                
                for fmt in formats:
                    created_files.append(f"{output_path}.{fmt}")
                
                self.close_plot()
                
                if (time_idx + 1) % 5 == 0 or time_idx == 0:
                    print(f"Completed {time_idx + 1}/{total_times} cross-sections (time index {time_idx})...")
                    
            except Exception as e:
                print(f"Error creating cross-section for time index {time_idx}: {e}")
                continue
        
        print(f"\nBatch cross-section processing completed. Created {len(created_files)} files.")
        return created_files