#!/usr/bin/env python3

"""
MPAS Vertical Cross-Section Visualization

This module provides specialized functionality for creating vertical cross-sections of 3D MPAS atmospheric data along user-defined transects through the atmosphere showing vertical structure and distribution of meteorological variables. It includes the MPASVerticalCrossSectionPlotter class that generates interpolated 2D vertical slices from 3D model output along great-circle paths between geographic coordinates, supporting multiple vertical coordinate systems (pressure levels, geometric height, model native levels), automatic variable-specific styling and colormaps, and professional contour rendering with filled contours or line contours. The plotter handles automatic data extraction and interpolation along paths using KDTree spatial search, converts vertical coordinates to height when possible using standard atmosphere approximations, applies unit conversion for display, and produces publication-quality cross-section plots with formatted axes, colorbars, and annotations. Core capabilities include great-circle path generation with spherical geometry, level-by-level data interpolation, coordinate transformation, and batch processing for creating time series of cross-section analyses suitable for mesoscale weather analysis and model evaluation.

Classes:
    MPASVerticalCrossSectionPlotter: Specialized class for creating vertical cross-section visualizations of 3D MPAS atmospheric data.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

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
    """
    Specialized plotter for creating vertical cross-section visualizations of 3D MPAS atmospheric data along user-defined transects through the atmosphere. This class extends MPASVisualizer to provide comprehensive functionality for extracting, interpolating, and rendering 2D vertical slices from 3D model output, showing the vertical structure and distribution of atmospheric variables like temperature, winds, moisture, and reflectivity along great-circle paths between specified geographic coordinates. The plotter supports multiple vertical coordinate systems (pressure levels, geometric height, model native levels), automatic variable-specific styling and colormaps, professional contour rendering with filled contours or line contours, and seamless integration with MPAS3DProcessor for data loading and manipulation. Cross-section plots include formatted axes with height/pressure labels, colorbars with unit information, optional transect path annotation, and timestamp/branding for publication-quality output suitable for mesoscale weather analysis and model evaluation.
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 12),
        dpi: int = 100
    ) -> None:
        """
        Initializes the vertical cross-section plotter with figure dimensions and resolution settings for creating 2D slices through 3D atmospheric data. This constructor inherits from MPASVisualizer to establish the base plotting framework and configures default figure size optimized for vertical cross-section aspect ratios (taller than wide) and high-resolution output suitable for publication and detailed analysis of atmospheric vertical structure.

        Parameters:
            figsize (Tuple[float, float]): Figure dimensions in inches as (width, height) (default: (10, 12)).
            dpi (int): Figure resolution in dots per inch for output quality (default: 300).

        Returns:
            None: Initializes instance attributes through parent class constructor.
        """
        super().__init__(figsize, dpi)
        
    def create_vertical_cross_section(
        self,
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
        **kwargs: Any
    ) -> Tuple[Figure, Axes]:
        """
        Creates a vertical cross-section plot showing the vertical structure of a 3D atmospheric variable along a great-circle transect between two geographic points. This method generates interpolated cross-section data along the specified path, converts vertical coordinates to height when possible, applies automatic unit conversion and variable-specific styling, and produces filled contour or contour plots with professional formatting including colorbar, gridlines, title with statistics, and optional path annotation. It returns a complete figure ready for display or saving.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance with loaded 3D dataset.
            var_name (str): Name of 3D atmospheric variable to plot (must have vertical dimension).
            start_point (Tuple[float, float]): Starting point as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point as (longitude, latitude) in degrees.
            time_index (int): Time index for data extraction from dataset (default: 0).
            vertical_coord (str): Vertical coordinate system 'pressure', 'height', or 'model_levels' (default: 'pressure').
            display_vertical (str): Optional display vertical coordinate. If provided, overrides automatic
                conversion and forces the y-axis to display as 'pressure', 'height', or 'model_levels'.
                If None (default), existing automatic behavior is preserved (pressure -> height conversion).
            num_points (int): Number of interpolation points along cross-section path (default: 100).
            levels (Optional[np.ndarray]): Explicit contour levels (default: auto-generated from data).
            colormap (Optional[Union[str, mcolors.Colormap]]): Colormap name or object (default: variable-specific).
            extend (str): Colorbar extension mode 'both', 'min', 'max', 'neither' (default: 'both').
            plot_type (str): Plot rendering type 'filled_contour', 'contour', 'pcolormesh' (default: 'filled_contour').
            save_path (Optional[str]): File path for saving figure (default: None).
            title (Optional[str]): Custom plot title (default: auto-generated with time and variable).
            max_height (Optional[float]): Maximum vertical extent in km for y-axis (default: auto from data).
            **kwargs: Additional keyword arguments passed to matplotlib plotting functions.

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes containing the cross-section plot.

        Raises:
            ValueError: If processor is invalid, data not loaded, variable not found, or not 3D variable.
            RuntimeError: If cross-section data generation or plotting fails.
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
                vertical_display = vertical_coords
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
            cbar = plt.colorbar(cs, ax=self.ax, orientation='vertical', pad=0.05, shrink=0.8)
            
            try:
                units = metadata.get('units', '')
                long_name = metadata.get('long_name', var_name)
                cbar_label = f"{long_name} [{units}]" if units else long_name
            except Exception:
                cbar_label = var_name
                
            cbar.set_label(cbar_label, fontsize=12)
            
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
        
    def _generate_cross_section_data(
        self,
        mpas_3d_processor: MPAS3DProcessor,
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        time_index: int,
        vertical_coord: str,
        num_points: int
    ) -> Dict[str, Any]:
        """
        Generates cross-section data by interpolating 3D atmospheric data along a great-circle path between start and end points. This internal method creates a transect path, extracts vertical levels from the dataset (pressure, height, or model levels), iterates through each vertical level to extract and interpolate horizontal data onto the path using nearest-neighbor KDTree interpolation, and assembles a complete 2D cross-section array. It returns a dictionary containing distances, vertical coordinates, interpolated data values, and path coordinates for plotting.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance with loaded dataset.
            var_name (str): 3D variable name to extract.
            start_point (Tuple[float, float]): Cross-section start point as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Cross-section end point as (longitude, latitude) in degrees.
            time_index (int): Time index for temporal data selection.
            vertical_coord (str): Vertical coordinate type ('pressure', 'height', 'model_levels').
            num_points (int): Number of interpolation points along the cross-section path.

        Returns:
            Dict[str, Any]: Dictionary with keys 'distances' (km), 'vertical_coords', 'data_values' (2D array), 'path_lons', 'path_lats', 'longitudes', 'vertical_coord_type'.
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
        
        for level_idx, level in enumerate(vertical_levels):
            try:
                if hasattr(mpas_3d_processor, 'get_3d_variable_data'):
                    level_data = mpas_3d_processor.get_3d_variable_data(var_name, level_idx, time_index)
                else:
                    var_data = mpas_3d_processor.dataset[var_name]
                    if 'Time' in var_data.sizes:
                        var_data = var_data.isel(Time=time_index)
                    
                    if 'nVertLevels' in var_data.sizes:
                        level_data = var_data.isel(nVertLevels=level_idx)
                    elif 'nVertLevelsP1' in var_data.sizes:
                        level_data = var_data.isel(nVertLevelsP1=level_idx)
                    else:
                        continue
                
                if hasattr(level_data, 'values'):
                    data_values = level_data.values
                else:
                    data_values = level_data
                
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
        
    def _generate_great_circle_path(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a great-circle path between two geographic points using spherical geometry and haversine distance calculation. This internal method computes the shortest path on a sphere between start and end points, calculates total distance using the haversine formula with Earth radius of 6371 km, performs spherical linear interpolation (slerp) to generate evenly spaced points along the great circle, and returns arrays of longitudes, latitudes, and cumulative distances suitable for cross-section path definition.

        Parameters:
            start_point (Tuple[float, float]): Starting point as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point as (longitude, latitude) in degrees.
            num_points (int): Number of evenly-spaced points to generate along the path.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Longitude array (degrees), latitude array (degrees), and distance array (km) along the path.
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
        
    def _interpolate_along_path(
        self,
        grid_lons: np.ndarray,
        grid_lats: np.ndarray,
        grid_data: Union[np.ndarray, xr.DataArray, Any],
        path_lons: np.ndarray,
        path_lats: np.ndarray
    ) -> np.ndarray:
        """
        Interpolates unstructured grid data along a specified path using nearest-neighbor spatial search with KDTree acceleration. This internal method converts geographic coordinates to 3D Cartesian coordinates for accurate spherical distance calculations, filters out NaN values from the source grid data, builds a KDTree spatial index for efficient nearest-neighbor queries, and interpolates values at each path point by finding the closest valid grid cell. It returns an array of interpolated values matching the path length.

        Parameters:
            grid_lons (np.ndarray): Unstructured grid longitude coordinates in degrees (1D flattened).
            grid_lats (np.ndarray): Unstructured grid latitude coordinates in degrees (1D flattened).
            grid_data (Union[np.ndarray, xr.DataArray, Any]): Grid data values corresponding to lon/lat coordinates.
            path_lons (np.ndarray): Interpolation path longitude coordinates in degrees.
            path_lats (np.ndarray): Interpolation path latitude coordinates in degrees.

        Returns:
            np.ndarray: Interpolated data values along the path (NaN where no valid data found).
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
        
    def _get_default_levels(
        self,
        data_values: Union[np.ndarray, xr.DataArray, Any],
        var_name: str
    ) -> np.ndarray:
        """
        Generates default contour levels for a variable based on data range and variable type with context-aware spacing strategies. This internal method analyzes the data distribution excluding NaN values, applies variable-specific level generation rules (temperature uses fixed intervals, pressure uses logarithmic spacing, wind uses symmetric levels around zero), and returns an array of contour levels optimized for the specific atmospheric variable. It provides reasonable defaults when explicit levels are not specified.

        Parameters:
            data_values (Union[np.ndarray, xr.DataArray, Any]): Data values for level range determination.
            var_name (str): Variable name for context-specific level selection.

        Returns:
            np.ndarray: Array of contour levels optimized for the variable type and data range.
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
    
    def _extract_height_from_dataset(
        self,
        mpas_3d_processor: MPAS3DProcessor,
        time_index: int,
        vertical_coords: np.ndarray,
        var_name: str
    ) -> Optional[np.ndarray]:
        """
        Extract and interpolate geometric height data from dataset variables for vertical coordinate conversion. This internal helper method attempts to access height variables (zgrid or height) from the MPAS dataset, handles mismatched array dimensions through interpolation or averaging (nVertLevelsP1 to nVertLevels conversion), and returns height values in meters suitable for pressure-to-height or model-level-to-height transformations. The method employs scipy interpolation when array sizes differ significantly and falls back to None if height data is unavailable or extraction fails. This enables flexible vertical coordinate handling across different MPAS output configurations.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance providing dataset access.
            time_index (int): Time dimension index for extracting time-dependent height fields.
            vertical_coords (np.ndarray): Target vertical coordinate array determining expected output length.
            var_name (str): Height variable name to extract, typically 'zgrid' or 'height'.

        Returns:
            Optional[np.ndarray]: Extracted height data in meters with length matching vertical_coords, or None if extraction fails or variable unavailable.
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
    
    def _convert_vertical_to_height(
        self,
        vertical_coords: np.ndarray,
        vertical_coord_type: str,
        mpas_3d_processor: MPAS3DProcessor,
        time_index: int
    ) -> Tuple[np.ndarray, str]:
        """
        Converts vertical coordinates to height in kilometers when possible using standard atmosphere approximations or geometric height variables. This internal method handles three coordinate types: direct height conversion by dividing by 1000, pressure-to-height conversion using standard atmosphere scale height formula (h = -H*ln(P/P0) where H=8.4km), and model level conversion by searching for geometric height variables (zgrid, height) in the dataset. It returns converted coordinates and an updated coordinate type string for axis labeling.

        Parameters:
            vertical_coords (np.ndarray): Original vertical coordinate values.
            vertical_coord_type (str): Input coordinate type ('height', 'pressure', 'model_levels').
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance for accessing geometric height data.
            time_index (int): Time index for height variable extraction.

        Returns:
            Tuple[np.ndarray, str]: Converted vertical coordinates and coordinate type name ('height_km', 'pressure_hPa', 'model_levels').
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
    
    def _setup_pressure_axis(
        self,
        vertical_coords: np.ndarray,
        use_standard_ticks: bool = True
    ) -> None:
        """
        Configure pressure axis with logarithmic scaling and meteorologically-appropriate tick marks for professional cross-section displays. This internal helper method sets up logarithmic y-axis scaling suitable for pressure coordinates, applies standard atmospheric pressure level tick marks (1000, 850, 700, 500, 300, 200, 100, 50 hPa) when use_standard_ticks is enabled, formats tick labels with appropriate precision, and handles edge cases where pressure values are non-positive by falling back to linear scaling. This method eliminates code duplication between pressure_hPa and pressure coordinate axis formatting. The logarithmic scaling properly represents the exponential nature of atmospheric pressure decrease with altitude.

        Parameters:
            vertical_coords (np.ndarray): Pressure coordinate values in hPa or Pa determining axis range and tick selection.
            use_standard_ticks (bool): Flag to apply meteorological standard pressure levels as major tick marks (default: True for standard levels, False for automatic ticks).

        Returns:
            None: Modifies self.ax y-axis properties directly including scale, locators, and formatters.
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
        
    def _format_cross_section_axes(
        self,
        longitudes: np.ndarray,
        vertical_coords: np.ndarray,
        vertical_coord_type: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        max_height: Optional[float] = None
    ) -> None:
        """
        Formats cross-section plot axes with appropriate labels, limits, and coordinate-specific settings including inverted pressure axes and logarithmic scaling. This internal method configures the x-axis for longitude with custom formatting, sets y-axis properties based on vertical coordinate type (height in km with upward orientation, pressure in hPa/Pa with inverted logarithmic scale, model levels with top-to-bottom ordering), applies optional maximum height limits with unit conversion, and adds a path information text box showing start and end coordinates.

        Parameters:
            longitudes (np.ndarray): Longitude values for x-axis range.
            vertical_coords (np.ndarray): Vertical coordinate values for y-axis range.
            vertical_coord_type (str): Vertical coordinate type name ('height_km', 'pressure_hPa', 'pressure', 'height', 'model_levels').
            start_point (Tuple[float, float]): Cross-section start point as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Cross-section end point as (longitude, latitude) in degrees.
            max_height (Optional[float]): Optional maximum height in km for y-axis upper limit (default: None).

        Returns:
            None: Modifies self.ax axes properties directly.
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
    
    def _get_time_string(
        self,
        mpas_3d_processor: MPAS3DProcessor,
        time_index: int
    ) -> str:
        """
        Retrieves and formats a time string from the 3D processor dataset for plot annotations using multiple fallback strategies. This internal method first attempts to use the processor's get_time_info method if available, then tries to extract and format timestamps from the dataset's Time coordinate using pandas datetime conversion with standard UTC format string, and falls back to displaying the time index number if no time information is accessible. It returns a human-readable time string suitable for plot titles and labels.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance with temporal data.
            time_index (int): Time index for timestamp extraction.

        Returns:
            str: Formatted time string ('Valid: YYYY-MM-DD HH:MM UTC' or 'Time Index: N').
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
    
    def create_batch_cross_section_plots(
        self,
        mpas_3d_processor: Any,
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
        formats: List[str] = ['png']
    ) -> List[str]:
        """
        Creates vertical cross-section plots for all time steps in the dataset using batch processing with consistent styling and automatic file naming. This method iterates through all available time steps, extracts timestamps for filename generation, creates individual cross-section plots with uniform vertical extent and contour levels, saves each plot in multiple formats with descriptive filenames including vertical coordinate and time information, provides progress updates every 5 steps, and returns a complete list of created file paths.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance with loaded 3D dataset.
            output_dir (str): Output directory path for saving plot files.
            var_name (str): 3D atmospheric variable name to plot.
            start_point (Tuple[float, float]): Cross-section start point as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Cross-section end point as (longitude, latitude) in degrees.
            vertical_coord (str): Vertical coordinate system ('pressure', 'height', 'model_levels') (default: 'pressure').
            num_points (int): Number of interpolation points along cross-section (default: 100).
            levels (Optional[np.ndarray]): Contour levels for all plots (default: auto-generated).
            colormap (Optional[Union[str, mcolors.Colormap]]): Colormap name or object (default: variable-specific).
            extend (str): Colorbar extension mode 'both', 'min', 'max', 'neither' (default: 'both').
            plot_type (str): Plot rendering type 'filled_contour', 'contour', 'pcolormesh' (default: 'filled_contour').
            max_height (Optional[float]): Maximum vertical extent in km for all plots (default: None).
            file_prefix (str): Filename prefix for output files (default: 'mpas_crosssection').
            formats (List[str]): Output file format list (default: ['png']).

        Returns:
            List[str]: Complete list of created file paths including all format variants.

        Raises:
            ValueError: If processor is invalid, data not loaded, or variable not found.
            RuntimeError: If batch processing encounters critical failures.
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