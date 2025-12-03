#!/usr/bin/env python3

"""
MPAS Surface Variable Visualization

This module provides specialized plotting functionality for MPAS surface variables including 2-meter temperature, sea-level pressure, humidity, and wind speed with flexible rendering options and comprehensive cartographic presentation. It implements the MPASSurfacePlotter class that creates professional geographic maps using both scatter plot rendering (direct MPAS cell display preserving unstructured mesh resolution) and contour/filled contour plots (interpolated to regular grids for smooth gradients), with automatic unit conversion from model output to display units, variable-specific colormap and contour level selection, and optional feature overlays including wind vectors and geographic elements. The plotter supports batch processing for creating time series of surface maps with consistent styling, adaptive marker sizing based on map extent and data density, multiple map projections via Cartopy, and handles both 2D surface data and automatic extraction of surface levels from 3D datasets. Core capabilities include scipy-based grid interpolation for contour plots, geographic extent validation, metadata-driven styling, and publication-quality output suitable for operational weather analysis and climate model diagnostics.

Classes:
    MPASSurfacePlotter: Specialized class for creating surface variable visualizations from MPAS model output with cartographic presentation.
    
Functions:
    create_surface_plot: Convenience function for quick surface map generation without class instantiation.
    
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
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
from typing import Tuple, Optional, List, Union, Any, cast

try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None

from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.visualization.wind import MPASWindPlotter


class MPASSurfacePlotter(MPASVisualizer):
    """
    Specialized plotter for creating professional cartographic visualizations of MPAS surface variables including 2-meter temperature, sea-level pressure, humidity, and wind speed with flexible rendering options. This class extends MPASVisualizer to provide comprehensive surface diagnostic plotting capabilities including both scatter plot rendering (direct MPAS cell display preserving unstructured mesh resolution) and contour/filled contour plots (interpolated to regular grids for smooth gradients), automatic unit conversion from model output to display units via UnitConverter, variable-specific colormap and contour level selection through MPASFileMetadata, and optional feature overlays (wind vectors, geographic features, custom surface annotations). The plotter supports batch processing for creating time series of surface maps with consistent styling, adaptive marker sizing based on map extent and data density, multiple map projections via Cartopy, and publication-quality output with timestamps, colorbars, and professional cartographic elements suitable for mesoscale weather analysis and model evaluation diagnostics.
    """
    
    def create_surface_map(self,
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
                         grid_resolution: Optional[int] = None,
                         grid_resolution_deg: Optional[float] = None,
                         wind_overlay: Optional[dict] = None,
                         surface_overlay: Optional[dict] = None,
                         level_index: Optional[int] = None,
                         level_value: Optional[float] = None) -> Tuple[Figure, Axes]:
        """
        Create professional cartographic map visualizations for MPAS surface variables with flexible rendering options and automatic metadata handling. This comprehensive method serves as the main entry point for surface variable plotting, supporting both scatter (direct cell rendering) and contour (interpolated) plot types for 2-meter temperature, surface pressure, humidity, wind speed, and other MPAS surface diagnostics. The method performs automatic unit conversion via UnitConverter, retrieves variable-specific colormaps and contour levels from MPASFileMetadata, handles geographic extent validation and map projection setup, and optionally overlays wind vectors or additional surface features. Grid interpolation for contour plots uses adaptive or user-specified resolution, with scipy griddata performing triangulation-based interpolation from MPAS unstructured mesh to regular grids for smooth contoured fields.

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS mesh cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS mesh cell centers.
            data (np.ndarray): 1D array of surface variable values in model units to be plotted.
            var_name (str): Variable name for metadata lookup and unit conversion (e.g., 't2m', 'mslp', 'q2').
            lon_min (float): Western boundary of map extent in degrees.
            lon_max (float): Eastern boundary of map extent in degrees.
            lat_min (float): Southern boundary of map extent in degrees.
            lat_max (float): Northern boundary of map extent in degrees.
            title (Optional[str]): Custom plot title string, auto-generated from metadata if None (default: None).
            plot_type (str): Rendering method ('scatter' for direct cell display, 'contour' for interpolated lines/fills) (default: 'scatter').
            colormap (Optional[str]): Custom matplotlib colormap name overriding variable-specific defaults (default: None).
            levels (Optional[List[float]]): Custom contour level array overriding metadata defaults (default: None).
            clim_min (Optional[float]): Minimum color limit to clip contour levels (default: None).
            clim_max (Optional[float]): Maximum color limit to clip contour levels (default: None).
            projection (str): Cartopy projection name ('PlateCarree', 'Mercator', 'LambertConformal') (default: 'PlateCarree').
            time_stamp (Optional[datetime]): Valid time for title annotation (default: None).
            data_array (Optional[xr.DataArray]): Source xarray DataArray for metadata attribute extraction (default: None).
            grid_resolution (Optional[int]): Number of grid points per axis for contour interpolation (default: None uses adaptive).
            grid_resolution_deg (Optional[float]): Grid spacing in degrees for contour interpolation, overrides grid_resolution if set (default: None).
            wind_overlay (Optional[dict]): Wind vector overlay configuration with keys 'u_data', 'v_data', 'plot_type' (default: None).
            surface_overlay (Optional[dict]): Additional surface feature overlay configuration placeholder (default: None).
            level_index (Optional[int]): For 3D data extraction, vertical level index to plot (default: None).
            level_value (Optional[float]): For 3D data extraction, pressure level value in hPa to plot (default: None).

        Returns:
            Tuple[Figure, Axes]: Two-element tuple containing (matplotlib_figure, cartopy_geoaxes) with rendered surface map.

        Raises:
            ValueError: If plot_type is not 'scatter' or 'contour', or if geographic extent is invalid.
        """
        if plot_type not in ['scatter', 'contour', 'both', 'contourf']:
            raise ValueError(f"plot_type must be 'scatter', 'contour', 'contourf', or 'both', got '{plot_type}'")
        
        if data.ndim > 1:
            if data.ndim > 3:
                raise ValueError(f"only 1D, 2D and 3D data are supported, got {data.ndim}D data with shape {data.shape}")
            
            if level_index is not None:
                if data.ndim == 2:
                    data = data[:, level_index]
                elif data.ndim == 3:
                    data = data[:, level_index, ...]
                    if data.ndim > 1:  
                        data = data[..., 0]
                print(f"Extracted 2D data from {data.ndim}D using level_index={level_index}")
            elif level_value is not None:
                if data.ndim == 2:
                    data = data[:, -1]
                elif data.ndim == 3:
                    data = data[:, -1, ...]
                    if data.ndim > 1:
                        data = data[..., 0]
                print(f"Extracted 2D data from multi-D using surface level (level_value={level_value} not yet implemented)")
            else:
                if data.ndim == 2:
                    data = data[:, -1]
                elif data.ndim == 3:
                    data = data[:, -1, ...]
                    if data.ndim > 1:
                        data = data[..., 0]
                print("Extracted 2D data from multi-D using surface level (default)")
            
            if data.ndim > 1:
                data = data.flatten()
                print(f"Flattened remaining dimensions to 1D, final shape: {data.shape}")
        
        if len(data) != len(lon) or len(data) != len(lat):
            raise ValueError(f"Data array length ({len(data)}) must match coordinate arrays length (lon: {len(lon)}, lat: {len(lat)})")
        
        var_metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        original_unit = None

        if data_array is not None:
            try:
                original_unit = getattr(data_array, 'attrs', {}).get('units')
            except Exception:
                original_unit = None

        if original_unit is None and hasattr(data, 'attrs'):
            original_unit = getattr(data, 'attrs', {}).get('units')

        if not original_unit:
            original_unit = var_metadata.get('original_units')

        if not original_unit:
            original_unit = var_metadata.get('units')

        display_unit = UnitConverter.get_display_units(var_name, original_unit or "")

        var_metadata['original_units'] = original_unit
        var_metadata['units'] = display_unit
        self._current_var_metadata = var_metadata

        if original_unit != display_unit:
            try:
                converted_data = UnitConverter.convert_units(
                    data, cast(str, original_unit or ""), display_unit
                )
                if isinstance(converted_data, xr.DataArray):
                    data = converted_data.values
                elif isinstance(converted_data, np.ndarray):
                    data = converted_data
                else:
                    data = np.asarray(converted_data)
                print(f"Converted {var_name} from {original_unit} to {display_unit}")
            except ValueError as e:
                print(f"Warning: Could not convert {var_name} from {original_unit} to {display_unit}: {e}")

        lon = self.convert_to_numpy(lon)
        lat = self.convert_to_numpy(lat)
        data = self.convert_to_numpy(data)
        
        if colormap is None:
            colormap = var_metadata['colormap']

        if levels is None:
            levels = var_metadata.get('levels', None)
        
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for cartopy plots"
        
        is_global_lon = (lon_max - lon_min) >= 359.0
        is_global_lat = (lat_max - lat_min) >= 179.0
        
        if is_global_lon and is_global_lat:
            filter_lon_min = max(lon_min, -179.99)
            filter_lon_max = min(lon_max, 179.99)
            filter_lat_min = max(lat_min, -89.99)
            filter_lat_max = min(lat_max, 89.99)
            self.ax.set_extent([filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max], crs=data_crs)
            print(f"Using global extent (adjusted to avoid dateline): [{filter_lon_min}, {filter_lon_max}, {filter_lat_min}, {filter_lat_max}]")
            
            filter_lon_min_data = -180.01 
            filter_lon_max_data = 180.01 
            filter_lat_min_data = -90.01
            filter_lat_max_data = 90.01
        else:
            filter_lon_min = lon_min
            filter_lon_max = lon_max
            filter_lat_min = lat_min
            filter_lat_max = lat_max
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
            
            filter_lon_min_data = lon_min
            filter_lon_max_data = lon_max
            filter_lat_min_data = lat_min
            filter_lat_max_data = lat_max
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
        
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        
        if clim_min is not None and clim_max is not None and levels is not None:
            levels = [level for level in levels if clim_min <= level <= clim_max]

            if clim_min not in levels:
                levels.insert(0, clim_min)

            if clim_max not in levels:
                levels.append(clim_max)

        try:
            cmap_obj = plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
        except Exception:
            cmap_obj = plt.get_cmap('viridis')
        
        if cmap_obj is None:
            cmap_obj = plt.get_cmap('viridis')

        norm = None

        if clim_min is not None or clim_max is not None:
            vmin = clim_min if clim_min is not None else float(np.nanmin(data))
            vmax = clim_max if clim_max is not None else float(np.nanmax(data))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            try:
                if levels is not None:
                    color_levels_sorted = sorted(set([v for v in levels if np.isfinite(v)]))
                    if color_levels_sorted:
                        last_bound = max(color_levels_sorted) + 1
                        bounds = [min(color_levels_sorted)] + color_levels_sorted + [last_bound]
                        norm = BoundaryNorm(bounds, ncolors=cmap_obj.N, clip=True)
            except Exception:
                norm = None
        
        print(f"DEBUG: filter_lon_min_data={filter_lon_min_data:.4f}, filter_lon_max_data={filter_lon_max_data:.4f}")
        print(f"DEBUG: lon range in data: [{np.min(lon):.4f}, {np.max(lon):.4f}]")
        
        valid_mask = (np.isfinite(data) & 
                     np.isfinite(lon) & np.isfinite(lat) &
                     (lon >= filter_lon_min_data) & (lon <= filter_lon_max_data) &
                     (lat >= filter_lat_min_data) & (lat <= filter_lat_max_data))

        try:
            if hasattr(valid_mask, 'compute'):
                valid_mask = cast(Any, valid_mask).compute()
        except Exception:
            pass

        valid_mask = np.asarray(valid_mask, dtype=bool)
        
        if not np.any(valid_mask):
            raise ValueError(f"No valid data points found within the specified map extent for {var_name}")
            
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        data_valid = data[valid_mask]
        
        print(f"Plotting {len(data_valid):,} data points for {var_name}")
        print(f"Data range: {data_valid.min():.3f} to {data_valid.max():.3f} {var_metadata['units']}")
        
        if plot_type == 'scatter':
            self._create_scatter_plot(lon_valid, lat_valid, data_valid, cmap_obj, norm, data_crs)
        elif plot_type == 'contour':            
            self._create_contour_plot(lon_valid, lat_valid, data_valid, 
                                    filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
                                    cmap_obj, norm, levels, data_crs,
                                    grid_resolution, grid_resolution_deg)
        elif plot_type == 'contourf':
            self._create_contourf_plot(lon_valid, lat_valid, data_valid, 
                                     filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
                                     cmap_obj, norm, levels, data_crs,
                                     grid_resolution, grid_resolution_deg)
        elif plot_type == 'both':
            self._create_contour_plot(lon_valid, lat_valid, data_valid, 
                                    filter_lon_min, filter_lon_max, filter_lat_min, filter_lat_max,
                                    cmap_obj, norm, levels, data_crs,
                                    grid_resolution, grid_resolution_deg)
            self._create_scatter_plot(lon_valid, lat_valid, data_valid, cmap_obj, norm, data_crs)
        
        time_in_title = False

        if title is None:
            title = f"MPAS {var_metadata['long_name']}"
            if time_stamp:
                time_str = time_stamp.strftime('%Y%m%dT%H')
                title += f" | Valid Time: {time_str}"
                time_in_title = True
        else:
            if time_stamp:
                time_str = time_stamp.strftime('%Y%m%dT%H')
                time_in_title = (time_str in title or 'Valid Time:' in title or 'Valid:' in title)

        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if time_stamp and not time_in_title:
            time_str = time_stamp.strftime('%Y%m%dT%H')
            self.ax.text(0.02, 0.98, f'Valid: {time_str}', 
                        transform=self.ax.transAxes, 
                        fontsize=12, fontweight='bold',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        gl = self.ax.gridlines(crs=data_crs, draw_labels=True, 
                             linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        gl.top_labels = False
        gl.right_labels = False

        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        plt.tight_layout()
        self.fig.subplots_adjust(bottom=-0.07)
        
        self.add_timestamp_and_branding()
        
        if wind_overlay is not None:
            try:
                wind_plotter = MPASWindPlotter()
                wind_plotter.add_wind_overlay(self.ax, lon, lat, wind_overlay)
                print("Added wind overlay to surface map")
            except ValueError:
                raise
            except Exception as e:
                print(f"Warning: Failed to add wind overlay: {e}")
        
        if surface_overlay is not None:
            try:
                self._add_surface_overlay(lon, lat, surface_overlay)
                print("Added surface overlay to surface map")
            except ValueError:
                raise
            except Exception as e:
                print(f"Warning: Failed to add surface overlay: {e}")
        
        return self.fig, self.ax

    def _interpolate_to_grid(self, lon: np.ndarray, lat: np.ndarray, data: np.ndarray,
                           lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                           grid_resolution: Optional[int] = None,
                           grid_resolution_deg: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper method to interpolate scattered data to a regular grid with adaptive resolution.
        This eliminates duplication between _create_contour_plot and _create_contourf_plot.
        
        Parameters:
            lon (np.ndarray): Source longitude array.
            lat (np.ndarray): Source latitude array.
            data (np.ndarray): Source data values.
            lon_min (float): Western grid bound.
            lon_max (float): Eastern grid bound.
            lat_min (float): Southern grid bound.
            lat_max (float): Northern grid bound.
            grid_resolution (Optional[int]): Grid points per axis.
            grid_resolution_deg (Optional[float]): Grid spacing in degrees.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: lon_mesh, lat_mesh, data_interp
        """
        if griddata is None:
            raise ImportError("scipy.interpolate.griddata is required for contour plotting")
        
        if grid_resolution_deg is not None:
            step = float(grid_resolution_deg)
            if step <= 0:
                raise ValueError("grid_resolution_deg must be > 0")

            lon_coords = np.arange(lon_min, lon_max + 1e-12, step)
            lat_coords = np.arange(lat_min, lat_max + 1e-12, step)

            nx = len(lon_coords)
            ny = len(lat_coords)
            max_points = 1200

            if nx > max_points or ny > max_points:
                lon_coords = np.linspace(lon_min, lon_max, min(nx, max_points))
                lat_coords = np.linspace(lat_min, lat_max, min(ny, max_points))
                print(f"Requested degree step {step}° produces >{max_points} points per axis; clipping to {len(lat_coords)}x{len(lon_coords)} grid")

            lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
            print(f"Interpolating {len(data)} points to {lon_mesh.shape[0]}x{lon_mesh.shape[1]} grid (~{step}° resolution)...")
        else:
            if grid_resolution is None:
                adaptive = int(np.sqrt(len(data)) / 9)
                grid_resolution = max(25, min(adaptive, 200))
                print(f"Auto-selected grid resolution: {grid_resolution}")

            lon_coords = np.linspace(lon_min, lon_max, grid_resolution)
            lat_coords = np.linspace(lat_min, lat_max, grid_resolution)
            lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
            print(f"Interpolating {len(data)} points to {grid_resolution}x{grid_resolution} grid...")

        data_interp = griddata((lon, lat), data, (lon_mesh, lat_mesh), method='linear')

        try:
            if np.any(np.isnan(data_interp)):
                data_interp_nearest = griddata((lon, lat), data, (lon_mesh, lat_mesh), method='nearest')
                data_interp[np.isnan(data_interp)] = data_interp_nearest[np.isnan(data_interp)]
                print("Filled NaN values in interpolated grid using nearest-neighbor interpolation to avoid edge clipping")
        except Exception:
            pass
        
        return lon_mesh, lat_mesh, data_interp

    def _add_colorbar_with_metadata(self, mappable) -> None:
        """
        Helper method to add colorbar with metadata-driven labels and formatting.
        This eliminates duplication between _create_scatter_plot and _create_contourf_plot.
        
        Parameters:
            mappable: Matplotlib mappable object (scatter or contourf result).
        """
        assert self.fig is not None, "Figure must be created before adding colorbar"
        
        cbar = self.fig.colorbar(mappable, ax=self.ax, orientation='horizontal', extend='both',
                               pad=0.06, shrink=0.8, aspect=30)
        
        if hasattr(self, '_current_var_metadata') and self._current_var_metadata:
            var_units = self._current_var_metadata.get('units', '')
            var_long_name = self._current_var_metadata.get('long_name', 'Value')
            if var_units and f'({var_units})' in var_long_name:
                cbar_label = var_long_name
            else:
                cbar_label = f"{var_long_name} ({var_units})" if var_units else var_long_name
            cbar.set_label(cbar_label, fontsize=12, fontweight='bold', labelpad=-50)
        
        try:
            ticks = cbar.get_ticks().tolist()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
            cbar.ax.tick_params(labelsize=8)
        except Exception:
            pass

    def _create_scatter_plot(self, lon: np.ndarray, lat: np.ndarray, data: np.ndarray,
                           cmap_obj: Union[str, mcolors.Colormap], norm: Optional[mcolors.Normalize],
                           data_crs: ccrs.CRS) -> None:
        """
        Renders a scatter plot of point data on the active cartographic axes with adaptive marker sizing
        and density-based transparency. This internal method calculates optimal marker sizes based on
        map extent and point density, sorts data values for proper color overlay ordering, and applies
        density-dependent alpha values to prevent overplotting. It generates a horizontal colorbar with
        metadata-driven labels formatted using the instance's current variable metadata if available.

        Parameters:
            lon (np.ndarray): 1D longitude array in degrees matching lat and data arrays.
            lat (np.ndarray): 1D latitude array in degrees matching lon and data arrays.
            data (np.ndarray): 1D array of data values used for color mapping.
            cmap_obj (Union[str, mcolors.Colormap]): Colormap name or instance for point coloring.
            norm (Optional[mcolors.Normalize]): Normalization for mapping data to colormap range.
            data_crs (ccrs.CRS): Coordinate reference system for input coordinates (typically PlateCarree).

        Returns:
            None: Draws scatter plot directly onto self.ax and updates self.fig with colorbar.
        """
        assert self.ax is not None, "Axes must be created before scatter plot"
        assert self.fig is not None, "Figure must be created before scatter plot"
        
        map_extent = (lon.min(), lon.max(), lat.min(), lat.max())
        fig_size = (self.figsize[0], self.figsize[1])
        marker_size = self.calculate_adaptive_marker_size(map_extent, len(data), fig_size)
        
        map_area = (map_extent[1] - map_extent[0]) * (map_extent[3] - map_extent[2])
        point_density = len(data) / map_area if map_area > 0 else 0
        
        if point_density > 1000:
            alpha_val = 0.8
        elif point_density > 100:
            alpha_val = 0.9
        else:
            alpha_val = 0.9
        
        sort_indices = np.argsort(data)
        lon_sorted = lon[sort_indices]
        lat_sorted = lat[sort_indices]
        data_sorted = data[sort_indices]
        
        scatter = self.ax.scatter(lon_sorted, lat_sorted, c=data_sorted,
                               cmap=cmap_obj, norm=norm, s=marker_size, alpha=alpha_val,
                               transform=data_crs, edgecolors='none')
        
        self._add_colorbar_with_metadata(scatter)
    
    def _create_contour_plot(self, lon: np.ndarray, lat: np.ndarray, data: np.ndarray,
                           lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                           cmap_obj: Union[str, mcolors.Colormap], norm: Optional[mcolors.Normalize],
                           levels: Optional[List[float]], data_crs: ccrs.CRS,
                           grid_resolution: Optional[int] = None,
                           grid_resolution_deg: Optional[float] = None) -> None:
        """
        Interpolates scattered point data to a regular grid using scipy griddata and renders
        line contours (matplotlib.contour) on the cartographic axes. This internal method
        supports fixed-resolution and degree-based grids, applies adaptive resolution when
        parameters are omitted, and performs linear interpolation using scipy's griddata.
        Unlike the filled-contour helper, this method draws contour lines and optionally
        labels them when contour levels are provided.

        Parameters:
            lon (np.ndarray): 1D source longitude array in degrees.
            lat (np.ndarray): 1D source latitude array in degrees.
            data (np.ndarray): 1D source data values corresponding to lon/lat points.
            lon_min (float): Western longitude bound in degrees for target grid.
            lon_max (float): Eastern longitude bound in degrees for target grid.
            lat_min (float): Southern latitude bound in degrees for target grid.
            lat_max (float): Northern latitude bound in degrees for target grid.
            cmap_obj (Union[str, mcolors.Colormap]): Colormap name or instance for contour line colors.
            norm (Optional[mcolors.Normalize]): Normalization for color mapping.
            levels (Optional[List[float]]): Explicit contour levels (default: automatic levels).
            data_crs (ccrs.CRS): Coordinate reference system for input coordinates (typically PlateCarree).
            grid_resolution (Optional[int]): Number of grid points per axis (default: adaptive from 25-200).
            grid_resolution_deg (Optional[float]): Grid spacing in degrees (overrides grid_resolution).

        Returns:
            None: Draws contour lines to self.ax and updates figure colorbar (if applicable).
        """
        assert self.ax is not None, "Axes must be created before contour plot"
        assert self.fig is not None, "Figure must be created before contour plot"
        
        lon_mesh, lat_mesh, data_interp = self._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, grid_resolution_deg
        )

        try:
            contour_color = 'black'
            try:
                if levels is not None:
                    cs = self.ax.contour(lon_mesh, lat_mesh, data_interp, levels=levels,
                                         colors=contour_color, linewidths=1.0, linestyles='solid',
                                         transform=data_crs)
                else:
                    cs = self.ax.contour(lon_mesh, lat_mesh, data_interp,
                                         colors=contour_color, linewidths=1.0, linestyles='solid',
                                         transform=data_crs)

                if levels is not None:
                    try:
                        self.ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
                    except Exception:
                        pass
            except Exception as e:
                raise RuntimeError(f"Contour plotting failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Contour plotting failed: {e}")
        
    def _create_contourf_plot(self, lon: np.ndarray, lat: np.ndarray, data: np.ndarray,
                             lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                             cmap_obj: Union[str, mcolors.Colormap], norm: Optional[mcolors.Normalize],
                             levels: Optional[List[float]], data_crs: ccrs.CRS,
                             grid_resolution: Optional[int] = None,
                             grid_resolution_deg: Optional[float] = None) -> None:
        """
        Interpolates scattered data to a regular grid and renders filled contours using matplotlib's contourf
        on the cartographic axes. This internal method parallels _create_contour_plot functionality but uses
        contourf specifically for filled contour rendering, supports both fixed-resolution and degree-based
        grid generation with adaptive resolution selection, and produces a horizontal colorbar with
        metadata-driven labels extracted from the instance's current variable metadata.

        Parameters:
            lon (np.ndarray): Input longitude array in degrees.
            lat (np.ndarray): Input latitude array in degrees.
            data (np.ndarray): Data values at corresponding lon/lat points.
            lon_min (float): Western longitude bound in degrees for interpolation grid.
            lon_max (float): Eastern longitude bound in degrees for interpolation grid.
            lat_min (float): Southern latitude bound in degrees for interpolation grid.
            lat_max (float): Northern latitude bound in degrees for interpolation grid.
            cmap_obj (Union[str, mcolors.Colormap]): Colormap name or instance.
            norm (Optional[mcolors.Normalize]): Normalization for mapping data values to colors.
            levels (Optional[List[float]]): Explicit contour levels (default: automatic levels).
            data_crs (ccrs.CRS): Coordinate reference system for source coordinates.
            grid_resolution (Optional[int]): Grid points per axis (default: adaptive from 25-200).
            grid_resolution_deg (Optional[float]): Grid spacing in degrees (overrides grid_resolution).

        Returns:
            None: Draws filled contours to self.ax and updates figure colorbar.
        """
        assert self.ax is not None, "Axes must be created before contourf plot"
        assert self.fig is not None, "Figure must be created before contourf plot"
        
        lon_mesh, lat_mesh, data_interp = self._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, grid_resolution_deg
        )

        if levels is not None:
            cs = self.ax.contourf(lon_mesh, lat_mesh, data_interp, levels=levels,
                                cmap=cmap_obj, norm=norm, transform=data_crs, extend='both')
        else:
            cs = self.ax.contourf(lon_mesh, lat_mesh, data_interp,
                                cmap=cmap_obj, norm=norm, transform=data_crs, extend='both')

        self._add_colorbar_with_metadata(cs)

    def _add_surface_overlay(self, lon: np.ndarray, lat: np.ndarray, surface_config: dict) -> None:
        """
        Adds an auxiliary contour or filled contour overlay to the current cartographic axes using
        interpolated scattered point data. This internal method processes a configuration dictionary
        specifying overlay data, variable name, plot type (contour/contourf), styling parameters
        (levels, colors, linewidth, alpha), and optional vertical level indexing for 3D data. It
        performs grid interpolation using scipy griddata and renders the overlay with optional contour labels.

        Parameters:
            lon (np.ndarray): Longitude array in degrees for overlay points.
            lat (np.ndarray): Latitude array in degrees for overlay points.
            surface_config (dict): Configuration dictionary with required key 'data' (ndarray) and optional keys 'var_name' (str), 'plot_type' ('contour'/'contourf'), 'levels' (list), 'colors' (str), 'linewidth' (float), 'alpha' (float), 'level_index' (int), 'add_labels' (bool).

        Returns:
            None: Draws overlay directly onto self.ax without returning objects.
        """
        assert self.ax is not None, "Axes must be created before surface overlay"
        
        if griddata is None:
            raise ImportError("scipy.interpolate.griddata is required for contour plotting")
        
        overlay_data = surface_config['data']
        var_name = surface_config.get('var_name', 'overlay')
        plot_type = surface_config.get('plot_type', 'contour')
        levels = surface_config.get('levels', None)
        colors = surface_config.get('colors', 'black')
        linewidth = surface_config.get('linewidth', 1.0)
        alpha = surface_config.get('alpha', 1.0)
        
        if plot_type not in ['contour', 'contourf']:
            raise ValueError(f"Unsupported surface overlay plot_type: {plot_type}")
        
        if overlay_data.ndim > 1:
            level_index = surface_config.get('level_index', None)
            if level_index is not None:
                overlay_data = overlay_data[:, level_index]
            else:
                overlay_data = overlay_data[:, -1]
        
        valid_mask = (np.isfinite(overlay_data) & 
                     np.isfinite(lon) & np.isfinite(lat))
        
        if not np.any(valid_mask):
            print(f"Warning: No valid overlay data found for {var_name}")
            return
        
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        data_valid = overlay_data[valid_mask]
        
        grid_resolution = 50
        lon_coords = np.linspace(lon.min(), lon.max(), grid_resolution)
        lat_coords = np.linspace(lat.min(), lat.max(), grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
        
        data_interp = griddata((lon_valid, lat_valid), data_valid, 
                              (lon_mesh, lat_mesh), method='linear')
        
        if plot_type == 'contour':
            if levels is not None:
                cs = self.ax.contour(lon_mesh, lat_mesh, data_interp, levels=levels,
                                   colors=colors, linewidths=linewidth, alpha=alpha,
                                   transform=ccrs.PlateCarree())
            else:
                cs = self.ax.contour(lon_mesh, lat_mesh, data_interp,
                                   colors=colors, linewidths=linewidth, alpha=alpha,
                                   transform=ccrs.PlateCarree())
            
            if surface_config.get('add_labels', False):
                self.ax.clabel(cs, inline=True, fontsize=8)
                
        elif plot_type == 'contourf':
            if levels is not None:
                cs = self.ax.contourf(lon_mesh, lat_mesh, data_interp, levels=levels,
                                    alpha=alpha, transform=ccrs.PlateCarree())
            else:
                cs = self.ax.contourf(lon_mesh, lat_mesh, data_interp,
                                    alpha=alpha, transform=ccrs.PlateCarree())
        
        print(f"Added {plot_type} surface overlay for {var_name}")

    @staticmethod
    def convert_to_numpy(x: Any) -> np.ndarray:
        """
        Convert xarray DataArray or dask-backed arrays to a plain NumPy array.

        This helper accepts xarray DataArray objects, dask-backed arrays, or any array-like
        object and returns a 1D or N-D NumPy ndarray suitable for downstream numeric
        operations and boolean indexing. It safely unwraps xarray DataArray objects to their
        underlying values, computes any dask-backed arrays to materialize them in memory,
        and finally coerces the result into a NumPy ndarray. Use this function when a
        NumPy array is required for masking, indexing, or plotting operations that do not
        support lazy/dask-backed arrays.

        Parameters:
            x (Any): Input array-like object. Typical values are an `xarray.DataArray`, a
                dask-backed array-like object, a NumPy ndarray, or other array-like values
                returned by processing routines.

        Returns:
            np.ndarray: A NumPy ndarray containing the computed/converted values. The shape
                and dtype match the input where possible; this array is safe for boolean
                indexing and other NumPy operations.
        """
        try:
            if isinstance(x, xr.DataArray):
                arr = x.values
            else:
                arr = x
        except Exception:
            arr = x

        try:
            if hasattr(arr, 'compute'):
                arr = cast(Any, arr).compute()
        except Exception:
            pass

        return np.asarray(arr)

    def create_batch_surface_maps(self, processor, output_dir: str,
                                 lon_min: float, lon_max: float,
                                 lat_min: float, lat_max: float,
                                 var_name: str = 't2m',
                                 plot_type: str = 'scatter',
                                 file_prefix: str = 'mpas_surface',
                                 formats: List[str] = ['png'],
                                 grid_resolution: Optional[int] = None,
                                 grid_resolution_deg: Optional[float] = None,
                                 clim_min: Optional[float] = None,
                                 clim_max: Optional[float] = None) -> List[str]:
        """
        Generates surface variable maps for all time steps in the dataset using batch processing with
        automatic coordinate extraction and metadata handling. This method iterates through all available
        time steps, extracts 2D variable data and coordinates using the processor instance, constructs
        descriptive titles with timestamp information, and saves plots in multiple formats. It provides
        progress updates every 10 steps and handles individual step failures gracefully.

        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset and time dimension.
            output_dir (str): Output directory path for saving plot files.
            lon_min (float): Western longitude bound in degrees.
            lon_max (float): Eastern longitude bound in degrees.
            lat_min (float): Southern latitude bound in degrees.
            lat_max (float): Northern latitude bound in degrees.
            var_name (str): 2D surface variable name to plot (default: 't2m').
            plot_type (str): Plot rendering type 'scatter' or 'contour' (default: 'scatter').
            file_prefix (str): Filename prefix for output files (default: 'mpas_surface').
            formats (List[str]): Output format list such as ['png', 'pdf'] (default: ['png']).
            grid_resolution (Optional[int]): Grid points per axis for interpolation (default: adaptive).
            grid_resolution_deg (Optional[float]): Grid spacing in degrees (default: None).
            clim_min (Optional[float]): Minimum color limit for all plots (default: None).
            clim_max (Optional[float]): Maximum color limit for all plots (default: None).

        Returns:
            List[str]: List of created file paths including all format variants.
        """
        if processor.dataset is None:
            raise ValueError("No data loaded in processor")

        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
        total_times = processor.dataset.sizes[time_dim]

        created_files = []
        print(f"\nCreating surface maps for {total_times} time steps...")

        for time_idx in range(total_times):
            try:
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                    time_end = pd.Timestamp(processor.dataset.Time.values[time_idx]).to_pydatetime()
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_end = None
                    time_str = f"t{time_idx:03d}"

                var_data = processor.get_2d_variable_data(var_name, time_idx)
                lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)

                title = f"MPAS Surface Map | Var: {var_name.upper()} | Valid: {time_str} | Type: {plot_type.title()}"

                fig, ax = self.create_surface_map(
                    lon, lat, var_data.values, var_name,
                    lon_min, lon_max, lat_min, lat_max,
                    title=title,
                    plot_type=plot_type,
                    time_stamp=time_end,
                    data_array=var_data,
                    grid_resolution=grid_resolution,
                    grid_resolution_deg=grid_resolution_deg,
                    clim_min=clim_min,
                    clim_max=clim_max
                )

                output_path = os.path.join(output_dir, f"{file_prefix}_{var_name}_{plot_type}_{time_str}")
                self.save_plot(output_path, formats=formats)

                for fmt in formats:
                    created_files.append(f"{output_path}.{fmt}")

                self.close_plot()

                if (time_idx + 1) % 10 == 0:
                    print(f"Completed {time_idx + 1}/{total_times} surface maps...")

            except Exception as e:
                print(f"Error creating surface map for time index {time_idx}: {e}")
                continue

        print(f"\nBatch processing completed. Created {len(created_files)} files.")
        return created_files

    def get_surface_colormap_and_levels(self, var_name: str, data_array: Optional[xr.DataArray] = None) -> Tuple[str, List[float]]:
        """
        Retrieves variable-specific colormap name and contour levels for 2D surface variables by querying
        the MPASFileMetadata system. This convenience method extracts colormap and level specifications
        from metadata definitions based on variable name and optional data array for automatic level
        detection, returning a tuple suitable for direct use in plotting functions. It provides consistent
        color mapping across all surface visualizations.

        Parameters:
            var_name (str): 2D surface variable name (e.g., 't2m', 'psfc', 'q2').
            data_array (Optional[xr.DataArray]): Optional data array for automatic level generation (default: None).

        Returns:
            Tuple[str, List[float]]: Colormap name and list of contour level values.
        """
        metadata = MPASFileMetadata.get_2d_variable_metadata(var_name, data_array)
        return metadata['colormap'], metadata['levels']

    def create_simple_scatter_plot(self,
                                 lon: np.ndarray,
                                 lat: np.ndarray,
                                 data: np.ndarray,
                                 title: str = "MPAS Surface Variable",
                                 colorbar_label: str = "Value",
                                 colormap: str = 'viridis',
                                 point_size: float = 2.0) -> Tuple[Figure, Axes]:
        """
        Creates a simple scatter plot without cartographic projections for quick visualization and debugging
        purposes. This lightweight method uses standard matplotlib axes rather than cartopy projections,
        filters invalid data points (NaN/inf), renders a basic scatter plot with colorbar, applies grid
        lines and axis labels, and adds timestamp/branding through the standard method. It returns figure
        and axes objects for further customization or immediate display.

        Parameters:
            lon (np.ndarray): Longitude coordinate array in degrees.
            lat (np.ndarray): Latitude coordinate array in degrees.
            data (np.ndarray): Data value array for color mapping.
            title (str): Plot title text (default: "MPAS Surface Variable").
            colorbar_label (str): Colorbar label text (default: "Value").
            colormap (str): Matplotlib colormap name (default: 'viridis').
            point_size (float): Scatter marker size in points (default: 2.0).

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects for the plot.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        valid_mask = np.isfinite(data) & np.isfinite(lon) & np.isfinite(lat)

        if not np.any(valid_mask):
            raise ValueError("No valid data points found")
            
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        data_valid = data[valid_mask]
        
        scatter = self.ax.scatter(lon_valid, lat_valid, c=data_valid, 
                                cmap=colormap, s=point_size, alpha=0.8)
        
        cbar = self.fig.colorbar(scatter, ax=self.ax)
        cbar.set_label(colorbar_label, fontsize=12)
        
        self.ax.set_xlabel('Longitude', fontsize=12)
        self.ax.set_ylabel('Latitude', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.add_timestamp_and_branding()
        
        return self.fig, self.ax


def create_surface_plot(lon: np.ndarray, lat: np.ndarray, data: np.ndarray,
                       var_name: str, extent: Tuple[float, float, float, float],
                       plot_type: str = 'scatter', title: Optional[str] = None,
                       colormap: Optional[str] = None, **kwargs) -> Tuple[Figure, Axes]:
    """
    Convenience function for quick surface plotting.
    
    Parameters:
        lon (np.ndarray): Longitude coordinates.
        lat (np.ndarray): Latitude coordinates.
        data (np.ndarray): Data values.
        var_name (str): Variable name.
        extent (Tuple[float, float, float, float]): Map extent (lon_min, lon_max, lat_min, lat_max).
        plot_type (str): Plot type ('scatter' or 'contour').
        title (Optional[str]): Plot title.
        colormap (Optional[str]): Colormap name.
        **kwargs: Additional arguments passed to create_surface_map.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
    """
    plotter = MPASSurfacePlotter()
    lon_min, lon_max, lat_min, lat_max = extent
    
    return plotter.create_surface_map(
        lon, lat, data, var_name,
        lon_min, lon_max, lat_min, lat_max,
        title=title, plot_type=plot_type, colormap=colormap,
        **kwargs
    )