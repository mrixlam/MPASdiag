#!/usr/bin/env python3

"""
MPAS Visualization Module

This module provides comprehensive visualization functionality for MPAS model output data,
including precipitation maps, scatter plots, and cartographic presentations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Last Modified: 2025-10-06

Features:
    - Professional cartographic precipitation maps
    - Scatter plot visualization for unstructured data
    - Customizable colormaps and color levels
    - High-quality output with multiple format support
    - Flexible map extents and projections
    - Comprehensive plot annotation and metadata
"""

import os
import re
import warnings
from datetime import datetime
from typing import Tuple, Optional, List, Any, Union

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .utils import get_accumulation_hours
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.geoaxes import GeoAxes

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False})

warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')


class MPASVisualizer:
    """
    Main class for visualizing MPAS model output data.
    
    This class provides methods for creating publication-quality maps and plots
    of MPAS unstructured mesh data with professional cartographic presentation.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 10), dpi: int = 300):
        """
        Initialize the MPAS visualizer.

        Parameters:
            figsize (Tuple[float, float]): Figure size in inches (width, height).
            dpi (int): Resolution for output images.

        Returns:
            None
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
    
    def add_timestamp_and_branding(self) -> None:
        """
        Add timestamp and MPASdiag branding to the bottom left of the current figure.

        Parameters:
            None

        Returns:
            None
        """
        if self.fig is not None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            self.fig.text(0.02, 0.09, f'Generated with MPASdiag on: {timestamp}', 
                         fontsize=8, alpha=0.7, transform=self.fig.transFigure)
    
    def create_colormap(self, accum: str = "a24h") -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        Create a discrete colormap and contour levels for precipitation plotting.

        Parameters:
            accum (str): Accumulation period (e.g., 'a24h', 'a01h') to determine appropriate levels.

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: Colormap and contour levels.
        """
        colors = [
            "white", "lightskyblue", "dodgerblue", "seagreen", "lawngreen",
            "darkorange", "red", "maroon", "darkviolet", "fuchsia", "violet",
        ]

        accum_key = (accum or "a24h").lower().strip()
        hours = None

        try:
            m = re.search(r"(\d+)", accum_key)
            if m:
                hours = int(m.group(1))
        except Exception:
            hours = None

        if hours is None or hours >= 12:
            levels = [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]
        else:
            levels = [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75]

        cmap = mcolors.ListedColormap(colors)
        return cmap, levels
    
    def format_latitude(self, value: float, _) -> str:
        """
        Format latitude value with N/S direction.

        Parameters:
            value (float): Latitude value in degrees.
            _ : Unused parameter (required by FuncFormatter signature).

        Returns:
            str: Formatted latitude string with N/S suffix (e.g., "45.0°N").
        """
        direction = 'N' if value >= 0 else 'S'
        return f"{abs(value):.1f}°{direction}"

    def format_longitude(self, value: float, _) -> str:
        """
        Format longitude value with E/W direction.

        Parameters:
            value (float): Longitude value in degrees.
            _ : Unused parameter (required by FuncFormatter signature).

        Returns:
            str: Formatted longitude string with E/W suffix (e.g., "120.0°W").
        """
        direction = 'E' if value >= 0 else 'W'
        return f"{abs(value):.1f}°{direction}"
    
    def calculate_adaptive_marker_size(self, map_extent: Tuple[float, float, float, float], 
                                     num_points: int, fig_size: Tuple[float, float] = (12, 10)) -> float:
        """
        Calculate adaptive marker size based on plot extent, data density, and figure size.
        
        Parameters:
            map_extent (Tuple[float, float, float, float]): Map bounds (lon_min, lon_max, lat_min, lat_max)
            num_points (int): Number of data points to plot
            fig_size (Tuple[float, float]): Figure size in inches (width, height)
            
        Returns:
            float: Appropriate marker size for scatter plot
            
        Notes:
            - Calculates marker size based on map extent area and point density.
            - Scales with figure size to maintain consistent visual appearance.
            - Provides reasonable bounds to avoid extremely small or large markers.
        """
        if map_extent is None:
            return 5.0  
        
        lon_min, lon_max, lat_min, lat_max = map_extent
        map_area = (lon_max - lon_min) * (lat_max - lat_min)
        
        if map_area > 0 and num_points > 0:
            point_density = num_points / map_area
        else:
            return 5.0 
        
        if point_density < 1:
            base_size = 8.0   
        elif point_density < 10:
            base_size = 3.0  
        elif point_density < 50:
            base_size = 1.5 
        elif point_density < 150:
            base_size = 0.8   
        elif point_density < 500:
            base_size = 0.4   
        else:
            base_size = 0.25 
        
        if map_area < 50: 
            area_scale = max(8.0, 400.0 / map_area) 
        elif map_area < 500:  
            area_scale = max(3.0, 150.0 / map_area)  
        elif map_area < 5000: 
            area_scale = max(1.5, 50.0 / map_area) 
        else:  
            area_scale = max(0.5, 10.0 / map_area)  
        
        fig_scale = min(1.2, max(0.8, (fig_size[0] * fig_size[1]) / (10 * 12)))
        
        marker_size = base_size * area_scale * fig_scale    
        marker_size = max(0.1, min(20.0, marker_size))
        
        print(f"Adaptive marker sizing: area={map_area:.1f}°², density={point_density:.2f} pts/°², base={base_size:.2f}, scales=({area_scale:.2f}×{fig_scale:.2f}), final_size={marker_size:.2f}")
        
        return marker_size

    def _format_ticks_dynamic(self, ticks: List[float]) -> List[str]:
        """
        Choose a sensible numeric format for tick labels based on tick spacing and magnitude.

        Parameters:
            ticks (list of float): Tick values to format.

        Returns:
            list of str: Formatted tick strings following heuristics for precision.

        Rules:
            - If spacing < 0.01 -> use 3 decimal places
            - If spacing < 0.1 -> use 2 decimal places
            - If spacing < 1 -> use 1 decimal place
            - Otherwise use no decimals unless values are fractional, then 2 decimals
        """
        if not ticks:
            return []

        t = np.array(ticks)
        if len(t) > 1:
            spacings = np.abs(np.diff(np.sort(t)))
            median_spacing = float(np.median(spacings[spacings > 0])) if np.any(spacings > 0) else 0.0
        else:
            median_spacing = 0.0

        if median_spacing > 0 and median_spacing < 0.01:
            fmt = '{:.3f}'
        elif median_spacing >= 0.01 and median_spacing < 0.1:
            fmt = '{:.2f}'
        elif median_spacing >= 0.1 and median_spacing < 1:
            fmt = '{:.1f}'
        else:
            if np.allclose(t, np.round(t), atol=1e-6):
                fmt = '{:.0f}'
            else:
                fmt = '{:.2f}'

        return [fmt.format(x) for x in ticks]

    def _choose_wind_subsample(self, lon: np.ndarray, lat: np.ndarray, plot_type: str = 'barbs', max_vectors: Optional[int] = None) -> int:
        """
        Choose an automatic subsample factor for wind plotting to limit plotted vectors.

        Parameters:
            lon (np.ndarray): Input longitude array (1D or 2D).
            lat (np.ndarray): Input latitude array (1D or 2D).
            plot_type (str): Plot type, e.g., 'barbs' or 'arrows'.
            max_vectors (Optional[int]): Maximum number of wind vectors to plot.

        Returns:
            int: Subsample step (>=1). A value of 1 means plot all vectors.
        """
        if max_vectors is None:
            if (plot_type or '').lower() == 'barbs':
                max_vectors = 30  
            else:
                max_vectors = 50  

        try:
            n_points = int(np.size(lon))
        except Exception:
            n_points = int(np.size(lat)) if lat is not None else 0

        if n_points <= 0:
            return 1

        if n_points <= max_vectors:
            return 1

        step = int(np.ceil(np.sqrt(n_points / max_vectors)))
        step = max(1, step)

        if n_points > max_vectors * 25:
            step = int(step * 2)

        print(f"Auto wind subsample: n_points={n_points}, max_vectors={max_vectors}, step={step}")
        return step
    
    def setup_map_projection(self, lon_min: float, lon_max: float, 
                           lat_min: float, lat_max: float,
                           projection: str = 'PlateCarree') -> Tuple[ccrs.Projection, ccrs.PlateCarree]:
        """
        Set up map projection and data coordinate system.

        Parameters:
            lon_min (float): Minimum longitude bound.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Minimum latitude bound.
            lat_max (float): Maximum latitude bound.
            projection (str): Map projection name.

        Returns:
            Tuple[ccrs.Projection, ccrs.PlateCarree]: Map projection and data CRS.
        """
        central_lon = (lon_min + lon_max) / 2
        central_lat = (lat_min + lat_max) / 2
        
        if projection.lower() == 'platecarree':
            map_proj = ccrs.PlateCarree(central_longitude=central_lon)
        elif projection.lower() == 'mercator':
            map_proj = ccrs.Mercator(central_longitude=central_lon)
        elif projection.lower() == 'lambertconformal':
            map_proj = ccrs.LambertConformal(central_longitude=central_lon, central_latitude=central_lat)
        else:
            map_proj = ccrs.PlateCarree(central_longitude=central_lon)
        
        data_crs = ccrs.PlateCarree()
        
        return map_proj, data_crs
    
    def create_precipitation_map(self, 
                               lon: np.ndarray, 
                               lat: np.ndarray, 
                               precip_data: np.ndarray,
                               lon_min: float, 
                               lon_max: float, 
                               lat_min: float, 
                               lat_max: float,
                               title: str = "MPAS Precipitation",
                               accum_period: str = "a01h",
                               colormap: Optional[str] = None,
                               levels: Optional[List[float]] = None,
                               clim_min: Optional[float] = None,
                               clim_max: Optional[float] = None,
                               projection: str = 'PlateCarree',
                               time_end: Optional[datetime] = None,
                               time_start: Optional[datetime] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a precipitation map with professional cartographic presentation.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            precip_data (np.ndarray): Precipitation data values.
            lon_min (float): Minimum longitude bound for map extent.
            lon_max (float): Maximum longitude bound for map extent.
            lat_min (float): Minimum latitude bound for map extent.
            lat_max (float): Maximum latitude bound for map extent.
            title (str): Plot title.
            accum_period (str): Accumulation period for colormap selection.
            colormap (Optional[str]): Custom colormap name.
            levels (Optional[List[float]]): Custom contour levels.
            clim_min (Optional[float]): Minimum color limit.
            clim_max (Optional[float]): Maximum color limit.
            projection (str): Map projection.
            time_end (Optional[datetime]): End time for accumulation period.
            time_start (Optional[datetime]): Start time for accumulation period.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)
        
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
        
        if colormap and levels:
            cmap = plt.get_cmap(colormap)
            color_levels = levels
        elif colormap:
            cmap = plt.get_cmap(colormap)
            cmap_obj, color_levels = self.create_colormap(accum_period)
        else:
            cmap, color_levels = self.create_colormap(accum_period)
        
        if clim_min is not None and clim_max is not None:
            color_levels = [level for level in color_levels if clim_min <= level <= clim_max]
            if clim_min not in color_levels:
                color_levels.insert(0, clim_min)
            if clim_max not in color_levels:
                color_levels.append(clim_max)
        
        color_levels_sorted = sorted(set([v for v in color_levels if np.isfinite(v)]))
        last_bound = max(color_levels_sorted) + 1
        bounds = [0] + color_levels_sorted + [last_bound]
        norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        
        valid_mask = (np.isfinite(precip_data) & 
                     (precip_data >= 0) & 
                     (precip_data < 1e5) &
                     (lon >= lon_min) & (lon <= lon_max) &
                     (lat >= lat_min) & (lat <= lat_max))
        
        if np.any(valid_mask):
            lon_valid = lon[valid_mask]
            lat_valid = lat[valid_mask]
            precip_valid = precip_data[valid_mask]
            
            map_extent = (lon_min, lon_max, lat_min, lat_max)
            fig_size = (self.figsize[0], self.figsize[1])
            marker_size = self.calculate_adaptive_marker_size(map_extent, len(precip_valid), fig_size)
            
            map_area = (lon_max - lon_min) * (lat_max - lat_min)
            point_density = len(precip_valid) / map_area if map_area > 0 else 0
            
            if point_density > 1000:
                alpha_val = 0.9   
            elif point_density > 100:
                alpha_val = 0.9   
            else:
                alpha_val = 0.9  
            
            sort_indices = np.argsort(precip_valid)
            lon_sorted = lon_valid[sort_indices]
            lat_sorted = lat_valid[sort_indices]
            precip_sorted = precip_valid[sort_indices]
            
            scatter = self.ax.scatter(lon_sorted, lat_sorted, c=precip_sorted, 
                                   cmap=cmap, norm=norm, s=marker_size, alpha=alpha_val, 
                                   transform=data_crs, edgecolors='none')
            
            cbar = self.fig.colorbar(scatter, ax=self.ax, orientation='horizontal', extend='both',
                                   pad=0.06, shrink=0.8, aspect=30)
            cbar.set_label('Precipitation (mm)', fontsize=12, fontweight='bold', labelpad=-50)
            cbar.ax.tick_params(labelsize=10)
            
            if len(color_levels_sorted) <= 15: 
                cbar.set_ticks(color_levels_sorted)
                cbar.set_ticklabels(self._format_ticks_dynamic(color_levels_sorted))
        
        gl = self.ax.gridlines(crs=data_crs, draw_labels=True, 
                             linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if time_end is not None:
            if time_start is None:
                n_hours = get_accumulation_hours(accum_period)
                time_start = time_end - pd.Timedelta(hours=n_hours)
            
            start_utc = time_start.strftime('%Y-%m-%d %H:%M UTC')
            end_utc = time_end.strftime('%Y-%m-%d %H:%M UTC')
            n_hours = int((time_end - time_start).total_seconds() / 3600)
            
            txt = f"Accumulation: {start_utc} to {end_utc} ({n_hours} h)"
            self.ax.text(0.01, 0.02, txt, transform=self.ax.transAxes, fontsize=9,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        elif accum_period:
            accum_hours_map = {'a01h': '1-h', 'a03h': '3-h', 'a06h': '6-h', 'a12h': '12-h', 'a24h': '24-h'}
            accum_display = accum_hours_map.get(accum_period, accum_period)
            txt = f"Accumulation: {accum_display}"
            self.ax.text(0.01, 0.02, txt, transform=self.ax.transAxes, fontsize=9,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def create_surface_map(self,
                         lon: np.ndarray,
                         lat: np.ndarray,
                         data: np.ndarray,
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
                         grid_resolution_deg: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a 2D surface map for any MPAS forecast variable.

        Supports scatter and contour plots (with interpolation to regular grids)
        for visualizing MPAS surface variables like temperature, humidity, pressure,
        and wind speed.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            data (np.ndarray): Variable data values.
            var_name (str): Variable name for metadata lookup.
            lon_min (float): Minimum longitude bound for map extent.
            lon_max (float): Maximum longitude bound for map extent.
            lat_min (float): Minimum latitude bound for map extent.
            lat_max (float): Maximum latitude bound for map extent.
            title (Optional[str]): Custom plot title (auto-generated if None).
            plot_type (str): 'scatter' or 'contour'.
            colormap (Optional[str]): Custom colormap name.
            levels (Optional[List[float]]): Custom contour levels.
            clim_min (Optional[float]): Minimum color limit.
            clim_max (Optional[float]): Maximum color limit.
            projection (str): Map projection.
            time_stamp (Optional[datetime]): Time stamp for title.
            data_array (Optional[xr.DataArray]): Original data array for metadata extraction.
            grid_resolution (Optional[int]): Grid resolution (points per axis) for interpolation.
            grid_resolution_deg (Optional[float]): Grid resolution in degrees for interpolation.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.

        Raises:
            ValueError: If plot_type is not 'scatter' or 'contour'.
        """
        if plot_type not in ['scatter', 'contour']:
            raise ValueError(f"plot_type must be 'scatter' or 'contour', got '{plot_type}'")
        
        var_metadata = get_variable_metadata(var_name, data_array)
        
        if colormap is None:
            colormap = var_metadata['colormap']
        if levels is None:
            levels = var_metadata['levels']
        
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)
        
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
        
        if clim_min is not None and clim_max is not None:
            levels = [level for level in levels if clim_min <= level <= clim_max]
            if clim_min not in levels:
                levels.insert(0, clim_min)
            if clim_max not in levels:
                levels.append(clim_max)

        try:
            cmap_obj = plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
        except Exception:
            cmap_obj = plt.get_cmap('viridis')

        norm = None
        if clim_min is not None or clim_max is not None:
            vmin = clim_min if clim_min is not None else float(np.nanmin(data))
            vmax = clim_max if clim_max is not None else float(np.nanmax(data))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            try:
                color_levels_sorted = sorted(set([v for v in levels if np.isfinite(v)]))
                if color_levels_sorted:
                    last_bound = max(color_levels_sorted) + 1
                    bounds = [min(color_levels_sorted)] + color_levels_sorted + [last_bound]
                    norm = BoundaryNorm(bounds, ncolors=cmap_obj.N, clip=True)
            except Exception:
                norm = None
        
        valid_mask = (np.isfinite(data) & 
                     np.isfinite(lon) & np.isfinite(lat) &
                     (lon >= lon_min) & (lon <= lon_max) &
                     (lat >= lat_min) & (lat <= lat_max))
        
        if not np.any(valid_mask):
            print(f"Warning: No valid data points found for {var_name}")
            return self.fig, self.ax
            
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        data_valid = data[valid_mask]
        
        print(f"Plotting {len(data_valid):,} data points for {var_name}")
        print(f"Data range: {data_valid.min():.3f} to {data_valid.max():.3f} {var_metadata['units']}")
        
        if plot_type == 'scatter':
            map_extent = (lon_min, lon_max, lat_min, lat_max)
            fig_size = (self.figsize[0], self.figsize[1])
            marker_size = self.calculate_adaptive_marker_size(map_extent, len(data_valid), fig_size)
            
            map_area = (lon_max - lon_min) * (lat_max - lat_min)
            point_density = len(data_valid) / map_area if map_area > 0 else 0
            
            if point_density > 1000:
                alpha_val = 0.8
            elif point_density > 100:
                alpha_val = 0.9
            else:
                alpha_val = 0.9
            
            sort_indices = np.argsort(data_valid)
            lon_sorted = lon_valid[sort_indices]
            lat_sorted = lat_valid[sort_indices]
            data_sorted = data_valid[sort_indices]
            
            scatter = self.ax.scatter(lon_sorted, lat_sorted, c=data_sorted,
                                   cmap=cmap_obj, norm=norm, s=marker_size, alpha=alpha_val,
                                   transform=data_crs, edgecolors='none')
            
            cbar = self.fig.colorbar(scatter, ax=self.ax, orientation='horizontal', extend='both',
                                   pad=0.06, shrink=0.8, aspect=30)
            
        elif plot_type == 'contour':            
            try:
                from scipy.interpolate import griddata

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
                    print(f"Interpolating {len(data_valid)} points to {lon_mesh.shape[0]}x{lon_mesh.shape[1]} grid (~{step}° resolution)...")

                else:
                    if grid_resolution is None:
                        adaptive = int(np.sqrt(len(data_valid)) / 9)
                        grid_resolution = int(np.clip(adaptive, 50, 200))
                    else:
                        grid_resolution = int(grid_resolution)

                    grid_resolution = max(40, min(400, grid_resolution))

                    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
                    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
                    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

                    print(f"Interpolating {len(data_valid)} points to {grid_resolution}x{grid_resolution} grid...")

                data_interp = griddata(
                    (lon_valid, lat_valid), data_valid,
                    (lon_mesh, lat_mesh), method='linear', fill_value=np.nan
                )
                
                contourf = self.ax.contourf(lon_mesh, lat_mesh, data_interp, levels=levels,
                                         cmap=cmap_obj, norm=norm, transform=data_crs, extend='both')
                
                contour_lines = self.ax.contour(lon_mesh, lat_mesh, data_interp, levels=levels,
                                              colors='black', linewidths=0.5, alpha=0.3,
                                              transform=data_crs)
                
                cbar = self.fig.colorbar(contourf, ax=self.ax, orientation='horizontal', extend='both',
                                       pad=0.06, shrink=0.8, aspect=30)
                try:
                    if levels is not None and len(levels) > 0:
                        tick_levels = list(levels)
                        cbar.set_ticks(tick_levels)
                        cbar.set_ticklabels(self._format_ticks_dynamic(tick_levels))
                    else:
                        ticks = cbar.get_ticks().tolist()
                        cbar.set_ticks(ticks)
                        cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
                except Exception:
                    pass
                
            except ImportError:
                print("Warning: scipy not available for contour plotting. Falling back to scatter plot.")
                return self.create_surface_map(lon, lat, data, var_name, lon_min, lon_max, 
                                             lat_min, lat_max, title, 'scatter', colormap, 
                                             levels, clim_min, clim_max, projection, time_stamp, data_array)
        
        units_str = f" ({var_metadata['units']})" if var_metadata['units'] else ""
        cbar.set_label(f"{var_metadata['long_name']}{units_str}", fontsize=12, fontweight='bold', labelpad=-50)
        cbar.ax.tick_params(labelsize=10)
        
        if plot_type == 'scatter' and levels is not None and len(levels) > 0:
            tick_levels = list(levels)
            cbar.set_ticks(tick_levels)
            cbar.set_ticklabels(self._format_ticks_dynamic(tick_levels))
        
        gl = self.ax.gridlines(crs=data_crs, draw_labels=True,
                             linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        if title is None:
            title = f"MPAS {var_metadata['long_name']}"
            if time_stamp:
                time_str = time_stamp.strftime('%Y%m%dT%H')
                title += f" | Valid Time: {time_str}"
            title += f" | Plot Type: {plot_type.title()}"
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def create_simple_scatter_plot(self,
                                 lon: np.ndarray,
                                 lat: np.ndarray, 
                                 data: np.ndarray,
                                 title: str = "MPAS Data",
                                 colorbar_label: str = "Value",
                                 colormap: str = 'viridis',
                                 point_size: float = 1.0) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a simple scatter plot of MPAS data.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            data (np.ndarray): Data values.
            title (str): Plot title.
            colorbar_label (str): Colorbar label.
            colormap (str): Colormap name.
            point_size (float): Size of scatter points.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        valid_mask = np.isfinite(data) & np.isfinite(lon) & np.isfinite(lat)
        
        if np.any(valid_mask):
            scatter = self.ax.scatter(lon[valid_mask], lat[valid_mask], 
                                   c=data[valid_mask], cmap=colormap, 
                                   s=point_size, alpha=0.7, edgecolors='none')
            
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            try:
                ticks = cbar.get_ticks().tolist()
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
            except Exception:
                pass
            cbar.set_label(colorbar_label, fontsize=12)
        
        self.ax.set_xlabel('Longitude', fontsize=12)
        self.ax.set_ylabel('Latitude', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def save_plot(self, 
                  output_path: str, 
                  formats: List[str] = ['png'],
                  bbox_inches: str = 'tight',
                  pad_inches: float = 0.1) -> None:
        """
        Save the current plot to file(s).

        Parameters:
            output_path (str): Base output path (without extension).
            formats (List[str]): List of output formats.
            bbox_inches (str): Bounding box mode.
            pad_inches (float): Padding around the figure.

        Returns:
            None
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            full_path = f"{output_path}.{fmt}"
            self.fig.savefig(full_path, dpi=self.dpi, bbox_inches=bbox_inches, 
                           pad_inches=pad_inches, format=fmt)
            print(f"Saved plot: {full_path}")
    
    def close_plot(self) -> None:
        """Close the current figure to free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def create_time_series_plot(self,
                              times: List[datetime],
                              values: List[float],
                              title: str = "Time Series",
                              ylabel: str = "Value",
                              xlabel: str = "Time") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a time series plot.

        Parameters:
            times (list of datetime): Time values.
            values (list of float): Data values.
            title (str): Plot title.
            ylabel (str): Y-axis label.
            xlabel (str): X-axis label.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        self.ax.plot(times, values, linewidth=2, marker='o', markersize=4)
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.autofmt_xdate()
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def create_histogram(self,
                        data: np.ndarray,
                        bins: Union[int, np.ndarray] = 50,
                        title: str = "Data Distribution",
                        xlabel: str = "Value",
                        ylabel: str = "Frequency",
                        log_scale: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a histogram of data values.

        Parameters:
            data (np.ndarray): Data values.
            bins (int or np.ndarray): Number of bins or bin edges.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            log_scale (bool): Use logarithmic scale for y-axis.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) > 0:
            n, bins, patches = self.ax.hist(valid_data, bins=bins, alpha=0.7, 
                                          edgecolor='black', linewidth=0.5)
            
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            self.ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.2f}')
            self.ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1, 
                          label=f'Mean ± Std: {mean_val-std_val:.2f} to {mean_val+std_val:.2f}')
            self.ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1)
            
            self.ax.legend()
        
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        if log_scale:
            self.ax.set_yscale('log')
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        
        return self.fig, self.ax

    def create_wind_plot(self, lon: np.ndarray, lat: np.ndarray, 
                        u_data: np.ndarray, v_data: np.ndarray,
                        lon_min: float, lon_max: float, 
                        lat_min: float, lat_max: float,
                        wind_level: str = "surface",
                        plot_type: str = "barbs",
                        subsample: int = 0,
                        scale: Optional[float] = None,
                        show_background: bool = False,
                        bg_colormap: str = "viridis",
                        title: Optional[str] = None,
                        time_stamp: Optional[object] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a wind vector plot using MPAS data.

        Parameters:
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            u_data (np.ndarray): U-component wind data.
            v_data (np.ndarray): V-component wind data.
            lon_min (float): Minimum longitude for extent.
            lon_max (float): Maximum longitude for extent.
            lat_min (float): Minimum latitude for extent.
            lat_max (float): Maximum latitude for extent.
            wind_level (str): Wind level description for labeling (default: 'surface').
            plot_type (str): 'barbs' or 'arrows' (default: 'barbs').
            subsample (int): Subsample factor for wind vectors (<=0 means auto).
            scale (Optional[float]): Scale factor for wind vectors (auto if None).
            show_background (bool): Show background wind speed contours.
            bg_colormap (str): Colormap for background.
            title (Optional[str]): Custom plot title.
            time_stamp (Optional[object]): Time stamp used for title formatting.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        wind_speed = np.sqrt(u_data**2 + v_data**2)

        is_gridded = all(arr is not None and getattr(arr, 'ndim', 1) == 2 for arr in (lon, lat, u_data, v_data))

        if show_background:
            if is_gridded:
                levels = np.linspace(0, np.percentile(wind_speed, 95), 15)
                cs = ax.contourf(lon, lat, wind_speed, levels=levels,
                                 cmap=bg_colormap, alpha=0.6, transform=ccrs.PlateCarree())

                cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
            else:
                lon_flat = np.ravel(lon)
                lat_flat = np.ravel(lat)
                wind_flat = np.ravel(wind_speed)

                sc = ax.scatter(lon_flat, lat_flat, c=wind_flat, cmap=bg_colormap,
                                alpha=0.6, s=12, transform=ccrs.PlateCarree(), edgecolors='none')
                cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)

            try:
                ticks = cbar.get_ticks().tolist()
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(self._format_ticks_dynamic(ticks))
            except Exception:
                pass
            cbar.set_label('Wind Speed (m/s)', fontsize=10)
        
        if subsample <= 0:
            subsample = self._choose_wind_subsample(lon, lat, plot_type=plot_type)
        subsample = max(1, int(subsample))

        if is_gridded:
            if subsample > 1:
                lon_sub = lon[::subsample, ::subsample]
                lat_sub = lat[::subsample, ::subsample]
                u_sub = u_data[::subsample, ::subsample]
                v_sub = v_data[::subsample, ::subsample]
            else:
                lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
        else:
            lon_flat = np.ravel(lon)
            lat_flat = np.ravel(lat)
            u_flat = np.ravel(u_data)
            v_flat = np.ravel(v_data)

            if subsample > 1:
                lon_sub = lon_flat[::subsample]
                lat_sub = lat_flat[::subsample]
                u_sub = u_flat[::subsample]
                v_sub = v_flat[::subsample]
            else:
                lon_sub, lat_sub, u_sub, v_sub = lon_flat, lat_flat, u_flat, v_flat
        
        if plot_type.lower() == "barbs":
            ax.barbs(lon_sub, lat_sub, u_sub, v_sub, 
                    length=6, barbcolor='black', flagcolor='red',
                    linewidth=0.8, transform=ccrs.PlateCarree())
        elif plot_type.lower() == "arrows":
            if scale is None:
                scale = float(np.percentile(wind_speed, 90)) * 15
                
            ax.quiver(lon_sub, lat_sub, u_sub, v_sub,
                     scale=scale, scale_units='xy', angles='xy',
                     color='black', width=0.003, alpha=0.8,
                     transform=ccrs.PlateCarree())
        
        if title is None:
            if time_stamp is not None:
                try:
                    ts = pd.to_datetime(time_stamp)
                    time_str = ts.strftime('%Y%m%dT%H')
                    title = f"MPAS {wind_level.title()} Wind Vectors | Valid: {time_str}"
                except Exception:
                    title = f"MPAS {wind_level.title()} Wind Vectors | Valid: {time_stamp}"
            else:
                title = f"MPAS {wind_level.title()} Wind Vectors"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        wind_stats = f"""Wind Statistics:
    Max: {wind_speed.max():.1f} m/s
    Mean: {wind_speed.mean():.1f} m/s
    Min: {wind_speed.min():.1f} m/s"""
        
        ax.text(0.02, 0.98, wind_stats, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
        
        if plot_type.lower() == "barbs":
            legend_text = """Wind Barbs:
    Half barb = 2.5 m/s (5 kt)
    Full barb = 5 m/s (10 kt)  
    Flag = 25 m/s (50 kt)"""
        else:
            legend_text = f"Wind Arrows\nScale: {scale:.0f} units"
        
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.fig = fig
        self.ax = ax
        
        self.add_timestamp_and_branding()
        
        plt.tight_layout()
        return fig, ax


def create_batch_precipitation_maps(processor, visualizer, output_dir: str,
                                  lon_min: float, lon_max: float,
                                  lat_min: float, lat_max: float,
                                  var_name: str = 'rainnc',
                                  accum_period: str = 'a01h',
                                  file_prefix: str = 'mpas_precipitation_map',
                                  formats: List[str] = ['png']) -> List[str]:
    """
    Create precipitation maps for all time steps in batch mode.

    Parameters:
        processor: MPASDataProcessor instance with loaded data.
        visualizer: MPASVisualizer instance.
        output_dir (str): Output directory for plots.
        lon_min (float): Minimum longitude bound.
        lon_max (float): Maximum longitude bound.
        lat_min (float): Minimum latitude bound.
        lat_max (float): Maximum latitude bound.
        var_name (str): Precipitation variable name.
        accum_period (str): Accumulation period.
        file_prefix (str): Prefix for output filenames.
        formats (List[str]): Output file formats.

    Returns:
        List[str]: List of created file paths.
    """
    if processor.dataset is None:
        raise ValueError("No data loaded in processor")
    
    lon, lat = processor.extract_spatial_coordinates()
    
    time_dim = 'Time' if 'Time' in processor.dataset.dims else 'time'
    total_times = processor.dataset.sizes[time_dim]
    
    created_files = []
    
    print(f"\nCreating precipitation maps for {total_times} time steps...")
    
    for time_idx in range(total_times):
        try:
            if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                time_end = pd.to_datetime(processor.dataset.Time.values[time_idx])
                time_str = time_end.strftime('%Y%m%dT%H')
            else:
                time_end = None
                time_str = f"t{time_idx:03d}"
            
            precip_data = processor.compute_precipitation_difference(time_idx, var_name)
            
            title = f"MPAS Precipitation | VarType: {var_name.upper()} | Valid Time: {time_str}"
            fig, ax = visualizer.create_precipitation_map(
                lon, lat, precip_data.values,
                lon_min, lon_max, lat_min, lat_max,
                title=title,
                accum_period=accum_period,
                time_end=time_end
            )
            
            output_path = os.path.join(output_dir, f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_point")
            visualizer.save_plot(output_path, formats=formats)
            
            for fmt in formats:
                created_files.append(f"{output_path}.{fmt}")
            
            visualizer.close_plot()
            
            if (time_idx + 1) % 10 == 0:
                print(f"Completed {time_idx + 1}/{total_times} maps...")
                
        except Exception as e:
            print(f"Error creating map for time index {time_idx}: {e}")
            continue
    
    print(f"\nBatch processing completed. Created {len(created_files)} files.")
    return created_files


def create_batch_surface_maps(processor, visualizer, output_dir: str,
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
    Create surface maps for all time steps in batch mode.

    Mirrors `create_batch_precipitation_maps` but calls `create_surface_map` for
    each time step.

    Parameters:
        processor: MPASDataProcessor instance with loaded data.
        visualizer: MPASVisualizer instance.
        output_dir (str): Output directory for plots.
        lon_min (float): Minimum longitude bound.
        lon_max (float): Maximum longitude bound.
        lat_min (float): Minimum latitude bound.
        lat_max (float): Maximum latitude bound.
        var_name (str): Variable name to plot.
        plot_type (str): 'scatter' or 'contour'.
        file_prefix (str): Prefix for output filenames.
        formats (List[str]): Output formats.
        grid_resolution (Optional[int]): Grid resolution for interpolation.
        grid_resolution_deg (Optional[float]): Grid resolution in degrees.
        clim_min (Optional[float]): Minimum color limit.
        clim_max (Optional[float]): Maximum color limit.

    Returns:
        List[str]: List of created file paths.
    """
    if processor.dataset is None:
        raise ValueError("No data loaded in processor")

    lon, lat = processor.extract_spatial_coordinates()

    time_dim = 'Time' if 'Time' in processor.dataset.dims else 'time'
    total_times = processor.dataset.sizes[time_dim]

    created_files = []
    print(f"\nCreating surface maps for {total_times} time steps...")

    for time_idx in range(total_times):
        try:
            if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                time_end = pd.to_datetime(processor.dataset.Time.values[time_idx])
                time_str = time_end.strftime('%Y%m%dT%H')
            else:
                time_end = None
                time_str = f"t{time_idx:03d}"

            var_data = processor.get_variable_data(var_name, time_idx)

            title = f"MPAS Surface | Var: {var_name} | Valid Time: {time_str}"

            data_array = None
            try:
                data_array = processor.dataset[var_name].isel(Time=time_idx) if hasattr(processor.dataset, var_name) else None
            except Exception:
                data_array = None

            fig, ax = visualizer.create_surface_map(
                lon, lat, var_data.values,
                var_name,
                lon_min, lon_max, lat_min, lat_max,
                title=title,
                plot_type=plot_type,
                time_stamp=time_end,
                data_array=data_array,
                grid_resolution=grid_resolution,
                grid_resolution_deg=grid_resolution_deg,
                clim_min=clim_min,
                clim_max=clim_max
            )

            output_path = os.path.join(output_dir, f"{file_prefix}_{var_name}_{plot_type}_{time_str}")
            visualizer.save_plot(output_path, formats=formats)

            for fmt in formats:
                created_files.append(f"{output_path}.{fmt}")

            visualizer.close_plot()

            if (time_idx + 1) % 10 == 0:
                print(f"Completed {time_idx + 1}/{total_times} maps...")

        except Exception as e:
            print(f"Error creating surface map for time index {time_idx}: {e}")
            continue

    print(f"\nBatch processing completed. Created {len(created_files)} files.")
    return created_files


def create_batch_wind_plots(processor, visualizer, output_dir: str,
                            lon_min: float, lon_max: float,
                            lat_min: float, lat_max: float,
                            u_variable: str = 'u10',
                            v_variable: str = 'v10',
                            plot_type: str = 'barbs',
                            file_prefix: str = 'mpas_wind',
                            formats: List[str] = ['png'],
                            subsample: int = 0,
                            scale: Optional[float] = None,
                            show_background: bool = False,
                            background_colormap: str = 'viridis') -> List[str]:
    """
    Create wind vector plots for all time steps in batch mode.

    Parameters:
        processor: MPASDataProcessor instance with loaded data.
        visualizer: MPASVisualizer instance.
        output_dir (str): Output directory for plots.
        lon_min (float): Minimum longitude bound.
        lon_max (float): Maximum longitude bound.
        lat_min (float): Minimum latitude bound.
        lat_max (float): Maximum latitude bound.
        u_variable (str): U-component variable name.
        v_variable (str): V-component variable name.
        plot_type (str): 'barbs' or 'arrows'.
        file_prefix (str): Prefix for output filenames.
        formats (List[str]): Output formats.
        subsample (int): Subsample factor for wind vectors (<=0 means auto).
        scale (Optional[float]): Scale for quiver arrows.
        show_background (bool): Whether to show wind speed background.
        background_colormap (str): Colormap for background wind speed.

    Returns:
        List[str]: List of created file paths.
    """
    if processor.dataset is None:
        raise ValueError("No data loaded in processor")

    lon, lat = processor.extract_spatial_coordinates()

    time_dim = 'Time' if 'Time' in processor.dataset.dims else 'time'
    total_times = processor.dataset.sizes[time_dim]

    created_files = []
    print(f"\nCreating wind plots for {total_times} time steps...")

    for time_idx in range(total_times):
        try:
            if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                time_end = pd.to_datetime(processor.dataset.Time.values[time_idx])
                time_str = time_end.strftime('%Y%m%dT%H')
            else:
                time_end = None
                time_str = f"t{time_idx:03d}"

            u_data, v_data = processor.get_wind_components(u_variable, v_variable, time_idx)

            title = f"MPAS {plot_type.title()} Wind | Valid: {time_str}"

            fig, ax = visualizer.create_wind_plot(
                lon, lat, u_data.values, v_data.values,
                lon_min, lon_max, lat_min, lat_max,
                wind_level='surface',
                plot_type=plot_type,
                title=title,
                subsample=subsample,
                scale=scale,
                show_background=show_background,
                bg_colormap=background_colormap,
                time_stamp=time_end
            )

            output_path = os.path.join(output_dir, f"{file_prefix}_{plot_type}_{time_str}")
            visualizer.save_plot(output_path, formats=formats)

            for fmt in formats:
                created_files.append(f"{output_path}.{fmt}")

            visualizer.close_plot()

            if (time_idx + 1) % 10 == 0:
                print(f"Completed {time_idx + 1}/{total_times} plots...")

        except Exception as e:
            print(f"Error creating wind plot for time index {time_idx}: {e}")
            continue

    print(f"\nBatch processing completed. Created {len(created_files)} files.")
    return created_files


def validate_plot_parameters(lon_min: float, lon_max: float, 
                           lat_min: float, lat_max: float) -> bool:
    """
    Validate plotting parameters.

    Parameters:
        lon_min (float): Minimum longitude bound.
        lon_max (float): Maximum longitude bound.
        lat_min (float): Minimum latitude bound.
        lat_max (float): Maximum latitude bound.

    Returns:
        bool: True if parameters are valid.
    """
    return (
        -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0
        and -90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0
        and lon_max > lon_min and lat_max > lat_min
    )


def get_color_levels_for_variable(var_name: str, accum_period: str = 'a01h') -> List[float]:
    """
    Get appropriate color levels for different variables and accumulation periods.

    Parameters:
        var_name (str): Variable name.
        accum_period (str): Accumulation period.

    Returns:
        List[float]: Color levels.
    """
    hours = 1
    try:
        m = re.search(r"(\d+)", accum_period)
        if m:
            hours = int(m.group(1))
    except:
        pass
    
    if var_name.lower() in ['rainc', 'rainnc', 'total']:
        if hours >= 12:
            return [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]
        else:
            return [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75]
    else:
        return [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]


def get_variable_metadata(var_name: str, data_array: Optional[xr.DataArray] = None) -> dict:
    """
    Get metadata for MPAS forecast variables including units, names, and colormaps.

    Parameters:
        var_name (str): Variable name.
        data_array (Optional[xr.DataArray]): Data array containing variable attributes.

    Returns:
        dict: Variable metadata including 'units', 'long_name', 'colormap', and 'levels'.
    """
    standard_metadata = {
        't2m': {'units': 'K', 'long_name': '2-meter Temperature', 'colormap': 'RdYlBu_r', 'levels': list(range(250, 320, 5))},
        'temperature': {'units': 'K', 'long_name': 'Temperature', 'colormap': 'RdYlBu_r', 'levels': list(range(250, 320, 5))},
        'theta': {'units': 'K', 'long_name': 'Potential Temperature', 'colormap': 'RdYlBu_r', 'levels': list(range(280, 350, 5))},
        
        'surface_pressure': {'units': 'Pa', 'long_name': 'Surface Pressure', 'colormap': 'viridis', 'levels': list(range(95000, 105000, 500))},
        'mslp': {'units': 'Pa', 'long_name': 'Mean Sea Level Pressure', 'colormap': 'viridis', 'levels': list(range(98000, 104000, 200))},
        'pressure': {'units': 'Pa', 'long_name': 'Pressure', 'colormap': 'viridis', 'levels': list(range(90000, 105000, 1000))},
        
        'q2': {'units': 'kg/kg', 'long_name': '2-meter Specific Humidity', 'colormap': 'Blues', 'levels': [i*0.002 for i in range(0, 15)]},
        'rh2m': {'units': '%', 'long_name': '2-meter Relative Humidity', 'colormap': 'Blues', 'levels': list(range(0, 105, 10))},
        'qv': {'units': 'kg/kg', 'long_name': 'Water Vapor Mixing Ratio', 'colormap': 'Blues', 'levels': [i*0.002 for i in range(0, 15)]},
        
        'u10': {'units': 'm/s', 'long_name': '10-meter U-wind', 'colormap': 'RdBu_r', 'levels': list(range(-20, 25, 2))},
        'v10': {'units': 'm/s', 'long_name': '10-meter V-wind', 'colormap': 'RdBu_r', 'levels': list(range(-20, 25, 2))},
        'wspd10': {'units': 'm/s', 'long_name': '10-meter Wind Speed', 'colormap': 'plasma', 'levels': list(range(0, 25, 2))},
        'uReconstructZonal': {'units': 'm/s', 'long_name': 'Zonal Wind', 'colormap': 'RdBu_r', 'levels': list(range(-30, 35, 5))},
        'uReconstructMeridional': {'units': 'm/s', 'long_name': 'Meridional Wind', 'colormap': 'RdBu_r', 'levels': list(range(-30, 35, 5))},
        
        'rainc': {'units': 'mm', 'long_name': 'Convective Precipitation', 'colormap': 'Blues', 'levels': [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]},
        'rainnc': {'units': 'mm', 'long_name': 'Non-Convective Precipitation', 'colormap': 'Blues', 'levels': [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]},
        
        'cldfrac': {'units': '', 'long_name': 'Cloud Fraction', 'colormap': 'gray', 'levels': [i*0.1 for i in range(0, 11)]},
        'cldtop': {'units': 'Pa', 'long_name': 'Cloud Top Pressure', 'colormap': 'viridis_r', 'levels': list(range(10000, 100000, 5000))},
        
        'swdown': {'units': 'W/m^2', 'long_name': 'Downward Shortwave Radiation', 'colormap': 'plasma', 'levels': list(range(0, 1000, 50))},
        'lwdown': {'units': 'W/m^2', 'long_name': 'Downward Longwave Radiation', 'colormap': 'inferno', 'levels': list(range(200, 500, 20))},
        'swup': {'units': 'W/m^2', 'long_name': 'Upward Shortwave Radiation', 'colormap': 'plasma', 'levels': list(range(0, 800, 40))},
        'lwup': {'units': 'W/m^2', 'long_name': 'Upward Longwave Radiation', 'colormap': 'inferno', 'levels': list(range(200, 600, 25))},
        
        'skintemp': {'units': 'K', 'long_name': 'Skin Temperature', 'colormap': 'RdYlBu_r', 'levels': list(range(250, 320, 5))},
        'sst': {'units': 'K', 'long_name': 'Sea Surface Temperature', 'colormap': 'RdYlBu_r', 'levels': list(range(270, 310, 2))},
        'snowc': {'units': '', 'long_name': 'Snow Cover', 'colormap': 'Blues_r', 'levels': [i*0.1 for i in range(0, 11)]},
        'snowh': {'units': 'm', 'long_name': 'Snow Height', 'colormap': 'Blues_r', 'levels': [i*0.1 for i in range(0, 21)]},
    }
    
    metadata = standard_metadata.get(var_name.lower(), {
        'units': '',
        'long_name': var_name,
        'colormap': 'viridis',
        'levels': list(range(0, 21))
    })
    
    if data_array is not None:
        if hasattr(data_array, 'attrs'):
            if 'units' in data_array.attrs:
                metadata['units'] = data_array.attrs['units']
            if 'long_name' in data_array.attrs:
                metadata['long_name'] = data_array.attrs['long_name']
            elif 'standard_name' in data_array.attrs:
                metadata['long_name'] = data_array.attrs['standard_name']
                
        try:
            data_values = data_array.values
            if hasattr(data_values, 'flatten'):
                flat_data = data_values.flatten()
                finite_data = flat_data[np.isfinite(flat_data)]
                if len(finite_data) > 0:
                    data_min = float(np.percentile(finite_data, 5))
                    data_max = float(np.percentile(finite_data, 95))
                    if data_max > data_min:
                        n_levels = 15
                        level_step = (data_max - data_min) / n_levels
                        metadata['levels'] = [data_min + i * level_step for i in range(n_levels + 1)]
        except Exception:
            pass
    
    return metadata


def get_surface_colormap_and_levels(var_name: str, data_array: Optional[xr.DataArray] = None) -> Tuple[str, List[float]]:
    """
    Get appropriate colormap and levels for surface variables.

    Parameters:
        var_name (str): Variable name.
        data_array (Optional[xr.DataArray]): Data array for automatic level detection.

    Returns:
        Tuple[str, List[float]]: Colormap name and contour levels.
    """
    metadata = get_variable_metadata(var_name, data_array)
    return metadata['colormap'], metadata['levels']
