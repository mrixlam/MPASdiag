#!/usr/bin/env python3

"""
MPAS Precipitation Visualization

This module provides specialized precipitation visualization functionality for MPAS model output implementing the exact plotting logic and color schemes from the original mpas_analysis module for backward compatibility. It includes the MPASPrecipitationPlotter class that creates professional cartographic precipitation maps with discrete colormaps optimized for meteorological precipitation display, accumulation-period-specific contour levels (hourly vs daily), automatic unit conversion from model output to millimeters, and seamless integration with the modern MPASdiag visualization architecture. The plotter supports single precipitation maps for individual time steps, batch processing for creating time series of precipitation analyses, and multi-panel comparison plots for model-observation evaluation. Core capabilities include period-aware color scheme selection, scatter plot rendering of MPAS unstructured mesh data, geographic feature overlays, and publication-quality output with timestamps and colorbars suitable for operational weather analysis and climate model validation.

Classes:
    MPASPrecipitationPlotter: Specialized class for creating precipitation visualizations from MPAS model output with meteorologically-appropriate styling.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from datetime import datetime
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
from typing import Tuple, Optional, List, Any, Union

from .base_visualizer import MPASVisualizer
from .styling import MPASVisualizationStyle
from ..processing.utils_unit import UnitConverter
from ..processing.utils_metadata import MPASFileMetadata
from ..diagnostics.precipitation import PrecipitationDiagnostics

warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')


class MPASPrecipitationPlotter(MPASVisualizer):
    """
    Specialized plotter for creating professional precipitation visualizations from MPAS model output with meteorologically-appropriate color schemes and accumulation period handling. This class extends MPASVisualizer to provide comprehensive functionality for rendering precipitation fields on cartographic maps using MPAS unstructured mesh data, implementing the exact visualization logic and color schemes from the original mpas_analysis module for backward compatibility while leveraging modern MPASdiag architecture. The plotter supports multiple accumulation periods (hourly, daily, etc.) with period-specific contour levels and discrete colormaps following meteorological conventions, automated unit conversion from model output to display units (mm), flexible map projections via Cartopy, and geographic feature overlays (coastlines, borders, terrain). Visualization outputs include publication-quality single-panel precipitation maps, multi-panel comparison plots for model-observation evaluation, and batch processing capabilities for creating time series of precipitation analyses with consistent styling and automatic file naming.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (10, 14), dpi: int = 100) -> None:
        """
        Initialize the MPAS precipitation plotter with figure dimensions and resolution settings for cartographic precipitation map generation. This constructor calls the parent MPASVisualizer initializer to set up the matplotlib figure and axes infrastructure, establishing default figure size optimized for vertical precipitation maps with colorbar legends and geographic context. The dpi parameter controls output resolution for both screen display and saved image files, with default 300 dpi suitable for publication-quality plots. This initialization prepares the plotter instance for creating single or batch precipitation visualizations using MPAS unstructured mesh data with proper map projections and precipitation-specific color schemes.

        Parameters:
            figsize (Tuple[float, float]): Figure dimensions as (width, height) in inches, optimized for precipitation maps with legends (default: (10, 14)).
            dpi (int): Figure resolution in dots-per-inch for rendering and output file quality (default: 300).

        Returns:
            None: Initializes parent class attributes including figure, axes, and style configurations.
        """
        super().__init__(figsize, dpi)
    
    def create_precip_colormap(self, accum: str = "a24h") -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        Create a discrete colormap and contour level specifications tailored for precipitation visualization with accumulation-period-specific color schemes. This method delegates to MPASVisualizationStyle to generate a ListedColormap and associated contour levels that follow meteorological conventions for precipitation display, using blue-to-purple color gradients optimized for rainfall intensity representation. The accumulation period string determines the contour level spacing and range, with finer intervals for hourly accumulations and broader intervals for daily totals. This colormap design ensures consistent, publication-quality precipitation visualization across all MPAS precipitation diagnostics, matching the original mpas_analysis module's precipitation color schemes for backward compatibility.

        Parameters:
            accum (str): Accumulation period identifier (e.g., 'a24h' for 24-hour, 'a01h' for 1-hour) determining contour level selection (default: "a24h").

        Returns:
            Tuple[matplotlib.colors.ListedColormap, List[float]]: Two-element tuple containing (discrete_colormap, contour_levels_array) for precipitation plotting with BoundaryNorm.
        """
        return MPASVisualizationStyle.create_precip_colormap(accum)
    
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
                               time_start: Optional[datetime] = None,
                               data_array: Optional[xr.DataArray] = None,
                               var_name: str = 'precipitation') -> Tuple[Figure, Axes]:
        """
        Create a professional cartographic precipitation map from MPAS unstructured mesh data with precipitation-specific color schemes and geographic context. This method implements the complete precipitation visualization workflow including unit conversion via UnitConverter, map projection setup with Cartopy, precipitation-specific colormap selection based on accumulation period, scatter plot rendering of MPAS cell values, discrete colorbar with meteorological contour levels, and geographic feature overlays (coastlines, borders, ocean, land). The method validates geographic extents, handles optional custom colormaps and contour levels, applies color limit clipping when specified, and adds time period annotations for accumulation context. This implementation maintains exact compatibility with the original mpas_analysis precipitation plotter while using modern MPASdiag visualization architecture.

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees [-180, 180] for MPAS mesh cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees [-90, 90] for MPAS mesh cell centers.
            precip_data (np.ndarray): 1D array of precipitation values (mm or model units) corresponding to lon/lat points.
            lon_min (float): Western boundary of map extent in degrees.
            lon_max (float): Eastern boundary of map extent in degrees.
            lat_min (float): Southern boundary of map extent in degrees.
            lat_max (float): Northern boundary of map extent in degrees.
            title (str): Plot title string, auto-generated from metadata if not provided (default: "MPAS Precipitation").
            accum_period (str): Accumulation period identifier for colormap selection (e.g., 'a01h', 'a24h') (default: "a01h").
            colormap (Optional[str]): Custom matplotlib colormap name overriding default precipitation colormap (default: None).
            levels (Optional[List[float]]): Custom contour levels overriding default precipitation levels (default: None).
            clim_min (Optional[float]): Minimum color limit to clip contour levels (default: None).
            clim_max (Optional[float]): Maximum color limit to clip contour levels (default: None).
            projection (str): Cartopy projection name ('PlateCarree', 'LambertConformal', etc.) (default: 'PlateCarree').
            time_end (Optional[datetime]): End datetime for accumulation period annotation (default: None).
            time_start (Optional[datetime]): Start datetime for accumulation period, derived from time_end and accum_period if None (default: None).
            data_array (Optional[xr.DataArray]): Source xarray DataArray for metadata extraction (units, long_name attributes) (default: None).
            var_name (str): Variable name for metadata lookup and unit conversion (default: 'precipitation').

        Returns:
            Tuple[Figure, Axes]: Two-element tuple containing (matplotlib_figure, cartopy_geoaxes) with rendered precipitation map.

        Raises:
            ValueError: If geographic extent parameters are invalid (out of range or improperly ordered).
        """
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180 and
                -90 <= lat_min <= 90 and -90 <= lat_max <= 90 and
                lon_max > lon_min and lat_max > lat_min):
            raise ValueError("Invalid plot extent parameters")
        
        if data_array is not None:
            try:
                if isinstance(precip_data, np.ndarray):
                    data_for_conversion = xr.DataArray(precip_data, attrs=data_array.attrs)
                else:
                    data_for_conversion = precip_data  
                converted_data, metadata = UnitConverter.convert_data_for_display(
                    data_for_conversion, var_name, data_array
                )
            except AttributeError:
                converted_data = precip_data
                metadata = {'units': getattr(data_array, 'units', 'mm'), 
                           'long_name': getattr(data_array, 'long_name', 'Precipitation')}
            if isinstance(converted_data, xr.DataArray):
                precip_data = converted_data.values
            elif isinstance(converted_data, np.ndarray):
                precip_data = converted_data
            else:
                precip_data = np.asarray(converted_data)
            unit_label = metadata.get('units', 'mm')
        else:
            unit_label = 'mm'
        
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for cartopy plots"
        
        is_global_lon = (lon_max - lon_min) >= 359.0
        is_global_lat = (lat_max - lat_min) >= 179.0
        
        if is_global_lon and is_global_lat:
            adjusted_lon_min = max(lon_min, -179.99)
            adjusted_lon_max = min(lon_max, 179.99)
            adjusted_lat_min = max(lat_min, -89.99)
            adjusted_lat_max = min(lat_max, 89.99)
            self.ax.set_extent([adjusted_lon_min, adjusted_lon_max, adjusted_lat_min, adjusted_lat_max], crs=data_crs)
            print(f"Using global extent (adjusted to avoid dateline): [{adjusted_lon_min}, {adjusted_lon_max}, {adjusted_lat_min}, {adjusted_lat_max}]")
        else:
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
        
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        
        if colormap and levels:
            cmap = plt.get_cmap(colormap)
            color_levels = levels
        elif colormap:
            cmap = plt.get_cmap(colormap)
            cmap_obj, color_levels = self.create_precip_colormap(accum_period)
        else:
            cmap, color_levels = self.create_precip_colormap(accum_period)
        
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
        
        precip_data_flat = np.asarray(precip_data).flatten()
        lon_flat = np.asarray(lon).flatten()
        lat_flat = np.asarray(lat).flatten()
        
        min_length = min(len(precip_data_flat), len(lon_flat), len(lat_flat))
        precip_data_flat = precip_data_flat[:min_length]
        lon_flat = lon_flat[:min_length]
        lat_flat = lat_flat[:min_length]
        
        valid_mask = (np.isfinite(precip_data_flat) & 
                     (precip_data_flat >= 0) & 
                     (precip_data_flat < 1e5) &
                     (lon_flat >= lon_min) & (lon_flat <= lon_max) &
                     (lat_flat >= lat_min) & (lat_flat <= lat_max))
        
        if np.any(valid_mask):
            lon_valid = lon_flat[valid_mask]
            lat_valid = lat_flat[valid_mask]
            precip_valid = precip_data_flat[valid_mask]
            
            map_extent = (lon_min, lon_max, lat_min, lat_max)
            fig_size = (self.figsize[0], self.figsize[1])
            marker_size = self.calculate_adaptive_marker_size(map_extent, len(precip_valid), fig_size)
            
            map_area = (lon_max - lon_min) * (lat_max - lat_min)
            point_density = len(precip_valid) / map_area if map_area > 0 else 0
            
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
            cbar.set_label(f'Precipitation ({unit_label})', fontsize=12, fontweight='bold', labelpad=-50)
            cbar.ax.tick_params(labelsize=8)
            
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
                accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
                n_hours = accum_hours_map.get(accum_period, 24)
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
        
        plt.tight_layout()
        self.add_timestamp_and_branding()
        
        return self.fig, self.ax
    
    def create_batch_precipitation_maps(self, 
                                      processor, 
                                      output_dir: str,
                                      lon_min: float, 
                                      lon_max: float,
                                      lat_min: float, 
                                      lat_max: float,
                                      var_name: str = 'rainnc',
                                      accum_period: str = 'a01h',
                                      file_prefix: str = 'mpas_precipitation_map',
                                      formats: List[str] = ['png'],
                                      custom_title_template: Optional[str] = None,
                                      colormap: Optional[str] = None,
                                      levels: Optional[List[float]] = None,
                                      time_indices: Optional[List[int]] = None) -> List[str]:
        """
        Generate precipitation maps for multiple time steps in batch processing mode with automatic accumulation calculation and file naming. This method iterates through time indices in the loaded MPAS dataset, extracts precipitation data at each time step, calculates accumulation periods by differencing with earlier time steps based on accum_period specification, creates individual precipitation maps using create_precipitation_map(), and saves each plot to output directory with standardized filenames encoding variable, accumulation type, and valid time. The batch processing handles time dimension detection, applies minimum time index constraints based on accumulation period requirements, and provides progress feedback for long-running batch operations. This workflow enables automated generation of complete precipitation analysis sequences matching the original mpas_analysis batch processing capabilities.

        Parameters:
            processor: MPAS2DProcessor instance with loaded precipitation dataset containing time series data.
            output_dir (str): Directory path for saving output plot files, created if it doesn't exist.
            lon_min (float): Western boundary of all maps in degrees.
            lon_max (float): Eastern boundary of all maps in degrees.
            lat_min (float): Southern boundary of all maps in degrees.
            lat_max (float): Northern boundary of all maps in degrees.
            var_name (str): Precipitation variable name in dataset (e.g., 'rainnc', 'rainc') (default: 'rainnc').
            accum_period (str): Accumulation period identifier ('a01h', 'a03h', 'a06h', 'a12h', 'a24h') (default: 'a01h').
            file_prefix (str): Prefix string for output filenames (default: 'mpas_precipitation_map').
            formats (List[str]): Output file format extensions (default: ['png']).
            custom_title_template (Optional[str]): Custom title template with {var_name}, {time_str} placeholders (default: None).
            colormap (Optional[str]): Custom colormap name overriding default precipitation colormap (default: None).
            levels (Optional[List[float]]): Custom contour levels overriding default levels (default: None).
            time_indices (Optional[List[int]]): Specific time indices to process, None processes all available times (default: None).

        Returns:
            List[str]: List of absolute file paths for all successfully created output files.

        Raises:
            ValueError: If processor is None or processor.dataset is None (no data loaded).
        """
        if processor is None:
            raise ValueError("Processor cannot be None")
        
        if processor.dataset is None:
            raise ValueError("No data loaded in processor")
        
        try:
            if hasattr(processor, 'extract_2d_coordinates_for_variable'):
                lon, lat = processor.extract_2d_coordinates_for_variable(var_name)
            elif hasattr(processor, 'extract_spatial_coordinates'):
                lon, lat = processor.extract_spatial_coordinates()
            else:
                lon = processor.dataset.lonCell.values
                lat = processor.dataset.latCell.values
        except AttributeError:
            lon = processor.dataset.lonCell.values
            lat = processor.dataset.latCell.values
        
        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'
        total_times = processor.dataset.sizes[time_dim]
        
        accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
        accum_hours = accum_hours_map.get(accum_period, 24)
        min_time_idx = accum_hours  
        
        if min_time_idx >= total_times:
            print(f"\nWarning: Accumulation period {accum_period} ({accum_hours} hours) requires at least {min_time_idx + 1} time steps.")
            print(f"Dataset only has {total_times} time steps. No plots will be generated.")
            return []
        
        if time_indices is None:
            time_indices = list(range(min_time_idx, total_times))
        else:
            time_indices = [idx for idx in time_indices if idx >= min_time_idx]
            if not time_indices:
                print(f"\nWarning: No valid time indices for accumulation period {accum_period}")
                return []
        
        actual_time_steps = len(time_indices)
        created_files = []
        
        print(f"\nCreating precipitation maps for {actual_time_steps} time steps...")
        print(f"Using accumulation period: {accum_period} ({accum_hours} hours)")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, time_idx in enumerate(time_indices):
            try:
                if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
                    time_end = pd.Timestamp(processor.dataset.Time.values[time_idx]).to_pydatetime()
                    time_str = time_end.strftime('%Y%m%dT%H')
                else:
                    time_end = None
                    time_str = f"t{time_idx:03d}"

                precip_diag = PrecipitationDiagnostics(verbose=False)
                precip_data = precip_diag.compute_precipitation_difference(
                    processor.dataset, time_idx, var_name, accum_period, processor.data_type
                )

                if custom_title_template:
                    title = custom_title_template.format(
                        var_name=var_name.upper(),
                        time_str=time_str,
                        accum_period=accum_period
                    )
                else:
                    title = f"MPAS Precipitation | VarType: {var_name.upper()} | Valid Time: {time_str}"

                fig, ax = self.create_precipitation_map(
                    lon, lat, precip_data.values,
                    lon_min, lon_max, lat_min, lat_max,
                    title=title,
                    accum_period=accum_period,
                    time_end=time_end,
                    colormap=colormap,
                    levels=levels,
                    data_array=precip_data,
                    var_name=var_name
                )
                
                output_path = os.path.join(
                    output_dir, 
                    f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_point"
                )
                
                self.save_plot(output_path, formats=formats)
                
                for fmt in formats:
                    created_files.append(f"{output_path}.{fmt}")
                
                self.close_plot()
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{actual_time_steps} maps (time index {time_idx})...")
                    
            except Exception as e:
                print(f"Error creating map for time index {time_idx}: {e}")
                continue
        
        print(f"\nBatch processing completed. Created {len(created_files)} files in: {output_dir}")
        return created_files
    
    def create_precipitation_comparison_plot(self,
                                           lon: np.ndarray,
                                           lat: np.ndarray,
                                           precip_data1: np.ndarray,
                                           precip_data2: np.ndarray,
                                           lon_min: float,
                                           lon_max: float,
                                           lat_min: float,
                                           lat_max: float,
                                           title1: str = "Precipitation 1",
                                           title2: str = "Precipitation 2",
                                           accum_period: str = "a01h",
                                           projection: str = 'PlateCarree') -> Tuple[Figure, List[Axes]]:
        """
        Create side-by-side precipitation comparison plots for analyzing differences between two precipitation datasets or accumulation periods. This method generates a two-panel figure with synchronized map projections, color scales, and geographic extents to enable direct visual comparison of precipitation patterns from different model runs, accumulation periods, or observational datasets. Both panels share the same precipitation-specific colormap and contour levels determined by accum_period, use identical map projections and geographic features (coastlines, borders, ocean, land), and are rendered with consistent marker sizes for MPAS unstructured mesh cells. A shared colorbar is positioned between or below the panels to facilitate quantitative comparison of precipitation intensities across the two datasets.

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS mesh cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS mesh cell centers.
            precip_data1 (np.ndarray): First precipitation dataset values (mm or display units) for left panel.
            precip_data2 (np.ndarray): Second precipitation dataset values (mm or display units) for right panel.
            lon_min (float): Western boundary of both maps in degrees.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Southern boundary of both maps in degrees.
            lat_max (float): Northern boundary of both maps in degrees.
            title1 (str): Title string for first (left) precipitation panel (default: "Precipitation 1").
            title2 (str): Title string for second (right) precipitation panel (default: "Precipitation 2").
            accum_period (str): Accumulation period identifier for shared colormap selection (default: "a01h").
            projection (str): Cartopy projection name applied to both panels (default: 'PlateCarree').

        Returns:
            Tuple[Figure, List[Axes]]: Two-element tuple containing (matplotlib_figure, list_of_two_geoaxes) for side-by-side comparison.
        """
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.8, self.figsize[1]), 
                                     dpi=self.dpi, subplot_kw={'projection': map_proj})
        
        axes = list(axes)
        
        cmap, color_levels = self.create_precip_colormap(accum_period)
        color_levels_sorted = sorted(set([v for v in color_levels if np.isfinite(v)]))
        last_bound = max(color_levels_sorted) + 1
        bounds = [0] + color_levels_sorted + [last_bound]
        norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        
        datasets = [precip_data1, precip_data2]
        titles = [title1, title2]
        
        is_global_lon = (lon_max - lon_min) >= 359.0
        is_global_lat = (lat_max - lat_min) >= 179.0
        
        scatter = None  

        for i, (ax, data, title) in enumerate(zip(axes, datasets, titles)):
            if is_global_lon and is_global_lat:
                adjusted_lon_min = max(lon_min, -179.99)
                adjusted_lon_max = min(lon_max, 179.99)
                adjusted_lat_min = max(lat_min, -89.99)
                adjusted_lat_max = min(lat_max, 89.99)
                ax.set_extent([adjusted_lon_min, adjusted_lon_max, adjusted_lat_min, adjusted_lat_max], crs=data_crs)
                if i == 0:
                    print(f"Using global extent (adjusted to avoid dateline): [{adjusted_lon_min}, {adjusted_lon_max}, {adjusted_lat_min}, {adjusted_lat_max}]")
            else:
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
            
            original_ax = self.ax

            self.ax = ax
            self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
            self.ax = original_ax
            
            data_flat = np.asarray(data).flatten()
            lon_flat = np.asarray(lon).flatten()
            lat_flat = np.asarray(lat).flatten()
            
            min_length = min(len(data_flat), len(lon_flat), len(lat_flat))
            data_flat = data_flat[:min_length]
            lon_flat = lon_flat[:min_length]
            lat_flat = lat_flat[:min_length]
            
            valid_mask = (np.isfinite(data_flat) & 
                         (data_flat >= 0) & 
                         (data_flat < 1e5) &
                         (lon_flat >= lon_min) & (lon_flat <= lon_max) &
                         (lat_flat >= lat_min) & (lat_flat <= lat_max))
            
            scatter = None
            if np.any(valid_mask):
                lon_valid = lon_flat[valid_mask]
                lat_valid = lat_flat[valid_mask]
                data_valid = data_flat[valid_mask]
                
                map_extent = (lon_min, lon_max, lat_min, lat_max)
                marker_size = self.calculate_adaptive_marker_size(map_extent, len(data_valid), self.figsize)
                
                sort_indices = np.argsort(data_valid)
                lon_sorted = lon_valid[sort_indices]
                lat_sorted = lat_valid[sort_indices]
                data_sorted = data_valid[sort_indices]
                
                scatter = ax.scatter(lon_sorted, lat_sorted, c=data_sorted, 
                                   cmap=cmap, norm=norm, s=marker_size, alpha=0.9, 
                                   transform=data_crs, edgecolors='none')
            
            gl = ax.gridlines(crs=data_crs, draw_labels=True, 
                             linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = (i == 1)  
            gl.left_labels = (i == 0)   
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
            gl.xformatter = FuncFormatter(self.format_longitude)
            gl.yformatter = FuncFormatter(self.format_latitude)
            
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        assert scatter is not None, "At least one subplot must have valid data for colorbar"
        cbar = self.fig.colorbar(scatter, ax=axes, orientation='horizontal', extend='both',
                               pad=0.08, shrink=0.6, aspect=30)
        cbar.set_label('Precipitation (mm)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        
        if len(color_levels_sorted) <= 15:
            cbar.set_ticks(color_levels_sorted)
            cbar.set_ticklabels(self._format_ticks_dynamic(color_levels_sorted))
        
        self.add_timestamp_and_branding()
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.15)
        
        return self.fig, axes
    
    def calculate_adaptive_marker_size(self, map_extent: Tuple[float, float, float, float],
                                       num_points: int,
                                       fig_size: Tuple[float, float] = (12, 10)) -> float:
        """
        Calculate optimal scatter plot marker size adaptively based on map area coverage and point density to ensure appropriate visual representation. This method delegates to MPASVisualizationStyle.calculate_adaptive_marker_size() which computes marker size using heuristics that consider the geographic extent in degrees squared, the number of plotted points, and the figure dimensions in inches. The adaptive sizing prevents marker overlap in high-resolution meshes while ensuring sufficient visibility in coarse meshes or large spatial domains. The returned marker size value (in points squared) is suitable for direct use in matplotlib scatter() calls for MPAS unstructured mesh visualization, providing consistent and professional-looking cartographic displays across varying mesh resolutions and spatial extents.

        Parameters:
            map_extent (Tuple[float, float, float, float]): Geographic extent as (lon_min, lon_max, lat_min, lat_max) defining plotted map area in degrees.
            num_points (int): Number of MPAS mesh cells (data points) to be plotted within the specified extent.
            fig_size (Tuple[float, float]): Figure dimensions as (width, height) in inches affecting marker size calculation (default: (12, 10)).

        Returns:
            float: Suggested scatter marker size in points squared (marker size units) optimized for the given extent, density, and figure size.
        """
        return MPASVisualizationStyle.calculate_adaptive_marker_size(map_extent, num_points, fig_size)
    
    def setup_map_projection(self, lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                             projection: str = 'PlateCarree') -> Tuple[ccrs.CRS, ccrs.CRS]:
        """
        Create Cartopy map projection and data coordinate reference systems for geographic plotting with automatic projection centering. This method calculates the central longitude and latitude from the provided extent bounds, instantiates the requested Cartopy projection (PlateCarree for equirectangular, Mercator for conformal cylindrical, or LambertConformal for conic projections), and returns both the map projection CRS for axes creation and the data CRS (PlateCarree) for plotting coordinate transformation. The central meridian and parallel are positioned at extent midpoints to minimize distortion within the region of interest. This projection setup is fundamental for all cartographic visualizations in MPASdiag, ensuring proper coordinate transformations between MPAS lat/lon data and projected map displays.

        Parameters:
            lon_min (float): Western boundary of map extent in degrees.
            lon_max (float): Eastern boundary of map extent in degrees.
            lat_min (float): Southern boundary of map extent in degrees.
            lat_max (float): Northern boundary of map extent in degrees.
            projection (str): Cartopy projection name ('PlateCarree', 'Mercator', 'LambertConformal'), defaults to PlateCarree if unrecognized (default: 'PlateCarree').

        Returns:
            Tuple[ccrs.CRS, ccrs.CRS]: Two-element tuple containing (map_projection_crs, data_crs_platecarree) for Cartopy GeoAxes creation and data transformation.
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
    
    def format_longitude(self, value: float, _: Any) -> str:
        """
        Format longitude values as human-readable axis tick labels with cardinal direction indicators for cartographic map displays. This method delegates to MPASVisualizationStyle.format_longitude() which converts numeric longitude degrees to formatted strings with degree symbols and hemisphere suffixes (E for eastern, W for western), handling special cases like the prime meridian (0°) and 180° meridian. The underscore parameter is required by matplotlib's FuncFormatter interface but unused in the implementation. This formatter is typically attached to map axes using set_major_formatter() to produce professional cartographic longitude labels on precipitation and other geographic visualizations.

        Parameters:
            value (float): Longitude coordinate value in degrees, typically in range [-180, 180].
            _ (Any): Unused secondary argument required by matplotlib FuncFormatter callback signature.

        Returns:
            str: Formatted longitude string with degree symbol and hemisphere indicator (e.g., '120°W', '45°E', '0°').
        """
        return MPASVisualizationStyle.format_longitude(value, _)
    
    def format_latitude(self, value: float, _: Any) -> str:
        """
        Format latitude values as human-readable axis tick labels with cardinal direction indicators for cartographic map displays. This method delegates to MPASVisualizationStyle.format_latitude() which converts numeric latitude degrees to formatted strings with degree symbols and hemisphere suffixes (N for northern, S for southern), handling special cases like the equator (0°) and poles (90°N, 90°S). The underscore parameter is required by matplotlib's FuncFormatter interface but unused in the implementation. This formatter is typically attached to map axes using set_major_formatter() to produce professional cartographic latitude labels on precipitation and other geographic visualizations, ensuring consistent and clear geographic context.

        Parameters:
            value (float): Latitude coordinate value in degrees, typically in range [-90, 90].
            _ (Any): Unused secondary argument required by matplotlib FuncFormatter callback signature.

        Returns:
            str: Formatted latitude string with degree symbol and hemisphere indicator (e.g., '45°N', '30°S', '0°').
        """
        return MPASVisualizationStyle.format_latitude(value, _)
    
    def add_timestamp_and_branding(self) -> None:
        """
        Add generation timestamp and lightweight MPASdiag branding text to the current figure for documentation and provenance tracking. This method places a small text annotation in the lower-left corner of the figure using fig.text() with the current datetime in UTC format and the MPASdiag project name. The text is rendered with reduced font size (8 points) and partial transparency (alpha=0.7) to provide attribution without distracting from the main visualization content. This timestamp serves as metadata for plot generation time, useful for version control, comparison of analysis runs, and documentation of when specific diagnostic plots were created. No action is taken if no figure is currently active in the plotter instance.

        Returns:
            None: Adds text annotation to self.fig if figure exists, otherwise no operation performed.
        """
        if self.fig is not None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            self.fig.text(0.02, 0.09, f'Generated with MPASdiag on: {timestamp}', 
                         fontsize=8, alpha=0.7, transform=self.fig.transFigure)
    
    def save_plot(self, output_path: str, formats: List[str] = ['png'],
                  bbox_inches: str = 'tight', pad_inches: float = 0.1) -> None:
        """
        Save the current matplotlib figure to disk in one or more file formats with configurable bounding box and padding options. This method writes the active figure stored in self.fig to files using the provided base output path, appending format-specific extensions for each requested format in the formats list. The bbox_inches='tight' option removes excess whitespace around the plot, while pad_inches controls additional padding around the figure edges. Multiple formats can be specified to generate publication-ready outputs (PNG for web/presentations, PDF/SVG for print), with each file saved at the configured DPI resolution. The method prints confirmation messages for each saved file and safely handles cases where no figure is active by returning without error.

        Parameters:
            output_path (str): Base file path without extension where plot files will be saved (e.g., '/output/precip_map').
            formats (List[str]): List of file format extensions to generate (e.g., ['png', 'pdf', 'svg', 'eps']) (default: ['png']).
            bbox_inches (str): Bounding box mode for savefig determining figure cropping ('tight' removes whitespace) (default: 'tight').
            pad_inches (float): Padding in inches around the figure edges when bbox_inches='tight' (default: 0.1).

        Returns:
            None: Writes files to disk and prints confirmation messages for each saved format.
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for fmt in formats:
            full_path = f"{output_path}.{fmt}"
            save_kwargs = {'dpi': self.dpi, 'bbox_inches': bbox_inches, 'pad_inches': pad_inches, 'format': fmt}
            if fmt.lower() == 'png':
                save_kwargs['pil_kwargs'] = {'compress_level': 1}
            self.fig.savefig(full_path, **save_kwargs)
            print(f"Saved plot: {full_path}")
    
    def close_plot(self) -> None:
        """
        Close the current matplotlib figure and release all associated references to free memory resources. This method calls plt.close() on the active figure stored in self.fig and sets both self.fig and self.ax to None to ensure proper garbage collection. Closing plots is essential in batch processing workflows to prevent memory accumulation when generating many sequential visualizations. This cleanup should be called after each plot is saved or when switching between different visualization tasks. The method safely handles cases where fig is already None, making it safe to call multiple times.

        Returns:
            None: Closes figure and clears instance attributes self.fig and self.ax.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _format_ticks_dynamic(self, ticks: List[float]) -> List[str]:
        """
        Format axis tick labels dynamically with intelligent precision selection based on the numeric range and magnitude of tick values. This method delegates to MPASVisualizationStyle.format_ticks_dynamic() which analyzes the tick value array to determine appropriate decimal places and formatting style (scientific notation for very large/small values, fixed decimal for typical ranges). The dynamic formatting ensures tick labels are concise yet informative, avoiding excessive decimal places for large numbers while maintaining necessary precision for small values. This adaptive approach produces clean, professional axis labels across diverse data ranges encountered in meteorological visualizations, from millimeter-scale precipitation to large geographic coordinates.

        Parameters:
            ticks (List[float]): Array of numeric tick values to be formatted for axis display.
            
        Returns:
            List[str]: Array of formatted tick label strings with appropriate precision and notation style.
        """
        return MPASVisualizationStyle.format_ticks_dynamic(ticks)
    
    def apply_style(self, style_name: str = 'default') -> None:
        """
        Apply a named visualization style to the plotter and active figure for consistent professional appearance across all precipitation plots. This method delegates to the style_manager if available to apply registered matplotlib style configurations (fonts, colors, line widths, etc.), then sets figure and axes background colors to standard defaults (light gray for axes, white for figure background). Style application affects all subsequent plotting operations and can be used to implement organization-specific branding, publication requirements, or different visual themes for presentation vs print media. The method safely handles cases where no style_manager is configured or no figure/axes are active, making it suitable for initialization and runtime style switching.

        Parameters:
            style_name (str): Name of the registered visualization style to apply from the style manager's style registry (default: 'default').

        Returns:
            None: Modifies appearance of self.ax and self.fig if they exist, applying colors and delegating to style_manager for broader matplotlib settings.
        """
        style_manager = getattr(self, 'style_manager', None)
        if style_manager:
            style_manager.apply_style(style_name)

        if self.ax is not None:
            self.ax.set_facecolor('#f0f0f0')

        if self.fig is not None:
            self.fig.patch.set_facecolor('white')