#!/usr/bin/env python3

"""
MPAS Base Visualization Framework

This module provides the foundational visualization infrastructure for creating professional cartographic plots from MPAS atmospheric model output with consistent styling and quality standards. It implements the abstract base class MPASVisualizer that serves as the parent for all specialized plotters (precipitation, surface, cross-section, wind), providing common functionality including map projection setup with Cartopy, coordinate axis formatting with geographic labels, adaptive marker sizing based on map extent and data density, timestamp and version branding for plot provenance, and standardized figure saving with optimized compression settings. The base visualizer establishes consistent plotting conventions across all MPASdiag visualization modules, handles matplotlib and Cartopy integration for unstructured mesh rendering, and provides utility methods for time series plotting, histograms, and statistical visualizations. This framework enables rapid development of new diagnostic plotters by inheriting common cartographic and styling functionality while allowing specialization for variable-specific rendering requirements.

Classes:
    MPASVisualizer: Abstract base class providing common visualization infrastructure for all MPAS diagnostic plotters.
    
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
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Optional, List, Any, Union, Sequence, cast

from ..processing.utils_unit import UnitConverter
from ..processing.utils_metadata import MPASFileMetadata
from ..processing.processors_3d import MPAS3DProcessor
from .styling import MPASVisualizationStyle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False})

warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')


class MPASVisualizer:
    """
    Base class for MPAS model output visualization with comprehensive functionality for publication-quality cartographic presentations of unstructured mesh data. This class provides common utilities and methods inherited by specialized plotting classes for different variable types, including surface diagnostics, wind vectors, precipitation, and cross-sections. Core capabilities include automatic marker sizing based on data density, professional map projection setup with cartopy integration, unit conversion and metadata management through centralized utilities, adaptive formatting for coordinate axes and color scales, and multi-format output generation. The visualizer emphasizes consistent styling across all plot types through delegation to MPASVisualizationStyle for branding, colormaps, and presentation standards while providing flexible customization through configurable figure size, DPI, and projection options.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (10, 14), dpi: int = 100, verbose: bool = True) -> None:
        """
        Initialize the MPAS visualizer with configurable figure dimensions and output resolution settings. This constructor establishes the base visualization parameters including figure size in inches and dots-per-inch (DPI) resolution, initializes instance variables for matplotlib Figure and Axes objects to None for lazy creation during plotting operations, and sets up the foundation for all subsequent plot generation through inherited specialized visualizer classes. The default figure size (10x14 inches) is optimized for portrait-oriented map displays typical of regional MPAS domains, while DPI=100 provides screen-quality output suitable for interactive analysis with options to increase to 300+ for publication-quality figures.

        Parameters:
            figsize (Tuple[float, float]): Figure size in inches as (width, height) tuple (default: (10, 14) for portrait orientation).
            dpi (int): Resolution in dots per inch for output images, use 100 for screen quality or 300+ for publication (default: 100).

        Returns:
            None: Initializes instance attributes self.figsize, self.dpi, self.fig, and self.ax.
        """
        self.figsize = figsize
        self.dpi = dpi
        self.verbose: bool = verbose
        self.fig: Optional[Figure] = None
        self.ax: Optional[Union[Axes, GeoAxes]] = None
    
    def add_timestamp_and_branding(self) -> None:
        """
        Adds timestamp and project branding to the active figure using the centralized style manager
        for consistent presentation across all plot types. This method delegates rendering to
        MPASVisualizationStyle to ensure uniform branding appearance including timestamp, project name,
        institution, and formatting. It operates as a no-op if no figure is active, providing safe
        integration into visualization workflows without requiring explicit figure existence checks.

        Returns:
            None: Modifies self.fig by adding text annotations for timestamp and branding.
        """
        assert self.fig is not None, "Figure must be created before adding branding"
        MPASVisualizationStyle.add_timestamp_and_branding(self.fig)
    
    def format_latitude(self, value: float, _) -> str:
        """
        Formats a numeric latitude value for axis tick labels using standardized geographic conventions
        with degree symbols and hemisphere indicators. This method delegates to MPASVisualizationStyle
        to ensure consistent latitude formatting across all plot types, converting decimal degrees to
        readable strings with N/S suffixes (e.g., 45.0 becomes '45°N', -30.0 becomes '30°S'). It
        accepts a placeholder second argument to satisfy matplotlib's FuncFormatter interface requirements.

        Parameters:
            value (float): Latitude value in decimal degrees (-90 to 90).
            _ (Any): Placeholder parameter required by matplotlib FuncFormatter (ignored).

        Returns:
            str: Formatted latitude string with degree symbol and hemisphere (e.g., '45°N', '30°S').
        """
        return MPASVisualizationStyle.format_latitude(value, _)

    def format_longitude(self, value: float, _) -> str:
        """
        Formats a numeric longitude value for axis tick labels using standardized geographic conventions
        with degree symbols and hemisphere indicators. This method delegates to MPASVisualizationStyle
        to ensure consistent longitude formatting across all plot types, converting decimal degrees to
        readable strings with E/W suffixes (e.g., -120.0 becomes '120°W', 75.0 becomes '75°E'). It
        accepts a placeholder second argument to satisfy matplotlib's FuncFormatter interface requirements.

        Parameters:
            value (float): Longitude value in decimal degrees (-180 to 180 or 0 to 360).
            _ (Any): Placeholder parameter required by matplotlib FuncFormatter (ignored).

        Returns:
            str: Formatted longitude string with degree symbol and hemisphere (e.g., '120°W', '75°E').
        """
        return MPASVisualizationStyle.format_longitude(value, _)
    
    def calculate_adaptive_marker_size(self, map_extent: Tuple[float, float, float, float], 
                                     num_points: int, fig_size: Tuple[float, float] = (12, 10)) -> float:
        """
        Calculate optimal marker size for scatter plots based on map geographic extent, data point density, and figure dimensions to maintain visual clarity. This method delegates to MPASVisualizationStyle to compute marker sizing using heuristics that balance coverage and overlap, accounting for the total map area in square degrees, the number of points to display, and the figure size in inches to scale appropriately for different output resolutions. The adaptive sizing prevents overcrowding in high-density regions while ensuring visibility in sparse data areas, with automatic bounds applied to avoid extremely small (<1 point) or excessively large (>200 points) markers. This dynamic sizing is essential for professional presentation of MPAS unstructured mesh data where point density can vary dramatically across regional and global domains.

        Parameters:
            map_extent (Tuple[float, float, float, float]): Geographic bounds as (lon_min, lon_max, lat_min, lat_max) in degrees for area calculation.
            num_points (int): Total number of data points to be plotted for density estimation.
            fig_size (Tuple[float, float]): Figure dimensions in inches as (width, height) for resolution-aware scaling (default: (12, 10)).

        Returns:
            float: Appropriate marker size in matplotlib points-squared units for scatter plot visualization.
        """
        return MPASVisualizationStyle.calculate_adaptive_marker_size(map_extent, num_points, fig_size)

    def _format_ticks_dynamic(self, ticks: List[float]) -> List[str]:
        """
        Choose sensible numeric formatting for axis tick labels based on value magnitude and range to optimize readability. This method delegates to MPASVisualizationStyle to apply formatting heuristics including scientific notation for very small (<1e-3) or very large (>=1e4) values, integer formatting when all values are near-integers, and magnitude-based decimal precision ranging from 0 decimals for hundreds to 3 decimals for sub-0.01 values. The adaptive formatting prevents cluttered axis labels with excessive precision while maintaining sufficient detail for data interpretation. This automatic formatting is applied to both geographic coordinate axes and data value colorbars across all MPAS visualization types.

        Parameters:
            ticks (List[float]): Tick values to format for axis labeling, can span any numeric range or magnitude.

        Returns:
            List[str]: Formatted tick label strings with appropriate precision and notation style for clean axis presentation.
        """
        return MPASVisualizationStyle.format_ticks_dynamic(ticks)
    
    def get_variable_specific_settings(self, var_name: str, data: np.ndarray) -> Tuple[Union[str, mcolors.ListedColormap], Optional[List[float]]]:
        """
        Retrieve variable-specific colormap and contour level specifications based on meteorological conventions and data characteristics. This method delegates to MPASVisualizationStyle to lookup optimal visualization settings for common MPAS variables, returning matplotlib colormap names (e.g., 'RdYlBu_r' for temperature, 'Blues' for precipitation, 'viridis' for pressure) and discrete contour level arrays tailored to variable type and data range. The settings ensure scientifically appropriate color representations following established meteorological conventions while adapting to actual data ranges when necessary. This centralized configuration promotes consistent visualization across different plotting functions and variable types throughout MPASdiag.

        Parameters:
            var_name (str): MPAS variable name for lookup in settings registry (e.g., 'temperature', 'pressure', 'wind_speed', 'rainnc').
            data (np.ndarray): Data array used to determine appropriate value ranges when dynamic level calculation is needed.

        Returns:
            Tuple[Union[str, mcolors.ListedColormap], Optional[List[float]]]: Two-element tuple containing (colormap_name_or_object, contour_levels_list) for matplotlib plotting functions.
        """
        return MPASVisualizationStyle.get_variable_specific_settings(var_name, data)
    
    def setup_map_projection(self, lon_min: float, lon_max: float, 
                           lat_min: float, lat_max: float,
                           projection: str = 'PlateCarree') -> Tuple[ccrs.CRS, ccrs.CRS]:
        """
        Returns a cartopy map projection and data coordinate reference system (CRS) suitable for plotting
        the specified geographic extent. This method delegates to MPASVisualizationStyle to select appropriate
        projection parameters based on the domain bounds, supports multiple projection types (PlateCarree,
        Mercator, LambertConformal), and returns both the map projection for axes creation and the data CRS
        (typically PlateCarree) for coordinate transformations during plotting operations.

        Parameters:
            lon_min (float): Western longitude bound in degrees for projection centering.
            lon_max (float): Eastern longitude bound in degrees for projection centering.
            lat_min (float): Southern latitude bound in degrees for projection parameters.
            lat_max (float): Northern latitude bound in degrees for projection parameters.
            projection (str): Projection name from cartopy (default: 'PlateCarree').

        Returns:
            Tuple[ccrs.CRS, ccrs.CRS]: Map projection for axes and data CRS for transformations (typically PlateCarree).
        """
        return MPASVisualizationStyle.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
    
    def add_regional_features(self, lon_min: float, lon_max: float, 
                             lat_min: float, lat_max: float) -> None:
        """
        Adds region-specific map features based on geographic extent with automatic feature selection
        for enhanced cartographic context. This method analyzes the map domain bounds to determine if
        the plot covers the continental United States (CONUS) region by checking latitude (15-60°N)
        and longitude (-135 to -55°E) ranges, automatically adds state boundaries as red line features
        when CONUS coverage is detected, and provides intelligent feature management for regional-scale
        visualizations without requiring manual feature specification.

        Parameters:
            lon_min (float): Western longitude bound in degrees for domain detection.
            lon_max (float): Eastern longitude bound in degrees for domain detection.
            lat_min (float): Southern latitude bound in degrees for domain detection.
            lat_max (float): Northern latitude bound in degrees for domain detection.

        Returns:
            None: Adds features directly to self.ax when conditions are met (no-op if ax is None).
        """
        if self.ax is None:
            return
            
        show_states = False

        if (lat_min >= 15 and lat_max <= 60 and 
            lon_min >= -135 and lon_max <= -55):
            show_states = True
        
        if show_states and isinstance(self.ax, GeoAxes):
            self.ax.add_feature(cfeature.STATES, linewidth=0.5,
                                edgecolor='red', facecolor='none')
    
    def save_plot(self, 
                  output_path: str, 
                  formats: List[str] = ['png'],
                  bbox_inches: str = 'tight',
                  pad_inches: float = 0.1) -> None:
        """
        Save the current matplotlib figure to one or more output file formats with configurable layout and quality settings. This method delegates to MPASVisualizationStyle to handle file writing with automatic format extension appending, tight bounding box calculation to minimize whitespace, and DPI application from instance settings for resolution control. The method supports multiple simultaneous format outputs (PNG, PDF, SVG, EPS) from a single render, enabling efficient generation of both screen-display and publication-ready versions. Directory creation is handled automatically when output paths contain non-existent directories.

        Parameters:
            output_path (str): Base file path without extension where plot(s) will be saved, extensions added automatically per format.
            formats (List[str]): List of output format strings like 'png', 'pdf', 'svg', 'eps' for multi-format export (default: ['png']).
            bbox_inches (str): Matplotlib bounding box mode, typically 'tight' to crop whitespace or standard bbox (default: 'tight').
            pad_inches (float): Padding in inches around figure content when using tight bbox mode (default: 0.1).

        Returns:
            None: Writes plot file(s) to disk, raises AssertionError if self.fig is None.
        """
        assert self.fig is not None, "Figure must be created before saving"
        MPASVisualizationStyle.save_plot(self.fig, output_path, formats, bbox_inches, pad_inches, self.dpi)
    
    def close_plot(self) -> None:
        """
        Close and release matplotlib figure resources to prevent memory leaks in batch processing workflows. This method properly closes the current figure using plt.close(), sets the figure and axes references to None to enable garbage collection, and ensures clean resource management when generating multiple plots sequentially. Calling this method is essential in loops or batch operations to avoid accumulating figure objects in memory, which can cause performance degradation or out-of-memory errors with large datasets. The method is safe to call multiple times or when no figure exists (no-op if self.fig is already None).

        Returns:
            None: Closes self.fig and resets self.fig and self.ax to None.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def create_time_series_plot(self,
                              times: List[datetime],
                              values: List[float],
                              title: str = "Time Series",
                              ylabel: str = "Value",
                              xlabel: str = "Time") -> Tuple[Figure, Axes]:
        """
        Create a time series line plot with automatic date formatting and statistical overlays for temporal analysis of MPAS output. This method generates a matplotlib figure with datetime-aware x-axis formatting, plots the time-value relationship with line and marker styling, adds grid lines for readability, applies automatic date label rotation, and includes project timestamp/branding. The visualization uses consistent styling with 2-point line width and 4-point circular markers, enabling quick identification of temporal patterns, trends, and anomalies in MPAS diagnostic time series such as domain-averaged quantities or point measurements.

        Parameters:
            times (List[datetime]): Time coordinate values as datetime objects for x-axis temporal positioning.
            values (List[float]): Data values corresponding to each time point for y-axis plotting.
            title (str): Plot title text displayed at top of figure (default: "Time Series").
            ylabel (str): Y-axis label describing the plotted variable and units (default: "Value").
            xlabel (str): X-axis label, typically "Time" for temporal plots (default: "Time").

        Returns:
            Tuple[Figure, Axes]: Two-element tuple containing (matplotlib Figure object, matplotlib Axes object) for further customization if needed.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        try:
            times_index = pd.to_datetime(times)
        except Exception:
            times_index = np.array(times, dtype='datetime64[ns]')

        self.ax.plot(times_index, values, linewidth=2, marker='o', markersize=4)
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.autofmt_xdate()
        plt.tight_layout()
        
        self.add_timestamp_and_branding()
        
        return self.fig, self.ax
    
    def create_histogram(self,
                        data: np.ndarray,
                        bins: Union[int, np.ndarray] = 50,
                        title: str = "Data Distribution",
                        xlabel: str = "Value",
                        ylabel: str = "Frequency",
                        log_scale: bool = False) -> Tuple[Figure, Axes]:
        """
        Create a histogram visualization of data distribution with automatic statistical summary overlays including mean and standard deviation markers. This method generates a matplotlib histogram plot filtering out non-finite values (NaN, Inf), computes and displays summary statistics with colored reference lines (red dashed for mean, orange dotted for ±1 std), supports optional logarithmic y-axis scaling for skewed distributions, and includes grid lines for readability. The visualization is essential for quality assessment of MPAS data, identifying outliers, understanding value distributions, and validating data ranges before detailed analysis or plotting operations.

        Parameters:
            data (np.ndarray): Data array to histogram, can be multi-dimensional (flattened automatically), non-finite values excluded from analysis.
            bins (Union[int, np.ndarray]): Number of histogram bins as integer or explicit bin edge array for custom binning (default: 50).
            title (str): Plot title text displayed at top of figure (default: "Data Distribution").
            xlabel (str): X-axis label describing the data variable and units (default: "Value").
            ylabel (str): Y-axis label, typically "Frequency" or "Count" (default: "Frequency").
            log_scale (bool): Apply logarithmic scale to y-axis for better visualization of skewed distributions (default: False).

        Returns:
            Tuple[Figure, Axes]: Two-element tuple containing (matplotlib Figure object, matplotlib Axes object) with histogram plot ready for display or saving.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) > 0:
            bins_arg = cast(Union[int, Sequence[float], str, None], bins)
            if isinstance(bins, np.ndarray):
                try:
                    bins_arg = bins.tolist()
                except Exception:
                    bins_arg = list(bins)

            n, bins, patches = self.ax.hist(valid_data, bins=bins_arg, alpha=0.7,  
                                          edgecolor='black', linewidth=0.5)
            
            mean_val = float(np.mean(valid_data))
            std_val = float(np.std(valid_data))
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
        
        plt.tight_layout()
        self.add_timestamp_and_branding()
        
        return self.fig, self.ax

    def extract_2d_from_3d(self, 
                          data_3d: Union[np.ndarray, xr.DataArray],
                          level_index: Optional[int] = None,
                          level_value: Optional[float] = None,
                          level_dim: str = 'nVertLevels',
                          method: str = 'nearest') -> np.ndarray:
        """
        Extract 2D horizontal slice from 3D MPAS atmospheric data at specified vertical level for surface plotting workflows. This method delegates to MPAS3DProcessor to handle vertical extraction using either direct level indexing (0-based integer) or value-based interpolation (e.g., 850 for 850 hPa), supports multiple vertical coordinate systems (model levels, pressure, height), applies nearest-neighbor or linear interpolation methods, and returns a 2D numpy array ready for cartographic visualization. This extraction is essential for creating horizontal maps of upper-air variables like temperature, winds, or moisture at constant pressure surfaces or fixed heights from full 3D model output.

        Parameters:
            data_3d (Union[np.ndarray, xr.DataArray]): 3D data array with vertical dimension to extract from, can be raw numpy or xarray DataArray.
            level_index (Optional[int]): Direct 0-based index of vertical level to extract, mutually exclusive with level_value (default: None).
            level_value (Optional[float]): Physical value to search for in vertical coordinate (e.g., 850 for 850 hPa), requires level_dim specification (default: None).
            level_dim (str): Name of vertical dimension in data array, such as 'nVertLevels', 'pressure', 'height' (default: 'nVertLevels').
            method (str): Interpolation method for value-based extraction, either 'nearest' or 'linear' (default: 'nearest').

        Returns:
            np.ndarray: 2D extracted data array with vertical dimension removed, ready for surface plotting on map projection.
        """
        return MPAS3DProcessor.extract_2d_from_3d(data_3d, level_index, level_value, level_dim, method)

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
                        time_stamp: Optional[datetime] = None,
                        projection: str = 'PlateCarree') -> Tuple[Figure, Axes]:
        """
        Create unified wind vector visualization using barbs or arrows with automatic subsampling and optional wind speed background shading. This method generates professional cartographic wind plots with geographic features (coastlines, borders, land/ocean), configures map projection and extent, filters data to domain bounds, applies intelligent automatic subsampling based on point density to prevent overcrowding (typically every Nth point where N scales with density), overlays wind vectors as either meteorological barbs or directional arrows, optionally adds filled contour background showing wind speed magnitude, and includes formatted gridlines with degree symbols. The visualization handles MPAS unstructured mesh wind data for any vertical level (surface, 850 hPa, etc.) with consistent styling across applications.

        Parameters:
            lon (np.ndarray): Longitude coordinates for wind vector positions in degrees.
            lat (np.ndarray): Latitude coordinates for wind vector positions in degrees.
            u_data (np.ndarray): U-component (eastward) wind data in m/s matching coordinate arrays.
            v_data (np.ndarray): V-component (northward) wind data in m/s matching coordinate arrays.
            lon_min (float): Western boundary of plot extent in degrees for domain cropping.
            lon_max (float): Eastern boundary of plot extent in degrees for domain cropping.
            lat_min (float): Southern boundary of plot extent in degrees for domain cropping.
            lat_max (float): Northern boundary of plot extent in degrees for domain cropping.
            wind_level (str): Descriptive label for wind level used in title (e.g., "surface", "850 hPa") (default: "surface").
            plot_type (str): Wind vector representation type, either 'barbs' (meteorological standard) or 'arrows' (directional) (default: "barbs").
            subsample (int): Subsampling factor for vectors, plot every Nth point; ≤0 triggers automatic density-based calculation (default: 0).
            scale (Optional[float]): Scale factor for arrow size when plot_type='arrows', auto-computed from 90th percentile if None (default: None).
            show_background (bool): Display filled contour background of wind speed magnitude for context (default: False).
            bg_colormap (str): Matplotlib colormap name for wind speed background when show_background=True (default: "viridis").
            title (Optional[str]): Custom plot title, auto-generated from wind_level and time_stamp if None (default: None).
            time_stamp (Optional[datetime]): Datetime for inclusion in auto-generated title as UTC timestamp (default: None).
            projection (str): Cartopy projection name for map axes (default: 'PlateCarree').

        Returns:
            Tuple[Figure, Axes]: Two-element tuple containing (matplotlib Figure, cartopy GeoAxes) with completed wind plot.
        """
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, projection=map_proj)
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for cartopy plots"
        
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        
        gl = self.ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        wind_speed = np.sqrt(u_data**2 + v_data**2)
        
        mask = ((lon >= lon_min) & (lon <= lon_max) & 
                (lat >= lat_min) & (lat <= lat_max) & 
                np.isfinite(u_data) & np.isfinite(v_data))
        
        if not np.any(mask):
            raise ValueError("No valid wind data points found within the specified map extent.")
        
        lon_filtered = lon[mask]
        lat_filtered = lat[mask]
        u_filtered = u_data[mask]
        v_filtered = v_data[mask]
        wind_speed_filtered = wind_speed[mask]
        
        print(f"Plotting {np.sum(mask)} wind vectors")
        print(f"Wind speed range: {np.min(wind_speed_filtered):.1f} to {np.max(wind_speed_filtered):.1f} m/s")

        if show_background:
            self._create_wind_background(lon_filtered, lat_filtered, wind_speed_filtered, 
                                       bg_colormap, data_crs)

        if subsample <= 0:
            map_area = (lon_max - lon_min) * (lat_max - lat_min)
            point_density = len(lon_filtered) / map_area
            
            if point_density > 20:
                subsample = max(2, int(np.sqrt(point_density / 10)))
            else:
                subsample = 1
            
            print(f"Auto-subsampling: using every {subsample} point(s)")

        if subsample > 1:
            indices = np.arange(0, len(lon_filtered), subsample)
            lon_plot = lon_filtered[indices]
            lat_plot = lat_filtered[indices] 
            u_plot = u_filtered[indices]
            v_plot = v_filtered[indices]
        else:
            lon_plot, lat_plot = lon_filtered, lat_filtered
            u_plot, v_plot = u_filtered, v_filtered

        if plot_type.lower() == 'barbs':
            self.ax.barbs(lon_plot, lat_plot, u_plot, v_plot,
                         length=7, barbcolor='black', flagcolor='red',
                         transform=data_crs, linewidth=0.8)
        elif plot_type.lower() == 'arrows':
            scale = scale or float(np.percentile(wind_speed_filtered, 90) * 10)
            self.ax.quiver(lon_plot, lat_plot, u_plot, v_plot,
                          scale=scale, width=0.003, headwidth=3,
                          color='black', transform=data_crs)
        else:
            raise ValueError(f"plot_type must be 'barbs' or 'arrows', got '{plot_type}'")

        if title is None:
            title = f"MPAS {wind_level.title()} Wind"
            if time_stamp:
                title += f" - {time_stamp.strftime('%Y-%m-%d %H:%M UTC')}"

        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        self.add_timestamp_and_branding()
        
        return self.fig, self.ax

    def _create_wind_background(self, lon: np.ndarray, lat: np.ndarray, 
                               wind_speed: np.ndarray, colormap: str, data_crs) -> None:
        """
        Create filled scatter plot background showing wind speed magnitude for enhanced wind vector visualization context. This helper method computes adaptive marker sizes based on data density using calculate_adaptive_marker_size(), creates a colored scatter plot with wind speed values mapped to the specified colormap, applies semi-transparent rendering (alpha=0.6) to allow vector overlay visibility, adds a vertical colorbar with wind speed labels and tick formatting, and integrates seamlessly with the main wind plot axes. This background layer provides intuitive visual representation of wind intensity patterns underlying the directional vector field, particularly useful for identifying regions of strong winds or calm conditions.

        Parameters:
            lon (np.ndarray): Longitude coordinates for scatter point positions in degrees.
            lat (np.ndarray): Latitude coordinates for scatter point positions in degrees.
            wind_speed (np.ndarray): Wind speed magnitude values in m/s for color mapping.
            colormap (str): Matplotlib colormap name for wind speed color representation (e.g., 'viridis', 'plasma').
            data_crs (ccrs.CRS): Cartopy coordinate reference system for scatter plot transformation, typically ccrs.PlateCarree().

        Returns:
            None: Modifies self.ax by adding scatter plot and colorbar, raises AssertionError if ax is None.
        """
        assert self.ax is not None, "Axes must be created before adding wind background"
        marker_size = self.calculate_adaptive_marker_size(
            (lon.min(), lon.max(), lat.min(), lat.max()), 
            len(wind_speed), self.figsize
        ) * 0.5  
        
        sc = self.ax.scatter(lon, lat, c=wind_speed, cmap=colormap,
                           alpha=0.6, s=marker_size, edgecolors='none',
                           transform=data_crs)
        
        cbar = plt.colorbar(sc, ax=self.ax, orientation='vertical', 
                          pad=0.05, shrink=0.8)
        cbar.set_label('Wind Speed (m/s)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)
