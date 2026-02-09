#!/usr/bin/env python3

"""
MPAS Precipitation Visualization

This module provides specialized precipitation visualization functionality for MPAS model output implementing the exact plotting logic and color schemes from the original mpas_analysis module for backward compatibility. It includes the MPASPrecipitationPlotter class that creates professional cartographic precipitation maps with discrete colormaps optimized for meteorological precipitation display, accumulation-period-specific contour levels (hourly vs daily), automatic unit conversion from model output to millimeters, and seamless integration with the modern MPASdiag visualization architecture. The plotter supports single precipitation maps for individual time steps, batch processing for creating time series of precipitation analyses, and multi-panel comparison plots for model-observation evaluation. Core capabilities include period-aware color scheme selection, scatter plot rendering of MPAS unstructured mesh data, geographic feature overlays, and publication-quality output with timestamps and colorbars suitable for operational weather analysis and climate model validation.

Classes:
    MPASPrecipitationPlotter: Specialized class for creating precipitation visualizations from MPAS model output with meteorologically-appropriate styling.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Import standard libraries and third-party dependencies for data handling, plotting, and geospatial processing
import os
import warnings
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
from typing import Tuple, Optional, List, Any, Union, Dict, cast

# Import relevant MPASdiag modules for visualization and processing
from .base_visualizer import MPASVisualizer
from .styling import MPASVisualizationStyle
from ..processing.utils_unit import UnitConverter
from ..processing.utils_metadata import MPASFileMetadata
from ..processing.remapping import remap_mpas_to_latlon_with_masking
from ..diagnostics.precipitation import PrecipitationDiagnostics

# Filter warnings from cartopy and shapely to avoid cluttering output with non-critical warnings during map rendering
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')


class MPASPrecipitationPlotter(MPASVisualizer):
    """
    Specialized plotter for creating professional precipitation visualizations from MPAS model output with meteorologically-appropriate color schemes and accumulation period handling. This class extends MPASVisualizer to provide comprehensive functionality for rendering precipitation fields on cartographic maps using MPAS unstructured mesh data, implementing the exact visualization logic and color schemes from the original mpas_analysis module for backward compatibility while leveraging modern MPASdiag architecture. The plotter supports multiple accumulation periods (hourly, daily, etc.) with period-specific contour levels and discrete colormaps following meteorological conventions, automated unit conversion from model output to display units (mm), flexible map projections via Cartopy, and geographic feature overlays (coastlines, borders, terrain). Visualization outputs include publication-quality single-panel precipitation maps, multi-panel comparison plots for model-observation evaluation, and batch processing capabilities for creating time series of precipitation analyses with consistent styling and automatic file naming.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (10, 14), dpi: int = 100) -> None:
        """
        Initialize the MPAS precipitation plotter with default figure size and resolution. This constructor delegates to the parent `MPASVisualizer` to establish plotting infrastructure including `self.fig` and `self.ax`. The instance is prepared to create single or batch precipitation visualizations with map projections and standardized styling. Use `figsize` to control the layout and `dpi` to control output resolution for saved images.

        Parameters:
            figsize (Tuple[float, float]): Figure dimensions as (width, height) in inches.
            dpi (int): Figure resolution in dots-per-inch used for rendering and saved output.

        Returns:
            None: The initializer sets up instance attributes and does not return a value.
        """
        super().__init__(figsize, dpi)
    
    def create_precip_colormap(
        self,
        accum: str = "a24h"
    ) -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        Generate a precipitation-specific discrete colormap and associated contour levels. This helper returns a `ListedColormap` and a list of contour level values appropriate for the specified accumulation period. The accumulation period controls spacing and extent of the returned levels so that hourly and daily totals use meteorologically sensible bins. The produced colormap and levels are suitable for use with `BoundaryNorm` and colorbar tick formatting.

        Parameters:
            accum (str): Accumulation period identifier (e.g., 'a24h' or 'a01h').

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: A tuple containing the discrete colormap and a sorted list of contour level values.
        """
        # Delegate to MPASVisualizationStyle to create a discrete colormap and contour levels based on the specified accumulation period
        return MPASVisualizationStyle.create_precip_colormap(accum)
    
    def _convert_precipitation_units(
        self,
        precip_data: np.ndarray,
        data_array: Optional[xr.DataArray],
        var_name: str
    ) -> Tuple[np.ndarray, str]:
        """
        Convert precipitation values to display units and return a representative unit label. This routine inspects optional xarray `DataArray` metadata to determine source units and uses `UnitConverter` to convert values (for example from model-native units to millimeters). If no metadata is provided, the input data are returned unchanged and a default unit label is supplied. The returned numpy array is suitable for downstream plotting routines.

        Parameters:
            precip_data (np.ndarray): 1D or flattened precipitation values to convert.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray providing metadata (units, long_name).
            var_name (str): Variable name used for unit lookup and conversion logic.

        Returns:
            Tuple[np.ndarray, str]: Two-element tuple containing the converted precipitation array as a numpy ndarray and a string label for the display units (e.g., 'mm').
        """
        # If data_array is not provided, return original data and default unit label 'mm' for precipitation
        if data_array is None:
            return precip_data, 'mm'
        
        try:
            # Prepare data for conversion, ensuring it is an xarray DataArray with appropriate attributes for UnitConverter
            if isinstance(precip_data, np.ndarray):
                data_for_conversion = xr.DataArray(precip_data, attrs=data_array.attrs)
            else:
                data_for_conversion = precip_data
            
            # Use UnitConverter to convert precipitation data to display units (mm) and extract metadata for colorbar labeling
            converted_data, metadata = UnitConverter.convert_data_for_display(
                data_for_conversion, var_name, data_array
            )
        except AttributeError:
            # If conversion fails due to missing attributes, return original data and default unit label
            converted_data = precip_data

            # Extract units and long_name from data_array attributes for colorbar labeling, defaulting to 'mm' and 'Precipitation' if not available
            metadata = {
                'units': getattr(data_array, 'units', 'mm'),
                'long_name': getattr(data_array, 'long_name', 'Precipitation')
            }
        
        # Extract converted data values as numpy array for plotting, ensuring compatibility with matplotlib functions
        if isinstance(converted_data, xr.DataArray):
            precip_data = converted_data.values
        elif isinstance(converted_data, np.ndarray):
            precip_data = converted_data
        else:
            precip_data = np.asarray(converted_data)
        
        # Check for negative precipitation values which are physically invalid and likely indicate data issues
        n_negative = np.sum(precip_data < 0)

        # If negative values are found, log a warning with the count and minimum value, then clip the data to 0 to ensure physical validity for precipitation fields
        if n_negative > 0:
            print(f"Warning: Found {n_negative:,} negative precipitation values (min: {np.nanmin(precip_data):.4f}). Clipping to 0 (physically invalid).")
            precip_data = np.clip(precip_data, 0, None)
        
        # Extract unit label for colorbar annotation, defaulting to 'mm' if not available
        unit_label = metadata.get('units', 'mm')

        # Return the converted precipitation data and the unit label for colorbar annotation
        return precip_data, unit_label
    
    def _setup_precipitation_figure(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        projection: str
    ) -> Tuple[ccrs.Projection, ccrs.CRS]:
        """
        Create and configure the matplotlib `Figure` and cartopy `GeoAxes` for a precipitation map. This method selects a map projection via `setup_map_projection`, instantiates `self.fig` and `self.ax` (a `GeoAxes`), sets the map extent, and adds standard geographic features such as coastlines, borders, land and ocean patches. Regional features appropriate to the provided bounds are also added. The configured projection and the data CRS used for plotting are returned for use by plotting helpers.

        Parameters:
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            projection (str): Name of the Cartopy projection to use (e.g., 'PlateCarree').

        Returns:
            Tuple[ccrs.Projection, ccrs.CRS]: A tuple containing the chosen map projection and the data CRS used for plotting transforms.
        """
        # Setup map projection and data CRS using parent class method
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)
        
        # Create figure and GeoAxes with the specified projection
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)

        # Assert ax is created and of correct type for cartopy plotting before adding features
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for cartopy plots"
        
        # Determine if the map is global based on extent to handle dateline issues appropriately
        is_global = (lon_max - lon_min) >= 359.0 and (lat_max - lat_min) >= 179.0

        # For global maps, adjust extent slightly to avoid dateline issues; for regional maps, set extent directly with validation
        if is_global:
            # Adjust global extent slightly to avoid dateline issues
            adjusted_lon_min = max(lon_min, -179.99)
            adjusted_lon_max = min(lon_max, 179.99)
            adjusted_lat_min = max(lat_min, -89.99)
            adjusted_lat_max = min(lat_max, 89.99)

            # Set global extent with slight adjustment to avoid dateline issues, ensuring proper rendering of global maps without artifacts
            self.ax.set_extent([adjusted_lon_min, adjusted_lon_max, adjusted_lat_min, adjusted_lat_max], crs=data_crs)
            print(f"Using global extent (adjusted to avoid dateline): [{adjusted_lon_min}, {adjusted_lon_max}, {adjusted_lat_min}, {adjusted_lat_max}]")
        else:
            # Set extent for regional maps, ensuring it is within valid ranges and properly ordered
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        # Add geographic features with styling for precipitation maps
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)

        # Add regional features if within the specified extent
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        
        # Return the map projection and data CRS 
        return map_proj, data_crs
    
    def _prepare_precipitation_colormap(
        self,
        colormap: Optional[str],
        levels: Optional[List[float]],
        accum_period: str,
        clim_min: Optional[float],
        clim_max: Optional[float]
    ) -> Tuple[mcolors.Colormap, BoundaryNorm, List[float]]:
        """
        Construct a colormap, corresponding `BoundaryNorm`, and sorted contour levels for precipitation plotting. This helper supports either user-specified colormap/levels or automatically generated defaults based on the accumulation period. Optional `clim_min` and `clim_max` parameters are used to clip and extend the returned level list so that color normalization matches requested color limits. The returned `BoundaryNorm` is ready to use with matplotlib artists for consistent discrete coloring.

        Parameters:
            colormap (Optional[str]): Optional name of a matplotlib colormap to use.
            levels (Optional[List[float]]): Optional explicit contour level values to use.
            accum_period (str): Accumulation period identifier used to choose default levels when `levels` is None.
            clim_min (Optional[float]): Optional minimum color limit to enforce on levels.
            clim_max (Optional[float]): Optional maximum color limit to enforce on levels.

        Returns:
            Tuple[mcolors.Colormap, BoundaryNorm, List[float]]: The selected colormap, a `BoundaryNorm` for discrete coloring, and the sorted list of contour levels.
        """
        # Select colormap and levels based on user input and accumulation period
        if colormap and levels:
            # Use custom colormap and levels directly, ensuring levels are sorted and finite for proper normalization and colorbar ticks
            cmap = plt.get_cmap(colormap)
            color_levels = levels
        elif colormap:
            # Use custom colormap with default levels based on accumulation period for proper colorbar ticks and normalization
            cmap = plt.get_cmap(colormap)
            _, color_levels = self.create_precip_colormap(accum_period)
        else:
            # Use default precipitation colormap and levels based on accumulation period
            cmap, color_levels = self.create_precip_colormap(accum_period)
        
        # Apply color limits if specified
        if clim_min is not None and clim_max is not None:
            # Filter color levels to those within the specified color limits for proper normalization and colorbar ticks
            color_levels = [level for level in color_levels if clim_min <= level <= clim_max]

            # Ensure clim_min is included as a level for proper colorbar ticks and normalization
            if clim_min not in color_levels:
                color_levels.insert(0, clim_min)

            # Ensure clim_max is included as a level for proper colorbar ticks and normalization
            if clim_max not in color_levels:
                color_levels.append(clim_max)
        
        # Sort and deduplicate color levels, ensuring they are finite values for proper normalization
        color_levels_sorted = sorted(set([v for v in color_levels if np.isfinite(v)]))

        # Add bounds for normalization: extend beyond the last level to ensure proper coloring of values above the highest level
        last_bound = max(color_levels_sorted) + 1
        bounds = [0] + color_levels_sorted + [last_bound]

        # Specify colors for each level, ensuring the number of colors matches the number of intervals defined by bounds
        norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        
        # Return the colormap, normalization, and sorted contour levels 
        return cmap, norm, color_levels_sorted
    
    def _prepare_precipitation_data(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        precip_data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        plot_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align and validate input longitude, latitude, and precipitation arrays for plotting. Arrays are flattened and trimmed to the same length; entries are filtered for finite, non-negative, and reasonable values. For scatter plots additional spatial bounds filtering is applied to ensure only points inside the specified extent are returned. Instead of raising an exception for no valid points, this function returns empty arrays so calling code can decide how to handle empty datasets.

        Parameters:
            lon (np.ndarray): Longitude coordinate array (1D or broadcastable to `precip_data`).
            lat (np.ndarray): Latitude coordinate array (1D or broadcastable to `precip_data`).
            precip_data (np.ndarray): Precipitation values corresponding to lon/lat.
            lon_min (float): Western extent bound in degrees.
            lon_max (float): Eastern extent bound in degrees.
            lat_min (float): Southern extent bound in degrees.
            lat_max (float): Northern extent bound in degrees.
            plot_type (str): Plot mode, one of 'scatter', 'contour', 'contourf'.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Three 1D numpy arrays (lon_valid, lat_valid, precip_valid) containing only the points that passed validation. Arrays may be empty if no valid points exist.
        """
        # Flatten arrays and ensure they are 1D for alignment
        precip_data_flat = np.asarray(precip_data).flatten()
        lon_flat = np.asarray(lon).flatten()
        lat_flat = np.asarray(lat).flatten()
        
        # Ensure all arrays have the same length for proper alignment
        min_length = min(len(precip_data_flat), len(lon_flat), len(lat_flat))

        # Truncate arrays to the same length to ensure proper alignment for plotting
        precip_data_flat = precip_data_flat[:min_length]
        lon_flat = lon_flat[:min_length]
        lat_flat = lat_flat[:min_length]
        
        # Create validation mask based on plot type
        if plot_type == 'scatter':
            # For scatter, we need to ensure points are within geographic extent and have valid precipitation values
            valid_mask = (
                np.isfinite(precip_data_flat) &
                (precip_data_flat >= 0) &
                (precip_data_flat < 1e5) &
                (lon_flat >= lon_min) & (lon_flat <= lon_max) &
                (lat_flat >= lat_min) & (lat_flat <= lat_max)
            )
        else:
            # For contour and contourf, we will handle geographic masking during interpolation, so only validate data values here
            valid_mask = (
                np.isfinite(precip_data_flat) &
                (precip_data_flat >= 0) &
                (precip_data_flat < 1e5)
            )
        
        # Return only valid points for plotting
        return lon_flat[valid_mask], lat_flat[valid_mask], precip_data_flat[valid_mask]
    
    def _add_time_annotation(
        self,
        time_end: Optional[datetime],
        time_start: Optional[datetime],
        accum_period: str
    ) -> None:
        """
        Add time period annotation text box to precipitation map.
        
        Parameters:
            time_end (Optional[datetime]): End datetime for accumulation.
            time_start (Optional[datetime]): Start datetime for accumulation.
            accum_period (str): Accumulation period identifier.
        """
        # Assert ax is created and of correct type for cartopy plotting before adding annotation
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for cartopy plots"

        # Determine annotation text based on provided time information and accumulation period
        if time_end is not None:
            # Derive time_start from time_end and accum_period if not provided
            if time_start is None:
                accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
                n_hours = accum_hours_map.get(accum_period, 24)
                time_start = time_end - pd.Timedelta(hours=n_hours)
            
            # Format annotation text with UTC times and accumulation period
            start_utc = time_start.strftime('%Y-%m-%d %H:%M UTC')
            end_utc = time_end.strftime('%Y-%m-%d %H:%M UTC')
            n_hours = int((time_end - time_start).total_seconds() / 3600)
            txt = f"Accumulation: {start_utc} to {end_utc} ({n_hours} h)"
        elif accum_period:
            # Use accumulation period string if time_end is not provided but accum_period is specified
            accum_hours_map = {'a01h': '1-h', 'a03h': '3-h', 'a06h': '6-h', 'a12h': '12-h', 'a24h': '24-h'}
            accum_display = accum_hours_map.get(accum_period, accum_period)
            txt = f"Accumulation: {accum_display}"
        else:
            return
        
        # Add annotation text box to the plot with styling
        self.ax.text(0.01, 0.02, txt, transform=self.ax.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def create_precipitation_map(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        precip_data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        title: str = "MPAS Precipitation",
        accum_period: str = "a01h",
        plot_type: str = 'scatter',
        colormap: Optional[str] = None,
        levels: Optional[List[float]] = None,
        clim_min: Optional[float] = None,
        clim_max: Optional[float] = None,
        projection: str = 'PlateCarree',
        time_end: Optional[datetime] = None,
        time_start: Optional[datetime] = None,
        data_array: Optional[xr.DataArray] = None,
        var_name: str = 'precipitation',
        grid_resolution: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> Tuple[Figure, Axes]:
        """
        Render a precipitation map for MPAS unstructured mesh data using cartopy and matplotlib. This method converts input units, configures the map projection and figure, selects an appropriate colormap and normalization for the accumulation period, and renders the data either as a scatter of MPAS cell values or as interpolated contours/filled contours. It also supports optional customization of colormap, levels, color limits, and grid resolution for interpolation. The function always returns a created `Figure` and `Axes` even when the input data contain no valid points; in that case, an empty map with geographic features is produced.

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS cell centers.
            precip_data (np.ndarray): Precipitation values corresponding to the lon/lat points (may be model units or already in display units).
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            title (str): Title to display on the map.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to select levels and colormap.
            plot_type (str): Rendering mode: 'scatter', 'contour', or 'contourf'.
            colormap (Optional[str]): Optional matplotlib colormap name to override defaults.
            levels (Optional[List[float]]): Optional list of contour levels to use instead of defaults.
            clim_min (Optional[float]): Optional minimum color limit to clip contour levels.
            clim_max (Optional[float]): Optional maximum color limit to clip contour levels.
            projection (str): Cartopy projection identifier (default: 'PlateCarree').
            time_end (Optional[datetime]): Optional end datetime for accumulation annotation.
            time_start (Optional[datetime]): Optional start datetime for accumulation annotation.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray providing metadata for unit conversion.
            var_name (str): Variable name for metadata lookup and unit conversion logic.
            grid_resolution (Optional[float]): Optional target grid resolution (degrees) for interpolation in contour/contourf.
            dataset (Optional[xr.Dataset]): Optional MPAS dataset for remapping when interpolating to a regular lat/lon grid.

        Returns:
            Tuple[Figure, Axes]: A tuple containing the created matplotlib `Figure` and the cartopy `Axes` used for plotting.

        Raises:
            ValueError: If `plot_type` is not one of 'scatter', 'contour', or 'contourf', or if geographic extent values are invalid.
        """
        # Validate plot type parameter to ensure it is one of the accepted values for rendering methods
        if plot_type not in ['scatter', 'contour', 'contourf']:
            raise ValueError(f"plot_type must be 'scatter', 'contour', or 'contourf', got '{plot_type}'")
        
        # Validate geographic extent parameters to ensure they are within acceptable ranges and properly ordered
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180 and
                -90 <= lat_min <= 90 and -90 <= lat_max <= 90 and
                lon_max > lon_min and lat_max > lat_min):
            raise ValueError("Invalid plot extent parameters")
        
        # Convert precipitation data units and extract unit label for colorbar
        precip_data, unit_label = self._convert_precipitation_units(precip_data, data_array, var_name)
        
        # Setup figure, axes, projection, and geographic features for precipitation map
        map_proj, data_crs = self._setup_precipitation_figure(lon_min, lon_max, lat_min, lat_max, projection)
        
        # Assert fig and ax are created and of correct type for cartopy plotting
        assert self.fig is not None, "Figure must be created by _setup_precipitation_figure"
        assert self.ax is not None, "Axes must be created by _setup_precipitation_figure"
        assert isinstance(self.ax, GeoAxes), "Axes must be GeoAxes for cartopy plots"
        
        # Prepare colormap, contour levels, and normalization for precipitation visualization
        cmap, norm, color_levels_sorted = self._prepare_precipitation_colormap(
            colormap, levels, accum_period, clim_min, clim_max
        )
        
        # Prepare and validate data for plotting, returning only valid points within geographic extent (for scatter) or valid values (for contour/contourf)
        lon_valid, lat_valid, precip_valid = self._prepare_precipitation_data(
            lon, lat, precip_data, lon_min, lon_max, lat_min, lat_max, plot_type
        )
        
        # Handle empty data case (all-NaN or no valid points)
        if len(precip_valid) > 0:
            print(f"Plotting {len(precip_valid):,} precipitation points for {var_name}")
            print(f"Precipitation range: {precip_valid.min():.3f} to {precip_valid.max():.3f} {unit_label}")
            
            # Render based on plot type using helper methods
            if plot_type == 'scatter':
                # Scatter plot can use original lon/lat points without interpolation, directly colored by precipitation values
                self._create_scatter_plot(lon_valid, lat_valid, precip_valid, cmap, norm, data_crs, unit_label, color_levels_sorted)
            elif plot_type == 'contour':
                # Contour requires interpolation to a regular grid, which is handled in the helper method using MPASRemapper
                self._create_contour_plot(lon_valid, lat_valid, precip_valid,
                                         lon_min, lon_max, lat_min, lat_max,
                                         cmap, norm, color_levels_sorted, data_crs,
                                         grid_resolution, dataset, unit_label)
            elif plot_type == 'contourf':
                # Contourf requires interpolation to a regular grid, which is handled in the helper method using MPASRemapper
                self._create_contourf_plot(lon_valid, lat_valid, precip_valid,
                                          lon_min, lon_max, lat_min, lat_max,
                                          cmap, norm, color_levels_sorted, data_crs,
                                          grid_resolution, dataset, unit_label)
        else:
            print(f"Warning: No valid precipitation data points found for {var_name}")
        
        # Add gridlines with labels and custom formatting
        gl = self.ax.gridlines(crs=data_crs, draw_labels=True,
                             linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        # Disable gridline labels on top and right to avoid clutter
        gl.top_labels = False
        gl.right_labels = False

        # Customize label styles and formats
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # Format longitude and latitude labels with degree symbols
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
        
        # Set title and add time annotation
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add time annotation for accumulation period
        self._add_time_annotation(time_end, time_start, accum_period)
        
        # Adjust layout to prevent overlap of elements
        plt.tight_layout()

        # Add timestamp and branding to the plot
        self.add_timestamp_and_branding()
        
        # Return the figure and axes for further manipulation or saving
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
        Interpolate scattered precipitation data onto a regular lat-lon grid using MPASRemapper.
        This helper method eliminates code duplication between contour and contourf methods.
        
        Parameters:
            lon (np.ndarray): 1D longitude array in degrees.
            lat (np.ndarray): 1D latitude array in degrees.
            data (np.ndarray): 1D precipitation data array.
            lon_min (float): Western longitude bound in degrees.
            lon_max (float): Eastern longitude bound in degrees.
            lat_min (float): Southern latitude bound in degrees.
            lat_max (float): Northern latitude bound in degrees.
            grid_resolution (Optional[float]): Grid spacing in degrees (default: None for adaptive).
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinates (default: None, auto-created).
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Three arrays (lon_grid, lat_grid, data_grid).
        """
        print(f"Interpolating {len(data)} precipitation points using linear interpolation...")
        
        # Determine grid resolution if not provided, using a heuristic based on geographic extent to balance detail and performance
        if grid_resolution is None:
            # Determine grid resolution based on geographic extent
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min

            # Estimate grid resolution based on extent and number of points
            grid_resolution = max(lon_range / 100, lat_range / 100)
            grid_resolution = max(0.1, min(grid_resolution, 1.0))
            print(f"Auto-selected grid resolution: {grid_resolution:.3f}°")
        
        if dataset is None:
            # Ensure lon and lat are numpy arrays for remapping
            lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
            lat_arr = lat if isinstance(lat, np.ndarray) else lat.values

            # Create a minimal xarray Dataset with lonCell and latCell for remapping
            dataset = xr.Dataset({
                'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
                'latCell': xr.DataArray(lat_arr, dims=['nCells'])
            })
        
        # Create DataArray for input data with appropriate dimensions for remapping
        data_xr = xr.DataArray(data, dims=['nCells'])

        # Use MPASRemapper to interpolate from unstructured MPAS mesh to regular lat-lon grid with masking of invalid points
        remapped = remap_mpas_to_latlon_with_masking(
            data=data_xr,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=grid_resolution,
            method='linear',
            apply_mask=True,
            lon_convention='auto'
        )
        
        # Extract regular grid coordinates and interpolated data values
        lon_grid = remapped.lon.values
        lat_grid = remapped.lat.values
        data_grid = remapped.values
        
        print(f"Remapped to {data_grid.shape[0]}x{data_grid.shape[1]} grid")

        # Return the regular grid coordinates and interpolated data
        return lon_grid, lat_grid, data_grid
    
    def _create_scatter_plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        cmap: mcolors.Colormap,
        norm: BoundaryNorm,
        data_crs: ccrs.CRS,
        unit_label: str,
        color_levels: List[float]
    ) -> None:
        """
        Render precipitation as a colored scatter plot on the configured map axes and add a matching colorbar. The function calculates an adaptive marker size based on map extent and point count, sorts points by data value to ensure correct layering, and creates a matplotlib `PathCollection` with the provided colormap and normalization. After plotting, a discrete precipitation colorbar is added using the provided `color_levels` and `unit_label`.

        Parameters:
            lon (np.ndarray): 1D array of validated longitude values for plotting.
            lat (np.ndarray): 1D array of validated latitude values for plotting.
            data (np.ndarray): 1D array of validated precipitation values corresponding to lon/lat.
            cmap (mcolors.Colormap): Matplotlib colormap used to map values to colors.
            norm (BoundaryNorm): Normalization object used for discrete color intervals.
            data_crs (ccrs.CRS): Coordinate reference system used for plotting transforms.
            unit_label (str): Label for the colorbar units (e.g., 'mm').
            color_levels (List[float]): Sorted list of contour/colorbar tick levels used for the colorbar.

        Returns:
            None: This function draws onto `self.ax` and does not return a value.
        """
        # Assert fig and ax are created
        assert self.ax is not None, "Axes must be created before scatter plot"
        assert self.fig is not None, "Figure must be created before scatter plot"
        
        # Determine map extent for adaptive marker sizing
        map_extent = (lon.min(), lon.max(), lat.min(), lat.max())

        # Calculate adaptive marker size based on map extent and number of points to ensure visibility without overcrowding
        marker_size = self.calculate_adaptive_marker_size(map_extent, len(data), self.figsize)
        
        # Sort data for better visualization (plot smaller values on top)
        sort_indices = np.argsort(data)
        lon_sorted = lon[sort_indices]
        lat_sorted = lat[sort_indices]
        data_sorted = data[sort_indices]
        
        # Render scatter points with specified colormap and normalization
        scatter = self.ax.scatter(
            lon_sorted, lat_sorted, c=data_sorted,
            cmap=cmap, norm=norm, s=marker_size, alpha=0.9,
            transform=data_crs, edgecolors='none'
        )
        
        # Add discrete colorbar with proper formatting
        self._add_precipitation_colorbar(scatter, unit_label, color_levels)
    
    def _create_contour_plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        cmap: mcolors.Colormap,
        norm: BoundaryNorm,
        color_levels: List[float],
        data_crs: ccrs.CRS,
        grid_resolution: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None,
        unit_label: str = 'mm'
    ) -> None:
        """
        This method first interpolates the scattered data onto a regular lat-lon grid using the `_interpolate_to_grid` helper, then uses `ax.contour` to draw contour lines at the specified `color_levels`. Contour labels are added using `ax.clabel` with a simple numeric format. The colorbar is added separately using the `_add_precipitation_colorbar` helper to ensure consistent formatting with scatter and contourf plots. 
        
        Parameters:
            lon (np.ndarray): Valid longitude coordinates.
            lat (np.ndarray): Valid latitude coordinates.
            data (np.ndarray): Valid precipitation data values.
            lon_min (float): Western bound in degrees.
            lon_max (float): Eastern bound in degrees.
            lat_min (float): Southern bound in degrees.
            lat_max (float): Northern bound in degrees.
            cmap (mcolors.Colormap): Colormap for contours.
            norm (BoundaryNorm): Color normalization.
            color_levels (List[float]): Explicit contour levels.
            data_crs (ccrs.CRS): Coordinate reference system.
            grid_resolution (Optional[float]): Grid spacing in degrees.
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinates.
            unit_label (str): Unit label for colorbar.
        """
        # Assert fig and ax are created
        assert self.ax is not None, "Axes must be created before contour plot"
        assert self.fig is not None, "Figure must be created before contour plot"
        
        # Interpolate data to regular grid for contour rendering
        lon_grid, lat_grid, data_grid = self._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, dataset
        )
        
        # Draw contour lines with labels
        cs = self.ax.contour(
            lon_grid, lat_grid, data_grid,
            levels=color_levels,
            colors='black',
            linewidths=1.0,
            linestyles='solid',
            transform=data_crs
        )
        
        # Add contour labels
        try:
            # Use '%g' format to avoid trailing zeros and ensure clean labels
            self.ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
        except Exception:
            pass
    
    def _create_contourf_plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        cmap: mcolors.Colormap,
        norm: BoundaryNorm,
        color_levels: List[float],
        data_crs: ccrs.CRS,
        grid_resolution: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None,
        unit_label: str = 'mm'
    ) -> None:
        """
        This method first interpolates the scattered data onto a regular lat-lon grid using the `_interpolate_to_grid` helper, then uses `ax.contourf` to draw filled contours at the specified `color_levels` with the provided `cmap` and `norm`. The colorbar is added separately using the `_add_precipitation_colorbar` helper to ensure consistent formatting with scatter and contour plots.
        
        Parameters:
            lon (np.ndarray): Valid longitude coordinates.
            lat (np.ndarray): Valid latitude coordinates.
            data (np.ndarray): Valid precipitation data values.
            lon_min (float): Western bound in degrees.
            lon_max (float): Eastern bound in degrees.
            lat_min (float): Southern bound in degrees.
            lat_max (float): Northern bound in degrees.
            cmap (mcolors.Colormap): Colormap for filled contours.
            norm (BoundaryNorm): Color normalization.
            color_levels (List[float]): Explicit contour levels.
            data_crs (ccrs.CRS): Coordinate reference system.
            grid_resolution (Optional[float]): Grid spacing in degrees.
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinates.
            unit_label (str): Unit label for colorbar.
        """
        # Assert fig and ax are created 
        assert self.ax is not None, "Axes must be created before contourf plot"
        assert self.fig is not None, "Figure must be created before contourf plot"
        
        # Interpolate data to regular grid for contourf rendering
        lon_grid, lat_grid, data_grid = self._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, dataset
        )
        
        # Draw filled contours with specified levels and colormap
        contourf = self.ax.contourf(
            lon_grid, lat_grid, data_grid,
            levels=color_levels,
            cmap=cmap,
            norm=norm,
            transform=data_crs,
            extend='both'
        )
        
        # Add discrete colorbar with proper formatting
        self._add_precipitation_colorbar(contourf, unit_label, color_levels)
    
    def _add_precipitation_colorbar(
        self,
        mappable: Any,
        unit_label: str,
        color_levels: List[float]
    ) -> None:
        """
        This method ensures the colorbar is added to the existing figure and axes, sets a descriptive label with units, and configures tick labels based on the provided contour levels for consistency across scatter, contour, and contourf plots.
        
        Parameters:
            mappable (Any): Matplotlib mappable object (scatter or contourf).
            unit_label (str): Unit label for colorbar.
            color_levels (List[float]): Contour levels for tick labels.
        """
        # Ensure figure exists before adding colorbar
        assert self.fig is not None, "Figure must exist before adding colorbar"
        
        # Add horizontal colorbar below the plot with specified padding and aspect ratio
        cbar = self.fig.colorbar(
            mappable, ax=self.ax, orientation='horizontal',
            extend='both', pad=0.06, shrink=0.8, aspect=30
        )

        # Set colorbar label with units and styling
        cbar.set_label(
            f'Precipitation [{unit_label}]',
            fontsize=12, fontweight='bold', labelpad=-50
        )

        # Adjust tick label size for readability
        cbar.ax.tick_params(labelsize=8)
        
        # Set colorbar ticks to contour levels if not too many levels for readability
        if len(color_levels) <= 15:
            cbar.set_ticks(color_levels)
            cbar.set_ticklabels(self._format_ticks_dynamic(color_levels))
    
    def _prepare_overlay_data(
        self,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        precip_data: np.ndarray,
        var_name: str,
        original_units: Optional[str],
        plot_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """
       This method handles unit conversion if necessary, flattens the input arrays for processing, and calculates the geographic bounds of the data for plotting. It also creates a validity mask based on the plot type and data quality to filter out invalid points. Instead of raising an exception for no valid points, this function returns empty arrays and bounds, allowing calling code to decide how to handle empty datasets.
        
        Parameters:
            lon: Longitude coordinates.
            lat: Latitude coordinates.
            precip_data: Precipitation data array.
            var_name (str): Variable name for unit conversion.
            original_units (Optional[str]): Original units of data.
            plot_type (str): Type of plot being created.
            
        Returns:
            Tuple containing (lon_valid, lat_valid, precip_valid, lon_min, lon_max, lat_min, lat_max).
        """
        # Convert to numpy arrays if they are xarray DataArrays
        lon = self.convert_to_numpy(lon)
        lat = self.convert_to_numpy(lat)
        precip_data = self.convert_to_numpy(precip_data)
        
        # Check for potential unit issues if original units are not provided
        if original_units is None:
            data_mean = np.nanmean(precip_data)
            if data_mean > 1000:
                print("Warning: Precipitation data may not be in mm. Consider specifying 'original_units' in config.")
        
        # Attempt unit conversion if original units are provided and differ from display units
        if original_units:
            # Get display units from UnitConverter based on variable name and original units
            display_units = UnitConverter.get_display_units(var_name, original_units)
            # Only attempt conversion if display units differ from original units
            if original_units != display_units:
                try:
                    # Attempt to convert units using UnitConverter
                    converted_data = UnitConverter.convert_units(precip_data, original_units, display_units)
                    precip_data = self.convert_to_numpy(converted_data)
                    print(f"Converted overlay {var_name} from {original_units} to {display_units}")
                except ValueError as e:
                    # If conversion fails, log a warning and proceed with original data
                    print(f"Warning: Could not convert overlay {var_name} from {original_units} to {display_units}: {e}")
        
        # Check for negative precipitation values which are physically invalid and likely indicate data issues
        n_negative = np.sum(precip_data < 0)

        # If negative values are found, log a warning with the count and minimum value, then clip to 0 to ensure physically valid precipitation values for plotting
        if n_negative > 0:
            print(f"Warning: Found {n_negative:,} negative precipitation values (min: {np.nanmin(precip_data):.4f}). Clipping to 0 (physically invalid).")
            precip_data = np.clip(precip_data, 0, None)
        
        # Flatten arrays for processing and ensure they are 1D
        precip_data_flat = precip_data.flatten()
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()
        
        # Ensure all arrays have the same length after flattening
        min_length = min(len(precip_data_flat), len(lon_flat), len(lat_flat))

        # Truncate arrays to minimum length to ensure alignment
        precip_data_flat = precip_data_flat[:min_length]
        lon_flat = lon_flat[:min_length]
        lat_flat = lat_flat[:min_length]
        
        # Determine bounds from valid longitude and latitude values
        lon_min = float(lon_flat[np.isfinite(lon_flat)].min())
        lon_max = float(lon_flat[np.isfinite(lon_flat)].max())
        lat_min = float(lat_flat[np.isfinite(lat_flat)].min())
        lat_max = float(lat_flat[np.isfinite(lat_flat)].max())
        
        # Create validity mask based on plot type and data quality
        if plot_type == 'scatter':
            # For scatter plots, we will filter out points outside the map extent to avoid plotting irrelevant data, in addition to filtering invalid precipitation values
            valid_mask = (
                np.isfinite(precip_data_flat) & 
                (precip_data_flat >= 0) & 
                (precip_data_flat < 1e5) &
                (lon_flat >= lon_min) & (lon_flat <= lon_max) &
                (lat_flat >= lat_min) & (lat_flat <= lat_max)
            )
        else: 
            # For contour and contourf, we will rely on remapping to handle masking, so we only filter out invalid precipitation values here
            valid_mask = (
                np.isfinite(precip_data_flat) & 
                (precip_data_flat >= 0) & 
                (precip_data_flat < 1e5)
            )
        
        # Check if any valid data points exist after filtering
        if not np.any(valid_mask):
            raise ValueError(f"No valid precipitation overlay data found for {var_name}")
        
        # Extract valid data for overlay rendering
        lon_valid = lon_flat[valid_mask]
        lat_valid = lat_flat[valid_mask]
        precip_valid = precip_data_flat[valid_mask]
        
        # Return valid data and bounds for overlay rendering
        return lon_valid, lat_valid, precip_valid, lon_min, lon_max, lat_min, lat_max
    
    def _setup_overlay_colormap(
        self,
        colormap: Optional[str],
        levels: Optional[List[float]],
        accum_period: str
    ) -> Tuple[mcolors.Colormap, BoundaryNorm, List[float]]:
        """
        This method determines the appropriate colormap and contour levels based on user input and defaults for the specified accumulation period. It also creates a `BoundaryNorm` with clipping to ensure that values outside the defined levels are colored appropriately. The function returns the configured colormap, normalization, and sorted levels for consistent use in overlay rendering and colorbar configuration.
        
        Parameters:
            colormap (Optional[str]): Custom colormap name.
            levels (Optional[List[float]]): Custom contour levels.
            accum_period (str): Accumulation period identifier.
            
        Returns:
            Tuple containing (colormap, normalization, sorted_levels).
        """
        # Determine colormap and levels based on user input and defaults
        if colormap and levels:
            # Use custom colormap and levels directly
            cmap = plt.get_cmap(colormap)
            color_levels = levels
        elif colormap:
            # Use custom colormap with default levels based on accumulation period
            cmap = plt.get_cmap(colormap)
            _, color_levels = self.create_precip_colormap(accum_period)
        else:
            # Use default precipitation colormap and levels based on accumulation period
            cmap, color_levels = self.create_precip_colormap(accum_period)
        
        # Filter and sort color levels for normalization and colorbar ticks
        color_levels_sorted = sorted(set([v for v in color_levels if np.isfinite(v)]))

        # Set bounds for BoundaryNorm (0 to max level + 1) to ensure proper color mapping with clipping
        last_bound = max(color_levels_sorted) + 1
        bounds = [0] + color_levels_sorted + [last_bound]

        # Use BoundaryNorm for discrete color mapping with clipping
        norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        
        # Return colormap, normalization, and sorted levels for consistent use in overlay rendering
        return cmap, norm, color_levels_sorted
    
    def _calculate_overlay_grid_resolution(
        self,
        grid_resolution_input: Optional[float],
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float
    ) -> float:
        """
        The method uses a heuristic based on the geographic extent of the data to determine an appropriate resolution that balances detail and performance. The resolution is calculated as 1% of the larger dimension of the map extent, with bounds to ensure it is suitable for precipitation maps (not too coarse or too fine). If the user provides a specific grid resolution, that value is used directly without modification.
        
        Parameters:
            grid_resolution_input (Optional[float]): User-specified resolution.
            lon_min (float): Western boundary.
            lon_max (float): Eastern boundary.
            lat_min (float): Southern boundary.
            lat_max (float): Northern boundary.
            
        Returns:
            float: Grid resolution in degrees.
        """
        # Use user-specified resolution if provided
        if grid_resolution_input is not None:
            return float(grid_resolution_input)
        
        # Calculate map extent
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        # Calculate resolution as 1% of the larger dimension of the map extent
        resolution = max(lon_range / 100, lat_range / 100)

        # Return resolution clipped to reasonable bounds for precipitation maps
        return max(0.1, min(resolution, 1.0))
    
    def _render_overlay_scatter(
        self,
        ax: Axes,
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        precip_valid: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        cmap: mcolors.Colormap,
        norm: BoundaryNorm,
        alpha: float
    ) -> None:
        """
        The method calculates an adaptive marker size based on the map extent and number of valid points to ensure visibility without overcrowding. It sorts the data by precipitation value to plot smaller values on top for better visibility of low precipitation areas. The scatter points are colored using the provided colormap and normalization, with a specified transparency level. This method does not add a colorbar; that is handled separately to ensure consistent formatting across different plot types.
        
        Parameters:
            ax (Axes): Target axes for overlay.
            lon_valid (np.ndarray): Valid longitude coordinates.
            lat_valid (np.ndarray): Valid latitude coordinates.
            precip_valid (np.ndarray): Valid precipitation data.
            lon_min (float): Western boundary.
            lon_max (float): Eastern boundary.
            lat_min (float): Southern boundary.
            lat_max (float): Northern boundary.
            cmap (mcolors.Colormap): Colormap for precipitation.
            norm (BoundaryNorm): Color normalization.
            alpha (float): Transparency level.
        """
        # Determine map extent for adaptive marker sizing
        map_extent = (lon_min, lon_max, lat_min, lat_max)

        # Calculate adaptive marker size based on map extent and number of points
        marker_size = self.calculate_adaptive_marker_size(map_extent, len(precip_valid), self.figsize)
        
        # Sort data for proper color overlay (plot smaller values on top for better visibility)
        sort_indices = np.argsort(precip_valid)
        lon_sorted = lon_valid[sort_indices]
        lat_sorted = lat_valid[sort_indices]
        precip_sorted = precip_valid[sort_indices]
        
        # Plot scatter points with appropriate color mapping and transparency
        ax.scatter(
            lon_sorted, lat_sorted, c=precip_sorted,
            cmap=cmap, norm=norm, s=marker_size, alpha=alpha,
            transform=ccrs.PlateCarree(), edgecolors='none'
        )
    
    def _render_overlay_interpolated(
        self,
        ax: Axes,
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        precip_valid: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        plot_type: str,
        resolution: float,
        var_name: str,
        dataset: Optional[xr.Dataset],
        cmap: mcolors.Colormap,
        norm: BoundaryNorm,
        color_levels_sorted: List[float],
        alpha: float,
        lon: np.ndarray,
        lat: np.ndarray
    ) -> None:
        """
        The method uses MPASRemapper to interpolate the scattered data onto a regular lat-lon grid, automatically handling masking of invalid points. The interpolation is performed with linear method for smooth results. After interpolation, the method extracts the regular grid coordinates and interpolated data values for plotting. Depending on the `plot_type`, it either draws contour lines with labels or filled contours, ensuring that the colorbar levels are consistent with the provided `color_levels_sorted`. The method includes logging of interpolation details and handles potential issues with contour labeling when there are many levels or masked data.
        
        Parameters:
            ax (Axes): Target axes for overlay.
            lon_valid (np.ndarray): Valid longitude coordinates.
            lat_valid (np.ndarray): Valid latitude coordinates.
            precip_valid (np.ndarray): Valid precipitation data.
            lon_min (float): Western boundary.
            lon_max (float): Eastern boundary.
            lat_min (float): Southern boundary.
            lat_max (float): Northern boundary.
            plot_type (str): Either 'contour' or 'contourf'.
            resolution (float): Grid resolution in degrees.
            var_name (str): Variable name for logging.
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinates.
            cmap (mcolors.Colormap): Colormap for precipitation.
            norm (BoundaryNorm): Color normalization.
            color_levels_sorted (List[float]): Sorted contour levels.
            alpha (float): Transparency level.
            lon (np.ndarray): Original longitude array.
            lat (np.ndarray): Original latitude array.
        """
        print(f"Interpolating {var_name} overlay using MPASRemapper (resolution: {resolution:.3f}°)")
        
        # Create dataset if not provided, ensuring lon and lat are numpy arrays for remapping
        if dataset is None:
            # Ensure lon and lat are numpy arrays for dataset creation
            lon_arr = lon if isinstance(lon, np.ndarray) else lon.values
            lat_arr = lat if isinstance(lat, np.ndarray) else lat.values

            # Create a minimal xarray Dataset with coordinate information for remapping
            dataset = xr.Dataset({
                'lonCell': xr.DataArray(lon_arr, dims=['nCells']),
                'latCell': xr.DataArray(lat_arr, dims=['nCells'])
            })
        
        # Create DataArray for remapping with proper dimensions
        data_xr = xr.DataArray(precip_valid, dims=['nCells'])

        # Use remap_mpas_to_latlon_with_masking for high-quality interpolation with automatic masking
        remapped_precip = remap_mpas_to_latlon_with_masking(
            data=data_xr,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            method='linear',
            apply_mask=True,
            lon_convention='auto'
        )
        
        # Extract grid and data for plotting
        lon_grid = remapped_precip.lon.values
        lat_grid = remapped_precip.lat.values
        precip_grid = remapped_precip.values
        
        print(f"Remapped to {precip_grid.shape[0]}x{precip_grid.shape[1]} grid")
        
        # Special handling for contour vs contourf to ensure colorbar consistency and proper rendering
        if plot_type == 'contour':
            # Draw contour lines with labels and handle potential issues with too many levels or masked data
            cs = ax.contour(
                lon_grid, lat_grid, precip_grid,
                levels=color_levels_sorted,
                colors='black',
                linewidths=1.0,
                linestyles='solid',
                alpha=alpha,
                transform=ccrs.PlateCarree()
            )
            try:
                # Add contour labels, handling potential issues with too many levels or masked data
                ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
            except Exception:
                pass
        else:  
            # For filled contours, we want to ensure that the colorbar reflects the same levels
            ax.contourf(
                lon_grid, lat_grid, precip_grid,
                levels=color_levels_sorted,
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                extend='both'
            )
    
    def add_precipitation_overlay(
        self,
        ax: Axes,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        precip_config: Dict[str, Any],
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None
    ) -> None:
        """
        Adds precipitation data as an overlay onto an existing map axes using configuration dictionary for flexible styling and data selection. This method enables overlaying precipitation fields on top of other meteorological variables (e.g., adding precipitation contours over temperature or wind maps). The method extracts precipitation data from the configuration, optionally regrids data to regular lat-lon grids using MPASRemapper for smooth interpolation, applies accumulation-period-specific colormaps and levels, filters valid precipitation values (non-negative, finite), and renders as scatter points or filled contours with customizable styling without creating a new figure. Map extent bounds are used for interpolation and filtering if provided, otherwise derived from data extent.

        Parameters:
            ax (Axes): Existing map axes (typically GeoAxes) to receive the precipitation overlay rendering.
            lon (Union[np.ndarray, xr.DataArray]): 1D longitude array in degrees for precipitation data positions.
            lat (Union[np.ndarray, xr.DataArray]): 1D latitude array in degrees for precipitation data positions.
            precip_config (Dict[str, Any]): Configuration dictionary with required key 'data' (ndarray) and optional keys:
                - 'data' (ndarray): Precipitation data array in mm or model units (REQUIRED)
                - 'accum_period' (str): Accumulation period ('a01h', 'a24h', etc.) for colormap selection (default: 'a01h')
                - 'plot_type' (str): 'scatter' for point display, 'contour' for lines, or 'contourf' for filled contours (default: 'scatter')
                - 'colormap' (str): Custom matplotlib colormap name overriding default (default: None uses precip colormap)
                - 'levels' (list): Explicit contour levels overriding defaults (default: None for automatic)
                - 'alpha' (float): Transparency level 0-1 (default: 0.7 for overlay visibility)
                - 'var_name' (str): Variable name for labeling (default: 'precipitation')
                - 'grid_resolution' (float): Target grid resolution in degrees for contour/contourf interpolation (default: None for adaptive)
                - 'original_units' (str): Original units of precipitation data for conversion (default: None for auto-detection)
            lon_min (Optional[float]): Western longitude bound for regridding and filtering (default: None derives from data).
            lon_max (Optional[float]): Eastern longitude bound for regridding and filtering (default: None derives from data).
            lat_min (Optional[float]): Southern latitude bound for regridding and filtering (default: None derives from data).
            lat_max (Optional[float]): Northern latitude bound for regridding and filtering (default: None derives from data).
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinate information, auto-created from lon/lat if not provided (default: None).

        Returns:
            None: Draws precipitation overlay directly onto provided axes without returning objects.

        Raises:
            ValueError: If plot_type is not 'scatter', 'contour', or 'contourf'.
        """
        # Extract precipitation data from configuration, ensuring required 'data' key is present
        precip_data = precip_config['data']

        # Get accumulation period for colormap selection, defaulting to 'a01h' if not specified
        accum_period = precip_config.get('accum_period', 'a01h')

        # Determine plot type for rendering (scatter, contour, or contourf)
        plot_type = precip_config.get('plot_type', 'scatter')

        # Get colormap from config or use default based on accumulation period
        colormap = precip_config.get('colormap', None)

        # Get contour levels from config or use defaults based on accumulation period
        levels = precip_config.get('levels', None)

        # Set default alpha for overlay visibility, can be overridden in config
        alpha = precip_config.get('alpha', 0.7)

        # Variable name for labeling and unit conversion
        var_name = precip_config.get('var_name', 'precipitation')

        # Calculate grid resolution for interpolation if provided in config, otherwise will be determined adaptively based on map extent and data density
        grid_resolution_input = precip_config.get('grid_resolution', None)

        # Retrieve original units for precipitation data conversion if provided
        original_units = precip_config.get('original_units', None)
        
        # Validate plot type for overlay rendering
        if plot_type not in ['scatter', 'contour', 'contourf']:
            raise ValueError(f"plot_type must be 'scatter', 'contour', or 'contourf', got '{plot_type}'")
        
        # Prepare data: convert units, flatten, validate, and determine bounds for interpolation
        try:
            # Special handling for overlay data preparation to ensure valid points and determine bounds for interpolation
            lon_valid, lat_valid, precip_valid, bounds_lon_min, bounds_lon_max, bounds_lat_min, bounds_lat_max = \
                self._prepare_overlay_data(lon, lat, precip_data, var_name, original_units, plot_type)
        except ValueError as e:
            # If no valid data points are found, print warning and skip rendering without raising exception to allow base map to display
            print(f"Warning: {e}")
            return
        
        # Use provided bounds or data-derived bounds for filtering and interpolation
        lon_min = lon_min if lon_min is not None else bounds_lon_min
        lon_max = lon_max if lon_max is not None else bounds_lon_max
        lat_min = lat_min if lat_min is not None else bounds_lat_min
        lat_max = lat_max if lat_max is not None else bounds_lat_max
        
        # Setup colormap and normalization for overlay based on accumulation period and configuration
        cmap, norm, color_levels_sorted = self._setup_overlay_colormap(colormap, levels, accum_period)
        
        print(f"Adding {len(precip_valid):,} precipitation points as {plot_type} overlay")
        
        # Render overlay based on plot type with appropriate handling for interpolation
        if plot_type == 'scatter':
            # Scatter plot can be rendered directly without interpolation
            self._render_overlay_scatter(
                ax, lon_valid, lat_valid, precip_valid,
                lon_min, lon_max, lat_min, lat_max,
                cmap, norm, alpha
            )
        else:  # (For contour or contourf)
            # Calculate grid resolution for interpolation if not provided
            resolution = self._calculate_overlay_grid_resolution(
                grid_resolution_input, lon_min, lon_max, lat_min, lat_max
            )

            # Convert original lon/lat to numpy for interpolation if they are xarray DataArrays
            lon_orig = self.convert_to_numpy(lon) if not isinstance(lon, np.ndarray) else lon
            lat_orig = self.convert_to_numpy(lat) if not isinstance(lat, np.ndarray) else lat
            
            # Render with interpolation using MPASRemapper for smooth contours
            self._render_overlay_interpolated(
                ax, lon_valid, lat_valid, precip_valid,
                lon_min, lon_max, lat_min, lat_max,
                plot_type, resolution, var_name, dataset,
                cmap, norm, color_levels_sorted, alpha,
                lon_orig, lat_orig
            )
        
        # Print summary of overlay addition for debugging
        print(f"Added {plot_type} precipitation overlay")
    
    def _extract_coordinates_from_processor(
        self,
        processor: Any,
        var_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The method first checks for a specific method to extract 2D coordinates for the given variable, then falls back to a general method for spatial coordinates, and finally defaults to direct dataset access if specific methods are not available. This approach ensures compatibility with different processor implementations while providing robust coordinate extraction for precipitation plotting.
        
        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            var_name (str): Variable name for coordinate extraction.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Longitude and latitude arrays.
        """
        try:
            # First try specific method for 2D coordinates if available
            if hasattr(processor, 'extract_2d_coordinates_for_variable'):
                return processor.extract_2d_coordinates_for_variable(var_name)
            
            # Next try general method for spatial coordinates if available
            elif hasattr(processor, 'extract_spatial_coordinates'):
                return processor.extract_spatial_coordinates()
            
            # Finally, fallback to direct dataset access if specific methods are not available
            else:
                return processor.dataset.lonCell.values, processor.dataset.latCell.values
        except AttributeError:
            # Fallback to direct dataset access if specific methods are not available
            return processor.dataset.lonCell.values, processor.dataset.latCell.values
    
    def _setup_batch_time_indices(
        self,
        processor: Any,
        accum_period: str,
        time_indices: Optional[List[int]]
    ) -> Tuple[List[int], int]:
        """
        The method determines the required number of hours for the specified accumulation period, checks if the dataset has enough time steps to support that accumulation, and filters user-specified time indices to ensure they are valid for the accumulation period. If no valid time indices are available, it returns an empty list and logs a warning, allowing calling code to handle the case of no available data gracefully without raising an exception. The method also returns the accumulation hours for use in title formatting and colormap selection.
        
        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            accum_period (str): Accumulation period identifier.
            time_indices (Optional[List[int]]): User-specified time indices or None for all.
            
        Returns:
            Tuple[List[int], int]: Validated time indices list and accumulation hours.
            
        Raises:
            ValueError: If no valid time indices are available for the accumulation period.
        """
        # Determine time dimension name (case-insensitive) and ensure it exists in the dataset
        time_dim = 'Time' if 'Time' in processor.dataset.sizes else 'time'

        # Determine total available time steps in the dataset for validation
        total_times = processor.dataset.sizes[time_dim]
        
        # Map accumulation period identifiers to required hours and corresponding time index offsets
        accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
        accum_hours = accum_hours_map.get(accum_period, 24)
        min_time_idx = accum_hours
        
        # Check if dataset has enough time steps for the accumulation period
        if min_time_idx >= total_times:
            print(f"\nWarning: Accumulation period {accum_period} ({accum_hours} hours) requires at least {min_time_idx + 1} time steps.")
            print(f"Dataset only has {total_times} time steps. No plots will be generated.")
            # Return empty list and accumulation hours for title formatting, even though no valid time indices are available
            return [], accum_hours
        
        # Validate and filter time indices based on accumulation period
        if time_indices is None:
            # Process all valid time indices starting from min_time_idx to total_times
            time_indices = list(range(min_time_idx, total_times))
        else:
            # Filter user-specified time indices to ensure they are valid for the accumulation period
            time_indices = [idx for idx in time_indices if idx >= min_time_idx]

            # Warn if user-specified time indices are invalid for the accumulation period
            if not time_indices:
                print(f"\nWarning: No valid time indices for accumulation period {accum_period}")
                return [], accum_hours
        
        # Return validated time indices and accumulation hours for title formatting
        return time_indices, accum_hours
    
    def _process_single_time_step(
        self,
        processor: Any,
        time_idx: int,
        lon: np.ndarray,
        lat: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        var_name: str,
        accum_period: str,
        plot_type: str,
        grid_resolution: Optional[float],
        colormap: Optional[str],
        levels: Optional[List[float]],
        custom_title_template: Optional[str],
        output_dir: str,
        file_prefix: str,
        formats: List[str]
    ) -> List[str]:
        """
        This method handles the entire workflow for one time step, including data extraction using PrecipitationDiagnostics, title generation with optional custom template, map creation with appropriate styling, and file saving with standardized naming. The method returns a list of created file paths for the processed time step, allowing batch processing to aggregate results. This encapsulation of single time step processing ensures that the batch method can focus on iteration and validation while this method handles the specifics of data handling and plotting for each time step.
        
        Parameters:
            processor: MPAS2DProcessor instance.
            time_idx (int): Time index to process.
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            lon_min (float): Western boundary.
            lon_max (float): Eastern boundary.
            lat_min (float): Southern boundary.
            lat_max (float): Northern boundary.
            var_name (str): Precipitation variable name.
            accum_period (str): Accumulation period identifier.
            plot_type (str): Rendering method.
            grid_resolution (Optional[float]): Grid resolution for interpolation.
            colormap (Optional[str]): Custom colormap name.
            levels (Optional[List[float]]): Custom contour levels.
            custom_title_template (Optional[str]): Custom title template.
            output_dir (str): Output directory path.
            file_prefix (str): Filename prefix.
            formats (List[str]): Output file formats.
            
        Returns:
            List[str]: List of created file paths.
        """
        # Extract precipitation data for this time step using PrecipitationDiagnostics
        if hasattr(processor.dataset, 'Time') and len(processor.dataset.Time) > time_idx:
            # Convert time coordinate to datetime for title formatting
            time_end = pd.Timestamp(processor.dataset.Time.values[time_idx]).to_pydatetime()
            time_str = time_end.strftime('%Y%m%dT%H')
        else:
            # Fallback to time index string if Time coordinate is not available
            time_end = None
            time_str = f"t{time_idx:03d}"
        
        # Create precipitation diagnostics instance for this time step
        precip_diag = PrecipitationDiagnostics(verbose=False)

        # Identify the correct method to compute precipitation difference based on processor capabilities
        precip_data = precip_diag.compute_precipitation_difference(
            processor.dataset, time_idx, var_name, accum_period, processor.data_type
        )
        
        # Generate title using custom template or default format
        if custom_title_template:
            # Use custom title template with placeholders for variable name, time string, accumulation period, and plot type
            title = custom_title_template.format(
                var_name=var_name.upper(),
                time_str=time_str,
                accum_period=accum_period,
                plot_type=plot_type
            )
        else:
            # Default title format matches original mpas_analysis style
            title = f"MPAS Precipitation | PlotType: {plot_type.upper()} | VarType: {var_name.upper()} | Valid Time: {time_str}"
        
        # Create precipitation map for this time step
        fig, ax = self.create_precipitation_map(
            lon, lat, precip_data.values,
            lon_min, lon_max, lat_min, lat_max,
            title=title,
            accum_period=accum_period,
            plot_type=plot_type,
            grid_resolution=grid_resolution,
            time_end=time_end,
            colormap=colormap,
            levels=levels,
            data_array=precip_data,
            var_name=var_name
        )
        
        # Construct output file path with standardized naming convention
        output_path = os.path.join(
            output_dir,
            f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_ptype_{plot_type}"
        )

        # Save plot to files
        self.save_plot(output_path, formats=formats)
        
        # Close the figure to free memory
        self.close_plot()
        
        # Return list of created file paths
        return [f"{output_path}.{fmt}" for fmt in formats]
    
    def create_batch_precipitation_maps(
        self,
        processor: Any,
        output_dir: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        var_name: str = 'rainnc',
        accum_period: str = 'a01h',
        plot_type: str = 'scatter',
        grid_resolution: Optional[float] = None,
        file_prefix: str = 'mpas_precipitation_map',
        formats: List[str] = ['png'],
        custom_title_template: Optional[str] = None,
        colormap: Optional[str] = None,
        levels: Optional[List[float]] = None,
        time_indices: Optional[List[int]] = None
    ) -> List[str]:
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
            plot_type (str): Rendering method - 'scatter' for direct cell display or 'contourf' for interpolated smooth fields (default: 'scatter').
            grid_resolution (Optional[float]): Target grid resolution in degrees for contourf interpolation (default: None uses adaptive).
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
        # Validate processor input
        if processor is None:
            raise ValueError("Processor cannot be None")
        
        # Validate that dataset is loaded in processor
        if processor.dataset is None:
            raise ValueError("No data loaded in processor")
        
        # Extract coordinates from processor dataset
        lon, lat = self._extract_coordinates_from_processor(processor, var_name)
        
        # Setup time indices for batch processing based on accumulation period requirements
        time_indices_to_process, accum_hours = self._setup_batch_time_indices(
            processor, accum_period, time_indices
        )
        
        # If no valid time indices, return empty list
        if not time_indices_to_process:
            return []
        
        # Estimate total time steps to process for progress tracking
        actual_time_steps = len(time_indices_to_process)

        # Initialize list to store created file paths
        created_files = []
        
        print(f"\nCreating precipitation maps for {actual_time_steps} time steps...")
        print(f"Using accumulation period: {accum_period} ({accum_hours} hours)")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate through time indices and create maps
        for i, time_idx in enumerate(time_indices_to_process):
            try:
                # Process single time step and collect created file paths
                files = self._process_single_time_step(
                    processor, time_idx, lon, lat,
                    lon_min, lon_max, lat_min, lat_max,
                    var_name, accum_period, plot_type,
                    grid_resolution, colormap, levels,
                    custom_title_template, output_dir,
                    file_prefix, formats
                )

                # Add created files to the list
                created_files.extend(files)
                
                # Show progress every 10 steps
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{actual_time_steps} maps (time index {time_idx})...")
                    
            except Exception as e:
                # Log error and continue with next time step
                print(f"Error creating map for time index {time_idx}: {e}")
                continue
        
        # Final progress message with total created files
        print(f"\nBatch processing completed. Created {len(created_files)} files in: {output_dir}")

        # Return list of created file paths
        return created_files
    
    def _setup_comparison_subplot(
        self,
        ax: GeoAxes,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        data_crs: ccrs.CRS,
        is_global: bool,
        panel_index: int
    ) -> None:
        """
        This method configures the GeoAxes with appropriate map features (coastlines, borders, land, ocean) and sets the extent based on provided longitude and latitude bounds. It includes special handling for global extents to avoid issues with the dateline in Cartopy by slightly adjusting the bounds. The method also adds gridlines with labels formatted as degrees and controls label visibility based on panel index to ensure a clean comparison layout. This setup allows for consistent styling across both panels in a side-by-side comparison of precipitation fields.
        
        Parameters:
            ax (Axes): The GeoAxes to configure.
            lon_min (float): Western boundary in degrees.
            lon_max (float): Eastern boundary in degrees.
            lat_min (float): Southern boundary in degrees.
            lat_max (float): Northern boundary in degrees.
            data_crs (ccrs.CRS): Coordinate reference system.
            is_global (bool): Whether the extent covers the globe.
            panel_index (int): Index of panel (0 for left, 1 for right) for label control.
        """
        # Set extent with dateline handling for global panels
        if is_global:
            # Adjust global extent slightly to avoid dateline issues with Cartopy
            adjusted_lon_min = max(lon_min, -179.99)
            adjusted_lon_max = min(lon_max, 179.99)
            adjusted_lat_min = max(lat_min, -89.99)
            adjusted_lat_max = min(lat_max, 89.99)

            # Set extent with adjusted bounds to avoid dateline issues
            ax.set_extent([adjusted_lon_min, adjusted_lon_max, adjusted_lat_min, adjusted_lat_max], crs=data_crs)

            # Print adjusted extent for global panel to inform about dateline handling
            if panel_index == 0:
                print(f"Using global extent (adjusted to avoid dateline): [{adjusted_lon_min}, {adjusted_lon_max}, {adjusted_lat_min}, {adjusted_lat_max}]")
        else:
            # For regional extents, use provided bounds directly
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        # Add geographic features with styling
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
        
        # Add regional features using existing method
        original_ax = self.ax
        self.ax = ax
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        self.ax = original_ax
        
        # Add gridlines with labels and formatting
        gl = ax.gridlines(crs=data_crs, draw_labels=True, 
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        # Only show top labels for the top row of panels (if there were more than 2 panels in a future extension)
        gl.top_labels = False

        # Control label visibility based on panel index (right labels for right panel, left labels for left panel)
        gl.right_labels = (panel_index == 1)
        gl.left_labels = (panel_index == 0)

        # Control label visibility and styling based on panel index
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # Format longitude and latitude labels
        gl.xformatter = FuncFormatter(self.format_longitude)
        gl.yformatter = FuncFormatter(self.format_latitude)
    
    def _plot_precipitation_data(
        self,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        cmap: mcolors.Colormap,
        norm: BoundaryNorm,
        data_crs: ccrs.CRS
    ) -> Optional[Any]:
        """
        The method first flattens the input data and coordinates, ensures they are of the same length, and creates a mask to filter out invalid data points (non-finite, negative, excessively large, or outside map extent). If no valid points remain after masking, it returns None to indicate that no plot was created. For valid data points, it calculates an adaptive marker size based on the map extent and number of points, sorts the data by value to ensure proper color overlay (lower values plotted first), and creates a scatter plot with the specified colormap and normalization. The method returns the scatter plot object for reference in colorbar creation.
        
        Parameters:
            ax (Axes): The GeoAxes to plot on.
            lon (np.ndarray): Longitude coordinates.
            lat (np.ndarray): Latitude coordinates.
            data (np.ndarray): Precipitation data values.
            lon_min (float): Western boundary in degrees.
            lon_max (float): Eastern boundary in degrees.
            lat_min (float): Southern boundary in degrees.
            lat_max (float): Northern boundary in degrees.
            cmap (mcolors.Colormap): Colormap for precipitation.
            norm (BoundaryNorm): Color normalization.
            data_crs (ccrs.CRS): Coordinate reference system.
            
        Returns:
            Optional[Any]: Scatter plot object if data is valid, None otherwise.
        """
        # Flatten data and coordinates for processing
        data_flat = np.asarray(data).flatten()
        lon_flat = np.asarray(lon).flatten()
        lat_flat = np.asarray(lat).flatten()
        
        # Ensure all arrays have the same length for proper masking and plotting
        min_length = min(len(data_flat), len(lon_flat), len(lat_flat))

        # Truncate arrays to the minimum length to avoid misalignment
        data_flat = data_flat[:min_length]
        lon_flat = lon_flat[:min_length]
        lat_flat = lat_flat[:min_length]
        
        # Create mask for valid data points (finite, non-negative, within reasonable bounds, and within map extent)
        valid_mask = (np.isfinite(data_flat) & 
                     (data_flat >= 0) & 
                     (data_flat < 1e5) &
                     (lon_flat >= lon_min) & (lon_flat <= lon_max) &
                     (lat_flat >= lat_min) & (lat_flat <= lat_max))
        
        # Check if there are valid points to plot
        if not np.any(valid_mask):
            return None
        
        # Extract valid data for plotting
        lon_valid = lon_flat[valid_mask]
        lat_valid = lat_flat[valid_mask]
        data_valid = data_flat[valid_mask]
        
        # Define map extent for marker size calculation
        map_extent = (lon_min, lon_max, lat_min, lat_max)

        # Calculate adaptive marker size based on map extent and number of valid points
        marker_size = self.calculate_adaptive_marker_size(map_extent, len(data_valid), self.figsize)
        
        # Sort by data values to ensure proper color overlay (lower values plotted first)
        sort_indices = np.argsort(data_valid)
        lon_sorted = lon_valid[sort_indices]
        lat_sorted = lat_valid[sort_indices]
        data_sorted = data_valid[sort_indices]
        
        # Plot scatter and return the scatter object for colorbar reference
        scatter = ax.scatter(lon_sorted, lat_sorted, c=data_sorted, 
                           cmap=cmap, norm=norm, s=marker_size, alpha=0.9, 
                           transform=data_crs, edgecolors='none')
        
        # Return scatter for colorbar reference
        return scatter
    
    def create_precipitation_comparison_plot(
        self,
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
        projection: str = 'PlateCarree'
    ) -> Tuple[Figure, List[Axes]]:
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
        # Setup map projection and coordinate reference system based on provided geographic bounds and projection type
        map_proj, data_crs = self.setup_map_projection(lon_min, lon_max, lat_min, lat_max, projection)

        # Create figure with two subplots sharing the same map projection and size
        self.fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.8, self.figsize[1]), 
                                     dpi=self.dpi, subplot_kw={'projection': map_proj})
        
        # Ensure axes is a list for consistent processing
        axes = list(axes)
        
        # Create shared colormap and normalization based on accumulation period
        cmap, color_levels = self.create_precip_colormap(accum_period)

        # Sort and filter color levels to ensure valid contour levels for normalization
        color_levels_sorted = sorted(set([v for v in color_levels if np.isfinite(v)]))

        # Add an upper bound to the color levels for normalization to ensure all data values are properly colored
        last_bound = max(color_levels_sorted) + 1
        bounds = [0] + color_levels_sorted + [last_bound]

        # Create normalization for colormap based on bounds and number of colors in the colormap
        norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        
        # Determine if the extent is global to adjust map features and avoid dateline issues
        is_global = (lon_max - lon_min) >= 359.0 and (lat_max - lat_min) >= 179.0
        
        # Initialize scatter variable for colorbar reference
        scatter = None

        # Loop through each subplot and corresponding data to setup map features and plot precipitation
        for i, (ax, data, title) in enumerate(zip(axes, [precip_data1, precip_data2], [title1, title2])):
            # Setup subplot with map features and extent
            self._setup_comparison_subplot(ax, lon_min, lon_max, lat_min, lat_max, 
                                          data_crs, is_global, i)
            
            # Plot precipitation data and capture scatter result for colorbar reference
            scatter_result = self._plot_precipitation_data(ax, lon, lat, data, 
                                                          lon_min, lon_max, lat_min, lat_max,
                                                          cmap, norm, data_crs)
            
            # Store scatter result for colorbar reference if valid
            if scatter_result is not None:
                scatter = scatter_result
            
            # Set subplot title with consistent styling
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        # Ensure at least one subplot has valid data for the colorbar
        assert scatter is not None, "At least one subplot must have valid data for colorbar"

        # Position colorbar below the two panels, centered, with appropriate padding and aspect ratio
        cbar = self.fig.colorbar(scatter, ax=axes, orientation='horizontal', extend='both',
                               pad=0.08, shrink=0.6, aspect=30)
        
        # Set colorbar label and tick parameters for readability
        cbar.set_label('Precipitation [mm]', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        
        if len(color_levels_sorted) <= 15:
            # Set colorbar ticks to match contour levels for direct interpretation
            cbar.set_ticks(color_levels_sorted)

            # Format tick labels dynamically based on value range and magnitude for better readability
            cbar.set_ticklabels(self._format_ticks_dynamic(color_levels_sorted))
        
        # Add timestamp and branding to the figure
        self.add_timestamp_and_branding()

        # Adjust layout to prevent overlap and ensure clear presentation
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.15)
        
        # Return the figure and axes for further manipulation or saving
        return self.fig, axes
    
    def save_plot(
        self,
        output_path: str,
        formats: List[str] = ['png'],
        bbox_inches: str = 'tight',
        pad_inches: float = 0.1
    ) -> None:
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
        # Check if there is an active figure to save
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")

        # Determine output directory from the provided output path
        output_dir = os.path.dirname(output_path)

        # Ensure output directory exists, creating it if necessary
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for fmt in formats:
            # Construct full file path with format extension
            full_path = f"{output_path}.{fmt}"

            # Define savefig options with specified format and bounding box settings
            save_kwargs = {'dpi': self.dpi, 'bbox_inches': bbox_inches, 'pad_inches': pad_inches, 'format': fmt}

            # Optimize PNG files with low compression for faster saving while maintaining quality, especially for large precipitation maps with many points.
            if fmt.lower() == 'png':
                save_kwargs['pil_kwargs'] = {'compress_level': 1}

            # Save the figure with the specified format and options
            self.fig.savefig(full_path, **save_kwargs)

            # Print confirmation message for the saved file
            print(f"Saved plot: {full_path}")
    
    def close_plot(self) -> None:
        """
        Close the current matplotlib figure and release all associated references to free memory resources. This method calls plt.close() on the active figure stored in self.fig and sets both self.fig and self.ax to None to ensure proper garbage collection. Closing plots is essential in batch processing workflows to prevent memory accumulation when generating many sequential visualizations. This cleanup should be called after each plot is saved or when switching between different visualization tasks. The method safely handles cases where fig is already None, making it safe to call multiple times.

        Returns:
            None: Closes figure and clears instance attributes self.fig and self.ax.
        """
        # Close the figure if it exists and clear references
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _format_ticks_dynamic(
        self,
        ticks: List[float]
    ) -> List[str]:
        """
        Format axis tick labels dynamically with intelligent precision selection based on the numeric range and magnitude of tick values. This method delegates to MPASVisualizationStyle.format_ticks_dynamic() which analyzes the tick value array to determine appropriate decimal places and formatting style (scientific notation for very large/small values, fixed decimal for typical ranges). The dynamic formatting ensures tick labels are concise yet informative, avoiding excessive decimal places for large numbers while maintaining necessary precision for small values. This adaptive approach produces clean, professional axis labels across diverse data ranges encountered in meteorological visualizations, from millimeter-scale precipitation to large geographic coordinates.

        Parameters:
            ticks (List[float]): Array of numeric tick values to be formatted for axis display.
            
        Returns:
            List[str]: Array of formatted tick label strings with appropriate precision and notation style.
        """
        # Delegate to MPASVisualizationStyle for dynamic tick formatting based on value range and magnitude
        return MPASVisualizationStyle.format_ticks_dynamic(ticks)
    
    def apply_style(
        self,
        style_name: str = 'default'
    ) -> None:
        """
        Apply a named visualization style to the plotter and active figure for consistent professional appearance across all precipitation plots. This method delegates to the style_manager if available to apply registered matplotlib style configurations (fonts, colors, line widths, etc.), then sets figure and axes background colors to standard defaults (light gray for axes, white for figure background). Style application affects all subsequent plotting operations and can be used to implement organization-specific branding, publication requirements, or different visual themes for presentation vs print media. The method safely handles cases where no style_manager is configured or no figure/axes are active, making it suitable for initialization and runtime style switching.

        Parameters:
            style_name (str): Name of the registered visualization style to apply from the style manager's style registry (default: 'default').

        Returns:
            None: Modifies appearance of self.ax and self.fig if they exist, applying colors and delegating to style_manager for broader matplotlib settings.
        """
        # Get style manager if it exists
        style_manager = getattr(self, 'style_manager', None)

        # Apply style from style manager if available
        if style_manager:
            style_manager.apply_style(style_name)

        # Set axes background to light gray for better contrast with precipitation colors
        if self.ax is not None:
            self.ax.set_facecolor('#f0f0f0')

        # Set figure background to white for better contrast and printing
        if self.fig is not None:
            self.fig.patch.set_facecolor('white')