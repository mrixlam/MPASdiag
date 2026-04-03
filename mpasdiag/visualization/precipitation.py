#!/usr/bin/env python3

"""
MPASdiag Core Visualization Module: Precipitation Plotting and Overlays

This module provides the MPASPrecipitationPlotter class, which specializes in creating high-quality cartographic visualizations of precipitation fields from MPAS unstructured mesh data. It supports multiple accumulation periods (e.g., hourly, daily), automatic unit conversion, and flexible rendering options including scatter plots of cell values and interpolated contour/filled contour maps. The plotter handles geographic features, map projections, and color mapping with meteorologically appropriate defaults while allowing extensive customization. It also includes robust data validation and formatting to ensure professional presentation of precipitation diagnostics. 
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Import standard libraries
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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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
    """ Specialized plotter for creating professional cartographic visualizations of MPAS precipitation fields with support for multiple accumulation periods, automatic unit conversion, and flexible rendering options. """
    
    def __init__(self: "MPASPrecipitationPlotter", 
                 figsize: Tuple[float, float] = (10, 14), 
                 dpi: int = 100) -> None:
        """
        This initializer sets up the MPASPrecipitationPlotter with default figure size and resolution. It calls the parent class initializer to ensure any necessary setup from MPASVisualizer is performed. The figsize parameter controls the dimensions of the output figure in inches, while dpi sets the resolution for rendering and saving the figure. These defaults can be overridden when instantiating the plotter to suit specific presentation or publication requirements. The initializer does not perform any plotting itself but prepares the instance for subsequent plotting operations.

        Parameters:
            figsize (Tuple[float, float]): Default figure size in inches (width, height).
            dpi (int): Default resolution in dots per inch for rendering and saving figures.

        Returns:
            None
        """
        super().__init__(figsize, dpi)
    
    def create_precip_colormap(self: "MPASPrecipitationPlotter",
                               accum: str = "a24h") -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        This method creates a discrete colormap and corresponding contour levels for precipitation visualization based on the specified accumulation period. It delegates to the MPASVisualizationStyle class to generate a colormap that is appropriate for the given accumulation period (e.g., hourly, daily) and returns both the colormap and a sorted list of contour level values that can be used for consistent coloring and colorbar ticks in precipitation maps. The accumulation period identifier (e.g., 'a24h' for 24-hour accumulation) is used to select predefined color schemes and levels that are meteorologically meaningful for visualizing precipitation fields. 

        Parameters:
            accum (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to select appropriate colormap and contour levels for precipitation visualization. 

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: A tuple containing the created discrete colormap and a sorted list of contour level values suitable for precipitation plotting. 
        """
        return MPASVisualizationStyle.create_precip_colormap(accum)
    
    def _convert_precipitation_units(self: "MPASPrecipitationPlotter",
                                     precip_data: np.ndarray,
                                     data_array: Optional[xr.DataArray],
                                     var_name: str) -> Tuple[np.ndarray, str]:
        """
        This method handles unit conversion for precipitation data if an xarray DataArray with appropriate metadata is provided. It uses the UnitConverter to convert the input precipitation data to display units (e.g., mm) based on the variable name and metadata attributes. If conversion fails due to missing attributes or incompatible units, it logs a warning and returns the original data with a default unit label of 'mm'. The method also checks for negative precipitation values, which are physically invalid, and clips them to 0 while logging a warning with the count and minimum value. The returned precipitation data is guaranteed to be in display units suitable for plotting, and the unit label can be used for colorbar annotation. 

        Parameters:
            precip_data (np.ndarray): Input precipitation data array, which may be in model units or already in display units.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing metadata attributes for unit conversion. If None, no conversion is attempted and original data is returned.
            var_name (str): Variable name used for metadata lookup and unit conversion logic. 

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the converted precipitation data array (or original if conversion fails) and a string representing the unit label for colorbar annotation (defaulting to 'mm' if conversion is not possible). 
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
    
    def _setup_precipitation_figure(self: "MPASPrecipitationPlotter",
                                    lon_min: float,
                                    lon_max: float,
                                    lat_min: float,
                                    lat_max: float,
                                    projection: str) -> Tuple[ccrs.Projection, ccrs.CRS]:
        """
        This method sets up the figure and GeoAxes for precipitation plotting using cartopy. It configures the map projection based on the specified projection name and geographic extent, and adds relevant geographic features such as coastlines, borders, ocean, and land with styling appropriate for precipitation maps. The method also handles global map extents by adjusting them slightly to avoid dateline issues. It returns the configured map projection and data CRS for use in plotting transforms. The created figure and axes are stored as instance attributes for use in subsequent plotting methods.  

        Parameters:
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            projection (str): Cartopy projection identifier (e.g., 'PlateCarree', 'Mercator') to use for the map. 

        Returns:
            Tuple[ccrs.Projection, ccrs.CRS]: A tuple containing the configured cartopy map projection for the axes and the data CRS for plotting transforms. The figure and axes are also stored as instance attributes for use in plotting.  
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
    
    def _prepare_precipitation_colormap(self: "MPASPrecipitationPlotter",
                                        colormap: Optional[str],
                                        levels: Optional[List[float]],
                                        accum_period: str,
                                        clim_min: Optional[float],
                                        clim_max: Optional[float]) -> Tuple[mcolors.Colormap, BoundaryNorm, List[float]]:
        """
        This method prepares the colormap, normalization, and contour levels for precipitation plotting based on user input and accumulation period. It allows for custom colormap and levels to be specified, or defaults to predefined schemes based on the accumulation period. The method also applies color limits if provided, ensuring that the levels used for coloring and colorbar ticks are consistent with the specified limits. It returns the selected colormap, a BoundaryNorm for discrete coloring, and a sorted list of contour levels that can be used for consistent visualization of precipitation fields. 

        Parameters:
            colormap (Optional[str]): Optional matplotlib colormap name to use for precipitation plotting. If None, a default colormap based on the accumulation period will be used.
            levels (Optional[List[float]]): Optional list of contour levels to use for coloring and colorbar ticks. If None, default levels based on the accumulation period will be used.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to select default colormap and levels if custom ones are not provided.
            clim_min (Optional[float]): Optional minimum color limit to clip contour levels for coloring and colorbar ticks.
            clim_max (Optional[float]): Optional maximum color limit to clip contour levels for coloring and colorbar ticks. 

        Returns:
            Tuple[mcolors.Colormap, BoundaryNorm, List[float]]: A tuple containing the selected colormap, a BoundaryNorm for discrete coloring based on the levels and color limits, and a sorted list of contour levels that are consistent with the specified limits and accumulation period.  
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
    
    def _prepare_precipitation_data(self: "MPASPrecipitationPlotter",
                                    lon: np.ndarray,
                                    lat: np.ndarray,
                                    precip_data: np.ndarray,
                                    lon_min: float,
                                    lon_max: float,
                                    lat_min: float,
                                    lat_max: float,
                                    plot_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method prepares and validates the longitude, latitude, and precipitation data for plotting. It flattens the input arrays to ensure they are 1D and properly aligned for scatter plotting. For scatter plots, it creates a validation mask that checks for finite precipitation values within a reasonable range (e.g., non-negative and below a high threshold) and ensures that the corresponding longitude and latitude points fall within the specified geographic extent. For contour and contourf plots, it focuses on validating the precipitation values since geographic masking will be handled during interpolation. The method returns only the valid points that can be plotted, ensuring that the visualization is based on clean and physically meaningful data. If no valid points exist after filtering, it returns empty arrays, allowing the plotting function to handle this case gracefully (e.g., by showing an empty map with features but no data). 

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS cell centers.
            precip_data (np.ndarray): Precipitation values corresponding to the lon/lat points (in display units).
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            plot_type (str): Rendering mode: 'scatter', 'contour', or 'contourf'. This determines the validation criteria applied to the data. 

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the valid longitude, latitude, and precipitation data arrays that can be plotted. If no valid points exist, empty arrays are returned. The returned arrays are 1D and aligned for plotting with matplotlib functions. 
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
    
    def _add_time_annotation(self: "MPASPrecipitationPlotter",
                             time_end: Optional[datetime],
                             time_start: Optional[datetime],
                             accum_period: str) -> None:
        """
        This method adds a time annotation to the precipitation plot based on the provided end time, start time, and accumulation period. If the end time is provided, it formats the annotation to show the accumulation period in terms of UTC times. If only the accumulation period is provided without specific times, it annotates with the accumulation period identifier. The annotation is added as a text box in the lower left corner of the plot with styling to ensure readability against the map features. The method asserts that the axes are properly set up for cartopy plotting before adding the annotation. This provides important context for interpreting the precipitation field in terms of its temporal coverage. 
        
        Parameters:
            time_end (Optional[datetime]): Optional end datetime for the accumulation period. If provided, it will be used to derive the start time based on the accumulation period if the start time is not explicitly provided.
            time_start (Optional[datetime]): Optional start datetime for the accumulation period. If not provided but time_end is given, it will be derived from time_end and accum_period.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to determine the duration of the accumulation for annotation purposes. 

        Returns:
            None
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
    
    def create_precipitation_map(self: "MPASPrecipitationPlotter",
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
                                 dataset: Optional[xr.Dataset] = None) -> Tuple[Figure, Axes]:
        """
        This method creates a precipitation map from MPAS unstructured mesh data using cartopy for geographic visualization. It supports multiple rendering options including scatter plots of cell values and interpolated contour/filled contour maps. The method handles unit conversion, colormap selection, data validation, and geographic features to produce a professional-quality visualization of precipitation fields. It also includes robust handling of edge cases such as empty or invalid data, and provides informative annotations about the accumulation period. The resulting figure and axes are returned for further customization or saving. 

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS cell centers.
            precip_data (np.ndarray): Precipitation values corresponding to the lon/lat points (in model or display units).
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            title (str): Title for the plot.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used for colormap selection and annotation.
            plot_type (str): Rendering mode: 'scatter', 'contour', or 'contourf'.
            colormap (Optional[str]): Optional matplotlib colormap name to use for precipitation plotting. If None, a default colormap based on the accumulation period will be used.
            levels (Optional[List[float]]): Optional list of contour levels to use for coloring and colorbar ticks. If None, default levels based on the accumulation period will be used.
            clim_min (Optional[float]): Optional minimum color limit to clip contour levels for coloring and colorbar ticks.
            clim_max (Optional[float]): Optional maximum color limit to clip contour levels for coloring and colorbar ticks.
            projection (str): Cartopy projection identifier (e.g., 'PlateCarree', 'Mercator') to use for the map.
            time_end (Optional[datetime]): Optional end datetime for the accumulation period. If provided, it will be used to derive the start time based on the accumulation period if the start time is not explicitly provided.
            time_start (Optional[datetime]): Optional start datetime for the accumulation period. If not provided but time_end is given, it will be derived from time_end and accum_period.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing metadata attributes for unit conversion. If None, no conversion is attempted and original data is used.
            var_name (str): Variable name used for metadata lookup and unit conversion logic.
            grid_resolution (Optional[float]): Optional grid resolution in degrees for contour/contourf interpolation. If None, a default resolution will be determined based on the data density and geographic extent.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset containing the original MPAS data, which may be needed for interpolation and remapping in contour/contourf plots.

        Returns:
            Tuple[Figure, Axes]: A tuple containing the created matplotlib Figure and Axes objects for the precipitation map. The figure and axes are also stored as instance attributes for further manipulation or saving.
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
                self._create_scatter_plot(
                    lon_valid, lat_valid, precip_valid, cmap, norm, data_crs,
                    colorbar_label=f'Precipitation [{unit_label}]',
                    colorbar_ticks=color_levels_sorted,
                )
            elif plot_type == 'contour':
                # Contour requires interpolation to a regular grid, which is handled in the helper method using MPASRemapper
                self._create_contour_plot(
                    lon_valid, lat_valid, precip_valid,
                    lon_min, lon_max, lat_min, lat_max,
                    cmap, norm, color_levels_sorted, data_crs,
                    grid_resolution, dataset,
                )
            elif plot_type == 'contourf':
                # Contourf requires interpolation to a regular grid, which is handled in the helper method using MPASRemapper
                self._create_contourf_plot(
                    lon_valid, lat_valid, precip_valid,
                    lon_min, lon_max, lat_min, lat_max,
                    cmap, norm, color_levels_sorted, data_crs,
                    grid_resolution, dataset,
                    colorbar_label=f'Precipitation [{unit_label}]',
                    colorbar_ticks=color_levels_sorted,
                )
        else:
            print(f"Warning: No valid precipitation data points found for {var_name}")
        
        # Add gridlines with labels and custom formatting
        self._add_gridlines(data_crs)
        
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
    
    def _interpolate_to_grid(self: "MPASPrecipitationPlotter",
                             lon: np.ndarray,
                             lat: np.ndarray,
                             data: np.ndarray,
                             lon_min: float,
                             lon_max: float,
                             lat_min: float,
                             lat_max: float,
                             grid_resolution: Optional[float] = None,
                             dataset: Optional[xr.Dataset] = None,
                             method: str = 'linear',
                             resolution_bounds: Optional[Tuple[float, float]] = (0.1, 1.0),) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method performs interpolation of the unstructured MPAS data onto a regular grid suitable for contour and contourf plotting. It uses the MPASRemapper to handle the interpolation, which is designed to work with MPAS datasets and can account for the unstructured nature of the data. The method takes into account the geographic extent and desired grid resolution, and allows for different interpolation methods (e.g., 'linear', 'nearest') as well as optional resolution bounds to prevent over-interpolation. The resulting gridded longitude, latitude, and interpolated precipitation data arrays are returned for use in contour plotting. 

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS cell centers.
            data (np.ndarray): Precipitation values corresponding to the lon/lat points (in display units).
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            grid_resolution (Optional[float]): Desired grid resolution in degrees for interpolation. If None, a default resolution will be determined based on data density and geographic extent.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset containing the original MPAS data, which may be needed for interpolation and remapping. If None, interpolation will be attempted with available data arrays.
            method (str): Interpolation method to use ('linear', 'nearest', etc.) for gridding the data. Default is 'linear'.
            resolution_bounds (Optional[Tuple[float, float]]): Optional tuple specifying minimum and maximum allowed grid resolution in degrees to prevent over-interpolation or under-sampling. Default is (0.1, 1.0) degrees.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the gridded longitude, latitude, and interpolated precipitation data arrays suitable for contour plotting. The longitude and latitude arrays will be 2D grids corresponding to the shape of the interpolated data array. 
        """
        return super()._interpolate_to_grid(
            lon, lat, data, lon_min, lon_max, lat_min, lat_max,
            grid_resolution, dataset, method=method,
            resolution_bounds=resolution_bounds,
        )
    
    def _prepare_overlay_data(self: "MPASPrecipitationPlotter",
                              lon: Union[np.ndarray, xr.DataArray],
                              lat: Union[np.ndarray, xr.DataArray],
                              precip_data: np.ndarray,
                              var_name: str,
                              original_units: Optional[str],
                              plot_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """
       This method prepares and validates the longitude, latitude, and precipitation data for overlay plotting. It handles unit conversion if original units are provided, checks for physically valid precipitation values (e.g., non-negative), and creates a validity mask based on the plot type. For scatter plots, it filters out points that fall outside the map extent in addition to invalid precipitation values. For contour and contourf plots, it focuses on validating the precipitation values since geographic masking will be handled during interpolation. The method returns only the valid points that can be plotted along with the geographic bounds for use in rendering the overlay. If no valid points exist after filtering, it raises a ValueError to indicate that the overlay cannot be plotted with the given data. 
        
        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates for the overlay data, which can be a numpy array or an xarray DataArray.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates for the overlay data, which can be a numpy array or an xarray DataArray.
            precip_data (np.ndarray): Precipitation values corresponding to the lon/lat points (in model or display units).
            var_name (str): Variable name used for metadata lookup and unit conversion logic.
            original_units (Optional[str]): Original units of the precipitation data, used for unit conversion if provided. If None, no conversion is attempted and original data is used.
            plot_type (str): Rendering mode: 'scatter', 'contour', or 'contourf'. This determines the validation criteria applied to the data.
            
        Returns:
            Tuple containing (lon_valid, lat_valid, precip_valid, lon_min, lon_max, lat_min, lat_max) where the first three are 1D arrays of valid points for plotting and the last four are the geographic bounds of the data for use in rendering the overlay. If no valid points exist, a ValueError is raised. 
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
    
    def _setup_overlay_colormap(self: "MPASPrecipitationPlotter",
                                colormap: Optional[str],
                                levels: Optional[List[float]],
                                accum_period: str) -> Tuple[mcolors.Colormap, BoundaryNorm, List[float]]:
        """
        This method sets up the colormap and normalization for the precipitation overlay based on user input and defaults. It allows for custom colormap and levels to be specified, but also provides sensible defaults based on the accumulation period if custom values are not provided. The method ensures that the levels used for coloring and colorbar ticks are consistent with the specified accumulation period and any provided color limits. It returns a colormap, a BoundaryNorm for discrete coloring, and a sorted list of contour levels that can be used for consistent visualization of the precipitation overlay. 
        
        Parameters:
            colormap (Optional[str]): Optional matplotlib colormap name to use for precipitation plotting. If None, a default colormap based on the accumulation period will be used.
            levels (Optional[List[float]]): Optional list of contour levels to use for coloring and colorbar ticks. If None, default levels based on the accumulation period will be used.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to select default colormap and levels if custom ones are not provided.
            
        Returns:
            Tuple[mcolors.Colormap, BoundaryNorm, List[float]]: A tuple containing the selected colormap, a BoundaryNorm for discrete coloring based on the levels and color limits, and a sorted list of contour levels that are consistent with the specified limits and accumulation period. 
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
    
    def _calculate_overlay_grid_resolution(self: "MPASPrecipitationPlotter",
                                           grid_resolution_input: Optional[float],
                                           lon_min: float,
                                           lon_max: float,
                                           lat_min: float,
                                           lat_max: float) -> float:
        """
        This method calculates an appropriate grid resolution for interpolating the precipitation overlay based on the geographic extent of the data and the number of valid points. If the user has provided a specific grid resolution, it will be used directly. Otherwise, the method calculates a default resolution as a percentage of the larger dimension of the map extent (longitude or latitude range) to ensure that the interpolation is neither too coarse nor too fine for the given data. The calculated resolution is then clipped to reasonable bounds (e.g., between 0.1 and 1 degree) to prevent over-interpolation or under-sampling, which can lead to poor visualization quality. This adaptive approach allows for better handling of different spatial scales and data densities in precipitation maps. 
        
        Parameters:
            grid_resolution_input (Optional[float]): User-specified grid resolution in degrees for interpolation. If provided, this value will be used directly without calculation.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees. 
            
        Returns:
            float: Calculated grid resolution in degrees for interpolation, either based on user input or adaptive calculation from the map extent. The returned value is clipped to reasonable bounds to ensure good visualization quality. 
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
    
    def _render_overlay_scatter(self: "MPASPrecipitationPlotter",
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
                                alpha: float) -> None:
        """
        This method renders the precipitation overlay as a scatter plot on the provided axes. It takes the valid longitude, latitude, and precipitation data points and plots them using a scatter plot with colors determined by the provided colormap and normalization. The method calculates an adaptive marker size based on the map extent and number of points to ensure that the scatter points are visible without being too large or too small. The points are plotted with a specified transparency level (alpha) to allow for better visualization of overlapping points and underlying map features. The method also sorts the data by precipitation values to ensure that smaller values are plotted on top for better visibility in cases of overlapping points. This approach allows for a clear visualization of the spatial distribution of precipitation values across the map. 
        
        Parameters:
            ax (Axes): Target axes for overlay.
            lon_valid (np.ndarray): Valid longitude coordinates for plotting.
            lat_valid (np.ndarray): Valid latitude coordinates for plotting.
            precip_valid (np.ndarray): Valid precipitation values for plotting.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            cmap (mcolors.Colormap): Colormap for precipitation values.
            norm (BoundaryNorm): Normalization for color mapping of precipitation values.
            alpha (float): Transparency level for scatter points (0 to 1) to allow for better visualization of overlapping points and underlying map features.

        Returns:
            None: This method modifies the provided axes in place by adding the scatter points for the precipitation overlay. It does not return any value.
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
    
    def _render_overlay_interpolated(self: "MPASPrecipitationPlotter",
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
                                     lat: np.ndarray) -> None:
        """
        This method renders the precipitation overlay as either contour lines or filled contours based on the specified plot type. It first performs interpolation of the valid precipitation data onto a regular lat-lon grid using the MPASRemapper, which is designed to handle the unstructured nature of MPAS data and can apply masking to ensure that only valid data points contribute to the interpolation. The method then extracts the gridded longitude, latitude, and interpolated precipitation values for plotting. For contour plots, it draws contour lines with specified levels and adds labels to the contours for better readability. For filled contour plots, it fills the contours with colors based on the provided colormap and normalization, ensuring that the colorbar reflects the same levels for consistency. The method also applies a specified transparency level (alpha) to allow for better visualization of underlying map features while still clearly showing the precipitation patterns. This approach allows for a smooth and visually appealing representation of precipitation fields derived from unstructured MPAS data. 
        
        Parameters:
            ax (Axes): Target axes for overlay.
            lon_valid (np.ndarray): Valid longitude coordinates for plotting.
            lat_valid (np.ndarray): Valid latitude coordinates for plotting.
            precip_valid (np.ndarray): Valid precipitation values for plotting.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            plot_type (str): Rendering mode: 'contour' for lines or 'contourf' for filled contours.
            resolution (float): Grid resolution in degrees for interpolation.
            var_name (str): Variable name for labeling and unit conversion logic.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset containing the original MPAS data, which may be needed for interpolation and remapping. If None, interpolation will be attempted with available data arrays.
            cmap (mcolors.Colormap): Colormap for precipitation values.
            norm (BoundaryNorm): Normalization for color mapping of precipitation values.
            color_levels_sorted (List[float]): Sorted list of contour levels for consistent coloring and colorbar ticks.
            alpha (float): Transparency level for contours (0 to 1) to allow for better visualization of underlying map features while still showing precipitation patterns.
            lon (np.ndarray): Original longitude array for the MPAS cell centers, used for interpolation.
            lat (np.ndarray): Original latitude array for the MPAS cell centers, used for interpolation.

        Returns:
            None: This method modifies the provided axes in place by adding the contour lines or filled contours for the precipitation overlay. It does not return any value.
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
    
    def add_precipitation_overlay(self: "MPASPrecipitationPlotter",
                                  ax: Axes,
                                  lon: Union[np.ndarray, xr.DataArray],
                                  lat: Union[np.ndarray, xr.DataArray],
                                  precip_config: Dict[str, Any],
                                  lon_min: Optional[float] = None,
                                  lon_max: Optional[float] = None,
                                  lat_min: Optional[float] = None,
                                  lat_max: Optional[float] = None,
                                  dataset: Optional[xr.Dataset] = None) -> None:
        """
        This method adds a precipitation overlay to an existing map axes (typically GeoAxes) using the provided longitude, latitude, and precipitation data along with a configuration dictionary. It supports rendering the overlay as either scatter points, contour lines, or filled contours based on the specified plot type in the configuration. The method handles unit conversion for the precipitation data if original units are provided, and it validates the data to ensure that only physically valid values are plotted. For scatter plots, it filters out points that fall outside the map extent to avoid plotting irrelevant data. For contour and contourf plots, it relies on interpolation and remapping to handle geographic masking. The method also sets up the colormap and normalization based on the accumulation period and any custom configuration provided. It calculates an appropriate grid resolution for interpolation if needed and uses the MPASRemapper for high-quality interpolation of unstructured MPAS data onto a regular grid suitable for contour plotting. Finally, it renders the overlay with a specified transparency level to allow for better visualization of underlying map features while still clearly showing the precipitation patterns. 

        Parameters:
            ax (Axes): Target axes for overlay.
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates for the overlay data, which can be a numpy array or an xarray DataArray.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates for the overlay data, which can be a numpy array or an xarray DataArray.
            precip_config (Dict[str, Any]): Configuration dictionary containing precipitation data and plotting options. Required keys include 'data' for precipitation values, and optional keys include 'accum_period', 'plot_type', 'colormap', 'levels', 'alpha', 'var_name', 'original_units', and 'grid_resolution'.
            lon_min (Optional[float]): Western boundary of the map extent in degrees. If None, it will be determined from the data bounds.
            lon_max (Optional[float]): Eastern boundary of the map extent in degrees. If None, it will be determined from the data bounds.
            lat_min (Optional[float]): Southern boundary of the map extent in degrees. If None, it will be determined from the data bounds.
            lat_max (Optional[float]): Northern boundary of the map extent in degrees. If None, it will be determined from the data bounds.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset containing the original MPAS data, which may be needed for interpolation and remapping. If None, interpolation will be attempted with available data arrays.

        Returns:
            None: This method modifies the provided axes in place by adding the precipitation overlay. It does not return any value.
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
    
    def _extract_coordinates_from_processor(self: "MPASPrecipitationPlotter",
                                            processor: Any,
                                            var_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method attempts to extract longitude and latitude coordinates from the provided MPAS2DProcessor instance using a series of methods in order of specificity. It first checks if the processor has a method specifically designed for extracting 2D coordinates for the given variable name, which would be ideal for handling cases where coordinates may differ based on the variable being plotted. If that method is not available, it falls back to a more general method for extracting spatial coordinates, which may be suitable for cases where the same coordinates are used across multiple variables. If neither of those methods is available, it finally attempts to access the longitude and latitude directly from the dataset attributes (lonCell and latCell), which is a common convention in MPAS datasets. This approach allows for maximum flexibility in handling different dataset structures and processor implementations while ensuring that valid coordinates are extracted for plotting. If all methods fail, it will raise an AttributeError, which can be caught by calling code to handle missing coordinate information gracefully. 
        
        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            var_name (str): Variable name for which to extract coordinates, used for method selection if specific coordinate extraction methods are available.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the longitude and latitude coordinates as numpy arrays extracted from the processor using the most specific available method. The first element is the longitude array and the second element is the latitude array. If specific methods are not available, it will attempt to access lonCell and latCell directly from the dataset. If all methods fail, an AttributeError will be raised. 
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
    
    def _setup_batch_time_indices(self: "MPASPrecipitationPlotter",
                                  processor: Any,
                                  accum_period: str,
                                  time_indices: Optional[List[int]]) -> Tuple[List[int], int]:
        """
        This method sets up the time indices for batch processing of precipitation plots based on the specified accumulation period and any user-specified time indices. It first determines the total number of available time steps in the dataset and checks if there are enough time steps to support the specified accumulation period (e.g., a 24-hour accumulation would require at least 25 time steps). If there are not enough time steps, it logs a warning and returns an empty list of time indices, which will result in no plots being generated but allows the program to continue without crashing. If there are enough time steps, it validates the user-specified time indices (if provided) to ensure they are valid for the accumulation period, filtering out any indices that do not meet the minimum requirement. If no user-specified indices are provided, it defaults to using all valid time indices starting from the minimum required index for the accumulation period. The method returns a list of validated time indices that can be used for batch processing, along with the number of hours corresponding to the accumulation period for use in title formatting and other plot annotations. This approach ensures that batch processing is robust and can handle cases where the dataset may not have sufficient temporal coverage for certain accumulation periods while still providing useful feedback to the user. 
        
        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to determine the minimum required time steps for validation.
            time_indices (Optional[List[int]]): User-specified list of time indices to process. If None, all valid time indices starting from the minimum required index will be used.
            
        Returns:
            Tuple[List[int], int]: A tuple containing a list of validated time indices that are suitable for batch processing based on the accumulation period and dataset coverage, and an integer representing the number of hours corresponding to the accumulation period for use in title formatting and plot annotations. If there are not enough time steps for the accumulation period, the list of time indices will be empty, and a warning will be logged. 
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
    
    def _process_single_time_step(self: "MPASPrecipitationPlotter",
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
                                  formats: List[str]) -> List[str]:
        """
        This method processes a single time step for precipitation plotting in batch mode. It extracts the precipitation data for the specified time index using the PrecipitationDiagnostics class, which handles the calculation of precipitation differences based on the accumulation period and variable type. It then generates a title for the plot using either a custom template provided by the user or a default format that includes the variable name, valid time, accumulation period, and plot type. The method creates a precipitation map for this time step using the create_precipitation_map() method, passing all necessary parameters including coordinates, data, and visualization options. Finally, it saves the generated plot to files with standardized naming conventions that encode the variable type, accumulation type, valid time, and plot type for easy identification. The method returns a list of file paths for the created plots in the specified formats. This workflow allows for automated generation of precipitation maps for each time step in a batch processing context while ensuring that each plot is properly labeled and saved with consistent naming conventions. 
        
        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            time_idx (int): Time index for which to process the precipitation plot.
            lon (np.ndarray): Longitude coordinates for the plot.
            lat (np.ndarray): Latitude coordinates for the plot.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            var_name (str): Precipitation variable name in the dataset (e.g., 'rainnc', 'rainc').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h').
            plot_type (str): Rendering method for the overlay ('scatter', 'contour', 'contourf').
            grid_resolution (Optional[float]): Grid resolution in degrees for interpolation if needed. If None, it will be determined adaptively based on map extent and data density.
            colormap (Optional[str]): Custom colormap name for precipitation values. If None, a default colormap based on the accumulation period will be used.
            levels (Optional[List[float]]): Custom contour levels for precipitation values. If None, default levels based on the accumulation period will be used.
            custom_title_template (Optional[str]): Custom title template with placeholders for variable name, time string, accumulation period, and plot type. If None, a default title format will be used.
            output_dir (str): Directory path for saving the output plot files.
            file_prefix (str): Prefix string for the output filenames to ensure consistent naming conventions.
            formats (List[str]): List of file format extensions for saving the plot (e.g., ['png', 'pdf']).
            
        Returns:
            List[str]: A list of absolute file paths for all successfully created output files for this time step, with standardized naming conventions that encode the variable type, accumulation type, valid time, and plot type for easy identification. If an error occurs during processing, an empty list will be returned and the error will be logged.
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
    
    def create_batch_precipitation_maps(self: "MPASPrecipitationPlotter",
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
                                        time_indices: Optional[List[int]] = None) -> List[str]:
        """
        This method orchestrates the batch processing of precipitation maps for multiple time steps in a dataset using the provided MPAS2DProcessor instance. It first validates the processor input to ensure that it is not None and that it contains a loaded dataset. It then extracts the longitude and latitude coordinates from the processor using a dedicated method that handles different processor implementations. The method sets up the time indices for batch processing based on the specified accumulation period and any user-specified time indices, ensuring that there are enough time steps in the dataset to support the accumulation period. If there are not enough time steps, it logs a warning and returns an empty list of created files. If valid time indices are available, it iterates through each time index, calling a helper method to process each time step individually. This helper method extracts the precipitation data for the specific time step, generates a title, creates the precipitation map, and saves it to files with standardized naming conventions. The main method collects all created file paths and returns them as a list at the end of the batch processing. This approach allows for efficient generation of multiple precipitation maps while providing robust error handling and informative logging throughout the process. 

        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            output_dir (str): Directory path for saving the output plot files.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            var_name (str): Precipitation variable name in the dataset (e.g., 'rainnc', 'rainc').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h').
            plot_type (str): Rendering method for the overlay ('scatter', 'contour', 'contourf').
            grid_resolution (Optional[float]): Grid resolution in degrees for interpolation if needed. If None, it will be determined adaptively based on map extent and data density.
            file_prefix (str): Prefix string for the output filenames to ensure consistent naming conventions.
            formats (List[str]): List of file format extensions for saving the plot (e.g., ['png', 'pdf']).
            custom_title_template (Optional[str]): Custom title template with placeholders for variable name, time string, accumulation period, and plot type. If None, a default title format will be used.
            colormap (Optional[str]): Custom colormap name for precipitation values. If None, a default colormap based on the accumulation period will be used.
            levels (Optional[List[float]]): Custom contour levels for precipitation values. If None, default levels based on the accumulation period will be used.
            time_indices (Optional[List[int]]): User-specified list of time indices to process. If None, all valid time indices starting from the minimum required index for the accumulation period will be used. 

        Returns:
            List[str]: A list of absolute file paths for all successfully created output files across all processed time steps, with standardized naming conventions that encode the variable type, accumulation type, valid time, and plot type for easy identification. If no valid time indices are available for processing, an empty list will be returned and a warning will be logged. 
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
    
    def _setup_comparison_subplot(self: "MPASPrecipitationPlotter",
                                  ax: GeoAxes,
                                  lon_min: float,
                                  lon_max: float,
                                  lat_min: float,
                                  lat_max: float,
                                  data_crs: ccrs.CRS,
                                  is_global: bool,
                                  panel_index: int) -> None:
        """
        This method configures a GeoAxes subplot for side-by-side precipitation comparison by setting the appropriate map extent, adding geographic features (coastlines, borders, ocean, land), and configuring gridlines with labels. It handles global extents by adjusting the longitude and latitude bounds slightly to avoid issues with the dateline in Cartopy. The method also controls the visibility of gridline labels based on the panel index (left or right) to ensure that labels do not overlap between the two panels. This setup allows for clear visual comparison of precipitation patterns across different datasets or accumulation periods while maintaining consistent geographic context and styling. 
        
        Parameters:
            ax (GeoAxes): The GeoAxes subplot to configure for comparison.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            data_crs (ccrs.CRS): Coordinate reference system of the input data for proper transformation.
            is_global (bool): Flag indicating whether the panel represents a global extent, which requires special handling for dateline issues.
            panel_index (int): Index of the panel (0 for left, 1 for right) used to control label visibility and styling.

        Returns:
            None: This method modifies the provided GeoAxes in place by setting the extent, adding geographic features, and configuring gridlines and labels. It does not return any value.
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
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
    
    def _plot_precipitation_data(self: "MPASPrecipitationPlotter",
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
                                 data_crs: ccrs.CRS) -> Optional[Any]:
        """
        This method plots precipitation data on a GeoAxes using a scatter plot. It first flattens the input data and coordinates to ensure they are 1D arrays for processing. It then creates a mask to filter out invalid data points, which includes checking for finite values, non-negative values, reasonable upper bounds (e.g., less than 100,000 mm), and ensuring that the points fall within the specified map extent. If there are no valid points to plot after masking, the method returns None to indicate that no scatter plot was created. If valid points are available, it extracts the valid longitude, latitude, and precipitation data for plotting. The method calculates an adaptive marker size based on the map extent and the number of valid points to ensure that the scatter plot is visually appropriate for both dense and sparse datasets. It sorts the valid data by precipitation values to ensure that lower values are plotted first, allowing higher values to be more visible on top. Finally, it creates a scatter plot on the provided GeoAxes with the specified colormap and normalization, and returns the scatter object for reference in creating a colorbar. This method allows for flexible and robust plotting of precipitation data while handling various edge cases related to data validity and visualization aesthetics. 
        
        Parameters:
            ax (Axes): The GeoAxes on which to plot the precipitation data.
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS mesh cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS mesh cell centers.
            data (np.ndarray): 1D array of precipitation values corresponding to the longitude and latitude coordinates, in mm or display units.
            lon_min (float): Western boundary of the map extent in degrees for filtering valid points.
            lon_max (float): Eastern boundary of the map extent in degrees for filtering valid points.
            lat_min (float): Southern boundary of the map extent in degrees for filtering valid points.
            lat_max (float): Northern boundary of the map extent in degrees for filtering valid points.
            cmap (mcolors.Colormap): Colormap instance to use for coloring the scatter points based on precipitation values.
            norm (BoundaryNorm): Normalization instance to use for mapping precipitation values to colors in the colormap.
            data_crs (ccrs.CRS): Coordinate reference system of the input data for proper transformation when plotting.
            
        Returns:
            Optional[Any]: The scatter object created by ax.scatter for the valid precipitation data points, which can be used for creating a colorbar. If no valid points are available to plot, the method returns None to indicate that no scatter plot was created. 
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
    
    def create_precipitation_comparison_plot(self: "MPASPrecipitationPlotter",
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
        This method creates a side-by-side comparison plot of two precipitation datasets on a shared map projection. It sets up a figure with two GeoAxes subplots, each configured with the same geographic extent and map features for consistent visual comparison. The method creates a shared colormap and normalization based on the specified accumulation period to ensure that both panels use the same color scale for accurate comparison. It handles global extents by adjusting the longitude and latitude bounds to avoid dateline issues in Cartopy. Each subplot is populated with the corresponding precipitation data using a scatter plot, and titles are set for each panel to clearly indicate which dataset is being displayed. A single colorbar is added below the two panels, centered, with consistent styling and tick formatting based on the color levels used in the plots. Finally, timestamp and branding are added to the figure, and the layout is adjusted to prevent overlap and ensure clear presentation. The method returns the figure and axes for further manipulation or saving as needed. This approach allows for effective visual comparison of two precipitation datasets while maintaining consistent geographic context and styling across both panels. 

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS mesh cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS mesh cell centers.
            precip_data1 (np.ndarray): 2D array of precipitation values for the first dataset, corresponding to the longitude and latitude coordinates, in mm or display units.
            precip_data2 (np.ndarray): 2D array of precipitation values for the second dataset, corresponding to the longitude and latitude coordinates, in mm or display units.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            title1 (str): Title for the first subplot (default: "Precipitation 1").
            title2 (str): Title for the second subplot (default: "Precipitation 2").
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to determine colormap and normalization (default: "a01h").
            projection (str): Map projection type to use for the GeoAxes (default: 'PlateCarree').

        Returns:
            Tuple[Figure, List[Axes]]: A tuple containing the created matplotlib Figure object and a list of the two GeoAxes subplots. The figure contains the side-by-side comparison of the two precipitation datasets with shared map features and a common colorbar. The axes can be further manipulated or saved as needed after the method returns.
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

        # Position colorbar below the two panels, centered, with consistent styling
        cbar = MPASVisualizationStyle.add_colorbar(
            self.fig, self.ax, scatter,
            label='Precipitation [mm]', orientation='horizontal',
            fraction=0.03, pad=0.08, shrink=0.6, fmt=None, labelpad=0,
            label_pos='top', tick_labelsize=8
        )

        if cbar is not None and len(color_levels_sorted) <= 15:
            try:
                cbar.set_ticks(color_levels_sorted)
                cbar.set_ticklabels(self._format_ticks_dynamic(color_levels_sorted))
            except Exception:
                pass
        
        # Add timestamp and branding to the figure
        self.add_timestamp_and_branding()

        # Adjust layout to prevent overlap and ensure clear presentation
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.15)
        
        # Return the figure and axes for further manipulation or saving
        return self.fig, axes
    
    def save_plot(self: "MPASPrecipitationPlotter",
                  output_path: str,
                  formats: List[str] = ['png'],
                  bbox_inches: str = 'tight',
                  pad_inches: float = 0.1) -> None:
        """
        This method saves the current matplotlib figure to disk in multiple specified formats with consistent styling and optimized settings for precipitation maps. It first checks if there is an active figure to save, raising an error if not. It then determines the output directory from the provided output path and ensures that it exists, creating it if necessary. For each specified format, it constructs the full file path with the appropriate extension and defines savefig options including DPI, bounding box settings, and format. For PNG files, it applies low compression settings to optimize saving speed while maintaining quality, especially for large precipitation maps with many points. The method saves the figure with the specified options and prints a confirmation message for each saved file. This approach allows for flexible saving of precipitation plots in various formats while ensuring that the output files are properly organized and named according to the provided output path. 

        Parameters:
            output_path (str): Base file path (without extension) for saving the plot files. The method will append the appropriate extension for each format specified in the formats list.
            formats (List[str]): List of file format extensions to save the plot in (e.g., ['png', 'pdf']). The method will save a separate file for each format in this list.
            bbox_inches (str): Bounding box setting for saving the figure, typically 'tight' to minimize whitespace around the plot.
            pad_inches (float): Padding in inches to add around the figure when bbox_inches is set to 'tight' to prevent clipping of labels and titles.

        Returns:
            None: Saves the current figure to disk in the specified formats with optimized settings for precipitation maps. Prints confirmation messages for each saved file. Raises an error if there is no active figure to save.
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
    
    def close_plot(self: "MPASPrecipitationPlotter") -> None:
        """
        This method closes the current matplotlib figure and clears the instance attributes self.fig and self.ax to free up memory and resources after saving or when the plot is no longer needed. It checks if there is an active figure (self.fig) and if so, it calls plt.close() on that figure to close it. After closing the figure, it sets self.fig and self.ax to None to clear references to the closed figure and axes, allowing for garbage collection. This method is important for managing memory usage when creating multiple plots in a batch process or when generating large figures that may consume significant resources. By closing the figure and clearing references, it helps prevent memory leaks and ensures that the system remains responsive during extended plotting sessions. 

        Parameters:
            None: This method does not take any parameters. It operates on the instance attributes self.fig and self.ax to manage the current figure and axes.

        Returns:
            None: This method does not return any value. It performs side effects by closing the current figure and clearing the instance attributes to free up memory and resources.
        """
        # Close the figure if it exists and clear references
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _format_ticks_dynamic(self: "MPASPrecipitationPlotter",
                              ticks: List[float]) -> List[str]:
        """
        This method formats tick labels dynamically based on the range and magnitude of the tick values for precipitation plots. It delegates to the MPASVisualizationStyle class to apply consistent formatting rules that adjust the number of decimal places, use scientific notation for very large or small values, and ensure that the tick labels are concise and readable. This dynamic formatting helps improve the clarity of the axis labels on precipitation maps, especially when dealing with a wide range of precipitation values that may include very small amounts (e.g., drizzle) or very large amounts (e.g., heavy rainfall). By centralizing the formatting logic in the MPASVisualizationStyle class, it allows for consistent styling across all plots and makes it easier to maintain and update the formatting rules in one place.  

        Parameters:
            ticks (List[float]): A list of tick values that need to be formatted for display on the axes. These values can vary widely in magnitude, and the formatting will adjust accordingly to ensure readability.
            
        Returns:
            List[str]: A list of formatted tick labels corresponding to the input tick values, formatted according to the dynamic rules defined in the MPASVisualizationStyle class. The formatting may include adjustments to decimal places, use of scientific notation, and other styling choices to enhance readability on precipitation plots. 
        """
        # Delegate to MPASVisualizationStyle for dynamic tick formatting based on value range and magnitude
        return MPASVisualizationStyle.format_ticks_dynamic(ticks)
    
    def apply_style(self: "MPASPrecipitationPlotter",
                    style_name: str = 'default') -> None:
        """
        This method applies a visualization style to the current plot by setting the axes background color to light gray for better contrast with precipitation colors and the figure background to white for improved contrast and printing. It also delegates to the style manager (if it exists) to apply broader matplotlib settings based on the specified style name. This allows for consistent styling across all precipitation plots while ensuring that the specific color choices enhance the visibility of precipitation data on the map. By centralizing style application in this method, it makes it easier to maintain and update the visual appearance of all plots in a consistent manner, while still allowing for customization through different registered styles in the style manager.  

        Parameters:
            style_name (str): The name of the style to apply, which corresponds to a registered style in the style manager. This allows for flexible styling options that can be easily switched by changing the style name. The method will apply the specified style settings to the plot if the style manager is available. 

        Returns:
            None
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