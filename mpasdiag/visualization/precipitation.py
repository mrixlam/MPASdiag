#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

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
import re
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
from matplotlib.lines import Line2D
from typing import Tuple, Optional, List, Any, Union, Dict, Sequence, cast
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Import relevant MPASdiag modules for visualization and processing
from .base_visualizer import MPASVisualizer, close_figure_on_error
from .styling import MPASVisualizationStyle
from dataclasses import dataclass
from ..processing.utils_unit import UnitConverter
from ..processing.utils_datetime import MPASDateTimeUtils
from ..processing.utils_geog import GeographicBounds, MPASGeographicUtils
from ..processing.utils_logger import get_logger
from ..diagnostics.precipitation import (
    InsufficientLookbackError,
    PrecipitationDiagnostics,
)

# Import remapping utilities and check for ESMPy availability
from ..processing.remapping import (
    remap_mpas_to_latlon_with_masking,
    dispatch_remap,
    ESMPY_AVAILABLE,
)

logger = get_logger(__name__)


@dataclass
class PrecipitationMapStyle:
    """Optional appearance and file-naming settings for batch precipitation maps, grouping the output filename prefix, custom title template, colormap, and contour levels into a single value object."""

    file_prefix: str = "mpas_precipitation_map"
    custom_title_template: Optional[str] = None
    colormap: Optional[str] = None
    levels: Optional[List[float]] = None
    display_units: str = "mm"
    threshold: Optional[float] = None


@dataclass
class PrecipitationRenderStyle:
    """Optional rendering and appearance settings for a single precipitation map, grouping the title, plot type, colormap, contour levels, color limits, projection, projection centering/keyword controls, grid resolution, display units ('mm', 'in' or 'cm'; the data, colorbar label, levels and color limits are all expressed in these units) and an optional threshold (in display units) above which cells are visibly outlined into a single value object."""

    title: str = "MPAS Precipitation"
    plot_type: str = "scatter"
    colormap: Optional[str] = None
    levels: Optional[List[float]] = None
    clim_min: Optional[float] = None
    clim_max: Optional[float] = None
    projection: str = "PlateCarree"
    central_longitude: Optional[float] = None
    central_latitude: Optional[float] = None
    proj_kwargs: Optional[Dict[str, Any]] = None
    grid_resolution: Optional[float] = None
    display_units: str = "mm"
    threshold: Optional[float] = None


@dataclass
class OverlayColorSpec:
    """Colormap, normalization, sorted contour levels, and transparency used to render an interpolated precipitation overlay."""

    cmap: mcolors.Colormap
    norm: BoundaryNorm
    color_levels_sorted: List[float]
    alpha: float


# Filter warnings from cartopy and shapely to avoid cluttering output with non-critical warnings during map rendering
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")


class MPASPrecipitationPlotter(MPASVisualizer):
    """Specialized plotter for creating professional cartographic visualizations of MPAS precipitation fields with support for multiple accumulation periods, automatic unit conversion, and flexible rendering options."""

    _GEOAXES_MSG = "Axes must be GeoAxes for cartopy plots"

    def __init__(
        self: "MPASPrecipitationPlotter",
        figsize: Tuple[float, float] = (10, 14),
        dpi: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        This initializer sets up the MPASPrecipitationPlotter with default figure size and resolution. It calls the parent class initializer to ensure any necessary setup from MPASVisualizer is performed. The figsize parameter controls the dimensions of the output figure in inches, while dpi sets the resolution for rendering and saving the figure, and verbose is forwarded to the base class so every plotter constructor accepts the same keyword. These defaults can be overridden when instantiating the plotter to suit specific presentation or publication requirements. The initializer does not perform any plotting itself but prepares the instance for subsequent plotting operations.

        Parameters:
            figsize (Tuple[float, float]): Default figure size in inches (width, height).
            dpi (int): Default resolution in dots per inch for rendering and saving figures.
            verbose (bool): Enable verbose informational output, forwarded to MPASVisualizer (default: True).

        Returns:
            None
        """
        super().__init__(figsize, dpi, verbose)

    def create_precip_colormap(
        self: "MPASPrecipitationPlotter", accum: str = "a24h"
    ) -> Tuple[mcolors.ListedColormap, List[float]]:
        """
        This method creates a discrete colormap and corresponding contour levels for precipitation visualization based on the specified accumulation period. It delegates to the MPASVisualizationStyle class to generate a colormap that is appropriate for the given accumulation period (e.g., hourly, daily) and returns both the colormap and a sorted list of contour level values that can be used for consistent coloring and colorbar ticks in precipitation maps. The accumulation period identifier (e.g., 'a24h' for 24-hour accumulation) is used to select predefined color schemes and levels that are meteorologically meaningful for visualizing precipitation fields.

        Parameters:
            accum (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to select appropriate colormap and contour levels for precipitation visualization.

        Returns:
            Tuple[mcolors.ListedColormap, List[float]]: A tuple containing the created discrete colormap and a sorted list of contour level values suitable for precipitation plotting.
        """
        # Delegate to MPASVisualizationStyle to create a discrete colormap and contour levels based on the specified accumulation period
        return MPASVisualizationStyle.create_precip_colormap(accum)

    _MM_PER_UNIT: Dict[str, float] = {"mm": 1.0, "cm": 10.0, "in": 25.4}

    @classmethod
    def _normalize_display_units(cls, display_units: Optional[str]) -> str:
        """
        This helper validates and normalizes the requested precipitation display units. The accepted values are 'mm', 'cm' and 'in' (the aliases 'inch', 'inches' and 'millimeters' style spellings are folded onto them, case-insensitively), and anything else raises a ValueError listing the valid choices instead of silently plotting in millimetres.

        Parameters:
            display_units (Optional[str]): Requested display units, or None for the millimetre default.

        Returns:
            str: One of 'mm', 'cm' or 'in'.
        """
        if display_units is None:
            return "mm"

        key = str(display_units).strip().lower()
        aliases = {
            "inch": "in",
            "inches": "in",
            '"': "in",
            "millimeter": "mm",
            "millimeters": "mm",
            "millimetre": "mm",
            "millimetres": "mm",
            "centimeter": "cm",
            "centimeters": "cm",
            "centimetre": "cm",
            "centimetres": "cm",
        }
        key = aliases.get(key, key)

        if key not in cls._MM_PER_UNIT:
            raise ValueError(
                f"display_units must be one of 'mm', 'in' or 'cm', got '{display_units}'"
            )

        return key

    @staticmethod
    def _mask_negative_precipitation(precip_data: np.ndarray) -> np.ndarray:
        """
        This helper handles physically invalid negative precipitation values by masking them to NaN (so they are simply not drawn) rather than clipping them to zero, which would paint them as dry cells; a warning with the count and minimum value is logged when any are found.

        Parameters:
            precip_data (np.ndarray): Precipitation values that may contain negatives.

        Returns:
            np.ndarray: Float array with negative values replaced by NaN (the input is returned unchanged when none are negative).
        """
        data = np.asarray(precip_data)

        with np.errstate(invalid="ignore"):
            negative = data < 0

        n_negative = int(np.sum(negative))

        if n_negative == 0:
            return data

        logger.warning(
            "Found %s negative precipitation values (min: %.4f). Masking them as missing (physically invalid).",
            f"{n_negative:,}",
            float(np.nanmin(data)),
        )

        masked = data.astype(float, copy=True)
        masked[negative] = np.nan
        return masked

    def _convert_precipitation_units(
        self: "MPASPrecipitationPlotter",
        precip_data: Union[np.ndarray, xr.DataArray],
        data_array: Optional[xr.DataArray],
        var_name: str,
        display_units: str = "mm",
    ) -> Tuple[np.ndarray, str]:
        """
        This method handles unit conversion for precipitation data. When an xarray DataArray with metadata is provided it uses the UnitConverter to convert the input to millimetres based on the variable name and attributes; if conversion fails due to missing attributes it logs nothing fatal and keeps the original data with the metadata units. The millimetre values are then converted to the requested display units ('mm', 'cm' or 'in'), so the returned data, and therefore the colorbar label, levels and color limits, are all in those units. Negative precipitation values are physically invalid and are masked to NaN with a warning (with and without metadata), so they are not drawn as dry cells. The returned unit label can be used directly for colorbar annotation.

        Parameters:
            precip_data (np.ndarray): Input precipitation data array, which may be in model units or already in display units.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing metadata attributes for unit conversion. If None, the data are assumed to already be in millimetres.
            var_name (str): Variable name used for metadata lookup and unit conversion logic.
            display_units (str): Requested display units, one of 'mm', 'cm' or 'in' (default: 'mm').

        Returns:
            Tuple[np.ndarray, str]: The converted precipitation data and the unit label (the display units) for colorbar annotation.
        """
        target = self._normalize_display_units(display_units)
        unit_label = "mm"

        if data_array is not None:
            try:
                # Prepare data for conversion, ensuring it is an xarray DataArray with appropriate attributes for UnitConverter
                if isinstance(precip_data, np.ndarray):
                    data_for_conversion = xr.DataArray(
                        precip_data, attrs=data_array.attrs
                    )
                else:
                    data_for_conversion = precip_data

                converted_data, metadata = UnitConverter.convert_data_for_display(
                    data_for_conversion, var_name, data_array
                )
            except AttributeError:
                # If conversion fails due to missing attributes, keep the original data and the metadata units
                converted_data = precip_data
                metadata = {
                    "units": getattr(data_array, "units", "mm"),
                    "long_name": getattr(data_array, "long_name", "Precipitation"),
                }

            # Extract converted data values as numpy array for plotting
            if isinstance(converted_data, xr.DataArray):
                precip_data = converted_data.values
            elif isinstance(converted_data, np.ndarray):
                precip_data = converted_data
            else:
                precip_data = np.asarray(converted_data)

            unit_label = metadata.get("units", "mm")
        else:
            precip_data = np.asarray(
                precip_data.values
                if isinstance(precip_data, xr.DataArray)
                else precip_data
            )

        # Convert millimetres to the requested display units
        if target != "mm":
            if str(unit_label).strip().lower() in ("mm", "millimeter", "millimeters"):
                precip_data = (
                    np.asarray(precip_data, dtype=float) / self._MM_PER_UNIT[target]
                )
                unit_label = target
            else:
                logger.warning(
                    "Cannot convert precipitation from '%s' to '%s'; keeping '%s'",
                    unit_label,
                    target,
                    unit_label,
                )
        else:
            unit_label = unit_label if data_array is not None else "mm"

        # Mask physically invalid negative values instead of painting them as dry cells
        precip_data = self._mask_negative_precipitation(precip_data)

        return precip_data, unit_label

    def _setup_precipitation_figure(
        self: "MPASPrecipitationPlotter",
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        projection: str,
        central_longitude: Optional[float] = None,
        central_latitude: Optional[float] = None,
        proj_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ccrs.Projection, ccrs.CRS]:
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
        map_proj, data_crs = self.setup_map_projection(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            projection,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            proj_kwargs=proj_kwargs,
        )

        # Create figure and GeoAxes with the specified projection
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.axes(projection=map_proj)

        # Assert ax is created and of correct type for cartopy plotting before adding features
        assert isinstance(self.ax, GeoAxes), self._GEOAXES_MSG

        # Determine if the map is global based on extent to handle dateline issues appropriately
        is_global = (lon_max - lon_min) >= 359.0 and (lat_max - lat_min) >= 179.0

        # For global maps, show the projection's full natural domain via set_global().
        if is_global:
            self.ax.set_global()
            logger.debug("Using global extent via set_global() for projection view")
        else:
            # Set extent for regional maps, ensuring it is within valid ranges and properly ordered
            self._apply_map_extent(
                self.ax, [lon_min, lon_max, lat_min, lat_max], data_crs
            )

        # Add geographic features with styling for precipitation maps
        self.ax.add_feature(
            cfeature.COASTLINE, linewidth=0.8, edgecolor="black", alpha=0.7
        )
        self.ax.add_feature(
            cfeature.BORDERS, linewidth=0.6, edgecolor="gray", alpha=0.7
        )
        self._add_ocean_feature(self.ax, facecolor="lightblue", alpha=0.3)
        self.ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)

        # Add regional features if within the specified extent
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)

        # Return the map projection and data CRS
        return map_proj, data_crs

    _DEFAULT_LEVELS_BY_UNIT: Dict[str, Tuple[List[float], List[float]]] = {
        "mm": (
            [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75],
            [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150],
        ),
        "in": (
            [0.01, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3],
            [0.01, 0.05, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6],
        ),
        "cm": (
            [0.01, 0.05, 0.25, 0.5, 1, 1.5, 2, 2.5, 5, 7.5],
            [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15],
        ),
    }

    @classmethod
    def _default_levels_for_units(
        cls, accum_period: str, display_units: str
    ) -> List[float]:
        """
        This helper returns the default contour levels for an accumulation period expressed in the requested display units. Millimetre levels are the standard scheme from the styling module (short periods below 12 h use the finer set); inch and centimetre levels are hand-picked round numbers with the same number of intervals so the eleven-colour scheme maps onto them one to one, rather than ugly converted values such as 0.0039 in.

        Parameters:
            accum_period (str): Accumulation period identifier such as 'a01h' or 'a24h'.
            display_units (str): One of 'mm', 'cm' or 'in'.

        Returns:
            List[float]: Ten ascending contour levels in the display units.
        """
        match = re.search(r"(\d+)", str(accum_period).lower())
        hours = int(match.group(1)) if match else None
        short, long = cls._DEFAULT_LEVELS_BY_UNIT[display_units]
        return list(long if hours is None or hours >= 12 else short)

    @staticmethod
    def _precip_bounds(levels: List[float], include_zero_floor: bool) -> List[float]:
        """
        This helper builds the strictly increasing BoundaryNorm bounds for a set of sorted, unique, finite contour levels: exactly the levels, so the colorbar spans the first to the last level and nothing beyond (values above the last level take the norm's over colour). When include_zero_floor is True and every level is positive, a lower bound of zero is prepended so values below the first level (dry cells) get the first colour of the scheme; a level at or below zero is never duplicated or followed by a hard-coded zero, so a color limit of 0 or a negative value yields valid, monotonic bounds.

        Parameters:
            levels (List[float]): Sorted unique finite levels (non-empty).
            include_zero_floor (bool): Whether to prepend a zero floor when the lowest level is positive.

        Returns:
            List[float]: Strictly increasing bounds.
        """
        bounds = list(levels)

        if include_zero_floor and bounds[0] > 0:
            bounds.insert(0, 0.0)

        return bounds

    @staticmethod
    def _precip_norm(bounds: List[float], cmap: mcolors.Colormap) -> BoundaryNorm:
        """
        This helper builds the discrete precipitation normalization for the given bounds. The norm has one colour region per interval plus an open-ended over region for values above the last bound (extend='max'), so the colours match the classic scheme where the top colour covers everything above the highest level, while the colorbar itself spans exactly the bounds. Values below the first bound take the colormap's under colour, which is its first colour. A single bound (one level) is widened by one unit so the norm stays valid.

        Parameters:
            bounds (List[float]): Strictly increasing bounds.
            cmap (mcolors.Colormap): Colormap whose colours are distributed over the regions.

        Returns:
            BoundaryNorm: The discrete normalization (extend='max').
        """
        if len(bounds) < 2:
            bounds = [bounds[0], bounds[0] + 1.0]

        return BoundaryNorm(bounds, ncolors=cmap.N, extend="max")

    def _select_precip_scheme(
        self: "MPASPrecipitationPlotter",
        colormap: Optional[str],
        levels: Optional[List[float]],
        accum_period: str,
        display_units: str,
    ) -> Tuple[mcolors.Colormap, List[float]]:
        """
        This helper chooses the colormap and the raw contour levels before color limits are applied: explicit levels always win, a named colormap replaces the default colours, and when no levels are given the default scheme for the accumulation period is used, taken from the unit-specific table when the display units are not millimetres.

        Parameters:
            colormap (Optional[str]): Optional matplotlib colormap name.
            levels (Optional[List[float]]): Optional explicit levels in display units.
            accum_period (str): Accumulation period identifier selecting the default scheme.
            display_units (str): Display units ('mm', 'cm' or 'in') selecting the default level table.

        Returns:
            Tuple[mcolors.Colormap, List[float]]: The colormap and the raw (unsorted) levels.
        """
        default_cmap, default_levels = self.create_precip_colormap(accum_period)
        cmap = plt.get_cmap(colormap) if colormap else default_cmap

        if levels:
            return cmap, list(levels)

        if display_units != "mm":
            return cmap, self._default_levels_for_units(accum_period, display_units)

        return cmap, list(default_levels)

    @staticmethod
    def _apply_color_limits(
        color_levels: List[float], clim_min: Optional[float], clim_max: Optional[float]
    ) -> List[float]:
        """
        This helper restricts contour levels to the color limits (either limit may be given alone) and inserts each given limit as a level so the colorbar starts and ends exactly at the limits.

        Parameters:
            color_levels (List[float]): Raw contour levels.
            clim_min (Optional[float]): Optional lower limit.
            clim_max (Optional[float]): Optional upper limit.

        Returns:
            List[float]: Levels inside the limits including the limits themselves.
        """
        kept = [
            level
            for level in color_levels
            if (clim_min is None or level >= clim_min)
            and (clim_max is None or level <= clim_max)
        ]

        if clim_min is not None and clim_min not in kept:
            kept.insert(0, clim_min)

        if clim_max is not None and clim_max not in kept:
            kept.append(clim_max)

        return kept

    def _prepare_precipitation_colormap(
        self: "MPASPrecipitationPlotter",
        colormap: Optional[str],
        levels: Optional[List[float]],
        accum_period: str,
        clim_min: Optional[float],
        clim_max: Optional[float],
        display_units: str = "mm",
    ) -> Tuple[mcolors.Colormap, BoundaryNorm, List[float]]:
        """
        This method prepares the colormap, normalization, and contour levels for precipitation plotting based on user input and accumulation period. It allows for custom colormap and levels to be specified, or defaults to predefined schemes based on the accumulation period and display units. It applies color limits if provided (one-sided limits are supported), ensuring that the levels used for coloring and colorbar ticks are consistent with the limits and that each remaining interval keeps the colour the default scheme assigns to it rather than the colours being stretched across fewer levels. The normalization bounds are built from the sorted unique levels without a hard-coded leading zero, so a color limit of 0 (or a negative one) produces valid, strictly increasing bounds; a minimum limit greater than or equal to the maximum raises a ValueError. It returns the selected colormap, a BoundaryNorm for discrete coloring, and a sorted list of contour levels.

        Parameters:
            colormap (Optional[str]): Optional matplotlib colormap name to use for precipitation plotting. If None, a default colormap based on the accumulation period will be used.
            levels (Optional[List[float]]): Optional list of contour levels (in display units) to use for coloring and colorbar ticks. If None, default levels based on the accumulation period and display units will be used.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used to select default colormap and levels if custom ones are not provided.
            clim_min (Optional[float]): Optional minimum color limit (display units) to clip contour levels for coloring and colorbar ticks; may be given without clim_max.
            clim_max (Optional[float]): Optional maximum color limit (display units) to clip contour levels for coloring and colorbar ticks; may be given without clim_min.
            display_units (str): Display units of the data, one of 'mm', 'cm' or 'in', selecting the default levels (default: 'mm').

        Returns:
            Tuple[mcolors.Colormap, BoundaryNorm, List[float]]: A tuple containing the selected colormap, a BoundaryNorm for discrete coloring based on the levels and color limits, and a sorted list of contour levels that are consistent with the specified limits and accumulation period.
        """
        if clim_min is not None and clim_max is not None and clim_min >= clim_max:
            raise ValueError(
                f"clim_min ({clim_min}) must be smaller than clim_max ({clim_max})"
            )

        cmap, color_levels = self._select_precip_scheme(
            colormap, levels, accum_period, display_units
        )

        # Remember the unclipped scheme so colours keep their level mapping when limits are applied
        original_levels = sorted({v for v in color_levels if np.isfinite(v)})
        original_cmap = cmap
        has_limits = clim_min is not None or clim_max is not None

        # Apply color limits if specified (either one alone is honoured)
        if has_limits:
            color_levels = self._apply_color_limits(color_levels, clim_min, clim_max)

        # Sort and deduplicate color levels, ensuring they are finite values for proper normalization
        color_levels_sorted = sorted({v for v in color_levels if np.isfinite(v)})

        if not color_levels_sorted:
            raise ValueError(
                "No finite contour levels available for precipitation plot"
            )

        # A one-sided limit beyond the whole level scheme leaves a single level, which cannot form a colour scale
        if has_limits and len(color_levels_sorted) < 2:
            raise ValueError(
                f"The color limits (clim_min={clim_min}, clim_max={clim_max}) leave fewer than two "
                f"contour levels ({color_levels_sorted}); choose limits inside the level range"
            )

        # A lower color limit is the lowest bound itself; otherwise a zero floor catches dry cells
        floor = clim_min is None
        bounds = self._precip_bounds(color_levels_sorted, floor)
        norm = self._precip_norm(bounds, cmap)

        # When limits changed the intervals, give each new interval the colour the unclipped scheme assigns to that value instead of stretching the colours over fewer levels
        if has_limits and original_levels:
            original_bounds = self._precip_bounds(original_levels, True)
            original_norm = self._precip_norm(original_bounds, original_cmap)
            interval_colors = []
            edges = list(bounds) + [float(np.nextafter(bounds[-1], np.inf))]
            for lower, upper in zip(edges[:-2], edges[1:-1]):
                midpoint = 0.5 * (lower + upper)
                interval_colors.append(original_cmap(int(original_norm(midpoint))))
            # The open-ended top region keeps the colour of the values just above the upper limit
            interval_colors.append(
                original_cmap(int(original_norm(float(edges[-1]) + 1e-9)))
            )
            cmap = mcolors.ListedColormap(interval_colors)
            norm = self._precip_norm(bounds, cmap)

        # Return the colormap, normalization, and sorted contour levels
        return cmap, norm, color_levels_sorted

    @staticmethod
    def _validate_precipitation_inputs(
        lon: Any, lat: Any, precip_data: Any, var_name: str
    ) -> None:
        """
        This helper rejects precipitation inputs that cannot produce a meaningful map, mirroring the checks of the surface plotter. Longitude, latitude, and data must have the same number of points, must not be empty, and the data must contain at least one finite value. Zero precipitation is valid data, and points falling outside the map extent are handled later (an empty region still yields a map), so only structurally broken or entirely missing input is rejected here. Previously such input was silently truncated to the shortest array or drawn as a blank map.

        Parameters:
            lon (Any): Longitude values (array-like).
            lat (Any): Latitude values (array-like).
            precip_data (Any): Precipitation values (array-like or DataArray).
            var_name (str): Variable name used in error messages.

        Returns:
            None: Raises ValueError when the inputs are unusable.
        """
        values = np.asarray(
            getattr(precip_data, "values", precip_data), dtype=float
        ).ravel()
        n_lon = np.asarray(lon).size
        n_lat = np.asarray(lat).size

        if values.size == 0 or n_lon == 0 or n_lat == 0:
            raise ValueError(
                f"No precipitation data to plot for {var_name}: empty input arrays"
            )

        if not (n_lon == n_lat == values.size):
            raise ValueError(
                f"Mismatched input sizes for {var_name}: lon={n_lon}, lat={n_lat}, data={values.size}"
            )

        if not np.isfinite(values).any():
            raise ValueError(
                f"No valid data points found: all {var_name} values are missing (NaN/inf)"
            )

    def _prepare_precipitation_data(
        self: "MPASPrecipitationPlotter",
        lon: np.ndarray,
        lat: np.ndarray,
        precip_data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        plot_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if plot_type == "scatter":
            # Re-express longitudes in the extent's frame so dateline-crossing boxes (e.g. 150..210) keep the cells east of 180°
            lon_flat = MPASGeographicUtils.wrap_longitude_to_extent(
                lon_flat, lon_min, lon_max
            )
            # For scatter, we need to ensure points are within geographic extent and have valid precipitation values
            valid_mask = (
                np.isfinite(precip_data_flat)
                & (precip_data_flat >= 0)
                & (precip_data_flat < 1e5)
                & (lon_flat >= lon_min)
                & (lon_flat <= lon_max)
                & (lat_flat >= lat_min)
                & (lat_flat <= lat_max)
            )
        else:
            # For contour and contourf, we will handle geographic masking during interpolation, so only validate data values here
            valid_mask = (
                np.isfinite(precip_data_flat)
                & (precip_data_flat >= 0)
                & (precip_data_flat < 1e5)
            )

        # Return only valid points for plotting
        return lon_flat[valid_mask], lat_flat[valid_mask], precip_data_flat[valid_mask]

    @staticmethod
    def _accum_period_hours(accum_period: str) -> int:
        """
        This helper converts an accumulation period identifier into its length in hours. Identifiers have the form 'aNNh' (for example 'a01h', 'a03h', 'a06h', 'a12h', 'a24h', 'a48h'), so any whole number of hours from 1 to 240 is accepted rather than only a fixed table. An identifier that does not match this form raises a ValueError listing the standard valid periods, instead of silently falling back to 24 hours and producing mislabelled accumulations.

        Parameters:
            accum_period (str): Accumulation period identifier such as 'a01h' or 'a24h'.

        Returns:
            int: Number of hours in the accumulation period.
        """
        match = re.fullmatch(r"a(\d{1,3})h", str(accum_period).strip().lower())

        if match is None or not 1 <= int(match.group(1)) <= 240:
            raise ValueError(
                f"Unknown accumulation period '{accum_period}'. Use the form 'aNNh' "
                "with 1 to 240 hours, e.g. 'a01h', 'a03h', 'a06h', 'a12h', 'a24h'"
            )

        return int(match.group(1))

    @staticmethod
    def _reconcile_accum_period(
        accum_period: str, data_array: Optional[xr.DataArray]
    ) -> str:
        """
        This helper reconciles the requested accumulation period with the one recorded in the data's metadata (the 'accumulation_period' or 'accumulation_hours' attribute written by the diagnostics). When the two differ, the colour scheme and levels would be chosen for one period while the annotation and the values describe another, so a warning is logged and the metadata period, which is what was actually plotted, is used for both the levels and the annotation. Missing or unparsable metadata leaves the requested period unchanged.

        Parameters:
            accum_period (str): Requested accumulation period identifier such as 'a01h' or 'a24h'.
            data_array (Optional[xr.DataArray]): DataArray whose attrs may carry the accumulation metadata.

        Returns:
            str: The accumulation period identifier to use for levels and annotation.
        """
        if data_array is None:
            return accum_period

        attrs = getattr(data_array, "attrs", {})
        hours: Optional[int] = None

        try:
            if attrs.get("accumulation_period") is not None:
                hours = MPASPrecipitationPlotter._accum_period_hours(
                    str(attrs["accumulation_period"])
                )
            elif attrs.get("accumulation_hours") is not None:
                hours = int(attrs["accumulation_hours"])
        except (ValueError, TypeError):
            return accum_period

        if hours is None or not 1 <= hours <= 240:
            return accum_period

        if hours != MPASPrecipitationPlotter._accum_period_hours(accum_period):
            reconciled = f"a{hours:02d}h"
            logger.warning(
                "accum_period '%s' does not match the %d-hour accumulation recorded in the data; "
                "using '%s' for the levels and the annotation",
                accum_period,
                hours,
                reconciled,
            )
            return reconciled

        return accum_period

    def _add_time_annotation(
        self: "MPASPrecipitationPlotter",
        time_end: Optional[datetime],
        time_start: Optional[datetime],
        accum_period: str,
    ) -> None:
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
        assert isinstance(self.ax, GeoAxes), self._GEOAXES_MSG

        # Determine annotation text based on provided time information and accumulation period
        if time_end is not None:
            # Derive time_start from time_end and accum_period if not provided
            if time_start is None:
                n_hours = self._accum_period_hours(accum_period)
                time_start = time_end - pd.Timedelta(hours=n_hours)

            # Format annotation text with UTC times and accumulation period
            start_utc = time_start.strftime("%Y-%m-%d %H:%M UTC")
            end_utc = time_end.strftime("%Y-%m-%d %H:%M UTC")
            n_hours = int((time_end - time_start).total_seconds() / 3600)
            annotation_text = f"Accumulation: {start_utc} to {end_utc} ({n_hours} h)"
        elif accum_period:
            # Use accumulation period string if time_end is not provided but accum_period is specified
            try:
                accum_display = f"{self._accum_period_hours(accum_period)}-h"
            except ValueError:
                accum_display = accum_period
            annotation_text = f"Accumulation: {accum_display}"
        else:
            return

        # Add annotation text box to the plot with styling
        self.ax.text(
            0.01,
            0.02,
            annotation_text,
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    @staticmethod
    def _read_time_metadata(
        data_array: Optional[xr.DataArray],
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        This static method reads time metadata from an xarray DataArray's attributes, specifically looking for 'valid_time' and 'accumulation_start_time'. It attempts to parse these attributes into Python datetime objects using pandas Timestamp for robust handling of various datetime formats. If the attributes are missing or cannot be parsed, it returns None for those values. This method provides a best-effort approach to extract time information that may have been attached by the diagnostics, allowing for more informative annotations on the precipitation plots when such metadata is available.

        Parameters:
            data_array (Optional[xr.DataArray]): DataArray whose attrs may carry 'valid_time' and 'accumulation_start_time'.

        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: The (meta_end, meta_start) datetimes parsed from metadata.
        """
        if data_array is None:
            return None, None

        def _parse(key: str) -> Optional[datetime]:
            """
            This helper function attempts to parse a datetime from the specified attribute key in the DataArray's attributes. It returns a Python datetime object if parsing is successful, or None if the attribute is missing or cannot be parsed. This allows for robust handling of time metadata that may be present in various formats.

            Parameters:
                key (str): The attribute key to look for in the DataArray's attributes (e.g., 'valid_time', 'accumulation_start_time').

            Returns:
                Optional[datetime]: The parsed datetime object if successful, or None if the attribute is missing or cannot be parsed.
            """
            value = data_array.attrs.get(key)

            if value is None:
                return None

            try:
                result: datetime = pd.Timestamp(value).to_pydatetime()
                return result
            except (ValueError, TypeError):
                return None

        return _parse("valid_time"), _parse("accumulation_start_time")

    def _resolve_accumulation_times(
        self: "MPASPrecipitationPlotter",
        time_end: Optional[datetime],
        time_start: Optional[datetime],
        data_array: Optional[xr.DataArray],
        dataset: Optional[xr.Dataset],
        config: Optional[Any],
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        This method resolves the accumulation end and start times for annotation by considering explicit parameters, metadata from the DataArray, and fallback logic based on the dataset and config. The explicit parameters take precedence, followed by metadata if it is consistent with the explicit values, and finally a fallback to the dataset's Time coordinate based on the config's time_index if necessary. This allows for flexible handling of time information while ensuring that any derived times are consistent and meaningful for annotating the precipitation plot. The method returns the resolved (time_end, time_start) pair that can be used for annotation or further processing.

        Parameters:
            time_end (Optional[datetime]): Explicit accumulation end time; takes precedence when provided.
            time_start (Optional[datetime]): Explicit accumulation start time; takes precedence when provided.
            data_array (Optional[xr.DataArray]): DataArray whose attrs may carry 'valid_time' and 'accumulation_start_time'.
            dataset (Optional[xr.Dataset]): Dataset providing the Time coordinate for the config fallback.
            config (Optional[Any]): Configuration object whose time_index drives the fallback derivation.

        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: The resolved (time_end, time_start) pair.
        """
        # Read any valid-time metadata attached by the diagnostics
        meta_end, meta_start = self._read_time_metadata(data_array)

        # Resolve the end time with precedence: explicit parameter > metadata > config fallback
        if time_end is None:
            time_end = meta_end

        # If time_end is still None, attempt to derive it from the dataset's Time coordinate using the config's time_index
        if time_end is None and dataset is not None and config is not None:
            time_idx = getattr(config, "time_index", None)
            if (
                time_idx is not None
                and hasattr(dataset, "Time")
                and len(dataset.Time) > time_idx
            ):
                time_end = pd.Timestamp(dataset.Time.values[time_idx]).to_pydatetime()

        # Resolve the start time from metadata only when it is consistent with the resolved end
        if time_start is None and meta_start is not None:
            consistent = (
                meta_end is None
                or time_end is None
                or pd.Timestamp(meta_end) == pd.Timestamp(time_end)
            )
            if consistent:
                time_start = meta_start

        # Return the resolved accumulation window endpoints
        return time_end, time_start

    @close_figure_on_error
    def create_precipitation_map(
        self: "MPASPrecipitationPlotter",
        lon: np.ndarray,
        lat: np.ndarray,
        precip_data: np.ndarray,
        bounds: GeographicBounds,
        accum_period: str = "a01h",
        style: Optional[PrecipitationRenderStyle] = None,
        time_end: Optional[datetime] = None,
        time_start: Optional[datetime] = None,
        data_array: Optional[xr.DataArray] = None,
        var_name: str = "precipitation",
        dataset: Optional[xr.Dataset] = None,
        config: Optional[Any] = None,
    ) -> Tuple[Figure, Axes]:
        """
        This method creates a precipitation map from MPAS unstructured mesh data using cartopy for geographic visualization. It supports multiple rendering options including scatter plots of cell values and interpolated contour/filled contour maps. The method handles unit conversion, colormap selection, data validation, and geographic features to produce a professional-quality visualization of precipitation fields. It also includes robust handling of edge cases such as empty or invalid data, and provides informative annotations about the accumulation period. The resulting figure and axes are returned for further customization or saving.

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS cell centers.
            precip_data (np.ndarray): Precipitation values corresponding to the lon/lat points (in model or display units).
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h') used for colormap selection and annotation; when data_array records a different accumulation period a warning is logged and the data's period is used for both.
            style (Optional[PrecipitationRenderStyle]): Rendering and appearance settings (title, plot_type, colormap, levels, clim_min, clim_max, projection, grid_resolution, display_units, threshold). Levels, color limits and the threshold are expressed in display_units ('mm', 'in' or 'cm'); cells at or above the threshold are outlined with bold black rings and counted in a legend entry. If None, defaults are used.
            time_end (Optional[datetime]): Optional end datetime for the accumulation period. If provided, it will be used to derive the start time based on the accumulation period if the start time is not explicitly provided.
            time_start (Optional[datetime]): Optional start datetime for the accumulation period. If not provided but time_end is given, it will be derived from time_end and accum_period.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing metadata attributes for unit conversion. If None, no conversion is attempted and original data is used.
            var_name (str): Variable name used for metadata lookup and unit conversion logic.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset containing the original MPAS data, which may be needed for interpolation and remapping in contour/contourf plots.

        Returns:
            Tuple[Figure, Axes]: A tuple containing the created matplotlib Figure and Axes objects for the precipitation map. The figure and axes are also stored as instance attributes for further manipulation or saving.
        """
        # Unpack the geographic extent
        lon_min, lon_max, lat_min, lat_max = bounds

        # Default style to PrecipitationRenderStyle if not specified
        if style is None:
            style = PrecipitationRenderStyle()

        title = style.title
        plot_type = style.plot_type
        colormap = style.colormap
        levels = style.levels
        clim_min = style.clim_min
        clim_max = style.clim_max
        projection = style.projection
        grid_resolution = style.grid_resolution
        display_units = self._normalize_display_units(style.display_units)
        threshold = style.threshold

        if threshold is not None and not np.isfinite(threshold):
            raise ValueError(f"threshold must be a finite number, got {threshold}")

        # Close the figure left over from an earlier call on this plotter so reusing the plotter does not leak figures
        self._release_previous_figure()

        # Validate the accumulation period (form 'aNNh' with a sensible number of hours) before any figure is created
        self._accum_period_hours(accum_period)

        # Follow the accumulation period recorded in the data, warning when it differs from the requested one
        accum_period = self._reconcile_accum_period(accum_period, data_array)

        # Validate plot type parameter to ensure it is one of the accepted values for rendering methods
        if plot_type not in ["scatter", "contour", "contourf"]:
            raise ValueError(
                f"plot_type must be 'scatter', 'contour', or 'contourf', got '{plot_type}'"
            )

        # Validate geographic extent parameters to ensure they are within acceptable ranges and properly ordered.
        # Longitudes may extend past ±180 (e.g. 150..210 or 0..360) so dateline-crossing boxes can be requested.
        if not (
            -360 <= lon_min <= 360
            and -360 <= lon_max <= 360
            and lon_max - lon_min <= 360
            and -90 <= lat_min <= 90
            and -90 <= lat_max <= 90
            and lon_max > lon_min
            and lat_max > lat_min
        ):
            raise ValueError("Invalid plot extent parameters")

        # Convert precipitation data units and extract unit label for colorbar
        precip_data, unit_label = self._convert_precipitation_units(
            precip_data, data_array, var_name, display_units
        )

        # Reject inputs that cannot produce a meaningful map before any figure is created (as the surface plotter does)
        self._validate_precipitation_inputs(lon, lat, precip_data, var_name)

        # Longitudes may arrive in 0..360 (e.g. lonCell in degrees); the extent filters, the remapper
        # back-mapping and the dataset alignment all work in [-180, 180]
        lon = MPASGeographicUtils.normalize_longitude(
            np.asarray(self.convert_to_numpy(lon), dtype=float)
        )

        # Setup figure, axes, projection, and geographic features for precipitation map
        _, data_crs = self._setup_precipitation_figure(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            projection,
            central_longitude=style.central_longitude,
            central_latitude=style.central_latitude,
            proj_kwargs=style.proj_kwargs,
        )

        # Assert fig and ax are created and of correct type for cartopy plotting
        assert (
            self.fig is not None
        ), "Figure must be created by _setup_precipitation_figure"
        assert (
            self.ax is not None
        ), "Axes must be created by _setup_precipitation_figure"
        assert isinstance(self.ax, GeoAxes), self._GEOAXES_MSG

        # Prepare colormap, contour levels, and normalization for precipitation visualization
        cmap, norm, color_levels_sorted = self._prepare_precipitation_colormap(
            colormap,
            levels,
            accum_period,
            clim_min,
            clim_max,
            display_units if unit_label == display_units else "mm",
        )

        # Prepare and validate data for plotting, returning only valid points within geographic extent (for scatter) or valid values (for contour/contourf)
        lon_valid, lat_valid, precip_valid = self._prepare_precipitation_data(
            lon, lat, precip_data, lon_min, lon_max, lat_min, lat_max, plot_type
        )

        # Handle empty data case (all-NaN or no valid points)
        if len(precip_valid) > 0:
            logger.info(
                "Plotting %s precipitation points for %s",
                f"{len(precip_valid):,}",
                var_name,
            )
            logger.debug(
                "Precipitation range: %.3f to %.3f %s",
                float(precip_valid.min()),
                float(precip_valid.max()),
                unit_label,
            )

            # Render based on plot type using helper methods
            if plot_type == "scatter":
                # Scatter plot can use original lon/lat points without interpolation, directly colored by precipitation values
                self._create_scatter_plot(
                    lon_valid,
                    lat_valid,
                    precip_valid,
                    cmap,
                    norm,
                    data_crs,
                    colorbar_label=f"Precipitation [{unit_label}]",
                    colorbar_ticks=color_levels_sorted,
                )
            elif plot_type == "contour":
                # Contour requires interpolation to a regular grid, which is handled in the helper method using MPASRemapper
                self._create_contour_plot(
                    lon_valid,
                    lat_valid,
                    precip_valid,
                    GeographicBounds(lon_min, lon_max, lat_min, lat_max),
                    cmap,
                    norm,
                    color_levels_sorted,
                    data_crs,
                    grid_resolution,
                    dataset,
                    config=config,
                )
            elif plot_type == "contourf":
                # Contourf requires interpolation to a regular grid, which is handled in the helper method using MPASRemapper
                self._create_contourf_plot(
                    lon_valid,
                    lat_valid,
                    precip_valid,
                    GeographicBounds(lon_min, lon_max, lat_min, lat_max),
                    cmap,
                    norm,
                    color_levels_sorted,
                    data_crs,
                    grid_resolution,
                    dataset,
                    colorbar_label=f"Precipitation [{unit_label}]",
                    colorbar_ticks=color_levels_sorted,
                    config=config,
                )

            if threshold is not None:
                self._highlight_threshold(
                    lon_valid,
                    lat_valid,
                    precip_valid,
                    threshold,
                    unit_label,
                    (lon_min, lon_max, lat_min, lat_max),
                    data_crs,
                )
        else:
            logger.warning("No valid precipitation data points found for %s", var_name)

        # Add gridlines with labels and custom formatting
        self._add_gridlines(data_crs)

        # Set title and add time annotation
        self.ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Resolve the accumulation window (valid-time metadata preferred, config.time_index as fallback)
        time_end, time_start = self._resolve_accumulation_times(
            time_end, time_start, data_array, dataset, config
        )

        # Add time annotation for accumulation period
        self._add_time_annotation(time_end, time_start, accum_period)

        # Adjust layout to prevent overlap of elements
        plt.tight_layout()

        # Add timestamp and branding to the plot
        self.add_timestamp_and_branding()

        # Return the figure and axes for further manipulation or saving
        return self.fig, self.ax

    def _highlight_threshold(
        self: "MPASPrecipitationPlotter",
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        precip_valid: np.ndarray,
        threshold: float,
        unit_label: str,
        extent: Tuple[float, float, float, float],
        data_crs: ccrs.CRS,
    ) -> None:
        """
        This method visibly highlights the cells whose precipitation is at or above a threshold by drawing bold black hollow rings around them on top of the filled map (for every plot type), and adds a legend entry stating the threshold, its units and how many plotted cells reach it. Only cells inside the map extent are counted and drawn, with longitudes re-expressed in the extent's frame so dateline-crossing boxes work; when no cell reaches the threshold the legend says so.

        Parameters:
            lon_valid (np.ndarray): Longitudes of the valid precipitation points in degrees.
            lat_valid (np.ndarray): Latitudes of the valid precipitation points in degrees.
            precip_valid (np.ndarray): Valid precipitation values in display units.
            threshold (float): Threshold in display units; cells with values greater than or equal to it are highlighted.
            unit_label (str): Display units used in the legend text.
            extent (Tuple[float, float, float, float]): Map extent as (lon_min, lon_max, lat_min, lat_max) in degrees.
            data_crs (ccrs.CRS): Coordinate reference system of the point coordinates.

        Returns:
            None
        """
        assert self.ax is not None, "Axes must be created before highlighting"

        lon_min, lon_max, lat_min, lat_max = extent
        lon_w = MPASGeographicUtils.wrap_longitude_to_extent(
            np.asarray(lon_valid), lon_min, lon_max
        )
        hit = (
            (precip_valid >= threshold)
            & (lon_w >= lon_min)
            & (lon_w <= lon_max)
            & (lat_valid >= lat_min)
            & (lat_valid <= lat_max)
        )
        n_hit = int(np.sum(hit))

        if n_hit > 0:
            size = self.calculate_adaptive_marker_size(
                (lon_min, lon_max, lat_min, lat_max),
                int(len(precip_valid)),
                (self.figsize[0], self.figsize[1]),
            )
            self.ax.scatter(
                lon_w[hit],
                lat_valid[hit],
                s=max(float(size), 4.0) * 1.8,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                transform=data_crs,
                zorder=6,
            )

        label = (
            f"Threshold \u2265 {threshold:.3g} {unit_label}: {n_hit:,} cells"
            if n_hit > 0
            else f"Threshold \u2265 {threshold:.3g} {unit_label}: no cells"
        )
        handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            label=label,
        )
        self.ax.legend(handles=[handle], loc="upper right", fontsize=9, framealpha=0.85)

    def _interpolate_to_grid(
        self: "MPASPrecipitationPlotter",
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        grid_resolution: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None,
        method: str = "linear",
        resolution_bounds: Optional[Tuple[float, float]] = (0.1, 1.0),
        comm: Optional[Any] = None,
        config: Optional[dict] = None,
        interpolation: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method performs interpolation of the unstructured MPAS data onto a regular grid defined by the specified geographic extent and grid resolution. It uses the MPASRemapper to handle the interpolation, which can be computationally intensive for large datasets, so it includes options for parallel processing with MPI. The method also allows for specifying the interpolation method (e.g., 'linear', 'nearest') and bounds on the grid resolution to ensure that the resulting grid is appropriate for the data density and geographic extent. The returned longitude, latitude, and interpolated data arrays are suitable for contour and filled contour plotting with cartopy.

        Parameters:
            lon (np.ndarray): 1D array of longitude coordinates in degrees for MPAS cell centers.
            lat (np.ndarray): 1D array of latitude coordinates in degrees for MPAS cell centers.
            data (np.ndarray): Data values corresponding to the lon/lat points (in display units) that need to be interpolated onto a regular grid.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            grid_resolution (Optional[float]): Optional grid resolution in degrees for the regular grid. If None, a default resolution will be determined based on the data density and geographic extent.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset containing the original MPAS data, which may be needed for interpolation and remapping. If None, the method will attempt to perform interpolation without the dataset context, which may limit the interpolation options available.
            method (str): Interpolation method to use (e.g., 'linear', 'nearest'). The choice of method can affect the appearance of the resulting contour plots, with 'linear' providing smoother results and 'nearest' preserving original data values but resulting in a more blocky appearance.
            resolution_bounds (Optional[Tuple[float, float]]): Optional tuple specifying the minimum and maximum grid resolution in degrees to use for interpolation. This can help ensure that the resulting grid is not too coarse (which can miss important features) or too fine (which can be computationally expensive and may not be justified by the data density). The method will select an appropriate grid resolution within these bounds based on the geographic extent and the number of data points, ensuring a balance between detail and performance in the resulting contour plots.
            comm (Optional[Any]): Optional MPI communicator for parallel processing of the interpolation. If provided, the method will use the communicator to distribute the interpolation workload across multiple processes, which can significantly reduce the time required for interpolation of large datasets. If None, the interpolation will be performed in a single process, which may be sufficient for smaller datasets but can be very slow for larger datasets.
            config (Optional[dict]): Optional configuration object with remap settings.
            interpolation (Optional[str]): Explicit gridding method ('nearest', 'linear' or 'cubic') overriding method and the config remap method (default: None).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the longitude, latitude, and interpolated data arrays on the regular grid. The longitude and latitude arrays define the coordinates of the grid points, and the data array contains the interpolated values at those grid points. These arrays are suitable for use in contour and filled contour plotting with cartopy, allowing for smooth visualization of the precipitation field across the specified geographic extent. The method handles the complexities of interpolation from an unstructured mesh to a regular grid, ensuring that the resulting data is appropriate for visualization while also providing options for performance optimization through parallel processing and resolution control.
        """
        return super()._interpolate_to_grid(
            lon,
            lat,
            data,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            grid_resolution=grid_resolution,
            dataset=dataset,
            method=method,
            resolution_bounds=resolution_bounds,
            comm=comm,
            config=config,
            interpolation=interpolation,
        )

    @staticmethod
    def _resolve_overlay_extent(
        extent: Optional[Tuple[Optional[float], ...]],
        data_bounds: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """
        This static helper merges a requested map extent with the overlay data's own bounds: every entry of the extent that is not None replaces the corresponding data bound, and a missing extent leaves the data bounds unchanged.

        Parameters:
            extent (Optional[Tuple[Optional[float], ...]]): Map extent as (lon_min, lon_max, lat_min, lat_max) with optional None entries, or None.
            data_bounds (Tuple[float, float, float, float]): Bounds derived from the overlay data in the same order.

        Returns:
            Tuple[float, float, float, float]: The resolved (lon_min, lon_max, lat_min, lat_max).
        """
        if extent is None:
            return data_bounds

        merged = [
            given if given is not None else fallback
            for given, fallback in zip(extent, data_bounds)
        ]
        return (merged[0], merged[1], merged[2], merged[3])

    def _convert_overlay_units(
        self: "MPASPrecipitationPlotter",
        precip_data: np.ndarray,
        var_name: str,
        original_units: Optional[str],
    ) -> np.ndarray:
        """
        This helper converts overlay precipitation values from their original units to the display units chosen by UnitConverter for the variable. Nothing happens when no original units are given or they already equal the display units, and a failed conversion is logged as a warning and leaves the data unchanged.

        Parameters:
            precip_data (np.ndarray): Precipitation values in the original units.
            var_name (str): Variable name used to pick the display units.
            original_units (Optional[str]): Original units, or None to skip conversion.

        Returns:
            np.ndarray: The converted values (or the input when no conversion applies or it fails).
        """
        if not original_units:
            return precip_data

        display_units = UnitConverter.get_display_units(var_name, original_units)

        if original_units == display_units:
            return precip_data

        try:
            converted_data = UnitConverter.convert_units(
                precip_data, original_units, display_units
            )
            logger.debug(
                "Converted overlay %s from %s to %s",
                var_name,
                original_units,
                display_units,
            )
            return self.convert_to_numpy(converted_data)
        except ValueError as e:
            logger.warning(
                "Could not convert overlay %s from %s to %s: %s",
                var_name,
                original_units,
                display_units,
                e,
            )
            return precip_data

    def _prepare_overlay_data(
        self: "MPASPrecipitationPlotter",
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        precip_data: np.ndarray,
        var_name: str,
        original_units: Optional[str],
        plot_type: str,
        extent: Optional[Tuple[Optional[float], ...]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """
        This method prepares and validates the longitude, latitude, and precipitation data for overlay plotting. It handles unit conversion if original units are provided, checks for physically valid precipitation values (e.g., non-negative), and creates a validity mask based on the plot type. For scatter plots, it filters out points that fall outside the map extent in addition to invalid precipitation values. For contour and contourf plots, it focuses on validating the precipitation values since geographic masking will be handled during interpolation. The method returns only the valid points that can be plotted along with the geographic bounds for use in rendering the overlay. If no valid points exist after filtering, it raises a ValueError to indicate that the overlay cannot be plotted with the given data.

         Parameters:
             lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates for the overlay data, which can be a numpy array or an xarray DataArray.
             lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates for the overlay data, which can be a numpy array or an xarray DataArray.
             precip_data (np.ndarray): Precipitation values corresponding to the lon/lat points (in model or display units).
             var_name (str): Variable name used for metadata lookup and unit conversion logic.
             original_units (Optional[str]): Original units of the precipitation data, used for unit conversion if provided. If None, no conversion is attempted and original data is used.
             plot_type (str): Rendering mode: 'scatter', 'contour', or 'contourf'. This determines the validation criteria applied to the data.
             extent (Optional[Tuple[Optional[float], ...]]): Map extent as (lon_min, lon_max, lat_min, lat_max); any None entry falls back to the overlay data's own bound. Scatter filtering and the returned bounds use the map extent so the overlay is clipped to the map rather than to its own data (default: None).

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
                logger.warning(
                    "Precipitation data may not be in mm. Consider specifying 'original_units' in config."
                )

        # Attempt unit conversion if original units are provided and differ from display units
        precip_data = self._convert_overlay_units(precip_data, var_name, original_units)

        # Mask physically invalid negative values (NaN, not drawn) with a warning
        precip_data = self._mask_negative_precipitation(precip_data)

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

        # Prefer the map extent over the overlay's own data bounds when supplied
        lon_min, lon_max, lat_min, lat_max = self._resolve_overlay_extent(
            extent, (lon_min, lon_max, lat_min, lat_max)
        )
        if extent is not None:
            # Re-express longitudes in the extent's frame so dateline-crossing boxes keep their eastern cells
            lon_flat = MPASGeographicUtils.wrap_longitude_to_extent(
                lon_flat, lon_min, lon_max
            )

        # Create validity mask based on plot type and data quality
        if plot_type == "scatter":
            # For scatter plots, we will filter out points outside the map extent to avoid plotting irrelevant data, in addition to filtering invalid precipitation values
            valid_mask = (
                np.isfinite(precip_data_flat)
                & (precip_data_flat >= 0)
                & (precip_data_flat < 1e5)
                & (lon_flat >= lon_min)
                & (lon_flat <= lon_max)
                & (lat_flat >= lat_min)
                & (lat_flat <= lat_max)
            )
        else:
            # For contour and contourf, we will rely on remapping to handle masking, so we only filter out invalid precipitation values here
            valid_mask = (
                np.isfinite(precip_data_flat)
                & (precip_data_flat >= 0)
                & (precip_data_flat < 1e5)
            )

        # Check if any valid data points exist after filtering
        if not np.any(valid_mask):
            raise ValueError(
                f"No valid precipitation overlay data found for {var_name}"
            )

        # Extract valid data for overlay rendering
        lon_valid = lon_flat[valid_mask]
        lat_valid = lat_flat[valid_mask]
        precip_valid = precip_data_flat[valid_mask]

        # Return valid data and bounds for overlay rendering
        return lon_valid, lat_valid, precip_valid, lon_min, lon_max, lat_min, lat_max

    def _setup_overlay_colormap(
        self: "MPASPrecipitationPlotter",
        colormap: Optional[str],
        levels: Optional[List[float]],
        accum_period: str,
    ) -> Tuple[mcolors.Colormap, BoundaryNorm, List[float]]:
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
        color_levels_sorted = sorted({v for v in color_levels if np.isfinite(v)})

        # Set bounds for BoundaryNorm (zero floor plus the levels) so the colours follow the levels exactly
        bounds = self._precip_bounds(color_levels_sorted, True)

        # Use BoundaryNorm for discrete color mapping; values above the top level take the over colour
        norm = self._precip_norm(bounds, cmap)

        # Return colormap, normalization, and sorted levels for consistent use in overlay rendering
        return cmap, norm, color_levels_sorted

    def _calculate_overlay_grid_resolution(
        self: "MPASPrecipitationPlotter",
        grid_resolution_input: Optional[float],
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> float:
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

    def _render_overlay_scatter(
        self: "MPASPrecipitationPlotter",
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
        alpha: float,
    ) -> None:
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
        marker_size = self.calculate_adaptive_marker_size(
            map_extent, len(precip_valid), self.figsize
        )

        # Sort data for proper color overlay (plot smaller values on top for better visibility)
        sort_indices = np.argsort(precip_valid)
        lon_sorted = lon_valid[sort_indices]
        lat_sorted = lat_valid[sort_indices]
        precip_sorted = precip_valid[sort_indices]

        # Plot scatter points with appropriate color mapping and transparency
        ax.scatter(
            lon_sorted,
            lat_sorted,
            c=precip_sorted,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            edgecolors="none",
        )

    def _render_overlay_interpolated(
        self: "MPASPrecipitationPlotter",
        ax: Axes,
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        precip_valid: np.ndarray,
        bounds: GeographicBounds,
        plot_type: str,
        resolution: float,
        var_name: str,
        dataset: Optional[xr.Dataset],
        color_spec: OverlayColorSpec,
        lon: np.ndarray,
        lat: np.ndarray,
        config: Optional[Any] = None,
    ) -> None:
        """
        This method renders the precipitation overlay as either contour lines or filled contours based on the specified plot type. It first performs interpolation of the valid precipitation data onto a regular lat-lon grid using the MPASRemapper, which is designed to handle the unstructured nature of MPAS data and can apply masking to ensure that only valid data points contribute to the interpolation. The method then extracts the gridded longitude, latitude, and interpolated precipitation values for plotting. For contour plots, it draws contour lines with specified levels and adds labels to the contours for better readability. For filled contour plots, it fills the contours with colors based on the provided colormap and normalization, ensuring that the colorbar reflects the same levels for consistency. The method also applies a specified transparency level (alpha) to allow for better visualization of underlying map features while still clearly showing the precipitation patterns. This approach allows for a smooth and visually appealing representation of precipitation fields derived from unstructured MPAS data.

        Parameters:
            ax (Axes): Target axes for overlay.
            lon_valid (np.ndarray): Valid longitude coordinates for plotting.
            lat_valid (np.ndarray): Valid latitude coordinates for plotting.
            precip_valid (np.ndarray): Valid precipitation values for plotting.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            plot_type (str): Rendering mode: 'contour' for lines or 'contourf' for filled contours.
            resolution (float): Grid resolution in degrees for interpolation.
            var_name (str): Variable name for labeling and unit conversion logic.
            dataset (Optional[xr.Dataset]): Optional xarray Dataset containing the original MPAS data, which may be needed for interpolation and remapping. If None, interpolation will be attempted with available data arrays.
            color_spec (OverlayColorSpec): Colormap, normalization, sorted contour levels, and transparency (alpha) used to render the overlay.
            lon (np.ndarray): Original longitude array for the MPAS cell centers, used for interpolation.
            lat (np.ndarray): Original latitude array for the MPAS cell centers, used for interpolation.
            config (Optional[Any]): Optional configuration object for additional settings.

        Returns:
            None: This method modifies the provided axes in place by adding the contour lines or filled contours for the precipitation overlay. It does not return any value.
        """
        # Unpack the geographic extent and color specification
        lon_min, lon_max, lat_min, lat_max = bounds
        cmap = color_spec.cmap
        norm = color_spec.norm
        color_levels_sorted = color_spec.color_levels_sorted
        alpha = color_spec.alpha

        # Perform interpolation of the valid precipitation data onto a regular lat-lon grid
        remapped_precip = self._remap_overlay_data(
            lon_valid,
            lat_valid,
            precip_valid,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            resolution,
            var_name,
            dataset,
            lon,
            lat,
            config,
        )

        # Extract grid and data for plotting
        lon_grid = remapped_precip.lon.values
        lat_grid = remapped_precip.lat.values
        precip_grid = remapped_precip.values

        logger.debug(
            "Remapped to %dx%d grid", precip_grid.shape[0], precip_grid.shape[1]
        )

        # Render contours based on plot type using the gridded data
        self._draw_overlay_contours(
            ax,
            lon_grid,
            lat_grid,
            precip_grid,
            plot_type,
            cmap,
            norm,
            color_levels_sorted,
            alpha,
        )

    def _remap_overlay_data(
        self: "MPASPrecipitationPlotter",
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        precip_valid: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        resolution: float,
        var_name: str,
        dataset: Optional[xr.Dataset],
        lon: np.ndarray,
        lat: np.ndarray,
        config: Optional[Any],
    ) -> xr.DataArray:
        """
        This method handles the remapping of the precipitation overlay data from the unstructured MPAS grid to a regular lat-lon grid suitable for contour plotting. It first determines whether to use the legacy ESMPy remapper based on the availability of boundary data and user configuration. If ESMPy is available and appropriate, it attempts to perform a conservative remap, which can provide better conservation properties for precipitation fields. If the ESMPy remap fails or is not suitable, it falls back to a KD-Tree based interpolation method, which is more robust but may not conserve quantities as well. The method ensures that a dataset with cell-center coordinates is available for remapping, either by using the provided dataset or by constructing one from the longitude and latitude arrays. The resulting remapped precipitation field is returned as an xarray DataArray on a regular lat-lon grid defined by the specified geographic extent and resolution, ready for contour plotting with cartopy.

        Parameters:
            lon_valid (np.ndarray): Valid longitude coordinates of the data points.
            lat_valid (np.ndarray): Valid latitude coordinates of the data points.
            precip_valid (np.ndarray): Valid precipitation values for the data points.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            resolution (float): Grid resolution in degrees for interpolation.
            var_name (str): Variable name used in log messages.
            dataset (Optional[xr.Dataset]): Optional dataset with the MPAS grid info.
            lon (np.ndarray): Original longitude array for the MPAS cell centers.
            lat (np.ndarray): Original latitude array for the MPAS cell centers.
            config (Optional[Any]): Optional configuration controlling remap engine and method.

        Returns:
            xr.DataArray: The remapped precipitation field on the regular lat-lon grid.
        """
        engine = getattr(config, "remap_engine", None)
        esmf_method = (
            getattr(config, "remap_method", "conservative")
            if config
            else "conservative"
        )
        kdtree_method = (
            getattr(config, "remap_method", "linear") if config else "linear"
        )

        use_esmf = engine == "esmf" or (
            engine is None
            and ESMPY_AVAILABLE
            and self._has_boundary_data(self._ensure_boundary_data(dataset))
        )

        lon_arr = lon if isinstance(lon, np.ndarray) else np.asarray(lon)
        lat_arr = lat if isinstance(lat, np.ndarray) else np.asarray(lat)

        remapped_precip = None

        if config is not None:
            remapped_precip = self._remap_overlay_via_config(
                precip_valid,
                dataset,
                lon_valid,
                lat_valid,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                resolution,
                config,
            )
        elif use_esmf:
            remapped_precip = self._remap_overlay_via_esmf(
                lon_valid,
                lat_valid,
                precip_valid,
                lon_arr,
                lat_arr,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                resolution,
                var_name,
                dataset,
                esmf_method,
            )

        if remapped_precip is None:
            remapped_precip = self._remap_overlay_via_kdtree(
                precip_valid,
                dataset,
                lon_valid,
                lat_valid,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                resolution,
                var_name,
                kdtree_method,
            )

        return remapped_precip

    @staticmethod
    def _ensure_overlay_dataset(
        dataset: Optional[xr.Dataset], lon_arr: np.ndarray, lat_arr: np.ndarray
    ) -> xr.Dataset:
        """
        This method ensures that an xarray Dataset with the necessary longitude and latitude coordinates for the MPAS cell centers is available for remapping. If a dataset is provided, it is returned as-is. If no dataset is provided, a minimal dataset is constructed using the provided longitude and latitude arrays, with dimensions named 'nCells' to match the expected structure for remapping. This allows the remapping functions to operate with a consistent dataset structure regardless of whether the user has provided an existing dataset or not.

        Parameters:
            dataset (Optional[xr.Dataset]): Existing dataset, or None to build one.
            lon_arr (np.ndarray): Longitude array for the MPAS cell centers.
            lat_arr (np.ndarray): Latitude array for the MPAS cell centers.

        Returns:
            xr.Dataset: The provided dataset, or a minimal one built from the arrays.
        """
        if dataset is not None:
            return dataset
        return xr.Dataset(
            {
                "lonCell": xr.DataArray(lon_arr, dims=["nCells"]),
                "latCell": xr.DataArray(lat_arr, dims=["nCells"]),
            }
        )

    def _remap_overlay_via_config(
        self: "MPASPrecipitationPlotter",
        precip_valid: np.ndarray,
        dataset: Optional[xr.Dataset],
        lon_arr: np.ndarray,
        lat_arr: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        resolution: float,
        config: Any,
    ) -> xr.DataArray:
        """
        This method handles remapping of the overlay data using a user-specified configuration for the remap engine and method. It ensures that a dataset with cell-center coordinates is available, then delegates the remapping to the dispatch_remap function, which will select the appropriate remapping backend based on the configuration. This allows users to specify their preferred remapping approach (e.g., ESMPy, KD-Tree) through the configuration, while still ensuring that the necessary data structure is in place for the remapping to occur. The resulting remapped precipitation field is returned as an xarray DataArray on a regular lat-lon grid defined by the specified geographic extent and resolution.

        Parameters:
            precip_valid (np.ndarray): Valid precipitation values for the data points.
            dataset (Optional[xr.Dataset]): Optional dataset with the MPAS grid info.
            lon_arr (np.ndarray): Longitudes of the valid points in precip_valid.
            lat_arr (np.ndarray): Latitudes of the valid points in precip_valid.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            resolution (float): Grid resolution in degrees for interpolation.
            config (Any): Configuration controlling remap engine and method.

        Returns:
            xr.DataArray: The remapped precipitation field on the regular lat-lon grid.
        """
        dataset = self._ensure_overlay_dataset(dataset, lon_arr, lat_arr)
        data_xr = xr.DataArray(
            self._align_to_dataset(dataset, lon_arr, lat_arr, precip_valid),
            dims=["nCells"],
        )
        return dispatch_remap(
            data=data_xr,
            dataset=dataset,
            config=config,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            apply_mask=True,
            lon_convention="auto",
        )

    def _remap_overlay_via_esmf(
        self: "MPASPrecipitationPlotter",
        lon_valid: np.ndarray,
        lat_valid: np.ndarray,
        precip_valid: np.ndarray,
        lon_arr: np.ndarray,
        lat_arr: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        resolution: float,
        var_name: str,
        dataset: Optional[xr.Dataset],
        esmf_method: str,
    ) -> Optional[xr.DataArray]:
        """
        This method attempts to remap the overlay precipitation data onto a regular lat-lon grid using the ESMPy remapper, which is designed to handle the unstructured nature of MPAS data and can apply masking to ensure that only valid data points contribute to the interpolation. It first ensures that a dataset with cell-center coordinates is available, then performs a conservative remap using the provided method. If the ESMPy remapping fails for any reason (e.g., missing boundary data, issues with the remapping process), it catches the exception, logs a warning, and returns None to indicate that the fallback KD-Tree method should be used instead. This allows for robust handling of remapping while still attempting to use the more sophisticated ESMPy approach when possible.

        Parameters:
            lon_valid (np.ndarray): Valid longitude coordinates of the data points.
            lat_valid (np.ndarray): Valid latitude coordinates of the data points.
            precip_valid (np.ndarray): Valid precipitation values for the data points.
            lon_arr (np.ndarray): Longitude array for the MPAS cell centers.
            lat_arr (np.ndarray): Latitude array for the MPAS cell centers.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            resolution (float): Grid resolution in degrees for interpolation.
            var_name (str): Variable name used in log messages.
            dataset (Optional[xr.Dataset]): Optional dataset with the MPAS grid info.
            esmf_method (str): The ESMPy remap method (e.g. 'conservative').

        Returns:
            Optional[xr.DataArray]: The remapped field on success, or None to fall back.
        """
        dataset = self._ensure_boundary_data(dataset)

        logger.info(
            "Interpolating %s overlay via ESMPy/%s (resolution: %.3f°)",
            var_name,
            esmf_method,
            resolution,
        )

        try:
            full_data = self._backmap_to_full_grid(
                lon_valid, lat_valid, precip_valid, lon_arr, lat_arr
            )
            return self._remap_conservative(
                full_data,
                lon_arr,
                lat_arr,
                dataset,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                resolution,
                method=esmf_method,
            )
        except Exception as exc:
            logger.warning(
                "ESMPy/%s overlay remapping failed (%s); falling back to KD-Tree.",
                esmf_method,
                exc,
            )
            return None

    def _remap_overlay_via_kdtree(
        self: "MPASPrecipitationPlotter",
        precip_valid: np.ndarray,
        dataset: Optional[xr.Dataset],
        lon_arr: np.ndarray,
        lat_arr: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        resolution: float,
        var_name: str,
        kdtree_method: str,
    ) -> xr.DataArray:
        """
        This method remaps the overlay precipitation data onto a regular lat-lon grid using a KD-Tree based interpolation method. It ensures that a dataset with cell-center coordinates is available, then performs the remapping using the remap_mpas_to_latlon_with_masking function, which applies masking to ensure that only valid data points contribute to the interpolation. The method logs the interpolation process and returns the remapped precipitation field as an xarray DataArray on a regular lat-lon grid defined by the specified geographic extent and resolution. This approach provides a robust fallback for remapping when ESMPy is not available or fails, allowing for effective visualization of precipitation fields derived from unstructured MPAS data.

        Parameters:
            precip_valid (np.ndarray): Valid precipitation values for the data points.
            dataset (Optional[xr.Dataset]): Optional dataset with the MPAS grid info.
            lon_arr (np.ndarray): Longitudes of the valid points in precip_valid.
            lat_arr (np.ndarray): Latitudes of the valid points in precip_valid.
            lon_min (float): Western boundary of the map extent in degrees.
            lon_max (float): Eastern boundary of the map extent in degrees.
            lat_min (float): Southern boundary of the map extent in degrees.
            lat_max (float): Northern boundary of the map extent in degrees.
            resolution (float): Grid resolution in degrees for interpolation.
            var_name (str): Variable name used in log messages.
            kdtree_method (str): The KD-Tree interpolation method (e.g. 'linear').

        Returns:
            xr.DataArray: The remapped precipitation field on the regular lat-lon grid.
        """
        logger.info(
            "Interpolating %s overlay using KD-Tree/%s (resolution: %.3f°)",
            var_name,
            kdtree_method,
            resolution,
        )

        dataset = self._ensure_overlay_dataset(dataset, lon_arr, lat_arr)
        data_xr = xr.DataArray(
            self._align_to_dataset(dataset, lon_arr, lat_arr, precip_valid),
            dims=["nCells"],
        )

        return remap_mpas_to_latlon_with_masking(
            data=data_xr,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            method=kdtree_method,
            apply_mask=True,
            lon_convention="auto",
        )

    @staticmethod
    def _draw_overlay_contours(
        ax: Axes,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        precip_grid: np.ndarray,
        plot_type: str,
        cmap: mcolors.Colormap,
        norm: BoundaryNorm,
        color_levels_sorted: List[float],
        alpha: float,
    ) -> None:
        """
        This method draws contour lines or filled contours for the precipitation overlay on the provided axes using the gridded longitude, latitude, and precipitation data. It checks the specified plot type to determine whether to draw contour lines (with labels) or filled contours (with color mapping). For contour lines, it uses a solid black line style and attempts to add labels to the contours for better readability, while handling potential issues with too many levels or masked data. For filled contours, it fills the areas between contour levels with colors based on the provided colormap and normalization, ensuring that the colorbar reflects the same levels for consistency. The method applies a specified transparency level (alpha) to allow for better visualization of underlying map features while still clearly showing the precipitation patterns. This approach allows for a clear and visually appealing representation of precipitation fields on the map.

        Parameters:
            ax (Axes): Target axes for the overlay.
            lon_grid (np.ndarray): Gridded longitude values from interpolation.
            lat_grid (np.ndarray): Gridded latitude values from interpolation.
            precip_grid (np.ndarray): Gridded precipitation values from interpolation.
            plot_type (str): 'contour' for lines or 'contourf' for filled contours.
            cmap (mcolors.Colormap): Colormap for filled contours.
            norm (BoundaryNorm): Normalization for color mapping of filled contours.
            color_levels_sorted (List[float]): Sorted contour levels.
            alpha (float): Transparency level for the contours.

        Returns:
            None: Modifies the provided axes in place.
        """
        # Special handling for contour vs contourf to ensure colorbar consistency and proper rendering
        if plot_type == "contour":
            # Draw contour lines with labels and handle potential issues with too many levels or masked data
            contour_set = ax.contour(
                lon_grid,
                lat_grid,
                precip_grid,
                levels=color_levels_sorted,
                colors="black",
                linewidths=1.0,
                linestyles="solid",
                alpha=alpha,
                transform=ccrs.PlateCarree(),
            )
            try:
                # Add contour labels, handling potential issues with too many levels or masked data
                ax.clabel(contour_set, inline=True, fontsize=8, fmt="%g")
            except Exception:
                pass
        else:
            # For filled contours, extend only toward the sides where the gridded data actually leaves the level range
            ax.contourf(
                lon_grid,
                lat_grid,
                precip_grid,
                levels=color_levels_sorted,
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                extend=MPASVisualizer._derive_extend(
                    precip_grid, color_levels_sorted[0], color_levels_sorted[-1]
                ),
            )

    def add_precipitation_overlay(
        self: "MPASPrecipitationPlotter",
        ax: Axes,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        precip_config: Dict[str, Any],
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None,
        config: Optional[Any] = None,
    ) -> None:
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
        precip_data = precip_config["data"]

        # Get accumulation period for colormap selection, defaulting to 'a01h' if not specified
        accum_period = precip_config.get("accum_period", "a01h")

        # Determine plot type for rendering (scatter, contour, or contourf)
        plot_type = precip_config.get("plot_type", "scatter")

        # Get colormap from config or use default based on accumulation period
        colormap = precip_config.get("colormap", None)

        # Get contour levels from config or use defaults based on accumulation period
        levels = precip_config.get("levels", None)

        # Set default alpha for overlay visibility, can be overridden in config
        alpha = precip_config.get("alpha", 0.7)

        # Variable name for labeling and unit conversion
        var_name = precip_config.get("var_name", "precipitation")

        # Calculate grid resolution for interpolation if provided in config, otherwise will be determined adaptively based on map extent and data density
        grid_resolution_input = precip_config.get("grid_resolution", None)

        # Retrieve original units for precipitation data conversion if provided
        original_units = precip_config.get("original_units", None)

        # Validate plot type for overlay rendering
        if plot_type not in ["scatter", "contour", "contourf"]:
            raise ValueError(
                f"plot_type must be 'scatter', 'contour', or 'contourf', got '{plot_type}'"
            )

        # Prepare data: convert units, flatten, validate, and determine bounds for interpolation
        try:
            # Special handling for overlay data preparation to ensure valid points and determine bounds for interpolation
            (
                lon_valid,
                lat_valid,
                precip_valid,
                bounds_lon_min,
                bounds_lon_max,
                bounds_lat_min,
                bounds_lat_max,
            ) = self._prepare_overlay_data(
                lon,
                lat,
                precip_data,
                var_name,
                original_units,
                plot_type,
                (lon_min, lon_max, lat_min, lat_max),
            )
        except ValueError as e:
            logger.warning("%s", e)
            return

        # Use provided bounds or data-derived bounds for filtering and interpolation
        lon_min = lon_min if lon_min is not None else bounds_lon_min
        lon_max = lon_max if lon_max is not None else bounds_lon_max
        lat_min = lat_min if lat_min is not None else bounds_lat_min
        lat_max = lat_max if lat_max is not None else bounds_lat_max

        # Setup colormap and normalization for overlay based on accumulation period and configuration
        cmap, norm, color_levels_sorted = self._setup_overlay_colormap(
            colormap, levels, accum_period
        )

        logger.info(
            "Adding %s precipitation points as %s overlay",
            f"{len(precip_valid):,}",
            plot_type,
        )

        # Render overlay based on plot type with appropriate handling for interpolation
        if plot_type == "scatter":
            # Scatter plot can be rendered directly without interpolation
            self._render_overlay_scatter(
                ax,
                lon_valid,
                lat_valid,
                precip_valid,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                cmap,
                norm,
                alpha,
            )
        else:  # (For contour or contourf)
            # Calculate grid resolution for interpolation if not provided
            resolution = self._calculate_overlay_grid_resolution(
                grid_resolution_input, lon_min, lon_max, lat_min, lat_max
            )

            # Convert original lon/lat to numpy for interpolation if they are xarray DataArrays
            lon_orig = (
                self.convert_to_numpy(lon) if not isinstance(lon, np.ndarray) else lon
            )
            lat_orig = (
                self.convert_to_numpy(lat) if not isinstance(lat, np.ndarray) else lat
            )

            # Render with interpolation using MPASRemapper for smooth contours
            self._render_overlay_interpolated(
                ax,
                lon_valid,
                lat_valid,
                precip_valid,
                GeographicBounds(lon_min, lon_max, lat_min, lat_max),
                plot_type,
                resolution,
                var_name,
                dataset,
                OverlayColorSpec(cmap, norm, color_levels_sorted, alpha),
                lon_orig,
                lat_orig,
                config=config,
            )

        logger.info("Added %s precipitation overlay", plot_type)

    def _extract_coordinates_from_processor(
        self: "MPASPrecipitationPlotter", processor: Any, var_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            if hasattr(processor, "extract_2d_coordinates_for_variable"):
                return cast(
                    Tuple[np.ndarray, np.ndarray],
                    processor.extract_2d_coordinates_for_variable(var_name),
                )

            # Next try general method for spatial coordinates if available
            elif hasattr(processor, "extract_spatial_coordinates"):
                return cast(
                    Tuple[np.ndarray, np.ndarray],
                    processor.extract_spatial_coordinates(),
                )

            # Finally, fallback to direct dataset access if specific methods are not available
            else:
                return (
                    processor.dataset.lonCell.values,
                    processor.dataset.latCell.values,
                )
        except AttributeError:
            # Fallback to direct dataset access if specific methods are not available
            return processor.dataset.lonCell.values, processor.dataset.latCell.values

    def _setup_batch_time_indices(
        self: "MPASPrecipitationPlotter",
        processor: Any,
        accum_period: str,
        time_indices: Optional[List[int]],
    ) -> Tuple[List[int], int]:
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
        time_dim = "Time" if "Time" in processor.dataset.sizes else "time"

        # Determine total available time steps in the dataset for validation
        total_times = processor.dataset.sizes[time_dim]

        # Map accumulation period identifiers to required hours and corresponding time index offsets
        accum_hours = self._accum_period_hours(accum_period)
        min_time_idx = accum_hours

        # Convert the accumulation length to a number of steps using the dataset's time spacing, so non-hourly output still starts at the first index with sufficient lookback
        try:
            step_hours = float(
                MPASDateTimeUtils.calculate_time_delta(processor.dataset)
                / pd.Timedelta(hours=1)
            )
            if np.isfinite(step_hours) and step_hours > 0:
                min_time_idx = max(1, int(np.ceil(accum_hours / step_hours - 1e-9)))
        except Exception:
            min_time_idx = accum_hours

        # Check if dataset has enough time steps for the accumulation period
        if min_time_idx >= total_times:
            logger.warning(
                "Accumulation period %s (%d hours) requires at least %d time steps",
                accum_period,
                accum_hours,
                min_time_idx + 1,
            )
            logger.warning(
                "Dataset only has %d time steps. No plots will be generated.",
                total_times,
            )
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
                logger.warning(
                    "No valid time indices for accumulation period %s",
                    accum_period,
                )
                return [], accum_hours

        # Return validated time indices and accumulation hours for title formatting
        return time_indices, accum_hours

    def _process_single_time_step(
        self: "MPASPrecipitationPlotter",
        processor: Any,
        time_idx: int,
        lon: np.ndarray,
        lat: np.ndarray,
        bounds: GeographicBounds,
        var_name: str,
        accum_period: str,
        plot_type: str,
        grid_resolution: Optional[float],
        style: PrecipitationMapStyle,
        output_dir: str,
        formats: List[str],
        config: Optional[Any] = None,
    ) -> List[str]:
        """
        This method processes a single time step for precipitation plotting in batch mode. It extracts the precipitation data for the specified time index using the PrecipitationDiagnostics class, which handles the calculation of precipitation differences based on the accumulation period and variable type. It then generates a title for the plot using either a custom template provided by the user or a default format that includes the variable name, valid time, accumulation period, and plot type. The method creates a precipitation map for this time step using the create_precipitation_map() method, passing all necessary parameters including coordinates, data, and visualization options. Finally, it saves the generated plot to files with standardized naming conventions that encode the variable type, accumulation type, valid time, and plot type for easy identification. The method returns a list of file paths for the created plots in the specified formats. This workflow allows for automated generation of precipitation maps for each time step in a batch processing context while ensuring that each plot is properly labeled and saved with consistent naming conventions.

        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            time_idx (int): Time index for which to process the precipitation plot.
            lon (np.ndarray): Longitude coordinates for the plot.
            lat (np.ndarray): Latitude coordinates for the plot.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            var_name (str): Precipitation variable name in the dataset (e.g., 'rainnc', 'rainc').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h').
            plot_type (str): Rendering method for the overlay ('scatter', 'contour', 'contourf').
            grid_resolution (Optional[float]): Grid resolution in degrees for interpolation if needed. If None, it will be determined adaptively based on map extent and data density.
            style (PrecipitationMapStyle): Appearance and file-naming settings (file_prefix, custom_title_template, colormap, levels).
            output_dir (str): Directory path for saving the output plot files.
            formats (List[str]): List of file format extensions for saving the plot (e.g., ['png', 'pdf']).
            config (Optional[Any]): Configuration object containing additional parameters for the plot. If None, default settings will be used.

        Returns:
            List[str]: A list of absolute file paths for all successfully created output files for this time step, with standardized naming conventions that encode the variable type, accumulation type, valid time, and plot type for easy identification. If an error occurs during processing, an empty list will be returned and the error will be logged.
        """
        # Unpack the geographic extent and styling options
        lon_min, lon_max, lat_min, lat_max = bounds
        colormap = style.colormap
        levels = style.levels
        custom_title_template = style.custom_title_template
        file_prefix = style.file_prefix

        # Extract precipitation data for this time step using PrecipitationDiagnostics
        if (
            hasattr(processor.dataset, "Time")
            and len(processor.dataset.Time) > time_idx
        ):
            # Convert time coordinate to datetime for title formatting
            time_end = pd.Timestamp(
                processor.dataset.Time.values[time_idx]
            ).to_pydatetime()
            time_str = MPASDateTimeUtils.format_time_tag(time_end)
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
                plot_type=plot_type,
            )
        else:
            # Default title format matches original mpas_analysis style
            title = f"MPAS Precipitation | PlotType: {plot_type.upper()} | VarType: {var_name.upper()} | Valid Time: {time_str}"

        # Create precipitation map for this time step
        _, _ = self.create_precipitation_map(
            lon,
            lat,
            precip_data.values,
            GeographicBounds(lon_min, lon_max, lat_min, lat_max),
            accum_period=accum_period,
            time_end=time_end,
            data_array=precip_data,
            var_name=var_name,
            dataset=processor.dataset,
            config=config,
            style=PrecipitationRenderStyle(
                title=title,
                plot_type=plot_type,
                colormap=colormap,
                levels=levels,
                grid_resolution=grid_resolution,
                display_units=style.display_units,
                threshold=style.threshold,
            ),
        )

        # Construct output file path with standardized naming convention
        output_path = os.path.join(
            output_dir,
            f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_ptype_{plot_type}",
        )

        # Save plot to files
        self.save_plot(output_path, formats=formats)

        # Close the figure to free memory
        self.close_plot()

        # Return list of created file paths
        return MPASVisualizationStyle.saved_file_paths(output_path, formats)

    def create_batch_precipitation_maps(
        self: "MPASPrecipitationPlotter",
        processor: Any,
        output_dir: str,
        bounds: GeographicBounds,
        var_name: str = "rainnc",
        accum_period: str = "a01h",
        plot_type: str = "scatter",
        grid_resolution: Optional[float] = None,
        formats: List[str] = ["png"],
        time_indices: Optional[List[int]] = None,
        config: Optional[Any] = None,
        style: Optional[PrecipitationMapStyle] = None,
    ) -> List[str]:
        """
        This method orchestrates the batch processing of precipitation maps for multiple time steps in a dataset using the provided MPAS2DProcessor instance. It first validates the processor input to ensure that it is not None and that it contains a loaded dataset. It then extracts the longitude and latitude coordinates from the processor using a dedicated method that handles different processor implementations. The method sets up the time indices for batch processing based on the specified accumulation period and any user-specified time indices, ensuring that there are enough time steps in the dataset to support the accumulation period. If there are not enough time steps, it logs a warning and returns an empty list of created files. If valid time indices are available, it iterates through each time index, calling a helper method to process each time step individually. This helper method extracts the precipitation data for the specific time step, generates a title, creates the precipitation map, and saves it to files with standardized naming conventions. The main method collects all created file paths and returns them as a list at the end of the batch processing. This approach allows for efficient generation of multiple precipitation maps while providing robust error handling and informative logging throughout the process.

        Parameters:
            processor: MPAS2DProcessor instance with loaded dataset.
            output_dir (str): Directory path for saving the output plot files.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            var_name (str): Precipitation variable name in the dataset (e.g., 'rainnc', 'rainc').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h').
            plot_type (str): Rendering method for the overlay ('scatter', 'contour', 'contourf').
            grid_resolution (Optional[float]): Grid resolution in degrees for interpolation if needed. If None, it will be determined adaptively based on map extent and data density.
            formats (List[str]): List of file format extensions for saving the plot (e.g., ['png', 'pdf']).
            time_indices (Optional[List[int]]): User-specified list of time indices to process. If None, all valid time indices starting from the minimum required index for the accumulation period will be used.
            config (Optional[Any]): Configuration object containing additional parameters for the plot. If None, default settings will be used.
            style (Optional[PrecipitationMapStyle]): Appearance and file-naming settings (file_prefix, custom_title_template, colormap, levels). If None, defaults are used.

        Returns:
            List[str]: A list of absolute file paths for all successfully created output files across all processed time steps, with standardized naming conventions that encode the variable type, accumulation type, valid time, and plot type for easy identification. If no valid time indices are available for processing, an empty list will be returned and a warning will be logged.
        """
        if style is None:
            style = PrecipitationMapStyle()

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

        logger.info("Creating precipitation maps for %d time steps", actual_time_steps)
        logger.info(
            "Using accumulation period: %s (%d hours)", accum_period, accum_hours
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through time indices and create maps
        for i, time_idx in enumerate(time_indices_to_process):
            try:
                # Process single time step and collect created file paths
                files = self._process_single_time_step(
                    processor,
                    time_idx,
                    lon,
                    lat,
                    bounds,
                    var_name,
                    accum_period,
                    plot_type,
                    grid_resolution,
                    style,
                    output_dir,
                    formats,
                    config=config,
                )

                # Add created files to the list
                created_files.extend(files)

                # Show progress every 10 steps
                if (i + 1) % 10 == 0:
                    logger.info(
                        "Completed %d/%d maps (time index %d)",
                        i + 1,
                        actual_time_steps,
                        time_idx,
                    )

            except InsufficientLookbackError as e:
                # Not enough preceding time steps for this accumulation: skip quietly rather than counting a failure
                logger.info(
                    "Skipping time index %d: insufficient lookback (%s)", time_idx, e
                )
                self.close_plot()
                continue

            except Exception as e:
                logger.exception(
                    "Error creating map for time index %d: %s", time_idx, e
                )

                # Close the partially built figure so failed steps do not leak open figures
                self.close_plot()
                continue

        logger.info(
            "Batch processing completed. Created %d files in: %s",
            len(created_files),
            output_dir,
        )

        # Return list of created file paths
        return created_files

    def _setup_comparison_subplot(
        self: "MPASPrecipitationPlotter",
        ax: GeoAxes,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        data_crs: ccrs.CRS,
        is_global: bool,
        panel_index: int,
    ) -> None:
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
        # Set extent for the panel. Global panels use set_global() so the full
        # projection domain is shown regardless of projection type or
        # central_longitude (a set_extent() with a lon/lat box collapses for
        # projected CRSs centered near the antimeridian); regional panels set the
        # requested extent directly.
        if is_global:
            ax.set_global()
            if panel_index == 0:
                logger.debug("Using global extent via set_global() for projection view")
        else:
            # For regional extents, use provided bounds directly
            self._apply_map_extent(ax, [lon_min, lon_max, lat_min, lat_max], data_crs)

        # Add geographic features with styling
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black", alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="gray", alpha=0.7)
        self._add_ocean_feature(ax, facecolor="lightblue", alpha=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)

        # Add regional features using existing method
        original_ax = self.ax
        self.ax = ax
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)
        self.ax = original_ax

        # Add gridlines with labels and formatting
        gl = ax.gridlines(
            crs=data_crs,
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

        # Only show top labels for the top row of panels (if there were more than 2 panels in a future extension)
        gl.top_labels = False

        # Control label visibility based on panel index (right labels for right panel, left labels for left panel)
        gl.right_labels = panel_index == 1
        gl.left_labels = panel_index == 0

        # Control label visibility and styling based on panel index
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}

        # Format longitude and latitude labels
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

    def _plot_precipitation_data(
        self: "MPASPrecipitationPlotter",
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
        data_crs: ccrs.CRS,
    ) -> Optional[Any]:
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

        # Re-express longitudes in the extent's frame so dateline-crossing boxes (e.g. 150..210) keep the cells east of 180°
        lon_flat = MPASGeographicUtils.wrap_longitude_to_extent(
            lon_flat, lon_min, lon_max
        )

        # Create mask for valid data points (finite, non-negative, within reasonable bounds, and within map extent)
        valid_mask = (
            np.isfinite(data_flat)
            & (data_flat >= 0)
            & (data_flat < 1e5)
            & (lon_flat >= lon_min)
            & (lon_flat <= lon_max)
            & (lat_flat >= lat_min)
            & (lat_flat <= lat_max)
        )

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
        marker_size = self.calculate_adaptive_marker_size(
            map_extent, len(data_valid), self.figsize
        )

        # Sort by data values to ensure proper color overlay (lower values plotted first)
        sort_indices = np.argsort(data_valid)
        lon_sorted = lon_valid[sort_indices]
        lat_sorted = lat_valid[sort_indices]
        data_sorted = data_valid[sort_indices]

        # Plot scatter and return the scatter object for colorbar reference
        scatter = ax.scatter(
            lon_sorted,
            lat_sorted,
            c=data_sorted,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            alpha=0.9,
            transform=data_crs,
            edgecolors="none",
        )

        # Return scatter for colorbar reference
        return scatter

    @close_figure_on_error
    def create_precipitation_comparison_plot(
        self: "MPASPrecipitationPlotter",
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
        projection: str = "PlateCarree",
        central_longitude: Optional[float] = None,
        central_latitude: Optional[float] = None,
        proj_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Figure, List[Axes]]:
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
            central_longitude (Optional[float]): Explicit central longitude in degrees, overriding the auto-derived value. Defaults to None.
            central_latitude (Optional[float]): Explicit central latitude in degrees for projections that accept it, overriding the auto-derived value. Defaults to None.
            proj_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments passed directly to the cartopy projection constructor. Defaults to None.

        Returns:
            Tuple[Figure, List[Axes]]: A tuple containing the created matplotlib Figure object and a list of the two GeoAxes subplots. The figure contains the side-by-side comparison of the two precipitation datasets with shared map features and a common colorbar. The axes can be further manipulated or saved as needed after the method returns.
        """
        # Setup map projection and coordinate reference system based on provided geographic bounds and projection type
        map_proj, data_crs = self.setup_map_projection(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            projection,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            proj_kwargs=proj_kwargs,
        )

        # Create figure with two subplots sharing the same map projection and size
        self.fig, axes = plt.subplots(
            1,
            2,
            figsize=(self.figsize[0] * 1.8, self.figsize[1]),
            dpi=self.dpi,
            subplot_kw={"projection": map_proj},
        )

        # Ensure axes is a list for consistent processing
        axes = list(axes)

        # Create shared colormap and normalization based on accumulation period
        cmap, color_levels = self.create_precip_colormap(accum_period)

        # Sort and filter color levels to ensure valid contour levels for normalization
        color_levels_sorted = sorted({v for v in color_levels if np.isfinite(v)})

        # Bounds are a zero floor plus the levels; values above the top level take the over colour
        bounds = self._precip_bounds(color_levels_sorted, True)

        # Create normalization for colormap based on bounds and number of colors in the colormap
        norm = self._precip_norm(bounds, cmap)

        # Determine if the extent is global to adjust map features and avoid dateline issues
        is_global = (lon_max - lon_min) >= 359.0 and (lat_max - lat_min) >= 179.0

        # Initialize scatter variable for colorbar reference
        scatter = None

        # Loop through each subplot and corresponding data to setup map features and plot precipitation
        for i, (ax, data, title) in enumerate(
            zip(axes, [precip_data1, precip_data2], [title1, title2])
        ):
            # Setup subplot with map features and extent
            self._setup_comparison_subplot(
                ax, lon_min, lon_max, lat_min, lat_max, data_crs, is_global, i
            )

            # Plot precipitation data and capture scatter result for colorbar reference
            scatter_result = self._plot_precipitation_data(
                ax,
                lon,
                lat,
                data,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                cmap,
                norm,
                data_crs,
            )

            # Store scatter result for colorbar reference if valid
            if scatter_result is not None:
                scatter = scatter_result

            # Set subplot title with consistent styling
            ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

        # Ensure at least one subplot has valid data for the colorbar
        assert (
            scatter is not None
        ), "At least one subplot must have valid data for colorbar"

        # Position colorbar below the two panels, centered, with consistent styling
        cbar = MPASVisualizationStyle.add_colorbar(
            cast(Figure, self.fig),
            cast(Axes, self.ax),
            scatter,
            label="Precipitation [mm]",
            orientation="horizontal",
            extend=MPASVisualizer._derive_extend(
                np.concatenate(
                    [
                        np.asarray(precip_data1, dtype=float).ravel(),
                        np.asarray(precip_data2, dtype=float).ravel(),
                    ]
                ),
                bounds[0],
                bounds[-1],
            ),
            fraction=0.03,
            pad=0.08,
            shrink=0.6,
            fmt=None,
            labelpad=0,
            label_pos="top",
            tick_labelsize=8,
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
        return cast(Figure, self.fig), axes

    def save_plot(
        self: "MPASPrecipitationPlotter",
        output_path: str,
        formats: Union[str, Sequence[str]] = ("png",),
        bbox_inches: str = "tight",
        pad_inches: float = 0.1,
    ) -> None:
        """
        This method saves the current matplotlib figure to disk in multiple specified formats with consistent styling and optimized settings for precipitation maps. It first checks if there is an active figure to save, raising an error if not. It then determines the output directory from the provided output path and ensures that it exists, creating it if necessary. For each specified format, it constructs the full file path with the appropriate extension and defines savefig options including DPI, bounding box settings, and format. For PNG files, it applies low compression settings to optimize saving speed while maintaining quality, especially for large precipitation maps with many points. The method saves the figure with the specified options and prints a confirmation message for each saved file. This approach allows for flexible saving of precipitation plots in various formats while ensuring that the output files are properly organized and named according to the provided output path.

        Parameters:
            output_path (str): Base file path for saving the plot files, with or without the extension; the extension of each format is appended unless the path already ends with it.
            formats (Union[str, Sequence[str]]): A format or list of formats to save the plot in (e.g., 'png' or ['png', 'pdf']); leading dots are stripped and names lowercased. An empty list or unsupported format raises a ValueError.
            bbox_inches (str): Bounding box setting for saving the figure, typically 'tight' to minimize whitespace around the plot.
            pad_inches (float): Padding in inches to add around the figure when bbox_inches is set to 'tight' to prevent clipping of labels and titles.

        Returns:
            None: Saves the current figure to disk in the specified formats with optimized settings for precipitation maps. Prints confirmation messages for each saved file. Raises an error if there is no active figure to save.
        """
        # Check if there is an active figure to save
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")

        # Delegate to the shared saver, which validates and normalizes the formats and never doubles an extension
        MPASVisualizationStyle.save_plot(
            self.fig, output_path, formats, bbox_inches, pad_inches, self.dpi
        )

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

    def _format_ticks_dynamic(
        self: "MPASPrecipitationPlotter", ticks: List[float]
    ) -> List[str]:
        """
        This method formats tick labels dynamically based on the range and magnitude of the tick values for precipitation plots. It delegates to the MPASVisualizationStyle class to apply consistent formatting rules that adjust the number of decimal places, use scientific notation for very large or small values, and ensure that the tick labels are concise and readable. This dynamic formatting helps improve the clarity of the axis labels on precipitation maps, especially when dealing with a wide range of precipitation values that may include very small amounts (e.g., drizzle) or very large amounts (e.g., heavy rainfall). By centralizing the formatting logic in the MPASVisualizationStyle class, it allows for consistent styling across all plots and makes it easier to maintain and update the formatting rules in one place.

        Parameters:
            ticks (List[float]): A list of tick values that need to be formatted for display on the axes. These values can vary widely in magnitude, and the formatting will adjust accordingly to ensure readability.

        Returns:
            List[str]: A list of formatted tick labels corresponding to the input tick values, formatted according to the dynamic rules defined in the MPASVisualizationStyle class. The formatting may include adjustments to decimal places, use of scientific notation, and other styling choices to enhance readability on precipitation plots.
        """
        # Delegate to MPASVisualizationStyle for dynamic tick formatting based on value range and magnitude
        return MPASVisualizationStyle.format_ticks_dynamic(ticks)

    def apply_style(
        self: "MPASPrecipitationPlotter", style_name: str = "default"
    ) -> None:
        """
        This method applies a visualization style to the current plot by setting the axes background color to light gray for better contrast with precipitation colors and the figure background to white for improved contrast and printing. It also delegates to the style manager (if it exists) to apply broader matplotlib settings based on the specified style name. This allows for consistent styling across all precipitation plots while ensuring that the specific color choices enhance the visibility of precipitation data on the map. By centralizing style application in this method, it makes it easier to maintain and update the visual appearance of all plots in a consistent manner, while still allowing for customization through different registered styles in the style manager.

        Parameters:
            style_name (str): The name of the style to apply, which corresponds to a registered style in the style manager. This allows for flexible styling options that can be easily switched by changing the style name. The method will apply the specified style settings to the plot if the style manager is available.

        Returns:
            None
        """
        # Get style manager if it exists
        style_manager = getattr(self, "style_manager", None)

        # Apply style from style manager if available
        if style_manager:
            style_manager.apply_style(style_name)

        # Set axes background to light gray for better contrast with precipitation colors
        if self.ax is not None:
            self.ax.set_facecolor("#f0f0f0")

        # Set figure background to white for better contrast and printing
        if self.fig is not None:
            self.fig.patch.set_facecolor("white")
