#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

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
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.spatial import KDTree
from dataclasses import dataclass
from math import radians, sin, cos, atan2
from typing import Tuple, Optional, List, Dict, Any, Union, cast

from .base_visualizer import MPASVisualizer, close_figure_on_error
from .styling import MPASVisualizationStyle
from ..processing.constants import NETCDF_ENGINE
from ..processing.processors_3d import MPAS3DProcessor
from ..processing.utils_geog import MPASGeographicUtils
from ..processing.utils_unit import UnitConverter
from ..processing.utils_datetime import MPASDateTimeUtils
from ..processing.utils_metadata import MPASFileMetadata
from ..processing.utils_logger import get_logger
from ..processing.utils_validator import DataValidator

logger = get_logger(__name__)

# Potential-temperature variables stay in kelvin (the standard unit) rather than being converted to degrees Celsius
_POTENTIAL_TEMPERATURE_VARS = ("theta", "theta_m", "theta_base", "theta_rho")

# Explicit wind-component variable names used to choose symmetric diverging contour levels
_WIND_VARS = (
    "u",
    "v",
    "w",
    "uzonal",
    "umeridional",
    "ureconstructzonal",
    "ureconstructmeridional",
    "u10",
    "v10",
)

_PATH_NEIGHBOR_CACHE: Dict[Any, np.ndarray] = {}
_PATH_NEIGHBOR_CACHE_MAX = 8


def _path_neighbor_fingerprint(
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    path_lons: np.ndarray,
    path_lats: np.ndarray,
    k: int,
) -> Tuple[Any, ...]:
    """
    This helper builds a cheap, collision-resistant fingerprint identifying a (grid, transect) pair for the path-neighbour cache. The grid is summarised by its shape, endpoint coordinates and coordinate sums, and the transect by its length, endpoint coordinates and coordinate sums, together with the neighbour count k; identical batches (same model grid, same transect) therefore map to the same key while different grids or transects do not. The sums are O(n_cells) but negligible beside the KDTree build they let us skip, and they guard against distinct grids that happen to share endpoints.

    Parameters:
        grid_lons (np.ndarray): Flattened grid longitudes in degrees.
        grid_lats (np.ndarray): Flattened grid latitudes in degrees.
        path_lons (np.ndarray): Flattened transect longitudes in degrees.
        path_lats (np.ndarray): Flattened transect latitudes in degrees.
        k (int): Number of nearest neighbours retained per path point.

    Returns:
        Tuple[Any, ...]: A hashable fingerprint suitable as a dict key.
    """
    return (
        grid_lons.shape,
        float(grid_lons[0]),
        float(grid_lons[-1]),
        float(grid_lats[0]),
        float(grid_lats[-1]),
        float(grid_lons.sum()),
        float(grid_lats.sum()),
        path_lons.shape,
        float(path_lons[0]),
        float(path_lons[-1]),
        float(path_lats[0]),
        float(path_lats[-1]),
        float(path_lons.sum()),
        float(path_lats.sum()),
        int(k),
    )


@dataclass
class CrossSectionStyle:
    """Optional appearance and file-naming settings for batch cross-section plots, grouping contour levels, colormap, colorbar extend direction, plot type, output filename prefix, and optional colour limits (clim_min/clim_max, in plotted units, bounding the automatic levels when levels is None) into a single value object."""

    levels: Optional[np.ndarray] = None
    colormap: Optional[Union[str, mcolors.Colormap]] = None
    extend: str = "both"
    plot_type: str = "contourf"
    file_prefix: str = "mpas_cross_section"
    clim_min: Optional[float] = None
    clim_max: Optional[float] = None


class MPASVerticalCrossSectionPlotter(MPASVisualizer):
    """Specialized plotter for creating vertical cross-section visualizations of 3D MPAS atmospheric data along user-defined transects through the atmosphere."""

    def __init__(
        self: "MPASVerticalCrossSectionPlotter",
        figsize: Tuple[float, float] = (10, 12),
        dpi: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        This constructor initializes the MPASVerticalCrossSectionPlotter instance with specified figure dimensions and resolution. It calls the parent class constructor to set up the base visualization environment, allowing for consistent styling and branding across different plot types. The figsize parameter determines the size of the output figure in inches, while the dpi parameter controls the resolution of the saved figure for high-quality output. This setup ensures that the vertical cross-section plots are rendered with appropriate dimensions and clarity for analysis and presentation purposes.

        Parameters:
            figsize (Tuple[float, float]): Figure size in inches (width, height) for the cross-section plot (default: (10, 12)).
            dpi (int): Dots per inch resolution for saved figure output (default: 100).
            verbose (bool): Enable verbose informational output, forwarded to MPASVisualizer so every plotter constructor accepts the same keyword (default: True).

        Returns:
            None
        """
        super().__init__(figsize, dpi, verbose)
        self._phase_timings: Dict[str, float] = {}

    def _validate_cross_section_inputs(
        self: "MPASVerticalCrossSectionPlotter", mpas_3d_processor: Any, var_name: str
    ) -> None:
        """
        This internal method validates the inputs for creating a vertical cross-section plot. It checks that the provided mpas_3d_processor is an instance of MPAS3DProcessor and that it has loaded data. It also verifies that the specified var_name exists in the dataset and is a 3D atmospheric variable with appropriate vertical dimensions. If any of these conditions are not met, it raises a ValueError with an informative message to guide the user in correcting the input. This validation step ensures that the subsequent processing and plotting steps have the necessary data and context to generate a meaningful vertical cross-section visualization.

        Parameters:
            mpas_3d_processor (Any): The processor object that should be an instance of MPAS3DProcessor with loaded dataset.
            var_name (str): The name of the variable to be plotted in the cross-section, which must exist in the dataset and have appropriate vertical dimensions.

        Returns:
            None
        """
        if not isinstance(mpas_3d_processor, MPAS3DProcessor):
            raise ValueError("mpas_3d_processor must be an instance of MPAS3DProcessor")

        if mpas_3d_processor.dataset is None:
            raise ValueError(
                "MPAS3DProcessor must have loaded data. Call load_3d_data() first."
            )

        if var_name not in mpas_3d_processor.dataset.data_vars:
            available_vars = list(mpas_3d_processor.dataset.data_vars.keys())
            raise ValueError(
                f"Variable '{var_name}' not found. Available variables: {available_vars[:10]}..."
            )

        var_dims = mpas_3d_processor.dataset[var_name].sizes

        if "nVertLevels" not in var_dims and "nVertLevelsP1" not in var_dims:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")

        # The transect is sampled at model cells, so only cell-centred variables can be used; edge or vertex fields would silently reuse their indices as cell indices
        spatial_dims = [
            str(d)
            for d in var_dims
            if d not in ("Time", "time", "nVertLevels", "nVertLevelsP1")
        ]

        if spatial_dims and "nCells" not in spatial_dims:
            hint = (
                " Use a cell-centred variable such as 'uReconstructZonal' / 'uReconstructMeridional' instead of the edge-normal wind 'u'."
                if "nEdges" in spatial_dims
                else ""
            )
            raise ValueError(
                f"Variable '{var_name}' lives on {spatial_dims} rather than nCells; vertical cross-sections require a cell-centred variable.{hint}"
            )

    def _convert_and_clip_data(
        self: "MPASVerticalCrossSectionPlotter",
        data_values: Union[np.ndarray, xr.DataArray, float],
        var_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        This internal method performs unit conversion and physical clipping on the data values for the specified variable. It retrieves the original units from the variable metadata and determines the appropriate display units using the UnitConverter utility. If a conversion is needed, it applies the conversion to the data values and updates the metadata accordingly (potential temperature is deliberately kept in kelvin). Additionally, if the variable is identified as a moisture variable (e.g., specific humidity, mixing ratio), it checks for any negative values in the data, which are physically invalid, and clips them to zero while issuing a warning with the count of negative values and their range. This method ensures that the data values are in the correct units for display and that any physically unrealistic values are handled appropriately before plotting. Accepts np.ndarray, xr.DataArray, or float as input for data_values.

        Parameters:
            data_values (Union[np.ndarray, xr.DataArray, float]): The data values to be processed.
            var_name (str): The name of the variable being processed.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The processed data values and associated metadata.
        """
        if isinstance(data_values, xr.DataArray):
            data_values = data_values.values
        elif not isinstance(data_values, np.ndarray):
            data_values = np.asarray(data_values)
        try:
            metadata = MPASFileMetadata.get_variable_metadata(var_name)
            original_units = metadata.get("units", "")
            display_units = UnitConverter.get_display_units(var_name, original_units)

            # Keep potential temperature in kelvin; only the other temperature variables are shown in degrees Celsius
            if var_name.lower() in _POTENTIAL_TEMPERATURE_VARS:
                display_units = original_units

            if original_units != display_units and original_units:
                logger.debug(
                    "Converting %s from %s to %s",
                    var_name,
                    original_units,
                    display_units,
                )
                data_values = UnitConverter.convert_units(
                    data_values, original_units, display_units
                )
                metadata["units"] = display_units
                logger.debug(
                    "Data range after conversion: %.4f to %.4f %s",
                    float(np.nanmin(data_values)),
                    float(np.nanmax(data_values)),
                    display_units,
                )
            else:
                logger.debug(
                    "No unit conversion needed for %s (units: %s)",
                    var_name,
                    original_units,
                )
        except Exception as e:
            logger.warning("Unit conversion failed for %s: %s", var_name, e)
            metadata = {"units": "", "long_name": var_name}

        moisture_vars = [
            "q2",
            "qv",
            "qc",
            "qr",
            "qi",
            "qs",
            "qg",
            "qv2m",
            "humidity",
            "mixing_ratio",
        ]

        if any(mv in var_name.lower() for mv in moisture_vars):
            n_negative = int(np.sum(data_values < 0))
            if n_negative > 0:
                logger.warning(
                    "Found %s negative %s values (min: %.4f). Clipping to 0 (physically invalid).",
                    f"{n_negative:,}",
                    var_name,
                    float(np.nanmin(data_values)),
                )
                data_values = np.clip(data_values, 0, None)

        return np.asarray(data_values), metadata

    def _resolve_vertical_display(
        self: "MPASVerticalCrossSectionPlotter",
        vertical_coords: np.ndarray,
        vertical_coord_type: str,
        desired_display: str,
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
    ) -> Tuple[np.ndarray, str]:
        """
        This internal method resolves the vertical coordinate values for display based on the desired vertical coordinate type. It accepts the original vertical coordinates and their type, along with the desired display type (e.g., 'height', 'pressure', 'modlev'). Depending on the desired display, it performs the necessary conversions using the MPAS3DProcessor methods to convert model levels to height or pressure as needed. If the desired display is 'pressure' but the vertical coordinates contain non-positive or non-finite values, it issues a warning and falls back to displaying model levels. The method returns the vertical coordinates formatted for display and a string indicating the type of vertical coordinate being displayed.

        Parameters:
            vertical_coords (np.ndarray): The original vertical coordinate values extracted from the dataset.
            vertical_coord_type (str): The type of the original vertical coordinates (e.g., 'pressure', 'height', 'modlev').
            desired_display (str): The desired vertical coordinate type for display ('pressure', 'height', 'modlev').
            mpas_3d_processor ('MPAS3DProcessor'): The processor instance used for any necessary conversions.
            time_index (int): The time index for selecting temporal data if needed for conversions.

        Returns:
            Tuple[np.ndarray, str]: The vertical coordinates formatted for display and the type of vertical coordinate being displayed.
        """
        logger.debug(
            "_resolve_vertical_display: desired_display='%s', vertical_coord_type='%s'",
            desired_display,
            vertical_coord_type,
        )
        logger.debug(
            "_resolve_vertical_display: vertical_coords shape=%s, min=%.4f, max=%.4f",
            np.asarray(vertical_coords).shape,
            float(np.nanmin(vertical_coords)),
            float(np.nanmax(vertical_coords)),
        )

        if desired_display == "height":
            result_coords, result_type = self._convert_vertical_to_height(
                vertical_coords, vertical_coord_type, mpas_3d_processor, time_index
            )
            logger.debug(
                "_resolve_vertical_display: after height conversion: type='%s', min=%.4f, max=%.4f",
                result_type,
                float(np.nanmin(result_coords)),
                float(np.nanmax(result_coords)),
            )
            return result_coords, result_type

        if desired_display == "pressure":
            try:
                pressure_values: np.ndarray = vertical_coords.astype(float).copy()
            except Exception:
                pressure_values = np.asarray(vertical_coords, dtype=float)
            if (
                not np.all(np.isfinite(pressure_values))
                or np.nanmin(pressure_values) <= 0
            ):
                logger.warning(
                    "Vertical coordinates contain non-positive or non-finite values; "
                    "cannot display as pressure. Falling back to model levels."
                )
                return np.arange(len(vertical_coords), dtype=float), "modlev"
            return (
                pressure_values / 100.0
                if np.nanmax(pressure_values) > 10000
                else pressure_values
            ), "pressure_hPa"

        if desired_display == "modlev":
            # Model levels are indices (0 at the lowest level), never the pressure or height values the levels were resolved from
            return np.arange(len(vertical_coords), dtype=float), "modlev"

        return self._convert_vertical_to_height(
            vertical_coords, vertical_coord_type, mpas_3d_processor, time_index
        )

    @staticmethod
    def _filter_columns_below(
        vertical_display: np.ndarray, data_values: np.ndarray, max_height: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This static helper applies the maximum-height filter to a per-column (two-dimensional) vertical coordinate. Rows are kept up to and including the first row that lies above the limit in every column, so each column still has a level on or above the limit and the filled field reaches the top of the axes instead of stopping at, say, 11.6 km under a 12 km limit; the axes limits then clip the drawing at the limit. Within each column, data on levels beyond the first level above the limit are masked (set to NaN) rather than rows being dropped for all columns at once. If no level lies at or below the limit the unfiltered arrays are returned after a warning.

        Parameters:
            vertical_display (np.ndarray): Vertical coordinate in km with shape (levels, points).
            data_values (np.ndarray): Data with the same shape.
            max_height (float): Maximum height in km to keep.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The filtered vertical coordinate and data (the data is a float copy with the masked entries set to NaN).
        """
        limit = float(max_height)

        with np.errstate(invalid="ignore"):
            below_any = np.nanmin(vertical_display, axis=1) <= limit

        if not np.any(below_any):
            logger.warning(
                "No vertical levels are below the requested max_height; showing full range"
            )
            return vertical_display, data_values

        # Keep every row up to one past the last row that is below the limit in some column (the bracketing row)
        last = int(np.max(np.nonzero(below_any)[0]))
        stop = min(last + 2, vertical_display.shape[0])
        vertical_kept = vertical_display[:stop, :]
        data_kept = np.array(data_values[:stop, :], dtype=float, copy=True)

        # Mask, per column, everything beyond the first level above the limit
        with np.errstate(invalid="ignore"):
            above = vertical_kept > limit

        prior_above = np.vstack([np.zeros((1, above.shape[1]), dtype=bool), above[:-1]])
        data_kept[above & prior_above] = np.nan

        return vertical_kept, data_kept

    def _apply_max_height_filter(
        self: "MPASVerticalCrossSectionPlotter",
        vertical_display: Union[np.ndarray, xr.DataArray, float],
        vertical_coord_display: str,
        data_values: Union[np.ndarray, xr.DataArray, float],
        max_height: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal method applies a maximum height filter to the vertical coordinates and corresponding data values for display. It checks if the vertical coordinate being displayed is in height units (e.g., 'height_km') and if a max_height value is provided. If so, it creates a boolean mask to filter out any vertical levels that exceed the specified maximum height (for a per-column two-dimensional vertical coordinate a level is kept when it lies below the limit in at least one column). The method then returns the filtered vertical coordinates and data values for plotting. If no filtering is applied (e.g., if the vertical coordinate is not in height units or if max_height is None), it returns the original vertical coordinates and data values without modification.

        Parameters:
            vertical_display (Union[np.ndarray, xr.DataArray, float]): The vertical coordinate values for display.
            vertical_coord_display (str): The type of vertical coordinate being displayed.
            data_values (Union[np.ndarray, xr.DataArray, float]): The data values corresponding to the vertical coordinates.
            max_height (Optional[float]): The maximum height (km) to display.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The filtered vertical coordinates and data values.
        """
        if isinstance(vertical_display, xr.DataArray):
            vertical_display = vertical_display.values
        elif not isinstance(vertical_display, np.ndarray):
            vertical_display = np.asarray(vertical_display)

        if isinstance(data_values, xr.DataArray):
            data_values = data_values.values
        elif not isinstance(data_values, np.ndarray):
            data_values = np.asarray(data_values)

        if vertical_coord_display != "height_km" or max_height is None:
            return vertical_display, data_values
        try:
            if np.asarray(vertical_display).ndim == 2:
                return self._filter_columns_below(
                    np.asarray(vertical_display), np.asarray(data_values), max_height
                )

            mask = np.asarray(vertical_display) <= float(max_height)
            if np.any(mask):
                return (
                    np.asarray(vertical_display)[mask],
                    np.asarray(data_values)[mask, :],
                )
            logger.warning(
                "No vertical levels are below the requested max_height; showing full range"
            )
        except Exception:
            pass

        return vertical_display, data_values

    @staticmethod
    def _visible_data_for_levels(
        vertical_display: np.ndarray,
        vertical_coord_display: str,
        data_values: np.ndarray,
        max_height: Optional[float],
    ) -> np.ndarray:
        """
        This static helper returns the data that actually lies inside the visible vertical range, so automatic contour levels are derived from what is drawn rather than from the whole column. For a height display the visible range is below max_height (km); for a pressure display it is the pressure above the barometric equivalent of max_height (the same conversion the pressure axis limit uses). Hidden values are set to NaN in a copy. Model-level displays have no height axis, so the data are returned unchanged (the caller warns that max_height does not apply), as they are when max_height is None or nothing would remain visible.

        Parameters:
            vertical_display (np.ndarray): Vertical coordinate, one-dimensional (levels) or two-dimensional (levels, points).
            vertical_coord_display (str): Displayed vertical coordinate type ('height_km', 'pressure_hPa', 'modlev', ...).
            data_values (np.ndarray): Data with shape (levels, points).
            max_height (Optional[float]): Maximum height in km shown on the vertical axis, or None.

        Returns:
            np.ndarray: The data with values outside the visible range masked as NaN, or the unchanged data.
        """
        if max_height is None:
            return data_values

        vertical = np.asarray(vertical_display, dtype=float)

        if vertical.ndim == 1:
            vertical = vertical[:, None]

        with np.errstate(invalid="ignore"):
            if vertical_coord_display == "height_km":
                visible = vertical <= float(max_height)
            elif vertical_coord_display == "pressure_hPa":
                visible = vertical >= 1013.25 * np.exp(-float(max_height) / 8.4)
            else:
                return data_values

        visible = np.broadcast_to(visible, data_values.shape)

        if not np.any(visible):
            return data_values

        return np.where(visible, np.asarray(data_values, dtype=float), np.nan)

    @staticmethod
    def _apply_clim_to_levels(
        levels: np.ndarray, clim_min: Optional[float], clim_max: Optional[float]
    ) -> np.ndarray:
        """
        This static helper bounds a set of contour levels to the requested colour limits: levels outside the limits are dropped and the limits themselves become the first and last level, so the colours span exactly [clim_min, clim_max] (values beyond them fall in the extend regions).

        Parameters:
            levels (np.ndarray): Contour levels in ascending order.
            clim_min (Optional[float]): Lower colour limit in plotted units, or None.
            clim_max (Optional[float]): Upper colour limit in plotted units, or None.

        Returns:
            np.ndarray: The bounded levels.
        """
        bounded = np.asarray(levels, dtype=float)

        if clim_min is not None:
            bounded = np.concatenate([[float(clim_min)], bounded[bounded > clim_min]])

        if clim_max is not None:
            bounded = np.concatenate([bounded[bounded < clim_max], [float(clim_max)]])

        return bounded

    @staticmethod
    def _ensure_contourable_levels(levels: Any) -> np.ndarray:
        """
        This static helper guarantees the contour levels can be drawn: contouring needs at least two levels, so a single level (as produced for a constant field) is widened into a small band around it, and an empty set becomes a band around zero.

        Parameters:
            levels (Any): Contour levels as any array-like.

        Returns:
            np.ndarray: Array of at least two contour levels.
        """
        levels_arr = np.asarray(levels)

        if levels_arr.size < 2:
            centre = float(levels_arr.ravel()[0]) if levels_arr.size else 0.0
            half_width = max(abs(centre) * 1e-3, 0.5)
            return np.array([centre - half_width, centre + half_width])

        return levels_arr

    @staticmethod
    def _validate_clim_range(
        clim_min: Optional[float], clim_max: Optional[float]
    ) -> None:
        """
        This helper method checks that the colour limits, when both are given, form a valid range and raises a ValueError when the lower limit is not smaller than the upper limit.

        Parameters:
            clim_min (Optional[float]): Lower colour limit in plotted units, or None.
            clim_max (Optional[float]): Upper colour limit in plotted units, or None.

        Returns:
            None
        """
        if (
            clim_min is not None
            and clim_max is not None
            and float(clim_min) >= float(clim_max)
        ):
            raise ValueError(
                f"clim_min ({clim_min}) must be smaller than clim_max ({clim_max})"
            )

    def _variable_style_defaults(
        self: "MPASVerticalCrossSectionPlotter",
        var_name: str,
        data_values: np.ndarray,
        colormap: Optional[Union[str, mcolors.Colormap]],
        levels: Optional[np.ndarray],
    ) -> Tuple[Any, Any]:
        """
        This helper method fills in the colormap and contour levels that the caller left unset, using the variable-specific style when it is available. If the style lookup fails, the colormap falls back to 'viridis' and the levels to the automatically determined default levels.

        Parameters:
            var_name (str): The name of the variable being plotted, used to determine styling.
            data_values (np.ndarray): The data values for the variable, used to determine default contour levels.
            colormap (Optional[Union[str, mcolors.Colormap]]): The user-specified colormap, or None.
            levels (Optional[np.ndarray]): The user-specified contour levels, or None.

        Returns:
            Tuple[Any, Any]: The colormap and levels, with every value that was None replaced by its default.
        """
        try:
            dummy_data = xr.DataArray(
                data_values, dims=["level", "distance"], name=var_name
            )
            style = MPASVisualizationStyle.get_variable_style(var_name, dummy_data)

            if colormap is None:
                colormap = style.get("colormap", "viridis")

            if levels is None:
                levels = style.get(
                    "levels", self._get_default_levels(data_values, var_name)
                )
        except Exception:
            if colormap is None:
                colormap = "viridis"

            if levels is None:
                levels = self._get_default_levels(data_values, var_name)

        return colormap, levels

    def _resolve_plot_style(
        self: "MPASVerticalCrossSectionPlotter",
        var_name: str,
        data_values: Union[np.ndarray, xr.DataArray, float],
        colormap: Optional[Union[str, mcolors.Colormap]],
        levels: Optional[np.ndarray],
        clim_min: Optional[float] = None,
        clim_max: Optional[float] = None,
    ) -> Tuple[Any, np.ndarray]:
        """
        This internal method resolves the plot style for the cross-section visualization by determining the appropriate colormap and contour levels based on the variable name and data values. It first checks if the data values are in a compatible format (np.ndarray or xr.DataArray) and converts them to np.ndarray if necessary. If both colormap and levels are provided, it returns them directly. Otherwise, it attempts to retrieve variable-specific styling information from the MPASVisualizationStyle class using a dummy DataArray with the same shape as the data values. If styling information is available, it uses the specified colormap and levels; if not, it falls back to default values (e.g., 'viridis' for colormap and automatically determined levels based on data range). This method ensures that the plot is styled appropriately for the variable being visualized while allowing for user overrides when desired.

        Parameters:
            var_name (str): The name of the variable being plotted, used to determine styling.
            data_values (Union[np.ndarray, xr.DataArray, float]): The data values for the variable, used to determine contour levels if not provided.
            colormap (Optional[Union[str, mcolors.Colormap]]): An optional colormap specified by the user; if None, a default or variable-specific colormap will be used.
            levels (Optional[np.ndarray]): An optional array of contour levels specified by the user; if None, levels will be automatically determined based on data range and variable type.
            clim_min (Optional[float]): Lower colour limit in plotted units; bounds the automatic levels (ignored when levels is given).
            clim_max (Optional[float]): Upper colour limit in plotted units; bounds the automatic levels (ignored when levels is given).

        Returns:
            Tuple[Any, np.ndarray]: The resolved colormap and contour levels for plotting the cross-section.
        """
        if isinstance(data_values, xr.DataArray):
            data_values = data_values.values
        elif not isinstance(data_values, np.ndarray):
            data_values = np.asarray(data_values)

        self._validate_clim_range(clim_min, clim_max)

        if colormap is not None and levels is not None:
            return colormap, self._ensure_contourable_levels(levels)

        levels_given = levels is not None

        if not levels_given and (clim_min is not None or clim_max is not None):
            data_values = np.clip(
                data_values,
                None if clim_min is None else float(clim_min),
                None if clim_max is None else float(clim_max),
            )

        colormap, levels = self._variable_style_defaults(
            var_name, data_values, colormap, levels
        )

        if levels is None:
            levels = self._get_default_levels(data_values, var_name)

        if not levels_given and (clim_min is not None or clim_max is not None):
            levels = self._apply_clim_to_levels(
                np.asarray(levels, dtype=float), clim_min, clim_max
            )

        levels = self._ensure_contourable_levels(levels)

        return colormap, levels

    def _render_cross_section_plot(
        self: "MPASVerticalCrossSectionPlotter",
        X: np.ndarray,
        Y: np.ndarray,
        data_values: np.ndarray,
        colormap: Any,
        levels: np.ndarray,
        extend: str,
        plot_type: str,
        metadata: Dict[str, Any],
        var_name: str,
        **kwargs: Any,
    ) -> None:
        """
        This internal method renders the cross-section plot using the specified plot type (filled contour, contour, or pcolormesh) with the provided data values, colormap, and contour levels. It first checks that the axes have been initialized before attempting to plot. Depending on the plot type, it uses the appropriate matplotlib function to create the visualization. For filled contours, it also adds contour lines and labels for clarity. If a colorbar is applicable (for filled contours and pcolormesh), it adds a colorbar with a label derived from the variable metadata. The method includes error handling for unknown plot types and ensures that the plot is rendered with the specified styling options.

        Parameters:
            X (np.ndarray): The 2D array of longitude coordinates for the cross-section plot.
            Y (np.ndarray): The 2D array of vertical coordinate values for the cross-section plot.
            data_values (np.ndarray): The 2D array of data values to be plotted in the cross-section.
            colormap (Any): The colormap to use for plotting the data values.
            levels (np.ndarray): The array of contour levels to use for plotting.
            extend (str): The contour extend option for out-of-range values when using filled contours ('neither', 'both', 'min', 'max').
            plot_type (str): The type of plot to create ('contourf', 'contour', 'pcolormesh').
            metadata (Dict[str, Any]): Metadata dictionary containing information about the variable, used for labeling the colorbar.
            var_name (str): The name of the variable being plotted, used for labeling and styling purposes.
            **kwargs: Additional keyword arguments passed to the underlying matplotlib plotting functions for further customization.

        Returns:
            None
        """
        assert self.ax is not None, "Axes must be initialized before rendering the plot"

        if plot_type == "contourf":
            contour_set = self.ax.contourf(
                X, Y, data_values, levels=levels, cmap=colormap, extend=extend, **kwargs
            )
            contour_lines = self.ax.contour(
                X,
                Y,
                data_values,
                levels=levels,
                colors="black",
                linewidths=0.5,
                alpha=0.6,
            )
            self.ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")
        elif plot_type == "contour":
            contour_set = self.ax.contour(
                X, Y, data_values, levels=levels, cmap=colormap, **kwargs
            )
            self.ax.clabel(contour_set, inline=True, fontsize=8, fmt="%.1f")
        elif plot_type == "pcolormesh":
            # Colour by the same level boundaries (and extend regions) the filled-contour plot uses
            if "norm" not in kwargs:
                cmap_obj = (
                    plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
                )
                kwargs["norm"] = mcolors.BoundaryNorm(
                    np.asarray(levels, dtype=float), cmap_obj.N, extend=extend
                )

            contour_set = self.ax.pcolormesh(X, Y, data_values, cmap=colormap, **kwargs)
        else:
            raise ValueError(
                f"Unknown plot_type: {plot_type}. Use 'contourf', 'contour', or 'pcolormesh'"
            )

        if plot_type in ["contourf", "contour", "pcolormesh"]:
            label = (
                f"{metadata.get('long_name', var_name)} [{metadata.get('units', '')}]"
                if metadata.get("units", "")
                else metadata.get("long_name", var_name)
            )

            assert (
                self.fig is not None
            ), "Figure must be initialized before adding a colorbar"

            cbar = MPASVisualizationStyle.add_colorbar(
                self.fig,
                self.ax,
                contour_set,
                label=label,
                orientation="vertical",
                fraction=0.03,
                pad=0.05,
                shrink=0.8,
                fmt=None,
                labelpad=4,
                label_pos="right",
                tick_labelsize=10,
            )

            # Keep the tick labels on the outer (right) side so they never overlap the axes frame
            if cbar is not None:
                cbar.ax.yaxis.set_ticks_position("right")
                cbar.ax.yaxis.set_label_position("right")

    def _generate_cross_section_title(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        time_index: int,
        title: Optional[str],
    ) -> str:
        """
        This internal method generates a title for the vertical cross-section plot based on the variable name and time information. If a custom title is provided, it returns that title directly. Otherwise, it attempts to retrieve a time string from the MPAS3DProcessor using the specified time index and formats a default title that includes the variable name and valid time. If there is an issue retrieving the time information, it falls back to a simpler title that only includes the variable name. This method ensures that the plot has an informative title that provides context about the variable being visualized and the time of the data, while also allowing for user customization when desired.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): The processor instance used to retrieve time information for the title.
            var_name (str): The name of the variable being plotted, used in the title.
            time_index (int): The time index for selecting temporal data from the dataset, used to retrieve valid time information for the title.
            title (Optional[str]): A custom title provided by the user; if None, a default title will be generated based on variable name and time information.

        Returns:
            str: The generated title for the vertical cross-section plot.
        """
        if title is not None:
            return title

        try:
            time_str = self._get_time_string(mpas_3d_processor, time_index)
            return f"Vertical Cross-Section: {var_name} | Valid Time: {time_str}"
        except Exception:
            return f"Vertical Cross-Section: {var_name}"

    def _save_cross_section(
        self: "MPASVerticalCrossSectionPlotter", save_path: str
    ) -> None:
        """
        This internal method saves the generated vertical cross-section figure to the specified file path. It checks that the figure has been initialized before attempting to save. The method determines the appropriate save parameters based on the file extension (e.g., using a lower compression level for PNG files) and saves the figure with the specified resolution and bounding box settings. After saving, it prints a confirmation message with the path to the saved figure.

        Parameters:
            save_path (str): The file path where the figure should be saved, including the file extension (e.g., 'cross_section.png').

        Returns:
            None
        """
        save_kwargs: Dict[str, Any] = {"dpi": self.dpi, "bbox_inches": "tight"}

        if save_path.lower().endswith(".png"):
            save_kwargs["pil_kwargs"] = {"compress_level": 1}

        assert self.fig is not None, "Figure must be initialized before saving"

        self.fig.savefig(save_path, **save_kwargs)
        logger.info("Vertical cross-section saved to: %s", save_path)

    @staticmethod
    def _normalize_time_index(
        mpas_3d_processor: "MPAS3DProcessor", var_name: str, time_index: int
    ) -> int:
        """
        This static helper validates the requested time index once, up front, and returns the equivalent non-negative index. Python-style negative indices are accepted; an index outside [-n, n) raises an IndexError stating the valid range (instead of being swallowed by the per-level extraction and failing later). Variables without a time dimension are returned unchanged.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): Processor whose dataset holds the variable.
            var_name (str): Name of the plotted variable.
            time_index (int): Requested time index.

        Returns:
            int: The validated non-negative time index.
        """
        dataset = mpas_3d_processor.dataset

        if dataset is None or not any(
            d in dataset[var_name].sizes for d in ("Time", "time")
        ):
            return time_index

        return int(MPASDateTimeUtils.validate_time_parameters(dataset, time_index)[1])

    @close_figure_on_error
    def create_vertical_cross_section(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: Any,
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        time_index: int = 0,
        vertical_coord: str = "pressure",
        display_vertical: Optional[str] = None,
        num_points: int = 100,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        max_height: Optional[float] = None,
        style: Optional[CrossSectionStyle] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        This method creates a vertical cross-section plot for a specified 3D atmospheric variable along a user-defined transect between two geographic points. It first validates the input processor and variable, then generates the cross-section data by interpolating the 3D variable along the great-circle path defined by the start and end points. The method handles unit conversions, applies variable-specific styling, and renders the plot using the specified plot type (filled contour, contour, or pcolormesh). It formats the axes based on the vertical coordinate system, adds titles and colorbars, and includes options for saving the figure. The method returns the matplotlib figure and axes objects containing the generated cross-section plot for further customization or display.

        Parameters:
            mpas_3d_processor (Any): MPAS3DProcessor instance with loaded dataset and processing capabilities.
            var_name (str): Name of the 3D variable to plot in the cross-section.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            time_index (int): Time index for selecting temporal data from the dataset (default: 0).
            vertical_coord (str): Type of vertical coordinate to use ('pressure', 'height', 'modlev') for data extraction (default: 'pressure').
            display_vertical (Optional[str]): Desired vertical coordinate type for display ('pressure', 'height', 'modlev'); if None, uses vertical_coord type (default: None).
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting (default: 100).
            save_path (Optional[str]): File path to save the generated figure; if None, the figure is not saved to disk (default: None).
            title (Optional[str]): Title for the plot; if None, a default title is generated based on variable name and time information (default: None).
            max_height (Optional[float]): Maximum height in kilometers to display on the vertical axis; if None, full height range is shown based on data and vertical coordinate system (default: None). Automatic contour levels are computed from the visible data only; it does not apply to a model-level display (a warning is logged).
            style (Optional[CrossSectionStyle]): Appearance settings (levels, colormap, extend, plot_type, clim_min, clim_max). If None, defaults are used.
            **kwargs: Additional keyword arguments passed to the underlying matplotlib plotting functions for further customization. May also include the advanced batch knob ``precomputed_levels`` (Optional[Tuple[np.ndarray, str]]): an already-resolved (vertical_levels, vertical_coord_type) pair to reuse for this plot. Resolving pressure levels reads and reduces the FULL 3D pressure field, so batch callers resolve once per batch and pass the result here; every frame then shares a consistent vertical axis and skips that read. If absent, levels are resolved from the dataset for this time index.

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects containing the generated vertical cross-section plot.
        """
        if style is None:
            style = CrossSectionStyle()

        precomputed_levels: Optional[Tuple[np.ndarray, str]] = kwargs.pop(
            "precomputed_levels", None
        )

        levels = style.levels
        colormap = style.colormap
        extend = style.extend
        plot_type = style.plot_type

        self._validate_cross_section_inputs(mpas_3d_processor, var_name)

        if num_points < 2:
            raise ValueError(f"num_points must be at least 2, got {num_points}")

        time_index = self._normalize_time_index(mpas_3d_processor, var_name, time_index)

        logger.info("Creating vertical cross-section for %s", var_name)
        logger.info(
            "Cross-section from (%.2f, %.2f) to (%.2f, %.2f)",
            start_point[0],
            start_point[1],
            end_point[0],
            end_point[1],
        )

        self._phase_timings = {
            "data_read": 0.0,
            "interpolate": 0.0,
            "render": 0.0,
            "save": 0.0,
        }

        DataValidator.enforce_size_limits(
            num_points=num_points, context="building the cross-section transect"
        )

        cross_section_data = self._generate_cross_section_data(
            mpas_3d_processor,
            var_name,
            start_point,
            end_point,
            time_index,
            vertical_coord,
            num_points,
            precomputed_levels=precomputed_levels,
        )

        render_start = time.perf_counter()

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)

        longitudes = cross_section_data["longitudes"]
        vertical_coords = cross_section_data["vertical_coords"]
        vertical_coord_type = cross_section_data.get(
            "vertical_coord_type", vertical_coord
        )
        data_values = cross_section_data["data_values"]

        data_values, metadata = self._convert_and_clip_data(data_values, var_name)

        desired_display = (
            display_vertical if display_vertical is not None else vertical_coord
        )
        vertical_display, vertical_coord_display = self._resolve_vertical_display(
            vertical_coords,
            vertical_coord_type,
            desired_display,
            mpas_3d_processor,
            time_index,
        )

        # Replace the 1-D vertical axis with each column's own pressure or height when it can be sampled at the path cells
        column_vertical = self._resolve_column_vertical(
            mpas_3d_processor,
            time_index,
            cross_section_data.get("path_cells"),
            data_values.shape[0],
            vertical_coord_display,
        )

        if column_vertical is not None and column_vertical.shape == data_values.shape:
            vertical_display = column_vertical

        vertical_display, data_values = self._apply_max_height_filter(
            vertical_display, vertical_coord_display, data_values, max_height
        )

        X: np.ndarray
        Y: np.ndarray
        distances = cross_section_data.get("distances")

        # Use along-path distance as the horizontal axis (longitude collapses for N-S sections and wraps at the dateline); fall back to longitude only for a degenerate zero-length path or data without distances
        use_distance = (
            distances is not None
            and len(distances) == len(longitudes)
            and float(np.nanmax(distances)) > 0.0
        )
        x_values = np.asarray(distances if use_distance else longitudes, dtype=float)

        if np.ndim(vertical_display) == 2:
            Y = np.asarray(vertical_display, dtype=float)
            X = np.broadcast_to(x_values, Y.shape)
        else:
            X, Y = np.meshgrid(x_values, vertical_display)

        if max_height is not None and vertical_coord_display == "modlev":
            logger.warning(
                "max_height=%s km does not apply to a model-level display and is ignored",
                max_height,
            )

        colormap, levels = self._resolve_plot_style(
            var_name,
            self._visible_data_for_levels(
                vertical_display, vertical_coord_display, data_values, max_height
            ),
            colormap,
            levels,
            clim_min=style.clim_min,
            clim_max=style.clim_max,
        )

        self._render_cross_section_plot(
            X,
            Y,
            data_values,
            colormap,
            levels,
            extend,
            plot_type,
            metadata,
            var_name,
            **kwargs,
        )

        self._shade_terrain(x_values, Y, vertical_coord_display)

        self._format_cross_section_axes(
            longitudes,
            vertical_display,
            vertical_coord_display,
            start_point,
            end_point,
            max_height,
            distances=distances if use_distance else None,
            path_lats=cross_section_data.get("path_lats"),
        )

        final_title = self._generate_cross_section_title(
            mpas_3d_processor, var_name, time_index, title
        )

        self.ax.set_title(final_title, fontsize=14, fontweight="bold", pad=20)
        self.ax.grid(True, alpha=0.3, linestyle="--")
        # tight_layout already fits the axis labels; keep only a slim margin for the branding footer
        plt.tight_layout()
        self.fig.subplots_adjust(bottom=max(self.fig.subplotpars.bottom, 0.06))
        self.add_timestamp_and_branding()

        self._phase_timings["render"] = time.perf_counter() - render_start

        if save_path:
            save_start = time.perf_counter()
            self._save_cross_section(save_path)
            self._phase_timings["save"] = time.perf_counter() - save_start

        return self.fig, self.ax

    def _resolve_vertical_levels(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        vertical_coord: str,
        time_index: int,
    ) -> Tuple[np.ndarray, str]:
        """
        This internal method resolves the vertical levels for the specified variable and vertical coordinate type. It attempts to retrieve the vertical levels from the MPAS3DProcessor using the get_vertical_levels method, which may return either pressure levels or model levels based on the vertical_coord argument. If the retrieved vertical levels are integer indices, it assumes they represent model levels and updates the vertical_coord accordingly. If there is an issue retrieving the vertical levels (e.g., due to missing metadata or unexpected data structure), it falls back to generating a default range of model level indices based on the dataset dimensions. The method returns the resolved vertical levels as a numpy array and a string indicating the type of vertical coordinate being used for display.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): The processor instance used to retrieve vertical levels from the dataset.
            var_name (str): The name of the variable for which to retrieve vertical levels.
            vertical_coord (str): The type of vertical coordinate to retrieve ('pressure' or 'modlev').
            time_index (int): The time index for selecting temporal data if needed for retrieving vertical levels.

        Returns:
            Tuple[np.ndarray, str]: The resolved vertical levels as a numpy array and the type of vertical coordinate being used for display.
        """
        try:
            return_pressure = vertical_coord in ("pressure", "height")

            vertical_levels = mpas_3d_processor.get_vertical_levels(
                var_name, return_pressure=return_pressure, time_index=time_index
            )

            if vertical_coord == "height":
                vertical_coord = "pressure"

            if vertical_coord not in ("pressure", "modlev"):
                vertical_coord = "modlev"

            levels_arr = np.array(vertical_levels)

            logger.debug(
                "_resolve_vertical_levels: return_pressure=%s", return_pressure
            )
            logger.debug(
                "_resolve_vertical_levels: vertical_coord_type='%s'", vertical_coord
            )
            logger.debug(
                "_resolve_vertical_levels: vertical_levels dtype=%s, shape=%s",
                levels_arr.dtype,
                levels_arr.shape,
            )
            logger.debug(
                "_resolve_vertical_levels: vertical_levels min=%.4f, max=%.4f",
                float(np.nanmin(levels_arr)),
                float(np.nanmax(levels_arr)),
            )

            if len(levels_arr) <= 60:
                logger.debug(
                    "_resolve_vertical_levels: all values (Pa if pressure): %s",
                    levels_arr,
                )
            else:
                logger.debug(
                    "_resolve_vertical_levels: first 5: %s, last 5: %s",
                    levels_arr[:5],
                    levels_arr[-5:],
                )

            if np.issubdtype(levels_arr.dtype, np.integer):
                vertical_coord = "modlev"
                if self.fig is not None and self.verbose:
                    logger.info(
                        "Vertical levels appear to be integer indices; "
                        "switching vertical_coord to 'modlev'"
                    )
            return levels_arr, vertical_coord
        except Exception as e:
            logger.warning("Could not get vertical levels, using indices: %s", e)
            dataset = mpas_3d_processor.dataset
            sizes = dataset.sizes if dataset is not None else {}
            n_levels = sizes.get("nVertLevels") or sizes.get("nVertLevelsP1") or 10
            return np.arange(n_levels), "modlev"

    def _extract_cross_section_coords(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        path_lons: np.ndarray,
        path_lats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal method extracts the longitude and latitude coordinates from the MPAS3DProcessor dataset for the specified variable. It first attempts to use the extract_2d_coordinates_for_variable method, which may provide variable-specific coordinate extraction logic. If that fails (e.g., due to missing metadata or unexpected data structure), it falls back to using the 'lonCell' and 'latCell' variables from the dataset as default coordinates. The processor's coordinates are already in degrees and are used as they are, while the raw radian-valued fallback is converted exactly once through MPASGeographicUtils.coordinates_to_degrees using the units attributes when available. Finally, it prints information about the grid domain and cross-section path, and checks whether the path extends outside the grid domain, issuing a warning if so. The method returns the longitude and latitude coordinates as numpy arrays for use in spatial interpolation along the cross-section path.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): The processor instance used to access the dataset and extract coordinates.
            var_name (str): The name of the variable for which to extract coordinates, used to determine if variable-specific coordinate extraction logic should be applied.
            path_lons (np.ndarray): The longitude coordinates of the cross-section path, used to check if the path extends outside the grid domain.
            path_lats (np.ndarray): The latitude coordinates of the cross-section path, used to check if the path extends outside the grid domain.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The longitude and latitude coordinates extracted from the dataset for the specified variable, formatted as numpy arrays.
        """
        dataset = mpas_3d_processor.dataset
        assert dataset is not None, "MPAS3DProcessor must have loaded data"

        try:
            var_da = dataset[var_name]
            lon_coords, lat_coords = (
                mpas_3d_processor.extract_2d_coordinates_for_variable(var_name, var_da)
            )
        except Exception:
            # The raw lonCell/latCell fallback is stored in radians, so convert it exactly once (honouring the units attributes) through the shared helper; coordinates returned by the processor are already in degrees and must not be converted again
            lon_coords, lat_coords = MPASGeographicUtils.coordinates_to_degrees(
                dataset["lonCell"].values,
                dataset["latCell"].values,
                dataset["lonCell"].attrs,
                dataset["latCell"].attrs,
            )

        logger.debug(
            "Grid domain: lon [%.2f, %.2f], lat [%.2f, %.2f]",
            float(np.min(lon_coords)),
            float(np.max(lon_coords)),
            float(np.min(lat_coords)),
            float(np.max(lat_coords)),
        )

        logger.debug(
            "Cross-section path: (%.2f, %.2f) to (%.2f, %.2f)",
            path_lons[0],
            path_lats[0],
            path_lons[-1],
            path_lats[-1],
        )

        # Compare every path longitude in the grid's own longitude frame so grids stored on [0, 360) and paths on [-180, 180] (or vice versa) are judged correctly
        grid_lon_min = float(np.min(lon_coords))
        grid_lon_max = float(np.max(lon_coords))
        path_lons_wrapped = (
            (np.asarray(path_lons) - grid_lon_min) % 360.0
        ) + grid_lon_min
        path_in_lon = bool(
            (grid_lon_max - grid_lon_min) >= 350.0
            or (
                np.all(path_lons_wrapped >= grid_lon_min - 1e-9)
                and np.all(path_lons_wrapped <= grid_lon_max + 1e-9)
            )
        )
        path_in_lat = min(path_lats[0], path_lats[-1]) >= np.min(lat_coords) and max(
            path_lats[0], path_lats[-1]
        ) <= np.max(lat_coords)

        if not (path_in_lon and path_in_lat):
            logger.warning("Cross-section path extends outside grid domain")
            logger.warning(
                "  Longitude OK: %s, Latitude OK: %s",
                path_in_lon,
                path_in_lat,
            )

        return lon_coords, lat_coords

    @staticmethod
    def _unwrap_dataset_var(
        mpas_3d_processor: "MPAS3DProcessor", var_name: str
    ) -> Tuple[xr.DataArray, Optional[str], Optional[str]]:
        """
        This static method unwraps the specified variable from the MPAS3DProcessor dataset and identifies the time and vertical dimension names. It first checks if the dataset is an xarray Dataset or can be converted to one, then retrieves the DataArray for the specified variable. It looks for common time dimension names ('Time' or 'time') and vertical dimension names ('nVertLevels' or 'nVertLevelsP1') in the variable's dimensions. The method returns the variable's DataArray along with the identified time and vertical dimension names, which are used later for indexing and interpolation when generating the cross-section data.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): The processor instance used to access the dataset.
            var_name (str): The name of the variable for which to extract data.

        Returns:
            Tuple[xr.DataArray, Optional[str], Optional[str]]: The data array for the variable, the time dimension name, and the vertical dimension name.
        """
        ds: Any = mpas_3d_processor.dataset

        if type(ds) is not xr.Dataset and isinstance(ds, xr.Dataset):
            ds = xr.Dataset(dict(ds.data_vars), coords=ds.coords, attrs=ds.attrs)

        var_da = ds[var_name]
        if "Time" in var_da.sizes:
            time_dim: Optional[str] = "Time"
        elif "time" in var_da.sizes:
            time_dim = "time"
        else:
            time_dim = None

        if "nVertLevels" in var_da.sizes:
            vert_dim: Optional[str] = "nVertLevels"
        elif "nVertLevelsP1" in var_da.sizes:
            vert_dim = "nVertLevelsP1"
        else:
            vert_dim = None

        return var_da, time_dim, vert_dim

    _SLAB_MAX_BYTES = 2 * 1024**3  # 2 GiB

    def _extract_variable_slab(
        self: "MPASVerticalCrossSectionPlotter",
        var_da: xr.DataArray,
        time_index: int,
        time_dim: Optional[str],
        vert_dim: Optional[str],
        expected_ncells: int,
        n_levels: int,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        This method reads the cross-section variable for a single timestep -- all vertical levels at once -- into a NumPy array, so the per-level interpolation loop can slice levels in memory instead of triggering a separate dask read for each level. Because MPAS 3D output is chunked with every vertical level in one on-disk chunk, the previous per-level .compute() re-read and discarded that whole chunk once per level (~N_levels× the necessary I/O per plot, multiplied again across timesteps and ranks); reading the (levels, nCells) slab a single time eliminates that redundancy while touching ONLY the one variable and one timestep this task needs -- not the rest of the lazily-opened dataset. To keep peak memory bounded it estimates the slab size first and returns None (so the caller falls back to per-level reads) when the slab would exceed _SLAB_MAX_BYTES, or when the variable lacks a vertical dimension or does not reduce to the expected (levels, nCells) two-dimensional shape.

        Parameters:
            var_da (xr.DataArray): The 3D variable to read (dims including time and vertical where present).
            time_index (int): Index along the time dimension to select for this plot.
            time_dim (Optional[str]): Name of the time dimension, or None if the variable has no time axis.
            vert_dim (Optional[str]): Name of the vertical dimension; None disables slab reading.
            expected_ncells (int): Number of grid cells the non-vertical axis must have for the slab to be valid.
            n_levels (int): Number of vertical levels, used with expected_ncells to estimate slab memory.

        Returns:
            Optional[Tuple[np.ndarray, int]]: The computed (levels, nCells) slab and the axis index of the vertical dimension within it, or None to signal the caller should fall back to per-level reads.
        """
        if vert_dim is None:
            return None

        itemsize = int(getattr(var_da.dtype, "itemsize", 8) or 8)

        if expected_ncells * max(n_levels, 1) * itemsize > self._SLAB_MAX_BYTES:
            return None

        slab_da = (
            var_da.isel({time_dim: time_index}) if time_dim is not None else var_da
        )

        if vert_dim not in slab_da.dims:
            return None
        vert_axis = slab_da.dims.index(vert_dim)

        if hasattr(slab_da, "compute"):
            slab_da = slab_da.compute()

        slab = np.asarray(slab_da.values if hasattr(slab_da, "values") else slab_da)

        if slab.ndim != 2 or slab.shape[1 - vert_axis] != expected_ncells:
            return None

        return slab, vert_axis

    def _extract_variable_cells(
        self: "MPASVerticalCrossSectionPlotter",
        var_da: xr.DataArray,
        time_index: int,
        time_dim: Optional[str],
        vert_dim: Optional[str],
        expected_ncells: int,
        n_levels: int,
        cell_lo: int,
        cell_hi: int,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        This method reads only the contiguous spatial-index range [cell_lo, cell_hi] of the cross-section variable for a single timestep -- all vertical levels -- into a NumPy array, so the gather can sample the transect from a small subset of the grid instead of the whole field. The transect samples at most num_points*k distinct cells; reading the index range that spans them (rather than the full nCells) is a strict subset of the full read and, because dask reads chunk-granular, a large I/O win whenever those cells cluster in index space (e.g. space-filling-curve-ordered MPAS meshes) and never more than the full read when they do not. It returns None -- so the caller falls back to the bounded full-grid read -- when the variable has no clean single spatial dimension matching the grid, when the range slice would still exceed _SLAB_MAX_BYTES, or when the result does not reduce to the expected (n_levels, n_range) two-dimensional shape.

        Parameters:
            var_da (xr.DataArray): The 3D variable to read (dims including time and vertical where present).
            time_index (int): Index along the time dimension to select for this plot.
            time_dim (Optional[str]): Name of the time dimension, or None if the variable has no time axis.
            vert_dim (Optional[str]): Name of the vertical dimension; None disables the range read.
            expected_ncells (int): Full grid cell count the variable's spatial axis must match for the indices to be valid.
            n_levels (int): Number of vertical levels the slab's vertical axis must have.
            cell_lo (int): Lowest spatial index across the transect's nearest cells (inclusive).
            cell_hi (int): Highest spatial index across the transect's nearest cells (inclusive).

        Returns:
            Optional[Tuple[np.ndarray, int]]: The computed (levels, n_range) block and the axis index of the vertical dimension within it, or None to signal the caller should fall back to the bounded full read.
        """
        if vert_dim is None:
            return None

        spatial_dims = [d for d in var_da.dims if d not in (time_dim, vert_dim)]

        if len(spatial_dims) != 1:
            return None

        spatial_dim = spatial_dims[0]

        if var_da.sizes.get(spatial_dim) != expected_ncells:
            return None

        n_cells_read = cell_hi - cell_lo + 1
        itemsize = int(getattr(var_da.dtype, "itemsize", 8) or 8)

        if n_cells_read * max(n_levels, 1) * itemsize > self._SLAB_MAX_BYTES:
            return None

        sel: Dict[str, Any] = {spatial_dim: slice(cell_lo, cell_hi + 1)}

        if time_dim is not None:
            sel[time_dim] = time_index

        block_da = var_da.isel(sel)

        if vert_dim not in block_da.dims:
            return None

        vert_axis = block_da.dims.index(vert_dim)

        if hasattr(block_da, "compute"):
            block_da = block_da.compute()

        block = np.asarray(block_da.values if hasattr(block_da, "values") else block_da)

        if (
            block.ndim != 2
            or block.shape[vert_axis] != n_levels
            or block.shape[1 - vert_axis] != n_cells_read
        ):
            return None

        return block, vert_axis

    @staticmethod
    def _extract_level_data(
        var_da: xr.DataArray, isel_dict: Dict[str, int], expected_ncells: int
    ) -> np.ndarray:
        """
        This static method extracts the data values for a specific vertical level from the variable's DataArray using the provided indexing dictionary. It first selects the appropriate slice of the DataArray based on the time and vertical indices specified in isel_dict. If the resulting level_data has a 'compute' method (e.g., if it's a Dask array), it computes it to get the actual data values. The method then checks if the extracted data is in a compatible format (either an xarray DataArray or a numpy array) and converts it to a numpy array if necessary. Finally, it verifies that the extracted data has the expected shape (1D array with length equal to the number of cells) before returning the data values for interpolation along the cross-section path.

        Parameters:
            var_da (xr.DataArray): The data array for the variable being plotted, containing the 3D data from which to extract the level data.
            isel_dict (Dict[str, int]): A dictionary specifying the indices for time and vertical dimensions to select the appropriate slice of the DataArray for the desired vertical level.
            expected_ncells (int): The expected number of cells in the extracted level data, used to validate the shape of the resulting data array.

        Returns:
            np.ndarray: A 1D array of data values for the specified vertical level, with length equal to the number of cells, ready for interpolation along the cross-section path.
        """
        level_data = var_da.isel(isel_dict)

        if hasattr(level_data, "compute"):
            level_data = level_data.compute()

        data_values = (
            level_data.values
            if hasattr(level_data, "values")
            else np.asarray(level_data)
        )

        if data_values.ndim != 1 or data_values.shape[0] != expected_ncells:
            raise ValueError(
                f"Level extraction produced shape {data_values.shape}, "
                f"expected ({expected_ncells},)"
            )

        return cast(np.ndarray, data_values)

    def _interpolate_all_levels(
        self: "MPASVerticalCrossSectionPlotter",
        var_da: xr.DataArray,
        vertical_levels: np.ndarray,
        time_index: int,
        time_dim: Optional[str],
        vert_dim: Optional[str],
        lon_coords: np.ndarray,
        lat_coords: np.ndarray,
        path_lons: np.ndarray,
        path_lats: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """
        This internal method performs interpolation of the specified variable's data across all vertical levels along the cross-section path. It initializes an array to hold the interpolated data for each vertical level and iterates through the vertical levels, extracting the corresponding data slice for each level using the identified time and vertical dimensions. For each level, it checks if the extracted data is in a compatible format and has the expected shape before performing spatial interpolation along the path defined by the longitude and latitude coordinates. The method handles exceptions during data extraction and interpolation, printing warnings as needed. After processing all levels, it checks for valid data points in the resulting cross-section data and prints summary statistics about the valid range of values before returning the final 2D array of interpolated data values for use in plotting.

        Parameters:
            var_da (xr.DataArray): The data array for the variable being plotted, containing the 3D data to be interpolated.
            vertical_levels (np.ndarray): The array of vertical levels for which to perform interpolation.
            time_index (int): The time index for selecting temporal data from the dataset.
            time_dim (Optional[str]): The name of the time dimension in the dataset, if present and applicable for indexing.
            vert_dim (Optional[str]): The name of the vertical dimension in the dataset, if present.
            lon_coords (np.ndarray): The longitude coordinates of the grid points in the dataset, used for spatial interpolation.
            lat_coords (np.ndarray): The latitude coordinates of the grid points in the dataset, used for spatial interpolation.
            path_lons (np.ndarray): The longitude coordinates along the cross-section path, used for spatial interpolation.
            path_lats (np.ndarray): The latitude coordinates along the cross-section path, used for spatial interpolation.
            num_points (int): The number of interpolation points along the cross-section path for smooth plotting.

        Returns:
            np.ndarray: A 2D array of interpolated data values along the cross-section path for each vertical level, with shape (num_vertical_levels, num_points).
        """
        n_levels = len(vertical_levels)
        phase = getattr(self, "_phase_timings", None)
        read_elapsed = 0.0
        gather_elapsed = 0.0

        precompute_t0 = time.perf_counter()
        path_indices = self._precompute_path_indices(
            lon_coords, lat_coords, path_lons, path_lats
        )
        precompute_elapsed = time.perf_counter() - precompute_t0

        cross_section_data: Optional[np.ndarray] = None

        if path_indices is not None:
            cell_lo = int(path_indices.min())
            cell_hi = int(path_indices.max())

            read_t0 = time.perf_counter()
            cell_block = self._extract_variable_cells(
                var_da,
                time_index,
                time_dim,
                vert_dim,
                lon_coords.shape[0],
                n_levels,
                cell_lo,
                cell_hi,
            )
            read_elapsed += time.perf_counter() - read_t0

            if cell_block is not None:
                block_values, block_vert_axis = cell_block
                local_neighbors: np.ndarray = path_indices - cell_lo

                gather_t0 = time.perf_counter()
                gathered, unresolved = self._gather_levels_from_block(
                    block_values, block_vert_axis, local_neighbors, n_levels, num_points
                )
                gather_elapsed += time.perf_counter() - gather_t0

                if not unresolved:
                    cross_section_data = gathered

        if cross_section_data is None:
            read_t0 = time.perf_counter()

            slab = self._extract_variable_slab(
                var_da, time_index, time_dim, vert_dim, lon_coords.shape[0], n_levels
            )

            read_elapsed += time.perf_counter() - read_t0
            gather_t0 = time.perf_counter()

            cross_section_data = self._interpolate_levels_full(
                var_da,
                vertical_levels,
                time_index,
                time_dim,
                vert_dim,
                lon_coords,
                lat_coords,
                path_lons,
                path_lats,
                num_points,
                path_indices,
                slab,
            )

            gather_elapsed += time.perf_counter() - gather_t0

        if phase is not None:
            phase["data_read"] = phase.get("data_read", 0.0) + read_elapsed
            phase["interpolate"] = (
                phase.get("interpolate", 0.0) + precompute_elapsed + gather_elapsed
            )

        valid_mask = ~np.isnan(cross_section_data)

        if np.any(valid_mask):
            data_min = float(np.min(cross_section_data[valid_mask]))
            data_max = float(np.max(cross_section_data[valid_mask]))
            logger.debug(
                "Final cross-section data: %.3f to %.3f (%d/%d valid points)",
                data_min,
                data_max,
                int(np.sum(valid_mask)),
                cross_section_data.size,
            )
        else:
            logger.warning("Final cross-section data contains NO valid values")

        return cross_section_data

    def _gather_levels_from_block(
        self: "MPASVerticalCrossSectionPlotter",
        block_values: np.ndarray,
        vert_axis: int,
        local_neighbors: np.ndarray,
        n_levels: int,
        num_points: int,
    ) -> Tuple[np.ndarray, bool]:
        """
        This method builds the (levels, path-points) cross-section from the reduced cell block read by _extract_variable_cells, using the precomputed nearest-neighbour table re-based onto the block. For each level it takes the level's values out of the block, then for each path point selects the first non-NaN among that point's k nearest cells (already distance-sorted) -- identical to the full-grid gather because the block holds exactly the same cell values, just restricted to the contiguous index range. It reports whether ANY path point at ANY level was left unresolved (all k candidates NaN), which the caller treats as a signal to fall back to the bounded full read so the global nearest-valid search can run; when nothing is unresolved the returned data is exactly what the full-grid path would have produced.

        Parameters:
            block_values (np.ndarray): The (levels, n_range) cell block (vertical axis given by vert_axis) read for the transect's index range.
            vert_axis (int): Axis of block_values corresponding to the vertical dimension.
            local_neighbors (np.ndarray): The (num_points, k) nearest-cell indices rebased into the block (full-grid index minus cell_lo), distance-sorted.
            n_levels (int): Number of vertical levels to fill.
            num_points (int): Number of path points per level.

        Returns:
            Tuple[np.ndarray, bool]: The (n_levels, num_points) cross-section and a flag that is True if any path point at any level could not be resolved from its k candidates.
        """
        cross_section_data = np.full((n_levels, num_points), np.nan)
        unresolved_any = False

        for level_idx in range(n_levels):
            level_values = np.take(block_values, level_idx, axis=vert_axis)
            candidate = level_values[local_neighbors]  # (num_points, k)
            valid = ~np.isnan(candidate)
            has_valid = valid.any(axis=1)

            rows = np.nonzero(has_valid)[0]
            if rows.size:
                first_valid_col = np.argmax(valid[rows], axis=1)
                cross_section_data[level_idx, rows] = candidate[rows, first_valid_col]

            if not bool(has_valid.all()):
                unresolved_any = True

        return cross_section_data, unresolved_any

    def _interpolate_levels_full(
        self: "MPASVerticalCrossSectionPlotter",
        var_da: xr.DataArray,
        vertical_levels: np.ndarray,
        time_index: int,
        time_dim: Optional[str],
        vert_dim: Optional[str],
        lon_coords: np.ndarray,
        lat_coords: np.ndarray,
        path_lons: np.ndarray,
        path_lats: np.ndarray,
        num_points: int,
        path_indices: Optional[np.ndarray],
        slab: Optional[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """
        This method interpolates the cross-section over every vertical level using the full grid -- the exact path that handles deep-terrain points via the global nearest-valid search in _interpolate_with_indices. It uses the pre-read full-grid slab when available (slicing each level with np.take), or reads each level individually when the slab was too large to materialise, and tolerates per-level extraction errors by leaving those levels NaN. This is the bounded full-read fallback invoked when the contiguous-range fast path is inapplicable or leaves any path point unresolved.

        Parameters:
            var_da (xr.DataArray): The 3D variable being interpolated.
            vertical_levels (np.ndarray): The vertical levels to interpolate.
            time_index (int): Time index selected for this plot.
            time_dim (Optional[str]): Name of the time dimension, or None.
            vert_dim (Optional[str]): Name of the vertical dimension, or None (no levels).
            lon_coords (np.ndarray): Full-grid cell longitudes in degrees.
            lat_coords (np.ndarray): Full-grid cell latitudes in degrees.
            path_lons (np.ndarray): Transect longitudes in degrees.
            path_lats (np.ndarray): Transect latitudes in degrees.
            num_points (int): Number of path points per level.
            path_indices (Optional[np.ndarray]): Precomputed (num_points, k) full-grid neighbour table, or None.
            slab (Optional[Tuple[np.ndarray, int]]): Pre-read full slab and its vertical axis, or None to read per level.

        Returns:
            np.ndarray: The (n_levels, num_points) interpolated cross-section.
        """
        cross_section_data: np.ndarray = np.full(
            (len(vertical_levels), num_points), np.nan
        )

        for level_idx, level in enumerate(vertical_levels):
            try:
                if vert_dim is None:
                    continue

                if slab is not None:
                    slab_values, vert_axis = slab
                    data_values = np.take(slab_values, level_idx, axis=vert_axis)
                else:
                    isel_dict: Dict[str, int] = {}

                    if time_dim is not None:
                        isel_dict[time_dim] = time_index

                    isel_dict[vert_dim] = level_idx
                    data_values = self._extract_level_data(
                        var_da, isel_dict, lon_coords.shape[0]
                    )

                cross_section_data[level_idx, :] = self._interpolate_with_indices(
                    lon_coords,
                    lat_coords,
                    data_values,
                    path_lons,
                    path_lats,
                    path_indices,
                )
            except Exception as e:
                logger.warning("Could not extract data for level %s: %s", level, e)

        return cross_section_data

    def _generate_cross_section_data(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        time_index: int,
        vertical_coord: str,
        num_points: int,
        precomputed_levels: Optional[Tuple[np.ndarray, str]] = None,
    ) -> Dict[str, Any]:
        """
        This internal method generates the cross-section data by interpolating the specified 3D variable along a great-circle path defined by the start and end points. It retrieves the vertical levels for the variable, extracts the corresponding data slices for each level, and performs spatial interpolation to obtain data values along the cross-section path. The method handles different vertical coordinate types, checks for valid data, and returns a dictionary containing the distances along the path, vertical coordinates, interpolated data values, and other relevant information for plotting.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): MPAS3DProcessor instance providing dataset access and processing capabilities.
            var_name (str): Name of the 3D variable to extract and interpolate for the cross-section.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            time_index (int): Time index for selecting temporal data from the dataset.
            vertical_coord (str): Type of vertical coordinate to use for data extraction ('pressure', 'height', 'modlev').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.
            precomputed_levels (Optional[Tuple[np.ndarray, str]]): Already-resolved (vertical_levels, vertical_coord_type) pair that skips the per-call level resolution (a full 3D pressure-field read for pressure coordinates); used by batch callers that resolve once and reuse across timesteps.

        Returns:
            Dict[str, Any]: Dictionary containing 'distances' (array of distances along the path), 'vertical_coords' (array of vertical coordinate values), 'data_values' (2D array of interpolated data values along the cross-section), 'path_lons' (array of longitude coordinates along the path), 'path_lats' (array of latitude coordinates along the path), 'path_cells' (nearest model cell index per path point, or None), and 'vertical_coord_type' (string indicating the type of vertical coordinate used).
        """
        path_lons, path_lats, distances = self._generate_great_circle_path(
            start_point, end_point, num_points
        )

        if precomputed_levels is not None:
            vertical_levels, vertical_coord = precomputed_levels
            vertical_levels = np.asarray(vertical_levels)
        else:
            vertical_levels, vertical_coord = self._resolve_vertical_levels(
                mpas_3d_processor, var_name, vertical_coord, time_index
            )

        phase = getattr(self, "_phase_timings", None)
        coords_t0 = time.perf_counter()

        lon_coords, lat_coords = self._extract_cross_section_coords(
            mpas_3d_processor, var_name, path_lons, path_lats
        )

        if phase is not None:
            phase["data_read"] = phase.get("data_read", 0.0) + (
                time.perf_counter() - coords_t0
            )

        logger.info("Interpolating %s data along cross-section", var_name)
        var_da, time_dim, vert_dim = self._unwrap_dataset_var(
            mpas_3d_processor, var_name
        )

        cross_section_data = self._interpolate_all_levels(
            var_da,
            vertical_levels,
            time_index,
            time_dim,
            vert_dim,
            lon_coords,
            lat_coords,
            path_lons,
            path_lats,
            num_points,
        )

        # Nearest model cell for every path point, used to read each column's own vertical coordinate
        path_neighbors = self._precompute_path_indices(
            lon_coords, lat_coords, path_lons, path_lats
        )
        path_cells = (
            np.asarray(path_neighbors)[:, 0] if path_neighbors is not None else None
        )

        return {
            "distances": distances,
            "path_cells": path_cells,
            "vertical_coords": np.array(vertical_levels),
            "data_values": cross_section_data,
            "path_lons": path_lons,
            "path_lats": path_lats,
            "longitudes": path_lons,
            "vertical_coord_type": vertical_coord,
        }

    def _generate_great_circle_path(
        self: "MPASVerticalCrossSectionPlotter",
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal method generates a great-circle path between two geographic points specified by their longitude and latitude coordinates. It calculates the intermediate points along the great-circle route, providing arrays of longitude, latitude, and distance along the path. The method uses spherical trigonometry to compute the path. Identical endpoints raise a ValueError; for antipodal endpoints (where every great circle is a valid route) the path deterministically follows the meridian from the start over the north pole; fewer than two points raise a ValueError. The output can be used for spatial interpolation of data along the cross-section path in the vertical cross-section plotting process.

        Parameters:
            start_point (Tuple[float, float]): Starting point of the path as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the path as (longitude, latitude) in degrees.
            num_points (int): Number of interpolation points along the path for smooth plotting.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of longitude coordinates, latitude coordinates, and distances along the path (in kilometers) for each interpolation point.
        """
        if num_points < 2:
            raise ValueError(f"num_points must be at least 2, got {num_points}")

        lon1, lat1 = radians(start_point[0]), radians(start_point[1])
        lon2, lat2 = radians(end_point[0]), radians(end_point[1])

        vec1 = np.array([cos(lat1) * cos(lon1), cos(lat1) * sin(lon1), sin(lat1)])
        vec2 = np.array([cos(lat2) * cos(lon2), cos(lat2) * sin(lon2), sin(lat2)])

        # atan2 of the cross and dot products stays accurate for both tiny and near-antipodal separations
        angular_distance = float(
            atan2(
                float(np.linalg.norm(np.cross(vec1, vec2))), float(np.dot(vec1, vec2))
            )
        )
        total_distance = angular_distance * 6371.0

        if total_distance < 1e-6:
            raise ValueError(
                "start_point and end_point are the same location; a cross-section needs two distinct points"
            )

        fractions = np.linspace(0.0, 1.0, num_points)

        if np.pi - angular_distance < 1e-6:
            # Antipodal endpoints: every great circle through the start also reaches the end, so the route is ambiguous (and slerp divides by sin(pi)). Deterministically follow the meridian from the start towards the north (over the north pole); at a pole this is the meridian of the start longitude's antipode.
            north = np.array(
                [-sin(lat1) * cos(lon1), -sin(lat1) * sin(lon1), cos(lat1)]
            )
            angle = np.pi * fractions
            points = np.outer(np.cos(angle), vec1) + np.outer(np.sin(angle), north)
        else:
            sin_d = sin(angular_distance)
            coeff_a = np.sin((1.0 - fractions) * angular_distance) / sin_d
            coeff_b = np.sin(fractions * angular_distance) / sin_d
            points = np.outer(coeff_a, vec1) + np.outer(coeff_b, vec2)

        lats = np.degrees(
            np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]))
        )
        lons = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        distances = fractions * total_distance

        # The endpoints are returned exactly as given
        lons[0], lats[0] = start_point[0], start_point[1]
        lons[-1], lats[-1] = end_point[0], end_point[1]
        distances[-1] = total_distance

        return lons, lats, distances

    def _interpolate_along_path(
        self: "MPASVerticalCrossSectionPlotter",
        grid_lons: np.ndarray,
        grid_lats: np.ndarray,
        grid_data: Union[np.ndarray, xr.DataArray, Any],
        path_lons: np.ndarray,
        path_lats: np.ndarray,
    ) -> np.ndarray:
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
            empty_result: np.ndarray = np.full(len(path_lons), np.nan)
            return empty_result

        grid_points = np.column_stack(
            [
                np.cos(np.radians(grid_lats_valid))
                * np.cos(np.radians(grid_lons_valid)),
                np.cos(np.radians(grid_lats_valid))
                * np.sin(np.radians(grid_lons_valid)),
                np.sin(np.radians(grid_lats_valid)),
            ]
        )

        path_points = np.column_stack(
            [
                np.cos(np.radians(path_lats)) * np.cos(np.radians(path_lons)),
                np.cos(np.radians(path_lats)) * np.sin(np.radians(path_lons)),
                np.sin(np.radians(path_lats)),
            ]
        )

        tree = KDTree(grid_points)
        _, indices = tree.query(path_points)

        return cast(np.ndarray, grid_data_valid[indices])

    _PATH_NEIGHBORS = 16

    def _precompute_path_indices(
        self: "MPASVerticalCrossSectionPlotter",
        grid_lons: np.ndarray,
        grid_lats: np.ndarray,
        path_lons: np.ndarray,
        path_lats: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        This method builds the nearest-neighbour mapping from the cross-section path points to the model grid, returning the K nearest grid-cell indices per path point (distance-sorted) so every vertical level can be interpolated without rebuilding a KDTree. Because the mapping depends only on the grid geometry and the transect -- not on the data values -- the previous code's per-level KDTree rebuild was pure waste, yet its NaN fast-path was defeated on real data: a single NaN anywhere in a level (e.g. terrain/below-ground masking, present at nearly every pressure level) forced the whole level back onto a fresh full-grid KDTree. Retaining K candidates lets each level take the first non-NaN among a path point's K nearest cells, which equals the original 'nearest valid cell' whenever at least one of the K is valid -- true for every path point except those whose own column is deeply terrain-masked at that level. The table is also (grid+transect)-invariant -- identical for every timestep in a batch -- so it is cached per process (keyed by a cheap grid+transect fingerprint) and the full-grid KDTree is built only once per worker/rank rather than once per file. The tree is built over the full (unmasked) grid on the unit sphere so Euclidean distance matches great-circle order; the tree itself is not retained because the rare all-NaN-neighbour case is handled by the level loop's exact fallback.

        Parameters:
            grid_lons (np.ndarray): Longitudes of the grid cells in degrees.
            grid_lats (np.ndarray): Latitudes of the grid cells in degrees.
            path_lons (np.ndarray): Longitudes of the path points in degrees.
            path_lats (np.ndarray): Latitudes of the path points in degrees.

        Returns:
            Optional[np.ndarray]: Array of shape (n_path, k) holding each path point's k nearest grid-cell indices sorted by distance, or None if the grid is empty.
        """
        grid_lons_flat = np.asarray(grid_lons).ravel()
        grid_lats_flat = np.asarray(grid_lats).ravel()

        if grid_lons_flat.size == 0:
            return None

        k = int(min(self._PATH_NEIGHBORS, grid_lons_flat.shape[0]))

        path_lons_flat = np.asarray(path_lons).ravel()
        path_lats_flat = np.asarray(path_lats).ravel()

        cache_key = _path_neighbor_fingerprint(
            grid_lons_flat, grid_lats_flat, path_lons_flat, path_lats_flat, k
        )

        cached = _PATH_NEIGHBOR_CACHE.get(cache_key)

        if cached is not None:
            return cached

        grid_lat_r = np.radians(grid_lats_flat)
        grid_lon_r = np.radians(grid_lons_flat)

        grid_points = np.column_stack(
            [
                np.cos(grid_lat_r) * np.cos(grid_lon_r),
                np.cos(grid_lat_r) * np.sin(grid_lon_r),
                np.sin(grid_lat_r),
            ]
        )

        path_lat_r = np.radians(path_lats_flat)
        path_lon_r = np.radians(path_lons_flat)

        path_points = np.column_stack(
            [
                np.cos(path_lat_r) * np.cos(path_lon_r),
                np.cos(path_lat_r) * np.sin(path_lon_r),
                np.sin(path_lat_r),
            ]
        )

        _, neighbors = KDTree(grid_points).query(path_points, k=k)

        neighbors = np.asarray(neighbors)

        if neighbors.ndim == 1:
            neighbors = neighbors[:, np.newaxis]

        if len(_PATH_NEIGHBOR_CACHE) >= _PATH_NEIGHBOR_CACHE_MAX:
            _PATH_NEIGHBOR_CACHE.clear()

        _PATH_NEIGHBOR_CACHE[cache_key] = neighbors

        return cast(np.ndarray, neighbors)

    def _interpolate_with_indices(
        self: "MPASVerticalCrossSectionPlotter",
        grid_lons: np.ndarray,
        grid_lats: np.ndarray,
        grid_data: Union[np.ndarray, xr.DataArray, Any],
        path_lons: np.ndarray,
        path_lats: np.ndarray,
        path_neighbors: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        This method maps one vertical level of grid data onto the cross-section path using the precomputed k-nearest-neighbour table, avoiding a KDTree rebuild in the common case. For each path point it scans that point's K nearest grid cells (already distance-sorted) and takes the first one whose value is not NaN; because the candidates are sorted by distance, this equals the 'nearest valid cell' the original per-level masked search returned, whenever at least one of the K candidates is valid. Crucially this is now correct even when the level contains NaNs elsewhere in the domain (the common case that previously forced a full-grid KDTree rebuild), because only the path points' own neighbourhoods matter. Only path points whose K candidates are ALL NaN -- a transect column deeply terrain-masked at this level -- need the original search, so those (and only those) are filled by a single delegation to _interpolate_along_path, which costs one valid-cell tree build for that level (never more than the old per-level cost, and skipped entirely for levels where the path is unmasked). A fully-NaN level or an absent table also falls back to _interpolate_along_path.

        Parameters:
            grid_lons (np.ndarray): Longitudes of the grid cells in degrees.
            grid_lats (np.ndarray): Latitudes of the grid cells in degrees.
            grid_data (Union[np.ndarray, xr.DataArray, Any]): Data values for this vertical level, one per grid cell.
            path_lons (np.ndarray): Longitudes of the path points in degrees.
            path_lats (np.ndarray): Latitudes of the path points in degrees.
            path_neighbors (Optional[np.ndarray]): Precomputed (n_path, k) nearest-cell index table from _precompute_path_indices, or None.

        Returns:
            np.ndarray: Interpolated data values along the path for this level.
        """
        if isinstance(grid_data, xr.DataArray):
            grid_data = grid_data.values
        elif not isinstance(grid_data, np.ndarray):
            grid_data = np.asarray(grid_data)

        grid_data_flat = grid_data.ravel()

        if path_neighbors is None:
            return self._interpolate_along_path(
                grid_lons, grid_lats, grid_data, path_lons, path_lats
            )

        candidate = grid_data_flat[path_neighbors]  # (n_path, k), distance-sorted
        valid = ~np.isnan(candidate)
        has_valid = valid.any(axis=1)

        result = np.full(path_neighbors.shape[0], np.nan, dtype=np.float64)
        rows = np.nonzero(has_valid)[0]

        if rows.size:
            first_valid_col = np.argmax(valid[rows], axis=1)
            result[rows] = candidate[rows, first_valid_col]

        if not bool(has_valid.all()):
            exact = self._interpolate_along_path(
                grid_lons, grid_lats, grid_data, path_lons, path_lats
            )
            unresolved = ~has_valid
            result[unresolved] = exact[unresolved]

        return cast(np.ndarray, result)

    def _compute_var_levels(
        self: "MPASVerticalCrossSectionPlotter",
        var_lower: str,
        var_name: str,
        data_min: float,
        data_max: float,
        data_range: float,
    ) -> np.ndarray:
        """
        This internal method computes contour levels for plotting based on the variable name and data range. It applies specific rules for different variable types (e.g., temperature, pressure, wind) to determine appropriate level spacing and range. For temperature variables, it uses a step of 5 or 2 depending on the data range. For pressure variables, it uses logarithmic spacing if the minimum value is greater than zero. For wind variables, it creates symmetric levels around zero if the data range includes both positive and negative values. For other variables, it defaults to linear spacing with 15 levels across the data range. This method ensures that the contour levels are meaningful and enhance the visualization of the variable in the cross-section plot.

        Parameters:
            var_lower (str): The lowercase version of the variable name, used for determining variable type.
            var_name (str): The original variable name (kept for API compatibility); wind variables are recognised by explicit names such as 'u', 'v', 'w', 'uReconstructZonal' or a name containing 'wind'.
            data_min (float): The minimum valid data value for the variable, used to determine the starting point of the levels.
            data_max (float): The maximum valid data value for the variable, used to determine the ending point of the levels.
            data_range (float): The range of valid data values (data_max - data_min), used to determine the spacing of the levels.

        Returns:
            np.ndarray: An array of contour levels computed based on the variable type and data range, optimized for visualization in the cross-section plot.
        """
        if "temperature" in var_lower or "temp" in var_lower:
            step = 5 if data_range > 50 else 2
            temp_levels: np.ndarray = np.arange(data_min, data_max + step, step)
            return temp_levels

        if "pressure" in var_lower:
            if data_min > 0:
                return cast(
                    np.ndarray, np.logspace(np.log10(data_min), np.log10(data_max), 15)
                )
            return cast(np.ndarray, np.linspace(data_min, data_max, 15))

        if "wind" in var_lower or var_lower in _WIND_VARS:
            max_abs = max(abs(data_min), abs(data_max))
            if data_min < 0 and data_max > 0:
                return cast(np.ndarray, np.linspace(-max_abs, max_abs, 21))
            return cast(np.ndarray, np.linspace(data_min, data_max, 15))

        return cast(np.ndarray, np.linspace(data_min, data_max, 15))

    def _get_default_levels(
        self: "MPASVerticalCrossSectionPlotter",
        data_values: Union[np.ndarray, xr.DataArray, float],
        var_name: str,
    ) -> np.ndarray:
        """
        This internal method determines default contour levels for plotting based on the data range and variable type. It handles different variable types (e.g., temperature, pressure, wind) by applying specific rules for level spacing and range. The method first checks for valid data values, computes the minimum and maximum, and then applies variable-specific logic to generate an array of contour levels that are optimized for the variable's typical range and variability. This ensures that the resulting cross-section plot has meaningful contours that enhance the visualization of the atmospheric variable along the transect.

        Parameters:
            data_values (Union[np.ndarray, xr.DataArray, float]): Array of data values for the variable being plotted, which may contain NaN values.
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
            return cast(np.ndarray, np.linspace(0, 1, 11))

        data_min, data_max = valid_data.min(), valid_data.max()
        data_range = data_max - data_min

        if data_range == 0:
            return cast(np.ndarray, np.array([data_min]))

        var_lower = var_name.lower()
        return self._compute_var_levels(
            var_lower, var_name, data_min, data_max, data_range
        )

    def _extract_height_from_dataset(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
        vertical_coords: np.ndarray,
        var_name: str,
    ) -> Optional[np.ndarray]:
        """
        This internal method attempts to extract geometric height data from the MPAS dataset for the specified variable name (e.g., 'zgrid' or 'height') at the given time index. It checks if the variable exists in the dataset, retrieves the height data for the first cell (assuming it's representative of the vertical structure), and processes it to match the expected length based on the vertical coordinates. If the height data has one more level than the vertical coordinates, it computes mid-level heights by averaging adjacent levels. If it matches the length of vertical coordinates, it returns it directly. If neither condition is met, it attempts to interpolate the height data to match the vertical coordinates. The method includes error handling to return None if extraction fails or if the variable is unavailable, allowing for fallback methods to determine height when necessary.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): MPAS3DProcessor instance providing access to the dataset for height variable extraction.
            time_index (int): Time index for selecting the appropriate temporal slice of the height variable.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the cross-section, which determines the expected length of the height data.
            var_name (str): Name of the height variable to extract from the dataset (e.g., 'zgrid' or 'height').

        Returns:
            Optional[np.ndarray]: Array of geometric height values in meters corresponding to the vertical coordinates, or None if extraction fails or variable is unavailable.
        """
        try:
            height_data = None
            dataset = mpas_3d_processor.dataset

            if dataset is not None and var_name in dataset.data_vars:
                height_data = dataset[var_name].isel(Time=time_index, nCells=0).values
            elif (
                hasattr(mpas_3d_processor, "grid_file") and mpas_3d_processor.grid_file
            ):
                height_data = self._load_height_var_from_grid_file(
                    mpas_3d_processor.grid_file, var_name
                )

            if height_data is None:
                return None

            return self._match_height_to_vertical_coords(height_data, vertical_coords)
        except Exception:
            return None

    @staticmethod
    def _load_height_var_from_grid_file(
        grid_file: str, var_name: str, cells: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        This static helper method attempts to load a height variable from the MPAS grid file. It tries to read only the specified variable to minimize memory usage, and if successful, it extracts the height values for the first cell, or for the given cells (one column per cell, shape (n_cells, n_vertical)) when cells is supplied. The method includes error handling to return None if the variable is not found or if any issues arise during file reading, allowing the calling method to proceed with alternative methods for determining height when necessary.

        Parameters:
            grid_file (str): Path to the MPAS grid file to read the height variable from.
            var_name (str): Name of the height variable to load (e.g., 'zgrid', 'height').
            cells (Optional[np.ndarray]): Optional model cell indices whose full columns are returned instead of the first cell (default: None).

        Returns:
            Optional[np.ndarray]: The height values for the first cell (or the requested cells), or None if the variable is absent or the read fails.
        """
        try:
            # Only load the needed height variable from the grid file
            open_kwargs: dict = {"decode_times": False}

            try:
                with xr.open_dataset(
                    grid_file, decode_times=False, engine=NETCDF_ENGINE
                ) as probe:
                    all_vars = list(probe.data_vars)

                vars_to_drop = [v for v in all_vars if v != var_name]

                if vars_to_drop:
                    open_kwargs["drop_variables"] = vars_to_drop
            except Exception:
                pass

            with xr.open_dataset(
                grid_file, **open_kwargs, engine=NETCDF_ENGINE
            ) as grid_ds:
                if var_name in grid_ds.data_vars:
                    if cells is not None:
                        picked = grid_ds[var_name].isel(
                            nCells=xr.DataArray(
                                np.asarray(cells, dtype=int), dims="path"
                            )
                        )
                        vert = [d for d in picked.dims if d != "path"][0]
                        return cast(np.ndarray, picked.transpose("path", vert).values)
                    return cast(np.ndarray, grid_ds[var_name].isel(nCells=0).values)
        except Exception:
            pass
        return None

    @staticmethod
    def _match_height_to_vertical_coords(
        height_data: np.ndarray, vertical_coords: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        This static helper method processes the raw height data to match the expected length based on the vertical coordinates. If the height data has one more level than the vertical coordinates, it computes mid-level heights by averaging adjacent levels. If the height data matches the length of vertical coordinates, it returns it directly. If neither condition is met, it attempts to interpolate the height data to match the vertical coordinates using scipy's interp1d function. The method includes error handling to return None if interpolation fails, allowing for fallback methods to determine height when necessary.

        Parameters:
            height_data (np.ndarray): Raw height values extracted from the dataset.
            vertical_coords (np.ndarray): Vertical coordinate values whose length sets the expected output length.

        Returns:
            Optional[np.ndarray]: Height values matching the vertical coordinates, or None if interpolation fails.
        """
        height_data = np.asarray(height_data, dtype=float)

        if len(height_data) == len(vertical_coords) + 1:
            return cast(np.ndarray, 0.5 * (height_data[:-1] + height_data[1:]))

        if len(height_data) == len(vertical_coords):
            return height_data

        try:
            from scipy.interpolate import interp1d

            interp_x_known = np.linspace(0, 1, len(height_data))
            interp_y_known = height_data

            height_interp_fn = interp1d(
                interp_x_known,
                interp_y_known,
                bounds_error=False,
                fill_value=cast(Any, "extrapolate"),
            )

            interp_x_query = np.linspace(0, 1, len(vertical_coords))
            return cast(np.ndarray, height_interp_fn(interp_x_query))
        except Exception:
            return None

    def _read_path_columns(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        time_index: int,
        cells: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        This internal helper reads the full vertical column of a dataset variable at each of the given model cells for one time step, returning an array of shape (n_points, n_vertical). Only the requested cells are read (vectorised point selection on the cell dimension, so lazily-opened datasets are not loaded whole). It returns None when the variable is absent or does not reduce to a single cell dimension plus one vertical dimension, which lets callers fall back to coarser vertical-coordinate estimates.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): Processor whose dataset holds the variable.
            var_name (str): Name of the variable to read (for example 'pressure' or 'zgrid').
            time_index (int): Time index to select when the variable has a time dimension.
            cells (np.ndarray): Model cell indices, one per path point.

        Returns:
            Optional[np.ndarray]: The (n_points, n_vertical) column values, or None if the variable cannot be read this way.
        """
        try:
            var_da, time_dim, vert_dim = self._unwrap_dataset_var(
                mpas_3d_processor, var_name
            )
        except Exception:
            return None

        if vert_dim is None:
            return None

        spatial_dims = [d for d in var_da.dims if d not in (time_dim, vert_dim)]

        if len(spatial_dims) != 1:
            return None

        selection: Dict[str, Any] = {
            spatial_dims[0]: xr.DataArray(np.asarray(cells, dtype=int), dims="path")
        }

        if time_dim is not None:
            selection[time_dim] = min(time_index, var_da.sizes[time_dim] - 1)

        try:
            columns = var_da.isel(selection).transpose("path", vert_dim)
            return np.asarray(columns.values, dtype=float)
        except Exception:
            return None

    @staticmethod
    def _columns_to_levels(columns: np.ndarray, n_levels: int) -> Optional[np.ndarray]:
        """
        This static helper reshapes per-column vertical values of shape (n_points, n_vertical) onto the n_levels levels the plotted variable lives on, returning (n_levels, n_points). Equal lengths pass through; one extra value per column (layer interfaces, such as zgrid on nVertLevelsP1) is averaged to layer mid-points; and one fewer value (mid-level pressure for a variable on interfaces) is extended to the interfaces by linear extrapolation at the two ends. Any other mismatch returns None.

        Parameters:
            columns (np.ndarray): Per-column values with shape (n_points, n_vertical).
            n_levels (int): Number of vertical levels of the plotted variable.

        Returns:
            Optional[np.ndarray]: Values with shape (n_levels, n_points), or None when the vertical sizes are incompatible.
        """
        n_vertical = columns.shape[1]

        if n_vertical == n_levels:
            return np.asarray(columns.T)

        if n_vertical == n_levels + 1:
            return np.asarray(0.5 * (columns[:, :-1] + columns[:, 1:]).T)

        if n_vertical == n_levels - 1 and n_vertical >= 2:
            interior = 0.5 * (columns[:, :-1] + columns[:, 1:])
            bottom = 1.5 * columns[:, :1] - 0.5 * columns[:, 1:2]
            top = 1.5 * columns[:, -1:] - 0.5 * columns[:, -2:-1]
            return np.asarray(np.concatenate([bottom, interior, top], axis=1).T)

        return None

    def _sample_column_pressure(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
        cells: np.ndarray,
        n_levels: int,
    ) -> Optional[np.ndarray]:
        """
        This internal method returns the actual pressure (Pa) of every model level in the column under each path point, shape (n_levels, n_points), taken from the 'pressure' variable or the sum of 'pressure_p' and 'pressure_base' at the sampled cells. Using each column's own pressure (instead of one global mean per level) lets the cross-section show the terrain-following structure, such as lower surface pressure over mountains. It returns None when no suitable pressure field exists.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): Processor whose dataset holds the pressure fields.
            time_index (int): Time index to sample.
            cells (np.ndarray): Model cell index for each path point.
            n_levels (int): Number of vertical levels of the plotted variable.

        Returns:
            Optional[np.ndarray]: Pressure in Pa with shape (n_levels, n_points), or None if unavailable.
        """
        dataset = mpas_3d_processor.dataset

        if dataset is None:
            return None

        if "pressure" in dataset.data_vars:
            columns = self._read_path_columns(
                mpas_3d_processor, "pressure", time_index, cells
            )
        elif "pressure_p" in dataset.data_vars and "pressure_base" in dataset.data_vars:
            perturbation = self._read_path_columns(
                mpas_3d_processor, "pressure_p", time_index, cells
            )
            base = self._read_path_columns(
                mpas_3d_processor, "pressure_base", time_index, cells
            )
            columns = (
                perturbation + base
                if perturbation is not None and base is not None
                else None
            )
        else:
            return None

        if columns is None:
            return None

        return self._columns_to_levels(columns, n_levels)

    def _sample_column_height(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
        cells: np.ndarray,
        n_levels: int,
    ) -> Optional[np.ndarray]:
        """
        This internal method returns the geometric height (m above mean sea level) of every model level in the column under each path point, shape (n_levels, n_points). It reads 'zgrid' (or 'height') at the sampled cells from the dataset or, when absent there, from the grid file, and averages the layer interfaces to mid-levels where the plotted variable is a mid-level field. Per-column heights keep terrain in the section instead of reusing the heights of cell 0 for every column. It returns None when no height field is available.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): Processor whose dataset or grid file holds the height field.
            time_index (int): Time index to sample (ignored for static height fields).
            cells (np.ndarray): Model cell index for each path point.
            n_levels (int): Number of vertical levels of the plotted variable.

        Returns:
            Optional[np.ndarray]: Height in metres with shape (n_levels, n_points), or None if unavailable.
        """
        for height_var in ("zgrid", "height"):
            columns = self._read_path_columns(
                mpas_3d_processor, height_var, time_index, cells
            )

            if columns is None:
                grid_file = getattr(mpas_3d_processor, "grid_file", None)

                if grid_file:
                    columns = self._load_height_var_from_grid_file(
                        grid_file, height_var, cells
                    )

            if columns is not None:
                levels = self._columns_to_levels(
                    np.asarray(columns, dtype=float), n_levels
                )

                if levels is not None:
                    return levels

        return None

    def _resolve_column_vertical(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
        cells: Optional[np.ndarray],
        n_levels: int,
        display_type: str,
    ) -> Optional[np.ndarray]:
        """
        This internal method builds the two-dimensional vertical coordinate (n_levels, n_points) used as the y-values of the cross-section, so each path point is drawn at the true pressure or height of its own model column. For 'pressure_hPa' it converts the per-column pressure to hPa; for 'height_km' it uses per-column geometric height in km, falling back to the standard-atmosphere height of the per-column pressure when no height field exists. It returns None for other display types or whenever the per-column information cannot be obtained, in which case the caller keeps the one-dimensional vertical axis.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): Processor providing the pressure and height fields.
            time_index (int): Time index to sample.
            cells (Optional[np.ndarray]): Model cell index for each path point, or None when unknown.
            n_levels (int): Number of vertical levels of the plotted variable.
            display_type (str): Vertical display type, 'pressure_hPa' or 'height_km'.

        Returns:
            Optional[np.ndarray]: The (n_levels, n_points) vertical coordinate in display units, or None.
        """
        if cells is None or display_type not in ("pressure_hPa", "height_km"):
            return None

        try:
            if display_type == "pressure_hPa":
                pressure_pa = self._sample_column_pressure(
                    mpas_3d_processor, time_index, cells, n_levels
                )
                return None if pressure_pa is None else pressure_pa / 100.0

            height_m = self._sample_column_height(
                mpas_3d_processor, time_index, cells, n_levels
            )

            if height_m is None:
                pressure_pa = self._sample_column_pressure(
                    mpas_3d_processor, time_index, cells, n_levels
                )

                if pressure_pa is None:
                    return None

                flat = np.clip(pressure_pa.ravel(), 1.0, None)
                height_m = self._std_atm_pressure_to_height(flat).reshape(
                    pressure_pa.shape
                )

            return np.asarray(height_m / 1000.0)
        except Exception as exc:
            logger.debug("_resolve_column_vertical: falling back to 1-D axis: %s", exc)
            return None

    def _try_extract_height_km(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
        vertical_coords: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, str]]:
        """
        This helper method attempts to extract geometric height from the dataset using known variable names ('zgrid' or 'height') and convert it to kilometers. It calls the _extract_height_from_dataset method for each variable name, and if successful, it converts the height from meters to kilometers and returns it along with the coordinate type string. If extraction fails for both variable names, it returns None, allowing the calling method to proceed with alternative methods for determining height. This approach provides a way to utilize available geometric height data in the dataset when pressure coordinates are used, enhancing the accuracy of the vertical axis in the cross-section plot.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): MPAS3DProcessor instance for accessing the dataset to extract height information.
            time_index (int): Time index for selecting the appropriate temporal slice when extracting height from the dataset.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the cross-section, which may be needed to determine the expected length of the height data.

        Returns:
            Optional[Tuple[np.ndarray, str]]: A tuple containing the extracted height values in kilometers and the string 'height_km' if extraction is successful, or None if extraction fails for both variable names.
        """
        logger.debug(
            "_try_extract_height_km: Attempting to extract geometric height from dataset"
        )
        try:
            for var in ("zgrid", "height"):
                logger.debug("_try_extract_height_km: Trying variable '%s'", var)
                height_m = self._extract_height_from_dataset(
                    mpas_3d_processor, time_index, vertical_coords, var
                )
                if height_m is not None:
                    logger.debug(
                        "_try_extract_height_km: SUCCESS with '%s': min=%.2f m, max=%.2f m",
                        var,
                        float(np.nanmin(height_m)),
                        float(np.nanmax(height_m)),
                    )
                    logger.debug(
                        "_try_extract_height_km: Converted to km: min=%.4f, max=%.4f",
                        float(np.nanmin(height_m / 1000.0)),
                        float(np.nanmax(height_m / 1000.0)),
                    )
                    return height_m / 1000.0, "height_km"
                else:
                    logger.debug(
                        "_try_extract_height_km: '%s' not found or extraction failed",
                        var,
                    )
        except Exception as e:
            logger.debug("_try_extract_height_km: Exception: %s", e)
        logger.debug(
            "_try_extract_height_km: No geometric height found, "
            "will fall back to barometric approximation"
        )
        return None

    @staticmethod
    def _std_atm_pressure_to_height(pressure_pa: np.ndarray) -> np.ndarray:
        """
        This static method converts pressure values in Pascals to geometric height in meters using the US Standard Atmosphere 1976 piecewise model. The method defines the standard atmospheric layers with their base heights, temperatures, lapse rates, and pressures. It then iterates through the input pressure values, determines which atmospheric layer each pressure falls into, and applies the appropriate formula (isothermal or with lapse rate) to compute the corresponding geometric height. The method handles potential issues with non-positive or non-finite pressure values by clipping them to a minimum positive value to avoid mathematical errors during logarithmic calculations. This conversion is used as a fallback when geometric height data is not available in the dataset, providing an approximate vertical axis for the cross-section plot based on standard atmospheric conditions.

        Parameters:
            pressure_pa (np.ndarray): 1-D array of pressure values in Pa.  Values must be positive and finite.

        Returns:
            np.ndarray: Corresponding geometric heights in metres.
        """
        g = 9.80665  # gravitational acceleration (m/s^2)
        R = 287.053  # specific gas constant for dry air (J/(kg*K))

        # (base_height_m, base_temp_K, lapse_rate_K_per_m, base_pressure_Pa)
        layers = [
            (0.0, 288.15, -0.0065, 101325.0),  # Troposphere: 0-11 km
            (11000.0, 216.65, 0.0, 22632.1),  # Tropopause: 11-20 km
            (20000.0, 216.65, 0.001, 5474.89),  # Stratosphere 1: 20-32 km
            (32000.0, 228.65, 0.0028, 868.02),  # Stratosphere 2: 32-47 km
            (47000.0, 270.65, 0.0, 110.91),  # Stratopause: 47-51 km
        ]

        height_m = np.empty_like(pressure_pa)

        for i in range(len(pressure_pa)):
            pressure_val = pressure_pa[i]

            # Find which layer this pressure falls in (layers sorted by decreasing base pressure)
            layer_idx = len(layers) - 1

            for j in range(len(layers) - 1):
                if pressure_val >= layers[j + 1][3]:
                    layer_idx = j
                    break

            # Extract layer parameters for the identified layer
            base_height, base_temp, lapse_rate, base_pressure = layers[layer_idx]

            if abs(lapse_rate) < 1e-10:
                # Isothermal layer: h = base_height - (R*base_temp/g) * ln(P/base_pressure)
                height_m[i] = base_height - (R * base_temp / g) * np.log(
                    pressure_val / base_pressure
                )
            else:
                # Layer with lapse rate: h = base_height + (base_temp/lapse_rate) * ((P/base_pressure)^(-R*lapse_rate/g) - 1)
                exponent = -R * lapse_rate / g
                height_m[i] = base_height + (base_temp / lapse_rate) * (
                    (pressure_val / base_pressure) ** exponent - 1.0
                )

        return cast(np.ndarray, height_m)

    def _pressure_to_height_approx(
        self: "MPASVerticalCrossSectionPlotter",
        vertical_coords: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """
        This internal method converts pressure values to geometric height in kilometers using the US Standard Atmosphere 1976 approximation. It first checks if the input pressure values are likely in hPa (by checking if the maximum value is less than 10000) and converts them to Pa if necessary. It then handles any non-positive or non-finite pressure values by clipping them to a minimum positive value to avoid issues with logarithmic calculations. The method applies the standard atmosphere model to convert the pressure values to height in meters, and then converts it to kilometers before returning. If any exceptions occur during this process, it falls back to returning the original vertical coordinates converted to hPa and indicates that the vertical coordinate type is 'pressure_hPa'. This method provides an approximate way to determine height when geometric height data is not available in the dataset, ensuring that the cross-section plot can still be generated with a reasonable vertical axis.

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values representing pressure levels, which may be in hPa or Pa and may contain non-positive or non-finite values.

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the converted height values in kilometers and the string 'height_km' if conversion is successful, or the original vertical coordinates converted to hPa and the string 'pressure_hPa' if an exception occurs during conversion.
        """
        try:
            pressure_pa: np.ndarray = vertical_coords.astype(float).copy()

            logger.debug("_pressure_to_height_approx: Input pressure values:")
            logger.debug(
                "_pressure_to_height_approx: dtype=%s, shape=%s",
                pressure_pa.dtype,
                pressure_pa.shape,
            )
            logger.debug(
                "_pressure_to_height_approx: min=%.4f, max=%.4f",
                float(np.nanmin(pressure_pa)),
                float(np.nanmax(pressure_pa)),
            )

            if np.nanmax(pressure_pa) < 10000:
                logger.debug(
                    "_pressure_to_height_approx: max < 10000 => treating as hPa, multiplying by 100"
                )
                pressure_pa = pressure_pa * 100.0
            else:
                logger.debug(
                    "_pressure_to_height_approx: max >= 10000 => treating as Pa (no conversion)"
                )

            min_positive = 1.0

            if np.any(pressure_pa <= 0) or np.any(~np.isfinite(pressure_pa)):
                logger.debug(
                    "_pressure_to_height_approx: non-positive/non-finite values detected, clipping"
                )
                if self.verbose:
                    logger.warning(
                        "Pressure levels contained non-positive or non-finite values; "
                        "clipping to minimum positive value to avoid log(0)"
                    )
                pressure_pa = np.where(
                    np.isfinite(pressure_pa), pressure_pa, min_positive
                )
                pressure_pa = np.clip(pressure_pa, min_positive, None)

            logger.debug(
                "_pressure_to_height_approx: Using US Standard Atmosphere 1976 (piecewise, 5 layers)"
            )
            logger.debug(
                "_pressure_to_height_approx: pressure_pa after processing: min=%.4f, max=%.4f",
                float(np.nanmin(pressure_pa)),
                float(np.nanmax(pressure_pa)),
            )

            height_m = self._std_atm_pressure_to_height(pressure_pa)
            height_km = np.maximum(height_m / 1000.0, 0.0)

            logger.debug(
                "_pressure_to_height_approx: height_km: min=%.4f, max=%.4f",
                float(np.nanmin(height_km)),
                float(np.nanmax(height_km)),
            )
            if len(height_km) <= 60:
                logger.debug(
                    "_pressure_to_height_approx: all height_km values: %s",
                    np.array2string(height_km, precision=3, separator=", "),
                )

            return height_km, "height_km"
        except Exception as e:
            logger.debug("_pressure_to_height_approx: EXCEPTION: %s", e)
            return vertical_coords / 100.0, "pressure_hPa"

    def _convert_vertical_to_height(
        self: "MPASVerticalCrossSectionPlotter",
        vertical_coords: np.ndarray,
        vertical_coord_type: str,
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
    ) -> Tuple[np.ndarray, str]:
        """
        This internal method converts vertical coordinate values to geometric height in kilometers based on the specified vertical coordinate type. If the input coordinate type is 'height', it simply converts from meters to kilometers. If the input type is 'pressure', it first attempts to extract geometric height from the dataset using known variable names ('zgrid' or 'height'). If successful, it converts to kilometers and returns. If extraction fails, it applies a standard atmosphere approximation to convert pressure to height, while handling potential issues with non-positive or non-finite pressure values. For 'modlev', it also tries to extract geometric height if available, otherwise it returns the original model level indices. The method ensures that the returned vertical coordinates are in a consistent format for plotting and includes error handling for various edge cases.

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values to convert.
            vertical_coord_type (str): Type of vertical coordinate ('height', 'pressure', 'modlev') indicating how to interpret the input coordinates.
            mpas_3d_processor ('MPAS3DProcessor'): MPAS3DProcessor instance for accessing the dataset to extract height information if needed.
            time_index (int): Time index for selecting the appropriate temporal slice when extracting height from the dataset.

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the converted vertical coordinates in kilometers and a string indicating the type of vertical coordinate used for display (e.g., 'height_km', 'pressure_hPa', or 'modlev').
        """
        logger.debug(
            "_convert_vertical_to_height: vertical_coord_type='%s'",
            vertical_coord_type,
        )
        logger.debug(
            "_convert_vertical_to_height: vertical_coords: shape=%s, min=%.4f, max=%.4f",
            vertical_coords.shape,
            float(np.nanmin(vertical_coords)),
            float(np.nanmax(vertical_coords)),
        )

        if vertical_coord_type == "height":
            logger.debug(
                "_convert_vertical_to_height: Path: direct height -> dividing by 1000 for km"
            )
            return vertical_coords / 1000.0, "height_km"

        if vertical_coord_type == "pressure":
            logger.debug(
                "_convert_vertical_to_height: Path: pressure -> trying geometric height first, "
                "then barometric approx"
            )
            result = self._try_extract_height_km(
                mpas_3d_processor, time_index, vertical_coords
            )
            if result is not None:
                logger.debug(
                    "_convert_vertical_to_height: Using extracted geometric height"
                )
                return result
            logger.debug(
                "_convert_vertical_to_height: Falling back to barometric approximation"
            )
            return self._pressure_to_height_approx(vertical_coords)

        logger.debug(
            "_convert_vertical_to_height: Path: modlev -> trying geometric height"
        )
        result = self._try_extract_height_km(
            mpas_3d_processor, time_index, vertical_coords
        )

        if result is not None:
            return result

        logger.debug(
            "_convert_vertical_to_height: No height conversion possible, returning model-level indices"
        )
        return np.arange(len(vertical_coords), dtype=float), "modlev"

    def _apply_standard_pressure_ticks(
        self: "MPASVerticalCrossSectionPlotter",
        vertical_coords: np.ndarray,
    ) -> None:
        """
        This internal method applies standard meteorological pressure levels as major ticks on the y-axis of the vertical cross-section plot when using pressure coordinates. It first checks if the axes have been created, then it defines a list of standard pressure levels commonly used in meteorology (e.g., 1000 hPa, 850 hPa, etc.). It determines which of these standard levels fall within the range of the provided vertical coordinate values and sets those as major ticks on the y-axis. The method also includes a custom formatter to display tick labels appropriately based on their magnitude (e.g., integers for values >= 1 and two decimal places for smaller values). Error handling is included to ensure that tick formatting does not fail if certain styling features are unavailable, providing a robust setup for clear visualization of pressure levels along the cross-section.

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values representing pressure levels, used to determine which standard pressure levels to apply as ticks on the y-axis.

        Returns:
            None: Modifies self.ax y-axis properties directly without returning a value.
        """
        if self.ax is None:
            return

        try:
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

            standard_ticks = [
                1000,
                850,
                700,
                600,
                500,
                400,
                300,
                250,
                200,
                150,
                100,
                70,
                50,
                30,
                20,
                10,
                1,
            ]
            # Ticks are chosen for the visible window (the axis limits), falling back to the data range
            y_lo, y_hi = sorted(self.ax.get_ylim())
            data_min = max(float(np.nanmin(vertical_coords)), y_lo)
            data_max = min(float(np.nanmax(vertical_coords)), y_hi)
            tick_vals: List[float] = [
                float(t) for t in standard_ticks if data_min <= t <= data_max
            ]

            if len(tick_vals) < 2 and data_min > 0 and data_max > data_min:
                # Narrow range: a few log-spaced, rounded ticks inside it instead of matplotlib's scientific-notation log labels
                step = 10.0 if data_max >= 200 else 5.0 if data_max >= 50 else 1.0
                tick_vals = sorted(
                    {
                        float(np.round(v / step) * step)
                        for v in np.geomspace(data_min, data_max, 5)
                        if data_min <= np.round(v / step) * step <= data_max
                    },
                    reverse=True,
                )

            if len(tick_vals) >= 2:
                self.ax.yaxis.set_major_locator(FixedLocator(tick_vals))

                def _fmt(x: float, pos: Any) -> str:
                    if x >= 1:
                        return f"{int(x):d}"
                    return f"{x:.2f}"

                self.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
                self.ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                self.ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: f"{x:g}")
                )
                self.ax.yaxis.set_minor_formatter(NullFormatter())
        except Exception:
            pass

    def _setup_pressure_axis(
        self: "MPASVerticalCrossSectionPlotter",
        vertical_coords: np.ndarray,
        use_standard_ticks: bool = True,
    ) -> None:
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
            vmin = float(np.nanmin(vertical_coords))
        except Exception:
            logger.warning(
                "Could not determine pressure coordinate min; using linear y-scale"
            )
            return

        if vmin <= 0:
            logger.warning(
                "Detected non-positive pressure coordinate values; "
                "using linear y-scale for pressure display"
            )
            return

        self.ax.set_yscale("log")

        if use_standard_ticks:
            self._apply_standard_pressure_ticks(vertical_coords)

    def _apply_x_axis_formatting(
        self: "MPASVerticalCrossSectionPlotter",
        longitudes: np.ndarray,
        distances: Optional[np.ndarray] = None,
        latitudes: Optional[np.ndarray] = None,
    ) -> None:
        """
        This internal method formats the x-axis of the vertical cross-section. When along-path distances are supplied (the normal case) the axis is distance along the great-circle path in km, so north-south and dateline-crossing sections are drawn with the correct geometry, and each tick carries a second line with the latitude and longitude at that distance (interpolated along the path with longitude unwrapped across the dateline). Without distances it falls back to the historical longitude axis with degree-formatted ticks. Formatting errors are swallowed so styling problems never prevent a plot.

        Parameters:
            longitudes (np.ndarray): Longitudes along the cross-section path in degrees.
            distances (Optional[np.ndarray]): Along-path distances in km for each path point; when given they define the x-axis (default: None).
            latitudes (Optional[np.ndarray]): Latitudes along the path in degrees, used for the secondary tick labels (default: None).

        Returns:
            None: Modifies self.ax x-axis properties directly without returning a value.
        """
        assert self.ax is not None, "Axes must be created before formatting"

        if distances is None:
            self.ax.set_xlabel("Longitude", fontsize=12, labelpad=10)
            self.ax.set_xlim(longitudes.min(), longitudes.max())

            try:
                from matplotlib.ticker import FuncFormatter

                lon_formatter = FuncFormatter(MPASVisualizationStyle.format_longitude)
                self.ax.xaxis.set_major_formatter(lon_formatter)
            except Exception:
                pass
            return

        dist = np.asarray(distances, dtype=float)
        self.ax.set_xlabel("Distance along path [km]", fontsize=12, labelpad=8)
        self.ax.set_xlim(float(dist.min()), float(dist.max()))

        try:
            from matplotlib.ticker import FuncFormatter, MaxNLocator

            x_locator = MaxNLocator(nbins=5)
            tick_step = float(np.diff(x_locator.tick_values(dist.min(), dist.max()))[0])
            # Short paths have sub-kilometre tick spacing; show enough decimals that neighbouring labels differ
            decimals = (
                0 if tick_step >= 1.0 else int(np.ceil(-np.log10(max(tick_step, 1e-9))))
            )

            lon_unwrapped = np.degrees(np.unwrap(np.radians(np.asarray(longitudes))))
            lat_path = None if latitudes is None else np.asarray(latitudes, dtype=float)

            # Short paths span fractions of a degree per tick, so the place labels need more decimals to stay distinct
            degree_span = float(
                max(
                    np.ptp(lon_unwrapped),
                    0.0 if lat_path is None else np.ptp(lat_path),
                )
            )
            degrees_per_tick = degree_span * tick_step / max(float(np.ptp(dist)), 1e-9)
            place_decimals = int(
                np.clip(np.ceil(-np.log10(max(degrees_per_tick, 1e-9))), 1, 4)
            )

            def _fmt(x: float, pos: Any) -> str:
                label = f"{x:.{decimals}f}"
                lon_here = float(np.interp(x, dist, lon_unwrapped))
                lon_here = ((lon_here + 180.0) % 360.0) - 180.0
                place = f"{abs(lon_here):.{place_decimals}f}\u00b0{'E' if lon_here >= 0 else 'W'}"
                if lat_path is not None:
                    lat_here = float(np.interp(x, dist, lat_path))
                    lat_text = f"{abs(lat_here):.{place_decimals}f}\u00b0{'N' if lat_here >= 0 else 'S'}"
                    place = f"{lat_text} {place}"
                return f"{label}\n{place}"

            self.ax.xaxis.set_major_locator(x_locator)
            self.ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        except Exception:
            pass

    def _shade_terrain(
        self: "MPASVerticalCrossSectionPlotter",
        x_values: np.ndarray,
        vertical_display: np.ndarray,
        vertical_coord_type: str,
    ) -> None:
        """
        This internal method shades the ground below the lowest model level of every column when a per-column (two-dimensional) vertical coordinate is displayed, making terrain visible in the section. For height axes the region below the lowest level is filled down to zero; for pressure axes (which are inverted) it is filled from the lowest-level pressure to the bottom of the axis. It does nothing for a one-dimensional vertical axis or other coordinate types.

        Parameters:
            x_values (np.ndarray): Horizontal coordinate of each path point.
            vertical_display (np.ndarray): Vertical coordinate; shading only applies when it is two-dimensional (levels, points).
            vertical_coord_type (str): Vertical display type ('height_km' or 'pressure_hPa').

        Returns:
            None: Adds a filled polygon to self.ax.
        """
        if self.ax is None or np.ndim(vertical_display) != 2:
            return

        surface = np.asarray(vertical_display)[0, :]

        if vertical_coord_type == "height_km":
            self.ax.fill_between(
                x_values, 0.0, surface, color="0.35", zorder=3, linewidth=0
            )
        elif vertical_coord_type == "pressure_hPa":
            self.ax.fill_between(
                x_values,
                surface,
                float(np.nanmax(vertical_display)) * 1.2,
                color="0.35",
                zorder=3,
                linewidth=0,
            )

    def _set_pressure_ylim(
        self: "MPASVerticalCrossSectionPlotter",
        vertical_coords: np.ndarray,
        max_height: Optional[float],
        p0: float,
        scale_height: float = 8.4,
    ) -> None:
        """
        This internal method sets the y-axis limits for pressure-based vertical cross-section plots based on the provided vertical coordinate values and an optional maximum height. If a maximum height is specified, it calculates the corresponding minimum pressure using the barometric formula and sets the y-axis limits accordingly. If no maximum height is provided, it defaults to setting the limits based on the range of the vertical coordinates. The method includes error handling to ensure that axis limits are set correctly even if certain styling features are unavailable, providing a robust setup for clear visualization of pressure levels along the cross-section. This method is essential for ensuring that the y-axis of pressure-based cross-section plots is appropriately scaled to the range of pressure levels being visualized, enhancing the readability and interpretability of the plot.

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values representing pressure levels, used to determine appropriate y-axis limits for the pressure display.
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data.
            p0 (float): Reference sea level pressure in Pa used for calculating the minimum pressure corresponding to the maximum height when max_height is provided.
            scale_height (float): Scale height in kilometers used in the barometric formula for converting height to pressure (default: 8.4 km).

        Returns:
            None: Modifies self.ax y-axis limits directly without returning a value.
        """
        assert self.ax is not None, "Axes must be created before setting limits"

        if max_height is not None:
            min_pressure = p0 * np.exp(-max_height / scale_height)
            valid_coords = vertical_coords[vertical_coords >= min_pressure]

            if len(valid_coords) > 0:
                self.ax.set_ylim(np.nanmax(valid_coords), min_pressure)
            else:
                self.ax.set_ylim(np.nanmax(vertical_coords), np.nanmin(vertical_coords))
        else:
            self.ax.set_ylim(np.nanmax(vertical_coords), np.nanmin(vertical_coords))

    def _format_cross_section_axes(
        self: "MPASVerticalCrossSectionPlotter",
        longitudes: np.ndarray,
        vertical_coords: np.ndarray,
        vertical_coord_type: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        max_height: Optional[float] = None,
        distances: Optional[np.ndarray] = None,
        path_lats: Optional[np.ndarray] = None,
    ) -> None:
        """
        This internal method formats the axes of the vertical cross-section plot by setting appropriate labels, limits, and tick formatting based on the longitude range and vertical coordinate type. It configures the x-axis to display longitude values with degree symbols and sets limits based on the provided longitude array. For the y-axis, it applies different formatting depending on whether the vertical coordinate is height (in km), pressure (in hPa or Pa), or model levels. It also adds a text box to indicate the start and end points of the cross-section path. The method includes error handling to ensure that axis formatting does not fail even if certain styling features are unavailable, providing a robust setup for clear visualization of the cross-section data.

        Parameters:
            longitudes (np.ndarray): Array of longitude values along the cross-section path, used to set x-axis limits and formatting.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the y-axis, with formatting determined by the vertical_coord_type.
            vertical_coord_type (str): Type of vertical coordinate ('height_km', 'pressure_hPa', 'pressure', 'height', 'modlev') that dictates y-axis labeling and limits.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees, used for annotating the plot.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees, used for annotating the plot.
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data.
            distances (Optional[np.ndarray]): Along-path distances in km; when given the x-axis shows distance with latitude/longitude on the tick labels instead of longitude (default: None).
            path_lats (Optional[np.ndarray]): Latitudes along the path in degrees for the secondary tick labels (default: None).

        Returns:
            None: Modifies self.ax properties directly without returning a value.
        """
        assert self.ax is not None, "Axes must be created before formatting"

        self._apply_x_axis_formatting(longitudes, distances, path_lats)

        if vertical_coord_type == "height_km":
            self.ax.set_ylabel("Height [km]", fontsize=12)
            y_max = (
                max_height
                if max_height is not None
                else float(np.nanmax(vertical_coords))
            )
            self.ax.set_ylim(0, y_max)
        elif vertical_coord_type == "pressure_hPa":
            self.ax.set_ylabel("Pressure [hPa]", fontsize=12)
            self._set_pressure_ylim(vertical_coords, max_height, p0=1013.25)
            self._setup_pressure_axis(vertical_coords, use_standard_ticks=True)
        elif vertical_coord_type == "pressure":
            self.ax.set_ylabel("Pressure [Pa]", fontsize=12)
            self._set_pressure_ylim(vertical_coords, max_height, p0=101325.0)
            self._setup_pressure_axis(vertical_coords, use_standard_ticks=False)
        elif vertical_coord_type == "height":
            self.ax.set_ylabel("Height [m]", fontsize=12)
            y_max = (
                max_height * 1000 if max_height is not None else vertical_coords.max()
            )
            self.ax.set_ylim(vertical_coords.min(), y_max)
        else:  # modlev
            self.ax.set_ylabel("Model Level", fontsize=12)
            try:
                self.ax.set_ylim(vertical_coords.min(), vertical_coords.max())
            except Exception:
                self.ax.set_ylim(vertical_coords.max(), vertical_coords.min())

        path_info = f"From ({start_point[0]:.1f}°, {start_point[1]:.1f}°) to ({end_point[0]:.1f}°, {end_point[1]:.1f}°)"
        self.ax.text(
            0.02,
            0.98,
            path_info,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    def _get_time_string(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        time_index: int,
    ) -> str:
        """
        This internal method retrieves a formatted time string for the given time index from the MPAS3DProcessor instance. It first checks if the processor has a method to get time information directly; if so, it uses that method. If not, it attempts to access a 'Time' variable in the dataset and convert it to a datetime object for formatting. If both methods fail, it falls back to returning a simple string with the time index. This approach ensures that the plot can display meaningful time information when available while maintaining robustness in cases where time data is not accessible or formatted differently.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): MPAS3DProcessor instance that may contain methods or variables for retrieving time information.
            time_index (int): Time index for which to retrieve the formatted time string.

        Returns:
            str: Formatted time string for display in the plot title or annotations, or a fallback string with the time index if specific time information is not available.
        """
        try:
            dataset = mpas_3d_processor.dataset
            if hasattr(mpas_3d_processor, "get_time_info"):
                return str(mpas_3d_processor.get_time_info(time_index))
            elif (
                dataset is not None
                and hasattr(dataset, "Time")
                and len(dataset.Time) > time_index
            ):
                time_value = pd.to_datetime(dataset.Time.values[time_index])
                return str(time_value.strftime("Valid: %Y-%m-%d %H:%M UTC"))
            else:
                return f"Time Index: {time_index}"
        except Exception:
            return f"Time Index: {time_index}"

    def _resolve_batch_time_str(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        time_idx: int,
    ) -> str:
        """
        This helper method resolves a formatted time string for a given time index during batch processing of cross-section plots. It first checks if the MPAS3DProcessor instance has a method to retrieve time information directly; if so, it uses that method. If not, it attempts to access a 'Time' variable in the dataset and convert it to a datetime object for formatting. If both methods fail, it falls back to returning a simple string with the time index. This method ensures that each plot generated in the batch process can include meaningful time information when available, while maintaining robustness in cases where time data is not accessible or formatted differently.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): MPAS3DProcessor instance that may contain methods or variables for retrieving time information.
            time_idx (int): Time index for which to resolve the formatted time string during batch processing.

        Returns:
            str: Formatted time string for display in the plot title or annotations during batch processing, or a fallback string with the time index if specific time information is not available.
        """
        try:
            dataset = mpas_3d_processor.dataset
            if (
                dataset is not None
                and hasattr(dataset, "Time")
                and len(dataset.Time) > time_idx
            ):
                time_value = pd.to_datetime(dataset.Time.values[time_idx])
                return MPASDateTimeUtils.format_time_tag(time_value)
        except Exception:
            pass
        return f"t{time_idx:03d}"

    def _process_batch_time_step(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        output_dir: str,
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        time_idx: int,
        total_times: int,
        vertical_coord: str,
        num_points: int,
        max_height: Optional[float],
        formats: List[str],
        style: CrossSectionStyle,
        precomputed_levels: Optional[Tuple[np.ndarray, str]] = None,
        display_vertical: Optional[str] = None,
    ) -> List[str]:
        """
        This helper method processes a single time step for creating a vertical cross-section plot in the batch processing workflow. It retrieves a formatted time string for the current time index, constructs a descriptive title for the plot, and calls the create_vertical_cross_section method with the appropriate parameters. After creating the plot, it constructs the output file path based on the variable name, vertical coordinate type, and time information, saves the plot in the specified formats, and returns a list of created file paths. This method encapsulates the logic for handling each time step in the batch process, including error handling and progress updates.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): MPAS3DProcessor instance with loaded 3D data for plotting.
            output_dir (str): Directory where the generated plot will be saved.
            var_name (str): Name of the variable to plot from the dataset.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            time_idx (int): Current time index being processed in the batch workflow.
            total_times (int): Total number of time steps to process, used for progress updates.
            vertical_coord (str): Type of vertical coordinate to use ('pressure', 'height', 'modlev').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.
            max_height (Optional[float]): Maximum height to display in the plot; if None, the full range is used.
            formats (List[str]): List of file formats to save the plot (e.g., ['png', 'pdf']).
            style (CrossSectionStyle): Appearance and file-naming settings (levels, colormap, extend, plot_type, file_prefix).
            precomputed_levels (Optional[Tuple[np.ndarray, str]]): Batch-wide (vertical_levels, vertical_coord_type) pair resolved once by the caller to fix the level count and coordinate type; the actual per-column pressure or height is still sampled separately for every time step.
            display_vertical (Optional[str]): Desired vertical coordinate for display ('pressure', 'height', 'modlev'); if None, vertical_coord is used (default: None).

        Returns:
            List[str]: A list of file paths for the created cross-section plot in the specified formats.
        """
        time_str = self._resolve_batch_time_str(mpas_3d_processor, time_idx)
        path_str = f"({start_point[0]:.1f}°, {start_point[1]:.1f}°) to ({end_point[0]:.1f}°, {end_point[1]:.1f}°)"

        title = f"Vertical Cross-Section: {var_name} | Valid Time: {time_str}\nPath: {path_str}"

        self.create_vertical_cross_section(
            mpas_3d_processor=mpas_3d_processor,
            var_name=var_name,
            start_point=start_point,
            end_point=end_point,
            time_index=time_idx,
            vertical_coord=vertical_coord,
            display_vertical=display_vertical,
            num_points=num_points,
            max_height=max_height,
            title=title,
            style=style,
            precomputed_levels=precomputed_levels,
        )

        # Keep fractional limits distinguishable in filenames (0.5 km -> 'maxh_0p5km') instead of truncating them to '0km'
        height_suffix = (
            f"_maxh_{format(max_height, 'g').replace('.', 'p')}km"
            if max_height is not None
            else ""
        )
        display_suffix = (
            f"_disp_{display_vertical}"
            if display_vertical and display_vertical != vertical_coord
            else ""
        )

        output_path = os.path.join(
            output_dir,
            f"{style.file_prefix}_{var_name}_vcrd_{vertical_coord}{display_suffix}_valid_{time_str}{height_suffix}",
        )

        self.save_plot(output_path, formats=formats)
        self.close_plot()

        if (time_idx + 1) % 5 == 0 or time_idx == 0:
            logger.info(
                "Completed %d/%d cross-sections (time index %d)",
                time_idx + 1,
                total_times,
                time_idx,
            )

        return [f"{output_path}.{fmt}" for fmt in formats]

    def _validate_batch_inputs(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        num_points: int,
    ) -> xr.Dataset:
        """
        This helper method validates the inputs of a batch cross-section run. It requires an MPAS3DProcessor with loaded data that contains the requested variable, applies the cross-section input checks, and requires at least two interpolation points, raising a ValueError describing the first problem found.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): The processor supplying the 3D data.
            var_name (str): Name of the variable to plot from the dataset.
            num_points (int): Number of interpolation points along the cross-section path.

        Returns:
            xr.Dataset: The loaded dataset of the processor.
        """
        if not isinstance(mpas_3d_processor, MPAS3DProcessor):
            raise ValueError("mpas_3d_processor must be an instance of MPAS3DProcessor")

        if mpas_3d_processor.dataset is None:
            raise ValueError(
                "MPAS3DProcessor must have loaded data. Call load_3d_data() first."
            )

        if var_name not in mpas_3d_processor.dataset.data_vars:
            available_vars = list(mpas_3d_processor.dataset.data_vars.keys())
            raise ValueError(
                f"Variable '{var_name}' not found. Available variables: {available_vars[:10]}..."
            )

        self._validate_cross_section_inputs(mpas_3d_processor, var_name)

        if num_points < 2:
            raise ValueError(f"num_points must be at least 2, got {num_points}")

        return mpas_3d_processor.dataset

    def create_batch_cross_section_plots(
        self: "MPASVerticalCrossSectionPlotter",
        mpas_3d_processor: "MPAS3DProcessor",
        output_dir: str,
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        vertical_coord: str = "pressure",
        num_points: int = 100,
        max_height: Optional[float] = None,
        formats: List[str] = ["png"],
        style: Optional[CrossSectionStyle] = None,
        display_vertical: Optional[str] = None,
    ) -> List[str]:
        """
        This method creates vertical cross-section plots for a specified 3D atmospheric variable across all available time steps in the MPAS dataset. It generates a plot for each time step, saving them to the specified output directory with filenames that include the variable name, vertical coordinate type, and valid time information. The method handles various vertical coordinate systems (pressure, height, model levels) and allows for customization of contour levels, colormap, plot type, and maximum height. It includes error handling to ensure that issues with data loading or variable availability are reported clearly, and it provides progress updates during batch processing. The resulting list of created file paths is returned upon completion.

        Parameters:
            mpas_3d_processor ('MPAS3DProcessor'): An instance of MPAS3DProcessor with loaded 3D data for plotting.
            output_dir (str): Directory where the generated plots will be saved.
            var_name (str): Name of the variable to plot from the dataset.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            vertical_coord (str): Type of vertical coordinate to use ('pressure', 'height', 'modlev').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data.
            formats (List[str]): List of file formats to save the plots in (e.g., ['png', 'pdf']).
            style (Optional[CrossSectionStyle]): Appearance and file-naming settings (levels, colormap, extend, plot_type, file_prefix). If None, defaults are used.
            display_vertical (Optional[str]): Desired vertical coordinate for display ('pressure', 'height', 'modlev'), passed through to every frame; if None, vertical_coord is used (default: None).

        Returns:
            List[str]: A list of file paths for the created cross-section plots in the specified formats.
        """
        if style is None:
            style = CrossSectionStyle()

        dataset = self._validate_batch_inputs(mpas_3d_processor, var_name, num_points)

        os.makedirs(output_dir, exist_ok=True)

        time_dim = "Time" if "Time" in dataset.sizes else "time"
        total_times = dataset.sizes[time_dim]

        created_files: List[str] = []
        logger.info(
            "Creating vertical cross-section plots for %d time steps",
            total_times,
        )
        logger.info("Variable: %s", var_name)
        logger.info(
            "Cross-section from (%.2f, %.2f) to (%.2f, %.2f)",
            start_point[0],
            start_point[1],
            end_point[0],
            end_point[1],
        )
        logger.info("Vertical coordinate: %s", vertical_coord)

        if max_height:
            logger.info("Maximum height: %s km", max_height)

        precomputed_levels: Optional[Tuple[np.ndarray, str]] = None

        try:
            batch_levels, batch_coord = self._resolve_vertical_levels(
                mpas_3d_processor, var_name, vertical_coord, 0
            )
            precomputed_levels = (np.asarray(batch_levels), batch_coord)
        except Exception as e:
            logger.warning(
                "Could not resolve vertical levels once for the batch: %s", e
            )

        for time_idx in range(total_times):
            try:
                step_files = self._process_batch_time_step(
                    mpas_3d_processor,
                    output_dir,
                    var_name,
                    start_point,
                    end_point,
                    time_idx,
                    total_times,
                    vertical_coord,
                    num_points,
                    max_height,
                    formats,
                    style,
                    precomputed_levels=precomputed_levels,
                    display_vertical=display_vertical,
                )
                created_files.extend(step_files)
            except Exception as e:
                logger.exception(
                    "Error creating cross-section for time index %d: %s",
                    time_idx,
                    e,
                )

                # Close the partially built figure so failed steps do not leak open figures
                self.close_plot()

        logger.info(
            "Batch cross-section processing completed. Created %d files.",
            len(created_files),
        )

        return created_files
