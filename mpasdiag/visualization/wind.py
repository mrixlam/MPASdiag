#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Visualization Module: Wind Vector Plotting and Overlay

This module defines the `MPASWindPlotter` class, a specialized visualizer for rendering wind vector fields from MPAS model output on cartographic maps using matplotlib and cartopy. The plotter supports various options for customizing the appearance of wind vectors, such as scaling, color coding, and density. It includes automated unit conversion from model output to display units (m/s), flexible map projections via Cartopy, and geographic feature overlays (coastlines, borders, terrain). Visualization outputs include publication-quality single-panel wind vector maps, multi-panel comparison plots for model-observation evaluation, and batch processing capabilities for creating time series of wind analyses with consistent styling and automatic file naming. The class extends the base `MPASVisualizer` to leverage common plotting infrastructure while implementing the specific logic and styling for wind vector visualization, ensuring backward compatibility with the original mpas_analysis module while utilizing modern MPASdiag architecture and best practices for data handling and visualization in Python.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cartopy.feature as cfeature
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
from typing import Optional, Dict, Any, Tuple, Union, List, cast

# Import MPASdiag modules for configuration, data processing, remapping, and visualization
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_geog import GeographicBounds, MPASGeographicUtils
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.visualization.base_visualizer import (
    MS_TO_KNOTS,
    MPASVisualizer,
    WindPlotStyle,
    close_figure_on_error,
    projection_central_longitude,
)
from mpasdiag.visualization.styling import MPASVisualizationStyle
from mpasdiag.processing.utils_logger import get_logger

logger = get_logger(__name__)


class MPASWindPlotter(MPASVisualizer):
    """Specialized class for creating wind vector visualizations from MPAS model output with cartographic presentation."""

    def __init__(
        self: "MPASWindPlotter",
        figsize: Tuple[float, float] = (12, 10),
        dpi: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        This constructor initializes the MPASWindPlotter instance with specified figure dimensions and resolution. It calls the parent class constructor to set up the plotting environment and styling defaults. The figsize parameter allows users to customize the size of the plot canvas in inches, while the dpi parameter controls the resolution of the output image, affecting the quality and detail level of the visualization. By providing these parameters at initialization, users can easily configure the visual appearance of their wind vector plots according to their needs for publication or presentation.

        Parameters:
            figsize (Tuple[float, float]): Figure dimensions in inches as (width, height) tuple (default: (12, 10)).
            dpi (int): Dots per inch for the output figure resolution, affecting quality and detail level (default: 100).
            verbose (bool): Enable verbose informational output, forwarded to MPASVisualizer so every plotter constructor accepts the same keyword (default: True).

        Returns:
            None: Initializes the MPASWindPlotter instance with specified figure size and resolution.
        """
        super().__init__(figsize=figsize, dpi=dpi, verbose=verbose)

    def _clip_to_extent(
        self: "MPASWindPlotter",
        lon: np.ndarray,
        lat: np.ndarray,
        u_data: np.ndarray,
        v_data: np.ndarray,
        bounds: Tuple[float, float, float, float],
        margin_fraction: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal helper keeps only the one-dimensional (unstructured mesh) vectors whose positions lie inside the map extent enlarged by a small margin, so that barbs and arrows are neither computed nor subsampled for cells that cannot be seen. Longitudes are re-expressed in the extent's frame with MPASGeographicUtils.wrap_longitude_to_extent, which makes boxes crossing the dateline (for example 150..210) work, and the wrapped longitudes are returned. The margin (a fraction of the extent size, at least 0.5 degrees) keeps vectors that straddle the edge. Non-finite positions and vectors (NaN or +/-inf) are dropped. Two-dimensional (gridded) input is returned unchanged because its structure must be preserved.

        Parameters:
            lon (np.ndarray): Longitude array in degrees east.
            lat (np.ndarray): Latitude array in degrees north.
            u_data (np.ndarray): U-component wind array.
            v_data (np.ndarray): V-component wind array.
            bounds (Tuple[float, float, float, float]): Map extent as (lon_min, lon_max, lat_min, lat_max) in degrees.
            margin_fraction (float): Margin around the extent as a fraction of its size (default: 0.05).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The clipped longitude, latitude, U and V arrays.
        """
        if lon.ndim != 1:
            return lon, lat, u_data, v_data

        lon_min, lon_max, lat_min, lat_max = bounds
        lon_margin = max(0.5, margin_fraction * (lon_max - lon_min))
        lat_margin = max(0.5, margin_fraction * (lat_max - lat_min))

        wrapped = MPASGeographicUtils.wrap_longitude_to_extent(
            lon, lon_min - lon_margin, lon_max + lon_margin
        )
        keep = (
            np.isfinite(wrapped)
            & np.isfinite(lat)
            & np.isfinite(u_data)
            & np.isfinite(v_data)
            & (wrapped >= lon_min - lon_margin)
            & (wrapped <= lon_max + lon_margin)
            & (lat >= lat_min - lat_margin)
            & (lat <= lat_max + lat_margin)
        )
        return wrapped[keep], lat[keep], u_data[keep], v_data[keep]

    def _prepare_wind_data(
        self: "MPASWindPlotter",
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
        subsample: int = 1,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal helper method prepares the wind data for plotting by converting input data to NumPy arrays if they are xarray DataArrays or dask arrays, applying subsampling to reduce data density for performance optimization, and filtering out any points where either U or V component is NaN to ensure that only valid vectors are plotted. When a map extent is supplied, unstructured (1D) vectors outside it are discarded before subsampling so the stride is spent on visible cells only, and non-finite values (NaN or +/-inf) are always dropped. The method handles both 1D arrays (for unstructured mesh data) and 2D arrays (for regridded data), applying subsampling appropriately while preserving the grid structure for 2D data. For 1D data, it applies stride-based subsampling and then filters out invalid points, while for 2D data, it applies subsampling along both dimensions and returns the resulting arrays without filtering to preserve the grid structure for quiver rendering. The prepared longitude, latitude, U component, and V component arrays are returned ready for use in the plotting functions.

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinate array for vector positions in degrees north.
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s.
            v_data (Union[np.ndarray, xr.DataArray]): V-component wind data array in m/s.
            subsample (int): Subsampling factor to reduce data density for plotting (default: 1, no subsampling).
            bounds (Optional[Tuple[float, float, float, float]]): Map extent as (lon_min, lon_max, lat_min, lat_max); when given, one-dimensional vectors outside the extent (plus a margin, dateline aware) are dropped before subsampling (default: None, keep all).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Prepared longitude, latitude, U component, and V component arrays ready for plotting.
        """
        # Convert input data to NumPy arrays if they are xarray DataArrays or dask arrays
        lon = self.convert_to_numpy(lon)
        lat = self.convert_to_numpy(lat)

        # Convert u and v data to NumPy arrays if they are xarray DataArrays or dask arrays
        u_data = self.convert_to_numpy(u_data)
        v_data = self.convert_to_numpy(v_data)

        # Determine if data is 2D (regridded) or 1D (unstructured mesh)
        is_2d = lon.ndim == 2

        # Clip unstructured vectors to the (margin-extended) extent before subsampling so only visible cells are thinned and drawn
        if bounds is not None and not is_2d:
            lon, lat, u_data, v_data = self._clip_to_extent(
                lon, lat, u_data, v_data, bounds
            )

        # Apply subsampling to reduce data density for performance optimization
        if is_2d:
            if subsample > 1:
                # For 2D data, apply subsampling along both dimensions while preserving grid structure for quiver rendering
                lon_sub = lon[::subsample, ::subsample]
                lat_sub = lat[::subsample, ::subsample]
                u_sub = u_data[::subsample, ::subsample]
                v_sub = v_data[::subsample, ::subsample]
            else:
                # No subsampling, return original 2D arrays
                lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data
            # For 2D data, we typically want to preserve the grid structure even if there are NaN values
            return lon_sub, lat_sub, u_sub, v_sub
        else:
            if subsample > 1:
                # For 1D data, apply stride-based subsampling to reduce the number of points while maintaining representative coverage of the spatial domain
                indices = np.arange(0, len(lon), subsample)
                lon_sub = lon[indices]
                lat_sub = lat[indices]
                u_sub = u_data[indices]
                v_sub = v_data[indices]
            else:
                # No subsampling, return original 1D arrays
                lon_sub, lat_sub, u_sub, v_sub = lon, lat, u_data, v_data

            # Filter out any points where the position or either wind component is not finite (NaN or +/-inf)
            valid_mask = (
                np.isfinite(u_sub)
                & np.isfinite(v_sub)
                & np.isfinite(lon_sub)
                & np.isfinite(lat_sub)
            )
            lon_valid = lon_sub[valid_mask]
            lat_valid = lat_sub[valid_mask]
            u_valid = u_sub[valid_mask]
            v_valid = v_sub[valid_mask]

            # Return the prepared wind data arrays ready for plotting, with subsampling applied and invalid values removed
            return lon_valid, lat_valid, u_valid, v_valid

    def _draw_speed_background(
        self: "MPASWindPlotter",
        lon: np.ndarray,
        lat: np.ndarray,
        u_data: np.ndarray,
        v_data: np.ndarray,
        bounds: Tuple[float, float, float, float],
        colormap: str,
    ) -> None:
        """
        This internal helper draws the wind-speed background behind the vectors, honouring the style's show_background and bg_colormap options for every plot type. For unstructured (1D) input it reuses the base class scatter background, so the look and the vertical wind-speed colorbar match MPASVisualizer.create_wind_plot; the points are the finite, in-extent vectors before any subsampling. For gridded (2D) input, which is what streamlines use, the speed is drawn as a pcolormesh with an identical colorbar. Nothing is drawn when no finite wind speed is available.

        Parameters:
            lon (np.ndarray): Longitudes of the vectors in degrees east (1D mesh points or a 2D grid).
            lat (np.ndarray): Latitudes of the vectors in degrees north (same shape as lon).
            u_data (np.ndarray): U-component wind in m/s.
            v_data (np.ndarray): V-component wind in m/s.
            bounds (Tuple[float, float, float, float]): Map extent as (lon_min, lon_max, lat_min, lat_max) in degrees.
            colormap (str): Colormap name for the speed background.

        Returns:
            None: Adds the background layer and its colorbar to self.ax.
        """
        assert self.ax is not None, "Axes must be created before adding wind background"

        lon_arr = self.convert_to_numpy(lon)
        lat_arr = self.convert_to_numpy(lat)
        speed = np.hypot(self.convert_to_numpy(u_data), self.convert_to_numpy(v_data))
        data_crs = ccrs.PlateCarree()

        if lon_arr.ndim == 1:
            lon_min, lon_max, lat_min, lat_max = bounds
            wrapped = MPASGeographicUtils.wrap_longitude_to_extent(
                lon_arr, lon_min, lon_max
            )
            keep = (
                np.isfinite(speed)
                & np.isfinite(wrapped)
                & (wrapped >= lon_min)
                & (wrapped <= lon_max)
                & (lat_arr >= lat_min)
                & (lat_arr <= lat_max)
            )

            if np.any(keep):
                self._create_wind_background(
                    wrapped[keep],
                    lat_arr[keep],
                    speed[keep],
                    colormap,
                    data_crs,
                    size_scale=4.0,
                )
            return

        if not np.any(np.isfinite(speed)):
            return

        mesh = self.ax.pcolormesh(
            lon_arr,
            lat_arr,
            np.ma.masked_invalid(speed),
            cmap=colormap,
            alpha=0.6,
            transform=data_crs,
            shading="auto",
        )
        colorbar = MPASVisualizationStyle.add_colorbar(
            self.fig,
            self.ax,
            mesh,
            label=MPASVisualizationStyle.build_colorbar_label(
                {"long_name": "Wind Speed", "units": "m s$^{-1}$"}
            ),
            orientation="vertical",
            fraction=0.03,
            pad=0.05,
            shrink=0.8,
            fmt=None,
            labelpad=4,
            label_pos="right",
            tick_labelsize=10,
        )

        if colorbar is not None:
            colorbar.ax.tick_params(labelsize=10)

    def _render_wind_vectors(
        self: "MPASWindPlotter",
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        u_data: np.ndarray,
        v_data: np.ndarray,
        plot_type: str = "barbs",
        color: str = "black",
        scale: Optional[float] = None,
        alpha: float = 1.0,
        speed_colored: bool = True,
    ) -> Optional[Any]:
        """
        This internal helper method renders the wind vectors on the provided Matplotlib axes based on the specified plot type (barbs, arrows, or streamlines). It uses cartopy's PlateCarree transformation for geographic positioning of the vectors. For 'barbs', it renders meteorological barbs that indicate wind speed and direction with flags representing speed thresholds; the m/s input components are converted to knots (x 1.943844) first because barb flags are defined in knots. For 'arrows', it uses quiver to render simple arrows where length and orientation indicate magnitude and direction, with an optional scale factor for arrow length. IVT vector fields have values in the range of hundreds to thousands of kg m⁻¹ s⁻¹, which are far larger than typical wind speeds in m/s. To keep the arrows at a visually sensible length, pass a larger scale value (e.g. 15000) when rendering IVT vectors — the returned Quiver object can then be passed to ax.quiverkey() to add a labeled reference arrow to the plot. For 'streamlines', it requires 2D gridded data to show continuous flow trajectories colored by wind speed, and it validates that the input data is 2D before rendering. The method also includes error handling for unsupported plot types to ensure users are aware of valid options. By centralizing the rendering logic in this method, it allows for consistent styling and rendering of wind vectors across different types of visualizations while leveraging cartopy's capabilities for geographic plotting.

        Parameters:
            ax (Axes): Matplotlib axes on which to render the wind vectors.
            lon (np.ndarray): Longitude array for vector positions in degrees east.
            lat (np.ndarray): Latitude array for vector positions in degrees north.
            u_data (np.ndarray): U-component wind data array in m/s.
            v_data (np.ndarray): V-component wind data array in m/s.
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine rendering method (default: 'barbs').
            color (str): Color for the barbs and arrows (default: 'black'); streamlines keep the speed colormap.
            alpha (float): Opacity of the vectors in [0, 1], applied to barbs, arrows and streamlines (default: 1.0).
            speed_colored (bool): For streamlines, color the lines by wind speed and add the speed colorbar (default: True); when False (a speed background with its own colorbar is drawn) the lines use the given color and no extra colorbar is added.
            scale (Optional[float]): Scale factor for arrow length when plot_type is 'arrows' (default: None, uses automatic scaling of 200). For high-magnitude fields such as IVT (kg m⁻¹ s⁻¹), use a much larger value (e.g. 15000) to keep arrows at a readable length.

        Returns:
            Optional[Any]: The Quiver object returned by ax.quiver() when plot_type is 'arrows', which can be used with ax.quiverkey() to add a labeled scale-reference arrow to the plot. Returns None for 'barbs' and 'streamlines' plot types.
        """
        # Render wind vectors based on the specified plot type using cartopy's PlateCarree transformation for geographic positioning
        if plot_type == "barbs":
            # Use barbs for meteorological convention showing wind speed and direction with flags indicating speed thresholds; barb flags are defined in knots, so convert the m/s components first.
            ax.barbs(
                lon,
                lat,
                np.asarray(u_data) * MS_TO_KNOTS,
                np.asarray(v_data) * MS_TO_KNOTS,
                transform=ccrs.PlateCarree(),
                color=color,
                length=6,
                alpha=alpha,
            )
            return None
        elif plot_type == "arrows":
            if scale is None:
                scale = 200
            # Use quiver for simple arrow representation of wind vectors where length and orientation indicate magnitude and direction.
            # The returned Quiver object can be passed to ax.quiverkey() to add a reference arrow that shows the scale of the vectors.
            # For high-magnitude fields such as IVT (kg m⁻¹ s⁻¹), use a larger scale value (e.g. 15000) to keep arrows at a sensible length.
            quiver_plot = ax.quiver(
                lon,
                lat,
                u_data,
                v_data,
                transform=ccrs.PlateCarree(),
                color=color,
                scale=scale,
                alpha=alpha,
            )
            return quiver_plot
        elif plot_type == "streamlines":
            # Streamlines require 2D gridded data to show continuous flow trajectories colored by wind speed. Validate that input data is 2D and raise error if not.
            if lon.ndim == 1:
                raise ValueError(
                    "Streamlines require gridded data. Use grid_resolution parameter to enable regridding."
                )

            # Identify 1D longitude and latitude arrays from 2D meshgrid for streamplot function
            lon_1d = lon[0, :] if lon.ndim == 2 else lon
            lat_1d = lat[:, 0] if lat.ndim == 2 else lat

            # Calculate wind speed for coloring streamlines by magnitude
            wind_speed = np.sqrt(u_data**2 + v_data**2)

            # Express the grid in a source CRS centered like the map so streamlines do not break at the ±180° seam
            lon_1d, u_data, v_data, wind_speed, src_crs = self._streamline_source_grid(
                ax, lon_1d, u_data, v_data, wind_speed
            )

            # Render streamlines with color mapping based on wind speed, using a colormap suitable for meteorological data
            stream_plot = ax.streamplot(
                lon_1d,
                lat_1d,
                u_data,
                v_data,
                transform=src_crs,
                color=wind_speed if speed_colored else color,
                **({"cmap": "viridis"} if speed_colored else {}),
                linewidth=1.5,
                density=2,
                arrowsize=1.5,
                arrowstyle="->",
                minlength=0.1,
            )
            if alpha < 1.0:
                stream_plot.lines.set_alpha(alpha)
                stream_plot.arrows.set_alpha(alpha)

            # Add a colorbar for streamlines to indicate wind speed values corresponding to streamline colors
            if speed_colored:
                MPASVisualizationStyle.add_colorbar(
                    plt.gcf(),
                    ax,
                    stream_plot.lines,
                    label="Wind Speed [m s$^{-1}$]",
                    orientation="horizontal",
                    fraction=0.03,
                    # Leave room for the longitude tick labels so the colorbar label (placed on top) does not overlap them
                    pad=0.09,
                    shrink=0.8,
                    fmt=None,
                    labelpad=10,
                    label_pos="top",
                    tick_labelsize=10,
                )
            return None
        else:
            # Raise an error for unsupported plot types to ensure users are aware of valid options and prevent silent failures
            raise ValueError(
                f"plot_type must be 'barbs', 'arrows', or 'streamlines', got '{plot_type}'"
            )

    @staticmethod
    def _streamline_source_grid(
        ax: Axes,
        lon_1d: np.ndarray,
        u_data: np.ndarray,
        v_data: np.ndarray,
        speed: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ccrs.CRS]:
        """
        This internal helper prepares a regular lon/lat grid for cartopy's streamplot so that streamlines are continuous across the ±180° dateline. Cartopy regrids streamline input by mapping target points back into the source CRS and interpolating there, so any part of the grid that lies outside the source CRS's [-180, 180] longitude window (e.g. a 150..210 box) or straddles its seam (a global grid on a map centered at 180°) comes back empty and leaves a gap. The helper therefore expresses the grid in a PlateCarree CRS centered on the map's central longitude, reorders the columns so longitudes increase monotonically in that frame, drops the duplicated seam column of a global grid, and repeats the first column at +180° so the grid closes. Grids that already lie within [-180, 180] on a map centered at 0° are returned unchanged, and if re-expressing the grid would make the column spacing uneven the original grid is returned.

        Parameters:
            ax (Axes): The GeoAxes the streamlines are drawn on, used for its central longitude.
            lon_1d (np.ndarray): 1D grid longitudes in degrees, increasing.
            u_data (np.ndarray): 2D U wind component with shape (n_lat, n_lon).
            v_data (np.ndarray): 2D V wind component with shape (n_lat, n_lon).
            speed (np.ndarray): 2D wind speed used to color the streamlines.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ccrs.CRS]: Grid longitudes in the source CRS frame, the reordered U, V and speed arrays, and the source CRS to pass as the streamplot transform.
        """
        projection = getattr(ax, "projection", None)
        center = projection_central_longitude(projection)
        lon_1d = np.asarray(lon_1d, dtype=float)

        if abs(center) < 1e-9 and lon_1d.min() >= -180.0 and lon_1d.max() <= 180.0:
            return lon_1d, u_data, v_data, speed, ccrs.PlateCarree()

        x = ((lon_1d - center + 180.0) % 360.0) - 180.0
        x, order = np.unique(x, return_index=True)
        u_out, v_out, s_out = u_data[:, order], v_data[:, order], speed[:, order]

        steps = np.diff(x)
        if len(steps) == 0 or not np.allclose(steps, steps[0]):
            return lon_1d, u_data, v_data, speed, ccrs.PlateCarree()

        if np.isclose(x[0], -180.0) and np.isclose(x[-1] + steps[0], 180.0):
            x = np.append(x, 180.0)
            u_out = np.concatenate([u_out, u_out[:, :1]], axis=1)
            v_out = np.concatenate([v_out, v_out[:, :1]], axis=1)
            s_out = np.concatenate([s_out, s_out[:, :1]], axis=1)

        return x, u_out, v_out, s_out, ccrs.PlateCarree(central_longitude=center)

    def _regrid_wind_components(
        self: "MPASWindPlotter",
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
        dataset: Optional[xr.Dataset],
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        grid_resolution: float,
        regrid_method: str = "linear",
        config: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal helper method performs regridding of the wind components from the original MPAS mesh to a regular latitude-longitude grid based on the specified geographic extent and grid resolution. It uses the remap_mpas_to_latlon_with_masking function to interpolate the U and V component data onto a regular grid defined by the provided longitude and latitude boundaries and desired grid resolution. The method converts input data to NumPy arrays if they are xarray DataArrays or dask arrays for consistent processing. It applies the specified interpolation method (e.g., 'linear', 'nearest', 'cubic') during regridding to ensure that the resulting gridded data is suitable for rendering with streamlines or quiver plots. The regridded longitude, latitude, U component, and V component arrays are returned as 2D arrays on the regular grid, ready for use in plotting functions that require gridded data.

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Original longitude coordinate array for the MPAS mesh in degrees east.
            lat (Union[np.ndarray, xr.DataArray]): Original latitude coordinate array for the MPAS mesh in degrees north.
            u_data (Union[np.ndarray, xr.DataArray]): Original U-component wind data array on the MPAS mesh in m/s.
            v_data (Union[np.ndarray, xr.DataArray]): Original V-component wind data array on the MPAS mesh in m/s.
            dataset (Optional[xr.Dataset]): Original xarray Dataset containing the MPAS data for metadata reference during regridding (default: None).
            lon_min (float): Minimum longitude of the regridded grid extent in degrees east.
            lon_max (float): Maximum longitude of the regridded grid extent in degrees east.
            lat_min (float): Minimum latitude of the regridded grid extent in degrees north.
            lat_max (float): Maximum latitude of the regridded grid extent in degrees north.
            grid_resolution (float): Desired grid resolution for the regular latitude-longitude grid in degrees (e.g., 0.1 for 0.1° grid).
            regrid_method (str): Interpolation method to use during regridding (e.g., 'linear', 'nearest', 'cubic') (default: 'linear').

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Regridded longitude, latitude, U component, and V component arrays on the regular grid ready for plotting.
        """
        logger.info(
            "Regridding wind components to %s° grid using %s interpolation",
            grid_resolution,
            regrid_method,
        )

        lon_arr = self.convert_to_numpy(lon)
        lat_arr = self.convert_to_numpy(lat)
        u_arr = self.convert_to_numpy(u_data)
        v_arr = self.convert_to_numpy(v_data)

        lon_2d, lat_2d, u_2d = self._interpolate_to_grid(
            lon_arr,
            lat_arr,
            u_arr,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            grid_resolution=grid_resolution,
            dataset=dataset,
            method=regrid_method,
            config=config,
        )
        _, _, v_2d = self._interpolate_to_grid(
            lon_arr,
            lat_arr,
            v_arr,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            grid_resolution=grid_resolution,
            dataset=dataset,
            method=regrid_method,
            config=config,
        )

        return lon_2d, lat_2d, u_2d, v_2d

    def _setup_wind_plot_figure(
        self: "MPASWindPlotter", map_proj: ccrs.Projection
    ) -> Tuple[Figure, GeoAxes]:
        """
        This internal helper method sets up the Matplotlib figure and GeoAxes for wind plotting using a pre-built Cartopy projection object. The projection is created by the shared setup_map_projection factory (which validates the projection name against a whitelist and applies any centering or passthrough keyword arguments), so this method simply creates a Matplotlib figure and GeoAxes with the provided projection, ensuring that the axes is a GeoAxes instance for compatibility with cartographic plotting. By centralizing the figure and axes setup in this method, it allows for consistent configuration of the plotting environment across different types of wind visualizations while leveraging Cartopy's capabilities for geographic projections.

        Parameters:
            map_proj (ccrs.Projection): Pre-built Cartopy projection object (from setup_map_projection) to use for the GeoAxes.

        Returns:
            Tuple[Figure, GeoAxes]: Matplotlib Figure and GeoAxes instances configured with the specified Cartopy projection for wind plotting.
        """
        # Create a matplotlib figure and GeoAxes with the provided projection
        fig, ax = plt.subplots(
            1,
            1,
            figsize=self.figsize,
            dpi=self.dpi,
            subplot_kw={"projection": map_proj},
        )

        # Validate that the created axes is indeed a GeoAxes instance to ensure compatibility
        assert isinstance(ax, GeoAxes), "Axes must be a GeoAxes instance"

        # Return the figure and axes for use in wind plotting
        return fig, ax

    # Upper limit on the number of grid points of an automatically chosen streamline grid
    _MAX_AUTO_STREAM_POINTS = 400_000

    def _handle_streamline_regridding(
        self: "MPASWindPlotter",
        plot_type: str,
        grid_resolution: Optional[float],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        lon: Optional[Union[np.ndarray, xr.DataArray]] = None,
        lat: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> Optional[float]:
        """
        This internal helper method checks if the plot type is 'streamlines' and if no grid resolution is provided, it chooses a default grid resolution for regridding the wind data so that streamlines can be rendered. Streamlines require gridded data to show continuous flow trajectories, so with 'streamlines' and no user resolution regridding is enabled automatically. The default is derived from the map extent and the source mesh rather than a fixed 0.1 degrees (which gave a 3600 x 1800 grid, about 7.5 GB, for a global map): it is the largest of 0.1 degrees, one 720th of the longest extent side, and half the typical spacing of the source cells inside the extent (estimated from their count and the extent area when lon and lat are given), and it is increased further if needed so that the grid never exceeds 400,000 points. A user-supplied resolution is returned unchanged.

        Parameters:
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine if regridding is required for streamlines.
            grid_resolution (Optional[float]): Original grid resolution provided by the user, which may be None if not specified.
            bounds (Optional[Tuple[float, float, float, float]]): Map extent as (lon_min, lon_max, lat_min, lat_max) in degrees used to derive the default resolution (default: None, which keeps the legacy 0.1 degrees).
            lon (Optional[Union[np.ndarray, xr.DataArray]]): Source longitudes, used with lat to estimate the mesh spacing (default: None).
            lat (Optional[Union[np.ndarray, xr.DataArray]]): Source latitudes, used with lon to estimate the mesh spacing (default: None).

        Returns:
            Optional[float]: Updated grid resolution for regridding if plot type is 'streamlines' and no resolution was provided, otherwise returns the original grid resolution.
        """
        # If the plot type is streamlines and no grid resolution is provided, derive a default resolution
        if plot_type == "streamlines" and grid_resolution is None:
            grid_resolution = 0.1

            if bounds is not None:
                lon_min, lon_max, lat_min, lat_max = bounds
                lon_span = float(lon_max - lon_min)
                lat_span = float(lat_max - lat_min)
                candidates = [0.1, max(lon_span, lat_span) / 720.0]

                # Estimate the source mesh spacing from the number of cells inside the extent
                if lon is not None and lat is not None:
                    lon_arr = self.convert_to_numpy(lon)
                    lat_arr = self.convert_to_numpy(lat)

                    if lon_arr.ndim == 1 and lon_arr.shape == lat_arr.shape:
                        wrapped = MPASGeographicUtils.wrap_longitude_to_extent(
                            lon_arr, lon_min, lon_max
                        )
                        inside = (
                            np.isfinite(wrapped)
                            & (wrapped >= lon_min)
                            & (wrapped <= lon_max)
                            & (lat_arr >= lat_min)
                            & (lat_arr <= lat_max)
                        )
                        count = int(np.sum(inside))

                        if count >= 4:
                            mean_lat = float(np.mean(lat_arr[inside]))
                            area = (
                                lon_span
                                * lat_span
                                * max(np.cos(np.radians(mean_lat)), 0.05)
                            )
                            candidates.append(0.5 * float(np.sqrt(area / count)))

                grid_resolution = max(candidates)

                # Cap the total number of grid points
                points = (lon_span / grid_resolution) * (lat_span / grid_resolution)
                if points > self._MAX_AUTO_STREAM_POINTS:
                    grid_resolution *= float(
                        np.sqrt(points / self._MAX_AUTO_STREAM_POINTS)
                    )

            logger.info(
                "Streamlines require gridded data. Auto-enabling regridding with resolution: %.3g°",
                grid_resolution,
            )
        # Return the (possibly updated) grid resolution for use in the plotting workflow
        return grid_resolution

    def _calculate_valid_point_count(
        self: "MPASWindPlotter",
        lon: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        lat: Optional[Union[np.ndarray, xr.DataArray]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> int:
        """
        This internal helper method counts the valid finite wind points that will actually be drawn, which drives the automatic subsampling density. It converts the inputs to NumPy arrays and keeps points where both the longitude and the U component are finite. When the latitude array and the map extent (lon_min, lon_max, lat_min, lat_max) are supplied, only points inside that extent are counted, with longitudes re-expressed in the extent's frame through MPASGeographicUtils.wrap_longitude_to_extent so boxes crossing the dateline count correctly; without them every finite point is counted, as before. Counting only the visible points prevents a small regional map of a global mesh from being thinned as if the whole globe were drawn.

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east.
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s.
            lat (Optional[Union[np.ndarray, xr.DataArray]]): Latitude coordinate array in degrees north, required together with bounds to restrict the count to the extent (default: None).
            bounds (Optional[Tuple[float, float, float, float]]): Map extent as (lon_min, lon_max, lat_min, lat_max) in degrees used to restrict the count (default: None, count all finite points).

        Returns:
            int: Count of valid finite data points (inside the extent when one is given).
        """
        # Convert input data to NumPy arrays if they are xarray DataArrays or dask arrays for consistent processing
        lon_converted = self.convert_to_numpy(lon)
        u_converted = self.convert_to_numpy(u_data)

        # A point is valid when both its longitude and U component are finite
        valid = np.isfinite(lon_converted) & np.isfinite(u_converted)

        # Restrict the count to the map extent when one is provided, wrapping longitudes into the extent's frame for dateline-crossing boxes
        if lat is not None and bounds is not None:
            ext_lon_min, ext_lon_max, ext_lat_min, ext_lat_max = bounds
            lat_converted = self.convert_to_numpy(lat)
            lon_wrapped = MPASGeographicUtils.wrap_longitude_to_extent(
                lon_converted, ext_lon_min, ext_lon_max
            )
            valid = (
                valid
                & (lon_wrapped >= ext_lon_min)
                & (lon_wrapped <= ext_lon_max)
                & (lat_converted >= ext_lat_min)
                & (lat_converted <= ext_lat_max)
            )

        return int(np.sum(valid))

    def _setup_map_extent(
        self: "MPASWindPlotter",
        ax: GeoAxes,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> None:
        """
        This internal helper method sets up the map extent for the GeoAxes based on the provided longitude and latitude boundaries. It determines if the specified boundaries indicate global coverage (e.g., longitude range of 360° and latitude range of 180°) and adjusts the extent accordingly to ensure proper rendering without dateline artifacts in cartopy projections. For global coverage, it slightly adjusts the longitude and latitude boundaries to avoid issues with rendering across the dateline, while for regional plots, it sets the extent directly using the provided boundaries. By centralizing the logic for setting map extent in this method, it ensures consistent handling of geographic extents across different types of wind visualizations while leveraging cartopy's capabilities for geographic plotting.

        Parameters:
            ax (GeoAxes): GeoAxes instance on which to set the map extent.
            lon_min (float): Minimum longitude boundary for the map extent in degrees east.
            lon_max (float): Maximum longitude boundary for the map extent in degrees east.
            lat_min (float): Minimum latitude boundary for the map extent in degrees north.
            lat_max (float): Maximum latitude boundary for the map extent in degrees north.

        Returns:
            None: Sets the map extent on the provided GeoAxes based on the specified longitude and latitude boundaries, with adjustments for global coverage if necessary.
        """
        # Determine if the provided longitude and latitude boundaries indicate global coverage
        is_global_lon = (lon_max - lon_min) >= 359.0
        is_global_lat = (lat_max - lat_min) >= 179.0

        if is_global_lon and is_global_lat:
            ax.set_global()
            logger.debug("Using global extent via set_global() for projection view")
        else:
            # Set the map extent directly using the provided longitude and latitude boundaries for regional plots
            self._apply_map_extent(
                ax, [lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree()
            )

    def _add_map_features(
        self: "MPASWindPlotter",
        ax: GeoAxes,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> None:
        """
        This internal helper method adds standard cartographic features such as coastlines, borders, ocean shading, and land shading to the GeoAxes for enhanced map aesthetics and geographic context. It uses cartopy's built-in features to add these elements with specified styling options (e.g., linewidth, color, alpha) to improve the visual appeal and readability of the wind vector plots. Additionally, it calls another method to add regional features such as lakes, rivers, and urban areas within the specified geographic extent for further detail in the map presentation. By centralizing the logic for adding map features in this method, it ensures consistent styling and inclusion of geographic context across different types of wind visualizations while leveraging cartopy's capabilities for feature rendering.

        Parameters:
            ax (GeoAxes): GeoAxes instance on which to add cartographic features.
            lon_min (float): Minimum longitude boundary for determining regional features in degrees east.
            lon_max (float): Maximum longitude boundary for determining regional features in degrees east.
            lat_min (float): Minimum latitude boundary for determining regional features in degrees north.
            lat_max (float): Maximum latitude boundary for determining regional features in degrees north.

        Returns:
            None: Adds standard cartographic features and regional features to the provided GeoAxes for enhanced map aesthetics and geographic context.
        """
        # Add standard cartographic features such as coastlines, borders, ocean, and land shading for enhanced map aesthetics and geographic context.
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color="gray")
        self._add_ocean_feature(ax, color="lightblue", alpha=0.3)
        ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)

        # Add regional features such as lakes, rivers, and urban areas for enhanced map detail within the specified extent
        self.add_regional_features(lon_min, lon_max, lat_min, lat_max)

    def _generate_wind_title(
        self: "MPASWindPlotter",
        u_valid: np.ndarray,
        v_valid: np.ndarray,
        custom_title: Optional[str],
        level_info: Optional[str],
        time_stamp: Optional[datetime],
        plot_type: Optional[str] = None,
        lon: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> str:
        """
        This internal helper method generates a title string for the wind plot based on the provided U and V component data, custom title, level information, and timestamp. If a custom title is provided, it uses that directly without calculating statistics. Otherwise, it calculates wind speed from the U and V components to compute statistics such as maximum and mean wind speed for inclusion in the title. Statistics ignore non-finite values, and when no finite value exists (for example an all-NaN regridded field) the title reports that there is no valid data instead of raising. The method constructs the title string with a base title, appends level information if provided, and formats the timestamp if available. Finally, it combines all parts into a complete title string that includes wind speed statistics for maximum and mean wind speed, which can be used for annotation in the plot to provide context about the conditions being visualized.

        Parameters:
            u_valid (np.ndarray): Valid U-component wind array used for calculating wind speed statistics.
            v_valid (np.ndarray): Valid V-component wind array used for calculating wind speed statistics.
            custom_title (Optional[str]): Custom title string provided by the user, which takes precedence over calculated statistics if provided.
            level_info (Optional[str]): String containing level information (e.g., "Surface", "850 hPa") to include in the title for context about the vertical level of the wind data.
            time_stamp (Optional[datetime]): Datetime object representing the timestamp of the wind data to include in the title for temporal context.
            plot_type (Optional[str]): Wind vector plot type; when 'barbs' the title notes that barbs are labelled in knots (kt) while the statistics stay in m/s (default: None).
            lon (Optional[np.ndarray]): Longitudes of the vectors (same shape as u_valid); with lat and bounds, the Max/Mean statistics only cover vectors inside the plotted extent, with longitudes re-expressed in the extent's frame so dateline-crossing boxes work (default: None, all vectors).
            lat (Optional[np.ndarray]): Latitudes of the vectors (same shape as u_valid) used together with lon and bounds (default: None).
            bounds (Optional[Tuple[float, float, float, float]]): Plotted extent as (lon_min, lon_max, lat_min, lat_max) in degrees (default: None).

        Returns:
            str: Generated title string for the wind plot, including custom title if provided or calculated statistics with level and time information if not.
        """
        # If a custom title is provided, use it directly without calculating statistics
        if custom_title is not None:
            return custom_title

        # Calculate wind speed from U and V components to compute statistics for the title, ignoring NaN/inf values that would otherwise skew or break the statistics.
        wind_speed = np.sqrt(np.asarray(u_valid) ** 2 + np.asarray(v_valid) ** 2)
        in_extent = np.ones(wind_speed.shape, dtype=bool)

        # Restrict the statistics to the vectors that fall inside the plotted extent
        if lon is not None and lat is not None and bounds is not None:
            lon_arr = np.asarray(lon, dtype=float)
            lat_arr = np.asarray(lat, dtype=float)

            if lon_arr.shape == wind_speed.shape and lat_arr.shape == wind_speed.shape:
                e_lon_min, e_lon_max, e_lat_min, e_lat_max = bounds
                lon_wrapped = MPASGeographicUtils.wrap_longitude_to_extent(
                    lon_arr, e_lon_min, e_lon_max
                )
                in_extent = (
                    (lon_wrapped >= e_lon_min)
                    & (lon_wrapped <= e_lon_max)
                    & (lat_arr >= e_lat_min)
                    & (lat_arr <= e_lat_max)
                )

        wind_speed_finite = wind_speed[np.isfinite(wind_speed) & in_extent]

        if wind_speed_finite.size > 0:
            max_speed = float(np.max(wind_speed_finite))
            mean_speed = float(np.mean(wind_speed_finite))
            stats_line = f"\nMax: {max_speed:.1f} m/s, Mean: {mean_speed:.1f} m/s"
        else:
            # All values are NaN (or the array is empty), so report that there is nothing to summarize
            stats_line = "\nMax: n/a, Mean: n/a (no valid data)"

        # Construct the title string with the base title
        title_parts = ["MPAS Wind Analysis"]

        # Append level information to the title if provided to indicate the vertical level of the wind data being visualized
        if level_info:
            title_parts.append(f"({level_info})")

        # Append formatted timestamp to the title if provided to indicate the time of the wind data being visualized.
        if time_stamp:
            time_str = time_stamp.strftime("%Y-%m-%d %H:%M UTC")
            title_parts.append(f"- {time_str}")

        # Combine the title parts into a single string and append wind speed statistics for maximum and mean wind speed
        title = " ".join(title_parts)
        title += stats_line

        # Barbs are drawn in knots, so state the barb unit alongside the m/s statistics
        if plot_type == "barbs":
            title += " | barbs: kt"

        # Return the complete title string for use in the plot annotation
        return title

    def _print_wind_diagnostics(
        self: "MPASWindPlotter",
        lon_valid: np.ndarray,
        u_valid: np.ndarray,
        v_valid: np.ndarray,
    ) -> None:
        """
        This internal helper method prints diagnostic messages to the console about the plotted wind vectors, including the number of vectors plotted and the range of wind speeds. It calculates wind speed from the U and V components to provide insight into the conditions being visualized. For 2D arrays, it calculates the number of vectors based on the grid dimensions and prints the range of wind speeds while ignoring NaN values to avoid skewing the statistics due to invalid points in the regridded data. For 1D arrays, it calculates the number of valid vectors directly from the length of the valid longitude array and prints the range of wind speeds for all valid points. This diagnostic information can be useful for debugging and understanding the density and characteristics of the wind data being visualized in the plot.

        Parameters:
            lon_valid (np.ndarray): Valid longitude array for the plotted wind vectors in degrees east.
            u_valid (np.ndarray): Valid U-component wind array used for calculating wind speed diagnostics.
            v_valid (np.ndarray): Valid V-component wind array used for calculating wind speed diagnostics.

        Returns:
            None: Prints diagnostic messages about the plotted wind vectors, including the number of vectors and the range of wind speeds, to the console for debugging purposes.
        """
        # Calculate wind speed from U and V components to provide diagnostic information about the plotted wind vectors
        wind_speed = np.sqrt(u_valid**2 + v_valid**2)

        if lon_valid.ndim == 2:
            num_vectors = lon_valid.shape[0] * lon_valid.shape[1]
            wind_speed_valid = wind_speed[np.isfinite(wind_speed)]
            logger.info(
                "Plotted %d wind vectors on %dx%d grid",
                num_vectors,
                lon_valid.shape[0],
                lon_valid.shape[1],
            )

            if len(wind_speed_valid) > 0:
                logger.debug(
                    "Wind speed range: %.1f to %.1f m/s",
                    float(np.min(wind_speed_valid)),
                    float(np.max(wind_speed_valid)),
                )
        else:
            logger.info("Plotted %d wind vectors", len(lon_valid))
            logger.debug(
                "Wind speed range: %.1f to %.1f m/s",
                float(np.min(wind_speed)),
                float(np.max(wind_speed)),
            )

    def _validate_wind_inputs(
        self: "MPASWindPlotter",
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
        plot_type: str = "barbs",
    ) -> None:
        """
        This internal helper checks the wind plot inputs before any figure is created: for streamlines none of the arrays may be empty (an empty field would otherwise surface as an obscure numpy reduction error from deep inside the regridding and stream-plot code; other plot types simply report that there is no valid data), the U and V components must have identical shapes, and per-point longitude/latitude arrays (those of equal shape) must have as many points as the components (one-dimensional grid axes are exempt). Any violation raises a ValueError naming the offending sizes.

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates of the vectors.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates of the vectors.
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data.
            v_data (Union[np.ndarray, xr.DataArray]): V-component wind data.
            plot_type (str): Plot type; empty inputs are rejected only for 'streamlines' (default: 'barbs').

        Returns:
            None
        """
        sizes = {
            "lon": int(np.size(lon)),
            "lat": int(np.size(lat)),
            "u": int(np.size(u_data)),
            "v": int(np.size(v_data)),
        }

        if min(sizes.values()) == 0 and plot_type == "streamlines":
            raise ValueError(
                f"Streamlines need non-empty wind inputs (sizes: {sizes}); nothing to regrid or plot"
            )

        # An all-missing field would regrid to an all-NaN grid and draw an empty streamline map
        if plot_type == "streamlines" and not (
            np.isfinite(np.asarray(u_data, dtype=float)).any()
            and np.isfinite(np.asarray(v_data, dtype=float)).any()
        ):
            raise ValueError(
                "Streamlines need at least one finite u and v value; all wind inputs are missing (NaN/inf)"
            )

        if sizes["u"] != sizes["v"] or np.shape(u_data) != np.shape(v_data):
            raise ValueError(
                f"u and v must have the same shape, got {np.shape(u_data)} and {np.shape(v_data)}"
            )

        # Per-point coordinate arrays must line up with the components (1D grid axes are exempt)
        if np.shape(lon) == np.shape(lat) and sizes["lon"] != sizes["u"]:
            raise ValueError(
                f"lon/lat ({sizes['lon']} points) and u/v ({sizes['u']} points) must have the same number of points"
            )

    @close_figure_on_error
    def create_wind_plot(
        self: "MPASWindPlotter",
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
        bounds: GeographicBounds,
        style: Optional[WindPlotStyle] = None,
        level_info: Optional[str] = None,
        grid_resolution: Optional[float] = None,
        regrid_method: str = "linear",
        dataset: Optional[xr.Dataset] = None,
        config: Optional[Any] = None,
    ) -> Tuple[Figure, Axes]:
        """
        This method creates a wind plot based on the provided longitude, latitude, U and V component data, and various plotting options. It sets up the figure and axes with the specified Cartopy projection, handles regridding for streamlines if necessary, clips unstructured vectors to the map extent (with a margin, dateline aware), prepares the wind data by applying subsampling and filtering out invalid values, sets the map extent and adds features before plotting vectors to ensure proper layering of elements on the map, renders the wind vectors based on the specified plot type (barbs, arrows, or streamlines), generates a title for the plot based on wind speed statistics or a custom title if provided, and prints diagnostic information about the plotted wind vectors. The method returns the Matplotlib figure and GeoAxes containing the wind plot for further customization or saving.

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinate array for vector positions in degrees north.
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s.
            v_data (Union[np.ndarray, xr.DataArray]): V-component wind data array in m/s.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            style (Optional[WindPlotStyle]): Styling settings (wind_level, plot_type, subsample, scale, show_background, bg_colormap, title, time_stamp, projection). If None, defaults are used (with subsample=1). When show_background is True a wind-speed background in bg_colormap with its own colorbar is drawn for barbs, arrows and streamlines (streamlines then use the vector color instead of speed colouring).
            level_info (Optional[str]): String containing level information to include in the title for context about the vertical level of the wind data being visualized (default: None). When omitted and style.wind_level is not 'surface', the wind level label is used.

        Returns:
            Tuple[Figure, Axes]: Matplotlib Figure and GeoAxes instances containing the wind plot for further customization or saving. The timestamp and MPASdiag branding footer is added exactly once by this method, and the figure is closed again if any exception escapes (mismatched or empty u/v/lon/lat inputs raise a ValueError before a figure is created).
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        if style is None:
            style = WindPlotStyle(subsample=1)

        # Validate that the coordinate and component arrays are non-empty and consistent before any figure is created
        self._validate_wind_inputs(lon, lat, u_data, v_data, style.plot_type)

        # Close the figure left over from an earlier call on this plotter so reusing the plotter does not leak figures
        self._release_previous_figure()

        # Fall back to the style's wind level label when the caller gave no explicit level text
        if level_info is None and str(style.wind_level).strip().lower() not in (
            "",
            "surface",
        ):
            level_info = str(style.wind_level)

        plot_type = style.plot_type
        subsample = style.subsample
        scale = style.scale
        title = style.title
        time_stamp = style.time_stamp
        projection = style.projection

        # Build the projection via the shared factory
        map_proj, _ = self.setup_map_projection(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            projection,
            central_longitude=style.central_longitude,
            central_latitude=style.central_latitude,
            proj_kwargs=style.proj_kwargs,
        )
        self.fig, self.ax = self._setup_wind_plot_figure(map_proj)

        # Longitudes may arrive in 0..360 (e.g. lonCell in degrees); the extent filters and the
        # regridding work in [-180, 180]
        lon = MPASGeographicUtils.normalize_longitude(
            np.asarray(self.convert_to_numpy(lon), dtype=float)
        )

        # Keep the source vectors for the title statistics, so the reported max/mean describe the field in the extent rather than the regridded or thinned vectors that are drawn
        stats_fields = tuple(
            np.ravel(self.convert_to_numpy(a)) for a in (lon, lat, u_data, v_data)
        )

        # Auto-enable regridding for streamlines if needed
        grid_resolution = self._handle_streamline_regridding(
            plot_type,
            grid_resolution,
            (lon_min, lon_max, lat_min, lat_max),
            lon,
            lat,
        )

        # Regrid wind components to a regular lat-lon grid if grid_resolution is specified, ensuring that the dataset is available for coordinate extraction and masking
        if grid_resolution is not None:
            lon, lat, u_data, v_data = self._regrid_wind_components(
                lon,
                lat,
                u_data,
                v_data,
                dataset,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                grid_resolution,
                regrid_method,
                config=config,
            )

        # Keep only the unstructured vectors that can be seen (extent plus margin, dateline aware) before any thinning or drawing
        lon_np = self.convert_to_numpy(lon)
        lat_np = self.convert_to_numpy(lat)
        u_np = self.convert_to_numpy(u_data)
        v_np = self.convert_to_numpy(v_data)
        lon, lat, u_data, v_data = self._clip_to_extent(
            lon_np, lat_np, u_np, v_np, (lon_min, lon_max, lat_min, lat_max)
        )

        # If subsample is set to -1, automatically calculate an optimal subsampling factor based on the number of valid points and the map extent
        if subsample == -1:
            num_points = self._calculate_valid_point_count(
                lon, u_data, lat, (lon_min, lon_max, lat_min, lat_max)
            )
            subsample = self.calculate_optimal_subsample(
                num_points=num_points,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                figsize=self.figsize,
                plot_type=plot_type,
                points_in_extent=True,
            )
            logger.debug(
                "Auto-calculated subsample factor: %d (from %d points)",
                subsample,
                num_points,
            )

        # Prepare wind data by applying subsampling and filtering out invalid values, returning only the valid points for plotting.
        lon_valid, lat_valid, u_valid, v_valid = self._prepare_wind_data(
            lon, lat, u_data, v_data, subsample
        )

        # Check if there are valid points to plot after preparation (including NaN-only grids) and print a warning if not
        # For gridded (2D) data the NaN points are kept, so also require at least one finite vector component pair
        if len(lon_valid) == 0 or not np.any(
            np.isfinite(u_valid) & np.isfinite(v_valid)
        ):
            logger.warning("No valid wind data found")
            self.add_timestamp_and_branding()
            return self.fig, self.ax

        # Set map extent and add features before plotting vectors to ensure proper layering of elements on the map
        self._setup_map_extent(self.ax, lon_min, lon_max, lat_min, lat_max)
        self._add_map_features(self.ax, lon_min, lon_max, lat_min, lat_max)

        # Draw the wind-speed background (with its own colorbar) underneath the vectors when requested
        if style.show_background:
            self._draw_speed_background(
                lon,
                lat,
                u_data,
                v_data,
                (lon_min, lon_max, lat_min, lat_max),
                style.bg_colormap,
            )

        # Set color for barbs and arrows; streamlines will use a colormap based on wind speed
        color = style.vector_color

        # For streamlines, color is determined by wind speed and colormap, so we only set color for barbs and arrows.
        self._render_wind_vectors(
            self.ax,
            lon_valid,
            lat_valid,
            u_valid,
            v_valid,
            plot_type,
            color,
            scale,
            style.vector_alpha,
            speed_colored=not style.show_background,
        )

        # Setup gridlines after features to ensure they are on top for better visibility
        self._add_gridlines(ccrs.PlateCarree())

        # Generate and set title with wind statistics or use custom title if provided
        title = self._generate_wind_title(
            stats_fields[2],
            stats_fields[3],
            title,
            level_info,
            time_stamp,
            plot_type,
            lon=stats_fields[0],
            lat=stats_fields[1],
            bounds=(lon_min, lon_max, lat_min, lat_max),
        )
        self.ax.set_title(title, fontsize=12, pad=20)

        # Print diagnostics about the plotted wind vectors for debugging and verification
        self._print_wind_diagnostics(lon_valid, u_valid, v_valid)

        # Add the timestamp and branding footer exactly once so every caller (serial CLI, batch, API) gets it
        self.add_timestamp_and_branding()

        # Return the figure and axes for display or saving
        return self.fig, self.ax

    def _extract_wind_config(
        self: "MPASWindPlotter", wind_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        This internal helper method extracts and normalizes the wind overlay configuration parameters from the provided configuration dictionary. It handles backward compatibility for color specification by checking for both 'color' and 'colors' keys, and it ensures that the subsample parameter is properly converted to an integer with a default value of 1 if conversion fails or if the value is invalid. The method returns a normalized configuration dictionary containing all necessary parameters for creating a wind overlay, including U and V component data, plot type, subsampling factor, color, scale, level index, grid resolution, regridding method, figure size, and original units for potential unit conversion. By centralizing the extraction and normalization of configuration parameters in this method, it allows for consistent handling of wind overlay settings across different parts of the codebase while providing flexibility for users to specify their desired options in a single configuration dictionary.

        Parameters:
            wind_config (Dict[str, Any]): Dictionary containing the wind overlay configuration parameters, which may include keys such as 'u_data', 'v_data', 'plot_type', 'subsample', 'color' or 'colors', 'scale', 'level_index', 'grid_resolution', 'regrid_method', 'figsize', and 'original_units'.

        Returns:
            Dict[str, Any]: Normalized dictionary containing the extracted and validated wind overlay configuration parameters for use in the overlay plotting workflow.
        """
        # Extract subsample parameter with default value of 1, allowing for automatic calculation when set to -1.
        subsample_raw = wind_config.get("subsample", 1)

        # Attempt to convert subsample to an integer, defaulting to 1 if conversion fails or if the value is invalid (e.g., negative or zero).
        try:
            subsample = int(subsample_raw)
        except Exception:
            subsample = 1

        # Extract color with backward compatibility for 'colors' key, defaulting to 'black' if not specified
        color = wind_config.get("color", None)

        # If 'color' is not specified, check for 'colors' key for backward compatibility, and default to 'black' if neither is provided.
        if color is None:
            color = wind_config.get("colors", "black")

        # Return a dictionary with all extracted and validated configuration parameters for use in the overlay plotting workflow
        return {
            "u_data": wind_config["u_data"],
            "v_data": wind_config["v_data"],
            "plot_type": wind_config.get("plot_type", "barbs"),
            "subsample": subsample,
            "color": color,
            "scale": wind_config.get("scale", None),
            "level_index": wind_config.get("level_index", None),
            "grid_resolution": wind_config.get("grid_resolution", None),
            "regrid_method": wind_config.get("regrid_method", "linear"),
            "figsize": wind_config.get("figsize", self.figsize),
            "original_units": wind_config.get("original_units", None),
        }

    def _convert_wind_units(
        self: "MPASWindPlotter",
        u_data: np.ndarray,
        v_data: np.ndarray,
        original_units: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal helper method converts the U and V component wind data from the original units specified in the configuration to the display units used for plotting (typically meters per second, m/s). If the original units are not specified, it attempts to auto-detect based on typical wind speed magnitudes and prints a warning if the data may not be in m/s. If the original units are already the same as the display units, it returns the data unchanged. Otherwise, it uses a UnitConverter utility to perform the conversion and handles any exceptions that may arise during conversion by printing a warning and returning the original data without modification. This method ensures that the wind data is in consistent units for accurate representation in the plot while providing flexibility for users to specify their original data units.

        Parameters:
            u_data (np.ndarray): U-component wind array to be converted.
            v_data (np.ndarray): V-component wind array to be converted.
            original_units (Optional[str]): String specifying the original units of the wind data (e.g., 'm/s', 'km/h', 'knots'); if None, it will attempt to auto-detect based on typical magnitudes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Converted U and V component wind arrays in display units (m/s) for use in the overlay plotting workflow.
        """
        # If original units are not specified, attempt to auto-detect based on typical wind speed magnitudes
        if original_units is None:
            u_mean = np.nanmean(np.abs(u_data))
            if u_mean > 100:
                logger.warning(
                    "Wind data may not be in m/s. Consider specifying 'original_units' in config."
                )
            return u_data, v_data

        # Define the display units for wind speed, which is typically meters per second (m/s) for meteorological data visualization.
        display_units = "m/s"

        # If the original units are already the same as display units, return the data unchanged
        if original_units == display_units:
            return u_data, v_data

        try:
            # Attempt to convert the U and V components from the original units to the display units using the UnitConverter utility
            u_converted = UnitConverter.convert_units(
                u_data, original_units, display_units
            )
            v_converted = UnitConverter.convert_units(
                v_data, original_units, display_units
            )

            # Convert the converted data to NumPy arrays if they are not already, ensuring that the returned data is in a consistent format
            u_data = self.convert_to_numpy(u_converted)
            v_data = self.convert_to_numpy(v_converted)
            logger.debug(
                "Converted overlay wind from %s to %s", original_units, display_units
            )
        except ValueError as e:
            logger.warning(
                "Could not convert overlay wind from %s to %s: %s",
                original_units,
                display_units,
                e,
            )

        # Return the (possibly converted) U and V component arrays for use in the overlay plotting workflow
        return u_data, v_data

    def _extract_2d_wind_slice(
        self: "MPASWindPlotter",
        u_data: np.ndarray,
        v_data: np.ndarray,
        level_index: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal helper method extracts a 2D horizontal slice of the U and V component wind data from potentially 3D arrays based on the specified vertical level index. If the input wind data is already 2D (i.e., has 2 or fewer dimensions), it returns the data directly without slicing. For 3D data, it checks if a level index is provided; if so, it extracts the corresponding horizontal slice at that level index. If no level index is provided, it defaults to model level 0, the lowest level (MPAS orders levels bottom-up), of the 3D arrays for plotting. This method ensures that the wind data is in the correct 2D format for plotting while providing flexibility for users to specify which vertical level to visualize when working with 3D wind data.

        Parameters:
            u_data (np.ndarray): U-component wind data array, which may be 2D or 3D.
            v_data (np.ndarray): V-component wind data array, which may be 2D or 3D.
            level_index (Optional[int]): Integer index specifying which vertical level to extract for 3D data; if None, defaults to model level 0, the lowest level.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 2D horizontal slices of the U and V component wind data for use in the overlay plotting workflow.
        """
        # If the wind data is already 2D (i.e., has 2 or fewer dimensions), return it directly without slicing.
        if getattr(u_data, "ndim", 1) <= 1:
            return u_data, v_data

        # For 3D data, extract the specified level index if provided, otherwise default to level 0 (the lowest model level)
        if level_index is not None:
            return u_data[:, level_index], v_data[:, level_index]

        # If no level index is provided, default to the lowest model level (index 0; MPAS orders levels bottom-up) of the 3D arrays for plotting
        return u_data[:, 0], v_data[:, 0]

    def _calculate_auto_subsample(
        self: "MPASWindPlotter",
        lon: Union[np.ndarray, xr.DataArray],
        u_data: np.ndarray,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        figsize: Tuple[float, float],
        plot_type: str,
        lat: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> int:
        """
        This internal helper method calculates an optimal subsampling factor for the wind overlay based on the number of valid points in the longitude and U component arrays, as well as the map extent and figure size. It first converts the input longitude and U component data to NumPy arrays if they are xarray DataArrays for consistent processing. Then, it counts the number of valid points by checking for finite values in both the longitude and U component arrays that also fall inside the map extent, which is important for determining how many vectors would be plotted without subsampling. Finally, it delegates to the existing calculate_optimal_subsample method to determine the appropriate subsampling factor based on the count of valid points and the map extent, and it logs the calculated subsample factor along with the number of valid points for diagnostics. This method allows for automatic optimization of vector density in the plot while ensuring that performance is maintained when dealing with large datasets.

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east, which may be 1D or 2D.
            u_data (np.ndarray): U-component wind data array in m/s, which may be 2D or 3D.
            lon_min (float): Minimum longitude boundary for the map extent in degrees east.
            lon_max (float): Maximum longitude boundary for the map extent in degrees east.
            lat_min (float): Minimum latitude boundary for the map extent in degrees north.
            lat_max (float): Maximum latitude boundary for the map extent in degrees north.
            figsize (Tuple[float, float]): Figure size in inches (width, height) for calculating subsampling based on the map area.
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine subsampling strategy based on the expected density of vectors for each plot type.
            lat (Optional[Union[np.ndarray, xr.DataArray]]): Latitude coordinate array in degrees north; when given, only points inside the map extent are counted (default: None, count all finite points).

        Returns:
            int: Calculated optimal subsampling factor for the wind overlay based on the number of valid points and map extent, which can be used to reduce the number of plotted vectors for performance optimization.
        """
        # Count the valid points inside the map extent (all finite points when no latitude is supplied)
        num_points = self._calculate_valid_point_count(
            lon, u_data, lat, (lon_min, lon_max, lat_min, lat_max)
        )

        # Delegate to the existing calculate_optimal_subsample method to determine the appropriate subsampling factor
        subsample = self.calculate_optimal_subsample(
            num_points=num_points,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            figsize=figsize,
            plot_type=plot_type,
            points_in_extent=True,
        )

        # Log the calculated subsample factor along with the number of valid points
        logger.debug(
            "Auto-calculated overlay subsample factor: %d (from %d points)",
            subsample,
            int(num_points),
        )

        # Return the calculated subsample factor
        return subsample

    def _validate_and_log_wind_overlay(
        self: "MPASWindPlotter", lon_valid: np.ndarray, lat_valid: np.ndarray
    ) -> bool:
        """
        This internal helper method validates the presence of valid wind data points for the overlay and logs the number of vectors that will be plotted. It checks the dimensions of the valid longitude array to determine how to count valid points, which is important for providing accurate diagnostics about the density of the wind vectors being plotted. For 2D arrays, it counts valid points by checking for finite values in the longitude array across the entire grid, while for 1D arrays it counts valid points directly from the length of the valid longitude array. If no valid points are found, it prints a warning message and returns False to indicate that the overlay cannot be plotted. If valid points are found, it logs the number of wind vectors that will be added as an overlay and returns True to indicate that the overlay can be plotted successfully. This method ensures that users are informed about the presence or absence of valid data for the wind overlay and provides insight into how many vectors will be visualized in the plot.

        Parameters:
            lon_valid (np.ndarray): Valid longitude array for the plotted wind vectors in degrees east, which may be 1D or 2D.
            lat_valid (np.ndarray): Valid latitude array for the plotted wind vectors in degrees north, which may be 1D or 2D.

        Returns:
            bool: True if valid wind data points are present for the overlay and it can be plotted successfully; False if no valid points are found, indicating that the overlay cannot be plotted.
        """
        if lon_valid.ndim == 2:
            # For 2D arrays, count valid points by checking for finite values in the longitude array
            num_vectors = int(np.sum(np.isfinite(lon_valid)))

            # If no valid points are found, print a warning and return False to indicate that the overlay cannot be plotted.
            if num_vectors == 0:
                logger.warning("No valid wind data for overlay")
                return False
            logger.info("Added %d wind vectors as overlay", int(num_vectors))
        else:
            if len(lon_valid) == 0:
                logger.warning("No valid wind data for overlay")
                return False
            logger.info("Added %d wind vectors as overlay", len(lon_valid))
        # If valid points are found, return True to indicate that the overlay can be plotted successfully.
        return True

    def add_wind_overlay(
        self: "MPASWindPlotter",
        ax: Axes,
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        wind_config: Dict[str, Any],
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        dataset: Optional[xr.Dataset] = None,
        config: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        This method adds a wind vector overlay to an existing map axes (typically GeoAxes) based on the provided longitude, latitude, and wind configuration. It extracts and validates the necessary parameters from the wind configuration dictionary, converts the input data to NumPy arrays if they are xarray DataArrays for consistent processing, converts wind component units from original units to display units (m/s) if necessary, extracts a 2D slice from multi-dimensional wind arrays if needed, regrids the wind components to a regular lat-lon grid if requested, calculates an optimal subsampling factor if subsample is set to -1, prepares the wind data by applying subsampling and filtering out invalid values, validates that there are valid points to plot and logs the count before rendering the overlay, and finally renders the wind vectors on the provided axes using the specified plot type and styling from the configuration. This method allows for flexible addition of wind overlays to existing maps while ensuring that the data is properly processed and visualized according to user specifications. For arrow-type overlays the method returns the matplotlib Quiver object so that callers can optionally add a scale-reference arrow via ax.quiverkey() — this is particularly useful for IVT vectors whose magnitudes (kg m⁻¹ s⁻¹) are otherwise difficult to interpret from arrow length alone.

        Parameters:
            ax (Axes): Matplotlib Axes (typically GeoAxes) on which to add the wind vector overlay.
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinate array for vector positions in degrees east, which may be 1D or 2D.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinate array for vector positions in degrees north, which may be 1D or 2D.
            wind_config (Dict[str, Any]): Dictionary containing the wind overlay configuration parameters, which may include keys such as 'u_data', 'v_data', 'plot_type', 'subsample', 'color' or 'colors', 'scale', 'level_index', 'grid_resolution', 'regrid_method', 'figsize', and 'original_units'. For IVT vector overlays, set 'scale' to a large value (e.g. 15000) to compensate for the high-magnitude IVT data and keep arrow lengths visually meaningful.
            lon_min (Optional[float]): Minimum longitude boundary for the map extent in degrees east, required if grid_resolution is specified for regridding (default: None).
            lon_max (Optional[float]): Maximum longitude boundary for the map extent in degrees east, required if grid_resolution is specified for regridding (default: None).
            lat_min (Optional[float]): Minimum latitude boundary for the map extent in degrees north, required if grid_resolution is specified for regridding (default: None).
            lat_max (Optional[float]): Maximum latitude boundary for the map extent in degrees north, required if grid_resolution is specified for regridding (default: None).
            dataset (Optional[xr.Dataset]): MPAS dataset with coordinate information, auto-created from lon/lat if not provided. Required for regridding to ensure that the remapping utilities have access to the necessary coordinate information (default: None).

        Returns:
            Optional[Any]: The matplotlib Quiver object when plot_type is 'arrows', which can be passed directly to ax.quiverkey() to add a labeled reference arrow showing the meaning of arrow length. Returns None for all other plot types or when no valid data points are found.
        """
        # Preserve the MPASConfig object before it is shadowed by the wind config dict below
        mpas_config = config

        # Extract and validate configuration parameters from the provided wind_config dictionary
        config = self._extract_wind_config(wind_config)

        # Effective regrid method: explicit wind_config key > mpas_config.remap_method > 'linear'
        effective_regrid_method = wind_config.get(
            "regrid_method",
            (
                getattr(mpas_config, "remap_method", "linear")
                if mpas_config is not None
                else "linear"
            ),
        )

        # Convert input data to NumPy arrays if they are xarray DataArrays for consistent processing in subsequent steps
        u_data = self.convert_to_numpy(config["u_data"])
        v_data = self.convert_to_numpy(config["v_data"])

        # Convert wind component units from original units to display units (m/s) using the _convert_wind_units helper method
        u_data, v_data = self._convert_wind_units(
            u_data, v_data, config["original_units"]
        )

        # Extract 2D slice from multi-dimensional wind arrays if necessary, using the provided level index or defaulting to the lowest model level (index 0) for 3D data
        u_data, v_data = self._extract_2d_wind_slice(
            u_data, v_data, config["level_index"]
        )

        # Regrid wind components if requested
        if config["grid_resolution"] is not None:
            # Validate that longitude and latitude boundaries are provided for regridding to ensure that the remapping utilities have the necessary information
            if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
                raise ValueError(
                    "lon_min, lon_max, lat_min, lat_max must be provided when grid_resolution is specified"
                )

            # Regrid the wind components to a regular lat-lon grid using the specified resolution and interpolation method
            lon, lat, u_data, v_data = self._regrid_wind_components(
                lon,
                lat,
                u_data,
                v_data,
                dataset,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                config["grid_resolution"],
                effective_regrid_method,
                config=mpas_config,
            )

        # Extract subsample factor from config, allowing for automatic calculation when set to -1.
        subsample = config["subsample"]

        # If subsample is set to -1, automatically calculate an optimal subsampling factor based on the number of valid points and the map extent
        if subsample == -1:
            # Validate that longitude and latitude boundaries are provided for subsample calculation
            if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
                raise ValueError(
                    "lon_min, lon_max, lat_min, lat_max must be provided when subsample=-1"
                )

            # Calculate the optimal subsample factor based on the number of valid points and the map extent to prevent visual clutter in the overlay.
            subsample = self._calculate_auto_subsample(
                lon,
                u_data,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                config["figsize"],
                config["plot_type"],
                lat,
            )

        # Prepare wind data by applying subsampling and filtering out invalid values, returning only the valid points for plotting.
        lon_valid, lat_valid, u_valid, v_valid = self._prepare_wind_data(
            lon, lat, u_data, v_data, subsample
        )

        # Validate that there are valid points to plot and log the count before rendering the overlay.
        if not self._validate_and_log_wind_overlay(lon_valid, lat_valid):
            return None

        # Render wind vectors on the provided axes using the specified plot type and styling from the configuration.
        # For 'arrows' plot type, _render_wind_vectors returns the Quiver object which is propagated back to the
        # caller so it can optionally add a quiverkey reference arrow (e.g. for IVT magnitude labelling).
        return self._render_wind_vectors(
            ax,
            lon_valid,
            lat_valid,
            u_valid,
            v_valid,
            config["plot_type"],
            config["color"],
            config["scale"],
        )

    @staticmethod
    def _valid_time_at(dataset: Any, time_idx: int) -> Optional[datetime]:
        """
        This static helper returns the valid time at a time index as a Python datetime, or None when the dataset has no usable Time coordinate (missing, too short or unparsable), so batch plots can label themselves with the real valid time when it is known.

        Parameters:
            dataset (Any): Dataset that may carry a 'Time' coordinate.
            time_idx (int): Zero-based time index.

        Returns:
            Optional[datetime]: The valid time, or None if it cannot be determined.
        """
        try:
            if hasattr(dataset, "Time") and len(dataset.Time) > time_idx:
                return cast(
                    datetime,
                    pd.Timestamp(dataset.Time.values[time_idx]).to_pydatetime(),
                )
        except Exception:
            pass

        return None

    def create_batch_wind_plots(
        self: "MPASWindPlotter",
        processor: Any,
        output_dir: str,
        bounds: GeographicBounds,
        u_variable: str = "u",
        v_variable: str = "v",
        plot_type: str = "barbs",
        formats: Optional[List[str]] = None,
        subsample: int = 1,
        scale: Optional[float] = None,
        show_background: bool = False,
        grid_resolution: Optional[float] = None,
        regrid_method: str = "linear",
        config: Optional[Any] = None,
    ) -> List[str]:
        """
        This method creates a series of wind plots for each time step in the MPAS dataset using the specified U and V component variable names. It iterates through each time step, extracts the relevant wind data and coordinates, creates a wind plot using the create_wind_plot method with the specified parameters, generates a descriptive filename based on the variable names, plot type, and time information, adds timestamps and branding to the plot, saves the plot in all specified formats, and keeps track of the created file paths. The method returns a list of base file paths (without format extensions) for all successfully created wind plot files corresponding to each time step. This allows users to generate a comprehensive set of wind visualizations across multiple time steps while maintaining consistent styling and organization of output files.

        Parameters:
            processor (Any): An instance of the MPASDataProcessor class that has loaded the dataset containing the U and V component variables for plotting.
            output_dir (str): Directory path where the generated wind plot files will be saved.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            u_variable (str): Name of the U-component variable in the dataset to be used for plotting (default: 'u').
            v_variable (str): Name of the V-component variable in the dataset to be used for plotting (default: 'v').
            plot_type (str): Type of wind vector plot ('barbs', 'arrows', or 'streamlines') to determine rendering method (default: 'barbs').
            formats (Optional[List[str]]): List of file formats (e.g., ['png', 'pdf']) to save the plots in; if None, defaults to ['png'] (default: None).
            subsample (int): Subsampling factor to reduce the number of plotted vectors for performance optimization; if set to -1, it will be automatically calculated based on the number of valid points and map extent (default: 1).
            scale (Optional[float]): Scale factor for arrow length when plot_type is 'arrows' (default: None, uses automatic scaling).
            show_background (bool): Whether to show a background colormap based on wind speed when plot_type is 'streamlines' (default: False).
            grid_resolution (Optional[float]): Grid resolution in degrees for regridding the wind data to a regular lat-lon grid; if None, no regridding is performed (default: None).
            regrid_method (str): Interpolation method to use for regridding the wind data when grid_resolution is specified (e.g., 'nearest', 'linear', 'cubic') (default: 'linear').
            config (Optional[Any]): Additional configuration object that may contain settings for regridding or other processing steps; passed to the regridding utilities if needed (default: None).

        Returns:
            List[str]: A list of full file paths (including the format extension, one per saved format) for all successfully created wind plot files corresponding to each time step, which can be used for reference or further processing.
        """
        # Unpack the geographic extent
        lon_min, lon_max, lat_min, lat_max = bounds

        # Set default formats to ['png']
        if formats is None:
            formats = ["png"]

        # Validate that the processor has a loaded dataset before attempting to access data for plotting
        if not hasattr(processor, "dataset") or processor.dataset is None:
            raise ValueError(
                "Processor has no loaded dataset. Call load_2d_data() first."
            )

        # Access the dataset from the processor to validate variable existence and time dimension for iteration.
        dataset = processor.dataset

        # An empty Time dimension has nothing to plot: report it clearly instead of failing on index 0
        for time_dim_name in ("Time", "time"):
            if time_dim_name in dataset.sizes and dataset.sizes[time_dim_name] == 0:
                raise ValueError(
                    f"Dataset '{time_dim_name}' dimension has no time steps; cannot create wind plots"
                )

        # Validate that the time dimension is properly defined for iteration.
        _, _, time_size = MPASDateTimeUtils.validate_time_parameters(dataset, 0, False)

        # Initialize an empty list to keep track of the base file paths for all created plot files
        created_files: List[str] = []

        for time_idx in range(time_size):
            try:
                # Extract 2D U and V wind component data for the current time index using the processor's data access methods
                u_data = processor.get_2d_variable_data(u_variable, time_idx)
                v_data = processor.get_2d_variable_data(v_variable, time_idx)

                # Extract longitude and latitude coordinates for the current time index using the processor's utility method
                lon, lat = processor.extract_2d_coordinates_for_variable(
                    u_variable, u_data
                )

                # Resolve the real valid time for the current index so the plot title reports it
                time_stamp = self._valid_time_at(dataset, time_idx)

                # Create the wind plot for the current time index using the extracted data and specified parameters
                _, _ = self.create_wind_plot(
                    lon,
                    lat,
                    u_data.values,
                    v_data.values,
                    GeographicBounds(lon_min, lon_max, lat_min, lat_max),
                    style=WindPlotStyle(
                        plot_type=plot_type,
                        subsample=subsample,
                        scale=scale,
                        show_background=show_background,
                        time_stamp=time_stamp,
                    ),
                    grid_resolution=grid_resolution,
                    regrid_method=regrid_method,
                    dataset=dataset,
                    config=config,
                )

                # Attempt to extract time information from the dataset for the current time index to include in the plot title and filename
                try:
                    time_str = MPASDateTimeUtils.get_time_info(
                        dataset, time_idx, var_context="wind", verbose=False
                    )
                except Exception:
                    time_str = f"time_{time_idx}"

                # Generate a descriptive base filename for the plot using variable names, plot type, and time information
                base_name = (
                    f"mpas_wind_{u_variable}_{v_variable}_{plot_type}_valid_{time_str}"
                )
                output_path = os.path.join(output_dir, base_name)

                # Timestamp and branding were already added by create_wind_plot (exactly once), so only save here.
                # Save the plot in all specified formats and close the figure to free memory before the next iteration.
                self.save_plot(output_path, formats=formats)
                self.close_plot()

                # Append the full file path of every saved format, as the surface and precipitation batch functions do
                created_files.extend(
                    MPASVisualizationStyle.saved_file_paths(output_path, formats)
                )
            except Exception as e:
                logger.exception(
                    "Error creating wind plot for time index %d: %s", time_idx, e
                )

                # Close the partially built figure so failed steps do not leak open figures
                self.close_plot()
                continue

        # After processing all time steps, return the list of created file paths for user reference and potential further processing.
        return created_files

    def extract_2d_from_3d_wind(
        self: "MPASWindPlotter",
        u_data_3d: Union[np.ndarray, xr.DataArray],
        v_data_3d: Union[np.ndarray, xr.DataArray],
        level_index: Optional[int] = None,
        level_value: Optional[float] = None,
        pressure_levels: Optional[np.ndarray] = None,
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        This method extracts a 2D horizontal slice of the U and V component wind data from potentially 3D arrays based on the specified vertical level index or pressure level value. If the input wind data is already 2D (i.e., has 2 or fewer dimensions), it returns the data directly without slicing. For 3D data, it checks if a level index is provided; if so, it extracts the corresponding horizontal slice at that level index. If no level index is provided but a specific pressure level value is given along with the corresponding pressure levels array, it finds the index of the nearest pressure level to the specified value and extracts that slice. If neither level selection parameter is provided, it defaults to model level 0, the lowest level (MPAS orders levels bottom-up), of the 3D arrays for plotting. This method ensures that the wind data is in the correct 2D format for plotting while providing flexibility for users to specify which vertical level to visualize when working with 3D wind data, whether by index or by pressure level.

        Parameters:
            u_data_3d (Union[np.ndarray, xr.DataArray]): U-component wind data array, which may be 2D or 3D.
            v_data_3d (Union[np.ndarray, xr.DataArray]): V-component wind data array, which may be 2D or 3D.
            level_index (Optional[int]): Integer index specifying which vertical level to extract for 3D data; if None, defaults to model level 0, the lowest level.
            level_value (Optional[float]): Specific pressure level value to extract for 3D data; if provided, it will find the nearest pressure level index based on the pressure_levels array.
            pressure_levels (Optional[np.ndarray]): Array of pressure levels corresponding to the vertical dimension of the wind data, required if level_value is provided to find the nearest pressure level index.

        Returns:
            Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]: 2D horizontal slices of the U and V component wind data for use in the overlay plotting workflow, returned as either NumPy arrays or xarray DataArrays depending on the input type.
        """
        # If the input wind data is already 2D (i.e., has 2 or fewer dimensions), return it directly without slicing.
        if level_index is not None:
            return u_data_3d[:, level_index], v_data_3d[:, level_index]

        # If a specific pressure level value is provided along with the corresponding pressure levels array, find the index of the nearest pressure level
        if level_value is not None and pressure_levels is not None:
            level_idx = np.argmin(np.abs(pressure_levels - level_value))
            return u_data_3d[:, level_idx], v_data_3d[:, level_idx]

        # Return the lowest model level (index 0; MPAS orders levels bottom-up) of the 3D arrays for plotting if no specific level selection parameters are provided, as a default behavior for multi-level data.
        return u_data_3d[:, 0], v_data_3d[:, 0]

    def compute_wind_speed_and_direction(
        self: "MPASWindPlotter",
        u_data: Union[np.ndarray, xr.DataArray],
        v_data: Union[np.ndarray, xr.DataArray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method computes the wind speed and meteorological direction from the U and V component wind data. Wind speed is calculated as the vector magnitude of the U and V components using the Pythagorean theorem, while wind direction is calculated in degrees using the arctan2 function to determine the angle of the wind vector relative to the north direction. The method converts mathematical angles to meteorological convention where 0° corresponds to north, 90° to east, 180° to south, and 270° to west. The resulting wind speed and direction arrays are returned for use in visualization or further analysis. This method allows users to derive meaningful meteorological information from the raw U and V component data, enabling more informative visualizations of wind patterns.

        Parameters:
            u_data (Union[np.ndarray, xr.DataArray]): U-component wind data array in m/s, which may be 2D or 3D.
            v_data (Union[np.ndarray, xr.DataArray]): V-component wind data array in m/s, which may be 2D or 3D.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the computed wind speed array and wind direction array in degrees, both returned as NumPy arrays for consistent processing in visualization workflows.
        """
        # Calculate wind speed as the vector magnitude of the U and V components using the Pythagorean theorem.
        wind_speed = np.sqrt(u_data**2 + v_data**2)

        # Calculate wind direction in degrees using arctan2, converting from mathematical angles to meteorological convention
        wind_direction = np.arctan2(v_data, u_data) * 180 / np.pi

        # Convert mathematical angles to meteorological direction where 0° is north, 90° is east, etc.
        wind_direction = (270 - wind_direction) % 360

        # Return the computed wind speed and direction arrays
        return wind_speed, wind_direction
