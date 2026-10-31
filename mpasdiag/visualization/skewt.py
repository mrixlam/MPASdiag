#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Visualization Module: Skew-T Log-P Diagrams

This module provides functionality for creating Skew-T Log-P diagrams from MPAS model output, which are essential tools for analyzing atmospheric soundings and vertical profiles of temperature, humidity, and wind. The module includes the MPASSkewTPlotter class, which specializes in extracting vertical profile data from MPAS datasets, performing necessary unit conversions, and rendering Skew-T diagrams with meteorological conventions. The plotter supports flexible styling options for temperature and dew point profiles, wind barbs, and stability indices, as well as the ability to overlay multiple soundings for comparison. It handles the complexities of unstructured mesh data by utilizing MPASRemapper for interpolation to regular pressure levels when needed. The module also includes error handling to ensure that it can gracefully manage cases where sounding data may be incomplete or missing. Core capabilities include customizable Skew-T diagram styling, support for both individual and multiple soundings, integration with MPASRemapper for data processing, and publication-quality output suitable for atmospheric science research and operational weather analysis.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""

# Load standard libraries
import importlib.util
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Optional, Dict, Any

from .base_visualizer import MPASVisualizer, close_figure_on_error
from .styling import MPASVisualizationStyle
from ..processing.utils_logger import get_logger
from ..processing.utils_path import safe_plot_text

logger = get_logger(__name__)

# MetPy is imported on first use: metpy.plots (in _create_metpy_skewt) costs ~10 s and metpy.calc/metpy.units
# about a second, which every CLI process, multiprocessing forkserver and MPI rank would otherwise pay even
# without a Skew-T. A missing MetPy is still reported when this module is imported.
_METPY_REQUIRED_MSG = (
    "MetPy is required for Skew-T diagrams. "
    "Install it with: pip install metpy  or  "
    "conda install -c conda-forge metpy"
)

if importlib.util.find_spec("metpy") is None:
    raise ImportError(_METPY_REQUIRED_MSG)


def __getattr__(name: str) -> Any:
    """
    This module-level attribute hook imports MetPy the first time the module attributes 'mpcalc' (metpy.calc) or 'munits' (the MetPy unit registry) are read, and stores them as ordinary module attributes so later reads, and tests that patch them, see plain values.

    Parameters:
        name (str): The attribute being read.

    Returns:
        Any: The metpy.calc module or the MetPy unit registry.
    """
    if name not in ("mpcalc", "munits"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import metpy.calc
        from metpy.units import units
    except ImportError as exc:
        raise ImportError(
            f"{_METPY_REQUIRED_MSG} (importing it failed: {exc})"
        ) from exc

    globals().update(mpcalc=metpy.calc, munits=units)
    return globals()[name]


def _metpy_calc() -> Any:
    """
    This helper returns metpy.calc through the module attribute, importing MetPy on first use.

    Returns:
        Any: The metpy.calc module, or whatever a test patched in its place.
    """
    return sys.modules[__name__].mpcalc


def _metpy_units() -> Any:
    """
    This helper returns the MetPy unit registry through the module attribute, importing MetPy on first use.

    Returns:
        Any: The MetPy unit registry, or whatever a test patched in its place.
    """
    return sys.modules[__name__].munits


class MPASSkewTPlotter(MPASVisualizer):
    """Specialized plotter for creating Skew-T Log-P diagrams from MPAS atmospheric sounding data."""

    def __init__(
        self: "MPASSkewTPlotter",
        figsize: Tuple[float, float] = (9, 12),
        dpi: int = 150,
        verbose: bool = True,
    ) -> None:
        """
        This constructor initializes the MPASSkewTPlotter with specified figure size, resolution, and verbosity settings. It calls the base class constructor to set up common visualization parameters and prepares the plotter for creating Skew-T diagrams. The default figure size is set to 9 inches wide by 12 inches tall, which is a common aspect ratio for Skew-T diagrams, and the default resolution is set to 150 DPI for clear output. Verbose mode is enabled by default to provide feedback during the plotting process.

        Parameters:
            figsize (Tuple[float, float]): Size of the figure in inches (width, height).
            dpi (int): Resolution of the figure in dots per inch.
            verbose (bool): Whether to print progress messages during plotting.

        Returns:
            None
        """
        super().__init__(figsize, dpi, verbose)

    @staticmethod
    def _strip_known_extension(save_path: str) -> str:
        """
        This helper removes a trailing image extension ('.png', '.pdf', '.svg', '.jpg', '.jpeg', case-insensitive) from a save path, because the diagram is written in both PNG and PDF and the formats are appended by the save routine; without this a path such as 'sounding.png' would produce 'sounding.png.png' and 'sounding.png.pdf'.

        Parameters:
            save_path (str): User supplied output path, with or without extension.

        Returns:
            str: The path without a known image extension.
        """
        root, ext = os.path.splitext(save_path)
        return (
            root
            if ext.lower() in (".png", ".pdf", ".svg", ".jpg", ".jpeg")
            else save_path
        )

    @staticmethod
    def _location_text(
        location: Optional[Dict[str, Any]],
        indices: Optional[Dict[str, Optional[float]]],
    ) -> Optional[str]:
        """
        This helper builds the one-line description of the requested sounding point and its distance to the model cell actually used, from a profile dictionary carrying 'requested_lon', 'requested_lat' and 'distance_km' (as returned by SoundingDiagnostics.extract_sounding_profile); the same keys are also looked up in the indices dictionary. None is returned when any of the three values is absent or not finite.

        Parameters:
            location (Optional[Dict[str, Any]]): Profile dictionary (or any mapping) with the location keys.
            indices (Optional[Dict[str, Optional[float]]]): Optional indices dictionary searched as a fallback.

        Returns:
            Optional[str]: Text such as 'Requested 97.50\u00b0W, 36.50\u00b0N | nearest cell 4.2 km away', or None.
        """
        values: Dict[str, Any] = {}

        for source in (indices, location):
            if source:
                for key in ("requested_lon", "requested_lat", "distance_km"):
                    if source.get(key) is not None:
                        values[key] = source[key]

        try:
            lon = float(values["requested_lon"])
            lat = float(values["requested_lat"])
            dist = float(values["distance_km"])
        except (KeyError, TypeError, ValueError):
            return None

        if not (np.isfinite(lon) and np.isfinite(lat) and np.isfinite(dist)):
            return None

        lon_text = f"{abs(lon):.2f}\u00b0{'W' if lon < 0 else 'E'}"
        lat_text = f"{abs(lat):.2f}\u00b0{'S' if lat < 0 else 'N'}"
        return f"Requested {lon_text}, {lat_text} | nearest cell {dist:.1f} km away"

    @staticmethod
    def _validated_sorted_profile(
        pressure: np.ndarray, temperature: np.ndarray, dewpoint: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This static helper validates the sounding profile before any figure is created and sorts it by descending pressure so that index 0 is the surface. Empty, length-mismatched, or all-NaN profiles raise a ValueError; NaN pressures sort last.

        Parameters:
            pressure (np.ndarray): Pressure levels in hPa.
            temperature (np.ndarray): Temperature profile in °C.
            dewpoint (np.ndarray): Dew point profile in °C.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The sorted pressure, temperature and dewpoint arrays and the sort order, which can be applied to other per-level arrays such as the winds.
        """
        pressure_arr = np.asarray(pressure, dtype=np.float64)
        temp_arr = np.asarray(temperature, dtype=np.float64)
        dewpoint_arr = np.asarray(dewpoint, dtype=np.float64)

        # Validate the profile before any figure is created
        if pressure_arr.size == 0 or temp_arr.size == 0 or dewpoint_arr.size == 0:
            raise ValueError("Skew-T profile arrays must not be empty")

        if not (pressure_arr.shape == temp_arr.shape == dewpoint_arr.shape):
            raise ValueError(
                "pressure, temperature and dewpoint must have the same shape, got "
                f"{pressure_arr.shape}, {temp_arr.shape} and {dewpoint_arr.shape}"
            )

        if not np.any(np.isfinite(pressure_arr) & np.isfinite(temp_arr)):
            raise ValueError("Skew-T profile has no finite pressure/temperature pair")

        # Sort the profile by descending pressure so index 0 is the surface (parcel lifting and CAPE shading assume this ordering); NaN pressures sort last
        order = np.argsort(
            -np.where(np.isnan(pressure_arr), -np.inf, pressure_arr), kind="stable"
        )
        pressure_arr = pressure_arr[order]
        temp_arr = temp_arr[order]
        dewpoint_arr = dewpoint_arr[order]

        return pressure_arr, temp_arr, dewpoint_arr, order

    @close_figure_on_error
    def create_skewt_diagram(
        self: "MPASSkewTPlotter",
        pressure: np.ndarray,
        temperature: np.ndarray,
        dewpoint: np.ndarray,
        u_wind: Optional[np.ndarray] = None,
        v_wind: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        indices: Optional[Dict[str, Optional[float]]] = None,
        show_parcel: bool = False,
        save_path: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Figure, Axes]:
        """
        This method creates a Skew-T Log-P diagram using the provided pressure, temperature, dew point, and optional wind data. It utilizes MetPy's SkewT class to render the diagram with meteorological conventions, including plotting the temperature and dew point profiles, optional wind barbs, and a lifted parcel profile if requested. The method also adds thermodynamic reference lines and level markers based on provided indices for CAPE/CIN shading. The resulting figure is styled according to MPAS visualization standards and can be saved to a specified path. The method returns the Matplotlib Figure and Axes objects for further customization or display.

        Parameters:
            pressure (np.ndarray): Pressure levels in hPa.
            temperature (np.ndarray): Temperature profile in °C.
            dewpoint (np.ndarray): Dew point profile in °C.
            u_wind (Optional[np.ndarray]): Zonal wind component in knots.
            v_wind (Optional[np.ndarray]): Meridional wind component in knots.
            title (Optional[str]): Title for the Skew-T diagram. Defaults to "MPAS Skew-T Log-P Diagram".
            indices (Optional[Dict[str, Optional[float]]]): Dictionary of thermodynamic indices for CAPE/CIN shading and level markers.
            show_parcel (bool): Whether to plot the lifted parcel profile. Defaults to False.
            save_path (Optional[str]): Path to save the figure (PNG and PDF are written). A trailing known image extension such as '.png' is removed first so the output is 'x.png' and 'x.pdf', never 'x.png.png'. If None, the figure is not saved.
            location (Optional[Dict[str, Any]]): Optional profile dictionary (as returned by SoundingDiagnostics.extract_sounding_profile) carrying 'requested_lon', 'requested_lat' and 'distance_km'; when present a line under the title states the requested point and the distance to the model cell used.

        Returns:
            Tuple[Figure, Axes]: Matplotlib Figure and Axes objects with the Skew-T diagram rendered. Empty, all-NaN or length-mismatched profiles raise a ValueError, the title is sanitised with safe_plot_text, and the figure is closed if any exception escapes.
        """
        pressure_arr, temp_arr, dewpoint_arr, order = self._validated_sorted_profile(
            pressure, temperature, dewpoint
        )

        if u_wind is not None and v_wind is not None:
            u_wind = np.asarray(u_wind, dtype=np.float64)[order]
            v_wind = np.asarray(v_wind, dtype=np.float64)[order]

        fig, ax = self._create_metpy_skewt(
            pressure_arr, temp_arr, dewpoint_arr, u_wind, v_wind, show_parcel, indices
        )

        if title is None:
            title = "MPAS Skew-T Log-P Diagram"

        location_text = self._location_text(location, indices)
        ax.set_title(
            safe_plot_text(title),
            fontsize=14,
            fontweight="bold",
            pad=30 if location_text else 15,
        )

        if location_text:
            ax.text(
                0.5,
                1.012,
                location_text,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        table_height = 0.0
        if indices is not None:
            table_height = self._add_indices_table(fig, indices)

        self.fig = fig
        self.ax = ax

        # Reserve room under the axes for the indices table plus the branding footer
        fig.subplots_adjust(bottom=max(0.18, table_height + 0.07))

        # Fit the temperature window last: the skew shift depends on the final axes shape
        try:
            self._fit_xlim(ax, pressure_arr, temp_arr, dewpoint_arr)
        except (TypeError, ValueError):
            # Fitting the window is cosmetic; keep the default limits if the axes cannot be queried
            pass
        self.add_timestamp_and_branding()

        if save_path:
            MPASVisualizationStyle.save_plot(
                fig,
                self._strip_known_extension(save_path),
                formats=["png", "pdf"],
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=self.dpi,
            )
            if self.verbose:
                logger.info("Skew-T diagram saved to: %s", save_path)

        return fig, ax

    @staticmethod
    def _thin_levels(
        pressure_hpa: np.ndarray, candidates: np.ndarray, min_spacing: float = 40.0
    ) -> np.ndarray:
        """
        This static helper thins a profile for wind-barb plotting: walking from the highest pressure upwards it keeps a candidate level only when it lies at least min_spacing hPa above the last kept level, so a model profile with hundreds of levels yields a readable column of roughly one barb per 40 hPa instead of a solid smear.

        Parameters:
            pressure_hpa (np.ndarray): Pressure of every level in hPa.
            candidates (np.ndarray): Boolean mask of levels eligible to be kept.
            min_spacing (float): Minimum pressure spacing between kept levels in hPa (default: 40.0).

        Returns:
            np.ndarray: Boolean mask of the kept levels (a subset of candidates).
        """
        keep = np.zeros(pressure_hpa.shape, dtype=bool)
        last = np.inf

        for idx in np.argsort(-np.where(candidates, pressure_hpa, -np.inf)):
            if not candidates[idx]:
                break

            if last - pressure_hpa[idx] >= min_spacing:
                keep[idx] = True
                last = pressure_hpa[idx]
            elif not np.isfinite(last):
                keep[idx] = True
                last = pressure_hpa[idx]

        return keep

    @staticmethod
    def _plot_wind_barbs(
        skew: Any,
        pressure_hpa: Any,
        u_wind: Optional[np.ndarray],
        v_wind: Optional[np.ndarray],
    ) -> None:
        """
        This static method plots wind barbs on the Skew-T diagram if both zonal and meridional wind components are provided. It converts the wind components from knots to the appropriate units for plotting and uses MetPy's plot_barbs function to add the wind barbs at the corresponding pressure levels. The method checks for the presence of both u_wind and v_wind before attempting to plot, ensuring that it only adds wind barbs when complete wind data is available. This method is typically called during the Skew-T diagram creation process after plotting the temperature and dew point profiles, and before adding any parcel profiles or thermodynamic overlays.

        Parameters:
            skew (Any): The SkewT object from MetPy on which to plot the wind barbs.
            pressure_hpa (Any): Pressure levels with units for plotting.
            u_wind (Optional[np.ndarray]): Zonal wind component in knots.
            v_wind (Optional[np.ndarray]): Meridional wind component in knots.

        Returns:
            None
        """
        if u_wind is not None and v_wind is not None:
            # Only draw barbs for levels inside the plotted pressure range so none protrude beyond the axes (MetPy already clips barbs to the axes)
            p_vals = np.asarray(pressure_hpa.magnitude, dtype=np.float64)
            in_range = (p_vals >= 100.0) & (p_vals <= 1050.0)
            in_range &= MPASSkewTPlotter._thin_levels(p_vals, in_range)
            u_knots = (
                np.asarray(u_wind, dtype=np.float64)[in_range] * _metpy_units().knots
            )
            v_knots = (
                np.asarray(v_wind, dtype=np.float64)[in_range] * _metpy_units().knots
            )
            skew.plot_barbs(pressure_hpa[in_range], u_knots, v_knots)

    def _plot_parcel_profile(
        self: "MPASSkewTPlotter",
        skew: Any,
        pressure_hpa: Any,
        temp_degc: Any,
        dewpoint_degc: Any,
        indices: Optional[Dict[str, Optional[float]]],
    ) -> None:
        """
        This private method calculates and plots the lifted parcel profile on the Skew-T diagram based on the surface conditions defined by the temperature and dew point at the lowest pressure level. It uses MetPy's parcel_profile function to compute the temperature of a parcel lifted from the surface through the atmosphere, and then plots this profile as a dashed line on the Skew-T diagram. If indices for CAPE and CIN are provided, it also shades the areas of positive CAPE and negative CIN accordingly. The method includes error handling to catch any exceptions that may occur during the parcel profile calculation or plotting, and it will print a warning message if verbose mode is enabled. This method is typically called during the Skew-T diagram creation process after plotting the main temperature and dew point profiles, and before adding any thermodynamic overlays or level markers.

        Parameters:
            skew (Any): The SkewT object from MetPy on which to plot the parcel profile.
            pressure_hpa (Any): Pressure levels with units for plotting.
            temp_degc (Any): Temperature profile with units for plotting.
            dewpoint_degc (Any): Dew point profile with units for plotting.
            indices (Optional[Dict[str, Optional[float]]]): Dictionary of thermodynamic indices for CAPE/CIN shading.

        Returns:
            None
        """
        try:
            lifted_parcel = (
                _metpy_calc()
                .parcel_profile(pressure_hpa, temp_degc[0], dewpoint_degc[0])
                .to("degC")
            )
            skew.plot(pressure_hpa, lifted_parcel, "k--", linewidth=1.5, label="Parcel")
            if indices is not None:
                cape_val = indices.get("cape")
                cin_val = indices.get("cin")
                if cape_val is not None and cape_val > 0:
                    skew.shade_cape(pressure_hpa, temp_degc, lifted_parcel)
                if cin_val is not None and cin_val < 0:
                    # The dewpoint limits the shading to the surface-to-EL layer
                    skew.shade_cin(
                        pressure_hpa, temp_degc, lifted_parcel, dewpoint_degc
                    )
        except Exception as exc:
            if self.verbose:
                logger.warning("Parcel profile plotting failed: %s", exc)

    def _create_metpy_skewt(
        self: "MPASSkewTPlotter",
        pressure_arr: np.ndarray,
        temp_arr: np.ndarray,
        dewpoint_arr: np.ndarray,
        u_wind: Optional[np.ndarray],
        v_wind: Optional[np.ndarray],
        show_parcel: bool,
        indices: Optional[Dict[str, Optional[float]]],
    ) -> Tuple[Figure, Axes]:
        """
        This private method creates the Skew-T diagram using MetPy's SkewT class. It takes the pressure, temperature, dew point, and optional wind data, converts them to the appropriate units, and plots the temperature and dew point profiles on the Skew-T diagram. It also calls helper methods to plot wind barbs, the lifted parcel profile, and thermodynamic overlays such as dry adiabats, moist adiabats, and mixing ratio lines. If indices for CAPE/CIN are provided, it adds level markers for LCL, LFC, and EL. The method sets the axes limits and legend before returning the Figure and Axes objects. This method is typically called by the main create_skewt_diagram method after preparing the input data.

        Parameters:
            pressure_arr (np.ndarray): Pressure levels in hPa.
            temp_arr (np.ndarray): Temperature profile in °C.
            dewpoint_arr (np.ndarray): Dew point profile in °C.
            u_wind (Optional[np.ndarray]): Zonal wind component in knots.
            v_wind (Optional[np.ndarray]): Meridional wind component in knots.
            show_parcel (bool): Whether to plot the lifted parcel profile.
            indices (Optional[Dict[str, Optional[float]]]): Dictionary of thermodynamic indices for CAPE/CIN shading and level markers.

        Returns:
            Tuple[Figure, Axes]: Matplotlib Figure and Axes objects with the Skew-T diagram rendered.
        """
        try:
            from metpy.plots import SkewT
        except ImportError as exc:
            raise ImportError(
                f"{_METPY_REQUIRED_MSG} (importing it failed: {exc})"
            ) from exc

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        skew = SkewT(fig, rotation=45)

        pressure_hpa = pressure_arr * _metpy_units().hPa
        temp_degc = temp_arr * _metpy_units().degC
        dewpoint_degc = dewpoint_arr * _metpy_units().degC

        skew.plot(pressure_hpa, temp_degc, "r", linewidth=2, label="Temperature")
        skew.plot(pressure_hpa, dewpoint_degc, "g", linewidth=2, label="Dewpoint")

        self._plot_wind_barbs(skew, pressure_hpa, u_wind, v_wind)

        if show_parcel:
            self._plot_parcel_profile(
                skew, pressure_hpa, temp_degc, dewpoint_degc, indices
            )

        self._add_thermodynamic_overlays(skew)

        if indices is not None:
            self._add_level_markers(skew.ax, indices)

        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 50)
        skew.ax.set_xlabel("Temperature [\u00b0C]")
        skew.ax.set_ylabel("Pressure [hPa]")
        skew.ax.legend(loc="upper left", fontsize=9)

        return fig, skew.ax

    @staticmethod
    def _fit_xlim(
        ax: Axes,
        pressure_hpa: np.ndarray,
        temp_degc: np.ndarray,
        dewpoint_degc: np.ndarray,
        pad: float = 5.0,
        min_span: float = 90.0,
    ) -> None:
        """
        This static helper widens or shifts the (skewed) temperature axis so the whole temperature and dewpoint profile is visible, instead of the fixed -40 to 50 window that clips cold polar or high-latitude winter profiles. The profile is projected through the axes' own skew transform (so the diagonal shift is accounted for), and the window keeps the default width of 90 degrees unless the profile needs more; it is centred on the profile with a small padding. Warm profiles that already fit keep the default window.

        Parameters:
            ax (Axes): The Skew-T axes with its y-limits already set.
            pressure_hpa (np.ndarray): Pressure of every level in hPa.
            temp_degc (np.ndarray): Temperature in degrees Celsius.
            dewpoint_degc (np.ndarray): Dewpoint in degrees Celsius.
            pad (float): Padding in degrees added on each side of the profile (default: 5.0).
            min_span (float): Minimum width of the temperature window in degrees (default: 90.0).

        Returns:
            None
        """
        p_lo, p_hi = sorted(ax.get_ylim())
        shown = (
            np.isfinite(pressure_hpa) & (pressure_hpa >= p_lo) & (pressure_hpa <= p_hi)
        )

        if not np.any(shown):
            return

        pts_t = np.concatenate([temp_degc[shown], dewpoint_degc[shown]])
        pts_p = np.concatenate([pressure_hpa[shown], pressure_hpa[shown]])
        finite = np.isfinite(pts_t)

        if not np.any(finite):
            return

        pts_t, pts_p = pts_t[finite], pts_p[finite]
        default = ax.get_xlim()

        # Iterate: the skew shift (in data units) depends on the window width
        for _ in range(4):
            lo, hi = ax.get_xlim()
            disp = ax.transData.transform(np.column_stack([pts_t, pts_p]))
            frac = ax.transAxes.inverted().transform(disp)[:, 0]
            effective = lo + frac * (hi - lo)
            need_lo, need_hi = effective.min() - pad, effective.max() + pad

            if lo <= need_lo and need_hi <= hi and (hi - lo) <= min_span + 1e-9:
                break

            span = max(min_span, need_hi - need_lo)
            centre = 0.5 * (need_lo + need_hi)

            # Prefer the default window when the profile already fits in it
            if default[0] <= need_lo and need_hi <= default[1]:
                ax.set_xlim(default)
                break

            ax.set_xlim(centre - 0.5 * span, centre + 0.5 * span)

    @staticmethod
    def _add_thermodynamic_overlays(skew: Any) -> None:
        """
        This static method adds thermodynamic reference lines to the Skew-T diagram, including dry adiabats, moist adiabats, and mixing ratio lines. These overlays are essential for interpreting the stability of the atmosphere and understanding the thermodynamic processes at play. The method uses MetPy's built-in functions to plot these lines with specified styling for clarity. It is typically called during the Skew-T diagram creation process after plotting the main temperature and dew point profiles, and before adding any level markers or indices tables.

        Parameters:
            skew (Any): The SkewT object from MetPy on which to add the thermodynamic overlays.

        Returns:
            None
        """
        skew.plot_dry_adiabats(alpha=0.3, linewidth=0.8)
        skew.plot_moist_adiabats(alpha=0.3, linewidth=0.8)
        skew.plot_mixing_lines(alpha=0.3, linewidth=0.8)

    @staticmethod
    def _add_level_markers(ax: Axes, indices: Dict[str, Optional[float]]) -> None:
        """
        This static method adds horizontal lines to the Skew-T diagram to indicate key pressure levels such as the Lifting Condensation Level (LCL), Level of Free Convection (LFC), and Equilibrium Level (EL). It checks the provided indices dictionary for the presence of these levels, and if they are available and valid, it draws dashed horizontal lines at the corresponding pressure values. Each line is labeled with the level name and pressure value for clarity. This method is typically called during the Skew-T diagram creation process after plotting the main profiles and thermodynamic overlays, and before adding any indices tables.

        Parameters:
            ax (Axes): The Matplotlib Axes object on which to add the level markers.
            indices (Dict[str, Optional[float]]): Dictionary containing the pressure values for LCL, LFC, and EL levels.

        Returns:
            None
        """
        marker_styles = {
            "lcl_pressure": ("LCL", "tab:blue", "--"),
            "lfc_pressure": ("LFC", "tab:orange", "--"),
            "el_pressure": ("EL", "tab:purple", "--"),
        }
        for key, (label, color, linestyle) in marker_styles.items():
            val = indices.get(key)
            if val is not None and np.isfinite(val):
                ax.axhline(
                    val, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.7
                )
                # Axes-fraction x (data y) keeps the label inside the axes, just left of the wind barbs
                ax.text(
                    0.86,
                    val,
                    f"{label} ({val:.0f} hPa)",
                    transform=ax.get_yaxis_transform(),
                    fontsize=8,
                    color=color,
                    va="bottom",
                    ha="right",
                    clip_on=True,
                )

    @staticmethod
    def _add_indices_table(
        fig: Figure,
        indices: Dict[str, Optional[float]],
    ) -> float:
        """
        This static method creates a table of sounding indices and stability parameters to be displayed below the Skew-T diagram. It takes a dictionary of indices, formats them into three columns (Parcel/Instability, Moisture/Thermo, Shear/Severe), and creates a Matplotlib table with styled headers and alternating row colors for readability. The method filters out any indices that are not available (None) and pads the columns to ensure they have equal length before creating the table. This table provides a convenient summary of key atmospheric parameters that can be used to assess the stability and severe weather potential of the sounding. It is typically called during the Skew-T diagram creation process after plotting the main profiles and thermodynamic overlays, and before finalizing the figure for display or saving.

        Parameters:
            fig (Figure): The Matplotlib Figure object on which to add the indices table.
            indices (Dict[str, Optional[float]]): Dictionary containing the sounding indices and stability parameters to be displayed in the table.

        Returns:
            float: Height of the table axes as a fraction of the figure height (0.0 when there is nothing to show), so the caller can reserve matching space below the plot.
        """

        def _fmt(key: str, label: str, unit: str = "") -> str:
            val = indices.get(key)
            if val is None:
                return ""
            # Adding 0.0 turns a rounded -0.0 into 0.0 so the table never shows "-0" or "-0.0"
            if unit:
                return f"{label}: {round(float(val)) + 0.0:.0f} {unit}"
            return f"{label}: {round(float(val), 1) + 0.0:.1f}"

        # Build rows for each column
        col_parcel = [
            _fmt("sbcape", "SBCAPE", "J/kg"),
            _fmt("sbcin", "SBCIN", "J/kg"),
            _fmt("mlcape", "MLCAPE", "J/kg"),
            _fmt("mlcin", "MLCIN", "J/kg"),
            _fmt("mucape", "MUCAPE", "J/kg"),
            _fmt("mucin", "MUCIN", "J/kg"),
            _fmt("lifted_index", "LI"),
            _fmt("lcl_pressure", "LCL", "hPa"),
            _fmt("lfc_pressure", "LFC", "hPa"),
            _fmt("el_pressure", "EL", "hPa"),
        ]

        col_moisture = [
            _fmt("precipitable_water", "PW", "mm"),
            _fmt("wet_bulb_zero_height", "WBZ", "m"),
            _fmt("k_index", "K-Index"),
            _fmt("total_totals", "TT"),
            _fmt("showalter_index", "SI"),
            _fmt("cross_totals", "CT"),
            _fmt("dcape", "DCAPE", "J/kg"),
        ]

        col_shear = [
            _fmt("bulk_shear_0_6km", "0-6 BS", "kt"),
            _fmt("bulk_shear_0_1km", "0-1 BS", "kt"),
            _fmt("srh_0_1km", "0-1 SRH", "m\u00b2/s\u00b2"),
            _fmt("srh_0_3km", "0-3 SRH", "m\u00b2/s\u00b2"),
            _fmt("stp", "STP"),
            _fmt("scp", "SCP"),
            _fmt("sweat_index", "SWEAT"),
        ]

        # Filter out empty strings
        col_parcel = [s for s in col_parcel if s]
        col_moisture = [s for s in col_moisture if s]
        col_shear = [s for s in col_shear if s]

        if not col_parcel and not col_moisture and not col_shear:
            return 0.0

        # Pad columns to equal length
        n_rows = max(len(col_parcel), len(col_moisture), len(col_shear))
        for col in (col_parcel, col_moisture, col_shear):
            col.extend([""] * (n_rows - len(col)))

        headers = ["Parcel / Instability", "Moisture / Thermo", "Shear / Severe"]

        # Size the table axes from the row count (about 0.21 in per row) so every row fits inside the figure
        table_height = min(0.4, (n_rows + 1) * 0.21 / fig.get_figheight())
        table_ax = fig.add_axes((0.05, 0.035, 0.90, table_height))
        table_ax.axis("off")

        cell_text = [
            [col_parcel[i], col_moisture[i], col_shear[i]] for i in range(n_rows)
        ]

        table = table_ax.table(
            cellText=cell_text,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Style header row
        for j in range(3):
            cell = table[0, j]
            cell.set_text_props(weight="bold", fontsize=9)
            cell.set_facecolor("#d9e2ec")
            cell.set_edgecolor("#8fa8c8")

        # Style data rows
        for i in range(1, n_rows + 1):
            for j in range(3):
                cell = table[i, j]
                cell.set_facecolor("#f7f9fb" if i % 2 == 0 else "white")
                cell.set_edgecolor("#c8d6e0")

        return float(table_height + 0.035)
