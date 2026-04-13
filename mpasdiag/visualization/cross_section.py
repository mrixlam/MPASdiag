#!/usr/bin/env python3

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
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from math import radians, degrees, sin, cos, atan2, sqrt, asin
from scipy.spatial import KDTree 
from typing import Tuple, Optional, List, Dict, Any, Union, cast
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .base_visualizer import MPASVisualizer
from .styling import MPASVisualizationStyle
from ..processing.processors_3d import MPAS3DProcessor
from ..processing.utils_unit import UnitConverter
from ..processing.utils_metadata import MPASFileMetadata


class MPASVerticalCrossSectionPlotter(MPASVisualizer):
    """ Specialized plotter for creating vertical cross-section visualizations of 3D MPAS atmospheric data along user-defined transects through the atmosphere. """
    
    def __init__(self: 'MPASVerticalCrossSectionPlotter', 
                 figsize: Tuple[float, float] = (10, 12), 
                 dpi: int = 100) -> None:
        """
        This constructor initializes the MPASVerticalCrossSectionPlotter instance with specified figure dimensions and resolution. It calls the parent class constructor to set up the base visualization environment, allowing for consistent styling and branding across different plot types. The figsize parameter determines the size of the output figure in inches, while the dpi parameter controls the resolution of the saved figure for high-quality output. This setup ensures that the vertical cross-section plots are rendered with appropriate dimensions and clarity for analysis and presentation purposes. 

        Parameters:
            figsize (Tuple[float, float]): Figure size in inches (width, height) for the cross-section plot (default: (10, 12)).
            dpi (int): Dots per inch resolution for saved figure output (default: 100).

        Returns:
            None
        """
        super().__init__(figsize, dpi)

    def _validate_cross_section_inputs(self: 'MPASVerticalCrossSectionPlotter',
                                       mpas_3d_processor: Any, 
                                       var_name: str) -> None:
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
            raise ValueError("MPAS3DProcessor must have loaded data. Call load_3d_data() first.")

        if var_name not in mpas_3d_processor.dataset.data_vars:
            available_vars = list(mpas_3d_processor.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars[:10]}...")

        var_dims = mpas_3d_processor.dataset[var_name].sizes

        if 'nVertLevels' not in var_dims and 'nVertLevelsP1' not in var_dims:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")

    def _convert_and_clip_data(self: 'MPASVerticalCrossSectionPlotter',
                               data_values: Union[np.ndarray, xr.DataArray, float],
                               var_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        This internal method performs unit conversion and physical clipping on the data values for the specified variable. It retrieves the original units from the variable metadata and determines the appropriate display units using the UnitConverter utility. If a conversion is needed, it applies the conversion to the data values and updates the metadata accordingly. Additionally, if the variable is identified as a moisture variable (e.g., specific humidity, mixing ratio), it checks for any negative values in the data, which are physically invalid, and clips them to zero while issuing a warning with the count of negative values and their range. This method ensures that the data values are in the correct units for display and that any physically unrealistic values are handled appropriately before plotting.
        Accepts np.ndarray, xr.DataArray, or float as input for data_values.

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
            original_units = metadata.get('units', '')
            display_units = UnitConverter.get_display_units(var_name, original_units)

            if original_units != display_units and original_units:
                print(f"Converting {var_name} from {original_units} to {display_units}")
                data_values = UnitConverter.convert_units(data_values, original_units, display_units)
                metadata['units'] = display_units
                print(f"Data range after conversion: {np.nanmin(data_values):.4f} to {np.nanmax(data_values):.4f} {display_units}")
            else:
                print(f"No unit conversion needed for {var_name} (units: {original_units})")
        except Exception as e:
            print(f"Warning: Unit conversion failed for {var_name}: {e}")
            metadata = {'units': '', 'long_name': var_name}

        moisture_vars = ['q2', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qv2m', 'humidity', 'mixing_ratio']

        if any(mv in var_name.lower() for mv in moisture_vars):
            n_negative = np.sum(data_values < 0)
            if n_negative > 0:
                print(f"Warning: Found {n_negative:,} negative {var_name} values (min: {np.nanmin(data_values):.4f}). Clipping to 0 (physically invalid).")
                data_values = np.clip(data_values, 0, None)

        return np.asarray(data_values), metadata

    def _resolve_vertical_display(self: 'MPASVerticalCrossSectionPlotter',
                                  vertical_coords: np.ndarray,
                                  vertical_coord_type: str,
                                  desired_display: str,
                                  mpas_3d_processor: MPAS3DProcessor,
                                  time_index: int) -> Tuple[np.ndarray, str]:
        """
        This internal method resolves the vertical coordinate values for display based on the desired vertical coordinate type. It accepts the original vertical coordinates and their type, along with the desired display type (e.g., 'height', 'pressure', 'modlev'). Depending on the desired display, it performs the necessary conversions using the MPAS3DProcessor methods to convert model levels to height or pressure as needed. If the desired display is 'pressure' but the vertical coordinates contain non-positive or non-finite values, it issues a warning and falls back to displaying model levels. The method returns the vertical coordinates formatted for display and a string indicating the type of vertical coordinate being displayed.

        Parameters:
            vertical_coords (np.ndarray): The original vertical coordinate values extracted from the dataset.
            vertical_coord_type (str): The type of the original vertical coordinates (e.g., 'pressure', 'height', 'modlev').
            desired_display (str): The desired vertical coordinate type for display ('pressure', 'height', 'modlev').
            mpas_3d_processor (MPAS3DProcessor): The processor instance used for any necessary conversions.
            time_index (int): The time index for selecting temporal data if needed for conversions. 

        Returns:
            Tuple[np.ndarray, str]: The vertical coordinates formatted for display and the type of vertical coordinate being displayed. 
        """
        # DEBUG: trace vertical display resolution
        print(f"[DEBUG _resolve_vertical_display] desired_display='{desired_display}', vertical_coord_type='{vertical_coord_type}'")
        print(f"[DEBUG _resolve_vertical_display] vertical_coords shape={np.asarray(vertical_coords).shape}, min={np.nanmin(vertical_coords):.4f}, max={np.nanmax(vertical_coords):.4f}")

        if desired_display == 'height':
            result_coords, result_type = self._convert_vertical_to_height(
                vertical_coords, vertical_coord_type, mpas_3d_processor, time_index
            )
            print(f"[DEBUG _resolve_vertical_display] after height conversion: type='{result_type}', min={np.nanmin(result_coords):.4f}, max={np.nanmax(result_coords):.4f}")
            return result_coords, result_type

        if desired_display == 'pressure':
            try:
                v = vertical_coords.astype(float).copy()
            except Exception:
                v = np.asarray(vertical_coords, dtype=float)
            if not np.all(np.isfinite(v)) or np.nanmin(v) <= 0:
                print("Warning: vertical coordinates contain non-positive or non-finite values; cannot display as pressure. Falling back to model levels.")
                return np.arange(len(vertical_coords), dtype=float), 'modlev'
            return (v / 100.0 if np.nanmax(v) > 10000 else v), 'pressure_hPa'

        if desired_display == 'modlev':
            return vertical_coords, 'modlev'

        return self._convert_vertical_to_height(
            vertical_coords, vertical_coord_type, mpas_3d_processor, time_index
        )

    def _apply_max_height_filter(self: 'MPASVerticalCrossSectionPlotter',
                                 vertical_display: Union[np.ndarray, xr.DataArray, float],
                                 vertical_coord_display: str,
                                 data_values: Union[np.ndarray, xr.DataArray, float],
                                 max_height: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal method applies a maximum height filter to the vertical coordinates and corresponding data values for display. It checks if the vertical coordinate being displayed is in height units (e.g., 'height_km') and if a max_height value is provided. If so, it creates a boolean mask to filter out any vertical levels that exceed the specified maximum height. The method then returns the filtered vertical coordinates and data values for plotting. If no filtering is applied (e.g., if the vertical coordinate is not in height units or if max_height is None), it returns the original vertical coordinates and data values without modification.

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

        if vertical_coord_display != 'height_km' or max_height is None:
            return vertical_display, data_values
        try:
            mask = np.asarray(vertical_display) <= float(max_height)
            if np.any(mask):
                return np.asarray(vertical_display)[mask], np.asarray(data_values)[mask, :]
            print("Warning: No vertical levels are below the requested max_height; showing full range")
        except Exception:
            pass

        return vertical_display, data_values

    def _resolve_plot_style(self: 'MPASVerticalCrossSectionPlotter', 
                            var_name: str,
                            data_values: Union[np.ndarray, xr.DataArray, float],
                            colormap: Optional[Union[str, mcolors.Colormap]],
                            levels: Optional[np.ndarray]) -> Tuple[Any, np.ndarray]:
        """
        This internal method resolves the plot style for the cross-section visualization by determining the appropriate colormap and contour levels based on the variable name and data values. It first checks if the data values are in a compatible format (np.ndarray or xr.DataArray) and converts them to np.ndarray if necessary. If both colormap and levels are provided, it returns them directly. Otherwise, it attempts to retrieve variable-specific styling information from the MPASVisualizationStyle class using a dummy DataArray with the same shape as the data values. If styling information is available, it uses the specified colormap and levels; if not, it falls back to default values (e.g., 'viridis' for colormap and automatically determined levels based on data range). This method ensures that the plot is styled appropriately for the variable being visualized while allowing for user overrides when desired.

        Parameters:
            var_name (str): The name of the variable being plotted, used to determine styling.
            data_values (Union[np.ndarray, xr.DataArray, float]): The data values for the variable, used to determine contour levels if not provided.
            colormap (Optional[Union[str, mcolors.Colormap]]): An optional colormap specified by the user; if None, a default or variable-specific colormap will be used.
            levels (Optional[np.ndarray]): An optional array of contour levels specified by the user; if None, levels will be automatically determined based on data range and variable type.

        Returns:
            Tuple[Any, np.ndarray]: The resolved colormap and contour levels for plotting the cross-section.
        """
        if isinstance(data_values, xr.DataArray):
            data_values = data_values.values
        elif not isinstance(data_values, np.ndarray):
            data_values = np.asarray(data_values)

        if colormap is not None and levels is not None:
            return colormap, levels

        try:
            dummy_data = xr.DataArray(data_values, dims=['level', 'distance'], name=var_name)
            style = MPASVisualizationStyle.get_variable_style(var_name, dummy_data)

            if colormap is None:
                colormap = style.get('colormap', 'viridis')

            if levels is None:
                levels = style.get('levels', self._get_default_levels(data_values, var_name))
        except Exception:
            if colormap is None:
                colormap = 'viridis'

            if levels is None:
                levels = self._get_default_levels(data_values, var_name)

        if levels is None:
            levels = self._get_default_levels(data_values, var_name)

        return colormap, levels

    def _render_cross_section_plot(self: 'MPASVerticalCrossSectionPlotter',
                                   X: np.ndarray,
                                   Y: np.ndarray,
                                   data_values: np.ndarray,
                                   colormap: Any,
                                   levels: np.ndarray,
                                   extend: str,
                                   plot_type: str,
                                   metadata: Dict[str, Any],
                                   var_name: str,
                                   **kwargs: Any) -> None:
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

        if plot_type == 'contourf':
            cs = self.ax.contourf(X, Y, data_values, levels=levels, cmap=colormap, extend=extend, **kwargs)
            cs_lines = self.ax.contour(X, Y, data_values, levels=levels, colors='black', linewidths=0.5, alpha=0.6)
            self.ax.clabel(cs_lines, inline=True, fontsize=8, fmt='%.1f')
        elif plot_type == 'contour':
            cs = self.ax.contour(X, Y, data_values, levels=levels, cmap=colormap, **kwargs)
            self.ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        elif plot_type == 'pcolormesh':
            cs = self.ax.pcolormesh(X, Y, data_values, cmap=colormap, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'contourf', 'contour', or 'pcolormesh'")
        
        if plot_type in ['contourf', 'pcolormesh']:
            label = (
                f"{metadata.get('long_name', var_name)} [{metadata.get('units', '')}]"
                if metadata.get('units', '') else metadata.get('long_name', var_name)
            )

            assert self.fig is not None, "Figure must be initialized before adding a colorbar"

            MPASVisualizationStyle.add_colorbar(
                self.fig, self.ax, cs,
                label=label,
                orientation='vertical', fraction=0.03, pad=0.05, shrink=0.8,
                fmt=None, labelpad=4, label_pos='right', tick_labelsize=10
            )

    def _generate_cross_section_title(self: 'MPASVerticalCrossSectionPlotter',
                                      mpas_3d_processor: MPAS3DProcessor,
                                      var_name: str,
                                      time_index: int,
                                      title: Optional[str]) -> str:
        """
        This internal method generates a title for the vertical cross-section plot based on the variable name and time information. If a custom title is provided, it returns that title directly. Otherwise, it attempts to retrieve a time string from the MPAS3DProcessor using the specified time index and formats a default title that includes the variable name and valid time. If there is an issue retrieving the time information, it falls back to a simpler title that only includes the variable name. This method ensures that the plot has an informative title that provides context about the variable being visualized and the time of the data, while also allowing for user customization when desired.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): The processor instance used to retrieve time information for the title.
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

    def _save_cross_section(self: 'MPASVerticalCrossSectionPlotter',
                            save_path: str) -> None:
        """
        This internal method saves the generated vertical cross-section figure to the specified file path. It checks that the figure has been initialized before attempting to save. The method determines the appropriate save parameters based on the file extension (e.g., using a lower compression level for PNG files) and saves the figure with the specified resolution and bounding box settings. After saving, it prints a confirmation message with the path to the saved figure.

        Parameters:
            save_path (str): The file path where the figure should be saved, including the file extension (e.g., 'cross_section.png').

        Returns:
            None
        """
        save_kwargs: Dict[str, Any] = {'dpi': self.dpi, 'bbox_inches': 'tight'}

        if save_path.lower().endswith('.png'):
            save_kwargs['pil_kwargs'] = {'compress_level': 1}

        assert self.fig is not None, "Figure must be initialized before saving"

        self.fig.savefig(save_path, **save_kwargs)
        print(f"Vertical cross-section saved to: {save_path}")

    def create_vertical_cross_section(self: 'MPASVerticalCrossSectionPlotter', 
                                      mpas_3d_processor: Any, 
                                      var_name: str, 
                                      start_point: Tuple[float, float], 
                                      end_point: Tuple[float, float], 
                                      time_index: int = 0, 
                                      vertical_coord: str = 'pressure', 
                                      display_vertical: Optional[str] = None, 
                                      num_points: int = 100, 
                                      levels: Optional[np.ndarray] = None, 
                                      colormap: Optional[Union[str, mcolors.Colormap]] = None, 
                                      extend: str = 'both', 
                                      plot_type: str = 'contourf', 
                                      save_path: Optional[str] = None, 
                                      title: Optional[str] = None, 
                                      max_height: Optional[float] = None, 
                                      **kwargs: Any) -> Tuple[Figure, Axes]:
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
            levels (Optional[np.ndarray]): Array of contour levels to use for plotting; if None, levels are automatically determined based on data range and variable type (default: None).
            colormap (Optional[Union[str, mcolors.Colormap]]): Colormap to use for plotting; if None, a default colormap is selected based on variable type and styling rules (default: None).
            extend (str): Contour extend option for out-of-range values ('neither', 'both', 'min', 'max') when using filled contours (default: 'both').
            plot_type (str): Type of plot to create ('contourf', 'contour', 'pcolormesh') with different rendering styles for the cross-section visualization (default: 'contourf').
            save_path (Optional[str]): File path to save the generated figure; if None, the figure is not saved to disk (default: None).
            title (Optional[str]): Title for the plot; if None, a default title is generated based on variable name and time information (default: None).
            max_height (Optional[float]): Maximum height in kilometers to display on the vertical axis; if None, full height range is shown based on data and vertical coordinate system (default: None).
            **kwargs: Additional keyword arguments passed to the underlying matplotlib plotting functions for further customization. 

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects containing the generated vertical cross-section plot. 
        """
        self._validate_cross_section_inputs(mpas_3d_processor, var_name)

        print(f"Creating vertical cross-section for {var_name}")
        print(f"Cross-section from ({start_point[0]:.2f}, {start_point[1]:.2f}) to ({end_point[0]:.2f}, {end_point[1]:.2f})")

        cross_section_data = self._generate_cross_section_data(
            mpas_3d_processor, var_name, start_point, end_point,
            time_index, vertical_coord, num_points
        )

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)

        longitudes = cross_section_data['longitudes']
        vertical_coords = cross_section_data['vertical_coords']
        vertical_coord_type = cross_section_data.get('vertical_coord_type', vertical_coord)
        data_values = cross_section_data['data_values']

        data_values, metadata = self._convert_and_clip_data(data_values, var_name)

        desired_display = display_vertical if display_vertical is not None else vertical_coord
        vertical_display, vertical_coord_display = self._resolve_vertical_display(
            vertical_coords, vertical_coord_type, desired_display, mpas_3d_processor, time_index
        )

        vertical_display, data_values = self._apply_max_height_filter(
            vertical_display, vertical_coord_display, data_values, max_height
        )

        X, Y = np.meshgrid(longitudes, vertical_display)
        colormap, levels = self._resolve_plot_style(var_name, data_values, colormap, levels)

        self._render_cross_section_plot(
            X, Y, data_values, colormap, levels, extend, plot_type, metadata, var_name, **kwargs
        )

        self._format_cross_section_axes(longitudes, vertical_display, vertical_coord_display,
                                        start_point, end_point, max_height)

        final_title = self._generate_cross_section_title(mpas_3d_processor, var_name, time_index, title)

        self.ax.set_title(final_title, fontsize=14, fontweight='bold', pad=20)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        self.fig.subplots_adjust(bottom=0.09)
        self.add_timestamp_and_branding()

        if save_path:
            self._save_cross_section(save_path)

        return self.fig, self.ax
        

    def _resolve_vertical_levels(self: 'MPASVerticalCrossSectionPlotter',
                                 mpas_3d_processor: MPAS3DProcessor,
                                 var_name: str,
                                 vertical_coord: str,
                                 time_index: int) -> Tuple[np.ndarray, str]:
        """
        This internal method resolves the vertical levels for the specified variable and vertical coordinate type. It attempts to retrieve the vertical levels from the MPAS3DProcessor using the get_vertical_levels method, which may return either pressure levels or model levels based on the vertical_coord argument. If the retrieved vertical levels are integer indices, it assumes they represent model levels and updates the vertical_coord accordingly. If there is an issue retrieving the vertical levels (e.g., due to missing metadata or unexpected data structure), it falls back to generating a default range of model level indices based on the dataset dimensions. The method returns the resolved vertical levels as a numpy array and a string indicating the type of vertical coordinate being used for display.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): The processor instance used to retrieve vertical levels from the dataset.
            var_name (str): The name of the variable for which to retrieve vertical levels.
            vertical_coord (str): The type of vertical coordinate to retrieve ('pressure' or 'modlev').
            time_index (int): The time index for selecting temporal data if needed for retrieving vertical levels.

        Returns:
            Tuple[np.ndarray, str]: The resolved vertical levels as a numpy array and the type of vertical coordinate being used for display.
        """
        try:
            return_pressure = vertical_coord in ('pressure', 'height')

            vertical_levels = mpas_3d_processor.get_vertical_levels(
                var_name, return_pressure=return_pressure, time_index=time_index
            )

            if vertical_coord == 'height':
                vertical_coord = 'pressure'

            if vertical_coord not in ('pressure', 'modlev'):
                vertical_coord = 'modlev'

            vertical_levels = np.array(vertical_levels)

            # DEBUG: trace vertical level resolution
            print(f"[DEBUG _resolve_vertical_levels] return_pressure={return_pressure}")
            print(f"[DEBUG _resolve_vertical_levels] vertical_coord_type='{vertical_coord}'")
            print(f"[DEBUG _resolve_vertical_levels] vertical_levels dtype={vertical_levels.dtype}, shape={vertical_levels.shape}")
            print(f"[DEBUG _resolve_vertical_levels] vertical_levels min={np.nanmin(vertical_levels):.4f}, max={np.nanmax(vertical_levels):.4f}")

            if len(vertical_levels) <= 60:
                print(f"[DEBUG _resolve_vertical_levels] all values (Pa if pressure): {vertical_levels}")
            else:
                print(f"[DEBUG _resolve_vertical_levels] first 5: {vertical_levels[:5]}, last 5: {vertical_levels[-5:]}")

            if np.issubdtype(vertical_levels.dtype, np.integer):
                vertical_coord = 'modlev'
                if self.fig is not None and self.verbose:
                    print("Note: vertical levels appear to be integer indices; switching vertical_coord to 'modlev'")
            return vertical_levels, vertical_coord
        except Exception as e:
            print(f"Warning: Could not get vertical levels, using indices: {e}")
            sizes = mpas_3d_processor.dataset[var_name].sizes
            n_levels = (
                mpas_3d_processor.dataset.sizes.get('nVertLevels')
                or mpas_3d_processor.dataset.sizes.get('nVertLevelsP1')
                or 10
            )
            return np.arange(n_levels), 'modlev'

    def _extract_cross_section_coords(self: 'MPASVerticalCrossSectionPlotter',
                                      mpas_3d_processor: MPAS3DProcessor,
                                      var_name: str,
                                      path_lons: np.ndarray,
                                      path_lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This internal method extracts the longitude and latitude coordinates from the MPAS3DProcessor dataset for the specified variable. It first attempts to use the extract_2d_coordinates_for_variable method, which may provide variable-specific coordinate extraction logic. If that fails (e.g., due to missing metadata or unexpected data structure), it falls back to using the 'lonCell' and 'latCell' variables from the dataset as default coordinates. The method also checks if the longitude values are in radians (i.e., within ±π) and converts them to degrees if necessary. Finally, it prints information about the grid domain and cross-section path, and checks whether the path extends outside the grid domain, issuing a warning if so. The method returns the longitude and latitude coordinates as numpy arrays for use in spatial interpolation along the cross-section path.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): The processor instance used to access the dataset and extract coordinates.
            var_name (str): The name of the variable for which to extract coordinates, used to determine if variable-specific coordinate extraction logic should be applied.
            path_lons (np.ndarray): The longitude coordinates of the cross-section path, used to check if the path extends outside the grid domain.
            path_lats (np.ndarray): The latitude coordinates of the cross-section path, used to check if the path extends outside the grid domain.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The longitude and latitude coordinates extracted from the dataset for the specified variable, formatted as numpy arrays.
        """
        try:
            var_da = mpas_3d_processor.dataset[var_name]
            lon_coords, lat_coords = mpas_3d_processor.extract_2d_coordinates_for_variable(var_name, var_da)
        except Exception:
            lon_coords = mpas_3d_processor.dataset['lonCell'].values
            lat_coords = mpas_3d_processor.dataset['latCell'].values

        if np.max(np.abs(lon_coords)) <= np.pi:
            lon_coords = np.degrees(lon_coords)
            lat_coords = np.degrees(lat_coords)

        print(f"Grid domain: lon [{np.min(lon_coords):.2f}, {np.max(lon_coords):.2f}], lat [{np.min(lat_coords):.2f}, {np.max(lat_coords):.2f}]")
        print(f"Cross-section path: ({path_lons[0]:.2f}, {path_lats[0]:.2f}) to ({path_lons[-1]:.2f}, {path_lats[-1]:.2f})")

        path_in_lon = path_lons[0] >= np.min(lon_coords) and path_lons[-1] <= np.max(lon_coords)
        path_in_lat = min(path_lats[0], path_lats[-1]) >= np.min(lat_coords) and max(path_lats[0], path_lats[-1]) <= np.max(lat_coords)

        if not (path_in_lon and path_in_lat):
            print("WARNING: Cross-section path extends outside grid domain!")
            print(f"  Longitude OK: {path_in_lon}, Latitude OK: {path_in_lat}")

        return lon_coords, lat_coords

    @staticmethod
    def _unwrap_dataset_var(mpas_3d_processor: MPAS3DProcessor,
                            var_name: str) -> Tuple[xr.DataArray, Optional[str], Optional[str]]:
        """
        This static method unwraps the specified variable from the MPAS3DProcessor dataset and identifies the time and vertical dimension names. It first checks if the dataset is an xarray Dataset or can be converted to one, then retrieves the DataArray for the specified variable. It looks for common time dimension names ('Time' or 'time') and vertical dimension names ('nVertLevels' or 'nVertLevelsP1') in the variable's dimensions. The method returns the variable's DataArray along with the identified time and vertical dimension names, which are used later for indexing and interpolation when generating the cross-section data. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): The processor instance used to access the dataset.
            var_name (str): The name of the variable for which to extract data.

        Returns:
            Tuple[xr.DataArray, Optional[str], Optional[str]]: The data array for the variable, the time dimension name, and the vertical dimension name.
        """
        ds = mpas_3d_processor.dataset

        if type(ds) is not xr.Dataset and isinstance(ds, xr.Dataset):
            ds = xr.Dataset(ds)

        var_da = ds[var_name]
        time_dim = 'Time' if 'Time' in var_da.sizes else ('time' if 'time' in var_da.sizes else None)

        if 'nVertLevels' in var_da.sizes:
            vert_dim: Optional[str] = 'nVertLevels'
        elif 'nVertLevelsP1' in var_da.sizes:
            vert_dim = 'nVertLevelsP1'
        else:
            vert_dim = None

        return var_da, time_dim, vert_dim

    @staticmethod
    def _extract_level_data(var_da: xr.DataArray, 
                            isel_dict: Dict[str, int],
                            expected_ncells: int) -> np.ndarray:
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

        if hasattr(level_data, 'compute'):
            level_data = level_data.compute()

        data_values = level_data.values if hasattr(level_data, 'values') else np.asarray(level_data)

        if data_values.ndim != 1 or data_values.shape[0] != expected_ncells:
            raise ValueError(
                f"Level extraction produced shape {data_values.shape}, "
                f"expected ({expected_ncells},)"
            )

        return data_values

    def _interpolate_all_levels(self: 'MPASVerticalCrossSectionPlotter',
                                var_da: xr.DataArray,
                                vertical_levels: np.ndarray,
                                time_index: int,
                                time_dim: Optional[str],
                                vert_dim: Optional[str],
                                lon_coords: np.ndarray,
                                lat_coords: np.ndarray,
                                path_lons: np.ndarray,
                                path_lats: np.ndarray,
                                num_points: int) -> np.ndarray:
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
        cross_section_data = np.full((len(vertical_levels), num_points), np.nan)
        
        for level_idx, level in enumerate(vertical_levels):
            try:
                isel_dict: Dict[str, int] = {}

                if time_dim is not None:
                    isel_dict[time_dim] = time_index

                if vert_dim is None:
                    continue

                isel_dict[vert_dim] = level_idx
                data_values = self._extract_level_data(var_da, isel_dict, lon_coords.shape[0])
                
                cross_section_data[level_idx, :] = self._interpolate_along_path(
                    lon_coords, lat_coords, data_values, path_lons, path_lats
                )
            except Exception as e:
                print(f"Warning: Could not extract data for level {level}: {e}")

        valid = ~np.isnan(cross_section_data)

        if np.any(valid):
            data_min = np.min(cross_section_data[valid])
            data_max = np.max(cross_section_data[valid])
            print(f"Final cross-section data: {data_min:.3f} to {data_max:.3f} ({np.sum(valid)}/{cross_section_data.size} valid points)")
        else:
            print("WARNING: Final cross-section data contains NO valid values!")

        return cross_section_data


    def _generate_cross_section_data(self: 'MPASVerticalCrossSectionPlotter', 
                                     mpas_3d_processor: MPAS3DProcessor, 
                                     var_name: str, 
                                     start_point: Tuple[float, float], 
                                     end_point: Tuple[float, float], 
                                     time_index: int, 
                                     vertical_coord: str, 
                                     num_points: int) -> Dict[str, Any]:
        """
        This internal method generates the cross-section data by interpolating the specified 3D variable along a great-circle path defined by the start and end points. It retrieves the vertical levels for the variable, extracts the corresponding data slices for each level, and performs spatial interpolation to obtain data values along the cross-section path. The method handles different vertical coordinate types, checks for valid data, and returns a dictionary containing the distances along the path, vertical coordinates, interpolated data values, and other relevant information for plotting. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance providing dataset access and processing capabilities.
            var_name (str): Name of the 3D variable to extract and interpolate for the cross-section.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            time_index (int): Time index for selecting temporal data from the dataset.
            vertical_coord (str): Type of vertical coordinate to use for data extraction ('pressure', 'height', 'modlev').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.

        Returns:
            Dict[str, Any]: Dictionary containing 'distances' (array of distances along the path), 'vertical_coords' (array of vertical coordinate values), 'data_values' (2D array of interpolated data values along the cross-section), 'path_lons' (array of longitude coordinates along the path), 'path_lats' (array of latitude coordinates along the path), and 'vertical_coord_type' (string indicating the type of vertical coordinate used). 
        """
        path_lons, path_lats, distances = self._generate_great_circle_path(
            start_point, end_point, num_points
        )

        vertical_levels, vertical_coord = self._resolve_vertical_levels(
            mpas_3d_processor, var_name, vertical_coord, time_index
        )

        lon_coords, lat_coords = self._extract_cross_section_coords(
            mpas_3d_processor, var_name, path_lons, path_lats
        )

        print(f"Interpolating {var_name} data along cross-section...")
        var_da, time_dim, vert_dim = self._unwrap_dataset_var(mpas_3d_processor, var_name)

        cross_section_data = self._interpolate_all_levels(
            var_da, vertical_levels, time_index, time_dim, vert_dim,
            lon_coords, lat_coords, path_lons, path_lats, num_points
        )

        return {
            'distances': distances,
            'vertical_coords': np.array(vertical_levels),
            'data_values': cross_section_data,
            'path_lons': path_lons,
            'path_lats': path_lats,
            'longitudes': path_lons,
            'vertical_coord_type': vertical_coord,
        }
        
    def _generate_great_circle_path(self: 'MPASVerticalCrossSectionPlotter', 
                                    start_point: Tuple[float, float], 
                                    end_point: Tuple[float, float], 
                                    num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This internal method generates a great-circle path between two geographic points specified by their longitude and latitude coordinates. It calculates the intermediate points along the great-circle route, providing arrays of longitude, latitude, and distance along the path. The method uses spherical trigonometry to compute the path and handles edge cases such as very short distances. The output can be used for spatial interpolation of data along the cross-section path in the vertical cross-section plotting process. 

        Parameters:
            start_point (Tuple[float, float]): Starting point of the path as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the path as (longitude, latitude) in degrees.
            num_points (int): Number of interpolation points along the path for smooth plotting. 

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of longitude coordinates, latitude coordinates, and distances along the path (in kilometers) for each interpolation point. 
        """
        lon1, lat1 = radians(start_point[0]), radians(start_point[1])
        lon2, lat2 = radians(end_point[0]), radians(end_point[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        total_distance = 2 * asin(sqrt(a)) * 6371.0  
        
        fractions = np.linspace(0, 1, num_points)
        lons = np.zeros(num_points)
        lats = np.zeros(num_points)
        distances = np.zeros(num_points)
        
        if total_distance < 1e-6:  
            lons.fill(start_point[0])
            lats.fill(start_point[1])
            distances.fill(0)
            return lons, lats, distances
        
        for i, f in enumerate(fractions):
            if f == 0:
                lons[i] = start_point[0]
                lats[i] = start_point[1]
                distances[i] = 0
            elif f == 1:
                lons[i] = end_point[0]
                lats[i] = end_point[1]
                distances[i] = total_distance
            else:
                angular_distance = total_distance / 6371.0  
                A = sin((1-f) * angular_distance) / sin(angular_distance)
                B = sin(f * angular_distance) / sin(angular_distance)
                
                x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
                y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
                z = A * sin(lat1) + B * sin(lat2)
                
                lats[i] = degrees(atan2(z, sqrt(x**2 + y**2)))
                lons[i] = degrees(atan2(y, x))
                distances[i] = f * total_distance
            
        return lons, lats, distances
        
    def _interpolate_along_path(self: 'MPASVerticalCrossSectionPlotter',
                                 grid_lons: np.ndarray, 
                                 grid_lats: np.ndarray, 
                                 grid_data: Union[np.ndarray, xr.DataArray, Any], 
                                 path_lons: np.ndarray, 
                                 path_lats: np.ndarray) -> np.ndarray:
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
            return np.full(len(path_lons), np.nan)
            
        grid_points = np.column_stack([
            np.cos(np.radians(grid_lats_valid)) * np.cos(np.radians(grid_lons_valid)),
            np.cos(np.radians(grid_lats_valid)) * np.sin(np.radians(grid_lons_valid)),
            np.sin(np.radians(grid_lats_valid))
        ])
        
        path_points = np.column_stack([
            np.cos(np.radians(path_lats)) * np.cos(np.radians(path_lons)),
            np.cos(np.radians(path_lats)) * np.sin(np.radians(path_lons)),
            np.sin(np.radians(path_lats))
        ])
        
        tree = KDTree(grid_points)
        distances, indices = tree.query(path_points)
        
        return grid_data_valid[indices]

    def _compute_var_levels(self: 'MPASVerticalCrossSectionPlotter',
                            var_lower: str,
                            var_name: str,
                            data_min: float,
                            data_max: float,
                            data_range: float,) -> np.ndarray:
        """
        This internal method computes contour levels for plotting based on the variable name and data range. It applies specific rules for different variable types (e.g., temperature, pressure, wind) to determine appropriate level spacing and range. For temperature variables, it uses a step of 5 or 2 depending on the data range. For pressure variables, it uses logarithmic spacing if the minimum value is greater than zero. For wind variables, it creates symmetric levels around zero if the data range includes both positive and negative values. For other variables, it defaults to linear spacing with 15 levels across the data range. This method ensures that the contour levels are meaningful and enhance the visualization of the variable in the cross-section plot. 

        Parameters:
            var_lower (str): The lowercase version of the variable name, used for determining variable type.
            var_name (str): The original variable name, used for additional checks (e.g., if it starts with 'u' or 'v' for wind variables).
            data_min (float): The minimum valid data value for the variable, used to determine the starting point of the levels.
            data_max (float): The maximum valid data value for the variable, used to determine the ending point of the levels.
            data_range (float): The range of valid data values (data_max - data_min), used to determine the spacing of the levels.

        Returns:
            np.ndarray: An array of contour levels computed based on the variable type and data range, optimized for visualization in the cross-section plot.   
        """
        if 'temperature' in var_lower or 'temp' in var_lower:
            step = 5 if data_range > 50 else 2
            return np.arange(data_min, data_max + step, step)
        
        if 'pressure' in var_lower:
            if data_min > 0:
                return np.logspace(np.log10(data_min), np.log10(data_max), 15)
            return np.linspace(data_min, data_max, 15)

        if 'wind' in var_lower or var_name.startswith('u') or var_name.startswith('v'):
            max_abs = max(abs(data_min), abs(data_max))
            if data_min < 0 and data_max > 0:
                return np.linspace(-max_abs, max_abs, 21)
            return np.linspace(data_min, data_max, 15)

        return np.linspace(data_min, data_max, 15)

    def _get_default_levels(self: 'MPASVerticalCrossSectionPlotter', 
                            data_values: Union[np.ndarray, xr.DataArray, float], 
                            var_name: str) -> np.ndarray:
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
            return np.linspace(0, 1, 11)
            
        data_min, data_max = valid_data.min(), valid_data.max()
        data_range = data_max - data_min
        
        if data_range == 0:
            return np.array([data_min])
            
        var_lower = var_name.lower()
        return self._compute_var_levels(var_lower, var_name, data_min, data_max, data_range)
                
    def _extract_height_from_dataset(self: 'MPASVerticalCrossSectionPlotter', 
                                     mpas_3d_processor: MPAS3DProcessor, 
                                     time_index: int, 
                                     vertical_coords: np.ndarray, 
                                     var_name: str) -> Optional[np.ndarray]:
        """
        This internal method attempts to extract geometric height data from the MPAS dataset for the specified variable name (e.g., 'zgrid' or 'height') at the given time index. It checks if the variable exists in the dataset, retrieves the height data for the first cell (assuming it's representative of the vertical structure), and processes it to match the expected length based on the vertical coordinates. If the height data has one more level than the vertical coordinates, it computes mid-level heights by averaging adjacent levels. If it matches the length of vertical coordinates, it returns it directly. If neither condition is met, it attempts to interpolate the height data to match the vertical coordinates. The method includes error handling to return None if extraction fails or if the variable is unavailable, allowing for fallback methods to determine height when necessary. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance providing access to the dataset for height variable extraction.
            time_index (int): Time index for selecting the appropriate temporal slice of the height variable.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the cross-section, which determines the expected length of the height data.
            var_name (str): Name of the height variable to extract from the dataset (e.g., 'zgrid' or 'height'). 

        Returns:
            Optional[np.ndarray]: Array of geometric height values in meters corresponding to the vertical coordinates, or None if extraction fails or variable is unavailable. 
        """
        try:
            height_data = None

            if var_name in mpas_3d_processor.dataset.data_vars:
                height_data = mpas_3d_processor.dataset[var_name].isel(Time=time_index, nCells=0).values
            elif hasattr(mpas_3d_processor, 'grid_file') and mpas_3d_processor.grid_file:
                try:
                    # Only load the needed height variable from the grid file
                    open_kwargs: dict = {'decode_times': False}
                    try:
                        with xr.open_dataset(mpas_3d_processor.grid_file, decode_times=False) as probe:
                            all_vars = list(probe.data_vars)
                        drop = [v for v in all_vars if v != var_name]
                        if drop:
                            open_kwargs['drop_variables'] = drop
                    except Exception:
                        pass
                    with xr.open_dataset(mpas_3d_processor.grid_file, **open_kwargs) as grid_ds:
                        if var_name in grid_ds.data_vars:
                            height_data = grid_ds[var_name].isel(nCells=0).values
                except Exception:
                    pass

            if height_data is None:
                return None

            height_data = np.asarray(height_data, dtype=float)
            
            if len(height_data) == len(vertical_coords) + 1:
                mid_heights = 0.5 * (height_data[:-1] + height_data[1:])
                return mid_heights
            elif len(height_data) == len(vertical_coords):
                return height_data
            else:
                try:
                    from scipy.interpolate import interp1d
                    xp = np.linspace(0, 1, len(height_data))
                    fp = height_data
                    f = interp1d(xp, fp, bounds_error=False, fill_value=cast(Any, 'extrapolate'))
                    xq = np.linspace(0, 1, len(vertical_coords))
                    return f(xq)
                except Exception:
                    return None
        except Exception:
            return None
    
    def _try_extract_height_km(self: 'MPASVerticalCrossSectionPlotter',
                               mpas_3d_processor: MPAS3DProcessor,
                               time_index: int,
                               vertical_coords: np.ndarray,) -> Optional[Tuple[np.ndarray, str]]:
        """
        This helper method attempts to extract geometric height from the dataset using known variable names ('zgrid' or 'height') and convert it to kilometers. It calls the _extract_height_from_dataset method for each variable name, and if successful, it converts the height from meters to kilometers and returns it along with the coordinate type string. If extraction fails for both variable names, it returns None, allowing the calling method to proceed with alternative methods for determining height. This approach provides a way to utilize available geometric height data in the dataset when pressure coordinates are used, enhancing the accuracy of the vertical axis in the cross-section plot. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance for accessing the dataset to extract height information.
            time_index (int): Time index for selecting the appropriate temporal slice when extracting height from the dataset.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the cross-section, which may be needed to determine the expected length of the height data.

        Returns:
            Optional[Tuple[np.ndarray, str]]: A tuple containing the extracted height values in kilometers and the string 'height_km' if extraction is successful, or None if extraction fails for both variable names.
        """
        print(f"[DEBUG _try_extract_height_km] Attempting to extract geometric height from dataset")
        try:
            for var in ('zgrid', 'height'):
                print(f"[DEBUG _try_extract_height_km] Trying variable '{var}'...")
                height_m = self._extract_height_from_dataset(
                    mpas_3d_processor, time_index, vertical_coords, var
                )
                if height_m is not None:
                    print(f"[DEBUG _try_extract_height_km] SUCCESS with '{var}': min={np.nanmin(height_m):.2f} m, max={np.nanmax(height_m):.2f} m")
                    print(f"[DEBUG _try_extract_height_km] Converted to km: min={np.nanmin(height_m/1000.0):.4f}, max={np.nanmax(height_m/1000.0):.4f}")
                    return height_m / 1000.0, 'height_km'
                else:
                    print(f"[DEBUG _try_extract_height_km] '{var}' not found or extraction failed")
        except Exception as e:
            print(f"[DEBUG _try_extract_height_km] Exception: {e}")
        print(f"[DEBUG _try_extract_height_km] No geometric height found, will fall back to barometric approximation")
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
        g = 9.80665    # gravitational acceleration (m/s^2)
        R = 287.053    # specific gas constant for dry air (J/(kg*K))

        # (base_height_m, base_temp_K, lapse_rate_K_per_m, base_pressure_Pa)
        layers = [
            (0.0,     288.15,  -0.0065,  101325.0),    # Troposphere: 0-11 km
            (11000.0, 216.65,   0.0,      22632.1),     # Tropopause: 11-20 km
            (20000.0, 216.65,   0.001,    5474.89),     # Stratosphere 1: 20-32 km
            (32000.0, 228.65,   0.0028,   868.02),      # Stratosphere 2: 32-47 km
            (47000.0, 270.65,   0.0,      110.91),      # Stratopause: 47-51 km
        ]

        height_m = np.empty_like(pressure_pa)

        for i in range(len(pressure_pa)):
            p = pressure_pa[i]

            # Find which layer this pressure falls in (layers sorted by decreasing base pressure)
            layer_idx = len(layers) - 1

            for j in range(len(layers) - 1):
                if p >= layers[j + 1][3]:
                    layer_idx = j
                    break
            
            # Extract layer parameters for the identified layer
            h_b, T_b, L, P_b = layers[layer_idx]

            if abs(L) < 1e-10:
                # Isothermal layer: h = h_b - (R*T_b/g) * ln(P/P_b)
                height_m[i] = h_b - (R * T_b / g) * np.log(p / P_b)
            else:
                # Layer with lapse rate: h = h_b + (T_b/L) * ((P/P_b)^(-R*L/g) - 1)
                exponent = -R * L / g
                height_m[i] = h_b + (T_b / L) * ((p / P_b) ** exponent - 1.0)

        return height_m

    def _pressure_to_height_approx(self: 'MPASVerticalCrossSectionPlotter',
                                   vertical_coords: np.ndarray,) -> Tuple[np.ndarray, str]:
        """
        This internal method converts pressure values to geometric height in kilometers using the US Standard Atmosphere 1976 approximation. It first checks if the input pressure values are likely in hPa (by checking if the maximum value is less than 10000) and converts them to Pa if necessary. It then handles any non-positive or non-finite pressure values by clipping them to a minimum positive value to avoid issues with logarithmic calculations. The method applies the standard atmosphere model to convert the pressure values to height in meters, and then converts it to kilometers before returning. If any exceptions occur during this process, it falls back to returning the original vertical coordinates converted to hPa and indicates that the vertical coordinate type is 'pressure_hPa'. This method provides an approximate way to determine height when geometric height data is not available in the dataset, ensuring that the cross-section plot can still be generated with a reasonable vertical axis.

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values representing pressure levels, which may be in hPa or Pa and may contain non-positive or non-finite values.

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the converted height values in kilometers and the string 'height_km' if conversion is successful, or the original vertical coordinates converted to hPa and the string 'pressure_hPa' if an exception occurs during conversion.
        """
        try:
            pressure_pa = vertical_coords.astype(float).copy()

            # DEBUG: input pressure values
            print(f"[DEBUG _pressure_to_height_approx] Input pressure values:")
            print(f"[DEBUG _pressure_to_height_approx]   dtype={pressure_pa.dtype}, shape={pressure_pa.shape}")
            print(f"[DEBUG _pressure_to_height_approx]   min={np.nanmin(pressure_pa):.4f}, max={np.nanmax(pressure_pa):.4f}")

            if np.nanmax(pressure_pa) < 10000:  # Likely in hPa
                print(f"[DEBUG _pressure_to_height_approx]   max < 10000 => treating as hPa, multiplying by 100")
                pressure_pa = pressure_pa * 100.0
            else:
                print(f"[DEBUG _pressure_to_height_approx]   max >= 10000 => treating as Pa (no conversion)")

            min_positive = 1.0

            if np.any(pressure_pa <= 0) or np.any(~np.isfinite(pressure_pa)):
                print(f"[DEBUG _pressure_to_height_approx]   WARNING: non-positive/non-finite values detected, clipping")
                if self.verbose:
                    print(
                        "Warning: pressure levels contained non-positive or non-finite values; "
                        "clipping to minimum positive value to avoid log(0)"
                    )
                pressure_pa = np.where(np.isfinite(pressure_pa), pressure_pa, min_positive)
                pressure_pa = np.clip(pressure_pa, min_positive, None)

            print(f"[DEBUG _pressure_to_height_approx]   Using US Standard Atmosphere 1976 (piecewise, 5 layers)")
            print(f"[DEBUG _pressure_to_height_approx]   pressure_pa after processing: min={np.nanmin(pressure_pa):.4f}, max={np.nanmax(pressure_pa):.4f}")

            height_m = self._std_atm_pressure_to_height(pressure_pa)
            height_km = np.maximum(height_m / 1000.0, 0.0)

            print(f"[DEBUG _pressure_to_height_approx]   height_km: min={np.nanmin(height_km):.4f}, max={np.nanmax(height_km):.4f}")
            if len(height_km) <= 60:
                print(f"[DEBUG _pressure_to_height_approx]   all height_km values: {np.array2string(height_km, precision=3, separator=', ')}")

            return height_km, 'height_km'
        except Exception as e:
            print(f"[DEBUG _pressure_to_height_approx] EXCEPTION: {e}")
            return vertical_coords / 100.0, 'pressure_hPa'

    def _convert_vertical_to_height(self: 'MPASVerticalCrossSectionPlotter', 
                                    vertical_coords: np.ndarray, 
                                    vertical_coord_type: str, 
                                    mpas_3d_processor: MPAS3DProcessor, 
                                    time_index: int) -> Tuple[np.ndarray, str]:
        """
        This internal method converts vertical coordinate values to geometric height in kilometers based on the specified vertical coordinate type. If the input coordinate type is 'height', it simply converts from meters to kilometers. If the input type is 'pressure', it first attempts to extract geometric height from the dataset using known variable names ('zgrid' or 'height'). If successful, it converts to kilometers and returns. If extraction fails, it applies a standard atmosphere approximation to convert pressure to height, while handling potential issues with non-positive or non-finite pressure values. For 'modlev', it also tries to extract geometric height if available, otherwise it returns the original model level indices. The method ensures that the returned vertical coordinates are in a consistent format for plotting and includes error handling for various edge cases. 

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values to convert.
            vertical_coord_type (str): Type of vertical coordinate ('height', 'pressure', 'modlev') indicating how to interpret the input coordinates.
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance for accessing the dataset to extract height information if needed.
            time_index (int): Time index for selecting the appropriate temporal slice when extracting height from the dataset. 

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the converted vertical coordinates in kilometers and a string indicating the type of vertical coordinate used for display (e.g., 'height_km', 'pressure_hPa', or 'modlev'). 
        """
        print(f"[DEBUG _convert_vertical_to_height] vertical_coord_type='{vertical_coord_type}'")
        print(f"[DEBUG _convert_vertical_to_height] vertical_coords: shape={vertical_coords.shape}, min={np.nanmin(vertical_coords):.4f}, max={np.nanmax(vertical_coords):.4f}")

        if vertical_coord_type == 'height':
            print(f"[DEBUG _convert_vertical_to_height] Path: direct height -> dividing by 1000 for km")
            return vertical_coords / 1000.0, 'height_km'

        if vertical_coord_type == 'pressure':
            print(f"[DEBUG _convert_vertical_to_height] Path: pressure -> trying geometric height first, then barometric approx")
            result = self._try_extract_height_km(mpas_3d_processor, time_index, vertical_coords)
            if result is not None:
                print(f"[DEBUG _convert_vertical_to_height] Using extracted geometric height")
                return result
            print(f"[DEBUG _convert_vertical_to_height] Falling back to barometric approximation")
            return self._pressure_to_height_approx(vertical_coords)

        print(f"[DEBUG _convert_vertical_to_height] Path: modlev -> trying geometric height")
        result = self._try_extract_height_km(mpas_3d_processor, time_index, vertical_coords)

        if result is not None:
            return result
        
        print(f"[DEBUG _convert_vertical_to_height] No height conversion possible, returning raw model levels")
        return vertical_coords, 'modlev'

    def _apply_standard_pressure_ticks(self: 'MPASVerticalCrossSectionPlotter',
                                       vertical_coords: np.ndarray,) -> None:
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
            from matplotlib.ticker import FixedLocator, FuncFormatter
            standard_ticks = [1000, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 1]
            data_min = np.nanmin(vertical_coords)
            data_max = np.nanmax(vertical_coords)
            tick_vals = [t for t in standard_ticks if data_min <= t <= data_max]
            if len(tick_vals) >= 2:
                self.ax.yaxis.set_major_locator(FixedLocator(tick_vals))
                def _fmt(x, pos):
                    if x >= 1:
                        return f"{int(x):d}"
                    return f"{x:.2f}"
                self.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
        except Exception:
            pass

    def _setup_pressure_axis(self: 'MPASVerticalCrossSectionPlotter', 
                             vertical_coords: np.ndarray, 
                             use_standard_ticks: bool = True) -> None:
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
            vmin = np.nanmin(vertical_coords)
        except Exception:
            print("Warning: could not determine pressure coordinate min; using linear y-scale")
            return

        if vmin <= 0:
            print("Warning: detected non-positive pressure coordinate values; using linear y-scale for pressure display")
            return

        self.ax.set_yscale('log')
        
        if use_standard_ticks:
            self._apply_standard_pressure_ticks(vertical_coords)

    def _apply_x_axis_formatting(self: 'MPASVerticalCrossSectionPlotter',
                                 longitudes: np.ndarray,) -> None:
        """
        This internal method applies formatting to the x-axis of the vertical cross-section plot by setting the label to 'Longitude', adjusting the limits based on the provided longitude array, and applying a custom tick formatter to display longitude values with degree symbols. It includes error handling to ensure that axis formatting does not fail if certain styling features are unavailable, providing a robust setup for clear visualization of longitude along the cross-section path. This method enhances the readability of the x-axis by using appropriate labels and formatting for geographic coordinates. 

        Parameters:
            longitudes (np.ndarray): Array of longitude values along the cross-section path, used to set x-axis limits and formatting for the longitude display.

        Returns:
            None: Modifies self.ax x-axis properties directly without returning a value.
        """
        assert self.ax is not None, "Axes must be created before formatting"

        self.ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
        self.ax.set_xlim(longitudes.min(), longitudes.max())

        try:
            from matplotlib.ticker import FuncFormatter
            lon_formatter = FuncFormatter(MPASVisualizationStyle.format_longitude)
            self.ax.xaxis.set_major_formatter(lon_formatter)
        except Exception:
            pass

    def _set_pressure_ylim(self: 'MPASVerticalCrossSectionPlotter',
                           vertical_coords: np.ndarray,
                           max_height: Optional[float],
                           P0: float,
                           H: float = 8.4,) -> None:
        """
        This internal method sets the y-axis limits for pressure-based vertical cross-section plots based on the provided vertical coordinate values and an optional maximum height. If a maximum height is specified, it calculates the corresponding minimum pressure using the barometric formula and sets the y-axis limits accordingly. If no maximum height is provided, it defaults to setting the limits based on the range of the vertical coordinates. The method includes error handling to ensure that axis limits are set correctly even if certain styling features are unavailable, providing a robust setup for clear visualization of pressure levels along the cross-section. This method is essential for ensuring that the y-axis of pressure-based cross-section plots is appropriately scaled to the range of pressure levels being visualized, enhancing the readability and interpretability of the plot.

        Parameters:
            vertical_coords (np.ndarray): Array of vertical coordinate values representing pressure levels, used to determine appropriate y-axis limits for the pressure display.
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data.
            P0 (float): Reference sea level pressure in Pa used for calculating the minimum pressure corresponding to the maximum height when max_height is provided.
            H (float): Scale height in kilometers used in the barometric formula for converting height to pressure (default: 8.4 km).

        Returns:
            None: Modifies self.ax y-axis limits directly without returning a value.
        """
        assert self.ax is not None, "Axes must be created before setting limits"

        if max_height is not None:
            min_pressure = P0 * np.exp(-max_height / H)
            valid_coords = vertical_coords[vertical_coords >= min_pressure]

            if len(valid_coords) > 0:
                self.ax.set_ylim(valid_coords.max(), min_pressure)
            else:
                self.ax.set_ylim(vertical_coords.max(), vertical_coords.min())
        else:
            self.ax.set_ylim(vertical_coords.max(), vertical_coords.min())

    def _format_cross_section_axes(self: 'MPASVerticalCrossSectionPlotter', 
                                   longitudes: np.ndarray, 
                                   vertical_coords: np.ndarray, 
                                   vertical_coord_type: str, 
                                   start_point: Tuple[float, float], 
                                   end_point: Tuple[float, float], 
                                   max_height: Optional[float] = None) -> None:
        """
        This internal method formats the axes of the vertical cross-section plot by setting appropriate labels, limits, and tick formatting based on the longitude range and vertical coordinate type. It configures the x-axis to display longitude values with degree symbols and sets limits based on the provided longitude array. For the y-axis, it applies different formatting depending on whether the vertical coordinate is height (in km), pressure (in hPa or Pa), or model levels. It also adds a text box to indicate the start and end points of the cross-section path. The method includes error handling to ensure that axis formatting does not fail even if certain styling features are unavailable, providing a robust setup for clear visualization of the cross-section data. 

        Parameters:
            longitudes (np.ndarray): Array of longitude values along the cross-section path, used to set x-axis limits and formatting.
            vertical_coords (np.ndarray): Array of vertical coordinate values used for the y-axis, with formatting determined by the vertical_coord_type.
            vertical_coord_type (str): Type of vertical coordinate ('height_km', 'pressure_hPa', 'pressure', 'height', 'modlev') that dictates y-axis labeling and limits.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees, used for annotating the plot.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees, used for annotating the plot.
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data. 

        Returns:
            None: Modifies self.ax properties directly without returning a value. 
        """
        assert self.ax is not None, "Axes must be created before formatting"

        self._apply_x_axis_formatting(longitudes)

        if vertical_coord_type == 'height_km':
            self.ax.set_ylabel('Height [km]', fontsize=12)
            y_max = max_height if max_height is not None else vertical_coords.max()
            self.ax.set_ylim(0, y_max)
        elif vertical_coord_type == 'pressure_hPa':
            self.ax.set_ylabel('Pressure [hPa]', fontsize=12)
            self._set_pressure_ylim(vertical_coords, max_height, P0=1013.25)
            self._setup_pressure_axis(vertical_coords, use_standard_ticks=True)
        elif vertical_coord_type == 'pressure':
            self.ax.set_ylabel('Pressure [Pa]', fontsize=12)
            self._set_pressure_ylim(vertical_coords, max_height, P0=101325.0)
            self._setup_pressure_axis(vertical_coords, use_standard_ticks=False)
        elif vertical_coord_type == 'height':
            self.ax.set_ylabel('Height [m]', fontsize=12)
            y_max = max_height * 1000 if max_height is not None else vertical_coords.max()
            self.ax.set_ylim(vertical_coords.min(), y_max)
        else:  # modlev
            self.ax.set_ylabel('Model Level', fontsize=12)
            try:
                self.ax.set_ylim(vertical_coords.min(), vertical_coords.max())
            except Exception:
                self.ax.set_ylim(vertical_coords.max(), vertical_coords.min())

        path_info = f"From ({start_point[0]:.1f}°, {start_point[1]:.1f}°) to ({end_point[0]:.1f}°, {end_point[1]:.1f}°)"
        self.ax.text(0.02, 0.98, path_info, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _get_time_string(self: 'MPASVerticalCrossSectionPlotter', 
                         mpas_3d_processor: MPAS3DProcessor, 
                         time_index: int) -> str:
        """
        This internal method retrieves a formatted time string for the given time index from the MPAS3DProcessor instance. It first checks if the processor has a method to get time information directly; if so, it uses that method. If not, it attempts to access a 'Time' variable in the dataset and convert it to a datetime object for formatting. If both methods fail, it falls back to returning a simple string with the time index. This approach ensures that the plot can display meaningful time information when available while maintaining robustness in cases where time data is not accessible or formatted differently. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance that may contain methods or variables for retrieving time information.
            time_index (int): Time index for which to retrieve the formatted time string. 

        Returns:
            str: Formatted time string for display in the plot title or annotations, or a fallback string with the time index if specific time information is not available. 
        """
        try:
            if hasattr(mpas_3d_processor, 'get_time_info'):
                return mpas_3d_processor.get_time_info(time_index)
            elif hasattr(mpas_3d_processor.dataset, 'Time') and len(mpas_3d_processor.dataset.Time) > time_index:
                time_value = pd.to_datetime(mpas_3d_processor.dataset.Time.values[time_index])
                return time_value.strftime('Valid: %Y-%m-%d %H:%M UTC')
            else:
                return f"Time Index: {time_index}"
        except Exception:
            return f"Time Index: {time_index}"

    def _resolve_batch_time_str(self: 'MPASVerticalCrossSectionPlotter',
                                mpas_3d_processor: MPAS3DProcessor,
                                time_idx: int,) -> str:
        """
        This helper method resolves a formatted time string for a given time index during batch processing of cross-section plots. It first checks if the MPAS3DProcessor instance has a method to retrieve time information directly; if so, it uses that method. If not, it attempts to access a 'Time' variable in the dataset and convert it to a datetime object for formatting. If both methods fail, it falls back to returning a simple string with the time index. This method ensures that each plot generated in the batch process can include meaningful time information when available, while maintaining robustness in cases where time data is not accessible or formatted differently. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance that may contain methods or variables for retrieving time information.
            time_idx (int): Time index for which to resolve the formatted time string during batch processing.

        Returns:
            str: Formatted time string for display in the plot title or annotations during batch processing, or a fallback string with the time index if specific time information is not available.
        """
        try:
            if hasattr(mpas_3d_processor.dataset, 'Time') and len(mpas_3d_processor.dataset.Time) > time_idx:
                time_value = pd.to_datetime(mpas_3d_processor.dataset.Time.values[time_idx])
                return time_value.strftime('%Y%m%dT%H')
        except Exception:
            pass
        return f"t{time_idx:03d}"

    def _process_batch_time_step(self: 'MPASVerticalCrossSectionPlotter',
                                 mpas_3d_processor: MPAS3DProcessor,
                                 output_dir: str,
                                 var_name: str,
                                 start_point: Tuple[float, float],
                                 end_point: Tuple[float, float],
                                 time_idx: int,
                                 total_times: int,
                                 vertical_coord: str,
                                 num_points: int,
                                 levels: Optional[np.ndarray],
                                 colormap: Optional[Union[str, mcolors.Colormap]],
                                 extend: str,
                                 plot_type: str,
                                 max_height: Optional[float],
                                 file_prefix: str,
                                 formats: List[str],) -> List[str]:
        """
        This helper method processes a single time step for creating a vertical cross-section plot in the batch processing workflow. It retrieves a formatted time string for the current time index, constructs a descriptive title for the plot, and calls the create_vertical_cross_section method with the appropriate parameters. After creating the plot, it constructs the output file path based on the variable name, vertical coordinate type, and time information, saves the plot in the specified formats, and returns a list of created file paths. This method encapsulates the logic for handling each time step in the batch process, including error handling and progress updates. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): MPAS3DProcessor instance with loaded 3D data for plotting.
            output_dir (str): Directory where the generated plot will be saved.
            var_name (str): Name of the variable to plot from the dataset.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            time_idx (int): Current time index being processed in the batch workflow.
            total_times (int): Total number of time steps to process, used for progress updates.
            vertical_coord (str): Type of vertical coordinate to use ('pressure', 'height', 'modlev').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.
            levels (Optional[np.ndarray]): Optional array of contour levels; if None, levels are determined automatically.
            colormap (Optional[Union[str, mcolors.Colormap]]): Colormap to use for filled contour plots; can be a string name or a Matplotlib colormap object.
            extend (str): Direction to extend the colormap for out-of-range values ('neither', 'both', 'min', 'max').
            plot_type (str): Type of plot to create ('contourf' or 'contour_lines').
            max_height (Optional[float]): Maximum height to display in the plot; if None, the full range is used.
            file_prefix (str): Prefix for the output file names.
            formats (List[str]): List of file formats to save the plot (e.g., ['png', 'pdf']).

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
            num_points=num_points,
            levels=levels,
            colormap=colormap,
            extend=extend,
            plot_type=plot_type,
            max_height=max_height,
            title=title,
        )

        height_suffix = f"_maxh_{int(max_height)}km" if max_height else ""

        output_path = os.path.join(
            output_dir,
            f"{file_prefix}_{var_name}_vcrd_{vertical_coord}_valid_{time_str}{height_suffix}",
        )

        self.save_plot(output_path, formats=formats)
        self.close_plot()

        if (time_idx + 1) % 5 == 0 or time_idx == 0:
            print(f"Completed {time_idx + 1}/{total_times} cross-sections (time index {time_idx})...")

        return [f"{output_path}.{fmt}" for fmt in formats]

    def create_batch_cross_section_plots(self: 'MPASVerticalCrossSectionPlotter', 
                                         mpas_3d_processor: MPAS3DProcessor, 
                                         output_dir: str, 
                                         var_name: str, 
                                         start_point: Tuple[float, float], 
                                         end_point: Tuple[float, float], 
                                         vertical_coord: str = 'pressure', 
                                         num_points: int = 100, 
                                         levels: Optional[np.ndarray] = None, 
                                         colormap: Optional[Union[str, mcolors.Colormap]] = None, 
                                         extend: str = 'both', 
                                         plot_type: str = 'contourf', 
                                         max_height: Optional[float] = None, 
                                         file_prefix: str = 'mpas_cross_section', 
                                         formats: List[str] = ['png']) -> List[str]:
        """
        This method creates vertical cross-section plots for a specified 3D atmospheric variable across all available time steps in the MPAS dataset. It generates a plot for each time step, saving them to the specified output directory with filenames that include the variable name, vertical coordinate type, and valid time information. The method handles various vertical coordinate systems (pressure, height, model levels) and allows for customization of contour levels, colormap, plot type, and maximum height. It includes error handling to ensure that issues with data loading or variable availability are reported clearly, and it provides progress updates during batch processing. The resulting list of created file paths is returned upon completion. 

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): An instance of MPAS3DProcessor with loaded 3D data for plotting.
            output_dir (str): Directory where the generated plots will be saved.
            var_name (str): Name of the variable to plot from the dataset.
            start_point (Tuple[float, float]): Starting point of the cross-section as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section as (longitude, latitude) in degrees.
            vertical_coord (str): Type of vertical coordinate to use ('pressure', 'height', 'modlev').
            num_points (int): Number of interpolation points along the cross-section path for smooth plotting.
            levels (Optional[np.ndarray]): Optional array of contour levels; if None, levels are determined automatically.
            colormap (Optional[Union[str, mcolors.Colormap]]): Colormap to use for filled contour plots; can be a string name or a Matplotlib colormap object.
            extend (str): Direction to extend the colormap for out-of-range values ('neither', 'both', 'min', 'max').
            plot_type (str): Type of plot to create ('contourf' or 'contour_lines').
            max_height (Optional[float]): Optional maximum height in kilometers to set as the upper limit for height-based y-axes; if None, limits are determined from data.
            file_prefix (str): Prefix for the output filenames; additional information will be appended to this prefix.
            formats (List[str]): List of file formats to save the plots in (e.g., ['png', 'pdf']). 

        Returns:
            List[str]: A list of file paths for the created cross-section plots in the specified formats.  
        """
        if not isinstance(mpas_3d_processor, MPAS3DProcessor):
            raise ValueError("mpas_3d_processor must be an instance of MPAS3DProcessor")

        if mpas_3d_processor.dataset is None:
            raise ValueError("MPAS3DProcessor must have loaded data. Call load_3d_data() first.")

        if var_name not in mpas_3d_processor.dataset.data_vars:
            available_vars = list(mpas_3d_processor.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars[:10]}...")

        os.makedirs(output_dir, exist_ok=True)

        time_dim = 'Time' if 'Time' in mpas_3d_processor.dataset.sizes else 'time'
        total_times = mpas_3d_processor.dataset.sizes[time_dim]

        created_files: List[str] = []
        print(f"\nCreating vertical cross-section plots for {total_times} time steps...")
        print(f"Variable: {var_name}")
        print(f"Cross-section from ({start_point[0]:.2f}, {start_point[1]:.2f}) to ({end_point[0]:.2f}, {end_point[1]:.2f})")
        print(f"Vertical coordinate: {vertical_coord}")

        if max_height:
            print(f"Maximum height: {max_height} km")
        print()

        for time_idx in range(total_times):
            try:
                step_files = self._process_batch_time_step(
                    mpas_3d_processor, output_dir, var_name,
                    start_point, end_point, time_idx, total_times,
                    vertical_coord, num_points, levels, colormap,
                    extend, plot_type, max_height, file_prefix, formats,
                )
                created_files.extend(step_files)
            except Exception as e:
                print(f"Error creating cross-section for time index {time_idx}: {e}")

        print(f"\nBatch cross-section processing completed. Created {len(created_files)} files.")

        return created_files