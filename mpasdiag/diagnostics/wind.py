#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Diagnostics Module: Wind Diagnostics

This module defines the WindDiagnostics class, which provides methods for computing various diagnostics related to wind fields in MPAS model output. The class includes functions to calculate horizontal wind speed, wind direction, wind shear, and perform comprehensive analysis of wind components. It also includes helper methods for validating 3D variables, determining vertical dimensions, and extracting variable slices at specified levels. The diagnostics are designed to be flexible and informative, with options for verbose output to assist with debugging and understanding the data. The module relies on xarray for data handling and numpy for numerical operations, and it is structured to integrate seamlessly with the overall MPASdiag processing framework.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Tuple, Union, Optional, Any, cast, Dict

from mpasdiag.processing.constants import WIND_SPEED_UNITS
from mpasdiag.processing.utils_logger import get_logger
from mpasdiag.diagnostics._vertical import (
    as_dataarray as _as_dataarray,
    check_pressure_level,
    is_lazy as _is_lazy,
    interpolate_to_pressure,
    plain_dataset,
)

logger = get_logger(__name__)

_WIND_COMPONENT_RANGE_MSG = "Wind component %s range: %.2f to %.2f m/s"


class WindDiagnostics:
    """Computes diagnostics related to wind fields in MPAS model output, including wind speed, direction, shear, and component analysis."""

    def __init__(self: "WindDiagnostics", verbose: bool = True) -> None:
        """
        This constructor initializes the WindDiagnostics class with an optional verbose flag that enables detailed output messages during wind calculations. The verbose mode provides insights into the ranges and means of computed wind diagnostics, which can be helpful for identifying potential issues with the input data or understanding the characteristics of the wind fields being analyzed. By default, verbose mode is enabled, but it can be turned off if a quieter output is preferred during processing.

        Parameters:
            verbose (bool): Enable verbose output messages during wind calculations (default: True).

        Returns:
            None: This constructor does not return a value.
        """
        # Store the verbose flag for use in other methods
        self.verbose = verbose

    def compute_wind_speed(
        self: "WindDiagnostics", u_component: Any, v_component: Any
    ) -> Any:
        """
        This function computes the horizontal wind speed magnitude from the U and V wind components using the Pythagorean theorem. The wind speed is calculated as the square root of the sum of squared U and V components. The resulting DataArray includes CF-compliant metadata for units, standard name, and long name to ensure that it is properly described for downstream analysis and visualization. If verbose mode is enabled, it prints the range and mean of the computed wind speed to help identify potential issues with the input data or calculation.

        Parameters:
            u_component (Union[xr.DataArray, np.ndarray]): U (zonal) wind component in meters per second; a numpy- or dask-backed DataArray or a plain array.
            v_component (Union[xr.DataArray, np.ndarray]): V (meridional) wind component in meters per second.

        Returns:
            Union[xr.DataArray, np.ndarray]: Horizontal wind speed magnitude in meters per second, a DataArray with CF-compliant attributes (lazy when the inputs are dask-backed) or a plain array when neither input is a DataArray.
        """
        return_array = not (
            isinstance(u_component, xr.DataArray)
            or isinstance(v_component, xr.DataArray)
        )
        u_component = _as_dataarray(u_component)
        v_component = _as_dataarray(v_component)

        # Calculate wind speed using the Pythagorean theorem
        wind_speed = xr.apply_ufunc(
            np.sqrt, u_component**2 + v_component**2, keep_attrs=True, dask="allowed"
        )

        # Add CF-compliant attributes to the wind speed DataArray
        wind_speed.attrs.update(
            {
                "units": WIND_SPEED_UNITS,
                "standard_name": "wind_speed",
                "long_name": "Horizontal Wind Speed",
            }
        )

        if self.verbose and not _is_lazy(wind_speed):
            # Calculate the range of the U and V components for diagnostic purposes
            u_min, u_max = float(u_component.min()), float(u_component.max())
            v_min, v_max = float(v_component.min()), float(v_component.max())

            # Calculate the range and mean of the computed wind speed
            speed_min, speed_max = float(wind_speed.min()), float(wind_speed.max())
            speed_mean = float(wind_speed.mean())

            # Log the ranges and mean of the U and V components
            logger.debug("Wind component U range: %.2f to %.2f m/s", u_min, u_max)
            logger.debug("Wind component V range: %.2f to %.2f m/s", v_min, v_max)

            # Log the range and mean of the computed wind speed
            logger.debug("Wind speed range: %.2f to %.2f m/s", speed_min, speed_max)
            logger.debug("Wind speed mean: %.2f m/s", speed_mean)

        # Return the computed wind speed DataArray with metadata
        return wind_speed.values if return_array else wind_speed

    def compute_wind_direction(
        self: "WindDiagnostics",
        u_component: Any,
        v_component: Any,
        degrees: bool = True,
    ) -> Any:
        """
        This function computes the wind direction from the U and V wind components using the arctangent function. The direction is calculated in meteorological convention, where 0 degrees corresponds to wind coming from the north, 90 degrees from the east, 180 degrees from the south, and 270 degrees from the west. The function can return the direction in either degrees or radians based on the `degrees` parameter. The resulting DataArray includes CF-compliant metadata for units, standard name, long name, and a note describing the convention used for wind direction. If verbose mode is enabled, it prints the range and mean of the computed wind direction to help identify potential issues with the input data or calculation.

        Parameters:
            u_component (Union[xr.DataArray, np.ndarray]): U (zonal) wind component in meters per second; a numpy- or dask-backed DataArray or a plain array.
            v_component (Union[xr.DataArray, np.ndarray]): V (meridional) wind component in meters per second.
            degrees (bool): Return wind direction in degrees if True, or radians if False (default: True).

        Returns:
            Union[xr.DataArray, np.ndarray]: Wind direction in degrees or radians, a DataArray with CF-compliant attributes (lazy when the inputs are dask-backed) or a plain array when neither input is a DataArray.
        """
        return_array = not (
            isinstance(u_component, xr.DataArray)
            or isinstance(v_component, xr.DataArray)
        )
        u_component = _as_dataarray(u_component)
        v_component = _as_dataarray(v_component)

        # Meteorological "from" direction: atan2(-u, -v) measured clockwise from north
        direction_rad = xr.apply_ufunc(
            np.arctan2, -u_component, -v_component, keep_attrs=True, dask="allowed"
        )

        # Calm winds (u == v == 0) have no defined direction; use the conventional 0
        calm = (u_component == 0) & (v_component == 0)
        direction_rad = direction_rad.where(~calm, 0.0)

        if degrees:
            # Convert radians to degrees
            direction_deg = xr.apply_ufunc(
                np.rad2deg, direction_rad, keep_attrs=True, dask="allowed"
            )

            # Ensure direction is within 0-360 degrees
            direction_deg = direction_deg % 360

            # Floating-point wrap-around can yield exactly 360; fold it back to 0
            direction_deg = direction_deg.where(~(direction_deg >= 360.0), 0.0)

            # Add CF-compliant attributes to the wind direction DataArray
            direction_deg.attrs.update(
                {
                    "units": "degrees",
                    "standard_name": "wind_from_direction",
                    "long_name": "Wind Direction (meteorological convention)",
                    "note": "Direction wind is coming from, 0=North, 90=East, 180=South, 270=West",
                }
            )

            if self.verbose and not _is_lazy(direction_deg):
                # Calculate the range and mean of the computed wind direction
                dir_min, dir_max = float(direction_deg.min()), float(
                    direction_deg.max()
                )
                dir_mean = float(direction_deg.mean())

                # Log the range and mean of the computed wind direction
                logger.debug(
                    "Wind direction range: %.1f to %.1f degrees", dir_min, dir_max
                )
                logger.debug("Wind direction mean: %.1f degrees", dir_mean)

            # Return the computed wind direction in degrees with metadata
            return direction_deg.values if return_array else direction_deg
        else:
            # Ensure direction is within 0-2π radians
            direction_rad = direction_rad % (2 * np.pi)

            # Floating-point wrap-around can yield exactly 2π; fold it back to 0
            direction_rad = direction_rad.where(~(direction_rad >= 2 * np.pi), 0.0)

            # Add CF-compliant attributes to the wind direction DataArray
            direction_rad.attrs.update(
                {
                    "units": "radians",
                    "standard_name": "wind_from_direction",
                    "long_name": "Wind Direction (meteorological convention)",
                    "note": "Direction wind is coming from, 0=North, π/2=East, π=South, 3π/2=West",
                }
            )

            if self.verbose and not _is_lazy(direction_rad):
                # Calculate the range and mean of the computed wind direction in radians
                dir_min, dir_max = float(direction_rad.min()), float(
                    direction_rad.max()
                )
                dir_mean = float(direction_rad.mean())

                # Log the range and mean of the computed wind direction in radians
                logger.debug(
                    "Wind direction range: %.3f to %.3f radians", dir_min, dir_max
                )
                logger.debug("Wind direction mean: %.3f radians", dir_mean)

            # Return the computed wind direction in radians with metadata
            return direction_rad.values if return_array else direction_rad

    def analyze_wind_components(
        self: "WindDiagnostics",
        u_component: xr.DataArray,
        v_component: xr.DataArray,
        w_component: Optional[xr.DataArray] = None,
    ) -> Dict[str, Any]:
        """
        This function performs a comprehensive analysis of the U, V, and optionally W wind components by calculating summary statistics (minimum, maximum, mean, standard deviation) for each component, as well as the horizontal wind speed and wind direction derived from U and V. If the W component is provided, it also calculates the total 3D wind speed. The results are returned in a dictionary format that includes the computed statistics and units for each variable. If verbose mode is enabled, it prints the ranges and means of the components, horizontal speed, and direction to help identify potential issues with the input data or calculations. This function provides a convenient way to summarize key characteristics of the wind fields for diagnostic purposes.

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.
            w_component (Optional[xr.DataArray]): W (vertical) wind component in meters per second (optional).

        Returns:
            Dict[str, Any]: A dictionary containing summary statistics and units for U, V, W (if provided), horizontal wind speed, and wind direction.
        """
        # Initialize a dictionary to store the analysis results
        analysis = {}

        # Calculate summary statistics for U components
        analysis["u_component"] = {
            "min": float(u_component.min()),
            "max": float(u_component.max()),
            "mean": float(u_component.mean()),
            "std": float(u_component.std()),
            "units": u_component.attrs.get("units", WIND_SPEED_UNITS),
        }

        # Calculate summary statistics for V components
        analysis["v_component"] = {
            "min": float(v_component.min()),
            "max": float(v_component.max()),
            "mean": float(v_component.mean()),
            "std": float(v_component.std()),
            "units": v_component.attrs.get("units", WIND_SPEED_UNITS),
        }

        # Calculate horizontal wind speed
        wind_speed = self.compute_wind_speed(u_component, v_component)

        # Calculate summary statistics for horizontal wind speed
        analysis["horizontal_speed"] = {
            "min": float(wind_speed.min()),
            "max": float(wind_speed.max()),
            "mean": float(wind_speed.mean()),
            "std": float(wind_speed.std()),
            "units": WIND_SPEED_UNITS,
        }

        # Calculate wind direction from U and V components
        wind_direction = self.compute_wind_direction(
            u_component, v_component, degrees=True
        )

        # Direction is circular: use the vector (circular) mean and circular standard deviation, not the arithmetic ones
        dir_rad = np.deg2rad(np.asarray(wind_direction, dtype=np.float64))
        dir_rad = dir_rad[np.isfinite(dir_rad)]
        mean_sin, mean_cos = float(np.mean(np.sin(dir_rad))), float(
            np.mean(np.cos(dir_rad))
        )
        resultant = min(float(np.hypot(mean_sin, mean_cos)), 1.0)

        # Calculate summary statistics for wind direction (min/max stay plain; mean/std are circular)
        analysis["direction"] = {
            "min": float(wind_direction.min()),
            "max": float(wind_direction.max()),
            "mean": float(
                np.round(np.rad2deg(np.arctan2(mean_sin, mean_cos)), 10) % 360.0
            ),
            "std": float(
                np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))
                if resultant > 0
                else np.nan
            ),
            "units": "degrees",
        }

        if w_component is not None:
            # Calculate summary statistics for W components
            analysis["w_component"] = {
                "min": float(w_component.min()),
                "max": float(w_component.max()),
                "mean": float(w_component.mean()),
                "std": float(w_component.std()),
                "units": w_component.attrs.get("units", WIND_SPEED_UNITS),
            }

            # Calculate total 3D wind speed from U, V, and W components
            wind_speed_3d = np.sqrt(u_component**2 + v_component**2 + w_component**2)

            # Calculate summary statistics for total 3D wind speed
            analysis["total_speed"] = {
                "min": float(wind_speed_3d.min()),
                "max": float(wind_speed_3d.max()),
                "mean": float(wind_speed_3d.mean()),
                "std": float(wind_speed_3d.std()),
                "units": WIND_SPEED_UNITS,
            }

        if self.verbose:
            logger.info("Wind Component Analysis:")

            # Log the ranges and means of the U component
            logger.info(
                "  U component: %.2f to %.2f m/s (mean: %.2f)",
                analysis["u_component"]["min"],
                analysis["u_component"]["max"],
                analysis["u_component"]["mean"],
            )

            # Log the ranges and means of the V component
            logger.info(
                "  V component: %.2f to %.2f m/s (mean: %.2f)",
                analysis["v_component"]["min"],
                analysis["v_component"]["max"],
                analysis["v_component"]["mean"],
            )

            # Log the range and mean of the horizontal wind speed
            logger.info(
                "  Horizontal speed: %.2f to %.2f m/s (mean: %.2f)",
                analysis["horizontal_speed"]["min"],
                analysis["horizontal_speed"]["max"],
                analysis["horizontal_speed"]["mean"],
            )

            # Log the range and mean of the wind direction
            logger.info(
                "  Direction: %.1f to %.1f degrees (mean: %.1f)",
                analysis["direction"]["min"],
                analysis["direction"]["max"],
                analysis["direction"]["mean"],
            )

            if w_component is not None:
                # Log the range and mean of the W component
                logger.info(
                    "  W component: %.2f to %.2f m/s (mean: %.2f)",
                    analysis["w_component"]["min"],
                    analysis["w_component"]["max"],
                    analysis["w_component"]["mean"],
                )

                # Log the range and mean of the total 3D wind speed
                logger.info(
                    "  Total 3D speed: %.2f to %.2f m/s (mean: %.2f)",
                    analysis["total_speed"]["min"],
                    analysis["total_speed"]["max"],
                    analysis["total_speed"]["mean"],
                )

        # Return the analysis results as a dictionary
        return analysis

    def compute_wind_shear(
        self: "WindDiagnostics",
        u_upper: Any,
        v_upper: Any,
        u_lower: Any,
        v_lower: Any,
    ) -> Tuple[Any, Any]:
        """
        This function computes the wind shear magnitude and direction between two vertical levels using the U and V wind components at the upper and lower levels. The shear magnitude is calculated as the square root of the sum of squared differences in U and V components between the two levels. The shear direction is computed using the arctangent function to determine the direction of the shear vector, following meteorological convention. Both the shear magnitude and direction are returned as xarray DataArrays with CF-compliant attributes for units, standard name, and long name. If verbose mode is enabled, it prints the range and mean of the computed wind shear magnitude to help identify potential issues with the input data or calculation.

        Parameters:
            u_upper (Union[xr.DataArray, np.ndarray]): U component at the upper level in meters per second; numpy- or dask-backed DataArrays or plain arrays are accepted.
            v_upper (Union[xr.DataArray, np.ndarray]): V component at the upper level in meters per second.
            u_lower (Union[xr.DataArray, np.ndarray]): U component at the lower level in meters per second.
            v_lower (Union[xr.DataArray, np.ndarray]): V component at the lower level in meters per second.

        Returns:
            Tuple[Union[xr.DataArray, np.ndarray], Union[xr.DataArray, np.ndarray]]: The wind shear magnitude (in m/s) and shear direction (in degrees), DataArrays with CF-compliant attributes (lazy when the inputs are dask-backed) or plain arrays when no input is a DataArray.
        """
        return_array = not any(
            isinstance(x, xr.DataArray) for x in (u_upper, v_upper, u_lower, v_lower)
        )
        u_upper, v_upper = _as_dataarray(u_upper), _as_dataarray(v_upper)
        u_lower, v_lower = _as_dataarray(u_lower), _as_dataarray(v_lower)

        # Calculate the differences in U components between the upper and lower levels
        delta_u = u_upper - u_lower

        # Calculate the differences in V components between the upper and lower levels
        delta_v = v_upper - v_lower

        # Calculate the wind shear magnitude using the Pythagorean theorem
        shear_magnitude = xr.apply_ufunc(
            np.sqrt, delta_u**2 + delta_v**2, keep_attrs=True, dask="allowed"
        )

        # Add CF-compliant attributes to the wind shear magnitude DataArray
        shear_magnitude.attrs.update(
            {
                "units": WIND_SPEED_UNITS,
                "standard_name": "wind_shear_magnitude",
                "long_name": "wind shear magnitude",
            }
        )

        # Calculate the wind shear direction using arctangent and convert to degrees
        shear_direction = self.compute_wind_direction(delta_u, delta_v, degrees=True)
        shear_direction.attrs["long_name"] = "wind shear direction"

        if self.verbose and not _is_lazy(shear_magnitude):
            # Calculate the range and mean of the computed wind shear magnitude
            shear_min, shear_max = float(shear_magnitude.min()), float(
                shear_magnitude.max()
            )
            shear_mean = float(shear_magnitude.mean())

            # Log the range and mean of the computed wind shear magnitude
            logger.debug(
                "Wind shear magnitude range: %.2f to %.2f m/s",
                shear_min,
                shear_max,
            )
            logger.debug("Wind shear magnitude mean: %.2f m/s", shear_mean)

        # Return the computed wind shear magnitude and direction as a tuple of DataArrays
        if return_array:
            return shear_magnitude.values, shear_direction.values
        return shear_magnitude, shear_direction

    def _validate_3d_variable(
        self: "WindDiagnostics", dataset: xr.Dataset, var_name: str
    ) -> None:
        """
        This helper function checks whether the specified variable in the dataset is a 3D atmospheric variable by verifying that it contains either 'nVertLevels' or 'nVertLevelsP1' as one of its dimensions. If the variable is not found in the dataset or does not have the required vertical dimension, the function raises a ValueError with an appropriate message. This validation step is crucial to ensure that subsequent calculations that rely on vertical indexing are performed on compatible variables, thereby preventing errors and ensuring the integrity of the diagnostics.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Variable name to validate.

        Returns:
            None: This function does not return a value but raises an error if validation fails.
        """
        # Check if the variable exists in the dataset
        if var_name not in dataset.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in dataset")

        # Check if the variable has the required vertical dimension
        var_dims = dataset[var_name].dims

        # Raise an error if neither 'nVertLevels' nor 'nVertLevelsP1' is found
        if "nVertLevels" not in var_dims and "nVertLevelsP1" not in var_dims:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")

    def _get_vertical_dimension(
        self: "WindDiagnostics", dataset: xr.Dataset, var_name: str
    ) -> str:
        """
        This helper function determines the name of the vertical dimension for a given variable in the dataset by checking for the presence of 'nVertLevels' or 'nVertLevelsP1' in the variable's dimensions. It returns the name of the vertical dimension that is found, which can then be used for indexing and slicing operations in subsequent calculations. This function assumes that the variable has already been validated as a 3D atmospheric variable using the `_validate_3d_variable` method, ensuring that one of the expected vertical dimensions is present.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Variable name for which to determine the vertical dimension.

        Returns:
            str: The name of the vertical dimension ('nVertLevels' or 'nVertLevelsP1').
        """
        # Assumes variable has already been validated as 3D
        var_dims = dataset[var_name].dims

        # Return the name of the vertical dimension that is found
        return "nVertLevels" if "nVertLevels" in var_dims else "nVertLevelsP1"

    @staticmethod
    def _n_selectable_levels(dataset: xr.Dataset, vertical_dim: str) -> int:
        """
        This helper returns the number of model levels that an integer index or 'top' can select along a vertical dimension: the size of nVertLevels for mass-level variables and nVertLevelsP1 - 1 for interface-level variables, whose level values are the average of two adjacent interfaces (one per mass level).

        Parameters:
            dataset (xr.Dataset): Input dataset.
            vertical_dim (str): Vertical dimension name ('nVertLevels' or 'nVertLevelsP1').

        Returns:
            int: Number of selectable levels.
        """
        size = int(dataset.sizes.get(vertical_dim, 0))
        return size - 1 if vertical_dim == "nVertLevelsP1" else size

    def _compute_level_index(
        self: "WindDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        level_spec: Union[str, int, float],
        time_dim: str,
        time_index: int,
    ) -> int:
        """
        This helper function computes the vertical level index based on the provided specification, which can be an integer index, a pressure value in Pascals, or a string identifier such as 'surface' or 'top'. If the specification is an integer, it checks that the index is within the available levels in the dataset. For a variable on interface levels (nVertLevelsP1, such as 'w') the index refers to a mass level: the level value is the average of interfaces k and k+1 (see _extract_variable_slice), so there are nVertLevelsP1 - 1 selectable levels and 'top' is the top mass level, not the model lid interface (where w is always 0); this co-locates u, v and w, as the pressure-level path does. A float is a pressure in Pa and cannot be reduced to one level index (the model level at a given pressure differs from column to column), so it is validated and rejected here; get_3d_variable_at_level interpolates pressure levels per column instead. If it is a string, it checks for 'surface' (which corresponds to index 0) and 'top' (which corresponds to the maximum vertical level index). If the specification does not match any of these formats, it raises a ValueError. This function provides a flexible way to specify vertical levels for diagnostics, allowing users to choose levels based on their preferred method of identification.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable and pressure diagnostics.
            var_name (str): Name of the variable for which to compute the level index.
            level_spec (Union[str, int, float]): Specification of the vertical level (integer index, pressure in Pa, or 'surface'/'top').
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Time index to use for selecting pressure data if needed.

        Returns:
            int: The computed vertical level index based on the provided specification.
        """
        # Determine the vertical dimension name for the variable
        vertical_dim = self._get_vertical_dimension(dataset, var_name)

        # Handle integer specifications by validating the index against the available levels in the dataset
        if isinstance(level_spec, (int, np.integer)) and not isinstance(
            level_spec, bool
        ):
            level_idx = int(level_spec)
            max_levels = self._n_selectable_levels(dataset, vertical_dim)
            if level_idx >= max_levels or level_idx < -max_levels:
                raise ValueError(
                    f"Model level {level_idx} exceeds available levels {max_levels}"
                )
            if level_idx < 0 and vertical_dim == "nVertLevelsP1":
                level_idx += max_levels
            return level_idx

        # A pressure has no single model level index (the level at a given pressure differs per column)
        elif isinstance(level_spec, (float, np.floating)) and not isinstance(
            level_spec, bool
        ):
            check_pressure_level(float(level_spec), var_name)
            raise ValueError(
                f"Pressure level {level_spec} Pa has no single model level index; "
                f"get_3d_variable_at_level interpolates to the pressure in every column."
            )

        # Handle string specifications for surface and top levels
        elif isinstance(level_spec, str):
            if level_spec.lower() == "surface":
                return 0
            elif level_spec.lower() == "top":
                return self._n_selectable_levels(dataset, vertical_dim) - 1
            else:
                raise ValueError(f"Unknown level specification: {level_spec}")
        else:
            raise ValueError(f"Invalid level specification: {level_spec}")

    def _extract_variable_slice(
        self: "WindDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        time_dim: str,
        time_index: int,
        vertical_dim: str,
        level_idx: int,
        data_type: str,
    ) -> xr.DataArray:
        """
        This helper function extracts a slice of the specified variable from the dataset at the given time index and vertical level index; for a variable on interface levels (nVertLevelsP1) the slice is the mass-level value, the mean of interfaces level_idx and level_idx + 1. It handles both 'xarray' and 'uxarray' dataset access styles by checking for the presence of the `__getitem__` method when using 'uxarray'. The function uses the `isel` method to select the appropriate time and vertical level indices, and it ensures that any lazy computations are executed by calling `compute()` if the resulting DataArray has that method. The extracted variable slice is returned as an xarray DataArray, ready for further analysis and diagnostics.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Name of the variable to extract.
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Time index to select.
            vertical_dim (str): Name of the vertical dimension in the dataset.
            level_idx (int): Vertical level index to select.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: The extracted variable slice at the specified time and vertical level, ready for analysis.
        """
        # Interface-level variables are destaggered to the mass level: mean of interfaces k and k+1
        indices: Any = (
            [level_idx, level_idx + 1] if vertical_dim == "nVertLevelsP1" else level_idx
        )

        # Extract the variable slice based on the specified data access style
        if data_type == "uxarray" and hasattr(dataset, "__getitem__"):
            var_data = dataset[var_name][time_index].isel({vertical_dim: indices})
        else:
            var_data = dataset[var_name].isel(
                {time_dim: time_index, vertical_dim: indices}
            )

        if vertical_dim == "nVertLevelsP1":
            attrs = dict(var_data.attrs)
            var_data = var_data.mean(vertical_dim, skipna=False)
            var_data.attrs.update(attrs)

        # Compute the variable if it is lazy (e.g., Dask array)
        if hasattr(var_data, "compute"):
            var_data = cast(Any, var_data).compute()

        # Return the extracted variable slice as an xarray DataArray
        return var_data

    def get_3d_variable_at_level(
        self: "WindDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        level: Union[str, int, float],
        time_index: int,
        data_type: str = "xarray",
    ) -> xr.DataArray:
        """
        This function retrieves a 3D variable from the dataset at a specified vertical level and time index, with support for both 'xarray' and 'uxarray' dataset access styles. It first validates that the variable is a 3D atmospheric variable using the `_validate_3d_variable` method. Then it determines the time dimension and validates the time index using the `MPASDateTimeUtils.validate_time_parameters` method. Next, a float level is treated as a pressure in Pascals and the field is interpolated to it in every column separately, linearly in log(pressure), using each column's own total pressure (NaN where the pressure lies below the ground or above the model top; the attrs record 'actual_pressure_level'); an integer index or 'surface'/'top' is resolved with the `_compute_level_index` method. For interface-level variables (such as 'w') an integer index or 'top' selects a mass level and the returned values are the average of the two bracketing interfaces, so u, v and w are co-located. Finally, it extracts the variable slice at the specified time and vertical level using the `_extract_variable_slice` method and returns it as an xarray DataArray with metadata attributes indicating the selected level and index. This function provides a flexible and robust way to access 3D variables at specific levels for wind diagnostics and other analyses.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Name of the variable to retrieve.
            level (Union[str, int, float]): Specification of the vertical level for retrieval (integer index, pressure in Pa, or 'surface'/'top').
            time_index (int): Time index to select for retrieval.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            xr.DataArray: The extracted variable slice at the specified time and vertical level, with metadata attributes for the selected level and index.
        """
        from mpasdiag.processing.utils_datetime import MPASDateTimeUtils

        # Validate that the specified variable is a 3D atmospheric variable
        self._validate_3d_variable(dataset, var_name)

        # Validate time parameters and determine the time dimension name
        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            dataset, time_index, self.verbose
        )

        # Pressure levels: interpolate in log(p) in every column (NaN below ground / above the model top)
        if isinstance(level, (float, np.floating)) and not isinstance(level, bool):
            return interpolate_to_pressure(
                dataset, var_name, float(level), time_dim, validated_time_index
            )

        # Compute the vertical level index based on the provided level specification
        level_idx = self._compute_level_index(
            dataset, var_name, level, time_dim, validated_time_index
        )

        # Determine the vertical dimension name for the variable
        vertical_dim = self._get_vertical_dimension(dataset, var_name)

        # Extract the variable slice at the specified time and vertical level
        var_data = self._extract_variable_slice(
            dataset,
            var_name,
            time_dim,
            validated_time_index,
            vertical_dim,
            level_idx,
            data_type,
        )

        # Add attributes to indicate the selected level and index
        var_data.attrs["selected_level"] = level
        var_data.attrs["level_index"] = level_idx

        # Return the extracted variable slice
        return var_data

    def _extract_w_component_with_fallback(
        self: "WindDiagnostics",
        dataset: xr.Dataset,
        w_variable: str,
        level: Union[str, int, float],
        time_index: int,
        data_type: str,
        u_data: xr.DataArray,
    ) -> xr.DataArray:
        """
        This function attempts to extract the W (vertical) wind component from the dataset at a specified vertical level and time index. If the extraction is successful, it returns the W component as an xarray DataArray. However, if the extraction fails due to missing variables, invalid level specifications, or other issues, it catches the exception and creates a fallback W component filled with zeros that has the same shape and metadata as the U component. The fallback W component is assigned appropriate attributes to indicate that it is a zero vertical velocity and includes a note about the failed extraction. This approach ensures that downstream diagnostics can still be performed even if the W component is not available in the dataset, while providing clear information about the reason for using the fallback data.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the wind variables.
            w_variable (str): Name of the W-component variable to extract.
            level (Union[str, int, float]): Specification of the vertical level for extraction (integer index, pressure in Pa, or 'surface'/'top').
            time_index (int): Time index to select for extraction.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.
            u_data (xr.DataArray): The U component DataArray used to create a zero-filled fallback if W extraction fails.

        Returns:
            xr.DataArray: The extracted W component if successful, or a zero-filled fallback DataArray with appropriate attributes if extraction fails.
        """
        try:
            # Extract the W component using the same method as U and V components
            return self.get_3d_variable_at_level(
                dataset, w_variable, level, time_index, data_type
            )
        except (ValueError, IndexError) as e:
            # Log a warning message about the failed extraction of the W component
            if self.verbose:
                logger.warning(
                    "Could not extract %s at level %s: %s",
                    w_variable,
                    level,
                    e,
                )
                logger.warning("Setting W component to zero")

            # Create a zero-filled DataArray with the same shape and metadata as the U component
            w_data = xr.zeros_like(u_data)
            w_data.attrs["units"] = WIND_SPEED_UNITS
            w_data.attrs["long_name"] = (
                f"Zero vertical velocity (could not extract {w_variable})"
            )

            # Return the zero-filled fallback W component
            return w_data

    def _print_wind_component_diagnostics(
        self: "WindDiagnostics",
        u_data: xr.DataArray,
        v_data: xr.DataArray,
        w_data: xr.DataArray,
        u_variable: str,
        v_variable: str,
        w_variable: str,
    ) -> None:
        """
        This helper function prints diagnostic information about the extracted U, V, and W wind components, including their ranges and units. It calculates the horizontal wind speed from the U and V components and also prints its range. The function checks if verbose mode is enabled before printing any information, allowing users to control the level of output during processing. This diagnostic output can help identify potential issues with the extracted wind components, such as unexpected ranges or missing data, and provides context for understanding the characteristics of the wind fields being analyzed.

        Parameters:
            u_data (xr.DataArray): Extracted U component DataArray.
            v_data (xr.DataArray): Extracted V component DataArray.
            w_data (xr.DataArray): Extracted W component DataArray (or fallback).
            u_variable (str): Name of the U component variable.
            v_variable (str): Name of the V component variable.
            w_variable (str): Name of the W component variable.

        Returns:
            None: This function does not return a value but prints diagnostic information to the console if verbose mode is enabled.
        """
        # Check if verbose mode is enabled before printing diagnostics
        if not self.verbose:
            return

        # Calculate the horizontal wind speed from the U and V components
        wind_speed = np.sqrt(u_data**2 + v_data**2)

        # Calculate the range of the U, V, and W components for diagnostics
        u_min, u_max = float(u_data.min()), float(u_data.max())
        v_min, v_max = float(v_data.min()), float(v_data.max())
        w_min, w_max = float(w_data.min()), float(w_data.max())

        # Calculate the range of the horizontal wind speed for diagnostics
        wind_min, wind_max = float(wind_speed.min()), float(wind_speed.max())

        # Log the ranges of the U, V, and W components along with the horizontal wind speed range
        logger.debug(_WIND_COMPONENT_RANGE_MSG, u_variable, u_min, u_max)
        logger.debug(_WIND_COMPONENT_RANGE_MSG, v_variable, v_min, v_max)
        logger.debug(_WIND_COMPONENT_RANGE_MSG, w_variable, w_min, w_max)
        logger.debug(
            "Horizontal wind speed range: %.2f to %.2f m/s", wind_min, wind_max
        )

        # Log the units of the U component for reference
        u_units = u_data.attrs.get("units", WIND_SPEED_UNITS)
        logger.debug("Units: %s", u_units)

    def get_3d_wind_components(
        self: "WindDiagnostics",
        dataset: xr.Dataset,
        u_variable: str,
        v_variable: str,
        w_variable: str = "w",
        level: Union[str, int, float] = 0,
        time_index: int = 0,
        data_type: str = "xarray",
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        This function extracts the U, V, and W wind components from a 3D MPAS dataset at a specified vertical level and time index. It validates the presence of the specified variables in the dataset, checks that the time index is within bounds, and retrieves the variable slices for each component using the `get_3d_variable_at_level` method. If the W component cannot be extracted due to missing variables or invalid level specifications, it creates a fallback W component filled with zeros using the `_extract_w_component_with_fallback` method. The function includes diagnostic print statements that provide information about the ranges of the extracted components and their units if verbose mode is enabled. The resulting U, V, and W components are returned as xarray DataArrays with CF-compliant attributes and metadata for the selected level. If any issues arise during extraction (e.g., missing variables, out-of-range time index), it raises appropriate exceptions with informative messages to help diagnose the problem.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the wind variables.
            u_variable (str): Name of the U-component variable to extract.
            v_variable (str): Name of the V-component variable to extract.
            w_variable (str): Name of the W-component variable to extract (default: 'w').
            level (Union[str, int, float]): Specification of the vertical level for extraction (integer index, pressure in Pa, or 'surface'/'top') (default: 0).
            time_index (int): Time index to select for extraction (default: 0).
            data_type (str): Dataset access style, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            Tuple[xr.DataArray, xr.DataArray, xr.DataArray]: A tuple containing the extracted U, V, and W components as xarray DataArrays with CF-compliant attributes and metadata for the selected level.
        """
        # Log the variables and level being extracted if verbose mode is enabled
        if self.verbose:
            logger.debug(
                "Extracting 3D wind components %s, %s, %s at level %s, time index %d",
                u_variable,
                v_variable,
                w_variable,
                level,
                time_index,
            )

        try:
            # Extract the U and V components at the specified level and time index
            u_data = self.get_3d_variable_at_level(
                dataset, u_variable, level, time_index, data_type
            )
            v_data = self.get_3d_variable_at_level(
                dataset, v_variable, level, time_index, data_type
            )

            # Extract the W component with a fallback to zeros if extraction fails
            w_data = self._extract_w_component_with_fallback(
                dataset, w_variable, level, time_index, data_type, u_data
            )

            # Print diagnostic information about the extracted wind components if verbose mode is enabled
            self._print_wind_component_diagnostics(
                u_data, v_data, w_data, u_variable, v_variable, w_variable
            )

            # Return the extracted U, V, and W components as a tuple of xarray DataArrays
            return u_data, v_data, w_data

        except ValueError as e:
            # Check which variables are available in the dataset
            available_vars = list(dataset.data_vars.keys())

            # Check which of the requested variables are missing from the dataset
            missing_vars = [
                var
                for var in [u_variable, v_variable, w_variable]
                if var not in available_vars
            ]

            # Raise a ValueError with information about missing variables if any are not found
            if missing_vars:
                raise ValueError(
                    f"3D wind variables {missing_vars} not found in dataset. Available variables: {available_vars[:20]}..."
                )
            else:
                raise e

    @staticmethod
    def _layer_pressure_weights(
        pressure: np.ndarray, p_bottom: float, p_top: float
    ) -> np.ndarray:
        """
        This helper computes, for every column, the pressure thickness that each model level contributes to the layer between p_top and p_bottom. Level interfaces in pressure are the midpoints between neighbouring mass-level pressures, extended half a level beyond the lowest and highest levels, and the weight of a level is the overlap of its interface pair with the requested pressure layer. Because the overlap is clipped to the column's own pressure range, a column whose surface pressure is below p_bottom (high terrain) simply has its layer start at the ground, and levels above p_top or below the ground get zero weight. Pressure is assumed to decrease with level index, as in MPAS.

        Parameters:
            pressure (np.ndarray): Total pressure in Pa with shape (n_columns, n_levels), decreasing with level index.
            p_bottom (float): Bottom pressure of the layer in Pa.
            p_top (float): Top pressure of the layer in Pa.

        Returns:
            np.ndarray: Non-negative weights in Pa with the same shape as pressure.
        """
        # Interface pressures: midpoints between mass levels, extrapolated half a level at the bottom and top
        mid = 0.5 * (pressure[:, :-1] + pressure[:, 1:])
        bottom = pressure[:, :1] + 0.5 * (pressure[:, :1] - pressure[:, 1:2])
        top = np.maximum(
            pressure[:, -1:] - 0.5 * (pressure[:, -2:-1] - pressure[:, -1:]), 0.0
        )
        interfaces = np.concatenate([bottom, mid, top], axis=1)

        # Overlap of each level's [upper, lower] interface pair with the requested layer
        lower = np.minimum(interfaces[:, :-1], p_bottom)
        upper = np.maximum(interfaces[:, 1:], p_top)
        return np.clip(lower - upper, 0.0, None)

    def compute_storm_motion(
        self: "WindDiagnostics",
        processor: Any,
        u_var: str = "uReconstructZonal",
        v_var: str = "uReconstructMeridional",
        time_index: int = 0,
        p_bottom: float = 100000.0,
        p_top: float = 50000.0,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This function computes the storm motion estimate (u_storm, v_storm) separately in every column as the pressure-weighted mean wind of the layer between p_bottom and p_top (default 1000 to 500 hPa, the usual layer-mean steering-flow proxy). Each column uses its own total pressure (pressure_p + pressure_base), so the layer follows the terrain: where the ground pressure is below p_bottom the layer starts at the surface, and columns whose layer is empty (ground above p_top) are NaN. The weight of each model level is its pressure thickness inside the layer, so the result is a true mass-weighted layer mean and not a domain-wide average, which would be meaningless on a global mesh. The two returned fields are DataArrays over the horizontal dimension (nCells) with units m s-1, positive eastward and northward, and attrs recording the layer. For a single sounding, select the cell of interest from the returned arrays.

        Parameters:
            processor (Any): An MPAS3DProcessor-like object whose dataset holds the wind components, 'pressure_p' and 'pressure_base'.
            u_var (str): Name of the U-component variable in the dataset (default: 'uReconstructZonal').
            v_var (str): Name of the V-component variable in the dataset (default: 'uReconstructMeridional').
            time_index (int): Time index to select for the calculation (default: 0).
            p_bottom (float): Bottom pressure of the layer in Pascals (default: 100000 Pa, i.e., 1000 hPa).
            p_top (float): Top pressure of the layer in Pascals (default: 50000 Pa, i.e., 500 hPa).

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: (u_storm, v_storm) per-column storm motion components in m s-1, positive eastward and northward respectively.
        """
        from mpasdiag.processing.utils_datetime import MPASDateTimeUtils

        if p_top >= p_bottom:
            raise ValueError(
                f"p_top ({p_top} Pa) must be smaller than p_bottom ({p_bottom} Pa)."
            )

        ds_raw = plain_dataset(processor.dataset)

        if "pressure_p" not in ds_raw or "pressure_base" not in ds_raw:
            raise ValueError(
                "Cannot compute storm motion - pressure data not available"
            )

        # Validate the time index and determine the time dimension name
        time_dim, t_idx, _ = MPASDateTimeUtils.validate_time_parameters(
            ds_raw, time_index, self.verbose
        )

        # Per-column total pressure and wind, arranged as (columns, levels)
        total_p = (
            ds_raw["pressure_p"].isel({time_dim: t_idx})
            + ds_raw["pressure_base"].isel({time_dim: t_idx})
        ).compute()
        horiz_dims = [d for d in total_p.dims if d != "nVertLevels"]
        p_vals = total_p.transpose(*horiz_dims, "nVertLevels").values
        n_lev = p_vals.shape[-1]
        p_2d = p_vals.reshape(-1, n_lev).astype(np.float64)

        # Pressure-thickness weights of every level inside the layer, per column
        weights = self._layer_pressure_weights(p_2d, p_bottom, p_top)
        w_sum = weights.sum(axis=1)

        results = []
        for name in (u_var, v_var):
            comp = ds_raw[name].isel({time_dim: t_idx}).compute()
            vals = comp.transpose(*horiz_dims, "nVertLevels").values
            vals_2d = vals.reshape(-1, n_lev).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(
                    w_sum > 0.0, (weights * vals_2d).sum(axis=1) / w_sum, np.nan
                )
            out = comp.isel({"nVertLevels": 0}, drop=True).copy(
                data=mean.reshape(vals.shape[:-1])
            )
            out.attrs = {
                "units": WIND_SPEED_UNITS,
                "long_name": f"storm motion ({name}), pressure-weighted layer mean",
                "layer_bottom_pa": float(p_bottom),
                "layer_top_pa": float(p_top),
            }
            results.append(out)

        # Print summary statistics of the calculated storm motion
        if self.verbose:
            logger.debug(
                "Storm motion (per-column pressure-weighted mean, %.0f-%.0f hPa): "
                "u_storm %+.2f..%+.2f m s-1, v_storm %+.2f..%+.2f m s-1 (%d of %d columns defined)",
                p_bottom / 100,
                p_top / 100,
                float(np.nanmin(results[0].values)) if np.any(w_sum > 0) else np.nan,
                float(np.nanmax(results[0].values)) if np.any(w_sum > 0) else np.nan,
                float(np.nanmin(results[1].values)) if np.any(w_sum > 0) else np.nan,
                float(np.nanmax(results[1].values)) if np.any(w_sum > 0) else np.nan,
                int(np.sum(w_sum > 0)),
                len(w_sum),
            )

        # Return the per-column storm motion components
        return results[0], results[1]

    def _require_wind_variables(
        self: "WindDiagnostics",
        dataset: xr.Dataset,
        u_variable: str,
        v_variable: str,
    ) -> None:
        """
        This function verifies that both wind component variables exist in the dataset and raises a ValueError listing the missing ones (and a sample of the available variables) otherwise.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the wind variables.
            u_variable (str): Name of the U-component variable.
            v_variable (str): Name of the V-component variable.

        Returns:
            None
        """
        available_vars = list(dataset.data_vars.keys())
        missing_vars = []

        # Check if the U variable is missing from the dataset
        if u_variable not in available_vars:
            missing_vars.append(u_variable)

        # Check if the V variable is missing from the dataset
        if v_variable not in available_vars:
            missing_vars.append(v_variable)

        # Raise a ValueError with information about missing variables
        if missing_vars:
            raise ValueError(
                f"Wind variables {missing_vars} not found in dataset. Available variables: {available_vars[:20]}..."
            )

    def _log_wind_component_diagnostics(
        self: "WindDiagnostics",
        u_data: xr.DataArray,
        v_data: xr.DataArray,
        u_variable: str,
        v_variable: str,
        u_range: Tuple[float, float],
        v_range: Tuple[float, float],
        speed_range: Tuple[float, float],
    ) -> None:
        """
        This function logs the value ranges of the U and V components and of the horizontal wind speed, together with their units, and warns when the two components carry different units.

        Parameters:
            u_data (xr.DataArray): Extracted U component.
            v_data (xr.DataArray): Extracted V component.
            u_variable (str): Name of the U-component variable.
            v_variable (str): Name of the V-component variable.
            u_range (Tuple[float, float]): Minimum and maximum of the U component.
            v_range (Tuple[float, float]): Minimum and maximum of the V component.
            speed_range (Tuple[float, float]): Minimum and maximum of the wind speed.

        Returns:
            None
        """
        u_min, u_max = u_range
        v_min, v_max = v_range
        wind_min, wind_max = speed_range

        # Log the range of the U component for diagnostics
        logger.debug(
            _WIND_COMPONENT_RANGE_MSG,
            u_variable,
            u_min,
            u_max,
        )

        # Log the range of the V component for diagnostics
        logger.debug(
            _WIND_COMPONENT_RANGE_MSG,
            v_variable,
            v_min,
            v_max,
        )

        # Log the range of the horizontal wind speed for diagnostics
        logger.debug("Wind speed range: %.2f to %.2f m/s", wind_min, wind_max)

        # Identify the units of the U and V components for diagnostics
        u_units = u_data.attrs.get("units", WIND_SPEED_UNITS)
        v_units = v_data.attrs.get("units", WIND_SPEED_UNITS)

        # Log the units of the U component for reference
        logger.debug("Units: %s", u_units)

        # Log a warning if the U and V components have different units
        if u_units != v_units:
            logger.warning(
                "U and V components have different units: %s vs %s",
                u_units,
                v_units,
            )

    def get_2d_wind_components(
        self: "WindDiagnostics",
        dataset: xr.Dataset,
        u_variable: str,
        v_variable: str,
        time_index: int = 0,
        data_type: str = "xarray",
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This function extracts the U and V wind components from a 2D MPAS dataset at a specified time index. It validates the presence of the specified variables in the dataset and checks that the time index is within bounds. The function retrieves the variable slices for U and V components using either direct indexing for 'uxarray' datasets or the `isel` method for 'xarray' datasets. It ensures that any lazy computations are executed by calling `compute()` if necessary. The function includes diagnostic print statements that provide information about the ranges of the extracted components and their units if verbose mode is enabled. The resulting U and V components are returned as xarray DataArrays with CF-compliant attributes. If any issues arise during extraction (e.g., missing variables, out-of-range time index), it raises appropriate exceptions with informative messages to help diagnose the problem.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the wind variables.
            u_variable (str): Name of the U-component variable to extract.
            v_variable (str): Name of the V-component variable to extract.
            time_index (int): Time index to select for extraction (default: 0).
            data_type (str): Dataset access style, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: A tuple containing the extracted U and V components as xarray DataArrays with CF-compliant attributes.
        """
        from mpasdiag.processing.utils_datetime import MPASDateTimeUtils

        if dataset is None:
            raise RuntimeError("No dataset provided for wind component extraction.")

        # Log the variables being extracted if verbose mode is enabled
        if self.verbose:
            logger.debug(
                "Extracting wind components %s, %s at time index %d",
                u_variable,
                v_variable,
                time_index,
            )

        # Check for the presence of the specified U and V variables in the dataset
        self._require_wind_variables(dataset, u_variable, v_variable)

        # Validate time parameters and determine the time dimension name
        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            dataset, time_index, self.verbose
        )

        try:
            if data_type == "uxarray":
                # Uxarray does not support .isel, so we use direct indexing for time selection
                u_data = dataset[u_variable][validated_time_index]
                v_data = dataset[v_variable][validated_time_index]
            else:
                # For xarray datasets, we can use .isel to select the time index
                u_data = dataset[u_variable].isel({time_dim: validated_time_index})
                v_data = dataset[v_variable].isel({time_dim: validated_time_index})

            # Compute the U component if it is lazy (e.g., Dask array)
            if hasattr(u_data, "compute"):
                u_data = cast(Any, u_data).compute()

            # Compute the V component if it is lazy (e.g., Dask array)
            if hasattr(v_data, "compute"):
                v_data = cast(Any, v_data).compute()

            # Calculate the range of the U and V components for diagnostics
            u_min, u_max = float(u_data.min()), float(u_data.max())
            v_min, v_max = float(v_data.min()), float(v_data.max())

            # Calculate the horizontal wind speed from the U and V components
            wind_speed = np.sqrt(u_data**2 + v_data**2)

            # Calculate the range of the horizontal wind speed for diagnostics
            wind_min, wind_max = float(wind_speed.min()), float(wind_speed.max())

            if self.verbose:
                self._log_wind_component_diagnostics(
                    u_data,
                    v_data,
                    u_variable,
                    v_variable,
                    (u_min, u_max),
                    (v_min, v_max),
                    (wind_min, wind_max),
                )

            # Return the extracted U and V components as a tuple of xarray DataArrays
            return u_data, v_data

        except KeyError as e:
            raise ValueError(f"Error accessing wind variables: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting 2D wind components: {e}")
