#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Processing Module: Data Validation Utilities

This module provides comprehensive data validation utilities for MPAS coordinate arrays and numerical data quality assurance with sanity checking capabilities. It includes methods to validate longitude and latitude coordinate arrays for MPAS unstructured mesh data, ensuring they have matching lengths, finite values, and valid geographic bounds. Additionally, it offers a method to validate numerical data arrays by checking for finite values, calculating basic statistics (min, max, mean, std, median), and identifying potential issues such as excessive missing values, uniform data artifacts, or out-of-range values based on optional thresholds. The validation results are returned in a structured format that includes overall validity status, detected issues, and computed statistics. This module is essential for ensuring the integrity of spatial and numerical data before any analysis or visualization steps in the MPAS diagnostic workflow.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import numpy as np
from typing import Dict, Any, Optional

from .constants import (
    MAX_SOURCE_CELLS,
    MAX_TARGET_POINTS,
    MAX_WEIGHTS_NNZ,
    MAX_NUM_POINTS,
)


class DataValidator:
    """Data validation utilities for MPAS coordinate arrays and numerical data quality assurance with comprehensive sanity checking capabilities."""

    @staticmethod
    def _resolve_size_limit(env_var: str, default: int) -> int:
        """
        This method resolves the effective size limit for a given parameter by checking if an environment variable is set to override the compiled-in default. If the environment variable is set, it attempts to convert its value to an integer and checks if it is a positive integer. If the conversion fails or the value is not positive, a ValueError is raised. If the environment variable is not set, the method returns the default limit.

        Parameters:
            env_var (str): Name of the MPASDIAG_MAX_* environment variable that may override the compiled-in default.
            default (int): Default limit from constants.py to use when the environment variable is unset.

        Returns:
            int: The effective positive integer limit.
        """
        raw = os.environ.get(env_var)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{env_var} must be a positive integer (got {raw!r})")
        if value <= 0:
            raise ValueError(f"{env_var} must be a positive integer (got {raw!r})")
        return value

    @staticmethod
    def enforce_size_limits(
        *,
        n_src: Optional[int] = None,
        n_tgt: Optional[int] = None,
        nnz: Optional[int] = None,
        num_points: Optional[int] = None,
        context: str = "",
    ) -> None:
        """
        This method enforces safety limits on various parameters related to MPAS data processing, such as the number of source mesh cells, target grid points, non-zero remap weight entries, and cross-section interpolation points. It checks if the provided values are negative or exceed their respective safety limits, which can be overridden by environment variables. If any value is invalid, a ValueError is raised with a descriptive message indicating the issue and suggesting how to raise the limit if the input is trusted.

        Parameters:
            n_src (Optional[int]): Number of source mesh cells / points, checked against MAX_SOURCE_CELLS (env MPASDIAG_MAX_SOURCE_CELLS).
            n_tgt (Optional[int]): Number of target grid points, checked against MAX_TARGET_POINTS (env MPASDIAG_MAX_TARGET_POINTS).
            nnz (Optional[int]): Number of non-zero remap weight entries, checked against MAX_WEIGHTS_NNZ (env MPASDIAG_MAX_WEIGHTS_NNZ).
            num_points (Optional[int]): Number of cross-section interpolation points, checked against MAX_NUM_POINTS (env MPASDIAG_MAX_NUM_POINTS).
            context (str): Optional short description of the operation, included in error messages for clarity.

        Returns:
            None
        """
        checks = (
            (n_src, "MPASDIAG_MAX_SOURCE_CELLS", MAX_SOURCE_CELLS, "source grid cells"),
            (
                n_tgt,
                "MPASDIAG_MAX_TARGET_POINTS",
                MAX_TARGET_POINTS,
                "target grid points",
            ),
            (nnz, "MPASDIAG_MAX_WEIGHTS_NNZ", MAX_WEIGHTS_NNZ, "remap weight entries"),
            (
                num_points,
                "MPASDIAG_MAX_NUM_POINTS",
                MAX_NUM_POINTS,
                "cross-section points",
            ),
        )
        where = f" while {context}" if context else ""
        for value, env_var, default, label in checks:
            if value is None:
                continue
            if value < 0:
                raise ValueError(f"Invalid negative {label}: {value}")
            limit = DataValidator._resolve_size_limit(env_var, default)
            if value > limit:
                raise ValueError(
                    f"{label} ({int(value):,}) exceeds the safety limit "
                    f"({limit:,}){where}. If this input is trusted, raise the "
                    f"limit via the {env_var} environment variable."
                )

    @staticmethod
    def validate_coordinates(lon: np.ndarray, lat: np.ndarray) -> bool:
        """
        This method validates longitude and latitude coordinate arrays for MPAS unstructured mesh data. It checks that the longitude and latitude arrays have matching lengths, contain only finite values, and that the longitude values are within the range of -180 to 180 degrees while latitude values are within -90 to 90 degrees. The method returns True if all validation checks pass, indicating that the coordinate arrays are valid for use in MPAS diagnostics, and False if any check fails, which would suggest issues with the coordinate data that need to be addressed before further processing or visualization steps.

        Parameters:
            lon (np.ndarray): 1D array of longitude values for MPAS grid points, expected to be in degrees and within the range of -180 to 180.
            lat (np.ndarray): 1D array of latitude values for MPAS grid points, expected to be in degrees and within the range of -90 to 90.

        Returns:
            bool: True if the coordinate arrays are valid (matching lengths, finite values, and within geographic bounds), False otherwise.
        """
        if len(lon) != len(lat):
            return False

        if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
            return False

        if not (-180 <= np.min(lon) and np.max(lon) <= 180):
            return False

        if not (-90 <= np.min(lat) and np.max(lat) <= 90):
            return False

        return True

    @staticmethod
    def validate_data_array(
        data: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        This method validates a numerical data array by checking for finite values, calculating basic statistics (min, max, mean, std, median), and identifying potential issues such as excessive missing values, uniform data artifacts, or out-of-range values based on optional minimum and maximum thresholds. The method returns a dictionary containing the overall validity status of the data array, a list of detected issues if any are found, and a nested dictionary of computed statistics including the total number of points, count and percentage of finite values, and the calculated min, max, mean, std, and median of the finite data. This comprehensive validation approach helps ensure that the numerical data is suitable for analysis or visualization in MPAS diagnostics and can highlight potential problems that may need to be addressed before further processing.

        Parameters:
            data (np.ndarray): Numerical data array to be validated, which may contain finite values, NaNs, or infinities.
            min_val (Optional[float]): Optional minimum threshold for valid data values; if provided, values below this threshold will be flagged as issues (default: None).
            max_val (Optional[float]): Optional maximum threshold for valid data values; if provided, values above this threshold will be flagged as issues (default: None).

        Returns:
            Dict[str, Any]: A dictionary containing the validation results with the following structure:
                {
                    "valid": bool,  # Overall validity status of the data array
                    "issues": List[str],  # List of detected issues if any are found
                    "stats": {  # Nested dictionary of computed statistics
                        "total_points": int,  # Total number of points in the data array
                        "finite_points": int,  # Count of finite values in the data array
                        "finite_percentage": float,  # Percentage of finite values relative to total points
                        "min": float,  # Minimum value among finite data points
                        "max": float,  # Maximum value among finite data points
                        "mean": float,  # Mean value among finite data points
                        "std": float,  # Standard deviation among finite data points
                        "median": float  # Median value among finite data points
                    }
                }
        """
        results: Dict[str, Any] = {"valid": True, "issues": [], "stats": {}}

        finite_mask = np.isfinite(data)
        finite_count = int(np.sum(finite_mask))
        total_count = len(data.flatten())

        results["stats"]["total_points"] = total_count
        results["stats"]["finite_points"] = finite_count
        results["stats"]["finite_percentage"] = (finite_count / total_count) * 100

        if finite_count == 0:
            results["valid"] = False
            results["issues"].append("No finite values found")
            return results

        finite_data = data[finite_mask]

        results["stats"]["min"] = float(np.min(finite_data))
        results["stats"]["max"] = float(np.max(finite_data))
        results["stats"]["mean"] = float(np.mean(finite_data))
        results["stats"]["std"] = float(np.std(finite_data))
        results["stats"]["median"] = float(np.median(finite_data))

        if min_val is not None and results["stats"]["min"] < min_val:
            results["issues"].append(
                f"Minimum value {results['stats']['min']:.2f} below expected {min_val}"
            )

        if max_val is not None and results["stats"]["max"] > max_val:
            results["issues"].append(
                f"Maximum value {results['stats']['max']:.2f} above expected {max_val}"
            )

        if results["stats"]["min"] == results["stats"]["max"]:
            results["issues"].append("All values are identical")

        if results["stats"]["std"] == 0:
            results["issues"].append("Zero standard deviation")

        if results["issues"]:
            results["valid"] = False

        return results
