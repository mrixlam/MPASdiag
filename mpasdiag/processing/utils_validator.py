#!/usr/bin/env python3

"""
MPAS Data Validation Utilities

This module provides comprehensive data validation functionality for MPAS unstructured mesh data through quality assurance checks performed before intensive analysis or visualization operations. It includes validation methods for geographic coordinate bounds verification, finite value completeness testing, statistical outlier detection, and threshold-based data range validation. The DataValidator class offers stateless static methods optimized for efficiency on large MPAS datasets while catching data integrity issues such as invalid coordinates, excessive missing values, uniform data artifacts, or unexpected value ranges. These lightweight validators serve as quality gates ensuring data correctness before downstream processing, providing detailed diagnostic information when problems are detected. The module integrates seamlessly with MPASdiag processing workflows to maintain robust data quality standards throughout the diagnostic pipeline.

Classes:
    DataValidator: Static utility class providing comprehensive validation methods for MPAS coordinate arrays and numerical data quality assurance.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional


class DataValidator:
    """
    Data validation utilities for MPAS coordinate arrays and numerical data quality assurance with comprehensive sanity checking capabilities. This class provides stateless static methods designed to perform quick pre-processing validation before intensive analysis or visualization operations on MPAS unstructured mesh data. Validation checks include geographic coordinate bounds verification, finite value completeness testing, statistical outlier detection, and threshold-based data range validation. These lightweight validators serve as quality gates catching data integrity issues such as invalid coordinates, excessive missing values, uniform data artifacts, or unexpected value ranges. The validators are optimized for efficiency on large arrays typical of MPAS datasets while providing detailed diagnostic information when problems are detected.
    """
    
    @staticmethod
    def validate_coordinates(lon: np.ndarray, lat: np.ndarray) -> bool:
        """
        Validate longitude and latitude coordinate arrays for geographic correctness and completeness. This method performs comprehensive checks ensuring coordinate arrays have matching lengths, contain only finite values (no NaN or Inf), and fall within valid geographic bounds (-180 to 180 for longitude, -90 to 90 for latitude). It serves as a pre-processing quality gate to catch data integrity issues before visualization or analysis operations. The validation is efficient and suitable for large unstructured mesh coordinates typical of MPAS datasets. Returns True only when all validation criteria pass, False otherwise.

        Parameters:
            lon (np.ndarray): One-dimensional array of longitude values in degrees East.
            lat (np.ndarray): One-dimensional array of latitude values in degrees North.

        Returns:
            bool: True if coordinate arrays pass all validation checks, False if any check fails.
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
    def validate_data_array(data: np.ndarray, 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation and statistical summary of numerical data arrays with optional threshold checking. This method analyzes data quality by computing summary statistics (min, max, mean, std, median) on finite values while tracking missing data through NaN/Inf counts. It validates data against optional minimum and maximum thresholds, flagging any exceedances as issues in the returned dictionary. The function treats all non-finite values (NaN, Inf, -Inf) as missing data and reports their percentage. This validation is essential for data quality assurance before visualization or analysis, helping identify outliers, data gaps, or processing artifacts in MPAS model output.

        Parameters:
            data (np.ndarray): Numerical data array to validate and summarize, can be multi-dimensional.
            min_val (Optional[float]): Optional minimum threshold - reports issue if observed minimum is below this value (default: None).
            max_val (Optional[float]): Optional maximum threshold - reports issue if observed maximum is above this value (default: None).

        Returns:
            dict: Dictionary with 'valid' (bool) overall status, 'issues' (list of str) detected problems, and 'stats' (dict) containing min, max, mean, std, median, total_points, finite_points, and finite_percentage.
        """
        results = {
            "valid": True,
            "issues": [],
            "stats": {}
        }
        
        finite_mask = np.isfinite(data)
        finite_count = np.sum(finite_mask)
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
            results["issues"].append(f"Minimum value {results['stats']['min']:.2f} below expected {min_val}")
        
        if max_val is not None and results["stats"]["max"] > max_val:
            results["issues"].append(f"Maximum value {results['stats']['max']:.2f} above expected {max_val}")
        
        if results["stats"]["min"] == results["stats"]["max"]:
            results["issues"].append("All values are identical")
        
        if results["stats"]["std"] == 0:
            results["issues"].append("Zero standard deviation")
        
        if results["issues"]:
            results["valid"] = False
        
        return results