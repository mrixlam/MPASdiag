#!/usr/bin/env python3

"""
MPAS Geographic and Spatial Utilities

This module provides comprehensive geographic and spatial utilities for MPAS model data processing including coordinate extraction from unstructured meshes, geographic extent validation, spatial filtering operations, and coordinate transformations. It implements the MPASGeographicUtils class with static methods for extracting longitude and latitude arrays from MPAS datasets with automatic unit detection and conversion (radians to degrees), validating and normalizing geographic extents for plotting and subsetting, computing spatial bounds and bounding boxes for data regions, and performing point-in-region filtering for spatial subsetting. The utilities handle MPAS-specific coordinate naming conventions (lonCell, latCell) and mesh geometries (Voronoi cells, dual mesh edges), support both cell-centered and vertex-based coordinate systems, normalize longitude values to standard [-180, 180] range for dateline handling, and provide robust error handling for missing or malformed coordinate data. Core capabilities include automatic coordinate flattening for multi-dimensional arrays, extent validation with range checking and dateline crossing detection, great-circle distance calculations for spatial proximity operations, and integration with cartographic projection systems suitable for visualizing MPAS unstructured mesh data on geographic maps.

Classes:
    MPASGeographicUtils: Utility class providing static methods for geographic operations on MPAS unstructured mesh datasets.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional


class MPASGeographicUtils:
    """
    Geographic utilities class for MPAS spatial operations.
    
    This class provides functionality for handling geographic coordinates,
    spatial extents, and coordinate transformations for MPAS unstructured mesh data.
    """
    
    @staticmethod
    def extract_spatial_coordinates(dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract longitude and latitude coordinate arrays from MPAS unstructured mesh datasets with automatic unit conversion and normalization. This method searches for coordinate variables using common MPAS naming conventions ('lonCell', 'latCell', 'longitude', 'latitude', 'lon', 'lat'), handles both radian and degree units by detecting value ranges and converting radians to degrees when necessary, and flattens multi-dimensional coordinate arrays to 1D for consistency. The method normalizes longitude values to the standard [-180, 180] range to ensure proper handling of dateline-crossing regions and global domains. This coordinate extraction is fundamental for all spatial operations including plotting, subsetting, and geographic analysis of MPAS model output on irregular Voronoi meshes.

        Parameters:
            dataset (xr.Dataset): MPAS xarray Dataset containing coordinate information in variables or coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two-element tuple containing (longitude_array, latitude_array) as flattened 1D numpy arrays in degrees with longitude normalized to [-180, 180] range.
            
        Raises:
            ValueError: If dataset is None or if no recognizable spatial coordinates are found in the dataset.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None.")
        
        lon_names = ['lonCell', 'longitude', 'lon']
        lat_names = ['latCell', 'latitude', 'lat']
        
        lon_coords = lat_coords = None
        
        for name in lon_names:
            if name in dataset.coords or name in dataset.data_vars:
                lon_coords = dataset[name].values
                break
                
        for name in lat_names:
            if name in dataset.coords or name in dataset.data_vars:
                lat_coords = dataset[name].values
                break
                
        if lon_coords is None or lat_coords is None:
            available_vars = list(dataset.coords.keys()) + list(dataset.data_vars.keys())
            raise ValueError(f"Could not find spatial coordinates. Available variables: {available_vars}")
        
        if np.nanmax(np.abs(lat_coords)) <= np.pi:
            lat_coords = lat_coords * 180.0 / np.pi
            lon_coords = lon_coords * 180.0 / np.pi
        
        lon_coords = lon_coords.ravel()
        lat_coords = lat_coords.ravel()
        
        lon_coords = MPASGeographicUtils.normalize_longitude(lon_coords)
        
        return lon_coords, lat_coords

    @staticmethod
    def filter_by_spatial_extent(data: xr.DataArray, dataset: xr.Dataset,
                                lon_min: float, lon_max: float, 
                                lat_min: float, lat_max: float) -> Tuple[xr.DataArray, np.ndarray]:
        """
        Apply geographic bounding box filtering to MPAS data arrays by masking cells outside specified longitude and latitude ranges. This method extracts spatial coordinates from the dataset, creates a boolean mask identifying cells within the specified rectangular geographic extent, and applies the mask to the data array using xarray's `where()` method to set out-of-bounds values to NaN. The filtering preserves the original data structure and dimensions while enabling regional analysis and visualization of MPAS model output. The method returns both the filtered data array and the boolean mask for potential reuse in subsequent operations or for tracking which cells were included in the spatial subset.

        Parameters:
            data (xr.DataArray): MPAS data array to filter, typically with 'nCells' spatial dimension.
            dataset (xr.Dataset): MPAS dataset containing coordinate information for spatial subsetting.
            lon_min (float): Minimum longitude bound in degrees [-180 to 180] for western edge of bounding box.
            lon_max (float): Maximum longitude bound in degrees [-180 to 180] for eastern edge of bounding box.
            lat_min (float): Minimum latitude bound in degrees [-90 to 90] for southern edge of bounding box.
            lat_max (float): Maximum latitude bound in degrees [-90 to 90] for northern edge of bounding box.

        Returns:
            Tuple[xr.DataArray, np.ndarray]: Two-element tuple containing (filtered_data_array, boolean_mask) where filtered_data has NaN outside extent and mask indicates included cells.
        """
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(dataset)
        
        mask = ((lon >= lon_min) & (lon <= lon_max) & 
                (lat >= lat_min) & (lat <= lat_max))
        
        if 'nCells' in data.sizes:
            filtered_data = data.where(mask)
        else:
            filtered_data = data
        
        return filtered_data, mask

    @staticmethod
    def normalize_longitude(lon: np.ndarray) -> np.ndarray:
        """
        Normalize longitude values to the standard geographic range of [-180, 180] degrees using modular arithmetic for consistent dateline handling. This method takes longitude arrays that may span [0, 360], [-180, 180], or any other range due to coordinate system differences or accumulated offsets, and converts all values to the standardized [-180, 180] convention used throughout MPASdiag. The normalization uses the formula `((lon + 180) % 360) - 180` to wrap longitude values around the international dateline while preserving geographic accuracy. This standardization is essential for proper visualization, spatial filtering, and coordinate comparisons across datasets that may use different longitude conventions.

        Parameters:
            lon (np.ndarray): Longitude array in degrees with values in any range (commonly [0, 360] or [-180, 180]).

        Returns:
            np.ndarray: Normalized longitude array with all values mapped to the range [-180, 180] degrees.
        """
        lon = np.asarray(lon)
        lon = ((lon + 180) % 360) - 180
        return lon

    @staticmethod
    def validate_geographic_extent(extent: Tuple[float, float, float, float]) -> bool:
        """
        Validate that geographic extent coordinates are within valid geographic ranges and properly ordered for bounding box operations. This method checks that longitude values fall within [-180, 180] degrees, latitude values fall within [-90, 90] degrees, and that maximum bounds are greater than minimum bounds in both dimensions to define a valid rectangular region. The validation ensures that user-provided or calculated geographic extents are physically meaningful before use in spatial filtering, plotting, or subsetting operations. This function is particularly useful for input validation in command-line interfaces and for detecting configuration errors that could cause incorrect spatial queries or visualization artifacts.

        Parameters:
            extent (Tuple[float, float, float, float]): Geographic extent tuple as (lon_min, lon_max, lat_min, lat_max) in degrees.

        Returns:
            bool: True if extent coordinates are within valid ranges and properly ordered (lon_max > lon_min, lat_max > lat_min), False otherwise.
        """
        lon_min, lon_max, lat_min, lat_max = extent
        return (
            -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0
            and -90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0
            and lon_max > lon_min and lat_max > lat_min
        )

    @staticmethod
    def get_extent_from_coordinates(lon: np.ndarray, lat: np.ndarray, 
                                   buffer: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Calculate the minimum bounding box geographic extent from coordinate arrays with optional buffer expansion for margin control in visualizations. This method determines the rectangular geographic extent that encompasses all valid (non-NaN) coordinate points by computing minimum and maximum longitude and latitude values, then optionally expands the extent by a specified buffer in degrees to add whitespace around the data in plots. The method filters out invalid coordinates using finite checks, applies the buffer symmetrically on all sides, and clamps the final extent to valid geographic ranges [-180, 180] for longitude and [-90, 90] for latitude. This automatic extent calculation is essential for creating properly sized map projections and ensuring all data points are visible in regional visualization workflows.

        Parameters:
            lon (np.ndarray): Longitude coordinate array in degrees, may contain NaN values.
            lat (np.ndarray): Latitude coordinate array in degrees, may contain NaN values.
            buffer (float): Buffer distance in degrees to add around extent on all sides for plot margins (default: 0.0).

        Returns:
            Tuple[float, float, float, float]: Geographic extent tuple as (lon_min, lon_max, lat_min, lat_max) in degrees, clamped to valid ranges and expanded by buffer.
        """
        valid_mask = np.isfinite(lon) & np.isfinite(lat)

        if not np.any(valid_mask):
            raise ValueError("No valid coordinates found.")
            
        valid_lon = lon[valid_mask]
        valid_lat = lat[valid_mask]
        
        lon_min = float(np.min(valid_lon)) - buffer
        lon_max = float(np.max(valid_lon)) + buffer
        lat_min = float(np.min(valid_lat)) - buffer
        lat_max = float(np.max(valid_lat)) + buffer
        
        lon_min = max(lon_min, -180.0)
        lon_max = min(lon_max, 180.0)
        lat_min = max(lat_min, -90.0)
        lat_max = min(lat_max, 90.0)
        
        return lon_min, lon_max, lat_min, lat_max

    @staticmethod
    def calculate_spatial_resolution(lon: np.ndarray, lat: np.ndarray, 
                                   sample_size: int = 1000) -> float:
        """
        Estimate the average spatial resolution of the MPAS unstructured mesh by computing median nearest-neighbor distances between sample coordinate points. This method randomly samples a subset of mesh cells (or uses all points if fewer than sample_size), calculates Euclidean distances between consecutive points in the sample, and returns the median distance as a representative resolution value in degrees. The median statistic provides robustness against outliers that may occur in variable-resolution MPAS meshes where refinement regions have dramatically different cell sizes. This resolution estimate is useful for selecting appropriate contouring levels, determining optimal plotting parameters, and understanding the effective scale of the model output data for quality control and analysis purposes.

        Parameters:
            lon (np.ndarray): Longitude coordinate array for MPAS mesh cells in degrees.
            lat (np.ndarray): Latitude coordinate array for MPAS mesh cells in degrees.
            sample_size (int): Number of coordinate points to sample for resolution estimation, smaller values faster but less accurate (default: 1000).

        Returns:
            float: Estimated spatial resolution in degrees representing median distance between neighboring mesh cells, returns 0.0 if resolution cannot be calculated.
        """
        if len(lon) < 2 or len(lat) < 2:
            return 0.0
            
        n_points = min(sample_size, len(lon))

        if n_points < len(lon):
            indices = np.random.choice(len(lon), n_points, replace=False)
            sample_lon = lon[indices]
            sample_lat = lat[indices] 
        else:
            sample_lon = lon
            sample_lat = lat
        
        distances = []

        for i in range(len(sample_lon) - 1):
            dlat = sample_lat[i+1] - sample_lat[i]
            dlon = sample_lon[i+1] - sample_lon[i]

            dist = np.sqrt(dlat**2 + dlon**2)

            if dist > 0:
                distances.append(dist)
        
        if not distances:
            return 0.0
            
        return float(np.median(distances))

    @staticmethod
    def is_global_extent(extent: Tuple[float, float, float, float], 
                        tolerance: float = 1.0) -> bool:
        """
        Determine if a geographic extent approximately covers the entire globe by comparing longitude and latitude spans to full Earth coverage with configurable tolerance. This method calculates the longitude span (lon_max - lon_min) and latitude span (lat_max - lat_min) from the provided extent, then checks if the spans are within tolerance of 360° and 180° respectively, which would indicate global coverage. The tolerance parameter allows flexibility for near-global domains that may have small gaps or overlap, with default 1° tolerance accommodating minor coordinate rounding or mesh boundary effects. This classification is useful for automatically selecting global vs regional map projections, determining appropriate plotting strategies, and optimizing data processing workflows that differ for global and regional MPAS simulations.

        Parameters:
            extent (Tuple[float, float, float, float]): Geographic extent tuple as (lon_min, lon_max, lat_min, lat_max) in degrees.
            tolerance (float): Tolerance in degrees for "global" determination, allowing spans slightly less than full Earth coverage (default: 1.0).

        Returns:
            bool: True if extent spans approximately the full globe (longitude span ≥ 360° - tolerance and latitude span ≥ 180° - tolerance), False otherwise.
        """
        lon_min, lon_max, lat_min, lat_max = extent
        
        lon_span = lon_max - lon_min
        lat_span = lat_max - lat_min
        
        is_global_lon = lon_span >= (360.0 - tolerance)
        is_global_lat = lat_span >= (180.0 - tolerance)
        
        return is_global_lon and is_global_lat