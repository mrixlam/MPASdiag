#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Processing Module: Geographic and Spatial Utilities

This module defines the MPASGeographicUtils class, which provides a collection of static methods for handling geographic and spatial operations on MPAS model output data. The MPASGeographicUtils class includes functions for extracting longitude and latitude coordinates from xarray Datasets, filtering data based on geographic extents, normalizing longitude values, validating geographic extents, calculating spatial resolution estimates, and determining if an extent covers the global domain. These utilities are designed to facilitate common spatial operations required in MPAS diagnostic processing workflows, enabling users to easily manage and analyze spatial data from MPAS simulations. By centralizing these geographic functions in a dedicated utility class, MPASdiag promotes code reuse, consistency, and maintainability across different diagnostic scripts and analyses that involve spatial data handling.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Any, Mapping, NamedTuple, Optional, Tuple


class GeographicBounds(NamedTuple):
    """Geographic map extent expressed as minimum/maximum longitude and latitude in degrees."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


class MPASGeographicUtils:
    """Utility class for geographic and spatial operations on MPAS model output data, providing static methods for coordinate extraction, spatial filtering, and resolution estimation."""

    @staticmethod
    def coordinates_to_degrees(
        lon: Any,
        lat: Any,
        lon_attrs: Optional[Mapping[str, Any]] = None,
        lat_attrs: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method converts a pair of longitude and latitude arrays to degrees using a single, centralized radians-versus-degrees decision shared by every coordinate-loading path in MPASdiag. When a 'units' attribute is available on either coordinate it is authoritative: values labelled 'radian', 'radians', or 'rad' are converted with a factor of 180/pi, while values labelled with any 'degree' variant are returned unchanged. Without an explicit unit the method falls back to a conservative magnitude heuristic that treats the coordinates as radians only when both the maximum absolute latitude is at most pi/2 and the maximum absolute longitude is at most 2*pi (each with a small tolerance). Requiring both conditions avoids mis-converting degree-valued meshes that happen to lie close to the equator and prime meridian, which the former single-axis test of 'max|lat| <= pi' routinely did. The inputs are never modified in place and are returned as float arrays.

        Parameters:
            lon (Any): Longitude values, either as an array-like object or a numpy array, in radians or degrees.
            lat (Any): Latitude values, either as an array-like object or a numpy array, in radians or degrees.
            lon_attrs (Optional[Mapping[str, Any]]): Attributes of the longitude variable that may contain a 'units' entry (default: None).
            lat_attrs (Optional[Mapping[str, Any]]): Attributes of the latitude variable that may contain a 'units' entry (default: None).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two-element tuple of (lon_degrees, lat_degrees) as numpy arrays in degrees.
        """
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)

        units = ""
        for attrs in (lat_attrs, lon_attrs):
            if attrs is not None and attrs.get("units") is not None:
                units = str(attrs["units"]).strip().lower()
                break

        if units in ("radian", "radians", "rad"):
            is_radians = True
        elif units.startswith("degree") or units in (
            "deg",
            "degs",
            "degrees_east",
            "degrees_north",
        ):
            is_radians = False
        elif (
            lat_arr.size == 0
            or lon_arr.size == 0
            or np.isnan(lat_arr).all()
            or np.isnan(lon_arr).all()
        ):
            is_radians = False
        else:
            tol = 1e-6
            is_radians = bool(
                np.nanmax(np.abs(lat_arr)) <= np.pi / 2 + tol
                and np.nanmax(np.abs(lon_arr)) <= 2 * np.pi + tol
            )

        if is_radians:
            return np.degrees(lon_arr), np.degrees(lat_arr)
        return lon_arr, lat_arr

    @staticmethod
    def extract_spatial_coordinates(
        dataset: xr.Dataset, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method extracts longitude and latitude coordinates from an MPAS xarray Dataset, handling various possible variable names and coordinate structures commonly found in MPAS model output. It searches for standard coordinate variable names such as 'lonCell', 'latCell', 'longitude', 'latitude', 'lon', and 'lat' in both the dataset's coordinates and data variables. If the coordinates are found in radians, it converts them to degrees. The method then flattens the longitude and latitude arrays into 1D numpy arrays for easier handling in subsequent spatial operations. If the normalize parameter is set to True, it also normalizes longitude values to the range [-180, 180] degrees to ensure consistency across datasets that may use different longitude conventions (e.g., [0, 360]). This function is essential for preparing spatial coordinate data for filtering, plotting, and analysis tasks in MPAS diagnostic workflows.

        Parameters:
            dataset (xr.Dataset): MPAS xarray Dataset containing coordinate information for spatial operations.
            normalize (bool): If True, normalizes longitude values to the range [-180, 180] degrees (default: True).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two-element tuple containing (lon_coords, lat_coords) as 1D numpy arrays of longitude and latitude values in degrees, normalized if specified.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None.")

        lon_names = ["lonCell", "longitude", "lon"]
        lat_names = ["latCell", "latitude", "lat"]

        lon_coords = lat_coords = None
        lon_attrs: Mapping[str, Any] = {}
        lat_attrs: Mapping[str, Any] = {}

        for name in lon_names:
            if name in dataset.coords or name in dataset.data_vars:
                lon_coords = dataset[name].values
                lon_attrs = dataset[name].attrs
                break

        for name in lat_names:
            if name in dataset.coords or name in dataset.data_vars:
                lat_coords = dataset[name].values
                lat_attrs = dataset[name].attrs
                break

        if lon_coords is None or lat_coords is None:
            available_vars = list(dataset.coords.keys()) + list(
                dataset.data_vars.keys()
            )
            raise ValueError(
                f"Could not find spatial coordinates. Available variables: {available_vars}"
            )

        lon_coords, lat_coords = MPASGeographicUtils.coordinates_to_degrees(
            lon_coords, lat_coords, lon_attrs, lat_attrs
        )

        lon_coords = lon_coords.ravel()
        lat_coords = lat_coords.ravel()

        if normalize:
            lon_coords = MPASGeographicUtils.normalize_longitude(lon_coords)

        return lon_coords, lat_coords

    @staticmethod
    def filter_by_spatial_extent(
        data: xr.DataArray,
        dataset: xr.Dataset,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> Tuple[xr.DataArray, np.ndarray]:
        """
        This method filters an xarray DataArray based on a specified geographic extent defined by longitude and latitude bounds. It first extracts the longitude and latitude coordinates from the provided dataset using the extract_spatial_coordinates method. Then, it creates a boolean mask to identify which cells fall within the specified bounding box defined by lon_min, lon_max, lat_min, and lat_max. If the DataArray has a spatial dimension named 'nCells', it applies the mask to set values outside the extent to NaN, effectively filtering the data to include only the cells within the geographic bounds. The method returns a tuple containing the filtered DataArray and the boolean mask used for filtering, allowing users to easily identify which cells were included in the spatial subset. This functionality is crucial for focusing analyses and visualizations on specific regions of interest within MPAS model output data.

        Parameters:
            data (xr.DataArray): The xarray DataArray to be filtered based on geographic extent.
            dataset (xr.Dataset): The xarray Dataset containing coordinate information for extracting longitude and latitude.
            lon_min (float): Minimum longitude bound of the geographic extent in degrees.
            lon_max (float): Maximum longitude bound of the geographic extent in degrees.
            lat_min (float): Minimum latitude bound of the geographic extent in degrees.
            lat_max (float): Maximum latitude bound of the geographic extent in degrees.

        Returns:
            Tuple[xr.DataArray, np.ndarray]: A tuple containing the filtered DataArray with values outside the extent set to NaN (if 'nCells' dimension exists) and the boolean mask array indicating which cells are within the specified geographic bounds.
        """
        # Pick the coordinate pair that matches the data's spatial dimension (cells, vertices or edges)
        spatial_dim = "nCells"
        coord_dataset = dataset

        for dim, lon_name, lat_name in (
            ("nVertices", "lonVertex", "latVertex"),
            ("nEdges", "lonEdge", "latEdge"),
        ):
            if dim in data.sizes:
                if lon_name not in dataset or lat_name not in dataset:
                    raise ValueError(
                        f"Cannot filter data on '{dim}' by spatial extent: the dataset has no "
                        f"'{lon_name}'/'{lat_name}' coordinates."
                    )
                spatial_dim = dim
                coord_dataset = xr.Dataset(
                    {"lon": dataset[lon_name], "lat": dataset[lat_name]}
                )
                break

        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(coord_dataset)
        lon = MPASGeographicUtils.wrap_longitude_to_extent(lon, lon_min, lon_max)

        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

        if spatial_dim in data.sizes:
            filtered_data = data.where(xr.DataArray(mask, dims=[spatial_dim]))
        else:
            filtered_data = data

        return filtered_data, mask

    @staticmethod
    def normalize_longitude(lon: np.ndarray) -> np.ndarray:
        """
        This method normalizes longitude values to the range [-180, 180] degrees, which is a common convention for geographic data. It takes an array of longitude values that may be in any range (commonly [0, 360] or already in [-180, 180]) and applies a modulo operation to wrap the values into the desired range. The formula used is ((lon + 180) % 360) - 180, which effectively shifts the longitude values by 180 degrees, applies modulo 360 to wrap around, and then shifts back by 180 degrees to ensure all values fall within the range of -180 to 180 degrees. This normalization is important for ensuring consistency in spatial analyses and visualizations, especially when combining datasets that may use different longitude conventions.

        Parameters:
            lon (np.ndarray): Array of longitude values in degrees, which may be in any range (e.g., [0, 360] or already in [-180, 180]).

        Returns:
            np.ndarray: Array of longitude values normalized to the range [-180, 180] degrees.
        """
        lon = np.asarray(lon)
        lon = ((lon + 180) % 360) - 180
        return lon

    @staticmethod
    def wrap_longitude_to_extent(
        lon: np.ndarray, lon_min: float, lon_max: float
    ) -> np.ndarray:
        """
        This method re-expresses longitude values in the frame of a requested longitude extent so that a plain lon_min <= lon <= lon_max comparison works for extents that cross the ±180° dateline. MPAS coordinates are normally normalized to [-180, 180], so a Pacific box written as (150, 210) would otherwise drop every cell between -180° and -150°. When the extent lies entirely within [-180, 180] the input is returned unchanged, so existing filtering behavior is preserved exactly. Otherwise each longitude is shifted by a multiple of 360° into the window [lon_min, lon_min + 360). Cartopy accepts longitudes outside [-180, 180] with a PlateCarree transform, so the shifted values can be plotted directly.

        Parameters:
            lon (np.ndarray): Array of longitude values in degrees.
            lon_min (float): Western bound of the requested extent in degrees (may be below -180).
            lon_max (float): Eastern bound of the requested extent in degrees (may exceed 180).

        Returns:
            np.ndarray: Longitudes in the frame of the requested extent; the original array when the extent does not cross the dateline.
        """
        lon = np.asarray(lon)
        if -180.0 <= lon_min and lon_max <= 180.0:
            return lon
        return ((lon - lon_min) % 360.0) + lon_min

    @staticmethod
    def validate_geographic_extent(extent: Tuple[float, float, float, float]) -> bool:
        """
        This method validates a geographic extent defined by a tuple of (lon_min, lon_max, lat_min, lat_max) to ensure that the longitude and latitude bounds are within valid ranges and properly ordered. It checks that longitude values are between -180 and 180 degrees, latitude values are between -90 and 90 degrees, and that the maximum longitude is greater than the minimum longitude, and the maximum latitude is greater than the minimum latitude. This validation is crucial for ensuring that geographic extents used in filtering, plotting, or spatial analyses are meaningful and do not contain invalid or nonsensical values that could lead to errors or incorrect results in MPAS diagnostic processing workflows. If the extent passes all checks, the method returns True; otherwise, it returns False to indicate an invalid geographic extent.

        Parameters:
            extent (Tuple[float, float, float, float]): Geographic extent tuple as (lon_min, lon_max, lat_min, lat_max) in degrees to be validated.

        Returns:
            bool: True if the geographic extent is valid (longitude between -180 and 180, latitude between -90 and 90, and properly ordered), False otherwise.
        """
        lon_min, lon_max, lat_min, lat_max = extent
        return (
            -180.0 <= lon_min <= 180.0
            and -180.0 <= lon_max <= 180.0
            and -90.0 <= lat_min <= 90.0
            and -90.0 <= lat_max <= 90.0
            and lon_max > lon_min
            and lat_max > lat_min
        )

    @staticmethod
    def get_extent_from_coordinates(
        lon: np.ndarray, lat: np.ndarray, buffer: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """
        This method calculates the geographic extent (bounding box) from given longitude and latitude coordinate arrays, optionally adding a buffer distance to expand the extent for plot margins or spatial analyses. It first creates a boolean mask to identify valid (finite) longitude and latitude values, then computes the minimum and maximum longitude and latitude from the valid coordinates. The method applies the specified buffer to expand the extent on all sides, ensuring that the resulting bounds are clamped to valid ranges for longitude (-180 to 180 degrees) and latitude (-90 to 90 degrees). This function is useful for automatically determining appropriate geographic extents for plotting or spatial filtering based on the actual distribution of coordinate data in MPAS datasets, while allowing for additional margin space as needed. If no valid coordinates are found, it raises a ValueError indicating that an extent cannot be calculated.

        Parameters:
            lon (np.ndarray): Array of longitude coordinates in degrees.
            lat (np.ndarray): Array of latitude coordinates in degrees.
            buffer (float): Optional buffer distance in degrees to expand the extent on all sides (default: 0.0).

        Returns:
            Tuple[float, float, float, float]: Geographic extent as a tuple (lon_min, lon_max, lat_min, lat_max) in degrees, expanded by the specified buffer and clamped to valid ranges.
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

        lat_min = max(lat_min, -90.0)
        lat_max = min(lat_max, 90.0)

        return lon_min, lon_max, lat_min, lat_max

    @staticmethod
    def calculate_spatial_resolution(
        lon: np.ndarray, lat: np.ndarray, sample_size: int = 1000
    ) -> float:
        """
        This method estimates the spatial resolution of an MPAS mesh by calculating the median distance between neighboring longitude and latitude coordinate points. It takes arrays of longitude and latitude coordinates, optionally samples a subset of points for faster computation, and computes the great-circle distance from each sampled point to its nearest neighbour among all points (via a KDTree on the unit sphere). The method then returns the median nearest-neighbour distance as an estimate of the spatial resolution in degrees. If there are fewer than 2 valid points, it returns 0.0 to indicate that resolution cannot be calculated. This function provides a simple way to estimate the effective spatial resolution of MPAS model output data based on the distribution of coordinate points, which can be useful for understanding the level of detail in spatial analyses and visualizations.

        Parameters:
            lon (np.ndarray): Array of longitude coordinates in degrees.
            lat (np.ndarray): Array of latitude coordinates in degrees.
            sample_size (int): Optional number of points to sample for distance calculation to improve performance (default: 1000). If the total number of points is less than sample_size, all points will be used.

        Returns:
            float: Estimated spatial resolution in degrees, calculated as the median distance between neighboring coordinate points. Returns 0.0 if there are fewer than 2 valid points.
        """
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        valid = np.isfinite(lon) & np.isfinite(lat)
        lon, lat = lon[valid], lat[valid]

        if len(lon) < 2:
            return 0.0

        n_points = min(sample_size, len(lon))

        if n_points < len(lon):
            rng = np.random.default_rng(0)
            indices = rng.choice(len(lon), n_points, replace=False)
        else:
            indices = np.arange(len(lon))

        # Nearest-neighbour distance on the unit sphere (chord length), converted to degrees of great-circle arc
        from scipy.spatial import cKDTree

        lon_rad, lat_rad = np.radians(lon), np.radians(lat)
        xyz = np.column_stack(
            [
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ]
        )
        distances, _ = cKDTree(xyz).query(xyz[indices], k=2)
        chord = distances[:, 1]
        chord = chord[chord > 0]

        if chord.size == 0:
            return 0.0

        arc = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
        return float(np.degrees(np.median(arc)))

    @staticmethod
    def is_global_extent(
        extent: Tuple[float, float, float, float], tolerance: float = 1.0
    ) -> bool:
        """
        This method determines whether a given geographic extent approximately covers the global domain by checking if the longitude span is close to 360 degrees and the latitude span is close to 180 degrees, within a specified tolerance. It calculates the longitude and latitude spans from the provided extent tuple (lon_min, lon_max, lat_min, lat_max) and compares them against the thresholds of 360 degrees for longitude and 180 degrees for latitude, allowing for a small tolerance to account for extents that may not perfectly match the full globe but still effectively cover it. If both the longitude and latitude spans meet or exceed their respective thresholds minus the tolerance, the method returns True, indicating that the extent can be considered global; otherwise, it returns False. This function is useful for identifying when a dataset or plot should be treated as global in scope based on its geographic coverage.

        Parameters:
            extent (Tuple[float, float, float, float]): Geographic extent as a tuple (lon_min, lon_max, lat_min, lat_max) in degrees to be evaluated for global coverage.
            tolerance (float): Tolerance in degrees for determining if the longitude and latitude spans are close enough to 360 and 180 degrees, respectively (default: 1.0 degree).

        Returns:
            bool: True if the extent is considered global (longitude span >= 360 - tolerance and latitude span >= 180 - tolerance), False otherwise.
        """
        lon_min, lon_max, lat_min, lat_max = extent

        lon_span = lon_max - lon_min
        lat_span = lat_max - lat_min

        is_global_lon = lon_span >= (360.0 - tolerance)
        is_global_lat = lat_span >= (180.0 - tolerance)

        return is_global_lon and is_global_lat
