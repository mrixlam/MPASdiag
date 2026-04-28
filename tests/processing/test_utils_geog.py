#!/usr/bin/env python3

"""
MPASdiag Test Suite: Geographic Utilities

This module contains unit tests for the MPASGeographicUtils class, which provides utility functions for handling geographic coordinates and spatial extents in MPAS diagnostic processing. The tests cover error handling, coordinate conversion, spatial filtering, extent validation, resolution calculation, and global extent detection. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.utils_geog import MPASGeographicUtils


@pytest.fixture
def degree_dataset() -> xr.Dataset:
    """ 
    This fixture creates a simple xarray Dataset with longitude and latitude coordinates in degrees, suitable for testing geographic utilities that need to handle degree-based coordinates. The dataset contains 50 evenly spaced points covering a range of longitudes from -120 to -80 degrees and latitudes from 25 to 55 degrees, which corresponds to a typical geographic extent for testing purposes. 

    Parameters:
        None

    Returns:
        xr.Dataset: An xarray Dataset with 'lonCell' and 'latCell' DataArrays containing longitude and latitude values in degrees.
    """
    n = 50
    return xr.Dataset({
        "lonCell": xr.DataArray(np.linspace(-120.0, -80.0, n)),
        "latCell": xr.DataArray(np.linspace(25.0, 55.0, n)),
    })


@pytest.fixture
def radian_dataset() -> xr.Dataset:
    """
    This fixture creates a simple xarray Dataset with longitude and latitude coordinates in radians, suitable for testing geographic utilities that need to handle radian-based coordinates. The dataset contains 50 evenly spaced points covering a range of longitudes from -3 to 3 radians and latitudes from -1.5 to 1.5 radians, which corresponds to the same geographic extent as the degree-based dataset but in radians.

    Parameters:
        None

    Returns:
        xr.Dataset: An xarray Dataset with 'lonCell' and 'latCell' DataArrays containing longitude and latitude values in radians.
    """
    n = 50
    lat_rad = np.linspace(-1.5, 1.5, n)
    lon_rad = np.linspace(-3.0, 3.0, n)
    return xr.Dataset({
        "lonCell": xr.DataArray(lon_rad),
        "latCell": xr.DataArray(lat_rad),
    })


class TestExtractSpatialCoordinates:
    """ Tests for MPASGeographicUtils.extract_spatial_coordinates error paths and radian conversion. """

    def test_none_dataset_raises(self: 'TestExtractSpatialCoordinates') -> None:
        """
        This test verifies that the extract_spatial_coordinates method raises a ValueError when the input dataset is None. The error message should indicate that the dataset cannot be None, ensuring that the method properly handles invalid input and provides clear feedback to the user. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            MPASGeographicUtils.extract_spatial_coordinates(None)  # type: ignore[arg-type]

    def test_missing_coords_raises(self: 'TestExtractSpatialCoordinates') -> None:
        """
        This test verifies that the extract_spatial_coordinates method raises a ValueError when the input dataset does not contain the required spatial coordinate variables (e.g., 'lonCell' and 'latCell'). The error message should indicate that spatial coordinates could not be found, ensuring that the method properly handles datasets that are missing necessary information and provides clear feedback to the user. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({"temperature": xr.DataArray(np.ones(10))})
        with pytest.raises(ValueError, match="Could not find spatial coordinates"):
            MPASGeographicUtils.extract_spatial_coordinates(ds)

    def test_radian_coords_converted_to_degrees(self: 'TestExtractSpatialCoordinates', 
                                                radian_dataset: xr.Dataset) -> None:
        """
        This test verifies that the extract_spatial_coordinates method correctly converts longitude and latitude values from radians to degrees when the input dataset contains coordinates in radians. The test checks that the maximum absolute value of the latitude in radians does not exceed π, confirming that the input is indeed in radians. It then compares the output of the method to the expected degree values calculated from the radian input, ensuring that the conversion is accurate and consistent with standard mathematical conversions from radians to degrees. 

        Parameters:
            radian_dataset (xr.Dataset): An xarray Dataset with 'lonCell' and 'latCell' DataArrays containing longitude and latitude values in radians.

        Returns:
            None
        """
        lat_rad = radian_dataset["latCell"].values
        lon_rad = radian_dataset["lonCell"].values
        assert np.nanmax(np.abs(lat_rad)) <= np.pi

        result_lon, result_lat = MPASGeographicUtils.extract_spatial_coordinates(radian_dataset)

        expected_lat = lat_rad * 180.0 / np.pi
        expected_lon = lon_rad * 180.0 / np.pi
        assert np.allclose(result_lat, expected_lat)
        assert np.allclose(result_lon, expected_lon)


class TestFilterBySpatialExtent:
    """Tests for MPASGeographicUtils.filter_by_spatial_extent."""

    def test_no_ncells_returns_data_unchanged(self: 'TestFilterBySpatialExtent', 
                                              degree_dataset: xr.Dataset) -> None:
        """
        This test verifies that the filter_by_spatial_extent method returns the input data unchanged when the dataset does not contain an nCells dimension. The method should simply return the original DataArray and a mask without applying any filtering, as there are no spatial coordinates to filter against. The test checks that the returned data is equal to the input data and that the mask is a boolean array, confirming that the method handles datasets without nCells correctly and does not modify the data unnecessarily. 

        Parameters:
            degree_dataset (xr.Dataset): An xarray Dataset with 'lonCell' and 'latCell' DataArrays containing longitude and latitude values in degrees.

        Returns:
            None
        """
        data = xr.DataArray(np.ones(10), dims=["time"])
        result_data, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, degree_dataset, lon_min=-130.0, lon_max=-70.0,
            lat_min=20.0, lat_max=60.0,
        )
        assert result_data.equals(data)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_ncells_dimension_applies_mask(self: 'TestFilterBySpatialExtent', 
                                          degree_dataset: xr.Dataset) -> None:
        """
        This test verifies that the filter_by_spatial_extent method correctly applies a mask to the input data when the dataset contains an nCells dimension. The method should return a DataArray where values outside the specified geographic extent are set to NaN, and the mask should indicate which cells are within the extent. The test checks that the returned DataArray has the same nCells dimension as the input data and that not all values in the result are finite, confirming that the method is correctly filtering the data based on spatial coordinates and applying the mask as expected. 

        Parameters:
            degree_dataset (xr.Dataset): An xarray Dataset with 'lonCell' and 'latCell' DataArrays containing longitude and latitude values in degrees.

        Returns:
            None
        """
        n = len(degree_dataset["lonCell"])
        data = xr.DataArray(np.ones(n), dims=["nCells"])
        result_data, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, degree_dataset, lon_min=-110.0, lon_max=-90.0,
            lat_min=30.0, lat_max=50.0,
        )
        assert "nCells" in result_data.dims
        assert not np.all(np.isfinite(result_data.values))


class TestValidateGeographicExtent:
    """Tests for MPASGeographicUtils.validate_geographic_extent."""

    def test_valid_extent(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a valid geographic extent with longitude values between -180 and 180 degrees and latitude values between -90 and 90 degrees. The test checks that the method returns True for a typical geographic extent, confirming that it properly validates the input values and ensures they fall within acceptable ranges for geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((-120.0, -80.0, 25.0, 55.0)) is True

    def test_boundary_extent(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a geographic extent that is exactly on the boundary of valid longitude and latitude values. The test checks that the method returns True for an extent with longitude values at -180 and 180 degrees and latitude values at -90 and 90 degrees, confirming that it properly includes the boundary values as valid geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((-180.0, 180.0, -90.0, 90.0)) is True

    def test_lon_min_out_of_range(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a geographic extent with a longitude minimum out of range. The test checks that the method returns False for an extent with a longitude minimum less than -180 degrees, confirming that it properly validates the input values and ensures they fall within acceptable ranges for geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((-181.0, 0.0, -45.0, 45.0)) is False

    def test_lon_max_out_of_range(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a geographic extent with a longitude maximum out of range. The test checks that the method returns False for an extent with a longitude maximum greater than 180 degrees, confirming that it properly validates the input values and ensures they fall within acceptable ranges for geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((0.0, 181.0, -45.0, 45.0)) is False

    def test_lat_min_out_of_range(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a geographic extent with a latitude minimum out of range. The test checks that the method returns False for an extent with a latitude minimum less than -90 degrees, confirming that it properly validates the input values and ensures they fall within acceptable ranges for geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((-90.0, 90.0, -91.0, 45.0)) is False

    def test_lat_max_out_of_range(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a geographic extent with a latitude maximum out of reflect. The test checks that the method returns False for an extent with a latitude maximum greater than 90 degrees, confirming that it properly validates the input values and ensures they fall with acceptable ranges for geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((-90.0, 90.0, -45.0, 91.0)) is False

    def test_lon_reversed(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a geographic extent with reversed longitude values. The test checks that the method returns False for an extent where the longitude minimum is greater than the longitude maximum, confirming that it properly validates the input values and ensures they are logically consistent for geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((10.0, 5.0, -45.0, 45.0)) is False

    def test_lat_reversed(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that the validate_geographic_extent method correctly identifies a geographic extent with reversed latitude values. The test checks that the method returns False for an extent where the latitude minimum is greater than the latitude maximum, confirming that it properly validates the input values and ensures they are logically consistent for geographic coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((-90.0, 90.0, 45.0, 10.0)) is False


class TestGetExtentFromCoordinates:
    """Tests for MPASGeographicUtils.get_extent_from_coordinates."""

    def test_all_nan_raises(self: 'TestGetExtentFromCoordinates') -> None:
        """
        This test verifies that the get_extent_from_coordinates method raises a ValueError when all coordinates are NaN. The error message should indicate that no valid coordinates were found, ensuring that the method properly handles cases where the input data is invalid and provides clear feedback to the user. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.full(10, np.nan)
        lat = np.full(10, np.nan)
        with pytest.raises(ValueError, match="No valid coordinates found"):
            MPASGeographicUtils.get_extent_from_coordinates(lon, lat)

    def test_all_inf_raises(self: 'TestGetExtentFromCoordinates') -> None:
        """
        This test verifies that the get_extent_from_coordinates method raises a ValueError when all coordinates are infinite. The error message should indicate that no valid coordinates were found, ensuring that the method properly handles cases where the input data is invalid and provides clear feedback to the user. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.full(5, np.inf)
        lat = np.full(5, -np.inf)
        with pytest.raises(ValueError, match="No valid coordinates found"):
            MPASGeographicUtils.get_extent_from_coordinates(lon, lat)

    def test_valid_coords_returns_bounding_box(self: 'TestGetExtentFromCoordinates') -> None:
        """
        This test verifies that the get_extent_from_coordinates method correctly calculates the bounding box (extent) from valid longitude and latitude coordinates. The test checks that the returned longitude minimum, longitude maximum, latitude minimum, and latitude maximum values are approximately equal to the expected values based on the input coordinates, confirming that the method accurately computes the geographic extent from the provided coordinate arrays. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.linspace(-120.0, -80.0, 50)
        lat = np.linspace(25.0, 55.0, 50)
        lon_min, lon_max, lat_min, lat_max = MPASGeographicUtils.get_extent_from_coordinates(lon, lat)
        assert lon_min == pytest.approx(-120.0)
        assert lon_max == pytest.approx(-80.0)
        assert lat_min == pytest.approx(25.0)
        assert lat_max == pytest.approx(55.0)

    def test_buffer_expands_extent(self: 'TestGetExtentFromCoordinates') -> None:
        """
        This test verifies that the get_extent_from_coordinates method correctly expands the calculated extent by a specified buffer distance. The test checks that the returned longitude minimum, longitude maximum, latitude minimum, and latitude maximum values are approximately equal to the expected values based on the input coordinates plus the buffer, confirming that the method accurately applies the buffer to expand the geographic extent as intended. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.linspace(-100.0, -90.0, 20)
        lat = np.linspace(30.0, 40.0, 20)
        lon_min, lon_max, lat_min, lat_max = MPASGeographicUtils.get_extent_from_coordinates(
            lon, lat, buffer=5.0
        )
        assert lon_min == pytest.approx(-105.0)
        assert lon_max == pytest.approx(-85.0)
        assert lat_min == pytest.approx(25.0)
        assert lat_max == pytest.approx(45.0)

    def test_lat_clamped_to_valid_range(self: 'TestGetExtentFromCoordinates') -> None:
        """
        This test verifies that the get_extent_from_coordinates method correctly clamps the latitude values to the valid range of -90 to 90 degrees when the input coordinates exceed these limits. The test checks that the returned latitude maximum value does not exceed 90 degrees, confirming that the method properly handles cases where the input coordinates are outside the valid geographic range and ensures that the calculated extent remains within acceptable bounds. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.linspace(-10.0, 10.0, 10)
        lat = np.array([85.0, 88.0, 89.5, 89.9, 90.0, 85.0, 87.0, 88.5, 89.0, 89.8])
        _, _, _, lat_max = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=5.0)
        assert lat_max <= 90.0


class TestCalculateSpatialResolution:
    """Tests for MPASGeographicUtils.calculate_spatial_resolution."""

    def test_single_point_returns_zero(self: 'TestCalculateSpatialResolution') -> None:
        """
        This test verifies that the calculate_spatial_resolution method returns zero when the input consists of a single point. Since spatial resolution is typically defined as the average distance between points, having only one point means there are no distances to calculate, and thus the method should return zero. The test checks that the result is approximately equal to zero, confirming that the method handles this edge case correctly and does not produce an error or an invalid value when given minimal input. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASGeographicUtils.calculate_spatial_resolution(
            np.array([45.0]), np.array([30.0])
        )
        assert result == pytest.approx(0.0)

    def test_normal_data_returns_positive(self: 'TestCalculateSpatialResolution') -> None:
        """
        This test verifies that the calculate_spatial_resolution method returns a positive value when given a typical set of longitude and latitude coordinates. The test checks that the result is greater than zero and is of type float, confirming that the method correctly calculates a meaningful spatial resolution based on the input coordinates and that it returns a valid numeric value representing the average distance between points in the geographic space.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.linspace(-120.0, -80.0, 100)
        lat = np.linspace(25.0, 55.0, 100)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)
        assert result > 0.0
        assert isinstance(result, float)

    def test_sampling_path_large_dataset(self: 'TestCalculateSpatialResolution') -> None:
        """
        This test verifies that the calculate_spatial_resolution method returns a positive value when given a large dataset of longitude and latitude coordinates and a specified sample size. The test checks that the result is greater than zero, confirming that the method correctly calculates the spatial resolution using a subset of the data when the sample size is smaller than the total number of points, and that it handles larger datasets efficiently without errors. 

        Parameters:
            None

        Returns:
            None
        """
        rng = np.random.default_rng(42)
        lon = rng.uniform(-120.0, -80.0, 2000)
        lat = rng.uniform(25.0, 55.0, 2000)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat, sample_size=100)
        assert result > 0.0

    def test_no_sampling_small_dataset(self: 'TestCalculateSpatialResolution') -> None:
        """
        This test verifies that the calculate_spatial_resolution method returns a positive value when given a small dataset of longitude and latitude coordinates and a specified sample size that exceeds the number of points. The test checks that the result is greater than zero, confirming that the method correctly calculates the spatial resolution using all available data points when the sample size is larger than the total number of points, and that it does not produce an error or invalid value in this scenario. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.linspace(-100.0, -90.0, 50)
        lat = np.linspace(30.0, 40.0, 50)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat, sample_size=1000)
        assert result > 0.0

    def test_identical_points_returns_zero(self: 'TestCalculateSpatialResolution') -> None:
        """
        This test verifies that the calculate_spatial_resolution method returns zero when all input longitude and latitude coordinates are identical. Since spatial resolution is based on the distance between points, if all points are the same, there is no distance to calculate, and thus the method should return zero. The test checks that the result is approximately equal to 0.0, confirming that the method correctly handles this edge case without producing an error or an invalid value. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.full(10, 45.0)
        lat = np.full(10, 30.0)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)
        assert result == pytest.approx(0.0)


class TestIsGlobalExtent:
    """Tests for MPASGeographicUtils.is_global_extent."""

    def test_global_extent_returns_true(self: 'TestIsGlobalExtent') -> None:
        """
        This test verifies that the is_global_extent method returns True for a geographic extent that covers the entire globe, with longitude values from -180 to 180 degrees and latitude values from -90 to 90 degrees. The test checks that the method correctly identifies this extent as global, confirming that it properly evaluates the input values against the criteria for a global geographic extent. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.is_global_extent((-180.0, 180.0, -90.0, 90.0)) is True

    def test_non_global_lon_returns_false(self: 'TestIsGlobalExtent') -> None:
        """
        This test verifies that the is_global_extent method returns False for a non-global longitude extent. The test checks that the method correctly identifies an extent with longitude values that do not cover the full range of -180 to 180 degrees as non-global, confirming that it properly evaluates the input values against the criteria for a global geographic extent. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.is_global_extent((-90.0, 90.0, -90.0, 90.0)) is False

    def test_non_global_lat_returns_false(self: 'TestIsGlobalExtent') -> None:
        """
        This test verifies that the is_global_extent method returns False for a non-global latitude extent. The test checks that the method correctly identifies an extent with latitude values that do not cover the full range of -90 to 90 degrees as non-global, confirming that it properly evaluates the input values against the criteria for a global geographic extent. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.is_global_extent((-180.0, 180.0, -45.0, 45.0)) is False

    def test_custom_tolerance_passes(self: 'TestIsGlobalExtent') -> None:
        """
        This test verifies that the is_global_extent method returns True when a custom tolerance is provided and the extent is within the tolerance of being global. The test checks that the method correctly identifies an extent that is close enough to the global range, based on the specified tolerance, as global, confirming that it properly evaluates the input values against the criteria for a global geographic extent while allowing for some flexibility with the tolerance parameter. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.is_global_extent(
            (-179.5, 179.5, -89.5, 89.5), tolerance=2.0
        ) is True

    def test_custom_tolerance_fails(self: 'TestIsGlobalExtent') -> None:
        """
        This test verifies that the is_global_extent method returns False when a custom tolerance is provided but the extent is not within the tolerance of being global. The test checks that the method correctly identifies an extent that is not close enough to the global range, based on the specified tolerance, as non-global, confirming that it properly evaluates the input values against the criteria for a global geographic extent while respecting the limits set by the tolerance parameter. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.is_global_extent(
            (-179.5, 179.5, -89.5, 89.5), tolerance=0.0
        ) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
